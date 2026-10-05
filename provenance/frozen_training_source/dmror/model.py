"""PyTorch implementation of the July 2026 DM-ROR manuscript, Sec. 3.

The manuscript specifies HGNN abstractly; here it is a dependency-weighted,
relation-specific message encoder. ``signals`` are externally encoded signal
vectors: this module is an adapter and does not claim to run a language model.
The paper's unbounded propagation intensity pi is retained as ``edge_score``;
``edge_prob = 1 - exp(-pi)`` repairs the otherwise invalid path BCE probability.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODES = (
    "full", "no_stm", "no_ltm", "no_time", "no_resgate", "no_overload",
    "avg_fusion", "scalar_gate", "concat",
)


def group_softmax(logits: Tensor, group: Tensor, num_groups: int) -> Tensor:
    """Stable softmax independently for each outgoing (source, relation) group."""
    if logits.numel() == 0:
        return logits
    maximum = logits.new_full((num_groups,), -torch.inf)
    maximum.scatter_reduce_(0, group, logits.detach(), reduce="amax", include_self=True)
    exponent = torch.exp(logits - maximum[group])
    denominator = logits.new_zeros(num_groups)
    denominator.index_add_(0, group, exponent)
    return exponent / denominator[group].clamp_min(torch.finfo(logits.dtype).tiny)


def masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Return zero weights for empty rows, avoiding softmax(-inf,...,-inf)."""
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    mask = mask.bool()
    has_signal = mask.any(dim=-1, keepdim=True)
    masked = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(has_signal, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=-1) * mask.to(logits.dtype)


class HeterogeneousLayer(nn.Module):
    """R-GCN-style HGNN with separate relation transformations and weighted means."""

    def __init__(self, hidden: int, num_relations: int, dropout: float):
        super().__init__()
        self.num_relations = num_relations
        self.relation_weight = nn.Parameter(torch.empty(num_relations, hidden, hidden))
        self.self_projection = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        for weight in self.relation_weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, h: Tensor, src: Tensor, dst: Tensor,
                edge_type: Tensor, dependency: Tensor) -> Tensor:
        n = h.shape[0]
        aggregate = torch.zeros_like(h)
        if src.numel():
            weights = dependency.clamp_min(0)
            group = dst * self.num_relations + edge_type
            degree = h.new_zeros(n * self.num_relations)
            degree.index_add_(0, group, weights)
            message = torch.bmm(h[src].unsqueeze(1), self.relation_weight[edge_type]).squeeze(1)
            message = message * (weights / degree[group].clamp_min(1e-12)).unsqueeze(-1)
            aggregate.index_add_(0, dst, message)
            active_relations = (degree.reshape(n, self.num_relations) > 0).sum(-1)
            aggregate = aggregate / active_relations.clamp_min(1).unsqueeze(-1)
        update = F.gelu(self.self_projection(h) + aggregate)
        return self.norm(h + self.dropout(update))


class DMROR(nn.Module):
    """Dual memory, vector fusion, resilience threshold and overload redistribution.

    Required batch tensors: node_features[N,F], node_type[N], src/dst/type[E],
    edge_dependency[E], edge_features[E,4], resilience[N,5], signals[N,K,D],
    delta/confidence/signal_mask[N,K]. ``delta`` is age in the dataset's time unit.
    ``no_resgate`` removes all resilience inputs and threshold/beta suppression.
    ``no_overload`` replaces residual-load transport with standard diffusion of
    sigmoid(load) through the learned outgoing weights; theta remains diagnostic.
    """

    def __init__(self, node_dim: int, signal_dim: int, num_node_types: int,
                 num_relations: int, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, mode: str = "full", time_decay: float = 0.05,
                 beta_penalty: float = 1.0, resilience_dim: int = 5,
                 edge_feature_dim: int = 4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose one of {MODES}")
        if hidden < 2 or num_relations < 1 or num_node_types < 1 or num_layers < 0:
            raise ValueError("Invalid hidden, type, relation or layer count")
        if time_decay < 0 or beta_penalty < 0:
            raise ValueError("time_decay and beta_penalty must be nonnegative")
        self.mode = mode
        self.hidden = hidden
        self.num_relations = num_relations
        self.time_decay = float(time_decay)
        self.beta_penalty = float(beta_penalty)
        self.node_projection = nn.Linear(node_dim, hidden)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.graph_layers = nn.ModuleList([
            HeterogeneousLayer(hidden, num_relations, dropout) for _ in range(num_layers)
        ])
        self.signal_projection = nn.Sequential(nn.Linear(signal_dim, hidden), nn.GELU(),
                                               nn.LayerNorm(hidden))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.short_projection = nn.Linear(hidden, hidden, bias=False)
        self.vector_gate = nn.Linear(2 * hidden + resilience_dim, hidden)
        self.scalar_gate = nn.Linear(2 * hidden + resilience_dim, 1)
        self.concat_projection = nn.Linear(2 * hidden, hidden)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.load_head = nn.Linear(hidden, 1)
        self.threshold_head = nn.Linear(hidden + resilience_dim, 1)
        self.relation_embedding = nn.Embedding(num_relations, hidden)
        self.buffer_head = nn.Linear(3 * hidden + 1 + edge_feature_dim, 1)
        self.redistribution_head = nn.Linear(3 * hidden + 2, 1)
        self.node_head = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))
        # A dead ReLU at initialization would prevent path supervision from ever
        # training the load/threshold heads. Start with a positive load margin.
        nn.init.normal_(self.load_head.weight, std=0.02)
        nn.init.constant_(self.load_head.bias, 1.0)
        nn.init.normal_(self.threshold_head.weight, std=0.02)
        nn.init.constant_(self.threshold_head.bias, -1.0)

    def _long_memory(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch["node_features"]
        if self.mode == "no_ltm":
            return x.new_zeros((x.shape[0], self.hidden))
        h = self.input_norm(self.node_projection(x) + self.node_type_embedding(batch["node_type"].long()))
        for layer in self.graph_layers:
            h = layer(h, batch["src"].long(), batch["dst"].long(),
                      batch["edge_type"].long(), batch["edge_dependency"].reshape(-1).to(x.dtype))
        return h

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Support one prediction time or [B,N,K,D] temporal batches.

        The structural graph is encoded once for a temporal minibatch because
        it is shared long-term memory. Dynamic outputs are stacked on axis B.
        """
        if batch["signals"].ndim != 4:
            return self._forward_single(batch)
        h_long = self._long_memory(batch)
        temporal_keys = ("signals", "delta", "confidence", "signal_mask")
        outputs = []
        for index in range(batch["signals"].shape[0]):
            single = dict(batch)
            for key in temporal_keys:
                single[key] = batch[key][index]
            if batch["resilience"].ndim == 3:
                single["resilience"] = batch["resilience"][index]
            outputs.append(self._forward_single(single, h_long))
        if not outputs:
            raise ValueError("Temporal minibatch must contain at least one prediction time")
        return {key: torch.stack([output[key] for output in outputs]) for key in outputs[0]}

    def _forward_single(self, batch: Dict[str, Tensor], cached_long: Tensor = None) -> Dict[str, Tensor]:
        x = batch["node_features"]
        node_type = batch["node_type"].long()
        src, dst = batch["src"].long(), batch["dst"].long()
        edge_type = batch["edge_type"].long()
        dependency = batch["edge_dependency"].reshape(-1).to(x.dtype)
        edge_features = batch["edge_features"].to(x.dtype)
        resilience = batch["resilience"].to(x.dtype)
        signals = batch["signals"].to(x.dtype)
        ages = batch["delta"].to(x.dtype)
        confidence = batch["confidence"].to(x.dtype)
        signal_mask = batch["signal_mask"].bool()
        n = x.shape[0]
        if signals.ndim != 3 or signal_mask.shape != signals.shape[:2]:
            raise ValueError("signals must be [N,K,D] with signal_mask [N,K]")
        if ages.shape != signal_mask.shape or confidence.shape != signal_mask.shape:
            raise ValueError("delta and confidence must match signal_mask")
        if (signal_mask & (ages < 0)).any():
            raise ValueError("Negative signal age denotes future evidence and is not allowed")
        if self.mode == "no_resgate":
            resilience = torch.zeros_like(resilience)
            edge_features = torch.zeros_like(edge_features)

        # Stage I: long-term heterogeneous structural memory.
        h_long = self._long_memory(batch) if cached_long is None else cached_long

        # Stage II: alpha = softmax(q dot k / sqrt(d) - eta*age + confidence).
        available = signal_mask.any(-1)
        if self.mode == "no_stm":
            attention = x.new_zeros(signal_mask.shape)
            m_short = torch.zeros_like(h_long)
            available = torch.zeros_like(available)
        else:
            # Clear padding before the projection: masked NaNs must not contaminate
            # either attention or weighted aggregation via 0 * NaN.
            safe_signals = torch.where(signal_mask.unsqueeze(-1), signals, torch.zeros_like(signals))
            z = self.signal_projection(safe_signals)
            logits = (self.query(h_long).unsqueeze(1) * self.key(z)).sum(-1) / math.sqrt(self.hidden)
            safe_ages = torch.where(signal_mask, ages, torch.zeros_like(ages))
            safe_confidence = torch.where(signal_mask, confidence, torch.zeros_like(confidence))
            if self.mode != "no_time":
                logits = logits - self.time_decay * safe_ages
            attention = masked_softmax(logits + safe_confidence, signal_mask)
            m_short = (attention.unsqueeze(-1) * z).sum(1)
        gate_input = torch.cat((h_long, m_short, resilience), dim=-1)
        short = self.short_projection(m_short)
        if self.mode == "avg_fusion":
            fusion_gate = torch.full_like(h_long, 0.5)
            fused = (h_long + short) / 2
        elif self.mode == "scalar_gate":
            fusion_gate = torch.sigmoid(self.scalar_gate(gate_input)).expand_as(h_long)
            fused = fusion_gate * h_long + (1 - fusion_gate) * short
        elif self.mode == "concat":
            fusion_gate = torch.full_like(h_long, 0.5)
            fused = self.concat_projection(torch.cat((h_long, short), dim=-1))
        elif self.mode == "no_stm":
            fusion_gate = torch.ones_like(h_long)
            fused = h_long
        elif self.mode == "no_ltm":
            fusion_gate = torch.zeros_like(h_long)
            fused = short
        else:
            fusion_gate = torch.sigmoid(self.vector_gate(gate_input))
            fused = fusion_gate * h_long + (1 - fusion_gate) * short
        h = self.fusion_norm(fused)

        # Stage III: ell = softplus(w_ell*h); theta = softplus(w_theta*[b;h]).
        load = F.softplus(self.load_head(h).squeeze(-1))
        threshold = F.softplus(self.threshold_head(torch.cat((resilience, h), -1)).squeeze(-1))
        if self.mode == "no_resgate":
            threshold = torch.zeros_like(threshold)
        overload = F.relu(load - threshold)
        relation = self.relation_embedding(edge_type)
        pair = torch.cat((h[src], h[dst], relation), dim=-1)
        beta_input = torch.cat((pair, dependency.unsqueeze(-1), edge_features), dim=-1)
        beta = torch.sigmoid(self.buffer_head(beta_input).squeeze(-1))
        if self.mode == "no_resgate":
            beta = torch.zeros_like(beta)
        relevance = F.cosine_similarity(m_short[src], m_short[dst], dim=-1, eps=1e-8)
        redistribution_input = torch.cat((pair, dependency.unsqueeze(-1), relevance.unsqueeze(-1)), -1)
        redistribution_logits = self.redistribution_head(redistribution_input).squeeze(-1)
        transport_beta = beta
        transport_load = overload
        if self.mode == "no_overload":
            # Standard graph diffusion baseline: bounded local score and no
            # thresholded residual or relation-level absorption in transport.
            transport_load = torch.sigmoid(load)
            transport_beta = torch.zeros_like(beta)
        redistribution_logits = redistribution_logits - self.beta_penalty * transport_beta
        groups = src * self.num_relations + edge_type
        delta_edge = group_softmax(redistribution_logits, groups, n * self.num_relations)
        transmitted = transport_load[src] * delta_edge * (1 - transport_beta)
        incoming = x.new_zeros(n)
        incoming.index_add_(0, dst, transmitted)
        node_input = torch.cat((h, load.unsqueeze(-1), incoming.unsqueeze(-1), threshold.unsqueeze(-1)), -1)
        node_logits = self.node_head(node_input).squeeze(-1)
        node_prob = torch.sigmoid(node_logits)
        edge_score = transmitted * node_prob[dst]
        edge_prob = -torch.expm1(-edge_score)
        return {
            "node_logits": node_logits, "node_prob": node_prob,
            "edge_prob": edge_prob, "edge_score": edge_score,
            "overload": overload, "incoming": incoming, "threshold": threshold,
            "beta": beta, "delta_edge": delta_edge, "h_long": h_long,
            "m_short": m_short, "attention": attention, "load": load,
            "h_fused": h, "fusion_gate": fusion_gate,
            "redistribution_logits": redistribution_logits,
            "context_available": available & (self.mode not in ("no_stm", "no_ltm")),
        }
