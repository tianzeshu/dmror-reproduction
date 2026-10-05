"""Manuscript Sec. 3.4 node, contrastive-memory and propagation supervision.

Paper pi is an intensity, not necessarily a probability. Supervised path BCE
therefore uses p=1-exp(-pi); raw pi is used for path ranking and beam search.
All objectives are means to make weights independent of graph/batch size.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def alignment_loss(m_short: Tensor, h_long: Tensor, node_mask: Optional[Tensor] = None,
                   temperature: float = 0.1) -> Tensor:
    """One-direction cosine InfoNCE matching the paper's L_align equation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if m_short.ndim == 3:
        # A temporal batch must not count the same entity at another timestamp
        # as a negative example in the manuscript's entity-only denominator.
        rows = [alignment_loss(m_short[index], h_long[index],
                               None if node_mask is None else node_mask[index], temperature)
                for index in range(m_short.shape[0])]
        return torch.stack(rows).mean() if rows else m_short.sum() * 0
    if node_mask is not None:
        m_short, h_long = m_short[node_mask.bool()], h_long[node_mask.bool()]
    if m_short.shape[0] < 2:
        return (m_short.sum() + h_long.sum()) * 0
    similarity = F.normalize(m_short, dim=-1) @ F.normalize(h_long, dim=-1).T
    target = torch.arange(m_short.shape[0], device=m_short.device)
    return F.cross_entropy(similarity / temperature, target)


def path_supervision_loss(edge_prob: Tensor, edge_labels: Tensor,
                          edge_mask: Optional[Tensor] = None) -> Tensor:
    """BCE on the explicitly bounded propagation probability, never raw pi."""
    if edge_mask is not None:
        edge_prob, edge_labels = edge_prob[edge_mask.bool()], edge_labels[edge_mask.bool()]
    if edge_prob.numel() == 0:
        return edge_prob.sum() * 0
    return F.binary_cross_entropy(edge_prob.clamp(1e-7, 1 - 1e-7), edge_labels.to(edge_prob.dtype))


def path_ranking_loss(positive_scores: Tensor, negative_scores: Tensor,
                      margin: float = 0.2) -> Tensor:
    """Optional ranking alternative for weakly observed positive/negative paths."""
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return (positive_scores.sum() + negative_scores.sum()) * 0
    return F.relu(margin - positive_scores.unsqueeze(-1) + negative_scores).mean()


def multitask_loss(outputs: Dict[str, Tensor], labels: Tensor,
                   node_mask: Optional[Tensor] = None,
                   edge_labels: Optional[Tensor] = None,
                   edge_mask: Optional[Tensor] = None,
                   lambda_align: float = 0.05, lambda_path: float = 0.1,
                   temperature: float = 0.1, model: Optional[nn.Module] = None,
                   lambda_reg: float = 0.0, pos_weight: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Return total/node/align/path/reg; masks restrict supervised split access."""
    logits = outputs["node_logits"]
    if node_mask is None:
        node_mask = labels >= 0
    node_mask = node_mask.bool()
    node = logits.sum() * 0
    if node_mask.any():
        node = F.binary_cross_entropy_with_logits(logits[node_mask], labels[node_mask].to(logits.dtype),
                                                 pos_weight=pos_weight)
    alignment_mask = node_mask & outputs.get("context_available", torch.ones_like(node_mask))
    align = alignment_loss(outputs["m_short"], outputs["h_long"], alignment_mask, temperature)
    path = logits.sum() * 0
    if edge_labels is not None:
        if edge_mask is None:
            edge_mask = edge_labels >= 0
        path = path_supervision_loss(outputs["edge_prob"], edge_labels, edge_mask)
    reg = logits.sum() * 0
    if model is not None and lambda_reg:
        reg = sum((parameter.square().sum() for parameter in model.parameters()), reg)
    total = node + lambda_align * align + lambda_path * path + lambda_reg * reg
    return {"total": total, "node": node, "align": align, "path": path, "reg": reg}


compute_loss = multitask_loss
