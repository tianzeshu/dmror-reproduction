"""Cycle-free beam search using the manuscript's product of raw edge intensities."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def beam_search_paths(src: Tensor, dst: Tensor, edge_score: Tensor, source: int,
                      beam_width: int = 10, max_hops: int = 4, top_k: int = 5,
                      node_prob: Optional[Tensor] = None, min_node_prob: float = 0.0,
                      min_hops: int = 1) -> list[dict]:
    """Rank candidate paths by sum(log(pi)); no repeated node in any path.

    Return nodes, edge_ids, score and log_score. Scores above one are legal.
    Candidates of different lengths use the paper's unnormalized product, so
    short/long preference depends on pi's scale; no length penalty is invented.
    Moving detached scores to CPU is inference-only and does not affect training.
    """
    if beam_width < 1 or max_hops < 1 or top_k < 1 or min_hops < 1 or min_hops > max_hops:
        raise ValueError("Invalid beam width, hop count or top_k")
    src_cpu = torch.as_tensor(src).detach().cpu().reshape(-1).tolist()
    dst_cpu = torch.as_tensor(dst).detach().cpu().reshape(-1).tolist()
    scores = torch.as_tensor(edge_score).detach().cpu().reshape(-1).tolist()
    if len(src_cpu) != len(dst_cpu) or len(src_cpu) != len(scores):
        raise ValueError("src, dst and edge_score must have equal length")
    probabilities = None if node_prob is None else torch.as_tensor(node_prob).detach().cpu().tolist()
    adjacency: dict[int, list[tuple[int, int, float]]] = {}
    for edge_id, (u, v, score) in enumerate(zip(src_cpu, dst_cpu, scores)):
        if not math.isfinite(score) or score < 0:
            raise ValueError("Propagation edge scores must be finite and nonnegative")
        if score == 0 or (probabilities is not None and probabilities[v] < min_node_prob):
            continue
        adjacency.setdefault(u, []).append((v, edge_id, math.log(score)))
    # (log_product, nodes_tuple, edge_ids_tuple)
    beam = [(0.0, (int(source),), ())]
    candidates = []
    for depth in range(1, max_hops + 1):
        expanded = []
        for log_score, nodes, edge_ids in beam:
            for target, edge_id, log_edge in adjacency.get(nodes[-1], []):
                if target in nodes:
                    continue
                expanded.append((log_score + log_edge, nodes + (target,), edge_ids + (edge_id,)))
        expanded.sort(key=lambda item: (-item[0], item[1], item[2]))
        beam = expanded[:beam_width]
        if depth >= min_hops:
            candidates.extend(beam)
        if not beam:
            break
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [{"nodes": list(nodes), "edge_ids": list(edge_ids),
             "score": math.exp(log_score) if log_score < 709 else float("inf"),
             "log_score": log_score} for log_score, nodes, edge_ids in candidates[:top_k]]


def path_log_scores(edge_score: Tensor, paths: list[list[int]], eps: float = 1e-12) -> Tensor:
    """Differentiable sums of log intensities for optional path-ranking training."""
    if not paths:
        return edge_score.new_empty(0)
    scores = []
    for edge_ids in paths:
        if not edge_ids:
            scores.append(edge_score.sum() * 0)
        else:
            index = torch.as_tensor(edge_ids, dtype=torch.long, device=edge_score.device)
            scores.append(edge_score[index].clamp_min(eps).log().sum())
    return torch.stack(scores)
