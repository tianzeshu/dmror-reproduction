"""Metrics with explicit unknown-label handling and day-level ranking denominators."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def binary_metrics(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels), np.asarray(scores)
    keep = (labels >= 0) & np.isfinite(scores)
    y, p = labels[keep].astype(int), scores[keep]
    if not len(y):
        return {"n_known": 0, "n_positive": 0, "auc": None, "auprc": None,
                "macro_f1": None, "positive_f1": None, "threshold": float(threshold)}
    two_classes = np.unique(y).size == 2
    return {"n_known": int(len(y)), "n_positive": int(y.sum()),
            "auc": float(roc_auc_score(y, p)) if two_classes else None,
            "auprc": float(average_precision_score(y, p)) if y.sum() else None,
            "macro_f1": float(f1_score(y, p >= threshold, labels=[0, 1],
                                       average="macro", zero_division=0)),
            "positive_f1": float(f1_score(y, p >= threshold, zero_division=0)),
            "threshold": float(threshold)}


def tune_threshold(labels, scores, objective="macro_f1"):
    """Choose a threshold using validation labels only; ties prefer 0.5 proximity."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    keep = (labels >= 0) & np.isfinite(scores)
    if not keep.any():
        return 0.5
    p = scores[keep]
    y = labels[keep].astype(int)
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91),
                                            np.quantile(p, np.linspace(0, 1, 101)), [0.5]]))
    if objective not in ("macro_f1", "positive_f1"):
        raise ValueError("threshold objective must be macro_f1 or positive_f1")
    # Avoid recomputing ranking AUC/AP for each candidate threshold; those
    # metrics cannot influence this F1-based selection.
    sorted_order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[sorted_order], y[sorted_order]
    cumulative_positive = np.r_[0, np.cumsum(sorted_y)]
    cuts = np.searchsorted(sorted_p, candidates, side="left")
    false_negative = cumulative_positive[cuts]
    true_positive = y.sum() - false_negative
    true_negative = cuts - false_negative
    false_positive = len(y) - cuts - true_positive
    denominator_positive = 2 * true_positive + false_positive + false_negative
    denominator_negative = 2 * true_negative + false_positive + false_negative
    f1_positive = np.divide(2 * true_positive, denominator_positive,
                            out=np.zeros_like(candidates), where=denominator_positive > 0)
    f1_negative = np.divide(2 * true_negative, denominator_negative,
                            out=np.zeros_like(candidates), where=denominator_negative > 0)
    values = (f1_positive + f1_negative) / 2 if objective == "macro_f1" else f1_positive
    best = max((float(value), -abs(t - .5), t) for t, value in zip(candidates, values))
    return float(best[2])


def ranking_metrics(labels, scores, k=None):
    """Rank only known nodes. Empty-positive days count as zero and are disclosed."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim == 1:
        labels, scores = labels[None, :], scores[None, :]
    k = min(labels.shape[1], max(5, math.ceil(labels.shape[1] * 0.1))) if k is None else int(k)
    rows = []
    for y, p in zip(labels, scores):
        known = (y >= 0) & np.isfinite(p)
        if not known.any():
            continue
        y, p = y[known].astype(int), p[known]
        kk = min(k, len(y))
        ranked = np.argsort(-p, kind="stable")[:kk]
        hits, positives = int(y[ranked].sum()), int(y.sum())
        discount = 1.0 / np.log2(np.arange(kk) + 2.0)
        dcg = float(np.sum(y[ranked] * discount))
        idcg = float(discount[:min(positives, kk)].sum())
        rows.append([hits / kk, hits / positives if positives else 0.0,
                     dcg / idcg if idcg else 0.0,
                     hits / (kk + positives - hits) if kk + positives - hits else 0.0,
                     positives])
    if not rows:
        return {"k": k, "n_days": 0, "n_positive_days": 0,
                "precision_at_k": None, "recall_at_k": None,
                "ndcg_at_k": None, "jaccard_at_k": None}
    rows = np.asarray(rows)
    return {"k": k, "n_days": len(rows), "n_positive_days": int((rows[:, 4] > 0).sum()),
            "precision_at_k": float(rows[:, 0].mean()),
            "recall_at_k": float(rows[:, 1].mean()),
            "ndcg_at_k": float(rows[:, 2].mean()), "jaccard_at_k": float(rows[:, 3].mean())}


def threshold_scope_metrics(labels, scores, threshold):
    """Day-level Jaccard of validation-thresholded risk set and known positives."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim == 1:
        labels, scores = labels[None, :], scores[None, :]
    values, positive_values = [], []
    empty_union_days = 0
    for y, p in zip(labels, scores):
        known = (y >= 0) & np.isfinite(p)
        if not known.any():
            continue
        truth, predicted = y[known] == 1, p[known] >= threshold
        union = int((truth | predicted).sum())
        if union == 0:
            empty_union_days += 1
        jaccard = float((truth & predicted).sum() / union) if union else 1.0
        values.append(jaccard)
        if truth.any():
            positive_values.append(jaccard)
    return {"jaccard": float(np.mean(values)) if values else None,
            "jaccard_positive_days": float(np.mean(positive_values)) if positive_values else None,
            "jaccard_threshold": float(threshold), "jaccard_n_days": len(values),
            "jaccard_n_positive_days": len(positive_values), "jaccard_empty_union_days": empty_union_days,
            "jaccard_empty_union_convention": "one (both sets empty)",
            "jaccard_definition": "known nodes with score >= validation-selected node threshold"}


def _extract_nodes(path):
    if isinstance(path, dict):
        return tuple(map(int, path.get("nodes", path.get("path", []))))
    if hasattr(path, "nodes"):
        return tuple(map(int, path.nodes))
    if isinstance(path, (tuple, list)) and path and isinstance(path[0], (tuple, list, np.ndarray)):
        return tuple(map(int, path[0]))
    return tuple(map(int, path))


def _true_paths(src, dst, edge_labels, sources, max_hops=4, cap=10000):
    adjacency = {}
    for u, v, label in zip(src, dst, edge_labels):
        if label == 1:
            adjacency.setdefault(int(u), []).append(int(v))
    paths, truncated = set(), False
    def visit(path):
        nonlocal truncated
        if len(paths) >= cap:
            truncated = True
            return
        if len(path) > 1:
            paths.add(tuple(path))
        if len(path) - 1 == max_hops:
            return
        for nxt in adjacency.get(path[-1], []):
            if nxt not in path:
                visit(path + [nxt])
    for source in sources:
        visit([int(source)])
    return paths, truncated


def path_metrics(src, dst, edge_labels, edge_scores, source_mask, beam_search,
                 node_scores=None, beam_width=10, max_hops=4, top_k=5):
    """Compare paths with paths induced by *labelled* positive directed edges.

    This is simulator path truth if labels were simulated, not evidence of real
    organisational causality. Every positive prefix is a valid truth path.
    Unknown edges are excluded; partially labelled graphs disclose coverage.
    """
    src, dst = np.asarray(src), np.asarray(dst)
    edge_labels, edge_scores = np.asarray(edge_labels), np.asarray(edge_scores)
    if edge_labels.ndim == 1:
        edge_labels, edge_scores = edge_labels[None, :], edge_scores[None, :]
        source_mask = np.asarray(source_mask)[None, :]
    if not (edge_labels >= 0).any():
        return {"status": "skipped_no_edge_labels", "path_at_k": None,
                "path_at_k_evaluable_sources": 0, "path_precision": None,
                "path_recall": None, "path_f1": None}
    values, details = [], []
    source_hits, evaluable_sources = 0, 0
    from .paths import beam_search_paths, prepare_adjacency
    for day in range(len(edge_labels)):
        truth_labels = edge_labels[day]
        sources = np.flatnonzero(np.asarray(source_mask)[day])
        if not len(sources) or not (truth_labels >= 0).any():
            continue
        truth, truncated = _true_paths(src, dst, truth_labels, sources, max_hops)
        predictions = set()
        source_details = []
        labelled_scores = np.where(truth_labels >= 0, edge_scores[day], 0.0)
        prepared = (prepare_adjacency(src, dst, labelled_scores,
                                      None if node_scores is None else node_scores[day], 0.0)
                    if beam_search is beam_search_paths else None)
        truth_by_source = {}
        for path in truth:
            truth_by_source.setdefault(path[0], set()).add(path)
        for source in sources:
            # Do not count paths through unknown edges as false positives:
            # evaluate both truth and prediction on the same labelled subgraph.
            cache_args = {"prepared_adjacency": prepared} if prepared is not None else {}
            paths = beam_search(src, dst, labelled_scores, int(source),
                                beam_width=beam_width, max_hops=max_hops, top_k=top_k,
                                node_prob=None if node_scores is None else node_scores[day],
                                min_node_prob=0.0, **cache_args)
            if isinstance(paths, dict):
                paths = paths.get("paths", [])
            predicted_for_source = {_extract_nodes(p) for p in paths if len(_extract_nodes(p)) > 1}
            predictions.update(predicted_for_source)
            true_for_source = truth_by_source.get(int(source), set())
            evaluable = bool(true_for_source)
            hit = bool(true_for_source & predicted_for_source)
            if evaluable:
                evaluable_sources += 1
                source_hits += int(hit)
            source_details.append({"source": int(source), "evaluable": evaluable, "hit_at_k": hit,
                                   "true_paths": len(true_for_source), "predicted_paths": len(predicted_for_source)})
        matched = truth & predictions
        precision = len(matched) / len(predictions) if predictions else 0.0
        recall = len(matched) / len(truth) if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        pred_edges = {(a, b) for path in predictions for a, b in zip(path, path[1:])}
        true_edges = {(a, b) for path in truth for a, b in zip(path, path[1:])}
        matched_edges = pred_edges & true_edges
        edge_precision = len(matched_edges) / len(pred_edges) if pred_edges else 0.0
        edge_recall = len(matched_edges) / len(true_edges) if true_edges else 0.0
        values.append([precision, recall, f1, edge_precision, edge_recall])
        details.append({"day_index": day, "true_paths": len(truth),
                        "predicted_paths": len(predictions), "matched_paths": len(matched),
                        "truth_enumeration_truncated": truncated,
                        "known_edge_fraction": float((truth_labels >= 0).mean()), "sources": source_details})
    if not values:
        return {"status": "skipped_no_source_days", "path_at_k": None,
                "path_at_k_evaluable_sources": 0, "path_precision": None,
                "path_recall": None, "path_f1": None}
    mean = np.mean(values, axis=0)
    return dict(zip(["path_precision", "path_recall", "path_f1", "path_edge_precision",
                     "path_edge_recall"], map(float, mean)), status="computed", n_days=len(values),
                max_hops=max_hops, top_k_per_source=top_k, beam_width=beam_width,
                path_at_k=source_hits / evaluable_sources if evaluable_sources else None,
                path_at_k_hits=source_hits, path_at_k_evaluable_sources=evaluable_sources,
                path_at_k_definition="fraction of source-times with >=1 exact labelled positive-prefix path in top-K; denominator excludes sources with no true path",
                truth_definition="all simple source-reachable positive-edge prefixes up to max_hops; node-sequence exact match; parallel relations collapse",
                evaluation_graph="known-label edges only",
                per_day=details)
