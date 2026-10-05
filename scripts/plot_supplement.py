"""Plot saved held-out predictions with explicit support and simulator scope.

No training, dataset edits, checkpoint loading or surrogate results are performed.
Run from the delivery root; see docs/FIGURE_CONTRACT.md for the figure contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43, 71, 101)
BLUE, ORANGE, GRAY = "#376A92", "#CC8648", "#B8BDC3"
STRATA = {
    "age": ("(0, 7] d", "(7, 14] d", "(14, 30] d", ">30 d", "No context"),
    "confidence": ("0.55", "0.80", "0.90", "Other", "No context"),
    "indegree": ("0", "1-2", "3-5", "6+"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_name(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        # Input can be relocated. The manifest should remain portable and must
        # not leak an absolute user/server directory into figure deliverables.
        return path.name


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")


def read_dataset(path, manifest):
    path = Path(path)
    fields = ("labels", "edge_labels", "split", "prediction_days", "target_days",
              "signal_mask", "delta", "confidence", "src", "dst", "source_mask")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in fields}
    data["sha256"] = file_hash(path)
    data["name"] = path.parent.name
    metadata = path.parent / "metadata.json"
    data["metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
    test = np.flatnonzero(data["split"] == 2)
    if not len(test):
        raise ValueError(f"{data['name']}: empty test split")
    if data["signal_mask"].shape != data["delta"].shape or data["delta"].shape != data["confidence"].shape:
        raise ValueError(f"{data['name']}: context array shapes differ")
    valid = data["signal_mask"].astype(bool)
    if (data["delta"][valid] <= 0).any() or not np.isfinite(data["delta"][valid]).all():
        raise ValueError(f"{data['name']}: contexts must strictly precede queries")
    if not np.isfinite(data["confidence"][valid]).all():
        raise ValueError(f"{data['name']}: nonfinite confidence")
    times = {}
    for split, name in ((0, "train"), (1, "validation"), (2, "test")):
        indices = np.flatnonzero(data["split"] == split)
        times[name] = {"snapshots": int(len(indices)),
                       "prediction_day_min": int(data["prediction_days"][indices].min()) if len(indices) else None,
                       "prediction_day_max": int(data["prediction_days"][indices].max()) if len(indices) else None,
                       "target_day_min": int(data["target_days"][indices].min()) if len(indices) else None,
                       "target_day_max": int(data["target_days"][indices].max()) if len(indices) else None}
    for older, newer in (("train", "validation"), ("validation", "test")):
        if times[older]["snapshots"] and times[newer]["snapshots"]:
            if times[older]["target_day_max"] >= times[newer]["prediction_day_min"]:
                raise ValueError(f"{data['name']}: target window crosses the {newer} split")
    manifest["datasets"].append({"dataset": data["name"], "file": relative_name(path),
                                  "sha256": data["sha256"], "metadata_sha256": file_hash(metadata),
                                  "data_kind": data["metadata"].get("data_kind"),
                                  "label_kind": data["metadata"].get("label_kind"),
                                  "text_encoder": data["metadata"].get("text_encoder"),
                                  "temporal_cutoffs": times, "test_indices": test.tolist()})
    return data


def read_runs(results, datasets, manifest):
    runs = {}
    for path in sorted(Path(results).rglob("result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") not in ("full", "no_stm"):
            continue
        name, seed = record.get("dataset"), record.get("seed")
        if name not in datasets or seed not in SEEDS:
            continue
        identity = (name, record["mode"], int(seed))
        if identity in runs:
            raise ValueError(f"Duplicate run identity: {identity}")
        prediction_path = path.parent / "predictions.npz"
        if record.get("status") != "completed" or not prediction_path.exists():
            manifest["skipped"].append({"dataset": name, "mode": record["mode"], "seed": seed,
                                        "reason": "run incomplete or saved predictions missing"})
            continue
        data = datasets[name]
        if record.get("config", {}).get("dataset_sha256") != data["sha256"]:
            raise ValueError(f"{identity}: result belongs to a different dataset hash")
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction = {key: archive[key].copy() for key in
                          ("test_indices", "test_labels", "test_node_prob", "test_edge_score")}
            for key, source_key in (("test_edge_labels", "edge_labels"),
                                    ("prediction_days", "prediction_days"),
                                    ("target_days", "target_days")):
                if key in archive:
                    expected = data[source_key][prediction["test_indices"]]
                    if not np.array_equal(archive[key], expected):
                        raise ValueError(f"{identity}: saved {key} differs from dataset")
        test = np.flatnonzero(data["split"] == 2)
        if not np.array_equal(prediction["test_indices"], test):
            raise ValueError(f"{identity}: saved test rows do not exactly match split==2")
        if not np.array_equal(prediction["test_labels"], data["labels"][test]):
            raise ValueError(f"{identity}: saved test labels differ from dataset")
        probabilities = prediction["test_node_prob"]
        if probabilities.shape != data["labels"][test].shape or not np.isfinite(probabilities).all():
            raise ValueError(f"{identity}: invalid node prediction shape or nonfinite probability")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{identity}: node predictions outside [0,1]")
        scores = prediction["test_edge_score"]
        if scores.shape != data["edge_labels"][test].shape or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError(f"{identity}: invalid saved propagation intensities")
        prediction["record"] = record
        prediction["file"] = relative_name(prediction_path)
        runs[identity] = prediction
        manifest["runs"].append({"dataset": name, "mode": record["mode"], "seed": int(seed),
                                 "result_file": relative_name(path), "result_sha256": file_hash(path),
                                 "prediction_file": relative_name(prediction_path),
                                 "prediction_sha256": file_hash(prediction_path),
                                 "dataset_sha256": data["sha256"], "test_rows": len(test)})
    if not runs:
        raise ValueError("No completed full/no_stm predictions for the expected seeds; no figures created")
    return runs


def context_strata(data, test):
    mask = data["signal_mask"][test].astype(bool)
    delta = data["delta"][test]
    has_context = mask.any(axis=-1)
    slot = np.where(mask, delta, np.inf).argmin(axis=-1)
    age = np.take_along_axis(delta, slot[..., None], axis=-1)[..., 0].astype(float)
    confidence = np.take_along_axis(data["confidence"][test], slot[..., None], axis=-1)[..., 0].astype(float)
    age[~has_context], confidence[~has_context] = np.nan, np.nan
    age_bin = np.full(age.shape, 4, dtype=int)
    age_bin[has_context & (age <= 7)] = 0
    age_bin[has_context & (age > 7) & (age <= 14)] = 1
    age_bin[has_context & (age > 14) & (age <= 30)] = 2
    age_bin[has_context & (age > 30)] = 3
    confidence_bin = np.full(age.shape, 3, dtype=int)
    for index, value in enumerate((.55, .8, .9)):
        confidence_bin[has_context & np.isclose(confidence, value, atol=1e-6, rtol=0)] = index
    confidence_bin[~has_context] = 4
    degree = np.bincount(data["dst"], minlength=data["labels"].shape[1])
    degree_bin = np.where(degree == 0, 0, np.where(degree <= 2, 1, np.where(degree <= 5, 2, 3)))
    return {"age": age_bin, "confidence": confidence_bin,
            "indegree": np.broadcast_to(degree_bin, age.shape)}, age, confidence, degree, slot, has_context


def metric_pair(labels, full, comparator, selected, minimum):
    known = selected & (labels >= 0)
    y = labels[known].astype(int)
    support, positives = len(y), int(y.sum())
    reason = None
    if support < minimum:
        reason = f"support_below_{minimum}"
    elif positives < 2 or support - positives < 2:
        reason = "fewer_than_two_positives_or_negatives"
    if reason:
        return {"n": support, "n_positive": positives, "n_negative": support - positives,
                "full_auprc": None, "no_stm_auprc": None, "difference": None, "na_reason": reason}
    a = float(average_precision_score(y, full[known]))
    b = float(average_precision_score(y, comparator[known]))
    return {"n": support, "n_positive": positives, "n_negative": support - positives,
            "full_auprc": a, "no_stm_auprc": b, "difference": a - b, "na_reason": None}


def collect_strata(datasets, runs, minimum, manifest):
    source, metrics, summaries = [], [], []
    for name, data in sorted(datasets.items()):
        test = np.flatnonzero(data["split"] == 2)
        labels = data["labels"][test]
        bins, age, confidence, degree, slot, has_context = context_strata(data, test)
        pairs = [seed for seed in SEEDS if (name, "full", seed) in runs and (name, "no_stm", seed) in runs]
        for seed in SEEDS:
            if seed not in pairs:
                manifest["skipped"].append({"dataset": name, "seed": seed, "reason": "full/no_stm pair incomplete"})
        for seed in pairs:
            full = runs[name, "full", seed]["test_node_prob"]
            comparator = runs[name, "no_stm", seed]["test_node_prob"]
            for day, node in zip(*np.where(labels >= 0)):
                source.append({"dataset": name, "seed": seed, "dataset_sha256": data["sha256"],
                               "snapshot_index": int(test[day]), "node_index": int(node),
                               "prediction_day": int(data["prediction_days"][test[day]]),
                               "target_day": int(data["target_days"][test[day]]), "label": int(labels[day, node]),
                               "full_probability": float(full[day, node]), "no_stm_probability": float(comparator[day, node]),
                               "has_context": bool(has_context[day, node]),
                               "newest_context_slot": int(slot[day, node]) if has_context[day, node] else None,
                               "newest_age_days": float(age[day, node]) if has_context[day, node] else None,
                               "newest_confidence_proxy": float(confidence[day, node]) if has_context[day, node] else None,
                               "in_degree": int(degree[node]),
                               **{kind + "_stratum": STRATA[kind][bins[kind][day, node]] for kind in STRATA}})
            for kind, names in STRATA.items():
                for index, stratum in enumerate(names):
                    row = metric_pair(labels, full, comparator, bins[kind] == index, minimum)
                    metrics.append({"dataset": name, "seed": seed, "kind": kind, "stratum": stratum,
                                    "dataset_sha256": data["sha256"], **row})
        for kind, names in STRATA.items():
            for index, stratum in enumerate(names):
                rows = [row for row in metrics if row["dataset"] == name and row["kind"] == kind and row["stratum"] == stratum]
                values = [row["difference"] for row in rows if row["difference"] is not None]
                support_mask = (labels >= 0) & (bins[kind] == index)
                support, positives = int(support_mask.sum()), int((labels[support_mask] == 1).sum())
                complete = len(values) == len(SEEDS) and set(pairs) == set(SEEDS)
                summaries.append({"dataset": name, "kind": kind, "stratum": stratum,
                                  "n": support, "n_positive": positives, "n_negative": support - positives,
                                  "available_pair_seeds": ";".join(map(str, pairs)),
                                  "valid_pair_seeds": ";".join(str(row["seed"]) for row in rows if row["difference"] is not None),
                                  "n_valid_pairs": len(values), "all_five_pairs_valid": complete,
                                  "mean_difference": float(np.mean(values)) if complete else None,
                                  "sample_sd_difference": float(np.std(values, ddof=1)) if complete else None,
                                  "status": "five_seed_summary" if complete else "NA" if not values else "partial_points_only"})
    return source, metrics, summaries


def plotting_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": .7, "legend.frameon": False})
    return plt


def save_figure(plt, fig, directory, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_strata(plt, datasets, metrics, summaries, output, captions):
    for name in sorted(datasets):
        rows = [row for row in metrics if row["dataset"] == name]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55))
        values = [abs(row["difference"]) for row in rows if row["difference"] is not None]
        bound = max(.05, min(1.0, max(values, default=.05) * 1.3))
        for panel, (kind, names), axis in zip("abc", STRATA.items(), axes):
            axis.axvline(0, color=GRAY, linewidth=.7)
            for index, stratum in enumerate(names):
                group = [row for row in rows if row["kind"] == kind and row["stratum"] == stratum]
                summary = next(row for row in summaries if row["dataset"] == name and row["kind"] == kind and row["stratum"] == stratum)
                valid = [row for row in group if row["difference"] is not None]
                for row in valid:
                    offset = (SEEDS.index(row["seed"]) - 2) * .045
                    axis.scatter(row["difference"], index + offset, s=11, c=BLUE, alpha=.65, zorder=3)
                if summary["all_five_pairs_valid"]:
                    axis.errorbar(summary["mean_difference"], index, xerr=summary["sample_sd_difference"],
                                  fmt="D", color=ORANGE, markersize=3.1, linewidth=.9, capsize=2, zorder=4)
                elif not valid:
                    axis.text(.02 * bound, index, "NA", va="center", color="#686868", fontsize=6)
                else:
                    axis.text(-.98 * bound, index + .22, f"{len(valid)}/5 pairs", color="#686868", fontsize=5.5)
            labels = []
            for stratum in names:
                row = next(item for item in summaries if item["dataset"] == name and item["kind"] == kind and item["stratum"] == stratum)
                labels.append(f"{stratum}\nn={row['n']:,}; +={row['n_positive']:,}")
            titles = {"age": "Newest context age", "confidence": "Confidence proxy", "indegree": "Observed in-degree"}
            axis.set(yticks=np.arange(len(names)), yticklabels=labels, xlim=(-bound, bound),
                     ylim=(len(names) - .5, -.5), xlabel="AUPRC(full) - AUPRC(no_stm)", title=titles[kind])
            axis.tick_params(axis="y", length=0, labelsize=6)
            axis.tick_params(axis="x", labelsize=6)
            axis.grid(axis="x", linewidth=.3, alpha=.25)
            axis.text(-.21, 1.07, panel, transform=axis.transAxes, weight="bold", fontsize=8)
        kind = datasets[name]["metadata"].get("data_kind", "unspecified")
        fig.suptitle(f"{name} | {kind}; held-out observational strata", fontsize=8, y=1.0)
        fig.text(.5, .02, "Blue: paired seeds. Orange: mean +/- sample SD only with all five valid pairs. n: node-times; +: positives.",
                 ha="center", fontsize=6)
        fig.tight_layout(rect=(0, .06, 1, .96), w_pad=1.6)
        save_figure(plt, fig, output, name + "_context_structure_strata")
        captions.append(f"**{name}_context_structure_strata.** Test AUPRC differences between full and no_stm in predeclared newest-context-age, confidence-proxy, and observed in-degree strata. Blue points are saved paired training-seed predictions (17, 29, 43, 71, 101); orange diamonds and error bars are mean +/- sample SD (ddof=1) only when all five pairs are valid. n denotes labelled node-times; + denotes positive node-times. Strata below the minimum support or with fewer than two examples of either class are NA. Latest text means the minimum strictly positive delta among valid slots. Confidence is a extraction/rule proxy, not calibrated quality. The observations are correlated across overlapping time windows; seed SD is not a confidence interval. These descriptive strata do not establish causal effects of context quality or controlled robustness. See source_data/strata_rows.csv, strata_metrics.csv and strata_summary.csv.")


def beam_paths_numpy(src, dst, scores, source, beam_width, max_hops, top_k):
    """Inference-only copy of the product-intensity policy; requires no Torch."""
    adjacency = defaultdict(list)
    for edge, (u, v, score) in enumerate(zip(src, dst, scores)):
        if score > 0:
            adjacency[int(u)].append((int(v), int(edge), math.log(float(score))))
    beam, candidates = [(0., (int(source),), ())], []
    for _ in range(max_hops):
        expanded = []
        for value, nodes, edges in beam:
            for nxt, edge, log_score in adjacency.get(nodes[-1], []):
                if nxt not in nodes:
                    expanded.append((value + log_score, nodes + (nxt,), edges + (edge,)))
        expanded.sort(key=lambda item: (-item[0], item[1], item[2]))
        beam = expanded[:beam_width]
        candidates.extend(beam)
        if not beam:
            break
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [{"rank": rank, "nodes": nodes, "edge_ids": edges, "log_product_intensity": value,
             "product_intensity": math.exp(value) if value < 709 else None}
            for rank, (value, nodes, edges) in enumerate(candidates[:top_k], 1)]


def reachable_layout(src, dst, source, hops):
    adjacency = defaultdict(set)
    for u, v in zip(src, dst):
        adjacency[int(u)].add(int(v))
    distance, queue = {int(source): 0}, deque([int(source)])
    while queue:
        u = queue.popleft()
        if distance[u] == hops:
            continue
        for v in sorted(adjacency[u]):
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    positions = {}
    for depth in range(hops + 1):
        nodes = sorted(node for node, value in distance.items() if value == depth)
        for index, node in enumerate(nodes):
            positions[node] = (depth, (len(nodes) - 1) / 2 - index)
    return distance, positions


def plot_case(plt, data, run, output, source_dir, args, manifest, captions):
    # Selection depends on availability of simulator truth, never prediction
    # quality. Keep the fixed rule in the output so every miss remains auditable.
    indices = run["test_indices"]
    candidates = [int(index) for index in indices if data["source_mask"][index].any()
                  and (data["edge_labels"][index] == 1).any()]
    if not candidates:
        manifest["skipped"].append({"dataset": data["name"], "figure": "simulator_case", "reason": "no source/positive-edge test snapshot"})
        return
    snapshot = candidates[0]
    local = int(np.flatnonzero(indices == snapshot)[0])
    source = int(np.flatnonzero(data["source_mask"][snapshot])[0])
    probability, intensity = run["test_node_prob"][local], run["test_edge_score"][local]
    threshold = float(run["record"]["test"]["node"]["threshold"])
    distance, positions = reachable_layout(data["src"], data["dst"], source, args.case_hops)
    nodes = sorted(positions)
    edges = [int(edge) for edge, (u, v) in enumerate(zip(data["src"], data["dst"])) if int(u) in positions and int(v) in positions]
    paths = beam_paths_numpy(data["src"], data["dst"], intensity, source,
                             args.beam_width, args.case_hops, args.path_top_k)
    path_edges = {edge for path in paths for edge in path["edge_ids"]}
    # A simple path of H hops must lie in the observed H-hop reachable graph.
    if not path_edges.issubset(edges):
        raise ValueError("Beam path exceeds the declared displayed graph")
    stem = data["name"] + "_simulator_case_seed17"
    node_rows = [{"dataset": data["name"], "seed": 17, "snapshot_index": snapshot,
                  "prediction_day": int(data["prediction_days"][snapshot]), "target_day": int(data["target_days"][snapshot]),
                  "node_index": node, "source": bool(data["source_mask"][snapshot, node]),
                  "selected_source": node == source, "hop_distance": distance[node],
                  "label": int(data["labels"][snapshot, node]), "probability": float(probability[node]),
                  "validation_selected_threshold": threshold, "predicted_positive": bool(probability[node] >= threshold),
                  "layout_x": positions[node][0], "layout_y": positions[node][1]}
                 for node in nodes]
    edge_rows = [{"dataset": data["name"], "snapshot_index": snapshot, "edge_index": edge,
                  "source_node": int(data["src"][edge]), "target_node": int(data["dst"][edge]),
                  "simulator_successful_transmission": int(data["edge_labels"][snapshot, edge]),
                  "model_intensity": float(intensity[edge]), "in_top_k_predicted_path": edge in path_edges}
                 for edge in edges]
    path_rows = [{"dataset": data["name"], "snapshot_index": snapshot, "source_node": source,
                  "beam_width": args.beam_width, "max_hops": args.case_hops, "top_k": args.path_top_k,
                  "rank": path["rank"], "nodes": ";".join(map(str, path["nodes"])),
                  "edge_indices": ";".join(map(str, path["edge_ids"])),
                  "product_intensity": path["product_intensity"], "log_product_intensity": path["log_product_intensity"],
                  "all_edges_simulator_positive": bool((data["edge_labels"][snapshot, list(path["edge_ids"])] == 1).all())}
                 for path in paths]
    write_csv(source_dir / (stem + "_nodes.csv"), node_rows, list(node_rows[0]))
    write_csv(source_dir / (stem + "_edges.csv"), edge_rows, list(edge_rows[0]) if edge_rows else
              ["dataset", "snapshot_index", "edge_index", "source_node", "target_node", "simulator_successful_transmission", "model_intensity", "in_top_k_predicted_path"])
    write_csv(source_dir / (stem + "_paths.csv"), path_rows, list(path_rows[0]) if path_rows else
              ["dataset", "snapshot_index", "source_node", "beam_width", "max_hops", "top_k", "rank", "nodes", "edge_indices", "product_intensity", "log_product_intensity", "all_edges_simulator_positive"])
    manifest["case"] = {"dataset": data["name"], "seed": 17, "snapshot_index": snapshot,
                         "source_node": source, "selection": "first test index with source and >=1 positive simulator edge; smallest observed source index",
                         "display": f"all nodes and edges in the source-directed {args.case_hops}-hop induced subgraph",
                         "display_nodes": len(nodes), "display_edges": len(edges), "path_policy": "unnormalized product of saved intensities; no repeated nodes",
                         "validation_selected_node_threshold": threshold, "prediction_day": int(data["prediction_days"][snapshot]),
                         "target_day": int(data["target_days"][snapshot]), "dataset_sha256": data["sha256"]}
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    cmap = LinearSegmentedColormap.from_list("risk_probability", ["#EDF2F6", BLUE])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, max(3.9, min(6.4, len(nodes) * .11 + 2.3))))
    scale = max([intensity[edge] for edge in path_edges], default=1.)
    for panel, axis in enumerate(axes):
        for edge in edges:
            u, v = int(data["src"][edge]), int(data["dst"][edge])
            axis.annotate("", xy=positions[v], xytext=positions[u],
                          arrowprops={"arrowstyle": "->", "color": "#D0D3D6", "lw": .45,
                                      "shrinkA": 6, "shrinkB": 6, "mutation_scale": 5}, zorder=0)
            if panel == 2 and edge in path_edges:
                axis.annotate("", xy=positions[v], xytext=positions[u],
                              arrowprops={"arrowstyle": "->", "color": BLUE,
                                          "lw": .8 + 2.2 * float(intensity[edge]) / max(float(scale), 1e-12),
                                          "shrinkA": 6, "shrinkB": 6, "mutation_scale": 6}, zorder=1)
            if panel == 2 and data["edge_labels"][snapshot, edge] == 1:
                axis.annotate("", xy=positions[v], xytext=positions[u],
                              arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.15, "linestyle": "dashed",
                                          "shrinkA": 6, "shrinkB": 6, "mutation_scale": 5}, zorder=2)
        for node in nodes:
            x, y = positions[node]
            failed = data["labels"][snapshot, node] == 1
            color = ORANGE if panel == 0 and failed else "#EEF0F2" if panel == 0 else cmap(float(probability[node]))
            marker = "*" if node == source else "o"
            axis.scatter(x, y, s=100 if node == source else 55, marker=marker, c=[color],
                         edgecolor=ORANGE if failed and panel != 0 else "#737B83", linewidth=1.3 if failed else .5, zorder=4)
            if panel != 0 and probability[node] >= threshold:
                axis.scatter(x, y, s=15, marker="+", c="#202830", linewidth=.7, zorder=5)
            axis.annotate(str(node), (x, y), xytext=(6, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=5.5, zorder=6)
        axis.set_xlim(-.35, args.case_hops + .4)
        ys = [y for _, y in positions.values()]
        axis.set_ylim(min(ys) - .6, max(ys) + .85)
        axis.axis("off")
        axis.set_title(("Simulator failures", "Saved node predictions", "Beam paths and simulator edges")[panel], fontsize=7, pad=12)
        axis.text(-.05, 1.06, "abc"[panel], transform=axis.transAxes, fontsize=8, weight="bold")
    fig.suptitle(f"{data['name']} | simulator case; query day {data['prediction_days'][snapshot]}, source {source}, seed 17", fontsize=8)
    legend = [Line2D([], [], marker="*", linestyle="none", markerfacecolor="#EEF0F2", markeredgecolor="#737B83", label="Selected observed source"),
              Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=ORANGE, label="Simulator failure"),
              Line2D([], [], marker="+", linestyle="none", color="#202830", label=f"Predicted positive (threshold={threshold:.3f})"),
              Line2D([], [], color=BLUE, lw=2, label="Top-K path (width: intensity)"),
              Line2D([], [], color=ORANGE, linestyle="dashed", label="Simulator successful transmission")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .04), ncol=2, fontsize=6)
    fig.text(.5, .012, "Fixed first eligible test snapshot; all displayed failures and misses retained. Layout is schematic; no real causal claim.", ha="center", fontsize=6)
    fig.subplots_adjust(left=.035, right=.90, top=.85, bottom=.24, wspace=.30)
    color_axis = fig.add_axes([.927, .28, .012, .49])
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=color_axis, label="Node probability")
    save_figure(plt, fig, output, stem)
    captions.append(f"**{stem}.** Fixed simulator-only case: first eligible test snapshot, index {snapshot}, query day {int(data['prediction_days'][snapshot])}, target day {int(data['target_days'][snapshot])}, lowest-index observed source {source}, full model seed 17. All {len(nodes)} nodes and {len(edges)} observed edges in the source-directed {args.case_hops}-hop induced graph are retained. (a) Orange nodes are independent simulator failures. (b) Fill is saved node probability; orange rings retain failed nodes, while plus marks identify predictions above the validation-selected threshold {threshold:.6g}. Failed nodes without plus marks are visible misses. (c) Blue edges belong to the top {args.path_top_k} cycle-free beam candidates (beam width {args.beam_width}, max hops {args.case_hops}); line width increases with saved model propagation intensity, which can exceed one and is not a probability. Dashed orange edges are simulator-successful transmissions, including those absent from predicted paths. Paths rank the unnormalized product of edge intensities, computed in log space. No model parameters or inferred gates are reconstructed from this plot. The layout is deterministic and has no geographic/economic-distance meaning. The simulated case cannot validate real organisational causal paths; ICKG-Weak has no labelled paths and receives no causal case figure. See source_data/{stem}_nodes.csv, _edges.csv and _paths.csv.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/formal")
    parser.add_argument("--data-root", default="data/prepared")
    parser.add_argument("--output", default="reports/supplement")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--case-hops", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--path-top-k", type=int, default=5)
    args = parser.parse_args()
    if args.min_support < 4 or args.case_hops < 1 or args.beam_width < 1 or args.path_top_k < 1:
        parser.error("minimum support >=4 and positive case/beam/path counts are required")
    manifest = {"script": relative_name(__file__), "script_sha256": file_hash(__file__),
                "expected_paired_seeds": list(SEEDS), "minimum_support": args.min_support,
                "minimum_positives": 2, "minimum_negatives": 2,
                "statistics": "paired seed differences; five-seed mean and sample SD only with all five valid pairs; no p-values",
                "datasets": [], "runs": [], "skipped": []}
    files = sorted(Path(args.data_root).glob("*/dataset.npz"))
    if not files:
        raise FileNotFoundError("No prepared dataset.npz files found")
    datasets = {path.parent.name: read_dataset(path, manifest) for path in files}
    runs = read_runs(args.results, datasets, manifest)
    output = Path(args.output)
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    rows, metrics, summaries = collect_strata(datasets, runs, args.min_support, manifest)
    write_csv(source_dir / "strata_rows.csv", rows,
              ["dataset", "seed", "dataset_sha256", "snapshot_index", "node_index", "prediction_day", "target_day", "label",
               "full_probability", "no_stm_probability", "has_context", "newest_context_slot", "newest_age_days",
               "newest_confidence_proxy", "in_degree", "age_stratum", "confidence_stratum", "indegree_stratum"])
    write_csv(source_dir / "strata_metrics.csv", metrics,
              ["dataset", "seed", "kind", "stratum", "dataset_sha256", "n", "n_positive", "n_negative",
               "full_auprc", "no_stm_auprc", "difference", "na_reason"])
    write_csv(source_dir / "strata_summary.csv", summaries,
              ["dataset", "kind", "stratum", "n", "n_positive", "n_negative", "available_pair_seeds", "valid_pair_seeds",
               "n_valid_pairs", "all_five_pairs_valid", "mean_difference", "sample_sd_difference", "status"])
    plt = plotting_setup()
    captions = ["# Supplemental figure captions", "", "All values come from saved held-out predictions. Source data and hashes are retained; manuscript benchmark numbers are not inserted.", ""]
    plot_strata(plt, datasets, metrics, summaries, output, captions)
    if "SIM-Semi" in datasets and ("SIM-Semi", "full", 17) in runs:
        plot_case(plt, datasets["SIM-Semi"], runs["SIM-Semi", "full", 17], output, source_dir, args, manifest, captions)
    else:
        manifest["skipped"].append({"dataset": "SIM-Semi", "figure": "simulator_case", "reason": "full seed17 predictions unavailable"})
    for name, data in datasets.items():
        if not (data["edge_labels"] >= 0).any():
            manifest["skipped"].append({"dataset": name, "figure": "causal_case", "reason": "no propagation edge ground truth; no case generated"})
    (output / "captions.md").write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    manifest["outputs"] = [{"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
                            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "input_manifest.json"]
    json_write(output / "input_manifest.json", manifest)
    print(f"Supplement figures saved to {output}; {len(rows)} actual paired node-time rows, {len(metrics)} stratum/seed estimates")


if __name__ == "__main__":
    main()
