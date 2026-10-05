"""Reproducible temporal training; validation selects checkpoint and thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from .metrics import binary_metrics, path_metrics, ranking_metrics, threshold_scope_metrics, tune_threshold

STATIC = ("node_features", "node_type", "src", "dst", "edge_type", "edge_dependency", "edge_features")
DYNAMIC = ("resilience", "signals", "delta", "confidence", "signal_mask", "source_mask")
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
BASELINES = ("text_mlp", "graph_only", "concat")
EXPERIMENT_MODES = ("full", "text_mlp", "graph_only", "concat", "no_stm", "no_ltm",
                    "no_time", "no_resgate", "no_overload", "avg_fusion", "scalar_gate",
                    "no_align", "no_path")


class IndependentBaseline(nn.Module):
    """Standalone classifiers, without DM-ROR load/buffer/transport machinery.

    Graph-only uses static structure and therefore predicts the same node risk
    at every time. Text-MLP uses masked mean frozen text embeddings. Concat
    joins these two representations. Their edge scores are an explicitly
    disclosed node-risk/dependency heuristic, never trained propagation.
    """
    def __init__(self, node_dim, signal_dim, num_node_types, num_relations,
                 hidden=64, num_layers=2, dropout=0.1, mode="concat", **unused):
        super().__init__()
        from .model import HeterogeneousLayer
        self.mode = mode
        self.hidden = hidden
        self.node_projection = nn.Linear(node_dim, hidden)
        self.type_embedding = nn.Embedding(num_node_types, hidden)
        self.graph_layers = nn.ModuleList([HeterogeneousLayer(hidden, num_relations, dropout)
                                          for _ in range(num_layers)]) if mode != "text_mlp" else nn.ModuleList()
        self.text_projection = nn.Sequential(nn.Linear(signal_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        input_dim = hidden * 2 if mode == "concat" else hidden
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, batch):
        is_batch = batch["signals"].ndim == 4
        if not is_batch:
            batch = dict(batch)
            for key in DYNAMIC:
                batch[key] = batch[key].unsqueeze(0)
        count, nodes = batch["signals"].shape[:2]
        graph = batch["node_features"].new_zeros(nodes, self.hidden)
        if self.mode != "text_mlp":
            graph = self.node_projection(batch["node_features"]) + self.type_embedding(batch["node_type"])
            for layer in self.graph_layers:
                graph = layer(graph, batch["src"], batch["dst"], batch["edge_type"], batch["edge_dependency"])
        graph = graph.unsqueeze(0).expand(count, -1, -1)
        mask = batch["signal_mask"].unsqueeze(-1)
        observed = torch.where(mask, batch["signals"], torch.zeros_like(batch["signals"]))
        pooled = observed.sum(2) / mask.sum(2).clamp_min(1)
        text = self.text_projection(pooled)
        text = text * batch["signal_mask"].any(-1, keepdim=True)
        representation = text if self.mode == "text_mlp" else graph if self.mode == "graph_only" else torch.cat((graph, text), -1)
        logits = self.classifier(representation).squeeze(-1)
        probability = torch.sigmoid(logits)
        edge = probability[:, batch["src"]] * probability[:, batch["dst"]] * batch["edge_dependency"].clamp(0, 1)
        result = {"node_logits": logits, "node_prob": probability, "edge_prob": edge,
                  "edge_score": edge, "m_short": text, "h_long": graph,
                  "context_available": torch.zeros_like(logits, dtype=torch.bool)}
        return result if is_batch else {key: value[0] for key, value in result.items()}


def temporal_loss(outputs, batch, model, node_weight, args):
    """Average per-time objectives; InfoNCE never treats other times as negatives."""
    from .losses import multitask_loss
    baseline = args.mode in BASELINES
    lambda_align = 0.0 if baseline or args.mode == "no_align" else args.lambda_align
    lambda_path = 0.0 if baseline or args.mode == "no_path" else args.lambda_path
    rows = []
    for index in range(batch["labels"].shape[0]):
        single = {key: value[index] for key, value in outputs.items()}
        rows.append(multitask_loss(single, batch["labels"][index],
                     node_mask=batch["labels"][index] >= 0,
                     edge_labels=batch["edge_labels"][index],
                     edge_mask=batch["edge_labels"][index] >= 0,
                     lambda_align=lambda_align, lambda_path=lambda_path,
                     temperature=args.temperature, pos_weight=node_weight)["total"])
    loss = torch.stack(rows).mean()
    if args.lambda_reg:
        loss = loss + args.lambda_reg * sum(p.square().sum() for p in model.parameters())
    return loss


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    import sklearn
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "sklearn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "deterministic_algorithms": "enabled_warn_only", "time_utc": datetime.now(timezone.utc).isoformat()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_scaler(data, train_idx, output):
    """Only observed train signals define text scaling; static nodes are transductive."""
    features = data["node_features"].astype(np.float32)
    node_mean, node_std = features.mean(0), features.std(0)
    node_std = np.where(node_std > 1e-6, node_std, 1.0)
    observed = data["signals"][train_idx][data["signal_mask"][train_idx].astype(bool)]
    if len(observed):
        signal_mean, signal_std = observed.mean(0), observed.std(0)
    else:
        signal_mean = np.zeros(data["signals"].shape[-1], np.float32)
        signal_std = np.ones_like(signal_mean)
    signal_std = np.where(signal_std > 1e-6, signal_std, 1.0)
    data["node_features"] = ((features - node_mean) / node_std).astype(np.float32)
    data["signals"] = ((data["signals"] - signal_mean) / signal_std).astype(np.float32)
    data["signals"] *= data["signal_mask"][..., None]
    np.savez_compressed(Path(output) / "scaler.npz", node_mean=node_mean, node_std=node_std,
                        signal_mean=signal_mean, signal_std=signal_std)
    return {"signal_fit": "observed train-split signals only", "node_fit": "static graph nodes (transductive)",
            "resilience_confidence_delta": "physical scale preserved"}


def validate_dataset(data):
    required = set(STATIC + DYNAMIC + ("labels", "edge_labels", "split", "prediction_days", "target_days"))
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing dataset fields: {sorted(missing)}")
    t, n = data["labels"].shape
    e = len(data["src"])
    if data["node_features"].shape[0] != n or data["signals"].shape[:2] != (t, n):
        raise ValueError("node and time dimensions disagree")
    if data["edge_labels"].shape != (t, e):
        raise ValueError("edge_labels must be [T,E]")
    if not np.isin(data["labels"], [-1, 0, 1]).all() or not np.isin(data["edge_labels"], [-1, 0, 1]).all():
        raise ValueError("Labels must be -1 (unknown), 0, or 1")
    if not np.isin(data["split"], [-1, 0, 1, 2]).all():
        raise ValueError("Split must be -1 (purged), 0, 1, or 2")
    for key in STATIC + DYNAMIC:
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite values in {key}")
    splits = [np.flatnonzero(data["split"] == i) for i in range(3)]
    if not all(len(i) for i in splits):
        raise ValueError("train, validation, and test splits must all be nonempty")
    if max(splits[0]) >= min(splits[1]) or max(splits[1]) >= min(splits[2]):
        raise ValueError("Temporal splits must be strictly ordered")
    # Labels ending after a later prediction window must have been purged.
    dates_p, dates_t = data["prediction_days"], data["target_days"]
    if np.any(dates_t[splits[0]] >= min(dates_p[splits[1]])):
        raise ValueError("Training target window overlaps validation predictions; purge boundary windows")
    if np.any(dates_t[splits[1]] >= min(dates_p[splits[2]])):
        raise ValueError("Validation target window overlaps test predictions; purge boundary windows")
    return splits


def make_tensors(data, device):
    integer = {"node_type", "src", "dst", "edge_type"}
    boolean = {"signal_mask", "source_mask"}
    result = {}
    for key in STATIC + DYNAMIC:
        dtype = torch.long if key in integer else torch.bool if key in boolean else torch.float32
        result[key] = torch.as_tensor(data[key], dtype=dtype, device=device)
    for key in ("labels", "edge_labels"):
        result[key] = torch.as_tensor(data[key], dtype=torch.float32, device=device)
    return result


def batch_at(tensors, indices):
    return {key: value if key in STATIC else value[indices] for key, value in tensors.items()}


@torch.no_grad()
def predict(model, tensors, indices, batch_size):
    model.eval()
    predictions = {"node_prob": [], "edge_prob": [], "edge_score": []}
    for start in range(0, len(indices), batch_size):
        out = model(batch_at(tensors, indices[start:start + batch_size]))
        for key in predictions:
            predictions[key].append(out[key].detach().cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in predictions.items()}


def class_weight(labels, device):
    known = labels[labels >= 0]
    positive, negative = (known == 1).sum(), (known == 0).sum()
    weight = float(negative / max(positive, 1))
    return torch.tensor(max(min(weight, 50.0), 0.02), device=device), {
        "negative": int(negative), "positive": int(positive), "pos_weight": max(min(weight, 50.0), 0.02)}


def evaluate_split(data, indices, predictions, node_threshold, edge_threshold, args):
    from .paths import beam_search_paths
    node = binary_metrics(data["labels"][indices], predictions["node_prob"], node_threshold)
    scope = ranking_metrics(data["labels"][indices], predictions["node_prob"], args.top_k)
    scope.update(threshold_scope_metrics(data["labels"][indices], predictions["node_prob"], node_threshold))
    node["recall_at_k"] = scope["recall_at_k"]
    node["k"] = scope["k"]
    edge = binary_metrics(data["edge_labels"][indices], predictions["edge_prob"], edge_threshold)
    source = data.get("path_source_mask", data["source_mask"])[indices]
    path = path_metrics(data["src"], data["dst"], data["edge_labels"][indices],
                        predictions["edge_score"], source, beam_search_paths,
                        predictions["node_prob"], args.beam_width, args.max_hops, args.path_top_k)
    return {"node": node, "scope": scope, "edge": edge, "path": path}


def train(args):
    from .model import DMROR
    from .losses import multitask_loss
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed run already exists: {output}; use --overwrite intentionally")
    dataset_path = Path(args.dataset).resolve()
    set_seed(args.seed, args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly select --device cpu")
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    train_idx, val_idx, test_idx = validate_dataset(data)
    metadata_path = dataset_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"status": "missing"}
    config = {**vars(args), "dataset": str(dataset_path), "dataset_sha256": sha256(dataset_path),
              "implementation_sha256": {file.name: sha256(file) for file in sorted(Path(__file__).parent.glob("*.py"))},
              "tensor_dimensions": {"node_dim": data["node_features"].shape[-1], "signal_dim": data["signals"].shape[-1],
                                    "num_node_types": int(data["node_type"].max()) + 1,
                                    "num_relations": int(data["edge_type"].max()) + 1},
              "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx),
                              "purged": int((data["split"] == -1).sum())},
              "model_selection": "validation AUPRC", "threshold_selection": "validation macro F1 for nodes; positive F1 for edges",
              "dataset_metadata": metadata, "environment": environment()}
    write_json(output / "config.json", config)
    if not (data["labels"][train_idx] >= 0).any():
        result = {"status": "skipped_no_train_node_labels", "dataset": dataset_path.parent.name,
                  "mode": args.mode, "seed": args.seed, "test": None, "config": config,
                  "runtime_seconds": time.perf_counter() - started}
        write_json(output / "result.json", result)
        return result
    config["scaler"] = fit_scaler(data, train_idx, output)
    tensors = make_tensors(data, device)
    model_class = IndependentBaseline if args.mode in BASELINES else DMROR
    model_mode = "full" if args.mode in ("no_align", "no_path") else args.mode
    model = model_class(data["node_features"].shape[-1], data["signals"].shape[-1],
                  int(data["node_type"].max()) + 1, int(data["edge_type"].max()) + 1,
                  hidden=args.hidden, num_layers=args.layers, dropout=args.dropout, mode=model_mode,
                  time_decay=args.time_decay, beta_penalty=args.beta_penalty).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    node_weight, node_balance = class_weight(data["labels"][train_idx], device)
    config["node_class_balance_train_only"] = node_balance
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    config["baseline_path_policy"] = ("untrained heuristic p(source)*p(destination)*dependency; not causal propagation"
                                      if args.mode in BASELINES else "trained DM-ROR propagation intensity")
    write_json(output / "config.json", config)
    best_score, best_epoch, stale, history = -float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        losses_epoch = []
        for start in range(0, len(order), args.batch_size):
            batch = batch_at(tensors, order[start:start + args.batch_size])
            outputs = model(batch)
            loss = temporal_loss(outputs, batch, model, node_weight, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses_epoch.append(float(loss.detach()))
        validation_prediction = predict(model, tensors, val_idx, args.batch_size)
        validation_metric = binary_metrics(data["labels"][val_idx], validation_prediction["node_prob"])
        score = validation_metric["auprc"]
        if score is None:
            raise ValueError("Validation split has no positive node labels; cannot select by AUPRC")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses_epoch)),
               "validation_auprc": score, "validation_auc": validation_metric["auc"]}
        history.append(row)
        if score > best_score + args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_auprc": score,
                        "config": config}, output / "checkpoint.pt")
        else:
            stale += 1
        write_json(output / "history.json", history)
        if epoch == 1 or epoch % 5 == 0 or stale >= args.patience:
            print(f"{dataset_path.parent.name} {args.mode} seed={args.seed} epoch={epoch} "
                  f"loss={row['train_loss']:.4f} val_auprc={score:.4f} best={best_score:.4f}", flush=True)
        if stale >= args.patience:
            break
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_prediction = predict(model, tensors, val_idx, args.batch_size)
    test_prediction = predict(model, tensors, test_idx, args.batch_size)
    node_threshold = tune_threshold(data["labels"][val_idx], val_prediction["node_prob"])
    edge_threshold = tune_threshold(data["edge_labels"][val_idx], val_prediction["edge_prob"], "positive_f1")
    results = {"status": "completed", "dataset": dataset_path.parent.name, "mode": args.mode,
               "seed": args.seed, "best_epoch": best_epoch, "epochs_run": len(history),
               "best_validation_auprc": best_score, "config": config,
               "validation": evaluate_split(data, val_idx, val_prediction, node_threshold, edge_threshold, args),
               "test": evaluate_split(data, test_idx, test_prediction, node_threshold, edge_threshold, args),
               "runtime_seconds": time.perf_counter() - started,
               "scientific_scope": "Metrics inherit the dataset provenance: simulated labels demonstrate implementation only; weak public labels do not establish causal propagation."}
    arrays = {"validation_indices": val_idx, "test_indices": test_idx,
              "validation_labels": data["labels"][val_idx], "test_labels": data["labels"][test_idx],
              "test_edge_labels": data["edge_labels"][test_idx],
              "prediction_days": data["prediction_days"][test_idx], "target_days": data["target_days"][test_idx]}
    arrays.update({"validation_" + k: v for k, v in val_prediction.items()})
    arrays.update({"test_" + k: v for k, v in test_prediction.items()})
    np.savez_compressed(output / "predictions.npz", **arrays)
    write_json(output / "result.json", results)
    print(f"completed {output} test_auprc={results['test']['node']['auprc']}", flush=True)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="full", choices=EXPERIMENT_MODES)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-align", type=float, default=0.05)
    p.add_argument("--lambda-path", type=float, default=0.1)
    p.add_argument("--lambda-reg", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--time-decay", type=float, default=0.05)
    p.add_argument("--beta-penalty", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=10)
    p.add_argument("--max-hops", type=int, default=4)
    p.add_argument("--path-top-k", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch-size must be positive")
    try:
        train(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                              "config": vars(args)})
        raise


if __name__ == "__main__":
    main()
