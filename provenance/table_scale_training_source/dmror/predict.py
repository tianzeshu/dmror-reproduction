"""Export risk nodes, scopes and beam paths from a saved checkpoint and scaler.

Input is a prepared dataset.npz in the documented tensor contract. Text must
already have the same frozen encoder as training. This CLI does not pretend
to extract a graph or calibrate factual resilience from arbitrary raw text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import DMROR
from .paths import beam_search_paths
from .train import BASELINES, IndependentBaseline, make_tensors, predict, set_seed, write_json


def load_run(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("Inference requires a completed training run")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    dimensions = config.get("tensor_dimensions")
    if dimensions:
        node_dim, signal_dim = dimensions["node_dim"], dimensions["signal_dim"]
        node_types, relations = dimensions["num_node_types"], dimensions["num_relations"]
    else:
        # Compatibility for initial smoke checkpoints. Current checkpoints
        # persist dimensions and work after relocating the whole project.
        weights = checkpoint["model"]
        node_dim = weights["node_projection.weight"].shape[-1]
        baseline = config["mode"] in BASELINES
        signal_dim = weights["text_projection.0.weight" if baseline else "signal_projection.0.weight"].shape[-1]
        node_types = weights["type_embedding.weight" if baseline else "node_type_embedding.weight"].shape[0]
        relations = weights["graph_layers.0.relation_weight"].shape[0] if baseline and "graph_layers.0.relation_weight" in weights else weights.get("relation_embedding.weight", torch.empty(1, 1)).shape[0]
    model_class = IndependentBaseline if config["mode"] in BASELINES else DMROR
    mode = "full" if config["mode"] in ("no_align", "no_path") else config["mode"]
    model = model_class(node_dim, signal_dim, node_types, relations, hidden=config["hidden"],
                        num_layers=config["layers"], dropout=config["dropout"], mode=mode,
                        time_decay=config["time_decay"], beta_penalty=config["beta_penalty"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, config, result


def inference(args):
    set_seed(17, args.threads)
    run_dir, output = Path(args.run_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, config, result = load_run(run_dir, args.device)
    metadata_path = Path(args.dataset).parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        original_encoder = config.get("dataset_metadata", {}).get("text_encoder")
        if original_encoder is not None and metadata.get("text_encoder") != original_encoder:
            raise ValueError("Inference text_encoder metadata differs from training; re-encode with the training encoder")
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with np.load(run_dir / "scaler.npz", allow_pickle=False) as scaler:
        if data["node_features"].shape[-1] != len(scaler["node_mean"]) or data["signals"].shape[-1] != len(scaler["signal_mean"]):
            raise ValueError("Inference feature dimensions differ from training encoder")
        data["node_features"] = ((data["node_features"] - scaler["node_mean"]) / scaler["node_std"]).astype(np.float32)
        data["signals"] = ((data["signals"] - scaler["signal_mean"]) / scaler["signal_std"]).astype(np.float32)
        data["signals"] *= data["signal_mask"][..., None]
    # Prediction does not inspect labels. Unlabelled new snapshots are valid.
    shape = data["signals"].shape[:2]
    data.setdefault("labels", np.full(shape, -1, np.float32))
    data.setdefault("edge_labels", np.full((shape[0], len(data["src"])), -1, np.float32))
    tensors = make_tensors(data, args.device)
    if args.split == "all":
        indices = np.arange(shape[0])
    else:
        if "split" not in data:
            raise ValueError("--split test requires split field; use --split all for new snapshots")
        indices = np.flatnonzero(data["split"] == 2)
    if args.indices:
        indices = np.asarray(args.indices, dtype=int)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= shape[0]):
        raise ValueError("No valid inference indices selected")
    predictions = predict(model, tensors, indices, args.batch_size)
    threshold = float(result["validation"]["node"]["threshold"])
    names = data.get("node_names", data.get("node_ids", np.arange(shape[1])))
    def node_label(index):
        return str(names[index])
    records = []
    for offset, time_index in enumerate(indices):
        probabilities = predictions["node_prob"][offset]
        ranked = np.argsort(-probabilities, kind="stable")[:min(args.top_k, len(probabilities))]
        paths = []
        for source in np.flatnonzero(data["source_mask"][time_index]):
            found = beam_search_paths(data["src"], data["dst"], predictions["edge_score"][offset],
                                      int(source), beam_width=args.beam_width, max_hops=args.max_hops,
                                      top_k=args.path_top_k, node_prob=probabilities)
            for path in found:
                path["node_labels"] = [node_label(node) for node in path["nodes"]]
                path["source"] = int(source)
                paths.append(path)
        records.append({"time_index": int(time_index),
                        "prediction_day": str(data.get("prediction_days", np.arange(shape[0]))[time_index]),
                        "node_threshold_validation_selected": threshold,
                        "risk_scope_indices": np.flatnonzero(probabilities >= threshold).tolist(),
                        "top_nodes": [{"index": int(node), "label": node_label(node), "probability": float(probabilities[node])} for node in ranked],
                        "paths": paths})
    np.savez_compressed(output / "predictions.npz", indices=indices, **predictions)
    write_json(output / "risk_predictions.json", {"mode": config["mode"], "scientific_scope": result["scientific_scope"],
                 "path_policy": config.get("baseline_path_policy"), "path_sources": "input observed source_mask; labels never select sources",
                 "threshold_policy": "retained training-validation threshold; no inference-label tuning", "records": records})
    print(f"Saved {len(records)} snapshots to {output}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--path-top-k", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=4)
    inference(parser.parse_args())


if __name__ == "__main__":
    main()
