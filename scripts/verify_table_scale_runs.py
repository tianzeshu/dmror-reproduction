"""Audit frozen scale-protocol runs against actual data and saved predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmror.metrics import binary_metrics, tune_threshold
from dmror.train import sha256, validate_dataset


def audit(protocol_path, results, output):
    protocol_path, results = Path(protocol_path), Path(results)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    data_cache, records, errors = {}, [], []
    for job in protocol["jobs"]:
        run = results / job["name"]
        try:
            for name in ("config.json", "checkpoint.pt", "history.json", "scaler.npz", "predictions.npz", "result.json"):
                if not (run / name).is_file():
                    raise ValueError(f"Missing artifact: {name}")
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            if result["status"] != "completed":
                raise ValueError("Run is not completed")
            digest_input = {key: value for key, value in config.items() if key != "configuration_sha256"}
            digest = hashlib.sha256(json.dumps(digest_input, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            if config["configuration_sha256"] != digest or config.get("protocol_sha256") != protocol_hash:
                raise ValueError("Configuration/protocol digest mismatch")
            for key, value in {**protocol["common"], **job["parameters"]}.items():
                if config.get(key) != value:
                    raise ValueError(f"Frozen parameter mismatch: {key}")
            dataset = ROOT / job["dataset"]
            if job["dataset"] not in data_cache:
                with np.load(dataset, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in ("labels", "edge_labels", "split", "prediction_days", "target_days",
                            "node_features", "node_type", "src", "dst", "edge_type", "edge_dependency", "edge_features",
                            "resilience", "signals", "delta", "confidence", "signal_mask", "source_mask")}
                indices = validate_dataset(data)
                # Keep only evaluation labels after schema/temporal checks.
                data_cache[job["dataset"]] = (sha256(dataset), data["labels"], data["edge_labels"], indices)
                del data
            data_hash, labels, edge_labels, indices = data_cache[job["dataset"]]
            if config["dataset_sha256"] != data_hash:
                raise ValueError("Dataset digest mismatch")
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
            if checkpoint["config"] != config or checkpoint["epoch"] != result["best_epoch"]:
                raise ValueError("Checkpoint identity does not match saved configuration/result")
            if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
                raise ValueError("Nonfinite checkpoint parameter")
            selected_epoch, selected_score = 0, -float("inf")
            for row in history:
                if row["validation_auprc"] > selected_score + config["min_delta"]:
                    selected_epoch, selected_score = row["epoch"], row["validation_auprc"]
            if selected_epoch != result["best_epoch"] or selected_score != result["best_validation_auprc"]:
                raise ValueError("Checkpoint was not selected by recorded validation AUPRC")
            with np.load(run / "predictions.npz", allow_pickle=False) as predictions:
                val, test = indices[1:]
                np.testing.assert_array_equal(predictions["validation_indices"], val)
                np.testing.assert_array_equal(predictions["test_indices"], test)
                np.testing.assert_array_equal(predictions["validation_labels"], labels[val])
                np.testing.assert_array_equal(predictions["test_labels"], labels[test])
                np.testing.assert_array_equal(predictions["test_edge_labels"], edge_labels[test])
                node_threshold = tune_threshold(labels[val], predictions["validation_node_prob"])
                edge_threshold = tune_threshold(edge_labels[val], predictions["validation_edge_prob"], "positive_f1")
                for split_name, split_indices in (("validation", val), ("test", test)):
                    for kind, y, threshold in (("node", labels[split_indices], node_threshold),
                                                ("edge", edge_labels[split_indices], edge_threshold)):
                        recomputed = binary_metrics(y, predictions[split_name + "_" + kind + "_prob"], threshold)
                        for key, value in recomputed.items():
                            saved = result[split_name][kind][key]
                            if value is None:
                                assert saved is None
                            else:
                                assert np.isclose(value, saved, rtol=1e-12, atol=1e-12), (split_name, kind, key)
            records.append({"run": job["name"], "status": "verified", "configuration_sha256": digest,
                            "dataset_sha256": data_hash, "checkpoint_sha256": sha256(run / "checkpoint.pt"),
                            "predictions_sha256": sha256(run / "predictions.npz"),
                            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                            "runtime_seconds": result["runtime_seconds"],
                            "cuda_peak_allocated_bytes": result.get("cuda_peak_allocated_bytes")})
        except Exception as error:
            errors.append({"run": job["name"], "error_type": type(error).__name__, "error": str(error)})
    audit_result = {"status": "passed" if not errors else "failed", "protocol_sha256": protocol_hash,
                    "expected_runs": len(protocol["jobs"]), "verified_runs": len(records),
                    "checks": ["complete trained artifacts", "data/configuration/protocol SHA256",
                               "checkpoint identity and finite weights", "validation-only checkpoint and thresholds",
                               "ordered target-purged temporal splits", "metrics recomputed from saved predictions"],
                    "runs": records, "errors": errors}
    Path(output).write_text(json.dumps(audit_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit_result[key] for key in ("status", "expected_runs", "verified_runs", "errors")}, ensure_ascii=False))
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(ROOT / "protocols/table_scale.json"))
    parser.add_argument("--results", default=str(ROOT / "results/table_scale"))
    parser.add_argument("--output", default=str(ROOT / "results/table_scale/verification.json"))
    arguments = parser.parse_args()
    if not audit(arguments.protocol, arguments.results, arguments.output):
        raise SystemExit(1)
