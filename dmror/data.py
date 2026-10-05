"""Portable dataset utilities. No executable pickle payloads are accepted."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def hash_text(text, dim=64):
    result = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+|[\u4e00-\u9fff]", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        result[index] += 1 if digest[4] & 1 else -1
    return result / max(float(np.linalg.norm(result)), 1.0)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        n, t, e = len(data["node_type"]), len(data["split"]), len(data["src"])
        assert data["labels"].shape == (t, n)
        assert data["edge_labels"].shape == (t, e)
        assert data["signals"].shape[:2] == (t, n)
        assert data["signal_mask"].shape == data["signals"].shape[:3]
        assert np.isin(data["labels"], [-1, 0, 1]).all()
        assert np.isin(data["edge_labels"], [-1, 0, 1]).all()
        assert (data["delta"][data["signal_mask"].astype(bool)] > 0).all(), "Signals must strictly precede query"
        assert data["src"].min() >= 0 and data["dst"].min() >= 0
        assert data["src"].max() < n and data["dst"].max() < n
        assert np.isfinite(data["signals"]).all()
        for older, newer in [(0, 1), (1, 2)]:
            left, right = data["split"] == older, data["split"] == newer
            assert left.any() and right.any()
            assert data["target_days"][left].max() < data["prediction_days"][right].min(), "Cross-split target leakage"
        return {"nodes": n, "edges": e, "snapshots": t, "splits": {str(k): int((data['split'] == k).sum()) for k in [-1, 0, 1, 2]}, "known_node_labels": int((data['labels'] >= 0).sum()), "positive_node_labels": int((data['labels'] == 1).sum()), "known_edge_labels": int((data['edge_labels'] >= 0).sum()), "sha256": sha256(path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset), indent=2))
