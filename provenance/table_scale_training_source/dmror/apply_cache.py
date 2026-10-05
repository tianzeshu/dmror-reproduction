"""Reapply audited frozen embeddings without model weights or a GPU.

Cache matching is by exact text, not by entity or dataset-specific signal IDs.
Only signals and encoder/statistics metadata change; labels and graph arrays
are preserved. Unknown evidence fails before any dataset is replaced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .data import audit_dataset, sha256, write_json


def _records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


def _identifier(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def load_cache(cache_dir):
    """Validate cache hash, row identities and projection/encoder dimensions."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "embedding_cache.npz"
    provenance = json.loads((cache_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("encoder") != "frozen_local_LLM":
        raise ValueError("Cache provenance must identify a frozen_local_LLM encoder")
    if provenance.get("cache_sha256") != sha256(cache_path):
        raise ValueError("Cache SHA-256 differs from provenance; refusing stale or changed vectors")
    with np.load(cache_path, allow_pickle=False) as archive:
        if "embeddings" not in archive or "projection" not in archive:
            raise ValueError("Cache requires embeddings and projection arrays")
        embeddings, projection = archive["embeddings"].copy(), archive["projection"].copy()
    if embeddings.ndim != 2 or projection.ndim != 2 or not embeddings.size:
        raise ValueError("Cache embeddings/projection must be nonempty 2-D arrays")
    if embeddings.dtype != np.float32 or projection.dtype != np.float32:
        raise ValueError("Portable cache arrays must be float32")
    if not np.isfinite(embeddings).all() or not np.isfinite(projection).all():
        raise ValueError("Cache contains nonfinite features")
    rows, dim = embeddings.shape
    if (projection.shape != (provenance.get("original_hidden_dim"), dim)
            or provenance.get("projection_dim") != dim
            or provenance.get("unique_evidence_segments") != rows):
        raise ValueError("Cache shapes differ from encoder provenance")
    row_text, text_index = {}, {}
    for record in _records(cache_dir / "texts.jsonl"):
        cache_id = _identifier(record.get("cache_id"), "cache_id")
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Each cache row requires nonempty exact text")
        if cache_id in row_text or text in text_index:
            raise ValueError("Cache IDs and exact texts must be unique")
        row_text[cache_id], text_index[text] = text, cache_id
    if set(row_text) != set(range(rows)):
        raise ValueError("Cache text IDs must cover every embedding row exactly once")
    return embeddings, text_index, provenance


def _prepare(dataset, embeddings, text_index, provenance):
    dataset = Path(dataset)
    npz_path, metadata_path = dataset / "dataset.npz", dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    before_sha = sha256(npz_path)
    expected_sha = metadata.get("statistics", {}).get("sha256")
    if expected_sha is not None and expected_sha != before_sha:
        raise ValueError(f"{dataset.name}: dataset SHA-256 differs from metadata")
    audit_dataset(npz_path)
    lookup = {}
    for record in _records(dataset / "signals.jsonl"):
        signal_id = _identifier(record.get("signal_id"), "signal_id")
        if signal_id in lookup:
            raise ValueError(f"{dataset.name}: duplicate signal_id {signal_id}")
        text = record.get("text")
        if not isinstance(text, str) or text not in text_index:
            raise ValueError(f"{dataset.name}: signal_id {signal_id} has text absent from frozen cache; encode new evidence separately")
        lookup[signal_id] = text_index[text]
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    indices, mask = data["memory_indices"], data["signal_mask"]
    if (indices.ndim != 3 or mask.shape != indices.shape
            or data["signals"].ndim != 4 or data["signals"].shape[:3] != indices.shape):
        raise ValueError(f"{dataset.name}: memory_indices, signal_mask and signals shapes differ")
    if not np.issubdtype(indices.dtype, np.integer) or not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{dataset.name}: memory IDs must be integers and mask must be binary")
    active = mask.astype(bool)
    active_ids = np.unique(indices[active])
    absent = [int(value) for value in active_ids if int(value) not in lookup]
    if absent:
        raise ValueError(f"{dataset.name}: active memory IDs absent from signals.jsonl: {absent[:8]}")
    # Inactive slots remain zero even if an input retains a stale memory ID.
    # No labels, entity attributes, timestamps or confidences enter encoding.
    result = np.zeros((*indices.shape, embeddings.shape[1]), np.float32)
    for signal_id in active_ids:
        result[active & (indices == signal_id)] = embeddings[lookup[int(signal_id)]]
    data["signals"] = result
    # Exact provenance equality lets relocated training checkpoints verify the
    # same encoder at inference. Application audit lives outside that entry.
    metadata["text_encoder"] = dict(provenance)
    metadata["cache_application"] = {
        "method": "exact_text_then_dataset_memory_indices",
        "input_dataset_sha256": before_sha,
        "signal_records": len(lookup),
        "active_signal_ids": len(active_ids),
        "active_memory_slots": int(active.sum()),
        "unique_cache_rows_referenced": len({lookup[int(value)] for value in active_ids}),
        "padding_slots_zeroed": int((~active).sum()),
    }
    return dataset, data, metadata


def _write_npz(handle, data):
    """Use NumPy's NPZ format with fixed Unix creator bytes on every OS.

    NumPy's default ZIP creator-system field differs on Windows/Linux even
    when all arrays and compressed member bytes are identical. Pinning that
    non-data field reproduces the original Linux archive SHA on Windows too.
    """
    with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for key, value in data.items():
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)


def apply_cache(paths, cache_dir):
    """Validate all paths before committing any, then replace each NPZ atomically."""
    embeddings, text_index, provenance = load_cache(cache_dir)
    paths = [Path(path) for path in paths]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Supply one or more distinct dataset directories")
    prepared = [_prepare(path, embeddings, text_index, provenance) for path in paths]
    cache_texts_sha = sha256(Path(cache_dir) / "texts.jsonl")
    results = []
    for dataset, data, metadata in prepared:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=dataset, suffix=".npz", delete=False) as handle:
                temporary_path = Path(handle.name)
                _write_npz(handle, data)
            statistics = audit_dataset(temporary_path)
            # Retain dataset-specific label counts, provenance and temporal
            # audits rather than discarding them during vector replacement.
            metadata["statistics"] = {**metadata.get("statistics", {}), **statistics}
            metadata["cache_application"]["cache_texts_sha256"] = cache_texts_sha
            os.replace(temporary_path, dataset / "dataset.npz")
            temporary_path = None
            write_json(dataset / "metadata.json", metadata)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        summary = {"dataset": dataset.name, **metadata["cache_application"], **statistics}
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Prepared dataset directories containing NPZ, signals JSONL and metadata")
    parser.add_argument("--cache", default="data/llm_cache")
    args = parser.parse_args()
    apply_cache(args.datasets, args.cache)


if __name__ == "__main__":
    main()
