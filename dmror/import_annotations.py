"""Import reviewer-signed outcome windows into a NEW prepared dataset.

Unannotated cells are UNKNOWN (-1), including missing negatives. Features,
timestamps, encoder provenance and cause/source masks are never inferred from
future outcomes. CSV validation establishes structure, not factual correctness.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data import audit_dataset, sha256, write_json


COMMON_FIELDS = {"query_day", "target_day", "label", "evidence_url",
                 "evidence_span", "reviewer", "verified"}
EPOCH = date(1970, 1, 1)


def _day(value, field):
    """Accept ISO dates or integer days since 1970-01-01; reject fuzzy dates."""
    value = str(value).strip()
    try:
        if value.lstrip("-").isdigit():
            return int(value)
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return (parsed - EPOCH).days
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}: expected YYYY-MM-DD or integer epoch day") from error


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _node_ids(folder, arrays):
    n = len(arrays["node_type"])
    if "node_ids" in arrays:
        if arrays["node_ids"].shape != (n,):
            raise ValueError("NPZ node_ids must be a one-dimensional node mapping")
        ids = [str(value) for value in arrays["node_ids"]]
    else:
        rows = _jsonl(folder / "nodes.jsonl")
        if len(rows) != n or {row["node_index"] for row in rows} != set(range(n)):
            raise ValueError("nodes.jsonl must map each node_index exactly once")
        ids = [str(row["entity_id"]) for row in sorted(rows, key=lambda row: row["node_index"])]
    if len(ids) != n or len(set(ids)) != n or any(not value.strip() for value in ids):
        raise ValueError("Node IDs must be nonempty, unique and cover the NPZ nodes")
    return ids


def _read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path}: missing CSV fields {sorted(required - set(reader.fieldnames or []))}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: duplicate column names")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{number}: malformed CSV row")
            row = {key: value.strip() for key, value in row.items()}
            if not any(row.values()):
                continue
            for field in required:
                if not row[field]:
                    raise ValueError(f"{path}:{number}: {field} must not be blank")
            if row["label"] not in {"0", "1"} or row["verified"] != "1":
                raise ValueError(f"{path}:{number}: label must be 0/1 and verified must be 1")
            row["label"] = int(row["label"])
            row["verified"] = 1
            row["query_day"] = _day(row["query_day"], "query_day")
            row["target_day"] = _day(row["target_day"], "target_day")
            if row["query_day"] >= row["target_day"]:
                raise ValueError(f"{path}:{number}: target_day must be after query_day")
            yield number, row


def _snapshot_index(arrays):
    lookup = {}
    for index, (query, target) in enumerate(zip(arrays["prediction_days"], arrays["target_days"])):
        key = (int(query), int(target))
        if key in lookup:
            raise ValueError("NPZ prediction/target windows must be unique")
        lookup[key] = index
    return lookup


def _window(row, lookup):
    key = (row["query_day"], row["target_day"])
    if key not in lookup:
        raise ValueError(f"Annotation window {key} must exactly match an NPZ query/target window")
    return lookup[key]


def _assign(array, cell, row, seen, description):
    if cell in seen:
        if seen[cell] != row["label"]:
            raise ValueError(f"Conflicting duplicate {description} annotation for {cell}")
        raise ValueError(f"Duplicate {description} annotation for {cell}; consolidate evidence into one row")
    seen[cell] = row["label"]
    array[cell] = row["label"]


def _edge_lookup(folder, arrays, ids, metadata):
    """Map directed entity IDs AND relation type; never collapse parallel edges."""
    names = metadata.get("relation_types")
    if not isinstance(names, list) or not names:
        raise ValueError("Edge annotations require metadata.relation_types")
    lookup = {}
    for index, (src, dst, kind) in enumerate(zip(arrays["src"], arrays["dst"], arrays["edge_type"])):
        if not 0 <= int(kind) < len(names):
            raise ValueError("NPZ edge_type is outside metadata.relation_types")
        key = (ids[int(src)], ids[int(dst)], str(names[int(kind)]))
        lookup.setdefault(key, []).append(index)
    return lookup


def import_annotations(dataset, node_labels, output, edge_labels=None):
    """Validate all records, then publish a fresh dataset directory atomically.

    Node positive rows require event_date inside [query_day,target_day).
    Node negative rows explicitly assert reviewed coverage of the entire exact
    window through verified=1; absence of a CSV row is never a negative.
    """
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a fresh directory")
    if output == dataset or dataset in output.parents:
        raise ValueError("Output must be separate from the source dataset")
    npz = dataset / "dataset.npz"
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    before_sha = sha256(npz)
    expected_sha = metadata.get("statistics", {}).get("sha256")
    if expected_sha is not None and before_sha != expected_sha:
        raise ValueError("Source dataset SHA-256 differs from metadata")
    audit_dataset(npz)
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    ids = _node_ids(dataset, arrays)
    node_index = {value: index for index, value in enumerate(ids)}
    windows = _snapshot_index(arrays)
    # Use an explicitly signed label type even if a source serialized binary
    # labels as uint8/bool, whose dtype cannot represent UNKNOWN (-1).
    arrays["labels"] = np.full(arrays["labels"].shape, -1, np.float32)
    arrays["edge_labels"] = np.full(arrays["edge_labels"].shape, -1, np.float32)
    evidence, node_seen, edge_seen = [], {}, {}
    for number, row in _read_csv(node_labels, COMMON_FIELDS | {"entity_id"}):
        if row["entity_id"] not in node_index:
            raise ValueError(f"Unknown entity_id: {row['entity_id']}")
        ti = _window(row, windows)
        if row["label"] == 1:
            if not row.get("event_date"):
                raise ValueError("Positive node label requires event_date")
            event_day = _day(row["event_date"], "event_date")
            if not row["query_day"] <= event_day < row["target_day"]:
                raise ValueError("Positive event_date must be inside [query_day,target_day)")
            row["event_day"] = event_day
        elif row.get("event_date"):
            raise ValueError("Negative node label must leave event_date blank")
        ni = node_index[row["entity_id"]]
        _assign(arrays["labels"], (ti, ni), row, node_seen, "node")
        evidence.append({"kind": "reviewer_signed_node_outcome_window", "csv_row": number,
                         "snapshot_index": ti, "node_index": ni, **row})
    if edge_labels is not None:
        lookup = _edge_lookup(dataset, arrays, ids, metadata)
        required = COMMON_FIELDS | {"source_id", "target_id", "relation_type"}
        for number, row in _read_csv(edge_labels, required):
            if row["source_id"] not in node_index or row["target_id"] not in node_index:
                raise ValueError("Unknown source_id or target_id in edge annotation")
            ti = _window(row, windows)
            key = (row["source_id"], row["target_id"], row["relation_type"])
            candidates = lookup.get(key, [])
            if len(candidates) != 1:
                raise ValueError(f"Edge triple must uniquely identify one directed relation: {key}")
            ei = candidates[0]
            _assign(arrays["edge_labels"], (ti, ei), row, edge_seen, "edge")
            evidence.append({"kind": "reviewer_signed_edge_outcome_window", "csv_row": number,
                             "snapshot_index": ti, "edge_index": ei, **row})
    if not node_seen and not edge_seen:
        raise ValueError("No reviewed annotations supplied; templates contain no observations")

    # Everything above is read-only; invalid CSV cannot leave a partial output.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".annotations-", dir=output.parent))
    try:
        shutil.copytree(dataset, temporary, dirs_exist_ok=True)
        np.savez_compressed(temporary / "dataset.npz", **arrays)
        stats = audit_dataset(temporary / "dataset.npz")
        stats["label_counts_by_split"] = {
            str(value): {"snapshots": int((arrays["split"] == value).sum()),
                         "positive": int((arrays["labels"][arrays["split"] == value] == 1).sum()),
                         "negative": int((arrays["labels"][arrays["split"] == value] == 0).sum()),
                         "unknown": int((arrays["labels"][arrays["split"] == value] == -1).sum())}
            for value in (-1, 0, 1, 2)}
        metadata["name"] = output.name
        metadata["data_kind"] = "reviewer_signed_outcomes_on_existing_features"
        metadata["is_disruption_gold_standard"] = False
        metadata["generator"] = "dmror.import_annotations"
        metadata["statistics"] = stats
        metadata["label_kind"] = "reviewer_signed_exact_window_outcomes"
        metadata["label_rule"] = {
            "positive": "Reviewer-signed event with event_date in [query_day,target_day)",
            "negative": "Reviewer explicitly signs no event over the ENTIRE exact target window",
            "missing": "UNKNOWN (-1); never inferred from absence of future text",
            "verified": "Reviewer assertion only; importer does not establish factual truth"}
        metadata["annotation_import"] = {
            "source_dataset_sha256": before_sha,
            "source_metadata_sha256": sha256(dataset / "metadata.json"),
            "node_csv_sha256": sha256(node_labels),
            "edge_csv_sha256": sha256(edge_labels) if edge_labels is not None else None,
            "node_annotation_count": len(node_seen), "edge_annotation_count": len(edge_seen),
            "original_label_kind": json.loads((dataset / "metadata.json").read_text(encoding="utf-8")).get("label_kind"),
            "original_label_evidence": "label_evidence.jsonl retained as ORIGINAL proxy provenance, not accepted outcome truth",
            "future_annotations_used_as_features": False,
            "source_mask_policy": "Copied unchanged; no future-label-derived cause inference",
            "fact_verification_performed_by_importer": False}
        metadata["path_ground_truth"] = (
            "Explicit reviewed edge cells only; all other edges unknown. Edge annotations do not establish complete causal path truth."
            if edge_seen else "Unknown: edge_labels all -1; original source masks copied unchanged.")
        metadata["scope_ground_truth"] = "Partial reviewed node outcomes only; incomplete annotation is not a measured full affected set."
        with (temporary / "annotation_evidence.jsonl").open("w", encoding="utf-8") as handle:
            for row in evidence:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        annotations_dir = temporary / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        shutil.copy2(node_labels, annotations_dir / "node_labels.csv")
        if edge_labels is not None:
            shutil.copy2(edge_labels, annotations_dir / "edge_labels.csv")
        write_json(temporary / "metadata.json", metadata)
        write_json(temporary / "audit.json", stats)
        temporary.rename(output)
    finally:
        if temporary.exists():
            if temporary.resolve().parent != output.parent or not temporary.name.startswith(".annotations-"):
                raise RuntimeError("Refusing cleanup outside the allocated annotation workspace")
            shutil.rmtree(temporary)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Existing prepared dataset directory")
    parser.add_argument("--node-labels", required=True, help="Reviewer-signed node CSV")
    parser.add_argument("--edge-labels", help="Optional reviewed directed relation CSV")
    parser.add_argument("--output", required=True, help="Fresh output directory; cannot overwrite")
    args = parser.parse_args()
    result = import_annotations(args.dataset, args.node_labels, args.output, args.edge_labels)
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
