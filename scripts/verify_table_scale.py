"""Independently audit table-sized SYNTHETIC data, never original SC recovery.

Counts are read from JSONL records and NPZ arrays rather than copied metadata.
The public CLI fixes the manuscript Table 1 counts. Small explicit specifications
are accepted by audit_one only to support corruption tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

TYPES = ("firm", "product", "material", "industry", "region")
START_DAY = (date(2018, 1, 1) - date(1970, 1, 1)).days
END_DAY = (date(2025, 12, 31) - date(1970, 1, 1)).days
SPECS = {
    "SC-Auto-Synthetic": {"entities": [6482, 1126, 438, 76, 42], "relations": 58734,
        "texts": 124680, "signals": 38912, "labels": 8426, "queries": 96},
    "SC-Semi-Synthetic": {"entities": [4935, 842, 316, 54, 37], "relations": 46218,
        "texts": 96440, "signals": 31705, "labels": 6913, "queries": 96},
    "SC-Energy-Synthetic": {"entities": [5714, 973, 512, 68, 51], "relations": 52906,
        "texts": 108375, "signals": 34286, "labels": 7584, "queries": 96},
}
FILES = {"entities": "entities.jsonl", "relations": "relations.jsonl", "texts": "texts.jsonl",
         "signals": "signals.jsonl", "labels": "risk_labels.jsonl", "queries": "queries.jsonl"}


class Audit:
    def __init__(self):
        self.errors = Counter()
        self.examples = {}

    def require(self, condition, code, detail):
        if not condition:
            self.errors[code] += 1
            examples = self.examples.setdefault(code, [])
            if len(examples) < 5:
                examples.append(str(detail))
        return bool(condition)

    def guarded(self, code, operation):
        try:
            return operation()
        except (KeyError, IndexError, TypeError, ValueError, AssertionError, OSError) as error:
            self.require(False, code, f"{type(error).__name__}: {error}")
            return None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def signed_hash(text, dimension):
    """Recompute the disclosed deterministic text feature contract."""
    vector = np.zeros(dimension, np.float32)
    for token in re.findall(r"\w+|[\u4e00-\u9fff]", text.lower()):
        token_hash = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(token_hash[:4], "little") % dimension] += 1 if token_hash[4] & 1 else -1
    return vector / max(float(np.linalg.norm(vector)), 1.0)


def records(path, audit):
    values = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    audit.require(False, "jsonl_blank_line", f"{path.name}:{line_number}")
                    continue
                try:
                    record = json.loads(line)
                    if audit.require(isinstance(record, dict), "jsonl_record", f"{path.name}:{line_number}"):
                        values.append(record)
                except json.JSONDecodeError as error:
                    audit.require(False, "jsonl_parse", f"{path.name}:{line_number}: {error}")
    except OSError as error:
        audit.require(False, "missing_file", f"{path.name}: {error}")
    return values


def sequence_ids(values, key, audit, label):
    seen = set()
    for index, record in enumerate(values):
        value = record.get(key)
        audit.require(integer(value) and value == index, "record_id", f"{label}[{index}].{key}={value}")
        if isinstance(value, (int, str)):
            audit.require(value not in seen, "duplicate_id", f"{label}.{key}={value}")
            seen.add(value)
        audit.require(record.get("synthetic") is True, "synthetic_provenance", f"{label}[{index}]")


def valid_day(record, field, audit, label, date_field="date"):
    day = record.get(field)
    valid = integer(day) and START_DAY <= day <= END_DAY
    audit.require(valid, "timestamp_bounds", f"{label}.{field}={day}")
    if date_field == "date":
        audit.require(isinstance(record.get(date_field), str), "timestamp_date", f"{label}.{date_field} missing")
    if date_field in record and integer(day):
        expected = date.fromordinal(date(1970, 1, 1).toordinal() + day).isoformat()
        audit.require(record[date_field] == expected, "timestamp_date", f"{label}.{date_field}")
    return valid


def check_records(values, specification, audit):
    entities, relations, texts, signals, labels, queries = (values[k] for k in FILES)
    n = len(entities)
    for key in FILES:
        expected = sum(specification["entities"]) if key == "entities" else specification[key]
        audit.require(len(values[key]) == expected, "exact_count", f"{key}: {len(values[key])} != {expected}")
    for key, id_key in [("entities", "entity_id"), ("relations", "relation_id"), ("texts", "text_id"),
                        ("signals", "signal_id"), ("labels", "label_id")]:
        sequence_ids(values[key], id_key, audit, key)

    counts = Counter(record.get("entity_type") for record in entities)
    for kind, expected in zip(TYPES, specification["entities"]):
        audit.require(counts[kind] == expected, "entity_type_count", f"{kind}: {counts[kind]} != {expected}")
    audit.require(set(counts) <= set(TYPES), "entity_type", f"Types: {list(counts)}")

    endpoints, triples, relation_types = set(), set(), {}
    ontology = {0: ("firm", "firm", "supplies"), 1: ("firm", "product", "produces"),
                2: ("material", "product", "input_to"), 3: ("firm", "industry", "member_of"),
                4: ("firm", "region", "located_in")}
    for index, record in enumerate(relations):
        src, dst, relation = record.get("src"), record.get("dst"), record.get("edge_type")
        legal = integer(src) and integer(dst) and 0 <= src < n and 0 <= dst < n
        audit.require(legal and src != dst, "relation_endpoints", f"relation {index}")
        if legal:
            endpoints.update((src, dst))
        audit.require(integer(relation) and relation >= 0, "relation_type", f"relation {index}")
        triple = (src, dst, relation)
        audit.require(triple not in triples, "duplicate_relation", f"relation {index}: {triple}")
        triples.add(triple)
        relation_name = record.get("relation_type")
        audit.require(isinstance(relation_name, str) and bool(relation_name), "relation_type", f"relation {index}")
        if relation in relation_types:
            audit.require(relation_types[relation] == relation_name, "relation_type_mapping", f"relation {index}")
        relation_types[relation] = relation_name
        if legal and relation in ontology:
            head_type, tail_type, expected_name = ontology[relation]
            audit.require(entities[src].get("entity_type") == head_type and entities[dst].get("entity_type") == tail_type
                          and str(relation_name).lower() == expected_name, "relation_ontology", f"relation {index}")
        else:
            audit.require(False, "relation_ontology", f"relation {index}: unsupported type {relation}")
        dependency, features = record.get("dependency"), record.get("features")
        audit.require(isinstance(dependency, (int, float)) and np.isfinite(dependency) and 0 <= dependency <= 1,
                      "relation_dependency", f"relation {index}")
        audit.require(isinstance(features, list) and len(features) == 4 and np.isfinite(features).all(),
                      "relation_features", f"relation {index}")
    audit.require(endpoints == set(range(n)), "graph_entity_coverage", f"covered {len(endpoints)} / {n}")

    text_entities = set()
    for index, record in enumerate(texts):
        entity = record.get("entity_id")
        valid = integer(entity) and 0 <= entity < n
        audit.require(valid, "text_entity_link", f"text {index}")
        if valid:
            text_entities.add(entity)
        valid_day(record, "timestamp_day", audit, f"text {index}")
        audit.require(isinstance(record.get("text"), str) and bool(record["text"].strip()), "text_content", f"text {index}")
        audit.require(record.get("source_kind") == "original_synthetic_template", "text_source", f"text {index}")
    audit.require(text_entities == set(range(n)), "text_entity_coverage", f"covered {len(text_entities)} / {n}")
    text_days = [record.get("timestamp_day") for record in texts if integer(record.get("timestamp_day"))]
    if specification["queries"] == 96 and text_days:
        audit.require(min(text_days) == START_DAY and max(text_days) == END_DAY, "text_time_span", "Texts must actually cover 2018-01-01 through 2025-12-31")

    signal_entities = set()
    for index, record in enumerate(signals):
        entity, text_id = record.get("entity_id"), record.get("text_id")
        valid_entity = integer(entity) and 0 <= entity < n
        audit.require(valid_entity, "signal_entity_link", f"signal {index}")
        if valid_entity:
            signal_entities.add(entity)
        valid_text = integer(text_id) and 0 <= text_id < len(texts)
        audit.require(valid_text, "signal_text_link", f"signal {index}")
        valid_day(record, "timestamp_day", audit, f"signal {index}")
        if valid_text:
            text = texts[text_id]
            audit.require(entity == text.get("entity_id") and record.get("timestamp_day") == text.get("timestamp_day"),
                          "signal_text_consistency", f"signal {index}")
        confidence = record.get("confidence")
        audit.require(isinstance(confidence, (int, float)) and np.isfinite(confidence) and 0 <= confidence <= 1,
                      "signal_confidence", f"signal {index}")
        audit.require(record.get("severity") in {"minor", "moderate", "severe"}, "signal_severity", f"signal {index}")
        audit.require(isinstance(record.get("risk_kind"), str) and bool(record["risk_kind"]), "signal_risk_kind", f"signal {index}")

    query_days = []
    for index, record in enumerate(queries):
        audit.require(record.get("snapshot") == index, "query_snapshot", f"query {index}")
        valid_day(record, "prediction_day", audit, f"query {index}")
        valid_day(record, "target_day", audit, f"query {index}", date_field="target_date")
        audit.require(record.get("target_day") == record.get("prediction_day", 0) + 30, "forecast_horizon", f"query {index}")
        audit.require(record.get("split") in {-1, 0, 1, 2}, "query_split", f"query {index}")
        query_days.append(record.get("prediction_day"))
    if query_days:
        audit.require(all(a < b for a, b in zip(query_days, query_days[1:])), "query_order", "Queries must be strictly chronological")
        if specification["queries"] == 96:
            monthly = [(date(year, month, 1) - date(1970, 1, 1)).days for year in range(2018, 2026) for month in range(1, 13)]
            audit.require(query_days == monthly, "query_calendar", "Expected monthly first-day queries for 2018--2025")

    positions, labeled_entities, labeled_types = set(), set(), Counter()
    for index, record in enumerate(labels):
        entity, snapshot, label = record.get("entity_id"), record.get("snapshot"), record.get("label")
        legal = integer(entity) and 0 <= entity < n and integer(snapshot) and 0 <= snapshot < len(queries)
        audit.require(legal, "label_position", f"label {index}")
        position = (snapshot, entity)
        audit.require(position not in positions, "duplicate_label_position", f"label {index}: {position}")
        positions.add(position)
        audit.require(integer(label) and label in {0, 1}, "label_value", f"label {index}")
        audit.require(record.get("label_kind") == "independent_delayed_discrete_cascade", "label_provenance", f"label {index}")
        if legal:
            query = queries[snapshot]
            labeled_entities.add(entity)
            labeled_types[entities[entity].get("entity_type")] += 1
            audit.require(query["split"] != -1, "purged_label", f"label {index}")
            audit.require(record.get("prediction_day") == query["prediction_day"] and record.get("target_day") == query["target_day"],
                          "label_query_time", f"label {index}")
    audit.require(labeled_entities == set(range(n)), "label_entity_coverage", f"covered {len(labeled_entities)} / {n}")
    return {"entity_types": dict(counts), "graph_entities": len(endpoints), "text_entities": len(text_entities),
            "signal_entities": len(signal_entities), "label_entities": len(labeled_entities), "label_entity_types": dict(labeled_types),
            "signal_entity_types_covered": dict(Counter(entities[i]["entity_type"] for i in signal_entities)),
            "text_min_day": min(text_days) if text_days else None, "text_max_day": max(text_days) if text_days else None,
            "source_kind": dict(Counter(r.get("source_kind") for r in texts)),
            "label_kind": dict(Counter(r.get("label_kind") for r in labels)), "label_values": dict(Counter(r.get("label") for r in labels)),
            "signal_risk_kind": dict(Counter(r.get("risk_kind") for r in signals))}


def check_npz(path, values, audit):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n, e, t = len(values["entities"]), len(values["relations"]), len(values["queries"])
    required = {"node_features", "node_type", "src", "dst", "edge_type", "edge_dependency", "edge_features",
                "resilience", "signals", "signal_mask", "delta", "confidence", "memory_indices", "labels",
                "edge_labels", "source_mask", "split", "prediction_days", "target_days", "simulation_labels"}
    if not audit.require(required <= data.keys(), "npz_keys", f"Missing {sorted(required - data.keys())}"):
        return None
    for key, array in data.items():
        audit.require(array.dtype.kind not in "OUS", "npz_dtype", f"{key}: {array.dtype}")
        if np.issubdtype(array.dtype, np.number):
            audit.require(np.isfinite(array).all(), "npz_finite", key)
    shapes = {"node_type": (n,), "src": (e,), "dst": (e,), "edge_type": (e,), "edge_dependency": (e,),
              "edge_features": (e, 4), "resilience": (t, n, 5), "labels": (t, n), "edge_labels": (t, e),
              "source_mask": (t, n), "split": (t,), "prediction_days": (t,), "target_days": (t,)}
    if not all(audit.require(data[k].shape == v, "npz_shape", f"{k}: {data[k].shape} != {v}") for k, v in shapes.items()):
        return None
    for key, shape in (("edge_features_observed_mask", (e, 4)), ("edge_dependency_observed_mask", (e,)),
                       ("resilience_observed_mask", (t, n, 5)), ("path_source_mask", (t, n))):
        if key in data:
            audit.require(data[key].shape == shape and np.isin(data[key], [0, 1]).all(), "npz_observation_mask", key)
    audit.require(data["node_features"].ndim == 2 and data["node_features"].shape[0] == n, "npz_shape", "node_features")
    if not audit.require(data["signals"].ndim == 4 and data["signals"].shape[:2] == (t, n), "npz_shape", "signals"):
        return None
    memory_shape = data["signals"].shape[:3]
    if not all(audit.require(data[k].shape == memory_shape, "npz_shape", k) for k in ("signal_mask", "delta", "confidence", "memory_indices")):
        return None
    for key in ("node_type", "src", "dst", "edge_type", "memory_indices", "split", "prediction_days", "target_days"):
        audit.require(np.issubdtype(data[key].dtype, np.integer), "npz_integer_index", key)
    expected_types = [TYPES.index(record["entity_type"]) for record in values["entities"]]
    audit.require(np.array_equal(data["node_type"], expected_types), "npz_entity_types", "NPZ and entities.jsonl differ")
    for key, field in (("src", "src"), ("dst", "dst"), ("edge_type", "edge_type")):
        audit.require(np.array_equal(data[key], [record[field] for record in values["relations"]]), "npz_relations", key)
    for key, field in (("edge_dependency", "dependency"), ("edge_features", "features")):
        audit.require(np.allclose(data[key], np.asarray([record[field] for record in values["relations"]]), atol=1e-6, rtol=1e-6), "npz_relation_features", key)
    for key in ("split", "prediction_days", "target_days"):
        field = {"prediction_days": "prediction_day", "target_days": "target_day"}.get(key, key)
        audit.require(np.array_equal(data[key], [record[field] for record in values["queries"]]), "npz_queries", key)
    audit.require(np.isin(data["labels"], [-1, 0, 1]).all() and np.isin(data["edge_labels"], [-1, 0, 1]).all(), "npz_label_values", "Unknown labels must be -1")
    audit.require(np.isin(data["source_mask"], [0, 1]).all(), "npz_mask", "source_mask")
    reconstructed = np.full((t, n), -1, dtype=np.int8)
    for record in values["labels"]:
        reconstructed[record["snapshot"], record["entity_id"]] = record["label"]
    audit.require(np.array_equal(data["labels"], reconstructed), "npz_label_instances", "Known 0/1 coordinates do not exactly equal risk_labels.jsonl")
    audit.require(int((data["labels"] >= 0).sum()) == len(values["labels"]), "npz_known_count", "Known labels are instance totals, not positive-only counts")
    audit.require((data["labels"][data["split"] == -1] == -1).all(), "npz_purged_labels", "Purged queries must have unknown node outcomes")
    if "simulation_labels" in data:
        truth = data["simulation_labels"]
        audit.require(truth.shape == (t, n) and np.isin(truth, [0, 1]).all(), "simulation_truth", "simulation_labels must be a full binary T x N ground-truth array")
        if truth.shape == (t, n):
            known = data["labels"] >= 0
            audit.require(np.array_equal(data["labels"][known], truth[known]), "simulation_known_consistency", "Observed known outcomes differ from full simulator ground truth")
    simulation_summary = check_simulation(directory=path.parent, data=data, values=values, audit=audit)

    split = data["split"]
    present = [np.flatnonzero(split == k) for k in range(3)]
    audit.require(all(len(rows) > 0 for rows in present), "split_coverage", "Train, validation and test must all exist")
    for index, rows in enumerate(present):
        audit.require((data["labels"][rows] >= 0).any(), "split_known_labels", f"Split {index} has no known outcomes")
    for older, newer in ((0, 1), (1, 2)):
        left, right = present[older], present[newer]
        if len(left) and len(right):
            audit.require(left.max() < right.min(), "split_chronology", f"{older}->{newer}")
            audit.require(data["target_days"][left].max() < data["prediction_days"][right].min(), "split_target_leak", f"{older}->{newer}")
    audit.require((data["target_days"] == data["prediction_days"] + 30).all(), "npz_forecast_horizon", "Expected 30 days")
    mask = data["signal_mask"].astype(bool)
    index = data["memory_indices"]
    audit.require(np.isin(data["signal_mask"], [0, 1]).all(), "npz_mask", "signal_mask")
    audit.require((~data["source_mask"].astype(bool) | mask.any(-1)).all(), "observed_source_memory", "Observed sources must have pre-query retrieved evidence")
    source_rows, source_nodes = np.where(data["source_mask"].astype(bool))
    audit.require(np.isin(data["node_type"][source_nodes], [0, 2]).all(), "observed_source_types", "Observed sources must be firm/material warnings")
    if "path_source_mask" in data:
        audit.require(np.array_equal(data["path_source_mask"], data["source_mask"]), "observed_path_sources", "Path extractor must receive the same observed sources")
    audit.require((index[~mask] == -1).all(), "memory_inactive_index", "Inactive slots must be -1")
    audit.require((data["delta"][~mask] == 0).all() and (data["confidence"][~mask] == 0).all()
                  and (data["signals"][~mask] == 0).all(), "memory_inactive_values", "Inactive slots must contain zero vectors/ages/confidence")
    ti, ni, ki = np.where(mask)
    selected = index[mask]
    legal = (selected >= 0) & (selected < len(values["signals"]))
    audit.require(legal.all(), "memory_index_bounds", "Active memory points outside signals.jsonl")
    if legal.all() and len(selected):
        signal_days = np.asarray([record["timestamp_day"] for record in values["signals"]])
        signal_entities = np.asarray([record["entity_id"] for record in values["signals"]])
        signal_confidence = np.asarray([record["confidence"] for record in values["signals"]])
        ages = data["prediction_days"][ti] - signal_days[selected]
        audit.require((ages > 0).all(), "memory_future_leak", "Retrieved signal is on/after its query")
        audit.require((ages <= 30).all(), "memory_lookback", "Retrieved signal older than 30 days")
        audit.require(np.array_equal(signal_entities[selected], ni), "memory_entity_link", "Retrieved signal belongs to another node")
        audit.require(np.allclose(data["delta"][mask], ages), "memory_delta", "Stored age differs from actual timestamp")
        audit.require(np.allclose(data["confidence"][mask], signal_confidence[selected], atol=1e-6), "memory_confidence", "Stored confidence differs from source signal")
        vectors = np.asarray([signed_hash(values["texts"][record["text_id"]]["text"], data["signals"].shape[-1]) for record in values["signals"]])
        audit.require(np.allclose(data["signals"][mask], vectors[selected], atol=1e-6, rtol=1e-6), "memory_text_features", "Retrieved vector does not equal the referenced synthetic text hash")
        for row in index.reshape(-1, memory_shape[-1]):
            active = row[row >= 0]
            if len(active) > 1:
                audit.require(len(set(active.tolist())) == len(active), "memory_duplicate", "A signal repeated in a node/query memory")
    return {"nodes": n, "edges": e, "snapshots": t, "retrieval_size": memory_shape[-1], "signal_dim": data["signals"].shape[-1],
            "known_node_labels": int((data["labels"] >= 0).sum()), "unknown_node_labels": int((data["labels"] == -1).sum()),
            "positive_node_labels": int((data["labels"] == 1).sum()), "negative_node_labels": int((data["labels"] == 0).sum()),
            "known_edge_labels": int((data["edge_labels"] >= 0).sum()), "unknown_edge_labels": int((data["edge_labels"] == -1).sum()),
            "observed_source_instances": int(data["source_mask"].sum()), "retrieved_signal_instances": int(mask.sum()),
            "simulation": simulation_summary,
            "splits": {str(k): int((split == k).sum()) for k in (-1, 0, 1, 2)}, "query_start_day": int(data["prediction_days"].min()),
            "query_end_day": int(data["prediction_days"].max())}


def check_simulation(directory, data, values, audit):
    """Derive full node/edge ground truth from shock and causal event records."""
    shocks = records(directory / "simulation_shocks.jsonl", audit)
    events = records(directory / "simulation_events.jsonl", audit)
    t, n = data["labels"].shape
    e = len(data["src"])
    reconstructed_nodes = np.zeros((t, n), np.int8)
    reconstructed_edges = np.zeros((t, e), np.int8)
    initial = [set() for _ in range(t)]
    shock_positions = set()
    for index, record in enumerate(shocks):
        snapshot, entity = record.get("snapshot"), record.get("entity_id")
        valid = integer(snapshot) and 0 <= snapshot < t and integer(entity) and 0 <= entity < n
        audit.require(valid, "simulation_shock_position", f"shock {index}")
        audit.require(record.get("synthetic") is True, "synthetic_provenance", f"shock {index}")
        if not valid:
            continue
        position = (snapshot, entity)
        audit.require(position not in shock_positions, "simulation_duplicate_shock", f"shock {index}")
        shock_positions.add(position)
        audit.require(record.get("event_day") == int(data["prediction_days"][snapshot]) + 1,
                      "simulation_shock_time", f"shock {index}")
        severity = record.get("severity")
        valid_severity = isinstance(severity, (int, float)) and np.isfinite(severity) and severity >= 0
        audit.require(valid_severity, "simulation_shock_severity", f"shock {index}")
        if valid_severity and severity > float(data["resilience"][snapshot, entity].mean()) + .25:
            initial[snapshot].add(entity)
            reconstructed_nodes[snapshot, entity] = 1
    causal = [[] for _ in range(t)]
    positions = set()
    for index, record in enumerate(events):
        snapshot, edge, stage = record.get("snapshot"), record.get("relation_id"), record.get("stage")
        valid = integer(snapshot) and 0 <= snapshot < t and integer(edge) and 0 <= edge < e and integer(stage) and 1 <= stage <= 3
        audit.require(valid, "simulation_event_position", f"event {index}")
        audit.require(record.get("synthetic") is True, "synthetic_provenance", f"event {index}")
        if not valid:
            continue
        position = (snapshot, edge)
        audit.require(position not in positions, "simulation_duplicate_event", f"event {index}")
        positions.add(position)
        audit.require(record.get("event_day") == int(data["prediction_days"][snapshot]) + stage * 10,
                      "simulation_event_time", f"event {index}")
        causal[snapshot].append((stage, edge))
        reconstructed_edges[snapshot, edge] = 1
        reconstructed_nodes[snapshot, int(data["dst"][edge])] = 1
    for snapshot in range(t):
        frontier, failed = initial[snapshot].copy(), initial[snapshot].copy()
        for stage in range(1, 4):
            stage_events = [edge for step, edge in causal[snapshot] if step == stage]
            newly_failed = set()
            for edge in stage_events:
                src, dst = int(data["src"][edge]), int(data["dst"][edge])
                audit.require(src in frontier and dst not in failed, "simulation_event_causality", f"snapshot {snapshot}, stage {stage}, relation {edge}")
                newly_failed.add(dst)
            failed.update(newly_failed)
            frontier = newly_failed
    audit.require(np.array_equal(data["edge_labels"], reconstructed_edges), "simulation_edge_truth", "NPZ edge_labels differ from causal event records")
    audit.require(np.array_equal(data["simulation_labels"], reconstructed_nodes), "simulation_node_truth", "NPZ simulation_labels differ from initial failures and event destinations")
    return {"shock_records": len(shocks), "initial_failed_instances": sum(len(x) for x in initial),
            "causal_event_records": len(events), "full_positive_node_instances": int(reconstructed_nodes.sum()),
            "full_negative_node_instances": int(reconstructed_nodes.size - reconstructed_nodes.sum())}


def audit_one(directory, specification):
    directory = Path(directory)
    audit = Audit()
    values = {key: records(directory / filename, audit) for key, filename in FILES.items()}
    actual = {key: len(value) for key, value in values.items()}
    sources = audit.guarded("record_structure", lambda: check_records(values, specification, audit))
    tensor = audit.guarded("npz_structure", lambda: check_npz(directory / "dataset.npz", values, audit))
    metadata = audit.guarded("metadata_structure", lambda: json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
    if isinstance(metadata, dict):
        audit.require(metadata.get("data_kind") == "synthetic" and metadata.get("is_real_world") is False,
                      "metadata_synthetic", "Metadata must explicitly deny real-world recovery")
    file_hashes = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in directory.iterdir()
                   if p.is_file() and p.suffix in {".json", ".jsonl", ".npz"}} if directory.exists() else {}
    return {"name": directory.name, "status": "failed" if audit.errors else "passed", "data_kind": "synthetic",
            "is_original_sc_recovery": False, "expected": specification, "actual_record_counts": actual,
            "source_summary": sources, "npz": tensor, "error_counts": dict(audit.errors), "error_examples": audit.examples,
            "files": file_hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/table_scale"))
    parser.add_argument("--output", type=Path, default=Path("reports/table_scale_validation.json"))
    args = parser.parse_args(argv)
    results = [audit_one(args.data_root / name, specification) for name, specification in SPECS.items()]
    report = {"status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
              "scope": "Actual JSONL/NPZ audit of Table 1 scale synthetic engineering datasets",
              "is_original_sc_recovery": False, "datasets": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "datasets": [{"name": r["name"], "status": r["status"],
                       "counts": r["actual_record_counts"], "errors": r["error_counts"]} for r in results]}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
