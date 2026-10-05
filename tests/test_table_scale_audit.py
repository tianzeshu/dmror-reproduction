"""Corruption tests for independent Table 1 scale SYNTHETIC data auditing."""
from datetime import date
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

script = Path(__file__).resolve().parents[1] / "scripts" / "verify_table_scale.py"
module_spec = importlib.util.spec_from_file_location("table_scale_audit", script)
audit_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(audit_module)

SPEC = {"entities": [2, 1, 1, 1, 1], "relations": 5, "texts": 6, "signals": 6, "labels": 6, "queries": 3}


def day(year, month, number):
    return (date(year, month, number) - date(1970, 1, 1)).days


def date_string(value):
    return date.fromordinal(date(1970, 1, 1).toordinal() + value).isoformat()


def write_records(folder, filename, values):
    (folder / filename).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def load_records(folder, filename):
    return [json.loads(line) for line in (folder / filename).read_text(encoding="utf-8").splitlines()]


def change_npz(folder, change):
    with np.load(folder / "dataset.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    change(arrays)
    np.savez_compressed(folder / "dataset.npz", **arrays)


@pytest.fixture
def synthetic_fixture(tmp_path):
    folder = tmp_path / "SC-Fixture-Synthetic"
    folder.mkdir()
    types = ["firm", "firm", "product", "material", "industry", "region"]
    entities = [{"entity_id": i, "entity_type": kind, "name": f"Fixture {i}", "synthetic": True}
                for i, kind in enumerate(types)]
    triples = [(0, 1, 0, "supplies"), (0, 2, 1, "produces"), (3, 2, 2, "input_to"),
               (0, 4, 3, "member_of"), (1, 5, 4, "located_in")]
    relations = [{"relation_id": i, "src": src, "dst": dst, "edge_type": kind, "relation_type": name,
                  "dependency": .8, "features": [.1] * 4, "synthetic": True}
                 for i, (src, dst, kind, name) in enumerate(triples)]
    timestamp = day(2018, 1, 2)
    texts = [{"text_id": i, "entity_id": i, "timestamp_day": timestamp, "date": date_string(timestamp),
              "text": f"Original synthetic test text for entity {i}; no actual company information.",
              "source_kind": "original_synthetic_template", "synthetic": True} for i in range(6)]
    signals = [{"signal_id": i, "text_id": i, "entity_id": i, "timestamp_day": timestamp,
                "date": date_string(timestamp), "confidence": .8, "risk_kind": "fixture_warning",
                "severity": "moderate", "synthetic": True} for i in range(6)]
    queries = [{"snapshot": i, "prediction_day": prediction, "target_day": prediction + 30,
                "split": i, "date": date_string(prediction)}
               for i, prediction in enumerate([day(2018, 2, 1), day(2018, 4, 1), day(2018, 6, 1)])]
    labels = [{"label_id": i * 2 + node, "snapshot": i, "entity_id": i * 2 + node,
               "prediction_day": query["prediction_day"], "target_day": query["target_day"],
               "label": 1 - node, "label_kind": "independent_delayed_discrete_cascade", "synthetic": True}
              for i, query in enumerate(queries) for node in range(2)]
    for filename, values in [("entities.jsonl", entities), ("relations.jsonl", relations), ("texts.jsonl", texts),
                             ("signals.jsonl", signals), ("queries.jsonl", queries), ("risk_labels.jsonl", labels)]:
        write_records(folder, filename, values)
    shocks = [{"snapshot": i, "entity_id": i * 2, "event_day": query["prediction_day"] + 1,
               "severity": .9, "synthetic": True} for i, query in enumerate(queries)]
    write_records(folder, "simulation_shocks.jsonl", shocks)
    write_records(folder, "simulation_events.jsonl", [])
    node_type = np.asarray([0, 0, 1, 2, 3, 4])
    masks = np.zeros((3, 6, 2), bool)
    masks[0, :, 0] = True
    memory = np.full(masks.shape, -1, np.int64)
    memory[0, :, 0] = np.arange(6)
    vectors = np.zeros((3, 6, 2, 4), np.float32)
    vectors[masks] = np.asarray([audit_module.signed_hash(row["text"], 4) for row in texts])
    delta = np.zeros(masks.shape, np.float32)
    delta[masks] = 30
    confidence = np.zeros(masks.shape, np.float32)
    confidence[masks] = .8
    known = np.full((3, 6), -1, np.float32)
    truth = np.zeros((3, 6), np.int8)
    for i in range(3):
        known[i, i * 2:i * 2 + 2] = [1, 0]
        truth[i, i * 2] = 1
    np.savez_compressed(folder / "dataset.npz", node_features=np.eye(5, dtype=np.float32)[node_type],
                        node_type=node_type, src=np.asarray([r[0] for r in triples]), dst=np.asarray([r[1] for r in triples]),
                        edge_type=np.arange(5), edge_dependency=np.full(5, .8, np.float32),
                        edge_features=np.full((5, 4), .1, np.float32), resilience=np.full((3, 6, 5), .5, np.float32),
                        signals=vectors, signal_mask=masks, delta=delta, confidence=confidence, memory_indices=memory,
                        labels=known, simulation_labels=truth, edge_labels=np.zeros((3, 5), np.float32), source_mask=np.zeros((3, 6), bool),
                        split=np.arange(3), prediction_days=np.asarray([q["prediction_day"] for q in queries]),
                        target_days=np.asarray([q["target_day"] for q in queries]))
    (folder / "metadata.json").write_text(json.dumps({"data_kind": "synthetic", "is_real_world": False,
                        "statistics": {"this_untrusted_number": 999999}}), encoding="utf-8")
    return folder


def test_counts_are_recomputed_instead_of_trusting_metadata(synthetic_fixture):
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert result["status"] == "passed", result["error_examples"]
    assert result["actual_record_counts"]["labels"] == 6
    assert result["npz"]["known_node_labels"] == 6
    assert result["npz"]["positive_node_labels"] == 3
    assert result["npz"]["unknown_node_labels"] == 12
    assert result["is_original_sc_recovery"] is False


def test_duplicate_relation_is_detected(synthetic_fixture):
    relations = load_records(synthetic_fixture, "relations.jsonl")
    relations[-1] = {**relations[0], "relation_id": 4}
    write_records(synthetic_fixture, "relations.jsonl", relations)
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert result["status"] == "failed"
    assert "duplicate_relation" in result["error_counts"]


def test_timestamped_future_signal_cannot_be_retrieved(synthetic_fixture):
    new_day = day(2018, 2, 2)
    for filename in ("texts.jsonl", "signals.jsonl"):
        rows = load_records(synthetic_fixture, filename)
        rows[0].update(timestamp_day=new_day, date=date_string(new_day))
        write_records(synthetic_fixture, filename, rows)
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "memory_future_leak" in result["error_counts"]


def test_unrecorded_known_negative_is_not_unknown(synthetic_fixture):
    change_npz(synthetic_fixture, lambda data: data["labels"].__setitem__((0, 2), 0))
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "npz_label_instances" in result["error_counts"]
    assert "npz_known_count" in result["error_counts"]


def test_wrong_entity_memory_is_detected(synthetic_fixture):
    change_npz(synthetic_fixture, lambda data: data["memory_indices"].__setitem__((0, 0, 0), 1))
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "memory_entity_link" in result["error_counts"]


def test_retrieved_numeric_features_must_match_referenced_text(synthetic_fixture):
    change_npz(synthetic_fixture, lambda data: data["signals"].__setitem__((0, 0, 0, 0), 123))
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "memory_text_features" in result["error_counts"]


def test_text_entity_and_date_are_independently_checked(synthetic_fixture):
    texts = load_records(synthetic_fixture, "texts.jsonl")
    texts[0]["entity_id"] = 999
    texts[0]["date"] = "2026-01-01"
    write_records(synthetic_fixture, "texts.jsonl", texts)
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "text_entity_link" in result["error_counts"]
    assert "timestamp_date" in result["error_counts"]


def test_target_window_overlap_is_detected(synthetic_fixture):
    queries = load_records(synthetic_fixture, "queries.jsonl")
    prediction = day(2018, 2, 20)
    queries[1].update(prediction_day=prediction, target_day=prediction + 30, date=date_string(prediction))
    write_records(synthetic_fixture, "queries.jsonl", queries)
    change_npz(synthetic_fixture, lambda data: (data["prediction_days"].__setitem__(1, prediction),
                                               data["target_days"].__setitem__(1, prediction + 30)))
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "split_target_leak" in result["error_counts"]


def test_nonfinite_features_are_detected(synthetic_fixture):
    change_npz(synthetic_fixture, lambda data: data["node_features"].__setitem__((0, 0), np.nan))
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "npz_finite" in result["error_counts"]


def test_exact_instance_count_does_not_replace_full_entity_coverage(synthetic_fixture):
    labels = load_records(synthetic_fixture, "risk_labels.jsonl")
    labels[-1]["entity_id"] = 0
    write_records(synthetic_fixture, "risk_labels.jsonl", labels)
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert result["actual_record_counts"]["labels"] == 6
    assert "label_entity_coverage" in result["error_counts"]


def test_unlogged_full_simulator_positive_is_detected(synthetic_fixture):
    change_npz(synthetic_fixture, lambda data: data["simulation_labels"].__setitem__((2, 2), 1))
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "simulation_node_truth" in result["error_counts"]


def test_event_log_requires_a_previously_failed_source(synthetic_fixture):
    event = {"snapshot": 0, "relation_id": 2, "stage": 1, "event_day": day(2018, 2, 1) + 10,
             "synthetic": True}  # Material node 3 never failed in this fixture.
    write_records(synthetic_fixture, "simulation_events.jsonl", [event])
    result = audit_module.audit_one(synthetic_fixture, SPEC)
    assert "simulation_event_causality" in result["error_counts"]
    assert "simulation_edge_truth" in result["error_counts"]


def test_public_cli_fails_for_missing_full_datasets(tmp_path):
    output = tmp_path / "report.json"
    assert audit_module.main(["--data-root", str(tmp_path / "missing"), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["status"] == "failed"
