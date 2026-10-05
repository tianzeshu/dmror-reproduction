"""Synthetic fixtures verify outcome import; they are not real annotations."""
import csv
import json

import numpy as np
import pytest

from dmror.data import audit_dataset, sha256, write_json
from dmror.import_annotations import import_annotations


def source(tmp_path):
    folder = tmp_path / "prepared"
    folder.mkdir()
    arrays = {
        "node_ids": np.array(["fixture_a", "fixture_b"]), "node_type": np.array([0, 1]),
        "node_features": np.eye(2, dtype=np.float32),
        "src": np.array([0, 0]), "dst": np.array([1, 1]), "edge_type": np.array([0, 1]),
        "edge_features": np.zeros((2, 4), np.float32),
        "edge_features_observed_mask": np.zeros((2, 4), bool),
        "resilience_observed_mask": np.zeros((3, 2, 5), bool),
        "signals": np.arange(24, dtype=np.float32).reshape(3, 2, 1, 4),
        "signal_mask": np.ones((3, 2, 1), bool), "delta": np.ones((3, 2, 1), np.float32),
        "memory_indices": np.arange(6).reshape(3, 2, 1),
        # Binary source dtypes must not turn UNKNOWN (-1) into 255/True.
        "labels": np.zeros((3, 2), np.uint8), "edge_labels": np.ones((3, 2), bool),
        "source_mask": np.zeros((3, 2), bool), "split": np.array([0, 1, 2]),
        "prediction_days": np.array([0, 40, 80]), "target_days": np.array([30, 70, 110]),
    }
    np.savez_compressed(folder / "dataset.npz", **arrays)
    write_json(folder / "metadata.json", {"name": "fixture", "label_kind": "future_text_proxy",
               "text_encoder": {"encoder": "frozen_fixture", "cache_sha256": "fixture-only"},
               "source_provenance": [{"fixture": True}], "relation_types": ["Supplies", "InvestsIn"],
               "statistics": audit_dataset(folder / "dataset.npz")})
    (folder / "label_evidence.jsonl").write_text('{"proxy_only": true}\n', encoding="utf-8")
    return folder, arrays


def node_row(**change):
    row = {"entity_id": "fixture_a", "query_day": "1970-01-01", "target_day": "1970-01-31",
           "label": "1", "event_date": "1970-01-10", "evidence_url": "https://fixture.invalid/evidence",
           "evidence_span": "TEST FIXTURE ONLY", "reviewer": "fixture_reviewer", "verified": "1"}
    return {**row, **change}


def csv_file(tmp_path, name, rows):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_preserves_features_source_and_encoder_with_unknown_labels(tmp_path):
    folder, original = source(tmp_path)
    original_sha = sha256(folder / "dataset.npz")
    labels = csv_file(tmp_path, "nodes.csv", [node_row(), node_row(entity_id="fixture_b", label="0", event_date="")])
    output = tmp_path / "reviewed"
    metadata = import_annotations(folder, labels, output)
    assert sha256(folder / "dataset.npz") == original_sha
    with np.load(output / "dataset.npz", allow_pickle=False) as data:
        for name, value in original.items():
            if name not in {"labels", "edge_labels"}:
                np.testing.assert_array_equal(data[name], value)
        np.testing.assert_array_equal(data["labels"], [[1, 0], [-1, -1], [-1, -1]])
        assert np.all(data["edge_labels"] == -1)
        assert (data["labels"] >= 0).sum() == 2  # exactly the reviewed cells can enter loss/evaluation
    assert metadata["text_encoder"]["cache_sha256"] == "fixture-only"
    assert metadata["annotation_import"]["source_dataset_sha256"] == original_sha
    assert metadata["statistics"]["sha256"] == sha256(output / "dataset.npz")
    assert metadata["is_disruption_gold_standard"] is False
    assert (output / "label_evidence.jsonl").read_bytes() == (folder / "label_evidence.jsonl").read_bytes()
    assert len((output / "annotation_evidence.jsonl").read_text().splitlines()) == 2


def test_precise_parallel_edge_and_nodes_jsonl_mapping(tmp_path):
    folder, original = source(tmp_path)
    del original["node_ids"]
    np.savez_compressed(folder / "dataset.npz", **original)
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["statistics"] = audit_dataset(folder / "dataset.npz")
    write_json(folder / "metadata.json", metadata)
    (folder / "nodes.jsonl").write_text('{"node_index":1,"entity_id":"fixture_b"}\n{"node_index":0,"entity_id":"fixture_a"}\n')
    nodes = csv_file(tmp_path, "nodes.csv", [node_row()])
    edges = csv_file(tmp_path, "edges.csv", [{"query_day": "0", "target_day": "30", "source_id": "fixture_a",
        "target_id": "fixture_b", "relation_type": "InvestsIn", "label": "1", "evidence_url": "fixture://edge",
        "evidence_span": "TEST ONLY", "reviewer": "fixture_reviewer", "verified": "1"}])
    output = tmp_path / "reviewed"
    import_annotations(folder, nodes, output, edges)
    with np.load(output / "dataset.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["edge_labels"], [[-1, 1], [-1, -1], [-1, -1]])
        assert not data["source_mask"].any()


@pytest.mark.parametrize("change,match", [
    ({"entity_id": "absent"}, "Unknown entity"),
    ({"target_day": "31"}, "exactly match"),
    ({"event_date": "1970-01-31"}, "inside"),
    ({"verified": "0"}, "verified must be 1"),
    ({"evidence_span": ""}, "must not be blank"),
    ({"label": "-1"}, "label must be 0/1"),
])
def test_invalid_annotations_fail_without_output_or_source_change(tmp_path, change, match):
    folder, _ = source(tmp_path)
    before = sha256(folder / "dataset.npz")
    nodes = csv_file(tmp_path, "nodes.csv", [node_row(**change)])
    with pytest.raises(ValueError, match=match):
        import_annotations(folder, nodes, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    assert sha256(folder / "dataset.npz") == before


def test_duplicate_conflict_existing_output_and_source_hash_refused(tmp_path):
    folder, _ = source(tmp_path)
    nodes = csv_file(tmp_path, "nodes.csv", [node_row(), node_row(label="0", event_date="")])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        import_annotations(folder, nodes, tmp_path / "refused")
    nodes = csv_file(tmp_path, "nodes.csv", [node_row()])
    with pytest.raises(ValueError, match="already exists"):
        import_annotations(folder, nodes, folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["statistics"]["sha256"] = "changed"
    write_json(folder / "metadata.json", metadata)
    with pytest.raises(ValueError, match="SHA-256"):
        import_annotations(folder, nodes, tmp_path / "refused")


def test_empty_template_is_not_observed_data(tmp_path):
    folder, _ = source(tmp_path)
    nodes = tmp_path / "empty.csv"
    nodes.write_text(",".join(node_row()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No reviewed annotations"):
        import_annotations(folder, nodes, tmp_path / "refused")
