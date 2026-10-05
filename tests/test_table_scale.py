"""Temporal/provenance and independent-simulation tests on reduced fixtures."""
import json

import numpy as np

from dmror.build_table_scale import (FIRST, LAST, SPECS, build, scaled_spec,
                                    simulate_cascade)
from dmror.data import audit_dataset


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_counts_and_entity_coverage_without_annotation_outcome_sampling(tmp_path):
    name = "SC-Auto-Synthetic"
    spec = scaled_spec(name, .008)
    metadata = build(tmp_path, name, scale=.008, signal_dim=16)
    root = tmp_path / name
    assert metadata["is_real_world"] is False
    assert metadata["table1_size_matched"] is False
    assert len(records(root / "entities.jsonl")) == spec.nodes
    assert len(records(root / "relations.jsonl")) == spec.relations
    assert len(records(root / "texts.jsonl")) == spec.texts
    assert len(records(root / "signals.jsonl")) == spec.risk_signals
    label_records = records(root / "risk_labels.jsonl")
    assert len(label_records) == spec.risk_labels
    assert {record["entity_id"] for record in label_records} == set(range(spec.nodes))
    texts = records(root / "texts.jsonl")
    assert min(record["timestamp_day"] for record in texts) == FIRST
    assert max(record["timestamp_day"] for record in texts) == LAST
    assert {record["entity_id"] for record in texts} == set(range(spec.nodes))
    audit = audit_dataset(root / "dataset.npz")
    assert audit["known_node_labels"] == spec.risk_labels
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        coordinates = np.nonzero(data["labels"] >= 0)
        assert np.array_equal(data["labels"][coordinates], data["simulation_labels"][coordinates])
        assert not data["signal_mask"][0].any()
        assert data["target_days"].max() <= LAST
        assert np.isin(data["node_type"], [0, 1, 2, 3, 4]).all()


def test_memory_links_are_real_past_text_observations_and_reproducible(tmp_path):
    name = "SC-Semi-Synthetic"
    first = build(tmp_path / "a", name, scale=.008, signal_dim=16)
    second = build(tmp_path / "b", name, scale=.008, signal_dim=16)
    assert first["file_sha256"] == second["file_sha256"]
    root = tmp_path / "a" / name
    observations = records(root / "signals.jsonl")
    texts = records(root / "texts.jsonl")
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        for ti, entity, slot in zip(*np.nonzero(data["signal_mask"])):
            identifier = int(data["memory_indices"][ti, entity, slot])
            record = observations[identifier]
            assert record["entity_id"] == entity
            text = texts[record["text_id"]]
            assert text["entity_id"] == entity
            assert text["timestamp_day"] == record["timestamp_day"]
            age = int(data["prediction_days"][ti] - record["timestamp_day"])
            assert 0 < age <= 30
            assert data["delta"][ti, entity, slot] == age
            assert np.isclose(data["confidence"][ti, entity, slot], record["confidence"])
        assert (data["memory_indices"][~data["signal_mask"]] == -1).all()


def test_supply_cascade_excludes_descriptive_membership_edges():
    spec = SPECS["SC-Auto-Synthetic"]
    # The firm source fails, and an industry membership edge cannot transmit a
    # delivery failure even with maximal dependency and no barriers.
    capacity = np.full((2, 5), .2, dtype=np.float32)
    failed, causal, events = simulate_cascade(spec, np.random.default_rng(14),
        np.array([0]), np.array([1]), np.array([3]), np.ones(1), np.zeros((1, 4)),
        capacity, np.array([0]), np.array([1.2]))
    assert failed.tolist() == [True, False]
    assert not causal.any()
    assert events == []
    empty, _, _ = simulate_cascade(spec, np.random.default_rng(14),
        np.array([0]), np.array([1]), np.array([0]), np.ones(1), np.zeros((1, 4)),
        capacity, np.array([], dtype=int), np.array([]))
    assert not empty.any()
