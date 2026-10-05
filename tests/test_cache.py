"""Frozen cache identity, strict failure and preservation checks; no LLM needed."""
import json

import numpy as np
import pytest

from dmror.apply_cache import apply_cache, load_cache
from dmror.data import audit_dataset, sha256, write_json


def fixture_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    vectors = np.array([[.2, .8], [-.6, .5]], np.float32)
    np.savez_compressed(cache / "embedding_cache.npz", embeddings=vectors,
                        projection=np.zeros((4, 2), np.float32))
    (cache / "texts.jsonl").write_text(
        '{"cache_id": 0, "text": "shared evidence"}\n'
        '{"cache_id": 1, "text": "other evidence"}\n', encoding="utf-8")
    provenance = {"encoder": "frozen_local_LLM", "model_identifier": "fixture",
                  "cache_sha256": sha256(cache / "embedding_cache.npz"),
                  "original_hidden_dim": 4, "projection_dim": 2,
                  "unique_evidence_segments": 2, "projection_seed": 731}
    write_json(cache / "provenance.json", provenance)
    return cache, vectors, provenance


def fixture_dataset(tmp_path, name="data", unknown=False):
    folder = tmp_path / name
    folder.mkdir()
    indices = np.array([[[10, 20], [20, -1]], [[30, -1], [10, -1]],
                        [[10, -1], [20, -1]]], np.int64)
    mask = indices >= 0
    mask[0, 0, 1] = False  # stale positive ID in padding must still yield zero
    data = {"node_type": np.array([0, 1]), "node_features": np.array([[1., 0.], [0., 1.]], np.float32),
            "src": np.array([0]), "dst": np.array([1]), "edge_type": np.array([0]),
            "labels": np.array([[1, 0], [0, 1], [-1, 1]], np.float32),
            "edge_labels": np.array([[1], [0], [-1]], np.float32),
            "split": np.array([0, 1, 2]), "prediction_days": np.array([0, 40, 80]),
            "target_days": np.array([10, 50, 90]), "signals": np.ones((*indices.shape, 4), np.float32),
            "signal_mask": mask, "memory_indices": indices,
            "delta": np.ones(indices.shape, np.float32),
            "confidence": np.full(indices.shape, .8, np.float32),
            "node_ids": np.array(["entity_a", "entity_b"]),
            "resilience_observed_mask": np.zeros((3, 2, 5), bool)}
    np.savez_compressed(folder / "dataset.npz", **data)
    records = [{"signal_id": 10, "entity_id": "entity_a", "text": "shared evidence"},
               {"signal_id": 20, "entity_id": "entity_b", "text": "shared evidence"},
               {"signal_id": 30, "entity_id": "entity_a", "text": "not cached" if unknown else "other evidence"}]
    (folder / "signals.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    write_json(folder / "metadata.json", {"name": name, "source_provenance": {"fixture": True},
               "statistics": {**audit_dataset(folder / "dataset.npz"), "label_counts_by_split": "retain_me"},
               "text_encoder": "diagnostic_hash"})
    return folder, data


def test_exact_text_cross_entity_vectors_padding_and_attributes(tmp_path):
    cache, vectors, provenance = fixture_cache(tmp_path)
    folder, original = fixture_dataset(tmp_path)
    text_before = (folder / "signals.jsonl").read_bytes()
    apply_cache([folder], cache)
    with np.load(folder / "dataset.npz", allow_pickle=False) as data:
        for key, value in original.items():
            if key != "signals":
                np.testing.assert_array_equal(data[key], value)
        np.testing.assert_array_equal(data["signals"][0, 0, 0], vectors[0])
        np.testing.assert_array_equal(data["signals"][0, 1, 0], vectors[0])
        np.testing.assert_array_equal(data["signals"][1, 0, 0], vectors[1])
        assert (data["signals"][~original["signal_mask"]] == 0).all()
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["text_encoder"] == provenance
    assert metadata["source_provenance"] == {"fixture": True}
    assert metadata["statistics"]["label_counts_by_split"] == "retain_me"
    assert metadata["statistics"]["sha256"] == sha256(folder / "dataset.npz")
    assert metadata["cache_application"]["unique_cache_rows_referenced"] == 2
    assert (folder / "signals.jsonl").read_bytes() == text_before


def test_unknown_text_fails_before_any_dataset_changes(tmp_path):
    cache, _, _ = fixture_cache(tmp_path)
    first, _ = fixture_dataset(tmp_path, "first")
    second, _ = fixture_dataset(tmp_path, "unknown", unknown=True)
    before = {path: sha256(path / "dataset.npz") for path in (first, second)}
    with pytest.raises(ValueError, match="absent from frozen cache"):
        apply_cache([first, second], cache)
    for path in before:
        assert sha256(path / "dataset.npz") == before[path]


def test_frozen_cache_hash_rejects_changed_vectors(tmp_path):
    cache, _, _ = fixture_cache(tmp_path)
    with (cache / "embedding_cache.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Cache SHA-256"):
        load_cache(cache)


def test_dataset_metadata_hash_mismatch_fails(tmp_path):
    cache, _, _ = fixture_cache(tmp_path)
    folder, _ = fixture_dataset(tmp_path)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    metadata["statistics"]["sha256"] = "stale"
    write_json(folder / "metadata.json", metadata)
    with pytest.raises(ValueError, match="dataset SHA-256"):
        apply_cache([folder], cache)


def test_cache_shape_provenance_mismatch_fails(tmp_path):
    cache, _, provenance = fixture_cache(tmp_path)
    provenance["projection_dim"] = 3
    write_json(cache / "provenance.json", provenance)
    with pytest.raises(ValueError, match="shapes differ"):
        load_cache(cache)


def test_missing_memory_signal_fails(tmp_path):
    cache, _, _ = fixture_cache(tmp_path)
    folder, original = fixture_dataset(tmp_path)
    original["memory_indices"][0, 0, 0] = 999
    np.savez_compressed(folder / "dataset.npz", **original)
    write_json(folder / "metadata.json", {"statistics": audit_dataset(folder / "dataset.npz")})
    with pytest.raises(ValueError, match="active memory IDs absent"):
        apply_cache([folder], cache)


def test_cache_text_row_identity_must_be_unique(tmp_path):
    cache, _, _ = fixture_cache(tmp_path)
    with (cache / "texts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"cache_id": 1, "text": "duplicate row"}\n')
    with pytest.raises(ValueError, match="must be unique"):
        load_cache(cache)


def test_portable_npz_replay_is_idempotent(tmp_path):
    import zipfile
    cache, _, _ = fixture_cache(tmp_path)
    folder, _ = fixture_dataset(tmp_path)
    apply_cache([folder], cache)
    first_sha = sha256(folder / "dataset.npz")
    apply_cache([folder], cache)
    assert sha256(folder / "dataset.npz") == first_sha
    with zipfile.ZipFile(folder / "dataset.npz") as archive:
        assert all(member.create_system == 3 and member.date_time == (1980, 1, 1, 0, 0, 0)
                   for member in archive.infolist())
