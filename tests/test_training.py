"""Verify temporal loss, independent baselines, label masks and report honesty."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dmror.losses import multitask_loss
from dmror.metrics import binary_metrics, path_metrics, threshold_scope_metrics, tune_threshold
from dmror.model import DMROR
from dmror.paths import beam_search_paths
from dmror.report import aggregate, stats
from dmror.train import IndependentBaseline, make_tensors, batch_at, temporal_loss, validate_dataset, train, parser


def tiny_batch():
    generator = np.random.default_rng(17)
    t, n, k = 3, 5, 2
    data = {"node_features": generator.normal(size=(n, 4)).astype(np.float32),
            "node_type": np.array([0, 0, 1, 1, 0]), "src": np.array([0, 1, 2, 3, 0]),
            "dst": np.array([1, 2, 3, 4, 4]), "edge_type": np.array([0, 1, 0, 1, 0]),
            "edge_dependency": np.ones(5, np.float32), "edge_features": np.ones((5, 4), np.float32),
            "resilience": generator.random((t, n, 5)).astype(np.float32),
            "signals": generator.normal(size=(t, n, k, 6)).astype(np.float32),
            "delta": np.ones((t, n, k), np.float32), "confidence": np.ones((t, n, k), np.float32),
            "signal_mask": np.ones((t, n, k), bool), "source_mask": np.zeros((t, n), bool),
            "labels": np.array([[1, 0, 0, 1, -1], [0, 1, 0, -1, 1], [0, 0, 1, 0, 1]]),
            "edge_labels": np.array([[1, 0, -1, 0, 0], [0, 1, 0, 0, -1], [0, 0, 1, 0, 0]])}
    return make_tensors(data, "cpu")


def test_info_nce_is_per_time():
    torch.manual_seed(17)
    batch = tiny_batch()
    model = DMROR(4, 6, 2, 2, hidden=8, num_layers=1, dropout=0)
    outputs = model(batch)
    args = SimpleNamespace(mode="full", lambda_align=.05, lambda_path=.1, lambda_reg=0., temperature=.1)
    actual = temporal_loss(outputs, batch, model, torch.tensor(2.), args)
    expected = []
    for index in range(3):
        single = {key: value[index] for key, value in outputs.items()}
        expected.append(multitask_loss(single, batch["labels"][index], batch["labels"][index] >= 0,
                                      batch["edge_labels"][index], batch["edge_labels"][index] >= 0,
                                      pos_weight=torch.tensor(2.))["total"])
    assert torch.allclose(actual, torch.stack(expected).mean())
    actual.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_independent_baselines_obey_inputs():
    batch = tiny_batch()
    for mode in ("graph_only", "text_mlp", "concat"):
        model = IndependentBaseline(4, 6, 2, 2, hidden=8, num_layers=1, dropout=0, mode=mode).eval()
        actual = model(batch)
        assert actual["node_prob"].shape == (3, 5)
        assert actual["edge_prob"].shape == (3, 5)
        assert "load_head" not in dict(model.named_modules())
        if mode == "graph_only":
            assert torch.allclose(actual["node_prob"][0], actual["node_prob"][1])
        elif mode == "text_mlp":
            altered = {**batch, "node_features": batch["node_features"] * 100}
            assert torch.allclose(actual["node_prob"], model(altered)["node_prob"])
        assert torch.allclose(actual["edge_score"], actual["node_prob"][:, batch["src"]] *
                              actual["node_prob"][:, batch["dst"]] * batch["edge_dependency"])


def test_unknown_edge_is_excluded_from_path_prediction():
    result = path_metrics(np.array([0, 0]), np.array([1, 2]), np.array([[1, -1]]),
                          np.array([[.2, .99]]), np.array([[True, False, False]]),
                          beam_search_paths, top_k=1)
    assert result["path_precision"] == 1
    assert result["path_recall"] == 1
    assert result["path_at_k"] == 1
    assert result["path_at_k_evaluable_sources"] == 1
    assert result["evaluation_graph"] == "known-label edges only"
    absent = path_metrics(np.array([0]), np.array([1]), np.array([[-1]]),
                         np.array([[.9]]), np.array([[True, False]]), beam_search_paths)
    assert absent["path_f1"] is None


def test_threshold_scope_does_not_force_top_k_predictions():
    result = threshold_scope_metrics([[1, 0, -1], [0, 0, -1]], [[.9, .2, .99], [.1, .2, .99]], .5)
    assert result["jaccard"] == 1
    assert result["jaccard_n_positive_days"] == 1
    assert result["jaccard_empty_union_days"] == 1


def test_unknown_node_labels_do_not_change_metrics_or_threshold():
    y, p = np.array([0, 1, 0, 1]), np.array([.1, .8, .2, .9])
    threshold = tune_threshold(y, p)
    assert threshold == tune_threshold(np.r_[y, -1], np.r_[p, .999])
    known, extra = binary_metrics(y, p), binary_metrics(np.r_[y, -1], np.r_[p, .999])
    assert known == extra


@pytest.mark.parametrize("objective", ["macro_f1", "positive_f1"])
def test_fast_threshold_matches_independent_metric_search(objective):
    generator = np.random.default_rng(12)
    y = generator.integers(0, 2, size=100)
    p = generator.random(100)
    candidates = np.unique(np.r_[np.linspace(.05, .95, 91), np.quantile(p, np.linspace(0, 1, 101)), .5])
    expected = max((binary_metrics(y, p, float(value))[objective], -abs(value - .5), value) for value in candidates)[2]
    assert tune_threshold(y, p, objective) == expected


def test_aggregation_preserves_missing_values_and_rejects_mixed_data():
    records = [{"dataset": "test", "mode": "full", "seed": seed, "config": {"dataset_sha256": "same"},
                "test": {"node": {"auprc": value}, "path": {"path_f1": None}}, "runtime_seconds": 1}
               for seed, value in [(17, .2), (29, .4)]]
    result = aggregate(records)[0]
    assert result["metrics"]["node.auprc"]["mean"] == pytest.approx(.3)
    assert result["metrics"]["node.auprc"]["std"] == pytest.approx(np.sqrt(.02))
    assert result["metrics"]["path.path_f1"]["mean"] is None
    assert stats([.3])["std"] is None
    records[1]["config"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="Mixed dataset"):
        aggregate(records)


def test_target_window_overlap_is_rejected():
    tensors = tiny_batch()
    data = {key: value.numpy() for key, value in tensors.items()}
    data.update(split=np.array([0, 1, 2]), prediction_days=np.array([0, 30, 60]), target_days=np.array([30, 40, 70]))
    with pytest.raises(ValueError, match="overlaps validation"):
        validate_dataset(data)


def test_train_restore_inference_agree_without_labels_or_original_dataset(tmp_path):
    import json
    from dmror.predict import inference
    tensors = tiny_batch()
    data = {key: value.numpy() for key, value in tensors.items()}
    data.update(split=np.array([0, 1, 2]), prediction_days=np.array([0, 30, 60]), target_days=np.array([10, 40, 70]))
    dataset = tmp_path / "train_dataset.npz"
    np.savez_compressed(dataset, **data)
    run = tmp_path / "run"
    args = parser().parse_args(["--dataset", str(dataset), "--output", str(run), "--mode", "full",
                               "--epochs", "1", "--hidden", "8", "--layers", "1", "--dropout", "0", "--batch-size", "1"])
    train(args)
    expected = np.load(run / "predictions.npz")["test_node_prob"].copy()
    config = json.loads((run / "config.json").read_text())
    config["dataset"] = "/does-not-exist/relocated-project.npz"
    (run / "config.json").write_text(json.dumps(config))
    data.pop("labels")
    data.pop("edge_labels")
    unlabelled = tmp_path / "unlabelled.npz"
    np.savez_compressed(unlabelled, **data)
    prediction_args = SimpleNamespace(run_dir=str(run), dataset=str(unlabelled), output=str(tmp_path / "predicted"),
                     device="cpu", threads=2, batch_size=1, split="test", indices=None,
                     top_k=3, path_top_k=2, beam_width=3, max_hops=3)
    inference(prediction_args)
    actual = np.load(tmp_path / "predicted" / "predictions.npz")["node_prob"]
    np.testing.assert_allclose(expected, actual, atol=1e-6)
