"""Protect mathematical equivalence of optimizations needed for table-size graphs."""
import copy
import hashlib
import json

import pytest
import torch
import torch.nn.functional as F

from dmror.model import HeterogeneousLayer
from dmror.paths import beam_search_paths, prepare_adjacency


def test_relation_projection_matches_per_edge_reference_and_gradients():
    torch.manual_seed(17)
    layer = HeterogeneousLayer(7, 3, 0).double()
    reference = copy.deepcopy(layer)
    h = torch.randn(8, 7, dtype=torch.float64, requires_grad=True)
    ref_h = h.detach().clone().requires_grad_(True)
    src = torch.tensor([0, 0, 1, 1, 2, 3, 5, 7, 7])
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 4, 0, 1])
    types = torch.tensor([0, 1, 2, 2, 0, 1, 1, 0, 2])
    weights = torch.tensor([.8, .3, .1, .7, 0., .9, .5, .4, .6], dtype=torch.float64)
    actual = layer(h, src, dst, types, weights)
    groups = dst * 3 + types
    degree = ref_h.new_zeros(8 * 3).index_add(0, groups, weights)
    messages = torch.bmm(ref_h[src].unsqueeze(1), reference.relation_weight[types]).squeeze(1)
    messages = messages * (weights / degree[groups].clamp_min(1e-12)).unsqueeze(-1)
    aggregate = torch.zeros_like(ref_h).index_add(0, dst, messages)
    aggregate = aggregate / (degree.reshape(8, 3) > 0).sum(-1).clamp_min(1).unsqueeze(-1)
    expected = reference.norm(ref_h + F.gelu(reference.self_projection(ref_h) + aggregate))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    coefficient = torch.randn_like(actual)
    (actual * coefficient).sum().backward()
    (expected * coefficient).sum().backward()
    torch.testing.assert_close(h.grad, ref_h.grad, atol=1e-11, rtol=1e-11)
    for parameter, ref_parameter in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("source", [0, 1, 2, 3, 7])
def test_prepared_paths_preserve_parallel_edge_ids_cycles_and_filter(source):
    src = torch.tensor([0, 0, 1, 1, 2, 3, 0])
    dst = torch.tensor([1, 1, 2, 3, 0, 4, 4])
    score = torch.tensor([.9, .8, 1.2, .5, .7, .0, .3])
    probability = torch.tensor([.9, .9, .7, .4, .8])
    adjacency = prepare_adjacency(src, dst, score, probability, .5)
    options = dict(beam_width=8, max_hops=4, top_k=20, node_prob=probability, min_node_prob=.5)
    expected = beam_search_paths(src, dst, score, source, **options)
    actual = beam_search_paths(src, dst, score, source, prepared_adjacency=adjacency, **options)
    assert actual == expected


def test_training_config_digest_matches_saved_config(tmp_path):
    from tests.test_training import tiny_batch
    from dmror.train import parser, train
    import numpy as np
    data = {key: value.numpy() for key, value in tiny_batch().items()}
    data.update(split=np.array([0, 1, 2]), prediction_days=np.array([0, 30, 60]), target_days=np.array([10, 40, 70]))
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, **data)
    args = parser().parse_args(["--dataset", str(dataset), "--output", str(tmp_path / "run"),
                               "--epochs", "1", "--hidden", "8", "--layers", "1", "--batch-size", "1"])
    train(args)
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    digest = config.pop("configuration_sha256")
    assert digest == hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def test_scale_run_audit_detects_changed_prediction(tmp_path):
    from tests.test_training import tiny_batch
    from dmror.train import parser, train, sha256
    from scripts.verify_table_scale_runs import audit
    import numpy as np
    data = {key: value.numpy() for key, value in tiny_batch().items()}
    data.update(split=np.array([0, 1, 2]), prediction_days=np.array([0, 30, 60]), target_days=np.array([10, 40, 70]))
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, **data)
    common = dict(epochs=1, hidden=8, layers=1, batch_size=1)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"common": common, "jobs": [{"name": "tiny/full/seed_17",
                         "dataset": str(dataset), "parameters": {"mode": "full", "seed": 17}}]}))
    results = tmp_path / "results"
    run = results / "tiny/full/seed_17"
    args = parser().parse_args(["--dataset", str(dataset), "--output", str(run),
                     "--epochs", "1", "--hidden", "8", "--layers", "1", "--batch-size", "1",
                     "--protocol-sha256", sha256(protocol)])
    train(args)
    audit_output = tmp_path / "audit.json"
    assert audit(protocol, results, audit_output)
    with np.load(run / "predictions.npz") as archive:
        changed = {key: archive[key] for key in archive.files}
    changed["test_node_prob"][:] = .123
    np.savez_compressed(run / "predictions.npz", **changed)
    assert not audit(protocol, results, audit_output)
    assert len(json.loads(audit_output.read_text())["errors"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_first_cuda_training_process_records_peak_memory(tmp_path):
    from tests.test_training import tiny_batch
    import numpy as np
    import subprocess
    import sys
    from pathlib import Path
    data = {key: value.numpy() for key, value in tiny_batch().items()}
    data.update(split=np.array([0, 1, 2]), prediction_days=np.array([0, 30, 60]), target_days=np.array([10, 40, 70]))
    dataset, run = tmp_path / "dataset.npz", tmp_path / "cuda_run"
    np.savez_compressed(dataset, **data)
    process = subprocess.run([sys.executable, "-m", "dmror.train", "--dataset", str(dataset),
                    "--output", str(run), "--device", "cuda:0", "--epochs", "1", "--hidden", "8",
                    "--layers", "1", "--batch-size", "1"], cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((run / "result.json").read_text())["cuda_peak_allocated_bytes"] > 0
