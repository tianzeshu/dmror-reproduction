"""Numerical and behavioral tests for the manuscript-to-code implementation."""

import unittest

import torch

from dmror.model import DMROR, MODES, group_softmax
from dmror.losses import alignment_loss, multitask_loss
from dmror.paths import beam_search_paths


def small_batch(device="cpu"):
    torch.manual_seed(17)
    n, k = 5, 3
    return {
        "node_features": torch.randn(n, 6, device=device),
        "node_type": torch.tensor([0, 1, 0, 1, 2], device=device),
        "src": torch.tensor([0, 0, 0, 1, 2, 3], device=device),
        "dst": torch.tensor([1, 2, 3, 2, 3, 4], device=device),
        "edge_type": torch.tensor([0, 0, 1, 0, 1, 0], device=device),
        "edge_dependency": torch.tensor([0.9, 0.2, 0.7, 0.5, 0.8, 0.6], device=device),
        "edge_features": torch.rand(6, 4, device=device),
        "resilience": torch.rand(n, 5, device=device),
        "signals": torch.randn(n, k, 8, device=device),
        "delta": torch.tensor([[0., 1., 10.]] * n, device=device),
        "confidence": torch.tensor([[0.9, 0.7, 0.6]] * n, device=device),
        "signal_mask": torch.tensor([[1, 1, 0], [0, 0, 0], [1, 0, 0],
                                      [1, 1, 1], [1, 0, 0]], dtype=torch.bool, device=device),
    }


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = DMROR(6, 8, 3, 2, hidden=16, dropout=0).eval()
        self.batch = small_batch()

    def test_shapes_probabilities_and_empty_signal_rows(self):
        output = self.model(self.batch)
        for name, tensor in output.items():
            self.assertTrue(torch.isfinite(tensor).all(), name)
        self.assertEqual(tuple(output["h_long"].shape), (5, 16))
        self.assertEqual(tuple(output["attention"].shape), (5, 3))
        self.assertEqual(tuple(output["node_prob"].shape), (5,))
        self.assertEqual(tuple(output["edge_score"].shape), (6,))
        self.assertTrue(torch.equal(output["attention"][1], torch.zeros(3)))
        self.assertTrue(torch.equal(output["m_short"][1], torch.zeros(16)))
        torch.testing.assert_close(output["attention"].sum(-1), torch.tensor([1., 0., 1., 1., 1.]))
        for name in ("node_prob", "edge_prob", "beta"):
            self.assertTrue(((output[name] >= 0) & (output[name] <= 1)).all(), name)
        torch.testing.assert_close(output["edge_prob"], -torch.expm1(-output["edge_score"]))

    def test_per_source_relation_normalization_and_exact_transport(self):
        output = self.model(self.batch)
        groups = self.batch["src"] * 2 + self.batch["edge_type"]
        for group in groups.unique():
            torch.testing.assert_close(output["delta_edge"][groups == group].sum(), torch.tensor(1.))
        transmitted = output["overload"][self.batch["src"]] * output["delta_edge"] * (1 - output["beta"])
        incoming = torch.zeros(5).index_add_(0, self.batch["dst"], transmitted)
        torch.testing.assert_close(output["incoming"], incoming)
        torch.testing.assert_close(output["edge_score"], transmitted * output["node_prob"][self.batch["dst"]])

    def test_numerically_stable_group_softmax(self):
        logits = torch.tensor([10000., 10001., -10000.], requires_grad=True)
        weights = group_softmax(logits, torch.tensor([0, 0, 1]), 2)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights[:2].sum(), torch.tensor(1.))
        weights[0].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_time_decay_and_masked_padding(self):
        batch = {key: value.clone() for key, value in self.batch.items()}
        batch["signals"][0] = batch["signals"][0, 0].clone().repeat(3, 1)
        batch["confidence"][0] = 0.8
        batch["signal_mask"][0] = True
        batch["delta"][0] = torch.tensor([0., 1., 10.])
        output = self.model(batch)
        self.assertGreater(output["attention"][0, 0].item(), output["attention"][0, 2].item())
        mask = ~batch["signal_mask"]
        batch["signals"][mask] = float("nan")
        batch["delta"][mask] = float("nan")
        batch["confidence"][mask] = float("nan")
        for value in self.model(batch).values():
            self.assertTrue(torch.isfinite(value).all())

    def test_future_signal_rejected(self):
        self.batch["delta"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "future evidence"):
            self.model(self.batch)

    def test_zero_signals_and_zero_edges(self):
        for name in ("signals", "delta", "confidence", "signal_mask"):
            self.batch[name] = self.batch[name][:, :0]
        for name in ("src", "dst", "edge_type", "edge_dependency", "edge_features"):
            self.batch[name] = self.batch[name][:0]
        output = self.model(self.batch)
        self.assertEqual(output["edge_score"].numel(), 0)
        self.assertTrue(torch.equal(output["incoming"], torch.zeros(5)))
        for tensor in output.values():
            self.assertTrue(torch.isfinite(tensor).all())

    def test_stronger_relation_buffers_suppress_transmission(self):
        with torch.no_grad():
            self.model.buffer_head.weight.zero_()
            self.model.buffer_head.weight[:, -4:] = 2.0
            self.model.buffer_head.bias.fill_(-2.0)
        batch = {key: value.clone() for key, value in self.batch.items()}
        batch["edge_features"].zero_()
        weak = self.model(batch)
        batch["edge_features"].fill_(1.0)
        strong = self.model(batch)
        # Identical buffers across edges shift all logits in a group equally,
        # keeping delta fixed while the explicit (1-beta) term suppresses load.
        torch.testing.assert_close(weak["delta_edge"], strong["delta_edge"])
        self.assertTrue((strong["incoming"] <= weak["incoming"] + 1e-7).all())
        self.assertGreater(weak["incoming"].sum().item(), strong["incoming"].sum().item())

    def test_stronger_node_buffers_raise_threshold_with_fixed_representation(self):
        with torch.no_grad():
            self.model.threshold_head.weight.zero_()
            self.model.threshold_head.weight[:, :5] = 1.0
            self.model.threshold_head.bias.zero_()
            self.model.load_head.weight.zero_()
            self.model.load_head.bias.fill_(3.0)
        self.batch["resilience"].zero_()
        weak = self.model(self.batch)
        self.batch["resilience"].fill_(1.0)
        strong = self.model(self.batch)
        self.assertTrue((strong["threshold"] > weak["threshold"]).all())
        self.assertTrue((strong["overload"] < weak["overload"]).all())

    def test_path_intensity_can_exceed_one_but_bce_stays_finite(self):
        with torch.no_grad():
            self.model.load_head.weight.zero_()
            self.model.load_head.bias.fill_(20.)
            self.model.threshold_head.weight.zero_()
            self.model.threshold_head.bias.fill_(-10.)
            self.model.buffer_head.weight.zero_()
            self.model.buffer_head.bias.fill_(-10.)
        output = self.model(self.batch)
        self.assertGreater(output["edge_score"].max().item(), 1)
        loss = multitask_loss(output, torch.tensor([1., 0., 1., 0., 1.]),
                              edge_labels=torch.tensor([1., 0., 1., 0., 1., 0.]))
        self.assertTrue(torch.isfinite(loss["total"]))

    def test_gradients_reach_both_memories_and_overload_heads(self):
        output = self.model(self.batch)
        self.assertTrue((output["overload"] > 0).all())
        losses = multitask_loss(output, torch.tensor([1., 0., 1., 0., 1.]),
                                edge_labels=torch.tensor([1., 0., 1., 0., 1., 0.]))
        losses["total"].backward()
        parameters = (self.model.node_projection.weight,
                      self.model.graph_layers[0].relation_weight,
                      self.model.signal_projection[0].weight, self.model.vector_gate.weight,
                      self.model.load_head.weight, self.model.threshold_head.weight,
                      self.model.buffer_head.weight, self.model.redistribution_head.weight)
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_alignment_excludes_empty_memory_and_empty_supervision(self):
        output = self.model(self.batch)
        losses = multitask_loss(output, torch.zeros(5), node_mask=torch.zeros(5, dtype=torch.bool),
                                edge_labels=torch.zeros(6), edge_mask=torch.zeros(6, dtype=torch.bool))
        self.assertEqual(losses["total"].item(), 0)
        self.assertEqual(alignment_loss(output["m_short"][:1], output["h_long"][:1]).item(), 0)

    def test_all_ablation_modes_run(self):
        for mode in MODES:
            model = DMROR(6, 8, 3, 2, hidden=16, dropout=0, mode=mode).eval()
            output = model(self.batch)
            for value in output.values():
                self.assertTrue(torch.isfinite(value).all(), mode)
            if mode == "no_resgate":
                self.assertTrue(torch.equal(output["beta"], torch.zeros(6)))
                self.assertTrue(torch.equal(output["threshold"], torch.zeros(5)))
            if mode in ("no_stm", "no_ltm"):
                self.assertFalse(output["context_available"].any())

    def test_temporal_batch_matches_single_day_and_trains(self):
        single = self.model(self.batch)
        batch = dict(self.batch)
        for key in ("resilience", "signals", "delta", "confidence", "signal_mask"):
            batch[key] = self.batch[key].unsqueeze(0).repeat(2, *([1] * self.batch[key].ndim))
        batched = self.model(batch)
        for key in single:
            torch.testing.assert_close(batched[key][0], single[key])
            torch.testing.assert_close(batched[key][1], single[key])
        losses = multitask_loss(batched, torch.tensor([[1., 0., 1., 0., 1.]] * 2),
                                node_mask=torch.ones(2, 5, dtype=torch.bool),
                                edge_labels=torch.tensor([[1., 0., 1., 0., 1., 0.]] * 2))
        losses["total"].backward()
        self.assertTrue(torch.isfinite(self.model.node_projection.weight.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_execution(self):
        output = self.model.cuda()(small_batch("cuda"))
        self.assertEqual(output["node_prob"].device.type, "cuda")
        output["node_prob"].sum().backward()


class PathTests(unittest.TestCase):
    def test_cycle_free_product_ranking(self):
        src = torch.tensor([0, 1, 2, 0, 3])
        dst = torch.tensor([1, 2, 0, 3, 4])
        score = torch.tensor([2., 3., 4., 0.8, 0.9])
        paths = beam_search_paths(src, dst, score, 0, beam_width=8, max_hops=5, top_k=20)
        self.assertEqual(paths[0]["nodes"], [0, 1, 2])
        self.assertAlmostEqual(paths[0]["score"], 6.0)
        for path in paths:
            self.assertEqual(len(path["nodes"]), len(set(path["nodes"])))
            expected = score[path["edge_ids"]].prod().item()
            self.assertAlmostEqual(path["score"], expected, places=6)

    def test_zero_edges_and_high_risk_filter(self):
        self.assertEqual(beam_search_paths(torch.tensor([], dtype=torch.long),
                                          torch.tensor([], dtype=torch.long), torch.tensor([]), 0), [])
        result = beam_search_paths(torch.tensor([0, 0]), torch.tensor([1, 2]),
                                   torch.tensor([0., 0.7]), 0,
                                   node_prob=torch.tensor([0.9, 0.9, 0.1]), min_node_prob=0.5)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
