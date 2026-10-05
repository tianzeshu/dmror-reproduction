"""Independent regression tests for manuscript alignment and unknown labels."""

import torch

from dmror.losses import alignment_loss, multitask_loss


def test_temporal_alignment_equals_mean_of_within_time_objectives():
    torch.manual_seed(83)
    short = torch.randn(3, 4, 6, requires_grad=True)
    long = torch.randn(3, 4, 6, requires_grad=True)
    mask = torch.tensor([[True, True, True, False],
                         [False, True, False, False],
                         [True, True, False, True]])
    expected = torch.stack([alignment_loss(short[index], long[index], mask[index], .3)
                            for index in range(3)]).mean()
    actual = alignment_loss(short, long, mask, .3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(short.grad).all()
    assert torch.isfinite(long.grad).all()
    assert (short.grad[~mask] == 0).all()
    assert (long.grad[~mask] == 0).all()


def test_duplicate_times_do_not_create_extra_entity_negatives():
    memory = torch.eye(4)
    single = alignment_loss(memory, memory, temperature=.2)
    duplicated = memory.unsqueeze(0).repeat(2, 1, 1)
    batched = alignment_loss(duplicated, duplicated, temperature=.2)
    torch.testing.assert_close(batched, single)
    # Flattening the two times would introduce a duplicate positive entity as
    # a false negative and add roughly log(2) to this aligned-memory example.
    flattened = alignment_loss(duplicated.reshape(8, 4), duplicated.reshape(8, 4), temperature=.2)
    assert flattened > batched + .6


def test_multitask_defaults_ignore_unknown_node_and_edge_labels():
    torch.manual_seed(89)
    logits = torch.tensor([.3, -.2, 8., .5], requires_grad=True)
    edge_probability = torch.tensor([.6, .99, .2], requires_grad=True)
    short = torch.randn(4, 5, requires_grad=True)
    long = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([1., 0., -1., 1.])
    edge_labels = torch.tensor([1., -1., 0.])
    outputs = {"node_logits": logits, "edge_prob": edge_probability,
               "m_short": short, "h_long": long,
               "context_available": torch.ones(4, dtype=torch.bool)}
    automatic = multitask_loss(outputs, labels, edge_labels=edge_labels)
    explicit = multitask_loss(outputs, labels, node_mask=labels >= 0,
                              edge_labels=edge_labels, edge_mask=edge_labels >= 0)
    for key in automatic:
        torch.testing.assert_close(automatic[key], explicit[key])
    altered = {key: value.detach().clone() for key, value in outputs.items()}
    altered["node_logits"][2] = -1000
    altered["edge_prob"][1] = .001
    altered["m_short"][2] = 500
    altered["h_long"][2] = -500
    ignored_changed = multitask_loss(altered, labels, edge_labels=edge_labels)
    for key in automatic:
        torch.testing.assert_close(automatic[key], ignored_changed[key])
    automatic["total"].backward()
    assert logits.grad[2] == 0
    assert edge_probability.grad[1] == 0
    assert (short.grad[2] == 0).all()
    assert (long.grad[2] == 0).all()


def test_all_unknown_supervision_returns_finite_zero_loss():
    logits = torch.tensor([.2, -.3], requires_grad=True)
    probability = torch.tensor([.4], requires_grad=True)
    outputs = {"node_logits": logits, "edge_prob": probability,
               "m_short": torch.randn(2, 3), "h_long": torch.randn(2, 3),
               "context_available": torch.ones(2, dtype=torch.bool)}
    result = multitask_loss(outputs, torch.full((2,), -1.),
                            edge_labels=torch.full((1,), -1.))
    assert all(torch.isfinite(value) and value == 0 for value in result.values())
    result["total"].backward()
    assert torch.equal(logits.grad, torch.zeros(2))
    assert torch.equal(probability.grad, torch.zeros(1))
