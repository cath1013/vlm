"""Pure loss regressions for CRN5-A; no model fitting or checkpoint loading."""
import torch
import pytest

from examples.train_collision_risk_net import window_balanced_masked_bce, window_max_risk_2s_bce


def outcome(bucket, observed=(True, True), pair_present=True):
    return dict(gt_bucket=bucket, observed_negative_buckets=list(observed),
                gt_pair_present=pair_present)


def test_window_bce_matches_max_cumulative_pair_risk_and_uses_gt_onset():
    logits = torch.tensor([[[0., 0.], [-2., -2.]],
                           [[-1., -1.], [1., 0.]],
                           [[0., -1.], [-2., -3.]]], dtype=torch.float64, requires_grad=True)
    pairs = torch.ones((3, 2), dtype=torch.bool)
    rows = [outcome(1), outcome(2), outcome(3)]
    loss = window_max_risk_2s_bce(logits, pairs, rows)
    risk = (1 - (1 - logits.sigmoid()).prod(-1)).max(dim=1).values
    expected = torch.nn.functional.binary_cross_entropy(risk, torch.tensor([1., 1., 0.], dtype=logits.dtype))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # Only each window's maximum-risk pair receives gradients.
    assert (logits.grad[0, 1] == 0).all()
    assert (logits.grad[1, 0] == 0).all()
    assert (logits.grad[2, 1] == 0).all()
    assert (logits.grad[0, 0] < 0).all()
    assert (logits.grad[1, 1] < 0).all()
    assert (logits.grad[2, 0] > 0).all()


def test_window_bce_excludes_padding_censoring_pairless_windows_and_later_hazards():
    logits = torch.tensor([[[0., 0., float('nan')], [float('nan')] * 3],
                           [[float('nan')] * 3, [float('nan')] * 3],
                           [[float('nan')] * 3, [float('nan')] * 3]], requires_grad=True)
    pairs = torch.tensor([[True, False], [True, False], [False, False]])
    rows = [outcome(0), outcome(0, observed=(True, False)), outcome(1)]
    loss = window_max_risk_2s_bce(logits, pairs, rows)
    torch.testing.assert_close(loss, -torch.log(torch.tensor(.25)))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[0, 0, :2] > 0).all()
    assert logits.grad[0, 0, 2] == 0
    assert (logits.grad[0, 1] == 0).all()
    assert (logits.grad[1:] == 0).all()


def test_window_bce_empty_supervision_is_differentiable_zero():
    for pair_count in (0, 2):
        logits = torch.full((1, pair_count, 4), float('nan'), requires_grad=True)
        pairs = torch.zeros((1, pair_count), dtype=torch.bool)
        loss = window_max_risk_2s_bce(logits, pairs, [outcome(2)])
        assert loss.item() == 0
        loss.backward()
        assert (logits.grad == 0).all()


def test_window_bce_is_finite_for_extreme_finite_logits():
    logits = torch.tensor([[[-1000., -1000.]], [[1000., 1000.]]], requires_grad=True)
    loss = window_max_risk_2s_bce(logits, torch.ones((2, 1), dtype=torch.bool),
                                [outcome(1), outcome(0)])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('bucket', [1, 2])
def test_missing_gt_pair_excluded_only_from_auxiliary_loss(bucket):
    logits = torch.tensor([[[0., 0.]], [[-2., -2.]]], requires_grad=True)
    rows = [outcome(0), outcome(bucket, pair_present=False)]
    pairs = torch.ones((2, 1), dtype=torch.bool)
    auxiliary = window_max_risk_2s_bce(logits, pairs, rows)
    torch.testing.assert_close(auxiliary, window_max_risk_2s_bce(logits[:1], pairs[:1], rows[:1]))
    auxiliary.backward()
    assert (logits.grad[1] == 0).all()
    logits.grad.zero_()
    # The existing hazard term still supervises observed negatives for the
    # missing-pair window; auxiliary eligibility must not alter that term.
    hazard = window_balanced_masked_bce(logits, torch.zeros_like(logits),
                                      torch.ones_like(logits, dtype=torch.bool))
    (hazard + window_max_risk_2s_bce(logits, pairs, rows)).backward()
    assert (logits.grad[1] > 0).all()


@pytest.mark.parametrize('bucket', [0, 3, 4, 5])
def test_identifiable_negatives_with_missing_gt_pair_contribute_gradients(bucket):
    logits = torch.tensor([[[0., -1.]]], requires_grad=True)
    pairs = torch.ones((1, 1), dtype=torch.bool)
    loss = window_max_risk_2s_bce(logits, pairs, [outcome(bucket, pair_present=False)])
    expected = torch.nn.functional.softplus(logits).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad > 0).all()


@pytest.mark.parametrize('value', [-100., -1000.])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_extremely_negative_positive_logits_keep_corrective_gradients(value, dtype):
    # The second valid pair has greater risk, even when explicitly formed
    # probabilities would both round to zero. Padding must never win.
    logits = torch.tensor([[[value - 10, value - 10], [value, value], [1000., 1000.]]],
                          dtype=dtype, requires_grad=True)
    pairs = torch.tensor([[True, True, False]])
    loss = window_max_risk_2s_bce(logits, pairs, [outcome(2)])
    torch.testing.assert_close(loss, torch.tensor(-value, dtype=dtype) - torch.log(torch.tensor(2., dtype=dtype)))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[0, 1], torch.full((2,), -.5, dtype=dtype))
    assert (logits.grad[0, 0] == 0).all()
    assert (logits.grad[0, 2] == 0).all()
