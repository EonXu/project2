#!/usr/bin/env python
"""Production-path checks for target-local capture outcome factor PPO loss."""

from __future__ import print_function

import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import (
    compute_capture_outcome_factor_ppo_loss,
)


def _loss(delta, factors=None, transition_mask=None, ratio=None):
    delta = torch.as_tensor(delta, dtype=torch.float32)
    if factors is None:
        factors = torch.ones_like(delta)
    else:
        factors = torch.as_tensor(factors, dtype=torch.float32)
    if transition_mask is None:
        transition_mask = torch.ones(
            (delta.shape[0], 1), dtype=torch.float32
        )
    else:
        transition_mask = torch.as_tensor(
            transition_mask, dtype=torch.float32
        )
    if ratio is None:
        ratio = torch.ones_like(delta)
    else:
        ratio = torch.as_tensor(ratio, dtype=torch.float32)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    return compute_capture_outcome_factor_ppo_loss(
        factor_imp_weights=ratio,
        clipped_factor_imp_weights=clipped,
        capture_outcome_local_delta=delta,
        factor_loss_mask=factors,
        transition_mask=transition_mask,
    )


def test_positive_and_negative_surrogate_directions():
    result = _loss([[0.3, 0.0], [-0.3, 0.0]])
    assert torch.allclose(result["loss"], torch.tensor(0.0), atol=1e-7)
    assert result["positive_loss"].item() < 0.0
    assert result["negative_loss"].item() > 0.0
    assert result["target_count"].item() == 2.0


def test_order2_and_order3_target_positions_are_supported():
    # Factor columns are identity slots; the helper deliberately does not
    # special-case order, so exact order2/order3 targets share signed semantics.
    result = _loss([[0.2, 0.0, -0.1]])
    assert result["target_count"].item() == 2.0
    assert result["positive_loss"].item() < 0.0
    assert result["negative_loss"].item() > 0.0


def test_unrelated_factor_count_cannot_dilute_episode_mass():
    small = _loss([[0.25, 0.0], [-0.25, 0.0]])
    large = _loss([
        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    for key in ("loss", "positive_loss", "negative_loss"):
        assert torch.allclose(small[key], large[key], atol=1e-7)
    assert large["factors_per_transition"].item() == 6.0


def test_transition_denominator_matches_graph_objective_scale():
    one = _loss([[0.2, 0.0]])
    two = _loss([[0.2, 0.0], [0.0, 0.0]])
    assert torch.allclose(
        two["positive_loss"], one["positive_loss"] * 0.5, atol=1e-7
    )
    assert two["valid_transition_count"].item() == 2.0


def test_invalid_or_padding_target_fails_loudly():
    try:
        _loss([[0.2, 0.0]], factors=[[0.0, 1.0]])
    except RuntimeError:
        return
    raise AssertionError("invalid target factor did not fail")


def test_nonfinite_input_fails_loudly():
    try:
        _loss([[float("nan"), 0.0]])
    except FloatingPointError:
        return
    raise AssertionError("non-finite local delta did not fail")


def test_disabled_outcome_is_exact_zero():
    result = _loss([[0.0, 0.0], [0.0, 0.0]])
    assert result["loss"].item() == 0.0
    assert result["positive_loss"].item() == 0.0
    assert result["negative_loss"].item() == 0.0
    assert result["target_count"].item() == 0.0


def test_importance_is_applied_once_and_gradients_are_signed():
    positive_log_ratio = torch.zeros((1, 2), requires_grad=True)
    positive_ratio = torch.exp(positive_log_ratio)
    positive = compute_capture_outcome_factor_ppo_loss(
        positive_ratio,
        torch.clamp(positive_ratio, 0.8, 1.2),
        torch.tensor([[0.2, 0.0]]),
        torch.ones((1, 2)),
        torch.ones((1, 1)),
    )
    positive["loss"].backward()
    assert positive_log_ratio.grad[0, 0].item() < 0.0
    assert positive_log_ratio.grad[0, 1].item() == 0.0

    negative_log_ratio = torch.zeros((1, 2), requires_grad=True)
    negative_ratio = torch.exp(negative_log_ratio)
    negative = compute_capture_outcome_factor_ppo_loss(
        negative_ratio,
        torch.clamp(negative_ratio, 0.8, 1.2),
        torch.tensor([[-0.2, 0.0]]),
        torch.ones((1, 2)),
        torch.ones((1, 1)),
    )
    negative["loss"].backward()
    assert negative_log_ratio.grad[0, 0].item() > 0.0
    assert negative_log_ratio.grad[0, 1].item() == 0.0

    unit = compute_capture_outcome_factor_ppo_loss(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.tensor([[0.1, 0.0]]),
        torch.ones((1, 2)),
        torch.ones((1, 1)),
    )
    doubled_ratio = torch.tensor([[2.0, 1.0]])
    doubled = compute_capture_outcome_factor_ppo_loss(
        doubled_ratio,
        doubled_ratio,
        torch.tensor([[0.1, 0.0]]),
        torch.ones((1, 2)),
        torch.ones((1, 1)),
    )
    assert torch.allclose(
        doubled["positive_loss"],
        unit["positive_loss"] * 2.0,
        atol=1e-7,
    ), "importance=2 must scale the branch once, not square it"


def test_zero_transition_denominator_is_explicit():
    try:
        _loss([[0.2]], transition_mask=[[0.0]])
    except RuntimeError:
        return
    raise AssertionError("target delta with zero transition denominator did not fail")


def main():
    tests = [
        test_positive_and_negative_surrogate_directions,
        test_order2_and_order3_target_positions_are_supported,
        test_unrelated_factor_count_cannot_dilute_episode_mass,
        test_transition_denominator_matches_graph_objective_scale,
        test_invalid_or_padding_target_fails_loudly,
        test_nonfinite_input_fails_loudly,
        test_disabled_outcome_is_exact_zero,
        test_importance_is_applied_once_and_gradients_are_signed,
        test_zero_transition_denominator_is_explicit,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} outcome factor-loss tests".format(len(tests)))


if __name__ == "__main__":
    main()
