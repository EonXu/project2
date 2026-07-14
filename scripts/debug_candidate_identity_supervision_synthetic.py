#!/usr/bin/env python
"""Production-path checks for exact candidate-only capture supervision."""

from __future__ import print_function

import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import (
    compute_capture_candidate_identity_loss,
)
from algorithms.sddfg.algorithm.adj_generator import Adj_Generator


def _loss(scores, delta, valid=None, transitions=None):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    delta = torch.as_tensor(delta, dtype=torch.float32)
    if valid is None:
        valid = torch.ones_like(scores)
    else:
        valid = torch.as_tensor(valid, dtype=torch.float32)
    if transitions is None:
        transitions = torch.ones((scores.shape[0], 1))
    else:
        transitions = torch.as_tensor(transitions, dtype=torch.float32)
    return compute_capture_candidate_identity_loss(
        scores, delta, valid, transitions
    )


def test_positive_and_negative_targets_have_opposite_gradients():
    positive_score = torch.tensor([[0.5, 0.5]], requires_grad=True)
    positive = compute_capture_candidate_identity_loss(
        positive_score,
        torch.tensor([[0.2, 0.0]]),
        torch.ones_like(positive_score),
        torch.ones((1, 1)),
    )
    positive["loss"].backward()
    assert positive_score.grad[0, 0].item() < 0.0
    assert positive_score.grad[0, 1].item() > 0.0

    negative_score = torch.tensor([[0.5, 0.5]], requires_grad=True)
    negative = compute_capture_candidate_identity_loss(
        negative_score,
        torch.tensor([[-0.2, 0.0]]),
        torch.ones_like(negative_score),
        torch.ones((1, 1)),
    )
    negative["loss"].backward()
    assert negative_score.grad[0, 0].item() > 0.0
    assert negative_score.grad[0, 1].item() < 0.0
    assert torch.allclose(
        positive_score.grad + negative_score.grad,
        torch.zeros_like(positive_score.grad),
        atol=1e-7,
    )


def test_signed_conditional_probability_definition_is_exact():
    scores = torch.tensor(
        [[0.25, 0.75], [0.25, 0.75]], requires_grad=True
    )
    delta = torch.tensor([[0.2, 0.0], [-0.2, 0.0]])
    result = compute_capture_candidate_identity_loss(
        scores,
        delta,
        torch.ones_like(scores),
        torch.ones((2, 1)),
    )
    expected_log_probability = torch.log(torch.tensor(0.25))
    expected_positive = -0.2 * expected_log_probability / 2.0
    expected_negative = 0.2 * expected_log_probability / 2.0
    assert torch.allclose(result["positive_loss"], expected_positive)
    assert torch.allclose(result["negative_loss"], expected_negative)
    assert torch.allclose(result["loss"], torch.tensor(0.0), atol=1e-7)
    result["loss"].backward()
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 1].item() > 0.0
    assert scores.grad[1, 0].item() > 0.0
    assert scores.grad[1, 1].item() < 0.0


def test_uniform_score_rescaling_does_not_change_loss():
    base = _loss(
        [[0.5, 1.5, 0.2, 0.7]], [[0.2, 0.0, 0.0, 0.0]]
    )
    scaled = _loss(
        [[5.0, 15.0, 2.0, 7.0]], [[0.2, 0.0, 0.0, 0.0]]
    )
    assert torch.allclose(base["loss"], scaled["loss"], atol=1e-7)


def test_active_match_is_excluded_from_candidate_branch():
    disabled = _loss([[0.5, 0.5]], [[0.0, 0.0]])
    assert disabled["loss"].item() == 0.0
    assert disabled["target_count"].item() == 0.0


def test_transition_normalization_is_explicit():
    one = _loss([[0.5, 0.5]], [[0.2, 0.0]])
    two = _loss(
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.2, 0.0], [0.0, 0.0]],
    )
    assert torch.allclose(two["loss"], one["loss"] * 0.5, atol=1e-7)
    assert two["valid_transition_count"].item() == 2.0


def test_optimizer_step_moves_target_relative_to_competitor():
    positive_scores = torch.nn.Parameter(torch.tensor([[0.5, 0.8]]))
    positive_optimizer = torch.optim.SGD([positive_scores], lr=0.1)
    positive_before = positive_scores.detach().clone()
    positive = compute_capture_candidate_identity_loss(
        positive_scores,
        torch.tensor([[0.2, 0.0]]),
        torch.ones_like(positive_scores),
        torch.ones((1, 1)),
    )
    positive_optimizer.zero_grad()
    positive["loss"].backward()
    positive_optimizer.step()
    assert positive_scores[0, 0].item() > positive_before[0, 0].item()
    assert positive_scores[0, 1].item() < positive_before[0, 1].item()

    negative_scores = torch.nn.Parameter(torch.tensor([[0.5, 0.8]]))
    negative_optimizer = torch.optim.SGD([negative_scores], lr=0.1)
    negative_before = negative_scores.detach().clone()
    negative = compute_capture_candidate_identity_loss(
        negative_scores,
        torch.tensor([[-0.2, 0.0]]),
        torch.ones_like(negative_scores),
        torch.ones((1, 1)),
    )
    negative_optimizer.zero_grad()
    negative["loss"].backward()
    negative_optimizer.step()
    assert negative_scores[0, 0].item() < negative_before[0, 0].item()
    assert negative_scores[0, 1].item() > negative_before[0, 1].item()


def test_canonical_rank_is_deterministic_and_invalid_safe():
    scores = torch.tensor([[0.5, 0.7, 0.7, 0.9]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    rank = Adj_Generator.canonical_candidate_ranks(scores, valid)
    assert rank.tolist() == [[3, 1, 2, 0]]


def test_extreme_positive_weights_are_finite_without_probability_clamp():
    scores = torch.tensor([[1e-8, 1.0], [1e6, 1.0]], requires_grad=True)
    result = compute_capture_candidate_identity_loss(
        scores,
        torch.tensor([[0.2, 0.0], [-0.2, 0.0]]),
        torch.ones_like(scores),
        torch.ones((2, 1)),
    )
    assert bool(torch.isfinite(result["loss"]).item())
    result["loss"].backward()
    assert bool(torch.isfinite(scores.grad).all().item())


def test_invalid_target_fails_loudly():
    try:
        _loss([[0.5]], [[0.2]], valid=[[0.0]])
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid candidate target did not fail")

    try:
        _loss([[0.5]], [[0.2]], transitions=[[0.0]])
    except RuntimeError as error:
        assert "padded or invalid transition" in str(error)
        return
    raise AssertionError("padded candidate transition did not fail")


def test_mixed_sign_mass_is_preserved():
    result = _loss(
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.2, 0.0], [-0.2, 0.0]],
    )
    assert torch.allclose(result["positive_mass"], torch.tensor(0.2))
    assert torch.allclose(result["negative_mass"], torch.tensor(0.2))
    assert result["positive_loss"].item() > 0.0
    assert result["negative_loss"].item() < 0.0


def test_nonfinite_nonpositive_target_scores_fail_loudly():
    for score in (float("nan"), -0.1, 0.0):
        try:
            _loss([[score]], [[0.2]])
        except (FloatingPointError, RuntimeError):
            continue
        raise AssertionError("invalid candidate score did not fail")
    unrelated_zero = _loss(
        [[0.5, 0.0]], [[0.2, 0.0]], valid=[[1.0, 0.0]]
    )
    assert bool(torch.isfinite(unrelated_zero["loss"]).item())


def main():
    tests = [
        test_positive_and_negative_targets_have_opposite_gradients,
        test_signed_conditional_probability_definition_is_exact,
        test_uniform_score_rescaling_does_not_change_loss,
        test_active_match_is_excluded_from_candidate_branch,
        test_transition_normalization_is_explicit,
        test_optimizer_step_moves_target_relative_to_competitor,
        test_canonical_rank_is_deterministic_and_invalid_safe,
        test_extreme_positive_weights_are_finite_without_probability_clamp,
        test_invalid_target_fails_loudly,
        test_mixed_sign_mass_is_preserved,
        test_nonfinite_nonpositive_target_scores_fail_loudly,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} candidate identity supervision tests".format(len(tests)))


if __name__ == "__main__":
    main()
