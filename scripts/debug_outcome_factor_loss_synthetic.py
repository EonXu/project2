#!/usr/bin/env python
"""Production checks for exact-identity local factor PPO losses."""

from __future__ import print_function

import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import (
    _base_factor_population_diagnostics,
    _pair_gradient_direction_diagnostics,
    _pair_realized_displacement_diagnostics,
    _pair_target_score_change_diagnostics,
    _validate_optimizer_step_pair_credit,
    compute_capture_outcome_factor_ppo_loss,
    compute_identity_local_factor_ppo_loss,
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
    return compute_identity_local_factor_ppo_loss(
        factor_imp_weights=ratio,
        clipped_factor_imp_weights=clipped,
        identity_local_delta=delta,
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


def test_sparse_pair_population_ignores_unrelated_transitions():
    ratio = torch.ones((3, 2), dtype=torch.float32)
    delta = torch.tensor([
        [0.2, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    sparse = compute_identity_local_factor_ppo_loss(
        factor_imp_weights=ratio,
        clipped_factor_imp_weights=ratio,
        identity_local_delta=delta,
        factor_loss_mask=torch.ones_like(delta),
        transition_mask=torch.ones((3, 1)),
        normalize_by_target_transitions=True,
    )
    single = compute_identity_local_factor_ppo_loss(
        factor_imp_weights=ratio[:1],
        clipped_factor_imp_weights=ratio[:1],
        identity_local_delta=delta[:1],
        factor_loss_mask=torch.ones_like(delta[:1]),
        transition_mask=torch.ones((1, 1)),
        normalize_by_target_transitions=True,
    )
    assert torch.allclose(sparse["loss"], single["loss"], atol=1e-7)
    assert sparse["target_transition_count"].item() == 1.0
    assert sparse["valid_transition_count"].item() == 3.0
    assert sparse["normalization_denominator"].item() == 1.0
    assert sparse["normalization_uses_target_transitions"] == 1.0


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
    positive = compute_identity_local_factor_ppo_loss(
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
    negative = compute_identity_local_factor_ppo_loss(
        negative_ratio,
        torch.clamp(negative_ratio, 0.8, 1.2),
        torch.tensor([[-0.2, 0.0]]),
        torch.ones((1, 2)),
        torch.ones((1, 1)),
    )
    negative["loss"].backward()
    assert negative_log_ratio.grad[0, 0].item() > 0.0
    assert negative_log_ratio.grad[0, 1].item() == 0.0

    unit = compute_identity_local_factor_ppo_loss(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        torch.tensor([[0.1, 0.0]]),
        torch.ones((1, 2)),
        torch.ones((1, 1)),
    )
    doubled_ratio = torch.tensor([[2.0, 1.0]])
    doubled = compute_identity_local_factor_ppo_loss(
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


def test_capture_outcome_public_entry_matches_generic_objective():
    ratio = torch.tensor([[1.0, 0.9]], dtype=torch.float32)
    delta = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    factor_mask = torch.ones_like(delta)
    transition_mask = torch.ones((1, 1), dtype=torch.float32)
    public = compute_capture_outcome_factor_ppo_loss(
        factor_imp_weights=ratio,
        clipped_factor_imp_weights=torch.clamp(ratio, 0.8, 1.2),
        capture_outcome_local_delta=delta,
        factor_loss_mask=factor_mask,
        transition_mask=transition_mask,
    )
    generic = compute_identity_local_factor_ppo_loss(
        factor_imp_weights=ratio,
        clipped_factor_imp_weights=torch.clamp(ratio, 0.8, 1.2),
        identity_local_delta=delta,
        factor_loss_mask=factor_mask,
        transition_mask=transition_mask,
    )
    for key in ("loss", "positive_loss", "negative_loss", "target_count"):
        assert torch.allclose(public[key], generic[key], atol=1e-7)


def test_optimizer_step_pair_credit_requires_centered_two_sided_mass():
    valid = _validate_optimizer_step_pair_credit(
        torch.tensor([[0.2, 0.0], [-0.2, 0.0]], dtype=torch.float32)
    )
    assert valid["class_complete"] == 1.0
    assert valid["contract_valid"] == 1.0
    assert valid["centered_error"].item() <= 1e-7

    zero = _validate_optimizer_step_pair_credit(
        torch.zeros((2, 2), dtype=torch.float32)
    )
    assert zero["class_complete"] == 0.0
    assert zero["contract_valid"] == 1.0

    for invalid in (
            torch.tensor([[0.2, 0.0]], dtype=torch.float32),
            torch.tensor([[0.2, 0.0], [-0.1, 0.0]], dtype=torch.float32)):
        try:
            _validate_optimizer_step_pair_credit(invalid)
        except RuntimeError:
            continue
        raise AssertionError(
            "invalid optimizer-step pair population did not fail loudly"
        )


def test_base_population_split_matches_hand_calculation():
    base_surr = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    factor_training_mask = torch.tensor([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    factor_loss_mask = torch.tensor([
        [1.0, 0.25],
        [0.5, 0.0],
        [0.0, 0.75],
    ])
    result = _base_factor_population_diagnostics(
        base_factor_min_surr=base_surr,
        factor_training_mask=factor_training_mask,
        factor_loss_mask=factor_loss_mask,
        transition_mask=torch.ones((3, 1)),
        replay_population_provenance=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ]),
    )
    expected = {
        "pair_valid_transition_count": 2.0,
        "non_pair_valid_transition_count": 1.0,
        "pair_valid_factor_mask_count": 3.0,
        "non_pair_valid_factor_mask_count": 1.0,
        "pair_weighted_denominator": 2.0,
        "non_pair_weighted_denominator": 0.5,
        "pair_numerator": -6.0,
        "non_pair_numerator": -1.5,
        "pair_episode_count": 2.0,
        "non_pair_episode_count": 1.0,
        "pair_episode_weight_mean": 0.4,
        "pair_episode_weight_min": 0.3,
        "pair_episode_weight_max": 0.5,
        "non_pair_episode_weight_mean": 0.2,
        "non_pair_episode_weight_min": 0.2,
        "non_pair_episode_weight_max": 0.2,
    }
    for key, value in expected.items():
        assert torch.allclose(
            result[key],
            torch.tensor(value),
            atol=1e-7,
        ), (key, result[key].item(), value)
    assert result["contract_valid"] == 1.0


def test_base_population_split_survives_float32_cancellation():
    base_surr = torch.tensor(
        [[1.0e8], [1.0], [-1.0e8], [1.0]],
        dtype=torch.float32,
    )
    factor_training_mask = torch.ones_like(base_surr)
    factor_loss_mask = torch.ones_like(base_surr)
    transition_mask = torch.ones_like(base_surr)
    replay_population_provenance = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    float32_pair = -base_surr[:2].sum()
    float32_non_pair = -base_surr[2:].sum()
    float32_full = -base_surr.sum()
    assert bool(
        ((float32_pair + float32_non_pair) != float32_full).item()
    )

    result = _base_factor_population_diagnostics(
        base_factor_min_surr=base_surr,
        factor_training_mask=factor_training_mask,
        factor_loss_mask=factor_loss_mask,
        transition_mask=transition_mask,
        replay_population_provenance=replay_population_provenance,
    )
    assert result["pair_numerator"].dtype == base_surr.dtype
    assert result["non_pair_numerator"].dtype == base_surr.dtype
    assert result["contract_valid"] == 1.0


def test_base_population_split_zero_and_single_valid_masks():
    zeros = torch.zeros((2, 3))
    zero_result = _base_factor_population_diagnostics(
        base_factor_min_surr=torch.ones_like(zeros),
        factor_training_mask=zeros,
        factor_loss_mask=zeros,
        transition_mask=torch.zeros((2, 1)),
        replay_population_provenance=torch.tensor([
            [1.0, 1.0, 0.0],
            [0.0, 3.0, 0.0],
        ]),
    )
    for key in (
            "pair_valid_transition_count",
            "non_pair_valid_transition_count",
            "pair_valid_factor_mask_count",
            "non_pair_valid_factor_mask_count",
            "pair_weighted_denominator",
            "non_pair_weighted_denominator",
            "pair_numerator",
            "non_pair_numerator"):
        assert zero_result[key].item() == 0.0

    single_result = _base_factor_population_diagnostics(
        base_factor_min_surr=torch.tensor([[7.0, 11.0], [13.0, 17.0]]),
        factor_training_mask=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        factor_loss_mask=torch.tensor([[0.0, 0.0], [0.125, 0.0]]),
        transition_mask=torch.tensor([[0.0], [1.0]]),
        replay_population_provenance=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
    )
    assert single_result["pair_valid_transition_count"].item() == 0.0
    assert single_result["non_pair_valid_transition_count"].item() == 1.0
    assert single_result["non_pair_valid_factor_mask_count"].item() == 1.0
    assert torch.allclose(
        single_result["non_pair_weighted_denominator"],
        torch.tensor(0.125),
        atol=1e-7,
    )
    assert torch.allclose(
        single_result["non_pair_numerator"],
        torch.tensor(-1.625),
        atol=1e-7,
    )


def test_base_population_split_rejects_nonbinary_provenance():
    try:
        _base_factor_population_diagnostics(
            base_factor_min_surr=torch.ones((1, 1)),
            factor_training_mask=torch.ones((1, 1)),
            factor_loss_mask=torch.ones((1, 1)),
            transition_mask=torch.ones((1, 1)),
            replay_population_provenance=torch.tensor([[0.5, 0.0, 0.0]]),
        )
    except RuntimeError:
        return
    raise AssertionError("non-binary pair-evidence provenance did not fail")


def test_base_population_diagnostics_do_not_change_training_gradient():
    base_surr = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    factor_mask = torch.tensor([[1.0, 0.25], [0.5, 0.0]])
    denominator = factor_mask.sum().clamp_min(1.0)
    training_loss = -(base_surr * factor_mask).sum() / denominator
    _base_factor_population_diagnostics(
        base_factor_min_surr=base_surr,
        factor_training_mask=(factor_mask > 0.0).float(),
        factor_loss_mask=factor_mask,
        transition_mask=torch.ones((2, 1)),
        replay_population_provenance=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
    )
    training_loss.backward()
    assert torch.allclose(
        base_surr.grad,
        -factor_mask / denominator,
        atol=1e-7,
    )


def test_pair_gradient_diagnostics_are_read_only_and_directional():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    pair_loss = (
        0.5 * parameter[0] * parameter[0]
        + 0.5 * parameter[1] * parameter[1]
    )
    base_loss = 2.0 * parameter[0] - parameter[1]
    pair_grads = torch.autograd.grad(
        pair_loss,
        (parameter,),
        retain_graph=True,
    )
    base_grads = torch.autograd.grad(
        base_loss,
        (parameter,),
        retain_graph=True,
    )
    assert parameter.grad is None
    result = _pair_gradient_direction_diagnostics(
        pair_grads=pair_grads,
        base_factor_grads=base_grads,
        reference=pair_loss,
    )
    assert torch.allclose(
        result["pair_norm"],
        torch.sqrt(torch.tensor(5.0)),
        atol=1e-7,
    )
    assert torch.allclose(
        result["base_factor_norm"],
        torch.sqrt(torch.tensor(5.0)),
        atol=1e-7,
    )
    assert torch.allclose(
        result["pair_base_dot"],
        torch.tensor(4.0),
        atol=1e-7,
    )
    assert torch.allclose(
        result["pair_base_cosine"],
        torch.tensor(0.8),
        atol=1e-7,
    )
    assert parameter.grad is None

    total_loss = pair_loss + base_loss
    total_loss.backward()
    assert torch.allclose(
        parameter.grad,
        torch.tensor([3.0, -3.0]),
        atol=1e-7,
    )


def test_pair_realized_displacement_matches_hand_calculation():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = [parameter.detach().clone()]
    pair_grads = (torch.tensor([0.3, 0.4]),)
    with torch.no_grad():
        parameter.copy_(torch.tensor([0.7, 1.6]))
    result = _pair_realized_displacement_diagnostics(
        parameters=(parameter,),
        parameter_before_step=before,
        pair_grads=pair_grads,
        reference=parameter,
    )
    assert torch.allclose(
        result["update_norm"],
        torch.tensor(0.5),
        atol=1e-7,
    )
    assert torch.allclose(
        result["descent_dot"],
        torch.tensor(0.25),
        atol=1e-7,
    )
    assert torch.allclose(
        result["descent_cosine"],
        torch.tensor(1.0),
        atol=1e-7,
    )


def test_pair_target_score_change_joins_exact_signed_targets():
    result = _pair_target_score_change_diagnostics(
        pre_factor_logp=torch.tensor([[-1.0, -2.0, -3.0]]),
        post_factor_logp=torch.tensor([[-0.8, -2.3, 7.0]]),
        pair_local_delta=torch.tensor([[0.1, -0.1, 0.0]]),
    )
    assert result["target_count"].item() == 2.0
    assert torch.allclose(
        result["signed_change_mean"],
        torch.tensor(0.25),
        atol=1e-7,
    )
    assert torch.allclose(
        result["positive_change_mean"],
        torch.tensor(0.2),
        atol=1e-7,
    )
    assert torch.allclose(
        result["negative_signed_change_mean"],
        torch.tensor(0.3),
        atol=1e-7,
    )


def test_pair_diagnostics_fail_loudly_on_invalid_targets():
    clipped = _pair_gradient_direction_diagnostics(
        pair_grads=(torch.zeros(2),),
        base_factor_grads=(torch.ones(2),),
        reference=torch.tensor(0.0),
    )
    assert clipped["pair_norm"].item() == 0.0
    assert clipped["pair_base_cosine"].item() == 0.0

    try:
        _pair_target_score_change_diagnostics(
            pre_factor_logp=torch.zeros((1, 2)),
            post_factor_logp=torch.zeros((1, 3)),
            pair_local_delta=torch.ones((1, 2)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned pair score tensors did not fail")

    try:
        _pair_target_score_change_diagnostics(
            pre_factor_logp=torch.zeros((1, 2)),
            post_factor_logp=torch.zeros((1, 2)),
            pair_local_delta=torch.zeros((1, 2)),
        )
    except RuntimeError:
        return
    raise AssertionError("empty pair score target did not fail loudly")


def main():
    tests = [
        test_positive_and_negative_surrogate_directions,
        test_order2_and_order3_target_positions_are_supported,
        test_unrelated_factor_count_cannot_dilute_episode_mass,
        test_transition_denominator_matches_graph_objective_scale,
        test_sparse_pair_population_ignores_unrelated_transitions,
        test_invalid_or_padding_target_fails_loudly,
        test_nonfinite_input_fails_loudly,
        test_disabled_outcome_is_exact_zero,
        test_importance_is_applied_once_and_gradients_are_signed,
        test_zero_transition_denominator_is_explicit,
        test_capture_outcome_public_entry_matches_generic_objective,
        test_optimizer_step_pair_credit_requires_centered_two_sided_mass,
        test_base_population_split_matches_hand_calculation,
        test_base_population_split_survives_float32_cancellation,
        test_base_population_split_zero_and_single_valid_masks,
        test_base_population_split_rejects_nonbinary_provenance,
        test_base_population_diagnostics_do_not_change_training_gradient,
        test_pair_gradient_diagnostics_are_read_only_and_directional,
        test_pair_realized_displacement_matches_hand_calculation,
        test_pair_target_score_change_joins_exact_signed_targets,
        test_pair_diagnostics_fail_loudly_on_invalid_targets,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} outcome factor-loss tests".format(len(tests)))


if __name__ == "__main__":
    main()
