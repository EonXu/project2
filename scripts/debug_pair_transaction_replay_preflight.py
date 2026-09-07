#!/usr/bin/env python
"""Fast transaction-combination replay before a long SDDFG training run.

The two failed run116 attempts did not persist their complete pre-step tensors,
so the checked-in fixture is explicit about being a minimal gradient-geometry
reconstruction.  It preserves the failure predicates and also records the
successful final-v11 candidate+pair+lifecycle transaction at step 176000.

This process never creates a run directory, opens a production CSV, or touches
a production model/optimizer.  Every scenario owns fresh tensors and optimizer
state; the launcher runs it in a separate process before training starts.
"""

from __future__ import print_function

import copy
import json
import os
import sys

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import (  # noqa: E402
    _gradient_tuple_dot,
    _joint_exact_constraint_acceptance,
    _maximize_joint_exact_backtracking_scale,
    _select_joint_exact_progress_direction,
    _select_boundary_progress_seed_members,
    _pair_boundary_deficit_aware_minimum_dots,
    _nonnegative_gradient_dot_with_tolerance,
    _project_gradient_tuple_to_minimum_dots,
    _project_with_current_candidate_priority,
    _strict_gradient_dot_floor,
    _sync_adam_first_moment_to_executed_update,
)


FIXTURE_PATH = os.path.join(
    SCRIPT_DIR, "fixtures", "run116_transaction_replays.json"
)


def _tensor_tuple(values, device):
    return (torch.tensor(values, dtype=torch.float32, device=device),)


def _case_map():
    with open(FIXTURE_PATH, "r") as fixture_file:
        payload = json.load(fixture_file)
    assert payload["schema_version"] == 1
    assert payload["source_run"] == "run116+run118"
    assert payload["exact_failed_tensor_dump_available"] is False
    return dict((case["name"], case) for case in payload["cases"])


def _assert_all_strict(proposed, constraints, reference):
    dots = [
        float(_gradient_tuple_dot(
            proposed, constraint, reference
        ).detach().cpu().item())
        for constraint in constraints
    ]
    assert dots
    assert min(dots) > 0.0, dots
    return dots


def replay_float32_lifecycle_classification(cases, device):
    residual = cases["run116_float32_lifecycle_residual"]
    material = cases["run116_material_lifecycle_reverse_control"]
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)

    residual_result = _nonnegative_gradient_dot_with_tolerance(
        proposed_grads=_tensor_tuple(residual["proposed"], device),
        constraint_grads=_tensor_tuple(residual["constraint"], device),
        reference=reference,
    )
    residual_dot = float(residual_result["dot"].detach().cpu().item())
    residual_tolerance = float(
        residual_result["tolerance"].detach().cpu().item()
    )
    assert residual_dot < 0.0
    assert residual_dot >= -residual_tolerance
    assert bool(residual_result["valid"].item())
    residual_exact_gap = (
        float(residual["exact_signed_margin_after"])
        - float(residual["exact_signed_floor"])
    )
    assert residual_exact_gap >= 0.0

    material_result = _nonnegative_gradient_dot_with_tolerance(
        proposed_grads=_tensor_tuple(material["proposed"], device),
        constraint_grads=_tensor_tuple(material["constraint"], device),
        reference=reference,
    )
    material_dot = float(material_result["dot"].detach().cpu().item())
    material_tolerance = float(
        material_result["tolerance"].detach().cpu().item()
    )
    assert material_dot < -material_tolerance
    assert not bool(material_result["valid"].item())
    material_exact_gap = (
        float(material["exact_signed_margin_after"])
        - float(material["exact_signed_floor"])
    )
    assert material_exact_gap < 0.0
    return {
        "rounding_dot": residual_dot,
        "rounding_tolerance": residual_tolerance,
        "material_dot": material_dot,
        "material_tolerance": material_tolerance,
    }


def replay_current_priority_ordering(cases, device):
    case = cases["run116_current_priority_ordering"]
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    proposed = _tensor_tuple(case["proposed"], device)
    priorities = [
        _tensor_tuple(values, device)
        for values in case["current_priorities"]
    ]
    lifecycle = [
        _tensor_tuple(values, device)
        for values in case["lifecycle_constraints"]
    ]
    before = [
        float(_gradient_tuple_dot(
            proposed, priority, reference
        ).detach().cpu().item())
        for priority in priorities
    ]
    negative = [index for index, value in enumerate(before) if value <= 0.0]
    assert negative == [case["expected_negative_priority_index_before"]]
    projected, accepted, superseded, info = (
        _project_with_current_candidate_priority(
            proposed_grads=proposed,
            candidate_grads=priorities[0],
            lifecycle_constraint_grads=lifecycle,
            reference=reference,
            additional_priority_grads=tuple(priorities[1:]),
        )
    )
    after = _assert_all_strict(projected, priorities, reference)
    assert accepted == [0]
    assert superseded == []
    assert info["current_priority_repair_intervened"] == 1.0
    assert min(after) > 0.0
    return {
        "min_priority_dot_before": min(before),
        "min_priority_dot_after": min(after),
    }


def replay_multitarget_pending(cases, device):
    case = cases["run115_multitarget_pending_equivalent"]
    target_count = int(case["target_count"])
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    target_constraints = []
    for index in range(target_count):
        vector = torch.zeros(
            target_count, dtype=torch.float32, device=device
        )
        vector[index] = 1.0
        target_constraints.append((vector,))
    proposed_vector = torch.tensor(
        [
            (1.0 + 0.05 * index) * (-1.0 if index % 2 else 1.0)
            for index in range(target_count)
        ],
        dtype=torch.float32,
        device=device,
    )
    proposed = (proposed_vector,)
    aggregate = (torch.ones_like(proposed_vector) / float(target_count),)
    constraints = list(target_constraints) + [aggregate]
    floors = [
        float(_strict_gradient_dot_floor(
            proposed_grads=proposed,
            constraint_grads=constraint,
            reference=reference,
        ).detach().cpu().item())
        for constraint in constraints
    ]
    projected, info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=proposed,
        constraint_grads=constraints,
        minimum_dots=floors,
        reference=reference,
        diagnostic_name="run115 multi-target pending replay",
    )
    dots = _assert_all_strict(projected, constraints, reference)
    assert info["intervened"] == 1.0
    assert len(dots[:-1]) == (
        int(case["positive_target_count"])
        + int(case["negative_target_count"])
    )
    return {
        "target_count": target_count,
        "min_target_dot_after": min(dots[:-1]),
        "aggregate_dot_after": dots[-1],
    }


def _combination_case(
        device,
        target_count,
        candidate,
        lifecycle,
        raw_conflict):
    width = 2 + int(target_count)
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    candidate_grad = torch.zeros(width, dtype=torch.float32, device=device)
    candidate_grad[0] = 1.0
    target_grads = []
    for index in range(int(target_count)):
        target = torch.zeros(width, dtype=torch.float32, device=device)
        target[1 + index] = 1.0
        target_grads.append((target,))
    aggregate = (
        sum((item[0] for item in target_grads), torch.zeros(
            width, dtype=torch.float32, device=device
        )) / float(max(1, target_count)),
    )
    proposed = torch.ones(width, dtype=torch.float32, device=device)
    if raw_conflict and target_count:
        proposed[-1] = -1.0
    priorities = []
    if candidate:
        priorities.append((candidate_grad,))
    priorities.append(aggregate)
    priorities.extend(target_grads)
    lifecycle_constraints = []
    if lifecycle:
        lifecycle_grad = torch.zeros(
            width, dtype=torch.float32, device=device
        )
        lifecycle_grad[0] = 1.0
        lifecycle_grad[-1] = 1.0
        lifecycle_constraints.append((lifecycle_grad,))
    projected, _accepted, _superseded, _info = (
        _project_with_current_candidate_priority(
            proposed_grads=(proposed,),
            candidate_grads=priorities[0],
            lifecycle_constraint_grads=lifecycle_constraints,
            reference=reference,
            additional_priority_grads=tuple(priorities[1:]),
        )
    )
    _assert_all_strict(projected, priorities, reference)
    for constraint in lifecycle_constraints:
        check = _nonnegative_gradient_dot_with_tolerance(
            proposed_grads=projected,
            constraint_grads=constraint,
            reference=reference,
        )
        assert bool(check["valid"].item())


def replay_combination_matrix(device):
    combinations = []
    for target_count in (1, 5):
        for candidate in (False, True):
            for lifecycle in (False, True):
                for raw_conflict in (False, True):
                    _combination_case(
                        device=device,
                        target_count=target_count,
                        candidate=candidate,
                        lifecycle=lifecycle,
                        raw_conflict=raw_conflict,
                    )
                    combinations.append(
                        (target_count, candidate, lifecycle, raw_conflict)
                    )

    # Forced/non-actionable evidence has no strict target and therefore must
    # not enter a target Jacobian or pair-direction guard.
    forced_actionable_targets = []
    assert not forced_actionable_targets

    # An unknown zero-Jacobian target is still a contract error.
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    try:
        _project_gradient_tuple_to_minimum_dots(
            proposed_grads=_tensor_tuple([1.0, 0.0], device),
            constraint_grads=[_tensor_tuple([0.0, 0.0], device)],
            minimum_dots=[1.0],
            reference=reference,
            diagnostic_name="unknown zero-Jacobian replay",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown zero-Jacobian target was accepted")

    # Mutually opposite strict current evidence is mathematically infeasible.
    parameter = torch.nn.Parameter(
        torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    parameter_before = parameter.detach().clone()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    cpu_rng_before = torch.get_rng_state().clone()
    numpy_rng_before = np.random.get_state()
    try:
        _project_with_current_candidate_priority(
            proposed_grads=_tensor_tuple([1.0, 0.0], device),
            candidate_grads=_tensor_tuple([1.0, 0.0], device),
            lifecycle_constraint_grads=[],
            reference=reference,
            additional_priority_grads=(
                _tensor_tuple([-1.0, 0.0], device),
            ),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("infeasible current priorities were accepted")
    assert torch.equal(parameter.detach(), parameter_before)
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    return combinations


def replay_poisoned_adam_state_sync(device):
    parameter = torch.nn.Parameter(
        torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([parameter], lr=1.0e-2)
    for _ in range(8):
        optimizer.zero_grad()
        parameter.grad = torch.tensor(
            [-1.0, 0.0], dtype=torch.float32, device=device
        )
        optimizer.step()
    before = parameter.detach().clone()
    optimizer.zero_grad()
    parameter.grad = torch.tensor(
        [1.0, 0.0], dtype=torch.float32, device=device
    )
    optimizer.step()
    raw_delta = (parameter.detach() - before).detach().clone()
    raw_descent = (-raw_delta,)
    candidate = _tensor_tuple([1.0, 0.0], device)
    pair_target = _tensor_tuple([-1.0, 2.0], device)
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    assert float(_gradient_tuple_dot(
        raw_descent, candidate, reference
    ).detach().cpu().item()) < 0.0
    safe, _accepted, _superseded, _info = (
        _project_with_current_candidate_priority(
            proposed_grads=raw_descent,
            candidate_grads=candidate,
            lifecycle_constraint_grads=[
                _tensor_tuple([0.0, 1.0], device)
            ],
            reference=reference,
            additional_priority_grads=(pair_target,),
        )
    )
    _assert_all_strict(safe, [candidate, pair_target], reference)
    with torch.no_grad():
        parameter.copy_(before - safe[0])
    sync = _sync_adam_first_moment_to_executed_update(
        optimizer=optimizer,
        parameters=[parameter],
        parameter_before_step=[before],
        raw_parameter_deltas=[raw_delta],
    )
    assert sync["raw_reconstruction_error_ratio"] <= 1.0
    assert sync["safe_reconstruction_error_ratio"] <= 1.0
    return sync


def validate_recorded_final_v11_transaction(cases):
    case = cases["run116_final_v11_candidate_pair_lifecycle"]
    assert case["candidate_active"] is True
    assert case["pair_target_count"] == 4
    assert case["lifecycle_gradient_projection_intervened"] is True
    assert case["lifecycle_actual_update_correction_intervened"] is True
    assert (
        case["pair_target_actual_min_descent_dot_after"]
        > case["pair_target_actual_min_descent_dot_before"]
    )
    assert case["score_correct_target_count"] == case["pair_target_count"]
    assert case["score_reverse_target_count"] == 0
    assert case["score_zero_target_count"] == 0
    assert case["rollback_occurred"] is False


def replay_run118_joint_exact_lifecycle(cases, device):
    case = cases["run118_boundary_after_step_lifecycle_violation"]
    boundary_change = torch.tensor(
        [case["boundary_signed_changes"]],
        dtype=torch.float32,
        device=device,
    )
    actionable = torch.ones_like(boundary_change)
    lifecycle_floor = torch.tensor(
        [case["lifecycle_signed_floor"]],
        dtype=torch.float32,
        device=device,
    )
    lifecycle_tolerance = torch.tensor(
        [case["lifecycle_tolerance"]],
        dtype=torch.float32,
        device=device,
    )
    lifecycle_mask = torch.ones_like(lifecycle_floor)
    full = _joint_exact_constraint_acceptance(
        signed_boundary_change=boundary_change,
        actionable_pair_mask=actionable,
        candidate_loss_change=torch.tensor(
            case["candidate_loss_change"],
            dtype=torch.float32,
            device=device,
        ),
        lifecycle_signed_margin=torch.tensor(
            [case["full_step_lifecycle_signed_margin"]],
            dtype=torch.float32,
            device=device,
        ),
        lifecycle_signed_floor=lifecycle_floor,
        lifecycle_target_mask=lifecycle_mask,
        lifecycle_tolerance=lifecycle_tolerance,
    )
    assert case["optimizer_step_occurred"] is True
    assert case["transaction_row_persisted"] is False
    assert not full["valid"]
    assert full["boundary_valid"]
    assert full["candidate_valid"]
    assert not full["lifecycle_valid"]
    assert (
        full["lifecycle_violation_count"]
        == case["observed_exact_lifecycle_violation_count"]
    )
    backtracked = _joint_exact_constraint_acceptance(
        signed_boundary_change=0.5 * boundary_change,
        actionable_pair_mask=actionable,
        candidate_loss_change=torch.tensor(
            0.5 * case["candidate_loss_change"],
            dtype=torch.float32,
            device=device,
        ),
        lifecycle_signed_margin=torch.tensor(
            [case["backtracked_lifecycle_signed_margin"]],
            dtype=torch.float32,
            device=device,
        ),
        lifecycle_signed_floor=lifecycle_floor,
        lifecycle_target_mask=lifecycle_mask,
        lifecycle_tolerance=lifecycle_tolerance,
    )
    assert backtracked["valid"]
    return {
        "full_valid": float(full["valid"]),
        "backtracked_valid": float(backtracked["valid"]),
    }


def replay_run118_boundary_deficit_allocation(cases, device):
    case = cases["run118_boundary_deficit_epoch_168800_3"]
    target_count = len(case["target_signs"])
    identity = torch.eye(target_count, dtype=torch.float32, device=device)
    constraints = tuple((identity[index],) for index in range(target_count))
    signs = torch.tensor(
        [case["target_signs"]], dtype=torch.float32, device=device
    )
    weights = torch.tensor(
        case["target_weights"], dtype=torch.float32, device=device
    )
    signed_margins = torch.tensor(
        [case["pre_signed_margins"]], dtype=torch.float32, device=device
    )
    pre_boundary = signs * signed_margins
    info = _pair_boundary_deficit_aware_minimum_dots(
        boundary_target_grads=constraints,
        base_minimum_dots=case["observed_signed_margin_improvements"],
        pair_target_weights=weights,
        pre_boundary=pre_boundary,
        pair_local_delta=signs * weights.view(1, -1),
        target_candidate_index=torch.arange(
            target_count, dtype=torch.float32, device=device
        ).view(1, target_count),
        proposed_descent=(torch.ones(
            target_count, dtype=torch.float32, device=device
        ),),
        reference=torch.tensor(0.0, dtype=torch.float32, device=device),
    )
    expected_index = int(case["expected_affordable_crossing_target"])
    assert info["deficit_target_count"] == float(
        case["expected_deficit_target_count"]
    )
    assert info["affordable_crossing_count"] == 1.0
    assert float(
        info["crossing_affordable"][expected_index].detach().cpu().item()
    ) == 1.0
    assert (
        info["minimum_dots"][expected_index]
        > -float(case["pre_signed_margins"][expected_index])
    )
    assert abs(
        sum(info["minimum_dots"])
        - sum(case["observed_signed_margin_improvements"])
    ) <= 1.0e-6
    assert info["budget_conservation_valid"] == 1.0
    return {
        "affordable_crossing_count": info["affordable_crossing_count"],
        "allocated_budget": info["allocated_budget"],
        "budget_conservation_valid": info["budget_conservation_valid"],
    }


def replay_run122_nearest_member_identity_progress(device):
    """Replay aggregate masking seen in run122 with one canonical identity."""
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    constraints = (
        _tensor_tuple([1.0, 0.0], device),
        _tensor_tuple([-0.2, 1.0], device),
        _tensor_tuple([-0.2, 1.0], device),
    )
    pair_delta = torch.tensor(
        [[0.02], [0.02], [0.02]], dtype=torch.float32, device=device
    )
    info = _pair_boundary_deficit_aware_minimum_dots(
        boundary_target_grads=constraints,
        base_minimum_dots=(0.0001, 0.0001, 0.0660),
        pair_target_weights=pair_delta.abs().view(-1),
        pre_boundary=torch.tensor(
            [[-2.50], [-0.32], [-0.26]],
            dtype=torch.float32,
            device=device,
        ),
        pair_local_delta=pair_delta,
        target_candidate_index=torch.zeros(
            (3, 1), dtype=torch.float32, device=device
        ),
        proposed_descent=_tensor_tuple([0.01, 0.01], device),
        reference=reference,
    )
    assert info["identity_group_member_indices"] == ((2,),)
    assert info["identity_group_progress_member_flags"] == (0, 0, 1)
    group_constraint = constraints[2]
    safe, _ = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=_tensor_tuple([0.01, 0.01], device),
        constraint_grads=list(constraints) + [group_constraint],
        minimum_dots=(
            list(info["projection_minimum_dots"])
            + list(info["identity_group_minimum_dots"])
        ),
        reference=reference,
        diagnostic_name="run122 nearest-member boundary replay",
    )
    row_dots = _assert_all_strict(safe, constraints, reference)
    group_dot = float(_gradient_tuple_dot(
        safe, group_constraint, reference
    ).detach().cpu().item())
    assert group_dot >= info["identity_group_minimum_dots"][0] - 1.0e-6
    assert abs(info["allocated_budget"] - 0.0662) <= 1.0e-6
    return {
        "row_min_dot": min(row_dots),
        "group_dot": group_dot,
        "group_budget": info["identity_group_minimum_dots"][0],
    }


def replay_run123_maximum_joint_exact_safe_scale(device):
    """Replay run123's coarse-scale loss with all exact gates represented."""
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    threshold = 0.4375

    def evaluate(scale):
        scale_tensor = reference.new_tensor(float(scale))
        signed_boundary = torch.stack((
            0.02 * scale_tensor,
            reference.new_tensor(threshold) - scale_tensor,
        )).reshape(1, 2)
        signed_exact = torch.stack((
            0.03 * scale_tensor,
            0.01 * scale_tensor,
        )).reshape(1, 2)
        lifecycle_floor = reference.new_tensor([[0.2]])
        lifecycle_margin = 0.2 + 0.01 * scale_tensor.reshape(1, 1)
        info = _joint_exact_constraint_acceptance(
            signed_boundary_change=signed_boundary,
            actionable_pair_mask=torch.ones_like(signed_boundary),
            signed_exact_score_change=signed_exact,
            candidate_loss_change=-0.01 * scale_tensor,
            lifecycle_signed_margin=lifecycle_margin,
            lifecycle_signed_floor=lifecycle_floor,
            lifecycle_target_mask=torch.ones_like(lifecycle_floor),
            lifecycle_tolerance=torch.full_like(lifecycle_floor, 1.0e-6),
        )
        info["scale"] = float(scale)
        return info

    result = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate,
        max_halvings=20,
        refinement_steps=12,
    )
    assert result["valid"]
    assert result["final_scale"] > 0.43
    assert result["final_scale"] < threshold
    assert result["invalid_upper_scale"] >= threshold
    assert result["info"]["boundary_valid"]
    assert result["info"]["exact_score_valid"]
    assert result["info"]["candidate_valid"]
    assert result["info"]["lifecycle_valid"]
    return result


def replay_run124_curved_progress_direction_selection(device):
    """A fixed-direction maximum scale can lose to a safer curved direction."""
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)

    def evaluator(direction_progress, safe_limit):
        def evaluate(scale):
            scale_tensor = reference.new_tensor(float(scale))
            signed_boundary = torch.stack((
                direction_progress * scale_tensor,
                reference.new_tensor(float(safe_limit)) - scale_tensor,
            )).reshape(1, 2)
            info = _joint_exact_constraint_acceptance(
                signed_boundary_change=signed_boundary,
                actionable_pair_mask=torch.ones_like(signed_boundary),
                signed_exact_score_change=torch.full_like(
                    signed_boundary, 0.01
                ) * scale_tensor,
            )
            info["progress_min_completion"] = float(
                signed_boundary.reshape(-1)[0].detach().cpu().item()
            )
            info["progress_mean_completion"] = info[
                "progress_min_completion"
            ]
            return info
        return evaluate

    fixed = evaluator(1.0, 0.26)
    curved = evaluator(0.60, 0.90)
    result = _select_joint_exact_progress_direction(
        direction_evaluators=((1.0, fixed), (0.5, curved)),
        max_halvings=20,
        refinement_steps=12,
    )
    assert result["valid"]
    assert result["selected_candidate_label"] == 0.5
    assert result["final_scale"] > 0.89
    assert result["info"]["progress_min_completion"] > 0.53
    assert result["invalid_upper_info"][
        "limiting_constraint_type"
    ] == "boundary"
    assert result["invalid_upper_info"]["limiting_target_ordinal"] == 1
    return result


def replay_run125_progress_tangent_direction(device):
    """Equivalent geometry: progress Jacobian supplies a distinct safe ray.

    The original run125 tensors were not serialized.  This minimal geometry
    preserves its relevant contract: one nearest-member progress floor, one
    non-progress non-regression boundary, and one exact-score halfspace.  It
    verifies that the new proposal is not a rescaling of the Adam projection
    and that both directions enter the real affine projection helper.
    """
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    adam = _tensor_tuple([0.0, 1.0], device)
    progress = _tensor_tuple([1.0, 0.0], device)
    constraints = (
        progress,
        _tensor_tuple([-0.2, 1.0], device),
        _tensor_tuple([0.1, 1.0], device),
    )
    floors = (0.4, 0.1, 0.1)
    adam_safe, adam_info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=adam,
        constraint_grads=constraints,
        minimum_dots=floors,
        reference=reference,
        diagnostic_name="run125 Adam progress direction replay",
    )
    balanced = _tensor_tuple([0.5, 0.5], device)
    tangent_safe, tangent_info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=balanced,
        constraint_grads=constraints,
        minimum_dots=floors,
        reference=reference,
        diagnostic_name="run125 progress-tangent direction replay",
    )
    _assert_all_strict(adam_safe, constraints, reference)
    _assert_all_strict(tangent_safe, constraints, reference)
    adam_norm = torch.sqrt(_gradient_tuple_dot(
        adam_safe, adam_safe, reference
    ))
    tangent_norm = torch.sqrt(_gradient_tuple_dot(
        tangent_safe, tangent_safe, reference
    ))
    cosine = _gradient_tuple_dot(
        adam_safe, tangent_safe, reference
    ) / (adam_norm * tangent_norm)
    cosine_value = float(cosine.detach().cpu().item())
    assert cosine_value < 0.999
    assert adam_info["active_constraint_indices"] == (0,)
    assert isinstance(tangent_info["active_constraint_indices"], tuple)
    return {
        "cosine": cosine_value,
        "adam_active": adam_info["active_constraint_count"],
        "tangent_active": tangent_info["active_constraint_count"],
    }


def replay_run126_deficit_only_seed_and_candidate_order(device):
    """Exclude zero-budget identities and prove candidate-order isolation."""
    seed = _select_boundary_progress_seed_members(
        identity_group_member_indices=((0,), (1, 2)),
        progress_member_flags=(1, 1, 0),
        identity_group_extra_budgets=(0.0, 0.4, 0.0),
    )
    assert seed["selected_member_ordinals"] == (1,)
    assert seed["excluded_zero_budget_member_ordinals"] == (0,)

    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    parameter = torch.nn.Parameter(torch.zeros(
        2, dtype=torch.float32, device=device
    ))
    optimizer = torch.optim.Adam((parameter,), lr=1.0e-3)
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda" else None
    )

    def evaluator(progress, safe_limit):
        def evaluate(scale):
            scale_value = reference.new_tensor(float(scale))
            with torch.no_grad():
                parameter.copy_(torch.stack((
                    reference.new_tensor(float(progress)),
                    reference.new_tensor(1.0),
                )) * scale_value)
            boundary = torch.stack((
                parameter[0].detach(),
                reference.new_tensor(float(safe_limit))
                - parameter[1].detach(),
            )).reshape(1, 2)
            info = _joint_exact_constraint_acceptance(
                signed_boundary_change=boundary,
                actionable_pair_mask=torch.ones_like(boundary),
                signed_exact_score_change=(
                    torch.full_like(boundary, 0.01) * scale_value
                ),
            )
            info["progress_target_present"] = True
            info["progress_worst_actual"] = float(
                boundary.reshape(-1)[0].detach().cpu().item()
            )
            info["progress_worst_required"] = 1.0
            info["progress_min_completion"] = info[
                "progress_worst_actual"
            ]
            info["progress_mean_completion"] = info[
                "progress_worst_actual"
            ]
            return info
        return evaluate

    fixed = ("fixed", evaluator(1.0, 0.26))
    curved = ("curved", evaluator(0.60, 0.90))
    normal = _select_joint_exact_progress_direction(
        direction_evaluators=(fixed, curved),
        max_halvings=20,
        refinement_steps=12,
    )
    normal_parameter = parameter.detach().clone()
    with torch.no_grad():
        parameter.zero_()
    reversed_order = _select_joint_exact_progress_direction(
        direction_evaluators=(curved, fixed),
        max_halvings=20,
        refinement_steps=12,
    )
    assert normal["selected_candidate_label"] == "curved"
    assert reversed_order["selected_candidate_label"] == "curved"
    assert abs(
        normal["final_scale"] - reversed_order["final_scale"]
    ) <= 1.0e-12
    assert abs(
        normal["info"]["progress_worst_actual"]
        - reversed_order["info"]["progress_worst_actual"]
    ) <= 1.0e-12
    assert torch.equal(normal_parameter, parameter.detach())
    assert optimizer.state_dict() == optimizer_state_before
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    return {
        "selected_seed": seed["selected_member_ordinals"][0],
        "excluded_seed": seed[
            "excluded_zero_budget_member_ordinals"
        ][0],
        "selected_scale": normal["final_scale"],
    }


def replay_run127_deficit_closure_scale_expansion(device):
    """Replay run127's observed scale cap with its logged deficit ratio.

    The production tensors were not checkpointed per transaction, so this
    fixture uses seq737's exact logged pre-deficit, original requirement, and
    scale-1 actual progress.  It verifies the code defect exposed by those
    values: a safe scale-1 result must continue through exact-forward expansion
    instead of being mislabeled as its own unsafe upper bound.
    """
    reference = torch.tensor(0.0, dtype=torch.float32, device=device)
    pre_deficit = 0.1089094877243042
    original_required = 0.0034369535424048
    actual_at_one = 0.0034365653991699
    scale_limit = pre_deficit / original_required

    def evaluator(unsafe_scale):
        def evaluate(scale):
            scale_tensor = reference.new_tensor(float(scale))
            signed_boundary = torch.stack((
                actual_at_one * scale_tensor,
                reference.new_tensor(float(unsafe_scale)) - scale_tensor,
            )).reshape(1, 2)
            info = _joint_exact_constraint_acceptance(
                signed_boundary_change=signed_boundary,
                actionable_pair_mask=torch.ones_like(signed_boundary),
                signed_exact_score_change=(
                    torch.full_like(signed_boundary, 0.01) * scale_tensor
                ),
            )
            actual = float(
                signed_boundary.reshape(-1)[0].detach().cpu().item()
            )
            info["progress_target_present"] = True
            info["progress_worst_actual"] = actual
            info["progress_worst_required"] = original_required
            info["progress_min_completion"] = actual / original_required
            info["progress_mean_completion"] = info[
                "progress_min_completion"
            ]
            return info
        return evaluate

    capped = evaluator(scale_limit + 1.0)(1.0)
    assert capped["valid"]
    assert capped["progress_worst_actual"] < pre_deficit

    expanded = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluator(scale_limit + 1.0),
        maximum_scale=scale_limit,
        max_expansions=20,
        max_halvings=20,
        refinement_steps=12,
    )
    assert expanded["valid"]
    assert expanded["final_scale"] > 30.0
    assert abs(expanded["final_scale"] - scale_limit) <= 1.0e-12
    assert not expanded["unsafe_upper_present"]
    assert expanded["expansion_count"] > 0
    assert (
        expanded["info"]["progress_worst_actual"]
        > 30.0 * capped["progress_worst_actual"]
    )

    bracketed = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluator(12.75),
        maximum_scale=scale_limit,
        max_expansions=20,
        max_halvings=20,
        refinement_steps=12,
    )
    assert bracketed["valid"]
    assert bracketed["unsafe_upper_present"]
    assert bracketed["final_scale"] < 12.75
    assert bracketed["invalid_upper_scale"] >= 12.75
    assert bracketed["invalid_upper_info"][
        "limiting_constraint_type"
    ] == "boundary"
    assert bracketed["invalid_upper_info"][
        "limiting_target_ordinal"
    ] == 1

    parameter = torch.nn.Parameter(torch.zeros(
        2, dtype=torch.float32, device=device
    ))
    optimizer = torch.optim.Adam((parameter,), lr=1.0e-3)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda" else None
    )

    def mutable_direction(progress, unsafe_scale):
        def evaluate(scale):
            scale_tensor = reference.new_tensor(float(scale))
            with torch.no_grad():
                parameter.copy_(torch.stack((
                    reference.new_tensor(float(progress)) * scale_tensor,
                    scale_tensor,
                )))
            boundary = torch.stack((
                parameter[0].detach(),
                reference.new_tensor(float(unsafe_scale))
                - parameter[1].detach(),
            )).reshape(1, 2)
            info = _joint_exact_constraint_acceptance(
                signed_boundary_change=boundary,
                actionable_pair_mask=torch.ones_like(boundary),
                signed_exact_score_change=(
                    torch.full_like(boundary, 0.01) * scale_tensor
                ),
            )
            actual = float(parameter[0].detach().cpu().item())
            info["progress_target_present"] = True
            info["progress_worst_actual"] = actual
            info["progress_worst_required"] = 1.0
            info["progress_min_completion"] = actual
            info["progress_mean_completion"] = actual
            return info
        return evaluate

    short = ("short", mutable_direction(1.0, 2.6))
    long = ("long", mutable_direction(0.8, 4.5))
    normal = _select_joint_exact_progress_direction(
        direction_evaluators=(short, long),
        maximum_scale=4.0,
        max_expansions=20,
    )
    normal_parameter = parameter.detach().clone()
    with torch.no_grad():
        parameter.zero_()
    reversed_order = _select_joint_exact_progress_direction(
        direction_evaluators=(long, short),
        maximum_scale=4.0,
        max_expansions=20,
    )
    assert normal["selected_candidate_label"] == "long"
    assert reversed_order["selected_candidate_label"] == "long"
    assert abs(
        normal["final_scale"] - reversed_order["final_scale"]
    ) <= 1.0e-12
    assert torch.equal(normal_parameter, parameter.detach())
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    return {
        "scale_limit": scale_limit,
        "expanded_scale": expanded["final_scale"],
        "bracketed_scale": bracketed["final_scale"],
        "order_selected": normal["selected_candidate_label"],
    }


def run_device(device):
    cases = _case_map()
    residual = replay_float32_lifecycle_classification(cases, device)
    ordering = replay_current_priority_ordering(cases, device)
    multitarget = replay_multitarget_pending(cases, device)
    combinations = replay_combination_matrix(device)
    sync = replay_poisoned_adam_state_sync(device)
    run118 = replay_run118_joint_exact_lifecycle(cases, device)
    run118_deficit = replay_run118_boundary_deficit_allocation(cases, device)
    run122_identity = replay_run122_nearest_member_identity_progress(device)
    run123_scale = replay_run123_maximum_joint_exact_safe_scale(device)
    run124_direction = replay_run124_curved_progress_direction_selection(
        device
    )
    run125_tangent = replay_run125_progress_tangent_direction(device)
    run126_seed = replay_run126_deficit_only_seed_and_candidate_order(device)
    run127_scale = replay_run127_deficit_closure_scale_expansion(device)
    validate_recorded_final_v11_transaction(cases)
    print(
        "PASS run116 transaction replay on {}: combinations={}, "
        "rounding_dot={:.9g}, rounding_tolerance={:.9g}, "
        "priority_before={:.9g}, priority_after={:.9g}, "
        "multitarget_min={:.9g}, state_sync_safe_error_ratio={:.9g}, "
        "run118_full_valid={:.0f}, run118_backtracked_valid={:.0f}, "
        "run118_affordable_crossing_count={:.0f}, "
        "run122_identity_member_dot={:.9g}, "
        "run123_refined_scale={:.9g}, "
        "run124_direction_scale={:.9g}, "
        "run125_tangent_cosine={:.9g}, "
        "run126_selected_seed={}, run126_excluded_seed={}, "
        "run126_selected_scale={:.9g}, run127_scale_limit={:.9g}, "
        "run127_bracketed_scale={:.9g}".format(
            device,
            len(combinations),
            residual["rounding_dot"],
            residual["rounding_tolerance"],
            ordering["min_priority_dot_before"],
            ordering["min_priority_dot_after"],
            multitarget["min_target_dot_after"],
            sync["safe_reconstruction_error_ratio"],
            run118["full_valid"],
            run118["backtracked_valid"],
            run118_deficit["affordable_crossing_count"],
            run122_identity["group_dot"],
            run123_scale["final_scale"],
            run124_direction["final_scale"],
            run125_tangent["cosine"],
            run126_seed["selected_seed"],
            run126_seed["excluded_seed"],
            run126_seed["selected_scale"],
            run127_scale["scale_limit"],
            run127_scale["bracketed_scale"],
        )
    )


def main():
    cpu_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    run_device(torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0]
    assert np.array_equal(after_numpy[1], numpy_rng[1])
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state().clone()
        run_device(torch.device("cuda"))
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    else:
        print("SKIP run116 CUDA transaction replay: unavailable")
    print("All run116 transaction replay preflight checks passed")


if __name__ == "__main__":
    main()
