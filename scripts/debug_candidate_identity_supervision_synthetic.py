#!/usr/bin/env python
"""Production-path checks for exact candidate-only capture supervision."""

from __future__ import print_function

import copy
import os
import sys
import tempfile
import types

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import wandb  # noqa: F401
except ImportError:
    # The synthetic checks use only pure runner helpers.  Keep local CPU test
    # environments without the optional experiment client usable.
    sys.modules["wandb"] = types.ModuleType("wandb")

from algorithms.sddfg.r_sddfg import (
    R_SDDFG,
    _candidate_identity_transaction_trace_rows,
    _candidate_lifecycle_behavioral_progress_mask,
    _candidate_rank_decomposition,
    _candidate_target_bearing_update_diagnostic,
    _adjacency_optimizer_checkpoint_state,
    _clear_parameter_gradients_to_none,
    _compose_adj_candidate_objective,
    _load_adjacency_optimizer_checkpoint_state,
    _project_gradients_onto_nonincreasing_halfspaces,
    _project_with_current_candidate_priority,
    _reshape_replay_population_provenance,
    _restore_scaled_transaction_parameter_state,
    _select_candidate_lifecycle_committed_info,
    _select_adjacency_optimizer,
    _successful_candidate_capture_boundary_diagnostics,
    _sync_adam_first_moment_to_executed_update,
    compute_capture_candidate_identity_active_competitor_loss,
)
from utils.candidate_score_sensitivity import (
    candidate_score_to_rank_sensitivity,
)
from algorithms.sddfg.algorithm.adj_generator import Adj_Generator
from runner.base_runner import (
    RecRunner,
    _CANDIDATE_IDENTITY_TRANSACTION_CSV_FIELDS,
    _build_candidate_identity_transaction_rows,
    _build_sddfg_checkpoint_metadata,
    _require_sddfg_optimizer_checkpoint,
    _validate_sddfg_checkpoint_metadata,
)


def _loss(margins, delta, reference=None, valid=None, transitions=None):
    margins = torch.as_tensor(margins, dtype=torch.float32)
    delta = torch.as_tensor(delta, dtype=torch.float32)
    if reference is None:
        reference = torch.zeros_like(margins)
    else:
        reference = torch.as_tensor(reference, dtype=torch.float32)
    if valid is None:
        valid = torch.ones_like(margins)
    else:
        valid = torch.as_tensor(valid, dtype=torch.float32)
    if transitions is None:
        transitions = torch.ones((margins.shape[0], 1))
    else:
        transitions = torch.as_tensor(transitions, dtype=torch.float32)
    return compute_capture_candidate_identity_active_competitor_loss(
        margins, reference, delta, valid, transitions
    )


def test_strict_pair_standard_exact_replay_restores_complete_origin():
    """Scale zero must restore every standard-Adam transaction parameter."""
    pair_sensitive = torch.nn.Parameter(torch.tensor([0.0]))
    base_only = torch.nn.Parameter(torch.tensor([0.0]))
    parameters = [pair_sensitive, base_only]
    complete_before = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        pair_sensitive.copy_(torch.tensor([0.25]))
        base_only.copy_(torch.tensor([-0.50]))
    complete_displacements = [
        parameter.detach() - before
        for parameter, before in zip(parameters, complete_before)
    ]

    # This is the old hybrid-origin replay: only the parameter with a
    # first-order pair gradient is restored.  The unrelated standard-Adam
    # parameter remains at its post-step value and can invalidate an exact
    # nonlinear boundary even at scale zero.
    _restore_scaled_transaction_parameter_state(
        parameters=parameters,
        parameter_before_step=[complete_before[0], None],
        parameter_displacements=[complete_displacements[0], None],
        scale=0.0,
        require_complete=False,
    )
    assert torch.equal(pair_sensitive.detach(), complete_before[0])
    assert not torch.equal(base_only.detach(), complete_before[1])

    with torch.no_grad():
        for parameter, before, displacement in zip(
                parameters, complete_before, complete_displacements):
            parameter.copy_(before + displacement)
    _restore_scaled_transaction_parameter_state(
        parameters=parameters,
        parameter_before_step=complete_before,
        parameter_displacements=complete_displacements,
        scale=0.0,
        require_complete=True,
    )
    assert all(
        torch.equal(parameter.detach(), before)
        for parameter, before in zip(parameters, complete_before)
    )

    _restore_scaled_transaction_parameter_state(
        parameters=parameters,
        parameter_before_step=complete_before,
        parameter_displacements=complete_displacements,
        scale=0.5,
        require_complete=True,
    )
    assert torch.allclose(pair_sensitive.detach(), torch.tensor([0.125]))
    assert torch.allclose(base_only.detach(), torch.tensor([-0.25]))

    # A diagnostic ray can be non-finite and must be rejected for positive
    # scale, but it must never make the bit-exact scale-zero origin non-finite.
    nonfinite_displacements = [
        torch.full_like(pair_sensitive, float("nan")),
        torch.full_like(base_only, float("inf")),
    ]
    _restore_scaled_transaction_parameter_state(
        parameters=parameters,
        parameter_before_step=complete_before,
        parameter_displacements=nonfinite_displacements,
        scale=0.0,
        require_complete=True,
    )
    assert all(
        torch.equal(parameter.detach(), before)
        for parameter, before in zip(parameters, complete_before)
    )

    try:
        _restore_scaled_transaction_parameter_state(
            parameters=parameters,
            parameter_before_step=[complete_before[0], None],
            parameter_displacements=[complete_displacements[0], None],
            scale=0.0,
            require_complete=True,
        )
    except RuntimeError as error:
        assert "sparse transaction origin" in str(error)
    else:
        raise AssertionError("standard exact replay accepted a sparse origin")


def test_replay_population_provenance_uses_three_column_training_schema():
    provenance_batch = torch.tensor([
        [
            [1.0, 4.0, 0.0],
            [1.0, 4.0, 1.0],
        ],
        [
            [0.0, 7.0, 0.0],
            [0.0, 7.0, 1.0],
        ],
    ])
    decoded = _reshape_replay_population_provenance(
        provenance_batch,
        batch_size=4,
    )
    assert decoded.shape == (4, 3)
    assert torch.equal(
        decoded,
        torch.tensor([
            [1.0, 4.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 7.0, 0.0],
            [0.0, 7.0, 1.0],
        ]),
    )
    try:
        _reshape_replay_population_provenance(
            provenance_batch[..., :2],
            batch_size=4,
        )
    except ValueError as error:
        assert "[4, 3]" in str(error)
    else:
        raise AssertionError(
            "legacy two-column replay provenance did not fail loudly"
        )


def test_positive_and_negative_targets_have_opposite_gradients():
    positive_margin = torch.tensor([[-0.5]], requires_grad=True)
    positive = compute_capture_candidate_identity_active_competitor_loss(
        positive_margin,
        torch.zeros_like(positive_margin),
        torch.tensor([[0.2]]),
        torch.ones_like(positive_margin),
        torch.ones((1, 1)),
    )
    positive["loss"].backward()
    assert positive_margin.grad[0, 0].item() < 0.0

    negative_margin = torch.tensor([[0.5]], requires_grad=True)
    negative = compute_capture_candidate_identity_active_competitor_loss(
        negative_margin,
        torch.zeros_like(negative_margin),
        torch.tensor([[-0.2]]),
        torch.ones_like(negative_margin),
        torch.ones((1, 1)),
    )
    negative["loss"].backward()
    assert negative_margin.grad[0, 0].item() > 0.0


def test_boundary_anchored_objective_is_symmetric_for_wrong_side_references():
    """Equal wrong-side distances must produce equal opposite gradients."""
    behavior_logits = torch.tensor([[-4.0], [4.0]])
    current_logits = behavior_logits.clone().requires_grad_(True)
    result = compute_capture_candidate_identity_active_competitor_loss(
        candidate_competitor_margins=current_logits,
        candidate_reference_margins=behavior_logits,
        candidate_identity_delta=torch.tensor([[0.2], [-0.2]]),
        candidate_valid_mask=torch.ones_like(current_logits),
        transition_mask=torch.ones((2, 1)),
    )
    result["loss"].backward()
    assert torch.allclose(
        current_logits.grad[0, 0].abs(),
        current_logits.grad[1, 0].abs(),
        atol=1e-7,
    )
    assert current_logits.grad[0, 0].item() < 0.0
    assert current_logits.grad[1, 0].item() > 0.0


def test_boundary_anchored_objective_requires_positive_active_insertion():
    behavior_logits = torch.tensor([[-4.0]])
    current_logits = behavior_logits.clone().requires_grad_(True)
    result = compute_capture_candidate_identity_active_competitor_loss(
        candidate_competitor_margins=current_logits,
        candidate_reference_margins=behavior_logits,
        candidate_identity_delta=torch.tensor([[0.2]]),
        candidate_valid_mask=torch.ones_like(current_logits),
        transition_mask=torch.ones((1, 1)),
    )
    expected = torch.tensor(0.8)
    assert torch.allclose(result["loss"], expected)
    result["loss"].backward()
    assert torch.allclose(
        current_logits.grad[0, 0],
        torch.tensor(-0.2),
    )


def test_boundary_anchored_objective_crosses_both_active_boundaries():
    behavior_logits = torch.tensor([[-4.0], [4.0]])
    current_logits = behavior_logits.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([current_logits], lr=2.5)
    for _ in range(6):
        optimizer.zero_grad()
        result = compute_capture_candidate_identity_active_competitor_loss(
            candidate_competitor_margins=current_logits,
            candidate_reference_margins=behavior_logits,
            candidate_identity_delta=torch.tensor([[1.0], [-1.0]]),
            candidate_valid_mask=torch.ones_like(current_logits),
            transition_mask=torch.ones((2, 1)),
        )
        result["loss"].backward()
        optimizer.step()
    assert current_logits[0, 0].item() > 0.0
    assert current_logits[1, 0].item() < 0.0


def test_boundary_anchored_objective_preserves_better_reference():
    behavior_logits = torch.tensor([[2.0], [-3.0]])
    current_logits = behavior_logits.clone().requires_grad_(True)
    result = compute_capture_candidate_identity_active_competitor_loss(
        candidate_competitor_margins=current_logits,
        candidate_reference_margins=behavior_logits,
        candidate_identity_delta=torch.tensor([[0.2], [-0.2]]),
        candidate_valid_mask=torch.ones_like(current_logits),
        transition_mask=torch.ones((2, 1)),
    )
    assert result["positive_loss"].item() == 0.0
    assert result["negative_loss"].item() == 0.0


def test_boundary_anchored_reference_is_not_optimized():
    cases = [
        (-4.0, -4.0, 0.2),
        (2.0, 2.0, 0.2),
        (4.0, 4.0, -0.2),
        (-3.0, -3.0, -0.2),
    ]
    for current_value, reference_value, delta_value in cases:
        current_logits = torch.tensor(
            [[current_value]], requires_grad=True
        )
        behavior_logits = torch.tensor(
            [[reference_value]], requires_grad=True
        )
        result = compute_capture_candidate_identity_active_competitor_loss(
            candidate_competitor_margins=current_logits,
            candidate_reference_margins=behavior_logits,
            candidate_identity_delta=torch.tensor([[delta_value]]),
            candidate_valid_mask=torch.ones_like(current_logits),
            transition_mask=torch.ones((1, 1)),
        )
        result["loss"].backward()
        assert behavior_logits.grad is None
        assert not result["reference_margin"].requires_grad
        wrong_side = delta_value * current_value < 0.0
        expected_sign = -1.0 if delta_value > 0.0 else 1.0
        if wrong_side:
            assert torch.sign(current_logits.grad[0, 0]).item() == expected_sign
        else:
            assert current_logits.grad[0, 0].item() == 0.0


def test_boundary_anchored_competitor_definition_is_exact():
    margins = torch.tensor([[-3.0], [-3.0]], requires_grad=True)
    references = torch.tensor([[-4.0], [-4.0]])
    delta = torch.tensor([[0.2], [-0.2]])
    result = compute_capture_candidate_identity_active_competitor_loss(
        margins,
        references,
        delta,
        torch.ones_like(margins),
        torch.ones((2, 1)),
    )
    expected_positive = torch.tensor(0.3)
    expected_negative = torch.tensor(0.1)
    assert torch.allclose(result["positive_loss"], expected_positive)
    assert torch.allclose(result["negative_loss"], expected_negative)
    assert torch.allclose(
        result["loss"], expected_positive + expected_negative
    )
    result["loss"].backward()
    assert margins.grad[0, 0].item() < 0.0
    assert margins.grad[1, 0].item() > 0.0


def _ranking_graph():
    graph = object.__new__(Adj_Generator)
    graph.sparsity = 1.0
    graph.num_factor = 2
    graph.highest_orders = 2
    graph.require_connected = True
    graph.current_sampling_temperature = 1.0
    graph.current_min_order3_ratio = 0.0
    graph.current_max_order3_ratio = 0.0
    graph.min_pair_ratio = 0.0
    graph.order3_quota_score_floor = 0.0
    graph.order3_quota_mode = "soft"
    graph.order3_soft_quota_coef = 0.0
    graph.current_order3_credit_gate = 1.0
    graph.exploration_mix = 0.0
    graph.current_greedy_sample_prob = 0.0
    graph.use_topology_persistence = False
    return graph


def _ranking_fixture(raw_scores):
    candidate_nodes = torch.tensor([
        [True, True, False],   # selected slot 0: (0, 1)
        [True, False, True],   # selected slot 1: (0, 2)
        [False, True, True],   # inactive target: (1, 2)
    ])
    exist_mask = torch.ones((1, 3))
    adj = torch.tensor([[
        [1, 1],
        [1, 0],
        [0, 1],
    ]])
    return _ranking_graph(), raw_scores, candidate_nodes, exist_mask, adj


def test_replay_prefix_competitor_margin_uses_earliest_reachable_slot():
    graph, scores, nodes, exist, adj = _ranking_fixture(
        torch.tensor([[10.0, 1.0, 3.0]], requires_grad=True)
    )
    margins, valid = graph._candidate_identity_replay_competitor_margins(
        scores, nodes, exist, adj
    )
    assert valid.tolist() == [[False, False, True]]
    # At the first replay slot all three pairs are feasible. Coverage weighting
    # gives effective policy weights [20, 2, 6], so the inactive target must
    # beat the hardest eligible competitor rather than rely on an optimistic
    # later slot or a replay-prefix replacement probability.
    assert torch.allclose(
        margins[0, 2],
        torch.log(torch.tensor(6.0 / 20.0)),
        atol=1e-6,
    )
    margins[0, 2].backward()
    assert scores.grad[0, 2].item() > 0.0
    assert scores.grad[0, 0].item() < 0.0
    assert abs(scores.grad[0, 1].item()) < 1e-7


def test_competitor_margin_tracks_greedy_boundary_not_temperature_schedule():
    graph, scores, nodes, exist, adj = _ranking_fixture(
        torch.tensor([[10.0, 1.0, 3.0]])
    )
    graph.current_sampling_temperature = 0.1
    cold, cold_valid = graph._candidate_identity_replay_competitor_margins(
        scores, nodes, exist, adj
    )
    graph.current_sampling_temperature = 10.0
    hot, hot_valid = graph._candidate_identity_replay_competitor_margins(
        scores, nodes, exist, adj
    )
    assert torch.equal(cold_valid, hot_valid)
    assert torch.allclose(cold, hot, atol=1e-7)


def test_achieved_candidate_goal_stops_auxiliary_training():
    margins = torch.tensor([[0.4], [-0.7]], requires_grad=True)
    references = margins.detach().clone()
    result = compute_capture_candidate_identity_active_competitor_loss(
        candidate_competitor_margins=margins,
        candidate_reference_margins=references,
        candidate_identity_delta=torch.tensor([[0.2], [-0.2]]),
        candidate_valid_mask=torch.ones_like(margins),
        transition_mask=torch.ones((2, 1)),
    )
    assert result["loss"].item() == 0.0
    assert result["unsatisfied_target_count"].item() == 0.0
    result["loss"].backward(retain_graph=True)
    assert torch.abs(margins.grad).sum().item() == 0.0
    margins.grad.zero_()
    result["per_transition_constraint"].sum().backward()
    # Lifecycle protection remains differentiable at the achieved reference
    # even though the training objective has stopped.
    assert margins.grad[0, 0].item() < 0.0
    assert margins.grad[1, 0].item() > 0.0


def test_candidate_residual_objective_excludes_base_and_entropy_gradients():
    base_parameter = torch.tensor(2.0, requires_grad=True)
    candidate_parameter = torch.tensor(-1.0, requires_grad=True)
    residual_loss = _compose_adj_candidate_objective(
        base_rl_loss=base_parameter.pow(2),
        candidate_loss=candidate_parameter.pow(2),
        entropy=base_parameter,
        entropy_coef=0.25,
        candidate_residual_only=True,
    )
    residual_loss.backward()
    assert base_parameter.grad is None or base_parameter.grad.abs().item() == 0.0
    assert candidate_parameter.grad is not None
    assert candidate_parameter.grad.abs().item() > 0.0


def test_standard_objective_keeps_base_candidate_and_entropy_terms():
    base_parameter = torch.tensor(2.0, requires_grad=True)
    candidate_parameter = torch.tensor(-1.0, requires_grad=True)
    full_loss = _compose_adj_candidate_objective(
        base_rl_loss=base_parameter.pow(2),
        candidate_loss=candidate_parameter.pow(2),
        entropy=base_parameter,
        entropy_coef=0.25,
        candidate_residual_only=False,
    )
    full_loss.backward()
    assert base_parameter.grad is not None
    assert candidate_parameter.grad is not None
    assert base_parameter.grad.abs().item() > 0.0
    assert candidate_parameter.grad.abs().item() > 0.0


def _run_candidate_residual_optimizer_isolation_case(device):
    base_parameter = torch.nn.Parameter(
        torch.tensor([1.0], device=device)
    )
    candidate_parameter = torch.nn.Parameter(
        torch.tensor([1.0], device=device)
    )
    parameters = [base_parameter, candidate_parameter]
    standard_optimizer = torch.optim.Adam(parameters, lr=0.1)
    residual_optimizer = torch.optim.Adam(parameters, lr=0.1)
    pair_optimizer = torch.optim.Adam(parameters, lr=0.1)

    # Prime only the standard optimizer with the ordinary joint objective.
    standard_optimizer.zero_grad()
    (base_parameter.pow(2) + candidate_parameter.pow(2)).sum().backward()
    standard_optimizer.step()
    base_before_residual = base_parameter.detach().clone()
    candidate_before_residual = candidate_parameter.detach().clone()
    standard_step_before = standard_optimizer.state[
        candidate_parameter
    ]["step"]

    active_optimizer = _select_adjacency_optimizer(
        candidate_residual_only=True,
        pair_only_objective=False,
        standard_optimizer=standard_optimizer,
        residual_optimizer=residual_optimizer,
        pair_optimizer=pair_optimizer,
    )
    assert active_optimizer is residual_optimizer
    active_optimizer.zero_grad()
    _clear_parameter_gradients_to_none(parameters)
    candidate_parameter.pow(2).sum().backward()
    assert base_parameter.grad is None
    active_optimizer.step()

    # The residual step must not reuse PPO Adam momentum and must not move a
    # parameter absent from the current candidate graph.
    assert torch.equal(base_parameter.detach(), base_before_residual)
    assert not torch.equal(
        candidate_parameter.detach(), candidate_before_residual
    )
    assert base_parameter not in residual_optimizer.state
    assert candidate_parameter in residual_optimizer.state
    assert standard_optimizer.state[candidate_parameter]["step"] == (
        standard_step_before
    )
    assert _select_adjacency_optimizer(
        candidate_residual_only=False,
        pair_only_objective=False,
        standard_optimizer=standard_optimizer,
        residual_optimizer=residual_optimizer,
        pair_optimizer=pair_optimizer,
    ) is standard_optimizer
    assert _select_adjacency_optimizer(
        candidate_residual_only=False,
        pair_only_objective=True,
        standard_optimizer=standard_optimizer,
        residual_optimizer=residual_optimizer,
        pair_optimizer=pair_optimizer,
    ) is pair_optimizer


def test_candidate_residual_uses_isolated_adam_and_skips_inactive_parameters():
    _run_candidate_residual_optimizer_isolation_case(torch.device("cpu"))
    if torch.cuda.is_available():
        _run_candidate_residual_optimizer_isolation_case(
            torch.device("cuda:0")
        )


def _run_standard_and_residual_adam_checkpoint_round_trip(device):
    first = torch.nn.Parameter(torch.tensor([1.0], device=device))
    second = torch.nn.Parameter(torch.tensor([2.0], device=device))
    parameters = [first, second]
    standard_optimizer = torch.optim.Adam(parameters, lr=0.03)
    residual_optimizer = torch.optim.Adam(parameters, lr=0.07)
    pair_optimizer = torch.optim.Adam(parameters, lr=0.09)

    standard_optimizer.zero_grad()
    (first.pow(2) + second.pow(2)).sum().backward()
    standard_optimizer.step()
    _clear_parameter_gradients_to_none(parameters)
    second.pow(2).sum().backward()
    residual_optimizer.step()
    _clear_parameter_gradients_to_none(parameters)
    first.pow(2).sum().backward()
    pair_optimizer.step()

    checkpoint = _adjacency_optimizer_checkpoint_state(
        standard_optimizer=standard_optimizer,
        residual_optimizer=residual_optimizer,
        pair_optimizer=pair_optimizer,
    )
    assert checkpoint["version"] == 2

    checkpoint_path = os.path.join(
        tempfile.mkdtemp(prefix="sddfg_optimizer_state_"),
        "adj_optimizer_state.pt",
    )
    try:
        torch.save(checkpoint, checkpoint_path)
        loaded_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.isdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)

    restored_first = torch.nn.Parameter(
        torch.tensor([1.0], device=device)
    )
    restored_second = torch.nn.Parameter(
        torch.tensor([2.0], device=device)
    )
    restored_parameters = [restored_first, restored_second]
    restored_standard = torch.optim.Adam(restored_parameters, lr=0.5)
    restored_residual = torch.optim.Adam(restored_parameters, lr=0.6)
    restored_pair = torch.optim.Adam(restored_parameters, lr=0.7)
    _load_adjacency_optimizer_checkpoint_state(
        checkpoint=loaded_checkpoint,
        standard_optimizer=restored_standard,
        residual_optimizer=restored_residual,
        pair_optimizer=restored_pair,
    )

    assert restored_standard.param_groups[0]["lr"] == 0.03
    assert restored_residual.param_groups[0]["lr"] == 0.07
    assert restored_pair.param_groups[0]["lr"] == 0.09
    assert restored_first in restored_standard.state
    assert restored_second in restored_standard.state
    assert restored_first not in restored_residual.state
    assert restored_second in restored_residual.state
    assert restored_first in restored_pair.state
    assert restored_second not in restored_pair.state
    for key in ("step", "exp_avg", "exp_avg_sq"):
        standard_expected = standard_optimizer.state[second][key]
        standard_actual = restored_standard.state[restored_second][key]
        residual_expected = residual_optimizer.state[second][key]
        residual_actual = restored_residual.state[restored_second][key]
        pair_expected = pair_optimizer.state[first][key]
        pair_actual = restored_pair.state[restored_first][key]
        if torch.is_tensor(standard_expected):
            assert torch.equal(standard_actual, standard_expected)
            assert torch.equal(residual_actual, residual_expected)
            assert torch.equal(pair_actual, pair_expected)
        else:
            assert standard_actual == standard_expected
            assert residual_actual == residual_expected
            assert pair_actual == pair_expected


def test_standard_and_residual_adam_checkpoint_round_trip():
    _run_standard_and_residual_adam_checkpoint_round_trip(
        torch.device("cpu")
    )
    if torch.cuda.is_available():
        _run_standard_and_residual_adam_checkpoint_round_trip(
            torch.device("cuda:0")
        )


def test_dual_adam_checkpoint_restore_is_transactional():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    standard_optimizer = torch.optim.Adam([parameter], lr=0.03)
    residual_optimizer = torch.optim.Adam([parameter], lr=0.07)
    pair_optimizer = torch.optim.Adam([parameter], lr=0.09)
    standard_optimizer.zero_grad()
    parameter.pow(2).sum().backward()
    standard_optimizer.step()
    standard_before = copy.deepcopy(standard_optimizer.state_dict())
    residual_before = copy.deepcopy(residual_optimizer.state_dict())
    pair_before = copy.deepcopy(pair_optimizer.state_dict())

    malformed = _adjacency_optimizer_checkpoint_state(
        standard_optimizer=standard_optimizer,
        residual_optimizer=residual_optimizer,
        pair_optimizer=pair_optimizer,
    )
    malformed["candidate_residual_optimizer"]["param_groups"][0][
        "params"
    ].append(999)
    try:
        _load_adjacency_optimizer_checkpoint_state(
            checkpoint=malformed,
            standard_optimizer=standard_optimizer,
            residual_optimizer=residual_optimizer,
            pair_optimizer=pair_optimizer,
        )
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("malformed residual Adam checkpoint was accepted")

    restored_standard = standard_optimizer.state_dict()
    restored_residual = residual_optimizer.state_dict()
    restored_pair = pair_optimizer.state_dict()
    assert restored_standard["param_groups"] == standard_before["param_groups"]
    assert restored_residual["param_groups"] == residual_before["param_groups"]
    assert restored_pair == pair_before
    standard_before_key = standard_before["param_groups"][0]["params"][0]
    restored_standard_key = restored_standard["param_groups"][0]["params"][0]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        expected = standard_before["state"][standard_before_key][key]
        actual = restored_standard["state"][restored_standard_key][key]
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected


def test_legacy_dual_adam_checkpoint_resets_new_pair_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    standard_optimizer = torch.optim.Adam([parameter], lr=0.03)
    residual_optimizer = torch.optim.Adam([parameter], lr=0.07)
    pair_optimizer = torch.optim.Adam([parameter], lr=0.09)
    for optimizer in (
            standard_optimizer, residual_optimizer, pair_optimizer):
        optimizer.zero_grad()
        parameter.pow(2).sum().backward()
        optimizer.step()
        _clear_parameter_gradients_to_none([parameter])
    legacy = {
        "version": 1,
        "standard_optimizer": copy.deepcopy(
            standard_optimizer.state_dict()
        ),
        "candidate_residual_optimizer": copy.deepcopy(
            residual_optimizer.state_dict()
        ),
    }
    _load_adjacency_optimizer_checkpoint_state(
        checkpoint=legacy,
        standard_optimizer=standard_optimizer,
        residual_optimizer=residual_optimizer,
        pair_optimizer=pair_optimizer,
    )
    assert standard_optimizer.state
    assert residual_optimizer.state
    assert not pair_optimizer.state
    assert pair_optimizer.param_groups[0]["lr"] == 0.09


def test_terminal_checkpoint_metadata_is_explicit_and_strict():
    metadata = _build_sddfg_checkpoint_metadata(
        total_env_steps=260000,
        target_env_steps=260000,
        checkpoint_kind="terminal",
    )
    assert metadata["runner_checkpoint_version"] == 1
    assert metadata["checkpoint_kind"] == "terminal"
    assert metadata["training_complete"] is True
    assert metadata["total_env_steps"] == 260000
    assert metadata["target_env_steps"] == 260000
    _validate_sddfg_checkpoint_metadata(metadata)

    incomplete = dict(metadata)
    incomplete["total_env_steps"] = 240000
    try:
        _validate_sddfg_checkpoint_metadata(incomplete)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete terminal checkpoint was accepted")


def test_missing_sddfg_optimizer_checkpoint_fails_loudly():
    checkpoint_dir = tempfile.mkdtemp(prefix="sddfg_missing_optimizer_")
    missing_path = os.path.join(
        checkpoint_dir,
        "adj_optimizer_state.pt",
    )
    try:
        try:
            _require_sddfg_optimizer_checkpoint(
                algorithm_name="sddfg",
                optimizer_state_path=missing_path,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "SDDFG restore silently accepted a missing dual-Adam state"
            )
        assert _require_sddfg_optimizer_checkpoint(
            algorithm_name="ddfg",
            optimizer_state_path=missing_path,
        ) is False
    finally:
        os.rmdir(checkpoint_dir)


def test_optimizer_checkpoint_is_the_last_committed_model_file():
    class _StateObject(object):
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    class _Trainer(object):
        def adjacency_optimizer_checkpoint_state(self):
            return {
                "version": 2,
                "standard_optimizer": {"state": {}, "param_groups": []},
                "candidate_residual_optimizer": {
                    "state": {},
                    "param_groups": [],
                },
                "pair_pending_optimizer": {
                    "state": {},
                    "param_groups": [],
                },
            }

    save_dir = tempfile.mkdtemp(prefix="sddfg_terminal_save_")
    runner = type("_Runner", (object,), {})()
    runner.save_dir = save_dir
    runner.adj_network = _StateObject()
    runner.trainer = _Trainer()
    runner.total_env_steps = 260000
    runner.num_env_steps = 260000
    runner._checkpoint_kind = "terminal"
    runner.policy_ids = []
    runner.policies = {}
    runner.algorithm_name = "sddfg"
    runner.use_vfunction = False
    runner.highest_orders = 3

    save_calls = []
    original_save = torch.save

    def _tracking_save(value, path, *args, **kwargs):
        save_calls.append(str(path))
        return original_save(value, path, *args, **kwargs)

    torch.save = _tracking_save
    try:
        RecRunner.save_q_mdfg_cent(runner)
        assert os.path.basename(save_calls[-1]) == "adj_optimizer_state.pt.tmp"
        checkpoint_path = os.path.join(
            save_dir,
            "adj_optimizer_state.pt",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        _validate_sddfg_checkpoint_metadata(checkpoint)
        assert checkpoint["checkpoint_kind"] == "terminal"
        assert checkpoint["training_complete"] is True
    finally:
        torch.save = original_save
        for filename in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, filename))
        os.rmdir(save_dir)


def test_lifecycle_signed_floor_supports_legacy_torch():
    from algorithms.sddfg import r_sddfg

    assert hasattr(r_sddfg, "_elementwise_minimum_compat")
    left = torch.tensor([1.0, 4.0, -2.0])
    right = torch.tensor([2.0, 3.0, -2.0])
    original_minimum = getattr(torch, "minimum", None)
    if original_minimum is not None:
        delattr(torch, "minimum")
    try:
        result = r_sddfg._elementwise_minimum_compat(left, right)
    finally:
        if original_minimum is not None:
            torch.minimum = original_minimum
    assert torch.equal(result, torch.tensor([1.0, 3.0, -2.0]))


def test_active_candidates_are_not_candidate_targets():
    graph, scores, nodes, exist, adj = _ranking_fixture(
        torch.tensor([[2.0, 1.0, 3.0]])
    )
    _, valid = graph._candidate_identity_replay_competitor_margins(
        scores, nodes, exist, adj
    )
    assert valid.tolist() == [[False, False, True]]


def test_all_active_candidates_produce_an_empty_competitor_mask():
    graph = _ranking_graph()
    graph.num_factor = 3
    scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    nodes = torch.tensor([
        [True, True, False],
        [True, False, True],
        [False, True, True],
    ])
    exist = torch.ones((1, 3))
    adj = torch.tensor([[
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]])
    margins, valid = graph._candidate_identity_replay_competitor_margins(
        scores, nodes, exist, adj
    )
    assert valid.tolist() == [[False, False, False]]
    assert margins.tolist() == [[0.0, 0.0, 0.0]]
    disabled = compute_capture_candidate_identity_active_competitor_loss(
        margins,
        torch.zeros_like(margins),
        torch.zeros_like(margins),
        valid.float(),
        torch.ones((1, 1)),
    )
    assert disabled["loss"].item() == 0.0
    assert disabled["target_count"].item() == 0.0


def test_active_match_is_excluded_from_candidate_branch():
    disabled = _loss([[0.5, -0.5]], [[0.0, 0.0]])
    assert disabled["loss"].item() == 0.0
    assert disabled["target_count"].item() == 0.0


def test_unrelated_transitions_do_not_dilute_sparse_supervision():
    one = _loss([[-0.5]], [[0.2]])
    two = _loss(
        [[-0.5], [0.5]],
        [[0.2], [0.0]],
    )
    assert torch.allclose(two["loss"], one["loss"], atol=1e-7)
    assert two["valid_transition_count"].item() == 2.0
    assert two["target_transition_count"].item() == 1.0
    assert two["target_transition_fraction"].item() == 0.5


def test_satisfied_targets_do_not_dilute_unsatisfied_supervision():
    one = _loss([[-0.5]], [[0.2]], reference=[[-0.5]])
    mixed = _loss(
        [[-0.5], [-0.5]],
        [[0.2], [-0.2]],
        reference=[[-0.5], [-0.5]],
    )
    assert torch.allclose(mixed["loss"], one["loss"], atol=1e-7)
    assert mixed["target_transition_count"].item() == 2.0
    assert mixed["unsatisfied_target_transition_count"].item() == 1.0


def test_optimizer_step_moves_active_competitor_margin():
    graph, positive_scores, nodes, exist, adj = _ranking_fixture(
        torch.nn.Parameter(torch.tensor([[10.0, 3.0, 1.0]]))
    )
    positive_optimizer = torch.optim.SGD([positive_scores], lr=0.1)
    positive_before = positive_scores.detach().clone()
    positive_margin, positive_valid = (
        graph._candidate_identity_replay_competitor_margins(
            positive_scores, nodes, exist, adj
        )
    )
    positive = compute_capture_candidate_identity_active_competitor_loss(
        positive_margin,
        positive_margin.detach().clone(),
        torch.tensor([[0.0, 0.0, 0.2]]),
        positive_valid.float(),
        torch.ones((1, 1)),
    )
    positive_optimizer.zero_grad()
    positive["loss"].backward()
    positive_optimizer.step()
    assert positive_scores[0, 2].item() > positive_before[0, 2].item()
    assert positive_scores[0, 0].item() < positive_before[0, 0].item()

    graph, negative_scores, nodes, exist, adj = _ranking_fixture(
        torch.nn.Parameter(torch.tensor([[1.0, 3.0, 10.0]]))
    )
    negative_optimizer = torch.optim.SGD([negative_scores], lr=0.1)
    negative_before = negative_scores.detach().clone()
    negative_margin, negative_valid = (
        graph._candidate_identity_replay_competitor_margins(
            negative_scores, nodes, exist, adj
        )
    )
    negative = compute_capture_candidate_identity_active_competitor_loss(
        negative_margin,
        negative_margin.detach().clone(),
        torch.tensor([[0.0, 0.0, -0.2]]),
        negative_valid.float(),
        torch.ones((1, 1)),
    )
    negative_optimizer.zero_grad()
    negative["loss"].backward()
    negative_optimizer.step()
    assert negative_scores[0, 2].item() < negative_before[0, 2].item()
    assert negative_scores[0, 1].item() > negative_before[0, 1].item()


def test_canonical_rank_is_deterministic_and_invalid_safe():
    scores = torch.tensor([[0.5, 0.7, 0.7, 0.9]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    rank = Adj_Generator.canonical_candidate_ranks(scores, valid)
    assert rank.tolist() == [[3, 1, 2, 0]]


def test_multi_positive_and_negative_targets_are_supported():
    margins = torch.tensor(
        [[-1.0, -0.5, 1.5, 0.25]], requires_grad=True
    )
    result = compute_capture_candidate_identity_active_competitor_loss(
        margins,
        margins.detach().clone(),
        torch.tensor([[0.2, 0.1, -0.3, -0.4]]),
        torch.ones_like(margins),
        torch.ones((1, 1)),
    )
    result["loss"].backward()
    assert torch.all(margins.grad[0, :2] < 0.0).item()
    assert torch.all(margins.grad[0, 2:] > 0.0).item()


def test_extreme_margins_are_numerically_stable():
    margins = torch.tensor(
        [[-1000.0], [1000.0]], requires_grad=True
    )
    result = compute_capture_candidate_identity_active_competitor_loss(
        margins,
        torch.zeros_like(margins),
        torch.tensor([[0.2], [-0.2]]),
        torch.ones_like(margins),
        torch.ones((2, 1)),
    )
    assert bool(torch.isfinite(result["loss"]).item())
    result["loss"].backward()
    assert bool(torch.isfinite(margins.grad).all().item())


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
        [[-0.5], [0.5]],
        [[0.2], [-0.2]],
    )
    assert torch.allclose(result["positive_mass"], torch.tensor(0.2))
    assert torch.allclose(result["negative_mass"], torch.tensor(0.2))
    assert result["positive_loss"].item() > 0.0
    assert result["negative_loss"].item() > 0.0


def test_nonfinite_margin_fails_loudly():
    for margin in (float("nan"), float("inf"), float("-inf")):
        try:
            _loss([[margin]], [[0.2]])
        except FloatingPointError:
            continue
        raise AssertionError("invalid candidate margin did not fail")


def test_target_bearing_diagnostic_does_not_require_lifecycle_cache():
    reference = torch.tensor(0.0)
    assert _candidate_target_bearing_update_diagnostic(
        True, reference
    ).item() == 1.0
    assert _candidate_target_bearing_update_diagnostic(
        False, reference
    ).item() == 0.0


def test_candidate_lifecycle_is_finite_and_does_not_refresh():
    trainer = R_SDDFG.__new__(R_SDDFG)
    trainer.candidate_identity_lifecycle_horizon = 3
    trainer._candidate_identity_lifecycle = {}
    trainer._candidate_identity_lifecycle_observations = {}
    rnn_obs = torch.tensor([[[0.1], [0.2]]])
    dones = torch.zeros((1, 2, 1))
    adj = torch.ones((1, 2, 1))
    previous_adj = torch.zeros_like(adj)
    delta = torch.tensor([[0.25, 0.0]])
    transition_mask = torch.ones((1, 1))
    behavior_version = torch.zeros((1, 2))
    reference_margin = torch.tensor([[-0.4, 0.0]])
    reference_rank = torch.tensor([[2, 1]])

    assert trainer._register_candidate_identity_lifecycle(
        rnn_obs,
        dones,
        adj,
        previous_adj,
        delta,
        transition_mask,
        torch.tensor([[1.0, 0.0]]),
        behavior_version,
        reference_margin,
        reference_rank,
        lifecycle_clock=0,
    ) == 1
    entry = next(iter(trainer._candidate_identity_lifecycle.values()))
    assert entry["created_at"] == 0
    assert entry["expires_at"] == 4
    assert torch.equal(entry["reference_margin"], reference_margin)
    reference_margin.fill_(9.0)
    assert torch.equal(
        entry["reference_margin"], torch.tensor([[-0.4, 0.0]])
    )
    assert len(trainer._candidate_identity_lifecycle_observations) == 1
    # Reusing the same PPO transition must neither create a second identity
    # nor refresh the finite expiration point.
    assert trainer._register_candidate_identity_lifecycle(
        rnn_obs,
        dones,
        adj,
        previous_adj,
        delta,
        transition_mask,
        torch.tensor([[1.0, 0.0]]),
        behavior_version,
        reference_margin,
        reference_rank,
        lifecycle_clock=1,
    ) == 0
    entry = next(iter(trainer._candidate_identity_lifecycle.values()))
    assert entry["expires_at"] == 4
    assert trainer._prune_candidate_identity_lifecycle(2) == 0
    assert trainer._prune_candidate_identity_lifecycle(3) == 0
    assert trainer._prune_candidate_identity_lifecycle(4) == 1
    assert not trainer._candidate_identity_lifecycle
    # Expiration stops training protection but preserves the detached row for
    # exact age-5/10 diagnostics.  It is removed after age 10.
    assert len(trainer._candidate_identity_lifecycle_observations) == 1
    assert trainer._register_candidate_identity_lifecycle(
        rnn_obs,
        dones,
        adj,
        previous_adj,
        delta,
        transition_mask,
        torch.tensor([[1.0, 0.0]]),
        behavior_version,
        reference_margin,
        reference_rank,
        lifecycle_clock=5,
    ) == 0
    assert next(iter(
        trainer._candidate_identity_lifecycle_observations.values()
    ))["created_at"] == 0
    assert trainer._prune_candidate_identity_lifecycle(10) == 0
    assert len(trainer._candidate_identity_lifecycle_observations) == 1
    assert trainer._prune_candidate_identity_lifecycle(11) == 0
    assert not trainer._candidate_identity_lifecycle_observations


def test_retention_archive_observes_exact_age_without_extending_constraint():
    class FixedCandidateNetwork(object):
        @staticmethod
        def evaluate_candidate_identity_active_competitor_margins(
                rnn_obs, dones, adj):
            batch = rnn_obs.shape[0]
            margins = torch.tensor(
                [[0.8, -0.2]], dtype=rnn_obs.dtype
            ).repeat(batch, 1)
            return margins, torch.ones_like(margins)

        @staticmethod
        def canonical_candidate_ranks(scores, valid_mask):
            order = torch.argsort(scores, dim=1, descending=True)
            ranks = torch.zeros_like(order)
            rank_values = torch.arange(
                1, scores.shape[1] + 1, dtype=order.dtype
            ).view(1, -1).repeat(scores.shape[0], 1)
            ranks.scatter_(1, order, rank_values)
            return ranks

    trainer = R_SDDFG.__new__(R_SDDFG)
    trainer.candidate_identity_lifecycle_horizon = 3
    trainer._candidate_identity_lifecycle = {}
    trainer._candidate_identity_lifecycle_observations = {}
    trainer.adj_network = FixedCandidateNetwork()
    rnn_obs = torch.tensor([[[0.1], [0.2]]])
    dones = torch.zeros((1, 2, 1))
    adj = torch.ones((1, 2, 1))
    previous_adj = torch.zeros_like(adj)
    delta = torch.tensor([[0.25, 0.0]])
    transition_mask = torch.ones((1, 1))
    behavior_version = torch.zeros((1, 2))
    reference_margin = torch.tensor([[-0.4, 0.0]])
    reference_rank = torch.tensor([[2, 1]])
    trainer._register_candidate_identity_lifecycle(
        rnn_obs,
        dones,
        adj,
        previous_adj,
        delta,
        transition_mask,
        torch.tensor([[1.0, 0.0]]),
        behavior_version,
        reference_margin,
        reference_rank,
        lifecycle_clock=0,
    )
    # The training constraint is gone at age 4, but exact age 5 and 10 remain
    # observable once each and never re-enter the protected cache.
    assert trainer._prune_candidate_identity_lifecycle(5) == 1
    assert not trainer._candidate_identity_lifecycle
    age5 = trainer._candidate_identity_lifecycle_retention_diagnostics(
        5, torch.tensor(0.0)
    )
    assert age5[5]["eligible_count"].item() == 1.0
    assert age5[5]["signed_margin_held_count"].item() == 1.0
    assert age5[1]["eligible_count"].item() == 0.0
    age10 = trainer._candidate_identity_lifecycle_retention_diagnostics(
        10, torch.tensor(0.0)
    )
    assert age10[10]["eligible_count"].item() == 1.0
    assert age10[10]["rank_held_count"].item() == 1.0
    assert not trainer._candidate_identity_lifecycle


def test_lifecycle_registers_only_rank_or_boundary_progress():
    trainer = R_SDDFG.__new__(R_SDDFG)
    trainer.candidate_identity_lifecycle_horizon = 3
    trainer._candidate_identity_lifecycle = {}
    trainer._candidate_identity_lifecycle_observations = {}
    rnn_obs = torch.tensor([[[0.1], [0.2]]])
    dones = torch.zeros((1, 2, 1))
    adj = torch.ones((1, 2, 1))
    previous_adj = torch.zeros_like(adj)
    delta = torch.tensor([[0.25, 0.0]])
    transition_mask = torch.ones((1, 1))
    behavior_version = torch.zeros((1, 2))
    reference_margin = torch.tensor([[-0.4, 0.0]])
    reference_rank = torch.tensor([[2, 1]])

    assert trainer._register_candidate_identity_lifecycle(
        rnn_obs,
        dones,
        adj,
        previous_adj,
        delta,
        transition_mask,
        torch.zeros_like(delta),
        behavior_version,
        reference_margin,
        reference_rank,
        lifecycle_clock=0,
    ) == 0
    assert not trainer._candidate_identity_lifecycle
    assert not trainer._candidate_identity_lifecycle_observations

    assert trainer._register_candidate_identity_lifecycle(
        rnn_obs,
        dones,
        adj,
        previous_adj,
        delta,
        transition_mask,
        torch.tensor([[1.0, 0.0]]),
        behavior_version,
        reference_margin,
        reference_rank,
        lifecycle_clock=1,
    ) == 1


def test_lifecycle_progress_mask_requires_rank_change_or_goal_crossing():
    identity_delta = torch.tensor([[0.2, -0.2, 0.2, -0.2, 0.2]])
    pre_unsatisfied = torch.ones_like(identity_delta)
    post_unsatisfied = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    pre_rank = torch.tensor([[3.0, 2.0, 4.0, 1.0, 2.0]])
    post_rank = torch.tensor([[2.0, 3.0, 4.0, 1.0, 3.0]])
    expected = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])

    actual = _candidate_lifecycle_behavioral_progress_mask(
        identity_delta=identity_delta,
        pre_unsatisfied_mask=pre_unsatisfied,
        post_unsatisfied_mask=post_unsatisfied,
        pre_rank=pre_rank,
        post_rank=post_rank,
    )
    assert torch.equal(actual, expected)

    # A target already satisfied before this update is never newly registered,
    # even if its canonical rank changes incidentally.
    pre_unsatisfied[:, 0] = 0.0
    actual = _candidate_lifecycle_behavioral_progress_mask(
        identity_delta=identity_delta,
        pre_unsatisfied_mask=pre_unsatisfied,
        post_unsatisfied_mask=post_unsatisfied,
        pre_rank=pre_rank,
        post_rank=post_rank,
    )
    assert actual[0, 0].item() == 0.0


def test_lifecycle_uses_final_committed_candidate_state():
    post_info = {
        "unsatisfied_target_mask": torch.tensor([[1.0]])
    }
    final_info = {
        "unsatisfied_target_mask": torch.tensor([[0.0]])
    }
    assert _select_candidate_lifecycle_committed_info(
        post_candidate_info=post_info,
        final_candidate_info=None,
        lifecycle_target_present=False,
    ) is post_info
    assert _select_candidate_lifecycle_committed_info(
        post_candidate_info=post_info,
        final_candidate_info=final_info,
        lifecycle_target_present=True,
    ) is final_info
    try:
        _select_candidate_lifecycle_committed_info(
            post_candidate_info=post_info,
            final_candidate_info=None,
            lifecycle_target_present=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "missing final lifecycle transaction state was accepted"
        )


def test_lifecycle_projection_protects_each_transition_not_only_sum():
    proposed = [torch.tensor([1.0, 0.0])]
    # The aggregate gradient is [1, 0], so the old sum-only check sees a safe
    # positive dot.  The first individual transition is nevertheless harmed.
    constraints = [
        [torch.tensor([-1.0, 1.0])],
        [torch.tensor([2.0, -1.0])],
    ]
    assert torch.dot(proposed[0], constraints[0][0]).item() < 0.0
    assert torch.dot(
        proposed[0], constraints[0][0] + constraints[1][0]
    ).item() > 0.0
    projected, info = _project_gradients_onto_nonincreasing_halfspaces(
        proposed,
        constraints,
        proposed[0],
    )
    assert info["constraint_count"] == 2.0
    assert info["active_constraint_count"] >= 1.0
    for constraint in constraints:
        assert torch.dot(projected[0], constraint[0]).item() >= -1e-6
    # The orthogonal base-learning component is retained; this is not a full
    # candidate-sensitive parameter rollback.
    assert projected[0].norm().item() > 0.0


def test_lifecycle_projection_handles_opposite_cached_constraints():
    proposed = [torch.tensor([1.0, 2.0])]
    constraints = [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 0.0])],
    ]
    projected, info = _project_gradients_onto_nonincreasing_halfspaces(
        proposed,
        constraints,
        proposed[0],
    )
    assert info["constraint_count"] == 2.0
    assert abs(projected[0][0].item()) <= 1e-6
    assert torch.allclose(projected[0][1], torch.tensor(2.0), atol=1e-6)
    for constraint in constraints:
        assert torch.dot(projected[0], constraint[0]).item() >= -1e-6


def test_current_candidate_supersedes_only_incompatible_lifecycle_constraint():
    proposed = [torch.tensor([1.0, 1.0])]
    current_candidate = [torch.tensor([1.0, 0.0])]
    constraints = [
        [torch.tensor([0.0, 1.0])],
        [torch.tensor([-1.0, 0.0])],
    ]
    projected, accepted, superseded, info = (
        _project_with_current_candidate_priority(
            proposed,
            current_candidate,
            constraints,
            proposed[0],
        )
    )
    assert accepted == [0]
    assert superseded == [1]
    assert info["superseded_constraint_count"] == 1.0
    assert torch.dot(projected[0], current_candidate[0]).item() > 0.0
    assert torch.dot(projected[0], constraints[0][0]).item() >= -1e-6


def test_realized_parameter_displacement_is_projected_in_descent_space():
    raw_parameter_delta = torch.tensor([1.0, -0.5])
    realized_descent = [-raw_parameter_delta]
    constraints = [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, -1.0])],
    ]
    safe_descent, info = _project_gradients_onto_nonincreasing_halfspaces(
        realized_descent,
        constraints,
        realized_descent[0],
    )
    safe_parameter_delta = -safe_descent[0]
    assert info["constraint_count"] == 2.0
    for constraint in constraints:
        # A loss gradient h is non-increasing under parameter delta iff
        # h dot delta <= 0, equivalently (-delta) dot h >= 0.
        assert torch.dot(
            constraint[0], safe_parameter_delta
        ).item() <= 1e-6


def test_standard_adam_state_sync_uses_implementation_equation():
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.7]))
    optimizer = torch.optim.Adam([parameter], lr=1e-3, eps=1e-5)
    optimizer.zero_grad()
    (parameter.pow(2).sum()).backward()
    before = parameter.detach().clone()
    optimizer.step()
    raw_delta = parameter.detach() - before
    # Exercise a real post-guard displacement while retaining Adam's second
    # moment.  The helper must reconstruct both the observed raw step and the
    # executed safe step from the standard torch.optim.Adam equation.
    with torch.no_grad():
        parameter.copy_(before + raw_delta * 0.5)
    result = _sync_adam_first_moment_to_executed_update(
        optimizer,
        [parameter],
        [before],
        [raw_delta],
    )
    assert result["update_equation_version"] == 2.0
    assert result["raw_reconstruction_error"] <= result[
        "reconstruction_tolerance"
    ]
    assert result["safe_reconstruction_error"] <= result[
        "reconstruction_tolerance"
    ]
    assert result["raw_reconstruction_error_ratio"] <= 1.0
    assert result["safe_reconstruction_error_ratio"] <= 1.0


def test_successful_candidate_capture_boundary_join_is_exact_and_read_only():
    context = torch.tensor([
        [1.0, 0.0],
        [0.0, 0.5],
        [1.0, 0.0],
    ])
    success = torch.tensor([[1.0], [1.0], [0.0]])
    behavior_margin = torch.tensor([
        [-1.0, -0.2],
        [-0.3, -0.1],
        [-2.0, -0.5],
    ])
    behavior_rank = torch.tensor([
        [6.0, 2.0],
        [3.0, 2.0],
        [8.0, 4.0],
    ])
    behavior_valid = torch.ones_like(behavior_margin)
    current_margin = torch.tensor([
        [-0.4, -0.2],
        [-0.3, 0.2],
        [-1.5, -0.5],
    ], requires_grad=True)
    current_rank = torch.tensor([
        [5.0, 2.0],
        [3.0, 1.0],
        [7.0, 4.0],
    ])
    current_valid = torch.ones_like(current_margin)
    identity_delta = torch.tensor([
        [0.2, 0.0],
        [0.0, 0.0],
        [0.3, 0.0],
    ])
    snapshots = [
        tensor.detach().clone()
        for tensor in (
            context,
            success,
            behavior_margin,
            behavior_rank,
            behavior_valid,
            current_margin,
            current_rank,
            current_valid,
            identity_delta,
        )
    ]
    rng_before = torch.get_rng_state().clone()
    result = _successful_candidate_capture_boundary_diagnostics(
        candidate_capture_context=context,
        episode_success_gate=success,
        behavior_candidate_margin=behavior_margin,
        behavior_candidate_rank=behavior_rank,
        behavior_candidate_valid=behavior_valid,
        current_candidate_margin=current_margin,
        current_candidate_rank=current_rank,
        current_candidate_valid=current_valid,
        candidate_identity_delta=identity_delta,
    )
    assert result["version"].item() == 1.0
    assert result["identity_count"].item() == 2.0
    assert result["identity_mass"].item() == 1.5
    assert torch.allclose(
        result["current_margin_mean"], torch.tensor(-0.2), atol=1e-6
    )
    assert torch.allclose(
        result["current_rank_mean"],
        torch.tensor((5.0 + 0.5) / 1.5),
        atol=1e-6,
    )
    assert torch.allclose(
        result["current_boundary_crossed_fraction"],
        torch.tensor(1.0 / 3.0),
        atol=1e-6,
    )
    assert torch.allclose(
        result["current_boundary_deficit_mean"],
        torch.tensor(0.4 / 1.5),
        atol=1e-6,
    )
    assert torch.allclose(
        result["rank_improved_fraction"], torch.tensor(1.0), atol=1e-6
    )
    assert torch.allclose(
        result["candidate_target_overlap_fraction"],
        torch.tensor(2.0 / 3.0),
        atol=1e-6,
    )
    assert torch.allclose(
        result["positive_candidate_target_overlap_fraction"],
        torch.tensor(2.0 / 3.0),
        atol=1e-6,
    )
    assert result["identity_join_valid"].item() == 1.0
    assert current_margin.grad is None
    assert not result["current_margin_mean"].requires_grad
    assert torch.equal(torch.get_rng_state(), rng_before)
    for current, snapshot in zip(
            (
                context,
                success,
                behavior_margin,
                behavior_rank,
                behavior_valid,
                current_margin,
                current_rank,
                current_valid,
                identity_delta,
            ),
            snapshots):
        assert torch.equal(current.detach(), snapshot)

    invalid_current = current_valid.clone()
    invalid_current[0, 0] = 0.0
    try:
        _successful_candidate_capture_boundary_diagnostics(
            context,
            success,
            behavior_margin,
            behavior_rank,
            behavior_valid,
            current_margin,
            current_rank,
            invalid_current,
            identity_delta,
        )
    except RuntimeError as exc:
        assert "disappeared" in str(exc)
    else:
        raise AssertionError("candidate identity/catalog mismatch must fail")


def test_candidate_identity_transaction_trace_closes_same_population_rank():
    pre_margin = torch.tensor([
        [0.6, -1.10, 0.3, 0.2, -0.4, -0.8],
        [0.5, 0.4, -0.05, 0.2, 0.1, -0.6],
    ])
    post_margin = torch.tensor([
        [0.6, -1.04, 0.3, 0.2, -0.4, -0.8],
        [0.5, 0.4, 0.25, 0.2, 0.1, -0.6],
    ])
    valid = torch.ones_like(pre_margin)
    pre_rank = Adj_Generator.canonical_candidate_ranks(pre_margin, valid)
    post_rank = Adj_Generator.canonical_candidate_ranks(post_margin, valid)
    delta = torch.zeros_like(pre_margin)
    delta[0, 1] = 1.0
    delta[1, 2] = 1.0
    capture_context = torch.zeros_like(pre_margin)
    capture_context[0, 1] = 2.0
    success_gate = torch.tensor([[1.0], [0.0]])
    behavior_margin = pre_margin - 0.1
    behavior_rank = pre_rank.to(pre_margin.dtype)
    behavior_valid = valid.clone()
    behavior_version = torch.full_like(pre_margin, 7.0)
    provenance = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    lifecycle_progress = torch.zeros_like(pre_margin)
    lifecycle_progress[1, 2] = 1.0
    snapshots = [
        tensor.clone()
        for tensor in (
            pre_margin,
            post_margin,
            valid,
            delta,
            capture_context,
            success_gate,
            behavior_margin,
            behavior_rank,
            behavior_valid,
            behavior_version,
            provenance,
            lifecycle_progress,
        )
    ]
    rng_before = torch.get_rng_state().clone()
    rows = _candidate_identity_transaction_trace_rows(
        candidate_identity_delta=delta,
        candidate_effective_delta=delta,
        candidate_capture_context=capture_context,
        episode_success_gate=success_gate,
        behavior_candidate_margin=behavior_margin,
        behavior_candidate_rank=behavior_rank,
        behavior_candidate_valid=behavior_valid,
        behavior_candidate_version=behavior_version,
        current_candidate_policy_version=10,
        pre_candidate_margin=pre_margin,
        pre_candidate_rank=pre_rank,
        pre_candidate_valid=valid,
        post_candidate_margin=post_margin,
        post_candidate_rank=post_rank,
        post_candidate_valid=valid,
        replay_population_provenance=provenance,
        candidate_lifecycle_progress_mask=lifecycle_progress,
        num_agents=4,
        highest_orders=2,
    )
    assert len(rows) == 2
    run105_equivalent = rows[0]
    assert run105_equivalent["canonical_identity"] == "0-2"
    assert run105_equivalent["successful_candidate_capture_overlap"] == 2.0
    assert run105_equivalent["signed_margin_change"] > 0.0
    assert run105_equivalent["pre_rank"] == run105_equivalent["post_rank"]
    assert run105_equivalent["rank_improved"] == 0
    assert run105_equivalent["boundary_crossing"] == 0
    assert run105_equivalent["post_boundary_deficit"] < (
        run105_equivalent["pre_boundary_deficit"]
    )
    assert run105_equivalent[
        "same_population_rank_reconstruction_valid"
    ] == 1
    crossed = rows[1]
    assert crossed["canonical_identity"] == "0-3"
    assert crossed["boundary_crossing"] == 1
    assert crossed["rank_improved"] == 1
    assert crossed["lifecycle_behavioral_progress"] == 1
    assert torch.equal(torch.get_rng_state(), rng_before)
    for tensor, snapshot in zip(
            (
                pre_margin,
                post_margin,
                valid,
                delta,
                capture_context,
                success_gate,
                behavior_margin,
                behavior_rank,
                behavior_valid,
                behavior_version,
                provenance,
                lifecycle_progress,
            ),
            snapshots):
        assert torch.equal(tensor, snapshot)


def test_candidate_rank_decomposition_excludes_invalid_and_handles_ties():
    margins = torch.tensor([0.5, 0.5, 9.0, 0.2])
    valid = torch.tensor([1.0, 1.0, 0.0, 1.0])
    first = _candidate_rank_decomposition(margins, valid, 0)
    second = _candidate_rank_decomposition(margins, valid, 1)
    assert first["valid_population_count"] == 3
    assert first["reconstructed_rank"] == 1
    assert second["reconstructed_rank"] == 2
    assert second["strictly_better_count"] == 0
    assert second["tie_precedes_count"] == 1
    assert second["next_better_candidate_index"] == 0
    assert second["next_better_margin_gap"] == 0.0


def test_score_to_rank_counterfactual_handles_positive_negative_and_ties():
    positive_scores = [-0.4, -0.1, -0.8, 9.0]
    valid = [True, True, True, False]
    positive = candidate_score_to_rank_sensitivity(
        scores=positive_scores,
        valid=valid,
        target_index=0,
        target_sign=1.0,
        observed_signed_improvement=0.01,
    )
    assert positive["pre_rank"] == 2
    assert positive["rank_competitor_index"] == 1
    assert positive["rank_after_virtual_step"] == 1
    assert positive["rank_virtual_step_improves"]
    assert positive["observed_updates_to_rank_step"] > 30
    assert positive["observed_updates_to_boundary"] > 40

    negative_scores = [0.5, 0.2, -0.1, 8.0]
    negative = candidate_score_to_rank_sensitivity(
        scores=negative_scores,
        valid=valid,
        target_index=0,
        target_sign=-1.0,
        observed_signed_improvement=0.02,
    )
    assert negative["pre_rank"] == 1
    assert negative["rank_competitor_index"] == 1
    assert negative["rank_after_virtual_step"] > negative["pre_rank"]
    assert negative["rank_virtual_step_improves"]
    assert negative["virtual_boundary_crossed"]

    tie = candidate_score_to_rank_sensitivity(
        scores=[0.0, 0.0, 100.0],
        valid=[True, True, False],
        target_index=1,
        target_sign=1.0,
        observed_signed_improvement=1e-3,
    )
    assert tie["pre_rank"] == 2
    assert tie["rank_competitor_index"] == 0
    assert tie["rank_after_virtual_step"] == 1

    assert positive_scores == [-0.4, -0.1, -0.8, 9.0]
    assert negative_scores == [0.5, 0.2, -0.1, 8.0]
    assert valid == [True, True, True, False]


def test_run106_score_to_rank_counterfactual_quantifies_reachability_gap():
    result = candidate_score_to_rank_sensitivity(
        scores=[
            -2.234462022781372,
            -2.16845965385437,
            -2.5,
        ],
        valid=[True, True, True],
        target_index=0,
        target_sign=1.0,
        observed_signed_improvement=5.626678466796875e-05,
    )
    assert result["pre_rank"] == 2
    assert result["rank_competitor_index"] == 1
    assert result["observed_updates_to_rank_step"] >= 1174
    assert result["observed_updates_to_boundary"] >= 39712
    assert result["rank_virtual_step_improves"]
    assert result["virtual_boundary_crossed"]


def test_candidate_identity_transaction_runner_join_is_exact():
    trace_rows = [{
        "diagnostic_version": 2,
        "target_row_sequence_within_transaction": 0,
        "selected_episode_ordinal": 1,
        "episode_transition_step": 12,
        "transition_index_in_partition": 12,
        "candidate_index": 3,
        "canonical_identity": "0-4",
        "factor_order": 2,
        "target_sign": 1.0,
        "target_weight": 1.0,
        "pair_evidence_transition": 0,
        "successful_candidate_capture_overlap": 1.0,
        "behavior_margin": -2.2,
        "behavior_rank": 23.0,
        "behavior_valid": 1.0,
        "behavior_policy_version": 4.0,
        "candidate_policy_age": 3.0,
        "pre_margin": -2.26,
        "post_margin": -2.20,
        "signed_margin_change": 0.06,
        "margin_direction_correct": 1,
        "pre_rank": 23,
        "post_rank": 23,
        "signed_rank_improvement": 0.0,
        "rank_improved": 0,
        "pre_signed_boundary": -2.26,
        "post_signed_boundary": -2.20,
        "pre_boundary_deficit": 2.26,
        "post_boundary_deficit": 2.20,
        "boundary_crossing": 0,
        "pre_valid_population_count": 30,
        "post_valid_population_count": 30,
        "pre_strictly_better_count": 22,
        "post_strictly_better_count": 22,
        "pre_tie_precedes_count": 0,
        "post_tie_precedes_count": 0,
        "pre_next_better_candidate_index": 9,
        "post_next_better_candidate_index": 9,
        "pre_next_better_margin_gap": 0.2,
        "post_next_better_margin_gap": 0.14,
        "same_population_rank_reconstruction_valid": 1,
        "lifecycle_behavioral_progress": 0,
    }]
    transaction = {
        "run_id": "run106",
        "env_step": 80800,
        "adjacency_update_round": 93,
        "ppo_epoch_index": 0,
        "policy_id": "policy_0",
        "partition_index": 1,
        "transaction_sequence_index": 376,
        "optimizer_step_before": 376,
        "optimizer_step_after": 377,
        "candidate_gradient_projection_intervened": 0,
        "lifecycle_gradient_projection_intervened": 0,
        "lifecycle_backtrack_count": 0,
        "lifecycle_reject_occurred": 0,
        "rollback_occurred": 0,
    }
    episode_rows = [{
        "selected_for_training": 1,
        "selected_episode_ordinal": 1,
        "episode_generation": 370,
        "replay_slot_index": 2,
        "episode_recency_age": 1,
        "base_selected": 0,
        "support_selected": 1,
        "outcome_support_used": 1,
        "candidate_identity_indices": "3;11",
        "candidate_factor_identities": "0-4;2-5",
    }]
    train_info = {
        "capture_candidate_identity_loss_contribution": 0.08,
        "capture_candidate_identity_gradient_norm": 0.05,
        "capture_candidate_identity_projected_gradient_dot": 0.003,
        "capture_candidate_identity_clipped_gradient_dot": 0.003,
        "capture_candidate_identity_actual_update_descent_dot_before": 1e-6,
        "capture_candidate_identity_actual_update_descent_dot_after": 1e-6,
        "capture_candidate_identity_actual_update_corrected": 0.0,
        "capture_candidate_identity_loss_optimizer_change": -4e-6,
    }
    rows = _build_candidate_identity_transaction_rows(
        transaction_row=transaction,
        trace_rows=trace_rows,
        episode_rows=episode_rows,
        train_adj_info=train_info,
        candidate_event_by_target={
            (1, 12, 3, 1): (
                {
                    "environment_episode_id": 9001,
                    "capture_event_id": 44,
                    "prey_id": 7,
                    "capture_step": 12,
                    "candidate_index": 3,
                    "candidate_identity": "0-4",
                    "candidate_order": 2,
                    "participant_slots": "0-4",
                    "static_dynamic_class": "dynamic",
                    "final_target_mass": 0.4,
                },
                {
                    "environment_episode_id": 9001,
                    "capture_event_id": 45,
                    "prey_id": 8,
                    "capture_step": 12,
                    "candidate_index": 3,
                    "candidate_identity": "0-4",
                    "candidate_order": 2,
                    "participant_slots": "0-4-5",
                    "static_dynamic_class": "dynamic",
                    "final_target_mass": 0.6,
                },
            ),
        },
    )
    assert len(rows) == 1
    assert tuple(rows[0].keys()) == (
        _CANDIDATE_IDENTITY_TRANSACTION_CSV_FIELDS
    )
    assert rows[0]["episode_generation"] == 370
    assert rows[0]["support_selected"] == 1
    assert rows[0]["canonical_identity"] == "0-4"
    assert rows[0]["capture_event_id"] == -1
    assert rows[0]["candidate_event_group_size"] == 2
    assert rows[0]["capture_event_ids"] == "44|45"
    assert rows[0]["capture_prey_ids"] == "7|8"
    assert rows[0]["candidate_event_target_masses"] == "0.40000000000000002|0.59999999999999998"
    bad_trace = [dict(trace_rows[0])]
    bad_trace[0]["candidate_index"] = 4
    try:
        _build_candidate_identity_transaction_rows(
            transaction,
            bad_trace,
            episode_rows,
            train_info,
            {},
        )
    except RuntimeError as exc:
        assert "event provenance" in str(exc)
    else:
        raise AssertionError("candidate/episode identity mismatch must fail")


def _candidate_trace_neutrality_case(device):
    device = torch.device(device)

    def _run(enable_diagnostic):
        parameter = torch.nn.Parameter(
            torch.tensor([0.3, -0.2], device=device)
        )
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        loss = (parameter.pow(2) * torch.tensor(
            [1.0, 3.0], device=device
        )).sum()
        rng_before = torch.get_rng_state().clone()
        cuda_rng_before = (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda"
            else None
        )
        if enable_diagnostic:
            pre_margin = torch.tensor(
                [[0.4, -0.8, 0.1]], device=device
            )
            post_margin = torch.tensor(
                [[0.4, -0.7, 0.1]], device=device
            )
            valid = torch.ones_like(pre_margin)
            pre_rank = Adj_Generator.canonical_candidate_ranks(
                pre_margin, valid
            )
            post_rank = Adj_Generator.canonical_candidate_ranks(
                post_margin, valid
            )
            delta = torch.tensor(
                [[0.0, 1.0, 0.0]], device=device
            )
            rows = _candidate_identity_transaction_trace_rows(
                candidate_identity_delta=delta,
                candidate_effective_delta=delta,
                candidate_capture_context=torch.zeros_like(delta),
                episode_success_gate=torch.ones((1, 1), device=device),
                behavior_candidate_margin=pre_margin,
                behavior_candidate_rank=pre_rank.to(pre_margin.dtype),
                behavior_candidate_valid=valid,
                behavior_candidate_version=torch.zeros_like(delta),
                current_candidate_policy_version=1,
                pre_candidate_margin=pre_margin,
                pre_candidate_rank=pre_rank,
                pre_candidate_valid=valid,
                post_candidate_margin=post_margin,
                post_candidate_rank=post_rank,
                post_candidate_valid=valid,
                replay_population_provenance=torch.tensor(
                    [[0.0, 0.0, 0.0]], device=device
                ),
                candidate_lifecycle_progress_mask=torch.zeros_like(delta),
                num_agents=3,
                highest_orders=2,
            )
            assert len(rows) == 1
        assert torch.equal(torch.get_rng_state(), rng_before)
        if device.type == "cuda":
            assert torch.equal(
                torch.cuda.get_rng_state(device),
                cuda_rng_before,
            )
        optimizer.zero_grad()
        loss.backward()
        gradient = parameter.grad.detach().clone()
        optimizer.step()
        state = optimizer.state[parameter]
        return {
            "loss": loss.detach().clone(),
            "gradient": gradient,
            "parameter": parameter.detach().clone(),
            "exp_avg": state["exp_avg"].detach().clone(),
            "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
            "step": float(state["step"]),
        }

    baseline = _run(False)
    diagnostic = _run(True)
    for key in (
            "loss",
            "gradient",
            "parameter",
            "exp_avg",
            "exp_avg_sq"):
        assert torch.equal(baseline[key], diagnostic[key])
    assert baseline["step"] == diagnostic["step"]


def test_candidate_identity_transaction_trace_cpu_is_trajectory_neutral():
    _candidate_trace_neutrality_case("cpu")


def test_candidate_identity_transaction_trace_cuda_is_trajectory_neutral():
    if not torch.cuda.is_available():
        print("SKIP CUDA candidate identity transaction trace")
        return
    _candidate_trace_neutrality_case("cuda:0")


def main():
    tests = [
        test_replay_population_provenance_uses_three_column_training_schema,
        test_positive_and_negative_targets_have_opposite_gradients,
        test_boundary_anchored_objective_is_symmetric_for_wrong_side_references,
        test_boundary_anchored_objective_requires_positive_active_insertion,
        test_boundary_anchored_objective_crosses_both_active_boundaries,
        test_boundary_anchored_objective_preserves_better_reference,
        test_boundary_anchored_reference_is_not_optimized,
        test_boundary_anchored_competitor_definition_is_exact,
        test_replay_prefix_competitor_margin_uses_earliest_reachable_slot,
        test_competitor_margin_tracks_greedy_boundary_not_temperature_schedule,
        test_achieved_candidate_goal_stops_auxiliary_training,
        test_candidate_residual_objective_excludes_base_and_entropy_gradients,
        test_standard_objective_keeps_base_candidate_and_entropy_terms,
        test_candidate_residual_uses_isolated_adam_and_skips_inactive_parameters,
        test_standard_and_residual_adam_checkpoint_round_trip,
        test_dual_adam_checkpoint_restore_is_transactional,
        test_legacy_dual_adam_checkpoint_resets_new_pair_optimizer,
        test_terminal_checkpoint_metadata_is_explicit_and_strict,
        test_missing_sddfg_optimizer_checkpoint_fails_loudly,
        test_optimizer_checkpoint_is_the_last_committed_model_file,
        test_lifecycle_signed_floor_supports_legacy_torch,
        test_active_candidates_are_not_candidate_targets,
        test_all_active_candidates_produce_an_empty_competitor_mask,
        test_active_match_is_excluded_from_candidate_branch,
        test_unrelated_transitions_do_not_dilute_sparse_supervision,
        test_satisfied_targets_do_not_dilute_unsatisfied_supervision,
        test_optimizer_step_moves_active_competitor_margin,
        test_canonical_rank_is_deterministic_and_invalid_safe,
        test_multi_positive_and_negative_targets_are_supported,
        test_extreme_margins_are_numerically_stable,
        test_invalid_target_fails_loudly,
        test_mixed_sign_mass_is_preserved,
        test_nonfinite_margin_fails_loudly,
        test_target_bearing_diagnostic_does_not_require_lifecycle_cache,
        test_candidate_lifecycle_is_finite_and_does_not_refresh,
        test_retention_archive_observes_exact_age_without_extending_constraint,
        test_lifecycle_registers_only_rank_or_boundary_progress,
        test_lifecycle_progress_mask_requires_rank_change_or_goal_crossing,
        test_lifecycle_uses_final_committed_candidate_state,
        test_lifecycle_projection_protects_each_transition_not_only_sum,
        test_lifecycle_projection_handles_opposite_cached_constraints,
        test_current_candidate_supersedes_only_incompatible_lifecycle_constraint,
        test_realized_parameter_displacement_is_projected_in_descent_space,
        test_standard_adam_state_sync_uses_implementation_equation,
        test_strict_pair_standard_exact_replay_restores_complete_origin,
        test_successful_candidate_capture_boundary_join_is_exact_and_read_only,
        test_candidate_identity_transaction_trace_closes_same_population_rank,
        test_candidate_rank_decomposition_excludes_invalid_and_handles_ties,
        test_score_to_rank_counterfactual_handles_positive_negative_and_ties,
        test_run106_score_to_rank_counterfactual_quantifies_reachability_gap,
        test_candidate_identity_transaction_runner_join_is_exact,
        test_candidate_identity_transaction_trace_cpu_is_trajectory_neutral,
        test_candidate_identity_transaction_trace_cuda_is_trajectory_neutral,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} candidate identity supervision tests".format(len(tests)))


if __name__ == "__main__":
    main()
