"""Directed tests for per-epoch adjacency optimizer transaction diagnostics."""

import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gym.spaces import Box, Discrete


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The transaction writer is tested without constructing a live runner, so its
# optional online logging dependencies are intentionally stubbed when absent.
try:
    import wandb  # noqa: F401
except ImportError:
    sys.modules["wandb"] = types.ModuleType("wandb")
try:
    import tensorboardX  # noqa: F401
except ImportError:
    tensorboard_module = types.ModuleType("tensorboardX")
    tensorboard_module.SummaryWriter = object
    sys.modules["tensorboardX"] = tensorboard_module
try:
    import pandas  # noqa: F401
except ImportError:
    sys.modules["pandas"] = types.ModuleType("pandas")

from algorithms.sddfg.r_sddfg import (  # noqa: E402
    PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION,
    R_SDDFG,
    _joint_exact_catalog_change_acceptance,
    _joint_exact_constraint_acceptance,
    _joint_exact_inactive_selected_factor_acceptance,
    _joint_exact_trial_trace_record,
    _boundary_write_resolution_scale_limit,
    _selection_boundary_replay_tolerance,
    _lifecycle_target_population_from_delta,
    _maximize_joint_exact_backtracking_scale,
    _select_joint_exact_progress_direction,
    _canonical_candidate_identity,
    _pair_selection_boundary_change_diagnostics,
    _pair_boundary_deficit_aware_minimum_dots,
    _adam_state_before_step_diagnostics,
    _clone_parameter_gradients,
    _displacement_delta_diagnostics,
    _gradient_projection_delta_diagnostics,
    _gradient_reconstruction_diagnostics,
    _gradient_tuple_direction_diagnostics,
    _nonnegative_gradient_dot_with_tolerance,
    _objective_gradient_decomposition_diagnostics,
    _exact_pair_target_score_gradient_constraints,
    _failed_boundary_limiter_ordinals,
    _normalized_gradient_seed_blend,
    _pair_target_score_change_diagnostics,
    _pair_target_score_nonregression_postcondition,
    _pair_target_mass_preserving_minimum_dots,
    _parameter_displacement_direction_diagnostics,
    _preserve_pair_gradient_in_standard_transaction,
    _recover_standard_zero_aggregate_pair_gradient,
    _project_gradient_tuple_to_minimum_dots,
    _project_with_current_candidate_priority,
    _separate_replay_graph_and_factor_advantages,
    _validate_adam_step_increment,
    _validate_exact_score_join_fields,
)
from algorithms.sddfg.algorithm.adj_generator import (  # noqa: E402
    Adj_Generator,
    SelectedFactorInactiveCandidateError,
)
from algorithms.sddfg.algorithm.rSDDFGPolicy import (  # noqa: E402
    R_SDDFGPolicy,
)
from utils.pair_credit import CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH  # noqa: E402
from utils.pair_pending import (  # noqa: E402
    PairOptimizerRecoverableNoOpError,
)
from utils.pair_direction import (  # noqa: E402
    PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION,
)
from runner.base_runner import (  # noqa: E402
    PAIR_EXACT_SCORE_RECORDING_CONTRACT_VERSION,
    RecRunner,
    _ADJ_TRANSACTION_CSV_BASENAME,
    _ADJ_TRANSACTION_CSV_FIELDS,
    _ADJ_TRANSACTION_TRAIN_INFO_FIELDS,
    _PAIR_SELECTION_BOUNDARY_CSV_BASENAME,
    _PAIR_SELECTION_BOUNDARY_TRACE_FIELDS,
    _PAIR_SELECTION_BOUNDARY_RETENTION_TRACE_FIELDS,
    _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_TRACE_FIELDS,
    _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_TRACE_FIELDS,
    _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_CSV_BASENAME,
    _PAIR_DIRECTION_CANDIDATE_TRACE_FIELDS,
    _STRICT_PAIR_EXACT_FAILURE_FIELDS,
    _build_pair_selection_boundary_retention_rows,
    _build_pair_selection_boundary_retention_component_rows,
    _build_pair_selection_boundary_policy_response_rows,
    _build_pair_direction_candidate_rows,
    _build_adj_transaction_row,
    _expected_adj_transaction_partition_count,
    _get_run_csv_name,
    _validate_adj_transaction_update_records,
)


def _assert_close(left, right, atol=1e-10):
    assert abs(float(left) - float(right)) <= float(atol), (
        left,
        right,
    )


def _assert_nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_pair_selection_boundary_crossings():
    pre_boundary = torch.tensor([[-0.2, 0.3]], dtype=torch.float32)
    post_boundary = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
    info = _pair_selection_boundary_change_diagnostics(
        pre_boundary=pre_boundary,
        post_boundary=post_boundary,
        pre_target_logp=torch.tensor([[-0.8, -0.4]]),
        post_target_logp=torch.tensor([[-0.5, -0.8]]),
        pre_competitor_logp=torch.tensor([[-0.6, -0.7]]),
        post_competitor_logp=torch.tensor([[-0.6, -0.7]]),
        pre_rank=torch.tensor([[2.0, 1.0]]),
        post_rank=torch.tensor([[1.0, 2.0]]),
        pre_selected_index=torch.tensor([[0.0, 1.0]]),
        post_selected_index=torch.tensor([[0.0, 1.0]]),
        pre_competitor_index=torch.tensor([[1.0, 0.0]]),
        post_competitor_index=torch.tensor([[1.0, 0.0]]),
        pre_valid=torch.ones((1, 2)),
        post_valid=torch.ones((1, 2)),
        pre_forced=torch.zeros((1, 2)),
        post_forced=torch.zeros((1, 2)),
        pair_local_delta=torch.tensor([[1.0, -1.0]]),
        num_agents=4,
        highest_orders=2,
    )
    assert float(info["target_count"]) == 2.0
    assert float(info["correct_count"]) == 2.0
    assert float(info["reverse_count"]) == 0.0
    assert float(info["zero_count"]) == 0.0
    assert float(info["crossing_count"]) == 2.0
    assert float(info["promotion_count"]) == 1.0
    assert float(info["eviction_count"]) == 1.0


class _RetentionReplayNetwork:
    highest_orders = 1

    def __init__(self, device):
        self.device = device
        self.call_index = 0
        self.rng = np.random.RandomState(37)
        self.eval_rng = np.random.RandomState(41)
        self.pair_credit_ema = torch.zeros(1, device=device)
        self.pair_credit_seen = torch.zeros(1, device=device)
        self.triplet_credit_ema = torch.zeros(1, device=device)
        self.triplet_credit_seen = torch.zeros(1, device=device)
        self.order3_credit_loss_ema = 0.0
        self.order3_credit_margin_ema = 0.0
        self.current_order3_credit_gate = 1.0

    def evaluate_selected_factor_replay_boundaries(
            self, rnn_obs, dones, adj):
        del dones, adj
        # Deliberately touch every RNG family. The observer must restore each
        # source even if a future production forward becomes stochastic.
        torch.rand(1, device=rnn_obs.device)
        np.random.rand()
        __import__("random").random()
        self.rng.rand()
        self.eval_rng.rand()
        count = rnn_obs.shape[0]
        if self.call_index == 0:
            margin = 0.08
            competitor = 1.0
        else:
            margin = 0.05
            competitor = 2.0
        self.call_index += 1
        shape = (count, 1)
        competitor_logp = torch.full(
            shape, -1.0, device=rnn_obs.device
        )
        selected_logp = competitor_logp + margin
        return {
            "selected_margin": torch.full(
                shape, margin, device=rnn_obs.device
            ),
            "selected_logp": selected_logp,
            "competitor_logp": competitor_logp,
            "selected_rank": torch.ones(shape, device=rnn_obs.device),
            "selected_candidate_index": torch.zeros(
                shape, device=rnn_obs.device
            ),
            "competitor_candidate_index": torch.full(
                shape, competitor, device=rnn_obs.device
            ),
            "valid": torch.ones(shape, device=rnn_obs.device),
            "forced": torch.zeros(shape, device=rnn_obs.device),
        }


def validate_pair_selection_boundary_retention_observer(device):
    trainer = object.__new__(R_SDDFG)
    trainer.num_agents = 3
    trainer.num_factor = 1
    trainer.adj_network = _RetentionReplayNetwork(device)
    optimizer_parameter = torch.nn.Parameter(torch.zeros(1, device=device))
    trainer.adj_optimizer = torch.optim.Adam((optimizer_parameter,))
    trainer.candidate_residual_optimizer = torch.optim.Adam(
        (optimizer_parameter,)
    )
    trainer.pair_pending_optimizer = torch.optim.Adam(
        (optimizer_parameter,)
    )
    trainer._pair_selection_boundary_retention_observations = {}
    trainer._pair_selection_boundary_retention_next_id = 0
    trainer._pair_selection_boundary_ordinary_update_clock = 0

    boundary = _pair_selection_boundary_change_diagnostics(
        pre_boundary=torch.tensor([[-0.2]], device=device),
        post_boundary=torch.tensor([[0.1]], device=device),
        pre_target_logp=torch.tensor([[-0.8]], device=device),
        post_target_logp=torch.tensor([[-0.5]], device=device),
        pre_competitor_logp=torch.tensor([[-0.6]], device=device),
        post_competitor_logp=torch.tensor([[-0.6]], device=device),
        pre_rank=torch.tensor([[2.0]], device=device),
        post_rank=torch.tensor([[1.0]], device=device),
        pre_selected_index=torch.tensor([[0.0]], device=device),
        post_selected_index=torch.tensor([[0.0]], device=device),
        pre_competitor_index=torch.tensor([[1.0]], device=device),
        post_competitor_index=torch.tensor([[1.0]], device=device),
        pre_valid=torch.ones((1, 1), device=device),
        post_valid=torch.ones((1, 1), device=device),
        pre_forced=torch.zeros((1, 1), device=device),
        post_forced=torch.zeros((1, 1), device=device),
        pair_local_delta=torch.tensor([[1.0]], device=device),
        num_agents=3,
        highest_orders=1,
    )
    rnn_obs = torch.zeros((1, 4), device=device)
    dones = torch.zeros((1, 3, 1), device=device)
    adj = torch.zeros((1, 3, 3), device=device)
    provenance = torch.tensor([[1.0, 7.0, 9.0]], device=device)
    registered = (
        trainer._register_pair_selection_boundary_retention_observations(
            trace_rows=boundary["rows"],
            rnn_obs=rnn_obs,
            dones=dones,
            adj=adj,
            replay_population_provenance=provenance,
            adjacency_update_round=173,
            policy_id="policy_0",
            transaction_sequence_index=693,
        )
    )
    assert registered == 1
    assert len(trainer._pair_selection_boundary_retention_observations) == 1

    # The train-time protection forward uses the same immutable tensors and
    # must be just as RNG-neutral as the read-only observer.  A changed digest
    # is a hard checkpoint/archive corruption, never a default-zero context.
    direct_rng_before = torch.get_rng_state().clone()
    direct_numpy_before = copy.deepcopy(np.random.get_state())
    direct_python_before = copy.deepcopy(__import__("random").getstate())
    direct_graph_before = copy.deepcopy(trainer.adj_network.rng.get_state())
    direct_eval_before = copy.deepcopy(
        trainer.adj_network.eval_rng.get_state()
    )
    keyed_entries = list(
        trainer._pair_selection_boundary_retention_observations.items()
    )
    trainer._pair_selection_boundary_retention_forward(keyed_entries)
    assert torch.equal(torch.get_rng_state(), direct_rng_before)
    np.testing.assert_equal(np.random.get_state(), direct_numpy_before)
    assert __import__("random").getstate() == direct_python_before
    np.testing.assert_equal(
        trainer.adj_network.rng.get_state(), direct_graph_before
    )
    np.testing.assert_equal(
        trainer.adj_network.eval_rng.get_state(), direct_eval_before
    )
    trainer.adj_network.call_index = 0
    source_entry = keyed_entries[0][1]
    original_rnn_obs = source_entry["rnn_obs"].clone()
    source_entry["rnn_obs"][0, 0] = 1.0
    try:
        trainer._pair_selection_boundary_retention_forward(keyed_entries)
    except RuntimeError as exc:
        assert "exact context SHA256 changed" in str(exc)
    else:
        raise AssertionError("changed retention context SHA256 was accepted")
    source_entry["rnn_obs"].copy_(original_rnn_obs)
    original_target_index = source_entry["target_candidate_index"]
    original_target_identity = source_entry["target_canonical_identity"]
    source_entry["target_candidate_index"] = 1
    _, source_entry["target_canonical_identity"] = (
        _canonical_candidate_identity(
            candidate_index=1,
            num_agents=3,
            highest_orders=trainer.adj_network.highest_orders,
        )
    )
    source_entry["selection_context_sha256"] = (
        trainer._pair_selection_boundary_retention_context_digest(
            rnn_obs=source_entry["rnn_obs"],
            dones=source_entry["dones"],
            adj=source_entry["adj"],
            factor_index=source_entry["factor_index"],
            target_candidate_index=source_entry[
                "target_candidate_index"
            ],
            target_sign=source_entry["target_sign"],
        )
    )
    invalid_population = (
        trainer._pair_selection_boundary_retention_constraint_population(
            reference=optimizer_parameter
        )
    )
    assert invalid_population["context_invalid_keys"] == [0]
    assert not invalid_population["constraint_grads"]
    trainer.adj_network.call_index = 0
    source_entry["target_candidate_index"] = original_target_index
    source_entry["target_canonical_identity"] = original_target_identity
    source_entry["selection_context_sha256"] = (
        trainer._pair_selection_boundary_retention_context_digest(
            rnn_obs=source_entry["rnn_obs"],
            dones=source_entry["dones"],
            adj=source_entry["adj"],
            factor_index=source_entry["factor_index"],
            target_candidate_index=source_entry[
                "target_candidate_index"
            ],
            target_sign=source_entry["target_sign"],
        )
    )

    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda" else None
    )
    numpy_rng_before = copy.deepcopy(np.random.get_state())
    python_rng_before = copy.deepcopy(__import__("random").getstate())
    graph_rng_before = copy.deepcopy(trainer.adj_network.rng.get_state())
    eval_rng_before = copy.deepcopy(
        trainer.adj_network.eval_rng.get_state()
    )

    trainer._pair_selection_boundary_ordinary_update_clock = 1
    age_one = trainer._pair_selection_boundary_retention_diagnostics()
    assert len(age_one) == 1
    assert tuple(age_one[0].keys()) == (
        _PAIR_SELECTION_BOUNDARY_RETENTION_TRACE_FIELDS
    )
    assert age_one[0]["ordinary_update_age"] == 1
    assert age_one[0]["context_valid"] == 1
    assert age_one[0]["competitor_changed"] == 0
    _assert_close(
        age_one[0]["retained_progress_fraction"],
        (0.08 - (-0.2)) / (0.1 - (-0.2)),
        atol=1e-6,
    )
    assert age_one[0]["rank_retained"] == 1
    assert age_one[0]["active_retained"] == 1

    transaction_row = {
        "run_id": "run129",
        "env_step": 145600,
        "adjacency_update_round": 174,
        "ppo_epoch_index": 0,
        "policy_id": "policy_0",
        "partition_index": 0,
        "transaction_sequence_index": 694,
        "optimizer_kind": "standard_adam",
    }
    persisted = _build_pair_selection_boundary_retention_rows(
        transaction_row, age_one
    )
    assert len(persisted) == 1
    assert persisted[0]["source_transaction_sequence_index"] == 693

    checkpoint = trainer.adjacency_optimizer_checkpoint_state()
    resumed = object.__new__(R_SDDFG)
    resumed_parameter = torch.nn.Parameter(torch.zeros(1, device=device))
    resumed.adj_optimizer = torch.optim.Adam((resumed_parameter,))
    resumed.candidate_residual_optimizer = torch.optim.Adam(
        (resumed_parameter,)
    )
    resumed.pair_pending_optimizer = torch.optim.Adam(
        (resumed_parameter,)
    )
    resumed.adj_network = _RetentionReplayNetwork(device)
    resumed._pair_selection_boundary_retention_observations = {}
    resumed._pair_selection_boundary_retention_next_id = 0
    resumed._pair_selection_boundary_ordinary_update_clock = 0
    resumed.load_adjacency_optimizer_checkpoint_state(checkpoint)
    assert resumed._pair_selection_boundary_retention_next_id == 1
    assert resumed._pair_selection_boundary_ordinary_update_clock == 1
    resumed_entry = resumed._pair_selection_boundary_retention_observations[0]
    source_entry = trainer._pair_selection_boundary_retention_observations[0]
    assert set(resumed_entry) == set(source_entry)
    for field in source_entry:
        if torch.is_tensor(source_entry[field]):
            assert torch.equal(source_entry[field], resumed_entry[field])
        else:
            assert source_entry[field] == resumed_entry[field]

    legacy_retention = copy.deepcopy(checkpoint)
    legacy_retention["pair_selection_boundary_retention_version"] = 5
    try:
        resumed.load_adjacency_optimizer_checkpoint_state(legacy_retention)
    except RuntimeError as exc:
        assert "fresh run" in str(exc)
    else:
        raise AssertionError("pre-intervening-write retention checkpoint resumed")
    incomplete_current = copy.deepcopy(checkpoint)
    del incomplete_current[
        "pair_selection_boundary_retention_observations"
    ][0]["selection_context_sha256"]
    try:
        resumed.load_adjacency_optimizer_checkpoint_state(incomplete_current)
    except RuntimeError as exc:
        assert "selection_context_sha256" in str(exc)
    else:
        raise AssertionError("incomplete current retention archive was loaded")

    historical = {
        key: copy.deepcopy(value)
        for key, value in checkpoint.items()
        if not key.startswith("pair_selection_boundary_retention_")
        and key != "pair_selection_boundary_ordinary_update_clock"
    }
    resumed.load_adjacency_optimizer_checkpoint_state(historical)
    assert not resumed._pair_selection_boundary_retention_observations
    assert resumed._pair_selection_boundary_retention_next_id == 0
    assert resumed._pair_selection_boundary_ordinary_update_clock == 0

    trainer._pair_selection_boundary_ordinary_update_clock = 2
    age_two = trainer._pair_selection_boundary_retention_diagnostics()
    assert len(age_two) == 1
    assert age_two[0]["ordinary_update_age"] == 2
    assert age_two[0]["competitor_changed"] == 1
    assert len(trainer._pair_selection_boundary_retention_observations) == 0
    _assert_close(
        age_two[0]["retained_progress_fraction"],
        (0.05 - (-0.2)) / (0.1 - (-0.2)),
        atol=1e-6,
    )

    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
    assert __import__("random").getstate() == python_rng_before
    np.testing.assert_equal(
        trainer.adj_network.rng.get_state(), graph_rng_before
    )
    np.testing.assert_equal(
        trainer.adj_network.eval_rng.get_state(), eval_rng_before
    )


def test_run118_joint_exact_boundary_lifecycle_acceptance():
    actionable = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    boundary_change = torch.tensor(
        [[0.004, 0.002]], dtype=torch.float32
    )
    lifecycle_floor = torch.tensor(
        [[0.25, -0.1]], dtype=torch.float32
    )
    lifecycle_mask = torch.tensor(
        [[1.0, 1.0]], dtype=torch.float32
    )
    lifecycle_tolerance = torch.full_like(lifecycle_floor, 1.0e-6)

    # run118-equivalent full step: every current boundary improves, but one
    # accepted historical lifecycle margin materially regresses.
    rejected = _joint_exact_constraint_acceptance(
        signed_boundary_change=boundary_change,
        actionable_pair_mask=actionable,
        candidate_loss_change=torch.tensor(-0.01),
        lifecycle_signed_margin=torch.tensor(
            [[0.249, -0.09]], dtype=torch.float32
        ),
        lifecycle_signed_floor=lifecycle_floor,
        lifecycle_target_mask=lifecycle_mask,
        lifecycle_tolerance=lifecycle_tolerance,
    )
    assert rejected["boundary_valid"]
    assert rejected["candidate_valid"]
    assert not rejected["lifecycle_valid"]
    assert not rejected["valid"]
    assert rejected["lifecycle_violation_count"] == 1

    # The same jointly-safe direction at a smaller finite scale retains strict
    # current progress and the exact historical floor.
    accepted = _joint_exact_constraint_acceptance(
        signed_boundary_change=0.5 * boundary_change,
        actionable_pair_mask=actionable,
        candidate_loss_change=torch.tensor(-0.005),
        lifecycle_signed_margin=torch.tensor(
            [[0.2500002, -0.09]], dtype=torch.float32
        ),
        lifecycle_signed_floor=lifecycle_floor,
        lifecycle_target_mask=lifecycle_mask,
        lifecycle_tolerance=lifecycle_tolerance,
    )
    assert accepted["valid"]
    assert accepted["boundary_valid"]
    assert accepted["candidate_valid"]
    assert accepted["lifecycle_valid"]


def test_run123_maximum_joint_exact_safe_scale_refinement():
    """Recover the safe interval discarded by dyadic-only backtracking."""
    threshold = 0.4375
    observed_scales = []

    def evaluate(scale):
        scale = float(scale)
        observed_scales.append(scale)
        # This stands for the unchanged production conjunction of exact score,
        # boundary, candidate, and lifecycle revalidation.  The upper edge is
        # deliberately not itself safe.
        return {
            "valid": bool(scale < threshold),
            "scale": scale,
        }

    result = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate,
        max_halvings=20,
        refinement_steps=12,
    )
    assert result["valid"]
    assert result["halving_count"] == 2
    assert result["refinement_count"] == 12
    assert 0.43 < result["final_scale"] < threshold
    assert threshold <= result["invalid_upper_scale"] < 0.44
    assert result["info"]["valid"]
    assert abs(result["info"]["scale"] - result["final_scale"]) < 1e-12
    assert observed_scales[-1] == result["final_scale"]


def test_nonmonotone_midpoint_safe_island_is_committable():
    """Regression for production search_missed_feasible_midpoint.

    Every ordinary dyadic endpoint is candidate-invalid, while the interior
    of the first interval is exact-valid.  The old failure-only audit observed
    these points but deliberately rolled the transaction back.  The production
    selector must now recover, refine, rank, and revalidate the same safe
    island without relaxing any contract.
    """
    observed = []

    def make_evaluator(lower, upper, progress_bias):
        def evaluate(scale):
            scale = float(scale)
            valid = bool(lower < scale < upper)
            observed.append((progress_bias, scale, valid))
            return {
                "valid": valid,
                "scale": scale,
                "boundary_valid": valid,
                "exact_score_valid": valid,
                "candidate_valid": valid,
                "lifecycle_valid": valid,
                "limiting_constraint_type": (
                    "none" if valid else "candidate"
                ),
                "limiting_target_ordinal": 0,
                "progress_target_present": True,
                "progress_worst_actual": scale,
                "progress_worst_required": 1.0,
                "progress_min_completion": progress_bias + scale,
                "progress_mean_completion": progress_bias + scale,
            }
        return evaluate

    result = _select_joint_exact_progress_direction(
        direction_evaluators=(
            ("candidate_0", make_evaluator(0.70, 0.80, 0.0)),
            ("candidate_1", make_evaluator(0.30, 0.32, 1.0)),
            ("candidate_2", make_evaluator(0.167, 0.177, 2.0)),
        ),
        max_halvings=4,
        refinement_steps=12,
        maximum_scale=1.0,
    )
    assert result["valid"]
    assert result["candidate_count"] == 3
    assert result["valid_candidate_count"] == 3
    assert result["selected_candidate_ordinal"] == 2
    assert 0.167 < result["final_scale"] < 0.177
    assert result["info"]["valid"]
    assert any(
        trial["evaluation_kind"] == "interval_subdivision_search"
        and trial["valid"]
        for candidate in result["candidate_results"]
        for trial in candidate["trial_trace"]
    )
    assert observed[-1][0] == 2.0
    assert abs(observed[-1][1] - result["final_scale"]) < 1.0e-12


def test_resolution_only_zero_exact_scores_use_nonregression_contract():
    """Regression for correct=0, reverse=0, zero=2 standard guard failure."""
    tolerance = float(128.0 * torch.finfo(torch.float32).eps)
    pre = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    pair_delta = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    preserved = _pair_target_score_change_diagnostics(
        pre_factor_logp=pre,
        post_factor_logp=pre.clone(),
        pair_local_delta=pair_delta,
        zero_tolerance=tolerance,
    )
    assert float(preserved["correct_direction_count"]) == 0.0
    assert float(preserved["reverse_direction_count"]) == 0.0
    assert float(preserved["approximately_zero_count"]) == 2.0
    assert _pair_target_score_nonregression_postcondition(preserved)

    actionable = torch.ones((1, 2), dtype=torch.float32)
    strict_old = _joint_exact_constraint_acceptance(
        signed_boundary_change=torch.full(
            (1, 2), 1.0e-4, dtype=torch.float32
        ),
        actionable_pair_mask=actionable,
        signed_exact_score_change=torch.zeros(
            (1, 2), dtype=torch.float32
        ),
        candidate_loss_change=torch.tensor(-1.0e-4),
    )
    assert not strict_old["valid"]
    assert strict_old["boundary_valid"]
    assert not strict_old["exact_score_valid"]

    corrected = _joint_exact_constraint_acceptance(
        signed_boundary_change=torch.full(
            (1, 2), 1.0e-4, dtype=torch.float32
        ),
        actionable_pair_mask=actionable,
        signed_exact_score_change=torch.zeros(
            (1, 2), dtype=torch.float32
        ),
        candidate_loss_change=torch.tensor(-1.0e-4),
        preservation_tolerance=tolerance,
    )
    assert corrected["valid"]
    assert corrected["boundary_valid"]
    assert corrected["exact_score_valid"]

    reversed_post = pre.clone()
    reversed_post[0, 0] -= 2.0 * tolerance
    reversed_info = _pair_target_score_change_diagnostics(
        pre_factor_logp=pre,
        post_factor_logp=reversed_post,
        pair_local_delta=pair_delta,
        zero_tolerance=tolerance,
    )
    assert float(reversed_info["reverse_direction_count"]) == 1.0
    assert not _pair_target_score_nonregression_postcondition(reversed_info)


def test_failed_boundary_limiter_tangent_recovers_distinct_ray():
    candidate_results = tuple({
        "info": {
            "limiting_constraint_type": "boundary",
            "limiting_target_ordinal": 0,
        },
        # The final endpoint reports member zero, but an earlier exact forward
        # proves that member one failed simultaneously.  Production must not
        # discard that second limiter when constructing tangent directions.
        "trial_trace": ({
            "limiting_constraint_type": "boundary",
            "limiting_target_ordinal": 0,
            "boundary_failed_target_ordinals": (0, 1),
        },),
    } for _ in range(5))
    limiter_ordinals = _failed_boundary_limiter_ordinals(
        candidate_results=candidate_results,
        boundary_target_count=2,
    )
    assert limiter_ordinals == (0, 1)

    trace_acceptance = _joint_exact_constraint_acceptance(
        signed_boundary_change=torch.zeros((1, 2), dtype=torch.float32),
        actionable_pair_mask=torch.ones((1, 2), dtype=torch.float32),
    )
    trace_row = _joint_exact_trial_trace_record(
        scale=0.5,
        info=trace_acceptance,
        evaluation_ordinal=0,
        evaluation_kind="fixture",
    )
    assert trace_row["boundary_failed_target_ordinals"] == (0, 1)
    assert "boundary_failed_target_ordinals" in (
        _STRICT_PAIR_EXACT_FAILURE_FIELDS
    )

    reference = torch.tensor(0.0, dtype=torch.float32)
    # Five floor variants of this same ray cannot move the first float32
    # production boundary away from an origin near one.
    adam_ray = (torch.tensor([1.0e-8, 1.0], dtype=torch.float32),)
    boundary_constraints = (
        (torch.tensor([1.0, 0.0], dtype=torch.float32),),
        (torch.tensor([0.0, 1.0], dtype=torch.float32),),
    )
    old_change = (
        torch.tensor(1.0, dtype=torch.float32) + adam_ray[0][0]
        - torch.tensor(1.0, dtype=torch.float32)
    )
    assert float(old_change) == 0.0

    limiter_proposal, seed_info = _normalized_gradient_seed_blend(
        proposed_grads=adam_ray,
        seed_grads=boundary_constraints[0],
        reference=reference,
        seed_fraction=0.5,
    )
    recovered, _projection_info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=limiter_proposal,
        constraint_grads=boundary_constraints,
        minimum_dots=(1.0e-7, 1.0e-7),
        reference=reference,
        diagnostic_name="fixture failed-boundary limiter direction",
    )
    assert seed_info["reference_norm"] > 0.0
    assert seed_info["seed_component_norm"] > 0.0
    for constraint in boundary_constraints:
        assert float(torch.dot(recovered[0], constraint[0])) > 0.0
    recovered_change = (
        torch.tensor(1.0, dtype=torch.float32) + recovered[0][0]
        - torch.tensor(1.0, dtype=torch.float32)
    )
    assert float(recovered_change) > 0.0

    full = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=lambda scale: {"valid": True, "scale": scale},
    )
    assert full["final_scale"] == 1.0
    assert not full["unsafe_upper_present"]
    assert full["refinement_count"] == 0

    expanded = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=lambda scale: {
            "valid": bool(float(scale) < 3.25),
            "scale": float(scale),
        },
        maximum_scale=8.0,
        max_expansions=20,
        refinement_steps=12,
    )
    assert expanded["valid"]
    assert expanded["unsafe_upper_present"]
    assert expanded["expansion_count"] == 2
    assert 3.24 < expanded["final_scale"] < 3.25
    assert 3.25 <= expanded["invalid_upper_scale"] < 3.26

    safe_cap = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=lambda scale: {"valid": True, "scale": float(scale)},
        maximum_scale=32.0,
        max_expansions=20,
    )
    assert safe_cap["valid"]
    assert safe_cap["final_scale"] == 32.0
    assert safe_cap["expansion_count"] == 5
    assert not safe_cap["unsafe_upper_present"]

    failed = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=lambda scale: {"valid": False, "scale": scale},
        max_halvings=3,
    )
    assert not failed["valid"]
    assert failed["final_scale"] == 0.125


def test_selection_boundary_replay_tolerance_uses_operand_resolution():
    """Near-zero crossings tolerate ulp noise, not material regression."""
    selected_logp = torch.tensor(-8.0, dtype=torch.float32)
    competitor_logp = torch.tensor(-8.0000002, dtype=torch.float32)
    commit_floor = 2.0e-7
    tolerance = _selection_boundary_replay_tolerance(
        selected_logp=selected_logp,
        competitor_logp=competitor_logp,
        commit_floor=commit_floor,
    )
    tolerance_value = float(tolerance.item())
    assert tolerance_value >= (
        16.0 * 1.1920928955078125e-7 * 8.0
    )
    # The old margin-scaled formula was below one operand ulp at a crossing.
    old_tolerance = (
        16.0
        * 1.1920928955078125e-7
        * (abs(commit_floor) + abs(commit_floor))
    )
    assert tolerance_value > old_tolerance * 1000000.0
    replay_roundoff = commit_floor - 0.5 * tolerance_value
    material_regression = commit_floor - 2.0 * tolerance_value
    assert replay_roundoff >= commit_floor - tolerance_value
    assert material_regression < commit_floor - tolerance_value

    double_tolerance = _selection_boundary_replay_tolerance(
        selected_logp=selected_logp.double(),
        competitor_logp=competitor_logp.double(),
        commit_floor=commit_floor,
    )
    assert float(double_tolerance.item()) < tolerance_value * 1.0e-6


def test_boundary_write_resolution_expands_after_invalid_unit_scale():
    """Recover a strict float32 boundary visible only above scale one."""
    reference = torch.tensor(0.0, dtype=torch.float32)
    selected_logp = torch.tensor([[-2.0]], dtype=torch.float32)
    competitor_logp = torch.tensor([[-3.0]], dtype=torch.float32)
    actionable = torch.ones((1, 1), dtype=torch.float32)
    boundary_dot = torch.tensor([5.0e-8], dtype=torch.float32)
    scale_limit = _boundary_write_resolution_scale_limit(
        selected_logp=selected_logp,
        competitor_logp=competitor_logp,
        actionable_pair_mask=actionable,
        direction_boundary_dots=(boundary_dot,),
        reference=reference,
        current_scale_limit=1.0,
    )
    assert 2.0 < scale_limit <= (2.0 ** 20)

    origin = torch.tensor(1.0, dtype=torch.float32)
    observed_scales = []

    def evaluate(scale):
        scale = float(scale)
        observed_scales.append(scale)
        realized_change_tensor = (
            origin
            + torch.tensor(scale * 5.0e-8, dtype=torch.float32)
            - origin
        )
        realized_change = float(realized_change_tensor)
        # Candidate/lifecycle stand-ins bound a real safe island.  Scale one
        # and every smaller backtracking point have a bitwise-zero boundary;
        # scale 1.5 is the first sampled point that satisfies every contract.
        acceptance = _joint_exact_constraint_acceptance(
            signed_boundary_change=realized_change_tensor.reshape(1, 1),
            actionable_pair_mask=actionable,
            signed_exact_score_change=torch.tensor(
                [[scale * 1.0e-4]], dtype=torch.float32
            ),
            candidate_loss_change=torch.tensor(
                -scale * 1.0e-4, dtype=torch.float32
            ),
            lifecycle_signed_margin=torch.tensor(
                [[0.1 if scale <= 1.75 else -0.1]], dtype=torch.float32
            ),
            lifecycle_signed_floor=torch.zeros(
                (1, 1), dtype=torch.float32
            ),
            lifecycle_target_mask=torch.ones(
                (1, 1), dtype=torch.float32
            ),
            lifecycle_tolerance=torch.zeros(
                (1, 1), dtype=torch.float32
            ),
        )
        acceptance["scale"] = scale
        acceptance["realized_change"] = realized_change
        return acceptance

    legacy_bound = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate,
        max_halvings=20,
        maximum_scale=1.0,
    )
    assert not legacy_bound["valid"]

    recovered = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate,
        max_halvings=20,
        maximum_scale=scale_limit,
        max_expansions=20,
    )
    assert recovered["valid"]
    assert recovered["final_scale"] == 1.5
    assert recovered["info"]["realized_change"] > 0.0
    assert recovered["halving_count"] == 20
    assert recovered["expansion_count"] > 0
    assert any(scale > 1.0 for scale in observed_scales)

    # Regression for the later production failure whose unit/dyadic scales and
    # above-one midpoint/endpoints were all invalid.  The legal interval is
    # narrow and centered on the quarter point of (1, 2); the authoritative
    # write-resolution search must own that point instead of reporting a
    # sampled-grid miss.
    narrow_island = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=lambda scale: {
            "valid": bool(1.20 <= float(scale) <= 1.30),
            "scale": float(scale),
        },
        max_halvings=20,
        maximum_scale=4.0,
        max_expansions=20,
    )
    assert narrow_island["valid"]
    assert narrow_island["final_scale"] == 1.25
    assert narrow_island["resolution_interval_probe_count"] > 0
    assert any(
        trial["evaluation_kind"]
        == "resolution_interval_subdivision_search"
        and trial["valid"]
        for trial in narrow_island["trial_trace"]
    )

    def evaluate_unbounded_safe_tail(scale):
        scale = float(scale)
        return {
            "valid": bool(scale >= 1.5),
            "scale": scale,
            "progress_target_present": False,
            "progress_min_completion": 1.0,
            "progress_mean_completion": 1.0,
        }

    minimum_resolution_commit = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate_unbounded_safe_tail,
        max_halvings=20,
        maximum_scale=4.0,
        max_expansions=20,
        select_by_progress=True,
    )
    assert minimum_resolution_commit["valid"]
    assert minimum_resolution_commit["safe_frontier_scale"] == 4.0
    assert minimum_resolution_commit["final_scale"] == 1.5


def test_run118_boundary_deficit_budget_prioritizes_reachable_crossing():
    """Reuse run118's 168.8k geometry without increasing its step budget."""
    reference = torch.tensor(0.0, dtype=torch.float32)
    constraints = tuple(
        (torch.eye(5, dtype=torch.float32)[index],)
        for index in range(5)
    )
    # Exact run118 epoch-3 identities/signs and pre-boundary margins:
    # one satisfied negative, two satisfied positives, one large positive gap,
    # and the closest positive gap (0.0544331).
    pair_delta = torch.tensor(
        [[-0.05, 0.0114425, 0.0117964, 0.0121612, 0.0145998]],
        dtype=torch.float32,
    )
    pre_boundary = torch.tensor(
        [[-0.1873236, 0.1016084, 0.0750946, -0.8997469, -0.0544331]],
        dtype=torch.float32,
    )
    observed_improvements = (
        0.0254290, 0.0150778, 0.0153400, 0.0253072, 0.0218880,
    )
    info = _pair_boundary_deficit_aware_minimum_dots(
        boundary_target_grads=constraints,
        base_minimum_dots=observed_improvements,
        pair_target_weights=pair_delta.abs().view(-1),
        pre_boundary=pre_boundary,
        pair_local_delta=pair_delta,
        target_candidate_index=torch.arange(
            5, dtype=torch.float32
        ).view(1, 5),
        proposed_descent=(torch.ones(5, dtype=torch.float32),),
        reference=reference,
    )
    floors = info["minimum_dots"]
    assert abs(sum(floors) - sum(observed_improvements)) <= 1e-6
    assert info["deficit_target_count"] == 2.0
    assert info["affordable_crossing_count"] == 1.0
    assert float(info["crossing_affordable"][4]) == 1.0
    assert floors[4] > 0.0544331
    assert 0.0 < floors[3] < 0.8997469
    # Already-satisfied targets retain strict direction without consuming their
    # former evidence-mass share of the finite crossing budget.
    assert all(0.0 < floors[index] < observed_improvements[index]
               for index in (0, 1, 2))
    assert info["budget_conservation_valid"] == 1.0


def test_run120_unaffordable_boundary_budget_waterfills_nearest_target():
    """Do not fragment a finite budget across unreachable rank boundaries."""
    reference = torch.tensor(0.0, dtype=torch.float32)
    constraints = tuple(
        (torch.eye(3, dtype=torch.float32)[index],)
        for index in range(3)
    )
    pair_delta = torch.tensor(
        [[-0.05, 0.0246193, 0.0253807]], dtype=torch.float32
    )
    pre_boundary = torch.tensor(
        [[-0.0599360, -2.2167721, -2.2089419]], dtype=torch.float32
    )
    # Exact v15 epoch-2 required improvements persisted by run120.
    base_floors = (
        0.0000198384, 0.0103878584, 0.0107447216,
    )
    info = _pair_boundary_deficit_aware_minimum_dots(
        boundary_target_grads=constraints,
        base_minimum_dots=base_floors,
        pair_target_weights=pair_delta.abs().view(-1),
        pre_boundary=pre_boundary,
        pair_local_delta=pair_delta,
        target_candidate_index=torch.arange(
            3, dtype=torch.float32
        ).view(1, 3),
        proposed_descent=(
            torch.full((3,), 1.0e-4, dtype=torch.float32),
        ),
        reference=reference,
    )
    floors = info["minimum_dots"]
    assert info["affordable_crossing_count"] == 0.0
    assert info["deficit_target_count"] == 2.0
    # The run120 satisfied target was already at its representable strict
    # floor; the useful change is concentrating the two deficit allocations.
    assert info["zero_deficit_reclaimed_budget"] == 0.0
    assert abs(sum(floors) - sum(base_floors)) <= 1.0e-6
    # Identity 2 has the smaller production boundary deficit, so it receives
    # all non-strict budget that cannot yet fund a complete crossing.
    assert floors[2] > 1.9 * base_floors[2]
    assert floors[1] < 1.0e-4
    assert info["budget_conservation_valid"] == 1.0


def test_run122_multi_exposure_identity_targets_nearest_member():
    """Do not let easy members mask the nearest boundary member."""
    reference = torch.tensor(0.0, dtype=torch.float32)
    constraints = (
        (torch.tensor([1.0, 0.0], dtype=torch.float32),),
        (torch.tensor([-0.2, 1.0], dtype=torch.float32),),
        (torch.tensor([-0.2, 1.0], dtype=torch.float32),),
    )
    pair_delta = torch.tensor(
        [[0.02], [0.02], [0.02]], dtype=torch.float32
    )
    pre_boundary = torch.tensor(
        [[-2.50], [-0.32], [-0.26]], dtype=torch.float32
    )
    base_floors = (0.0001, 0.0001, 0.0660)
    proposed = (torch.tensor([0.01, 0.01], dtype=torch.float32),)
    info = _pair_boundary_deficit_aware_minimum_dots(
        boundary_target_grads=constraints,
        base_minimum_dots=base_floors,
        pair_target_weights=pair_delta.abs().view(-1),
        pre_boundary=pre_boundary,
        pair_local_delta=pair_delta,
        target_candidate_index=torch.zeros(
            (3, 1), dtype=torch.float32
        ),
        proposed_descent=proposed,
        reference=reference,
    )
    assert info["identity_group_count"] == 1.0
    assert info["multi_exposure_identity_group_count"] == 1.0
    assert info["identity_group_member_indices"] == ((2,),)
    assert info["identity_group_progress_member_flags"] == (0, 0, 1)
    assert info["identity_group_progress_member_ordinals"] == (2, 2, 2)
    assert all(
        projection_floor == strict_floor
        for projection_floor, strict_floor in zip(
            info["projection_minimum_dots"],
            info["target_strict_floors"],
        )
    )
    group_constraint = constraints[2]
    safe, _ = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=proposed,
        constraint_grads=list(constraints) + [group_constraint],
        minimum_dots=(
            list(info["projection_minimum_dots"])
            + list(info["identity_group_minimum_dots"])
        ),
        reference=reference,
        diagnostic_name="run121 identity boundary replay",
    )
    row_dots = [
        float(_gradient_tuple_dot_for_test(safe, constraint))
        for constraint in constraints
    ]
    assert all(
        dot >= floor - 1.0e-6
        for dot, floor in zip(row_dots, info["target_strict_floors"])
    )
    group_dot = float(_gradient_tuple_dot_for_test(safe, group_constraint))
    assert group_dot >= info["identity_group_minimum_dots"][0] - 1.0e-6
    assert group_dot > row_dots[0]
    assert abs(
        info["identity_group_extra_budgets"][2]
        - (
            info["allocated_budget"]
            - info["identity_group_strict_budgets"][2]
        )
    ) <= 1.0e-6
    assert abs(
        info["allocated_budget"] - sum(base_floors)
    ) <= 1.0e-6


def _gradient_tuple_dot_for_test(left, right):
    total = torch.tensor(0.0, dtype=torch.float32)
    for left_value, right_value in zip(left, right):
        if left_value is not None and right_value is not None:
            total = total + (left_value * right_value).sum()
    return total


def _fake_train_info(step_before, class_complete=1.0):
    info = {
        train_key: 0.0
        for train_key in set(_ADJ_TRANSACTION_TRAIN_INFO_FIELDS.values())
    }
    info.update({
        "pair_optimizer_transaction_enabled": 1.0,
        "pair_optimizer_transaction_diagnostic_version": float(
            PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION
        ),
        "pair_optimizer_transaction_support_version": 6.0,
        "pair_optimizer_transaction_nonzero_pair": 1.0,
        "pair_optimizer_transaction_pair_grad_finite": 1.0,
        "pair_optimizer_transaction_pair_grad_zero": 0.0,
        "pair_optimizer_transaction_pair_grad_norm": 0.25,
        "pair_optimizer_transaction_gradient_clip_scale": 1.0,
        "pair_optimizer_transaction_final_parameters_equal_raw_adam": 1.0,
        "pair_optimizer_transaction_pair_optimizer_isolated": 0.0,
        "pair_optimizer_transaction_pair_actual_update_direction_guard_applied":
            0.0,
        "pair_optimizer_transaction_pair_optimizer_state_sync_applied": 0.0,
        "pair_optimizer_transaction_pair_target_gradient_constraint_count":
            2.0,
        "pair_optimizer_transaction_pair_target_gradient_min_dot_after": 1e-6,
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_after":
            1e-6,
        "pair_optimizer_transaction_pair_boundary_diagnostic_version": 8.0,
        "pair_optimizer_transaction_pair_boundary_gradient_constraint_count":
            2.0,
        "pair_optimizer_transaction_pair_boundary_gradient_min_dot_after": 1e-6,
        "pair_optimizer_transaction_pair_boundary_actual_min_descent_dot_after":
            1e-6,
        "pair_optimizer_transaction_pair_boundary_target_count": 2.0,
        "pair_optimizer_transaction_pair_boundary_correct_direction_count": 2.0,
        "pair_optimizer_transaction_positive_target_count": 1.0,
        "pair_optimizer_transaction_negative_target_count": 1.0,
        "pair_optimizer_transaction_score_correct_direction_target_count": 2.0,
        "pair_optimizer_transaction_score_before_after_join_valid": 1.0,
        "pair_optimizer_transaction_score_zero_tolerance": 1e-12,
        "pair_optimizer_transaction_objective_scalar_reconstruction_valid":
            1.0,
        "pair_optimizer_transaction_independent_sum_reconstruction_valid":
            1.0,
        "pair_optimizer_transaction_raw_combined_grad_norm_from_backward":
            0.5,
        "pair_optimizer_transaction_pre_projection_combined_grad_norm": 0.5,
        "pair_optimizer_transaction_post_projection_combined_grad_norm": 0.5,
        "pair_optimizer_transaction_combined_grad_norm_preclip": 0.5,
        "pair_optimizer_transaction_optimizer_step_before": float(
            step_before
        ),
        "pair_optimizer_transaction_optimizer_step_before_min": float(
            step_before
        ),
        "pair_optimizer_transaction_optimizer_step_before_max": float(
            step_before
        ),
        "pair_optimizer_transaction_optimizer_step_after": float(
            step_before + 1
        ),
        "pair_optimizer_transaction_optimizer_step_after_min": float(
            step_before + 1
        ),
        "pair_optimizer_transaction_optimizer_step_after_max": float(
            step_before + 1
        ),
        "pair_optimizer_class_complete": float(class_complete),
        "pair_pursuit_factor_loss_target_transition_count": 1.0,
        "pair_optimizer_positive_mass": 0.5,
        "pair_optimizer_negative_mass": 0.5,
        "pair_optimizer_centered_error": 0.0,
        "pair_pending_objective_scope_pair_only": 0.0,
    })
    info["_pair_selection_boundary_rows"] = [
        {
            field: (
                1
                if field in (
                    "margin_direction_correct",
                    "valid",
                    "linearized_budget_conservation_valid",
                )
                else 0
            )
            for field in _PAIR_SELECTION_BOUNDARY_TRACE_FIELDS
        }
        for _ in range(2)
    ]
    for sequence_index, row in enumerate(
            info["_pair_selection_boundary_rows"]):
        row["target_row_sequence_within_transaction"] = sequence_index
        row["diagnostic_version"] = 8
        row["target_candidate_index"] = sequence_index
        row["target_canonical_identity"] = "order2:0-{}".format(
            sequence_index + 1
        )
        row["target_sign"] = 1 if sequence_index == 0 else -1
        row["target_weight"] = 1.0
        row["pre_competitor_candidate_index"] = sequence_index + 2
        row["pre_competitor_canonical_identity"] = "order2:1-2"
        row["post_competitor_candidate_index"] = sequence_index + 2
        row["post_competitor_canonical_identity"] = "order2:1-2"
        row["signed_margin_change"] = 1e-4
    return info


def validate_two_epoch_unaggregated_csv():
    temp_root = PROJECT_ROOT / ".codex_tmp"
    temp_root.mkdir(exist_ok=True)
    run_dir = temp_root / "transaction_diag_test" / "run100"
    if run_dir.exists():
        resolved = run_dir.resolve()
        assert str(resolved).startswith(str(temp_root.resolve()))
        shutil.rmtree(str(resolved))
    run_dir.mkdir(parents=True)
    try:
        runner = RecRunner.__new__(RecRunner)
        runner.run_dir = run_dir
        runner.total_env_steps = 77600
        runner._adj_transaction_log_initialized = False
        policy_buffer = SimpleNamespace(
            last_sample_episode_count=10,
            last_sample_selected_chunk_count=100,
            last_sample_pair_optimizer_atomic_partition=1.0,
        )
        sample = (np.zeros((100, 1), dtype=np.float32),)
        records = []
        for ppo_epoch in (0, 1):
            records.append(runner._record_adj_transaction(
                train_adj_info=_fake_train_info(20 + ppo_epoch),
                policy_buffer=policy_buffer,
                sample=sample,
                adjacency_update_round=8,
                ppo_epoch_index=ppo_epoch,
                policy_id="policy_0",
                partition_index=0,
            ))
        _validate_adj_transaction_update_records(records, sequence_start=0)
        filename = _get_run_csv_name(
            run_dir,
            _ADJ_TRANSACTION_CSV_BASENAME,
        )
        with open(str(run_dir / filename), "r", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert len(rows) == 2
        assert len(rows[0]) == len(_ADJ_TRANSACTION_CSV_FIELDS)
        assert [int(row["ppo_epoch_index"]) for row in rows] == [0, 1]
        assert all(
            float(row["diagnostic_version"])
            == float(PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION)
            and float(row["objective_scalar_reconstruction_valid"]) == 1.0
            and float(row["independent_sum_reconstruction_valid"]) == 1.0
            for row in rows
        )
        assert [int(row["transaction_sequence_index"]) for row in rows] == [
            0,
            1,
        ]
        boundary_filename = _get_run_csv_name(
            run_dir,
            _PAIR_SELECTION_BOUNDARY_CSV_BASENAME,
        )
        with open(
                str(run_dir / boundary_filename),
                "r",
                newline="") as csv_file:
            boundary_rows = list(csv.DictReader(csv_file))
        assert len(boundary_rows) == 4
        assert all(
            int(row["margin_direction_correct"]) == 1
            and int(row["margin_direction_reverse"]) == 0
            and int(row["margin_direction_zero"]) == 0
            for row in boundary_rows
        )
        assert [float(row["optimizer_step_before"]) for row in rows] == [
            20.0,
            21.0,
        ]
        pair_info = _fake_train_info(0)
        pair_info["pair_pending_objective_scope_pair_only"] = 1.0
        pair_info[
            "pair_optimizer_transaction_pair_optimizer_isolated"
        ] = 1.0
        pair_info[
            "pair_optimizer_transaction_final_pair_descent_dot"
        ] = 1e-6
        pair_row = _build_adj_transaction_row(
            run_id="run112",
            env_step=144800,
            adjacency_update_round=173,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
            transaction_sequence_index=0,
            selected_episode_count=2,
            selected_chunk_count=20,
            transaction_chunk_count=20,
            train_adj_info=pair_info,
        )
        assert pair_row["optimizer_kind"] == "pair_pending_adam"
        assert pair_row["pair_optimizer_isolated"] == 1.0
        assert pair_row["pair_target_gradient_constraint_count"] == 2.0
        assert pair_row["score_correct_direction_target_count"] == 2.0
        assert _expected_adj_transaction_partition_count(
            class_complete=True,
            selected_chunk_count=100,
            num_mini_batch=2,
        ) == 1
        assert _expected_adj_transaction_partition_count(
            class_complete=False,
            selected_chunk_count=100,
            num_mini_batch=2,
        ) == 2

        ordinary_records = []
        for epoch in (0, 1):
            for partition in (0, 1):
                row = dict(records[0])
                row["ppo_epoch_index"] = epoch
                row["partition_index"] = partition
                row["class_complete"] = 0.0
                row["transaction_chunk_count"] = 50.0
                row["transaction_sequence_index"] = len(ordinary_records)
                ordinary_records.append(row)
        _validate_adj_transaction_update_records(
            ordinary_records,
            sequence_start=0,
        )
        assert len(ordinary_records) == 4

        duplicated = [dict(record) for record in ordinary_records]
        duplicated[1]["transaction_sequence_index"] = 0
        try:
            _validate_adj_transaction_update_records(
                duplicated,
                sequence_start=0,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate transaction sequence was accepted")
    finally:
        parent = run_dir.parent
        if parent.exists():
            resolved_parent = parent.resolve()
            assert str(resolved_parent).startswith(str(temp_root.resolve()))
            shutil.rmtree(str(resolved_parent))


def validate_pair_direction_candidate_schema():
    transaction = {
        "run_id": "run_test",
        "env_step": 150400,
        "adjacency_update_round": 188,
        "ppo_epoch_index": 3,
        "policy_id": "policy_0",
        "partition_index": 0,
        "transaction_sequence_index": 721,
        "optimizer_kind": "pair_pending_adam",
        "pair_boundary_direction_candidate_count": 6.0,
        "pair_boundary_direction_valid_candidate_count": 6.0,
        "pair_boundary_selected_progress_floor_fraction": 1.0,
    }
    rows = []
    direction_kinds = (
        "adam_projection",
        "deficit_progress_seed",
        "boundary_limiter_blend",
        "boundary_limiter_tangent",
        "boundary_limiter_bundle_blend",
        "boundary_limiter_bundle_tangent",
    )
    for ordinal, kind in enumerate(direction_kinds):
        fraction = 0.5 if kind == "deficit_progress_seed" else 1.0
        selected = int(kind == "boundary_limiter_bundle_tangent")
        required = 0.5
        completion = 0.2 + 0.1 * ordinal
        row = {
            "diagnostic_version": (
                PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION
            ),
            "candidate_ordinal": ordinal,
            "direction_kind": kind,
            "progress_floor_fraction": fraction,
            "direction_norm": 0.1 + 0.01 * ordinal,
            "cosine_vs_full": 1.0 if ordinal == 0 else 0.91,
            "progress_seed_member_count": 1,
            "progress_seed_member_ordinals": (
                "0" if kind.startswith("boundary_limiter") else "1"
            ),
            "progress_seed_zero_budget_excluded_count": 1,
            "progress_seed_zero_budget_excluded_ordinals": "0",
            "progress_seed_raw_norm": 0.2,
            "adam_reference_norm": 0.1,
            "progress_component_norm": 0.1,
            "active_constraint_count": 1,
            "active_constraint_ordinals": str(ordinal),
            "valid": 1,
            "halving_count": 0,
            "expansion_count": 0,
            "refinement_count": 0,
            "safe_lower_scale": 1.0,
            "safe_frontier_scale": 1.0,
            "unsafe_upper_scale": 1.0,
            "unsafe_upper_present": 0,
            "scale_limit": 1.0,
            "progress_min_completion": completion,
            "progress_mean_completion": completion,
            "progress_target_present": 1,
            "progress_worst_actual": required * completion,
            "progress_worst_required": required,
            "limiting_constraint_code": 0,
            "limiting_target_ordinal": -1,
            "selected": selected,
        }
        assert tuple(row.keys()) == _PAIR_DIRECTION_CANDIDATE_TRACE_FIELDS
        rows.append(row)
    output = _build_pair_direction_candidate_rows(transaction, rows)
    assert len(output) == 6
    assert sum(int(row["selected"]) for row in output) == 1
    assert tuple(row["direction_kind"] for row in output) == direction_kinds

    invalid_deficit_overlap = [dict(row) for row in rows]
    invalid_deficit_overlap[1]["progress_seed_member_ordinals"] = "0"
    try:
        _build_pair_direction_candidate_rows(
            transaction, invalid_deficit_overlap
        )
    except RuntimeError as error:
        assert "seed includes a zero-budget member" in str(error)
    else:
        raise AssertionError(
            "deficit seed accepted an excluded zero-budget member"
        )

    invalid = [dict(row) for row in rows]
    invalid[0]["direction_kind"] = "unversioned_future_direction"
    try:
        _build_pair_direction_candidate_rows(transaction, invalid)
    except RuntimeError as error:
        assert "unknown strict-pair direction kind" in str(error)
    else:
        raise AssertionError("unknown direction kind was accepted")


def validate_episode_reject_reason_fixed_csv():
    temp_root = PROJECT_ROOT / ".codex_tmp"
    temp_root.mkdir(exist_ok=True)
    run_dir = temp_root / "episode_funnel_csv_test" / "run105"
    if run_dir.exists():
        resolved = run_dir.resolve()
        assert str(resolved).startswith(str(temp_root.resolve()))
        shutil.rmtree(str(resolved))
    run_dir.mkdir(parents=True)
    try:
        runner = RecRunner.__new__(RecRunner)
        runner.run_dir = run_dir
        rows = [
            {
                "run_id": "run105",
                "env_step": 80800,
                "adjacency_update_round": 93,
                "policy_id": "policy_0",
                "episode_generation": 370,
                "reject_reason": "CANDIDATE_ONLY_NOT_ACTIVE",
            },
            {
                "run_id": "run105",
                "env_step": 80800,
                "adjacency_update_round": 93,
                "policy_id": "policy_0",
                "episode_generation": 369,
                "reject_reason": "NOT_A_SUCCESSFUL_CAPTURE_GAP",
            },
        ]
        runner._append_fixed_rows_csv(
            "progress_train_pair_evidence_episode.csv",
            rows,
        )
        runner._append_fixed_rows_csv(
            "progress_train_pair_evidence_episode.csv",
            [dict(rows[0], env_step=81600)],
        )
        filename = _get_run_csv_name(
            run_dir,
            "progress_train_pair_evidence_episode.csv",
        )
        with open(str(run_dir / filename), "r", newline="") as csv_file:
            persisted = list(csv.DictReader(csv_file))
        assert len(persisted) == 3
        assert list(persisted[0].keys()) == list(rows[0].keys())
        assert persisted[0]["reject_reason"] == "CANDIDATE_ONLY_NOT_ACTIVE"
        try:
            runner._append_fixed_rows_csv(
                "progress_train_pair_evidence_episode.csv",
                [{"run_id": "run105", "env_step": 82400}],
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "incompatible episode diagnostic schema was accepted"
            )
    finally:
        parent = run_dir.parent
        if parent.exists():
            resolved_parent = parent.resolve()
            assert str(resolved_parent).startswith(str(temp_root.resolve()))
            shutil.rmtree(str(resolved_parent))


def validate_combined_gradient_directions():
    reference = torch.tensor(0.0)
    pair = (torch.tensor([1.0, 0.0]),)
    aligned = (torch.tensor([2.0, 0.0]),)
    opposed = (torch.tensor([-3.0, 0.0]),)
    orthogonal = (torch.tensor([0.0, 4.0]),)
    aligned_info = _gradient_tuple_direction_diagnostics(
        pair,
        aligned,
        reference,
    )
    opposed_info = _gradient_tuple_direction_diagnostics(
        pair,
        opposed,
        reference,
    )
    orthogonal_info = _gradient_tuple_direction_diagnostics(
        pair,
        orthogonal,
        reference,
    )
    _assert_close(aligned_info["cosine"], 1.0)
    assert float(aligned_info["descent_component"]) > 0.0
    _assert_close(opposed_info["cosine"], -1.0)
    assert float(opposed_info["descent_component"]) < 0.0
    _assert_close(orthogonal_info["cosine"], 0.0)

    base = (torch.tensor([2.0, 0.0]),)
    other = (torch.tensor([-4.0, 0.0]),)
    combined = (base[0] + other[0],)
    base_info = _gradient_tuple_direction_diagnostics(
        pair,
        base,
        reference,
    )
    combined_info = _gradient_tuple_direction_diagnostics(
        pair,
        combined,
        reference,
    )
    assert float(base_info["dot"]) > 0.0
    assert float(combined_info["dot"]) < 0.0


def _linear_objective_decomposition(
        objective_vectors,
        active_overrides=None,
        device=torch.device("cpu")):
    parameter = torch.nn.Parameter(torch.tensor(
        [0.4, -0.3],
        dtype=torch.float32,
        device=device,
    ))
    active_overrides = active_overrides or {}
    objective_specs = []
    losses = []
    for objective_name in (
            "graph",
            "base_factor",
            "capture_outcome",
            "pair",
            "candidate",
            "entropy"):
        vector = torch.tensor(
            objective_vectors.get(objective_name, (0.0, 0.0)),
            dtype=parameter.dtype,
            device=device,
        )
        component_loss = (parameter * vector).sum()
        active = bool(active_overrides.get(
            objective_name,
            bool(torch.any(vector != 0.0).item()),
        ))
        objective_specs.append((
            objective_name,
            component_loss,
            active,
        ))
        losses.append(component_loss)
    total_loss = sum(losses[1:], losses[0])
    return parameter, _objective_gradient_decomposition_diagnostics(
        parameters=[parameter],
        objective_specs=tuple(objective_specs),
        total_loss=total_loss,
    )


def validate_per_objective_gradient_hand_calculation():
    vectors = {
        "graph": (0.25, 0.5),
        "base_factor": (0.5, -0.25),
        "capture_outcome": (-0.125, 0.25),
        "pair": (1.0, 0.0),
        "candidate": (-0.75, 0.125),
        # This is the already-signed effective ``-beta * entropy`` gradient.
        "entropy": (-0.2, 0.4),
    }
    _parameter, info = _linear_objective_decomposition(vectors)
    for objective_name, vector in vectors.items():
        expected_norm = float(np.linalg.norm(np.asarray(vector)))
        expected_pair_dot = float(vector[0])
        _assert_close(
            info["objectives"][objective_name]["grad_norm"],
            expected_norm,
            atol=1e-6,
        )
        _assert_close(
            info["objectives"][objective_name]["pair_dot"],
            expected_pair_dot,
            atol=1e-6,
        )
    expected_sum = np.sum(
        np.asarray(list(vectors.values()), dtype=np.float32),
        axis=0,
    )
    _assert_close(
        info["independent_sum_norm"],
        np.linalg.norm(expected_sum),
        atol=1e-6,
    )
    _assert_close(
        info["pair_independent_sum_dot"],
        expected_sum[0],
        atol=1e-6,
    )
    assert float(info["objective_scalar_reconstruction_valid"]) == 1.0
    assert float(info["independent_sum_reconstruction_valid"]) == 1.0

    beta = 0.25
    entropy_raw_vector = np.asarray([0.8, -0.4], dtype=np.float32)
    vectors["entropy"] = tuple((-beta * entropy_raw_vector).tolist())
    _parameter, entropy_info = _linear_objective_decomposition(vectors)
    entropy_objective = entropy_info["objectives"]["entropy"]
    _assert_close(
        entropy_objective["pair_dot"],
        -beta * entropy_raw_vector[0],
        atol=1e-6,
    )


def _assert_pair_flip_source(vectors, expected_source_names):
    _parameter, info = _linear_objective_decomposition(vectors)
    pair_info = info["objectives"]["pair"]
    base_info = info["objectives"]["base_factor"]
    assert float(pair_info["pair_dot"]) > 0.0
    assert float(base_info["pair_dot"]) > 0.0
    assert float(info["pair_independent_sum_dot"]) < 0.0
    negative_sources = {
        name
        for name, objective_info in info["objectives"].items()
        if float(objective_info["pair_dot"]) < 0.0
    }
    assert negative_sources == set(expected_source_names)
    largest_negative = min(
        expected_source_names,
        key=lambda name: float(info["objectives"][name]["pair_dot"]),
    )
    assert float(info["objectives"][largest_negative]["pair_dot"]) == min(
        float(info["objectives"][name]["pair_dot"])
        for name in expected_source_names
    )


def validate_run100_equivalent_objective_flip_cases():
    common = {
        "pair": (1.0, 0.0),
        "base_factor": (0.25, 0.0),
    }
    graph_flip = dict(common)
    graph_flip["graph"] = (-2.0, 0.0)
    _assert_pair_flip_source(graph_flip, ("graph",))

    outcome_flip = dict(common)
    outcome_flip["capture_outcome"] = (-2.0, 0.0)
    _assert_pair_flip_source(outcome_flip, ("capture_outcome",))

    candidate_flip = dict(common)
    candidate_flip["candidate"] = (-2.0, 0.0)
    _assert_pair_flip_source(candidate_flip, ("candidate",))

    joint_flip = dict(common)
    joint_flip.update({
        "graph": (-0.8, 0.0),
        "capture_outcome": (-0.7, 0.0),
        "candidate": (-0.6, 0.0),
    })
    _assert_pair_flip_source(
        joint_flip,
        ("graph", "capture_outcome", "candidate"),
    )


def validate_graph_return_advantage_isolation():
    """Replay the run102 graph-local leakage and its corrected direction."""
    factor_advantage = torch.tensor([[-0.5, 3.5]], dtype=torch.float32)
    replay_graph_advantage = torch.tensor([[-0.5]], dtype=torch.float32)
    factor_mask = torch.ones_like(factor_advantage)
    split = _separate_replay_graph_and_factor_advantages(
        factor_advantage=factor_advantage,
        replay_graph_advantage=replay_graph_advantage,
        factor_mask=factor_mask,
        graph_return_coefficient=1.0,
    )
    _assert_close(split["graph_advantage"], -0.5)
    _assert_close(split["legacy_graph_advantage"], 1.5)
    _assert_close(split["graph_advantage_contamination"], 2.0)
    assert torch.equal(
        split["local_factor_advantage"],
        torch.tensor([[0.0, 4.0]]),
    )

    # Pair and base-factor gradients agree. The legacy factor mean broadcasts a
    # large local credit into graph PPO and reverses their combined direction;
    # the dedicated replay graph return preserves the declared graph objective.
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    pair_loss = parameter
    base_factor_loss = 0.25 * parameter
    ratio = torch.exp(parameter)
    legacy_graph_loss = -ratio * split["legacy_graph_advantage"].reshape(())
    corrected_graph_loss = -ratio * split["graph_advantage"].reshape(())
    pair_grad = torch.autograd.grad(
        pair_loss,
        parameter,
        retain_graph=True,
    )[0]
    base_grad = torch.autograd.grad(
        base_factor_loss,
        parameter,
        retain_graph=True,
    )[0]
    legacy_graph_grad = torch.autograd.grad(
        legacy_graph_loss,
        parameter,
        retain_graph=True,
    )[0]
    corrected_graph_grad = torch.autograd.grad(
        corrected_graph_loss,
        parameter,
        retain_graph=True,
    )[0]
    assert float(pair_grad * base_grad) > 0.0
    assert float(pair_grad * legacy_graph_grad) < 0.0
    assert float(
        pair_grad * (pair_grad + base_grad + legacy_graph_grad)
    ) < 0.0
    assert float(pair_grad * corrected_graph_grad) > 0.0
    assert float(
        pair_grad * (pair_grad + base_grad + corrected_graph_grad)
    ) > 0.0

    # Padding/inactive factors cannot contaminate the graph-return source.
    padded = _separate_replay_graph_and_factor_advantages(
        factor_advantage=torch.tensor([[-0.5, 100.0]]),
        replay_graph_advantage=replay_graph_advantage,
        factor_mask=torch.tensor([[1.0, 0.0]]),
        graph_return_coefficient=1.0,
    )
    _assert_close(padded["legacy_graph_advantage"], -0.5)
    _assert_close(padded["graph_advantage_contamination"], 0.0)
    assert torch.equal(
        padded["local_factor_advantage"],
        torch.zeros_like(factor_advantage),
    )
    empty = _separate_replay_graph_and_factor_advantages(
        factor_advantage=torch.tensor([[7.0, -9.0]]),
        replay_graph_advantage=replay_graph_advantage,
        factor_mask=torch.zeros_like(factor_advantage),
        graph_return_coefficient=1.0,
    )
    _assert_close(empty["legacy_graph_advantage"], 0.0)
    _assert_close(empty["graph_advantage_contamination"], 0.0)
    assert torch.equal(
        empty["local_factor_advantage"],
        torch.zeros_like(factor_advantage),
    )

    scaled = _separate_replay_graph_and_factor_advantages(
        factor_advantage=torch.tensor([[-0.25, -0.25]]),
        replay_graph_advantage=replay_graph_advantage,
        factor_mask=factor_mask,
        graph_return_coefficient=0.5,
    )
    _assert_close(scaled["graph_advantage"], -0.25)
    _assert_close(scaled["graph_advantage_contamination"], 0.0)

    # A legal non-pair graph transition still has a non-zero graph PPO gradient.
    non_pair_parameter = torch.nn.Parameter(torch.tensor(0.0))
    non_pair_graph_loss = (
        -torch.exp(non_pair_parameter)
        * scaled["graph_advantage"].reshape(())
    )
    non_pair_graph_grad = torch.autograd.grad(
        non_pair_graph_loss,
        non_pair_parameter,
    )[0]
    assert float(non_pair_graph_grad.abs()) > 0.0

    for invalid_kwargs in (
            {
                "factor_advantage": torch.zeros(1, 2, 1),
                "replay_graph_advantage": torch.zeros(1, 1),
                "factor_mask": torch.zeros(1, 2),
                "graph_return_coefficient": 1.0,
            },
            {
                "factor_advantage": torch.zeros(1, 2),
                "replay_graph_advantage": torch.zeros(1, 2),
                "factor_mask": torch.zeros(1, 2),
                "graph_return_coefficient": 1.0,
            },
            {
                "factor_advantage": torch.zeros(1, 2),
                "replay_graph_advantage": torch.zeros(1, 1),
                "factor_mask": torch.zeros(1, 3),
                "graph_return_coefficient": 1.0,
            }):
        try:
            _separate_replay_graph_and_factor_advantages(**invalid_kwargs)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid graph/factor advantage shape accepted")


def validate_projection_stage_direction_changes():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameters = [parameter]
    pair = (torch.tensor([1.0, 0.0]),)
    reference = parameter.sum() * 0.0

    pre_correct = (torch.tensor([1.0, 0.0]),)
    post_reverse = (torch.tensor([-1.0, 0.0]),)
    changed_bad = _gradient_projection_delta_diagnostics(
        parameters,
        pre_correct,
        post_reverse,
        pair,
        reference,
    )
    assert float(changed_bad["pair_pre_projection_dot"]) > 0.0
    assert float(changed_bad["pair_delta_dot"]) < 0.0
    assert float(changed_bad["pair_post_projection_dot"]) < 0.0

    pre_reverse = (torch.tensor([-1.0, 0.0]),)
    post_correct = (torch.tensor([1.0, 0.0]),)
    changed_good = _gradient_projection_delta_diagnostics(
        parameters,
        pre_reverse,
        post_correct,
        pair,
        reference,
    )
    assert float(changed_good["pair_pre_projection_dot"]) < 0.0
    assert float(changed_good["pair_delta_dot"]) > 0.0
    assert float(changed_good["pair_post_projection_dot"]) > 0.0

    unchanged = _gradient_projection_delta_diagnostics(
        parameters,
        pre_correct,
        pre_correct,
        pair,
        reference,
    )
    assert float(unchanged["delta_norm"]) == 0.0


def validate_standard_pair_gradient_projection():
    reference = torch.tensor(0.0)
    pair_grads = (torch.tensor([1.0, 0.0]),)
    candidate_grads = (torch.tensor([0.0, 1.0]),)
    proposed = (torch.tensor([-1.0, 1.0]),)
    protected, info = _preserve_pair_gradient_in_standard_transaction(
        proposed_grads=proposed,
        pair_grads=pair_grads,
        candidate_grads=candidate_grads,
        reference=reference,
    )
    assert info["intervened"] == 1.0
    assert float(torch.dot(protected[0], pair_grads[0])) > 0.0
    assert float(torch.dot(protected[0], candidate_grads[0])) > 0.0
    assert torch.equal(protected[0], torch.tensor([1.0, 1.0]))

    nonconflicting = (torch.tensor([3.0, 1.0]),)
    unchanged, unchanged_info = (
        _preserve_pair_gradient_in_standard_transaction(
            proposed_grads=nonconflicting,
            pair_grads=pair_grads,
            candidate_grads=candidate_grads,
            reference=reference,
        )
    )
    assert unchanged_info["intervened"] == 0.0
    assert torch.equal(unchanged[0], nonconflicting[0])

    # Sequential candidate-then-pair PCGrad used to reject this production
    # shape: the pair-only result [1, 0.5] opposes candidate [-1, 1], although
    # the joint feasible cone is non-empty (for example [1, 2]).  The standard
    # transaction must repair both current objectives together rather than
    # aborting on the ordering artifact.
    feasible_conflict_pair = (torch.tensor([1.0, 0.0]),)
    feasible_conflict_candidate = (torch.tensor([-1.0, 1.0]),)
    feasible_conflict_proposed = (torch.tensor([0.0, 0.5]),)
    jointly_repaired, jointly_repaired_info = (
        _preserve_pair_gradient_in_standard_transaction(
            proposed_grads=feasible_conflict_proposed,
            pair_grads=feasible_conflict_pair,
            candidate_grads=feasible_conflict_candidate,
            reference=reference,
        )
    )
    assert jointly_repaired_info["intervened"] == 1.0
    assert jointly_repaired_info["joint_repair_intervened"] == 1.0
    assert float(torch.dot(
        jointly_repaired[0], feasible_conflict_pair[0]
    )) > 0.0
    assert float(torch.dot(
        jointly_repaired[0], feasible_conflict_candidate[0]
    )) > 0.0

    try:
        _preserve_pair_gradient_in_standard_transaction(
            proposed_grads=proposed,
            pair_grads=pair_grads,
            candidate_grads=(torch.tensor([-1.0, 0.0]),),
            reference=reference,
        )
    except RuntimeError as exc:
        assert "candidate descent" in str(exc)
    else:
        raise AssertionError(
            "incompatible current pair/candidate gradients were accepted"
        )


def validate_standard_zero_aggregate_pair_gradient_recovery():
    reference = torch.tensor(0.0)
    zero_pair = (torch.zeros(2),)
    target_grads = (
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([0.0, 1.0]),),
    )
    boundary_grads = ((torch.tensor([1.0, 1.0]),),)
    try:
        _preserve_pair_gradient_in_standard_transaction(
            proposed_grads=(torch.tensor([-1.0, -1.0]),),
            pair_grads=zero_pair,
            candidate_grads=(None,),
            reference=reference,
        )
    except RuntimeError as exc:
        assert "no adjacency gradient" in str(exc)
    else:
        raise AssertionError(
            "raw zero aggregate unexpectedly passed the old guard"
        )
    recovered, info = _recover_standard_zero_aggregate_pair_gradient(
        pair_grads=zero_pair,
        pair_target_score_grads=target_grads,
        pair_boundary_target_grads=boundary_grads,
        reference=reference,
    )
    assert info["recovered"] == 1.0
    assert info["recovered_norm"] > 0.0
    assert info["minimum_dot"] > 0.0
    for constraint in target_grads + boundary_grads:
        assert float(torch.dot(recovered[0], constraint[0])) > 0.0

    # The recovered common direction enters the existing standard PCGrad and
    # mass-preserving target-floor paths without weakening either contract.
    proposed = (torch.tensor([-1.0, -1.0]),)
    protected, _projection_info = (
        _preserve_pair_gradient_in_standard_transaction(
            proposed_grads=proposed,
            pair_grads=recovered,
            candidate_grads=(None,),
            reference=reference,
        )
    )
    assert float(torch.dot(protected[0], recovered[0])) > 0.0
    floors = _pair_target_mass_preserving_minimum_dots(
        pair_target_score_grads=target_grads,
        pair_target_weights=torch.tensor([0.5, 0.5]),
        pair_grads=recovered,
        reference=reference,
    )
    assert len(floors) == 2
    assert all(float(value) > 0.0 for value in floors)

    # Equal-and-opposite exact labels remain contradictory and fail before an
    # optimizer step; the standard recovery must not become a skip/no-op.
    try:
        _recover_standard_zero_aggregate_pair_gradient(
            pair_grads=zero_pair,
            pair_target_score_grads=(
                (torch.tensor([1.0, 0.0]),),
                (torch.tensor([-1.0, 0.0]),),
            ),
            pair_boundary_target_grads=(),
            reference=reference,
        )
    except RuntimeError as exc:
        assert "jointly infeasible" in str(exc)
    else:
        raise AssertionError(
            "contradictory zero-aggregate pair targets were accepted"
        )


def validate_target_local_pair_score_jacobian_constraints():
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    scores = torch.stack([
        parameter[0],
        parameter[0] + 0.1 * parameter[1],
        parameter[1],
    ])
    local_delta = torch.tensor([0.5, -0.25, -0.25])
    target_info = _exact_pair_target_score_gradient_constraints(
        target_factor_logp=scores,
        pair_local_delta=local_delta,
        parameters=(parameter,),
    )
    target_grads = target_info["constraints"]
    assert len(target_grads) == 3
    pair_grad = tuple(
        sum(
            float(weight) * grad
            for weight, (grad,) in zip(
                target_info["weights"], target_grads
            )
        )
        for _parameter_index in (0,)
    )
    pair_grads = (pair_grad[0],)
    floors = _pair_target_mass_preserving_minimum_dots(
        pair_target_score_grads=target_grads,
        pair_target_weights=target_info["weights"],
        pair_grads=pair_grads,
        reference=scores,
    )
    proposed = (torch.tensor([-0.25, 0.275]),)
    before_dots = [
        float(torch.dot(proposed[0], target_grad[0]))
        for target_grad in target_grads
    ]
    assert any(dot <= 0.0 for dot in before_dots)
    protected, info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=proposed,
        constraint_grads=target_grads,
        minimum_dots=floors,
        reference=scores,
        diagnostic_name="synthetic target-local pair gradient",
    )
    after_dots = [
        float(torch.dot(protected[0], target_grad[0]))
        for target_grad in target_grads
    ]
    assert info["intervened"] == 1.0
    assert all(
        dot > 0.0 and dot + 1.0e-5 >= floor
        for dot, floor in zip(after_dots, floors)
    )

    # Regression for the first v6 production failure: an infinitesimal
    # candidate floor lets the minimum-norm pair solution erase a real
    # candidate descent, which can then round through zero in float32.
    candidate_grad = (torch.tensor([0.0, 1.0]),)
    candidate_descent_before = 0.25
    jointly_protected, joint_info = (
        _project_gradient_tuple_to_minimum_dots(
            proposed_grads=(torch.tensor([0.0, candidate_descent_before]),),
            constraint_grads=(
                (torch.tensor([1.0, 1.0]),),
                (torch.tensor([1.0, -1.0]),),
                candidate_grad,
            ),
            minimum_dots=(0.5, 0.5, candidate_descent_before),
            reference=scores,
            diagnostic_name=(
                "synthetic exact-pair targets with real candidate floor"
            ),
        )
    )
    assert joint_info["intervened"] == 1.0
    assert float(torch.dot(
        jointly_protected[0], candidate_grad[0]
    )) >= candidate_descent_before
    assert float(torch.dot(
        jointly_protected[0], torch.tensor([1.0, 1.0])
    )) >= 0.5
    assert float(torch.dot(
        jointly_protected[0], torch.tensor([1.0, -1.0])
    )) >= 0.5

    unchanged, unchanged_info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=protected,
        constraint_grads=target_grads,
        minimum_dots=floors,
        reference=scores,
        diagnostic_name="synthetic nonconflicting target-local pair gradient",
    )
    assert unchanged_info["intervened"] == 0.0
    assert torch.equal(unchanged[0], protected[0])

    try:
        _project_gradient_tuple_to_minimum_dots(
            proposed_grads=(torch.zeros(2),),
            constraint_grads=(
                (torch.tensor([1.0, 0.0]),),
                (torch.tensor([-1.0, 0.0]),),
            ),
            minimum_dots=(0.1, 0.1),
            reference=scores,
            diagnostic_name="synthetic infeasible exact pair targets",
        )
    except RuntimeError as exc:
        assert "infeasible" in str(exc)
    else:
        raise AssertionError(
            "mutually incompatible exact pair target directions were accepted"
        )


def validate_pair_priority_supersedes_incompatible_lifecycle():
    reference = torch.tensor(0.0)
    candidate_grads = (torch.tensor([0.0, 1.0]),)
    pair_grads = (torch.tensor([1.0, 0.0]),)
    proposed = (torch.tensor([1.0, 1.0]),)
    lifecycle_constraints = [
        (torch.tensor([1.0, 1.0]),),
        (torch.tensor([-1.0, 0.0]),),
    ]
    projected, accepted, superseded, _info = (
        _project_with_current_candidate_priority(
            proposed_grads=proposed,
            candidate_grads=candidate_grads,
            lifecycle_constraint_grads=lifecycle_constraints,
            reference=reference,
            additional_priority_grads=(pair_grads,),
        )
    )
    assert accepted == [0]
    assert superseded == [1]
    assert float(torch.dot(projected[0], candidate_grads[0])) > 0.0
    assert float(torch.dot(projected[0], pair_grads[0])) > 0.0

    # run163 production regression: current pair priority superseded one cached
    # lifecycle row.  The delta was zeroed, but the old target mask still made
    # exact scale-zero replay compare sign(0) * margin with a positive old floor,
    # yielding the same material negative gap at every trial scale.
    lifecycle_delta = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    stale_target_mask, _ = _lifecycle_target_population_from_delta(
        lifecycle_delta
    )
    lifecycle_delta = lifecycle_delta * torch.tensor(
        [[0.0], [1.0]], dtype=torch.float32
    )
    lifecycle_margins = torch.tensor([[3.0], [1.0]], dtype=torch.float32)
    lifecycle_floor = torch.tensor(
        [[2.2804315090179443], [1.0]], dtype=torch.float32
    )
    lifecycle_signed_margin = torch.sign(lifecycle_delta) * lifecycle_margins
    lifecycle_tolerance = torch.zeros_like(lifecycle_floor)
    stale = _joint_exact_constraint_acceptance(
        signed_boundary_change=torch.tensor([[1.0]], dtype=torch.float32),
        actionable_pair_mask=torch.tensor([[1.0]], dtype=torch.float32),
        lifecycle_signed_margin=lifecycle_signed_margin,
        lifecycle_signed_floor=lifecycle_floor,
        lifecycle_target_mask=stale_target_mask,
        lifecycle_tolerance=lifecycle_tolerance,
    )
    assert not stale["lifecycle_valid"]
    _assert_close(stale["lifecycle_min_signed_gap"], -2.2804315090179443)

    refreshed_target_mask, refreshed_present = (
        _lifecycle_target_population_from_delta(lifecycle_delta)
    )
    assert refreshed_present
    refreshed = _joint_exact_constraint_acceptance(
        signed_boundary_change=torch.tensor([[1.0]], dtype=torch.float32),
        actionable_pair_mask=torch.tensor([[1.0]], dtype=torch.float32),
        lifecycle_signed_margin=lifecycle_signed_margin,
        lifecycle_signed_floor=lifecycle_floor,
        lifecycle_target_mask=refreshed_target_mask,
        lifecycle_tolerance=lifecycle_tolerance,
    )
    assert refreshed["valid"]
    assert refreshed["lifecycle_valid"]
    _assert_close(refreshed["lifecycle_min_signed_gap"], 0.0)

    empty_mask, empty_present = _lifecycle_target_population_from_delta(
        torch.zeros_like(lifecycle_delta)
    )
    assert not empty_present
    assert not bool(torch.any(empty_mask).item())


def validate_reversed_adam_priority_is_repaired_before_lifecycle():
    """The raw Adam displacement may oppose one current pair target."""
    reference = torch.tensor(0.0, dtype=torch.float32)
    proposed = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
    candidate_grads = (
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    reversed_pair_target = (
        torch.tensor([-1.0, 2.0], dtype=torch.float32),
    )
    lifecycle_constraints = [
        (torch.tensor([0.0, 1.0], dtype=torch.float32),),
    ]
    projected, accepted, superseded, info = (
        _project_with_current_candidate_priority(
            proposed_grads=proposed,
            candidate_grads=candidate_grads,
            lifecycle_constraint_grads=lifecycle_constraints,
            reference=reference,
            additional_priority_grads=(reversed_pair_target,),
        )
    )
    assert accepted == [0]
    assert superseded == []
    assert info["current_priority_repair_intervened"] == 1.0
    assert info["current_priority_min_dot_before"] < 0.0
    assert info["current_priority_min_dot_after_repair"] > 0.0
    assert float(torch.dot(projected[0], candidate_grads[0])) > 0.0
    assert float(torch.dot(projected[0], reversed_pair_target[0])) > 0.0
    assert float(torch.dot(projected[0], lifecycle_constraints[0][0])) >= 0.0


def validate_lifecycle_final_dot_uses_float32_resolution():
    """Regression for the post-pair lifecycle false failure.

    The joint projection accepts residuals at norm-scaled float32 precision.
    The final parameter-write check must use the same contract, while a
    material opposing displacement must still fail.
    """
    reference = torch.tensor(0.0, dtype=torch.float32)
    proposed = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
    roundoff_constraint = (
        torch.tensor([-1.0e-8, 1.0], dtype=torch.float32),
    )
    roundoff = _nonnegative_gradient_dot_with_tolerance(
        proposed_grads=proposed,
        constraint_grads=roundoff_constraint,
        reference=reference,
    )
    assert float(roundoff["dot"]) < 0.0
    assert float(roundoff["tolerance"]) > abs(float(roundoff["dot"]))
    assert bool(roundoff["valid"].item())

    material_constraint = (
        torch.tensor([-1.0e-2, 1.0], dtype=torch.float32),
    )
    material = _nonnegative_gradient_dot_with_tolerance(
        proposed_grads=proposed,
        constraint_grads=material_constraint,
        reference=reference,
    )
    assert float(material["dot"]) < -float(material["tolerance"])
    assert not bool(material["valid"].item())


def validate_objective_none_inactive_zero_and_fail_loud():
    first = torch.nn.Parameter(torch.tensor([0.5]))
    second = torch.nn.Parameter(torch.tensor([-0.25]))
    zero_first = first.sum() * 0.0
    zero_second = second.sum() * 0.0
    pair_loss = first.sum()
    graph_loss = 2.0 * second.sum()
    objective_specs = (
        ("graph", graph_loss, True),
        ("base_factor", zero_first, False),
        ("capture_outcome", zero_first, False),
        ("pair", pair_loss, True),
        ("candidate", zero_second, False),
        ("entropy", zero_second, False),
    )
    info = _objective_gradient_decomposition_diagnostics(
        parameters=[first, second],
        objective_specs=objective_specs,
        total_loss=pair_loss + graph_loss,
    )
    assert info["objectives"]["pair"]["grads"][0] is not None
    assert info["objectives"]["pair"]["grads"][1] is None
    assert info["objectives"]["graph"]["grads"][0] is None
    assert info["objectives"]["graph"]["grads"][1] is not None
    inactive = info["objectives"]["candidate"]
    assert float(inactive["active"]) == 0.0
    assert float(inactive["grad_norm"]) == 0.0
    assert float(inactive["pair_dot"]) == 0.0

    zero_parameter = torch.nn.Parameter(torch.tensor([0.75]))
    zero = zero_parameter.sum() * 0.0
    all_zero_specs = tuple(
        (name, zero, name == "graph")
        for name in (
            "graph",
            "base_factor",
            "capture_outcome",
            "pair",
            "candidate",
            "entropy",
        )
    )
    all_zero = _objective_gradient_decomposition_diagnostics(
        parameters=[zero_parameter],
        objective_specs=all_zero_specs,
        total_loss=zero,
    )
    assert float(all_zero["raw_combined_grad_norm"]) == 0.0
    assert float(all_zero["independent_sum_norm"]) == 0.0
    assert float(all_zero["objectives"]["graph"]["active"]) == 1.0
    assert float(all_zero["objectives"]["graph"]["grad_norm"]) == 0.0

    bad_scalar_specs = list(all_zero_specs)
    try:
        _objective_gradient_decomposition_diagnostics(
            parameters=[zero_parameter],
            objective_specs=tuple(bad_scalar_specs),
            total_loss=zero + 1.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("scalar objective mismatch was accepted")

    try:
        _gradient_reconstruction_diagnostics(
            parameters=[zero_parameter],
            reconstructed_grads=(torch.tensor([1.0]),),
            reference_grads=(torch.tensor([2.0]),),
            reference=zero,
            diagnostic_name="intentional_mismatch",
            fail_loud=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("gradient reconstruction mismatch was accepted")


def _run_instrumented_transaction(enable_diagnostics, device):
    torch.manual_seed(171)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(171)
    parameter = torch.nn.Parameter(torch.tensor(
        [0.4, -0.2, 0.7],
        dtype=torch.float32,
        device=device,
    ))
    optimizer = torch.optim.Adam([parameter], lr=0.01, eps=1e-5)
    pair_vector = torch.tensor(
        [0.5, -0.25, 0.75],
        dtype=parameter.dtype,
        device=device,
    )
    other_vector = torch.tensor(
        [0.1, 0.2, -0.3],
        dtype=parameter.dtype,
        device=device,
    )
    pair_loss = (parameter * pair_vector).sum()
    other_loss = (parameter * other_vector).sum()
    total_loss = pair_loss + other_loss
    pair_grads = torch.autograd.grad(
        pair_loss,
        [parameter],
        retain_graph=True,
        allow_unused=True,
    )
    optimizer.zero_grad()
    total_loss.backward()
    preclip_grad = parameter.grad.detach().clone()
    diagnostic_values = {}
    if enable_diagnostics:
        combined_pre = _clone_parameter_gradients([parameter])
        diagnostic_values["combined_pre"] = (
            _gradient_tuple_direction_diagnostics(
                pair_grads,
                combined_pre,
                total_loss,
            )
        )
    clip_return = torch.nn.utils.clip_grad_norm_([parameter], 0.25)
    postclip_grad = parameter.grad.detach().clone()
    parameter_before = [parameter.detach().clone()]
    if enable_diagnostics:
        combined_post = _clone_parameter_gradients([parameter])
        diagnostic_values["combined_post"] = (
            _gradient_tuple_direction_diagnostics(
                pair_grads,
                combined_post,
                total_loss,
            )
        )
        adam_before = _adam_state_before_step_diagnostics(
            optimizer,
            [parameter],
            pair_grads,
            total_loss,
        )
    optimizer.step()
    if enable_diagnostics:
        raw = _parameter_displacement_direction_diagnostics(
            [parameter],
            parameter_before,
            pair_grads,
            total_loss,
        )
        _validate_adam_step_increment(
            optimizer,
            [parameter],
            adam_before["parameter_steps_before"],
        )
        final = _parameter_displacement_direction_diagnostics(
            [parameter],
            parameter_before,
            pair_grads,
            total_loss,
        )
        delta = _displacement_delta_diagnostics(
            [parameter],
            parameter_before,
            raw["displacements"],
            total_loss,
        )
        diagnostic_values.update({
            "raw": raw,
            "final": final,
            "delta": delta,
        })
    optimizer_state = optimizer.state[parameter]
    cpu_rng_state = torch.get_rng_state().clone()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda" else None
    )
    return {
        "loss": total_loss.detach().cpu(),
        "preclip_grad": preclip_grad.detach().cpu(),
        "postclip_grad": postclip_grad.detach().cpu(),
        "clip_return": float(
            clip_return.detach().cpu().item()
            if torch.is_tensor(clip_return) else clip_return
        ),
        "parameter": parameter.detach().cpu(),
        "exp_avg": optimizer_state["exp_avg"].detach().cpu(),
        "exp_avg_sq": optimizer_state["exp_avg_sq"].detach().cpu(),
        "step": int(float(optimizer_state["step"])),
        "cpu_rng_state": cpu_rng_state,
        "cuda_rng_state": (
            cuda_rng_state.detach().cpu()
            if cuda_rng_state is not None else None
        ),
        "candidate_lifecycle_state": {"clock": 7, "cache_size": 3},
        "diagnostic_values": diagnostic_values,
    }


def validate_trajectory_neutrality(device):
    without = _run_instrumented_transaction(False, device)
    with_diagnostics = _run_instrumented_transaction(True, device)
    for field in (
            "loss",
            "preclip_grad",
            "postclip_grad",
            "parameter",
            "exp_avg",
            "exp_avg_sq",
            "cpu_rng_state"):
        assert torch.equal(without[field], with_diagnostics[field]), field
    _assert_close(without["clip_return"], with_diagnostics["clip_return"])
    assert without["step"] == with_diagnostics["step"]
    assert (
        without["candidate_lifecycle_state"]
        == with_diagnostics["candidate_lifecycle_state"]
    )
    if device.type == "cuda":
        assert torch.equal(
            without["cuda_rng_state"],
            with_diagnostics["cuda_rng_state"],
        )
    assert float(
        with_diagnostics["diagnostic_values"]["delta"]["delta_norm"]
    ) == 0.0
    assert float(
        with_diagnostics["diagnostic_values"]["delta"]["exact_equal"]
    ) == 1.0


class _MinimalPolicy(object):
    def __init__(self, device):
        self.act_dim = 7
        self.module = torch.nn.Linear(3, 7).to(device)

    def parameters(self):
        return list(self.module.parameters())

    def critic_fv_parameters(self):
        return []

    def critic_vtot_parameters(self):
        return []


def _full_transaction_args():
    return SimpleNamespace(
        max_player_num=4,
        num_factor=4,
        highest_orders=3,
        sparsity=1.0,
        seed=193,
        hidden_size=16,
        gat_heads=4,
        gat_negative_slope=0.2,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        adj_anneal_time=1000,
        require_connected_adj=False,
        use_popart=False,
        use_value_active_masks=False,
        use_per=False,
        per_eps=1e-6,
        use_huber_loss=False,
        huber_delta=10.0,
        clip_param=0.2,
        use_vfunction=False,
        lr=5e-4,
        critic_lr=5e-4,
        adj_lr=3e-4,
        tau=0.005,
        opti_eps=1e-5,
        weight_decay=0.0,
        use_dyn_graph=True,
        entropy_coef=0.01,
        adj_entropy_coef=0.0002,
        use_valuenorm=False,
        max_grad_norm=10.0,
        adj_max_grad_norm=10.0,
        gamma=0.99,
        use_adj_pair_triplet_complementary_credit=True,
        adj_pair_pursuit_credit_coef=0.10,
        adj_return_adv_coef=1.0,
        use_adj_ppo_stale_trust=False,
    )


def _make_full_transaction_batch(
        graph,
        device,
        candidate_target=False,
        candidate_target_sign=1.0,
        successful_candidate_context=False,
        successful_episode_context=False):
    chunks = 1
    steps = 2
    num_agents = 4
    num_factors = 4
    hidden_size = 16
    batch_size = chunks * steps
    adj = torch.zeros(
        batch_size,
        num_agents,
        num_factors,
        dtype=torch.float32,
        device=device,
    )
    pair_members = ((0, 1), (1, 2), (2, 3), (0, 3))
    for factor_index, members in enumerate(pair_members):
        for member in members:
            adj[:, member, factor_index] = 1.0
    rnn_obs = torch.linspace(
        -0.6,
        0.8,
        steps=batch_size * num_agents * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(batch_size, num_agents, hidden_size)
    dones = torch.zeros(
        batch_size,
        num_agents,
        1,
        dtype=torch.bool,
        device=device,
    )
    obs = torch.zeros(
        batch_size,
        num_agents,
        1,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        behavior_prob, _ = graph.evaluate_prob(
            obs=obs,
            rnn_obs=rnn_obs,
            use_adj_init=False,
            dones=dones,
            adj=adj,
            previous_adj=adj,
        )

    factor_shape = (chunks, steps, num_factors, 1)
    scalar_shape = (chunks, steps, 1)
    f_advts = np.asarray(
        [[[[0.3], [-0.2], [0.1], [-0.1]],
          [[-0.1], [0.2], [-0.3], [0.4]]]],
        dtype=np.float32,
    )
    pair_credit = np.zeros(factor_shape, dtype=np.float32)
    pair_credit[0, 0, 0, 0] = 0.1
    pair_credit[0, 1, 0, 0] = -0.1
    pair_quality = np.zeros(factor_shape, dtype=np.float32)
    pair_quality[0, 0, 0, 0] = 1.0
    pair_quality[0, 1, 0, 0] = -1.0
    num_candidate_factors = (
        num_agents * (num_agents - 1) // 2
        + num_agents * (num_agents - 1) * (num_agents - 2) // 6
    )
    provenance = np.asarray(
        [[[[1.0, 0.0, 700.0]], [[1.0, 0.0, 700.0]]]],
        dtype=np.float32,
    ).reshape(chunks, steps, 3)
    zeros_factor = np.zeros(factor_shape, dtype=np.float32)
    zeros_scalar = np.zeros(scalar_shape, dtype=np.float32)
    candidate_identity_delta = np.zeros(
        (chunks, steps, num_candidate_factors, 1),
        dtype=np.float32,
    )
    candidate_capture_context = np.zeros_like(candidate_identity_delta)
    success_gate = zeros_scalar.copy()
    candidate_behavior = np.zeros(
        (chunks, steps, num_candidate_factors, 4),
        dtype=np.float32,
    )
    if bool(candidate_target):
        # Canonical candidate index 1 is feasible for this fixed graph.  A unit
        # signed behavior margin keeps either target strictly unsatisfied and
        # therefore exercises the real candidate projection.
        candidate_identity_delta[0, 0, 1, 0] = (
            0.2 * float(candidate_target_sign)
        )
        candidate_behavior[0, 0, 1, :] = np.asarray(
            [
                float(candidate_target_sign),
                1.0,
                1.0,
                float(graph.candidate_policy_version),
            ],
            dtype=np.float32,
        )
    if bool(successful_candidate_context):
        candidate_capture_context[0, 0, 1, 0] = 1.0
        candidate_behavior[0, 0, 1, :] = np.asarray(
            [
                -1.0,
                1.0,
                1.0,
                float(graph.candidate_policy_version),
            ],
            dtype=np.float32,
        )
    if bool(successful_candidate_context or successful_episode_context):
        success_gate[0, 0, 0] = 1.0
    return (
        obs.reshape(chunks, steps, num_agents, 1).cpu().numpy(),
        np.zeros((chunks, steps, num_agents, 1), dtype=np.float32),
        dones.float().reshape(
            chunks, steps, num_agents, 1
        ).cpu().numpy(),
        np.zeros((chunks, steps, 1), dtype=np.float32),
        adj.reshape(
            chunks, steps, num_agents, num_factors
        ).cpu().numpy(),
        behavior_prob.reshape(
            chunks, steps, num_agents, num_factors
        ).cpu().numpy(),
        np.zeros((chunks, steps, 1), dtype=np.float32),
        f_advts,
        zeros_factor.copy(),
        zeros_factor.copy(),
        zeros_factor.copy(),
        zeros_factor.copy(),
        zeros_factor.copy(),
        zeros_factor.copy(),
        zeros_factor.copy(),
        pair_credit,
        pair_quality,
        zeros_factor.copy(),
        zeros_factor.copy(),
        rnn_obs.reshape(
            chunks, steps, num_agents, hidden_size
        ).cpu().numpy(),
        zeros_factor.copy(),
        zeros_scalar.copy(),
        zeros_scalar.copy(),
        zeros_scalar.copy(),
        zeros_scalar.copy(),
        zeros_scalar.copy(),
        adj.reshape(
            chunks, steps, num_agents, num_factors
        ).cpu().numpy(),
        success_gate,
        zeros_scalar.copy(),
        np.zeros(
            (chunks, steps, CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH),
            dtype=np.float32,
        ),
        candidate_identity_delta,
        candidate_capture_context,
        candidate_behavior,
        provenance,
        np.ones(
            (chunks, steps, num_agents, 7),
            dtype=np.float32,
        ),
    )


def _make_exact_context_retention_case(
        device, current_pair=False, selection_state_sensitive=False):
    torch.manual_seed(211)
    np.random.seed(211)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(211)
    args = _full_transaction_args()
    if bool(selection_state_sensitive):
        args.use_adj_advantage_triplet_scorer = True
        args.adj_triplet_credit_ema_alpha = 1.0
        args.adj_triplet_credit_score_scale = 0.05
        args.use_adj_triplet_credit_direct_rank = True
        args.adj_triplet_credit_rank_coef = 1.25
        args.adj_triplet_credit_min_multiplier = 0.1
        args.adj_triplet_credit_max_multiplier = 2.5
        args.adj_triplet_credit_negative_rank_scale = 1.0
        args.adj_triplet_credit_min_positive_fraction = 0.0
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )
    trainer = R_SDDFG(
        args=args,
        num_agents=4,
        policies={"policy_0": _MinimalPolicy(device)},
        adj_network=graph,
        policy_mapping_fn=lambda _agent_id: "policy_0",
        device=device,
        episode_length=2,
    )
    if bool(selection_state_sensitive):
        with torch.no_grad():
            graph.pair_credit_ema.zero_()
            graph.triplet_credit_ema.fill_(-0.05)
            graph.pair_credit_seen.fill_(100.0)
            graph.triplet_credit_seen.fill_(100.0)
    batch = list(_make_full_transaction_batch(graph, device))
    # This fixture is an ordinary PPO adjacency update.  There is no new pair
    # event which could legitimately supersede the saved crossing.
    if not bool(current_pair):
        batch[15] = np.zeros_like(batch[15])
        batch[16] = np.zeros_like(batch[16])
    if bool(selection_state_sensitive):
        batch[7] = np.full_like(batch[7], -1.0)
    batch = tuple(batch)
    rnn_obs = torch.as_tensor(batch[19], device=device).reshape(
        2, 4, 16
    )
    dones = torch.as_tensor(batch[2], device=device).reshape(
        2, 4, 1
    ).bool()
    adj = torch.as_tensor(batch[4], device=device).reshape(2, 4, 4)
    return trainer, batch, rnn_obs, dones, adj


def validate_exact_behavior_score_replay_join(device):
    """Boundary replay must expose the exact PPO behavior score it guards.

    A scheduled temperature plus a persistence point mass deliberately makes
    the pure selected-policy log-probability differ from the behavior
    likelihood.  The fused boundary replay must still equal ``evaluate_prob``
    exactly, which is the run-time join that prevents a searched candidate from
    passing one score and reversing the committed PPO exact target.
    """
    torch.manual_seed(229)
    np.random.seed(229)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(229)
    args = _full_transaction_args()
    args.use_adj_topology_persistence = True
    args.adj_exploration_mix = 0.10
    args.adj_sampling_temperature_start = 0.65
    args.adj_sampling_temperature_final = 0.65
    args.adj_greedy_sample_prob_start = 0.35
    args.adj_greedy_sample_prob_final = 0.35
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )
    batch = _make_full_transaction_batch(graph, device)
    obs = torch.as_tensor(batch[0], device=device).reshape(2, 4, 1)
    dones = torch.as_tensor(batch[2], device=device).reshape(2, 4, 1).bool()
    adj = torch.as_tensor(batch[4], device=device).reshape(2, 4, 4)
    rnn_obs = torch.as_tensor(batch[19], device=device).reshape(2, 4, 16)
    previous_adj = torch.as_tensor(
        batch[26], device=device
    ).reshape(2, 4, 4)
    with torch.no_grad():
        prob_adj, _ = graph.evaluate_prob(
            obs=obs,
            rnn_obs=rnn_obs,
            use_adj_init=False,
            dones=dones,
            adj=adj,
            previous_adj=previous_adj,
        )
        target_prob = torch.where(
            adj == 1,
            prob_adj,
            torch.ones_like(prob_adj),
        ).clamp_min(1e-8)
        factor_den = adj.sum(dim=1).clamp_min(1.0)
        expected_behavior_logp = (
            (torch.log(target_prob) * adj).sum(dim=1) / factor_den
        )
        replay = graph.evaluate_selected_factor_replay_boundaries(
            rnn_obs=rnn_obs,
            dones=dones,
            adj=adj,
            previous_adj=previous_adj,
            include_behavior_logp=True,
        )
    assert torch.allclose(
        replay["behavior_selected_logp"],
        expected_behavior_logp,
        atol=2e-6,
        rtol=2e-6,
    )
    assert float(torch.max(torch.abs(
        replay["behavior_selected_logp"] - replay["selected_logp"]
    )).detach().cpu().item()) > 1e-5


def validate_inactive_selected_factor_is_an_exact_search_boundary(device):
    """An oversized trial is unsafe, while direct production replay fails."""
    torch.manual_seed(241)
    np.random.seed(241)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(241)
    args = _full_transaction_args()
    args.highest_orders = 2
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )
    selected_score_active = [False]

    def fixed_pair_score(_self, attention, exist_mask):
        batch_size, num_agents = attention.shape[:2]
        pair_mask = (
            (exist_mask > 0.5).unsqueeze(2)
            & (exist_mask > 0.5).unsqueeze(1)
        )
        diagonal = torch.eye(
            num_agents, dtype=torch.bool, device=device
        ).reshape(1, num_agents, num_agents)
        pair_mask = pair_mask & (~diagonal)
        pair_score = torch.ones(
            batch_size,
            num_agents,
            num_agents,
            dtype=attention.dtype,
            device=device,
        ) * pair_mask.to(attention.dtype)
        if not bool(selected_score_active[0]):
            pair_score[:, 0, 1] = 0.0
            pair_score[:, 1, 0] = 0.0
        return pair_score, pair_mask

    graph._pair_score = types.MethodType(fixed_pair_score, graph)
    rnn_obs = torch.zeros(1, 4, 16, device=device)
    dones = torch.zeros(1, 4, 1, dtype=torch.bool, device=device)
    adj = torch.zeros(1, 4, 4, device=device)
    adj[0, (0, 1), 0] = 1.0
    try:
        graph.evaluate_selected_factor_replay_boundaries(
            rnn_obs=rnn_obs,
            dones=dones,
            adj=adj,
        )
    except SelectedFactorInactiveCandidateError as error:
        assert error.batch_index == 0
        assert error.factor_index == 0
        assert error.candidate_index == 0
    else:
        raise AssertionError(
            "direct production replay accepted an inactive selected factor"
        )

    inactive_error = SelectedFactorInactiveCandidateError(0, 0, 0)
    actionable = torch.ones(1, dtype=torch.bool, device=device)

    def evaluate_scale(scale):
        scale = float(scale)
        if scale > 0.75:
            return _joint_exact_inactive_selected_factor_acceptance(
                inactive_error
            )
        acceptance = _joint_exact_constraint_acceptance(
            signed_boundary_change=torch.full(
                (1,), scale, dtype=torch.float32, device=device
            ),
            actionable_pair_mask=actionable,
            signed_exact_score_change=torch.zeros(
                1, dtype=torch.float32, device=device
            ),
            preservation_tolerance=1.0e-6,
        )
        acceptance.update({
            "progress_target_present": True,
            "progress_worst_actual": scale,
            "progress_worst_required": 1.0,
            "progress_min_completion": scale,
            "progress_mean_completion": scale,
            "competitor_candidate_indices": (1,),
            "target_ranks": (1,),
            "target_active": (1,),
        })
        return acceptance

    result = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate_scale,
        max_halvings=20,
        refinement_steps=12,
        maximum_scale=1.0,
        max_expansions=20,
        select_by_progress=True,
    )
    assert bool(result["valid"])
    assert 0.749 < float(result["final_scale"]) <= 0.75
    assert result["invalid_upper_info"][
        "limiting_constraint_type"
    ] == "selected_factor_inactive"
    assert any(
        row["limiting_constraint_type"] == "selected_factor_inactive"
        for row in result["trial_trace"]
    )

    # Reproduce the production call site from the traceback: every scale at or
    # below one is still float32-quantized/invalid, so the search reaches the
    # resolution-expansion probes above one.  An inactive endpoint must bracket
    # the safe midpoint instead of aborting the entire transaction.
    def evaluate_resolution_scale(scale):
        scale = float(scale)
        if scale >= 2.0:
            oversized = _joint_exact_constraint_acceptance(
                signed_boundary_change=torch.ones(
                    1, dtype=torch.float32, device=device
                ),
                actionable_pair_mask=actionable,
                signed_exact_score_change=torch.zeros(
                    1, dtype=torch.float32, device=device
                ),
                preservation_tolerance=1.0e-6,
            )
            return _joint_exact_catalog_change_acceptance(
                acceptance=oversized,
                catalog_kind="candidate_catalog_changed",
            )
        signed_change = 1.0 if scale >= 1.25 else 0.0
        acceptance = _joint_exact_constraint_acceptance(
            signed_boundary_change=torch.full(
                (1,), signed_change, dtype=torch.float32, device=device
            ),
            actionable_pair_mask=actionable,
            signed_exact_score_change=torch.zeros(
                1, dtype=torch.float32, device=device
            ),
            preservation_tolerance=1.0e-6,
        )
        acceptance.update({
            "progress_target_present": True,
            "progress_worst_actual": scale,
            "progress_worst_required": 1.0,
            "progress_min_completion": scale,
            "progress_mean_completion": scale,
            "competitor_candidate_indices": (1,),
            "target_ranks": (1,),
            "target_active": (1,),
        })
        return acceptance

    resolution_result = _maximize_joint_exact_backtracking_scale(
        evaluate_scale=evaluate_resolution_scale,
        max_halvings=20,
        refinement_steps=12,
        maximum_scale=2.0,
        max_expansions=20,
        select_by_progress=True,
    )
    assert bool(resolution_result["valid"])
    assert 1.25 <= float(resolution_result["final_scale"]) < 2.0
    assert int(resolution_result["expansion_count"]) == 1
    assert any(
        row["limiting_constraint_type"] == "candidate_catalog_changed"
        and float(row["scale"]) == 2.0
        for row in resolution_result["trial_trace"]
    )
    assert not bool(resolution_result["invalid_upper_info"][
        "candidate_valid"
    ])
    assert resolution_result["invalid_upper_info"][
        "lifecycle_valid"
    ] is None

    lifecycle_origin = _joint_exact_constraint_acceptance(
        signed_boundary_change=torch.ones(
            1, dtype=torch.float32, device=device
        ),
        actionable_pair_mask=actionable,
        signed_exact_score_change=torch.zeros(
            1, dtype=torch.float32, device=device
        ),
        preservation_tolerance=1.0e-6,
    )
    lifecycle_crossing = _joint_exact_catalog_change_acceptance(
        acceptance=lifecycle_origin,
        catalog_kind="lifecycle_catalog_changed",
    )
    assert not bool(lifecycle_crossing["valid"])
    assert bool(lifecycle_crossing["candidate_valid"])
    assert not bool(lifecycle_crossing["lifecycle_valid"])
    assert lifecycle_crossing[
        "limiting_constraint_type"
    ] == "lifecycle_catalog_changed"

    # Replaying the unchanged origin still succeeds after the rejected probe;
    # no fallback is installed in the graph's production evaluator.
    selected_score_active[0] = True
    replay = graph.evaluate_selected_factor_replay_boundaries(
        rnn_obs=rnn_obs,
        dones=dones,
        adj=adj,
    )
    assert int(replay["selected_candidate_index"][0, 0].item()) == 0
    assert bool((replay["valid"][0, 0] > 0.0).item())


def validate_selection_boundary_policy_response_diagnostic(device):
    """The exact active-factor counterfactual is read-only and schema fixed."""
    torch.manual_seed(137)
    np.random.seed(137)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(137)
    args = _full_transaction_args()
    args.prev_act_inp = False
    args.epsilon_anneal_time = 1000
    args.use_ReLU = True
    args.use_orthogonal = True
    args.use_feature_normalization = True
    args.gain = 0.01
    args.msg_iterations = 4
    args.msg_normalized = True
    args.msg_anytime = True
    args.num_rank = 3
    args.lamda = 0.0
    policy = R_SDDFGPolicy(
        config={
            "args": args,
            "device": device,
            "num_agents": 4,
        },
        policy_config={
            "obs_space": Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "act_space": Discrete(7),
            "cent_obs_dim": 1,
        },
        train=True,
    )
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )
    trainer = R_SDDFG(
        args=args,
        num_agents=4,
        policies={"policy_0": policy},
        adj_network=graph,
        policy_mapping_fn=lambda _agent_id: "policy_0",
        device=device,
        episode_length=2,
    )
    obs = torch.linspace(
        -0.5, 0.5, steps=4, device=device
    ).reshape(1, 4, 1)
    rnn_obs = torch.linspace(
        -0.8, 0.9, steps=4 * 16, device=device
    ).reshape(1, 4, 16)
    dones = torch.zeros(1, 4, 1, dtype=torch.bool, device=device)
    adj = torch.zeros(1, 4, 4, device=device)
    for factor_index, nodes in enumerate(
            ((0, 1), (1, 2), (2, 3), (0, 3))):
        adj[0, list(nodes), factor_index] = 1.0
    available_actions = torch.ones(1, 4, 7, device=device)
    available_actions[0, 0, 0] = 0.0
    trace_row = {
        "target_row_sequence_within_transaction": 0,
        "transition_index_in_partition": 0,
        "factor_index": 0,
        "target_candidate_index": 0,
        "target_canonical_identity": "0-1",
        "target_sign": 1.0,
        "pre_competitor_candidate_index": 1,
        "pre_competitor_canonical_identity": "0-2",
        "post_competitor_candidate_index": 1,
        "post_competitor_canonical_identity": "0-2",
        "boundary_crossing": 1,
        "positive_promotion": 1,
        "negative_eviction": 0,
    }
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda" else None
    )
    numpy_rng_before = copy.deepcopy(np.random.get_state())
    policy_rng_before = copy.deepcopy(policy.rng.get_state())
    state_before = trainer._pair_selection_boundary_policy_state_digest(
        policy
    )
    rows = trainer._pair_selection_boundary_policy_response_diagnostics(
        trace_rows=[trace_row],
        obs=obs,
        rnn_obs=rnn_obs,
        dones=dones,
        adj=adj,
        available_actions=available_actions,
        policy_id="policy_0",
    )
    assert len(rows) == 1
    response = rows[0]
    assert tuple(response.keys()) == (
        _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_TRACE_FIELDS
    )
    assert response["crossing_kind"] == "promotion"
    assert response["pre_active_canonical_identity"] == "0-2"
    assert response["post_active_canonical_identity"] == "0-1"
    assert float(response["structure_input_diff_norm"]) > 0.0
    assert float(response["observation_input_diff_norm"]) == 0.0
    assert float(response["rnn_state_input_diff_norm"]) == 0.0
    assert int(response["policy_response_nonzero"]) == 1
    assert int(response["rng_neutral"]) == 1
    assert int(response["state_neutral"]) == 1
    assert trainer._pair_selection_boundary_policy_state_digest(
        policy
    ) == state_before
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    if device.type == "cuda":
        assert torch.equal(
            torch.cuda.get_rng_state(device), cuda_rng_before
        )
    np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
    np.testing.assert_equal(policy.rng.get_state(), policy_rng_before)
    transaction_row = {
        "run_id": "run137-policy-response",
        "env_step": 153600,
        "adjacency_update_round": 77,
        "ppo_epoch_index": 1,
        "policy_id": "policy_0",
        "partition_index": 0,
        "transaction_sequence_index": 731,
        "optimizer_kind": "standard_adam",
        "pair_boundary_rank_crossing_count": 1.0,
    }
    persisted = _build_pair_selection_boundary_policy_response_rows(
        transaction_row=transaction_row,
        trace_rows=rows,
    )
    assert len(persisted) == 1
    assert int(persisted[0]["env_step"]) == 153600
    temp_root = PROJECT_ROOT / ".codex_tmp"
    temp_root.mkdir(exist_ok=True)
    run_dir = temp_root / "policy_response_diag_test" / "run137"
    if run_dir.exists():
        resolved = run_dir.resolve()
        assert str(resolved).startswith(str(temp_root.resolve()))
        shutil.rmtree(str(resolved))
    run_dir.mkdir(parents=True)
    runner = RecRunner.__new__(RecRunner)
    runner.run_dir = run_dir
    runner.total_env_steps = 153600
    runner._adj_transaction_log_initialized = False
    runner._adj_transaction_sequence_index = 731
    train_info = _fake_train_info(40)
    train_info[
        "pair_optimizer_transaction_pair_boundary_rank_crossing_count"
    ] = 1.0
    train_info[
        "pair_optimizer_transaction_pair_boundary_positive_promotion_count"
    ] = 1.0
    train_info["_pair_selection_boundary_rows"][0][
        "boundary_crossing"
    ] = 1
    train_info["_pair_selection_boundary_rows"][0][
        "positive_promotion"
    ] = 1
    train_info["_pair_selection_boundary_rows"][0][
        "pre_active_at_replay_boundary"
    ] = 0
    train_info["_pair_selection_boundary_rows"][0][
        "post_active_at_replay_boundary"
    ] = 1
    train_info["_pair_selection_boundary_policy_response_rows"] = rows
    runner._record_adj_transaction(
        train_adj_info=train_info,
        policy_buffer=SimpleNamespace(
            last_sample_episode_count=1,
            last_sample_selected_chunk_count=1,
            last_sample_pair_optimizer_atomic_partition=1.0,
        ),
        sample=(np.zeros((1, 1), dtype=np.float32),),
        adjacency_update_round=77,
        ppo_epoch_index=1,
        policy_id="policy_0",
        partition_index=0,
    )
    response_filename = _get_run_csv_name(
        run_dir,
        _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_CSV_BASENAME,
    )
    with open(
            str(run_dir / response_filename),
            "r",
            newline="") as csv_file:
        recorded_rows = list(csv.DictReader(csv_file))
    assert len(recorded_rows) == 1
    assert int(recorded_rows[0]["transaction_sequence_index"]) == 0
    assert recorded_rows[0]["policy_context_sha256"] == (
        response["policy_context_sha256"]
    )
    malformed = dict(rows[0])
    malformed["diagnostic_version"] = 2
    try:
        _build_pair_selection_boundary_policy_response_rows(
            transaction_row=transaction_row,
            trace_rows=[malformed],
        )
    except RuntimeError as exc:
        assert "unexpected selection-boundary policy response" in str(exc)
    else:
        raise AssertionError("unknown policy response schema was accepted")
    no_crossing = trainer._pair_selection_boundary_policy_response_diagnostics(
        trace_rows=[],
        obs=obs,
        rnn_obs=rnn_obs,
        dones=dones,
        adj=adj,
        available_actions=available_actions,
        policy_id="policy_0",
    )
    assert no_crossing == []
    cleanup_root = (temp_root / "policy_response_diag_test").resolve()
    assert str(cleanup_root).startswith(str(temp_root.resolve()))
    shutil.rmtree(str(cleanup_root))
    return response


def _retention_control_flow_fingerprint(trainer):
    def _update(digest, value):
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                digest.update(str(key).encode("utf-8"))
                _update(digest, value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                _update(digest, item)
        else:
            digest.update(repr(value).encode("utf-8"))

    def _digest(value):
        result = hashlib.sha256()
        _update(result, value)
        return result.hexdigest()

    observations = trainer._pair_selection_boundary_retention_observations
    active_entries = [
        entry for _key, entry in sorted(observations.items())
        if not bool(entry.get("protection_stopped", False))
    ]
    return {
        "parameter_sha256": _digest([
            parameter.detach() for parameter in trainer.adj_parameters
        ]),
        "optimizer_sha256": _digest(
            trainer.adj_optimizer.state_dict()
        ),
        "archive_sha256": _digest(observations),
        "selection_state_sha256": _digest(
            trainer
            ._pair_selection_boundary_retention_selection_state_snapshot()
        ),
        "pair_credit_seen": tuple(
            float(value)
            for value in trainer.adj_network.pair_credit_seen.detach()
            .cpu().reshape(-1).tolist()
        ),
        "triplet_credit_seen": tuple(
            float(value)
            for value in trainer.adj_network.triplet_credit_seen.detach()
            .cpu().reshape(-1).tolist()
        ),
        "context_sha256": tuple(
            str(entry["selection_context_sha256"])
            for entry in active_entries
        ),
        "commit_floors": tuple(
            float(entry["commit_signed_margin"])
            for entry in active_entries
        ),
        "ordinary_clock": int(
            trainer._pair_selection_boundary_ordinary_update_clock
        ),
        "ages": tuple(
            int(trainer._pair_selection_boundary_ordinary_update_clock)
            - int(entry["created_at"])
            for entry in active_entries
        ),
    }


def _run_exact_context_retention_case(
        device,
        protect,
        current_pair=False,
        transition_index=1,
        factor_index=3,
        force_exact_reject=False,
        force_first_exact_false_accept=False,
        force_precheck_floor_loss=False,
        selection_state_sensitive=False,
        target_sign=-1.0,
        force_late_selection_state_reject=False,
        transaction_modes=None):
    trainer, batch, rnn_obs, dones, adj = (
        _make_exact_context_retention_case(
            device,
            current_pair=current_pair,
            selection_state_sensitive=selection_state_sensitive,
        )
    )
    with torch.no_grad():
        commit = (
            trainer.adj_network
            .evaluate_selected_factor_replay_boundaries(
                rnn_obs=rnn_obs,
                dones=dones,
                adj=adj,
            )
        )
    if bool(protect):
        # This deterministic production batch loses the signed eviction
        # margin for transition 1/factor 3 without v25.  Archive exactly that
        # immutable context as an already-realized negative crossing.
        target_sign = float(target_sign)
        target_index = int(commit[
            "selected_candidate_index"
        ][transition_index, factor_index].item())
        competitor_index = int(commit[
            "competitor_candidate_index"
        ][transition_index, factor_index].item())
        _, target_identity = _canonical_candidate_identity(
            candidate_index=target_index,
            num_agents=4,
            highest_orders=trainer.adj_network.highest_orders,
        )
        _, competitor_identity = _canonical_candidate_identity(
            candidate_index=competitor_index,
            num_agents=4,
            highest_orders=trainer.adj_network.highest_orders,
        )
        commit_signed_margin = float(
            target_sign
            * commit["selected_margin"][
                transition_index, factor_index
            ].item()
        )
        commit_rank = int(commit[
            "selected_rank"
        ][transition_index, factor_index].item())
        assert commit_signed_margin > 0.0
        if target_sign > 0.0:
            assert commit_rank == 1
        else:
            assert commit_rank > 1
        trainer._pair_selection_boundary_retention_observations[0] = {
            "observation_id": 0,
            "source_policy_id": "policy_0",
            "source_transaction_sequence_index": -1,
            "source_adjacency_update_round": 0,
            "source_episode_ordinal": 0,
            "source_episode_step": 700,
            "selection_context_sha256": (
                trainer
                ._pair_selection_boundary_retention_context_digest(
                    rnn_obs=rnn_obs[
                        transition_index:transition_index + 1
                    ],
                    dones=dones[transition_index:transition_index + 1],
                    adj=adj[transition_index:transition_index + 1],
                    factor_index=factor_index,
                    target_candidate_index=target_index,
                    target_sign=target_sign,
                )
            ),
            "rnn_obs": rnn_obs[
                transition_index:transition_index + 1
            ].detach().clone(),
            "dones": dones[
                transition_index:transition_index + 1
            ].detach().clone(),
            "adj": adj[
                transition_index:transition_index + 1
            ].detach().clone(),
            "factor_index": factor_index,
            "target_candidate_index": target_index,
            "target_canonical_identity": target_identity,
            "target_sign": target_sign,
            "commit_competitor_candidate_index": competitor_index,
            "commit_competitor_canonical_identity": competitor_identity,
            "pre_signed_margin": -0.1,
            "commit_signed_margin": commit_signed_margin,
            "commit_rank": commit_rank,
            "commit_active": int(commit_rank == 1),
            "created_at": 0,
            "protection_stopped": False,
            "protection_stop_reason": "",
            "protection_stop_clock": -1,
        }
        trainer._pair_selection_boundary_retention_next_id = 1
    else:
        target_sign = float(target_sign)
        commit_signed_margin = float(
            target_sign
            * commit["selected_margin"][
                transition_index, factor_index
            ].item()
        )

    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda" else None
    )
    numpy_rng_before = copy.deepcopy(np.random.get_state())
    graph_rng_before = copy.deepcopy(trainer.adj_network.rng.get_state())
    graph_eval_rng_before = copy.deepcopy(
        trainer.adj_network.eval_rng.get_state()
    )
    if bool(force_precheck_floor_loss):
        state_before = trainer.pair_pending_outer_transaction_state()
        trainer._pair_selection_boundary_retention_observations[0][
            "commit_signed_margin"
        ] = commit_signed_margin + 1.0
        state_before = trainer.pair_pending_outer_transaction_state()
        try:
            trainer.train_adj_on_batch(
                batch,
                use_adj_init=False,
                adj_update_round=1,
                enable_optimizer_transaction_diagnostics=True,
                diagnostic_policy_id="policy_0",
                diagnostic_transaction_sequence_index=0,
            )
        except RuntimeError as exc:
            assert "floor was already lost before adjacency parameter update" in str(exc)
        else:
            raise AssertionError("lost retention floor did not fail loudly")
        _assert_nested_equal(
            state_before,
            trainer.pair_pending_outer_transaction_state(),
        )
        assert torch.equal(torch.get_rng_state(), cpu_rng_before)
        if device.type == "cuda":
            assert torch.equal(
                torch.cuda.get_rng_state(device), cuda_rng_before
            )
        np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
        np.testing.assert_equal(
            trainer.adj_network.rng.get_state(), graph_rng_before
        )
        np.testing.assert_equal(
            trainer.adj_network.eval_rng.get_state(),
            graph_eval_rng_before,
        )
        return {"precheck_rollback_verified": True}
    if bool(force_exact_reject):
        state_before = trainer.pair_pending_outer_transaction_state()
        original_acceptance = (
            trainer._pair_selection_boundary_retention_exact_acceptance
        )

        def reject_exact_writeback(_self, _keyed_entries):
            return {
                "valid": False,
                "context_invalid_keys": [],
                "violation_count": 1,
                "min_signed_gap": None,
                "max_tolerance": None,
            }

        trainer._pair_selection_boundary_retention_exact_acceptance = (
            types.MethodType(reject_exact_writeback, trainer)
        )
        try:
            trainer.train_adj_on_batch(
                batch,
                use_adj_init=False,
                adj_update_round=1,
                enable_optimizer_transaction_diagnostics=True,
                diagnostic_policy_id="policy_0",
                diagnostic_transaction_sequence_index=0,
            )
        except RuntimeError as exc:
            assert "no finite exact writeback" in str(exc)
        else:
            raise AssertionError("invalid retention writeback was committed")
        finally:
            trainer._pair_selection_boundary_retention_exact_acceptance = (
                original_acceptance
            )
        _assert_nested_equal(
            state_before,
            trainer.pair_pending_outer_transaction_state(),
        )
        assert torch.equal(torch.get_rng_state(), cpu_rng_before)
        if device.type == "cuda":
            assert torch.equal(
                torch.cuda.get_rng_state(device), cuda_rng_before
            )
        np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
        np.testing.assert_equal(
            trainer.adj_network.rng.get_state(), graph_rng_before
        )
        np.testing.assert_equal(
            trainer.adj_network.eval_rng.get_state(),
            graph_eval_rng_before,
        )
        return {"rollback_verified": True}
    if bool(force_late_selection_state_reject):
        if not bool(selection_state_sensitive):
            raise AssertionError(
                "late selection-state rejection requires a sensitive fixture"
            )
        state_before = trainer.pair_pending_outer_transaction_state()
        original_interpolate = (
            trainer
            ._interpolate_pair_selection_boundary_retention_selection_state
        )

        def keep_unsafe_selection_state(_self, before, after, scale):
            del before, scale
            _self._restore_pair_selection_boundary_retention_selection_state(
                after
            )

        trainer._interpolate_pair_selection_boundary_retention_selection_state = (
            types.MethodType(keep_unsafe_selection_state, trainer)
        )
        try:
            trainer.train_adj_on_batch(
                batch,
                use_adj_init=False,
                adj_update_round=1,
                enable_optimizer_transaction_diagnostics=True,
                diagnostic_policy_id="policy_0",
                diagnostic_transaction_sequence_index=694,
            )
        except PairOptimizerRecoverableNoOpError as exc:
            assert exc.reason == (
                "selection_state_no_compatible_exact_writeback"
            )
            assert int(exc.reason_code) == 9
            assert bool(exc.atomic_rollback_complete)
            assert float(exc.diagnostics[
                "selection_state_trial_count"
            ]) == 26.0
        else:
            raise AssertionError(
                "incompatible late production state was committed"
            )
        finally:
            trainer._interpolate_pair_selection_boundary_retention_selection_state = (
                original_interpolate
            )
        _assert_nested_equal(
            state_before,
            trainer.pair_pending_outer_transaction_state(),
        )
        assert torch.equal(torch.get_rng_state(), cpu_rng_before)
        if device.type == "cuda":
            assert torch.equal(
                torch.cuda.get_rng_state(device), cuda_rng_before
            )
        np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
        continuation_info, _priorities, _idxes = trainer.train_adj_on_batch(
            batch,
            use_adj_init=False,
            adj_update_round=1,
            enable_optimizer_transaction_diagnostics=True,
            diagnostic_policy_id="policy_0",
            diagnostic_transaction_sequence_index=695,
        )
        assert float(continuation_info[
            "pair_optimizer_transaction_diagnostic_version"
        ]) == float(PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION)
        return {
            "late_selection_state_rollback_verified": True,
            "late_selection_state_continuation_verified": True,
        }
    first_acceptance_call_count = [0]
    original_acceptance = None
    if bool(force_first_exact_false_accept):
        # Reproduce the run130 control-flow signature: the first ordinary
        # transaction's intermediate acceptance claims success although the
        # production context remains below its commit floor.  v25 returned that
        # transaction and failed only at the next pre-check.  v26 must recheck
        # the complete transaction, backtrack it, and never expose the bad age1
        # state.  This is fault injection around the real production forward and
        # optimizer path, not a scalar-only synthetic margin test.
        original_acceptance = (
            trainer._pair_selection_boundary_retention_exact_acceptance
        )

        def false_accept_first_writeback(_self, keyed_entries):
            actual = original_acceptance(keyed_entries)
            first_acceptance_call_count[0] += 1
            if first_acceptance_call_count[0] == 1:
                false_result = dict(actual)
                false_result["valid"] = True
                false_result["violation_count"] = 0
                return false_result
            return actual

        trainer._pair_selection_boundary_retention_exact_acceptance = (
            types.MethodType(false_accept_first_writeback, trainer)
        )
    observations = []
    train_infos = []
    control_flow_trace = []
    active_transaction_sequence = [-1]
    original_constraint_population = None
    original_factor_credit_update = None
    original_credit_gate_update = None
    if bool(selection_state_sensitive):
        original_constraint_population = (
            trainer
            ._pair_selection_boundary_retention_constraint_population
        )
        original_factor_credit_update = (
            trainer.adj_network.update_factor_credit_memory
        )
        original_credit_gate_update = (
            trainer.adj_network.update_order3_credit_gate
        )

        def traced_constraint_population(_self, reference):
            control_flow_trace.append((
                "precheck_{}".format(active_transaction_sequence[0]),
                _retention_control_flow_fingerprint(trainer),
            ))
            return original_constraint_population(reference)

        def traced_factor_credit_update(*args, **kwargs):
            control_flow_trace.append((
                "selection_state_before_{}".format(
                    active_transaction_sequence[0]
                ),
                _retention_control_flow_fingerprint(trainer),
            ))
            result = original_factor_credit_update(*args, **kwargs)
            control_flow_trace.append((
                "factor_credit_after_{}".format(
                    active_transaction_sequence[0]
                ),
                _retention_control_flow_fingerprint(trainer),
            ))
            return result

        def traced_credit_gate_update(*args, **kwargs):
            result = original_credit_gate_update(*args, **kwargs)
            control_flow_trace.append((
                "selection_state_full_after_{}".format(
                    active_transaction_sequence[0]
                ),
                _retention_control_flow_fingerprint(trainer),
            ))
            return result

        trainer._pair_selection_boundary_retention_constraint_population = (
            types.MethodType(traced_constraint_population, trainer)
        )
        trainer.adj_network.update_factor_credit_memory = (
            traced_factor_credit_update
        )
        trainer.adj_network.update_order3_credit_gate = (
            traced_credit_gate_update
        )
    if transaction_modes is None:
        transaction_modes = (False, False)
    transaction_modes = tuple(bool(value) for value in transaction_modes)
    if len(transaction_modes) != 2:
        raise AssertionError("retention fixture requires exactly two writes")
    if any(transaction_modes) and not bool(current_pair):
        raise AssertionError(
            "pair-only retention fixture requires current pair evidence"
        )
    original_policy_response_diagnostics = None
    if any(transaction_modes) or bool(current_pair):
        # The minimal fixture policy intentionally omits the production Q API.
        # Policy-response rows are read-only and orthogonal to the retention
        # write contract under test, so suppress only that diagnostic while
        # keeping the real exact boundary/Adam path and transaction clocks.
        original_policy_response_diagnostics = (
            trainer._pair_selection_boundary_policy_response_diagnostics
        )

        def no_policy_response(_self, **_kwargs):
            return []

        trainer._pair_selection_boundary_policy_response_diagnostics = (
            types.MethodType(no_policy_response, trainer)
        )
    try:
        for transaction_sequence_index, pair_only_objective in enumerate(
                transaction_modes):
            active_transaction_sequence[0] = 694 + transaction_sequence_index
            train_info, _priorities, _idxes = trainer.train_adj_on_batch(
                batch,
                use_adj_init=False,
                adj_update_round=transaction_sequence_index + 1,
                pair_only_objective=pair_only_objective,
                consume_factor_credit_observations=(
                    transaction_sequence_index == 0
                    and not pair_only_objective
                ),
                enable_optimizer_transaction_diagnostics=True,
                diagnostic_policy_id="policy_0",
                diagnostic_transaction_sequence_index=(
                    transaction_sequence_index
                ),
            )
            train_infos.append(train_info)
            control_flow_trace.append((
                "return_{}".format(active_transaction_sequence[0]),
                _retention_control_flow_fingerprint(trainer),
            ))
            with torch.no_grad():
                current = (
                    trainer.adj_network
                    .evaluate_selected_factor_replay_boundaries(
                        rnn_obs=rnn_obs,
                        dones=dones,
                        adj=adj,
                    )
                )
            observations.append({
                "signed_margin": float(
                    target_sign
                    * current["selected_margin"][
                        transition_index, factor_index
                    ].item()
                ),
                "rank": int(current[
                    "selected_rank"
                ][transition_index, factor_index].item()),
                "active": int(
                    current["selected_rank"][
                        transition_index, factor_index
                    ].item() == 1.0
                ),
                "competitor": int(current[
                    "competitor_candidate_index"
                ][transition_index, factor_index].item()),
            })
    finally:
        if original_acceptance is not None:
            trainer._pair_selection_boundary_retention_exact_acceptance = (
                original_acceptance
            )
        if original_constraint_population is not None:
            trainer._pair_selection_boundary_retention_constraint_population = (
                original_constraint_population
            )
            trainer.adj_network.update_factor_credit_memory = (
                original_factor_credit_update
            )
            trainer.adj_network.update_order3_credit_gate = (
                original_credit_gate_update
            )
        if original_policy_response_diagnostics is not None:
            trainer._pair_selection_boundary_policy_response_diagnostics = (
                original_policy_response_diagnostics
            )
    return {
        "trainer": trainer,
        "commit_signed_margin": commit_signed_margin,
        "observations": observations,
        "train_infos": train_infos,
        "cpu_rng_before": cpu_rng_before,
        "cpu_rng_after": torch.get_rng_state().clone(),
        "cuda_rng_before": cuda_rng_before,
        "cuda_rng_after": (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda" else None
        ),
        "numpy_rng_before": numpy_rng_before,
        "numpy_rng_after": copy.deepcopy(np.random.get_state()),
        "graph_rng_before": graph_rng_before,
        "graph_rng_after": copy.deepcopy(
            trainer.adj_network.rng.get_state()
        ),
        "graph_eval_rng_before": graph_eval_rng_before,
        "graph_eval_rng_after": copy.deepcopy(
            trainer.adj_network.eval_rng.get_state()
        ),
        "first_acceptance_call_count": int(
            first_acceptance_call_count[0]
        ),
        "control_flow_trace": control_flow_trace,
    }


def validate_exact_context_retention_floor(device):
    unprotected = _run_exact_context_retention_case(device, protect=False)
    protected = _run_exact_context_retention_case(device, protect=True)
    commit_floor = protected["commit_signed_margin"]
    assert unprotected["observations"][0]["signed_margin"] < commit_floor
    for age_index, observation in enumerate(
            protected["observations"], start=1):
        assert observation["signed_margin"] >= commit_floor - 1e-6
        assert observation["rank"] > 1
        assert observation["active"] == 0
        rows = protected["train_infos"][age_index - 1][
            "_pair_selection_boundary_retention_rows"
        ]
        assert len(rows) == 1
        assert int(rows[0]["ordinary_update_age"]) == age_index
        assert int(rows[0]["context_valid"]) == 1
        assert int(rows[0]["margin_nonregression"]) == 1
        assert int(rows[0]["rank_retained"]) == 1
        assert int(rows[0]["active_retained"]) == 1
        assert int(rows[0]["protection_stopped"]) == 0
        info = protected["train_infos"][age_index - 1]
        assert float(info[
            "pair_selection_boundary_retention_protected_target_count"
        ]) == 1.0
        assert float(info[
            "pair_selection_boundary_retention_superseded_count"
        ]) == 0.0
        assert not info.get("_candidate_evidence_consumption_rows", ())
    assert float(protected["train_infos"][0][
        "pair_selection_boundary_retention_actual_projection_corrected"
    ]) == 1.0
    assert float(protected["train_infos"][0][
        "pair_selection_boundary_retention_optimizer_state_sync_applied"
    ]) == 1.0
    assert not protected[
        "trainer"
    ]._pair_selection_boundary_retention_observations
    assert torch.equal(
        protected["cpu_rng_before"], protected["cpu_rng_after"]
    )
    if device.type == "cuda":
        assert torch.equal(
            protected["cuda_rng_before"], protected["cuda_rng_after"]
        )
    np.testing.assert_equal(
        protected["numpy_rng_before"], protected["numpy_rng_after"]
    )
    np.testing.assert_equal(
        protected["graph_rng_before"], protected["graph_rng_after"]
    )
    np.testing.assert_equal(
        protected["graph_eval_rng_before"],
        protected["graph_eval_rng_after"],
    )

    # A genuinely current strict-pair target keeps priority.  The incompatible
    # historical floor is stopped instead of replaying or overriding the new
    # evidence, and its generation is still consumed only by the current path.
    superseded = _run_exact_context_retention_case(
        device,
        protect=True,
        current_pair=True,
        transition_index=0,
        factor_index=0,
    )
    superseded_info = superseded["train_infos"][0]
    assert float(superseded_info[
        "pair_selection_boundary_retention_superseded_count"
    ]) == 1.0
    assert float(superseded_info[
        "pair_selection_boundary_retention_protected_target_count"
    ]) == 0.0
    superseded_rows = superseded_info[
        "_pair_selection_boundary_retention_rows"
    ]
    assert len(superseded_rows) == 1
    assert int(superseded_rows[0]["protection_stopped"]) == 1
    assert superseded_rows[0]["protection_stop_reason"] == (
        "incompatible_current_actionable_evidence"
    )
    aligned_current_pair = _run_exact_context_retention_case(
        device,
        protect=True,
        current_pair=True,
        transition_index=1,
        factor_index=0,
    )
    assert all(
        observation["signed_margin"]
        >= aligned_current_pair["commit_signed_margin"] - 1e-6
        for observation in aligned_current_pair["observations"]
    )
    assert float(aligned_current_pair["train_infos"][0][
        "pair_selection_boundary_retention_superseded_count"
    ]) == 0.0
    # The production runner executes bounded-pending pair-only writes after the
    # ordinary PPO transactions in the same adjacency round.  They do not age
    # the two-ordinary-transaction archive, but they do mutate the same
    # adjacency Parameters.  An aligned pending write must therefore preserve
    # the archived floor, and the following ordinary write must not discover a
    # floor that was lost between ordinary clocks.
    interleaved_pair_only = _run_exact_context_retention_case(
        device,
        protect=True,
        current_pair=True,
        transition_index=1,
        factor_index=0,
        transaction_modes=(True, False),
    )
    assert all(
        observation["signed_margin"]
        >= interleaved_pair_only["commit_signed_margin"] - 1e-6
        for observation in interleaved_pair_only["observations"]
    )
    pair_only_info, ordinary_info = interleaved_pair_only["train_infos"]
    assert float(pair_only_info[
        "pair_selection_boundary_retention_protected_target_count"
    ]) == 1.0
    assert float(pair_only_info[
        "pair_selection_boundary_retention_superseded_count"
    ]) == 0.0
    assert not pair_only_info["_pair_selection_boundary_retention_rows"]
    assert int(interleaved_pair_only["trainer"].
               _pair_selection_boundary_ordinary_update_clock) == 1
    ordinary_rows = ordinary_info[
        "_pair_selection_boundary_retention_rows"
    ]
    assert len(ordinary_rows) == 1
    assert int(ordinary_rows[0]["ordinary_update_age"]) == 1
    assert int(ordinary_rows[0]["margin_nonregression"]) == 1
    # This exact production context is intentionally incompatible with the
    # pending target.  Before v34 the pending write silently crossed the saved
    # floor and the next ordinary pre-check raised the run-time error reproduced
    # by run163.  The current target now wins explicitly in the pair-only
    # transaction, and the historical floor is marked superseded atomically.
    interleaved_pair_only_conflict = _run_exact_context_retention_case(
        device,
        protect=True,
        current_pair=True,
        transition_index=0,
        factor_index=0,
        transaction_modes=(True, False),
    )
    conflict_pair_info = interleaved_pair_only_conflict["train_infos"][0]
    assert float(conflict_pair_info[
        "pair_selection_boundary_retention_superseded_count"
    ]) == 1.0
    conflict_entry = interleaved_pair_only_conflict[
        "trainer"
    ]._pair_selection_boundary_retention_observations[0]
    assert bool(conflict_entry["protection_stopped"])
    assert conflict_entry["protection_stop_reason"] == (
        "incompatible_current_actionable_evidence"
    )
    assert int(conflict_entry["protection_stop_clock"]) == 0
    run130_commit_gap = _run_exact_context_retention_case(
        device,
        protect=True,
        force_first_exact_false_accept=True,
    )
    assert run130_commit_gap["first_acceptance_call_count"] >= 2
    assert all(
        observation["signed_margin"]
        >= run130_commit_gap["commit_signed_margin"] - 1e-6
        for observation in run130_commit_gap["observations"]
    )
    assert all(
        int(row["margin_nonregression"]) == 1
        for info in run130_commit_gap["train_infos"]
        for row in info["_pair_selection_boundary_retention_rows"]
    )
    run130_precheck = _run_exact_context_retention_case(
        device,
        protect=True,
        force_precheck_floor_loss=True,
    )
    assert run130_precheck["precheck_rollback_verified"]
    rejected = _run_exact_context_retention_case(
        device,
        protect=True,
        force_exact_reject=True,
    )
    assert rejected["rollback_verified"]


def validate_run131_late_selection_state_postcondition(device):
    result = _run_exact_context_retention_case(
        device,
        protect=True,
        transition_index=0,
        factor_index=0,
        selection_state_sensitive=True,
        target_sign=1.0,
    )
    first_info = result["train_infos"][0]
    assert float(first_info[
        "pair_selection_boundary_retention_final_postcondition_entered"
    ]) == 1.0
    assert float(first_info[
        "pair_selection_boundary_retention_final_postcondition_target_count"
    ]) == 1.0
    assert float(first_info[
        "pair_selection_boundary_retention_selection_state_backtrack_count"
    ]) > 0.0
    state_scale = float(first_info[
        "pair_selection_boundary_retention_selection_state_final_scale"
    ])
    assert 0.0 <= state_scale < 1.0
    backtrack_count = int(first_info[
        "pair_selection_boundary_retention_selection_state_backtrack_count"
    ])
    refinement_count = int(first_info[
        "pair_selection_boundary_retention_selection_state_refinement_count"
    ])
    unsafe_upper_scale = float(first_info[
        "pair_selection_boundary_retention_selection_state_unsafe_upper_scale"
    ])
    first_dyadic_safe = 0.5 ** backtrack_count
    assert refinement_count == 12
    assert first_dyadic_safe <= state_scale < unsafe_upper_scale
    assert unsafe_upper_scale - state_scale <= (
        first_dyadic_safe / (2.0 ** refinement_count)
        + 1e-12
    )
    assert float(first_info[
        "pair_selection_boundary_retention_selection_state_seen_count_delta"
    ]) > 0.0
    assert float(first_info[
        "pair_selection_boundary_retention_selection_state_seen_count_integral_valid"
    ]) == 1.0
    assert float(first_info[
        "adv_factor_credit_memory_observation_consumed"
    ]) == 1.0
    assert float(result["train_infos"][1][
        "adv_factor_credit_memory_observation_consumed"
    ]) == 0.0
    assert float(result["train_infos"][1][
        "pair_selection_boundary_retention_selection_state_seen_count_delta"
    ]) == 0.0
    assert all(
        observation["signed_margin"]
        >= result["commit_signed_margin"] - 1e-6
        for observation in result["observations"]
    )
    assert all(
        int(row["margin_nonregression"]) == 1
        and int(row["rank_retained"]) == 1
        and int(row["active_retained"]) == 1
        for info in result["train_infos"]
        for row in info["_pair_selection_boundary_retention_rows"]
    )
    assert not any(
        info.get("_candidate_evidence_consumption_rows", ())
        for info in result["train_infos"]
    )
    component_rows = first_info[
        "_pair_selection_boundary_retention_component_rows"
    ]
    assert len(component_rows) == 6
    assert all(
        tuple(row.keys())
        == _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_TRACE_FIELDS
        for row in component_rows
    )
    assert {
        row["component"] for row in component_rows
    } == {
        "pair_credit_ema",
        "triplet_credit_ema",
        "order3_credit_loss_ema",
        "order3_credit_margin_ema",
        "current_order3_credit_gate",
        "all_continuous",
    }
    assert any(
        int(row["floor_retained"]) == 0
        for row in component_rows
        if float(row["component_delta_norm"]) > 0.0
    )
    persisted_component_rows = (
        _build_pair_selection_boundary_retention_component_rows(
            transaction_row={
                "run_id": "run133_fixture",
                "env_step": 145600,
                "adjacency_update_round": 174,
                "ppo_epoch_index": 0,
                "policy_id": "policy_0",
                "partition_index": 0,
                "transaction_sequence_index": 694,
                "optimizer_kind": "standard_adam",
            },
            trace_rows=component_rows,
        )
    )
    assert len(persisted_component_rows) == len(component_rows)

    trace = dict(result["control_flow_trace"])
    before = trace["selection_state_before_694"]
    full_after = trace["selection_state_full_after_694"]
    returned = trace["return_694"]
    next_precheck = trace["precheck_695"]
    for field in ("parameter_sha256", "optimizer_sha256"):
        assert before[field] == full_after[field]
        assert full_after[field] == returned[field]
        assert returned[field] == next_precheck[field]
    assert before["selection_state_sha256"] != full_after[
        "selection_state_sha256"
    ]
    assert returned["selection_state_sha256"] != full_after[
        "selection_state_sha256"
    ]
    assert returned["selection_state_sha256"] == next_precheck[
        "selection_state_sha256"
    ]


    # The continuous credit/gate proposal is backtracked, but each real
    # factor observation is still counted once.  This is the run132
    # regression: v27 multiplied the count delta by 1/2048 and 1/32768.
    assert before["pair_credit_seen"] != full_after["pair_credit_seen"]
    assert returned["pair_credit_seen"] == full_after["pair_credit_seen"]
    assert returned["triplet_credit_seen"] == full_after[
        "triplet_credit_seen"
    ]
    assert returned["pair_credit_seen"] == next_precheck[
        "pair_credit_seen"
    ]
    assert returned["triplet_credit_seen"] == next_precheck[
        "triplet_credit_seen"
    ]
    for seen_counts in (
            returned["pair_credit_seen"],
            returned["triplet_credit_seen"]):
        assert all(
            abs(float(value) - round(float(value))) <= 1e-6
            for value in seen_counts
        )
    assert returned["archive_sha256"] == next_precheck["archive_sha256"]
    assert returned["context_sha256"] == next_precheck["context_sha256"]
    assert returned["commit_floors"] == next_precheck["commit_floors"]
    assert returned["ordinary_clock"] == next_precheck["ordinary_clock"]
    assert returned["ages"] == next_precheck["ages"]
    checkpoint = result["trainer"].adjacency_optimizer_checkpoint_state()
    resumed, _batch, _rnn_obs, _dones, _adj = (
        _make_exact_context_retention_case(
            device,
            selection_state_sensitive=True,
        )
    )
    resumed.load_adjacency_optimizer_checkpoint_state(checkpoint)
    _assert_nested_equal(
        result["trainer"]
        ._pair_selection_boundary_retention_selection_state_snapshot(),
        resumed
        ._pair_selection_boundary_retention_selection_state_snapshot(),
    )
    rejected = _run_exact_context_retention_case(
        device,
        protect=True,
        transition_index=0,
        factor_index=0,
        selection_state_sensitive=True,
        target_sign=1.0,
        force_late_selection_state_reject=True,
    )
    assert rejected["late_selection_state_rollback_verified"]
    assert rejected["late_selection_state_continuation_verified"]


def validate_current_transaction_final_crossing_floor(device):
    """A new floor must equal the complete transaction's returned state."""
    trainer, batch, _rnn_obs, _dones, _adj = (
        _make_exact_context_retention_case(
            device,
            current_pair=True,
            selection_state_sensitive=True,
        )
    )
    # The compact fixture policy deliberately has no production action API;
    # policy-response rows are read-only and orthogonal to this optimizer and
    # retention contract.
    original_policy_response = (
        trainer._pair_selection_boundary_policy_response_diagnostics
    )

    def no_policy_response(_self, **_kwargs):
        return []

    trainer._pair_selection_boundary_policy_response_diagnostics = (
        types.MethodType(no_policy_response, trainer)
    )
    try:
        first_info, _priorities, _idxes = trainer.train_adj_on_batch(
            batch,
            use_adj_init=False,
            adj_update_round=1,
            consume_factor_credit_observations=True,
            enable_optimizer_transaction_diagnostics=True,
            diagnostic_policy_id="policy_0",
            diagnostic_transaction_sequence_index=0,
        )
        assert float(first_info[
            "pair_selection_boundary_retention_new_count"
        ]) > 0.0
        keyed_entries = list(
            trainer._pair_selection_boundary_retention_observations.items()
        )
        assert keyed_entries
        with torch.no_grad():
            returned = (
                trainer._pair_selection_boundary_retention_forward(
                    keyed_entries
                )
            )
        for row_index, (_key, entry) in enumerate(keyed_entries):
            location = (row_index, int(entry["factor_index"]))
            returned_signed_margin = float(
                float(entry["target_sign"])
                * returned["selected_margin"][location]
                .detach().cpu().item()
            )
            _assert_close(
                returned_signed_margin,
                float(entry["commit_signed_margin"]),
                atol=2e-6,
            )

        # The next ordinary production update must see the exact same saved
        # floor at precheck, then preserve or explicitly supersede it.  Before
        # v40 the archive contained the pre-credit-state intermediate margin,
        # so this boundary could fail before optimizer.step.
        second_info, _priorities, _idxes = trainer.train_adj_on_batch(
            batch,
            use_adj_init=False,
            adj_update_round=2,
            consume_factor_credit_observations=False,
            enable_optimizer_transaction_diagnostics=True,
            diagnostic_policy_id="policy_0",
            diagnostic_transaction_sequence_index=1,
        )
        assert float(second_info[
            "pair_selection_boundary_retention_protected_target_count"
        ]) > 0.0
        assert all(
            int(row["margin_nonregression"]) == 1
            for row in second_info[
                "_pair_selection_boundary_retention_rows"
            ]
        )
    finally:
        trainer._pair_selection_boundary_policy_response_diagnostics = (
            original_policy_response
        )


def _recoverable_noop_state_snapshot(trainer):
    return {
        "parameters": tuple(
            parameter.detach().clone()
            for parameter in trainer.adj_parameters
        ),
        "optimizer": copy.deepcopy(trainer.adj_optimizer.state_dict()),
        "lifecycle": copy.deepcopy(trainer._candidate_identity_lifecycle),
        "lifecycle_observations": copy.deepcopy(
            trainer._candidate_identity_lifecycle_observations
        ),
        "selection_state": copy.deepcopy(
            trainer
            ._pair_selection_boundary_retention_selection_state_snapshot()
        ),
        "retention": copy.deepcopy(
            trainer._pair_selection_boundary_retention_observations
        ),
        "retention_next_id": int(
            trainer._pair_selection_boundary_retention_next_id
        ),
        "ordinary_clock": int(
            trainer._pair_selection_boundary_ordinary_update_clock
        ),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() else None
        ),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
        "python_rng": copy.deepcopy(__import__("random").getstate()),
    }


def _assert_recoverable_noop_state_equal(left, right):
    for left_parameter, right_parameter in zip(
            left["parameters"], right["parameters"]):
        assert torch.equal(left_parameter, right_parameter)
    for field in (
            "optimizer",
            "lifecycle",
            "lifecycle_observations",
            "selection_state",
            "retention",
            "retention_next_id",
            "ordinary_clock",
            "cpu_rng",
            "cuda_rng",
            "numpy_rng",
            "python_rng"):
        _assert_nested_equal(left[field], right[field])


def validate_pair_adam_zero_displacement_is_recoverable(device):
    """Production regression for the former line-13817 hard failure."""
    trainer, batch, _rnn_obs, _dones, _adj = (
        _make_exact_context_retention_case(
            device,
            current_pair=True,
            selection_state_sensitive=True,
        )
    )
    original_policy_response = (
        trainer._pair_selection_boundary_policy_response_diagnostics
    )

    def no_policy_response(_self, **_kwargs):
        return []

    trainer._pair_selection_boundary_policy_response_diagnostics = (
        types.MethodType(no_policy_response, trainer)
    )
    original_step = trainer.adj_optimizer.step

    def finite_zero_displacement_step(*args, **kwargs):
        before = tuple(
            parameter.detach().clone()
            for parameter in trainer.adj_parameters
        )
        result = original_step(*args, **kwargs)
        # Faithfully reproduce a finite float32-quantized Adam no-op while
        # leaving Adam's attempted state mutation in place. Production must
        # restore both populations atomically before reporting a no-op.
        with torch.no_grad():
            for parameter, origin in zip(trainer.adj_parameters, before):
                parameter.copy_(origin)
        return result

    trainer.adj_optimizer.step = finite_zero_displacement_step
    runner = object.__new__(RecRunner)
    runner.trainer = trainer
    try:
        stable_origin = _recoverable_noop_state_snapshot(trainer)
        for transaction_index in range(10):
            train_info, priorities, idxes = (
                runner._train_adj_on_batch_with_exact_failure_logging(
                    sample=batch,
                    use_adj_init=False,
                    adjacency_update_round=1,
                    consume_factor_credit_observations=True,
                    policy_id="policy_0",
                    transaction_sequence_index=transaction_index,
                    ppo_epoch_index=0,
                    partition_index=0,
                )
            )
            assert priorities is None and idxes is None
            assert float(train_info[
                "_pair_optimizer_recoverable_noop"
            ]) == 1.0
            assert float(train_info[
                "pair_optimizer_recoverable_noop_reason_code"
            ]) == 1.0
            assert float(train_info[
                "pair_optimizer_recoverable_noop_pair_update_norm_sq"
            ]) == 0.0
            _assert_recoverable_noop_state_equal(
                stable_origin,
                _recoverable_noop_state_snapshot(trainer),
            )

        # A recoverable partition must not poison the next optimizer update.
        trainer.adj_optimizer.step = original_step
        committed_info, _priorities, _idxes = trainer.train_adj_on_batch(
            batch,
            use_adj_init=False,
            adj_update_round=2,
            consume_factor_credit_observations=True,
            enable_optimizer_transaction_diagnostics=True,
            diagnostic_policy_id="policy_0",
            diagnostic_transaction_sequence_index=10,
        )
        assert float(committed_info[
            "pair_optimizer_transaction_final_pair_descent_dot"
        ]) > 0.0
        assert any(
            not torch.equal(before, parameter.detach())
            for before, parameter in zip(
                stable_origin["parameters"], trainer.adj_parameters
            )
        )
    finally:
        trainer.adj_optimizer.step = original_step
        trainer._pair_selection_boundary_policy_response_diagnostics = (
            original_policy_response
        )


def validate_pair_adam_nonfinite_remains_fatal(device):
    trainer, batch, _rnn_obs, _dones, _adj = (
        _make_exact_context_retention_case(
            device,
            current_pair=True,
            selection_state_sensitive=False,
        )
    )
    original_policy_response = (
        trainer._pair_selection_boundary_policy_response_diagnostics
    )
    trainer._pair_selection_boundary_policy_response_diagnostics = (
        types.MethodType(lambda _self, **_kwargs: [], trainer)
    )
    original_step = trainer.adj_optimizer.step

    def nonfinite_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            for parameter in trainer.adj_parameters:
                parameter.reshape(-1)[0] = float("nan")
        return result

    trainer.adj_optimizer.step = nonfinite_step
    origin = _recoverable_noop_state_snapshot(trainer)
    try:
        try:
            trainer.train_adj_on_batch(
                batch,
                use_adj_init=False,
                adj_update_round=1,
                consume_factor_credit_observations=True,
                enable_optimizer_transaction_diagnostics=True,
                diagnostic_policy_id="policy_0",
                diagnostic_transaction_sequence_index=0,
            )
        except PairOptimizerRecoverableNoOpError:
            raise AssertionError("non-finite update was treated as a no-op")
        except FloatingPointError:
            pass
        else:
            raise AssertionError("non-finite pair update did not fail loud")
        _assert_recoverable_noop_state_equal(
            origin,
            _recoverable_noop_state_snapshot(trainer),
        )
    finally:
        trainer.adj_optimizer.step = original_step
        trainer._pair_selection_boundary_policy_response_diagnostics = (
            original_policy_response
        )


def validate_run134_factor_credit_batch_semantics(device):
    """Run134 regression: one candidate gets one EMA write per fresh batch."""
    args = _full_transaction_args()
    args.use_adj_advantage_triplet_scorer = True
    args.adj_triplet_credit_ema_alpha = 0.05

    def _make_graph():
        graph = Adj_Generator(
            args=args,
            obs_dim=1,
            state_dim=1,
            act_dim=1,
            device=device,
        )
        graph.to(device)
        return graph

    def _apply(graph, credits, order):
        batch_size = len(credits)
        adj = torch.zeros(
            batch_size, graph.num_variable, 1, device=device
        )
        adj[:, :order, 0] = 1.0
        local = torch.as_tensor(
            credits, dtype=torch.float32, device=device
        ).reshape(batch_size, 1)
        graph_advantage = torch.ones(
            batch_size, dtype=torch.float32, device=device
        )
        mask = torch.ones(batch_size, 1, device=device)
        return graph.update_factor_credit_memory(
            adj, local, graph_advantage, mask
        )

    # Run134 had about 156 pair observations per canonical pair in each
    # constrained transaction (2344 / 15).  Opposite batch orders used to
    # leave almost opposite EMA endpoints because alpha was applied 156 times.
    credits = [1.0] * 78 + [-1.0] * 78
    forward = _make_graph()
    reverse = _make_graph()
    pair_id = forward._pair_index_map[(0, 1)]
    with torch.no_grad():
        forward.pair_credit_ema[pair_id] = 0.2
        reverse.pair_credit_ema[pair_id] = 0.2
        forward.pair_credit_seen[pair_id] = 100.0
        reverse.pair_credit_seen[pair_id] = 100.0
    forward_info = _apply(forward, credits, order=2)
    reverse_info = _apply(reverse, list(reversed(credits)), order=2)
    expected = 0.95 * 0.2 + 0.05 * 0.0
    _assert_close(forward.pair_credit_ema[pair_id], expected, atol=1e-6)
    _assert_close(reverse.pair_credit_ema[pair_id], expected, atol=1e-6)
    assert torch.equal(forward.pair_credit_ema, reverse.pair_credit_ema)
    assert float(forward.pair_credit_seen[pair_id].item()) == 256.0
    assert float(forward_info["adv_triplet_credit_pair_updates"]) == 156.0
    assert float(forward_info[
        "adv_triplet_credit_pair_state_updates"
    ]) == 1.0

    legacy_forward = 0.2
    legacy_reverse = 0.2
    for credit in credits:
        legacy_forward = 0.95 * legacy_forward + 0.05 * credit
    for credit in reversed(credits):
        legacy_reverse = 0.95 * legacy_reverse + 0.05 * credit
    assert abs(legacy_forward - legacy_reverse) > 1.9
    assert abs(float(forward.pair_credit_ema[pair_id]) - 0.2) < (
        abs(legacy_forward - 0.2) / 100.0
    )

    # Use the real production resolver (not a linear score approximation) to
    # show why this update granularity matters.  The legacy tail-dominated pair
    # endpoint switches the competitor to an order-3 factor and reverses the
    # active-side margin; the batch-mean endpoint is safe at full scale.
    scorer_trainer, _batch, rnn_obs, dones, adj = (
        _make_exact_context_retention_case(
            device,
            selection_state_sensitive=True,
        )
    )
    scorer = scorer_trainer.adj_network
    scorer_pair_id = scorer._pair_index_map[(0, 1)]

    def _production_margin_at(endpoint_scale):
        with torch.no_grad():
            scorer.pair_credit_ema[scorer_pair_id] = (
                float(endpoint_scale) * float(legacy_forward)
            )
            resolved = scorer.evaluate_selected_factor_replay_boundaries(
                rnn_obs=rnn_obs,
                dones=dones,
                adj=adj,
            )
        return (
            float(resolved["selected_margin"][0, 0].item()),
            int(resolved["selected_rank"][0, 0].item()),
            int(resolved[
                "competitor_candidate_index"
            ][0, 0].item()),
        )

    baseline_margin, baseline_rank, baseline_competitor = (
        _production_margin_at(0.0)
    )
    legacy_margin, legacy_rank, legacy_competitor = (
        _production_margin_at(1.0)
    )
    assert baseline_margin > 0.0 and baseline_rank == 1
    assert legacy_margin < 0.0 and legacy_rank > 1
    assert legacy_competitor != baseline_competitor
    lower = 0.0
    upper = 1.0
    for _ in range(24):
        midpoint = 0.5 * (lower + upper)
        margin, rank, _competitor = _production_margin_at(midpoint)
        if margin >= 0.0 and rank == 1:
            lower = midpoint
        else:
            upper = midpoint
    assert 0.10 < lower < 0.12
    # The batch mean is zero for this balanced observation population, so the
    # corrected full update is the baseline endpoint and remains exact-safe.
    corrected_margin, corrected_rank, corrected_competitor = (
        _production_margin_at(0.0)
    )
    assert corrected_margin == baseline_margin
    assert corrected_rank == baseline_rank
    assert corrected_competitor == baseline_competitor

    fixture_path = (
        PROJECT_ROOT / "scripts" / "fixtures"
        / "run134_credit_component_conflict.json"
    )
    with fixture_path.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    assert fixture["run_id"] == "run134"
    assert fixture["protected_identity"] not in fixture[
        "candidate_evidence_identities"
    ]
    assert fixture["pair_observations_per_transaction"] == 2344
    for row in fixture["transactions"]:
        pair_damage = abs(float(row["pair_margin_delta"]))
        all_damage = abs(float(row["all_continuous_margin_delta"]))
        assert pair_damage / all_damage > 0.99
        assert row["pair_component_competitor"] == "0-1-5"
        assert abs(float(row["gate_margin_delta"])) == 0.0


def _execute_full_train_adj_transaction(
        enable_diagnostics,
        device,
        candidate_target=False,
        candidate_target_sign=1.0,
        successful_candidate_context=False,
        successful_episode_context=False,
        consume_factor_credit_observations=True,
        use_credit_scorer=None,
        behavior_mixture=False):
    torch.manual_seed(211)
    np.random.seed(211)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(211)
    args = _full_transaction_args()
    if bool(behavior_mixture):
        args.use_adj_topology_persistence = True
        args.adj_exploration_mix = 0.10
        args.adj_sampling_temperature_start = 0.65
        args.adj_sampling_temperature_final = 0.65
        args.adj_greedy_sample_prob_start = 0.35
        args.adj_greedy_sample_prob_final = 0.35
    if use_credit_scorer is not None:
        args.use_adj_advantage_triplet_scorer = bool(use_credit_scorer)
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )
    policy = _MinimalPolicy(device)
    trainer = R_SDDFG(
        args=args,
        num_agents=4,
        policies={"policy_0": policy},
        adj_network=graph,
        policy_mapping_fn=lambda _agent_id: "policy_0",
        device=device,
        episode_length=2,
    )
    trainer.adj_network.train()
    batch = _make_full_transaction_batch(
        graph,
        device,
        candidate_target=candidate_target,
        candidate_target_sign=candidate_target_sign,
        successful_candidate_context=successful_candidate_context,
        successful_episode_context=successful_episode_context,
    )
    train_info, _priorities, _idxes = trainer.train_adj_on_batch(
        batch,
        use_adj_init=False,
        adj_update_round=1,
        consume_factor_credit_observations=bool(
            consume_factor_credit_observations
        ),
        enable_optimizer_transaction_diagnostics=enable_diagnostics,
        diagnostic_policy_id="policy_0",
        diagnostic_transaction_sequence_index=0,
    )
    transaction_train_info = {
        key: value
        for key, value in train_info.items()
        if (
            key.startswith("pair_optimizer_transaction_")
            or key in set(_ADJ_TRANSACTION_TRAIN_INFO_FIELDS.values())
        )
    }
    parameters = list(trainer.adj_parameters)
    optimizer_states = []
    for parameter in parameters:
        state = trainer.adj_optimizer.state.get(parameter, {})
        optimizer_states.append({
            key: (
                value.detach().cpu().clone()
                if torch.is_tensor(value) else copy.deepcopy(value)
            )
            for key, value in state.items()
        })
    return {
        "raw_train_info": train_info,
        "batch": batch,
        "train_info": {
            key: value
            for key, value in train_info.items()
            if (
                not key.startswith("pair_optimizer_transaction_")
                and not key.startswith("pair_selection_boundary_")
                and key not in (
                    "_candidate_identity_transaction_rows",
                    "_candidate_evidence_consumption_rows",
                    "_pair_selection_boundary_retention_rows",
                )
            )
        },
        "candidate_identity_transaction_rows": list(train_info.get(
            "_candidate_identity_transaction_rows",
            (),
        )),
        "transaction_train_info": transaction_train_info,
        "parameters": [
            parameter.detach().cpu().clone()
            for parameter in parameters
        ],
        "gradients": [
            (
                parameter.grad.detach().cpu().clone()
                if parameter.grad is not None else None
            )
            for parameter in parameters
        ],
        "optimizer_states": optimizer_states,
        "candidate_lifecycle": copy.deepcopy(
            trainer._candidate_identity_lifecycle
        ),
        "candidate_policy_version": int(
            graph.candidate_policy_version
        ),
        "candidate_lifecycle_clock": int(
            graph.candidate_lifecycle_clock
        ),
        "cpu_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device).cpu().clone()
            if device.type == "cuda" else None
        ),
        "numpy_rng_state": np.random.get_state(),
        "graph_rng_state": graph.rng.get_state(),
        "graph_eval_rng_state": graph.eval_rng.get_state(),
    }


def _assert_nested_optimizer_state_equal(left, right):
    assert len(left) == len(right)
    for left_state, right_state in zip(left, right):
        assert set(left_state.keys()) == set(right_state.keys())
        for key in left_state:
            if torch.is_tensor(left_state[key]):
                assert torch.equal(left_state[key], right_state[key]), key
            else:
                assert left_state[key] == right_state[key], key


def validate_full_train_adj_on_batch_trajectory_neutrality(
        device,
        candidate_target=False,
        candidate_target_sign=1.0,
        behavior_mixture=False):
    without = _execute_full_train_adj_transaction(
        False,
        device,
        candidate_target=candidate_target,
        candidate_target_sign=candidate_target_sign,
        behavior_mixture=behavior_mixture,
    )
    with_diagnostics = _execute_full_train_adj_transaction(
        True,
        device,
        candidate_target=candidate_target,
        candidate_target_sign=candidate_target_sign,
        behavior_mixture=behavior_mixture,
    )
    assert without["train_info"] == with_diagnostics["train_info"]
    assert not without["candidate_identity_transaction_rows"]
    if bool(candidate_target):
        assert with_diagnostics["candidate_identity_transaction_rows"]
    else:
        assert not with_diagnostics["candidate_identity_transaction_rows"]
    transaction_info = with_diagnostics["transaction_train_info"]
    missing_transaction_fields = (
        set(_ADJ_TRANSACTION_TRAIN_INFO_FIELDS.values())
        - set(transaction_info.keys())
    )
    assert not missing_transaction_fields, sorted(
        missing_transaction_fields
    )
    if bool(candidate_target):
        assert float(transaction_info[
            "pair_optimizer_transaction_objective_candidate_active"
        ]) == 1.0
        assert float(transaction_info[
            "pair_optimizer_transaction_objective_candidate_grad_norm"
        ]) > 0.0
        if float(candidate_target_sign) < 0.0:
            assert float(transaction_info[
                "pair_optimizer_transaction_gradient_projection_intervened"
            ]) == 1.0
            assert float(transaction_info[
                "pair_optimizer_transaction_projection_delta_norm"
            ]) > 0.0
    assert float(transaction_info[
        "pair_optimizer_transaction_objective_scalar_reconstruction_valid"
    ]) == 1.0
    assert float(transaction_info[
        "pair_optimizer_transaction_independent_sum_reconstruction_valid"
    ]) == 1.0
    objective_scalar_sum = sum(
        float(transaction_info[
            "pair_optimizer_transaction_objective_{}_scalar_loss".format(
                objective_name
            )
        ])
        for objective_name in (
            "graph",
            "base_factor",
            "capture_outcome",
            "pair",
            "candidate",
            "entropy",
        )
    )
    _assert_close(
        objective_scalar_sum,
        transaction_info[
            "pair_optimizer_transaction_total_adjacency_loss"
        ],
        atol=1e-6,
    )
    _assert_close(
        transaction_info[
            "pair_optimizer_transaction_raw_combined_grad_norm_from_backward"
        ],
        transaction_info[
            "pair_optimizer_transaction_pre_projection_combined_grad_norm"
        ],
        atol=1e-7,
    )
    _assert_close(
        transaction_info[
            "pair_optimizer_transaction_post_projection_combined_grad_norm"
        ],
        transaction_info[
            "pair_optimizer_transaction_combined_grad_norm_preclip"
        ],
        atol=1e-7,
    )
    for left, right in zip(
            without["parameters"],
            with_diagnostics["parameters"]):
        assert torch.equal(left, right)
    for left, right in zip(
            without["gradients"],
            with_diagnostics["gradients"]):
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert torch.equal(left, right)
    _assert_nested_optimizer_state_equal(
        without["optimizer_states"],
        with_diagnostics["optimizer_states"],
    )
    assert (
        without["candidate_lifecycle"]
        == with_diagnostics["candidate_lifecycle"]
    )
    assert (
        without["candidate_policy_version"]
        == with_diagnostics["candidate_policy_version"]
    )
    assert (
        without["candidate_lifecycle_clock"]
        == with_diagnostics["candidate_lifecycle_clock"]
    )
    assert torch.equal(
        without["cpu_rng_state"],
        with_diagnostics["cpu_rng_state"],
    )
    assert without["numpy_rng_state"][0] == with_diagnostics[
        "numpy_rng_state"
    ][0]
    assert np.array_equal(
        without["numpy_rng_state"][1],
        with_diagnostics["numpy_rng_state"][1],
    )
    assert without["numpy_rng_state"][2:] == with_diagnostics[
        "numpy_rng_state"
    ][2:]
    for rng_field in ("graph_rng_state", "graph_eval_rng_state"):
        assert without[rng_field][0] == with_diagnostics[rng_field][0]
        assert np.array_equal(
            without[rng_field][1],
            with_diagnostics[rng_field][1],
        )
        assert without[rng_field][2:] == with_diagnostics[rng_field][2:]
    if device.type == "cuda":
        assert torch.equal(
            without["cuda_rng_state"],
            with_diagnostics["cuda_rng_state"],
        )


def validate_successful_candidate_context_trajectory_neutrality(device):
    baseline = _execute_full_train_adj_transaction(
        False,
        device,
        successful_candidate_context=False,
        successful_episode_context=True,
    )
    diagnosed = _execute_full_train_adj_transaction(
        False,
        device,
        successful_candidate_context=True,
    )
    diagnostic_prefix = "successful_candidate_capture_boundary_"
    baseline_train = {
        key: value for key, value in baseline["train_info"].items()
        if not key.startswith(diagnostic_prefix)
    }
    diagnosed_train = {
        key: value for key, value in diagnosed["train_info"].items()
        if not key.startswith(diagnostic_prefix)
    }
    assert baseline_train == diagnosed_train
    assert diagnosed["train_info"][
        diagnostic_prefix + "identity_count"
    ] == 1.0
    assert diagnosed["train_info"][
        diagnostic_prefix + "identity_join_valid"
    ] == 1.0
    for left, right in zip(
            baseline["parameters"], diagnosed["parameters"]):
        assert torch.equal(left, right)
    for left, right in zip(
            baseline["gradients"], diagnosed["gradients"]):
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert torch.equal(left, right)
    _assert_nested_optimizer_state_equal(
        baseline["optimizer_states"],
        diagnosed["optimizer_states"],
    )
    assert baseline["candidate_lifecycle"] == diagnosed[
        "candidate_lifecycle"
    ]
    assert baseline["candidate_policy_version"] == diagnosed[
        "candidate_policy_version"
    ]
    assert baseline["candidate_lifecycle_clock"] == diagnosed[
        "candidate_lifecycle_clock"
    ]
    assert torch.equal(
        baseline["cpu_rng_state"], diagnosed["cpu_rng_state"]
    )
    assert np.array_equal(
        baseline["numpy_rng_state"][1],
        diagnosed["numpy_rng_state"][1],
    )
    for rng_field in ("graph_rng_state", "graph_eval_rng_state"):
        assert np.array_equal(
            baseline[rng_field][1], diagnosed[rng_field][1]
        )
    if device.type == "cuda":
        assert torch.equal(
            baseline["cuda_rng_state"], diagnosed["cuda_rng_state"]
        )


def _adam_direction_case(historical_gradient, current_gradient):
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if historical_gradient is not None:
        parameter.grad = torch.tensor([float(historical_gradient)])
        optimizer.step()
        optimizer.zero_grad()
    pair_grads = (torch.tensor([1.0]),)
    parameter.grad = torch.tensor([float(current_gradient)])
    reference = parameter.sum() * 0.0
    combined = _clone_parameter_gradients([parameter])
    combined_info = _gradient_tuple_direction_diagnostics(
        pair_grads,
        combined,
        reference,
    )
    before = [parameter.detach().clone()]
    adam_before = _adam_state_before_step_diagnostics(
        optimizer,
        [parameter],
        pair_grads,
        reference,
    )
    optimizer.step()
    raw = _parameter_displacement_direction_diagnostics(
        [parameter],
        before,
        pair_grads,
        reference,
    )
    _validate_adam_step_increment(
        optimizer,
        [parameter],
        adam_before["parameter_steps_before"],
    )
    return combined_info, adam_before, raw


def validate_adam_moment_direction_reversal():
    combined, adam_before, raw = _adam_direction_case(-10.0, 1.0)
    assert float(combined["descent_component"]) > 0.0
    assert float(adam_before["exp_avg_pair_dot"]) < 0.0
    assert float(raw["pair_descent_dot"]) < 0.0

    combined, _adam_before, raw = _adam_direction_case(None, 1.0)
    assert float(combined["descent_component"]) > 0.0
    assert float(raw["pair_descent_dot"]) > 0.0


def validate_raw_and_final_displacement():
    reference = torch.tensor(0.0)
    pair_grads = (torch.tensor([1.0, -1.0]),)
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    before = [parameter.detach().clone()]
    with torch.no_grad():
        parameter.add_(torch.tensor([-0.2, 0.1]))
    raw = _parameter_displacement_direction_diagnostics(
        [parameter],
        before,
        pair_grads,
        reference,
    )
    unchanged = _displacement_delta_diagnostics(
        [parameter],
        before,
        raw["displacements"],
        reference,
    )
    assert float(unchanged["delta_norm"]) == 0.0
    assert float(unchanged["exact_equal"]) == 1.0

    with torch.no_grad():
        parameter.add_(torch.tensor([0.05, 0.0]))
    candidate_corrected = _displacement_delta_diagnostics(
        [parameter],
        before,
        raw["displacements"],
        reference,
    )
    assert float(candidate_corrected["delta_norm"]) > 0.0
    assert float(candidate_corrected["exact_equal"]) == 0.0

    with torch.no_grad():
        parameter.copy_(before[0] + 0.5 * raw["displacements"][0])
    lifecycle_backtracked = _displacement_delta_diagnostics(
        [parameter],
        before,
        raw["displacements"],
        reference,
    )
    assert float(lifecycle_backtracked["delta_norm"]) > 0.0

    with torch.no_grad():
        parameter.copy_(before[0])
    rolled_back = _parameter_displacement_direction_diagnostics(
        [parameter],
        before,
        pair_grads,
        reference,
    )
    assert float(rolled_back["displacement_norm"]) == 0.0


def validate_exact_score_directions_and_join():
    pre = torch.zeros(4)
    post = torch.tensor([0.2, -0.2, -0.1, 0.0])
    delta = torch.tensor([1.0, -1.0, 1.0, -1.0])
    info = _pair_target_score_change_diagnostics(
        pre,
        post,
        delta,
    )
    assert float(info["correct_direction_count"]) == 2.0
    assert float(info["reverse_direction_count"]) == 1.0
    assert float(info["approximately_zero_count"]) == 1.0
    assert float(info["positive_target_count"]) == 2.0
    assert float(info["negative_target_count"]) == 2.0

    pre_join = {
        "transition": torch.tensor([4, 5]),
        "identity_order_slot": torch.tensor([[2, 1], [3, 0]]),
        "valid_mask": torch.tensor([1.0, 1.0]),
    }
    post_join = {
        key: value.clone()
        for key, value in pre_join.items()
    }
    assert _validate_exact_score_join_fields(pre_join, post_join)
    post_join["identity_order_slot"][1, 1] = 1
    try:
        _validate_exact_score_join_fields(pre_join, post_join)
    except RuntimeError:
        pass
    else:
        raise AssertionError("identity/order/slot mismatch was accepted")


def validate_row_fail_loud():
    assert PAIR_EXACT_SCORE_RECORDING_CONTRACT_VERSION == 2
    info = _fake_train_info(3)
    row = _build_adj_transaction_row(
        run_id="run101",
        env_step=108000,
        adjacency_update_round=11,
        ppo_epoch_index=0,
        policy_id="policy_0",
        partition_index=0,
        transaction_sequence_index=0,
        selected_episode_count=10,
        selected_chunk_count=100,
        transaction_chunk_count=100,
        train_adj_info=info,
    )
    assert row["optimizer_step_after"] == 4.0
    assert row["diagnostic_version"] == float(
        PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION
    )

    preserved = copy.deepcopy(info)
    preserved[
        "pair_optimizer_transaction_score_correct_direction_target_count"
    ] = 1.0
    preserved[
        "pair_optimizer_transaction_score_reverse_direction_target_count"
    ] = 0.0
    preserved[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] = 1.0
    preserved[
        "pair_optimizer_transaction_score_zero_tolerance"
    ] = float(128.0 * torch.finfo(torch.float32).eps)
    preserved_row = _build_adj_transaction_row(
        run_id="run154-preservation-fixture",
        env_step=176800,
        adjacency_update_round=213,
        ppo_epoch_index=0,
        policy_id="policy_0",
        partition_index=0,
        transaction_sequence_index=840,
        selected_episode_count=5,
        selected_chunk_count=100,
        transaction_chunk_count=100,
        train_adj_info=preserved,
    )
    assert preserved_row[
        "score_correct_direction_target_count"
    ] == 1.0
    assert preserved_row[
        "score_approximately_zero_target_count"
    ] == 1.0
    assert preserved_row["score_reverse_direction_target_count"] == 0.0

    reversed_target = copy.deepcopy(preserved)
    reversed_target[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] = 0.0
    reversed_target[
        "pair_optimizer_transaction_score_reverse_direction_target_count"
    ] = 1.0
    try:
        _build_adj_transaction_row(
            run_id="run154-reverse-fixture",
            env_step=176800,
            adjacency_update_round=213,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
            transaction_sequence_index=840,
            selected_episode_count=5,
            selected_chunk_count=100,
            transaction_chunk_count=100,
            train_adj_info=reversed_target,
        )
    except RuntimeError as exc:
        assert "reversed an exact target beyond" in str(exc)
    else:
        raise AssertionError(
            "tolerance-exceeding exact-target reversal was accepted"
        )

    incomplete_partition = copy.deepcopy(preserved)
    incomplete_partition[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] = 0.0
    try:
        _build_adj_transaction_row(
            run_id="run154-partition-fixture",
            env_step=176800,
            adjacency_update_round=213,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
            transaction_sequence_index=840,
            selected_episode_count=5,
            selected_chunk_count=100,
            transaction_chunk_count=100,
            train_adj_info=incomplete_partition,
        )
    except RuntimeError as exc:
        assert "do not partition the target population" in str(exc)
    else:
        raise AssertionError(
            "incomplete exact-target score classification was accepted"
        )

    invalid = copy.deepcopy(info)
    invalid["pair_optimizer_transaction_optimizer_step_after"] = 5.0
    invalid["pair_optimizer_transaction_optimizer_step_after_min"] = 5.0
    invalid["pair_optimizer_transaction_optimizer_step_after_max"] = 5.0
    try:
        _build_adj_transaction_row(
            run_id="run101",
            env_step=108000,
            adjacency_update_round=11,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
            transaction_sequence_index=0,
            selected_episode_count=10,
            selected_chunk_count=100,
            transaction_chunk_count=100,
            train_adj_info=invalid,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-unit Adam transaction was accepted")

    unknown = copy.deepcopy(info)
    unknown["pair_optimizer_transaction_diagnostic_version"] = float(
        PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION + 1
    )
    try:
        _build_adj_transaction_row(
            run_id="run135",
            env_step=7200,
            adjacency_update_round=1,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
            transaction_sequence_index=0,
            selected_episode_count=1,
            selected_chunk_count=1,
            transaction_chunk_count=2,
            train_adj_info=unknown,
        )
    except RuntimeError as exc:
        assert str(exc) == "unexpected pair optimizer diagnostic version"
    else:
        raise AssertionError("unknown transaction schema was accepted")

    missing = copy.deepcopy(info)
    del missing["adv_triplet_credit_pair_state_updates"]
    try:
        _build_adj_transaction_row(
            run_id="run135",
            env_step=7200,
            adjacency_update_round=1,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
            transaction_sequence_index=0,
            selected_episode_count=1,
            selected_chunk_count=1,
            transaction_chunk_count=2,
            train_adj_info=missing,
        )
    except RuntimeError as exc:
        assert "adv_triplet_credit_pair_state_updates" in str(exc)
    else:
        raise AssertionError(
            "malformed current transaction schema was accepted"
        )


def validate_current_producer_runner_csv_contract(device):
    result = _execute_full_train_adj_transaction(
        True,
        device,
        consume_factor_credit_observations=True,
        use_credit_scorer=True,
    )
    train_info = result["raw_train_info"]
    expected = {
        "credit_observation_consumed":
            "adv_factor_credit_memory_observation_consumed",
        "pair_credit_raw_observation_count":
            "adv_triplet_credit_pair_updates",
        "triplet_credit_raw_observation_count":
            "adv_triplet_credit_triplet_updates",
        "pair_credit_state_update_count":
            "adv_triplet_credit_pair_state_updates",
        "triplet_credit_state_update_count":
            "adv_triplet_credit_triplet_state_updates",
    }
    assert float(train_info[
        "pair_optimizer_transaction_diagnostic_version"
    ]) == float(PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION)
    assert float(train_info[
        "adv_factor_credit_memory_observation_consumed"
    ]) == 1.0
    assert float(train_info["adv_triplet_credit_pair_updates"]) > 0.0
    assert float(train_info[
        "adv_triplet_credit_pair_state_updates"
    ]) > 0.0

    temp_root = PROJECT_ROOT / ".codex_tmp"
    temp_root.mkdir(exist_ok=True)
    run_dir = temp_root / (
        "current_schema_production_{}".format(device.type)
    ) / "run135"
    if run_dir.exists():
        resolved = run_dir.resolve()
        assert str(resolved).startswith(str(temp_root.resolve()))
        shutil.rmtree(str(resolved))
    run_dir.mkdir(parents=True)
    try:
        runner = RecRunner.__new__(RecRunner)
        runner.run_dir = run_dir
        runner.total_env_steps = 7200
        runner._adj_transaction_log_initialized = False
        policy_buffer = SimpleNamespace(
            last_sample_episode_count=1,
            last_sample_selected_chunk_count=int(
                result["batch"][0].shape[0]
            ),
            last_sample_pair_optimizer_atomic_partition=float(
                train_info["pair_optimizer_class_complete"]
            ),
            last_sample_candidate_evidence_provenance_rows=(),
            last_sample_pair_evidence_episode_rows=(),
        )
        row = runner._record_adj_transaction(
            train_adj_info=copy.deepcopy(train_info),
            policy_buffer=policy_buffer,
            sample=result["batch"],
            adjacency_update_round=1,
            ppo_epoch_index=0,
            policy_id="policy_0",
            partition_index=0,
        )
        for csv_key, train_key in expected.items():
            assert float(row[csv_key]) == float(train_info[train_key])
        filename = _get_run_csv_name(
            run_dir,
            _ADJ_TRANSACTION_CSV_BASENAME,
        )
        with open(str(run_dir / filename), "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            persisted = list(reader)
            assert tuple(reader.fieldnames or ()) == _ADJ_TRANSACTION_CSV_FIELDS
        assert len(persisted) == 1
        assert len(persisted[0]) == len(_ADJ_TRANSACTION_CSV_FIELDS)
        for csv_key, train_key in expected.items():
            assert float(persisted[0][csv_key]) == float(
                train_info[train_key]
            )
    finally:
        resolved = run_dir.parent.resolve()
        assert str(resolved).startswith(str(temp_root.resolve()))
        shutil.rmtree(str(resolved))

    unconsumed = _execute_full_train_adj_transaction(
        True,
        device,
        consume_factor_credit_observations=False,
        use_credit_scorer=True,
    )["raw_train_info"]
    assert float(unconsumed[
        "adv_factor_credit_memory_observation_consumed"
    ]) == 0.0
    assert all(
        float(unconsumed[key]) == 0.0
        for key in (
            "adv_triplet_credit_pair_updates",
            "adv_triplet_credit_triplet_updates",
            "adv_triplet_credit_pair_state_updates",
            "adv_triplet_credit_triplet_state_updates",
        )
    )
    unconsumed_row = _build_adj_transaction_row(
        run_id="run135",
        env_step=7200,
        adjacency_update_round=1,
        ppo_epoch_index=1,
        policy_id="policy_0",
        partition_index=0,
        transaction_sequence_index=1,
        selected_episode_count=1,
        selected_chunk_count=int(result["batch"][0].shape[0]),
        transaction_chunk_count=int(result["batch"][0].shape[0]),
        train_adj_info=unconsumed,
    )
    assert unconsumed_row["credit_observation_consumed"] == 0.0

    default_off = _execute_full_train_adj_transaction(
        True,
        device,
        consume_factor_credit_observations=True,
        use_credit_scorer=False,
    )["raw_train_info"]
    assert float(default_off[
        "adv_factor_credit_memory_observation_consumed"
    ]) == 0.0
    assert all(
        float(default_off[key]) == 0.0
        for key in (
            "adv_triplet_credit_pair_updates",
            "adv_triplet_credit_triplet_updates",
            "adv_triplet_credit_pair_state_updates",
            "adv_triplet_credit_triplet_state_updates",
        )
    )
    default_off_row = _build_adj_transaction_row(
        run_id="run135-default-off",
        env_step=7200,
        adjacency_update_round=1,
        ppo_epoch_index=0,
        policy_id="policy_0",
        partition_index=0,
        transaction_sequence_index=0,
        selected_episode_count=1,
        selected_chunk_count=int(result["batch"][0].shape[0]),
        transaction_chunk_count=int(result["batch"][0].shape[0]),
        train_adj_info=default_off,
    )
    assert default_off_row["credit_observation_consumed"] == 0.0


def main():
    test_pair_selection_boundary_crossings()
    test_selection_boundary_replay_tolerance_uses_operand_resolution()
    validate_pair_selection_boundary_retention_observer(
        torch.device("cpu")
    )
    test_run118_joint_exact_boundary_lifecycle_acceptance()
    test_run123_maximum_joint_exact_safe_scale_refinement()
    test_nonmonotone_midpoint_safe_island_is_committable()
    test_resolution_only_zero_exact_scores_use_nonregression_contract()
    test_failed_boundary_limiter_tangent_recovers_distinct_ray()
    test_boundary_write_resolution_expands_after_invalid_unit_scale()
    test_run118_boundary_deficit_budget_prioritizes_reachable_crossing()
    test_run120_unaffordable_boundary_budget_waterfills_nearest_target()
    test_run122_multi_exposure_identity_targets_nearest_member()
    validate_two_epoch_unaggregated_csv()
    validate_pair_direction_candidate_schema()
    validate_exact_behavior_score_replay_join(torch.device("cpu"))
    validate_inactive_selected_factor_is_an_exact_search_boundary(
        torch.device("cpu")
    )
    validate_episode_reject_reason_fixed_csv()
    validate_combined_gradient_directions()
    validate_per_objective_gradient_hand_calculation()
    validate_run100_equivalent_objective_flip_cases()
    validate_graph_return_advantage_isolation()
    validate_projection_stage_direction_changes()
    validate_standard_pair_gradient_projection()
    validate_standard_zero_aggregate_pair_gradient_recovery()
    validate_target_local_pair_score_jacobian_constraints()
    validate_pair_priority_supersedes_incompatible_lifecycle()
    validate_reversed_adam_priority_is_repaired_before_lifecycle()
    validate_lifecycle_final_dot_uses_float32_resolution()
    validate_objective_none_inactive_zero_and_fail_loud()
    validate_trajectory_neutrality(torch.device("cpu"))
    validate_full_train_adj_on_batch_trajectory_neutrality(
        torch.device("cpu")
    )
    validate_full_train_adj_on_batch_trajectory_neutrality(
        torch.device("cpu"),
        behavior_mixture=True,
    )
    validate_full_train_adj_on_batch_trajectory_neutrality(
        torch.device("cpu"),
        candidate_target=True,
        candidate_target_sign=-1.0,
    )
    validate_exact_context_retention_floor(torch.device("cpu"))
    validate_selection_boundary_policy_response_diagnostic(
        torch.device("cpu")
    )
    validate_run131_late_selection_state_postcondition(
        torch.device("cpu")
    )
    validate_current_transaction_final_crossing_floor(
        torch.device("cpu")
    )
    validate_pair_adam_zero_displacement_is_recoverable(
        torch.device("cpu")
    )
    validate_pair_adam_nonfinite_remains_fatal(torch.device("cpu"))
    validate_run134_factor_credit_batch_semantics(torch.device("cpu"))
    validate_successful_candidate_context_trajectory_neutrality(
        torch.device("cpu")
    )
    validate_adam_moment_direction_reversal()
    validate_raw_and_final_displacement()
    validate_exact_score_directions_and_join()
    validate_row_fail_loud()
    validate_current_producer_runner_csv_contract(torch.device("cpu"))
    cuda_status = "skipped (CUDA unavailable)"
    if torch.cuda.is_available():
        validate_exact_behavior_score_replay_join(torch.device("cuda"))
        validate_inactive_selected_factor_is_an_exact_search_boundary(
            torch.device("cuda")
        )
        validate_pair_selection_boundary_retention_observer(
            torch.device("cuda")
        )
        validate_trajectory_neutrality(torch.device("cuda"))
        validate_full_train_adj_on_batch_trajectory_neutrality(
            torch.device("cuda")
        )
        validate_full_train_adj_on_batch_trajectory_neutrality(
            torch.device("cuda"),
            behavior_mixture=True,
        )
        validate_full_train_adj_on_batch_trajectory_neutrality(
            torch.device("cuda"),
            candidate_target=True,
            candidate_target_sign=-1.0,
        )
        validate_exact_context_retention_floor(torch.device("cuda"))
        validate_selection_boundary_policy_response_diagnostic(
            torch.device("cuda")
        )
        validate_run131_late_selection_state_postcondition(
            torch.device("cuda")
        )
        validate_current_transaction_final_crossing_floor(
            torch.device("cuda")
        )
        validate_pair_adam_zero_displacement_is_recoverable(
            torch.device("cuda")
        )
        validate_pair_adam_nonfinite_remains_fatal(
            torch.device("cuda")
        )
        validate_run134_factor_credit_batch_semantics(
            torch.device("cuda")
        )
        validate_successful_candidate_context_trajectory_neutrality(
            torch.device("cuda")
        )
        validate_current_producer_runner_csv_contract(torch.device("cuda"))
        cuda_status = "passed"
    print("pair optimizer transaction diagnostics: passed")
    print("per-objective gradient decomposition and reconstruction: passed")
    print("run100-equivalent objective/projection counterexamples: passed")
    print("run102 graph-return/factor-local advantage isolation: passed")
    print("CPU trajectory-neutral transaction: passed")
    print("CPU selection-boundary retention observer: passed")
    print("CPU full train_adj_on_batch trajectory neutrality: passed")
    print("CPU candidate-active projection trajectory neutrality: passed")
    print("CPU exact-context crossing retention floor: passed")
    print("CPU inactive selected-factor exact-search boundary: passed")
    print("CPU selection-boundary policy counterfactual: passed")
    print("CPU run131 late selection-state postcondition: passed")
    print("CPU complete-transaction crossing floor: passed")
    print("CPU pair Adam recoverable no-op and continuation: passed")
    print("CPU pair Adam non-finite fail-loud: passed")
    print("CPU run134 factor-credit batch semantics: passed")
    print("CPU current producer-runner-CSV schema contract: passed")
    print("CPU successful-candidate funnel trajectory neutrality: passed")
    print("CUDA trajectory-neutral transaction: {}".format(cuda_status))
    if torch.cuda.is_available():
        print("CUDA selection-boundary retention observer: passed")
        print("CUDA full train_adj_on_batch trajectory neutrality: passed")
        print("CUDA candidate-active projection trajectory neutrality: passed")
        print("CUDA exact-context crossing retention floor: passed")
        print("CUDA inactive selected-factor exact-search boundary: passed")
        print("CUDA selection-boundary policy counterfactual: passed")
        print("CUDA run131 late selection-state postcondition: passed")
        print("CUDA complete-transaction crossing floor: passed")
        print("CUDA pair Adam recoverable no-op and continuation: passed")
        print("CUDA pair Adam non-finite fail-loud: passed")
        print("CUDA run134 factor-credit batch semantics: passed")
        print("CUDA current producer-runner-CSV schema contract: passed")
        print("CUDA successful-candidate funnel trajectory neutrality: passed")


if __name__ == "__main__":
    main()
