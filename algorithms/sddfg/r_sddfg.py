import torch
import copy
import hashlib
import math
import random
import torch.nn.functional as F
from utils.util import soft_update, huber_loss, mse_loss, to_torch
import numpy as np
from utils.popart import PopArt
from utils.pair_credit import CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH
from utils.pair_pending import (
    PairOptimizerRecoverableNoOpError,
    PairPendingZeroGradientError,
)
from utils.pair_direction import (
    PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION,
    validate_pair_direction_candidate_seed_contract,
    validate_pair_direction_candidate_kind,
)
from algorithms.sddfg.algorithm.adj_generator import (
    SelectedFactorInactiveCandidateError,
)
from utils.joint_exploration import (
    build_batched_wolfpack_frontier_action_mask_from_local_obs,
)

PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION = 45
CANDIDATE_IDENTITY_TRANSACTION_DIAGNOSTIC_VERSION = 2
CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION = 1
PAIR_SELECTION_BOUNDARY_DIAGNOSTIC_VERSION = 8
PAIR_SELECTION_BOUNDARY_RETENTION_DIAGNOSTIC_VERSION = 3
PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_DIAGNOSTIC_VERSION = 1
PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_DIAGNOSTIC_VERSION = 1
STRICT_PAIR_EXACT_FAILURE_DIAGNOSTIC_VERSION = 5
CANDIDATE_EVIDENCE_CONSUMPTION_FIELDS = (
    "diagnostic_version",
    "selected_episode_ordinal",
    "episode_transition_step",
    "candidate_index",
    "target_sign",
    "target_weight",
    "behavior_policy_version",
)
PAIR_OPTIMIZER_OBJECTIVE_NAMES = (
    "graph",
    "base_factor",
    "capture_outcome",
    "pair",
    "candidate",
    "entropy",
)


class StrictPairExactInfeasibleError(RuntimeError):
    """Fail-loud strict-pair error carrying rollback-safe diagnostics.

    The production exact search mutates only temporary parameter state and the
    caller rolls the transaction back before raising this exception.  Keeping
    the diagnostic payload on the exception lets the runner persist the failed
    transaction without converting an infeasible optimizer transaction into a
    successful/no-op transaction.
    """

    def __init__(self, message, diagnostic_rows):
        super().__init__(message)
        self.strict_pair_exact_failure_rows = tuple(
            dict(row) for row in diagnostic_rows
        )
        shared = (
            self.strict_pair_exact_failure_rows[0]
            if self.strict_pair_exact_failure_rows else {}
        )
        shared_keys = (
            "diagnostic_version",
            "optimizer_kind",
            "failure_classification",
            "origin_preservation_valid",
            "diagnostic_probe_valid_count",
            "bounded_search_exhaustive",
        )
        shared_contract = bool(
            self.strict_pair_exact_failure_rows
            and all(
                all(row.get(key) == shared.get(key) for key in shared_keys)
                for row in self.strict_pair_exact_failure_rows
            )
        )
        # This is not a successful/no-op transaction.  It is the one typed
        # outcome for which the finite production search has proved that no
        # positive step satisfies every hard contract, while scale zero is a
        # valid preservation origin.  The caller may therefore keep the
        # already-completed atomic rollback and defer the unconsumed evidence.
        # Origin drift, a missed valid probe, older/incomplete searches, and
        # every other exact failure remain fail-loud.
        self.strict_pair_exact_bounded_deferral_safe = bool(
            shared_contract
            and int(shared.get("diagnostic_version", -1))
            == STRICT_PAIR_EXACT_FAILURE_DIAGNOSTIC_VERSION
            and str(shared.get("optimizer_kind", "")) in (
                "standard_adam",
                "pair_pending_adam",
            )
            and str(shared.get("failure_classification", ""))
            == "sampled_grid_has_no_feasible_point"
            and int(shared.get("origin_preservation_valid", 0)) == 1
            and int(shared.get("diagnostic_probe_valid_count", -1)) == 0
            and int(shared.get("bounded_search_exhaustive", 0)) == 1
        )


def _to_float(x):
    """兼容 torch 标量 / python float / numpy float"""
    if x is None:
        return 0.0
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    try:
        return float(x)
    except Exception:
        return 0.0


def _transaction_state_equal(left, right):
    """Exact recursive equality for an atomic rollback postcondition."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.device == right.device
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return bool(
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left.keys()) == set(right.keys())
            and all(
                _transaction_state_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return bool(
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _transaction_state_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return bool(left == right)


def _transaction_state_is_finite(value):
    """Return whether every numeric leaf in a transaction state is finite."""
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(
            _transaction_state_is_finite(item) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return all(_transaction_state_is_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _build_n_step_td_targets(
        rewards,
        dones_env,
        next_values,
        gamma,
        n_step):
    """Build finite-horizon TD targets without crossing episode terminals.

    ``next_values[:, t]`` is the target value of ``s[t + 1]``.  Therefore an
    n-step target bootstraps from ``next_values[:, t + n - 1]`` after adding
    rewards ``t .. t + n - 1``.  Tail transitions that do not have a complete
    horizon deliberately do not bootstrap past the stored episode boundary.
    """
    if rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise RuntimeError("SDDFG n-step rewards must have shape [B, T, 1]")
    if dones_env.ndim == 2:
        dones_env = dones_env.unsqueeze(-1)
    if dones_env.shape != rewards.shape:
        raise RuntimeError("SDDFG n-step done mask shape does not match rewards")
    if next_values.shape != rewards.shape:
        raise RuntimeError("SDDFG n-step bootstrap shape does not match rewards")
    n_step = int(n_step)
    if n_step <= 0:
        raise ValueError("SDDFG q_n_step must be positive")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("SDDFG gamma must be finite and non-negative")

    # Keep the established one-step expression bit-for-bit on the default
    # path.  This also makes q_n_step=1 a strict compatibility fixture.
    if n_step == 1:
        return rewards + (1.0 - dones_env) * gamma * next_values

    transition_count = int(rewards.shape[1])
    horizon = min(n_step, transition_count)
    targets = torch.zeros_like(rewards)
    survival = torch.ones_like(rewards)
    for offset in range(horizon):
        width = transition_count - offset
        aligned_survival = survival[:, :width]
        aligned_rewards = rewards[:, offset:]
        targets[:, :width] = (
            targets[:, :width]
            + (gamma ** offset) * aligned_survival * aligned_rewards
        )
        surviving_after_reward = (
            aligned_survival * (1.0 - dones_env[:, offset:])
        )
        survival[:, :width] = surviving_after_reward
        if offset + 1 == n_step:
            targets[:, :width] = (
                targets[:, :width]
                + (gamma ** (offset + 1))
                * surviving_after_reward
                * next_values[:, offset:]
            )

    if not torch.isfinite(targets).all():
        raise FloatingPointError("non-finite SDDFG n-step TD target")
    return targets


def _build_terminal_gated_n_step_td_targets(
        rewards,
        dones_env,
        next_values,
        terminal_win_rewards,
        gamma,
        n_step):
    """Use an n-step return only when its reachable window contains a win.

    run157 showed that applying a 24-step return to every transition summed
    ordinary standardized step rewards into a target whose standard deviation
    was more than five times the one-step baseline, before a single terminal
    transition existed.  The intended intervention was narrower: let a real
    terminal completion update its preceding 24 transitions in one replay.

    The one-step target therefore remains bit-for-bit authoritative for every
    transition without a reachable terminal-win marker.  A transition whose
    next ``n_step`` rewards contain that marker receives the exact ordinary
    n-step target (all intervening rewards plus the aligned bootstrap), so the
    completion reward is neither fabricated nor counted twice.
    """
    one_step_targets = _build_n_step_td_targets(
        rewards, dones_env, next_values, gamma, 1
    )
    n_step = int(n_step)
    gate = torch.zeros_like(rewards, dtype=torch.bool)
    if n_step == 1:
        return one_step_targets, gate, one_step_targets
    if terminal_win_rewards is None:
        raise RuntimeError(
            "SDDFG terminal-gated n-step target requires replay win provenance"
        )
    if terminal_win_rewards.shape != rewards.shape:
        raise RuntimeError(
            "SDDFG terminal-win provenance shape does not match rewards"
        )
    if not torch.isfinite(terminal_win_rewards).all():
        raise FloatingPointError("non-finite SDDFG terminal-win provenance")

    terminal_markers = terminal_win_rewards.abs().gt(0.0)
    if not bool(terminal_markers.any().item()):
        return one_step_targets, gate, one_step_targets

    transition_count = int(rewards.shape[1])
    horizon = min(n_step, transition_count)
    survival = torch.ones_like(rewards)
    for offset in range(horizon):
        width = transition_count - offset
        aligned_survival = survival[:, :width]
        gate[:, :width] = gate[:, :width] | (
            aligned_survival.gt(0.5) & terminal_markers[:, offset:]
        )
        survival[:, :width] = (
            aligned_survival * (1.0 - dones_env[:, offset:])
        )

    # Formal Wolfpack runs use ``continue_after_success`` so the environment
    # done bit deliberately remains false at the first win.  That continuation
    # is useful for trajectory diagnostics, but it is not part of the task-
    # completion return.  Treat the exact win provenance as a semantic return
    # boundary: predecessors receive the completion reward, never rewards or
    # bootstrap values from the diagnostic continuation after success.
    semantic_dones = torch.where(
        terminal_markers,
        torch.ones_like(dones_env),
        dones_env,
    )
    n_step_targets = _build_n_step_td_targets(
        rewards, semantic_dones, next_values, gamma, n_step
    )
    targets = torch.where(gate, n_step_targets, one_step_targets)
    if not torch.isfinite(targets).all():
        raise FloatingPointError(
            "non-finite SDDFG terminal-gated n-step TD target"
        )
    return targets, gate, one_step_targets


def _build_terminal_replay_loss_mask(
        valid_transition_mask,
        q_n_step_gate,
        terminal_replay_lane_episode_mask):
    """Keep uniform loss intact and admit only gated auxiliary transitions."""
    loss_mask, _, _, _, lane_episode_count = (
        _build_terminal_replay_loss_population(
            valid_transition_mask,
            q_n_step_gate,
            terminal_replay_lane_episode_mask,
            1.0,
        )
    )
    return loss_mask, lane_episode_count


def _build_terminal_replay_loss_population(
        valid_transition_mask,
        q_n_step_gate,
        terminal_replay_lane_episode_mask,
        terminal_replay_loss_weight):
    """Build separate uniform and weighted auxiliary loss populations.

    run159 showed that repeatedly appending one successful episode with the
    same MSE weight as ordinary uniform transitions raised median post-lane Q
    loss from 0.90 to 3.50 before first-capture geometry could improve.  Keep
    the uniform objective unchanged, exclude the auxiliary
    episode's 176 ordinary steps, and assign only its 24 completion-bearing
    transitions the explicit replay-lane weight.
    """
    terminal_replay_loss_weight = float(terminal_replay_loss_weight)
    if (
            not math.isfinite(terminal_replay_loss_weight)
            or terminal_replay_loss_weight <= 0.0
            or terminal_replay_loss_weight > 1.0):
        raise ValueError(
            "SDDFG terminal replay loss weight must be finite in (0, 1]"
        )
    if terminal_replay_lane_episode_mask is None:
        zero_mask = torch.zeros_like(valid_transition_mask)
        return (
            valid_transition_mask,
            valid_transition_mask,
            valid_transition_mask,
            zero_mask,
            valid_transition_mask.new_tensor(0.0),
        )
    batch_size = int(valid_transition_mask.shape[0])
    lane_mask = terminal_replay_lane_episode_mask.reshape(-1)
    if lane_mask.shape[0] != batch_size:
        raise RuntimeError("SDDFG terminal replay lane batch size mismatch")
    lane_episode_mask = lane_mask.gt(0.5).view(batch_size, 1, 1)
    uniform_transition_mask = (
        valid_transition_mask * (~lane_episode_mask).float()
    )
    auxiliary_transition_mask = (
        valid_transition_mask
        * lane_episode_mask.float()
        * q_n_step_gate.float()
    )
    loss_mask = uniform_transition_mask + auxiliary_transition_mask
    loss_weight = (
        uniform_transition_mask
        + terminal_replay_loss_weight * auxiliary_transition_mask
    )
    return (
        loss_mask,
        loss_weight,
        uniform_transition_mask,
        auxiliary_transition_mask,
        lane_episode_mask.float().sum(),
    )


def _candidate_target_bearing_update_diagnostic(
        candidate_target_present,
        reference):
    """Report current candidate supervision independently of cache state."""
    return reference.new_tensor(float(bool(candidate_target_present)))


def _reshape_replay_population_provenance(
        replay_population_provenance_batch,
        batch_size):
    """Decode the fixed three-column replay provenance without changing it."""
    provenance = to_torch(replay_population_provenance_batch)
    expected_element_count = int(batch_size) * 3
    if provenance.numel() != expected_element_count:
        raise ValueError(
            "replay population provenance must contain {} values for shape "
            "[{}, 3], got {} values".format(
                expected_element_count,
                int(batch_size),
                int(provenance.numel()),
            )
        )
    provenance = provenance.reshape(int(batch_size), 3)
    if provenance.shape != (int(batch_size), 3):
        raise RuntimeError(
            "replay population provenance reshape contract failed"
        )
    return provenance


def _successful_candidate_capture_boundary_diagnostics(
        candidate_capture_context,
        episode_success_gate,
        behavior_candidate_margin,
        behavior_candidate_rank,
        behavior_candidate_valid,
        current_candidate_margin,
        current_candidate_rank,
        current_candidate_valid,
        candidate_identity_delta):
    """Join successful candidate-only captures to the current active boundary.

    Every input is already produced by the normal adjacency forward or replay
    decode.  The helper detaches all values, performs no forward/backward/RNG
    work, and never changes the candidate objective.  Exact identity alignment
    is the shared canonical candidate axis.
    """
    context = candidate_capture_context.detach()
    success_gate = episode_success_gate.detach()
    behavior_margin = behavior_candidate_margin.detach()
    behavior_rank = behavior_candidate_rank.detach()
    behavior_valid = behavior_candidate_valid.detach()
    current_margin = current_candidate_margin.detach()
    current_rank = current_candidate_rank.detach()
    current_valid = current_candidate_valid.detach()
    identity_delta = candidate_identity_delta.detach()
    expected_shape = current_margin.shape
    for name, tensor in (
            ("candidate_capture_context", context),
            ("behavior_candidate_margin", behavior_margin),
            ("behavior_candidate_rank", behavior_rank),
            ("behavior_candidate_valid", behavior_valid),
            ("current_candidate_rank", current_rank),
            ("current_candidate_valid", current_valid),
            ("candidate_identity_delta", identity_delta)):
        if tensor.shape != expected_shape:
            raise ValueError(
                "{} shape {} does not match candidate margin {}"
                .format(name, tuple(tensor.shape), tuple(expected_shape))
            )
    if success_gate.ndim != 2 or success_gate.shape != (
            expected_shape[0], 1):
        raise ValueError(
            "episode success gate shape {} does not match candidate batch {}"
            .format(tuple(success_gate.shape), expected_shape[0])
        )
    finite_tensors = (
        context,
        success_gate,
        behavior_margin,
        behavior_rank,
        behavior_valid,
        current_margin,
        current_rank,
        current_valid,
        identity_delta,
    )
    if not all(bool(torch.isfinite(tensor).all().item())
               for tensor in finite_tensors):
        raise FloatingPointError(
            "non-finite successful candidate capture boundary diagnostic"
        )
    if bool(torch.any(context < 0.0).item()):
        raise ValueError("candidate capture context must be non-negative")

    successful_weight = (
        context
        * (success_gate > 0.0).to(context.dtype)
    )
    target_mask = successful_weight > 0.0
    if bool(torch.any(target_mask & (behavior_valid <= 0.0)).item()):
        raise RuntimeError(
            "successful candidate capture has invalid rollout identity metadata"
        )
    if bool(torch.any(target_mask & (current_valid <= 0.0)).item()):
        raise RuntimeError(
            "successful candidate capture identity disappeared from the "
            "current canonical catalog"
        )
    if bool(torch.any(target_mask & (behavior_rank < 1.0)).item()):
        raise RuntimeError(
            "successful candidate capture has no positive rollout rank"
        )
    if bool(torch.any(target_mask & (current_rank < 1.0)).item()):
        raise RuntimeError(
            "successful candidate capture has no positive current rank"
        )

    reference = current_margin
    identity_mass = successful_weight.sum()
    identity_count = target_mask.to(current_margin.dtype).sum()
    denominator = identity_mass.clamp_min(1.0)

    def _weighted_mean(values):
        return (values * successful_weight).sum() / denominator

    def _masked_extreme(values, maximum):
        if not bool(torch.any(target_mask).item()):
            return reference.new_tensor(0.0)
        selected = values[target_mask]
        return selected.max() if maximum else selected.min()

    current_deficit = torch.clamp(-current_margin, min=0.0)
    target_overlap = (identity_delta.abs() > 0.0).to(context.dtype)
    positive_target_overlap = (identity_delta > 0.0).to(context.dtype)
    return {
        "version": reference.new_tensor(1.0),
        "identity_count": identity_count,
        "identity_mass": identity_mass,
        "behavior_margin_mean": _weighted_mean(behavior_margin),
        "behavior_rank_mean": _weighted_mean(behavior_rank),
        "current_margin_mean": _weighted_mean(current_margin),
        "current_margin_min": _masked_extreme(
            current_margin, maximum=False
        ),
        "current_margin_max": _masked_extreme(
            current_margin, maximum=True
        ),
        "current_rank_mean": _weighted_mean(current_rank),
        "current_rank_min": _masked_extreme(current_rank, maximum=False),
        "current_rank_max": _masked_extreme(current_rank, maximum=True),
        "current_boundary_deficit_mean": _weighted_mean(current_deficit),
        "current_boundary_crossed_fraction": _weighted_mean(
            (current_margin > 0.0).to(context.dtype)
        ),
        "current_rank1_fraction": _weighted_mean(
            (current_rank <= 1.0).to(context.dtype)
        ),
        "margin_change_mean": _weighted_mean(
            current_margin - behavior_margin
        ),
        "rank_improved_fraction": _weighted_mean(
            (current_rank < behavior_rank).to(context.dtype)
        ),
        "candidate_target_overlap_fraction": _weighted_mean(target_overlap),
        "positive_candidate_target_overlap_fraction": _weighted_mean(
            positive_target_overlap
        ),
        "identity_join_valid": reference.new_tensor(1.0),
    }


def _canonical_candidate_identity(candidate_index, num_agents, highest_orders):
    """Decode the fixed pair/triplet catalog without a model forward."""
    candidate_index = int(candidate_index)
    num_agents = int(num_agents)
    highest_orders = int(highest_orders)
    if candidate_index < 0 or num_agents < 2:
        raise ValueError("invalid canonical candidate catalog request")
    catalog_index = 0
    for first in range(num_agents):
        for second in range(first + 1, num_agents):
            if catalog_index == candidate_index:
                return 2, "{}-{}".format(first, second)
            catalog_index += 1
    if highest_orders >= 3:
        for first in range(num_agents):
            for second in range(first + 1, num_agents):
                for third in range(second + 1, num_agents):
                    if catalog_index == candidate_index:
                        return 3, "{}-{}-{}".format(
                            first,
                            second,
                            third,
                        )
                    catalog_index += 1
    raise ValueError(
        "canonical candidate index {} exceeds catalog size {}"
        .format(candidate_index, catalog_index)
    )


def _candidate_rank_decomposition(
        candidate_margins,
        candidate_valid,
        candidate_index):
    """Explain one deterministic canonical rank in its exact valid population."""
    margins = candidate_margins.detach()
    valid = candidate_valid.detach() > 0.0
    candidate_index = int(candidate_index)
    if margins.dim() != 1 or valid.dim() != 1:
        raise ValueError("candidate rank decomposition inputs must be vectors")
    if margins.shape != valid.shape:
        raise ValueError(
            "candidate rank decomposition shapes differ: {} vs {}"
            .format(tuple(margins.shape), tuple(valid.shape))
        )
    if (
            candidate_index < 0
            or candidate_index >= int(margins.shape[0])
            or not bool(valid[candidate_index].item())):
        raise RuntimeError(
            "rank decomposition target is outside the valid candidate population"
        )
    if not bool(torch.isfinite(margins).all().item()):
        raise FloatingPointError(
            "non-finite candidate rank decomposition margin"
        )
    target_margin = margins[candidate_index]
    catalog_indices = torch.arange(
        int(margins.shape[0]),
        device=margins.device,
    )
    strictly_better = valid & (margins > target_margin)
    tie_precedes = (
        valid
        & (margins == target_margin)
        & (catalog_indices < candidate_index)
    )
    precedes = strictly_better | tie_precedes
    strict_count = int(strictly_better.long().sum().item())
    tie_count = int(tie_precedes.long().sum().item())
    reconstructed_rank = 1 + strict_count + tie_count
    valid_population_count = int(valid.long().sum().item())
    next_better_gap = 0.0
    next_better_index = -1
    if bool(torch.any(precedes).item()):
        gaps = torch.where(
            precedes,
            margins - target_margin,
            torch.full_like(margins, float("inf")),
        )
        next_better_index = int(torch.argmin(gaps).item())
        next_better_gap = float(
            gaps[next_better_index].detach().cpu().item()
        )
    return {
        "valid_population_count": valid_population_count,
        "strictly_better_count": strict_count,
        "tie_precedes_count": tie_count,
        "reconstructed_rank": reconstructed_rank,
        "next_better_candidate_index": next_better_index,
        "next_better_margin_gap": next_better_gap,
    }


def _candidate_identity_transaction_trace_rows(
        candidate_identity_delta,
        candidate_effective_delta,
        candidate_capture_context,
        episode_success_gate,
        behavior_candidate_margin,
        behavior_candidate_rank,
        behavior_candidate_valid,
        behavior_candidate_version,
        current_candidate_policy_version,
        pre_candidate_margin,
        pre_candidate_rank,
        pre_candidate_valid,
        post_candidate_margin,
        post_candidate_rank,
        post_candidate_valid,
        replay_population_provenance,
        candidate_lifecycle_progress_mask,
        num_agents,
        highest_orders):
    """Build detached per-target rows from already-computed transaction tensors.

    This diagnostic performs no forward, backward, RNG, optimizer, or lifecycle
    operation.  It explains score-to-rank conversion in exactly the same valid
    canonical population before and after the committed optimizer transaction.
    """
    tensors = (
        candidate_identity_delta,
        candidate_effective_delta,
        candidate_capture_context,
        behavior_candidate_margin,
        behavior_candidate_rank,
        behavior_candidate_valid,
        behavior_candidate_version,
        pre_candidate_margin,
        pre_candidate_rank,
        pre_candidate_valid,
        post_candidate_margin,
        post_candidate_rank,
        post_candidate_valid,
        candidate_lifecycle_progress_mask,
    )
    expected_shape = candidate_identity_delta.shape
    for tensor in tensors:
        if tensor.shape != expected_shape:
            raise ValueError(
                "candidate identity transaction trace shape {} does not match {}"
                .format(tuple(tensor.shape), tuple(expected_shape))
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise FloatingPointError(
                "non-finite candidate identity transaction trace input"
            )
    success_gate = episode_success_gate.detach()
    provenance = replay_population_provenance.detach()
    if success_gate.shape != (expected_shape[0], 1):
        raise ValueError(
            "candidate trace success gate has shape {}, expected {}"
            .format(tuple(success_gate.shape), (expected_shape[0], 1))
        )
    if provenance.shape != (expected_shape[0], 3):
        raise ValueError(
            "candidate trace replay provenance has shape {}, expected {}"
            .format(tuple(provenance.shape), (expected_shape[0], 3))
        )
    if not bool(torch.isfinite(success_gate).all().item()):
        raise FloatingPointError("non-finite candidate trace success gate")
    if not bool(torch.isfinite(provenance).all().item()):
        raise FloatingPointError("non-finite candidate trace provenance")
    if not bool(torch.equal(
            pre_candidate_valid.detach() > 0.0,
            post_candidate_valid.detach() > 0.0,
    )):
        raise RuntimeError(
            "candidate valid population changed inside one optimizer transaction"
        )

    delta = candidate_identity_delta.detach()
    effective_delta = candidate_effective_delta.detach()
    capture_context = candidate_capture_context.detach()
    behavior_margin = behavior_candidate_margin.detach()
    behavior_rank = behavior_candidate_rank.detach()
    behavior_valid = behavior_candidate_valid.detach()
    behavior_version = behavior_candidate_version.detach()
    current_policy_version = int(current_candidate_policy_version)
    if current_policy_version < 0:
        raise RuntimeError("candidate trace current policy version is negative")
    pre_margin = pre_candidate_margin.detach()
    pre_rank = pre_candidate_rank.detach()
    pre_valid = pre_candidate_valid.detach()
    post_margin = post_candidate_margin.detach()
    post_rank = post_candidate_rank.detach()
    post_valid = post_candidate_valid.detach()
    lifecycle_progress = candidate_lifecycle_progress_mask.detach()
    # PyTorch 1.3.1 does not support the newer ``as_tuple`` keyword.
    target_locations = torch.nonzero(effective_delta != 0.0)
    rows = []
    for target_location in target_locations.detach().cpu().tolist():
        transition_index = int(target_location[0])
        candidate_index = int(target_location[1])
        target_sign = 1.0 if float(
            effective_delta[
                transition_index, candidate_index
            ].detach().cpu().item()
        ) > 0.0 else -1.0
        pre_parts = _candidate_rank_decomposition(
            candidate_margins=pre_margin[transition_index],
            candidate_valid=pre_valid[transition_index],
            candidate_index=candidate_index,
        )
        post_parts = _candidate_rank_decomposition(
            candidate_margins=post_margin[transition_index],
            candidate_valid=post_valid[transition_index],
            candidate_index=candidate_index,
        )
        observed_pre_rank = int(round(float(
            pre_rank[transition_index, candidate_index]
            .detach().cpu().item()
        )))
        observed_post_rank = int(round(float(
            post_rank[transition_index, candidate_index]
            .detach().cpu().item()
        )))
        if observed_pre_rank != pre_parts["reconstructed_rank"]:
            raise RuntimeError(
                "pre-update candidate rank does not reconstruct from the "
                "same valid population"
            )
        if observed_post_rank != post_parts["reconstructed_rank"]:
            raise RuntimeError(
                "post-update candidate rank does not reconstruct from the "
                "same valid population"
            )
        factor_order, canonical_identity = _canonical_candidate_identity(
            candidate_index=candidate_index,
            num_agents=num_agents,
            highest_orders=highest_orders,
        )
        pre_value = float(
            pre_margin[transition_index, candidate_index]
            .detach().cpu().item()
        )
        post_value = float(
            post_margin[transition_index, candidate_index]
            .detach().cpu().item()
        )
        signed_margin_change = target_sign * (post_value - pre_value)
        signed_rank_improvement = (
            target_sign * float(observed_pre_rank - observed_post_rank)
        )
        pre_signed_boundary = target_sign * pre_value
        post_signed_boundary = target_sign * post_value
        target_weight = abs(float(
            delta[transition_index, candidate_index]
            .detach().cpu().item()
        ))
        episode_ordinal_value = float(
            provenance[transition_index, 1].detach().cpu().item()
        )
        episode_ordinal = int(round(episode_ordinal_value))
        if abs(episode_ordinal_value - float(episode_ordinal)) > 1e-6:
            raise RuntimeError(
                "candidate trace episode ordinal is not integral"
            )
        episode_step_value = float(
            provenance[transition_index, 2].detach().cpu().item()
        )
        episode_transition_step = int(round(episode_step_value))
        if (
                abs(
                    episode_step_value - float(episode_transition_step)
                ) > 1e-6
                or episode_transition_step < 0):
            raise RuntimeError(
                "candidate trace episode transition step is invalid"
            )
        successful_capture_weight = float(
            (
                capture_context[transition_index, candidate_index]
                * (success_gate[transition_index, 0] > 0.0).to(
                    capture_context.dtype
                )
            ).detach().cpu().item()
        )
        rows.append({
            "diagnostic_version": int(
                CANDIDATE_IDENTITY_TRANSACTION_DIAGNOSTIC_VERSION
            ),
            "target_row_sequence_within_transaction": int(len(rows)),
            "selected_episode_ordinal": episode_ordinal,
            "episode_transition_step": episode_transition_step,
            "transition_index_in_partition": transition_index,
            "candidate_index": candidate_index,
            "canonical_identity": canonical_identity,
            "factor_order": factor_order,
            "target_sign": target_sign,
            "target_weight": target_weight,
            "pair_evidence_transition": int(round(float(
                provenance[transition_index, 0].detach().cpu().item()
            ))),
            "successful_candidate_capture_overlap": (
                successful_capture_weight
            ),
            "behavior_margin": float(
                behavior_margin[transition_index, candidate_index]
                .detach().cpu().item()
            ),
            "behavior_rank": float(
                behavior_rank[transition_index, candidate_index]
                .detach().cpu().item()
            ),
            "behavior_valid": float(
                behavior_valid[transition_index, candidate_index]
                .detach().cpu().item()
            ),
            "behavior_policy_version": float(
                behavior_version[transition_index, candidate_index]
                .detach().cpu().item()
            ),
            "candidate_policy_age": (
                float(current_policy_version)
                - float(
                    behavior_version[transition_index, candidate_index]
                    .detach().cpu().item()
                )
            ),
            "pre_margin": pre_value,
            "post_margin": post_value,
            "signed_margin_change": signed_margin_change,
            "margin_direction_correct": int(signed_margin_change > 0.0),
            "pre_rank": observed_pre_rank,
            "post_rank": observed_post_rank,
            "signed_rank_improvement": signed_rank_improvement,
            "rank_improved": int(signed_rank_improvement > 0.0),
            "pre_signed_boundary": pre_signed_boundary,
            "post_signed_boundary": post_signed_boundary,
            "pre_boundary_deficit": max(-pre_signed_boundary, 0.0),
            "post_boundary_deficit": max(-post_signed_boundary, 0.0),
            "boundary_crossing": int(
                pre_signed_boundary <= 0.0
                and post_signed_boundary > 0.0
            ),
            "pre_valid_population_count": (
                pre_parts["valid_population_count"]
            ),
            "post_valid_population_count": (
                post_parts["valid_population_count"]
            ),
            "pre_strictly_better_count": (
                pre_parts["strictly_better_count"]
            ),
            "post_strictly_better_count": (
                post_parts["strictly_better_count"]
            ),
            "pre_tie_precedes_count": pre_parts["tie_precedes_count"],
            "post_tie_precedes_count": post_parts["tie_precedes_count"],
            "pre_next_better_candidate_index": (
                pre_parts["next_better_candidate_index"]
            ),
            "post_next_better_candidate_index": (
                post_parts["next_better_candidate_index"]
            ),
            "pre_next_better_margin_gap": (
                pre_parts["next_better_margin_gap"]
            ),
            "post_next_better_margin_gap": (
                post_parts["next_better_margin_gap"]
            ),
            "same_population_rank_reconstruction_valid": 1,
            "lifecycle_behavioral_progress": int(
                lifecycle_progress[
                    transition_index, candidate_index
                ].detach().cpu().item() > 0.0
            ),
        })
    expected_row_count = int(
        (effective_delta != 0.0).long().sum().detach().cpu().item()
    )
    if len(rows) != expected_row_count:
        raise RuntimeError(
            "candidate identity transaction trace dropped a target row"
        )
    return rows


def _candidate_evidence_consumption_trace_rows(
        candidate_identity_delta,
        behavior_candidate_version,
        replay_population_provenance):
    """Return detached keys for every target consumed by this transaction."""
    delta = candidate_identity_delta.detach()
    behavior_version = behavior_candidate_version.detach()
    provenance = replay_population_provenance.detach()
    if delta.shape != behavior_version.shape:
        raise ValueError(
            "candidate evidence delta/version shapes must match"
        )
    if provenance.shape != (delta.shape[0], 3):
        raise ValueError(
            "candidate evidence replay provenance must have shape [{}, 3]"
            .format(delta.shape[0])
        )
    if not bool(
            torch.isfinite(delta).all().item()
            and torch.isfinite(behavior_version).all().item()
            and torch.isfinite(provenance).all().item()):
        raise FloatingPointError(
            "candidate evidence consumption trace is non-finite"
        )
    locations = torch.nonzero(delta != 0.0)
    rows = []
    for transition_index, candidate_index in (
            locations.detach().cpu().tolist()):
        episode_ordinal_value = float(
            provenance[transition_index, 1].detach().cpu().item()
        )
        episode_step_value = float(
            provenance[transition_index, 2].detach().cpu().item()
        )
        episode_ordinal = int(round(episode_ordinal_value))
        episode_step = int(round(episode_step_value))
        if (
                abs(episode_ordinal_value - episode_ordinal) > 1e-6
                or abs(episode_step_value - episode_step) > 1e-6
                or episode_ordinal < 0
                or episode_step < 0):
            raise RuntimeError(
                "candidate evidence consumption provenance is not integral"
            )
        delta_value = float(
            delta[transition_index, candidate_index]
            .detach().cpu().item()
        )
        row = {
            "diagnostic_version": int(
                CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION
            ),
            "selected_episode_ordinal": episode_ordinal,
            "episode_transition_step": episode_step,
            "candidate_index": int(candidate_index),
            "target_sign": int(1 if delta_value > 0.0 else -1),
            "target_weight": abs(delta_value),
            "behavior_policy_version": float(
                behavior_version[
                    transition_index,
                    candidate_index,
                ].detach().cpu().item()
            ),
        }
        if tuple(row.keys()) != CANDIDATE_EVIDENCE_CONSUMPTION_FIELDS:
            raise RuntimeError(
                "candidate evidence consumption trace schema diverged"
            )
        rows.append(row)
    return rows


def _elementwise_minimum_compat(left, right):
    """Elementwise minimum compatible with PyTorch before torch.minimum."""
    return torch.where(left <= right, left, right)


def _compose_adj_candidate_objective(
        base_rl_loss,
        candidate_loss,
        entropy,
        entropy_coef,
        candidate_residual_only):
    """Compose either the full graph objective or a candidate-only residual.

    Adjacency PPO early stopping is defined by graph/factor importance ratios.
    The candidate competitor hinge has no PPO ratio, so tying it to that stop
    silently removes its remaining configured same-update epochs.  A residual
    epoch therefore optimizes only the exact candidate hinge.  It never applies
    another PPO, entropy, factor-credit, or outcome-factor gradient.
    """
    if candidate_residual_only:
        return candidate_loss
    return base_rl_loss + candidate_loss - float(entropy_coef) * entropy


def _clear_parameter_gradients_to_none(parameters):
    """Clear shared parameter gradients with legacy-PyTorch semantics.

    PyTorch 1.3 ``Optimizer.zero_grad`` leaves zero tensors in ``.grad``.
    Adam treats those tensors as real gradients and therefore advances old
    momentum even when the current objective does not use the parameter.
    Candidate-only residual steps require an explicit ``None`` so inactive
    graph parameters are skipped transactionally.
    """
    for parameter in parameters:
        parameter.grad = None


def _select_adjacency_optimizer(
        candidate_residual_only,
        pair_only_objective,
        standard_optimizer,
        residual_optimizer,
        pair_optimizer):
    """Select the optimizer whose state belongs to the active objective."""
    if bool(candidate_residual_only) and bool(pair_only_objective):
        raise RuntimeError(
            "candidate-residual and pair-only optimizer scopes overlap"
        )
    if candidate_residual_only:
        if residual_optimizer is None:
            raise RuntimeError(
                "candidate residual optimizer is not initialized"
            )
        return residual_optimizer
    if pair_only_objective:
        if pair_optimizer is None:
            raise RuntimeError(
                "pair pending optimizer is not initialized"
            )
        return pair_optimizer
    return standard_optimizer


def _adjacency_optimizer_checkpoint_state(
        standard_optimizer,
        residual_optimizer,
        pair_optimizer):
    """Return all adjacency Adam states as one versioned checkpoint.

    Standard, candidate-residual, and pending-pair objectives intentionally own
    disjoint Adam moments for the same Parameter objects. Saving only model
    weights, or only a subset of these optimizers, changes the update rule after
    restoration.
    """
    if (
            standard_optimizer is None
            or residual_optimizer is None
            or pair_optimizer is None):
        raise RuntimeError(
            "all adjacency optimizers are required for checkpointing"
        )
    return {
        "version": 2,
        "standard_optimizer": standard_optimizer.state_dict(),
        "candidate_residual_optimizer": residual_optimizer.state_dict(),
        "pair_pending_optimizer": pair_optimizer.state_dict(),
    }


def _load_adjacency_optimizer_checkpoint_state(
        checkpoint,
        standard_optimizer,
        residual_optimizer,
        pair_optimizer):
    """Transactionally restore all adjacency Adam states.

    Version 1 checkpoints predate the isolated pair optimizer. They remain
    readable: the two historical optimizers are restored exactly and the pair
    optimizer is reset to its freshly initialized empty state. Version 2
    checkpoints require and restore every optimizer, so a resumed run cannot
    silently lose pair momentum.
    """
    if not isinstance(checkpoint, dict):
        raise RuntimeError("adjacency optimizer checkpoint must be a dict")
    version = int(checkpoint.get("version", -1))
    if version not in (1, 2):
        raise RuntimeError(
            "unsupported adjacency optimizer checkpoint version"
        )
    required = [
        "standard_optimizer",
        "candidate_residual_optimizer",
    ]
    if version >= 2:
        required.append("pair_pending_optimizer")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(
            "adjacency optimizer checkpoint is missing {}".format(
                ", ".join(missing)
            )
        )
    if (
            standard_optimizer is None
            or residual_optimizer is None
            or pair_optimizer is None):
        raise RuntimeError(
            "all adjacency optimizers are required for restoration"
        )

    standard_before = copy.deepcopy(standard_optimizer.state_dict())
    residual_before = copy.deepcopy(residual_optimizer.state_dict())
    pair_before = copy.deepcopy(pair_optimizer.state_dict())
    try:
        standard_optimizer.load_state_dict(
            checkpoint["standard_optimizer"]
        )
        residual_optimizer.load_state_dict(
            checkpoint["candidate_residual_optimizer"]
        )
        if version >= 2:
            pair_optimizer.load_state_dict(
                checkpoint["pair_pending_optimizer"]
            )
        else:
            empty_pair_state = copy.deepcopy(pair_before)
            empty_pair_state["state"] = {}
            pair_optimizer.load_state_dict(empty_pair_state)
    except Exception:
        standard_optimizer.load_state_dict(standard_before)
        residual_optimizer.load_state_dict(residual_before)
        pair_optimizer.load_state_dict(pair_before)
        raise


def _candidate_lifecycle_behavioral_progress_mask(
        identity_delta,
        pre_unsatisfied_mask,
        post_unsatisfied_mask,
        pre_rank,
        post_rank):
    """Select candidate updates that changed a real behavior boundary.

    A lifecycle constraint is useful only after the committed update either
    improves canonical rank in the target direction or reaches the signed
    competitor goal.  Protecting every tiny margin movement consumed task
    gradient in run83 without producing a single positive boundary crossing.
    """
    expected_shape = identity_delta.shape
    for name, tensor in (
            ("pre_unsatisfied_mask", pre_unsatisfied_mask),
            ("post_unsatisfied_mask", post_unsatisfied_mask),
            ("pre_rank", pre_rank),
            ("post_rank", post_rank)):
        if tensor.shape != expected_shape:
            raise ValueError(
                "{} shape {} does not match identity delta {}"
                .format(name, tuple(tensor.shape), tuple(expected_shape))
            )
    target = identity_delta != 0.0
    pre_unsatisfied = pre_unsatisfied_mask > 0.0
    post_unsatisfied = post_unsatisfied_mask > 0.0
    rank_improved = (
        ((identity_delta > 0.0) & (post_rank < pre_rank))
        | ((identity_delta < 0.0) & (post_rank > pre_rank))
    )
    reached_goal = pre_unsatisfied & (~post_unsatisfied)
    return (
        target & pre_unsatisfied & (rank_improved | reached_goal)
    ).to(identity_delta.dtype)


def _select_candidate_lifecycle_committed_info(
        post_candidate_info,
        final_candidate_info,
        lifecycle_target_present):
    """Select diagnostics for the parameters that were actually committed.

    Without an existing lifecycle constraint, the immediate post-optimizer
    candidate state is final.  When projection, backtracking, or rollback can
    modify that optimizer step, registration must use the state recomputed
    after the complete lifecycle transaction.
    """
    selected = (
        final_candidate_info
        if bool(lifecycle_target_present)
        else post_candidate_info
    )
    if selected is None:
        raise RuntimeError(
            "candidate lifecycle committed state was not evaluated"
        )
    return selected


def _nan_to_num_compat(x, nan=0.0, posinf=None, neginf=None):
    """
    兼容旧版本 PyTorch 的 nan_to_num。
    旧版 torch 没有 torch.nan_to_num，因此用 torch.where 手动替换 NaN/Inf。
    """
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    out = x

    # 替换 NaN
    out = torch.where(
        torch.isnan(out),
        torch.full_like(out, float(nan)),
        out
    )

    # 替换 +Inf
    if posinf is not None:
        out = torch.where(
            torch.isinf(out) & (out > 0),
            torch.full_like(out, float(posinf)),
            out
        )

    # 替换 -Inf
    if neginf is not None:
        out = torch.where(
            torch.isinf(out) & (out < 0),
            torch.full_like(out, float(neginf)),
            out
        )

    return out


def _parameter_grad_norm(parameters):
    """Return the finite L2 norm of an explicit parameter collection."""
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).cpu().item())
        total_sq += grad_norm * grad_norm
    return float(total_sq ** 0.5)


def _gradient_tuple_dot(left, right, reference):
    """Dot product between two parameter-gradient tuples."""
    result = reference.new_tensor(0.0)
    for left_grad, right_grad in zip(left, right):
        if left_grad is not None and right_grad is not None:
            result = result + (left_grad * right_grad).sum()
    return result


def _floating_dtype_epsilon(reference):
    """Return the write-precision epsilon on legacy PyTorch releases."""
    return (
        2.220446049250313e-16
        if reference.dtype == torch.float64
        else 1.1920928955078125e-7
    )


def _selection_boundary_replay_tolerance(
        selected_logp,
        competitor_logp,
        commit_floor,
        tolerance_multiplier=16.0):
    """Bound exact-boundary replay noise by its log-probability operands.

    A selected-factor boundary is ``selected_logp - competitor_logp``. Near a
    real rank crossing those two O(1) operands almost cancel, so scaling a
    tolerance by the tiny resulting margin makes the tolerance smaller than
    one float32 ulp. Replaying the same archived context on CUDA can then look
    like a lost floor even though no adjacency write occurred. Measure write
    resolution from the operands, matching the exact boundary search contract,
    while keeping genuine finite regressions fail-loud.
    """
    if selected_logp.shape != competitor_logp.shape:
        raise ValueError(
            "selection-boundary replay log-probability shapes differ"
        )
    if not all(bool(torch.isfinite(value).all().item()) for value in (
            selected_logp, competitor_logp)):
        raise FloatingPointError(
            "non-finite selection-boundary replay log-probability"
        )
    commit_floor = float(commit_floor)
    tolerance_multiplier = float(tolerance_multiplier)
    if (
            not np.isfinite(commit_floor)
            or not np.isfinite(tolerance_multiplier)
            or tolerance_multiplier <= 0.0):
        raise ValueError("invalid selection-boundary replay tolerance input")
    operand_scale = torch.max(
        selected_logp.detach().abs(),
        competitor_logp.detach().abs(),
    ).clamp_min(max(1.0, abs(commit_floor)))
    return (
        tolerance_multiplier
        * _floating_dtype_epsilon(selected_logp)
        * operand_scale
    )


def _boundary_write_resolution_scale_limit(
        selected_logp,
        competitor_logp,
        actionable_pair_mask,
        direction_boundary_dots,
        reference,
        current_scale_limit=1.0,
        maximum_scale=2.0 ** 20,
        resolution_multiplier=16.0):
    """Bound a scale search by the first representable boundary change.

    A strict pair boundary is the difference of two float32 log-probabilities.
    When an otherwise valid projected displacement has a small positive
    Jacobian dot, every exact forward at scales ``1, 1/2, ...`` can therefore
    reproduce the origin bit-for-bit.  Backtracking alone can never make that
    positive change observable.  Derive the smallest useful scale from the
    write precision of the two boundary operands and the best complete
    positive boundary-dot vector.  The returned value is only a finite search
    cap: the production exact forward remains the sole commit authority.

    This helper is called only after the complete scale-at-most-one grid has
    failed.  Testing at least scale two is consequently evidence-driven rather
    than a change to the ordinary Adam path.
    """
    current_scale_limit = float(current_scale_limit)
    maximum_scale = float(maximum_scale)
    resolution_multiplier = float(resolution_multiplier)
    if (
            not np.isfinite(current_scale_limit)
            or current_scale_limit < 1.0
            or not np.isfinite(maximum_scale)
            or maximum_scale < current_scale_limit
            or not np.isfinite(resolution_multiplier)
            or resolution_multiplier <= 0.0):
        raise ValueError("invalid boundary write-resolution search bound")
    if (
            selected_logp.shape != competitor_logp.shape
            or selected_logp.shape != actionable_pair_mask.shape):
        raise ValueError("boundary write-resolution tensor shapes differ")
    if not all(bool(torch.isfinite(value).all().item()) for value in (
            selected_logp, competitor_logp)):
        raise FloatingPointError("non-finite boundary write-resolution input")
    actionable = actionable_pair_mask > 0.0
    if not bool(torch.any(actionable).item()):
        raise RuntimeError("boundary write-resolution has no actionable target")
    selected_values = selected_logp[actionable].detach()
    competitor_values = competitor_logp[actionable].detach()
    # A subtraction can remain unchanged until either float32 operand moves by
    # at least one ulp.  The lifecycle exact contract already uses a 16-epsilon
    # guard, so reuse that conservative write-resolution convention here.
    operand_scale = torch.max(
        selected_values.abs(), competitor_values.abs()
    ).clamp_min(1.0)
    required_change = (
        resolution_multiplier
        * _floating_dtype_epsilon(reference)
        * operand_scale
    )
    best_required_scale = None
    target_count = int(required_change.numel())
    for raw_dots in direction_boundary_dots:
        dots = raw_dots.detach().reshape(-1)
        if dots.numel() != target_count:
            raise ValueError(
                "boundary write-resolution direction population differs"
            )
        if not bool(torch.isfinite(dots).all().item()):
            raise FloatingPointError(
                "non-finite boundary write-resolution direction dot"
            )
        if not bool(torch.all(dots > 0.0).item()):
            raise RuntimeError(
                "boundary write-resolution direction lost strict linear dot"
            )
        required_scale = float(
            (required_change / dots).max().detach().cpu().item()
        )
        if not np.isfinite(required_scale) or required_scale < 0.0:
            raise FloatingPointError(
                "invalid boundary write-resolution required scale"
            )
        if (
                best_required_scale is None
                or required_scale < best_required_scale):
            best_required_scale = required_scale
    if best_required_scale is None:
        raise ValueError("boundary write-resolution direction set is empty")
    resolution_limit = max(
        current_scale_limit,
        2.0,
        best_required_scale,
    )
    return float(min(resolution_limit, maximum_scale))


def _gradient_dot_tolerance(
        proposed_grads,
        constraint_grads,
        reference,
        tolerance_multiplier):
    """Scale one first-order tolerance by both participating tuple norms."""
    proposed_norm_sq = _gradient_tuple_dot(
        proposed_grads, proposed_grads, reference
    )
    constraint_norm_sq = _gradient_tuple_dot(
        constraint_grads, constraint_grads, reference
    )
    if not bool(torch.isfinite(
            proposed_norm_sq + constraint_norm_sq).item()):
        raise FloatingPointError("non-finite gradient-dot tolerance norm")
    return (
        float(tolerance_multiplier)
        * _floating_dtype_epsilon(reference)
        * torch.sqrt(
            proposed_norm_sq.clamp_min(0.0)
            * constraint_norm_sq.clamp_min(0.0)
        )
    )


def _strict_gradient_dot_floor(
        proposed_grads,
        constraint_grads,
        reference,
        floor_multiplier=512.0):
    """Return a representable strict floor instead of a symbolic tiny value.

    The additional constraint norm keeps the floor positive even when the raw
    proposed tuple is zero.  This is a numerical resolution bound, not a loss
    coefficient; real candidate and target floors are preserved separately.
    """
    proposed_norm_sq = _gradient_tuple_dot(
        proposed_grads, proposed_grads, reference
    )
    constraint_norm_sq = _gradient_tuple_dot(
        constraint_grads, constraint_grads, reference
    )
    if not bool(torch.isfinite(
            proposed_norm_sq + constraint_norm_sq).item()):
        raise FloatingPointError("non-finite strict gradient-dot floor norm")
    if not bool((constraint_norm_sq > 0.0).item()):
        raise RuntimeError("strict gradient-dot floor has zero constraint norm")
    proposed_norm = torch.sqrt(proposed_norm_sq.clamp_min(0.0))
    constraint_norm = torch.sqrt(constraint_norm_sq)
    return (
        float(floor_multiplier)
        * _floating_dtype_epsilon(reference)
        * (proposed_norm + constraint_norm)
        * constraint_norm
    )


def _nonnegative_gradient_dot_with_tolerance(
        proposed_grads,
        constraint_grads,
        reference,
        tolerance_multiplier=128.0):
    """Validate one non-increasing constraint at the tensor write precision.

    Projection operates on float32 tensors and already permits residuals at
    the norm-scaled machine precision.  Requiring an exact ``dot >= 0`` after
    writing the projected displacement back to parameters contradicts that
    contract and can reject a mathematically accepted lifecycle constraint.
    This helper uses the same scale-aware bound while still rejecting every
    direction whose violation is larger than floating-point resolution.
    """
    dot = _gradient_tuple_dot(
        proposed_grads,
        constraint_grads,
        reference,
    )
    proposed_norm_sq = _gradient_tuple_dot(
        proposed_grads,
        proposed_grads,
        reference,
    )
    constraint_norm_sq = _gradient_tuple_dot(
        constraint_grads,
        constraint_grads,
        reference,
    )
    if not bool(torch.isfinite(
            dot + proposed_norm_sq + constraint_norm_sq).item()):
        raise FloatingPointError(
            "non-finite nonnegative gradient-dot validation"
        )
    tolerance = _gradient_dot_tolerance(
        proposed_grads=proposed_grads,
        constraint_grads=constraint_grads,
        reference=reference,
        tolerance_multiplier=tolerance_multiplier,
    )
    return {
        "dot": dot,
        "tolerance": tolerance,
        "valid": dot >= -tolerance,
    }


def _pair_gradient_direction_diagnostics(
        pair_grads,
        base_factor_grads,
        reference):
    """Measure pair/base directions without modifying either gradient tuple."""
    pair_norm_sq = _gradient_tuple_dot(
        pair_grads,
        pair_grads,
        reference,
    )
    base_norm_sq = _gradient_tuple_dot(
        base_factor_grads,
        base_factor_grads,
        reference,
    )
    pair_base_dot = _gradient_tuple_dot(
        pair_grads,
        base_factor_grads,
        reference,
    )
    if not bool(torch.isfinite(
            pair_norm_sq + base_norm_sq + pair_base_dot).item()):
        raise FloatingPointError("non-finite pair/base gradient diagnostic")
    pair_norm = torch.sqrt(pair_norm_sq)
    base_norm = torch.sqrt(base_norm_sq)
    cosine = (
        pair_base_dot
        / (pair_norm * base_norm).clamp_min(1e-12)
        if bool(
            (pair_norm_sq > 0.0).item()
            and (base_norm_sq > 0.0).item()
        )
        else reference.new_tensor(0.0)
    )
    return {
        "pair_norm": pair_norm,
        "base_factor_norm": base_norm,
        "pair_base_dot": pair_base_dot,
        "pair_base_cosine": cosine,
    }


def _gradient_tuple_direction_diagnostics(
        reference_grads,
        measured_grads,
        reference):
    """Measure an executed gradient tuple against a fixed reference tuple.

    ``descent_component`` is ``g_reference dot g_measured``.  It is positive
    exactly when the first-order displacement ``-g_measured`` descends the
    reference objective.  All inputs are read-only detached tensors.
    """
    if len(reference_grads) != len(measured_grads):
        raise ValueError("gradient diagnostic tuples have different lengths")
    reference_norm_sq = _gradient_tuple_dot(
        reference_grads,
        reference_grads,
        reference,
    )
    measured_norm_sq = _gradient_tuple_dot(
        measured_grads,
        measured_grads,
        reference,
    )
    direction_dot = _gradient_tuple_dot(
        reference_grads,
        measured_grads,
        reference,
    )
    if not bool(torch.isfinite(
            reference_norm_sq + measured_norm_sq + direction_dot).item()):
        raise FloatingPointError("non-finite combined-gradient diagnostic")
    reference_norm = torch.sqrt(reference_norm_sq)
    measured_norm = torch.sqrt(measured_norm_sq)
    cosine = (
        direction_dot
        / (reference_norm * measured_norm).clamp_min(1e-12)
        if bool(
            (reference_norm_sq > 0.0).item()
            and (measured_norm_sq > 0.0).item()
        )
        else reference.new_tensor(0.0)
    )
    return {
        "reference_norm": reference_norm,
        "measured_norm": measured_norm,
        "dot": direction_dot,
        "cosine": cosine,
        "descent_component": direction_dot,
    }


def _clone_parameter_gradients(parameters):
    """Clone the optimizer-visible ``.grad`` tuple without modifying it."""
    return tuple(
        (
            parameter.grad.detach().clone()
            if parameter.grad is not None else None
        )
        for parameter in parameters
    )


def _gradient_tuple_is_finite(gradients):
    """Return whether every present tensor in a fixed-order tuple is finite."""
    return all(
        gradient is None
        or bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    )


def _validate_fixed_parameter_gradient_tuple(
        parameters,
        gradients,
        diagnostic_name):
    """Validate one read-only gradient tuple against a fixed parameter order."""
    if len(parameters) != len(gradients):
        raise RuntimeError(
            "{} gradient count does not match adjacency parameter count"
            .format(diagnostic_name)
        )
    for parameter_index, (parameter, gradient) in enumerate(zip(
            parameters, gradients)):
        if gradient is None:
            continue
        if gradient.shape != parameter.shape:
            raise RuntimeError(
                "{} gradient shape mismatch at adjacency parameter {}"
                .format(diagnostic_name, parameter_index)
            )
        if gradient.device != parameter.device:
            raise RuntimeError(
                "{} gradient device mismatch at adjacency parameter {}"
                .format(diagnostic_name, parameter_index)
            )
        if gradient.dtype != parameter.dtype:
            raise RuntimeError(
                "{} gradient dtype mismatch at adjacency parameter {}"
                .format(diagnostic_name, parameter_index)
            )
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError(
                "non-finite {} gradient at adjacency parameter {}"
                .format(diagnostic_name, parameter_index)
            )
    return True


def _sum_fixed_parameter_gradient_tuples(
        parameters,
        gradient_tuples,
        diagnostic_name):
    """Sum fixed-order tuples while treating ``None`` as an exact zero."""
    for tuple_index, gradients in enumerate(gradient_tuples):
        _validate_fixed_parameter_gradient_tuple(
            parameters,
            gradients,
            "{}_component_{}".format(diagnostic_name, tuple_index),
        )
    summed = []
    for parameter_index, parameter in enumerate(parameters):
        parameter_sum = None
        for gradients in gradient_tuples:
            gradient = gradients[parameter_index]
            if gradient is None:
                continue
            parameter_sum = (
                gradient.detach().clone()
                if parameter_sum is None
                else parameter_sum + gradient.detach()
            )
        summed.append(parameter_sum)
    result = tuple(summed)
    _validate_fixed_parameter_gradient_tuple(
        parameters,
        result,
        diagnostic_name,
    )
    return result


def _select_boundary_progress_seed_members(
        identity_group_member_indices,
        progress_member_flags,
        identity_group_extra_budgets):
    """Select only deficit-bearing representatives for a progress proposal.

    v21 summed the first member of *every* identity group.  Run126 showed that
    every multi-direction transaction also contained a zero-deficit group;
    those groups had no extra budget, but their Jacobians still rotated the
    progress-directed proposal.  This reintroduced exactly the target class
    whose large boundary allocation v15 had removed.  Per-member exact-score
    and boundary non-regression constraints remain in the hard halfspaces;
    this helper changes only which representatives define extra progress.
    """
    flags = tuple(int(value) for value in progress_member_flags)
    extra = tuple(float(value) for value in identity_group_extra_budgets)
    if len(flags) != len(extra):
        raise ValueError(
            "boundary progress flags and extra budgets differ in length"
        )
    selected = []
    excluded_zero_budget = []
    seen = set()
    for group_ordinal, raw_members in enumerate(
            identity_group_member_indices):
        members = tuple(int(value) for value in raw_members)
        if not members:
            raise RuntimeError(
                "boundary identity group {} has no members".format(
                    group_ordinal
                )
            )
        progress_members = [
            member for member in members if flags[member] == 1
        ]
        if len(progress_members) != 1:
            raise RuntimeError(
                "boundary identity group {} must have one progress member, "
                "got {}".format(group_ordinal, len(progress_members))
            )
        progress_member = int(progress_members[0])
        if progress_member in seen:
            raise RuntimeError(
                "boundary progress member appears in multiple groups"
            )
        seen.add(progress_member)
        if extra[progress_member] > 0.0:
            selected.append(progress_member)
        else:
            excluded_zero_budget.append(progress_member)
    return {
        "selected_member_ordinals": tuple(selected),
        "excluded_zero_budget_member_ordinals": tuple(
            excluded_zero_budget
        ),
    }


def _failed_boundary_limiter_ordinals(
        candidate_results,
        boundary_target_count):
    """Return every observed exact boundary member limiting failed rays.

    The nonlinear selector can contain several linear-floor variants of the
    same underlying Adam ray.  When all of them fail, their exact-forward
    trace is the only production evidence about which boundaries need a
    genuinely different tangent proposal.  Earlier code retained only the
    final/lowest-scale argmin from each ray.  That loses other simultaneously
    failed members and every limiter observed at write-resolution scales above
    one.  Preserve ray/trial/member order so the fallback remains deterministic
    and trajectory neutral.
    """
    boundary_target_count = int(boundary_target_count)
    if boundary_target_count <= 0:
        raise ValueError("boundary target count must be positive")
    selected = []
    seen = set()
    def _append(raw_ordinal):
        ordinal = int(raw_ordinal)
        if ordinal < 0 or ordinal >= boundary_target_count:
            raise RuntimeError(
                "failed boundary limiter ordinal is outside target population"
            )
        if ordinal not in seen:
            seen.add(ordinal)
            selected.append(ordinal)

    for result in candidate_results:
        info = result.get("info")
        if (
                isinstance(info, dict)
                and info.get("limiting_constraint_type") == "boundary"):
            _append(info.get("limiting_target_ordinal", -1))
        for trial in result.get("trial_trace", ()):
            if not isinstance(trial, dict):
                continue
            failed_ordinals = trial.get(
                "boundary_failed_target_ordinals", ()
            )
            for ordinal in failed_ordinals:
                _append(ordinal)
            if (
                    not failed_ordinals
                    and trial.get("limiting_constraint_type") == "boundary"):
                _append(trial.get("limiting_target_ordinal", -1))
    return tuple(selected)


def _normalized_gradient_seed_blend(
        proposed_grads,
        seed_grads,
        reference,
        seed_fraction=0.5):
    """Blend a new gradient ray at the realized proposal's norm."""
    if len(proposed_grads) != len(seed_grads):
        raise ValueError("normalized gradient seed tuple lengths differ")
    seed_fraction = float(seed_fraction)
    if (
            not np.isfinite(seed_fraction)
            or seed_fraction <= 0.0
            or seed_fraction > 1.0):
        raise ValueError("normalized gradient seed fraction must be in (0, 1]")
    proposed_norm_sq = _gradient_tuple_dot(
        proposed_grads, proposed_grads, reference
    )
    seed_norm_sq = _gradient_tuple_dot(seed_grads, seed_grads, reference)
    if not bool(
            torch.isfinite(proposed_norm_sq + seed_norm_sq).item()
            and (proposed_norm_sq > 0.0).item()
            and (seed_norm_sq > 0.0).item()):
        raise RuntimeError("normalized gradient seed has no finite direction")
    seed_scale = torch.sqrt(proposed_norm_sq / seed_norm_sq)
    blended = []
    for proposed_grad, seed_grad in zip(proposed_grads, seed_grads):
        if proposed_grad is None and seed_grad is None:
            blended.append(None)
        elif proposed_grad is None:
            blended.append(seed_fraction * seed_scale * seed_grad.detach())
        elif seed_grad is None:
            blended.append((1.0 - seed_fraction) * proposed_grad.detach())
        else:
            blended.append(
                (1.0 - seed_fraction) * proposed_grad.detach()
                + seed_fraction * seed_scale * seed_grad.detach()
            )
    return tuple(blended), {
        "seed_raw_norm": float(torch.sqrt(
            seed_norm_sq
        ).detach().cpu().item()),
        "reference_norm": float(torch.sqrt(
            proposed_norm_sq
        ).detach().cpu().item()),
        "seed_component_norm": float(torch.sqrt(
            seed_norm_sq * seed_scale * seed_scale
        ).detach().cpu().item()) * seed_fraction,
        "seed_fraction": seed_fraction,
    }


def _subtract_fixed_parameter_gradient_tuples(
        parameters,
        left,
        right,
        diagnostic_name):
    """Return ``left - right`` without collapsing unused parameter positions."""
    _validate_fixed_parameter_gradient_tuple(
        parameters,
        left,
        "{}_left".format(diagnostic_name),
    )
    _validate_fixed_parameter_gradient_tuple(
        parameters,
        right,
        "{}_right".format(diagnostic_name),
    )
    difference = []
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is None and right_gradient is None:
            difference.append(None)
        elif left_gradient is None:
            difference.append(-right_gradient.detach().clone())
        elif right_gradient is None:
            difference.append(left_gradient.detach().clone())
        else:
            difference.append(
                left_gradient.detach() - right_gradient.detach()
            )
    result = tuple(difference)
    _validate_fixed_parameter_gradient_tuple(
        parameters,
        result,
        diagnostic_name,
    )
    return result


def _gradient_reconstruction_diagnostics(
        parameters,
        reconstructed_grads,
        reference_grads,
        reference,
        diagnostic_name,
        absolute_tolerance=1e-7,
        relative_tolerance=1e-5,
        fail_loud=True):
    """Measure and optionally enforce a fixed-order gradient reconstruction."""
    delta_grads = _subtract_fixed_parameter_gradient_tuples(
        parameters,
        reconstructed_grads,
        reference_grads,
        "{}_delta".format(diagnostic_name),
    )
    reconstructed_info = _gradient_tuple_direction_diagnostics(
        reference_grads=reference_grads,
        measured_grads=reconstructed_grads,
        reference=reference,
    )
    reference_norm = reconstructed_info["reference_norm"]
    reconstructed_norm = reconstructed_info["measured_norm"]
    delta_norm_sq = _gradient_tuple_dot(
        delta_grads,
        delta_grads,
        reference,
    )
    delta_norm = torch.sqrt(delta_norm_sq)
    denominator = reference_norm.clamp_min(1e-12)
    relative_error = delta_norm / denominator
    tolerance = (
        reference.new_tensor(float(absolute_tolerance))
        + float(relative_tolerance) * reference_norm
    )
    valid = bool((delta_norm <= tolerance).item())
    if bool(fail_loud) and not valid:
        raise RuntimeError(
            "{} gradient reconstruction failed: delta_norm={}, "
            "reference_norm={}, reconstructed_norm={}, tolerance={}"
            .format(
                diagnostic_name,
                _to_float(delta_norm),
                _to_float(reference_norm),
                _to_float(reconstructed_norm),
                _to_float(tolerance),
            )
        )
    return {
        "delta_grads": delta_grads,
        "delta_norm": delta_norm,
        "relative_error": relative_error,
        "valid": reference.new_tensor(float(valid)),
        "reference_norm": reference_norm,
        "reconstructed_norm": reconstructed_norm,
    }


def _objective_gradient_decomposition_diagnostics(
        parameters,
        objective_specs,
        total_loss,
        pair_objective_name="pair",
        scalar_absolute_tolerance=1e-7,
        scalar_relative_tolerance=1e-6,
        gradient_absolute_tolerance=1e-7,
        gradient_relative_tolerance=1e-5):
    """Decompose the exact standard adjacency objective without touching ``.grad``.

    ``objective_specs`` contains ``(name, effective_loss, active)`` tuples.
    Every effective loss is the tensor exactly as it enters ``total_loss``;
    in particular, the entropy component already includes its negative sign
    and coefficient.  The helper retains the graph only for the caller's real
    backward path and never writes optimizer-visible gradients.
    """
    names = tuple(spec[0] for spec in objective_specs)
    if names != tuple(PAIR_OPTIMIZER_OBJECTIVE_NAMES):
        raise RuntimeError(
            "adjacency objective diagnostic schema mismatch: {}"
            .format(names)
        )
    if pair_objective_name not in names:
        raise RuntimeError("pair objective is absent from decomposition")
    if total_loss.numel() != 1:
        raise RuntimeError("total adjacency loss must be scalar")

    objective_loss_sum = total_loss.new_tensor(0.0)
    objective_gradients = {}
    objective_diagnostics = {}
    for objective_name, objective_loss, active in objective_specs:
        if objective_loss.numel() != 1:
            raise RuntimeError(
                "{} adjacency objective must be scalar".format(
                    objective_name
                )
            )
        if not bool(torch.isfinite(objective_loss).item()):
            raise FloatingPointError(
                "non-finite {} adjacency objective".format(objective_name)
            )
        objective_loss_sum = objective_loss_sum + objective_loss
        if bool(active):
            if not objective_loss.requires_grad:
                raise RuntimeError(
                    "active {} adjacency objective has no autograd graph"
                    .format(objective_name)
                )
            gradients = torch.autograd.grad(
                objective_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            gradients = tuple(None for _ in parameters)
        _validate_fixed_parameter_gradient_tuple(
            parameters,
            gradients,
            objective_name,
        )
        objective_gradients[objective_name] = gradients
        objective_diagnostics[objective_name] = {
            "active": total_loss.new_tensor(float(bool(active))),
            "scalar_loss": objective_loss.detach(),
        }

    scalar_error = (objective_loss_sum - total_loss).abs()
    scalar_tolerance = (
        total_loss.new_tensor(float(scalar_absolute_tolerance))
        + float(scalar_relative_tolerance) * total_loss.detach().abs()
    )
    scalar_valid = bool((scalar_error <= scalar_tolerance).item())
    if not scalar_valid:
        raise RuntimeError(
            "adjacency objective scalar reconstruction failed: error={}, "
            "tolerance={}".format(
                _to_float(scalar_error),
                _to_float(scalar_tolerance),
            )
        )

    raw_combined_grads = torch.autograd.grad(
        total_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    _validate_fixed_parameter_gradient_tuple(
        parameters,
        raw_combined_grads,
        "raw_total_objective",
    )
    independent_sum_grads = _sum_fixed_parameter_gradient_tuples(
        parameters,
        [
            objective_gradients[objective_name]
            for objective_name in PAIR_OPTIMIZER_OBJECTIVE_NAMES
        ],
        "independent_objective_sum",
    )
    reconstruction = _gradient_reconstruction_diagnostics(
        parameters=parameters,
        reconstructed_grads=independent_sum_grads,
        reference_grads=raw_combined_grads,
        reference=total_loss,
        diagnostic_name="independent_objective_sum",
        absolute_tolerance=gradient_absolute_tolerance,
        relative_tolerance=gradient_relative_tolerance,
        fail_loud=True,
    )

    pair_grads = objective_gradients[pair_objective_name]
    for objective_name in PAIR_OPTIMIZER_OBJECTIVE_NAMES:
        direction = _gradient_tuple_direction_diagnostics(
            reference_grads=pair_grads,
            measured_grads=objective_gradients[objective_name],
            reference=total_loss,
        )
        objective_diagnostics[objective_name].update({
            "grad_norm": direction["measured_norm"],
            "pair_dot": direction["dot"],
            "pair_cosine": direction["cosine"],
            "pair_descent_component": direction["descent_component"],
            "grads": objective_gradients[objective_name],
        })
    independent_sum_direction = _gradient_tuple_direction_diagnostics(
        reference_grads=pair_grads,
        measured_grads=independent_sum_grads,
        reference=total_loss,
    )
    raw_combined_direction = _gradient_tuple_direction_diagnostics(
        reference_grads=pair_grads,
        measured_grads=raw_combined_grads,
        reference=total_loss,
    )
    return {
        "objectives": objective_diagnostics,
        "objective_scalar_reconstruction_error": scalar_error,
        "objective_scalar_reconstruction_valid": total_loss.new_tensor(
            float(scalar_valid)
        ),
        "independent_sum_grads": independent_sum_grads,
        "independent_sum_norm": independent_sum_direction["measured_norm"],
        "pair_independent_sum_dot": independent_sum_direction["dot"],
        "pair_independent_sum_cosine": independent_sum_direction["cosine"],
        "raw_combined_grads": raw_combined_grads,
        "raw_combined_grad_norm": raw_combined_direction["measured_norm"],
        "pair_raw_combined_dot": raw_combined_direction["dot"],
        "pair_raw_combined_cosine": raw_combined_direction["cosine"],
        "independent_sum_vs_raw_delta_norm": reconstruction["delta_norm"],
        "independent_sum_vs_raw_relative_error": reconstruction[
            "relative_error"
        ],
        "independent_sum_reconstruction_valid": reconstruction["valid"],
    }


def _gradient_projection_delta_diagnostics(
        parameters,
        pre_projection_grads,
        post_projection_grads,
        pair_grads,
        reference):
    """Measure the exact gradient-stage change before clipping."""
    projection_delta = _subtract_fixed_parameter_gradient_tuples(
        parameters,
        post_projection_grads,
        pre_projection_grads,
        "gradient_projection_delta",
    )
    pre_direction = _gradient_tuple_direction_diagnostics(
        reference_grads=pair_grads,
        measured_grads=pre_projection_grads,
        reference=reference,
    )
    post_direction = _gradient_tuple_direction_diagnostics(
        reference_grads=pair_grads,
        measured_grads=post_projection_grads,
        reference=reference,
    )
    delta_direction = _gradient_tuple_direction_diagnostics(
        reference_grads=pair_grads,
        measured_grads=projection_delta,
        reference=reference,
    )
    reconstruction = _sum_fixed_parameter_gradient_tuples(
        parameters,
        [pre_projection_grads, projection_delta],
        "projection_reconstructed_post",
    )
    _gradient_reconstruction_diagnostics(
        parameters=parameters,
        reconstructed_grads=reconstruction,
        reference_grads=post_projection_grads,
        reference=reference,
        diagnostic_name="projection_delta",
        fail_loud=True,
    )
    return {
        "delta_grads": projection_delta,
        "delta_norm": delta_direction["measured_norm"],
        "pair_delta_dot": delta_direction["dot"],
        "pair_delta_cosine": delta_direction["cosine"],
        "pre_projection_norm": pre_direction["measured_norm"],
        "post_projection_norm": post_direction["measured_norm"],
        "pair_pre_projection_dot": pre_direction["dot"],
        "pair_pre_projection_cosine": pre_direction["cosine"],
        "pair_post_projection_dot": post_direction["dot"],
        "pair_post_projection_cosine": post_direction["cosine"],
    }


def _optimizer_parameter_group_hyperparameters(optimizer, parameters):
    """Read one scalar Adam configuration for the actual parameter collection."""
    parameter_ids = set(id(parameter) for parameter in parameters)
    seen_parameter_ids = set()
    signatures = []
    for group in optimizer.param_groups:
        group_parameter_ids = set(
            id(parameter)
            for parameter in group["params"]
            if id(parameter) in parameter_ids
        )
        duplicate_ids = seen_parameter_ids.intersection(group_parameter_ids)
        if duplicate_ids:
            raise RuntimeError(
                "adjacency parameter appears in multiple Adam groups"
            )
        seen_parameter_ids.update(group_parameter_ids)
        if not group_parameter_ids:
            continue
        beta1, beta2 = group["betas"]
        signatures.append((
            float(group["lr"]),
            float(beta1),
            float(beta2),
            float(group["eps"]),
            float(group.get("weight_decay", 0.0)),
            float(bool(group.get("amsgrad", False))),
        ))
    if seen_parameter_ids != parameter_ids:
        raise RuntimeError(
            "standard adjacency Adam does not own every adjacency parameter"
        )
    if not signatures:
        raise RuntimeError("standard adjacency Adam has no parameter group")
    first_signature = signatures[0]
    if any(signature != first_signature for signature in signatures[1:]):
        raise RuntimeError(
            "per-transaction CSV cannot represent heterogeneous Adam groups"
        )
    return {
        "learning_rate": first_signature[0],
        "beta1": first_signature[1],
        "beta2": first_signature[2],
        "eps": first_signature[3],
        "weight_decay": first_signature[4],
        "amsgrad": first_signature[5],
    }


def _adam_state_before_step_diagnostics(
        optimizer,
        parameters,
        pair_grads,
        reference):
    """Read Adam moments and per-parameter step counters without mutation."""
    if len(parameters) != len(pair_grads):
        raise ValueError("Adam state and pair gradient lengths differ")
    hyperparameters = _optimizer_parameter_group_hyperparameters(
        optimizer,
        parameters,
    )
    exp_avg_norm_sq = reference.new_tensor(0.0)
    exp_avg_sq_norm_sq = reference.new_tensor(0.0)
    exp_avg_sq_sum = reference.new_tensor(0.0)
    exp_avg_pair_dot = reference.new_tensor(0.0)
    pair_norm_sq = _gradient_tuple_dot(
        pair_grads,
        pair_grads,
        reference,
    )
    parameter_steps = []
    for parameter, pair_grad in zip(parameters, pair_grads):
        if parameter.grad is None:
            parameter_steps.append(None)
            continue
        state = optimizer.state.get(parameter, {})
        parameter_steps.append(int(_to_float(state.get("step", 0.0))))
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is not None:
            detached_exp_avg = exp_avg.detach()
            exp_avg_norm_sq = (
                exp_avg_norm_sq + detached_exp_avg.pow(2).sum()
            )
            if pair_grad is not None:
                exp_avg_pair_dot = (
                    exp_avg_pair_dot
                    + (detached_exp_avg * pair_grad).sum()
                )
        if exp_avg_sq is not None:
            detached_exp_avg_sq = exp_avg_sq.detach()
            if bool((detached_exp_avg_sq < 0.0).any().item()):
                raise FloatingPointError("Adam exp_avg_sq is negative")
            exp_avg_sq_norm_sq = (
                exp_avg_sq_norm_sq + detached_exp_avg_sq.pow(2).sum()
            )
            exp_avg_sq_sum = exp_avg_sq_sum + detached_exp_avg_sq.sum()
    active_steps = [
        step for step in parameter_steps
        if step is not None
    ]
    if not active_steps:
        raise RuntimeError(
            "standard adjacency transaction has no optimizer-visible gradient"
        )
    finite_total = (
        exp_avg_norm_sq
        + exp_avg_sq_norm_sq
        + exp_avg_sq_sum
        + exp_avg_pair_dot
        + pair_norm_sq
    )
    if not bool(torch.isfinite(finite_total).item()):
        raise FloatingPointError("non-finite Adam state diagnostic")
    exp_avg_norm = torch.sqrt(exp_avg_norm_sq)
    pair_norm = torch.sqrt(pair_norm_sq)
    exp_avg_pair_cosine = (
        exp_avg_pair_dot
        / (exp_avg_norm * pair_norm).clamp_min(1e-12)
        if bool(
            (exp_avg_norm_sq > 0.0).item()
            and (pair_norm_sq > 0.0).item()
        )
        else reference.new_tensor(0.0)
    )
    return {
        "exp_avg_norm": exp_avg_norm,
        "exp_avg_sq_norm": torch.sqrt(exp_avg_sq_norm_sq),
        "exp_avg_sq_sqrt_sum": torch.sqrt(exp_avg_sq_sum),
        "exp_avg_pair_dot": exp_avg_pair_dot,
        "exp_avg_pair_cosine": exp_avg_pair_cosine,
        "optimizer_step_before": float(max(active_steps)),
        "optimizer_step_before_min": float(min(active_steps)),
        "optimizer_step_before_max": float(max(active_steps)),
        "parameter_steps_before": tuple(parameter_steps),
        "learning_rate": hyperparameters["learning_rate"],
        "beta1": hyperparameters["beta1"],
        "beta2": hyperparameters["beta2"],
        "eps": hyperparameters["eps"],
        "weight_decay": hyperparameters["weight_decay"],
        "amsgrad": hyperparameters["amsgrad"],
    }


def _validate_adam_step_increment(
        optimizer,
        parameters,
        parameter_steps_before):
    """Require exactly one Adam step for every participating parameter."""
    if len(parameters) != len(parameter_steps_before):
        raise ValueError("Adam step diagnostic lengths differ")
    steps_after = []
    for parameter, step_before in zip(parameters, parameter_steps_before):
        if step_before is None:
            continue
        state = optimizer.state.get(parameter, {})
        step_after = int(_to_float(state.get("step", -1.0)))
        if step_after != int(step_before) + 1:
            raise RuntimeError(
                "standard adjacency Adam parameter step did not increase by "
                "exactly one: before={}, after={}".format(
                    step_before,
                    step_after,
                )
            )
        steps_after.append(step_after)
    if not steps_after:
        raise RuntimeError("standard adjacency Adam advanced no parameter")
    return {
        "optimizer_step_after": float(max(steps_after)),
        "optimizer_step_after_min": float(min(steps_after)),
        "optimizer_step_after_max": float(max(steps_after)),
    }


def _parameter_displacement_direction_diagnostics(
        parameters,
        parameter_before_step,
        pair_grads,
        reference):
    """Measure a real parameter displacement over the complete parameter set."""
    if not (
            len(parameters)
            == len(parameter_before_step)
            == len(pair_grads)):
        raise ValueError("parameter displacement diagnostic lengths differ")
    displacement_norm_sq = reference.new_tensor(0.0)
    displacement_pair_dot = reference.new_tensor(0.0)
    displacements = []
    for parameter, before, pair_grad in zip(
            parameters,
            parameter_before_step,
            pair_grads):
        if before is None:
            raise RuntimeError(
                "standard Adam displacement is missing a parameter snapshot"
            )
        displacement = parameter.detach() - before
        displacements.append(displacement.detach().clone())
        displacement_norm_sq = (
            displacement_norm_sq + displacement.pow(2).sum()
        )
        if pair_grad is not None:
            displacement_pair_dot = (
                displacement_pair_dot + (displacement * pair_grad).sum()
            )
    pair_norm_sq = _gradient_tuple_dot(
        pair_grads,
        pair_grads,
        reference,
    )
    finite_total = (
        displacement_norm_sq + displacement_pair_dot + pair_norm_sq
    )
    if not bool(torch.isfinite(finite_total).item()):
        raise FloatingPointError("non-finite parameter displacement diagnostic")
    displacement_norm = torch.sqrt(displacement_norm_sq)
    pair_norm = torch.sqrt(pair_norm_sq)
    descent_dot = -displacement_pair_dot
    descent_cosine = (
        descent_dot
        / (displacement_norm * pair_norm).clamp_min(1e-12)
        if bool(
            (displacement_norm_sq > 0.0).item()
            and (pair_norm_sq > 0.0).item()
        )
        else reference.new_tensor(0.0)
    )
    return {
        "displacement_norm": displacement_norm,
        "pair_dot": displacement_pair_dot,
        "pair_descent_dot": descent_dot,
        "pair_descent_cosine": descent_cosine,
        "displacements": tuple(displacements),
    }


def _restore_scaled_transaction_parameter_state(
        parameters,
        parameter_before_step,
        parameter_displacements,
        scale,
        require_complete):
    """Restore one exact-search ray from its real transaction origin."""
    if not (
            len(parameters)
            == len(parameter_before_step)
            == len(parameter_displacements)):
        raise ValueError("exact transaction replay lengths differ")
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("exact transaction replay scale is non-finite")
    with torch.no_grad():
        for parameter, before, displacement in zip(
                parameters,
                parameter_before_step,
                parameter_displacements):
            if before is None and displacement is None:
                if require_complete:
                    raise RuntimeError(
                        "standard exact replay received a sparse transaction "
                        "origin"
                    )
                continue
            if before is None or displacement is None:
                raise RuntimeError(
                    "exact transaction replay origin and displacement masks "
                    "differ"
                )
            # Scale zero is the transaction origin, not an arithmetic point on
            # the ray.  Copy it bit-for-bit: ``0 * displacement`` would allow a
            # non-finite trial direction to contaminate the origin and makes the
            # strongest preservation invariant depend on direction contents.
            if scale == 0.0:
                parameter.copy_(before)
            else:
                parameter.copy_(before + scale * displacement)


def _displacement_delta_diagnostics(
        parameters,
        parameter_before_step,
        raw_displacements,
        reference):
    """Compare final committed parameters with the raw Adam displacement."""
    if not (
            len(parameters)
            == len(parameter_before_step)
            == len(raw_displacements)):
        raise ValueError("raw/final displacement diagnostic lengths differ")
    delta_norm_sq = reference.new_tensor(0.0)
    exact_equal = True
    for parameter, before, raw_displacement in zip(
            parameters,
            parameter_before_step,
            raw_displacements):
        final_displacement = parameter.detach() - before
        difference = final_displacement - raw_displacement
        delta_norm_sq = delta_norm_sq + difference.pow(2).sum()
        exact_equal = exact_equal and bool(torch.equal(
            final_displacement,
            raw_displacement,
        ))
    if not bool(torch.isfinite(delta_norm_sq).item()):
        raise FloatingPointError("non-finite Adam-to-final displacement delta")
    return {
        "delta_norm": torch.sqrt(delta_norm_sq),
        "exact_equal": reference.new_tensor(float(exact_equal)),
    }


def _pair_realized_displacement_diagnostics(
        parameters,
        parameter_before_step,
        pair_grads,
        reference):
    """Measure the final committed parameter displacement along pair descent."""
    displacement_norm_sq = reference.new_tensor(0.0)
    displacement_pair_dot = reference.new_tensor(0.0)
    for parameter, before, pair_grad in zip(
            parameters,
            parameter_before_step,
            pair_grads):
        if before is None or pair_grad is None:
            continue
        displacement = parameter.detach() - before
        displacement_norm_sq = (
            displacement_norm_sq + displacement.pow(2).sum()
        )
        displacement_pair_dot = (
            displacement_pair_dot + (displacement * pair_grad).sum()
        )
    pair_norm_sq = _gradient_tuple_dot(
        pair_grads,
        pair_grads,
        reference,
    )
    if not bool(torch.isfinite(
            displacement_norm_sq
            + displacement_pair_dot
            + pair_norm_sq).item()):
        raise FloatingPointError(
            "non-finite pair realized-displacement diagnostic"
        )
    displacement_norm = torch.sqrt(displacement_norm_sq)
    pair_norm = torch.sqrt(pair_norm_sq)
    descent_dot = -displacement_pair_dot
    descent_cosine = (
        descent_dot
        / (displacement_norm * pair_norm).clamp_min(1e-12)
        if bool(
            (displacement_norm_sq > 0.0).item()
            and (pair_norm_sq > 0.0).item()
        )
        else reference.new_tensor(0.0)
    )
    return {
        "update_norm": displacement_norm,
        "descent_dot": descent_dot,
        "descent_cosine": descent_cosine,
    }


def _validate_exact_score_join_fields(pre_join_fields, post_join_fields):
    """Require the exact transition/identity/mask join across an optimizer step."""
    if set(pre_join_fields.keys()) != set(post_join_fields.keys()):
        raise RuntimeError("pair score join field names differ")
    for field_name in sorted(pre_join_fields.keys()):
        pre_value = pre_join_fields[field_name]
        post_value = post_join_fields[field_name]
        if not torch.is_tensor(pre_value) or not torch.is_tensor(post_value):
            raise TypeError(
                "pair score join field {} is not a tensor".format(field_name)
            )
        if (
                pre_value.shape != post_value.shape
                or pre_value.dtype != post_value.dtype
                or pre_value.device != post_value.device
                or not bool(torch.equal(pre_value, post_value))):
            raise RuntimeError(
                "pair score before/after join changed field {}".format(
                    field_name
                )
            )
    return True


def _pair_target_score_change_diagnostics(
        pre_factor_logp,
        post_factor_logp,
        pair_local_delta,
        zero_tolerance=1e-12):
    """Join exact pair targets to their optimizer-before/after log-probability."""
    if (
            pre_factor_logp.shape != post_factor_logp.shape
            or pre_factor_logp.shape != pair_local_delta.shape):
        raise ValueError(
            "pair score diagnostic shapes differ: {}, {}, {}".format(
                tuple(pre_factor_logp.shape),
                tuple(post_factor_logp.shape),
                tuple(pair_local_delta.shape),
            )
        )
    if not bool(torch.isfinite(
            pre_factor_logp
            + post_factor_logp
            + pair_local_delta).all().item()):
        raise FloatingPointError("non-finite pair target score diagnostic")
    target_mask = pair_local_delta != 0.0
    target_count = target_mask.float().sum()
    if not bool((target_count > 0.0).item()):
        raise RuntimeError("pair score diagnostic has no exact target")
    score_change = post_factor_logp - pre_factor_logp
    positive_mask = (pair_local_delta > 0.0).float()
    negative_mask = (pair_local_delta < 0.0).float()
    positive_count = positive_mask.sum()
    negative_count = negative_mask.sum()
    signed_score_change = (
        score_change * pair_local_delta.sign()
    )
    zero_tolerance = float(zero_tolerance)
    if zero_tolerance < 0.0:
        raise ValueError("pair score zero tolerance must be nonnegative")
    correct_mask = target_mask & (signed_score_change > zero_tolerance)
    reverse_mask = target_mask & (signed_score_change < -zero_tolerance)
    zero_mask = target_mask & ~(correct_mask | reverse_mask)
    return {
        "target_count": target_count,
        "positive_target_count": positive_count,
        "negative_target_count": negative_count,
        "signed_change_mean": (
            signed_score_change * target_mask.float()
        ).sum() / target_count,
        "positive_change_mean": (
            score_change * positive_mask
        ).sum() / positive_count.clamp_min(1.0),
        "negative_signed_change_mean": (
            -score_change * negative_mask
        ).sum() / negative_count.clamp_min(1.0),
        "correct_direction_count": correct_mask.float().sum(),
        "reverse_direction_count": reverse_mask.float().sum(),
        "approximately_zero_count": zero_mask.float().sum(),
        "before_after_join_valid": target_count.new_tensor(1.0),
        "zero_tolerance": target_count.new_tensor(zero_tolerance),
    }


def _pair_target_score_nonregression_postcondition(score_info):
    """Accept a complete exact-score population unless a member regresses.

    Selected-factor boundaries carry the strict positive progress contract.
    Exact selected scores instead use the documented dtype-scaled preservation
    contract because a legal parameter displacement can quantize their
    float32 writeback to zero.  Keep this producer postcondition identical to
    the transaction CSV consumer: correct plus approximately-zero must
    partition the population and no member may reverse beyond tolerance.
    """
    required = (
        "target_count",
        "correct_direction_count",
        "reverse_direction_count",
        "approximately_zero_count",
    )
    missing = [key for key in required if key not in score_info]
    if missing:
        raise KeyError(
            "pair exact-score postcondition fields are missing: {}".format(
                ",".join(missing)
            )
        )
    target_count = score_info["target_count"]
    correct_count = score_info["correct_direction_count"]
    reverse_count = score_info["reverse_direction_count"]
    zero_count = score_info["approximately_zero_count"]
    classified_count = correct_count + reverse_count + zero_count
    return bool((
        (target_count > 0.0)
        & (reverse_count == 0.0)
        & (classified_count == target_count)
        & ((correct_count + zero_count) == target_count)
    ).item())


def _pair_selection_boundary_change_diagnostics(
        pre_boundary,
        post_boundary,
        pre_target_logp,
        post_target_logp,
        pre_competitor_logp,
        post_competitor_logp,
        pre_rank,
        post_rank,
        pre_selected_index,
        post_selected_index,
        pre_competitor_index,
        post_competitor_index,
        pre_valid,
        post_valid,
        pre_forced,
        post_forced,
        pair_local_delta,
        num_agents,
        highest_orders,
        linearized_required_improvement=None,
        linearized_crossing_affordable=None,
        linearized_allocation_info=None,
        zero_tolerance=1e-12):
    """Validate and trace strict pair targets at the real selection boundary."""
    tensors = (
        pre_boundary,
        post_boundary,
        pre_target_logp,
        post_target_logp,
        pre_competitor_logp,
        post_competitor_logp,
        pre_rank,
        post_rank,
        pre_selected_index,
        post_selected_index,
        pre_competitor_index,
        post_competitor_index,
        pre_valid,
        post_valid,
        pre_forced,
        post_forced,
        pair_local_delta,
    )
    expected_shape = pair_local_delta.shape
    for tensor in tensors:
        if tensor.shape != expected_shape:
            raise ValueError(
                "pair selection-boundary shape {} does not match {}".format(
                    tuple(tensor.shape), tuple(expected_shape)
                )
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise FloatingPointError(
                "non-finite pair selection-boundary diagnostic"
            )
    target_locations = torch.nonzero(
        pair_local_delta.detach() != 0.0
    ).detach().cpu().tolist()
    if not target_locations:
        raise RuntimeError(
            "pair selection-boundary diagnostic has no exact target"
        )
    target_count = len(target_locations)
    if linearized_required_improvement is None:
        linearized_required_improvement = pair_local_delta.new_zeros(
            (target_count,)
        )
    if linearized_crossing_affordable is None:
        linearized_crossing_affordable = pair_local_delta.new_zeros(
            (target_count,)
        )
    if (
            int(linearized_required_improvement.numel()) != target_count
            or int(linearized_crossing_affordable.numel()) != target_count):
        raise ValueError(
            "pair boundary allocation diagnostic population differs"
        )
    if not bool(torch.isfinite(
            linearized_required_improvement
            + linearized_crossing_affordable).all().item()):
        raise FloatingPointError(
            "non-finite pair boundary allocation diagnostic"
        )
    if linearized_allocation_info is None:
        linearized_allocation_info = {
            "base_budget": 0.0,
            "strict_budget": 0.0,
            "allocated_budget": 0.0,
            "zero_deficit_reclaimed_budget": 0.0,
            "deficit_target_count": 0.0,
            "affordable_crossing_count": 0.0,
            "budget_tolerance": 0.0,
            "identity_group_count": 0.0,
            "multi_exposure_identity_group_count": 0.0,
            "target_strict_floors": tuple(0.0 for _ in range(target_count)),
            "target_waterfill_allocations": tuple(
                0.0 for _ in range(target_count)
            ),
            "identity_group_ordinals": tuple(-1 for _ in range(target_count)),
            "identity_group_exposure_counts": tuple(
                1 for _ in range(target_count)
            ),
            "identity_group_allocated_budgets": tuple(
                0.0 for _ in range(target_count)
            ),
            "identity_group_strict_budgets": tuple(
                0.0 for _ in range(target_count)
            ),
            "identity_group_extra_budgets": tuple(
                0.0 for _ in range(target_count)
            ),
            "identity_group_progress_member_ordinals": tuple(
                -1 for _ in range(target_count)
            ),
            "identity_group_progress_member_flags": tuple(
                0 for _ in range(target_count)
            ),
            "identity_group_progress_required": tuple(
                0.0 for _ in range(target_count)
            ),
            "selected_progress_floor_fraction": 1.0,
            "progress_min_completion": 0.0,
            "progress_mean_completion": 0.0,
            "limiting_constraint_code": 0.0,
            "limiting_target_ordinal": -1.0,
            "budget_conservation_valid": 0.0,
        }
    required_allocation_fields = (
        "base_budget",
        "strict_budget",
        "allocated_budget",
        "zero_deficit_reclaimed_budget",
        "deficit_target_count",
        "affordable_crossing_count",
        "budget_tolerance",
        "identity_group_count",
        "multi_exposure_identity_group_count",
        "budget_conservation_valid",
    )
    for field in required_allocation_fields:
        if field not in linearized_allocation_info:
            raise KeyError(
                "missing pair boundary allocation diagnostic {}".format(
                    field
                )
            )
        value = float(linearized_allocation_info[field])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "invalid pair boundary allocation diagnostic {}".format(
                    field
                )
            )
    required_target_allocation_fields = (
        "target_strict_floors",
        "target_waterfill_allocations",
        "identity_group_ordinals",
        "identity_group_exposure_counts",
        "identity_group_allocated_budgets",
        "identity_group_strict_budgets",
        "identity_group_extra_budgets",
        "identity_group_progress_member_ordinals",
        "identity_group_progress_member_flags",
        "identity_group_progress_required",
    )
    for field in required_target_allocation_fields:
        if field not in linearized_allocation_info:
            raise KeyError(
                "missing pair boundary target allocation diagnostic {}".format(
                    field
                )
            )
        values = tuple(linearized_allocation_info[field])
        if len(values) != target_count:
            raise ValueError(
                "pair boundary target allocation population differs: {}"
                .format(field)
            )
        if not all(np.isfinite(float(value)) for value in values):
            raise FloatingPointError(
                "non-finite pair boundary target allocation {}".format(field)
            )
    for field in (
            "selected_progress_floor_fraction",
            "progress_min_completion",
            "progress_mean_completion",
            "limiting_constraint_code",
            "limiting_target_ordinal"):
        if field not in linearized_allocation_info:
            raise KeyError(
                "missing pair boundary nonlinear diagnostic {}".format(field)
            )
        if not np.isfinite(float(linearized_allocation_info[field])):
            raise FloatingPointError(
                "non-finite pair boundary nonlinear diagnostic {}".format(
                    field
                )
            )
    zero_tolerance = float(zero_tolerance)
    rows = []
    signed_changes = []
    for target_ordinal, raw_location in enumerate(target_locations):
        transition_index = int(raw_location[0])
        factor_index = int(raw_location[1])
        location = (transition_index, factor_index)
        target_delta = float(
            pair_local_delta[location].detach().cpu().item()
        )
        target_sign = 1.0 if target_delta > 0.0 else -1.0
        if not bool(
                (pre_valid[location] > 0.0).item()
                and (post_valid[location] > 0.0).item()):
            raise RuntimeError(
                "actionable pair target has no real selection competitor: "
                "index={}".format(location)
            )
        if bool(
                (pre_forced[location] > 0.0).item()
                or (post_forced[location] > 0.0).item()):
            raise RuntimeError(
                "forced pair target entered selection-boundary supervision: "
                "index={}".format(location)
            )
        selected_pre = int(round(float(
            pre_selected_index[location].detach().cpu().item()
        )))
        selected_post = int(round(float(
            post_selected_index[location].detach().cpu().item()
        )))
        if selected_pre != selected_post or selected_pre < 0:
            raise RuntimeError(
                "pair target canonical identity changed inside transaction"
            )
        competitor_pre = int(round(float(
            pre_competitor_index[location].detach().cpu().item()
        )))
        competitor_post = int(round(float(
            post_competitor_index[location].detach().cpu().item()
        )))
        if competitor_pre < 0 or competitor_post < 0:
            raise RuntimeError(
                "actionable pair target competitor identity is missing"
            )
        pre_value = float(pre_boundary[location].detach().cpu().item())
        post_value = float(post_boundary[location].detach().cpu().item())
        pre_signed = target_sign * pre_value
        post_signed = target_sign * post_value
        signed_change = post_signed - pre_signed
        pre_deficit = max(-pre_signed, 0.0)
        post_deficit = max(-post_signed, 0.0)
        deficit_reduction = pre_deficit - post_deficit
        signed_changes.append(
            pair_local_delta.new_tensor(signed_change)
        )
        pre_rank_value = int(round(float(
            pre_rank[location].detach().cpu().item()
        )))
        post_rank_value = int(round(float(
            post_rank[location].detach().cpu().item()
        )))
        if pre_rank_value <= 0 or post_rank_value <= 0:
            raise RuntimeError("pair target boundary rank is not positive")
        _, target_identity = _canonical_candidate_identity(
            candidate_index=selected_pre,
            num_agents=num_agents,
            highest_orders=highest_orders,
        )
        _, pre_competitor_identity = _canonical_candidate_identity(
            candidate_index=competitor_pre,
            num_agents=num_agents,
            highest_orders=highest_orders,
        )
        _, post_competitor_identity = _canonical_candidate_identity(
            candidate_index=competitor_post,
            num_agents=num_agents,
            highest_orders=highest_orders,
        )
        pre_active = int(pre_rank_value == 1)
        post_active = int(post_rank_value == 1)
        positive_promotion = int(
            target_sign > 0.0 and pre_active == 0 and post_active == 1
        )
        negative_eviction = int(
            target_sign < 0.0 and pre_active == 1 and post_active == 0
        )
        rows.append({
            "diagnostic_version": int(
                PAIR_SELECTION_BOUNDARY_DIAGNOSTIC_VERSION
            ),
            "target_row_sequence_within_transaction": int(len(rows)),
            "transition_index_in_partition": transition_index,
            "factor_index": factor_index,
            "target_candidate_index": selected_pre,
            "target_canonical_identity": target_identity,
            "target_sign": target_sign,
            "target_weight": abs(target_delta),
            "pre_competitor_candidate_index": competitor_pre,
            "pre_competitor_canonical_identity": pre_competitor_identity,
            "post_competitor_candidate_index": competitor_post,
            "post_competitor_canonical_identity": post_competitor_identity,
            "pre_margin": pre_value,
            "post_margin": post_value,
            "pre_target_logp": float(
                pre_target_logp[location].detach().cpu().item()
            ),
            "post_target_logp": float(
                post_target_logp[location].detach().cpu().item()
            ),
            "pre_competitor_logp": float(
                pre_competitor_logp[location].detach().cpu().item()
            ),
            "post_competitor_logp": float(
                post_competitor_logp[location].detach().cpu().item()
            ),
            "pre_signed_margin": pre_signed,
            "post_signed_margin": post_signed,
            "signed_margin_change": signed_change,
            "margin_direction_correct": int(
                signed_change > zero_tolerance
            ),
            "margin_direction_reverse": int(
                signed_change < -zero_tolerance
            ),
            "margin_direction_zero": int(
                abs(signed_change) <= zero_tolerance
            ),
            "pre_rank": pre_rank_value,
            "post_rank": post_rank_value,
            "signed_rank_improvement": (
                target_sign
                * float(pre_rank_value - post_rank_value)
            ),
            "pre_boundary_deficit": pre_deficit,
            "post_boundary_deficit": post_deficit,
            "boundary_deficit_reduction": deficit_reduction,
            "boundary_deficit_reduction_fraction": (
                deficit_reduction / pre_deficit
                if pre_deficit > 0.0 else 0.0
            ),
            "boundary_deficit_reduction_fraction_valid": int(
                pre_deficit > 0.0
            ),
            "linearized_required_margin_improvement": float(
                linearized_required_improvement[target_ordinal]
                .detach().cpu().item()
            ),
            "linearized_crossing_affordable": int(
                linearized_crossing_affordable[target_ordinal]
                .detach().cpu().item() > 0.5
            ),
            "linearized_original_boundary_budget": float(
                linearized_allocation_info["base_budget"]
            ),
            "linearized_strict_floor_budget": float(
                linearized_allocation_info["strict_budget"]
            ),
            "linearized_allocated_boundary_budget": float(
                linearized_allocation_info["allocated_budget"]
            ),
            "linearized_zero_deficit_reclaimed_budget": float(
                linearized_allocation_info[
                    "zero_deficit_reclaimed_budget"
                ]
            ),
            "linearized_deficit_target_count": int(round(float(
                linearized_allocation_info["deficit_target_count"]
            ))),
            "linearized_affordable_crossing_count": int(round(float(
                linearized_allocation_info["affordable_crossing_count"]
            ))),
            "linearized_budget_conservation_valid": int(round(float(
                linearized_allocation_info["budget_conservation_valid"]
            ))),
            "linearized_budget_tolerance": float(
                linearized_allocation_info["budget_tolerance"]
            ),
            "linearized_target_strict_floor": float(
                linearized_allocation_info["target_strict_floors"][
                    target_ordinal
                ]
            ),
            "linearized_target_waterfill_allocation": float(
                linearized_allocation_info["target_waterfill_allocations"][
                    target_ordinal
                ]
            ),
            "identity_group_ordinal": int(round(float(
                linearized_allocation_info["identity_group_ordinals"][
                    target_ordinal
                ]
            ))),
            "identity_group_exposure_count": int(round(float(
                linearized_allocation_info["identity_group_exposure_counts"][
                    target_ordinal
                ]
            ))),
            "identity_group_strict_budget": float(
                linearized_allocation_info["identity_group_strict_budgets"][
                    target_ordinal
                ]
            ),
            "identity_group_allocated_budget": float(
                linearized_allocation_info[
                    "identity_group_allocated_budgets"
                ][target_ordinal]
            ),
            "identity_group_extra_budget": float(
                linearized_allocation_info[
                    "identity_group_extra_budgets"
                ][target_ordinal]
            ),
            "identity_group_progress_member_ordinal": int(round(float(
                linearized_allocation_info[
                    "identity_group_progress_member_ordinals"
                ][target_ordinal]
            ))),
            "identity_group_progress_member": int(round(float(
                linearized_allocation_info[
                    "identity_group_progress_member_flags"
                ][target_ordinal]
            ))),
            "identity_group_progress_required": float(
                linearized_allocation_info[
                    "identity_group_progress_required"
                ][target_ordinal]
            ),
            "identity_group_actual_signed_margin_change": 0.0,
            "identity_group_actual_completion_ratio": 0.0,
            "identity_group_worst_member_signed_margin_change": 0.0,
            "selected_progress_floor_fraction": float(
                linearized_allocation_info[
                    "selected_progress_floor_fraction"
                ]
            ),
            "progress_min_completion": float(
                linearized_allocation_info["progress_min_completion"]
            ),
            "progress_mean_completion": float(
                linearized_allocation_info["progress_mean_completion"]
            ),
            "limiting_constraint_code": float(
                linearized_allocation_info["limiting_constraint_code"]
            ),
            "limiting_target_ordinal": int(round(float(
                linearized_allocation_info["limiting_target_ordinal"]
            ))),
            "limiting_target": int(
                target_ordinal == int(round(float(
                    linearized_allocation_info[
                        "limiting_target_ordinal"
                    ]
                )))
            ),
            "identity_group_count": int(round(float(
                linearized_allocation_info["identity_group_count"]
            ))),
            "multi_exposure_identity_group_count": int(round(float(
                linearized_allocation_info[
                    "multi_exposure_identity_group_count"
                ]
            ))),
            "boundary_crossing": int(
                pre_signed <= 0.0 and post_signed > 0.0
            ),
            "pre_active_at_replay_boundary": pre_active,
            "post_active_at_replay_boundary": post_active,
            "positive_promotion": positive_promotion,
            "negative_eviction": negative_eviction,
            "valid": 1,
        })
    group_member_changes = {}
    for row in rows:
        group_ordinal = int(row["identity_group_ordinal"])
        group_member_changes.setdefault(group_ordinal, []).append(row)
    for row in rows:
        members = group_member_changes[int(row["identity_group_ordinal"])]
        progress_members = [
            member for member in members
            if int(member["identity_group_progress_member"]) == 1
        ]
        if not progress_members and all(
                float(member["identity_group_progress_required"]) == 0.0
                for member in members):
            # Backward-compatible diagnostic-only callers do not allocate a
            # boundary budget.  They still need a deterministic row for the
            # group summary, but production nonzero groups always provide an
            # explicit progress member.
            progress_members = [members[0]]
        if len(progress_members) != 1:
            raise RuntimeError(
                "pair boundary identity group has no unique progress member"
            )
        group_actual = float(progress_members[0]["signed_margin_change"])
        group_required = float(row["identity_group_progress_required"])
        row["identity_group_actual_signed_margin_change"] = group_actual
        row["identity_group_actual_completion_ratio"] = (
            group_actual / group_required
            if group_required > 0.0 else 0.0
        )
        row["identity_group_worst_member_signed_margin_change"] = min(
            float(member["signed_margin_change"])
            for member in members
        )
    signed_change_tensor = torch.stack(signed_changes)
    correct = signed_change_tensor > zero_tolerance
    reverse = signed_change_tensor < -zero_tolerance
    zero = ~(correct | reverse)
    return {
        "rows": rows,
        "target_count": signed_change_tensor.new_tensor(
            float(len(rows))
        ),
        "signed_change_mean": signed_change_tensor.mean(),
        "signed_change_median": signed_change_tensor.median(),
        "signed_change_worst": signed_change_tensor.min(),
        "correct_count": correct.float().sum(),
        "reverse_count": reverse.float().sum(),
        "zero_count": zero.float().sum(),
        "crossing_count": signed_change_tensor.new_tensor(float(sum(
            row["boundary_crossing"] for row in rows
        ))),
        "promotion_count": signed_change_tensor.new_tensor(float(sum(
            row["positive_promotion"] for row in rows
        ))),
        "eviction_count": signed_change_tensor.new_tensor(float(sum(
            row["negative_eviction"] for row in rows
        ))),
    }


def _lifecycle_target_population_from_delta(lifecycle_delta):
    """Return the exact active lifecycle population encoded by identity delta."""
    if not torch.is_tensor(lifecycle_delta):
        raise TypeError("candidate lifecycle delta must be a tensor")
    if lifecycle_delta.ndim != 2:
        raise ValueError("candidate lifecycle delta must be rank two")
    target_mask = lifecycle_delta.abs() > 0.0
    return target_mask, bool(torch.any(target_mask).item())


def _joint_exact_constraint_acceptance(
        signed_boundary_change,
        actionable_pair_mask,
        boundary_strict_floor=1.0e-12,
        signed_exact_score_change=None,
        candidate_loss_change=None,
        lifecycle_signed_margin=None,
        lifecycle_signed_floor=None,
        lifecycle_target_mask=None,
        lifecycle_tolerance=None,
        preservation_tolerance=None):
    """Classify the exact post-writeback constraints of one pair transaction.

    First-order Jacobian projections are necessary but not sufficient for the
    piecewise selection boundary and cached lifecycle margins.  This helper is
    shared by production backtracking and the startup replay suite so the
    acceptance contract cannot drift into a test-only approximation.
    """
    if signed_boundary_change.shape != actionable_pair_mask.shape:
        raise ValueError(
            "pair boundary exact-acceptance shapes differ: {} and {}".format(
                tuple(signed_boundary_change.shape),
                tuple(actionable_pair_mask.shape),
            )
        )
    if not bool(torch.isfinite(signed_boundary_change).all().item()):
        raise FloatingPointError(
            "non-finite pair boundary exact-acceptance change"
        )
    actionable = actionable_pair_mask > 0.0
    if not bool(torch.any(actionable).item()):
        raise RuntimeError(
            "pair boundary exact acceptance has no actionable target"
        )
    boundary_values = signed_boundary_change[actionable]
    boundary_min_ordinal = int(
        torch.argmin(boundary_values).detach().cpu().item()
    )
    boundary_valid = bool(torch.all(
        boundary_values > float(boundary_strict_floor)
    ).item())
    if preservation_tolerance is not None:
        preservation_tolerance = float(preservation_tolerance)
        if (
                not np.isfinite(preservation_tolerance)
                or preservation_tolerance < 0.0):
            raise ValueError(
                "joint exact preservation tolerance must be finite and "
                "nonnegative"
            )

    exact_score_valid = True
    exact_score_min_signed_change = None
    exact_score_min_ordinal = -1
    if signed_exact_score_change is not None:
        if signed_exact_score_change.shape != actionable_pair_mask.shape:
            raise ValueError(
                "pair exact-score acceptance shapes differ: {} and {}".format(
                    tuple(signed_exact_score_change.shape),
                    tuple(actionable_pair_mask.shape),
                )
            )
        if not bool(torch.isfinite(
                signed_exact_score_change).all().item()):
            raise FloatingPointError(
                "non-finite pair exact-score acceptance change"
            )
        exact_score_values = signed_exact_score_change[actionable]
        if preservation_tolerance is None:
            exact_score_valid = bool(torch.all(
                exact_score_values > float(boundary_strict_floor)
            ).item())
        else:
            exact_score_valid = bool(torch.all(
                exact_score_values >= -preservation_tolerance
            ).item())
        exact_score_min_signed_change = exact_score_values.min()
        exact_score_min_ordinal = int(
            torch.argmin(exact_score_values).detach().cpu().item()
        )

    candidate_valid = True
    if candidate_loss_change is not None:
        if not bool(torch.isfinite(candidate_loss_change).item()):
            raise FloatingPointError(
                "non-finite current-candidate exact loss change"
            )
        candidate_valid = bool((candidate_loss_change < 0.0).item())

    lifecycle_valid = True
    lifecycle_violation_count = 0
    lifecycle_min_signed_gap = None
    lifecycle_min_ordinal = -1
    lifecycle_arguments = (
        lifecycle_signed_margin,
        lifecycle_signed_floor,
        lifecycle_target_mask,
        lifecycle_tolerance,
    )
    if any(value is not None for value in lifecycle_arguments):
        if not all(value is not None for value in lifecycle_arguments):
            raise ValueError(
                "incomplete lifecycle exact-acceptance population"
            )
        expected_shape = lifecycle_signed_margin.shape
        for value in lifecycle_arguments[1:]:
            if value.shape != expected_shape:
                raise ValueError(
                    "lifecycle exact-acceptance shapes differ: {} and {}"
                    .format(tuple(expected_shape), tuple(value.shape))
                )
        if not bool(torch.isfinite(
                lifecycle_signed_margin
                + lifecycle_signed_floor
                + lifecycle_tolerance
        ).all().item()):
            raise FloatingPointError(
                "non-finite lifecycle exact-acceptance input"
            )
        lifecycle_targets = lifecycle_target_mask > 0.0
        if not bool(torch.any(lifecycle_targets).item()):
            raise RuntimeError(
                "lifecycle exact acceptance has no target"
            )
        lifecycle_gap = (
            lifecycle_signed_margin - lifecycle_signed_floor
        )
        lifecycle_violation = (
            lifecycle_targets
            & (lifecycle_gap < -lifecycle_tolerance)
        )
        lifecycle_violation_count = int(
            lifecycle_violation.any(dim=1).float().sum().item()
        )
        lifecycle_valid = lifecycle_violation_count == 0
        lifecycle_min_signed_gap = lifecycle_gap[
            lifecycle_targets
        ].min()
        lifecycle_min_ordinal = int(torch.argmin(
            lifecycle_gap[lifecycle_targets]
        ).detach().cpu().item())

    limiting_constraint_type = "none"
    limiting_target_ordinal = -1
    if not boundary_valid:
        limiting_constraint_type = "boundary"
        limiting_target_ordinal = boundary_min_ordinal
    elif not exact_score_valid:
        limiting_constraint_type = "exact_score"
        limiting_target_ordinal = exact_score_min_ordinal
    elif not candidate_valid:
        limiting_constraint_type = "candidate"
    elif not lifecycle_valid:
        limiting_constraint_type = "lifecycle"
        limiting_target_ordinal = lifecycle_min_ordinal

    return {
        "valid": bool(
            boundary_valid
            and exact_score_valid
            and candidate_valid
            and lifecycle_valid
        ),
        "boundary_valid": boundary_valid,
        "boundary_strict_floor": float(boundary_strict_floor),
        "preservation_tolerance": preservation_tolerance,
        "exact_score_valid": exact_score_valid,
        "candidate_valid": candidate_valid,
        "lifecycle_valid": lifecycle_valid,
        "lifecycle_violation_count": lifecycle_violation_count,
        "boundary_min_signed_change": boundary_values.min(),
        "boundary_min_target_ordinal": boundary_min_ordinal,
        "exact_score_min_signed_change": exact_score_min_signed_change,
        "exact_score_min_target_ordinal": exact_score_min_ordinal,
        "lifecycle_min_signed_gap": lifecycle_min_signed_gap,
        "lifecycle_min_target_ordinal": lifecycle_min_ordinal,
        "limiting_constraint_type": limiting_constraint_type,
        "limiting_target_ordinal": limiting_target_ordinal,
        "signed_boundary_change_values": boundary_values.detach().clone(),
        "signed_exact_score_change_values": (
            None
            if signed_exact_score_change is None
            else signed_exact_score_change[actionable].detach().clone()
        ),
        "candidate_loss_change": (
            None
            if candidate_loss_change is None
            else candidate_loss_change.detach().clone()
        ),
    }


def _joint_exact_inactive_selected_factor_acceptance(error):
    """Classify one temporary exact-search catalog crossing as unsafe.

    The graph replay's default behavior remains fail-loud.  This finite result
    exists only for a nonzero temporary parameter probe whose selected factor
    score crossed to zero.  The exact search can then backtrack to a compatible
    writeback instead of aborting before it observes the safe interval.
    """
    if not isinstance(error, SelectedFactorInactiveCandidateError):
        raise TypeError(
            "inactive selected-factor acceptance requires its typed error"
        )
    return {
        "valid": False,
        "boundary_valid": False,
        "boundary_strict_floor": 1.0e-12,
        "preservation_tolerance": None,
        "exact_score_valid": False,
        "candidate_valid": None,
        "lifecycle_valid": None,
        "lifecycle_violation_count": 0,
        "boundary_min_signed_change": None,
        "boundary_min_target_ordinal": -1,
        "exact_score_min_signed_change": None,
        "exact_score_min_target_ordinal": -1,
        "lifecycle_min_signed_gap": None,
        "lifecycle_min_target_ordinal": -1,
        "limiting_constraint_type": "selected_factor_inactive",
        "limiting_target_ordinal": -1,
        "signed_boundary_change_values": None,
        "signed_exact_score_change_values": None,
        "candidate_loss_change": None,
        "progress_target_present": None,
        "progress_worst_actual": None,
        "progress_worst_required": None,
        "progress_min_completion": None,
        "progress_mean_completion": None,
        "competitor_candidate_indices": (),
        "target_ranks": (),
        "target_active": (),
        "inactive_selected_factor_batch_index": int(error.batch_index),
        "inactive_selected_factor_factor_index": int(error.factor_index),
        "inactive_selected_factor_candidate_index": int(error.candidate_index),
    }


def _joint_exact_catalog_change_acceptance(acceptance, catalog_kind):
    """Turn one parameter-dependent trial catalog crossing into a hard miss."""
    if not isinstance(acceptance, dict) or "valid" not in acceptance:
        raise TypeError("catalog-change acceptance requires an exact result")
    catalog_kind = str(catalog_kind)
    if catalog_kind not in (
            "candidate_catalog_changed",
            "lifecycle_catalog_changed"):
        raise ValueError(
            "unknown joint exact catalog-change kind {}".format(catalog_kind)
        )
    result = dict(acceptance)
    result["valid"] = False
    result["limiting_constraint_type"] = catalog_kind
    result["limiting_target_ordinal"] = -1
    if catalog_kind == "candidate_catalog_changed":
        result["candidate_valid"] = False
        # The lifecycle population was intentionally not evaluated after an
        # earlier candidate-catalog failure.  Do not report a fictitious pass.
        result["lifecycle_valid"] = None
        result["lifecycle_violation_count"] = 0
        result["lifecycle_min_signed_gap"] = None
        result["lifecycle_min_target_ordinal"] = -1
    else:
        result["lifecycle_valid"] = False
        result["lifecycle_violation_count"] = 0
        result["lifecycle_min_signed_gap"] = None
        result["lifecycle_min_target_ordinal"] = -1
    return result


def _joint_exact_origin_preservation_acceptance(info, reference):
    """Classify scale-zero preservation separately from strict progress.

    ``_joint_exact_constraint_acceptance`` always requires every pair boundary
    and the candidate objective to improve strictly.  During progress-member
    search it may apply the audited dtype preservation tolerance to selected
    exact scores; without that explicit tolerance, exact scores remain strict.
    At scale zero all changes are exactly zero, so the joint helper's ``valid``
    result must not be used to claim that the transaction origin itself was
    illegal.  This helper checks only finite, tolerance-bounded non-regression
    plus the lifecycle floor.  It is diagnostic-only and never authorizes a
    commit.
    """
    if not torch.is_tensor(reference):
        raise TypeError("joint exact origin reference must be a tensor")
    tolerance = float(
        128.0 * torch.finfo(reference.dtype).eps
    )
    boundary_values = info["signed_boundary_change_values"]
    exact_values = info["signed_exact_score_change_values"]
    candidate_change = info["candidate_loss_change"]
    boundary_valid = bool(torch.all(
        boundary_values >= -tolerance
    ).item())
    exact_valid = bool(
        exact_values is None
        or torch.all(exact_values >= -tolerance).item()
    )
    candidate_valid = bool(
        candidate_change is None
        or (candidate_change <= tolerance).item()
    )
    return {
        "valid": bool(
            boundary_valid
            and exact_valid
            and candidate_valid
            and bool(info["lifecycle_valid"])
        ),
        "boundary_valid": boundary_valid,
        "exact_score_valid": exact_valid,
        "candidate_valid": candidate_valid,
        "lifecycle_valid": bool(info["lifecycle_valid"]),
        "tolerance": tolerance,
    }


def _joint_exact_trial_trace_record(
        scale, info, evaluation_ordinal, evaluation_kind):
    """Detach one exact production evaluation into a fixed diagnostic row."""
    if "valid" not in info:
        raise KeyError("joint exact evaluator omitted valid")

    def _optional_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    "joint exact scalar diagnostic is not scalar"
                )
            return float(value.detach().cpu().item())
        return float(value)

    boundary_failed_target_ordinals = ()
    boundary_values = info.get("signed_boundary_change_values")
    if boundary_values is not None:
        boundary_floor = float(info.get("boundary_strict_floor", 1.0e-12))
        failed_locations = torch.nonzero(
            boundary_values <= boundary_floor
        ).detach().cpu().reshape(-1).tolist()
        boundary_failed_target_ordinals = tuple(
            int(value) for value in failed_locations
        )

    return {
        "evaluation_ordinal": int(evaluation_ordinal),
        "evaluation_kind": str(evaluation_kind),
        "scale": float(scale),
        "valid": bool(info["valid"]),
        "boundary_valid": info.get("boundary_valid"),
        "boundary_min_signed_change": _optional_float(
            info.get("boundary_min_signed_change")
        ),
        "exact_score_valid": info.get("exact_score_valid"),
        "exact_score_min_signed_change": _optional_float(
            info.get("exact_score_min_signed_change")
        ),
        "candidate_valid": info.get("candidate_valid"),
        "candidate_loss_change": _optional_float(
            info.get("candidate_loss_change")
        ),
        "lifecycle_valid": info.get("lifecycle_valid"),
        "lifecycle_violation_count": info.get(
            "lifecycle_violation_count"
        ),
        "lifecycle_min_signed_gap": _optional_float(
            info.get("lifecycle_min_signed_gap")
        ),
        "limiting_constraint_type": info.get(
            "limiting_constraint_type"
        ),
        "limiting_target_ordinal": info.get(
            "limiting_target_ordinal"
        ),
        "boundary_failed_target_ordinals": (
            boundary_failed_target_ordinals
        ),
        "progress_target_present": info.get("progress_target_present"),
        "progress_worst_actual": info.get("progress_worst_actual"),
        "progress_worst_required": info.get("progress_worst_required"),
        "progress_min_completion": info.get("progress_min_completion"),
        "progress_mean_completion": info.get("progress_mean_completion"),
        "competitor_candidate_indices": tuple(info.get(
            "competitor_candidate_indices", ()
        )),
        "target_ranks": tuple(info.get("target_ranks", ())),
        "target_active": tuple(info.get("target_active", ())),
    }


def _maximize_joint_exact_backtracking_scale(
        evaluate_scale,
        max_halvings=20,
        refinement_steps=12,
        maximum_scale=1.0,
        max_expansions=20,
        select_by_progress=False):
    """Find the largest validated scale in one bounded displacement interval.

    The old production loop accepted the first dyadic scale (1, 1/2, 1/4,
    ...).  Run123 reached the same 0.5 and 0.03125 scales as run121 even after
    the progress-member objective was corrected, leaving only 50.3% and 3.12%
    of its required progress.  Once a valid dyadic lower bound is found, a
    bounded deterministic bisection recovers the unused safe interval without
    changing the displacement direction, any objective, or the boundary
    budget.  Run127 then showed the complementary failure: every selected
    candidate was valid at scale 1, so the routine returned without ever
    observing an unsafe upper bound even though the nearest remaining deficit
    was over thirty times the realized update.  ``maximum_scale`` lets the
    caller provide a finite, evidence-derived upper bound.  Safe scale-1 rays
    are doubled up to that bound; the first exact-forward failure brackets the
    same deterministic refinement used by the original backtracking path.
    A complementary float32 case has an invalid unit scale because the strict
    positive boundary change is still bitwise zero, while a slightly larger
    scale is both representable and fully legal.  After ordinary backtracking
    has exhausted every scale at or below one, a bounded upward scan samples
    the midpoint and endpoint of each doubling interval.  This is not a
    relaxed acceptance path: every sampled point runs the identical exact
    production forward and all original hard contracts.  A later production
    transaction exposed the inverse non-monotone case: every dyadic endpoint
    failed, while exact-valid points existed strictly inside three adjacent
    intervals.  If all endpoint searches fail, a bounded three-level dyadic
    subdivision now searches each interval before declaring infeasibility.
    Any recovered safe island is refined and selected by the same exact
    production authority and real-progress key as every ordinary candidate.

    ``evaluate_scale`` is the production exact forward and therefore remains
    the sole acceptance authority.  When ``select_by_progress`` is enabled,
    every safe trial competes by worst then mean original-required completion;
    a larger safe scale cannot win after its real progress turns backward.
    ``safe_frontier_scale`` remains the largest validated scale independently
    of the selected progress scale.  A returned ``unsafe_upper_present`` flag
    distinguishes a genuinely evaluated failure from a safe search cap.
    """
    max_halvings = int(max_halvings)
    refinement_steps = int(refinement_steps)
    max_expansions = int(max_expansions)
    maximum_scale = float(maximum_scale)
    if (
            max_halvings < 0
            or refinement_steps < 0
            or max_expansions < 0):
        raise ValueError("joint exact backtracking bounds must be nonnegative")
    if not np.isfinite(maximum_scale) or maximum_scale < 1.0:
        raise ValueError(
            "joint exact maximum scale must be finite and at least one"
        )

    best_safe_scale = None
    best_safe_key = None
    trial_trace = []

    def _evaluate(scale, evaluation_kind="search"):
        info = evaluate_scale(float(scale))
        trial_trace.append(_joint_exact_trial_trace_record(
            scale=scale,
            info=info,
            evaluation_ordinal=len(trial_trace),
            evaluation_kind=evaluation_kind,
        ))
        return info

    def _remember_safe(scale, info):
        nonlocal best_safe_scale, best_safe_key
        if not bool(info["valid"]):
            raise RuntimeError("cannot remember an unsafe exact-search point")
        scale = float(scale)
        if bool(select_by_progress):
            minimum = float(info["progress_min_completion"])
            mean = float(info["progress_mean_completion"])
            if not np.isfinite(minimum) or not np.isfinite(mean):
                raise FloatingPointError(
                    "non-finite within-direction progress score"
                )
            # A resolution-only transaction has no nonlinear deficit to close.
            # Once its strict boundary becomes representably positive, prefer
            # the smallest exact-valid displacement instead of overshooting to
            # the finite write-resolution cap.  Real progress-member searches
            # retain the established largest/best-progress tie break.
            scale_key = (
                scale
                if bool(info.get("progress_target_present", True))
                else -scale
            )
            key = (minimum, mean, scale_key)
        else:
            key = (scale,)
        if best_safe_key is None or key > best_safe_key:
            best_safe_key = key
            best_safe_scale = scale

    full_info = _evaluate(1.0)
    if bool(full_info["valid"]):
        _remember_safe(1.0, full_info)
        valid_lower = 1.0
        valid_info = full_info
        expansion_count = 0
        while valid_lower < maximum_scale:
            if expansion_count >= max_expansions:
                raise RuntimeError(
                    "joint exact expansion did not reach its bounded scale"
                )
            trial_scale = min(2.0 * valid_lower, maximum_scale)
            if not trial_scale > valid_lower:
                raise RuntimeError(
                    "joint exact expansion scale failed to advance"
                )
            trial_info = _evaluate(trial_scale)
            expansion_count += 1
            if not bool(trial_info["valid"]):
                invalid_upper = trial_scale
                invalid_upper_info = trial_info
                refinement_count = 0
                for _ in range(refinement_steps):
                    midpoint = 0.5 * (valid_lower + invalid_upper)
                    midpoint_info = _evaluate(midpoint)
                    refinement_count += 1
                    if bool(midpoint_info["valid"]):
                        valid_lower = midpoint
                        valid_info = midpoint_info
                        _remember_safe(midpoint, midpoint_info)
                    else:
                        invalid_upper = midpoint
                        invalid_upper_info = midpoint_info
                valid_info = _evaluate(valid_lower)
                if not bool(valid_info["valid"]):
                    raise RuntimeError(
                        "joint exact expansion lost its validated lower bound"
                    )
                _remember_safe(valid_lower, valid_info)
                selected_info = _evaluate(best_safe_scale)
                if not bool(selected_info["valid"]):
                    raise RuntimeError(
                        "joint exact expansion lost its selected progress point"
                    )
                return {
                    "valid": True,
                    "final_scale": float(best_safe_scale),
                    "safe_frontier_scale": float(valid_lower),
                    "invalid_upper_scale": float(invalid_upper),
                    "unsafe_upper_present": True,
                    "scale_limit": float(maximum_scale),
                    "expansion_count": int(expansion_count),
                    "halving_count": 0,
                    "refinement_count": int(refinement_count),
                    "info": selected_info,
                    "invalid_upper_info": invalid_upper_info,
                    "trial_trace": tuple(trial_trace),
                }
            valid_lower = trial_scale
            valid_info = trial_info
            _remember_safe(trial_scale, trial_info)
        valid_info = _evaluate(valid_lower)
        if not bool(valid_info["valid"]):
            raise RuntimeError(
                "joint exact expansion lost its safe bounded endpoint"
            )
        _remember_safe(valid_lower, valid_info)
        selected_info = _evaluate(best_safe_scale)
        if not bool(selected_info["valid"]):
            raise RuntimeError(
                "joint exact expansion lost its selected bounded point"
            )
        return {
            "valid": True,
            "final_scale": float(best_safe_scale),
            "safe_frontier_scale": float(valid_lower),
            "invalid_upper_scale": float(maximum_scale),
            "unsafe_upper_present": False,
            "scale_limit": float(maximum_scale),
            "expansion_count": int(expansion_count),
            "halving_count": 0,
            "refinement_count": 0,
            "info": selected_info,
            "invalid_upper_info": None,
            "trial_trace": tuple(trial_trace),
        }

    invalid_upper = 1.0
    invalid_upper_info = full_info
    sampled_invalid_points = [(1.0, full_info)]
    valid_lower = None
    valid_info = None
    halving_count = 0
    last_info = full_info
    last_scale = 1.0
    for halving_index in range(1, max_halvings + 1):
        trial_scale = 0.5 ** halving_index
        trial_info = _evaluate(trial_scale)
        halving_count = halving_index
        last_info = trial_info
        last_scale = trial_scale
        if bool(trial_info["valid"]):
            valid_lower = trial_scale
            valid_info = trial_info
            _remember_safe(trial_scale, trial_info)
            break
        invalid_upper = trial_scale
        invalid_upper_info = trial_info
        sampled_invalid_points.append((float(trial_scale), trial_info))
    failed_endpoint_scale = last_scale
    failed_endpoint_info = last_info

    resolution_expansion_count = 0
    resolution_interval_probe_count = 0
    resolution_intervals = []
    if valid_lower is None and maximum_scale > 1.0:
        # Backtracking cannot recover a strict float32 change that is still
        # quantized to zero at scale one.  Search only to the caller's finite,
        # evidence-derived resolution cap.  Midpoints retain narrow safe
        # islands such as (1, 2) which endpoint-only doubling would miss.
        expansion_lower = 1.0
        sampled_safe_scales = []
        resolution_invalid_points = [(1.0, full_info)]
        while expansion_lower < maximum_scale:
            if resolution_expansion_count >= max_expansions:
                raise RuntimeError(
                    "joint exact resolution expansion did not reach its "
                    "bounded scale"
                )
            expansion_upper = min(2.0 * expansion_lower, maximum_scale)
            if not expansion_upper > expansion_lower:
                raise RuntimeError(
                    "joint exact resolution expansion failed to advance"
                )
            resolution_intervals.append((
                float(expansion_lower), float(expansion_upper)
            ))
            midpoint = 0.5 * (expansion_lower + expansion_upper)
            for probe_scale in (midpoint, expansion_upper):
                probe_info = _evaluate(probe_scale)
                last_info = probe_info
                last_scale = probe_scale
                if bool(probe_info["valid"]):
                    _remember_safe(probe_scale, probe_info)
                    sampled_safe_scales.append(float(probe_scale))
                else:
                    resolution_invalid_points.append((
                        float(probe_scale), probe_info
                    ))
                    invalid_upper = float(probe_scale)
                    invalid_upper_info = probe_info
            resolution_expansion_count += 1
            expansion_lower = expansion_upper

        if best_safe_scale is None:
            # The midpoint/endpoint scan above is only the first subdivision
            # level.  A production transaction can have a narrow exact-valid
            # island above scale one after float32 writeback becomes visible but
            # before candidate/lifecycle curvature closes the interval.  The
            # former three-level lattice covered only dyadic intervals below
            # one, so it could incorrectly declare this bounded region
            # infeasible.  Complete the same quarter/eighth lattice over every
            # evidence-bounded write-resolution interval.  Every probe still
            # uses the unchanged production exact forward.
            for subdivision_depth in (2, 3):
                denominator = 2 ** subdivision_depth
                for expansion_lower, expansion_upper in resolution_intervals:
                    for numerator in range(1, denominator, 2):
                        fraction = float(numerator) / float(denominator)
                        probe_scale = (
                            expansion_lower
                            + fraction * (expansion_upper - expansion_lower)
                        )
                        probe_info = _evaluate(
                            probe_scale,
                            evaluation_kind=(
                                "resolution_interval_subdivision_search"
                            ),
                        )
                        resolution_interval_probe_count += 1
                        last_info = probe_info
                        last_scale = probe_scale
                        if bool(probe_info["valid"]):
                            _remember_safe(probe_scale, probe_info)
                            sampled_safe_scales.append(float(probe_scale))
                        else:
                            resolution_invalid_points.append((
                                float(probe_scale), probe_info
                            ))
                if best_safe_scale is not None:
                    break

        if best_safe_scale is not None:
            safe_frontier = max(sampled_safe_scales)
            invalid_above = sorted(
                (
                    (scale, info)
                    for scale, info in resolution_invalid_points
                    if scale > safe_frontier
                ),
                key=lambda item: item[0],
            )
            selected_info = _evaluate(best_safe_scale)
            if not bool(selected_info["valid"]):
                raise RuntimeError(
                    "joint exact resolution expansion lost its selected point"
                )
            if invalid_above:
                invalid_upper, invalid_upper_info = invalid_above[0]
                unsafe_upper_present = True
            else:
                invalid_upper = maximum_scale
                invalid_upper_info = None
                unsafe_upper_present = False
            return {
                "valid": True,
                "final_scale": float(best_safe_scale),
                "safe_frontier_scale": float(safe_frontier),
                "invalid_upper_scale": float(invalid_upper),
                "unsafe_upper_present": bool(unsafe_upper_present),
                "scale_limit": float(maximum_scale),
                "expansion_count": int(resolution_expansion_count),
                "halving_count": int(halving_count),
                "refinement_count": 0,
                "resolution_interval_probe_count": int(
                    resolution_interval_probe_count
                ),
                "info": selected_info,
                "invalid_upper_info": invalid_upper_info,
                "trial_trace": tuple(trial_trace),
            }

    if valid_lower is None:
        # Exact validity is a conjunction of nonlinear boundary, candidate,
        # lifecycle, and score contracts; it is not monotone in displacement
        # scale.  Endpoint-only backtracking can therefore miss a bounded safe
        # island.  Search the interior of every adjacent dyadic interval on a
        # deterministic three-level lattice (midpoint, quarter, eighth).  This
        # path runs only after all ordinary endpoint and write-resolution
        # searches fail, and every point uses the unchanged production exact
        # forward.  The final interval extends once below the smallest endpoint
        # exactly as the former failure-only midpoint audit did.
        interval_safe_scales = []
        interval_probe_count = 0
        interval_subdivision_depth = 3
        for subdivision_depth in range(1, interval_subdivision_depth + 1):
            for interval_ordinal in range(max_halvings + 1):
                upper = 0.5 ** interval_ordinal
                lower = 0.5 ** (interval_ordinal + 1)
                denominator = 2 ** subdivision_depth
                for numerator in range(1, denominator, 2):
                    fraction = float(numerator) / float(denominator)
                    probe_scale = lower + fraction * (upper - lower)
                    probe_info = _evaluate(
                        probe_scale,
                        evaluation_kind="interval_subdivision_search",
                    )
                    interval_probe_count += 1
                    last_info = probe_info
                    last_scale = probe_scale
                    if bool(probe_info["valid"]):
                        _remember_safe(probe_scale, probe_info)
                        interval_safe_scales.append(float(probe_scale))
                    else:
                        sampled_invalid_points.append((
                            float(probe_scale), probe_info
                        ))
            # Refine the first lattice resolution that proves any safe island.
            # This keeps the formerly failing path bounded to 21 midpoint
            # forwards in the common recovered case, while quarter/eighth
            # levels remain available for narrower islands.
            if interval_safe_scales:
                break

        if interval_safe_scales:
            safe_frontier = max(interval_safe_scales)
            invalid_above = sorted(
                (
                    (scale, info)
                    for scale, info in sampled_invalid_points
                    if scale > safe_frontier
                ),
                key=lambda item: item[0],
            )
            refinement_count = 0
            if invalid_above:
                invalid_upper, invalid_upper_info = invalid_above[0]
                valid_lower = safe_frontier
                for _ in range(refinement_steps):
                    midpoint = 0.5 * (valid_lower + invalid_upper)
                    midpoint_info = _evaluate(
                        midpoint,
                        evaluation_kind="interval_island_refinement",
                    )
                    refinement_count += 1
                    if bool(midpoint_info["valid"]):
                        valid_lower = midpoint
                        _remember_safe(midpoint, midpoint_info)
                    else:
                        invalid_upper = midpoint
                        invalid_upper_info = midpoint_info
                safe_frontier = valid_lower
                unsafe_upper_present = True
            else:
                invalid_upper = 1.0
                invalid_upper_info = full_info
                unsafe_upper_present = False

            selected_info = _evaluate(
                best_safe_scale,
                evaluation_kind="interval_island_selection",
            )
            if not bool(selected_info["valid"]):
                raise RuntimeError(
                    "joint exact interval search lost its selected safe point"
                )
            return {
                "valid": True,
                "final_scale": float(best_safe_scale),
                "safe_frontier_scale": float(safe_frontier),
                "invalid_upper_scale": float(invalid_upper),
                "unsafe_upper_present": bool(unsafe_upper_present),
                "scale_limit": float(maximum_scale),
                "expansion_count": int(resolution_expansion_count),
                "halving_count": int(halving_count),
                "refinement_count": int(refinement_count),
                "interval_probe_count": int(interval_probe_count),
                "interval_valid_probe_count": int(
                    len(interval_safe_scales)
                ),
                "info": selected_info,
                "invalid_upper_info": invalid_upper_info,
                "trial_trace": tuple(trial_trace),
            }

        return {
            "valid": False,
            "final_scale": failed_endpoint_scale,
            "safe_frontier_scale": 0.0,
            "invalid_upper_scale": invalid_upper,
            "unsafe_upper_present": True,
            "scale_limit": float(maximum_scale),
            "expansion_count": int(resolution_expansion_count),
            "halving_count": halving_count,
            "refinement_count": 0,
            "interval_probe_count": int(interval_probe_count),
            "interval_valid_probe_count": 0,
            "resolution_interval_probe_count": int(
                resolution_interval_probe_count
            ),
            "info": failed_endpoint_info,
            "invalid_upper_info": invalid_upper_info,
            "trial_trace": tuple(trial_trace),
        }

    refinement_count = 0
    for _ in range(refinement_steps):
        midpoint = 0.5 * (valid_lower + invalid_upper)
        midpoint_info = _evaluate(midpoint)
        refinement_count += 1
        if bool(midpoint_info["valid"]):
            valid_lower = midpoint
            valid_info = midpoint_info
            _remember_safe(midpoint, midpoint_info)
        else:
            invalid_upper = midpoint
            invalid_upper_info = midpoint_info

    # The last midpoint can be invalid.  Reapply and revalidate the largest
    # accepted scale so parameter state and diagnostics refer to one exact
    # committed point.
    valid_info = _evaluate(valid_lower)
    if not bool(valid_info["valid"]):
        raise RuntimeError(
            "joint exact backtracking lost its validated lower bound"
        )
    _remember_safe(valid_lower, valid_info)
    selected_info = _evaluate(best_safe_scale)
    if not bool(selected_info["valid"]):
        raise RuntimeError(
            "joint exact backtracking lost its selected progress point"
        )
    return {
        "valid": True,
        "final_scale": float(best_safe_scale),
        "safe_frontier_scale": float(valid_lower),
        "invalid_upper_scale": float(invalid_upper),
        "unsafe_upper_present": True,
        "scale_limit": float(maximum_scale),
        "expansion_count": 0,
        "halving_count": int(halving_count),
        "refinement_count": int(refinement_count),
        "info": selected_info,
        "invalid_upper_info": invalid_upper_info,
        "trial_trace": tuple(trial_trace),
    }


def _select_joint_exact_progress_direction(
        direction_evaluators,
        max_halvings=20,
        refinement_steps=12,
        maximum_scale=1.0,
        max_expansions=20):
    """Choose the direction with the best *real* progress-member result.

    Run124 showed that bisection can recover the maximum safe scale on a fixed
    direction while still realizing only 69%/2.5% of the progress-member
    requirement.  The cause is not missing line-search precision: the affine
    projection can point into a curved all-member non-regression boundary.
    Evaluate a small deterministic family of directions which spend no more
    than the existing boundary budget, and choose by post-forward progress
    against the original (unscaled) requirement.  Exact score, every member's
    boundary, candidate, lifecycle, and finite-value gates remain hard.

    Each entry is ``(label, evaluate_scale)``.  The callback must return the
    production exact-acceptance dictionary plus ``progress_min_completion``
    and ``progress_mean_completion``.  The selected callback is re-evaluated
    by the caller before commit because other candidates may have changed the
    temporary parameter state.
    """
    candidates = list(direction_evaluators)
    if not candidates:
        raise ValueError("joint exact progress direction set is empty")
    results = []
    candidate_results = []
    for candidate_ordinal, (label, evaluate_scale) in enumerate(candidates):
        result = _maximize_joint_exact_backtracking_scale(
            evaluate_scale=evaluate_scale,
            max_halvings=max_halvings,
            refinement_steps=refinement_steps,
            maximum_scale=maximum_scale,
            max_expansions=max_expansions,
            select_by_progress=True,
        )
        candidate_result = dict(result)
        candidate_result["candidate_ordinal"] = int(candidate_ordinal)
        candidate_result["candidate_label"] = label
        candidate_results.append(candidate_result)
        if not bool(result["valid"]):
            continue
        info = result["info"]
        minimum = float(info["progress_min_completion"])
        mean = float(info["progress_mean_completion"])
        if not np.isfinite(minimum) or not np.isfinite(mean):
            raise FloatingPointError(
                "non-finite joint exact progress-direction score"
            )
        # Stable final tie-breaks prefer the larger verified displacement and
        # then the earlier candidate (the v19 direction is ordinal zero).
        scale_key = (
            float(result["final_scale"])
            if bool(info.get("progress_target_present", True))
            else -float(result["final_scale"])
        )
        key = (
            minimum,
            mean,
            scale_key,
            -int(candidate_ordinal),
        )
        results.append((key, candidate_ordinal, label, evaluate_scale, result))
    if not results:
        return {
            "valid": False,
            "candidate_count": len(candidates),
            "valid_candidate_count": 0,
            "candidate_results": tuple(candidate_results),
        }
    selected = max(results, key=lambda item: item[0])
    _key, ordinal, label, evaluate_scale, result = selected
    selected_info = evaluate_scale(result["final_scale"])
    if not bool(selected_info["valid"]):
        raise RuntimeError(
            "joint exact progress direction lost its validated scale"
        )
    result = dict(result)
    result["info"] = selected_info
    result["candidate_count"] = len(candidates)
    result["valid_candidate_count"] = len(results)
    result["selected_candidate_ordinal"] = int(ordinal)
    result["selected_candidate_label"] = label
    result["candidate_results"] = tuple(candidate_results)
    return result


def _exact_pair_target_score_gradient_constraints(
        target_factor_logp,
        pair_local_delta,
        parameters):
    """Build one signed exact-score Jacobian constraint per strict pair target.

    The aggregate pair loss can be a descent direction while shared parameters
    move individual identities backwards.  Each returned gradient is for
    ``-sign * log p(exact factor)``.  A positive optimizer-gradient dot with
    that gradient therefore gives a first-order improvement of the exact
    signed score.  Target mass remains in the separate weights and is never
    duplicated as an additional loss.
    """
    if target_factor_logp.shape != pair_local_delta.shape:
        raise ValueError(
            "pair target score-gradient shapes differ: {} and {}".format(
                tuple(target_factor_logp.shape),
                tuple(pair_local_delta.shape),
            )
        )
    if not bool(torch.isfinite(
            target_factor_logp + pair_local_delta).all().item()):
        raise FloatingPointError(
            "non-finite pair target score-gradient input"
        )
    target_indices = torch.nonzero(
        pair_local_delta != 0.0,
    ).detach().cpu().tolist()
    if not target_indices:
        raise RuntimeError(
            "pair target score-gradient constraints have no exact target"
        )
    constraints = []
    target_weights = []
    target_signs = []
    for raw_index in target_indices:
        index = tuple(int(item) for item in raw_index)
        target_delta = pair_local_delta[index].detach()
        target_sign = torch.sign(target_delta)
        score_loss = -target_sign * target_factor_logp[index]
        grads = torch.autograd.grad(
            score_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        detached_grads = tuple(
            None if grad is None else grad.detach().clone()
            for grad in grads
        )
        norm_sq = _gradient_tuple_dot(
            detached_grads,
            detached_grads,
            target_factor_logp,
        )
        if not bool(torch.isfinite(norm_sq).item()):
            raise FloatingPointError(
                "non-finite exact pair target score Jacobian"
            )
        if not bool((norm_sq > 0.0).item()):
            raise RuntimeError(
                "exact pair target score has zero adjacency Jacobian: "
                "index={}".format(index)
            )
        constraints.append(detached_grads)
        target_weights.append(target_delta.abs())
        target_signs.append(target_sign)
    return {
        "constraints": tuple(constraints),
        "weights": torch.stack(target_weights),
        "signs": torch.stack(target_signs),
        "target_count": float(len(constraints)),
    }


def _project_gradient_tuple_to_minimum_dots(
        proposed_grads,
        constraint_grads,
        minimum_dots,
        reference,
        diagnostic_name):
    """Return the minimum-change projection onto affine gradient halfspaces.

    Dykstra's deterministic cyclic projection avoids a new optimizer, a loss
    coefficient, and unsupported linear algebra in PyTorch 1.3.1.  It either
    satisfies every target-local direction or fails loudly before Adam runs.
    """
    if len(constraint_grads) != len(minimum_dots):
        raise ValueError(
            "{} constraint/floor lengths differ".format(diagnostic_name)
        )
    original = tuple(
        None if grad is None else grad.detach().clone()
        for grad in proposed_grads
    )
    if not constraint_grads:
        return original, {
            "constraint_count": 0.0,
            "active_constraint_count": 0.0,
            "active_constraint_indices": (),
            "intervened": 0.0,
            "min_dot_before": 0.0,
            "min_dot_after": 0.0,
            "projection_delta_norm": 0.0,
        }
    constraints = []
    floors = []
    for constraint, minimum_dot in zip(constraint_grads, minimum_dots):
        detached = tuple(
            None if grad is None else grad.detach().clone()
            for grad in constraint
        )
        norm_sq = _gradient_tuple_dot(detached, detached, reference)
        floor = reference.new_tensor(float(minimum_dot))
        if not bool(
                torch.isfinite(norm_sq).item()
                and torch.isfinite(floor).item()):
            raise FloatingPointError(
                "non-finite {} constraint".format(diagnostic_name)
            )
        if not bool((norm_sq > 0.0).item()):
            raise RuntimeError(
                "{} constraint has zero norm".format(diagnostic_name)
            )
        if bool((floor < 0.0).item()):
            raise ValueError(
                "{} minimum dot must be nonnegative".format(diagnostic_name)
            )
        constraints.append((detached, norm_sq))
        floors.append(floor)

    def _dots(grads):
        return torch.stack([
            _gradient_tuple_dot(grads, constraint, reference)
            for constraint, _norm_sq in constraints
        ])

    before_dots = _dots(original)
    dtype_epsilon = _floating_dtype_epsilon(reference)
    floor_tensor = torch.stack(floors)
    gram = [
        [
            float(_gradient_tuple_dot(
                left_constraint,
                right_constraint,
                reference,
            ).detach().cpu().item())
            for right_constraint, _right_norm_sq in constraints
        ]
        for left_constraint, _left_norm_sq in constraints
    ]
    rhs = [
        float(value)
        for value in (
            floor_tensor - before_dots
        ).detach().cpu().tolist()
    ]

    def _solve_active_system(active_indices):
        size = len(active_indices)
        augmented = [
            [
                float(gram[row_index][column_index])
                for column_index in active_indices
            ] + [float(rhs[row_index])]
            for row_index in active_indices
        ]
        matrix_scale = max(
            [abs(value) for row in augmented for value in row[:-1]]
            + [1.0e-300]
        )
        pivot_tolerance = 1024.0 * 2.220446049250313e-16 * matrix_scale
        for column in range(size):
            pivot_row = max(
                range(column, size),
                key=lambda row: abs(augmented[row][column]),
            )
            pivot = augmented[pivot_row][column]
            if abs(pivot) <= pivot_tolerance:
                return None
            if pivot_row != column:
                augmented[column], augmented[pivot_row] = (
                    augmented[pivot_row], augmented[column]
                )
            pivot = augmented[column][column]
            for item in range(column, size + 1):
                augmented[column][item] /= pivot
            for row in range(size):
                if row == column:
                    continue
                scale = augmented[row][column]
                if scale == 0.0:
                    continue
                for item in range(column, size + 1):
                    augmented[row][item] -= (
                        scale * augmented[column][item]
                    )
        return [augmented[row][-1] for row in range(size)]

    active = []
    projected = tuple(
        None if grad is None else grad.detach().clone()
        for grad in original
    )
    converged = False
    max_active_iterations = max(32, 8 * len(constraints) ** 2)
    for _active_iteration in range(max_active_iterations):
        coefficients = [0.0] * len(constraints)
        if active:
            active_coefficients = _solve_active_system(active)
            if active_coefficients is None:
                break
            coefficient_tolerance = (
                256.0 * 2.220446049250313e-16
                * max(
                    1.0,
                    max(abs(value) for value in active_coefficients),
                )
            )
            if any(
                    value < -coefficient_tolerance
                    for value in active_coefficients):
                remove_position = min(
                    range(len(active_coefficients)),
                    key=lambda index: active_coefficients[index],
                )
                del active[remove_position]
                continue
            for index, value in zip(active, active_coefficients):
                coefficients[index] = max(value, 0.0)
        projected_values = []
        for parameter_index, original_grad in enumerate(original):
            value = (
                None
                if original_grad is None
                else original_grad.detach().clone()
            )
            for coefficient, (constraint, _norm_sq) in zip(
                    coefficients, constraints):
                constraint_grad = constraint[parameter_index]
                if constraint_grad is None or coefficient == 0.0:
                    continue
                addition = float(coefficient) * constraint_grad
                value = addition if value is None else value + addition
            projected_values.append(value)
        projected = tuple(projected_values)
        after_dots = _dots(projected)
        projected_norm_sq = _gradient_tuple_dot(
            projected, projected, reference
        )
        tolerances = torch.stack([
            _gradient_dot_tolerance(
                proposed_grads=projected,
                constraint_grads=constraint,
                reference=reference,
                tolerance_multiplier=128.0,
            )
            for constraint, _norm_sq in constraints
        ])
        violations = floor_tensor - after_dots
        if bool(torch.all(violations <= tolerances).item()):
            converged = True
            break
        worst_index = int(
            torch.argmax(violations - tolerances).detach().cpu().item()
        )
        if worst_index in active:
            break
        active.append(worst_index)
    if not converged:
        raise RuntimeError(
            "{} constraints are jointly infeasible".format(diagnostic_name)
        )
    after_dots = _dots(projected)
    positive_floor_mask = floor_tensor > 0.0
    if (
            bool(torch.any(positive_floor_mask).item())
            and bool(torch.any(
                after_dots[positive_floor_mask] <= 0.0
            ).item())):
        raise RuntimeError(
            "{} lost strict target descent".format(diagnostic_name)
        )
    delta = []
    for projected_grad, original_grad in zip(projected, original):
        if projected_grad is None and original_grad is None:
            delta.append(None)
        elif projected_grad is None:
            delta.append(-original_grad)
        elif original_grad is None:
            delta.append(projected_grad)
        else:
            delta.append(projected_grad - original_grad)
    delta_norm_sq = _gradient_tuple_dot(delta, delta, reference)
    return projected, {
        "constraint_count": float(len(constraints)),
        "active_constraint_count": float(len(active)),
        "active_constraint_indices": tuple(int(index) for index in active),
        "intervened": float(bool((delta_norm_sq > 0.0).item())),
        "min_dot_before": float(before_dots.min().detach().cpu().item()),
        "min_dot_after": float(after_dots.min().detach().cpu().item()),
        "projection_delta_norm": float(
            torch.sqrt(delta_norm_sq).detach().cpu().item()
        ),
    }


def _pair_target_mass_preserving_minimum_dots(
        pair_target_score_grads,
        pair_target_weights,
        pair_grads,
        reference):
    """Allocate strict target-direction floors by the existing signed mass."""
    if len(pair_target_score_grads) != int(pair_target_weights.numel()):
        raise ValueError("pair target gradient/mass counts differ")
    if not bool(torch.isfinite(pair_target_weights).all().item()):
        raise FloatingPointError("non-finite pair target mass")
    total_mass = pair_target_weights.sum()
    if not bool((total_mass > 0.0).item()):
        raise RuntimeError("pair target direction has no signed mass")
    pair_norm_sq = _gradient_tuple_dot(pair_grads, pair_grads, reference)
    if not bool(
            torch.isfinite(pair_norm_sq).item()
            and (pair_norm_sq > 0.0).item()):
        raise RuntimeError("pair target direction has no aggregate gradient")
    pair_norm = torch.sqrt(pair_norm_sq)
    floors = []
    for target_grads, target_weight in zip(
            pair_target_score_grads, pair_target_weights):
        target_norm_sq = _gradient_tuple_dot(
            target_grads,
            target_grads,
            reference,
        )
        target_norm = torch.sqrt(target_norm_sq)
        floor = (
            target_weight / total_mass * pair_norm * target_norm
        )
        floors.append(float(floor.detach().cpu().item()))
    return tuple(floors)


def _pair_boundary_deficit_aware_minimum_dots(
        boundary_target_grads,
        base_minimum_dots,
        pair_target_weights,
        pre_boundary,
        pair_local_delta,
        target_candidate_index,
        proposed_descent,
        reference):
    """Reallocate, but never enlarge, the current boundary-descent budget.

    The v13/v14 floor followed evidence mass even when a target was already on
    the correct side of its real selection boundary.  Run118 consequently
    spent 12/26 target-epoch boundary floors on zero-deficit targets while no
    deficit-bearing target crossed.  This helper preserves the total existing
    first-order budget and every target's representable strict floor, then:

    * funds the closest linearly affordable crossing first; and
    * water-fills any insufficient remainder from the nearest real boundary
      outward instead of fragmenting it across several unreachable targets.

    It does not add a loss coefficient, increase the aggregate budget, require
    an exact nonlinear crossing, or admit forced/non-actionable targets.
    """
    target_count = len(boundary_target_grads)
    if (
            target_count != len(base_minimum_dots)
            or target_count != int(pair_target_weights.numel())):
        raise ValueError(
            "pair boundary gradient/floor/mass counts differ"
        )
    if pre_boundary.shape != pair_local_delta.shape:
        raise ValueError(
            "pair boundary deficit shapes differ: {} and {}".format(
                tuple(pre_boundary.shape), tuple(pair_local_delta.shape)
            )
        )
    if target_candidate_index.shape != pair_local_delta.shape:
        raise ValueError(
            "pair boundary canonical-index shape differs from target shape"
        )
    target_indices = torch.nonzero(
        pair_local_delta != 0.0
    ).detach().cpu().tolist()
    if len(target_indices) != target_count:
        raise RuntimeError(
            "pair boundary deficit population differs from target constraints"
        )
    if not bool(torch.isfinite(
            pre_boundary + pair_local_delta + target_candidate_index
    ).all().item()):
        raise FloatingPointError("non-finite pair boundary deficit input")
    if not bool(torch.isfinite(pair_target_weights).all().item()):
        raise FloatingPointError("non-finite pair boundary target mass")

    deficits = []
    strict_floors = []
    base_floors = []
    for raw_index, constraint, raw_base_floor in zip(
            target_indices, boundary_target_grads, base_minimum_dots):
        index = tuple(int(item) for item in raw_index)
        signed_margin = (
            torch.sign(pair_local_delta[index]) * pre_boundary[index]
        ).detach()
        deficits.append(torch.relu(-signed_margin))
        strict_floor = _strict_gradient_dot_floor(
            proposed_grads=proposed_descent,
            constraint_grads=constraint,
            reference=reference,
        )
        strict_floors.append(float(strict_floor.detach().cpu().item()))
        base_floor = float(raw_base_floor)
        if not np.isfinite(base_floor) or base_floor < 0.0:
            raise ValueError("invalid base pair boundary minimum dot")
        base_floors.append(base_floor)

    deficit_tensor = torch.stack(deficits)
    if not bool(torch.isfinite(deficit_tensor).all().item()):
        raise FloatingPointError("non-finite pair boundary deficit")
    floors = list(strict_floors)
    base_budget = float(sum(base_floors))
    strict_budget = float(sum(strict_floors))
    total_budget = max(base_budget, strict_budget)
    remaining_budget = max(total_budget - strict_budget, 0.0)
    affordable = [0.0 for _ in range(target_count)]
    deficit_indices = [
        index for index in range(target_count)
        if float(deficit_tensor[index].detach().cpu().item()) > 0.0
    ]

    if not deficit_indices:
        # With no rank-boundary gap, retain the prior mass-preserving contract.
        floors = [
            max(base_floor, strict_floor)
            for base_floor, strict_floor in zip(base_floors, strict_floors)
        ]
    else:
        # A stable index tie-break makes the allocation deterministic on legacy
        # PyTorch versions and independent of dictionary/catalog iteration.
        deficit_indices.sort(key=lambda index: (
            float(deficit_tensor[index].detach().cpu().item()), index
        ))
        comparison_tolerance = (
            64.0 * _floating_dtype_epsilon(reference)
            * max(total_budget, 1.0e-12)
        )
        for index in deficit_indices:
            crossing_floor = (
                float(deficit_tensor[index].detach().cpu().item())
                + strict_floors[index]
            )
            extra = max(crossing_floor - floors[index], 0.0)
            if extra <= remaining_budget + comparison_tolerance:
                floors[index] = crossing_floor
                remaining_budget = max(remaining_budget - extra, 0.0)
                affordable[index] = 1.0

        # When no complete crossing is affordable, v15 divided the remainder
        # between every deficit-bearing target.  Run120 then realized those
        # tiny requirements accurately but obtained zero rank crossings.  A
        # closest-first water-fill preserves exactly the same total budget and
        # all strict floors while making finite progress toward the first
        # reachable production boundary.  It remains bounded and does not make
        # crossing a transaction commit condition.
        unfunded = [
            index for index in deficit_indices if affordable[index] == 0.0
        ]
        for index in unfunded:
            if remaining_budget <= 0.0:
                break
            crossing_floor = (
                float(deficit_tensor[index].detach().cpu().item())
                + strict_floors[index]
            )
            extra_needed = max(crossing_floor - floors[index], 0.0)
            allocation = min(remaining_budget, extra_needed)
            floors[index] += allocation
            remaining_budget = max(remaining_budget - allocation, 0.0)

    if not all(np.isfinite(value) and value > 0.0 for value in floors):
        raise RuntimeError(
            "pair boundary deficit allocation lost strict target descent"
        )
    # The only permitted increase is the machine-representable strict floor
    # when the legacy base budget was itself below write precision.
    allowed_budget = max(base_budget, strict_budget)
    allocated_budget = float(sum(floors))
    budget_tolerance = (
        256.0 * _floating_dtype_epsilon(reference)
        * max(allowed_budget, 1.0e-12)
    )
    if allocated_budget > allowed_budget + budget_tolerance:
        raise RuntimeError(
            "pair boundary deficit allocation increased its descent budget"
        )
    zero_deficit_reclaimed_budget = float(sum(
        max(base_floors[index] - strict_floors[index], 0.0)
        for index in range(target_count)
        if index not in deficit_indices
    ))
    # Boundary observations are transition-local, but repeated replay rows can
    # supervise the same canonical factor and sign.  Run122 showed that summing
    # their Jacobians is not a production selection quantity: easy members
    # satisfied nearly 100% of the group sum while the water-fill-designated
    # nearest member achieved only 29% of its requirement.  Keep every row's
    # exact-score and boundary non-regression contract, but place the identity's
    # *additional* progress budget on the closest real boundary member.  This
    # prevents aggregate masking without restoring a full progress floor on
    # every exposure.
    group_members = {}
    for target_ordinal, raw_index in enumerate(target_indices):
        target_location = tuple(int(item) for item in raw_index)
        raw_candidate_index = float(
            target_candidate_index[target_location].detach().cpu().item()
        )
        candidate_index = int(round(raw_candidate_index))
        if candidate_index < 0 or abs(
                raw_candidate_index - float(candidate_index)) > 1.0e-6:
            raise RuntimeError(
                "pair boundary target has invalid canonical candidate index"
            )
        target_sign = int(torch.sign(
            pair_local_delta[target_location]
        ).detach().cpu().item())
        group_key = (candidate_index, target_sign)
        group_members.setdefault(group_key, []).append(target_ordinal)

    projection_minimum_dots = list(floors)
    identity_group_member_indices = []
    identity_group_minimum_dots = []
    identity_group_ordinals = [-1 for _ in range(target_count)]
    identity_group_exposure_counts = [1 for _ in range(target_count)]
    identity_group_allocated_budgets = [0.0 for _ in range(target_count)]
    identity_group_strict_budgets = [0.0 for _ in range(target_count)]
    identity_group_extra_budgets = [0.0 for _ in range(target_count)]
    identity_group_progress_member_ordinals = [-1 for _ in range(target_count)]
    identity_group_progress_member_flags = [0 for _ in range(target_count)]
    identity_group_progress_required = [0.0 for _ in range(target_count)]
    target_strict_floors = list(strict_floors)
    target_waterfill_allocations = [
        max(floor - strict_floor, 0.0)
        for floor, strict_floor in zip(floors, strict_floors)
    ]
    ordered_groups = sorted(group_members.items(), key=lambda item: item[0])
    for group_ordinal, (_, member_indices) in enumerate(ordered_groups):
        group_allocated_budget = float(sum(
            floors[index] for index in member_indices
        ))
        group_strict_budget = float(sum(
            strict_floors[index] for index in member_indices
        ))
        group_extra_budget = max(
            group_allocated_budget - group_strict_budget, 0.0
        )
        progress_member = min(member_indices, key=lambda index: (
            float(deficit_tensor[index].detach().cpu().item())
            if float(deficit_tensor[index].detach().cpu().item()) > 0.0
            else float("inf"),
            -target_waterfill_allocations[index],
            index,
        ))
        if not any(
                float(deficit_tensor[index].detach().cpu().item()) > 0.0
                for index in member_indices):
            progress_member = min(member_indices)
        progress_required = (
            strict_floors[progress_member] + group_extra_budget
        )
        for index in member_indices:
            identity_group_ordinals[index] = int(group_ordinal)
            identity_group_exposure_counts[index] = int(len(member_indices))
            identity_group_allocated_budgets[index] = group_allocated_budget
            identity_group_strict_budgets[index] = group_strict_budget
            identity_group_extra_budgets[index] = group_extra_budget
            identity_group_progress_member_ordinals[index] = int(
                progress_member
            )
            identity_group_progress_member_flags[index] = int(
                index == progress_member
            )
            identity_group_progress_required[index] = progress_required
        if len(member_indices) <= 1:
            identity_group_progress_member_flags[progress_member] = 1
            continue
        projection_minimum_dots = list(projection_minimum_dots)
        for index in member_indices:
            projection_minimum_dots[index] = strict_floors[index]
        identity_group_member_indices.append((progress_member,))
        identity_group_minimum_dots.append(progress_required)

    return {
        "minimum_dots": tuple(floors),
        "projection_minimum_dots": tuple(projection_minimum_dots),
        "target_strict_floors": tuple(target_strict_floors),
        "target_waterfill_allocations": tuple(target_waterfill_allocations),
        "identity_group_member_indices": tuple(identity_group_member_indices),
        "identity_group_minimum_dots": tuple(identity_group_minimum_dots),
        "identity_group_ordinals": tuple(identity_group_ordinals),
        "identity_group_exposure_counts": tuple(
            identity_group_exposure_counts
        ),
        "identity_group_allocated_budgets": tuple(
            identity_group_allocated_budgets
        ),
        "identity_group_strict_budgets": tuple(
            identity_group_strict_budgets
        ),
        "identity_group_extra_budgets": tuple(
            identity_group_extra_budgets
        ),
        "identity_group_progress_member_ordinals": tuple(
            identity_group_progress_member_ordinals
        ),
        "identity_group_progress_member_flags": tuple(
            identity_group_progress_member_flags
        ),
        "identity_group_progress_required": tuple(
            identity_group_progress_required
        ),
        "pre_deficits": deficit_tensor,
        "crossing_affordable": reference.new_tensor(affordable),
        "base_budget": base_budget,
        "strict_budget": strict_budget,
        "allocated_budget": allocated_budget,
        "budget_tolerance": budget_tolerance,
        "zero_deficit_reclaimed_budget": zero_deficit_reclaimed_budget,
        "deficit_target_count": float(len(deficit_indices)),
        "affordable_crossing_count": float(sum(affordable)),
        "identity_group_count": float(len(ordered_groups)),
        "multi_exposure_identity_group_count": float(sum(
            1 for _, member_indices in ordered_groups
            if len(member_indices) > 1
        )),
        "selected_progress_floor_fraction": 1.0,
        "progress_min_completion": 0.0,
        "progress_mean_completion": 0.0,
        "limiting_constraint_code": 0.0,
        "limiting_target_ordinal": -1.0,
        "budget_conservation_valid": 1.0,
    }


def _mean_gradient_tuples(gradient_tuples, parameter_count):
    """Average detached per-transition gradients without a second backward."""
    if not gradient_tuples:
        return tuple(None for _ in range(parameter_count))
    result = []
    denominator = float(len(gradient_tuples))
    for parameter_index in range(parameter_count):
        values = [
            grads[parameter_index]
            for grads in gradient_tuples
            if grads[parameter_index] is not None
        ]
        if not values:
            result.append(None)
            continue
        total = values[0].detach().clone()
        for value in values[1:]:
            total = total + value.detach()
        result.append(total / denominator)
    return tuple(result)


def _recover_standard_zero_aggregate_pair_gradient(
        pair_grads,
        pair_target_score_grads,
        pair_boundary_target_grads,
        reference):
    """Recover a minimal jointly strict direction for a standard transaction.

    The clipped aggregate pair PPO term can have an exactly zero gradient even
    though every target-local score and selection-boundary Jacobian is finite
    and the strict constraints are jointly feasible.  Rejecting at the
    aggregate norm check loses real current evidence before the existing exact
    feasibility guard can inspect it.  For a *standard* transaction only,
    project the zero tuple onto machine-resolvable positive floors for every
    current exact constraint.  This introduces no loss coefficient and returns
    the minimum-norm common direction.  Contradictory targets remain fail-loud
    through the same joint projection.

    Pending pair-only transactions deliberately keep their typed zero-gradient
    deferral contract and must not call this helper.
    """
    constraints = (
        tuple(pair_target_score_grads)
        + tuple(pair_boundary_target_grads)
    )
    if not constraints:
        raise RuntimeError(
            "standard zero-aggregate pair target has no exact constraint"
        )
    if any(len(constraint) != len(pair_grads) for constraint in constraints):
        raise ValueError(
            "standard zero-aggregate pair gradient tuple lengths differ"
        )
    zero_grads = []
    for parameter_index in range(len(pair_grads)):
        values = [
            constraint[parameter_index]
            for constraint in constraints
            if constraint[parameter_index] is not None
        ]
        zero_grads.append(
            None if not values else torch.zeros_like(values[0])
        )
    zero_grads = tuple(zero_grads)
    floors = tuple(
        float(_strict_gradient_dot_floor(
            proposed_grads=zero_grads,
            constraint_grads=constraint,
            reference=reference,
        ).detach().cpu().item())
        for constraint in constraints
    )
    recovered, projection_info = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=zero_grads,
        constraint_grads=constraints,
        minimum_dots=floors,
        reference=reference,
        diagnostic_name=(
            "standard zero-aggregate exact-pair target gradient"
        ),
    )
    recovered_norm_sq = _gradient_tuple_dot(
        recovered, recovered, reference
    )
    recovered_dots = tuple(
        _gradient_tuple_dot(recovered, constraint, reference)
        for constraint in constraints
    )
    if not bool(
            torch.isfinite(recovered_norm_sq).item()
            and (recovered_norm_sq > 0.0).item()
            and all(
                bool(torch.isfinite(dot).item() and (dot > 0.0).item())
                for dot in recovered_dots
            )):
        raise RuntimeError(
            "standard zero-aggregate pair recovery lost strict descent"
        )
    info = dict(projection_info)
    info["recovered"] = 1.0
    info["recovered_norm"] = float(
        torch.sqrt(recovered_norm_sq).detach().cpu().item()
    )
    info["minimum_dot"] = float(min(
        dot.detach().cpu().item() for dot in recovered_dots
    ))
    return tuple(recovered), info


def _preserve_pair_gradient_in_standard_transaction(
        proposed_grads,
        pair_grads,
        candidate_grads,
        reference):
    """Remove only the ordinary-objective component opposing strict pair credit.

    A normal adjacency transaction contains the pair objective exactly once.
    Treating the complete gradient as a monolith allowed graph/candidate
    objectives to overwhelm that component in run113: the first transaction
    after the pending commit had a negative combined/pair dot and all five
    exact pair targets moved backwards. Decompose ``proposed = base + pair``,
    project only the conflicting component of ``base`` off ``pair``, and add
    the unchanged pair gradient back. This is coefficient-free PCGrad and
    leaves a non-conflicting transaction bit-identical.

    Candidate supervision is another current, real capture objective. The
    caller has already protected it.  A pair-only projection can nevertheless
    rotate that already-safe direction back through zero even when a direction
    satisfying both current objectives exists.  In that case repair pair and
    candidate jointly with the existing minimum-change halfspace solver.  A
    genuinely incompatible pair/candidate label set still fails loudly.
    """
    if not (
            len(proposed_grads)
            == len(pair_grads)
            == len(candidate_grads)):
        raise ValueError("standard pair gradient tuples have different lengths")
    pair_norm_sq = _gradient_tuple_dot(
        pair_grads,
        pair_grads,
        reference,
    )
    if not bool(torch.isfinite(pair_norm_sq).item()):
        raise FloatingPointError("non-finite standard pair gradient norm")
    if not bool((pair_norm_sq > 0.0).item()):
        raise RuntimeError("standard pair target produced no adjacency gradient")

    base_grads = []
    for proposed_grad, pair_grad in zip(proposed_grads, pair_grads):
        if proposed_grad is None and pair_grad is None:
            base_grads.append(None)
        elif proposed_grad is None:
            base_grads.append(-pair_grad.detach())
        elif pair_grad is None:
            base_grads.append(proposed_grad.detach().clone())
        else:
            base_grads.append(
                proposed_grad.detach().clone() - pair_grad.detach()
            )
    base_pair_dot = _gradient_tuple_dot(
        base_grads,
        pair_grads,
        reference,
    )
    if not bool(torch.isfinite(base_pair_dot).item()):
        raise FloatingPointError(
            "non-finite ordinary/pair gradient conflict diagnostic"
        )
    intervened = bool((base_pair_dot < 0.0).item())
    if intervened:
        projection_scale = base_pair_dot / pair_norm_sq
        base_grads = [
            (
                base_grad - projection_scale * pair_grad
                if base_grad is not None and pair_grad is not None
                else base_grad
            )
            for base_grad, pair_grad in zip(base_grads, pair_grads)
        ]
    protected_grads = []
    for base_grad, pair_grad in zip(base_grads, pair_grads):
        if base_grad is None:
            protected_grads.append(
                None if pair_grad is None else pair_grad.detach().clone()
            )
        elif pair_grad is None:
            protected_grads.append(base_grad)
        else:
            protected_grads.append(base_grad + pair_grad.detach())

    pair_dot_before = _gradient_tuple_dot(
        proposed_grads,
        pair_grads,
        reference,
    )
    pair_dot_after = _gradient_tuple_dot(
        protected_grads,
        pair_grads,
        reference,
    )
    if not bool(
            torch.isfinite(pair_dot_before).item()
            and torch.isfinite(pair_dot_after).item()):
        raise FloatingPointError(
            "non-finite protected standard pair gradient direction"
        )
    if not bool((pair_dot_after > 0.0).item()):
        raise RuntimeError(
            "standard adjacency gradient does not preserve strict pair descent"
        )

    candidate_norm_sq = _gradient_tuple_dot(
        candidate_grads,
        candidate_grads,
        reference,
    )
    candidate_dot_before = _gradient_tuple_dot(
        proposed_grads,
        candidate_grads,
        reference,
    )
    candidate_dot_after = _gradient_tuple_dot(
        protected_grads,
        candidate_grads,
        reference,
    )
    if not bool(
            torch.isfinite(candidate_norm_sq).item()
            and torch.isfinite(candidate_dot_before).item()
            and torch.isfinite(candidate_dot_after).item()):
        raise FloatingPointError(
            "non-finite joint pair/candidate gradient direction"
        )
    joint_repair_info = {
        "intervened": 0.0,
        "projection_delta_norm": 0.0,
    }
    if (
            bool((candidate_norm_sq > 0.0).item())
            and not bool((candidate_dot_after > 0.0).item())):
        if not bool((candidate_dot_before > 0.0).item()):
            raise RuntimeError(
                "standard pair projection received no protected current "
                "candidate descent"
            )
        current_constraints = (pair_grads, candidate_grads)
        current_floors = tuple(
            float(_strict_gradient_dot_floor(
                proposed_grads=protected_grads,
                constraint_grads=constraint,
                reference=reference,
            ).detach().cpu().item())
            for constraint in current_constraints
        )
        try:
            protected_grads, joint_repair_info = (
                _project_gradient_tuple_to_minimum_dots(
                    proposed_grads=protected_grads,
                    constraint_grads=current_constraints,
                    minimum_dots=current_floors,
                    reference=reference,
                    diagnostic_name=(
                        "standard current pair/candidate gradient repair"
                    ),
                )
            )
        except RuntimeError as error:
            raise RuntimeError(
                "standard pair projection conflicts with current candidate "
                "descent: {}".format(str(error))
            )
        pair_dot_after = _gradient_tuple_dot(
            protected_grads,
            pair_grads,
            reference,
        )
        candidate_dot_after = _gradient_tuple_dot(
            protected_grads,
            candidate_grads,
            reference,
        )
        if not bool(
                torch.isfinite(pair_dot_after).item()
                and torch.isfinite(candidate_dot_after).item()
                and (pair_dot_after > 0.0).item()
                and (candidate_dot_after > 0.0).item()):
            raise RuntimeError(
                "joint standard pair/candidate repair lost strict descent"
            )
    return tuple(protected_grads), {
        "intervened": float(
            intervened or bool(joint_repair_info["intervened"])
        ),
        "joint_repair_intervened": float(
            joint_repair_info["intervened"]
        ),
        "joint_repair_projection_delta_norm": float(
            joint_repair_info["projection_delta_norm"]
        ),
        "base_pair_dot": base_pair_dot,
        "pair_dot_before": pair_dot_before,
        "pair_dot_after": pair_dot_after,
        "candidate_dot_before": candidate_dot_before,
        "candidate_dot_after": candidate_dot_after,
    }


def _project_gradients_onto_nonincreasing_halfspaces(
        proposed_grads,
        constraint_grads,
        reference):
    """Preserve every cached objective without replaying any cached loss.

    For a proposed descent gradient ``g`` and cached objective gradients
    ``h_i``, a first-order no-forget update requires ``g' dot h_i >= 0`` for
    every cache entry.  The old implementation projected only against
    ``sum_i h_i``; that can improve the aggregate while worsening individual
    cached transitions.

    This routine deterministically grows an active set of violated constraints.
    For an active set ``A`` it removes only the component of the *original*
    proposed gradient in ``span({h_i | i in A})``::

        g' = g - H_A^T pinv(H_A H_A^T) H_A g.

    Active constraints therefore have zero first-order change, while the
    orthogonal base-learning component is retained.  New violated constraints
    are added until all individual inequalities hold.  No cached gradient is
    added as a descent objective, so outcome mass is not replayed.
    """
    original = [
        grad.detach().clone() if grad is not None else None
        for grad in proposed_grads
    ]
    constraints = []
    for grads in constraint_grads:
        detached = [
            grad.detach().clone() if grad is not None else None
            for grad in grads
        ]
        norm_sq = _gradient_tuple_dot(detached, detached, reference)
        if bool(torch.isfinite(norm_sq).item()) and bool((norm_sq > 0.0).item()):
            constraints.append((detached, norm_sq))
    if not constraints:
        return original, {
            "constraint_count": 0.0,
            "active_constraint_count": 0.0,
            "min_dot_before": 0.0,
            "min_dot_after": 0.0,
            "fallback": 0.0,
        }

    dtype_epsilon = _floating_dtype_epsilon(reference)
    original_norm_sq = _gradient_tuple_dot(original, original, reference)

    def _dots(grads):
        return torch.stack([
            _gradient_tuple_dot(grads, item[0], reference)
            for item in constraints
        ])

    before_dots = _dots(original)
    active = []
    projected = original
    fallback = False
    for _ in range(len(constraints)):
        dots = _dots(projected)
        tolerances = torch.stack([
            _gradient_dot_tolerance(
                proposed_grads=original,
                constraint_grads=item[0],
                reference=reference,
                tolerance_multiplier=64.0,
            )
            for item in constraints
        ])
        violated = torch.where(dots < -tolerances)[0]
        if violated.numel() == 0:
            break
        # Add every currently violated constraint.  At least one new index is
        # added per pass, so the loop is bounded by the number of cache rows.
        for index in violated.detach().cpu().tolist():
            if index not in active:
                active.append(index)
        if not active:
            break
        # Modified Gram-Schmidt avoids an SVD/pseudoinverse in the GPU
        # training path.  Besides being cheaper for the small cache, it remains
        # compatible with the older deterministic PyTorch build on the server.
        basis = []
        for index in active:
            residual = [
                grad.detach().clone() if grad is not None else None
                for grad in constraints[index][0]
            ]
            for basis_vector in basis:
                coefficient = _gradient_tuple_dot(
                    residual, basis_vector, reference
                )
                residual = [
                    (
                        grad - coefficient * basis_grad
                        if grad is not None and basis_grad is not None
                        else grad
                    )
                    for grad, basis_grad in zip(residual, basis_vector)
                ]
            residual_norm_sq = _gradient_tuple_dot(
                residual, residual, reference
            )
            independence_tolerance = (
                64.0
                * dtype_epsilon
                * constraints[index][1]
            )
            if bool((residual_norm_sq > independence_tolerance).item()):
                residual_norm = torch.sqrt(residual_norm_sq)
                basis.append([
                    (
                        grad / residual_norm if grad is not None else None
                    )
                    for grad in residual
                ])
        projected = [
            grad.detach().clone() if grad is not None else None
            for grad in original
        ]
        for basis_vector in basis:
            coefficient = _gradient_tuple_dot(
                original, basis_vector, reference
            )
            for parameter_index, basis_grad in enumerate(basis_vector):
                if basis_grad is None:
                    continue
                if projected[parameter_index] is None:
                    projected[parameter_index] = -coefficient * basis_grad
                else:
                    projected[parameter_index] = (
                        projected[parameter_index]
                        - coefficient * basis_grad
                    )

    after_dots = _dots(projected)
    tolerances = torch.stack([
        _gradient_dot_tolerance(
            proposed_grads=original,
            constraint_grads=item[0],
            reference=reference,
            tolerance_multiplier=64.0,
        )
        for item in constraints
    ])
    if bool(torch.any(after_dots < -tolerances).item()):
        # Singular or nearly collinear constraints can make the pinverse
        # residual larger than machine precision.  A zero update on only the
        # constraint-sensitive parameter tensors is always feasible; unrelated
        # base-only tensors retain their original gradient.
        fallback = True
        constrained_parameter = [False] * len(original)
        for constraint, _ in constraints:
            for index, grad in enumerate(constraint):
                constrained_parameter[index] |= grad is not None
        projected = [
            (
                torch.zeros_like(grad)
                if grad is not None and constrained_parameter[index]
                else grad
            )
            for index, grad in enumerate(original)
        ]
        after_dots = _dots(projected)
        if bool(torch.any(after_dots < -tolerances).item()):
            raise RuntimeError(
                "candidate lifecycle halfspace fallback is not feasible"
            )

    return projected, {
        "constraint_count": float(len(constraints)),
        "active_constraint_count": float(len(active)),
        "min_dot_before": float(before_dots.min().detach().cpu().item()),
        "min_dot_after": float(after_dots.min().detach().cpu().item()),
        "fallback": float(fallback),
    }


def _project_with_current_candidate_priority(
        proposed_grads,
        candidate_grads,
        lifecycle_constraint_grads,
        reference,
        additional_priority_grads=()):
    """Keep only lifecycle guards compatible with strict current descent.

    A finite no-forget constraint can oppose newer real capture evidence so
    exactly that their joint homogeneous feasible cone contains no strict
    descent direction for the current candidate objective.  The current real
    target has priority.  Starting with the newest cache row, accept a lifecycle
    constraint only if the joint projection still retains a machine-resolvable
    positive dot with every current priority gradient.  ``candidate_grads`` is
    the primary current objective; ``additional_priority_grads`` carries other
    strict current evidence such as a simultaneous pair target.  Rejected rows
    are returned to the caller for explicit cache supersession; they are never
    replayed as losses.
    """
    priority_grads = [candidate_grads] + list(additional_priority_grads)
    dtype_epsilon = (
        2.220446049250313e-16
        if reference.dtype == torch.float64
        else 1.1920928955078125e-7
    )
    proposed_norm_sq = _gradient_tuple_dot(
        proposed_grads, proposed_grads, reference
    )
    if not bool(torch.isfinite(proposed_norm_sq).item()):
        raise FloatingPointError(
            "non-finite current lifecycle proposed-gradient norm"
        )
    priority_norm_squares = []
    proposed_priority_dots = []
    for priority_index, priority_gradient in enumerate(priority_grads):
        priority_norm_sq = _gradient_tuple_dot(
            priority_gradient, priority_gradient, reference
        )
        proposed_priority_dot = _gradient_tuple_dot(
            proposed_grads, priority_gradient, reference
        )
        if not bool(
                torch.isfinite(priority_norm_sq).item()
                and torch.isfinite(proposed_priority_dot).item()):
            raise FloatingPointError(
                "non-finite current lifecycle priority diagnostic"
            )
        if not bool((priority_norm_sq > 0.0).item()):
            raise RuntimeError(
                "current lifecycle priority {} has no adjacency gradient".format(
                    priority_index
                )
            )
        priority_norm_squares.append(priority_norm_sq)
        proposed_priority_dots.append(proposed_priority_dot)

    # A realized Adam displacement is not guaranteed to descend every current
    # target even when the clipped backpropagated gradient does.  In
    # particular, per-target pair guards run after the raw Adam step.  Repair
    # all current priorities jointly *before* deciding which older lifecycle
    # rows are compatible; rejecting the raw displacement here makes the
    # lifecycle pre-check race the later pair guard (run115 priority-4
    # failure).  The floors are numerical strictness margins only.  The
    # downstream candidate/pair guard retains the real achieved descent floors.
    priority_scale = torch.sqrt(proposed_norm_sq.clamp_min(0.0))
    for priority_norm_sq in priority_norm_squares:
        priority_scale = (
            priority_scale
            + torch.sqrt(priority_norm_sq.clamp_min(0.0))
        )
    priority_minimum_dots = [
        float(
            (
                512.0
                * dtype_epsilon
                * priority_scale
                * torch.sqrt(priority_norm_sq)
            ).detach().cpu().item()
        )
        for priority_norm_sq in priority_norm_squares
    ]
    (
        priority_safe_grads,
        priority_repair_info,
    ) = _project_gradient_tuple_to_minimum_dots(
        proposed_grads=proposed_grads,
        constraint_grads=priority_grads,
        minimum_dots=priority_minimum_dots,
        reference=reference,
        diagnostic_name="current lifecycle priority repair",
    )
    accepted_indices = []
    superseded_indices = []
    projected = [
        grad.detach().clone() if grad is not None else None
        for grad in priority_safe_grads
    ]
    projection_info = {
        "constraint_count": 0.0,
        "active_constraint_count": 0.0,
        "min_dot_before": 0.0,
        "min_dot_after": 0.0,
        "fallback": 0.0,
    }
    # Cache order is insertion order; prefer the most recent finite evidence.
    for index in reversed(range(len(lifecycle_constraint_grads))):
        trial_indices = [index] + accepted_indices
        trial_projected, trial_info = (
            _project_gradients_onto_nonincreasing_halfspaces(
                proposed_grads=priority_safe_grads,
                constraint_grads=[
                    lifecycle_constraint_grads[item]
                    for item in trial_indices
                ],
                reference=reference,
            )
        )
        trial_norm_sq = _gradient_tuple_dot(
            trial_projected, trial_projected, reference
        )
        trial_priority_valid = True
        for priority_gradient, priority_norm_sq in zip(
                priority_grads, priority_norm_squares):
            trial_priority_dot = _gradient_tuple_dot(
                trial_projected, priority_gradient, reference
            )
            strict_tolerance = _gradient_dot_tolerance(
                proposed_grads=trial_projected,
                constraint_grads=priority_gradient,
                reference=reference,
                tolerance_multiplier=64.0,
            )
            if not bool(
                    torch.isfinite(trial_priority_dot).item()
                    and (trial_priority_dot > strict_tolerance).item()):
                trial_priority_valid = False
                break
        if trial_priority_valid:
            accepted_indices = trial_indices
            projected = trial_projected
            projection_info = trial_info
        else:
            superseded_indices.append(index)

    accepted_indices = sorted(accepted_indices)
    superseded_indices = sorted(superseded_indices)
    final_priority_dots = []
    for priority_gradient in priority_grads:
        final_priority_dot = _gradient_tuple_dot(
            projected, priority_gradient, reference
        )
        if not bool((final_priority_dot > 0.0).item()):
            raise RuntimeError(
                "current-priority projection lost strict current descent"
            )
        final_priority_dots.append(final_priority_dot)
    projection_info = dict(projection_info)
    projection_info["superseded_constraint_count"] = float(
        len(superseded_indices)
    )
    projection_info["current_candidate_dot_after"] = float(
        final_priority_dots[0].detach().cpu().item()
    )
    projection_info["current_priority_repair_intervened"] = float(
        priority_repair_info["intervened"]
    )
    projection_info["current_priority_min_dot_before"] = float(
        min(
            float(dot.detach().cpu().item())
            for dot in proposed_priority_dots
        )
    )
    projection_info["current_priority_min_dot_after_repair"] = float(
        min(
            float(_gradient_tuple_dot(
                priority_safe_grads,
                priority_gradient,
                reference,
            ).detach().cpu().item())
            for priority_gradient in priority_grads
        )
    )
    return (
        projected,
        accepted_indices,
        superseded_indices,
        projection_info,
    )


def _sync_adam_first_moment_to_executed_update(
        optimizer,
        parameters,
        parameter_before_step,
        raw_parameter_deltas):
    """Make Adam's first moment represent a post-guard parameter update.

    The candidate-safe guard may replace Adam's raw displacement after
    ``optimizer.step``.  Leaving ``exp_avg`` unchanged makes the next Adam
    update follow momentum for a displacement that was never executed.  For
    every corrected parameter, solve Adam's own update equation for the first
    moment which produces the *executed* displacement while retaining the
    current second moment and step counter::

        delta_safe = -step_size * exp_avg_safe / denom

    The supported optimizer is ``torch.optim.Adam`` with its standard
    bias-corrected denominator

        ``denom = sqrt(exp_avg_sq / bias_correction2) + eps``.

    The observed raw displacement is used only as a validation oracle.  It is
    deliberately *not* used to choose between algebraic candidates: fitting a
    formula from a small float32 displacement can switch layouts because of
    rounding rather than because the optimizer implementation changed.
    AMSGrad uses ``max_exp_avg_sq`` in the same equation.  This operation is
    only called for parameters carrying candidate gradients; base-only
    parameters and ``exp_avg_sq`` are untouched.
    """
    group_by_parameter = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in group_by_parameter:
                raise RuntimeError(
                    "Adam parameter appears in more than one parameter group"
                )
            group_by_parameter[parameter_id] = group

    if optimizer.__class__ is not torch.optim.Adam:
        raise RuntimeError(
            "candidate-safe state synchronization requires torch.optim.Adam"
        )

    records = []
    reconstruction_tolerance = 0.0
    for parameter, before, raw_delta in zip(
            parameters, parameter_before_step, raw_parameter_deltas):
        if before is None:
            continue
        group = group_by_parameter.get(id(parameter))
        if group is None:
            raise RuntimeError(
                "candidate parameter is absent from the Adam optimizer"
            )
        if bool(group.get("maximize", False)):
            raise RuntimeError(
                "candidate-safe Adam state synchronization does not "
                "support maximize=True"
            )
        for unsupported_flag in ("foreach", "fused", "capturable", "differentiable"):
            if bool(group.get(unsupported_flag, False)):
                raise RuntimeError(
                    "candidate-safe Adam state synchronization does not "
                    "support {}=True".format(unsupported_flag)
                )
        state = optimizer.state.get(parameter)
        if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError(
                "candidate-safe update requires initialized Adam state"
            )
        step_value = state.get("step", 0)
        if torch.is_tensor(step_value):
            step_value = int(step_value.detach().cpu().item())
        else:
            step_value = int(step_value)
        if step_value <= 0:
            raise RuntimeError(
                "candidate-safe update encountered a non-positive Adam step"
            )
        beta1, beta2 = group["betas"]
        bias_correction1 = 1.0 - float(beta1) ** step_value
        bias_correction2 = 1.0 - float(beta2) ** step_value
        learning_rate = float(group["lr"])
        if learning_rate <= 0.0 or bias_correction1 <= 0.0 \
                or bias_correction2 <= 0.0:
            raise RuntimeError(
                "invalid Adam scale for candidate-safe state synchronization"
            )
        second_moment = state["exp_avg_sq"]
        if bool(group.get("amsgrad", False)):
            if "max_exp_avg_sq" not in state:
                raise RuntimeError(
                    "AMSGrad candidate-safe update is missing max_exp_avg_sq"
                )
            second_moment = state["max_exp_avg_sq"]
        sqrt_second_moment = second_moment.sqrt()
        eps = float(group["eps"])
        corrected_denom = (
            sqrt_second_moment.div(math.sqrt(bias_correction2)).add(eps)
        )
        corrected_step_size = learning_rate / bias_correction1
        old_exp_avg = state["exp_avg"].detach().clone()
        if parameter.dtype == torch.float64:
            dtype_epsilon = 2.220446049250313e-16
            dtype_tiny = 2.2250738585072014e-308
        elif parameter.dtype == torch.float16:
            dtype_epsilon = 9.765625e-4
            dtype_tiny = 5.960464477539063e-8
        else:
            # Candidate graph parameters are float32 in the supported runs.
            # This also safely covers older torch builds without torch.finfo.
            dtype_epsilon = 1.1920928955078125e-7
            dtype_tiny = 1.1754943508222875e-38
        records.append({
            "parameter": parameter,
            "before": before,
            "raw_delta": raw_delta,
            "state": state,
            "corrected_denom": corrected_denom,
            "corrected_step_size": corrected_step_size,
            "old_exp_avg": old_exp_avg,
            "dtype_epsilon": dtype_epsilon,
            "dtype_tiny": dtype_tiny,
        })

    synchronized_count = 0
    raw_reconstruction_error = 0.0
    safe_reconstruction_error = 0.0
    raw_reconstruction_error_ratio = 0.0
    safe_reconstruction_error_ratio = 0.0
    exp_avg_change_norm_sq = 0.0
    with torch.no_grad():
        for record in records:
            parameter = record["parameter"]
            before = record["before"]
            raw_delta = record["raw_delta"]
            state = record["state"]
            old_exp_avg = record["old_exp_avg"]
            denom = record["corrected_denom"]
            step_size = record["corrected_step_size"]
            dtype_epsilon = record["dtype_epsilon"]
            dtype_tiny = record["dtype_tiny"]
            reconstructed_raw_delta = (
                -step_size * old_exp_avg / denom
            )
            raw_error_tensor = (
                reconstructed_raw_delta - raw_delta
            ).abs()
            reconstructed_raw_after = before + reconstructed_raw_delta
            raw_tolerance_tensor = (
                16.0
                * dtype_epsilon
                * (
                    before.abs()
                    + reconstructed_raw_after.abs()
                    + raw_delta.abs()
                    + reconstructed_raw_delta.abs()
                )
                + dtype_tiny
            )
            raw_error = raw_error_tensor.max().item()
            raw_ratio = (
                raw_error_tensor / raw_tolerance_tensor
            ).max().item()
            raw_reconstruction_error = max(
                raw_reconstruction_error, float(raw_error)
            )
            raw_reconstruction_error_ratio = max(
                raw_reconstruction_error_ratio, float(raw_ratio)
            )
            reconstruction_tolerance = max(
                reconstruction_tolerance,
                float(raw_tolerance_tensor.max().item()),
            )

            safe_delta = parameter.detach() - before
            safe_exp_avg = -safe_delta * denom / step_size
            state["exp_avg"].copy_(safe_exp_avg)
            exp_avg_change_norm_sq += float(
                (safe_exp_avg - old_exp_avg).pow(2).sum().cpu().item()
            )
            reconstructed_safe_delta = (
                -step_size * state["exp_avg"] / denom
            )
            safe_error_tensor = (
                reconstructed_safe_delta - safe_delta
            ).abs()
            safe_tolerance_tensor = (
                16.0
                * dtype_epsilon
                * (
                    before.abs()
                    + parameter.detach().abs()
                    + safe_delta.abs()
                    + reconstructed_safe_delta.abs()
                )
                + dtype_tiny
            )
            safe_error = safe_error_tensor.max().item()
            safe_ratio = (
                safe_error_tensor / safe_tolerance_tensor
            ).max().item()
            safe_reconstruction_error = max(
                safe_reconstruction_error, float(safe_error)
            )
            safe_reconstruction_error_ratio = max(
                safe_reconstruction_error_ratio, float(safe_ratio)
            )
            reconstruction_tolerance = max(
                reconstruction_tolerance,
                float(safe_tolerance_tensor.max().item()),
            )
            synchronized_count += 1

    if synchronized_count <= 0:
        raise RuntimeError(
            "candidate-safe correction did not synchronize any Adam parameter"
        )
    return {
        "parameter_count": float(synchronized_count),
        "update_equation_version": 2.0,
        "raw_reconstruction_error": float(raw_reconstruction_error),
        "safe_reconstruction_error": float(safe_reconstruction_error),
        "raw_reconstruction_error_ratio": float(
            raw_reconstruction_error_ratio
        ),
        "safe_reconstruction_error_ratio": float(
            safe_reconstruction_error_ratio
        ),
        "reconstruction_tolerance": float(reconstruction_tolerance),
        "exp_avg_change_norm": float(exp_avg_change_norm_sq ** 0.5),
    }


def _validate_optimizer_step_pair_credit(pair_local_delta):
    """Fail unless one Adam transaction sees a centered pair contrast."""
    if not bool(torch.isfinite(pair_local_delta).all().item()):
        raise FloatingPointError("non-finite optimizer-step signed pair credit")
    positive_mass = torch.clamp(pair_local_delta, min=0.0).sum()
    negative_mass = torch.clamp(-pair_local_delta, min=0.0).sum()
    absolute_mass = positive_mass + negative_mass
    centered_error = torch.abs(positive_mass - negative_mass)
    nonzero = bool((absolute_mass > 0.0).item())
    class_complete = bool(
        (positive_mass > 0.0).item() and (negative_mass > 0.0).item()
    )
    tolerance = (
        absolute_mass.new_tensor(1.0e-5)
        * torch.clamp(absolute_mass, min=1.0)
    )
    if nonzero and not class_complete:
        raise RuntimeError(
            "one-sided optimizer transaction received non-zero pair loss"
        )
    if bool((centered_error > tolerance).item()):
        raise RuntimeError(
            "optimizer-step pair credit lost signed mass conservation"
        )
    return {
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "centered_error": centered_error,
        "class_complete": float(class_complete),
        "contract_valid": 1.0,
    }


def compute_identity_local_factor_ppo_loss(
        factor_imp_weights,
        clipped_factor_imp_weights,
        identity_local_delta,
        factor_loss_mask,
        transition_mask,
        normalize_by_target_transitions=False):
    """Return an exact-identity factor PPO term with transition normalization.

    Identity-local outcome mass is normalized before it reaches the trainer.
    Dividing its sparse factor surrogate by *all* selected factors makes the
    same episode-level supervision shrink whenever unrelated factor slots are
    added.  Capture outcome keeps the graph-compatible valid-transition
    denominator.  Sparse signed-pair supervision instead opts into the exact
    target-bearing-transition population so unrelated episodes cannot dilute
    the only updates that distinguish successful and failed capture backbones.

    Outcome sign is preserved by the standard PPO min surrogate.  The helper
    never changes reward, Q targets, priorities, or non-target factor values.
    """
    tensors = (
        factor_imp_weights,
        clipped_factor_imp_weights,
        identity_local_delta,
        factor_loss_mask,
    )
    reference_shape = factor_imp_weights.shape
    if any(tensor.shape != reference_shape for tensor in tensors[1:]):
        raise ValueError(
            "identity-local factor PPO tensors must share shape {}, got {}"
            .format(
                reference_shape,
                [tuple(tensor.shape) for tensor in tensors],
            )
        )
    if transition_mask.dim() == 1:
        transition_mask = transition_mask.unsqueeze(-1)
    if (
            transition_mask.dim() != 2
            or transition_mask.shape[0] != reference_shape[0]
            or transition_mask.shape[1] != 1):
        raise ValueError(
            "transition_mask must have shape [{}, 1], got {}".format(
                reference_shape[0], tuple(transition_mask.shape)
            )
        )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise FloatingPointError("non-finite identity-local factor PPO input")
    if not bool(torch.isfinite(transition_mask).all().item()):
        raise FloatingPointError("non-finite identity-local transition mask")
    target_mask = identity_local_delta.abs() > 0.0
    if bool(torch.any(target_mask & (factor_loss_mask <= 0.0)).item()):
        raise RuntimeError(
            "identity-local delta reached a padded or invalid factor"
        )
    valid_transition_count = transition_mask.sum()
    target_transition_mask = (
        (target_mask.float().sum(dim=-1, keepdim=True) > 0.0).float()
        * (transition_mask > 0.0).float()
    )
    target_transition_count = target_transition_mask.sum()
    if (
            bool(torch.any(target_mask).item())
            and not bool((valid_transition_count > 0.0).item())):
        raise RuntimeError(
            "identity-local delta has no valid transition denominator"
        )

    surr1 = factor_imp_weights * identity_local_delta
    surr2 = clipped_factor_imp_weights * identity_local_delta
    min_surr = torch.min(surr1, surr2)
    if (
            bool(torch.any(target_mask).item())
            and not bool((target_transition_count > 0.0).item())):
        raise RuntimeError(
            "identity-local delta has no target-bearing transition"
        )
    normalization_denominator = (
        target_transition_count
        if bool(normalize_by_target_transitions)
        else valid_transition_count
    )
    denominator = normalization_denominator.clamp_min(1.0)
    masked_surr = min_surr * factor_loss_mask
    loss = -masked_surr.sum() / denominator
    positive_loss = -(
        masked_surr * (identity_local_delta > 0.0).float()
    ).sum() / denominator
    negative_loss = -(
        masked_surr * (identity_local_delta < 0.0).float()
    ).sum() / denominator
    target_count = (
        target_mask.float() * (factor_loss_mask > 0.0).float()
    ).sum()
    valid_factor_count = (
        (factor_loss_mask > 0.0).float().sum()
    )
    factors_per_transition = (
        valid_factor_count / valid_transition_count.clamp_min(1.0)
    )
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "min_surr": min_surr,
        "target_count": target_count,
        "target_transition_count": target_transition_count,
        "valid_transition_count": valid_transition_count,
        "normalization_denominator": normalization_denominator,
        "normalization_uses_target_transitions": float(
            bool(normalize_by_target_transitions)
        ),
        "factors_per_transition": factors_per_transition,
    }


def _base_factor_population_diagnostics(
        base_factor_min_surr,
        factor_training_mask,
        factor_loss_mask,
        transition_mask,
        replay_population_provenance):
    """Split the base-factor population without changing its objective.

    This helper is deliberately diagnostics-only: it partitions the exact
    numerator and denominator already used by ``base_factor_rl_loss`` into
    pair-evidence and non-pair replay populations.  It never modifies or feeds
    a tensor back into the training path; all diagnostic views are detached
    before any extra reduction is built.
    """
    base_factor_min_surr = base_factor_min_surr.detach()
    factor_training_mask = factor_training_mask.detach()
    factor_loss_mask = factor_loss_mask.detach()
    transition_mask = transition_mask.detach()
    replay_population_provenance = replay_population_provenance.detach()
    tensors = (
        base_factor_min_surr,
        factor_training_mask,
        factor_loss_mask,
        transition_mask,
        replay_population_provenance,
    )
    if base_factor_min_surr.dim() != 2:
        raise ValueError(
            "base_factor_min_surr must be rank 2, got {}".format(
                tuple(base_factor_min_surr.shape)
            )
        )
    reference_shape = base_factor_min_surr.shape
    for name, tensor in (
            ("factor_training_mask", factor_training_mask),
            ("factor_loss_mask", factor_loss_mask)):
        if tensor.shape != reference_shape:
            raise ValueError(
                "{} must have shape {}, got {}".format(
                    name,
                    tuple(reference_shape),
                    tuple(tensor.shape),
                )
            )
    if (
            transition_mask.dim() != 2
            or transition_mask.shape[0] != reference_shape[0]
            or transition_mask.shape[1] != 1):
        raise ValueError(
            "transition_mask must have shape [{}, 1], got {}".format(
                reference_shape[0],
                tuple(transition_mask.shape),
            )
        )
    if (
            replay_population_provenance.dim() != 2
            or replay_population_provenance.shape[0] != reference_shape[0]
            or replay_population_provenance.shape[1] != 3):
        raise ValueError(
            "replay_population_provenance must have shape [{}, 3], got {}"
            .format(
                reference_shape[0],
                tuple(replay_population_provenance.shape),
            )
        )
    pair_evidence_transition = replay_population_provenance[:, 0:1]
    episode_ordinal_transition = replay_population_provenance[:, 1:2]
    for name, tensor in (
            ("pair_evidence_transition", pair_evidence_transition),
            ("episode_ordinal_transition", episode_ordinal_transition)):
        if (
                tensor.dim() != 2
                or tensor.shape[0] != reference_shape[0]
                or tensor.shape[1] != 1):
            raise ValueError(
                "{} must have shape [{}, 1], got {}".format(
                    name,
                    reference_shape[0],
                    tuple(tensor.shape),
                )
            )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise FloatingPointError(
            "non-finite base-factor population diagnostic input"
        )
    binary_pair_evidence = (
        (pair_evidence_transition == 0.0)
        | (pair_evidence_transition == 1.0)
    )
    if not bool(binary_pair_evidence.all().item()):
        raise RuntimeError(
            "pair-evidence transition provenance must be exactly binary"
        )
    if (
            bool(torch.any(episode_ordinal_transition < 0.0).item())
            or not bool(torch.all(
                episode_ordinal_transition
                == torch.round(episode_ordinal_transition)
            ).item())):
        raise RuntimeError(
            "episode ordinal transition provenance must be non-negative "
            "integers"
        )

    # These tensors are detached diagnostics only.  Accumulate the population
    # split in float64 so that strong cancellation cannot make the partition
    # check depend on float32 reduction order.  The training loss dtype and
    # gradient path are unchanged.
    diagnostic_output_dtype = base_factor_min_surr.dtype
    diagnostic_dtype = torch.float64
    base_factor_min_surr = base_factor_min_surr.to(
        dtype=diagnostic_dtype
    )
    factor_training_mask = factor_training_mask.to(
        dtype=diagnostic_dtype
    )
    factor_loss_mask = factor_loss_mask.to(dtype=diagnostic_dtype)
    transition_mask = transition_mask.to(dtype=diagnostic_dtype)
    pair_evidence_transition = pair_evidence_transition.to(
        dtype=diagnostic_dtype
    )

    pair_transition_selector = pair_evidence_transition
    non_pair_transition_selector = 1.0 - pair_transition_selector
    pair_factor_selector = pair_transition_selector.expand_as(
        factor_training_mask
    )
    non_pair_factor_selector = non_pair_transition_selector.expand_as(
        factor_training_mask
    )
    valid_transition_mask = (
        transition_mask > 0.0
    ).to(dtype=diagnostic_dtype)
    valid_factor_mask = (
        factor_training_mask > 0.0
    ).to(dtype=diagnostic_dtype)
    weighted_surr = base_factor_min_surr * factor_loss_mask

    pair_valid_transition_count = (
        valid_transition_mask * pair_transition_selector
    ).sum()
    non_pair_valid_transition_count = (
        valid_transition_mask * non_pair_transition_selector
    ).sum()
    pair_valid_factor_mask_count = (
        valid_factor_mask * pair_factor_selector
    ).sum()
    non_pair_valid_factor_mask_count = (
        valid_factor_mask * non_pair_factor_selector
    ).sum()
    pair_weighted_denominator = (
        factor_loss_mask * pair_factor_selector
    ).sum()
    non_pair_weighted_denominator = (
        factor_loss_mask * non_pair_factor_selector
    ).sum()
    pair_numerator = -(
        weighted_surr * pair_factor_selector
    ).sum()
    non_pair_numerator = -(
        weighted_surr * non_pair_factor_selector
    ).sum()

    episode_ordinals = sorted(set(
        int(value)
        for value in episode_ordinal_transition.view(-1).cpu().tolist()
    ))
    if not episode_ordinals:
        raise RuntimeError("replay population provenance is empty")
    episode_weighted_denominators = []
    episode_pair_flags = []
    for episode_ordinal in episode_ordinals:
        episode_transition_selector = (
            episode_ordinal_transition == float(episode_ordinal)
        ).to(dtype=diagnostic_dtype)
        episode_factor_selector = episode_transition_selector.expand_as(
            factor_loss_mask
        )
        episode_weighted_denominators.append(
            (factor_loss_mask * episode_factor_selector).sum()
        )
        episode_pair_values = pair_evidence_transition[
            episode_transition_selector > 0.0
        ]
        if not bool(torch.all(
                episode_pair_values == episode_pair_values[0]).item()):
            raise RuntimeError(
                "pair-evidence provenance changed within an episode"
            )
        episode_pair_flags.append(episode_pair_values[0])
    episode_weighted_denominators = torch.stack(
        episode_weighted_denominators
    )
    episode_pair_flags = torch.stack(episode_pair_flags)
    total_weight_for_fraction = factor_loss_mask.sum().clamp_min(1.0)
    normalized_episode_weights = (
        episode_weighted_denominators / total_weight_for_fraction
    )
    pair_episode_mask = episode_pair_flags > 0.0
    non_pair_episode_mask = ~pair_episode_mask

    def _episode_weight_stats(group_mask):
        group_weights = normalized_episode_weights[group_mask]
        if group_weights.numel() == 0:
            zero = normalized_episode_weights.new_tensor(0.0)
            return zero, zero, zero, zero
        return (
            group_weights.new_tensor(float(group_weights.numel())),
            group_weights.mean(),
            group_weights.min(),
            group_weights.max(),
        )

    pair_episode_count, pair_episode_weight_mean, \
        pair_episode_weight_min, pair_episode_weight_max = (
            _episode_weight_stats(pair_episode_mask)
        )
    non_pair_episode_count, non_pair_episode_weight_mean, \
        non_pair_episode_weight_min, non_pair_episode_weight_max = (
            _episode_weight_stats(non_pair_episode_mask)
        )

    total_valid_transition_count = valid_transition_mask.sum()
    total_valid_factor_mask_count = valid_factor_mask.sum()
    total_weighted_denominator = factor_loss_mask.sum()
    total_numerator = -weighted_surr.sum()
    decompositions = (
        (
            "valid_transition_count",
            pair_valid_transition_count + non_pair_valid_transition_count,
            total_valid_transition_count,
            valid_transition_mask.abs().sum(),
        ),
        (
            "valid_factor_mask_count",
            pair_valid_factor_mask_count + non_pair_valid_factor_mask_count,
            total_valid_factor_mask_count,
            valid_factor_mask.abs().sum(),
        ),
        (
            "weighted_denominator",
            pair_weighted_denominator + non_pair_weighted_denominator,
            total_weighted_denominator,
            factor_loss_mask.abs().sum(),
        ),
        (
            "numerator",
            pair_numerator + non_pair_numerator,
            total_numerator,
            weighted_surr.abs().sum(),
        ),
    )
    for (
            decomposition_name,
            split_total,
            full_total,
            reduction_scale) in decompositions:
        tolerance = 1e-10 * max(
            float(reduction_scale.item()),
            1.0,
        )
        difference = abs(float((split_total - full_total).item()))
        if difference > tolerance:
            raise RuntimeError(
                "pair/non-pair base-factor population split failed to "
                "reconstruct the full transaction "
                "(decomposition={}, difference={}, tolerance={})".format(
                    decomposition_name,
                    difference,
                    tolerance,
                )
            )

    def _restore_output_dtype(value):
        return value.to(dtype=diagnostic_output_dtype)

    return {
        "pair_valid_transition_count": _restore_output_dtype(
            pair_valid_transition_count
        ),
        "non_pair_valid_transition_count": _restore_output_dtype(
            non_pair_valid_transition_count
        ),
        "pair_valid_factor_mask_count": _restore_output_dtype(
            pair_valid_factor_mask_count
        ),
        "non_pair_valid_factor_mask_count": _restore_output_dtype(
            non_pair_valid_factor_mask_count
        ),
        "pair_weighted_denominator": _restore_output_dtype(
            pair_weighted_denominator
        ),
        "non_pair_weighted_denominator": _restore_output_dtype(
            non_pair_weighted_denominator
        ),
        "pair_numerator": _restore_output_dtype(pair_numerator),
        "non_pair_numerator": _restore_output_dtype(non_pair_numerator),
        "pair_episode_count": _restore_output_dtype(pair_episode_count),
        "non_pair_episode_count": _restore_output_dtype(
            non_pair_episode_count
        ),
        "pair_episode_weight_mean": _restore_output_dtype(
            pair_episode_weight_mean
        ),
        "pair_episode_weight_min": _restore_output_dtype(
            pair_episode_weight_min
        ),
        "pair_episode_weight_max": _restore_output_dtype(
            pair_episode_weight_max
        ),
        "non_pair_episode_weight_mean": _restore_output_dtype(
            non_pair_episode_weight_mean
        ),
        "non_pair_episode_weight_min": _restore_output_dtype(
            non_pair_episode_weight_min
        ),
        "non_pair_episode_weight_max": _restore_output_dtype(
            non_pair_episode_weight_max
        ),
        "contract_valid": 1.0,
    }


def _separate_replay_graph_and_factor_advantages(
        factor_advantage,
        replay_graph_advantage,
        factor_mask,
        graph_return_coefficient):
    """Keep graph-return PPO separate from factor-local replay credit.

    ``AdjBuffer`` stores standardized graph-return advantage in its own replay
    field. ``factor_advantage`` is a combined field: repeated graph return plus
    centered factor-Q and delayed identity-local credit. Averaging that combined
    field silently broadcasts any non-zero-mean local credit into structured
    graph PPO.
    """
    if factor_advantage.dim() != 2:
        raise RuntimeError(
            "factor advantage must have shape [batch, factor], got {}"
            .format(tuple(factor_advantage.shape))
        )
    if replay_graph_advantage.dim() != 2:
        raise RuntimeError(
            "replay graph advantage must have shape [batch, 1], got {}"
            .format(tuple(replay_graph_advantage.shape))
        )
    if replay_graph_advantage.shape[0] != factor_advantage.shape[0]:
        raise RuntimeError(
            "graph/factor replay batch sizes differ: {} vs {}"
            .format(
                replay_graph_advantage.shape[0],
                factor_advantage.shape[0],
            )
        )
    if replay_graph_advantage.shape[1] != 1:
        raise RuntimeError(
            "replay graph advantage must contain one value per transition"
        )
    if factor_mask.shape != factor_advantage.shape:
        raise RuntimeError(
            "factor advantage/mask shapes differ: {} vs {}"
            .format(
                tuple(factor_advantage.shape),
                tuple(factor_mask.shape),
            )
        )
    coefficient = float(graph_return_coefficient)
    if not np.isfinite(coefficient):
        raise FloatingPointError("graph return advantage coefficient is non-finite")
    for name, tensor in (
            ("factor advantage", factor_advantage),
            ("replay graph advantage", replay_graph_advantage),
            ("factor mask", factor_mask)):
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError("non-finite {}".format(name))

    graph_advantage = replay_graph_advantage * coefficient
    local_factor_advantage = (
        factor_advantage - graph_advantage
    ) * factor_mask
    valid_factor_count = factor_mask.sum(dim=-1, keepdim=True)
    legacy_graph_advantage = (
        (factor_advantage * factor_mask).sum(dim=-1, keepdim=True)
        / valid_factor_count.clamp_min(1.0)
    )
    has_valid_factor = valid_factor_count > 0.0
    legacy_graph_advantage = torch.where(
        has_valid_factor,
        legacy_graph_advantage,
        torch.zeros_like(legacy_graph_advantage),
    )
    graph_advantage_contamination = torch.where(
        has_valid_factor,
        legacy_graph_advantage - graph_advantage,
        torch.zeros_like(graph_advantage),
    )
    return {
        "graph_advantage": graph_advantage,
        "local_factor_advantage": local_factor_advantage,
        "legacy_graph_advantage": legacy_graph_advantage,
        "graph_advantage_contamination": graph_advantage_contamination,
    }


def compute_capture_outcome_factor_ppo_loss(
        factor_imp_weights,
        clipped_factor_imp_weights,
        capture_outcome_local_delta,
        factor_loss_mask,
        transition_mask):
    """Stable public entry for the exact-identity local factor objective.

    Existing server preflight scripts import this name and pass the historical
    ``capture_outcome_local_delta`` keyword.  Forward it explicitly to the
    generic implementation so old callers and the new signed-pair branch use
    exactly the same mathematics; this is not an algorithm fallback.
    """
    return compute_identity_local_factor_ppo_loss(
        factor_imp_weights=factor_imp_weights,
        clipped_factor_imp_weights=clipped_factor_imp_weights,
        identity_local_delta=capture_outcome_local_delta,
        factor_loss_mask=factor_loss_mask,
        transition_mask=transition_mask,
        normalize_by_target_transitions=False,
    )


def compute_capture_candidate_identity_active_competitor_loss(
        candidate_competitor_margins,
        candidate_reference_margins,
        candidate_identity_delta,
        candidate_valid_mask,
        transition_mask):
    """One-sided first-reachable competitor loss for candidate identities.

    ``candidate_competitor_margins`` compares an inactive target with the
    hardest legal alternative at the first replay prefix where the target can
    enter. Positive margin means the target is the greedy choice at that slot.
    Negative targets use the same geometry with the sign reversed.

    A wrong-side behavior reference is anchored at the real competitor boundary
    zero; a better reference is preserved:

        signed_goal = max(sign(delta) * reference_margin, 0)
        violation = relu(
            signed_goal - sign(delta) * current_margin
        )
        loss = abs(delta) * violation.

    Unlike the previous softplus objective, the auxiliary gradient is exactly
    zero once the target reaches its boundary/reference. This prevents already
    achieved sparse targets from continuing to displace the task gradient.
    Lifecycle protection uses the separately returned signed-margin constraint,
    whose gradient remains defined even at an achieved reference.
    """
    if (
            candidate_competitor_margins.shape != candidate_identity_delta.shape
            or candidate_competitor_margins.shape
            != candidate_reference_margins.shape
            or candidate_competitor_margins.shape != candidate_valid_mask.shape):
        raise ValueError(
            "candidate competitor/reference/delta/mask shapes must match: "
            "{}, {}, {}, {}"
            .format(
                tuple(candidate_competitor_margins.shape),
                tuple(candidate_reference_margins.shape),
                tuple(candidate_identity_delta.shape),
                tuple(candidate_valid_mask.shape),
            )
        )
    if transition_mask.dim() == 1:
        transition_mask = transition_mask.unsqueeze(-1)
    if (
            transition_mask.dim() != 2
            or transition_mask.shape[0] != candidate_competitor_margins.shape[0]
            or transition_mask.shape[1] != 1):
        raise ValueError(
            "candidate transition_mask must have shape [{}, 1], got {}"
            .format(
                candidate_competitor_margins.shape[0],
                tuple(transition_mask.shape),
            )
        )
    for tensor in (
            candidate_competitor_margins,
            candidate_reference_margins,
            candidate_identity_delta,
            candidate_valid_mask,
            transition_mask):
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError("non-finite candidate identity loss input")
    target_mask = candidate_identity_delta.abs() > 0.0
    if bool(torch.any(target_mask & (candidate_valid_mask <= 0.0)).item()):
        raise RuntimeError(
            "candidate-only outcome delta has no first-reachable active "
            "competitor"
        )
    if bool(torch.any(target_mask & (transition_mask <= 0.0)).item()):
        raise RuntimeError(
            "candidate-only outcome delta targets a padded or invalid "
            "transition"
        )
    valid_transition_count = transition_mask.sum()
    effective_valid = (
        (candidate_valid_mask > 0.0)
        & (transition_mask > 0.0)
    )
    target_transition = target_mask.any(dim=1)
    target_transition_mask = (
        target_transition.unsqueeze(-1) & (transition_mask > 0.0)
    )
    target_transition_count = target_transition_mask.float().sum()
    if (
            bool(torch.any(target_mask).item())
            and not bool((target_transition_count > 0.0).item())):
        raise RuntimeError(
            "candidate-only outcome delta has no supervised transition "
            "denominator"
        )
    valid_candidate_count = effective_valid.long().sum(dim=1)
    if bool(torch.any(target_transition & (valid_candidate_count <= 0)).item()):
        raise RuntimeError(
            "candidate-only outcome target has no valid conditional "
            "candidate set"
        )

    positive_weight = candidate_identity_delta.clamp_min(0.0)
    negative_weight = (-candidate_identity_delta).clamp_min(0.0)
    target_weight = candidate_identity_delta.abs()
    target_sign = torch.sign(candidate_identity_delta)
    detached_reference_margins = candidate_reference_margins.detach()
    reference_signed_margin = target_sign * detached_reference_margins
    required_signed_margin = reference_signed_margin.clamp_min(0.0)
    goal_margins = target_sign * required_signed_margin
    relative_margin = candidate_competitor_margins - goal_margins
    signed_margin = target_sign * relative_margin
    effective_mask = effective_valid.to(candidate_competitor_margins.dtype)
    violation = torch.relu(-signed_margin)
    unsatisfied_target_mask = (
        (violation > 0.0) & (effective_mask > 0.0)
    )
    unsatisfied_target_transition_count = (
        unsatisfied_target_mask.any(dim=1).float().sum()
    )
    per_target_loss = target_weight * violation
    per_target_constraint = (
        -target_weight * target_sign * candidate_competitor_margins
    )
    denominator = unsatisfied_target_transition_count.clamp_min(1.0)
    positive_loss = (
        per_target_loss
        * (candidate_identity_delta > 0.0).to(per_target_loss.dtype)
        * effective_mask
    ).sum() / denominator
    negative_loss = (
        per_target_loss
        * (candidate_identity_delta < 0.0).to(per_target_loss.dtype)
        * effective_mask
    ).sum() / denominator
    loss = positive_loss + negative_loss
    positive_target_mask = (
        (candidate_identity_delta > 0.0).float() * effective_mask
    )
    negative_target_mask = (
        (candidate_identity_delta < 0.0).float() * effective_mask
    )
    positive_denominator = positive_target_mask.sum().clamp_min(1.0)
    negative_denominator = negative_target_mask.sum().clamp_min(1.0)
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "target_count": (
            target_mask.float() * (effective_mask > 0.0).float()
        ).sum(),
        "valid_transition_count": valid_transition_count,
        "target_transition_count": target_transition_count,
        "target_transition_fraction": (
            target_transition_count / valid_transition_count.clamp_min(1.0)
        ),
        "positive_mass": positive_weight.sum(),
        "negative_mass": negative_weight.sum(),
        "margin": candidate_competitor_margins,
        "reference_margin": detached_reference_margins,
        "goal_margin": goal_margins,
        "relative_margin": relative_margin,
        "competitor_win_probability": torch.sigmoid(
            candidate_competitor_margins
        ),
        "signed_margin": signed_margin,
        "unsatisfied_target_count": (
            unsatisfied_target_mask.float().sum()
        ),
        "positive_unsatisfied_target_count": (
            unsatisfied_target_mask
            & (candidate_identity_delta > 0.0)
        ).float().sum(),
        "negative_unsatisfied_target_count": (
            unsatisfied_target_mask
            & (candidate_identity_delta < 0.0)
        ).float().sum(),
        "unsatisfied_target_transition_count": (
            unsatisfied_target_transition_count
        ),
        "unsatisfied_target_mask": unsatisfied_target_mask,
        "per_transition_loss": (
            per_target_loss * effective_mask
        ).sum(dim=1),
        "per_transition_constraint": (
            per_target_constraint * effective_mask
        ).sum(dim=1),
        "positive_margin_mean": (
            (relative_margin * positive_target_mask).sum()
            / positive_denominator
        ),
        "negative_margin_mean": (
            (relative_margin * negative_target_mask).sum()
            / negative_denominator
        ),
        "positive_signed_margin_mean": (
            (signed_margin * positive_target_mask).sum()
            / positive_denominator
        ),
        "negative_signed_margin_mean": (
            (signed_margin * negative_target_mask).sum()
            / negative_denominator
        ),
        "positive_boundary_crossed_fraction": (
            (
            (candidate_competitor_margins > 0.0).float()
                * positive_target_mask
            ).sum()
            / positive_denominator
        ),
        "negative_boundary_respected_fraction": (
            (
            (candidate_competitor_margins < 0.0).float()
                * negative_target_mask
            ).sum()
            / negative_denominator
        ),
    }


def _infer_inactive_dones_from_obs(obs_seq: torch.Tensor) -> torch.Tensor:
    """
    Infer fixed-capacity inactive slots from Wolfpack padded observations.

    In intra-episode dynamic Wolfpack, inactive / empty capacity slots are
    represented by an all -1 observation vector.  The replay buffer stores
    agent dones for the *next* state only, so the initial state s0 must be
    reconstructed from observations; otherwise the learner treats empty
    capacity slots as live agents at the beginning of every episode.

    Args:
        obs_seq: [B, T, N, obs_dim] or [B, N, obs_dim].
    Returns:
        Float mask with shape obs_seq[..., :1], where 1 means inactive/done.
    """
    obs_seq = to_torch(obs_seq)
    finite_obs = torch.where(torch.isfinite(obs_seq), obs_seq, torch.zeros_like(obs_seq))
    return torch.all(finite_obs <= -0.999, dim=-1, keepdim=True).float()


class R_SDDFG:
    def __init__(self, args, num_agents, policies, adj_network, policy_mapping_fn, device=torch.device("cuda:0"),
                 episode_length=25, vdn=False):
        """
        Trainer class for QMix with MLP policies. See parent class for more information.
        :param vdn: (bool) whether the algorithm in use is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.clip_param = self.args.clip_param
        self.adj_order_adv_coef = max(
            0.0,
            float(getattr(self.args, "adj_order_adv_coef", 0.0)),
        )
        self.adj_order_adv_positive_only = bool(
            getattr(self.args, "adj_order_adv_positive_only", False)
        )
        self.adj_order_adv_negative_coef = max(
            0.0,
            float(getattr(self.args, "adj_order_adv_negative_coef", 0.0)),
        )
        self.adj_order_adv_require_positive_graph_adv = bool(
            getattr(
                self.args,
                "adj_order_adv_require_positive_graph_adv",
                False,
            )
        )
        graph_gate_mode = str(
            getattr(self.args, "adj_order_adv_graph_gate_mode", "binary")
        ).lower()
        if graph_gate_mode not in ("binary", "soft"):
            graph_gate_mode = "binary"
        self.adj_order_adv_graph_gate_mode = graph_gate_mode
        self.adj_order_adv_graph_gate_scale = max(
            1e-6,
            float(getattr(self.args, "adj_order_adv_graph_gate_scale", 1.0)),
        )
        self.use_adj_triplet_graph_return_credit = bool(
            getattr(self.args, "use_adj_triplet_graph_return_credit", False)
        )
        self.adj_triplet_graph_return_credit_coef = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_triplet_graph_return_credit_coef",
                    0.0,
                )
            ),
        )
        self.adj_triplet_graph_return_credit_cap = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_triplet_graph_return_credit_cap",
                    0.0,
                )
            ),
        )
        self.adj_triplet_graph_return_credit_min_graph_adv = float(
            getattr(
                self.args,
                "adj_triplet_graph_return_credit_min_graph_adv",
                0.0,
            )
        )
        self.adj_triplet_graph_return_credit_raw_gate_scale = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_triplet_graph_return_credit_raw_gate_scale",
                    0.0,
                )
            ),
        )
        self.adj_triplet_graph_return_credit_require_delayed_gate = bool(
            getattr(
                self.args,
                "adj_triplet_graph_return_credit_require_delayed_gate",
                False,
            )
        )
        self.use_adj_delayed_triplet_credit = bool(
            getattr(self.args, "use_adj_delayed_triplet_credit", False)
        )
        self.adj_delayed_triplet_credit_coef = max(
            0.0,
            float(getattr(self.args, "adj_delayed_triplet_credit_coef", 0.0)),
        )
        self.adj_delayed_triplet_credit_window = max(
            0,
            int(getattr(self.args, "adj_delayed_triplet_credit_window", 0)),
        )
        self.adj_delayed_triplet_credit_cap = max(
            0.0,
            float(getattr(self.args, "adj_delayed_triplet_credit_cap", 0.0)),
        )
        self.adj_delayed_triplet_credit_min_reward = float(
            getattr(self.args, "adj_delayed_triplet_credit_min_reward", 0.0)
        )
        self.adj_delayed_triplet_credit_positive_only = bool(
            getattr(
                self.args,
                "adj_delayed_triplet_credit_positive_only",
                False,
            )
        )
        self.adj_delayed_triplet_credit_min_adv = max(
            0.0,
            float(getattr(self.args, "adj_delayed_triplet_credit_min_adv", 0.0)),
        )
        self.adj_delayed_triplet_credit_require_future_match = bool(
            getattr(
                self.args,
                "adj_delayed_triplet_credit_require_future_match",
                False,
            )
        )
        self.use_adj_delayed_triplet_success_gate = bool(
            getattr(self.args, "use_adj_delayed_triplet_success_gate", False)
        )
        self.adj_delayed_triplet_success_gate_min_adv = float(
            getattr(
                self.args,
                "adj_delayed_triplet_success_gate_min_adv",
                0.0,
            )
        )
        self.adj_delayed_triplet_success_gate_scale = max(
            1e-6,
            float(
                getattr(
                    self.args,
                    "adj_delayed_triplet_success_gate_scale",
                    1.0,
                )
            ),
        )
        self.adj_delayed_triplet_success_gate_floor = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "adj_delayed_triplet_success_gate_floor",
                        0.0,
                    )
                ),
            ),
        )
        self.adj_delayed_triplet_future_overlap_min_nodes = min(
            3,
            max(
                1,
                int(
                    getattr(
                        self.args,
                        "adj_delayed_triplet_future_overlap_min_nodes",
                        3,
                    )
                ),
            ),
        )
        self.adj_delayed_triplet_partial_match_weight = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "adj_delayed_triplet_partial_match_weight",
                        0.5,
                    )
                ),
            ),
        )
        self.use_adj_capture_to_win_credit = bool(
            getattr(self.args, "use_adj_capture_to_win_credit", False)
        )
        self.adj_capture_to_win_credit_coef = max(
            0.0,
            float(getattr(self.args, "adj_capture_to_win_credit_coef", 0.0)),
        )
        self.adj_capture_to_win_credit_min_outcome_adv = float(
            getattr(
                self.args,
                "adj_capture_to_win_credit_min_outcome_adv",
                0.5,
            )
        )
        self.adj_capture_to_win_credit_scale = max(
            1e-6,
            float(getattr(self.args, "adj_capture_to_win_credit_scale", 0.75)),
        )
        self.adj_capture_to_win_credit_cap = max(
            0.0,
            float(getattr(self.args, "adj_capture_to_win_credit_cap", 0.35)),
        )
        self.adj_capture_to_win_credit_require_future_match = bool(
            getattr(
                self.args,
                "adj_capture_to_win_credit_require_future_match",
                False,
            )
        )
        self.use_adj_pair_triplet_complementary_credit = bool(
            getattr(
                self.args,
                "use_adj_pair_triplet_complementary_credit",
                False,
            )
        )
        self.adj_pair_pursuit_credit_coef = max(
            0.0,
            float(getattr(self.args, "adj_pair_pursuit_credit_coef", 0.0)),
        )
        self.adj_pair_pursuit_credit_window = max(
            1,
            int(getattr(self.args, "adj_pair_pursuit_credit_window", 20)),
        )
        self.adj_pair_pursuit_credit_cap = max(
            0.0,
            float(getattr(self.args, "adj_pair_pursuit_credit_cap", 0.20)),
        )
        self.adj_pair_pursuit_credit_min_reward = float(
            getattr(self.args, "adj_pair_pursuit_credit_min_reward", 0.0)
        )
        self.adj_return_adv_coef = float(
            getattr(self.args, "adj_return_adv_coef", 1.0)
        )
        if not np.isfinite(self.adj_return_adv_coef):
            raise FloatingPointError("adj_return_adv_coef must be finite")
        self.use_adj_ppo_stale_trust = bool(
            getattr(self.args, "use_adj_ppo_stale_trust", False)
        )
        self.adj_ppo_stale_trust_clip = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_ppo_stale_trust_clip",
                    self.clip_param,
                )
            ),
        )
        self.adj_ppo_stale_trust_scale = max(
            1e-6,
            float(getattr(self.args, "adj_ppo_stale_trust_scale", 0.25)),
        )
        self.adj_ppo_stale_trust_min_weight = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.args,
                        "adj_ppo_stale_trust_min_weight",
                        0.25,
                    )
                ),
            ),
        )
        self.use_vfunction = self.args.use_vfunction
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.critic_lr = self.args.critic_lr
        self.adj_lr = self.args.adj_lr
        self.tau = self.args.tau
        self.q_n_step = int(getattr(self.args, "q_n_step", 1))
        if self.q_n_step <= 0 or self.q_n_step > int(episode_length):
            raise ValueError(
                "SDDFG q_n_step must be in [1, episode_length]"
            )
        self.q_terminal_replay_loss_weight = float(getattr(
            self.args, "q_terminal_replay_loss_weight", 0.10
        ))
        if (
                not math.isfinite(self.q_terminal_replay_loss_weight)
                or self.q_terminal_replay_loss_weight <= 0.0
                or self.q_terminal_replay_loss_weight > 1.0):
            raise ValueError(
                "SDDFG terminal replay loss weight must be finite in (0, 1]"
            )
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.highest_orders = self.args.highest_orders
        self.use_dyn_graph = self.args.use_dyn_graph
        self.num_factor = self.args.num_factor
        # ``entropy_coef`` was historically reused as the SDDFG graph entropy
        # weight.  In the intra-episode dynamic Wolfpack runs, run21 showed
        # that the legacy value (1e-3) keeps the graph distribution close to
        # uniform at 200k steps.  Keep the old fallback, but allow the graph
        # policy to use its own smaller coefficient.
        configured_adj_entropy_coef = float(
            getattr(self.args, "adj_entropy_coef", -1.0)
        )
        self.entropy_coef = (
            float(self.args.entropy_coef)
            if configured_adj_entropy_coef < 0.0
            else configured_adj_entropy_coef
        )
        self.adj_entropy_coef = float(self.entropy_coef)
        configured_entropy_final = float(
            getattr(self.args, "adj_entropy_coef_final", -1.0)
        )
        self.adj_entropy_coef_final = (
            self.adj_entropy_coef
            if configured_entropy_final < 0.0
            else configured_entropy_final
        )
        self.adj_entropy_anneal_steps = max(
            0,
            int(getattr(self.args, "adj_entropy_anneal_steps", 0)),
        )
        self._use_valuenorm = self.args.use_valuenorm
        self.adj_max_grad_norm = self.args.adj_max_grad_norm
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for
            policy_id in
            self.policies.keys()}
        if self._use_valuenorm:
            self.value_normalizer = {policy_id: PopArt(1, self.device).to(self.device) for policy_id in
                                     self.policies.keys()}

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # target policies/networks
        self.adj_network = adj_network
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}

        self.policy_parameters = []
        for policy in self.policies.values():
            self.policy_parameters += policy.parameters()
        self.policy_optimizer = torch.optim.Adam(params=self.policy_parameters, lr=self.lr, eps=self.opti_eps)

        self.critic_fv_parameters = []
        self.critic_vtot_parameters = []
        self.critic_fv_optimizer = None
        self.critic_vtot_optimizer = None
        if self.use_vfunction:
            for policy in self.policies.values():
                self.critic_fv_parameters += policy.critic_fv_parameters()
                self.critic_vtot_parameters += policy.critic_vtot_parameters()
            if not self.critic_fv_parameters or not self.critic_vtot_parameters:
                raise RuntimeError(
                    "use_vfunction=True but SDDFG critic parameters are empty"
                )
            self.critic_fv_optimizer = torch.optim.Adam(
                params=self.critic_fv_parameters,
                lr=self.critic_lr,
                eps=self.opti_eps,
            )
            self.critic_vtot_optimizer = torch.optim.Adam(
                params=self.critic_vtot_parameters,
                lr=self.critic_lr,
                eps=self.opti_eps,
            )

        self.adj_parameters = []
        self.adj_parameters += self.adj_network.parameters()
        self.adj_optimizer = torch.optim.Adam(
            params=self.adj_parameters,
            lr=self.adj_lr,
            eps=self.opti_eps,
        )
        # Residual candidate epochs must not inherit PPO/factor Adam momentum.
        # Both optimizers reference the same graph parameters, but their state
        # is disjoint.  Standard adjacency epochs use ``adj_optimizer``;
        # candidate-only completion epochs use this optimizer exclusively.
        self.candidate_residual_optimizer = torch.optim.Adam(
            params=self.adj_parameters,
            lr=self.adj_lr,
            eps=self.opti_eps,
        )
        # Pending pair-only transactions are exact replay targets, not samples
        # from the ordinary graph/base PPO population. Run110 showed that the
        # standard optimizer's accumulated PPO moment reversed the current
        # pair-gradient direction in six consecutive epochs. Keep pair moments
        # isolated while retaining the same Adam hyperparameters and parameters.
        self.pair_pending_optimizer = torch.optim.Adam(
            params=self.adj_parameters,
            lr=self.adj_lr,
            eps=self.opti_eps,
        )
        # Pin the supported implementation to Adam's deterministic
        # single-tensor path when these optional flags exist.  Older server
        # builds do not expose them; newer builds may otherwise auto-select a
        # foreach kernel when the value is None.  The state reconstruction
        # formula below is derived from this explicit path, never fitted from
        # a rounded observed displacement.
        for optimizer in (
                self.adj_optimizer,
                self.candidate_residual_optimizer,
                self.pair_pending_optimizer):
            for optimizer_group in optimizer.param_groups:
                for optional_path in ("foreach", "fused"):
                    if optional_path in optimizer_group:
                        if optimizer_group[optional_path] not in (None, False):
                            raise RuntimeError(
                                "candidate-safe Adam requires {}=False".format(
                                    optional_path
                                )
                            )
                        optimizer_group[optional_path] = False

                for unsupported_mode in (
                        "capturable", "differentiable", "maximize"):
                    if bool(optimizer_group.get(unsupported_mode, False)):
                        raise RuntimeError(
                            "candidate-safe Adam does not support {}=True"
                            .format(unsupported_mode)
                        )
        print(
            "pair optimizer transaction diagnostics version={}; ".format(
                PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION
            )
            + "per_epoch_transaction_csv=true; "
            "combined_gradient_projection=true; "
            "adam_raw_displacement=true; "
            "per_objective_gradient_decomposition=true; "
            "objective_gradient_reconstruction=true; "
            "projection_delta_tracking=true; "
            "pair_pending_optimizer=isolated_adam; "
            "pair_actual_update_direction_guard=true; "
            "pair_finite_zero_update=verified_atomic_noop; "
            "selection_state_no_compatible_writeback="
            "verified_atomic_noop; "
            "standard_pair_gradient_projection=true; "
            "standard_pair_actual_update_direction_guard=true; "
            "standard_pair_target_score_jacobian_constraints=true; "
            "standard_pair_target_actual_update_guard=true; "
            "pending_pair_target_score_jacobian_constraints=true; "
            "pending_pair_target_actual_update_guard=true; "
            "selection_boundary_target_constraints=true; "
            "selection_boundary_exact_revalidation=true; "
            "selection_boundary_exact_score_contract="
            "dtype_tolerant_nonregression; "
            "selection_boundary_member_margin_contract=strict_positive; "
            "selection_boundary_target_trace=true; "
            "selection_boundary_joint_exact_backtracking=true; "
            "selection_boundary_max_safe_scale_refinement=true; "
            "selection_boundary_direction_search=deficit_closure_write_resolution_full_three_level_interval_lattice_exact_with_all_failed_boundary_limiter_bundle_tangents; "
            "selection_boundary_progress_seed=positive_deficit_extra_budget_groups_only; "
            "selection_boundary_direction_candidate_trace=true; "
            "selection_boundary_unsafe_limiter_trace=true; "
            "selection_boundary_inactive_selected_factor_probe="
            "unsafe_backtrack; "
            "selection_boundary_parameter_catalog_change_probe="
            "unsafe_backtrack; "
            "selection_boundary_deficit_budget=canonical_identity_nearest_member_waterfill; "
            "selection_boundary_exposure_nonregression=true; "
            "selection_boundary_crossing_is_not_hard_commit=true; "
            "selection_boundary_crossing_retention="
            "exact_context_commit_margin_floor_next_two_ordinary_adam_transactions; "
            "selection_boundary_retention_intervening_pair_only_writes="
            "preserve_or_supersede; "
            "selection_boundary_retention_replays_old_loss=false; "
            "selection_boundary_retention_current_competitor=true; "
            "selection_boundary_retention_incompatible_current_evidence_supersedes=true; "
            "selection_boundary_retention_final_transaction_postcondition=true; "
            "selection_boundary_retention_final_selection_state_postcondition=true; "
            "selection_boundary_retention_selection_state_frontier_refinement=true; "
            "selection_boundary_retention_seen_counts=discrete_full_observation_commit; "
            "selection_boundary_retention_component_attribution=exact_context_full_delta; "
            "selection_boundary_policy_response="
            "exact_available_action_counterfactual_v1; "
            "factor_credit_memory=batch_mean_once_per_fresh_rollout_partition; "
            "pair_failure_atomic_rollback=true; "
            "strict_pair_exact_infeasibility="
            "exhaustive_origin_valid_atomic_deferral; "
            "lifecycle_actual_pair_priority=true; "
            "lifecycle_current_priority_pre_repair=true; "
            "gradient_dot_tolerance=unified_norm_scaled; "
            "symbolic_strict_floor=false; "
            "lifecycle_final_dot_tolerance=norm_scaled; "
            "lifecycle_final_exact_revalidation=true"
        )
        print(
            "pair evidence funnel diagnostics version=2; "
            "successful_candidate_boundary_join=true; "
            "reject_reason_partition=true; "
            "terminal_gap_timing=true; "
            "episode_level_reject_reason_csv=true; "
            "exact_active_capture_identity_export=true"
        )
        print(
            "candidate identity transaction diagnostics version=2; "
            "per_target_same_population_rank_trace=true; "
            "multi_event_target_group=true; "
            "no_additional_forward=true; "
            "trajectory_neutral=true"
        )
        print(
            "candidate evidence provenance diagnostics version=1; "
            "event_identity_rows=true; "
            "generation_event_dedup=true; "
            "transaction_join=true; "
            "no_additional_forward=true; "
            "trajectory_neutral=true"
        )

        # A real candidate-only capture is rare and the replay support is
        # intentionally consumable.  Once its loss has produced a safe update,
        # later base-only graph PPO steps must not immediately undo that update.
        # Keep a finite, detached constraint cache for exactly the adjacency
        # update rounds covered by the configured recent replay window.  Cached targets
        # are never added to the loss again: they only remove a base-gradient or
        # Adam-displacement component that would increase the previously
        # observed signed candidate objective.  Outcome mass is therefore not
        # replayed or amplified.
        recent_window = max(
            1,
            int(getattr(self.args, "adj_recent_episode_window", 1)),
        )
        # TTL is expressed in adjacency-update rounds, not optimizer
        # mini-batches.  Run73 used configured epochs*mini-batches even though
        # PPO early-stop reduced 29/64 logged updates to one epoch, so H=16
        # lasted up to eight adjacency rounds instead of the intended four.
        self.candidate_identity_lifecycle_horizon = recent_window
        self._candidate_identity_lifecycle = {}
        # Read-only observations outlive the training constraint just long
        # enough to measure exact 1/5/10-round retention.  They are never used
        # in a loss, gradient projection, replay cohort, or cache registration.
        self._candidate_identity_lifecycle_observations = {}
        # Selection-boundary crossings are retained through exactly the next
        # two ordinary adjacency Adam transactions.  Every intervening write
        # to the shared adjacency Parameters, including bounded-pending
        # pair-only Adam, must preserve or explicitly supersede the floor but
        # does not itself advance the ordinary age clock.  The archive never re-enters
        # the old pair loss, required progress, replay cohort, evidence store,
        # or active-selection executor.  It contributes only an exact-context
        # no-regression halfspace and nonlinear commit-margin floor, using the
        # current production competitor on every forward.  Incompatible new
        # actionable evidence may explicitly supersede a row.
        self._pair_selection_boundary_retention_observations = {}
        self._pair_selection_boundary_retention_next_id = 0
        self._pair_selection_boundary_ordinary_update_clock = 0

    def _pair_selection_boundary_retention_selection_state_snapshot(self):
        """Clone mutable non-parameter state which changes production ranking."""
        tensor_names = (
            "pair_credit_ema",
            "pair_credit_seen",
            "triplet_credit_ema",
            "triplet_credit_seen",
        )
        scalar_names = (
            "order3_credit_loss_ema",
            "order3_credit_margin_ema",
            "current_order3_credit_gate",
        )
        missing = [
            name for name in tensor_names + scalar_names
            if not hasattr(self.adj_network, name)
        ]
        if missing:
            raise RuntimeError(
                "selection-boundary retention production state is missing "
                + ", ".join(missing)
            )
        tensors = {}
        for name in tensor_names:
            value = getattr(self.adj_network, name)
            if not torch.is_tensor(value):
                raise RuntimeError(
                    "selection-boundary retention state {} is not a tensor"
                    .format(name)
                )
            tensors[name] = value.detach().clone()
        scalars = {}
        for name in scalar_names:
            value = float(getattr(self.adj_network, name))
            if not np.isfinite(value):
                raise FloatingPointError(
                    "non-finite selection-boundary retention state {}"
                    .format(name)
                )
            scalars[name] = value
        return {"tensors": tensors, "scalars": scalars}

    def _restore_pair_selection_boundary_retention_selection_state(
            self, state):
        if set(state.keys()) != {"tensors", "scalars"}:
            raise RuntimeError(
                "selection-boundary retention production-state snapshot is "
                "incomplete"
            )
        with torch.no_grad():
            for name, before in state["tensors"].items():
                current = getattr(self.adj_network, name, None)
                if not torch.is_tensor(current) or current.shape != before.shape:
                    raise RuntimeError(
                        "selection-boundary retention tensor state changed "
                        "shape for {}".format(name)
                    )
                current.copy_(before)
        for name, before in state["scalars"].items():
            setattr(self.adj_network, name, float(before))

    def _interpolate_pair_selection_boundary_retention_selection_state(
            self, before, after, scale):
        """Commit one exact-safe fraction of continuous production state.

        Credit EMAs and gate state are continuous learner memory.  The
        ``*_credit_seen`` buffers are different: ``update_factor_credit_memory``
        increments them by exactly one for every real observed factor and the
        production scorer interprets them as observation counts.  A retained
        fraction such as run132's 1/2048 must therefore never turn one real
        observation into a fractional count.  Every trial commits the complete
        post-observation count endpoint and scales only continuous state.
        """
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                "selection-boundary retention production-state scale is out "
                "of range"
            )
        if (
                set(before["tensors"].keys())
                != set(after["tensors"].keys())
                or set(before["scalars"].keys())
                != set(after["scalars"].keys())):
            raise RuntimeError(
                "selection-boundary retention production-state snapshots "
                "differ"
            )
        continuous_tensor_names = (
            "pair_credit_ema",
            "triplet_credit_ema",
        )
        discrete_tensor_names = (
            "pair_credit_seen",
            "triplet_credit_seen",
        )
        if set(before["tensors"].keys()) != set(
                continuous_tensor_names + discrete_tensor_names):
            raise RuntimeError(
                "selection-boundary retention production tensor-state "
                "semantics are incomplete"
            )
        with torch.no_grad():
            for name in continuous_tensor_names:
                start = before["tensors"][name]
                end = after["tensors"][name]
                current = getattr(self.adj_network, name)
                current.copy_(start + scale * (end - start))
            for name in discrete_tensor_names:
                start = before["tensors"][name]
                end = after["tensors"][name]
                if (
                        not bool(torch.isfinite(start).all().item())
                        or not bool(torch.isfinite(end).all().item())):
                    raise FloatingPointError(
                        "selection-boundary retention seen count is non-finite "
                        "for {}".format(name)
                    )
                start_integrality = float(
                    (start - start.round()).abs().max().detach().cpu().item()
                )
                end_integrality = float(
                    (end - end.round()).abs().max().detach().cpu().item()
                )
                delta = end - start
                delta_integrality = float(
                    (delta - delta.round()).abs().max().detach().cpu().item()
                )
                if (
                        start_integrality > 1e-6
                        or end_integrality > 1e-6
                        or delta_integrality > 1e-6
                        or float(delta.min().detach().cpu().item()) < 0.0):
                    raise RuntimeError(
                        "selection-boundary retention seen count is not a "
                        "nonnegative integral observation delta for {}"
                        .format(name)
                    )
                current = getattr(self.adj_network, name)
                current.copy_(end)
        for name, start in before["scalars"].items():
            end = after["scalars"][name]
            setattr(self.adj_network, name, start + scale * (end - start))

    def adjacency_optimizer_checkpoint_state(self):
        """Serialize every adjacency optimizer state."""
        state = _adjacency_optimizer_checkpoint_state(
            standard_optimizer=self.adj_optimizer,
            residual_optimizer=self.candidate_residual_optimizer,
            pair_optimizer=self.pair_pending_optimizer,
        )
        state["pair_selection_boundary_retention_version"] = 6
        state["pair_selection_boundary_retention_observations"] = (
            copy.deepcopy(
                self._pair_selection_boundary_retention_observations
            )
        )
        state["pair_selection_boundary_retention_next_id"] = int(
            self._pair_selection_boundary_retention_next_id
        )
        state["pair_selection_boundary_ordinary_update_clock"] = int(
            self._pair_selection_boundary_ordinary_update_clock
        )
        state["pair_selection_boundary_retention_selection_state"] = (
            self._pair_selection_boundary_retention_selection_state_snapshot()
        )
        return state

    def load_adjacency_optimizer_checkpoint_state(self, checkpoint):
        """Restore adjacency optimizer states without cross-contamination."""
        _load_adjacency_optimizer_checkpoint_state(
            checkpoint=checkpoint,
            standard_optimizer=self.adj_optimizer,
            residual_optimizer=self.candidate_residual_optimizer,
            pair_optimizer=self.pair_pending_optimizer,
        )
        retention_fields = (
            "pair_selection_boundary_retention_version",
            "pair_selection_boundary_retention_observations",
            "pair_selection_boundary_retention_next_id",
            "pair_selection_boundary_ordinary_update_clock",
            "pair_selection_boundary_retention_selection_state",
        )
        present = [field in checkpoint for field in retention_fields]
        if any(present) and not all(present):
            raise RuntimeError(
                "incomplete selection-boundary retention checkpoint"
            )
        if all(present):
            if int(checkpoint[retention_fields[0]]) != 6:
                raise RuntimeError(
                    "unsupported selection-boundary retention checkpoint; "
                    "intervening-write retention semantics require a fresh run"
                )
            observations = copy.deepcopy(checkpoint[retention_fields[1]])
            next_id = int(checkpoint[retention_fields[2]])
            ordinary_clock = int(checkpoint[retention_fields[3]])
            if not isinstance(observations, dict):
                raise RuntimeError(
                    "selection-boundary retention observations must be a dict"
                )
            if next_id < 0 or ordinary_clock < 0:
                raise RuntimeError(
                    "selection-boundary retention checkpoint clocks are "
                    "negative"
                )
            if observations and next_id <= max(
                    int(key) for key in observations):
                raise RuntimeError(
                    "selection-boundary retention next id is not monotonic"
                )
            required_entry_fields = (
                "observation_id",
                "source_policy_id",
                "source_transaction_sequence_index",
                "source_adjacency_update_round",
                "source_episode_ordinal",
                "source_episode_step",
                "selection_context_sha256",
                "rnn_obs",
                "dones",
                "adj",
                "factor_index",
                "target_candidate_index",
                "target_canonical_identity",
                "target_sign",
                "commit_competitor_candidate_index",
                "commit_competitor_canonical_identity",
                "pre_signed_margin",
                "commit_signed_margin",
                "commit_rank",
                "commit_active",
                "created_at",
                "protection_stopped",
                "protection_stop_reason",
                "protection_stop_clock",
            )
            for key, entry in observations.items():
                missing = [
                    field for field in required_entry_fields
                    if field not in entry
                ]
                if missing:
                    raise RuntimeError(
                        "selection-boundary retention checkpoint entry {} "
                        "is missing {}".format(
                            key, ", ".join(missing)
                        )
                    )
            self._pair_selection_boundary_retention_observations = (
                observations
            )
            self._pair_selection_boundary_retention_next_id = next_id
            self._pair_selection_boundary_ordinary_update_clock = (
                ordinary_clock
            )
            self._restore_pair_selection_boundary_retention_selection_state(
                copy.deepcopy(checkpoint[retention_fields[4]])
            )
        else:
            # Historical checkpoints contain no read-only observation state.
            # Restoring their optimizer tensors remains backward-compatible,
            # but retention evidence begins only with future crossings.
            self._pair_selection_boundary_retention_observations = {}
            self._pair_selection_boundary_retention_next_id = 0
            self._pair_selection_boundary_ordinary_update_clock = 0

    def pair_pending_outer_transaction_state(self):
        """Snapshot every trainer state a pair-only Adam step may mutate."""
        network_attrs = {}
        for name in (
                "candidate_policy_version",
                "candidate_lifecycle_clock",
                "order3_credit_gate",
                "order3_credit_loss_ema",
                "order3_credit_margin_ema",
                "current_order3_credit_gate"):
            if hasattr(self.adj_network, name):
                network_attrs[name] = copy.deepcopy(
                    getattr(self.adj_network, name)
                )
        network_rng_states = {}
        for name in ("rng", "eval_rng"):
            rng = getattr(self.adj_network, name, None)
            if rng is not None and hasattr(rng, "get_state"):
                network_rng_states[name] = copy.deepcopy(rng.get_state())
        return {
            "version": 1,
            "adj_network": copy.deepcopy(
                self.adj_network.state_dict()
            ),
            "adj_optimizers": copy.deepcopy(
                self.adjacency_optimizer_checkpoint_state()
            ),
            "candidate_lifecycle": copy.deepcopy(
                self._candidate_identity_lifecycle
            ),
            "candidate_lifecycle_observations": copy.deepcopy(
                self._candidate_identity_lifecycle_observations
            ),
            "pair_boundary_retention_observations": copy.deepcopy(
                self._pair_selection_boundary_retention_observations
            ),
            "pair_boundary_retention_next_id": int(
                self._pair_selection_boundary_retention_next_id
            ),
            "pair_boundary_ordinary_update_clock": int(
                self._pair_selection_boundary_ordinary_update_clock
            ),
            "network_attrs": network_attrs,
            "network_rng_states": network_rng_states,
        }

    def restore_pair_pending_outer_transaction_state(self, state):
        if int(state.get("version", -1)) != 1:
            raise RuntimeError(
                "unsupported pair pending outer transaction state"
            )
        self.adj_network.load_state_dict(
            copy.deepcopy(state["adj_network"])
        )
        self.load_adjacency_optimizer_checkpoint_state(
            copy.deepcopy(state["adj_optimizers"])
        )
        self._candidate_identity_lifecycle = copy.deepcopy(
            state["candidate_lifecycle"]
        )
        self._candidate_identity_lifecycle_observations = copy.deepcopy(
            state["candidate_lifecycle_observations"]
        )
        self._pair_selection_boundary_retention_observations = copy.deepcopy(
            state["pair_boundary_retention_observations"]
        )
        self._pair_selection_boundary_retention_next_id = int(
            state["pair_boundary_retention_next_id"]
        )
        self._pair_selection_boundary_ordinary_update_clock = int(
            state["pair_boundary_ordinary_update_clock"]
        )
        for name, value in state["network_attrs"].items():
            setattr(self.adj_network, name, copy.deepcopy(value))
        for name, rng_state in state.get("network_rng_states", {}).items():
            rng = getattr(self.adj_network, name, None)
            if rng is None or not hasattr(rng, "set_state"):
                raise RuntimeError(
                    "pair pending rollback cannot restore adjacency RNG {}"
                    .format(name)
                )
            rng.set_state(copy.deepcopy(rng_state))

    @staticmethod
    def _candidate_identity_lifecycle_key(
            rnn_obs,
            dones,
            adj,
            previous_adj,
            identity_delta,
            behavior_policy_version):
        """Stable key for one exact replay transition across PPO epochs."""
        digest = hashlib.sha256()
        for tensor in (
                rnn_obs,
                dones,
                adj,
                previous_adj,
                identity_delta,
                behavior_policy_version):
            value = tensor.detach().contiguous().cpu().numpy()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes())
        return digest.digest()

    def _prune_candidate_identity_lifecycle(self, lifecycle_clock):
        expired = [
            key for key, entry in self._candidate_identity_lifecycle.items()
            if int(lifecycle_clock) >= int(entry["expires_at"])
        ]
        for key in expired:
            del self._candidate_identity_lifecycle[key]
        observation_archive = getattr(
            self, "_candidate_identity_lifecycle_observations", {}
        )
        observed = [
            key for key, entry in observation_archive.items()
            if int(lifecycle_clock) - int(entry["created_at"]) > 10
        ]
        for key in observed:
            del observation_archive[key]
        return len(expired)

    def _candidate_identity_lifecycle_retention_diagnostics(
            self, lifecycle_clock, reference):
        """Measure exact age-1/5/10 retention without extending protection."""
        observations = getattr(
            self, "_candidate_identity_lifecycle_observations", {}
        )
        result = {}
        for retention_age in (1, 5, 10):
            result[retention_age] = {
                "eligible_count": reference.new_tensor(0.0),
                "signed_margin_fraction": reference.new_tensor(0.0),
                "rank_fraction": reference.new_tensor(0.0),
                "signed_margin_held_count": reference.new_tensor(0.0),
                "rank_held_count": reference.new_tensor(0.0),
                "positive_eligible_count": reference.new_tensor(0.0),
                "positive_signed_margin_held_count": reference.new_tensor(0.0),
                "positive_rank_held_count": reference.new_tensor(0.0),
                "negative_eligible_count": reference.new_tensor(0.0),
                "negative_signed_margin_held_count": reference.new_tensor(0.0),
                "negative_rank_held_count": reference.new_tensor(0.0),
            }
        if not observations:
            return result

        entries = list(observations.values())
        ages = [
            int(lifecycle_clock) - int(entry["created_at"])
            for entry in entries
        ]
        selected_indices = [
            index for index, age in enumerate(ages)
            if age in (1, 5, 10)
        ]
        if not selected_indices:
            return result
        selected_entries = [entries[index] for index in selected_indices]
        selected_ages = reference.new_tensor([
            ages[index] for index in selected_indices
        ])
        rnn_obs = torch.cat(
            [entry["rnn_obs"] for entry in selected_entries], dim=0
        )
        dones = torch.cat(
            [entry["dones"] for entry in selected_entries], dim=0
        )
        adj = torch.cat(
            [entry["adj"] for entry in selected_entries], dim=0
        )
        previous_adj = torch.cat(
            [entry["previous_adj"] for entry in selected_entries], dim=0
        )
        identity_delta = torch.cat(
            [entry["identity_delta"] for entry in selected_entries], dim=0
        )
        reference_margin = torch.cat(
            [
                entry["reference_margin"]
                for entry in selected_entries
            ],
            dim=0,
        )
        reference_rank = torch.cat(
            [entry["reference_rank"] for entry in selected_entries], dim=0
        )
        with torch.no_grad():
            margins, valid_mask = (
                self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                    rnn_obs=rnn_obs,
                    dones=dones.bool(),
                    adj=adj,
                )
            )
            transition_mask = torch.ones(
                (margins.shape[0], 1),
                device=margins.device,
                dtype=margins.dtype,
            )
            loss_info = compute_capture_candidate_identity_active_competitor_loss(
                candidate_competitor_margins=margins,
                candidate_reference_margins=reference_margin,
                candidate_identity_delta=identity_delta,
                candidate_valid_mask=valid_mask,
                transition_mask=transition_mask,
            )
            current_margin = loss_info["margin"]
            current_rank = self.adj_network.canonical_candidate_ranks(
                margins, valid_mask
            )
            margin_tolerance = (
                8.0 * 2.220446049250313e-16
                if current_margin.dtype == torch.float64
                else 8.0 * 1.1920928955078125e-7
            )
            target_mask = identity_delta.abs() > 0.0
            positive_mask = identity_delta > 0.0
            negative_mask = identity_delta < 0.0
            # Retention means no forgetting relative to the registered
            # post-update state. Boundary attainment is intentionally reported
            # separately by the current candidate boundary-crossing metrics.
            signed_margin_held = (
                torch.sign(identity_delta) * current_margin
                >= (
                    torch.sign(identity_delta) * reference_margin
                    - margin_tolerance
                )
            )
            rank_held = (
                (positive_mask & (current_rank <= reference_rank))
                | (negative_mask & (current_rank >= reference_rank))
            )
            for retention_age in (1, 5, 10):
                # Exact-age evaluation prevents one transition from being
                # counted repeatedly at every later optimizer update.
                eligible = target_mask & (
                    selected_ages == float(retention_age)
                ).unsqueeze(-1)
                eligible_count = eligible.float().sum()
                positive_eligible = eligible & positive_mask
                negative_eligible = eligible & negative_mask
                signed_margin_held_count = (
                    signed_margin_held & eligible
                ).float().sum()
                rank_held_count = (rank_held & eligible).float().sum()
                result[retention_age] = {
                    "eligible_count": eligible_count,
                    "signed_margin_fraction": (
                        signed_margin_held_count
                        / eligible_count.clamp_min(1.0)
                    ),
                    "rank_fraction": (
                        rank_held_count / eligible_count.clamp_min(1.0)
                    ),
                    "signed_margin_held_count": signed_margin_held_count,
                    "rank_held_count": rank_held_count,
                    "positive_eligible_count": (
                        positive_eligible.float().sum()
                    ),
                    "positive_signed_margin_held_count": (
                        signed_margin_held & positive_eligible
                    ).float().sum(),
                    "positive_rank_held_count": (
                        rank_held & positive_eligible
                    ).float().sum(),
                    "negative_eligible_count": (
                        negative_eligible.float().sum()
                    ),
                    "negative_signed_margin_held_count": (
                        signed_margin_held & negative_eligible
                    ).float().sum(),
                    "negative_rank_held_count": (
                        rank_held & negative_eligible
                    ).float().sum(),
                }
        return result

    @staticmethod
    def _pair_selection_boundary_retention_context_digest(
            rnn_obs, dones, adj, factor_index, target_candidate_index,
            target_sign):
        digest = hashlib.sha256()
        for tensor in (rnn_obs, dones, adj):
            value = tensor.detach().contiguous().cpu().numpy()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes())
        digest.update(str(int(factor_index)).encode("ascii"))
        digest.update(str(int(target_candidate_index)).encode("ascii"))
        digest.update(str(float(target_sign)).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _pair_selection_boundary_policy_state_digest(policy):
        """Fingerprint the policy state touched by a counterfactual forward."""
        digest = hashlib.sha256()
        parameters = list(policy.parameters())
        for ordinal, parameter in enumerate(parameters):
            value = parameter.detach().contiguous().cpu().numpy()
            digest.update(str(int(ordinal)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _pair_selection_boundary_policy_context_digest(
            obs, rnn_obs, dones, available_actions, pre_adj, post_adj,
            factor_index, target_candidate_index, target_sign):
        digest = hashlib.sha256()
        for tensor in (
                obs,
                rnn_obs,
                dones,
                available_actions,
                pre_adj,
                post_adj):
            value = tensor.detach().contiguous().cpu().numpy()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes())
        digest.update(str(int(factor_index)).encode("ascii"))
        digest.update(str(int(target_candidate_index)).encode("ascii"))
        digest.update(str(float(target_sign)).encode("ascii"))
        return digest.hexdigest()

    def _pair_selection_boundary_policy_response_diagnostics(
            self,
            trace_rows,
            obs,
            rnn_obs,
            dones,
            adj,
            available_actions,
            policy_id):
        """Replay the policy with only the crossing's active factor changed.

        This is a read-only, exact-context diagnostic.  The recurrent feature
        input and availability mask are held fixed; only the selected dynamic
        factor at ``factor_index`` changes from the pre-crossing active
        candidate to the post-crossing active candidate.  It never supplies a
        loss, gradient, replay priority, lifecycle event, or evidence credit.
        """
        crossing_rows = [
            row for row in trace_rows
            if int(row["boundary_crossing"]) == 1
        ]
        if not crossing_rows:
            return []
        if policy_id not in self.policies:
            raise RuntimeError(
                "policy counterfactual has no matching production policy"
            )
        policy = self.policies[policy_id]
        required_policy_methods = ("get_rnn_batch", "get_actions")
        missing_methods = [
            name for name in required_policy_methods
            if not callable(getattr(policy, name, None))
        ]
        if missing_methods:
            raise RuntimeError(
                "policy counterfactual is missing {}".format(
                    ", ".join(missing_methods)
                )
            )
        input_tensors = {
            "obs": obs,
            "rnn_obs": rnn_obs,
            "dones": dones,
            "adj": adj,
            "available_actions": available_actions,
        }
        if any(tensor.dim() != 3 for tensor in input_tensors.values()):
            raise RuntimeError(
                "policy counterfactual inputs must all be rank three"
            )
        expected_shapes = {
            name: (int(tensor.shape[0]), int(tensor.shape[1]))
            for name, tensor in input_tensors.items()
        }
        if len({shape[0] for shape in expected_shapes.values()}) != 1:
            raise RuntimeError(
                "policy counterfactual batch axes do not match"
            )
        if any(shape[1] != self.num_agents
               for shape in expected_shapes.values()):
            raise RuntimeError(
                "policy counterfactual agent axes do not match"
            )
        if available_actions.dim() != 3:
            raise RuntimeError(
                "policy counterfactual availability must be [B,N,A]"
            )
        if (
                not bool(torch.isfinite(available_actions).all().item())
                or bool(torch.any(available_actions < 0.0).item())
                or bool(torch.any(available_actions > 1.0).item())):
            raise RuntimeError(
                "policy counterfactual availability mask is invalid"
            )

        def _candidate_nodes(candidate_index):
            _order, identity = _canonical_candidate_identity(
                candidate_index=candidate_index,
                num_agents=self.num_agents,
                highest_orders=self.adj_network.highest_orders,
            )
            nodes = tuple(int(value) for value in identity.split("-"))
            if len(nodes) != int(_order):
                raise RuntimeError(
                    "policy counterfactual candidate identity is malformed"
                )
            return int(_order), identity, nodes

        def _replace_factor(structure, factor_index, candidate_index):
            result = structure.detach().clone()
            order, identity, nodes = _candidate_nodes(candidate_index)
            result[:, :, int(factor_index)] = 0.0
            result[:, list(nodes), int(factor_index)] = 1.0
            return result, order, identity

        def _with_alive_self_factors(dynamic_adj, row_dones):
            dynamic_binary = (dynamic_adj > 0.5).to(torch.int64)
            alive = (~row_dones.bool()[..., 0]).to(torch.int64)
            eye = torch.eye(
                self.num_agents,
                device=dynamic_adj.device,
                dtype=torch.int64,
            ).unsqueeze(0)
            eye = eye * alive.unsqueeze(-1)
            return torch.cat((dynamic_binary, eye), dim=2)

        def _factor_q(policy_instance, row_obs, row_rnn, row_dones,
                      full_adj, factor_index):
            q_batch, idx_node_order, _adj_out, _num_edges = (
                policy_instance.get_rnn_batch(
                    row_obs,
                    row_rnn,
                    adj_input=full_adj,
                    batch_size=1,
                    no_sequence=False,
                    dones=row_dones,
                )
            )
            matches = []
            for order_offset, indices in enumerate(idx_node_order):
                if len(indices) == 0:
                    continue
                selected = torch.where(
                    indices[:, 1] == int(factor_index)
                )[0]
                for position in selected.tolist():
                    matches.append((order_offset + 1, position))
            if len(matches) != 1:
                raise RuntimeError(
                    "policy counterfactual factor slot is not unique"
                )
            order, position = matches[0]
            return q_batch[order - 1][position].detach().reshape(-1), order

        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        )
        numpy_rng_before = copy.deepcopy(np.random.get_state())
        python_rng_before = random.getstate()
        policy_rng = getattr(policy, "rng", None)
        policy_rng_before = (
            copy.deepcopy(policy_rng.get_state())
            if policy_rng is not None and hasattr(policy_rng, "get_state")
            else None
        )
        state_before = self._pair_selection_boundary_policy_state_digest(
            policy
        )
        rows = []
        try:
            with torch.no_grad():
                for trace_row in crossing_rows:
                    transition_index = int(
                        trace_row["transition_index_in_partition"]
                    )
                    factor_index = int(trace_row["factor_index"])
                    target_index = int(trace_row["target_candidate_index"])
                    if (
                            transition_index < 0
                            or transition_index >= int(obs.shape[0])
                            or factor_index < 0
                            or factor_index >= int(adj.shape[2])):
                        raise RuntimeError(
                            "policy counterfactual crossing index is invalid"
                        )
                    target_order, target_identity, target_nodes = (
                        _candidate_nodes(target_index)
                    )
                    if target_identity != str(
                            trace_row["target_canonical_identity"]):
                        raise RuntimeError(
                            "policy counterfactual target identity changed"
                        )
                    row_obs = obs[
                        transition_index:transition_index + 1
                    ].detach().clone()
                    row_rnn = rnn_obs[
                        transition_index:transition_index + 1
                    ].detach().clone()
                    row_dones = dones[
                        transition_index:transition_index + 1
                    ].detach().clone()
                    row_adj = adj[
                        transition_index:transition_index + 1
                    ].detach().clone()
                    row_available = available_actions[
                        transition_index:transition_index + 1
                    ].detach().clone()
                    replay_nodes = tuple(torch.where(
                        row_adj[0, :, factor_index] > 0.5
                    )[0].detach().cpu().tolist())
                    if replay_nodes != target_nodes:
                        raise RuntimeError(
                            "policy counterfactual replay factor is not the "
                            "crossing target"
                        )
                    positive_promotion = int(
                        trace_row["positive_promotion"]
                    )
                    negative_eviction = int(trace_row["negative_eviction"])
                    if positive_promotion + negative_eviction != 1:
                        raise RuntimeError(
                            "policy counterfactual crossing kind is invalid"
                        )
                    if positive_promotion == 1:
                        pre_index = int(
                            trace_row["pre_competitor_candidate_index"]
                        )
                        post_index = target_index
                        crossing_kind = "promotion"
                    else:
                        pre_index = target_index
                        post_index = int(
                            trace_row["post_competitor_candidate_index"]
                        )
                        crossing_kind = "eviction"
                    pre_adj, pre_order, pre_identity = _replace_factor(
                        row_adj, factor_index, pre_index
                    )
                    post_adj, post_order, post_identity = _replace_factor(
                        row_adj, factor_index, post_index
                    )
                    pre_full_adj = _with_alive_self_factors(
                        pre_adj, row_dones
                    )
                    post_full_adj = _with_alive_self_factors(
                        post_adj, row_dones
                    )
                    pre_q, pre_q_order = _factor_q(
                        policy, row_obs, row_rnn, row_dones,
                        pre_full_adj, factor_index
                    )
                    post_q, post_q_order = _factor_q(
                        policy, row_obs, row_rnn, row_dones,
                        post_full_adj, factor_index
                    )
                    if pre_q_order != pre_order or post_q_order != post_order:
                        raise RuntimeError(
                            "policy counterfactual factor-Q order changed"
                        )
                    factor_q_comparable = int(pre_q.shape == post_q.shape)
                    factor_q_diff = (
                        float(torch.norm(post_q - pre_q).item())
                        if factor_q_comparable else None
                    )
                    pre_action, pre_best, _pre_margin, pre_factor_value = (
                        policy.get_actions(
                            row_obs,
                            row_rnn,
                            available_actions=row_available,
                            explore=False,
                            adj_input=pre_full_adj,
                            no_sequence=False,
                            dones=row_dones,
                        )
                    )
                    post_action, post_best, _post_margin, post_factor_value = (
                        policy.get_actions(
                            row_obs,
                            row_rnn,
                            available_actions=row_available,
                            explore=False,
                            adj_input=post_full_adj,
                            no_sequence=False,
                            dones=row_dones,
                        )
                    )
                    pre_best_value = float(
                        pre_best.detach().reshape(-1)[0].cpu().item()
                    )
                    post_best_value = float(
                        post_best.detach().reshape(-1)[0].cpu().item()
                    )
                    pre_selected_factor_value = float(
                        pre_factor_value.detach()[0, factor_index, 0]
                        .cpu().item()
                    )
                    post_selected_factor_value = float(
                        post_factor_value.detach()[0, factor_index, 0]
                        .cpu().item()
                    )
                    finite_values = (
                        pre_best_value,
                        post_best_value,
                        pre_selected_factor_value,
                        post_selected_factor_value,
                        float(torch.norm(pre_q).item()),
                        float(torch.norm(post_q).item()),
                    )
                    if not all(np.isfinite(value) for value in finite_values):
                        raise FloatingPointError(
                            "non-finite policy counterfactual response"
                        )
                    pre_action_index = np.asarray(pre_action).reshape(
                        self.num_agents, -1
                    ).argmax(axis=-1)
                    post_action_index = np.asarray(post_action).reshape(
                        self.num_agents, -1
                    ).argmax(axis=-1)
                    changed_count = int(np.sum(
                        pre_action_index != post_action_index
                    ))
                    structure_diff = float(torch.norm(
                        post_full_adj.float() - pre_full_adj.float()
                    ).item())
                    best_value_delta = post_best_value - pre_best_value
                    selected_factor_delta = (
                        post_selected_factor_value
                        - pre_selected_factor_value
                    )
                    response_nonzero = int(
                        changed_count > 0
                        or abs(best_value_delta) > 1.0e-8
                        or abs(selected_factor_delta) > 1.0e-8
                        or (
                            factor_q_diff is not None
                            and factor_q_diff > 1.0e-8
                        )
                    )
                    rows.append({
                        "diagnostic_version": int(
                            PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_DIAGNOSTIC_VERSION
                        ),
                        "target_row_sequence_within_transaction": int(
                            trace_row[
                                "target_row_sequence_within_transaction"
                            ]
                        ),
                        "transition_index_in_partition": transition_index,
                        "factor_index": factor_index,
                        "target_candidate_index": target_index,
                        "target_canonical_identity": target_identity,
                        "target_sign": float(trace_row["target_sign"]),
                        "crossing_kind": crossing_kind,
                        "pre_active_candidate_index": pre_index,
                        "pre_active_canonical_identity": pre_identity,
                        "post_active_candidate_index": post_index,
                        "post_active_canonical_identity": post_identity,
                        "policy_context_sha256": (
                            self
                            ._pair_selection_boundary_policy_context_digest(
                                obs=row_obs,
                                rnn_obs=row_rnn,
                                dones=row_dones,
                                available_actions=row_available,
                                pre_adj=pre_full_adj,
                                post_adj=post_full_adj,
                                factor_index=factor_index,
                                target_candidate_index=target_index,
                                target_sign=float(trace_row["target_sign"]),
                            )
                        ),
                        "policy_state_sha256": state_before,
                        "available_action_count": int(round(float(
                            row_available.sum().detach().cpu().item()
                        ))),
                        "pre_factor_order": pre_order,
                        "post_factor_order": post_order,
                        "structure_input_diff_norm": structure_diff,
                        "observation_input_diff_norm": 0.0,
                        "rnn_state_input_diff_norm": 0.0,
                        "factor_q_comparable": factor_q_comparable,
                        "pre_factor_q_norm": finite_values[4],
                        "post_factor_q_norm": finite_values[5],
                        "factor_q_diff_norm": factor_q_diff,
                        "pre_best_value": pre_best_value,
                        "post_best_value": post_best_value,
                        "best_value_delta": best_value_delta,
                        "pre_selected_factor_value": (
                            pre_selected_factor_value
                        ),
                        "post_selected_factor_value": (
                            post_selected_factor_value
                        ),
                        "selected_factor_value_delta": selected_factor_delta,
                        "greedy_action_changed_count": changed_count,
                        "greedy_action_changed_fraction": float(
                            changed_count / float(self.num_agents)
                        ),
                        "policy_response_nonzero": response_nonzero,
                        "rng_neutral": 1,
                        "state_neutral": 1,
                        "valid": 1,
                    })
        finally:
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            np.random.set_state(numpy_rng_before)
            random.setstate(python_rng_before)
            if policy_rng_before is not None:
                policy_rng.set_state(copy.deepcopy(policy_rng_before))
        state_after = self._pair_selection_boundary_policy_state_digest(
            policy
        )
        if state_after != state_before:
            raise RuntimeError(
                "policy counterfactual changed production policy state"
            )
        return rows

    def _pair_selection_boundary_retention_due_entries(self):
        """Return floors due across writes before ordinary ages one and two."""
        ordinary_clock = int(
            self._pair_selection_boundary_ordinary_update_clock
        )
        result = []
        for key, entry in sorted(
                self._pair_selection_boundary_retention_observations.items()):
            if bool(entry.get("protection_stopped", False)):
                continue
            age_before_update = ordinary_clock - int(entry["created_at"])
            if age_before_update in (0, 1):
                result.append((key, entry))
        return result

    def _pair_selection_boundary_retention_forward(self, keyed_entries):
        """Run the real selected-factor boundary on archived exact contexts."""
        if not keyed_entries:
            return None
        entries = [entry for _key, entry in keyed_entries]
        for entry in entries:
            actual_digest = (
                self._pair_selection_boundary_retention_context_digest(
                    rnn_obs=entry["rnn_obs"],
                    dones=entry["dones"],
                    adj=entry["adj"],
                    factor_index=entry["factor_index"],
                    target_candidate_index=(
                        entry["target_candidate_index"]
                    ),
                    target_sign=entry["target_sign"],
                )
            )
            if actual_digest != str(entry["selection_context_sha256"]):
                raise RuntimeError(
                    "selection-boundary retention exact context SHA256 "
                    "changed for observation_id={}".format(
                        int(entry["observation_id"])
                    )
                )
        rnn_obs = torch.cat([entry["rnn_obs"] for entry in entries], dim=0)
        dones = torch.cat([entry["dones"] for entry in entries], dim=0)
        adj = torch.cat([entry["adj"] for entry in entries], dim=0)
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        )
        numpy_rng_before = copy.deepcopy(np.random.get_state())
        python_rng_before = random.getstate()
        network_rng_before = {}
        for name in ("rng", "eval_rng"):
            rng = getattr(self.adj_network, name, None)
            if rng is not None and hasattr(rng, "get_state"):
                network_rng_before[name] = copy.deepcopy(rng.get_state())
        try:
            current = (
                self.adj_network
                .evaluate_selected_factor_replay_boundaries(
                    rnn_obs=rnn_obs,
                    dones=dones.bool(),
                    adj=adj,
                )
            )
        finally:
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            np.random.set_state(numpy_rng_before)
            random.setstate(python_rng_before)
            for name, rng_state in network_rng_before.items():
                rng = getattr(self.adj_network, name, None)
                if rng is None or not hasattr(rng, "set_state"):
                    raise RuntimeError(
                        "selection-boundary retention cannot restore "
                        "adjacency RNG {}".format(name)
                    )
                rng.set_state(copy.deepcopy(rng_state))
        required = (
            "selected_margin",
            "selected_logp",
            "competitor_logp",
            "selected_rank",
            "selected_candidate_index",
            "competitor_candidate_index",
            "valid",
            "forced",
        )
        missing = [field for field in required if field not in current]
        if missing:
            raise RuntimeError(
                "selection-boundary retention production replay is missing "
                + ", ".join(missing)
            )
        return current

    def _pair_selection_boundary_retention_constraint_population(
            self, reference):
        """Build no-regression gradients without replaying any old objective."""
        keyed_entries = self._pair_selection_boundary_retention_due_entries()
        if not keyed_entries:
            return {
                "keyed_entries": [],
                "constraint_grads": [],
                "context_invalid_keys": [],
                "current_signed_margins": [],
            }
        current = self._pair_selection_boundary_retention_forward(
            keyed_entries
        )
        constraint_entries = []
        constraint_grads = []
        invalid_keys = []
        current_signed_margins = []
        for row_index, (key, entry) in enumerate(keyed_entries):
            factor_index = int(entry["factor_index"])
            location = (row_index, factor_index)
            valid = bool((current["valid"][location] > 0.0).item())
            forced = bool((current["forced"][location] > 0.0).item())
            target_index = int(round(float(
                current["selected_candidate_index"][location]
                .detach().cpu().item()
            )))
            competitor_index = int(round(float(
                current["competitor_candidate_index"][location]
                .detach().cpu().item()
            )))
            target_identity = "INVALID"
            if target_index >= 0:
                _, target_identity = _canonical_candidate_identity(
                    candidate_index=target_index,
                    num_agents=self.num_agents,
                    highest_orders=self.adj_network.highest_orders,
                )
            context_valid = bool(
                valid
                and not forced
                and target_index == int(entry["target_candidate_index"])
                and target_identity == str(entry["target_canonical_identity"])
                and competitor_index >= 0
            )
            if not context_valid:
                invalid_keys.append(key)
                continue
            target_sign = float(entry["target_sign"])
            signed_margin = (
                target_sign * current["selected_margin"][location]
            )
            commit_floor = float(entry["commit_signed_margin"])
            tolerance = _selection_boundary_replay_tolerance(
                selected_logp=current["selected_logp"][location],
                competitor_logp=current["competitor_logp"][location],
                commit_floor=commit_floor,
            )
            if bool((
                    signed_margin.detach()
                    < commit_floor - tolerance
            ).item()):
                raise RuntimeError(
                    "selection-boundary retention floor was already lost "
                    "before adjacency parameter update: observation_id={}, "
                    "signed_margin={:.17g}, commit_floor={:.17g}, "
                    "tolerance={:.17g}, selected_logp={:.17g}, "
                    "competitor_logp={:.17g}".format(
                        int(entry["observation_id"]),
                        float(signed_margin.detach().cpu().item()),
                        commit_floor,
                        float(tolerance.detach().cpu().item()),
                        float(
                            current["selected_logp"][location]
                            .detach().cpu().item()
                        ),
                        float(
                            current["competitor_logp"][location]
                            .detach().cpu().item()
                        ),
                    )
                )
            constraint = -signed_margin
            grads = torch.autograd.grad(
                constraint,
                self.adj_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            norm_sq = _gradient_tuple_dot(grads, grads, reference)
            if not bool(
                    torch.isfinite(norm_sq).item()
                    and (norm_sq > 0.0).item()):
                raise RuntimeError(
                    "selection-boundary retention target has no finite "
                    "adjacency gradient"
                )
            constraint_entries.append((key, entry))
            constraint_grads.append(grads)
            current_signed_margins.append(signed_margin.detach())
        return {
            "keyed_entries": constraint_entries,
            "constraint_grads": constraint_grads,
            "context_invalid_keys": invalid_keys,
            "current_signed_margins": current_signed_margins,
        }

    def _pair_selection_boundary_retention_exact_acceptance(
            self, keyed_entries):
        """Revalidate current-competitor commit floors after writeback."""
        if not keyed_entries:
            return {
                "valid": True,
                "context_invalid_keys": [],
                "violation_count": 0,
                "min_signed_gap": None,
                "max_tolerance": None,
            }
        with torch.no_grad():
            current = self._pair_selection_boundary_retention_forward(
                keyed_entries
            )
        invalid_keys = []
        gaps = []
        tolerances = []
        violation_count = 0
        for row_index, (key, entry) in enumerate(keyed_entries):
            factor_index = int(entry["factor_index"])
            location = (row_index, factor_index)
            valid = bool((current["valid"][location] > 0.0).item())
            forced = bool((current["forced"][location] > 0.0).item())
            target_index = int(round(float(
                current["selected_candidate_index"][location]
                .detach().cpu().item()
            )))
            competitor_index = int(round(float(
                current["competitor_candidate_index"][location]
                .detach().cpu().item()
            )))
            target_identity = "INVALID"
            if target_index >= 0:
                _, target_identity = _canonical_candidate_identity(
                    candidate_index=target_index,
                    num_agents=self.num_agents,
                    highest_orders=self.adj_network.highest_orders,
                )
            context_valid = bool(
                valid
                and not forced
                and target_index == int(entry["target_candidate_index"])
                and target_identity == str(entry["target_canonical_identity"])
                and competitor_index >= 0
            )
            if not context_valid:
                invalid_keys.append(key)
                continue
            signed_margin = float(entry["target_sign"]) * current[
                "selected_margin"
            ][location]
            commit_floor = float(entry["commit_signed_margin"])
            tolerance = _selection_boundary_replay_tolerance(
                selected_logp=current["selected_logp"][location],
                competitor_logp=current["competitor_logp"][location],
                commit_floor=commit_floor,
            )
            gap = signed_margin - commit_floor
            gaps.append(gap)
            tolerances.append(tolerance)
            if bool((gap < -tolerance).item()):
                violation_count += 1
        return {
            "valid": violation_count == 0,
            "context_invalid_keys": invalid_keys,
            "violation_count": int(violation_count),
            "min_signed_gap": (
                torch.stack(gaps).min() if gaps else None
            ),
            "max_tolerance": (
                torch.stack(tolerances).max() if tolerances else None
            ),
        }

    def _pair_selection_boundary_retention_component_rows(
            self,
            keyed_entries,
            baseline,
            component,
            component_name,
            component_delta_norm,
            joint_objectives_valid):
        """Compare one full continuous-state component on exact contexts."""
        rows = []

        def _read_state(current, row_index, entry):
            factor_index = int(entry["factor_index"])
            location = (row_index, factor_index)
            valid = bool((current["valid"][location] > 0.0).item())
            forced = bool((current["forced"][location] > 0.0).item())
            target_index = int(round(float(
                current["selected_candidate_index"][location]
                .detach().cpu().item()
            )))
            competitor_index = int(round(float(
                current["competitor_candidate_index"][location]
                .detach().cpu().item()
            )))
            target_identity = "INVALID"
            competitor_identity = "INVALID"
            if target_index >= 0:
                _, target_identity = _canonical_candidate_identity(
                    candidate_index=target_index,
                    num_agents=self.num_agents,
                    highest_orders=self.adj_network.highest_orders,
                )
            if competitor_index >= 0:
                _, competitor_identity = _canonical_candidate_identity(
                    candidate_index=competitor_index,
                    num_agents=self.num_agents,
                    highest_orders=self.adj_network.highest_orders,
                )
            context_valid = bool(
                valid
                and not forced
                and target_index == int(entry["target_candidate_index"])
                and target_identity == str(entry["target_canonical_identity"])
                and competitor_index >= 0
            )
            if not context_valid:
                return {
                    "context_valid": 0,
                    "competitor_candidate_index": None,
                    "competitor_canonical_identity": None,
                    "signed_margin": None,
                    "margin_tolerance": None,
                    "rank": None,
                    "active": None,
                }
            signed_margin = float(entry["target_sign"]) * float(
                current["selected_margin"][location].detach().cpu().item()
            )
            margin_tolerance = float(
                _selection_boundary_replay_tolerance(
                    selected_logp=current["selected_logp"][location],
                    competitor_logp=current["competitor_logp"][location],
                    commit_floor=entry["commit_signed_margin"],
                ).detach().cpu().item()
            )
            rank = int(round(float(
                current["selected_rank"][location].detach().cpu().item()
            )))
            if rank <= 0:
                raise RuntimeError(
                    "selection-state component attribution rank is not positive"
                )
            return {
                "context_valid": 1,
                "competitor_candidate_index": competitor_index,
                "competitor_canonical_identity": competitor_identity,
                "signed_margin": signed_margin,
                "margin_tolerance": margin_tolerance,
                "rank": rank,
                "active": int(rank == 1),
            }

        component_delta_norm = float(component_delta_norm)
        if not np.isfinite(component_delta_norm) or component_delta_norm < 0.0:
            raise FloatingPointError(
                "selection-state component delta norm is invalid"
            )
        for row_index, (_key, entry) in enumerate(keyed_entries):
            baseline_state = _read_state(baseline, row_index, entry)
            component_state = _read_state(component, row_index, entry)
            context_valid = int(
                baseline_state["context_valid"] == 1
                and component_state["context_valid"] == 1
            )
            signed_margin_delta = None
            floor_retained = 0
            rank_retained = 0
            active_retained = 0
            competitor_changed = 0
            if context_valid == 1:
                baseline_margin = float(baseline_state["signed_margin"])
                component_margin = float(component_state["signed_margin"])
                signed_margin_delta = component_margin - baseline_margin
                commit_floor = float(entry["commit_signed_margin"])
                margin_tolerance = float(
                    component_state["margin_tolerance"]
                )
                floor_retained = int(
                    component_margin >= commit_floor - margin_tolerance
                )
                target_sign = float(entry["target_sign"])
                if target_sign > 0.0:
                    rank_retained = int(
                        int(component_state["rank"])
                        <= int(entry["commit_rank"])
                    )
                    active_retained = int(component_state["active"] == 1)
                elif target_sign < 0.0:
                    rank_retained = int(
                        int(component_state["rank"])
                        >= int(entry["commit_rank"])
                    )
                    active_retained = int(component_state["active"] == 0)
                else:
                    raise RuntimeError(
                        "selection-state component attribution sign is zero"
                    )
                competitor_changed = int(
                    baseline_state["competitor_candidate_index"]
                    != component_state["competitor_candidate_index"]
                )
            source_kind = (
                "ordinary_factor_credit_memory"
                if component_name in (
                    "pair_credit_ema",
                    "triplet_credit_ema",
                )
                else (
                    "ordinary_order3_gate_statistics"
                    if component_name != "all_continuous"
                    else "combined_continuous_selection_state"
                )
            )
            rows.append({
                "diagnostic_version": int(
                    PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_DIAGNOSTIC_VERSION
                ),
                "observation_id": int(entry["observation_id"]),
                "component": str(component_name),
                "source_kind": source_kind,
                "component_delta_norm": component_delta_norm,
                "commit_floor": float(entry["commit_signed_margin"]),
                "baseline_competitor_candidate_index": (
                    baseline_state["competitor_candidate_index"]
                ),
                "baseline_competitor_canonical_identity": (
                    baseline_state["competitor_canonical_identity"]
                ),
                "component_competitor_candidate_index": (
                    component_state["competitor_candidate_index"]
                ),
                "component_competitor_canonical_identity": (
                    component_state["competitor_canonical_identity"]
                ),
                "baseline_signed_margin": baseline_state["signed_margin"],
                "component_signed_margin": component_state["signed_margin"],
                "signed_margin_delta": signed_margin_delta,
                "baseline_rank": baseline_state["rank"],
                "component_rank": component_state["rank"],
                "baseline_active": baseline_state["active"],
                "component_active": component_state["active"],
                "competitor_changed": competitor_changed,
                "baseline_context_valid": int(
                    baseline_state["context_valid"]
                ),
                "component_context_valid": int(
                    component_state["context_valid"]
                ),
                "context_valid": context_valid,
                "floor_retained": floor_retained,
                "rank_retained": rank_retained,
                "active_retained": active_retained,
                "joint_objectives_valid": int(bool(joint_objectives_valid)),
            })
        return rows

    def _register_pair_selection_boundary_retention_observations(
            self,
            trace_rows,
            rnn_obs,
            dones,
            adj,
            replay_population_provenance,
            adjacency_update_round,
            policy_id,
            transaction_sequence_index):
        """Archive exact crossing contexts without protecting or replaying them."""
        crossing_rows = [
            row for row in trace_rows
            if int(row["boundary_crossing"]) == 1
        ]
        if not crossing_rows:
            return 0
        if policy_id is None or transaction_sequence_index is None:
            raise RuntimeError(
                "crossing retention requires policy and transaction identity"
            )
        if replay_population_provenance.shape != (rnn_obs.shape[0], 3):
            raise RuntimeError(
                "crossing retention replay provenance has an invalid shape"
            )
        ordinary_clock = int(
            self._pair_selection_boundary_ordinary_update_clock
        )
        next_id = int(self._pair_selection_boundary_retention_next_id)
        pending_entries = []
        for row in crossing_rows:
            transition_index = int(row["transition_index_in_partition"])
            factor_index = int(row["factor_index"])
            target_candidate_index = int(row["target_candidate_index"])
            target_sign = float(row["target_sign"])
            if transition_index < 0 or transition_index >= rnn_obs.shape[0]:
                raise RuntimeError(
                    "crossing retention transition index is out of range"
                )
            if factor_index < 0 or factor_index >= self.num_factor:
                raise RuntimeError(
                    "crossing retention factor index is out of range"
                )
            _, target_identity = _canonical_candidate_identity(
                candidate_index=target_candidate_index,
                num_agents=self.num_agents,
                highest_orders=self.adj_network.highest_orders,
            )
            if target_identity != str(row["target_canonical_identity"]):
                raise RuntimeError(
                    "crossing retention canonical identity changed"
                )
            pre_signed_margin = float(row["pre_signed_margin"])
            commit_signed_margin = float(row["post_signed_margin"])
            if not (
                    np.isfinite(pre_signed_margin)
                    and np.isfinite(commit_signed_margin)
                    and pre_signed_margin <= 0.0
                    and commit_signed_margin > 0.0):
                raise RuntimeError(
                    "crossing retention row does not cross the boundary"
                )
            commit_rank = int(row["post_rank"])
            commit_active = int(row["post_active_at_replay_boundary"])
            if commit_rank <= 0 or commit_active != int(commit_rank == 1):
                raise RuntimeError(
                    "crossing retention commit rank/active state differs"
                )
            if target_sign > 0.0:
                if int(row["positive_promotion"]) != 1:
                    raise RuntimeError(
                        "positive crossing retention row is not a promotion"
                    )
            elif target_sign < 0.0:
                if int(row["negative_eviction"]) != 1:
                    raise RuntimeError(
                        "negative crossing retention row is not an eviction"
                    )
            else:
                raise RuntimeError(
                    "crossing retention target sign is zero"
                )
            row_rnn_obs = rnn_obs[
                transition_index:transition_index + 1
            ].detach().clone()
            row_dones = dones[
                transition_index:transition_index + 1
            ].detach().clone()
            row_adj = adj[
                transition_index:transition_index + 1
            ].detach().clone()
            provenance = replay_population_provenance[transition_index]
            observation_id = next_id + len(pending_entries)
            entry = {
                "observation_id": int(observation_id),
                "source_policy_id": str(policy_id),
                "source_transaction_sequence_index": int(
                    transaction_sequence_index
                ),
                "source_adjacency_update_round": int(
                    adjacency_update_round
                ),
                "source_episode_ordinal": int(round(float(
                    provenance[1].detach().cpu().item()
                ))),
                "source_episode_step": int(round(float(
                    provenance[2].detach().cpu().item()
                ))),
                "selection_context_sha256": (
                    self._pair_selection_boundary_retention_context_digest(
                        rnn_obs=row_rnn_obs,
                        dones=row_dones,
                        adj=row_adj,
                        factor_index=factor_index,
                        target_candidate_index=target_candidate_index,
                        target_sign=target_sign,
                    )
                ),
                "rnn_obs": row_rnn_obs,
                "dones": row_dones,
                "adj": row_adj,
                "factor_index": factor_index,
                "target_candidate_index": target_candidate_index,
                "target_canonical_identity": target_identity,
                "target_sign": target_sign,
                "commit_competitor_candidate_index": int(
                    row["post_competitor_candidate_index"]
                ),
                "commit_competitor_canonical_identity": str(
                    row["post_competitor_canonical_identity"]
                ),
                "pre_signed_margin": pre_signed_margin,
                "commit_signed_margin": commit_signed_margin,
                "commit_rank": commit_rank,
                "commit_active": commit_active,
                "created_at": ordinary_clock,
                "protection_stopped": False,
                "protection_stop_reason": "",
                "protection_stop_clock": -1,
            }
            pending_entries.append((observation_id, entry))
        observations = self._pair_selection_boundary_retention_observations
        for observation_id, _entry in pending_entries:
            if observation_id in observations:
                raise RuntimeError(
                    "duplicate crossing retention observation id"
                )
        for observation_id, entry in pending_entries:
            observations[observation_id] = entry
        self._pair_selection_boundary_retention_next_id = (
            next_id + len(pending_entries)
        )
        return len(pending_entries)

    def _pair_selection_boundary_retention_diagnostics(self):
        """Replay exact crossing contexts after ordinary Adam updates 1 and 2."""
        observations = self._pair_selection_boundary_retention_observations
        if not observations:
            return []
        ordinary_clock = int(
            self._pair_selection_boundary_ordinary_update_clock
        )
        selected = [
            entry for _key, entry in sorted(observations.items())
            if ordinary_clock - int(entry["created_at"]) in (1, 2)
        ]
        if not selected:
            return []
        rnn_obs = torch.cat([entry["rnn_obs"] for entry in selected], dim=0)
        dones = torch.cat([entry["dones"] for entry in selected], dim=0)
        adj = torch.cat([entry["adj"] for entry in selected], dim=0)

        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        )
        numpy_rng_before = copy.deepcopy(np.random.get_state())
        python_rng_before = random.getstate()
        network_rng_before = {}
        for name in ("rng", "eval_rng"):
            rng = getattr(self.adj_network, name, None)
            if rng is not None and hasattr(rng, "get_state"):
                network_rng_before[name] = copy.deepcopy(rng.get_state())
        try:
            with torch.no_grad():
                current = (
                    self.adj_network
                    .evaluate_selected_factor_replay_boundaries(
                        rnn_obs=rnn_obs,
                        dones=dones.bool(),
                        adj=adj,
                    )
                )
        finally:
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            np.random.set_state(numpy_rng_before)
            random.setstate(python_rng_before)
            for name, rng_state in network_rng_before.items():
                rng = getattr(self.adj_network, name, None)
                if rng is None or not hasattr(rng, "set_state"):
                    raise RuntimeError(
                        "crossing retention cannot restore adjacency RNG {}"
                        .format(name)
                    )
                rng.set_state(copy.deepcopy(rng_state))

        required = (
            "selected_margin",
            "selected_logp",
            "competitor_logp",
            "selected_rank",
            "selected_candidate_index",
            "competitor_candidate_index",
            "valid",
            "forced",
        )
        missing = [field for field in required if field not in current]
        if missing:
            raise RuntimeError(
                "crossing retention production replay is missing {}".format(
                    ", ".join(missing)
                )
            )
        rows = []
        for index, entry in enumerate(selected):
            factor_index = int(entry["factor_index"])
            location = (index, factor_index)
            valid = bool((current["valid"][location] > 0.0).item())
            forced = bool((current["forced"][location] > 0.0).item())
            current_target_index = int(round(float(
                current["selected_candidate_index"][location]
                .detach().cpu().item()
            )))
            current_competitor_index = int(round(float(
                current["competitor_candidate_index"][location]
                .detach().cpu().item()
            )))
            target_identity_valid = current_target_index >= 0
            competitor_valid = current_competitor_index >= 0
            current_target_identity = "INVALID"
            current_competitor_identity = "INVALID"
            if target_identity_valid:
                _, current_target_identity = _canonical_candidate_identity(
                    candidate_index=current_target_index,
                    num_agents=self.num_agents,
                    highest_orders=self.adj_network.highest_orders,
                )
            if competitor_valid:
                _, current_competitor_identity = (
                    _canonical_candidate_identity(
                        candidate_index=current_competitor_index,
                        num_agents=self.num_agents,
                        highest_orders=self.adj_network.highest_orders,
                    )
                )
            context_valid = bool(
                valid
                and not forced
                and current_target_index
                == int(entry["target_candidate_index"])
                and current_target_identity
                == str(entry["target_canonical_identity"])
                and competitor_valid
            )
            competitor_changed = int(
                not competitor_valid
                or current_competitor_index
                != int(entry["commit_competitor_candidate_index"])
            )
            current_signed_margin = None
            current_rank = None
            current_active = None
            retained_progress_fraction = None
            margin_nonregression = 0
            rank_retained = 0
            active_retained = 0
            if context_valid:
                target_sign = float(entry["target_sign"])
                current_signed_margin = target_sign * float(
                    current["selected_margin"][location]
                    .detach().cpu().item()
                )
                current_rank = int(round(float(
                    current["selected_rank"][location]
                    .detach().cpu().item()
                )))
                if current_rank <= 0:
                    raise RuntimeError(
                        "crossing retention current rank is not positive"
                    )
                current_active = int(current_rank == 1)
                denominator = (
                    float(entry["commit_signed_margin"])
                    - float(entry["pre_signed_margin"])
                )
                if not denominator > 0.0:
                    raise RuntimeError(
                        "crossing retention commit progress is not positive"
                    )
                retained_progress_fraction = (
                    current_signed_margin
                    - float(entry["pre_signed_margin"])
                ) / denominator
                if not np.isfinite(retained_progress_fraction):
                    raise FloatingPointError(
                        "non-finite crossing retention fraction"
                    )
                margin_tolerance = float(
                    _selection_boundary_replay_tolerance(
                        selected_logp=current["selected_logp"][location],
                        competitor_logp=(
                            current["competitor_logp"][location]
                        ),
                        commit_floor=entry["commit_signed_margin"],
                    ).detach().cpu().item()
                )
                margin_nonregression = int(
                    current_signed_margin
                    >= float(entry["commit_signed_margin"])
                    - margin_tolerance
                )
                if target_sign > 0.0:
                    rank_retained = int(
                        current_rank <= int(entry["commit_rank"])
                    )
                    active_retained = int(current_active == 1)
                else:
                    rank_retained = int(
                        current_rank >= int(entry["commit_rank"])
                    )
                    active_retained = int(current_active == 0)
            age = ordinary_clock - int(entry["created_at"])
            rows.append({
                "diagnostic_version": int(
                    PAIR_SELECTION_BOUNDARY_RETENTION_DIAGNOSTIC_VERSION
                ),
                "observation_id": int(entry["observation_id"]),
                "source_policy_id": str(entry["source_policy_id"]),
                "source_transaction_sequence_index": int(
                    entry["source_transaction_sequence_index"]
                ),
                "source_adjacency_update_round": int(
                    entry["source_adjacency_update_round"]
                ),
                "ordinary_update_age": int(age),
                "source_episode_ordinal": int(
                    entry["source_episode_ordinal"]
                ),
                "source_episode_step": int(entry["source_episode_step"]),
                "selection_context_sha256": str(
                    entry["selection_context_sha256"]
                ),
                "factor_index": factor_index,
                "target_candidate_index": int(
                    entry["target_candidate_index"]
                ),
                "target_canonical_identity": str(
                    entry["target_canonical_identity"]
                ),
                "target_sign": float(entry["target_sign"]),
                "commit_competitor_candidate_index": int(
                    entry["commit_competitor_candidate_index"]
                ),
                "commit_competitor_canonical_identity": str(
                    entry["commit_competitor_canonical_identity"]
                ),
                "current_competitor_candidate_index": (
                    current_competitor_index
                ),
                "current_competitor_canonical_identity": (
                    current_competitor_identity
                ),
                "pre_signed_margin": float(entry["pre_signed_margin"]),
                "commit_signed_margin": float(
                    entry["commit_signed_margin"]
                ),
                "current_signed_margin": current_signed_margin,
                "retained_progress_fraction": retained_progress_fraction,
                "commit_rank": int(entry["commit_rank"]),
                "current_rank": current_rank,
                "commit_active": int(entry["commit_active"]),
                "current_active": current_active,
                "competitor_changed": competitor_changed,
                "context_valid": int(context_valid),
                "margin_nonregression": margin_nonregression,
                "rank_retained": rank_retained,
                "active_retained": active_retained,
                "protection_stopped": int(bool(
                    entry.get("protection_stopped", False)
                )),
                "protection_stop_reason": str(
                    entry.get("protection_stop_reason", "")
                ),
                "protection_stop_clock": int(
                    entry.get("protection_stop_clock", -1)
                ),
            })
        expired = [
            key for key, entry in observations.items()
            if ordinary_clock - int(entry["created_at"]) >= 2
        ]
        for key in expired:
            del observations[key]
        return rows

    def _register_candidate_identity_lifecycle(
            self,
            rnn_obs,
            dones,
            adj,
            previous_adj,
            identity_delta,
            transition_mask,
            behavioral_progress_mask,
            behavior_policy_version,
            reference_margin,
            reference_rank,
            lifecycle_clock):
        """Register behaviorally improved target rows without refreshing TTL."""
        if behavioral_progress_mask.shape != identity_delta.shape:
            raise ValueError(
                "candidate lifecycle progress mask shape {} does not match {}"
                .format(
                    tuple(behavioral_progress_mask.shape),
                    tuple(identity_delta.shape),
                )
            )
        protected_delta = (
            identity_delta
            * (behavioral_progress_mask > 0.0).to(identity_delta.dtype)
        )
        target_rows = (
            (protected_delta.abs().sum(dim=1) > 0.0)
            & (transition_mask.reshape(-1) > 0.0)
        )
        new_count = 0
        for row in torch.where(target_rows)[0].detach().cpu().tolist():
            row_rnn = rnn_obs[row:row + 1].detach().clone()
            row_dones = dones[row:row + 1].detach().clone()
            row_adj = adj[row:row + 1].detach().clone()
            row_previous_adj = (
                previous_adj[row:row + 1].detach().clone()
            )
            row_delta = protected_delta[row:row + 1].detach().clone()
            row_behavior_version = (
                behavior_policy_version[row:row + 1].detach().clone()
            )
            row_reference_margin = (
                reference_margin[row:row + 1].detach().clone()
            )
            row_reference_rank = (
                reference_rank[row:row + 1].detach().clone()
            )
            key = self._candidate_identity_lifecycle_key(
                row_rnn,
                row_dones,
                row_adj,
                row_previous_adj,
                row_delta,
                row_behavior_version,
            )
            if (
                    key in self._candidate_identity_lifecycle
                    or key in self._candidate_identity_lifecycle_observations):
                continue
            entry = {
                "rnn_obs": row_rnn,
                "dones": row_dones,
                "adj": row_adj,
                "previous_adj": row_previous_adj,
                "identity_delta": row_delta,
                "behavior_policy_version": row_behavior_version,
                "reference_margin": row_reference_margin,
                "reference_rank": row_reference_rank,
                "created_at": int(lifecycle_clock),
                # The creation round is age 0.  A horizon H protects the next
                # H adjacency-update rounds (ages 1..H) and expires before
                # age H+1.  Using created_at + H here expired the entry before
                # the age-H update and silently shortened a configured horizon
                # of four rounds to only three subsequent protected rounds.
                "expires_at": int(lifecycle_clock)
                + int(self.candidate_identity_lifecycle_horizon)
                + 1,
            }
            self._candidate_identity_lifecycle[key] = entry
            self._candidate_identity_lifecycle_observations[key] = entry
            new_count += 1
        return new_count

    def _set_optimizer_lr_with_floor(self, optimizer, init_lr, episode, episodes, floor_lr):
        """
        线性衰减，但不低于 floor_lr。
        """
        if episodes <= 0:
            lr = init_lr
        else:
            frac = 1.0 - float(episode) / float(episodes)
            frac = max(0.0, min(1.0, frac))
            lr = max(float(init_lr) * frac, float(floor_lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def lr_decay(self, episode, episodes):
        """
        Decay policy/critic lr, optionally decay adj/GAT lr.

        关键变化：
        1. policy/critic 仍然随 --use_linear_lr_decay 衰减；
        2. adj/GAT 默认不衰减，避免后期 adj_lr 过低；
        3. 如果显式传 --use_adj_linear_lr_decay，则 adj/GAT 衰减但不低于 adj_lr_decay_floor。
        """
        policy_floor = float(getattr(self.args, "policy_lr_decay_floor", 1e-5))
        critic_floor = float(getattr(self.args, "critic_lr_decay_floor", 1e-5))
        use_adj_decay = bool(getattr(self.args, "use_adj_linear_lr_decay", False))

        # policy / critic 继续衰减，但设置 floor，避免后期完全学不动
        self._set_optimizer_lr_with_floor(
            self.policy_optimizer,
            self.lr,
            episode,
            episodes,
            policy_floor
        )
        if self.use_vfunction:
            self._set_optimizer_lr_with_floor(
                self.critic_fv_optimizer,
                self.critic_lr,
                episode,
                episodes,
                critic_floor
            )
            self._set_optimizer_lr_with_floor(
                self.critic_vtot_optimizer,
                self.critic_lr,
                episode,
                episodes,
                critic_floor
            )

        # adj/GAT 默认不跟随全局 linear decay
        if use_adj_decay:
            self.adj_lr_decay(episode)
        else:
            for optimizer in (
                    self.adj_optimizer,
                    self.candidate_residual_optimizer):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = float(self.adj_lr)

    def adj_lr_decay(self, env_step):
        """Decay only graph parameters on a fixed environment-step clock.

        Dynamic topology inference remains active after annealing; only the
        cumulative drift of the GAT/hyperedge parameters is reduced. The
        caller supplies an explicit environment-step horizon so experiments
        can either preserve a common prefix or anneal across the full run.
        """
        anneal_steps = max(
            1,
            int(getattr(self.args, "adj_lr_anneal_steps", 500000)),
        )
        adj_floor = float(getattr(self.args, "adj_lr_decay_floor", 2e-5))
        current_lr = self._set_optimizer_lr_with_floor(
            self.adj_optimizer,
            self.adj_lr,
            env_step,
            anneal_steps,
            adj_floor,
        )
        self._set_optimizer_lr_with_floor(
            self.candidate_residual_optimizer,
            self.adj_lr,
            env_step,
            anneal_steps,
            adj_floor,
        )
        if self.adj_entropy_anneal_steps > 0:
            progress = min(
                max(
                    float(env_step)
                    / float(self.adj_entropy_anneal_steps),
                    0.0,
                ),
                1.0,
            )
            self.adj_entropy_coef = (
                self.entropy_coef
                + progress
                * (self.adj_entropy_coef_final - self.entropy_coef)
            )
        else:
            self.adj_entropy_coef = float(self.entropy_coef)
        return current_lr

    def train_policy_on_batch(self, batch, use_same_share_obs=None):
        """See parent class."""

        terminal_win_reward_batch = None
        terminal_replay_lane_episode_mask_batch = None
        if len(batch) == 12:
            obs_batch, cent_obs_batch, \
                act_batch, rew_batch, \
                dones_batch, dones_env_batch, \
                avail_act_batch, adj, \
                prob_adj, idxes, terminal_win_reward_batch, \
                terminal_replay_lane_episode_mask_batch = batch
        elif len(batch) == 11:
            obs_batch, cent_obs_batch, \
                act_batch, rew_batch, \
                dones_batch, dones_env_batch, \
                avail_act_batch, adj, \
                prob_adj, idxes, terminal_win_reward_batch = batch
        else:
            obs_batch, cent_obs_batch, \
                act_batch, rew_batch, \
                dones_batch, dones_env_batch, \
                avail_act_batch, adj, \
                prob_adj, idxes = batch
        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).transpose(0, 1).to(**self.tpdv)
        terminal_win_rewards = None
        if terminal_win_reward_batch is not None:
            terminal_win_rewards = to_torch(
                terminal_win_reward_batch[self.policy_ids[0]]
            ).to(**self.tpdv)
        terminal_replay_lane_episode_mask = None
        if terminal_replay_lane_episode_mask_batch is not None:
            terminal_replay_lane_episode_mask = to_torch(
                terminal_replay_lane_episode_mask_batch[self.policy_ids[0]]
            ).to(**self.tpdv).reshape(-1)

        # individual agent q values: each element is of shape (batch_size, 1)
        qs = []
        target_qs = []
        vtot = []
        target_vtot = []
        fv = []
        q_target_frontier_diagnostic = {
            "state_count": 0,
            "pre_capture_state_count": 0,
            "eligible_slot_count": 0,
            "constrained_slot_count": 0,
            "conflict_slot_count": 0,
            "reranked_state_count": 0,
            "reranked_slot_count": 0,
        }

        for p_id in self.policy_ids:

            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            stored_next_dones = to_torch(dones_batch[p_id]).permute(1, 2, 0, 3).to(self.device)
            rewards = to_torch(rew_batch[p_id][0]).transpose(0, 1).to(**self.tpdv)  # [25,32,1]
            curr_obs_batch = to_torch(obs_batch[p_id]).transpose(0, 2)  # [3,26,32,18]
            curr_act_batch = to_torch(act_batch[p_id]).transpose(0, 2).to(**self.tpdv)  # [3,25,32,5]
            state_obs_batch = to_torch(cent_obs_batch[p_id]).to(**self.tpdv)
            adj = to_torch(adj[p_id])

            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id]).transpose(0, 2).to(self.device)
            else:
                curr_avail_act_batch = None

            act_dim = curr_act_batch.shape[3]
            step = rewards.shape[1] + 1
            batch_size = curr_obs_batch.shape[0]

            # Intra-episode dynamic fix:
            # replay stores agent dones as next-state masks.  For the current
            # state sequence [s0...sT], prepend the true s0 inactive mask
            # inferred from padded observations instead of all-zero dones.
            # OR with observation-derived masks for all states as a guard
            # against any stale buffer transition around join/recover events.
            obs_inactive_dones = _infer_inactive_dones_from_obs(curr_obs_batch).permute(1, 0, 2, 3).to(**self.tpdv)
            if obs_inactive_dones.shape[0] != stored_next_dones.shape[0] + 1:
                raise RuntimeError(
                    "SDDFG replay mask length mismatch: "
                    f"obs states={obs_inactive_dones.shape[0]}, "
                    f"stored next dones={stored_next_dones.shape[0]}"
                )
            dones = torch.cat((obs_inactive_dones[:1], stored_next_dones), dim=0).float()
            obs_inactive_dones = obs_inactive_dones.float()
            # Old PyTorch on the server does not provide torch.maximum.
            # This is elementwise max(a, b): keep a slot done if either replay
            # next-done or observation-derived inactive mask marks it inactive.
            dones = torch.where(dones >= obs_inactive_dones, dones, obs_inactive_dones)

            dones_batch = torch.cat((torch.zeros(batch_size, 1, 1).to(**self.tpdv), dones_env_batch), dim=1)
            bad_transitions_mask = dones_batch[:, :-1]
            active_transition_mask = (
                (1.0 - dones[:-1].float()).sum(dim=2).gt(0.0).float().transpose(0, 1)
            )
            valid_transition_mask = (1.0 - bad_transitions_mask) * active_transition_mask

            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2)  # [25,256,5]
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2)  # [26,256,48]
            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, batch_size * self.num_agents, act_dim).to(**self.tpdv),
                                                 stacked_act_batch))  # [26,256,5]
            stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1]  # [25,256]

            alive_mask = (1.0 - dones.float()).clamp(0.0, 1.0)

            eye = torch.eye(
                self.num_agents,
                dtype=torch.int64,
                device=self.device
            ).view(1, 1, self.num_agents, self.num_agents).repeat(step, batch_size, 1, 1)
            eye = eye * alive_mask.long()

            if self.use_dyn_graph:
                adj_dyn = adj.to(self.device).long()

                factor_size = adj_dyn.float().sum(dim=2, keepdim=True)
                alive_count = (adj_dyn.float() * alive_mask).sum(dim=2, keepdim=True)
                factor_valid = ((factor_size > 0.0) & (alive_count == factor_size)).long()

                adj_dyn = adj_dyn * alive_mask.long() * factor_valid
                adj_input = torch.cat([adj_dyn, eye], dim=3).to(self.device)
            else:
                adj_input = eye.to(self.device)

            rnn_states_1 = policy.init_hidden(self.num_agents, batch_size)
            target_rnn_states = target_policy.init_hidden(self.num_agents, batch_size)

            dones_seq = dones.reshape(step, batch_size * self.num_agents, 1)

            rnn_obs_batch_1, _, no_sequence = policy.get_hidden_states(
                stacked_obs_batch, pol_prev_act_buffer_seq, rnn_states_1, dones=dones_seq
            )
            target_rnn_obs_batch, _, _ = target_policy.get_hidden_states(
                stacked_obs_batch, pol_prev_act_buffer_seq, target_rnn_states, dones=dones_seq
            )

            curr_act_batch_ind = stacked_act_batch_ind.reshape((step - 1) * batch_size, self.num_agents, -1)
            rnn_obs_q = rnn_obs_batch_1[:-1].reshape((step - 1) * batch_size, self.num_agents, -1)
            obs_q = to_torch(stacked_obs_batch[:-1]).reshape((step - 1) * batch_size, self.num_agents, -1)
            adj_input_q = adj_input[:-1].reshape((step - 1) * batch_size, self.num_agents, -1)
            dones_q = dones[:-1].reshape((step - 1) * batch_size, self.num_agents, -1)
            policy_qs = policy.get_q_values(obs_q, rnn_obs_q, curr_act_batch_ind, adj_input_q, no_sequence, dones_q)

            rnn_obs_qtot = rnn_obs_batch_1[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            obs_qtot = to_torch(stacked_obs_batch[1:]).reshape((step - 1) * batch_size, self.num_agents, -1)

            adj_input_qtot = adj_input[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            dones_qtot = dones[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            rnn_target_obs = target_rnn_obs_batch[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            if curr_avail_act_batch is None:
                curr_avail_act = None
            else:
                curr_avail_act = curr_avail_act_batch.transpose(
                    0, 1
                )[1:].reshape(
                    (step - 1) * batch_size,
                    self.num_agents,
                    -1,
                )
            with torch.no_grad():
                target_frontier_mask = None
                if bool(getattr(
                        policy,
                        "pre_capture_visible_prey_quorum_greedy_frontier_guard",
                        False)):
                    if curr_avail_act is None:
                        raise RuntimeError(
                            "frontier-aligned Q targets require replayed "
                            "Wolfpack action legality"
                        )
                    (
                        target_frontier_mask,
                        target_frontier_eligible,
                        target_frontier_constrained,
                        target_frontier_conflicts,
                        target_frontier_pre_capture,
                    ) = build_batched_wolfpack_frontier_action_mask_from_local_obs(
                        obs_batch=obs_qtot.detach().cpu().numpy(),
                        dones_batch=dones_qtot.detach().cpu().numpy(),
                        available_actions=(
                            curr_avail_act.detach().cpu().numpy()
                        ),
                        num_agents=self.num_agents,
                        max_food_num=int(getattr(
                            self.args, "max_food_num", 0
                        )),
                        sight_radius=int(getattr(
                            self.args, "sight_radius", 0
                        )),
                        prey_max_step=1,
                        capture_quorum=2,
                    )
                    q_target_frontier_diagnostic[
                        "state_count"
                    ] += int(target_frontier_mask.shape[0])
                    q_target_frontier_diagnostic[
                        "pre_capture_state_count"
                    ] += int(target_frontier_pre_capture.sum())
                    q_target_frontier_diagnostic[
                        "eligible_slot_count"
                    ] += int(target_frontier_eligible.sum())
                    q_target_frontier_diagnostic[
                        "constrained_slot_count"
                    ] += int(target_frontier_constrained.sum())
                    q_target_frontier_diagnostic[
                        "conflict_slot_count"
                    ] += int(target_frontier_conflicts.sum())
                greedy, _, _, _ = policy.get_actions(
                    obs_qtot,
                    rnn_obs_qtot,
                    curr_avail_act,
                    None,
                    False,
                    adj_input_qtot,
                    no_sequence,
                    dones_qtot,
                    precomputed_greedy_frontier_action_mask=(
                        target_frontier_mask
                    ),
                )
                if target_frontier_mask is not None:
                    selection_diagnostic = getattr(
                        policy,
                        "last_greedy_frontier_batch_diagnostic",
                        None,
                    )
                    if not isinstance(selection_diagnostic, dict):
                        raise RuntimeError(
                            "frontier-aligned Q target did not report its "
                            "selection exposure"
                        )
                    if int(selection_diagnostic.get(
                            "state_count", -1)) != int(
                                target_frontier_mask.shape[0]):
                        raise RuntimeError(
                            "frontier-aligned Q target exposure mismatch"
                        )
                    q_target_frontier_diagnostic[
                        "reranked_state_count"
                    ] += int(selection_diagnostic[
                        "reranked_state_count"
                    ])
                    q_target_frontier_diagnostic[
                        "reranked_slot_count"
                    ] += int(selection_diagnostic[
                        "reranked_slot_count"
                    ])
                curr_nact_batch_ind = torch.from_numpy(greedy).max(dim=-1)[1].to(self.device)
                target_policy_qs = target_policy.get_q_values(obs_qtot, rnn_target_obs,
                                                              curr_nact_batch_ind.unsqueeze(dim=-1), adj_input_qtot,
                                                              no_sequence, dones_qtot)

            qs.append(policy_qs.reshape(step - 1, batch_size).transpose(0, 1))
            target_qs.append(target_policy_qs.reshape(step - 1, batch_size).transpose(0, 1))
            if self.use_vfunction:
                rnn_critic = policy.init_hidden(1, batch_size)
                policy_v_tot = policy.get_vtot(state_obs_batch, rnn_critic).reshape(step, batch_size)
                target_rnn_critic = target_policy.init_hidden(1, batch_size)
                with torch.no_grad():
                    target_policy_v_tot = target_policy.get_vtot(
                        state_obs_batch,
                        target_rnn_critic
                    ).reshape(step, batch_size)
                policy_v = policy.get_v_values(rnn_obs_q.detach(),
                                               state_obs_batch[:-1].reshape((step - 1) * batch_size, 1, -1),
                                               adj_input_q, no_sequence, dones_q)
                '''with torch.no_grad():
                    target_policy_v =target_policy.get_v_values(rnn_target_obs,state_obs_batch[1:].reshape((step-1)*batch_size,1,-1), adj_input_qtot,no_sequence,dones_qtot)'''
                vtot.append(policy_v_tot[:-1].transpose(0, 1))
                target_vtot.append(target_policy_v_tot[1:].transpose(0, 1))
                fv.append(policy_v.reshape(step - 1, batch_size).transpose(0, 1))
                # target_fv.append(target_policy_v.reshape(step-1,batch_size).transpose(0,1))

        # combine the agent q value sequences to feed into mixer networks
        curr_Q_tot = torch.cat(qs, dim=-1).unsqueeze(-1)
        next_step_Q_tot = torch.cat(target_qs, dim=-1).unsqueeze(-1)

        # all agents must share reward, so get the reward sequence for an agent
        if self._use_valuenorm:
            Q_tot_targets, q_n_step_gate, q_one_step_targets = (
                _build_terminal_gated_n_step_td_targets(
                rewards,
                dones_env_batch,
                self.value_normalizer[p_id].denormalize(next_step_Q_tot),
                terminal_win_rewards,
                self.args.gamma,
                self.q_n_step,
            )
            )
            Q_tot_targets = self.value_normalizer[p_id](Q_tot_targets)
            q_one_step_targets = self.value_normalizer[p_id](
                q_one_step_targets
            )
        else:
            Q_tot_targets, q_n_step_gate, q_one_step_targets = (
                _build_terminal_gated_n_step_td_targets(
                rewards,
                dones_env_batch,
                next_step_Q_tot,
                terminal_win_rewards,
                self.args.gamma,
                self.q_n_step,
            )
            )
        # The uniform sample remains authoritative for ordinary one-step
        # learning.  An appended terminal episode contributes only its gated
        # completion-credit transitions, never its other 176 steps.
        q_loss_transition_mask, q_loss_transition_weight, \
            q_uniform_transition_mask, q_auxiliary_transition_mask, \
            terminal_replay_lane_episode_count = (
            _build_terminal_replay_loss_population(
                valid_transition_mask,
                q_n_step_gate,
                terminal_replay_lane_episode_mask,
                self.q_terminal_replay_loss_weight,
            )
        )
        raw_error = curr_Q_tot - Q_tot_targets.detach()
        error = raw_error * q_loss_transition_mask
        valid_denom = q_loss_transition_mask.sum().clamp_min(1.0)
        # Keep the original uniform objective's denominator exactly intact.
        # The forced completion lane is an additive, explicitly weighted
        # correction rather than an extra population that can dilute uniform
        # one-step learning merely by being appended to the batch.
        uniform_denom = q_uniform_transition_mask.sum().clamp_min(1.0)

        if self.use_huber_loss:
            q_loss_elements = huber_loss(raw_error, self.huber_delta)
        else:
            q_loss_elements = mse_loss(raw_error)

        if self.use_per:
            loss = (
                q_loss_elements.flatten()
                * q_loss_transition_weight.flatten()
                * to_torch(importance_weights).to(**self.tpdv)
            ).sum() / uniform_denom
            new_priorities = error.abs().cpu().detach().numpy().flatten() + self.per_eps
        else:
            loss = (
                q_loss_elements * q_loss_transition_weight
            ).sum() / uniform_denom
            new_priorities = None

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite SDDFG policy loss")

        if self.use_vfunction:
            curr_v_tot = torch.cat(vtot, dim=-1).unsqueeze(-1)
            next_step_v_tot = torch.cat(target_vtot, dim=-1).unsqueeze(-1)
            v_tot_targets, _, _ = _build_terminal_gated_n_step_td_targets(
                rewards,
                dones_env_batch,
                next_step_v_tot,
                terminal_win_rewards,
                self.args.gamma,
                self.q_n_step,
            )
            raw_error_v = curr_v_tot - v_tot_targets.detach()
            error_v = raw_error_v * q_loss_transition_mask
            loss_v = (
                mse_loss(raw_error_v) * q_loss_transition_weight
            ).sum() / uniform_denom

            f_v_tot = torch.cat(fv, dim=-1).unsqueeze(-1)
            raw_error_fv = f_v_tot - curr_v_tot.detach()
            error_fv = raw_error_fv * q_loss_transition_mask
            loss_fv = (
                mse_loss(raw_error_fv) * q_loss_transition_weight
            ).sum() / uniform_denom
            if not torch.isfinite(loss_v):
                raise FloatingPointError("non-finite SDDFG total-value loss")
            if not torch.isfinite(loss_fv):
                raise FloatingPointError("non-finite SDDFG factor-value loss")

        self.policy_optimizer.zero_grad()
        loss.backward()
        q_grad_parameters = []
        rnn_grad_parameters = []
        for policy in self.policies.values():
            rnn_grad_parameters += list(policy.rnn_network.parameters())
            for order in range(1, self.highest_orders + 1):
                q_grad_parameters += list(
                    policy.q_network[order].parameters()
                )
        q_network_grad_norm = _parameter_grad_norm(q_grad_parameters)
        rnn_network_grad_norm = _parameter_grad_norm(rnn_grad_parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.max_grad_norm)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite SDDFG policy gradient norm")
        self.policy_optimizer.step()
        if self.use_vfunction:
            self.critic_vtot_optimizer.zero_grad()
            loss_v.backward()
            vtot_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_vtot_parameters, self.args.max_grad_norm
            )
            if not torch.isfinite(torch.as_tensor(vtot_grad_norm)):
                raise FloatingPointError("non-finite SDDFG total-value gradient norm")
            self.critic_vtot_optimizer.step()

            self.critic_fv_optimizer.zero_grad()
            loss_fv.backward()
            fv_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_fv_parameters, self.args.max_grad_norm
            )
            if not torch.isfinite(torch.as_tensor(fv_grad_norm)):
                raise FloatingPointError("non-finite SDDFG factor-value gradient norm")
            self.critic_fv_optimizer.step()

        train_info = {}
        train_info['loss'] = _to_float(loss)
        train_info['loss_v'] = _to_float(loss_v) if self.use_vfunction else 0.0
        train_info['loss_fv'] = _to_float(loss_fv) if self.use_vfunction else 0.0
        train_info['policy_grad_norm'] = _to_float(grad_norm)
        train_info['q_network_grad_norm'] = q_network_grad_norm
        train_info['rnn_network_grad_norm'] = rnn_network_grad_norm
        train_info['vtot_grad_norm'] = _to_float(vtot_grad_norm) if self.use_vfunction else 0.0
        train_info['fv_grad_norm'] = _to_float(fv_grad_norm) if self.use_vfunction else 0.0
        train_info['valid_transition_ratio'] = _to_float(
            q_loss_transition_mask.mean()
        )
        train_info['active_transition_ratio'] = _to_float(active_transition_mask.mean())
        train_info['q_tot_mean'] = _to_float(
            (curr_Q_tot.detach() * q_loss_transition_mask).sum()
            / valid_denom
        )
        train_info['q_target_mean'] = _to_float(
            (Q_tot_targets.detach() * q_loss_transition_mask).sum()
            / valid_denom
        )
        train_info['td_abs_mean'] = _to_float(
            error.detach().abs().sum() / valid_denom
        )
        valid_sample_mask = q_loss_transition_mask.detach().gt(0.5)
        valid_sampled_rewards = rewards.detach()[valid_sample_mask]
        valid_q_targets = Q_tot_targets.detach()[valid_sample_mask]
        valid_td_errors = error.detach().abs()[valid_sample_mask]
        if valid_sampled_rewards.numel() <= 0:
            raise RuntimeError("SDDFG policy batch has no valid reward sample")
        sampled_reward_mean = valid_sampled_rewards.mean()
        sampled_reward_variance = (
            (valid_sampled_rewards - sampled_reward_mean).pow(2).mean()
        )
        q_target_mean_for_variance = valid_q_targets.mean()
        q_target_variance = (
            (valid_q_targets - q_target_mean_for_variance).pow(2).mean()
        )
        train_info['sampled_reward_mean'] = _to_float(sampled_reward_mean)
        train_info['sampled_reward_std'] = _to_float(
            sampled_reward_variance.sqrt()
        )
        train_info['sampled_reward_min'] = _to_float(
            valid_sampled_rewards.min()
        )
        train_info['sampled_reward_max'] = _to_float(
            valid_sampled_rewards.max()
        )
        train_info['q_target_std'] = _to_float(q_target_variance.sqrt())
        train_info['q_target_min'] = _to_float(valid_q_targets.min())
        train_info['q_target_max'] = _to_float(valid_q_targets.max())
        train_info['td_abs_max'] = _to_float(valid_td_errors.max())
        train_info['policy_lr'] = float(
            self.policy_optimizer.param_groups[0]['lr']
        )
        train_info['policy_grad_was_clipped'] = float(
            _to_float(grad_norm) > float(self.args.max_grad_norm)
        )
        train_info['q_target_n_step'] = float(self.q_n_step)
        valid_q_n_step_gate = (
            q_n_step_gate.float() * q_loss_transition_mask
        )
        gated_transition_count = valid_q_n_step_gate.sum()
        train_info['q_target_terminal_gated'] = float(self.q_n_step > 1)
        frontier_state_count = float(
            q_target_frontier_diagnostic["state_count"]
        )
        frontier_pre_capture_count = float(
            q_target_frontier_diagnostic["pre_capture_state_count"]
        )
        train_info['q_target_frontier_alignment_enabled'] = float(
            bool(getattr(
                self.args,
                "pre_capture_visible_prey_quorum_greedy_frontier_guard",
                False,
            ))
        )
        train_info['q_target_frontier_state_count'] = frontier_state_count
        train_info[
            'q_target_frontier_pre_capture_state_count'
        ] = frontier_pre_capture_count
        train_info[
            'q_target_frontier_pre_capture_state_fraction'
        ] = (
            frontier_pre_capture_count / max(frontier_state_count, 1.0)
        )
        for diagnostic_name in (
                "eligible_slot_count",
                "constrained_slot_count",
                "conflict_slot_count",
                "reranked_state_count",
                "reranked_slot_count"):
            train_info[
                "q_target_frontier_{}".format(diagnostic_name)
            ] = float(q_target_frontier_diagnostic[diagnostic_name])
        train_info['q_target_n_step_gated_transition_count'] = _to_float(
            gated_transition_count
        )
        train_info['q_target_n_step_gated_transition_fraction'] = _to_float(
            gated_transition_count / valid_denom
        )
        train_info['q_terminal_replay_lane_episode_count'] = _to_float(
            terminal_replay_lane_episode_count
        )
        uniform_transition_count = q_uniform_transition_mask.sum()
        auxiliary_transition_count = q_auxiliary_transition_mask.sum()
        detached_q_loss_elements = q_loss_elements.detach()
        train_info['q_terminal_replay_loss_weight'] = float(
            self.q_terminal_replay_loss_weight
        )
        train_info['q_terminal_replay_uniform_transition_count'] = _to_float(
            uniform_transition_count
        )
        train_info['q_terminal_replay_aux_transition_count'] = _to_float(
            auxiliary_transition_count
        )
        train_info['q_terminal_replay_uniform_loss_mean'] = _to_float(
            (detached_q_loss_elements * q_uniform_transition_mask).sum()
            / uniform_transition_count.clamp_min(1.0)
        )
        train_info['q_terminal_replay_aux_loss_mean_unweighted'] = _to_float(
            (detached_q_loss_elements * q_auxiliary_transition_mask).sum()
            / auxiliary_transition_count.clamp_min(1.0)
        )
        train_info[
            'q_terminal_replay_aux_weighted_loss_contribution'
        ] = _to_float(
            (
                detached_q_loss_elements
                * q_auxiliary_transition_mask
                * self.q_terminal_replay_loss_weight
            ).sum()
            / uniform_denom
        )
        train_info['q_terminal_replay_weighted_transition_mass'] = _to_float(
            q_loss_transition_weight.sum()
        )
        train_info['q_terminal_replay_objective_denom'] = _to_float(
            uniform_denom
        )
        gated_target_delta = (
            (Q_tot_targets.detach() - q_one_step_targets.detach()).abs()
            * valid_q_n_step_gate
        )
        train_info['q_target_n_step_gain_abs_mean'] = _to_float(
            gated_target_delta.sum() / gated_transition_count.clamp_min(1.0)
        )
        train_info['q_target_n_step_gain_abs_max'] = _to_float(
            gated_target_delta.max()
        )
        return train_info, new_priorities, idxes

    def train_adj_on_batch(
            self,
            batch,
            use_adj_init,
            use_same_share_obs=None,
            adj_update_round=None,
            candidate_residual_only=False,
            pair_only_objective=False,
            consume_factor_credit_observations=True,
            enable_optimizer_transaction_diagnostics=True,
            diagnostic_policy_id=None,
            diagnostic_transaction_sequence_index=None):
        """Train the full adjacency objective or one finite candidate residual.

        ``candidate_residual_only`` is used only for configured PPO epochs that
        graph/factor clipping stopped.  The caller must replay a sample already
        observed in the current adjacency update; cross-update outcome replay is
        intentionally unsupported.

        ``pair_only_objective`` is reserved for immutable bounded-pending
        strict-pair evidence. It re-evaluates the current adjacency policy but
        permits only the identity-local pair loss to contribute a gradient.

        ``consume_factor_credit_observations`` is true only for the first PPO
        pass over each fresh rollout partition.  Later PPO epochs and finite
        residual/pending replays must not count the same factor outcomes again.
        """

        if bool(candidate_residual_only) and bool(pair_only_objective):
            raise ValueError(
                "candidate residual and pending pair-only modes are mutually "
                "exclusive"
            )

        if len(batch) != 35:
            raise RuntimeError(
                "SDDFG adjacency training requires the current 35-field replay "
                "schema, got {} fields".format(len(batch))
            )
        obs_batch, _share_obs_batch, dones_batch, \
            dones_env_batch, adj_batch, prob_adj_batch, \
            advantages_batch, f_advts_batch, \
            delayed_triplet_credit_batch, \
            delayed_triplet_success_gate_batch, \
            delayed_triplet_future_match_batch, \
            delayed_triplet_future_exact_batch, \
            delayed_triplet_future_partial_batch, \
            capture_to_win_triplet_credit_batch, \
            capture_to_win_quality_gate_batch, \
            pair_pursuit_credit_batch, \
            pair_pursuit_quality_batch, \
            pair_to_triplet_transition_score_batch, \
            triplet_capture_quality_batch, \
            rnn_obs_batch, \
            pair_transition_delay_batch, \
            capture_counts_batch, \
            capture_matched_count_batch, \
            positive_reward_without_capture_batch, \
            offset0_candidate_count_batch, \
            positive_reward_step_batch, \
            previous_adj_batch, \
            capture_to_win_episode_success_gate_batch, \
            failed_episode_capture_count_batch, \
            capture_outcome_diagnostics_batch, \
            candidate_identity_delta_batch, \
            candidate_capture_context_batch, \
            candidate_behavior_batch, \
            replay_population_provenance_batch, \
            available_actions_batch = batch
        tarprob_adj = []
        adj_entropy = []
        adj = to_torch(adj_batch)  # [batch_size,step,num_agent,-1]
        batch_size = adj.shape[0] * adj.shape[1]
        adj = adj.reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        dones = to_torch(dones_batch).reshape(batch_size, self.num_agents, -1).to(self.device)
        dones_env = to_torch(dones_env_batch).reshape(batch_size, -1).to(**self.tpdv)
        prob_adj = to_torch(prob_adj_batch).reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        previous_adj = None
        if previous_adj_batch is not None:
            previous_adj = (
                to_torch(previous_adj_batch)
                .reshape(batch_size, self.num_agents, -1)
                .to(**self.tpdv)
            )
        f_advts = to_torch(f_advts_batch).reshape(batch_size, self.num_factor, -1).to(**self.tpdv)
        replay_graph_advantage = (
            to_torch(advantages_batch)
            .reshape(batch_size, -1)
            .to(**self.tpdv)
        )
        if replay_graph_advantage.shape[-1] != 1:
            raise RuntimeError(
                "graph replay advantage must have one value per transition, "
                "got {}".format(tuple(replay_graph_advantage.shape))
            )
        if delayed_triplet_credit_batch is None:
            delayed_triplet_credit = torch.zeros_like(f_advts)
        else:
            delayed_triplet_credit = (
                to_torch(delayed_triplet_credit_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_success_gate_batch is None:
            delayed_triplet_success_gate = torch.zeros_like(f_advts)
        else:
            delayed_triplet_success_gate = (
                to_torch(delayed_triplet_success_gate_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_future_match_batch is None:
            delayed_triplet_future_match = torch.zeros_like(f_advts)
        else:
            delayed_triplet_future_match = (
                to_torch(delayed_triplet_future_match_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_future_exact_batch is None:
            delayed_triplet_future_exact = torch.zeros_like(f_advts)
        else:
            delayed_triplet_future_exact = (
                to_torch(delayed_triplet_future_exact_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_future_partial_batch is None:
            delayed_triplet_future_partial = torch.zeros_like(f_advts)
        else:
            delayed_triplet_future_partial = (
                to_torch(delayed_triplet_future_partial_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if capture_to_win_triplet_credit_batch is None:
            capture_to_win_triplet_credit = torch.zeros_like(f_advts)
        else:
            capture_to_win_triplet_credit = (
                to_torch(capture_to_win_triplet_credit_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if capture_to_win_quality_gate_batch is None:
            capture_to_win_quality_gate = torch.zeros_like(f_advts)
        else:
            capture_to_win_quality_gate = (
                to_torch(capture_to_win_quality_gate_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if pair_pursuit_credit_batch is None:
            pair_pursuit_credit = torch.zeros_like(f_advts)
        else:
            pair_pursuit_credit = (
                to_torch(pair_pursuit_credit_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if pair_pursuit_quality_batch is None:
            pair_pursuit_quality = torch.zeros_like(f_advts)
        else:
            pair_pursuit_quality = (
                to_torch(pair_pursuit_quality_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if pair_to_triplet_transition_score_batch is None:
            pair_to_triplet_transition_score = torch.zeros_like(f_advts)
        else:
            pair_to_triplet_transition_score = (
                to_torch(pair_to_triplet_transition_score_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if triplet_capture_quality_batch is None:
            triplet_capture_quality = torch.zeros_like(f_advts)
        else:
            triplet_capture_quality = (
                to_torch(triplet_capture_quality_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        num_candidate_factor = (
            self.num_agents * (self.num_agents - 1) // 2
            + self.num_agents * (self.num_agents - 1)
            * (self.num_agents - 2) // 6
        )
        if candidate_identity_delta_batch is None:
            candidate_identity_delta = torch.zeros(
                batch_size,
                num_candidate_factor,
                1,
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            candidate_identity_delta = (
                to_torch(candidate_identity_delta_batch)
                .reshape(batch_size, num_candidate_factor, -1)
                .to(**self.tpdv)
            )
        if bool(pair_only_objective):
            # Pending replay is not ordinary PPO replay. Preserve the real
            # inputs needed to evaluate the current exact factor score, while
            # making every non-pair target structurally inactive.
            replay_graph_advantage = torch.zeros_like(
                replay_graph_advantage
            )
            f_advts = torch.zeros_like(f_advts)
            delayed_triplet_credit = torch.zeros_like(
                delayed_triplet_credit
            )
            capture_to_win_triplet_credit = torch.zeros_like(
                capture_to_win_triplet_credit
            )
            capture_to_win_quality_gate = torch.zeros_like(
                capture_to_win_quality_gate
            )
            candidate_identity_delta = torch.zeros_like(
                candidate_identity_delta
            )
        if candidate_capture_context_batch is None:
            candidate_capture_context = torch.zeros(
                batch_size,
                num_candidate_factor,
                1,
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            candidate_capture_context = (
                to_torch(candidate_capture_context_batch)
                .reshape(batch_size, num_candidate_factor, -1)
                .to(**self.tpdv)
            )
        if candidate_capture_context.shape != candidate_identity_delta.shape:
            raise RuntimeError(
                "candidate capture context shape {} does not match identity "
                "delta {}".format(
                    tuple(candidate_capture_context.shape),
                    tuple(candidate_identity_delta.shape),
                )
            )
        if (
                not bool(torch.isfinite(
                    candidate_capture_context).all().item())
                or bool(torch.any(candidate_capture_context < 0.0).item())):
            raise FloatingPointError(
                "candidate capture context must be finite and non-negative"
            )
        if candidate_behavior_batch is None:
            candidate_behavior = torch.zeros(
                batch_size,
                num_candidate_factor,
                4,
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            candidate_behavior = (
                to_torch(candidate_behavior_batch)
                .reshape(batch_size, num_candidate_factor, 4)
                .to(**self.tpdv)
            )
        if not bool(torch.isfinite(candidate_behavior).all().item()):
            raise FloatingPointError(
                "non-finite rollout candidate behavior metadata"
            )
        replay_population_provenance = (
            _reshape_replay_population_provenance(
                replay_population_provenance_batch,
                batch_size,
            )
            .to(**self.tpdv)
        )
        if not bool(torch.isfinite(
                replay_population_provenance).all().item()):
            raise FloatingPointError(
                "non-finite replay population provenance"
            )
        if pair_transition_delay_batch is None:
            pair_transition_delay = torch.zeros_like(f_advts)
        else:
            pair_transition_delay = (
                to_torch(pair_transition_delay_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )

        def _graph_diagnostic_tensor(values):
            if values is None:
                return torch.zeros_like(dones_env)
            return (
                to_torch(values)
                .reshape(batch_size, -1)
                .to(**self.tpdv)
            )

        capture_counts = _graph_diagnostic_tensor(capture_counts_batch)
        capture_matched_count = _graph_diagnostic_tensor(
            capture_matched_count_batch
        )
        positive_reward_without_capture = _graph_diagnostic_tensor(
            positive_reward_without_capture_batch
        )
        offset0_candidate_count = _graph_diagnostic_tensor(
            offset0_candidate_count_batch
        )
        positive_reward_step = _graph_diagnostic_tensor(
            positive_reward_step_batch
        )
        capture_to_win_episode_success_gate = _graph_diagnostic_tensor(
            capture_to_win_episode_success_gate_batch
        )
        failed_episode_capture_count = _graph_diagnostic_tensor(
            failed_episode_capture_count_batch
        )
        if capture_outcome_diagnostics_batch is None:
            capture_outcome_diagnostics = torch.zeros(
                (batch_size, CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH),
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            capture_outcome_diagnostics = (
                to_torch(capture_outcome_diagnostics_batch)
                .reshape(batch_size, -1)
                .to(**self.tpdv)
            )
            if (
                capture_outcome_diagnostics.shape[-1]
                != CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH
            ):
                raise RuntimeError(
                    "capture outcome diagnostics must have width {}, got {}"
                    .format(
                        CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
                        capture_outcome_diagnostics.shape[-1],
                    )
                )
            if not bool(torch.isfinite(capture_outcome_diagnostics).all().item()):
                raise FloatingPointError(
                    "non-finite capture outcome diagnostics in adjacency batch"
                )
        rnn_obs = to_torch(rnn_obs_batch).reshape(batch_size, self.num_agents, -1).to(
            **self.tpdv)  # [batch_size,step-1,1]
        obs = to_torch(obs_batch).reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        available_actions = (
            to_torch(available_actions_batch)
            .reshape(batch_size, self.num_agents, -1)
            .to(**self.tpdv)
        )

        for _ in self.policy_ids:
            target_prob_adj, entropy = self.adj_network.evaluate_prob(
                obs=obs,
                rnn_obs=rnn_obs,
                use_adj_init=use_adj_init,
                dones=dones.bool(),
                adj=adj,
                previous_adj=previous_adj,
            )

            target_prob = torch.where(
                adj == 1,
                target_prob_adj,
                torch.ones_like(target_prob_adj, dtype=torch.float32)
            ).clamp_min(1e-8)

            adj_entropy.append(entropy)
            tarprob_adj.append(torch.log(target_prob))

        tarlog_prob_adj = torch.cat(tarprob_adj, dim=-1)
        adj_entropy_batch = torch.cat(adj_entropy, dim=-1).unsqueeze(-1)
        candidate_identity_margins, candidate_identity_valid_mask = (
            self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                rnn_obs=rnn_obs,
                dones=dones.bool(),
                adj=adj,
            )
        )
        if candidate_identity_margins.shape != candidate_identity_delta[..., 0].shape:
            raise RuntimeError(
                "candidate identity replay/catalog shape mismatch: {} vs {}"
                .format(
                    tuple(candidate_identity_delta[..., 0].shape),
                    tuple(candidate_identity_margins.shape),
                )
            )
        candidate_identity_current_rank = (
            self.adj_network.canonical_candidate_ranks(
                candidate_identity_margins,
                candidate_identity_valid_mask,
            ).to(candidate_identity_margins.dtype)
        )
        successful_candidate_capture_boundary = (
            _successful_candidate_capture_boundary_diagnostics(
                candidate_capture_context=candidate_capture_context[..., 0],
                episode_success_gate=(
                    capture_to_win_episode_success_gate
                ),
                behavior_candidate_margin=candidate_behavior[..., 0],
                behavior_candidate_rank=candidate_behavior[..., 1],
                behavior_candidate_valid=candidate_behavior[..., 2],
                current_candidate_margin=candidate_identity_margins,
                current_candidate_rank=candidate_identity_current_rank,
                current_candidate_valid=candidate_identity_valid_mask,
                candidate_identity_delta=candidate_identity_delta[..., 0],
            )
        )

        prob_adj_safe = torch.where(
            adj == 1,
            prob_adj,
            torch.ones_like(prob_adj, dtype=torch.float32)
        ).clamp_min(1e-8)
        log_prob_adj = torch.log(prob_adj_safe)

        active_masks = (1.0 - dones.float()).clamp(0.0, 1.0)  # [B, N, 1]

        adj_binary = (adj > 0.5).float()  # [B, N, F]
        factor_size = adj_binary.sum(dim=1, keepdim=True)  # [B, 1, F]
        alive_count = (adj_binary * active_masks).sum(dim=1, keepdim=True)

        # 只有 factor 中所有参与节点均存活，该 factor 才有效
        valid_factor_masks = ((factor_size > 0.0) & (alive_count == factor_size)).float()  # [B, 1, F]

        tarlog_prob_adj = tarlog_prob_adj * active_masks * valid_factor_masks
        log_prob_adj = log_prob_adj * active_masks * valid_factor_masks

        # 同步过滤 factor advantage
        f_advts = f_advts * valid_factor_masks.transpose(1, 2)

        # 后续 order 判定必须使用 valid_adj
        valid_adj = adj_binary * active_masks * valid_factor_masks

        # Each selected column is one pair/triplet factor action. The generator
        # stores the same categorical factor probability on every participating
        # node, so average node log-probability recovers the joint factor
        # log-probability. The previous order-specific square/cube logic counted
        # one factor multiple times and caused most adjacency updates to clip.
        factor_den = factor_size.squeeze(1).clamp_min(1.0)
        target_factor_logp = (
            (tarlog_prob_adj * valid_adj).sum(dim=1) / factor_den
        )
        behavior_factor_logp = (
            (log_prob_adj * valid_adj).sum(dim=1) / factor_den
        )
        factor_mask = valid_factor_masks.transpose(1, 2)
        factor_mask_2d = factor_mask.squeeze(-1)

        # The ordered factor sequence is one structured graph action. Its
        # probability is the product of the conditional factor probabilities,
        # so PPO must use the sum of their log-probabilities. Treating every
        # factor as an independent action gives the wrong objective and makes
        # roster sizes with more factor slots receive larger updates.
        target_graph_logp = (
            target_factor_logp * factor_mask_2d
        ).sum(dim=-1, keepdim=True)
        behavior_graph_logp = (
            behavior_factor_logp * factor_mask_2d
        ).sum(dim=-1, keepdim=True)
        graph_diff_log = (
            target_graph_logp - behavior_graph_logp
        ).clamp(min=-10.0, max=10.0)
        graph_imp_weights = torch.exp(graph_diff_log)
        graph_imp_weights = _nan_to_num_compat(
            graph_imp_weights,
            nan=1.0,
            posinf=1.0 + self.clip_param,
            neginf=1.0 - self.clip_param,
        )

        bad_transitions_mask = dones_env
        clipped_graph_imp_weights = torch.clamp(
            graph_imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        )

        factor_advantage = f_advts.squeeze(-1)
        replay_advantage_split = (
            _separate_replay_graph_and_factor_advantages(
                factor_advantage=factor_advantage,
                replay_graph_advantage=replay_graph_advantage,
                factor_mask=factor_mask_2d,
                graph_return_coefficient=self.adj_return_adv_coef,
            )
        )
        graph_advantage = replay_advantage_split["graph_advantage"]
        legacy_graph_advantage = replay_advantage_split[
            "legacy_graph_advantage"
        ]
        graph_advantage_contamination = replay_advantage_split[
            "graph_advantage_contamination"
        ]
        surr1 = graph_imp_weights * graph_advantage
        surr2 = clipped_graph_imp_weights * graph_advantage
        per_transition_surr = torch.min(surr1, surr2)

        has_valid_factor = (
                factor_mask.sum(dim=-2) > 0
        ).float()

        transition_mask = (
                (1.0 - bad_transitions_mask)
                * has_valid_factor
        )
        graph_trust_weights = torch.ones_like(graph_imp_weights)
        if self.use_adj_ppo_stale_trust:
            # run37 showed that high-clamp early stop reliably stops after one
            # epoch, but the first epoch is already far outside the PPO trust
            # region.  That means the buffer is stale before any replay update
            # happens.  Down-weight those out-of-distribution graph decisions
            # instead of letting clipped stale samples dominate the gradient.
            graph_ratio_deviation = (
                graph_imp_weights.detach() - 1.0
            ).abs()
            graph_stale_excess = (
                graph_ratio_deviation
                - float(self.adj_ppo_stale_trust_clip)
            ).clamp_min(0.0)
            graph_trust_weights = torch.exp(
                -graph_stale_excess
                / float(self.adj_ppo_stale_trust_scale)
            )
            graph_trust_weights = torch.clamp(
                graph_trust_weights,
                min=float(self.adj_ppo_stale_trust_min_weight),
                max=1.0,
            ).detach()
        graph_loss_mask = transition_mask * graph_trust_weights

        graph_rl_loss = -(
                per_transition_surr * graph_loss_mask
        ).sum() / graph_loss_mask.sum().clamp_min(1.0)

        # ``AdjBuffer`` centers the auxiliary factor Q(-V) advantage within
        # each selected graph. Averaging factors into ``graph_advantage``
        # therefore cancels that signal exactly. Apply the centered residual
        # to each sequential conditional factor decision, while retaining the
        # graph-level PPO objective for the trajectory return advantage.
        factor_training_mask = (
            factor_mask_2d * (1.0 - bad_transitions_mask)
        )
        factor_order_2d = factor_size.squeeze(1)
        raw_local_factor_advantage = (
            replay_advantage_split["local_factor_advantage"]
            * (1.0 - bad_transitions_mask)
        )
        order_extra = torch.clamp(factor_order_2d - 2.0, min=0.0)
        positive_order_credit_weight = (
            1.0 + self.adj_order_adv_coef * order_extra
        )
        triplet_order_mask = order_extra > 0.0
        capture_factor_order_mask = (
            ((factor_order_2d == 2.0) | (factor_order_2d == 3.0))
            & (factor_training_mask > 0.0)
        )
        triplet_graph_return_credit = torch.zeros_like(
            raw_local_factor_advantage
        )
        triplet_graph_return_credit_gate = torch.ones_like(
            raw_local_factor_advantage
        )
        graph_return_credit_strength = torch.zeros_like(graph_advantage)
        credit_local_factor_advantage = raw_local_factor_advantage
        triplet_success_selective_gate = (
            delayed_triplet_success_gate.squeeze(-1).clamp(0.0, 1.0)
        )
        if self.use_adj_delayed_triplet_success_gate:
            gate_floor = float(self.adj_delayed_triplet_success_gate_floor)
            if 0.0 < gate_floor < 1.0:
                triplet_success_selective_gate = (
                    (triplet_success_selective_gate - gate_floor)
                    / (1.0 - gate_floor)
                ).clamp(0.0, 1.0)
            elif gate_floor >= 1.0:
                triplet_success_selective_gate = torch.zeros_like(
                    triplet_success_selective_gate
                )
        triplet_graph_return_success_gate = torch.ones_like(
            raw_local_factor_advantage
        )
        if self.adj_triplet_graph_return_credit_require_delayed_gate:
            triplet_graph_return_success_gate = triplet_success_selective_gate
        if (
            self.use_adj_triplet_graph_return_credit
            and self.adj_triplet_graph_return_credit_coef > 0.0
        ):
            # run42 reached high capture counts but its triplet marginal EMA
            # became increasingly negative.  The local factor residual alone
            # misses delayed graph-level payoff, so selected triplets in
            # positive-return graphs receive a bounded supplemental credit.
            graph_return_credit = torch.clamp(
                graph_advantage
                - float(self.adj_triplet_graph_return_credit_min_graph_adv),
                min=0.0,
            )
            graph_return_credit = graph_return_credit * transition_mask
            graph_return_abs_scale = (
                (graph_advantage.abs() * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            ).clamp_min(1e-6)
            graph_return_credit = (
                graph_return_credit
                * float(self.adj_triplet_graph_return_credit_coef)
            )
            if self.adj_triplet_graph_return_credit_cap > 0.0:
                graph_return_credit_cap = (
                    graph_return_abs_scale
                    * float(self.adj_triplet_graph_return_credit_cap)
                )
                graph_return_credit_cap = (
                    torch.ones_like(graph_return_credit)
                    * graph_return_credit_cap
                )
                graph_return_credit = torch.where(
                    graph_return_credit > graph_return_credit_cap,
                    graph_return_credit_cap,
                    graph_return_credit,
                )
            graph_return_credit_strength = (
                graph_return_credit / graph_return_abs_scale
            ).clamp(0.0, 1.0)
            if self.adj_triplet_graph_return_credit_raw_gate_scale > 0.0:
                gate_scale = (
                    graph_return_abs_scale
                    * float(
                        self.adj_triplet_graph_return_credit_raw_gate_scale
                    )
                ).clamp_min(1e-6)
                triplet_graph_return_credit_gate = (
                    (
                        raw_local_factor_advantage
                        + gate_scale
                    )
                    / gate_scale
                ).clamp(0.0, 1.0)
            triplet_graph_return_credit = (
                graph_return_credit
                * triplet_order_mask.to(graph_return_credit.dtype)
                * triplet_graph_return_credit_gate
                * triplet_graph_return_success_gate
                * factor_training_mask
            )
            credit_local_factor_advantage = (
                raw_local_factor_advantage
                + triplet_graph_return_credit
            )
        # Outcome credit has an exact capture-factor identity.  Keep it out of
        # AdjBuffer's shared f_advt tensor and inject it only after the graph
        # mean has been removed.  Otherwise one credited capture factor changes the
        # graph mean and silently broadcasts the opposite residual to every
        # unrelated pair/triplet in the same transition.
        capture_outcome_local_delta = (
            capture_to_win_triplet_credit.squeeze(-1)
            * factor_training_mask
        )
        if bool(torch.any(
                (capture_outcome_local_delta.abs() > 0.0)
                & ~capture_factor_order_mask
        ).item()):
            raise RuntimeError(
                "capture outcome credit leaked into a non-pair/non-triplet "
                "or invalid factor"
            )
        # Signed pair credit has the same exact-identity semantics.  It must not
        # enter f_advt: graph averaging would broadcast the opposite residual
        # to every unrelated factor in this transition.  Keep it as a sparse
        # order-2 factor objective and normalize by target-bearing transitions.
        pair_pursuit_local_delta = (
            pair_pursuit_credit.squeeze(-1)
            * factor_training_mask
        )
        pair_factor_order_mask = (
            (factor_order_2d == 2.0)
            & (factor_training_mask > 0.0)
        )
        if bool(torch.any(
                (pair_pursuit_local_delta.abs() > 0.0)
                & ~pair_factor_order_mask
        ).item()):
            raise RuntimeError(
                "signed pair credit leaked into a non-pair or invalid factor"
            )
        # Pair contrast is consumed by this train_adj_on_batch call, which is
        # one Adam transaction. Update-level centering is insufficient when
        # positive and negative evidence is split across mini-batches: Adam
        # then applies two one-sided steps in sequence. Enforce conservation on
        # the population that actually reaches this optimizer step.
        pair_optimizer_contract = _validate_optimizer_step_pair_credit(
            pair_pursuit_local_delta
        )
        pair_optimizer_positive_mass = pair_optimizer_contract[
            "positive_mass"
        ]
        pair_optimizer_negative_mass = pair_optimizer_contract[
            "negative_mass"
        ]
        pair_optimizer_centered_error = pair_optimizer_contract[
            "centered_error"
        ]
        pair_optimizer_class_complete = pair_optimizer_contract[
            "class_complete"
        ]
        pair_optimizer_contract_valid = pair_optimizer_contract[
            "contract_valid"
        ]
        # Keep the capture-outcome objective separate from generic order credit.
        # Its event/episode mass is already normalized and its identity is exact;
        # mixing it into the shared residual makes later order weighting and the
        # all-factor loss denominator change that mass.
        positive_residual_mask = credit_local_factor_advantage > 0.0
        graph_promotion_strength = torch.ones_like(graph_advantage)
        if self.adj_order_adv_require_positive_graph_adv:
            # A triplet that is locally better than the average factor inside
            # a bad graph can still be a bad coordination choice.  The first
            # binary gate fixed that direction, but run34 showed it was too
            # noisy: promoted fractions collapsed while graph PPO clamp ratios
            # grew.  Use a continuous positive graph-advantage strength when
            # requested; negative-return graphs still get zero positive
            # triplet credit, while mildly positive graphs get a proportional
            # rather than all-or-nothing promotion.
            if self.adj_order_adv_graph_gate_mode == "soft":
                graph_adv_abs_scale = (
                    (graph_advantage.abs() * transition_mask).sum()
                    / transition_mask.sum().clamp_min(1.0)
                ).clamp_min(1e-6)
                graph_adv_abs_scale = (
                    graph_adv_abs_scale
                    * float(self.adj_order_adv_graph_gate_scale)
                ).clamp_min(1e-6)
                positive_graph_advantage = torch.clamp(
                    graph_advantage,
                    min=0.0,
                )
                graph_promotion_strength = (
                    positive_graph_advantage
                    / (
                        positive_graph_advantage
                        + graph_adv_abs_scale
                    )
                )
            else:
                graph_promotion_strength = (
                    graph_advantage > 0.0
                ).float()
        promoted_positive_adv_mask = (
            positive_residual_mask
            & (
                graph_promotion_strength
                > 0.0
            )
        )
        if self.adj_order_adv_positive_only:
            negative_order_credit_weight = (
                1.0
                + self.adj_order_adv_negative_coef
                * order_extra
            )
            # run32 still had a persistent positive order3 PPO loss even
            # though the structure ratio was already pair-heavy.  The gate
            # was partly reacting to negative triplet residuals that had
            # been multiplied by the same order-aware factor used to promote
            # useful triplets.  Split the two directions: positive triplet
            # residuals keep the strong promotion signal, while negative
            # residuals are suppressed at a configurable, usually smaller,
            # scale so the graph learner is driven by marginal gains instead
            # of amplified early triplet noise.
            triplet_positive_weight = (
                positive_order_credit_weight
                * graph_promotion_strength
            )
            positive_credit_weight = torch.where(
                triplet_order_mask,
                triplet_positive_weight,
                torch.ones_like(positive_order_credit_weight),
            )
            order_credit_weight = torch.where(
                positive_residual_mask,
                positive_credit_weight,
                negative_order_credit_weight,
            )
        else:
            order_credit_weight = positive_order_credit_weight
        order_credit_weight = torch.where(
            factor_training_mask > 0.0,
            order_credit_weight,
            torch.ones_like(order_credit_weight),
        )
        base_local_factor_advantage = (
            credit_local_factor_advantage
            * order_credit_weight
        )
        capture_outcome_weighted_delta = capture_outcome_local_delta
        local_factor_advantage = (
            base_local_factor_advantage
            + capture_outcome_weighted_delta
            + pair_pursuit_local_delta
        )
        factor_diff_log = (
            target_factor_logp - behavior_factor_logp
        ).clamp(min=-10.0, max=10.0)
        factor_imp_weights = torch.exp(factor_diff_log)
        factor_imp_weights = _nan_to_num_compat(
            factor_imp_weights,
            nan=1.0,
            posinf=1.0 + self.clip_param,
            neginf=1.0 - self.clip_param,
        )
        clipped_factor_imp_weights = torch.clamp(
            factor_imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        )
        factor_trust_weights = torch.ones_like(factor_imp_weights)
        if self.use_adj_ppo_stale_trust:
            factor_ratio_deviation = (
                factor_imp_weights.detach() - 1.0
            ).abs()
            factor_stale_excess = (
                factor_ratio_deviation
                - float(self.adj_ppo_stale_trust_clip)
            ).clamp_min(0.0)
            factor_trust_weights = torch.exp(
                -factor_stale_excess
                / float(self.adj_ppo_stale_trust_scale)
            )
            factor_trust_weights = torch.clamp(
                factor_trust_weights,
                min=float(self.adj_ppo_stale_trust_min_weight),
                max=1.0,
            ).detach()
        factor_loss_mask = factor_training_mask * factor_trust_weights
        pair_positive_target_mask = (
            pair_pursuit_local_delta > 0.0
        ).to(pair_pursuit_local_delta.dtype)
        pair_negative_target_mask = (
            pair_pursuit_local_delta < 0.0
        ).to(pair_pursuit_local_delta.dtype)
        pair_target_control_mask = (
            (pair_pursuit_local_delta.abs() > 0.0).to(
                pair_pursuit_local_delta.dtype
            )
            * factor_training_mask
        )
        pair_target_clip_indicator = (
            (factor_imp_weights > 1.0 + self.clip_param)
            | (factor_imp_weights < 1.0 - self.clip_param)
        ).to(pair_pursuit_local_delta.dtype)
        pair_target_control_raw_numerator = (
            pair_target_clip_indicator * pair_target_control_mask
        ).sum()
        pair_target_control_raw_denominator = (
            pair_target_control_mask.sum()
        )
        pair_target_control_trusted_mask = (
            pair_target_control_mask * factor_trust_weights
        )
        pair_target_control_trusted_numerator = (
            pair_target_clip_indicator * pair_target_control_trusted_mask
        ).sum()
        pair_target_control_trusted_denominator = (
            pair_target_control_trusted_mask.sum()
        )
        pair_target_control_raw_ratio = (
            pair_target_control_raw_numerator
            / pair_target_control_raw_denominator.clamp_min(1.0)
        )
        pair_target_control_trusted_ratio = (
            pair_target_control_trusted_numerator
            / pair_target_control_trusted_denominator.clamp_min(1.0)
        )
        pair_effective_positive_mass = (
            pair_pursuit_local_delta.clamp_min(0.0)
            * factor_trust_weights
        ).sum()
        pair_effective_negative_mass = (
            (-pair_pursuit_local_delta).clamp_min(0.0)
            * factor_trust_weights
        ).sum()
        pair_positive_trust_mean = (
            (factor_trust_weights * pair_positive_target_mask).sum()
            / pair_positive_target_mask.sum().clamp_min(1.0)
        )
        pair_negative_trust_mean = (
            (factor_trust_weights * pair_negative_target_mask).sum()
            / pair_negative_target_mask.sum().clamp_min(1.0)
        )
        factor_surr1 = factor_imp_weights * local_factor_advantage
        factor_surr2 = (
            clipped_factor_imp_weights * local_factor_advantage
        )
        factor_min_surr = torch.min(factor_surr1, factor_surr2)
        base_factor_surr1 = (
            factor_imp_weights * base_local_factor_advantage
        )
        base_factor_surr2 = (
            clipped_factor_imp_weights * base_local_factor_advantage
        )
        base_factor_min_surr = torch.min(
            base_factor_surr1,
            base_factor_surr2,
        )
        raw_factor_surr1 = (
            factor_imp_weights * raw_local_factor_advantage
        )
        raw_factor_surr2 = (
            clipped_factor_imp_weights * raw_local_factor_advantage
        )
        raw_factor_min_surr = torch.min(
            raw_factor_surr1,
            raw_factor_surr2,
        )
        credit_factor_surr1 = (
            factor_imp_weights * credit_local_factor_advantage
        )
        credit_factor_surr2 = (
            clipped_factor_imp_weights * credit_local_factor_advantage
        )
        credit_factor_min_surr = torch.min(
            credit_factor_surr1,
            credit_factor_surr2,
        )
        base_factor_rl_loss = -(
            base_factor_min_surr
            * factor_loss_mask
        ).sum() / factor_loss_mask.sum().clamp_min(1.0)
        base_factor_population_info = _base_factor_population_diagnostics(
            base_factor_min_surr=base_factor_min_surr,
            factor_training_mask=factor_training_mask,
            factor_loss_mask=factor_loss_mask,
            transition_mask=transition_mask,
            replay_population_provenance=replay_population_provenance,
        )
        capture_outcome_loss_info = compute_identity_local_factor_ppo_loss(
            factor_imp_weights=factor_imp_weights,
            clipped_factor_imp_weights=clipped_factor_imp_weights,
            identity_local_delta=capture_outcome_weighted_delta,
            factor_loss_mask=factor_loss_mask,
            transition_mask=transition_mask,
        )
        capture_outcome_factor_loss_contribution = (
            capture_outcome_loss_info["loss"]
        )
        capture_outcome_positive_factor_loss_contribution = (
            capture_outcome_loss_info["positive_loss"]
        )
        capture_outcome_negative_factor_loss_contribution = (
            capture_outcome_loss_info["negative_loss"]
        )
        capture_outcome_factor_loss_target_count = (
            capture_outcome_loss_info["target_count"]
        )
        capture_outcome_factor_loss_valid_transition_count = (
            capture_outcome_loss_info["valid_transition_count"]
        )
        capture_outcome_factor_loss_factors_per_transition = (
            capture_outcome_loss_info["factors_per_transition"]
        )
        capture_outcome_factor_loss_normalization_version = 2.0
        pair_pursuit_loss_info = compute_identity_local_factor_ppo_loss(
            factor_imp_weights=factor_imp_weights,
            clipped_factor_imp_weights=clipped_factor_imp_weights,
            identity_local_delta=pair_pursuit_local_delta,
            factor_loss_mask=factor_loss_mask,
            transition_mask=transition_mask,
            normalize_by_target_transitions=True,
        )
        pair_pursuit_factor_loss_contribution = (
            pair_pursuit_loss_info["loss"]
        )
        pair_pursuit_positive_factor_loss_contribution = (
            pair_pursuit_loss_info["positive_loss"]
        )
        pair_pursuit_negative_factor_loss_contribution = (
            pair_pursuit_loss_info["negative_loss"]
        )
        pair_pursuit_factor_loss_target_count = (
            pair_pursuit_loss_info["target_count"]
        )
        pair_pursuit_factor_loss_target_transition_count = (
            pair_pursuit_loss_info["target_transition_count"]
        )
        pair_pursuit_factor_loss_valid_transition_count = (
            pair_pursuit_loss_info["valid_transition_count"]
        )
        pair_pursuit_factor_loss_factors_per_transition = (
            pair_pursuit_loss_info["factors_per_transition"]
        )
        pair_pursuit_factor_loss_normalization_denominator = (
            pair_pursuit_loss_info["normalization_denominator"]
        )
        pair_pursuit_factor_loss_normalization_version = 2.0
        candidate_target_mask = candidate_identity_delta[..., 0].abs() > 0.0
        behavior_candidate_margin = candidate_behavior[..., 0]
        behavior_candidate_rank = candidate_behavior[..., 1]
        behavior_candidate_valid = candidate_behavior[..., 2]
        behavior_candidate_version = candidate_behavior[..., 3]
        candidate_identity_loss_info = (
            compute_capture_candidate_identity_active_competitor_loss(
            candidate_competitor_margins=candidate_identity_margins,
            candidate_reference_margins=behavior_candidate_margin,
            candidate_identity_delta=candidate_identity_delta[..., 0],
            candidate_valid_mask=candidate_identity_valid_mask,
            transition_mask=transition_mask,
        ))
        if bool(torch.any(candidate_target_mask & (behavior_candidate_valid <= 0.0)).item()):
            raise RuntimeError(
                "candidate-only outcome target had no feasible rollout "
                "active-slot competitor"
            )
        if bool(torch.any(candidate_target_mask & (behavior_candidate_rank < 1.0)).item()):
            raise RuntimeError(
                "candidate-only outcome target has no rollout policy rank"
            )
        if not bool(torch.all(
                behavior_candidate_version
                == torch.round(behavior_candidate_version)
        ).item()):
            raise RuntimeError("candidate graph-policy version must be integral")
        current_candidate_policy_version = int(getattr(
            self.adj_network,
            "candidate_policy_version",
            0,
        ))
        candidate_policy_age = (
            float(current_candidate_policy_version)
            - behavior_candidate_version
        )
        if bool(torch.any(candidate_target_mask & (candidate_policy_age < 0.0)).item()):
            raise RuntimeError(
                "candidate replay metadata comes from a future graph-policy "
                "version"
            )
        candidate_target_float = candidate_target_mask.float()
        candidate_positive_target = (
            candidate_identity_delta[..., 0] > 0.0
        ).float()
        candidate_negative_target = (
            candidate_identity_delta[..., 0] < 0.0
        ).float()
        candidate_unsatisfied_target = candidate_identity_loss_info[
            "unsatisfied_target_mask"
        ].to(candidate_identity_delta.dtype)
        candidate_effective_delta = (
            candidate_identity_delta[..., 0] * candidate_unsatisfied_target
        )
        candidate_optimized_positive_target = (
            candidate_effective_delta > 0.0
        ).float()
        candidate_optimized_negative_target = (
            candidate_effective_delta < 0.0
        ).float()
        candidate_target_denominator = candidate_target_float.sum().clamp_min(1.0)
        candidate_positive_denominator = candidate_positive_target.sum().clamp_min(1.0)
        candidate_negative_denominator = candidate_negative_target.sum().clamp_min(1.0)
        candidate_optimized_positive_denominator = (
            candidate_optimized_positive_target.sum().clamp_min(1.0)
        )
        candidate_optimized_negative_denominator = (
            candidate_optimized_negative_target.sum().clamp_min(1.0)
        )
        capture_candidate_identity_behavior_margin_mean = (
            (behavior_candidate_margin * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_behavior_rank_mean = (
            (behavior_candidate_rank * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_positive_behavior_margin_mean = (
            (behavior_candidate_margin * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_behavior_margin_mean = (
            (behavior_candidate_margin * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_behavior_rank_mean = (
            (behavior_candidate_rank * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_behavior_rank_mean = (
            (behavior_candidate_rank * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_current_rank_mean = (
            (candidate_identity_current_rank * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_current_rank_mean = (
            (candidate_identity_current_rank * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        current_margin = candidate_identity_loss_info["margin"]
        candidate_margin_change = current_margin - behavior_candidate_margin
        candidate_signed_margin_change = (
            torch.sign(candidate_identity_delta[..., 0])
            * candidate_margin_change
        )
        capture_candidate_identity_positive_margin_change_mean = (
            (candidate_margin_change * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_margin_change_mean = (
            (candidate_margin_change * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_signed_margin_change_mean = (
            (candidate_signed_margin_change * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_signed_margin_change_mean = (
            (candidate_signed_margin_change * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_margin_improved_fraction = (
            (
                (candidate_margin_change > 0.0).float()
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_margin_reduced_fraction = (
            (
                (candidate_margin_change < 0.0).float()
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_boundary_crossed_fraction = (
            (
                (current_margin > 0.0).float()
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_boundary_respected_fraction = (
            (
                (current_margin < 0.0).float()
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_rank_improved_fraction = (
            (
                (
                    candidate_identity_current_rank
                    < behavior_candidate_rank
                ).float()
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_rank_reduced_fraction = (
            (
                (
                    candidate_identity_current_rank
                    > behavior_candidate_rank
                ).float()
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_behavior_valid_fraction = (
            (behavior_candidate_valid * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_policy_age_mean = (
            (candidate_policy_age * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_policy_age_max = torch.where(
            candidate_target_mask,
            candidate_policy_age,
            torch.zeros_like(candidate_policy_age),
        ).max()
        capture_candidate_identity_loss_contribution = (
            candidate_identity_loss_info["loss"]
        )
        capture_candidate_identity_positive_loss_contribution = (
            candidate_identity_loss_info["positive_loss"]
        )
        capture_candidate_identity_negative_loss_contribution = (
            candidate_identity_loss_info["negative_loss"]
        )
        capture_candidate_identity_target_count = (
            candidate_identity_loss_info["target_count"]
        )
        capture_candidate_identity_valid_transition_count = (
            candidate_identity_loss_info["valid_transition_count"]
        )
        capture_candidate_identity_target_transition_count = (
            candidate_identity_loss_info["target_transition_count"]
        )
        capture_candidate_identity_unsatisfied_target_count = (
            candidate_identity_loss_info["unsatisfied_target_count"]
        )
        capture_candidate_identity_positive_unsatisfied_target_count = (
            candidate_identity_loss_info[
                "positive_unsatisfied_target_count"
            ]
        )
        capture_candidate_identity_negative_unsatisfied_target_count = (
            candidate_identity_loss_info[
                "negative_unsatisfied_target_count"
            ]
        )
        capture_candidate_identity_target_transition_fraction = (
            candidate_identity_loss_info["target_transition_fraction"]
        )
        capture_candidate_identity_positive_mass = (
            candidate_identity_loss_info["positive_mass"]
        )
        capture_candidate_identity_negative_mass = (
            candidate_identity_loss_info["negative_mass"]
        )
        capture_candidate_identity_positive_margin_mean = (
            candidate_identity_loss_info["positive_margin_mean"]
        )
        capture_candidate_identity_negative_margin_mean = (
            candidate_identity_loss_info["negative_margin_mean"]
        )
        capture_candidate_identity_positive_signed_margin_mean = (
            candidate_identity_loss_info["positive_signed_margin_mean"]
        )
        capture_candidate_identity_negative_signed_margin_mean = (
            candidate_identity_loss_info["negative_signed_margin_mean"]
        )
        candidate_valid_transition_mask = (
            (candidate_identity_valid_mask > 0.0)
            & (transition_mask > 0.0)
        )
        candidate_valid_margins = candidate_identity_margins[
            candidate_valid_transition_mask
        ]
        if candidate_valid_margins.numel() > 0:
            capture_candidate_identity_valid_margin_mean = (
                candidate_valid_margins.mean()
            )
            capture_candidate_identity_valid_margin_min = (
                candidate_valid_margins.min()
            )
            capture_candidate_identity_valid_margin_max = (
                candidate_valid_margins.max()
            )
        else:
            capture_candidate_identity_valid_margin_mean = (
                candidate_identity_margins.new_tensor(0.0)
            )
            capture_candidate_identity_valid_margin_min = (
                candidate_identity_margins.new_tensor(0.0)
            )
            capture_candidate_identity_valid_margin_max = (
                candidate_identity_margins.new_tensor(0.0)
            )
        base_factor_rl_loss_with_identity_credit = (
            base_factor_rl_loss
            + capture_outcome_factor_loss_contribution
            + pair_pursuit_factor_loss_contribution
        )
        factor_rl_loss = (
            base_factor_rl_loss_with_identity_credit
            + capture_candidate_identity_loss_contribution
        )
        base_rl_loss = (
            graph_rl_loss + base_factor_rl_loss_with_identity_credit
        )
        rl_loss = base_rl_loss + capture_candidate_identity_loss_contribution

        # 仅对环境未结束且在线的 agent 计算图熵。
        valid_env_masks = (1.0 - bad_transitions_mask).view(-1, 1, 1)
        valid_agent_masks = active_masks * valid_env_masks

        # 先对每个 transition 的在线 agent 求平均，再对有效 transition 求平均。
        active_count = active_masks.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        entropy_per_transition = (
                (adj_entropy_batch * active_masks).sum(
                    dim=1, keepdim=True
                )
                / active_count
        )

        entropy_transition_mask = (
                (1.0 - bad_transitions_mask)
                .view(-1, 1, 1)
                * (active_count > 0).float()
        )

        entropy_loss = (
            entropy_per_transition * entropy_transition_mask
        ).sum() / entropy_transition_mask.sum().clamp_min(1.0)

        if bool(pair_only_objective):
            if not bool((
                    pair_pursuit_factor_loss_target_transition_count > 0.0
            ).item()):
                raise RuntimeError(
                    "pending pair-only objective has no target-bearing "
                    "transition"
                )
            zero_objective = (
                pair_pursuit_factor_loss_contribution * 0.0
            )
            graph_rl_loss = zero_objective
            base_factor_rl_loss = zero_objective
            capture_outcome_factor_loss_contribution = zero_objective
            capture_candidate_identity_loss_contribution = zero_objective
            entropy_loss = zero_objective
            base_factor_rl_loss_with_identity_credit = (
                pair_pursuit_factor_loss_contribution
            )
            factor_rl_loss = pair_pursuit_factor_loss_contribution
            base_rl_loss = pair_pursuit_factor_loss_contribution
            rl_loss = pair_pursuit_factor_loss_contribution
            loss = pair_pursuit_factor_loss_contribution
        else:
            loss = _compose_adj_candidate_objective(
                base_rl_loss=base_rl_loss,
                candidate_loss=capture_candidate_identity_loss_contribution,
                entropy=entropy_loss,
                entropy_coef=self.adj_entropy_coef,
                candidate_residual_only=bool(candidate_residual_only),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "non-finite SDDFG adjacency loss"
            )

        active_adj_optimizer = _select_adjacency_optimizer(
            candidate_residual_only=bool(candidate_residual_only),
            pair_only_objective=bool(pair_only_objective),
            standard_optimizer=self.adj_optimizer,
            residual_optimizer=self.candidate_residual_optimizer,
            pair_optimizer=self.pair_pending_optimizer,
        )
        active_adj_optimizer.zero_grad()
        # Optimizers share Parameter objects.  Legacy ``zero_grad`` leaves
        # zero tensors behind, which would let either Adam reuse the other
        # objective's momentum on parameters absent from this graph.
        _clear_parameter_gradients_to_none(self.adj_parameters)

        # Exact candidate-only identity supervision and the on-policy graph
        # objective share the same scoring parameters.  A successful capture
        # factor is absent from the behavior graph by definition, so the base
        # PPO gradient can directly oppose the gradient that would improve its
        # first-reachable competitor margin. Preserve the identity
        # gradient without introducing a coefficient: when the two objectives
        # conflict, remove only the component of the base gradient that points
        # against the candidate gradient (PCGrad-style projection).
        candidate_gradient_norm = loss.new_tensor(0.0)
        candidate_base_gradient_norm = loss.new_tensor(0.0)
        candidate_base_gradient_cosine = loss.new_tensor(0.0)
        candidate_gradient_conflict = loss.new_tensor(0.0)
        candidate_projected_gradient_dot = loss.new_tensor(0.0)
        candidate_total_gradient_norm_ratio = loss.new_tensor(0.0)
        candidate_base_gradient_removed_norm_fraction = loss.new_tensor(0.0)
        candidate_clipped_gradient_dot = loss.new_tensor(0.0)
        candidate_actual_update_descent_dot_before = loss.new_tensor(0.0)
        candidate_actual_update_descent_dot_after = loss.new_tensor(0.0)
        candidate_actual_update_corrected = loss.new_tensor(0.0)
        candidate_actual_update_norm = loss.new_tensor(0.0)
        candidate_actual_update_correction_norm = loss.new_tensor(0.0)
        candidate_actual_update_correction_norm_ratio = loss.new_tensor(0.0)
        candidate_optimizer_state_sync_applied = loss.new_tensor(0.0)
        candidate_optimizer_state_sync_parameter_count = loss.new_tensor(0.0)
        candidate_optimizer_state_update_equation_version = loss.new_tensor(0.0)
        candidate_optimizer_state_raw_reconstruction_error = loss.new_tensor(0.0)
        candidate_optimizer_state_safe_reconstruction_error = loss.new_tensor(0.0)
        candidate_optimizer_state_raw_reconstruction_error_ratio = loss.new_tensor(0.0)
        candidate_optimizer_state_safe_reconstruction_error_ratio = loss.new_tensor(0.0)
        candidate_optimizer_state_reconstruction_tolerance = loss.new_tensor(0.0)
        candidate_optimizer_state_exp_avg_change_norm = loss.new_tensor(0.0)
        candidate_loss_optimizer_change = loss.new_tensor(0.0)
        candidate_positive_optimizer_signed_margin_change = (
            loss.new_tensor(0.0)
        )
        candidate_negative_optimizer_signed_margin_change = (
            loss.new_tensor(0.0)
        )
        candidate_positive_optimizer_rank_improved_fraction = (
            loss.new_tensor(0.0)
        )
        candidate_negative_optimizer_rank_reduced_fraction = (
            loss.new_tensor(0.0)
        )
        pair_gradient_norm = loss.new_tensor(0.0)
        pair_base_factor_gradient_norm = loss.new_tensor(0.0)
        pair_base_factor_gradient_dot = loss.new_tensor(0.0)
        pair_base_factor_gradient_cosine = loss.new_tensor(0.0)
        pair_actual_update_norm = loss.new_tensor(0.0)
        pair_actual_update_descent_dot = loss.new_tensor(0.0)
        pair_actual_update_descent_cosine = loss.new_tensor(0.0)
        pair_target_score_signed_change_mean = loss.new_tensor(0.0)
        pair_target_score_positive_change_mean = loss.new_tensor(0.0)
        pair_target_score_negative_signed_change_mean = loss.new_tensor(0.0)
        pair_target_score_positive_target_count = loss.new_tensor(0.0)
        pair_target_score_negative_target_count = loss.new_tensor(0.0)
        pair_target_score_correct_direction_count = loss.new_tensor(0.0)
        pair_target_score_reverse_direction_count = loss.new_tensor(0.0)
        pair_target_score_approximately_zero_count = loss.new_tensor(0.0)
        pair_target_score_before_after_join_valid = loss.new_tensor(0.0)
        pair_target_score_zero_tolerance = loss.new_tensor(1e-12)
        pair_diagnostic_target_present = bool(
            not candidate_residual_only
            and (
                pair_pursuit_factor_loss_target_transition_count > 0.0
            ).item()
        )
        pair_grads = tuple(None for _ in self.adj_parameters)
        pair_target_score_grads = ()
        pair_target_score_weights = loss.new_zeros((0,))
        pair_boundary_target_grads = ()
        pair_boundary_target_weights = loss.new_zeros((0,))
        pair_boundary_before = None
        pair_boundary_linearized_required_improvement = loss.new_zeros((0,))
        pair_boundary_linearized_crossing_affordable = loss.new_zeros((0,))
        pair_boundary_linearized_allocation_info = None
        pair_boundary_trace_rows = []
        pair_complete_production_boundary = None
        pair_direction_candidate_trace_rows = []
        pair_boundary_gradient_constraint_count = loss.new_tensor(0.0)
        pair_boundary_gradient_projection_intervened = loss.new_tensor(0.0)
        pair_boundary_gradient_min_dot_before = loss.new_tensor(0.0)
        pair_boundary_gradient_min_dot_after = loss.new_tensor(0.0)
        pair_boundary_actual_direction_guard_applied = loss.new_tensor(0.0)
        pair_boundary_actual_min_descent_dot_before = loss.new_tensor(0.0)
        pair_boundary_actual_min_descent_dot_after = loss.new_tensor(0.0)
        pair_boundary_target_count = loss.new_tensor(0.0)
        pair_boundary_correct_direction_count = loss.new_tensor(0.0)
        pair_boundary_reverse_direction_count = loss.new_tensor(0.0)
        pair_boundary_approximately_zero_count = loss.new_tensor(0.0)
        pair_boundary_signed_margin_change_mean = loss.new_tensor(0.0)
        pair_boundary_signed_margin_change_median = loss.new_tensor(0.0)
        pair_boundary_signed_margin_change_worst = loss.new_tensor(0.0)
        pair_boundary_rank_crossing_count = loss.new_tensor(0.0)
        pair_boundary_positive_promotion_count = loss.new_tensor(0.0)
        pair_boundary_negative_eviction_count = loss.new_tensor(0.0)
        pair_boundary_nonlinear_backtrack_count = loss.new_tensor(0.0)
        pair_boundary_nonlinear_backtrack_final_scale = loss.new_tensor(1.0)
        pair_boundary_nonlinear_refinement_count = loss.new_tensor(0.0)
        pair_boundary_nonlinear_invalid_upper_scale = loss.new_tensor(1.0)
        pair_boundary_direction_candidate_count = loss.new_tensor(0.0)
        pair_boundary_direction_valid_candidate_count = loss.new_tensor(0.0)
        pair_boundary_selected_progress_floor_fraction = loss.new_tensor(1.0)
        pair_boundary_progress_min_completion = loss.new_tensor(0.0)
        pair_boundary_progress_mean_completion = loss.new_tensor(0.0)
        pair_boundary_limiting_constraint_code = loss.new_tensor(0.0)
        pair_boundary_limiting_target_ordinal = loss.new_tensor(-1.0)
        pair_boundary_joint_candidate_exact_valid = loss.new_tensor(0.0)
        pair_boundary_joint_lifecycle_exact_valid = loss.new_tensor(0.0)
        pair_target_gradient_constraint_count = loss.new_tensor(0.0)
        pair_target_gradient_projection_intervened = loss.new_tensor(0.0)
        pair_target_gradient_min_dot_before = loss.new_tensor(0.0)
        pair_target_gradient_min_dot_after = loss.new_tensor(0.0)
        pair_target_gradient_projection_delta_norm = loss.new_tensor(0.0)
        pair_target_actual_direction_guard_applied = loss.new_tensor(0.0)
        pair_target_actual_min_descent_dot_before = loss.new_tensor(0.0)
        pair_target_actual_min_descent_dot_after = loss.new_tensor(0.0)
        pair_target_optimizer_state_sync_applied = loss.new_tensor(0.0)
        pair_parameter_before_step = [
            None for _ in self.adj_parameters
        ]
        pair_score_join_before = None
        transaction_diagnostics_enabled = bool(
            enable_optimizer_transaction_diagnostics
            and not candidate_residual_only
        )
        objective_gradient_diagnostics = {
            objective_name: {
                "active": loss.new_tensor(0.0),
                "scalar_loss": loss.new_tensor(0.0),
                "grad_norm": loss.new_tensor(0.0),
                "pair_dot": loss.new_tensor(0.0),
                "pair_cosine": loss.new_tensor(0.0),
                "pair_descent_component": loss.new_tensor(0.0),
                "grads": tuple(None for _ in self.adj_parameters),
            }
            for objective_name in PAIR_OPTIMIZER_OBJECTIVE_NAMES
        }
        objective_scalar_reconstruction_error = loss.new_tensor(0.0)
        objective_scalar_reconstruction_valid = loss.new_tensor(0.0)
        all_objectives_independent_grad_sum_norm = loss.new_tensor(0.0)
        pair_independent_grad_sum_dot = loss.new_tensor(0.0)
        pair_independent_grad_sum_cosine = loss.new_tensor(0.0)
        raw_combined_grad_norm_from_backward = loss.new_tensor(0.0)
        independent_sum_vs_raw_combined_delta_norm = loss.new_tensor(0.0)
        independent_sum_vs_raw_combined_relative_error = loss.new_tensor(0.0)
        independent_sum_reconstruction_valid = loss.new_tensor(0.0)
        pre_projection_combined_grads = tuple(
            None for _ in self.adj_parameters
        )
        pre_projection_combined_grad_norm = loss.new_tensor(0.0)
        post_projection_combined_grad_norm = loss.new_tensor(0.0)
        pair_pre_projection_dot = loss.new_tensor(0.0)
        pair_pre_projection_cosine = loss.new_tensor(0.0)
        pair_post_projection_dot = loss.new_tensor(0.0)
        pair_post_projection_cosine = loss.new_tensor(0.0)
        projection_delta_norm = loss.new_tensor(0.0)
        pair_projection_delta_dot = loss.new_tensor(0.0)
        pair_projection_delta_cosine = loss.new_tensor(0.0)
        gradient_projection_intervened = loss.new_tensor(0.0)
        standard_pair_gradient_projection_intervened = loss.new_tensor(0.0)
        pair_gradient_finite = loss.new_tensor(1.0)
        pair_gradient_zero = loss.new_tensor(1.0)
        pair_raw_gradient_norm = loss.new_tensor(0.0)
        pair_raw_gradient_zero = loss.new_tensor(1.0)
        pair_zero_aggregate_recovered = loss.new_tensor(0.0)
        pair_zero_aggregate_recovered_norm = loss.new_tensor(0.0)
        pair_zero_aggregate_recovered_min_dot = loss.new_tensor(0.0)
        combined_grad_norm_preclip = loss.new_tensor(0.0)
        pair_combined_grad_dot_preclip = loss.new_tensor(0.0)
        pair_combined_grad_cosine_preclip = loss.new_tensor(0.0)
        pair_combined_descent_component_preclip = loss.new_tensor(0.0)
        combined_grad_norm_postclip = loss.new_tensor(0.0)
        pair_combined_grad_dot_postclip = loss.new_tensor(0.0)
        pair_combined_grad_cosine_postclip = loss.new_tensor(0.0)
        pair_combined_descent_component_postclip = loss.new_tensor(0.0)
        gradient_clip_applied = loss.new_tensor(0.0)
        gradient_clip_scale = loss.new_tensor(1.0)
        adam_exp_avg_norm = loss.new_tensor(0.0)
        adam_exp_avg_sq_norm = loss.new_tensor(0.0)
        adam_exp_avg_sq_sqrt_sum = loss.new_tensor(0.0)
        adam_exp_avg_pair_dot = loss.new_tensor(0.0)
        adam_exp_avg_pair_cosine = loss.new_tensor(0.0)
        optimizer_step_before = 0.0
        optimizer_step_before_min = 0.0
        optimizer_step_before_max = 0.0
        optimizer_step_after = 0.0
        optimizer_step_after_min = 0.0
        optimizer_step_after_max = 0.0
        transaction_learning_rate = 0.0
        transaction_adam_beta1 = 0.0
        transaction_adam_beta2 = 0.0
        transaction_adam_eps = 0.0
        transaction_adam_weight_decay = 0.0
        transaction_adam_amsgrad = 0.0
        adam_raw_displacement_norm = loss.new_tensor(0.0)
        adam_raw_pair_dot = loss.new_tensor(0.0)
        adam_raw_pair_descent_dot = loss.new_tensor(0.0)
        adam_raw_pair_descent_cosine = loss.new_tensor(0.0)
        final_displacement_norm = loss.new_tensor(0.0)
        final_pair_dot = loss.new_tensor(0.0)
        final_pair_descent_dot = loss.new_tensor(0.0)
        final_pair_descent_cosine = loss.new_tensor(0.0)
        adam_to_final_displacement_delta_norm = loss.new_tensor(0.0)
        final_parameters_equal_raw_adam = loss.new_tensor(1.0)
        pair_optimizer_isolated = loss.new_tensor(float(
            bool(pair_only_objective)
        ))
        pair_actual_update_direction_guard_applied = loss.new_tensor(0.0)
        pair_optimizer_state_sync_applied = loss.new_tensor(0.0)
        transaction_parameter_before_standard_adam = None
        transaction_adam_raw_displacements = None
        transaction_parameter_steps_before = None
        candidate_target_present = bool(
            (
                candidate_identity_loss_info["target_transition_count"]
                > 0.0
            ).item()
        )
        candidate_objective_active = bool(
            (
                candidate_identity_loss_info["unsatisfied_target_count"]
                > 0.0
            ).item()
        )
        if bool(candidate_residual_only) and not candidate_objective_active:
            # The preceding full epoch may already have reached the exact
            # competitor goal.  A zero-gradient Adam step would still advance
            # momentum and parameters, so skip it transactionally.
            return {
                "candidate_residual_only": 1.0,
                "candidate_residual_optimizer_isolated": 1.0,
                "candidate_residual_inactive_parameter_count": 0.0,
                "candidate_residual_inactive_parameter_update_norm": 0.0,
                "candidate_residual_skipped_satisfied": 1.0,
                "capture_candidate_identity_target_count": _to_float(
                    capture_candidate_identity_target_count
                ),
                "capture_candidate_identity_unsatisfied_target_count": 0.0,
                "capture_candidate_identity_loss_contribution": 0.0,
                "capture_candidate_identity_gradient_norm": 0.0,
            }, None, None
        current_candidate_policy_version = int(getattr(
            self.adj_network,
            "candidate_policy_version",
            0,
        ))
        if current_candidate_policy_version < 0:
            raise RuntimeError("candidate graph-policy version is negative")
        explicit_adj_update_round = adj_update_round is not None
        current_candidate_lifecycle_clock = int(
            adj_update_round
            if explicit_adj_update_round
            else getattr(
                self.adj_network,
                "candidate_lifecycle_clock",
                current_candidate_policy_version,
            )
        )
        if current_candidate_lifecycle_clock < 0:
            raise RuntimeError("candidate lifecycle clock is negative")
        pair_transaction_lifecycle_cache_before = None
        pair_transaction_lifecycle_observations_before = None
        pair_transaction_candidate_policy_version_before = None
        pair_transaction_lifecycle_clock_before = None
        pair_transaction_cpu_rng_before = None
        pair_transaction_cuda_rng_before = None
        pair_transaction_numpy_rng_before = None
        pair_transaction_python_rng_before = None
        pair_transaction_boundary_retention_before = None
        pair_transaction_boundary_retention_next_id_before = None
        pair_transaction_boundary_ordinary_clock_before = None
        pair_transaction_boundary_selection_state_before = None
        pair_transaction_atomic_state_required = bool(
            pair_diagnostic_target_present
            or self._pair_selection_boundary_retention_observations
        )
        if pair_transaction_atomic_state_required:
            # A strict-pair transaction or a later retention observation can
            # fail only after a real Adam step.  Snapshot the small lifecycle
            # state before pruning or supersession, plus every RNG source, so
            # a fail-loud late validation is a genuinely atomic abort rather
            # than a partially committed process-local mutation.
            pair_transaction_lifecycle_cache_before = copy.deepcopy(
                self._candidate_identity_lifecycle
            )
            pair_transaction_lifecycle_observations_before = copy.deepcopy(
                self._candidate_identity_lifecycle_observations
            )
            pair_transaction_candidate_policy_version_before = int(getattr(
                self.adj_network,
                "candidate_policy_version",
                0,
            ))
            pair_transaction_lifecycle_clock_before = int(getattr(
                self.adj_network,
                "candidate_lifecycle_clock",
                0,
            ))
            pair_transaction_cpu_rng_before = torch.get_rng_state().clone()
            if torch.cuda.is_available():
                pair_transaction_cuda_rng_before = [
                    state.clone() for state in torch.cuda.get_rng_state_all()
                ]
            pair_transaction_numpy_rng_before = copy.deepcopy(
                np.random.get_state()
            )
            pair_transaction_python_rng_before = random.getstate()
            pair_transaction_boundary_retention_before = copy.deepcopy(
                self._pair_selection_boundary_retention_observations
            )
            pair_transaction_boundary_retention_next_id_before = int(
                self._pair_selection_boundary_retention_next_id
            )
            pair_transaction_boundary_ordinary_clock_before = int(
                self._pair_selection_boundary_ordinary_update_clock
            )
            pair_transaction_boundary_selection_state_before = (
                self
                ._pair_selection_boundary_retention_selection_state_snapshot()
            )
        lifecycle_expired_count = (
            0
            if bool(pair_only_objective)
            else self._prune_candidate_identity_lifecycle(
                current_candidate_lifecycle_clock
            )
        )
        lifecycle_cache_size = len(self._candidate_identity_lifecycle)
        lifecycle_observation_archive_size = len(
            self._candidate_identity_lifecycle_observations
        )
        lifecycle_new_count = 0
        lifecycle_duplicate_prevented_count = 0.0
        lifecycle_behavioral_progress_transition_count = 0.0
        lifecycle_no_progress_skipped_transition_count = 0.0
        candidate_lifecycle_behavioral_progress_mask = torch.zeros_like(
            candidate_effective_delta
        )
        post_candidate_info = None
        final_candidate_info = None
        lifecycle_protected_target_count = loss.new_tensor(0.0)
        lifecycle_gradient_norm = loss.new_tensor(0.0)
        lifecycle_base_gradient_cosine = loss.new_tensor(0.0)
        lifecycle_gradient_conflict = loss.new_tensor(0.0)
        lifecycle_projected_gradient_dot = loss.new_tensor(0.0)
        lifecycle_actual_update_descent_dot_before = loss.new_tensor(0.0)
        lifecycle_actual_update_descent_dot_after = loss.new_tensor(0.0)
        lifecycle_actual_update_corrected = loss.new_tensor(0.0)
        lifecycle_update_rejected = loss.new_tensor(0.0)
        lifecycle_violation_count = loss.new_tensor(0.0)
        lifecycle_attempted_loss_optimizer_change = loss.new_tensor(0.0)
        lifecycle_loss_optimizer_change = loss.new_tensor(0.0)
        lifecycle_state_sync_applied = loss.new_tensor(0.0)
        lifecycle_age_mean = loss.new_tensor(0.0)
        lifecycle_retention_1_count = loss.new_tensor(0.0)
        lifecycle_retention_5_count = loss.new_tensor(0.0)
        lifecycle_retention_10_count = loss.new_tensor(0.0)
        lifecycle_retention_1_fraction = loss.new_tensor(0.0)
        lifecycle_retention_5_fraction = loss.new_tensor(0.0)
        lifecycle_retention_10_fraction = loss.new_tensor(0.0)
        lifecycle_rank_retention_1_fraction = loss.new_tensor(0.0)
        lifecycle_rank_retention_5_fraction = loss.new_tensor(0.0)
        lifecycle_rank_retention_10_fraction = loss.new_tensor(0.0)
        lifecycle_retention_held_counts = {
            age: {
                "signed_margin_held_count": loss.new_tensor(0.0),
                "rank_held_count": loss.new_tensor(0.0),
                "positive_eligible_count": loss.new_tensor(0.0),
                "positive_signed_margin_held_count": loss.new_tensor(0.0),
                "positive_rank_held_count": loss.new_tensor(0.0),
                "negative_eligible_count": loss.new_tensor(0.0),
                "negative_signed_margin_held_count": loss.new_tensor(0.0),
                "negative_rank_held_count": loss.new_tensor(0.0),
            }
            for age in (1, 5, 10)
        }
        lifecycle_constraint_count = loss.new_tensor(0.0)
        lifecycle_active_constraint_count = loss.new_tensor(0.0)
        lifecycle_min_constraint_dot_before = loss.new_tensor(0.0)
        lifecycle_min_constraint_dot_after = loss.new_tensor(0.0)
        lifecycle_projection_fallback = loss.new_tensor(0.0)
        lifecycle_superseded_constraint_count = loss.new_tensor(0.0)
        lifecycle_actual_min_constraint_dot_before = loss.new_tensor(0.0)
        lifecycle_actual_min_constraint_dot_after = loss.new_tensor(0.0)
        lifecycle_actual_negative_constraint_count_before = loss.new_tensor(0.0)
        lifecycle_actual_negative_constraint_count_after = loss.new_tensor(0.0)
        lifecycle_actual_projection_corrected = loss.new_tensor(0.0)
        lifecycle_actual_projection_correction_norm_ratio = loss.new_tensor(0.0)
        lifecycle_current_priority_repair_intervened = loss.new_tensor(0.0)
        lifecycle_current_priority_min_dot_before = loss.new_tensor(0.0)
        lifecycle_current_priority_min_dot_after = loss.new_tensor(0.0)
        lifecycle_final_linear_min_dot = loss.new_tensor(0.0)
        lifecycle_final_linear_max_tolerance = loss.new_tensor(0.0)
        lifecycle_final_linear_max_normalized_violation = loss.new_tensor(0.0)
        lifecycle_final_linear_rounding_residual_count = loss.new_tensor(0.0)
        lifecycle_final_exact_revalidation_valid = loss.new_tensor(0.0)
        lifecycle_final_exact_min_signed_gap = loss.new_tensor(0.0)
        lifecycle_final_exact_max_tolerance = loss.new_tensor(0.0)
        lifecycle_nonlinear_backtrack_count = loss.new_tensor(0.0)
        lifecycle_current_candidate_nonlinear_violation = loss.new_tensor(0.0)
        lifecycle_target_bearing_update = (
            _candidate_target_bearing_update_diagnostic(
                candidate_target_present,
                loss,
            )
        )
        lifecycle_policy_version_advanced = loss.new_tensor(0.0)
        lifecycle_loss_info = None
        lifecycle_grads = None
        lifecycle_constraint_grads = []
        lifecycle_rnn_obs = None
        lifecycle_dones = None
        lifecycle_adj = None
        lifecycle_delta = None
        lifecycle_transition_mask = None
        lifecycle_target_present = False
        pair_boundary_retention_keyed_entries = []
        pair_boundary_retention_constraint_grads = []
        pair_boundary_retention_initial_keyed_entries = []
        pair_boundary_retention_stop_keys = set()
        pair_boundary_retention_superseded_keys = set()
        pair_boundary_retention_gradient_projection_intervened = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_protected_target_count = loss.new_tensor(0.0)
        pair_boundary_retention_actual_projection_corrected = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_nonlinear_backtrack_count = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_final_scale = loss.new_tensor(1.0)
        pair_boundary_retention_optimizer_state_sync_applied = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_final_exact_min_signed_gap = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_final_exact_max_tolerance = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_final_postcondition_entered = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_final_postcondition_target_count = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_selection_state_backtrack_count = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_selection_state_refinement_count = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_selection_state_final_scale = (
            loss.new_tensor(1.0)
        )
        pair_boundary_retention_selection_state_unsafe_upper_scale = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_selection_state_seen_count_delta = (
            loss.new_tensor(0.0)
        )
        pair_boundary_retention_selection_state_seen_count_integral_valid = (
            loss.new_tensor(1.0)
        )
        pair_boundary_retention_selection_state_component_rows = []
        # Every later optimizer step must respect an existing no-forget cache.
        # In run73 target-bearing mini-batches skipped this block, so a later
        # target in the same adjacency update could erase a just-created target
        # before the first base-only protected update.  The cache remains a
        # constraint only; it is never added to the training objective.
        if lifecycle_cache_size > 0 and not bool(pair_only_objective):
            lifecycle_entry_keys = list(
                self._candidate_identity_lifecycle.keys()
            )
            lifecycle_entries = list(
                self._candidate_identity_lifecycle.values()
            )
            lifecycle_rnn_obs = torch.cat(
                [entry["rnn_obs"] for entry in lifecycle_entries],
                dim=0,
            )
            lifecycle_dones = torch.cat(
                [entry["dones"] for entry in lifecycle_entries],
                dim=0,
            )
            lifecycle_adj = torch.cat(
                [entry["adj"] for entry in lifecycle_entries],
                dim=0,
            )
            lifecycle_previous_adj = torch.cat(
                [entry["previous_adj"] for entry in lifecycle_entries],
                dim=0,
            )
            lifecycle_delta = torch.cat(
                [entry["identity_delta"] for entry in lifecycle_entries],
                dim=0,
            )
            lifecycle_reference_margin = torch.cat(
                [
                    entry["reference_margin"]
                    for entry in lifecycle_entries
                ],
                dim=0,
            )
            lifecycle_reference_rank = torch.cat(
                [entry["reference_rank"] for entry in lifecycle_entries],
                dim=0,
            )
            lifecycle_age = loss.new_tensor([
                current_candidate_lifecycle_clock - int(entry["created_at"])
                for entry in lifecycle_entries
            ])
            lifecycle_margins, lifecycle_valid_mask = (
                self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                    rnn_obs=lifecycle_rnn_obs,
                    dones=lifecycle_dones.bool(),
                    adj=lifecycle_adj,
                )
            )
            lifecycle_transition_mask = torch.ones(
                (lifecycle_margins.shape[0], 1),
                device=lifecycle_margins.device,
                dtype=lifecycle_margins.dtype,
            )
            lifecycle_loss_info = compute_capture_candidate_identity_active_competitor_loss(
                candidate_competitor_margins=lifecycle_margins,
                candidate_reference_margins=lifecycle_reference_margin,
                candidate_identity_delta=lifecycle_delta,
                candidate_valid_mask=lifecycle_valid_mask,
                transition_mask=lifecycle_transition_mask,
            )
            lifecycle_target_present = bool(
                (
                    lifecycle_loss_info["target_transition_count"] > 0.0
                ).item()
            )
            lifecycle_protected_target_count = lifecycle_loss_info[
                "target_count"
            ]
            lifecycle_per_transition_loss = lifecycle_loss_info[
                "per_transition_loss"
            ]
            lifecycle_per_transition_constraint = lifecycle_loss_info[
                "per_transition_constraint"
            ]
            lifecycle_target_rows_for_grad = torch.where(
                lifecycle_delta.abs().sum(dim=1) > 0.0
            )[0]
            lifecycle_constraint_row_indices = (
                lifecycle_target_rows_for_grad.detach().cpu().tolist()
            )
            if lifecycle_constraint_row_indices != list(
                    range(len(lifecycle_entries))):
                raise RuntimeError(
                    "candidate lifecycle cache entries must each contain one "
                    "target transition"
                )
            lifecycle_constraint_entry_keys = list(lifecycle_entry_keys)
            for row in lifecycle_target_rows_for_grad.detach().cpu().tolist():
                lifecycle_constraint_grads.append(torch.autograd.grad(
                    lifecycle_per_transition_constraint[row],
                    self.adj_parameters,
                    retain_graph=True,
                    allow_unused=True,
                ))
            with torch.no_grad():
                lifecycle_target_mask, _lifecycle_population_present = (
                    _lifecycle_target_population_from_delta(lifecycle_delta)
                )
                lifecycle_positive_mask = lifecycle_delta > 0.0
                lifecycle_negative_mask = lifecycle_delta < 0.0
                lifecycle_current_margin = lifecycle_loss_info["margin"]
                lifecycle_current_signed_value = (
                    torch.sign(lifecycle_delta) * lifecycle_current_margin
                )
                lifecycle_reference_signed_value = (
                    torch.sign(lifecycle_delta) * lifecycle_reference_margin
                )
                # If a pre-existing numerical violation is observed, rollback
                # can only preserve the pre-step value. Otherwise protect the
                # registered post-update reference directly.
                lifecycle_signed_floor = _elementwise_minimum_compat(
                    lifecycle_current_signed_value,
                    lifecycle_reference_signed_value,
                )
                lifecycle_current_rank = (
                    self.adj_network.canonical_candidate_ranks(
                        lifecycle_margins,
                        lifecycle_valid_mask,
                    )
                )
                lifecycle_margin_tolerance = (
                    8.0 * 2.220446049250313e-16
                    if lifecycle_current_margin.dtype == torch.float64
                    else 8.0 * 1.1920928955078125e-7
                )
                lifecycle_signed_margin_held = (
                    torch.sign(lifecycle_delta) * lifecycle_current_margin
                    >= (
                        torch.sign(lifecycle_delta)
                        * lifecycle_reference_margin
                        - lifecycle_margin_tolerance
                    )
                )
                lifecycle_rank_held = (
                    (
                        lifecycle_positive_mask
                        & (lifecycle_current_rank <= lifecycle_reference_rank)
                    )
                    | (
                        lifecycle_negative_mask
                        & (lifecycle_current_rank >= lifecycle_reference_rank)
                    )
                )
                lifecycle_age_mean = lifecycle_age.mean()
                retention_values = {}
                for retention_age in (1, 5, 10):
                    eligible = (
                        lifecycle_target_mask
                        & (
                            lifecycle_age >= float(retention_age)
                        ).unsqueeze(-1)
                    )
                    eligible_count = eligible.float().sum()
                    signed_margin_fraction = (
                        (
                            lifecycle_signed_margin_held & eligible
                        ).float().sum()
                        / eligible_count.clamp_min(1.0)
                    )
                    rank_fraction = (
                        (lifecycle_rank_held & eligible).float().sum()
                        / eligible_count.clamp_min(1.0)
                    )
                    positive_eligible = eligible & lifecycle_positive_mask
                    negative_eligible = eligible & lifecycle_negative_mask
                    lifecycle_retention_held_counts[retention_age] = {
                        "signed_margin_held_count": (
                            lifecycle_signed_margin_held & eligible
                        ).float().sum(),
                        "rank_held_count": (
                            lifecycle_rank_held & eligible
                        ).float().sum(),
                        "positive_eligible_count": positive_eligible.float().sum(),
                        "positive_signed_margin_held_count": (
                            lifecycle_signed_margin_held & positive_eligible
                        ).float().sum(),
                        "positive_rank_held_count": (
                            lifecycle_rank_held & positive_eligible
                        ).float().sum(),
                        "negative_eligible_count": negative_eligible.float().sum(),
                        "negative_signed_margin_held_count": (
                            lifecycle_signed_margin_held & negative_eligible
                        ).float().sum(),
                        "negative_rank_held_count": (
                            lifecycle_rank_held & negative_eligible
                        ).float().sum(),
                    }
                    retention_values[retention_age] = (
                        eligible_count,
                        signed_margin_fraction,
                        rank_fraction,
                    )
                (
                    lifecycle_retention_1_count,
                    lifecycle_retention_1_fraction,
                    lifecycle_rank_retention_1_fraction,
                ) = retention_values[1]
                (
                    lifecycle_retention_5_count,
                    lifecycle_retention_5_fraction,
                    lifecycle_rank_retention_5_fraction,
                ) = retention_values[5]
                (
                    lifecycle_retention_10_count,
                    lifecycle_retention_10_fraction,
                    lifecycle_rank_retention_10_fraction,
                ) = retention_values[10]
        # The constraint cache expires after the finite protection horizon, so
        # it cannot define 5/10-round retention.  Override the legacy live-cache
        # diagnostics with exact-age observations from the read-only archive.
        if bool(pair_only_objective):
            lifecycle_retention_diagnostics = {
                age: {
                    "eligible_count": loss.new_tensor(0.0),
                    "signed_margin_fraction": loss.new_tensor(0.0),
                    "rank_fraction": loss.new_tensor(0.0),
                    "signed_margin_held_count": loss.new_tensor(0.0),
                    "rank_held_count": loss.new_tensor(0.0),
                    "positive_eligible_count": loss.new_tensor(0.0),
                    "positive_signed_margin_held_count": (
                        loss.new_tensor(0.0)
                    ),
                    "positive_rank_held_count": loss.new_tensor(0.0),
                    "negative_eligible_count": loss.new_tensor(0.0),
                    "negative_signed_margin_held_count": (
                        loss.new_tensor(0.0)
                    ),
                    "negative_rank_held_count": loss.new_tensor(0.0),
                }
                for age in (1, 5, 10)
            }
        else:
            lifecycle_retention_diagnostics = (
                self._candidate_identity_lifecycle_retention_diagnostics(
                    current_candidate_lifecycle_clock,
                    loss,
                )
            )
        for retention_age in (1, 5, 10):
            retention = lifecycle_retention_diagnostics[retention_age]
            lifecycle_retention_held_counts[retention_age] = {
                key: retention[key]
                for key in (
                    "signed_margin_held_count",
                    "rank_held_count",
                    "positive_eligible_count",
                    "positive_signed_margin_held_count",
                    "positive_rank_held_count",
                    "negative_eligible_count",
                    "negative_signed_margin_held_count",
                    "negative_rank_held_count",
                )
            }
        lifecycle_retention_1_count = lifecycle_retention_diagnostics[1][
            "eligible_count"
        ]
        lifecycle_retention_1_fraction = lifecycle_retention_diagnostics[1][
            "signed_margin_fraction"
        ]
        lifecycle_rank_retention_1_fraction = lifecycle_retention_diagnostics[1][
            "rank_fraction"
        ]
        lifecycle_retention_5_count = lifecycle_retention_diagnostics[5][
            "eligible_count"
        ]
        lifecycle_retention_5_fraction = lifecycle_retention_diagnostics[5][
            "signed_margin_fraction"
        ]
        lifecycle_rank_retention_5_fraction = lifecycle_retention_diagnostics[5][
            "rank_fraction"
        ]
        lifecycle_retention_10_count = lifecycle_retention_diagnostics[10][
            "eligible_count"
        ]
        lifecycle_retention_10_fraction = lifecycle_retention_diagnostics[10][
            "signed_margin_fraction"
        ]
        lifecycle_rank_retention_10_fraction = lifecycle_retention_diagnostics[10][
            "rank_fraction"
        ]
        if transaction_diagnostics_enabled:
            effective_entropy_objective = (
                -float(self.adj_entropy_coef) * entropy_loss
            )
            objective_decomposition = (
                _objective_gradient_decomposition_diagnostics(
                    parameters=self.adj_parameters,
                    objective_specs=(
                        (
                            "graph",
                            graph_rl_loss,
                            bool(
                                not pair_only_objective
                                and (graph_loss_mask.sum() > 0.0).item()
                            ),
                        ),
                        (
                            "base_factor",
                            base_factor_rl_loss,
                            bool(
                                not pair_only_objective
                                and (factor_loss_mask.sum() > 0.0).item()
                            ),
                        ),
                        (
                            "capture_outcome",
                            capture_outcome_factor_loss_contribution,
                            bool(not pair_only_objective and (
                                capture_outcome_factor_loss_target_count
                                > 0.0
                            ).item()),
                        ),
                        (
                            "pair",
                            pair_pursuit_factor_loss_contribution,
                            pair_diagnostic_target_present,
                        ),
                        (
                            "candidate",
                            capture_candidate_identity_loss_contribution,
                            candidate_objective_active,
                        ),
                        (
                            "entropy",
                            effective_entropy_objective,
                            bool(
                                not pair_only_objective
                                and
                                float(self.adj_entropy_coef) != 0.0
                                and (
                                    entropy_transition_mask.sum() > 0.0
                                ).item()
                            ),
                        ),
                    ),
                    total_loss=loss,
                )
            )
            objective_gradient_diagnostics = objective_decomposition[
                "objectives"
            ]
            objective_scalar_reconstruction_error = objective_decomposition[
                "objective_scalar_reconstruction_error"
            ]
            objective_scalar_reconstruction_valid = objective_decomposition[
                "objective_scalar_reconstruction_valid"
            ]
            all_objectives_independent_grad_sum_norm = (
                objective_decomposition["independent_sum_norm"]
            )
            pair_independent_grad_sum_dot = objective_decomposition[
                "pair_independent_sum_dot"
            ]
            pair_independent_grad_sum_cosine = objective_decomposition[
                "pair_independent_sum_cosine"
            ]
            raw_combined_grad_norm_from_backward = objective_decomposition[
                "raw_combined_grad_norm"
            ]
            independent_sum_vs_raw_combined_delta_norm = (
                objective_decomposition[
                    "independent_sum_vs_raw_delta_norm"
                ]
            )
            independent_sum_vs_raw_combined_relative_error = (
                objective_decomposition[
                    "independent_sum_vs_raw_relative_error"
                ]
            )
            independent_sum_reconstruction_valid = objective_decomposition[
                "independent_sum_reconstruction_valid"
            ]
            pre_projection_combined_grads = objective_decomposition[
                "raw_combined_grads"
            ]
            pre_projection_combined_grad_norm = objective_decomposition[
                "raw_combined_grad_norm"
            ]
            pair_pre_projection_dot = objective_decomposition[
                "pair_raw_combined_dot"
            ]
            pair_pre_projection_cosine = objective_decomposition[
                "pair_raw_combined_cosine"
            ]
            pair_grads = objective_gradient_diagnostics["pair"]["grads"]
        if pair_diagnostic_target_present:
            if transaction_diagnostics_enabled:
                pair_base_factor_grads = (
                    objective_gradient_diagnostics["base_factor"]["grads"]
                )
            else:
                pair_grads = torch.autograd.grad(
                    pair_pursuit_factor_loss_contribution,
                    self.adj_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                pair_base_factor_grads = torch.autograd.grad(
                    base_factor_rl_loss,
                    self.adj_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            pair_target_gradient_info = (
                _exact_pair_target_score_gradient_constraints(
                    target_factor_logp=target_factor_logp,
                    pair_local_delta=pair_pursuit_local_delta,
                    parameters=self.adj_parameters,
                )
            )
            pair_target_score_grads = pair_target_gradient_info[
                "constraints"
            ]
            pair_target_score_weights = pair_target_gradient_info[
                "weights"
            ]
            pair_boundary_before = (
                self.adj_network.evaluate_selected_factor_replay_boundaries(
                    rnn_obs=rnn_obs,
                    dones=dones.bool(),
                    adj=adj,
                )
            )
            actionable_pair_mask = pair_pursuit_local_delta != 0.0
            boundary_valid = pair_boundary_before["valid"] > 0.5
            boundary_forced = pair_boundary_before["forced"] > 0.5
            if bool(torch.any(
                    actionable_pair_mask
                    & ((~boundary_valid) | boundary_forced)).item()):
                raise RuntimeError(
                    "actionable strict-pair target has no real selection-"
                    "boundary competitor"
                )
            pair_boundary_gradient_info = (
                _exact_pair_target_score_gradient_constraints(
                    target_factor_logp=pair_boundary_before["selected_margin"],
                    pair_local_delta=pair_pursuit_local_delta,
                    parameters=self.adj_parameters,
                )
            )
            pair_boundary_target_grads = pair_boundary_gradient_info[
                "constraints"
            ]
            pair_boundary_target_weights = pair_boundary_gradient_info[
                "weights"
            ]
            pair_boundary_gradient_constraint_count = loss.new_tensor(
                pair_boundary_gradient_info["target_count"]
            )
            if (
                    pair_boundary_gradient_info["target_count"]
                    != pair_target_gradient_info["target_count"]):
                raise RuntimeError(
                    "strict-pair exact-score and selection-boundary target "
                    "populations differ"
                )
            pair_target_gradient_constraint_count = loss.new_tensor(
                pair_target_gradient_info["target_count"]
            )
            raw_pair_gradient_info = _pair_gradient_direction_diagnostics(
                pair_grads=pair_grads,
                base_factor_grads=pair_base_factor_grads,
                reference=loss,
            )
            pair_raw_gradient_norm = raw_pair_gradient_info["pair_norm"]
            pair_raw_gradient_zero = loss.new_tensor(float(
                not bool((pair_raw_gradient_norm > 0.0).item())
            ))
            pair_gradient_finite = loss.new_tensor(float(
                _gradient_tuple_is_finite(pair_grads)
            ))
            if not bool((pair_gradient_finite > 0.0).item()):
                raise FloatingPointError(
                    "non-finite strict pair adjacency gradient"
                )
            if (
                    bool(pair_only_objective)
                    and bool((pair_raw_gradient_zero > 0.0).item())):
                raise PairPendingZeroGradientError(
                    "pending pair-only target produced zero adjacency "
                    "gradient"
                )
            if (
                    not bool(pair_only_objective)
                    and bool((pair_raw_gradient_zero > 0.0).item())):
                pair_grads, zero_recovery_info = (
                    _recover_standard_zero_aggregate_pair_gradient(
                        pair_grads=pair_grads,
                        pair_target_score_grads=pair_target_score_grads,
                        pair_boundary_target_grads=(
                            pair_boundary_target_grads
                        ),
                        reference=loss,
                    )
                )
                pair_zero_aggregate_recovered = loss.new_tensor(1.0)
                pair_zero_aggregate_recovered_norm = loss.new_tensor(
                    zero_recovery_info["recovered_norm"]
                )
                pair_zero_aggregate_recovered_min_dot = loss.new_tensor(
                    zero_recovery_info["minimum_dot"]
                )
            pair_gradient_info = _pair_gradient_direction_diagnostics(
                pair_grads=pair_grads,
                base_factor_grads=pair_base_factor_grads,
                reference=loss,
            )
            pair_gradient_norm = pair_gradient_info["pair_norm"]
            pair_gradient_zero = loss.new_tensor(float(
                not bool((pair_gradient_norm > 0.0).item())
            ))
            if bool((pair_gradient_zero > 0.0).item()):
                raise RuntimeError(
                    "effective strict pair direction has zero adjacency "
                    "gradient"
                )
            pair_base_factor_gradient_norm = pair_gradient_info[
                "base_factor_norm"
            ]
            pair_base_factor_gradient_dot = pair_gradient_info[
                "pair_base_dot"
            ]
            pair_base_factor_gradient_cosine = pair_gradient_info[
                "pair_base_cosine"
            ]
            pair_parameter_before_step = [
                (
                    parameter.detach().clone()
                    if pair_grad is not None else None
                )
                for parameter, pair_grad in zip(
                    self.adj_parameters,
                    pair_grads,
                )
            ]
            if transaction_diagnostics_enabled:
                pair_score_join_before = {
                    "transition_factor_adjacency": adj.detach().clone(),
                    "active_mask": active_masks.detach().clone(),
                    "valid_factor_mask": valid_factor_masks.detach().clone(),
                    "valid_adjacency": valid_adj.detach().clone(),
                    "factor_denominator": factor_den.detach().clone(),
                    "pair_local_delta": (
                        pair_pursuit_local_delta.detach().clone()
                    ),
                }
        if candidate_objective_active:
            candidate_grads = torch.autograd.grad(
                capture_candidate_identity_loss_contribution,
                self.adj_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            base_loss_without_candidate = (
                loss - capture_candidate_identity_loss_contribution
            )
            base_grads = torch.autograd.grad(
                base_loss_without_candidate,
                self.adj_parameters,
                allow_unused=True,
            )
            candidate_norm_sq = loss.new_tensor(0.0)
            base_norm_sq = loss.new_tensor(0.0)
            base_candidate_dot = loss.new_tensor(0.0)
            for base_grad, candidate_grad in zip(
                    base_grads, candidate_grads):
                if base_grad is not None:
                    base_norm_sq = base_norm_sq + base_grad.pow(2).sum()
                if candidate_grad is not None:
                    candidate_norm_sq = (
                        candidate_norm_sq + candidate_grad.pow(2).sum()
                    )
                if base_grad is not None and candidate_grad is not None:
                    base_candidate_dot = (
                        base_candidate_dot
                        + (base_grad * candidate_grad).sum()
                    )
            if not bool(torch.isfinite(candidate_norm_sq).item()):
                raise FloatingPointError(
                    "non-finite candidate identity gradient norm"
                )
            if not bool((candidate_norm_sq > 0.0).item()):
                raise RuntimeError(
                    "candidate identity target produced no adjacency gradient"
                )
            if not bool(
                    torch.isfinite(base_norm_sq).item()
                    and torch.isfinite(base_candidate_dot).item()):
                raise FloatingPointError(
                    "non-finite base/candidate adjacency gradient diagnostic"
                )
            conflict = bool((base_candidate_dot < 0.0).item())
            projection_scale = loss.new_tensor(0.0)
            if conflict:
                projection_scale = base_candidate_dot / candidate_norm_sq
                candidate_gradient_conflict = loss.new_tensor(1.0)
            for parameter, base_grad, candidate_grad in zip(
                    self.adj_parameters, base_grads, candidate_grads):
                combined_grad = None
                if base_grad is not None:
                    combined_grad = base_grad
                    if conflict and candidate_grad is not None:
                        combined_grad = (
                            combined_grad
                            - projection_scale * candidate_grad
                        )
                if candidate_grad is not None:
                    combined_grad = (
                        candidate_grad
                        if combined_grad is None
                        else combined_grad + candidate_grad
                    )
                if combined_grad is not None:
                    parameter.grad = combined_grad.detach()
            candidate_gradient_norm = torch.sqrt(candidate_norm_sq)
            candidate_base_gradient_norm = torch.sqrt(base_norm_sq)
            candidate_base_gradient_cosine = (
                base_candidate_dot
                / (
                    candidate_gradient_norm
                    * candidate_base_gradient_norm
                ).clamp_min(1e-12)
            )
            combined_norm_sq = loss.new_tensor(0.0)
            combined_candidate_dot = loss.new_tensor(0.0)
            for parameter, candidate_grad in zip(
                    self.adj_parameters, candidate_grads):
                if parameter.grad is not None:
                    combined_norm_sq = (
                        combined_norm_sq + parameter.grad.pow(2).sum()
                    )
                    if candidate_grad is not None:
                        combined_candidate_dot = (
                            combined_candidate_dot
                            + (parameter.grad * candidate_grad).sum()
                        )
            candidate_projected_gradient_dot = combined_candidate_dot
            candidate_total_gradient_norm_ratio = (
                candidate_gradient_norm
                / torch.sqrt(combined_norm_sq).clamp_min(1e-12)
            )
            if conflict:
                candidate_base_gradient_removed_norm_fraction = (
                    projection_scale.abs() * candidate_gradient_norm
                    / candidate_base_gradient_norm.clamp_min(1e-12)
                )
            if not bool((combined_candidate_dot > 0.0).item()):
                raise RuntimeError(
                    "projected adjacency gradient does not preserve candidate "
                    "identity descent direction"
                )
        elif lifecycle_target_present:
            # No current candidate target: start from the ordinary graph loss.
            # The individual lifecycle constraints are applied below, through
            # the same path also used by target-bearing updates.
            base_grads = torch.autograd.grad(
                loss,
                self.adj_parameters,
                allow_unused=True,
            )
            for parameter, base_grad in zip(self.adj_parameters, base_grads):
                if base_grad is not None:
                    parameter.grad = base_grad.detach()
        else:
            loss.backward()

        if pair_diagnostic_target_present:
            # Protect the exact current pair term before any cached lifecycle
            # projection.  The lifecycle step below then treats it as a strict
            # current priority, so stale cache rows cannot silently turn it
            # back into an ascent direction.
            if not bool(pair_only_objective):
                (
                    protected_standard_pair_grads,
                    standard_pair_projection_info,
                ) = _preserve_pair_gradient_in_standard_transaction(
                    proposed_grads=[
                        parameter.grad for parameter in self.adj_parameters
                    ],
                    pair_grads=pair_grads,
                    candidate_grads=(
                        candidate_grads
                        if candidate_objective_active
                        else tuple(None for _ in self.adj_parameters)
                    ),
                    reference=loss,
                )
                for parameter, protected_grad in zip(
                        self.adj_parameters,
                        protected_standard_pair_grads):
                    parameter.grad = (
                        None
                        if protected_grad is None
                        else protected_grad.detach()
                    )
                standard_pair_gradient_projection_intervened = (
                    loss.new_tensor(
                        standard_pair_projection_info["intervened"]
                    )
                )
            pair_target_minimum_dots = (
                _pair_target_mass_preserving_minimum_dots(
                    pair_target_score_grads=pair_target_score_grads,
                    pair_target_weights=pair_target_score_weights,
                    pair_grads=pair_grads,
                    reference=loss,
                )
            )
            pair_boundary_minimum_dots = (
                _pair_target_mass_preserving_minimum_dots(
                    pair_target_score_grads=pair_boundary_target_grads,
                    pair_target_weights=pair_boundary_target_weights,
                    pair_grads=pair_grads,
                    reference=loss,
                )
            )
            target_proposed_grads = tuple(
                parameter.grad for parameter in self.adj_parameters
            )
            target_dots_before = torch.stack([
                _gradient_tuple_dot(
                    target_proposed_grads,
                    target_grads,
                    loss,
                )
                for target_grads in pair_target_score_grads
            ])
            boundary_dots_before = torch.stack([
                _gradient_tuple_dot(
                    target_proposed_grads,
                    boundary_target_grads,
                    loss,
                )
                for boundary_target_grads in pair_boundary_target_grads
            ])
            target_constraints = (
                list(pair_target_score_grads)
                + list(pair_boundary_target_grads)
            )
            target_constraint_floors = (
                list(pair_target_minimum_dots)
                + list(pair_boundary_minimum_dots)
            )
            # Retain strict aggregate-pair and current-candidate descent while
            # resolving the target-local shared-Jacobian conflicts.
            target_constraints.append(pair_grads)
            target_constraint_floors.append(float(
                _strict_gradient_dot_floor(
                    proposed_grads=target_proposed_grads,
                    constraint_grads=pair_grads,
                    reference=loss,
                ).detach().cpu().item()
            ))
            if candidate_objective_active:
                candidate_gradient_dot_before_target_projection = (
                    _gradient_tuple_dot(
                        target_proposed_grads,
                        candidate_grads,
                        loss,
                    )
                )
                if not bool(
                        (
                            candidate_gradient_dot_before_target_projection
                            > 0.0
                        ).item()):
                    raise RuntimeError(
                        "standard exact-pair target projection has no current "
                        "candidate gradient descent to preserve"
                    )
                target_constraints.append(candidate_grads)
                # Preserve the candidate descent already established by the
                # ordinary-gradient guard.  A near-zero symbolic floor can
                # round back through zero when the projected float32
                # displacement is written to parameters.
                target_constraint_floors.append(float(
                    candidate_gradient_dot_before_target_projection
                    .detach().cpu().item()
                ))
            (
                target_protected_grads,
                target_projection_info,
            ) = _project_gradient_tuple_to_minimum_dots(
                proposed_grads=target_proposed_grads,
                constraint_grads=target_constraints,
                minimum_dots=target_constraint_floors,
                reference=loss,
                diagnostic_name=(
                    "pending exact-pair target gradient"
                    if bool(pair_only_objective)
                    else "standard exact-pair target gradient"
                ),
            )
            target_dots_after = torch.stack([
                _gradient_tuple_dot(
                    target_protected_grads,
                    target_grads,
                    loss,
                )
                for target_grads in pair_target_score_grads
            ])
            boundary_dots_after = torch.stack([
                _gradient_tuple_dot(
                    target_protected_grads,
                    boundary_target_grads,
                    loss,
                )
                for boundary_target_grads in pair_boundary_target_grads
            ])
            if not bool(
                    torch.all(target_dots_after > 0.0).item()
                    and torch.all(boundary_dots_after > 0.0).item()):
                raise RuntimeError(
                    "{} exact-pair target or selection-boundary gradient "
                    "lost strict descent".format(
                        "pending"
                        if bool(pair_only_objective)
                        else "standard"
                    )
                )
            for parameter, protected_grad in zip(
                    self.adj_parameters, target_protected_grads):
                parameter.grad = (
                    None if protected_grad is None
                    else protected_grad.detach()
                )
            pair_target_gradient_projection_intervened = loss.new_tensor(
                target_projection_info["intervened"]
            )
            pair_target_gradient_min_dot_before = target_dots_before.min()
            pair_target_gradient_min_dot_after = target_dots_after.min()
            pair_boundary_gradient_projection_intervened = loss.new_tensor(
                float(bool(torch.any(
                    boundary_dots_before <= 0.0
                ).item()))
            )
            pair_boundary_gradient_min_dot_before = (
                boundary_dots_before.min()
            )
            pair_boundary_gradient_min_dot_after = boundary_dots_after.min()
            pair_target_gradient_projection_delta_norm = loss.new_tensor(
                target_projection_info["projection_delta_norm"]
            )

        if lifecycle_target_present:
            # Aggregate gradient fields are retained for backward-compatible
            # diagnostics, but the actual projection enforces every cached
            # transition separately.  This closes the run73 failure where the
            # aggregate lifecycle loss decreased while 139 individual cached
            # transitions worsened and triggered 67 coarse rollbacks.
            proposed_grads = [parameter.grad for parameter in self.adj_parameters]
            current_priority_grads = None
            additional_priority_grads = ()
            if candidate_objective_active:
                current_priority_grads = candidate_grads
                if (
                        pair_diagnostic_target_present
                        and not bool(pair_only_objective)):
                    additional_priority_grads = (
                        (pair_grads,)
                        + tuple(pair_target_score_grads)
                        + tuple(pair_boundary_target_grads)
                    )
            elif (
                    pair_diagnostic_target_present
                    and not bool(pair_only_objective)):
                current_priority_grads = pair_grads
                additional_priority_grads = (
                    tuple(pair_target_score_grads)
                    + tuple(pair_boundary_target_grads)
                )
            if current_priority_grads is not None:
                if len(lifecycle_constraint_grads) != len(
                        lifecycle_constraint_entry_keys):
                    raise RuntimeError(
                        "candidate lifecycle cache rows and gradient constraints "
                        "are misaligned"
                    )
                (
                    projected_grads,
                    enforced_constraint_indices,
                    superseded_constraint_indices,
                    lifecycle_projection_info,
                ) = _project_with_current_candidate_priority(
                    proposed_grads=proposed_grads,
                    candidate_grads=current_priority_grads,
                    lifecycle_constraint_grads=lifecycle_constraint_grads,
                    reference=loss,
                    additional_priority_grads=additional_priority_grads,
                )
                lifecycle_superseded_constraint_count = loss.new_tensor(
                    float(len(superseded_constraint_indices))
                )
                if superseded_constraint_indices:
                    # Each cache entry owns one exact target transition.  New
                    # real evidence supersedes only rows proven incompatible
                    # with its strict descent direction.
                    for index in superseded_constraint_indices:
                        del self._candidate_identity_lifecycle[
                            lifecycle_constraint_entry_keys[index]
                        ]
                    enforced_row_mask = torch.zeros(
                        lifecycle_delta.shape[0],
                        device=lifecycle_delta.device,
                        dtype=lifecycle_delta.dtype,
                    )
                    for index in enforced_constraint_indices:
                        row = lifecycle_constraint_row_indices[index]
                        enforced_row_mask[row] = 1.0
                    lifecycle_delta = (
                        lifecycle_delta * enforced_row_mask.unsqueeze(-1)
                    )
                    lifecycle_loss_info = (
                        compute_capture_candidate_identity_active_competitor_loss(
                            candidate_competitor_margins=lifecycle_margins,
                            candidate_reference_margins=lifecycle_reference_margin,
                            candidate_identity_delta=lifecycle_delta,
                            candidate_valid_mask=lifecycle_valid_mask,
                            transition_mask=lifecycle_transition_mask,
                        )
                    )
                    lifecycle_per_transition_loss = lifecycle_loss_info[
                        "per_transition_loss"
                    ]
                    lifecycle_constraint_grads = [
                        lifecycle_constraint_grads[index]
                        for index in enforced_constraint_indices
                    ]
                    lifecycle_constraint_row_indices = [
                        lifecycle_constraint_row_indices[index]
                        for index in enforced_constraint_indices
                    ]
                    lifecycle_constraint_entry_keys = [
                        lifecycle_constraint_entry_keys[index]
                        for index in enforced_constraint_indices
                    ]
                    lifecycle_protected_target_count = lifecycle_loss_info[
                        "target_count"
                    ]
                    lifecycle_cache_size = len(
                        self._candidate_identity_lifecycle
                    )
            else:
                projected_grads, lifecycle_projection_info = (
                    _project_gradients_onto_nonincreasing_halfspaces(
                        proposed_grads=proposed_grads,
                        constraint_grads=lifecycle_constraint_grads,
                        reference=loss,
                    )
                )
            lifecycle_grads = _mean_gradient_tuples(
                lifecycle_constraint_grads,
                len(self.adj_parameters),
            )
            lifecycle_norm_sq = _gradient_tuple_dot(
                lifecycle_grads, lifecycle_grads, loss
            )
            proposed_norm_sq = _gradient_tuple_dot(
                proposed_grads, proposed_grads, loss
            )
            lifecycle_proposed_dot = _gradient_tuple_dot(
                proposed_grads, lifecycle_grads, loss
            )
            if not bool(
                    torch.isfinite(lifecycle_norm_sq).item()
                    and torch.isfinite(lifecycle_proposed_dot).item()):
                raise FloatingPointError(
                    "non-finite candidate lifecycle gradient diagnostic"
                )
            if (
                    lifecycle_constraint_grads
                    and not bool((lifecycle_norm_sq > 0.0).item())):
                raise RuntimeError(
                    "candidate lifecycle target produced no adjacency gradient"
                )
            lifecycle_gradient_norm = torch.sqrt(lifecycle_norm_sq)
            lifecycle_base_gradient_cosine = (
                lifecycle_proposed_dot
                / (
                    torch.sqrt(proposed_norm_sq) * lifecycle_gradient_norm
                ).clamp_min(1e-12)
            )
            for parameter, projected_grad in zip(
                    self.adj_parameters, projected_grads):
                parameter.grad = (
                    None if projected_grad is None else projected_grad.detach()
                )
            lifecycle_constraint_count = loss.new_tensor(
                lifecycle_projection_info["constraint_count"]
            )
            lifecycle_active_constraint_count = loss.new_tensor(
                lifecycle_projection_info["active_constraint_count"]
            )
            lifecycle_min_constraint_dot_before = loss.new_tensor(
                lifecycle_projection_info["min_dot_before"]
            )
            lifecycle_min_constraint_dot_after = loss.new_tensor(
                lifecycle_projection_info["min_dot_after"]
            )
            lifecycle_projection_fallback = loss.new_tensor(
                lifecycle_projection_info["fallback"]
            )
            lifecycle_gradient_conflict = loss.new_tensor(float(
                lifecycle_projection_info["min_dot_before"] < 0.0
            ))
            lifecycle_projected_gradient_dot = _gradient_tuple_dot(
                [parameter.grad for parameter in self.adj_parameters],
                lifecycle_grads,
                loss,
            )
            if candidate_objective_active:
                candidate_projected_gradient_dot = _gradient_tuple_dot(
                    [parameter.grad for parameter in self.adj_parameters],
                    candidate_grads,
                    loss,
                )
                if not bool((candidate_projected_gradient_dot > 0.0).item()):
                    raise RuntimeError(
                        "joint candidate/lifecycle constraints have no strict "
                        "current-candidate descent direction"
                    )
        if self._pair_selection_boundary_retention_observations:
            try:
                retention_population = (
                    self
                    ._pair_selection_boundary_retention_constraint_population(
                        reference=loss
                    )
                )
            except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError):
                # This pre-check runs before ``optimizer.step`` and therefore
                # precedes the full parameter/optimizer snapshot below.  It can
                # still follow lifecycle pruning, gradient construction, and
                # exact-context forwards.  Restore every process-local state
                # captured before those operations so a fail-loud pre-check is
                # atomic instead of leaving a half-mutated trainer in memory.
                if not pair_transaction_atomic_state_required:
                    raise RuntimeError(
                        "retention pre-check failed without atomic state"
                    )
                self._candidate_identity_lifecycle = copy.deepcopy(
                    pair_transaction_lifecycle_cache_before
                )
                self._candidate_identity_lifecycle_observations = (
                    copy.deepcopy(
                        pair_transaction_lifecycle_observations_before
                    )
                )
                self.adj_network.candidate_policy_version = int(
                    pair_transaction_candidate_policy_version_before
                )
                self.adj_network.candidate_lifecycle_clock = int(
                    pair_transaction_lifecycle_clock_before
                )
                torch.set_rng_state(pair_transaction_cpu_rng_before)
                if pair_transaction_cuda_rng_before is not None:
                    torch.cuda.set_rng_state_all(
                        pair_transaction_cuda_rng_before
                    )
                np.random.set_state(pair_transaction_numpy_rng_before)
                random.setstate(pair_transaction_python_rng_before)
                self._pair_selection_boundary_retention_observations = (
                    copy.deepcopy(pair_transaction_boundary_retention_before)
                )
                self._pair_selection_boundary_retention_next_id = int(
                    pair_transaction_boundary_retention_next_id_before
                )
                self._pair_selection_boundary_ordinary_update_clock = int(
                    pair_transaction_boundary_ordinary_clock_before
                )
                self._restore_pair_selection_boundary_retention_selection_state(
                    pair_transaction_boundary_selection_state_before
                )
                _clear_parameter_gradients_to_none(self.adj_parameters)
                raise
            pair_boundary_retention_keyed_entries = list(
                retention_population["keyed_entries"]
            )
            pair_boundary_retention_constraint_grads = list(
                retention_population["constraint_grads"]
            )
            pair_boundary_retention_initial_keyed_entries = list(
                pair_boundary_retention_keyed_entries
            )
            pair_boundary_retention_stop_keys.update(
                retention_population["context_invalid_keys"]
            )
            if pair_boundary_retention_constraint_grads:
                proposed_grads = [
                    parameter.grad for parameter in self.adj_parameters
                ]
                current_priority_grads = None
                additional_priority_grads = ()
                if candidate_objective_active:
                    current_priority_grads = candidate_grads
                    if pair_diagnostic_target_present:
                        additional_priority_grads = (
                            (pair_grads,)
                            + tuple(pair_target_score_grads)
                            + tuple(pair_boundary_target_grads)
                        )
                elif pair_diagnostic_target_present:
                    current_priority_grads = pair_grads
                    additional_priority_grads = (
                        tuple(pair_target_score_grads)
                        + tuple(pair_boundary_target_grads)
                    )
                if current_priority_grads is not None:
                    (
                        retention_projected_grads,
                        retention_enforced_indices,
                        retention_superseded_indices,
                        retention_projection_info,
                    ) = _project_with_current_candidate_priority(
                        proposed_grads=proposed_grads,
                        candidate_grads=current_priority_grads,
                        lifecycle_constraint_grads=(
                            pair_boundary_retention_constraint_grads
                        ),
                        reference=loss,
                        additional_priority_grads=additional_priority_grads,
                    )
                else:
                    retention_projected_grads, retention_projection_info = (
                        _project_gradients_onto_nonincreasing_halfspaces(
                            proposed_grads=proposed_grads,
                            constraint_grads=(
                                pair_boundary_retention_constraint_grads
                            ),
                            reference=loss,
                        )
                    )
                    retention_enforced_indices = list(range(len(
                        pair_boundary_retention_constraint_grads
                    )))
                    retention_superseded_indices = []
                for index in retention_superseded_indices:
                    pair_boundary_retention_superseded_keys.add(
                        pair_boundary_retention_keyed_entries[index][0]
                    )
                pair_boundary_retention_keyed_entries = [
                    pair_boundary_retention_keyed_entries[index]
                    for index in retention_enforced_indices
                ]
                pair_boundary_retention_constraint_grads = [
                    pair_boundary_retention_constraint_grads[index]
                    for index in retention_enforced_indices
                ]
                for parameter, projected_grad in zip(
                        self.adj_parameters, retention_projected_grads):
                    parameter.grad = (
                        None
                        if projected_grad is None
                        else projected_grad.detach()
                    )
                pair_boundary_retention_gradient_projection_intervened = (
                    loss.new_tensor(float(
                        retention_projection_info[
                            "active_constraint_count"
                        ] > 0.0
                        or retention_projection_info["fallback"] > 0.0
                    ))
                )
                pair_boundary_retention_protected_target_count = (
                    loss.new_tensor(float(len(
                        pair_boundary_retention_constraint_grads
                    )))
                )
        if transaction_diagnostics_enabled:
            combined_grads_preclip = _clone_parameter_gradients(
                self.adj_parameters
            )
            projection_info = _gradient_projection_delta_diagnostics(
                parameters=self.adj_parameters,
                pre_projection_grads=pre_projection_combined_grads,
                post_projection_grads=combined_grads_preclip,
                pair_grads=pair_grads,
                reference=loss,
            )
            projection_delta_norm = projection_info["delta_norm"]
            pair_projection_delta_dot = projection_info["pair_delta_dot"]
            pair_projection_delta_cosine = projection_info[
                "pair_delta_cosine"
            ]
            pre_projection_combined_grad_norm = projection_info[
                "pre_projection_norm"
            ]
            post_projection_combined_grad_norm = projection_info[
                "post_projection_norm"
            ]
            pair_pre_projection_dot = projection_info[
                "pair_pre_projection_dot"
            ]
            pair_pre_projection_cosine = projection_info[
                "pair_pre_projection_cosine"
            ]
            pair_post_projection_dot = projection_info[
                "pair_post_projection_dot"
            ]
            pair_post_projection_cosine = projection_info[
                "pair_post_projection_cosine"
            ]
            gradient_projection_intervened = loss.new_tensor(float(
                bool(
                    (candidate_gradient_conflict > 0.0).item()
                    or (lifecycle_gradient_conflict > 0.0).item()
                    or (
                        standard_pair_gradient_projection_intervened > 0.0
                    ).item()
                    or (
                        pair_target_gradient_projection_intervened > 0.0
                    ).item()
                    or (
                        pair_boundary_retention_gradient_projection_intervened
                        > 0.0
                    ).item()
                )
            ))
            if not bool((gradient_projection_intervened > 0.0).item()):
                _gradient_reconstruction_diagnostics(
                    parameters=self.adj_parameters,
                    reconstructed_grads=combined_grads_preclip,
                    reference_grads=pre_projection_combined_grads,
                    reference=loss,
                    diagnostic_name="unprojected_optimizer_gradient",
                    fail_loud=True,
                )
            combined_preclip_info = _gradient_tuple_direction_diagnostics(
                reference_grads=pair_grads,
                measured_grads=combined_grads_preclip,
                reference=loss,
            )
            combined_grad_norm_preclip = combined_preclip_info[
                "measured_norm"
            ]
            pair_combined_grad_dot_preclip = combined_preclip_info["dot"]
            pair_combined_grad_cosine_preclip = combined_preclip_info[
                "cosine"
            ]
            pair_combined_descent_component_preclip = (
                combined_preclip_info["descent_component"]
            )
            if not _gradient_tuple_is_finite(combined_grads_preclip):
                raise FloatingPointError(
                    "non-finite pre-clip combined adjacency gradient"
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adj_parameters, self.adj_max_grad_norm)
        if transaction_diagnostics_enabled:
            combined_grads_postclip = _clone_parameter_gradients(
                self.adj_parameters
            )
            combined_postclip_info = _gradient_tuple_direction_diagnostics(
                reference_grads=pair_grads,
                measured_grads=combined_grads_postclip,
                reference=loss,
            )
            combined_grad_norm_postclip = combined_postclip_info[
                "measured_norm"
            ]
            pair_combined_grad_dot_postclip = combined_postclip_info["dot"]
            pair_combined_grad_cosine_postclip = combined_postclip_info[
                "cosine"
            ]
            pair_combined_descent_component_postclip = (
                combined_postclip_info["descent_component"]
            )
            if bool((combined_grad_norm_preclip > 0.0).item()):
                gradient_clip_scale = (
                    combined_grad_norm_postclip
                    / combined_grad_norm_preclip
                )
            else:
                gradient_clip_scale = loss.new_tensor(1.0)
            gradient_clip_applied = loss.new_tensor(float(
                bool((gradient_clip_scale < (1.0 - 1e-7)).item())
            ))
            transaction_parameter_before_standard_adam = [
                parameter.detach().clone()
                for parameter in self.adj_parameters
            ]
            adam_state_info = _adam_state_before_step_diagnostics(
                optimizer=active_adj_optimizer,
                parameters=self.adj_parameters,
                pair_grads=pair_grads,
                reference=loss,
            )
            adam_exp_avg_norm = adam_state_info["exp_avg_norm"]
            adam_exp_avg_sq_norm = adam_state_info["exp_avg_sq_norm"]
            adam_exp_avg_sq_sqrt_sum = adam_state_info[
                "exp_avg_sq_sqrt_sum"
            ]
            adam_exp_avg_pair_dot = adam_state_info["exp_avg_pair_dot"]
            adam_exp_avg_pair_cosine = adam_state_info[
                "exp_avg_pair_cosine"
            ]
            optimizer_step_before = adam_state_info[
                "optimizer_step_before"
            ]
            optimizer_step_before_min = adam_state_info[
                "optimizer_step_before_min"
            ]
            optimizer_step_before_max = adam_state_info[
                "optimizer_step_before_max"
            ]
            transaction_parameter_steps_before = adam_state_info[
                "parameter_steps_before"
            ]
            transaction_learning_rate = adam_state_info["learning_rate"]
            transaction_adam_beta1 = adam_state_info["beta1"]
            transaction_adam_beta2 = adam_state_info["beta2"]
            transaction_adam_eps = adam_state_info["eps"]
            transaction_adam_weight_decay = adam_state_info["weight_decay"]
            transaction_adam_amsgrad = adam_state_info["amsgrad"]
        candidate_parameter_before_step = None
        candidate_raw_parameter_deltas = None
        lifecycle_parameter_before_step = None
        lifecycle_optimizer_state_before_step = None
        lifecycle_raw_parameter_deltas = None
        lifecycle_optimizer_state_after_raw_step = None
        residual_inactive_parameter_before_step = None
        pair_guard_raw_parameter_deltas = None
        pair_optimizer_state_after_raw_step = None
        pair_exact_raw_parameter_deltas = None
        pair_exact_optimizer_state_after_raw_step = None
        pair_transaction_all_parameters_before_step = None
        pair_transaction_active_optimizer_before_step = None
        clipped_combined_norm_sq = loss.new_tensor(0.0)
        if candidate_objective_active:
            candidate_parameter_before_step = []
            for parameter, candidate_grad in zip(
                    self.adj_parameters, candidate_grads):
                if candidate_grad is not None:
                    candidate_parameter_before_step.append(
                        parameter.detach().clone()
                    )
                    if parameter.grad is not None:
                        clipped_combined_norm_sq = (
                            clipped_combined_norm_sq
                            + parameter.grad.pow(2).sum()
                        )
                        candidate_clipped_gradient_dot = (
                            candidate_clipped_gradient_dot
                            + (parameter.grad * candidate_grad).sum()
                        )
                else:
                    candidate_parameter_before_step.append(None)
            if not bool((candidate_clipped_gradient_dot > 0.0).item()):
                raise RuntimeError(
                    "gradient clipping removed candidate identity descent "
                    "direction"
                )
        if lifecycle_target_present:
            lifecycle_parameter_before_step = []
            lifecycle_optimizer_state_before_step = []
            for parameter_index, (parameter, lifecycle_grad) in enumerate(zip(
                    self.adj_parameters, lifecycle_grads)):
                candidate_grad = (
                    candidate_grads[parameter_index]
                    if candidate_objective_active else None
                )
                lifecycle_sensitive = (
                    lifecycle_grad is not None or candidate_grad is not None
                )
                lifecycle_parameter_before_step.append(
                    parameter.detach().clone()
                    if lifecycle_sensitive else None
                )
                lifecycle_optimizer_state_before_step.append(
                    copy.deepcopy(active_adj_optimizer.state.get(parameter, {}))
                    if lifecycle_sensitive else None
                )
        if bool(candidate_residual_only):
            residual_inactive_parameter_before_step = [
                (
                    parameter.detach().clone()
                    if parameter.grad is None else None
                )
                for parameter in self.adj_parameters
            ]
        if pair_transaction_atomic_state_required:
            pair_transaction_all_parameters_before_step = [
                parameter.detach().clone()
                for parameter in self.adj_parameters
            ]
            pair_transaction_active_optimizer_before_step = copy.deepcopy(
                active_adj_optimizer.state_dict()
            )
        if pair_diagnostic_target_present:
            if pair_transaction_all_parameters_before_step is None:
                raise RuntimeError(
                    "strict-pair exact origin requires a complete atomic "
                    "parameter snapshot"
                )
            # Anchor every nonlinear preservation comparison to one fresh
            # production replay at the exact atomic parameter origin.  The
            # differentiable boundary/candidate/lifecycle forwards above are
            # intentionally earlier because they build Jacobians.  Reusing
            # their floating values as the later scale-zero baseline makes two
            # separate production forwards masquerade as one transaction
            # origin and can classify a legal origin as a reverse move.  This
            # refresh changes no gradient and relaxes no positive-scale gate.
            with torch.no_grad():
                pair_boundary_exact_origin = (
                    self.adj_network
                    .evaluate_selected_factor_replay_boundaries(
                        rnn_obs=rnn_obs,
                        dones=dones.bool(),
                        adj=adj,
                        previous_adj=previous_adj,
                        include_behavior_logp=True,
                    )
                )
                for catalog_field in (
                        "selected_candidate_index",
                        "competitor_candidate_index",
                        "valid",
                        "forced",
                        "factor_order",
                        "selected_rank"):
                    if not bool(torch.equal(
                            pair_boundary_exact_origin[catalog_field],
                            pair_boundary_before[catalog_field].detach(),
                    )):
                        raise RuntimeError(
                            "strict-pair selection-boundary catalog changed "
                            "before the atomic optimizer step: field={}".format(
                                catalog_field
                            )
                        )
                pair_boundary_before = {
                    field_name: field_value.detach().clone()
                    for field_name, field_value in (
                        pair_boundary_exact_origin.items()
                    )
                }
                pair_exact_behavior_origin = pair_boundary_before[
                    "behavior_selected_logp"
                ]
                if pair_exact_behavior_origin.shape != target_factor_logp.shape:
                    raise RuntimeError(
                        "strict-pair behavior-score replay shape differs from "
                        "the PPO exact-score population: {} vs {}".format(
                            tuple(pair_exact_behavior_origin.shape),
                            tuple(target_factor_logp.shape),
                        )
                    )
                pair_exact_origin_score_info = (
                    _pair_target_score_change_diagnostics(
                        pre_factor_logp=target_factor_logp.detach(),
                        post_factor_logp=pair_exact_behavior_origin,
                        pair_local_delta=pair_pursuit_local_delta.detach(),
                        zero_tolerance=float(
                            128.0 * _floating_dtype_epsilon(loss)
                        ),
                    )
                )
                pair_exact_origin_nonzero_count = (
                    pair_exact_origin_score_info["correct_direction_count"]
                    + pair_exact_origin_score_info["reverse_direction_count"]
                )
                if bool((pair_exact_origin_nonzero_count > 0.0).item()):
                    raise RuntimeError(
                        "strict-pair atomic behavior-score origin does not "
                        "replay the PPO exact-score baseline"
                    )

                pair_exact_candidate_loss_origin = None
                if candidate_objective_active:
                    (
                        pair_exact_candidate_margins,
                        pair_exact_candidate_valid,
                    ) = (
                        self.adj_network
                        .evaluate_candidate_identity_active_competitor_margins(
                            rnn_obs=rnn_obs,
                            dones=dones.bool(),
                            adj=adj,
                        )
                    )
                    if not bool(torch.equal(
                            pair_exact_candidate_valid > 0.0,
                            candidate_identity_valid_mask > 0.0,
                    )):
                        raise RuntimeError(
                            "current candidate catalog changed before the "
                            "atomic strict-pair optimizer step"
                        )
                    pair_exact_candidate_info = (
                        compute_capture_candidate_identity_active_competitor_loss(
                            candidate_competitor_margins=(
                                pair_exact_candidate_margins
                            ),
                            candidate_reference_margins=(
                                behavior_candidate_margin
                            ),
                            candidate_identity_delta=(
                                candidate_identity_delta[..., 0]
                            ),
                            candidate_valid_mask=pair_exact_candidate_valid,
                            transition_mask=transition_mask,
                        )
                    )
                    pair_exact_candidate_loss_origin = (
                        pair_exact_candidate_info["loss"].detach().clone()
                    )

                if lifecycle_target_present:
                    (
                        pair_exact_lifecycle_margins,
                        pair_exact_lifecycle_valid,
                    ) = (
                        self.adj_network
                        .evaluate_candidate_identity_active_competitor_margins(
                            rnn_obs=lifecycle_rnn_obs,
                            dones=lifecycle_dones.bool(),
                            adj=lifecycle_adj,
                        )
                    )
                    if not bool(torch.equal(
                            pair_exact_lifecycle_valid > 0.0,
                            lifecycle_valid_mask > 0.0,
                    )):
                        raise RuntimeError(
                            "candidate lifecycle catalog changed before the "
                            "atomic strict-pair optimizer step"
                        )
                    pair_exact_lifecycle_signed_origin = (
                        torch.sign(lifecycle_delta)
                        * pair_exact_lifecycle_margins
                    )
                    # A preservation floor cannot be above the state it is
                    # supposed to preserve.  Keep the registered reference when
                    # it is weaker, otherwise rebase only to the actual atomic
                    # origin.  Positive-scale lifecycle acceptance remains exact.
                    lifecycle_signed_floor = _elementwise_minimum_compat(
                        lifecycle_signed_floor,
                        pair_exact_lifecycle_signed_origin,
                    )

        def _rollback_failed_pair_transaction():
            if not pair_transaction_atomic_state_required:
                raise RuntimeError(
                    "pair transaction rollback requested without atomic state"
                )
            if (
                    pair_transaction_all_parameters_before_step is None
                    or pair_transaction_active_optimizer_before_step is None
                    or pair_transaction_lifecycle_cache_before is None
                    or pair_transaction_lifecycle_observations_before is None
                    or pair_transaction_cpu_rng_before is None
                    or pair_transaction_numpy_rng_before is None
                    or pair_transaction_python_rng_before is None
                    or pair_transaction_boundary_retention_before is None
                    or pair_transaction_boundary_retention_next_id_before
                    is None
                    or pair_transaction_boundary_ordinary_clock_before
                    is None
                    or pair_transaction_boundary_selection_state_before
                    is None):
                raise RuntimeError(
                    "pair transaction rollback snapshot is incomplete"
                )
            with torch.no_grad():
                for parameter, before in zip(
                        self.adj_parameters,
                        pair_transaction_all_parameters_before_step):
                    parameter.copy_(before)
            active_adj_optimizer.load_state_dict(copy.deepcopy(
                pair_transaction_active_optimizer_before_step
            ))
            self._candidate_identity_lifecycle = copy.deepcopy(
                pair_transaction_lifecycle_cache_before
            )
            self._candidate_identity_lifecycle_observations = copy.deepcopy(
                pair_transaction_lifecycle_observations_before
            )
            self.adj_network.candidate_policy_version = int(
                pair_transaction_candidate_policy_version_before
            )
            self.adj_network.candidate_lifecycle_clock = int(
                pair_transaction_lifecycle_clock_before
            )
            torch.set_rng_state(pair_transaction_cpu_rng_before)
            if pair_transaction_cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(
                    pair_transaction_cuda_rng_before
                )
            np.random.set_state(pair_transaction_numpy_rng_before)
            random.setstate(pair_transaction_python_rng_before)
            self._pair_selection_boundary_retention_observations = (
                copy.deepcopy(pair_transaction_boundary_retention_before)
            )
            self._pair_selection_boundary_retention_next_id = int(
                pair_transaction_boundary_retention_next_id_before
            )
            self._pair_selection_boundary_ordinary_update_clock = int(
                pair_transaction_boundary_ordinary_clock_before
            )
            self._restore_pair_selection_boundary_retention_selection_state(
                pair_transaction_boundary_selection_state_before
            )
            rollback_failures = []
            if any(
                    not torch.equal(parameter.detach(), before)
                    for parameter, before in zip(
                        self.adj_parameters,
                        pair_transaction_all_parameters_before_step,
                    )):
                rollback_failures.append("parameters")
            if not _transaction_state_equal(
                    active_adj_optimizer.state_dict(),
                    pair_transaction_active_optimizer_before_step):
                rollback_failures.append("optimizer")
            if not _transaction_state_equal(
                    self._candidate_identity_lifecycle,
                    pair_transaction_lifecycle_cache_before):
                rollback_failures.append("lifecycle_cache")
            if not _transaction_state_equal(
                    self._candidate_identity_lifecycle_observations,
                    pair_transaction_lifecycle_observations_before):
                rollback_failures.append("lifecycle_observations")
            if (
                    int(self.adj_network.candidate_policy_version)
                    != int(pair_transaction_candidate_policy_version_before)):
                rollback_failures.append("candidate_policy_version")
            if (
                    int(self.adj_network.candidate_lifecycle_clock)
                    != int(pair_transaction_lifecycle_clock_before)):
                rollback_failures.append("candidate_lifecycle_clock")
            if not torch.equal(
                    torch.get_rng_state(), pair_transaction_cpu_rng_before):
                rollback_failures.append("torch_cpu_rng")
            if pair_transaction_cuda_rng_before is not None:
                if not _transaction_state_equal(
                        torch.cuda.get_rng_state_all(),
                        pair_transaction_cuda_rng_before):
                    rollback_failures.append("torch_cuda_rng")
            if not _transaction_state_equal(
                    np.random.get_state(),
                    pair_transaction_numpy_rng_before):
                rollback_failures.append("numpy_rng")
            if not _transaction_state_equal(
                    random.getstate(),
                    pair_transaction_python_rng_before):
                rollback_failures.append("python_rng")
            if not _transaction_state_equal(
                    self._pair_selection_boundary_retention_observations,
                    pair_transaction_boundary_retention_before):
                rollback_failures.append("boundary_retention")
            if (
                    int(self._pair_selection_boundary_retention_next_id)
                    != int(pair_transaction_boundary_retention_next_id_before)):
                rollback_failures.append("boundary_retention_next_id")
            if (
                    int(self._pair_selection_boundary_ordinary_update_clock)
                    != int(pair_transaction_boundary_ordinary_clock_before)):
                rollback_failures.append("boundary_ordinary_clock")
            if not _transaction_state_equal(
                    self
                    ._pair_selection_boundary_retention_selection_state_snapshot(),
                    pair_transaction_boundary_selection_state_before):
                rollback_failures.append("boundary_selection_state")
            if rollback_failures:
                raise RuntimeError(
                    "pair transaction rollback failed to restore: {}".format(
                        ",".join(rollback_failures)
                    )
                )
            return True

        def _raise_recoverable_pair_optimizer_noop(reason, diagnostics):
            """Atomically reject one finite, non-committable pair proposal."""
            _rollback_failed_pair_transaction()
            _clear_parameter_gradients_to_none(self.adj_parameters)
            raise PairOptimizerRecoverableNoOpError(
                reason=reason,
                diagnostics=diagnostics,
                target_count=float(
                    pair_target_gradient_constraint_count
                    .detach().cpu().item()
                ),
            )

        active_adj_optimizer.step()
        if pair_diagnostic_target_present:
            pair_guard_raw_parameter_deltas = [
                (
                    (parameter.detach() - before).detach().clone()
                    if before is not None else None
                )
                for parameter, before in zip(
                    self.adj_parameters,
                    pair_parameter_before_step,
                )
            ]
            if not bool(pair_only_objective):
                pair_optimizer_state_after_raw_step = [
                    (
                        copy.deepcopy(
                            active_adj_optimizer.state.get(parameter, {})
                        )
                        if before is not None else None
                    )
                    for parameter, before in zip(
                        self.adj_parameters,
                        pair_parameter_before_step,
                    )
                ]
            if pair_transaction_all_parameters_before_step is None:
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "strict-pair exact search is missing the complete "
                    "transaction parameter origin"
                )
            # The nonlinear exact validator must replay the complete optimizer
            # transaction.  pair_parameter_before_step intentionally masks
            # parameters whose first-order pair gradient is None; using that
            # sparse snapshot as the ray origin leaves all other parameters at
            # their post-Adam values when scale=0.  The resulting hybrid model
            # is not the transaction origin and can falsely report
            # origin_preservation_valid=0.  Preserve the full raw displacement
            # and raw Adam state so any exact projection/backtrack can be
            # validated and synchronized as one atomic update.
            pair_exact_raw_parameter_deltas = [
                (parameter.detach() - before).detach().clone()
                for parameter, before in zip(
                    self.adj_parameters,
                    pair_transaction_all_parameters_before_step,
                )
            ]
            if not bool(pair_only_objective):
                pair_exact_optimizer_state_after_raw_step = [
                    (
                        copy.deepcopy(
                            active_adj_optimizer.state.get(parameter, {})
                        )
                        if parameter.grad is not None else None
                    )
                    for parameter in self.adj_parameters
                ]
            if (
                    not all(
                        bool(torch.isfinite(parameter).all().item())
                        for parameter in self.adj_parameters
                    )
                    or not _transaction_state_is_finite(
                        active_adj_optimizer.state_dict()
                    )):
                _rollback_failed_pair_transaction()
                raise FloatingPointError(
                    "non-finite parameter or optimizer state after pair Adam"
                )
        if transaction_diagnostics_enabled:
            adam_raw_info = _parameter_displacement_direction_diagnostics(
                parameters=self.adj_parameters,
                parameter_before_step=(
                    transaction_parameter_before_standard_adam
                ),
                pair_grads=pair_grads,
                reference=loss,
            )
            adam_raw_displacement_norm = adam_raw_info[
                "displacement_norm"
            ]
            adam_raw_pair_dot = adam_raw_info["pair_dot"]
            adam_raw_pair_descent_dot = adam_raw_info[
                "pair_descent_dot"
            ]
            adam_raw_pair_descent_cosine = adam_raw_info[
                "pair_descent_cosine"
            ]
            transaction_adam_raw_displacements = adam_raw_info[
                "displacements"
            ]
            adam_step_after_info = _validate_adam_step_increment(
                optimizer=active_adj_optimizer,
                parameters=self.adj_parameters,
                parameter_steps_before=transaction_parameter_steps_before,
            )
            optimizer_step_after = adam_step_after_info[
                "optimizer_step_after"
            ]
            optimizer_step_after_min = adam_step_after_info[
                "optimizer_step_after_min"
            ]
            optimizer_step_after_max = adam_step_after_info[
                "optimizer_step_after_max"
            ]
            if optimizer_step_after != optimizer_step_before + 1.0:
                raise RuntimeError(
                    "standard adjacency Adam transaction step must increase "
                    "by exactly one"
                )
        if lifecycle_target_present:
            # Preserve Adam's unmodified displacement before either the current
            # candidate guard or the lifecycle displacement projection changes
            # parameters.  Final state synchronization validates against this
            # single real optimizer step.
            lifecycle_raw_parameter_deltas = [
                (
                    (parameter.detach() - before).detach().clone()
                    if before is not None else None
                )
                for parameter, before in zip(
                    self.adj_parameters, lifecycle_parameter_before_step
                )
            ]
            lifecycle_optimizer_state_after_raw_step = [
                (
                    copy.deepcopy(active_adj_optimizer.state.get(parameter, {}))
                    if before is not None else None
                )
                for parameter, before in zip(
                    self.adj_parameters, lifecycle_parameter_before_step
                )
            ]
        if candidate_objective_active:
            # Adam's momentum and coordinate-wise preconditioner need not
            # preserve a positive Euclidean gradient dot.  Inspect the actual
            # parameter displacement.  If it would increase the candidate
            # objective to first order, replace only its conflicting component
            # along the exact pre-step candidate gradient.  The desired safe
            # component uses the optimizer's own realized update norm, so this
            # introduces no coefficient or learning-rate surrogate.  Guard v2
            # then solves Adam's update equation for an exp_avg consistent with
            # the displacement that was actually executed; exp_avg_sq remains
            # the second moment of the clipped backpropagated gradient.
            actual_update_dot = loss.new_tensor(0.0)
            actual_update_norm_sq = loss.new_tensor(0.0)
            candidate_raw_parameter_deltas = []
            for parameter, before, candidate_grad in zip(
                    self.adj_parameters,
                    candidate_parameter_before_step,
                    candidate_grads):
                if before is None:
                    candidate_raw_parameter_deltas.append(None)
                    continue
                parameter_delta = parameter.detach() - before
                candidate_raw_parameter_deltas.append(
                    parameter_delta.detach().clone()
                )
                actual_update_norm_sq = (
                    actual_update_norm_sq + parameter_delta.pow(2).sum()
                )
                actual_update_dot = (
                    actual_update_dot
                    + (parameter_delta * candidate_grad).sum()
                )
            candidate_actual_update_norm = torch.sqrt(actual_update_norm_sq)
            candidate_actual_update_descent_dot_before = -actual_update_dot
            if not bool(torch.isfinite(actual_update_dot).item()):
                if pair_transaction_atomic_state_required:
                    _rollback_failed_pair_transaction()
                raise FloatingPointError(
                    "non-finite Adam candidate identity update dot"
                )
            if bool((actual_update_dot >= 0.0).item()):
                clipped_combined_norm = torch.sqrt(
                    clipped_combined_norm_sq
                )
                if not bool(
                        (candidate_actual_update_norm > 0.0).item()
                        and (clipped_combined_norm > 0.0).item()):
                    if pair_diagnostic_target_present:
                        candidate_noop_reason = (
                            "zero_candidate_adam_displacement"
                            if bool((
                                candidate_actual_update_norm == 0.0
                            ).item())
                            else "zero_candidate_clipped_gradient"
                        )
                        _raise_recoverable_pair_optimizer_noop(
                            candidate_noop_reason,
                            {
                                "pair_update_dot": float(
                                    actual_update_dot.detach().cpu().item()
                                ),
                                "pair_update_norm_sq": float(
                                    actual_update_norm_sq
                                    .detach().cpu().item()
                                ),
                                "pair_gradient_norm_sq": float(
                                    candidate_norm_sq.detach().cpu().item()
                                ),
                                "clipped_pair_dot": float(
                                    candidate_clipped_gradient_dot
                                    .detach().cpu().item()
                                ),
                                "clipped_gradient_norm_sq": float(
                                    clipped_combined_norm_sq
                                    .detach().cpu().item()
                                ),
                            },
                        )
                    raise RuntimeError(
                        "Adam produced no usable update for candidate-safe "
                        "correction"
                    )
                desired_descent_dot = (
                    candidate_clipped_gradient_dot
                    * candidate_actual_update_norm
                    / clipped_combined_norm
                )
                correction_scale = (
                    actual_update_dot + desired_descent_dot
                ) / candidate_norm_sq
                candidate_actual_update_correction_norm = (
                    correction_scale.abs() * candidate_gradient_norm
                )
                candidate_actual_update_correction_norm_ratio = (
                    candidate_actual_update_correction_norm
                    / candidate_actual_update_norm.clamp_min(1e-12)
                )
                with torch.no_grad():
                    for parameter, before, candidate_grad in zip(
                            self.adj_parameters,
                            candidate_parameter_before_step,
                            candidate_grads):
                        if before is not None and candidate_grad is not None:
                            safe_delta = (
                                parameter.detach() - before
                                - correction_scale * candidate_grad
                            )
                            parameter.copy_(before + safe_delta)
                candidate_actual_update_corrected = loss.new_tensor(1.0)
                state_sync_info = _sync_adam_first_moment_to_executed_update(
                    optimizer=active_adj_optimizer,
                    parameters=self.adj_parameters,
                    parameter_before_step=candidate_parameter_before_step,
                    raw_parameter_deltas=candidate_raw_parameter_deltas,
                )
                candidate_optimizer_state_sync_applied = loss.new_tensor(1.0)
                candidate_optimizer_state_sync_parameter_count = loss.new_tensor(
                    state_sync_info["parameter_count"]
                )
                candidate_optimizer_state_update_equation_version = (
                    loss.new_tensor(
                        state_sync_info["update_equation_version"]
                    )
                )
                candidate_optimizer_state_raw_reconstruction_error = (
                    loss.new_tensor(
                        state_sync_info["raw_reconstruction_error"]
                    )
                )
                candidate_optimizer_state_safe_reconstruction_error = (
                    loss.new_tensor(
                        state_sync_info["safe_reconstruction_error"]
                    )
                )
                candidate_optimizer_state_raw_reconstruction_error_ratio = (
                    loss.new_tensor(
                        state_sync_info["raw_reconstruction_error_ratio"]
                    )
                )
                candidate_optimizer_state_safe_reconstruction_error_ratio = (
                    loss.new_tensor(
                        state_sync_info["safe_reconstruction_error_ratio"]
                    )
                )
                candidate_optimizer_state_reconstruction_tolerance = (
                    loss.new_tensor(
                        state_sync_info["reconstruction_tolerance"]
                    )
                )
                candidate_optimizer_state_exp_avg_change_norm = loss.new_tensor(
                    state_sync_info["exp_avg_change_norm"]
                )
                if (
                        state_sync_info["raw_reconstruction_error_ratio"]
                        > 1.0):
                    raise RuntimeError(
                        "Adam raw update reconstruction is inconsistent with "
                        "the optimizer state: error={:.9g}, tolerance={:.9g}, "
                        "error_ratio={:.9g}, equation_version={:.0f}".format(
                            state_sync_info["raw_reconstruction_error"],
                            state_sync_info["reconstruction_tolerance"],
                            state_sync_info[
                                "raw_reconstruction_error_ratio"
                            ],
                            state_sync_info["update_equation_version"],
                        )
                    )
                if (
                        state_sync_info["safe_reconstruction_error_ratio"]
                        > 1.0):
                    raise RuntimeError(
                        "candidate-safe Adam state does not reconstruct the "
                        "executed parameter update: error={:.9g}, "
                        "tolerance={:.9g}, error_ratio={:.9g}".format(
                            state_sync_info["safe_reconstruction_error"],
                            state_sync_info["reconstruction_tolerance"],
                            state_sync_info[
                                "safe_reconstruction_error_ratio"
                            ],
                        )
                    )
            final_actual_update_dot = loss.new_tensor(0.0)
            for parameter, before, candidate_grad in zip(
                    self.adj_parameters,
                    candidate_parameter_before_step,
                    candidate_grads):
                if before is not None:
                    final_actual_update_dot = (
                        final_actual_update_dot
                        + (
                            (parameter.detach() - before) * candidate_grad
                        ).sum()
                    )
            candidate_actual_update_descent_dot_after = (
                -final_actual_update_dot
            )
            if not bool(torch.isfinite(
                    candidate_actual_update_descent_dot_after).item()):
                if pair_transaction_atomic_state_required:
                    _rollback_failed_pair_transaction()
                raise FloatingPointError(
                    "non-finite final candidate identity update direction"
                )
            if bool((
                    candidate_actual_update_descent_dot_after == 0.0
            ).item()) and pair_diagnostic_target_present:
                _raise_recoverable_pair_optimizer_noop(
                    "zero_candidate_final_descent",
                    {
                        "pair_update_dot": float(
                            final_actual_update_dot.detach().cpu().item()
                        ),
                        "pair_update_norm_sq": float(
                            actual_update_norm_sq.detach().cpu().item()
                        ),
                        "pair_gradient_norm_sq": float(
                            candidate_norm_sq.detach().cpu().item()
                        ),
                        "clipped_pair_dot": float(
                            candidate_clipped_gradient_dot
                            .detach().cpu().item()
                        ),
                        "clipped_gradient_norm_sq": float(
                            clipped_combined_norm_sq
                            .detach().cpu().item()
                        ),
                    },
                )
            if bool((
                    candidate_actual_update_descent_dot_after == 0.0
            ).item()):
                raise RuntimeError(
                    "actual adjacency parameter update has zero candidate "
                    "identity descent"
                )
            if bool((
                    candidate_actual_update_descent_dot_after < 0.0
            ).item()):
                if pair_transaction_atomic_state_required:
                    _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "actual adjacency parameter update does not preserve "
                    "candidate identity descent direction"
                )
            # Behavior-to-current diagnostics include every graph update that
            # occurred while a replay episode aged.  Re-evaluate immediately
            # after this optimizer step so run logs can verify the causal
            # delta produced by the candidate objective itself.
            with torch.no_grad():
                post_candidate_margins, post_candidate_valid_mask = (
                    self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                        rnn_obs=rnn_obs,
                        dones=dones.bool(),
                        adj=adj,
                    )
                )
                if not bool(torch.equal(
                        post_candidate_valid_mask > 0.0,
                        candidate_identity_valid_mask > 0.0,
                )):
                    raise RuntimeError(
                        "candidate valid catalog changed within one adjacency "
                        "optimizer step"
                    )
                post_candidate_info = (
                    compute_capture_candidate_identity_active_competitor_loss(
                    candidate_competitor_margins=post_candidate_margins,
                    candidate_reference_margins=behavior_candidate_margin,
                    candidate_identity_delta=candidate_identity_delta[..., 0],
                    candidate_valid_mask=post_candidate_valid_mask,
                    transition_mask=transition_mask,
                ))
                candidate_loss_optimizer_change = (
                    post_candidate_info["loss"]
                    - candidate_identity_loss_info["loss"].detach()
                )
                post_margin = post_candidate_info["margin"]
                post_rank = self.adj_network.canonical_candidate_ranks(
                    post_candidate_margins,
                    post_candidate_valid_mask,
                ).to(post_candidate_margins.dtype)
                optimizer_signed_margin_change = (
                    torch.sign(candidate_identity_delta[..., 0])
                    * (post_margin - current_margin.detach())
                )
                candidate_positive_optimizer_signed_margin_change = (
                    (
                        optimizer_signed_margin_change
                        * candidate_optimized_positive_target
                    ).sum()
                    / candidate_optimized_positive_denominator
                )
                candidate_negative_optimizer_signed_margin_change = (
                    (
                        optimizer_signed_margin_change
                        * candidate_optimized_negative_target
                    ).sum()
                    / candidate_optimized_negative_denominator
                )
                candidate_positive_optimizer_rank_improved_fraction = (
                    (
                        (post_rank < candidate_identity_current_rank).float()
                        * candidate_optimized_positive_target
                    ).sum()
                    / candidate_optimized_positive_denominator
                )
                candidate_negative_optimizer_rank_reduced_fraction = (
                    (
                        (post_rank > candidate_identity_current_rank).float()
                        * candidate_optimized_negative_target
                    ).sum()
                    / candidate_optimized_negative_denominator
                )
        if lifecycle_target_present:
            # Euclidean gradient safety does not imply safety of Adam's
            # momentum/preconditioned parameter displacement.  Project the
            # *realized descent displacement* through the same transition-local
            # halfspaces before evaluating nonlinear losses.  This prevents the
            # run74 failure mode where the gradient dots were non-negative but
            # one third of protected updates were discarded afterwards.
            actual_descent_grads = [
                (
                    -(parameter.detach() - before)
                    if before is not None else None
                )
                for parameter, before in zip(
                    self.adj_parameters, lifecycle_parameter_before_step
                )
            ]
            actual_descent_norm_sq = _gradient_tuple_dot(
                actual_descent_grads, actual_descent_grads, loss
            )
            lifecycle_actual_update_descent_dot_before = _gradient_tuple_dot(
                actual_descent_grads, lifecycle_grads, loss
            )
            if lifecycle_constraint_grads:
                actual_dots_before = torch.stack([
                    _gradient_tuple_dot(
                        actual_descent_grads, constraint, loss
                    )
                    for constraint in lifecycle_constraint_grads
                ])
                actual_tolerances_before = torch.stack([
                    _gradient_dot_tolerance(
                        proposed_grads=actual_descent_grads,
                        constraint_grads=constraint,
                        reference=loss,
                        tolerance_multiplier=64.0,
                    )
                    for constraint in lifecycle_constraint_grads
                ])
                lifecycle_actual_min_constraint_dot_before = (
                    actual_dots_before.min()
                )
                lifecycle_actual_negative_constraint_count_before = (
                    (
                        actual_dots_before < -actual_tolerances_before
                    ).float().sum()
                )
            actual_current_priority_grads = None
            actual_additional_priority_grads = ()
            if candidate_objective_active:
                actual_current_priority_grads = candidate_grads
                if pair_diagnostic_target_present:
                    actual_additional_priority_grads = (
                        (pair_grads,)
                        + tuple(pair_target_score_grads)
                        + tuple(pair_boundary_target_grads)
                    )
            elif pair_diagnostic_target_present:
                actual_current_priority_grads = pair_grads
                actual_additional_priority_grads = (
                    tuple(pair_target_score_grads)
                    + tuple(pair_boundary_target_grads)
                )
            if actual_current_priority_grads is not None:
                (
                    safe_actual_descent_grads,
                    actual_enforced_indices,
                    actual_superseded_indices,
                    actual_projection_info,
                ) = _project_with_current_candidate_priority(
                    proposed_grads=actual_descent_grads,
                    candidate_grads=actual_current_priority_grads,
                    lifecycle_constraint_grads=lifecycle_constraint_grads,
                    reference=loss,
                    additional_priority_grads=(
                        actual_additional_priority_grads
                    ),
                )
                lifecycle_current_priority_repair_intervened = loss.new_tensor(
                    actual_projection_info[
                        "current_priority_repair_intervened"
                    ]
                )
                lifecycle_current_priority_min_dot_before = loss.new_tensor(
                    actual_projection_info["current_priority_min_dot_before"]
                )
                lifecycle_current_priority_min_dot_after = loss.new_tensor(
                    actual_projection_info[
                        "current_priority_min_dot_after_repair"
                    ]
                )
                if actual_superseded_indices:
                    lifecycle_superseded_constraint_count = (
                        lifecycle_superseded_constraint_count
                        + loss.new_tensor(float(len(actual_superseded_indices)))
                    )
                    for index in actual_superseded_indices:
                        del self._candidate_identity_lifecycle[
                            lifecycle_constraint_entry_keys[index]
                        ]
                    enforced_row_mask = torch.zeros(
                        lifecycle_delta.shape[0],
                        device=lifecycle_delta.device,
                        dtype=lifecycle_delta.dtype,
                    )
                    for index in actual_enforced_indices:
                        enforced_row_mask[
                            lifecycle_constraint_row_indices[index]
                        ] = 1.0
                    lifecycle_delta = (
                        lifecycle_delta * enforced_row_mask.unsqueeze(-1)
                    )
                    # Superseded cache entries are no longer lifecycle targets.
                    # Keeping the pre-supersession mask makes exact replay test
                    # sign(0) * margin against the deleted entry's old positive
                    # floor.  The resulting constant negative gap cannot be
                    # repaired by any scale and falsely invalidates scale zero.
                    (
                        lifecycle_target_mask,
                        lifecycle_target_present,
                    ) = _lifecycle_target_population_from_delta(
                        lifecycle_delta
                    )
                    lifecycle_constraint_grads = [
                        lifecycle_constraint_grads[index]
                        for index in actual_enforced_indices
                    ]
                    lifecycle_constraint_row_indices = [
                        lifecycle_constraint_row_indices[index]
                        for index in actual_enforced_indices
                    ]
                    lifecycle_constraint_entry_keys = [
                        lifecycle_constraint_entry_keys[index]
                        for index in actual_enforced_indices
                    ]
                    lifecycle_loss_info = (
                        compute_capture_candidate_identity_active_competitor_loss(
                            candidate_competitor_margins=lifecycle_margins,
                            candidate_reference_margins=lifecycle_reference_margin,
                            candidate_identity_delta=lifecycle_delta,
                            candidate_valid_mask=lifecycle_valid_mask,
                            transition_mask=lifecycle_transition_mask,
                        )
                    )
                    lifecycle_per_transition_loss = lifecycle_loss_info[
                        "per_transition_loss"
                    ]
                    lifecycle_grads = _mean_gradient_tuples(
                        lifecycle_constraint_grads,
                        len(self.adj_parameters),
                    )
                    lifecycle_protected_target_count = lifecycle_loss_info[
                        "target_count"
                    ]
                    lifecycle_cache_size = len(
                        self._candidate_identity_lifecycle
                    )
            else:
                safe_actual_descent_grads, actual_projection_info = (
                    _project_gradients_onto_nonincreasing_halfspaces(
                        proposed_grads=actual_descent_grads,
                        constraint_grads=lifecycle_constraint_grads,
                        reference=loss,
                    )
                )
            actual_projection_correction_norm_sq = loss.new_tensor(0.0)
            with torch.no_grad():
                for parameter, before, original_descent, safe_descent in zip(
                        self.adj_parameters,
                        lifecycle_parameter_before_step,
                        actual_descent_grads,
                        safe_actual_descent_grads):
                    if before is None or safe_descent is None:
                        continue
                    if original_descent is not None:
                        actual_projection_correction_norm_sq = (
                            actual_projection_correction_norm_sq
                            + (safe_descent - original_descent).pow(2).sum()
                        )
                    parameter.copy_(before - safe_descent)
            if bool((actual_projection_correction_norm_sq > 0.0).item()):
                lifecycle_actual_projection_corrected = loss.new_tensor(1.0)
            lifecycle_actual_projection_correction_norm_ratio = (
                torch.sqrt(actual_projection_correction_norm_sq)
                / torch.sqrt(actual_descent_norm_sq).clamp_min(1.0e-12)
            )
            if lifecycle_constraint_grads:
                actual_dots_after = torch.stack([
                    _gradient_tuple_dot(
                        safe_actual_descent_grads, constraint, loss
                    )
                    for constraint in lifecycle_constraint_grads
                ])
                actual_tolerances_after = torch.stack([
                    _gradient_dot_tolerance(
                        proposed_grads=safe_actual_descent_grads,
                        constraint_grads=constraint,
                        reference=loss,
                        tolerance_multiplier=64.0,
                    )
                    for constraint in lifecycle_constraint_grads
                ])
                lifecycle_actual_min_constraint_dot_after = (
                    actual_dots_after.min()
                )
                lifecycle_actual_negative_constraint_count_after = (
                    (
                        actual_dots_after < -actual_tolerances_after
                    ).float().sum()
                )
            lifecycle_actual_dot = loss.new_tensor(0.0)
            for parameter, before, lifecycle_grad in zip(
                    self.adj_parameters,
                    lifecycle_parameter_before_step,
                    lifecycle_grads):
                if before is None or lifecycle_grad is None:
                    continue
                parameter_delta = parameter.detach() - before
                lifecycle_actual_dot = (
                    lifecycle_actual_dot
                    + (parameter_delta * lifecycle_grad).sum()
                )
            if not bool(torch.isfinite(lifecycle_actual_dot).item()):
                raise FloatingPointError(
                    "non-finite Adam candidate lifecycle update dot"
                )
            with torch.no_grad():
                lifecycle_attempted_margins, lifecycle_attempted_valid = (
                    self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                        rnn_obs=lifecycle_rnn_obs,
                        dones=lifecycle_dones.bool(),
                        adj=lifecycle_adj,
                    )
                )
                if not bool(torch.equal(
                        lifecycle_attempted_valid > 0.0,
                        lifecycle_valid_mask > 0.0,
                )):
                    raise RuntimeError(
                        "candidate lifecycle catalog changed within one "
                        "optimizer step"
                    )
                lifecycle_attempted_info = (
                    compute_capture_candidate_identity_active_competitor_loss(
                        candidate_competitor_margins=lifecycle_attempted_margins,
                        candidate_reference_margins=lifecycle_reference_margin,
                        candidate_identity_delta=lifecycle_delta,
                        candidate_valid_mask=lifecycle_attempted_valid,
                        transition_mask=lifecycle_transition_mask,
                    )
                )
                lifecycle_attempted_loss_optimizer_change = (
                    lifecycle_attempted_info["loss"]
                    - lifecycle_loss_info["loss"].detach()
                )
                lifecycle_pre_per_transition = lifecycle_loss_info[
                    "per_transition_loss"
                ].detach()
                lifecycle_post_per_transition = lifecycle_attempted_info[
                    "per_transition_loss"
                ]
                lifecycle_target_rows = (
                    lifecycle_delta.abs().sum(dim=1) > 0.0
                )
                lifecycle_dtype_epsilon = (
                    2.220446049250313e-16
                    if lifecycle_pre_per_transition.dtype == torch.float64
                    else 1.1920928955078125e-7
                )
                lifecycle_exact_tolerance = (
                    16.0
                    * lifecycle_dtype_epsilon
                    * (
                        lifecycle_signed_floor.abs()
                        + (
                            torch.sign(lifecycle_delta)
                            * lifecycle_attempted_margins
                        ).abs()
                    ).clamp_min(1.0e-12)
                )
                lifecycle_target_violation = (
                    lifecycle_target_mask
                    & (
                        torch.sign(lifecycle_delta)
                        * lifecycle_attempted_margins
                        < lifecycle_signed_floor - lifecycle_exact_tolerance
                    )
                )
                lifecycle_violation_mask = (
                    lifecycle_target_violation.any(dim=1)
                    & lifecycle_target_rows
                )
                lifecycle_violation_count = (
                    lifecycle_violation_mask.float().sum()
                )
                current_candidate_safe = True
                if candidate_objective_active:
                    lifecycle_candidate_margins, lifecycle_candidate_valid = (
                        self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                            rnn_obs=rnn_obs,
                            dones=dones.bool(),
                            adj=adj,
                        )
                    )
                    if not bool(torch.equal(
                            lifecycle_candidate_valid > 0.0,
                            candidate_identity_valid_mask > 0.0,
                    )):
                        raise RuntimeError(
                            "current candidate catalog changed after lifecycle "
                            "displacement projection"
                        )
                    lifecycle_candidate_info = (
                        compute_capture_candidate_identity_active_competitor_loss(
                            candidate_competitor_margins=lifecycle_candidate_margins,
                            candidate_reference_margins=behavior_candidate_margin,
                            candidate_identity_delta=(
                                candidate_identity_delta[..., 0]
                            ),
                            candidate_valid_mask=lifecycle_candidate_valid,
                            transition_mask=transition_mask,
                        )
                    )
                    current_candidate_safe = bool((
                        lifecycle_candidate_info["loss"]
                        < candidate_identity_loss_info["loss"].detach()
                    ).item())
                    lifecycle_current_candidate_nonlinear_violation = (
                        loss.new_tensor(float(not current_candidate_safe))
                    )
            lifecycle_transaction_violation = (
                bool(torch.any(lifecycle_violation_mask).item())
                or not current_candidate_safe
            )
            if lifecycle_transaction_violation:
                # The transition-local gradient and realized-displacement
                # halfspaces are first-order constraints.  Curvature can still
                # make a finite Adam step increase an old exact objective.
                # Backtrack only along the already projected safe displacement;
                # this adds no replay mass and keeps every unrelated/base-only
                # parameter direction intact.  A failed finite search still
                # falls through to the exact transaction rollback below.
                projected_parameter_deltas = [
                    (
                        (parameter.detach() - before).detach().clone()
                        if before is not None else None
                    )
                    for parameter, before in zip(
                        self.adj_parameters,
                        lifecycle_parameter_before_step,
                    )
                ]
                for backtrack_index in range(1, 25):
                    backtrack_scale = 0.5 ** backtrack_index
                    with torch.no_grad():
                        for parameter, before, projected_delta in zip(
                                self.adj_parameters,
                                lifecycle_parameter_before_step,
                                projected_parameter_deltas):
                            if before is not None and projected_delta is not None:
                                parameter.copy_(
                                    before + backtrack_scale * projected_delta
                                )
                        trial_margins, trial_valid = (
                            self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                                rnn_obs=lifecycle_rnn_obs,
                                dones=lifecycle_dones.bool(),
                                adj=lifecycle_adj,
                            )
                        )
                        if not bool(torch.equal(
                                trial_valid > 0.0,
                                lifecycle_valid_mask > 0.0,
                        )):
                            raise RuntimeError(
                                "candidate lifecycle catalog changed during "
                                "nonlinear backtracking"
                            )
                        trial_info = compute_capture_candidate_identity_active_competitor_loss(
                            candidate_competitor_margins=trial_margins,
                            candidate_reference_margins=lifecycle_reference_margin,
                            candidate_identity_delta=lifecycle_delta,
                            candidate_valid_mask=trial_valid,
                            transition_mask=lifecycle_transition_mask,
                        )
                        trial_per_transition = trial_info[
                            "per_transition_loss"
                        ]
                        trial_tolerance = (
                            16.0
                            * lifecycle_dtype_epsilon
                            * (
                                lifecycle_signed_floor.abs()
                                + (
                                    torch.sign(lifecycle_delta)
                                    * trial_margins
                                ).abs()
                            ).clamp_min(1.0e-12)
                        )
                        trial_target_violation = (
                            lifecycle_target_mask
                            & (
                                torch.sign(lifecycle_delta) * trial_margins
                                < lifecycle_signed_floor - trial_tolerance
                            )
                        )
                        trial_violation_mask = (
                            trial_target_violation.any(dim=1)
                            & lifecycle_target_rows
                        )
                        current_candidate_safe = True
                        if candidate_objective_active:
                            trial_candidate_margins, trial_candidate_valid = (
                                self.adj_network
                                .evaluate_candidate_identity_active_competitor_margins(
                                    rnn_obs=rnn_obs,
                                    dones=dones.bool(),
                                    adj=adj,
                                )
                            )
                            if not bool(torch.equal(
                                    trial_candidate_valid > 0.0,
                                    candidate_identity_valid_mask > 0.0,
                            )):
                                raise RuntimeError(
                                    "current candidate catalog changed during "
                                    "lifecycle backtracking"
                                )
                            trial_candidate_info = (
                                compute_capture_candidate_identity_active_competitor_loss(
                                    candidate_competitor_margins=trial_candidate_margins,
                                    candidate_reference_margins=behavior_candidate_margin,
                                    candidate_identity_delta=(
                                        candidate_identity_delta[..., 0]
                                    ),
                                    candidate_valid_mask=trial_candidate_valid,
                                    transition_mask=transition_mask,
                                )
                            )
                            current_candidate_safe = bool((
                                trial_candidate_info["loss"]
                                < candidate_identity_loss_info["loss"].detach()
                            ).item())
                    if (
                            not bool(torch.any(trial_violation_mask).item())
                            and current_candidate_safe):
                        lifecycle_nonlinear_backtrack_count = loss.new_tensor(
                            float(backtrack_index)
                        )
                        lifecycle_attempted_info = trial_info
                        lifecycle_post_per_transition = trial_per_transition
                        lifecycle_exact_tolerance = trial_tolerance
                        lifecycle_violation_mask = trial_violation_mask
                        lifecycle_violation_count = loss.new_tensor(0.0)
                        lifecycle_attempted_loss_optimizer_change = (
                            lifecycle_attempted_info["loss"]
                            - lifecycle_loss_info["loss"].detach()
                        )
                        lifecycle_actual_update_corrected = loss.new_tensor(1.0)
                        lifecycle_current_candidate_nonlinear_violation = (
                            loss.new_tensor(0.0)
                        )
                        lifecycle_transaction_violation = False
                        break

            lifecycle_final_delta_diff_norm_sq = loss.new_tensor(0.0)
            for parameter, before, raw_delta in zip(
                    self.adj_parameters,
                    lifecycle_parameter_before_step,
                    lifecycle_raw_parameter_deltas):
                if before is not None and raw_delta is not None:
                    lifecycle_final_delta_diff_norm_sq = (
                        lifecycle_final_delta_diff_norm_sq
                        + (
                            parameter.detach() - before - raw_delta
                        ).pow(2).sum()
                    )
            if (
                    not lifecycle_transaction_violation
                    and bool((lifecycle_final_delta_diff_norm_sq > 0.0).item())):
                # Candidate and lifecycle guards may both have edited the raw
                # Adam displacement.  Restore Adam's untouched post-step state,
                # then synchronize it once to the single final displacement.
                for parameter, state_after_raw in zip(
                        self.adj_parameters,
                        lifecycle_optimizer_state_after_raw_step):
                    if state_after_raw is None:
                        continue
                    state = active_adj_optimizer.state[parameter]
                    state.clear()
                    state.update(copy.deepcopy(state_after_raw))
                lifecycle_sync_info = (
                    _sync_adam_first_moment_to_executed_update(
                        optimizer=active_adj_optimizer,
                        parameters=self.adj_parameters,
                        parameter_before_step=lifecycle_parameter_before_step,
                        raw_parameter_deltas=lifecycle_raw_parameter_deltas,
                    )
                )
                if (
                        lifecycle_sync_info["raw_reconstruction_error_ratio"] > 1.0
                        or lifecycle_sync_info[
                            "safe_reconstruction_error_ratio"
                        ] > 1.0):
                    raise RuntimeError(
                        "lifecycle-safe Adam state does not reconstruct the "
                        "raw and final executed updates"
                    )
                lifecycle_state_sync_applied = loss.new_tensor(1.0)
                if candidate_objective_active:
                    candidate_optimizer_state_sync_applied = loss.new_tensor(1.0)
                    candidate_optimizer_state_sync_parameter_count = (
                        loss.new_tensor(lifecycle_sync_info["parameter_count"])
                    )
                    candidate_optimizer_state_update_equation_version = (
                        loss.new_tensor(
                            lifecycle_sync_info["update_equation_version"]
                        )
                    )
                    candidate_optimizer_state_raw_reconstruction_error = (
                        loss.new_tensor(
                            lifecycle_sync_info["raw_reconstruction_error"]
                        )
                    )
                    candidate_optimizer_state_safe_reconstruction_error = (
                        loss.new_tensor(
                            lifecycle_sync_info["safe_reconstruction_error"]
                        )
                    )
                    candidate_optimizer_state_raw_reconstruction_error_ratio = (
                        loss.new_tensor(
                            lifecycle_sync_info[
                                "raw_reconstruction_error_ratio"
                            ]
                        )
                    )
                    candidate_optimizer_state_safe_reconstruction_error_ratio = (
                        loss.new_tensor(
                            lifecycle_sync_info[
                                "safe_reconstruction_error_ratio"
                            ]
                        )
                    )
                    candidate_optimizer_state_reconstruction_tolerance = (
                        loss.new_tensor(
                            lifecycle_sync_info["reconstruction_tolerance"]
                        )
                    )
                    candidate_optimizer_state_exp_avg_change_norm = (
                        loss.new_tensor(
                            lifecycle_sync_info["exp_avg_change_norm"]
                        )
                    )
            if lifecycle_transaction_violation:
                # This is a no-forget constraint, not replayed supervision.
                # Reject the Adam update on candidate-sensitive parameter
                # tensors and restore their complete pre-step Adam state.
                # A post-step projection followed by exp_avg-only repair would
                # make exp_avg and exp_avg_sq describe different gradient
                # histories.  Transactional rollback gives an exact zero
                # lifecycle displacement while base-only parameter tensors
                # retain their normal update.
                with torch.no_grad():
                    for parameter, before, state_before in zip(
                            self.adj_parameters,
                            lifecycle_parameter_before_step,
                            lifecycle_optimizer_state_before_step):
                        if before is None:
                            continue
                        parameter.copy_(before)
                        state = active_adj_optimizer.state[parameter]
                        state.clear()
                        state.update(copy.deepcopy(state_before))
                lifecycle_actual_update_corrected = loss.new_tensor(1.0)
                lifecycle_update_rejected = loss.new_tensor(1.0)
                if candidate_objective_active:
                    # The candidate-sensitive part of the target-bearing step
                    # was part of the same rejected transaction.  Do not claim
                    # that its post-Adam state synchronization survived.
                    candidate_optimizer_state_sync_applied = loss.new_tensor(0.0)
                    candidate_optimizer_state_sync_parameter_count = (
                        loss.new_tensor(0.0)
                    )
                with torch.no_grad():
                    lifecycle_restored_margins, lifecycle_restored_valid = (
                        self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                            rnn_obs=lifecycle_rnn_obs,
                            dones=lifecycle_dones.bool(),
                            adj=lifecycle_adj,
                        )
                    )
                    if not bool(torch.equal(
                            lifecycle_restored_valid > 0.0,
                            lifecycle_valid_mask > 0.0,
                    )):
                        raise RuntimeError(
                            "candidate lifecycle catalog was not restored by "
                            "transaction rollback"
                        )
                    lifecycle_restored_info = (
                        compute_capture_candidate_identity_active_competitor_loss(
                            candidate_competitor_margins=lifecycle_restored_margins,
                            candidate_reference_margins=lifecycle_reference_margin,
                            candidate_identity_delta=lifecycle_delta,
                            candidate_valid_mask=lifecycle_restored_valid,
                            transition_mask=lifecycle_transition_mask,
                        )
                    )
                    lifecycle_restored_change = (
                        lifecycle_restored_info["loss"]
                        - lifecycle_loss_info["loss"].detach()
                    )
                    lifecycle_restore_tolerance = (
                        64.0
                        * (
                            2.220446049250313e-16
                            if lifecycle_restored_change.dtype == torch.float64
                            else 1.1920928955078125e-7
                        )
                        * (
                            lifecycle_restored_info["loss"].abs()
                            + lifecycle_loss_info["loss"].detach().abs()
                        ).clamp_min(1.0e-12)
                    )
                    if bool(
                            (
                                lifecycle_restored_change.abs()
                                > lifecycle_restore_tolerance
                            ).item()):
                        raise RuntimeError(
                            "candidate lifecycle transaction rollback did not "
                            "restore the pre-step objective"
                        )
            lifecycle_final_dot = loss.new_tensor(0.0)
            for parameter, before, lifecycle_grad in zip(
                    self.adj_parameters,
                    lifecycle_parameter_before_step,
                    lifecycle_grads):
                if before is not None and lifecycle_grad is not None:
                    lifecycle_final_dot = (
                        lifecycle_final_dot
                        + ((parameter.detach() - before) * lifecycle_grad).sum()
                    )
            lifecycle_actual_update_descent_dot_after = -lifecycle_final_dot
            lifecycle_loss_optimizer_change = (
                loss.new_tensor(0.0)
                if bool(lifecycle_update_rejected.item())
                else lifecycle_attempted_loss_optimizer_change
            )

        if pair_diagnostic_target_present:
            # Both isolated pending transactions and ordinary class-complete
            # transactions must commit a strict first-order pair descent.
            # Standard Adam can reverse a protected gradient through stale
            # moments (run113 epoch 1), so correct only that conflicting
            # component and synchronize the active Adam first moment to the
            # displacement that was actually executed.
            pair_update_dot = loss.new_tensor(0.0)
            pair_update_norm_sq = loss.new_tensor(0.0)
            pair_gradient_norm_sq = loss.new_tensor(0.0)
            clipped_pair_dot = loss.new_tensor(0.0)
            clipped_gradient_norm_sq = loss.new_tensor(0.0)
            for parameter, before, pair_grad in zip(
                    self.adj_parameters,
                    pair_parameter_before_step,
                    pair_grads):
                if before is None or pair_grad is None:
                    continue
                parameter_delta = parameter.detach() - before
                pair_update_dot = (
                    pair_update_dot + (parameter_delta * pair_grad).sum()
                )
                pair_update_norm_sq = (
                    pair_update_norm_sq + parameter_delta.pow(2).sum()
                )
                pair_gradient_norm_sq = (
                    pair_gradient_norm_sq + pair_grad.pow(2).sum()
                )
                if parameter.grad is not None:
                    clipped_pair_dot = (
                        clipped_pair_dot
                        + (parameter.grad.detach() * pair_grad).sum()
                    )
                    clipped_gradient_norm_sq = (
                        clipped_gradient_norm_sq
                        + parameter.grad.detach().pow(2).sum()
                    )
            pair_adam_guard_scalars = {
                "pair_update_dot": pair_update_dot,
                "pair_update_norm_sq": pair_update_norm_sq,
                "pair_gradient_norm_sq": pair_gradient_norm_sq,
                "clipped_pair_dot": clipped_pair_dot,
                "clipped_gradient_norm_sq": clipped_gradient_norm_sq,
            }
            nonfinite_pair_adam_fields = [
                field_name
                for field_name, field_value in pair_adam_guard_scalars.items()
                if not bool(torch.isfinite(field_value).item())
            ]
            if nonfinite_pair_adam_fields:
                _rollback_failed_pair_transaction()
                raise FloatingPointError(
                    "non-finite pair Adam update state: {}".format(
                        ",".join(nonfinite_pair_adam_fields)
                    )
                )
            # The pair Jacobian was already checked/recovered above and every
            # target-local projection was required to preserve it.  A negative
            # norm or pair dot here is therefore an implementation invariant
            # violation, not an optimizer rejection.
            if bool((pair_gradient_norm_sq <= 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "pair Adam guard lost the validated pair gradient"
                )
            if bool((pair_update_norm_sq < 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError("pair Adam update norm is negative")
            if bool((clipped_gradient_norm_sq < 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError("clipped adjacency gradient norm is negative")
            if bool((clipped_pair_dot < 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "clipped adjacency gradient reversed the validated pair "
                    "direction"
                )

            # A finite zero displacement is not data corruption.  Adam can
            # legitimately quantize a tiny update to the current float32
            # parameter value; an equally finite clipped direction can also
            # underflow to zero.  No positive strict-pair transaction exists
            # in that case.  Restore and verify the complete atomic snapshot,
            # leave evidence unconsumed, and let the runner continue with the
            # next replay partition.  Nonzero directions still pass through
            # the exact nonlinear candidate/backtracking contract below.
            recoverable_noop_reason = None
            if bool((pair_update_norm_sq == 0.0).item()):
                recoverable_noop_reason = "zero_adam_displacement"
            elif bool((clipped_gradient_norm_sq == 0.0).item()):
                recoverable_noop_reason = "zero_clipped_gradient"
            elif bool((clipped_pair_dot == 0.0).item()):
                recoverable_noop_reason = "zero_clipped_pair_dot"
            if recoverable_noop_reason is not None:
                recoverable_diagnostics = {
                    field_name: float(
                        field_value.detach().cpu().item()
                    )
                    for field_name, field_value
                    in pair_adam_guard_scalars.items()
                }
                _raise_recoverable_pair_optimizer_noop(
                    recoverable_noop_reason,
                    recoverable_diagnostics,
                )
            if bool((pair_update_dot >= 0.0).item()):
                desired_descent_dot = (
                    clipped_pair_dot
                    * torch.sqrt(pair_update_norm_sq)
                    / torch.sqrt(clipped_gradient_norm_sq)
                )
                correction_scale = (
                    pair_update_dot + desired_descent_dot
                ) / pair_gradient_norm_sq
                with torch.no_grad():
                    for parameter, before, pair_grad in zip(
                            self.adj_parameters,
                            pair_parameter_before_step,
                            pair_grads):
                        if before is None or pair_grad is None:
                            continue
                        corrected_delta = (
                            parameter.detach() - before
                            - correction_scale * pair_grad
                        )
                        parameter.copy_(before + corrected_delta)
                pair_actual_update_direction_guard_applied = loss.new_tensor(
                    1.0
                )
            if pair_transaction_all_parameters_before_step is None:
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "strict-pair exact search is missing the complete "
                    "transaction parameter origin"
                )
            pair_exact_search_parameter_before_step = (
                pair_parameter_before_step
                if bool(pair_only_objective)
                else pair_transaction_all_parameters_before_step
            )
            current_effective_descent = tuple(
                (
                    None if before is None
                    else before - parameter.detach()
                )
                for parameter, before in zip(
                    self.adj_parameters,
                    pair_exact_search_parameter_before_step,
                )
            )
            all_pair_target_grads = (
                tuple(pair_target_score_grads)
                + tuple(pair_boundary_target_grads)
            )
            clipped_all_target_dots = torch.stack([
                _gradient_tuple_dot(
                    tuple(
                        parameter.grad
                        for parameter in self.adj_parameters
                    ),
                    target_grads,
                    loss,
                )
                for target_grads in all_pair_target_grads
            ])
            if not bool(torch.all(
                    clipped_all_target_dots > 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "clipped {} gradient lost an exact pair score or "
                    "selection-boundary direction".format(
                        "pending"
                        if bool(pair_only_objective)
                        else "standard"
                    )
                )
            current_descent_norm_sq = _gradient_tuple_dot(
                current_effective_descent,
                current_effective_descent,
                loss,
            )
            target_update_scale = (
                torch.sqrt(current_descent_norm_sq)
                / torch.sqrt(clipped_gradient_norm_sq)
            )
            target_actual_floors = [
                float(
                    (dot * target_update_scale).detach().cpu().item()
                )
                for dot in clipped_all_target_dots
            ]
            boundary_start = len(pair_target_score_grads)
            boundary_floor_info = (
                _pair_boundary_deficit_aware_minimum_dots(
                    boundary_target_grads=pair_boundary_target_grads,
                    base_minimum_dots=target_actual_floors[boundary_start:],
                    pair_target_weights=pair_boundary_target_weights,
                    pre_boundary=pair_boundary_before["selected_margin"],
                    pair_local_delta=pair_pursuit_local_delta,
                    target_candidate_index=(
                        pair_boundary_before["selected_candidate_index"]
                    ),
                    proposed_descent=current_effective_descent,
                    reference=loss,
                )
            )
            target_actual_floors = (
                target_actual_floors[:boundary_start]
                + list(boundary_floor_info["projection_minimum_dots"])
            )
            pair_boundary_linearized_required_improvement = (
                loss.new_tensor(boundary_floor_info["minimum_dots"])
            )
            pair_boundary_linearized_crossing_affordable = (
                boundary_floor_info["crossing_affordable"]
            )
            pair_boundary_linearized_allocation_info = boundary_floor_info
            actual_all_target_dots_before = torch.stack([
                _gradient_tuple_dot(
                    current_effective_descent,
                    target_grads,
                    loss,
                )
                for target_grads in all_pair_target_grads
            ])
            actual_constraints = list(all_pair_target_grads)
            actual_floors = list(target_actual_floors)
            boundary_identity_group_constraints = []
            boundary_identity_group_floors = list(
                boundary_floor_info["identity_group_minimum_dots"]
            )
            for group_ordinal, member_indices in enumerate(
                    boundary_floor_info["identity_group_member_indices"]):
                group_constraint = _sum_fixed_parameter_gradient_tuples(
                    parameters=self.adj_parameters,
                    gradient_tuples=[
                        pair_boundary_target_grads[index]
                        for index in member_indices
                    ],
                    diagnostic_name=(
                        "pair_boundary_identity_group_{}".format(
                            group_ordinal
                        )
                    ),
                )
                boundary_identity_group_constraints.append(group_constraint)
                actual_constraints.append(group_constraint)
                actual_floors.append(
                    boundary_identity_group_floors[group_ordinal]
                )
            actual_constraints.append(pair_grads)
            actual_floors.append(float(
                _strict_gradient_dot_floor(
                    proposed_grads=current_effective_descent,
                    constraint_grads=pair_grads,
                    reference=loss,
                ).detach().cpu().item()
            ))
            if candidate_objective_active:
                if not bool((
                        candidate_actual_update_descent_dot_after > 0.0
                ).item()):
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "standard exact-pair target Adam projection has "
                        "no current candidate descent to preserve"
                    )
                actual_constraints.append(candidate_grads)
                # Keep the real candidate descent achieved by its Adam guard,
                # not merely an infinitesimal positive dot.
                actual_floors.append(float(
                    candidate_actual_update_descent_dot_after
                    .detach().cpu().item()
                ))
            for lifecycle_constraint_grad in lifecycle_constraint_grads:
                actual_constraints.append(lifecycle_constraint_grad)
                actual_floors.append(0.0)
            (
                target_safe_descent,
                target_actual_projection_info,
            ) = _project_gradient_tuple_to_minimum_dots(
                proposed_grads=current_effective_descent,
                constraint_grads=actual_constraints,
                minimum_dots=actual_floors,
                reference=loss,
                diagnostic_name=(
                    "pending exact-pair target Adam update"
                    if bool(pair_only_objective)
                    else "standard exact-pair target Adam update"
                ),
            )
            actual_all_target_dots_after = torch.stack([
                _gradient_tuple_dot(
                    target_safe_descent,
                    target_grads,
                    loss,
                )
                for target_grads in all_pair_target_grads
            ])
            boundary_identity_group_dots_before = [
                _gradient_tuple_dot(
                    current_effective_descent,
                    group_constraint,
                    loss,
                )
                for group_constraint in boundary_identity_group_constraints
            ]
            boundary_identity_group_dots_after = [
                _gradient_tuple_dot(
                    target_safe_descent,
                    group_constraint,
                    loss,
                )
                for group_constraint in boundary_identity_group_constraints
            ]
            if not bool(torch.all(
                    actual_all_target_dots_after > 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "{} exact-pair score or selection-boundary Adam "
                    "update lost strict descent".format(
                        "pending"
                        if bool(pair_only_objective)
                        else "standard"
                    )
                )
            identity_group_floor_violations = []
            for group_dot, group_floor, group_constraint in zip(
                    boundary_identity_group_dots_after,
                    boundary_identity_group_floors,
                    boundary_identity_group_constraints):
                group_tolerance = float(
                    _gradient_dot_tolerance(
                        target_safe_descent,
                        group_constraint,
                        loss,
                        tolerance_multiplier=128.0,
                    ).detach().cpu().item()
                )
                identity_group_floor_violations.append(bool(
                    (group_dot < group_floor - group_tolerance).item()
                ))
            if any(identity_group_floor_violations):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "canonical-identity boundary budget was not preserved"
                )
            if target_actual_projection_info["intervened"] > 0.0:
                with torch.no_grad():
                    for parameter, before, safe_descent in zip(
                            self.adj_parameters,
                            pair_parameter_before_step,
                            target_safe_descent):
                        if before is not None and safe_descent is not None:
                            parameter.copy_(before - safe_descent)
                pair_target_actual_direction_guard_applied = (
                    loss.new_tensor(1.0)
                )
                boundary_required = loss.new_tensor(
                    target_actual_floors[boundary_start:]
                )
                identity_group_guard_applied = any(
                    bool((group_dot < group_floor).item())
                    for group_dot, group_floor in zip(
                        boundary_identity_group_dots_before,
                        boundary_identity_group_floors,
                    )
                )
                pair_boundary_actual_direction_guard_applied = (
                    loss.new_tensor(float(bool(torch.any(
                        actual_all_target_dots_before[boundary_start:]
                        < boundary_required
                    ).item()) or identity_group_guard_applied))
                )
            pair_target_actual_min_descent_dot_before = (
                actual_all_target_dots_before[
                    :len(pair_target_score_grads)
                ].min()
            )
            pair_target_actual_min_descent_dot_after = (
                actual_all_target_dots_after[
                    :len(pair_target_score_grads)
                ].min()
            )
            pair_boundary_actual_min_descent_dot_before = (
                actual_all_target_dots_before[
                    len(pair_target_score_grads):
                ].min()
            )
            pair_boundary_actual_min_descent_dot_after = (
                actual_all_target_dots_after[
                    len(pair_target_score_grads):
                ].min()
            )
            # The real boundary is a max over the reachable competitors and is
            # therefore piecewise smooth.  A finite Adam step can cross into a
            # different competitor region even when every pre-step Jacobian dot
            # is strictly correct.  Backtrack the *whole* jointly-safe
            # displacement, preserving all current linear priorities, until the
            # committed generator boundary improves for every target.
            boundary_final_displacements = [
                (
                    None
                    if before is None
                    else parameter.detach() - before
                )
                for parameter, before in zip(
                    self.adj_parameters,
                    pair_exact_search_parameter_before_step,
                )
            ]
            progress_member_ordinals = sorted(set(
                int(ordinal)
                for ordinal, (is_progress, extra_budget) in enumerate(zip(
                    boundary_floor_info[
                        "identity_group_progress_member_flags"
                    ],
                    boundary_floor_info["identity_group_extra_budgets"],
                ))
                if int(is_progress) == 1 and float(extra_budget) > 0.0
            ))
            progress_member_requirements = [
                float(boundary_floor_info[
                    "identity_group_progress_required"
                ][ordinal])
                for ordinal in progress_member_ordinals
            ]
            exact_target_locations = torch.nonzero(
                actionable_pair_mask
            ).detach().cpu().tolist()
            if len(exact_target_locations) != len(
                    boundary_floor_info["target_strict_floors"]):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "joint exact target population differs from the boundary "
                    "floor population"
                )
            # Every selected-score halfspace is a hard non-regression
            # constraint, not an additional nonlinear progress objective.
            # Run140 demonstrated this for transactions with nonlinear progress
            # members; a later resolution-only standard transaction exposed the
            # same float32 writeback quantization with two scores preserved at
            # zero.  Apply the documented preservation contract to every exact
            # score population.  Boundary, candidate, lifecycle, and any real
            # progress-member gates retain their existing strict contracts.
            joint_exact_preservation_tolerance = float(
                128.0 * _floating_dtype_epsilon(loss)
            )
            progress_search_scale_limit = 1.0
            for ordinal, requirement in zip(
                    progress_member_ordinals,
                    progress_member_requirements):
                pre_deficit = float(
                    boundary_floor_info["pre_deficits"][ordinal]
                    .detach().cpu().item()
                )
                if pre_deficit <= 0.0:
                    continue
                if not np.isfinite(requirement) or requirement <= 0.0:
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "positive boundary deficit has no finite progress "
                        "requirement"
                    )
                closure_scale = pre_deficit / requirement
                if not np.isfinite(closure_scale) or closure_scale < 0.0:
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "boundary deficit closure scale is invalid"
                    )
                progress_search_scale_limit = max(
                    progress_search_scale_limit,
                    closure_scale,
                )
            if progress_search_scale_limit > (2.0 ** 20):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "boundary deficit closure scale exceeds bounded exact "
                    "search capacity"
                )

            def _evaluate_joint_pair_exact_scale(
                    boundary_trial_scale,
                    trial_displacements=None,
                    trial_parameter_before=None):
                if trial_displacements is None:
                    trial_displacements = boundary_final_displacements
                if trial_parameter_before is None:
                    trial_parameter_before = (
                        pair_exact_search_parameter_before_step
                    )
                try:
                    _restore_scaled_transaction_parameter_state(
                        parameters=self.adj_parameters,
                        parameter_before_step=trial_parameter_before,
                        parameter_displacements=trial_displacements,
                        scale=boundary_trial_scale,
                        require_complete=not bool(pair_only_objective),
                    )
                except (RuntimeError, ValueError):
                    _rollback_failed_pair_transaction()
                    raise
                with torch.no_grad():
                    try:
                        boundary_revalidation = (
                            self.adj_network
                            .evaluate_selected_factor_replay_boundaries(
                                rnn_obs=rnn_obs,
                                dones=dones.bool(),
                                adj=adj,
                                previous_adj=previous_adj,
                                include_behavior_logp=True,
                            )
                        )
                    except SelectedFactorInactiveCandidateError as error:
                        # Scale zero is the immutable transaction origin and
                        # was already required to satisfy the production
                        # catalog.  Never hide origin drift.  At any nonzero
                        # trial scale, an inactive selected factor is exactly a
                        # hard line-search boundary: reject this probe and let
                        # the bounded search restore/evaluate another scale.
                        if float(boundary_trial_scale) == 0.0:
                            _rollback_failed_pair_transaction()
                            raise
                        return (
                            _joint_exact_inactive_selected_factor_acceptance(
                                error
                            )
                        )
                    signed_boundary_change = (
                        torch.sign(pair_pursuit_local_delta)
                        * (
                            boundary_revalidation["selected_margin"]
                            - pair_boundary_before["selected_margin"].detach()
                        )
                    )
                    signed_exact_score_change = (
                        torch.sign(pair_pursuit_local_delta)
                        * (
                            boundary_revalidation["behavior_selected_logp"]
                            - pair_boundary_before[
                                "behavior_selected_logp"
                            ].detach()
                        )
                    )
                    candidate_loss_change = None
                    exact_catalog_change_kind = None
                    if candidate_objective_active:
                        (
                            boundary_candidate_margins,
                            boundary_candidate_valid,
                        ) = (
                            self.adj_network
                            .evaluate_candidate_identity_active_competitor_margins(
                                rnn_obs=rnn_obs,
                                dones=dones.bool(),
                                adj=adj,
                            )
                        )
                        if not bool(torch.equal(
                                boundary_candidate_valid > 0.0,
                                candidate_identity_valid_mask > 0.0,
                        )):
                            if float(boundary_trial_scale) == 0.0:
                                _rollback_failed_pair_transaction()
                                raise RuntimeError(
                                    "current candidate catalog changed at "
                                    "joint pair exact transaction origin"
                                )
                            exact_catalog_change_kind = (
                                "candidate_catalog_changed"
                            )
                        else:
                            boundary_candidate_info = (
                                compute_capture_candidate_identity_active_competitor_loss(
                                    candidate_competitor_margins=(
                                        boundary_candidate_margins
                                    ),
                                    candidate_reference_margins=(
                                        behavior_candidate_margin
                                    ),
                                    candidate_identity_delta=(
                                        candidate_identity_delta[..., 0]
                                    ),
                                    candidate_valid_mask=(
                                        boundary_candidate_valid
                                    ),
                                    transition_mask=transition_mask,
                                )
                            )
                            candidate_loss_change = (
                                boundary_candidate_info["loss"]
                                - pair_exact_candidate_loss_origin
                            )

                    lifecycle_signed_margin = None
                    lifecycle_tolerance = None
                    if (
                            lifecycle_target_present
                            and exact_catalog_change_kind is None):
                        (
                            boundary_lifecycle_margins,
                            boundary_lifecycle_valid,
                        ) = (
                            self.adj_network
                            .evaluate_candidate_identity_active_competitor_margins(
                                rnn_obs=lifecycle_rnn_obs,
                                dones=lifecycle_dones.bool(),
                                adj=lifecycle_adj,
                            )
                        )
                        if not bool(torch.equal(
                                boundary_lifecycle_valid > 0.0,
                                lifecycle_valid_mask > 0.0,
                        )):
                            if float(boundary_trial_scale) == 0.0:
                                _rollback_failed_pair_transaction()
                                raise RuntimeError(
                                    "candidate lifecycle catalog changed at "
                                    "joint pair exact transaction origin"
                                )
                            exact_catalog_change_kind = (
                                "lifecycle_catalog_changed"
                            )
                        else:
                            lifecycle_signed_margin = (
                                torch.sign(lifecycle_delta)
                                * boundary_lifecycle_margins
                            )
                            lifecycle_tolerance = (
                                16.0
                                * lifecycle_dtype_epsilon
                                * (
                                    lifecycle_signed_floor.abs()
                                    + lifecycle_signed_margin.abs()
                                ).clamp_min(1.0e-12)
                            )

                    lifecycle_exact_available = (
                        lifecycle_target_present
                        and lifecycle_signed_margin is not None
                    )
                    acceptance = _joint_exact_constraint_acceptance(
                        signed_boundary_change=signed_boundary_change,
                        actionable_pair_mask=actionable_pair_mask,
                        signed_exact_score_change=signed_exact_score_change,
                        candidate_loss_change=candidate_loss_change,
                        lifecycle_signed_margin=lifecycle_signed_margin,
                        lifecycle_signed_floor=(
                            lifecycle_signed_floor
                            if lifecycle_exact_available else None
                        ),
                        lifecycle_target_mask=(
                            lifecycle_target_mask
                            if lifecycle_exact_available else None
                        ),
                        lifecycle_tolerance=lifecycle_tolerance,
                        preservation_tolerance=(
                            joint_exact_preservation_tolerance
                        ),
                    )
                    if exact_catalog_change_kind is not None:
                        acceptance = _joint_exact_catalog_change_acceptance(
                            acceptance=acceptance,
                            catalog_kind=exact_catalog_change_kind,
                        )
                    # Reuse this authoritative production catalog when the
                    # complete transaction trace is persisted.  Returning the
                    # detached/no-grad tensors avoids another expensive GAT
                    # and candidate-catalog forward for diagnostics alone.
                    acceptance["boundary_revalidation"] = (
                        boundary_revalidation
                    )
                    acceptance["competitor_candidate_indices"] = tuple(
                        int(round(float(
                            boundary_revalidation[
                                "competitor_candidate_index"
                            ][tuple(int(item) for item in location)]
                            .detach().cpu().item()
                        )))
                        for location in exact_target_locations
                    )
                    acceptance["target_ranks"] = tuple(
                        int(round(float(
                            boundary_revalidation["selected_rank"][
                                tuple(int(item) for item in location)
                            ].detach().cpu().item()
                        )))
                        for location in exact_target_locations
                    )
                    acceptance["target_active"] = tuple(
                        int(rank == 1)
                        for rank in acceptance["target_ranks"]
                    )
                    if progress_member_ordinals:
                        progress_values = acceptance[
                            "signed_boundary_change_values"
                        ]
                        progress_actual_values = [
                            float(progress_values[ordinal]
                                  .detach().cpu().item())
                            for ordinal in progress_member_ordinals
                        ]
                        completion_values = [
                            actual / requirement
                            for actual, requirement in zip(
                                progress_actual_values,
                                progress_member_requirements,
                            )
                        ]
                        worst_progress_index = min(
                            range(len(completion_values)),
                            key=lambda index: (
                                completion_values[index], index
                            ),
                        )
                        acceptance["progress_target_present"] = True
                        acceptance["progress_worst_actual"] = (
                            progress_actual_values[worst_progress_index]
                        )
                        acceptance["progress_worst_required"] = (
                            progress_member_requirements[
                                worst_progress_index
                            ]
                        )
                        acceptance["progress_min_completion"] = min(
                            completion_values
                        )
                        acceptance["progress_mean_completion"] = (
                            float(sum(completion_values))
                            / float(len(completion_values))
                        )
                    else:
                        acceptance["progress_target_present"] = False
                        acceptance["progress_worst_actual"] = None
                        acceptance["progress_worst_required"] = None
                        acceptance["progress_min_completion"] = 1.0
                        acceptance["progress_mean_completion"] = 1.0
                    return acceptance

            # v19 recovered the largest safe scale on one affine-projected
            # direction, but run124 still realized only 69% and 2.5% of the
            # designated progress requirement.  Search a small deterministic
            # family which relaxes only the *linear extra-progress floor* for
            # multi-exposure groups.  Every member's strict floor, every exact
            # score, pair aggregate, candidate, lifecycle, and the original
            # direction-construction budget remain unchanged.  The subsequent
            # exact search may expand a safe ray only to the current
            # deficit/required closure bound.  Selection is by real
            # post-forward completion against the original unscaled
            # requirement, so neither a relaxed direction nor a larger scale
            # is accepted for a cosmetically easier contract.
            direction_records = [{
                "direction_kind": "adam_projection",
                "progress_floor_fraction": 1.0,
                "safe_descent": target_safe_descent,
                "projection_info": target_actual_projection_info,
                "all_target_dots": actual_all_target_dots_after,
                "group_dots": boundary_identity_group_dots_after,
                "displacements": tuple(boundary_final_displacements),
            }]
            progress_seed_info = {
                "selected_member_ordinals": (),
                "excluded_zero_budget_member_ordinals": (),
            }
            progress_seed_raw_norm = 0.0
            adam_reference_norm = float(torch.sqrt(
                _gradient_tuple_dot(
                    current_effective_descent,
                    current_effective_descent,
                    loss,
                )
            ).detach().cpu().item())
            progress_component_norm = 0.0
            if boundary_identity_group_constraints:
                group_constraint_start = len(all_pair_target_grads)

                def _progress_fraction_floors(progress_floor_fraction):
                    candidate_floors = list(actual_floors)
                    for group_ordinal, member_indices in enumerate(
                            boundary_floor_info[
                                "identity_group_member_indices"
                            ]):
                        progress_member = int(member_indices[0])
                        strict_floor = float(
                            boundary_floor_info["target_strict_floors"][
                                progress_member
                            ]
                        )
                        extra_budget = float(
                            boundary_floor_info[
                                "identity_group_extra_budgets"
                            ][progress_member]
                        )
                        candidate_floors[
                            group_constraint_start + group_ordinal
                        ] = (
                            strict_floor
                            + float(progress_floor_fraction) * extra_budget
                        )
                    return candidate_floors

                for progress_floor_fraction in (0.75, 0.5, 0.25, 0.125):
                    alternative_floors = _progress_fraction_floors(
                        progress_floor_fraction
                    )
                    (
                        alternative_descent,
                        alternative_projection_info,
                    ) = _project_gradient_tuple_to_minimum_dots(
                        proposed_grads=current_effective_descent,
                        constraint_grads=actual_constraints,
                        minimum_dots=alternative_floors,
                        reference=loss,
                        diagnostic_name=(
                            "pending nonlinear pair progress direction"
                            if bool(pair_only_objective)
                            else "standard nonlinear pair progress direction"
                        ),
                    )
                    alternative_target_dots = torch.stack([
                        _gradient_tuple_dot(
                            alternative_descent,
                            target_grads,
                            loss,
                        )
                        for target_grads in all_pair_target_grads
                    ])
                    if not bool(torch.all(
                            alternative_target_dots > 0.0).item()):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "nonlinear progress direction lost an exact-score "
                            "or member-boundary constraint"
                        )
                    alternative_group_dots = [
                        _gradient_tuple_dot(
                            alternative_descent,
                            group_constraint,
                            loss,
                        )
                        for group_constraint in (
                            boundary_identity_group_constraints
                        )
                    ]
                    alternative_displacements = tuple(
                        None if descent is None else -descent
                        for descent in alternative_descent
                    )
                    direction_records.append({
                        "direction_kind": "adam_projection",
                        "progress_floor_fraction": float(
                            progress_floor_fraction
                        ),
                        "safe_descent": alternative_descent,
                        "projection_info": alternative_projection_info,
                        "all_target_dots": alternative_target_dots,
                        "group_dots": alternative_group_dots,
                        "displacements": alternative_displacements,
                    })

                # v20 proved that lowering a progress floor can expose a
                # different feasible ray, but it still starts every candidate
                # from the same Adam proposal.  In run125 seq721 that recovered
                # only 24.6% of the original requirement and left a 0.184
                # production-boundary deficit.  Add a genuinely different
                # proposal: blend the Adam descent with the normalized sum of
                # the designated progress-member Jacobians, then solve the
                # *same* all-member/candidate/lifecycle halfspaces.  This does
                # not enlarge any floor or budget.  Exact production forward
                # validation and the existing best-real-progress selector
                # remain the only commit authority.
                progress_seed_info = _select_boundary_progress_seed_members(
                    identity_group_member_indices=boundary_floor_info[
                        "identity_group_member_indices"
                    ],
                    progress_member_flags=boundary_floor_info[
                        "identity_group_progress_member_flags"
                    ],
                    identity_group_extra_budgets=boundary_floor_info[
                        "identity_group_extra_budgets"
                    ],
                )
                progress_seed_ordinals = progress_seed_info[
                    "selected_member_ordinals"
                ]
                progress_seed = None
                progress_seed_norm_sq = loss.new_tensor(0.0)
                if progress_seed_ordinals:
                    progress_seed = _sum_fixed_parameter_gradient_tuples(
                        parameters=self.adj_parameters,
                        gradient_tuples=[
                            pair_boundary_target_grads[int(ordinal)]
                            for ordinal in progress_seed_ordinals
                        ],
                        diagnostic_name="pair progress-member search seed",
                    )
                    progress_seed_norm_sq = _gradient_tuple_dot(
                        progress_seed, progress_seed, loss
                    )
                adam_descent_norm_sq = _gradient_tuple_dot(
                    current_effective_descent,
                    current_effective_descent,
                    loss,
                )
                progress_seed_raw_norm = float(torch.sqrt(
                    progress_seed_norm_sq
                ).detach().cpu().item())
                if bool(
                        (progress_seed_norm_sq > 0.0).item()
                        and (adam_descent_norm_sq > 0.0).item()):
                    seed_scale = torch.sqrt(
                        adam_descent_norm_sq / progress_seed_norm_sq
                    )
                    progress_component_norm = float(torch.sqrt(
                        progress_seed_norm_sq * seed_scale * seed_scale
                    ).detach().cpu().item())
                    progress_balanced_proposal = tuple(
                        (
                            None
                            if adam_grad is None and progress_grad is None
                            else (
                                0.5 * adam_grad
                                + 0.5 * seed_scale * progress_grad
                                if adam_grad is not None
                                and progress_grad is not None
                                else (
                                    0.5 * adam_grad
                                    if adam_grad is not None
                                    else 0.5 * seed_scale * progress_grad
                                )
                            )
                        )
                        for adam_grad, progress_grad in zip(
                            current_effective_descent,
                            progress_seed,
                        )
                    )
                    for progress_floor_fraction in (1.0, 0.5, 0.25):
                        tangent_floors = _progress_fraction_floors(
                            progress_floor_fraction
                        )
                        (
                            tangent_descent,
                            tangent_projection_info,
                        ) = _project_gradient_tuple_to_minimum_dots(
                            proposed_grads=progress_balanced_proposal,
                            constraint_grads=actual_constraints,
                            minimum_dots=tangent_floors,
                            reference=loss,
                            diagnostic_name=(
                                "pending progress-member tangent direction"
                                if bool(pair_only_objective)
                                else "standard progress-member tangent direction"
                            ),
                        )
                        tangent_target_dots = torch.stack([
                            _gradient_tuple_dot(
                                tangent_descent,
                                target_grads,
                                loss,
                            )
                            for target_grads in all_pair_target_grads
                        ])
                        if not bool(torch.all(
                                tangent_target_dots > 0.0).item()):
                            _rollback_failed_pair_transaction()
                            raise RuntimeError(
                                "progress-member tangent direction lost an "
                                "exact-score or member-boundary constraint"
                            )
                        tangent_group_dots = [
                            _gradient_tuple_dot(
                                tangent_descent,
                                group_constraint,
                                loss,
                            )
                            for group_constraint in (
                                boundary_identity_group_constraints
                            )
                        ]
                        direction_records.append({
                            "direction_kind": "deficit_progress_seed",
                            "progress_floor_fraction": float(
                                progress_floor_fraction
                            ),
                            "safe_descent": tangent_descent,
                            "projection_info": tangent_projection_info,
                            "all_target_dots": tangent_target_dots,
                            "group_dots": tangent_group_dots,
                            "displacements": tuple(
                                None if descent is None else -descent
                                for descent in tangent_descent
                            ),
                        })

            full_direction = direction_records[0]["safe_descent"]
            full_direction_norm_sq = _gradient_tuple_dot(
                full_direction, full_direction, loss
            )
            for record in direction_records:
                direction_norm_sq = _gradient_tuple_dot(
                    record["safe_descent"], record["safe_descent"], loss
                )
                cosine = _gradient_tuple_dot(
                    record["safe_descent"], full_direction, loss
                ) / torch.sqrt(
                    direction_norm_sq * full_direction_norm_sq
                )
                record["direction_norm"] = float(
                    torch.sqrt(direction_norm_sq).detach().cpu().item()
                )
                record["cosine_vs_full"] = float(
                    cosine.detach().cpu().item()
                )
                record["progress_seed_member_ordinals"] = tuple(
                    progress_seed_info["selected_member_ordinals"]
                )
                record["progress_seed_member_count"] = len(
                    progress_seed_info["selected_member_ordinals"]
                )
                record[
                    "progress_seed_zero_budget_excluded_ordinals"
                ] = tuple(progress_seed_info[
                    "excluded_zero_budget_member_ordinals"
                ])
                record[
                    "progress_seed_zero_budget_excluded_count"
                ] = len(progress_seed_info[
                    "excluded_zero_budget_member_ordinals"
                ])
                record["progress_seed_raw_norm"] = float(
                    progress_seed_raw_norm
                )
                record["adam_reference_norm"] = float(
                    adam_reference_norm
                )
                record["progress_component_norm"] = float(
                    progress_component_norm
                )

            direction_evaluators = []
            for record in direction_records:
                trial_displacements = record["displacements"]

                def _evaluate_direction_scale(
                        trial_scale,
                        fixed_displacements=trial_displacements):
                    return _evaluate_joint_pair_exact_scale(
                        trial_scale,
                        trial_displacements=fixed_displacements,
                    )

                direction_evaluators.append((
                    record["progress_floor_fraction"],
                    _evaluate_direction_scale,
                ))
            boundary_scale_info = _select_joint_exact_progress_direction(
                direction_evaluators=direction_evaluators,
                max_halvings=20,
                refinement_steps=12,
                maximum_scale=progress_search_scale_limit,
                max_expansions=20,
            )
            if not bool(boundary_scale_info["valid"]):
                # The floor variants above may all project onto the same Adam
                # ray (notably when every identity group has zero extra budget).
                # A reported candidate_count therefore need not represent that
                # many independent nonlinear directions.  Use the exact-forward
                # boundary limiters from the failed rays to build a bounded,
                # deterministic tangent family.  Every tangent is projected
                # through the unchanged exact-score, member-boundary, aggregate
                # pair, candidate, lifecycle, and group-budget halfspaces before
                # the production forward can authorize it.
                limiter_ordinals = _failed_boundary_limiter_ordinals(
                    candidate_results=boundary_scale_info[
                        "candidate_results"
                    ],
                    boundary_target_count=len(pair_boundary_target_grads),
                )
                limiter_seed_specs = [
                    (
                        (int(limiter_ordinal),),
                        pair_boundary_target_grads[limiter_ordinal],
                        "boundary_limiter",
                    )
                    for limiter_ordinal in limiter_ordinals
                ]
                if len(limiter_ordinals) > 1:
                    limiter_seed_specs.append((
                        tuple(int(value) for value in limiter_ordinals),
                        _sum_fixed_parameter_gradient_tuples(
                            parameters=self.adj_parameters,
                            gradient_tuples=[
                                pair_boundary_target_grads[ordinal]
                                for ordinal in limiter_ordinals
                            ],
                            diagnostic_name=(
                                "failed_boundary_limiter_bundle"
                            ),
                        ),
                        "boundary_limiter_bundle",
                    ))
                for (
                        limiter_member_ordinals,
                        limiter_seed_grads,
                        limiter_kind_prefix,
                ) in limiter_seed_specs:
                    for seed_fraction, direction_suffix in (
                            (0.5, "blend"),
                            (1.0, "tangent")):
                        direction_kind = validate_pair_direction_candidate_kind(
                            "{}_{}".format(
                                limiter_kind_prefix, direction_suffix
                            )
                        )
                        limiter_proposal, limiter_seed_info = (
                            _normalized_gradient_seed_blend(
                                proposed_grads=current_effective_descent,
                                seed_grads=limiter_seed_grads,
                                reference=loss,
                                seed_fraction=seed_fraction,
                            )
                        )
                        (
                            limiter_descent,
                            limiter_projection_info,
                        ) = _project_gradient_tuple_to_minimum_dots(
                            proposed_grads=limiter_proposal,
                            constraint_grads=actual_constraints,
                            minimum_dots=actual_floors,
                            reference=loss,
                            diagnostic_name=(
                                "pending failed-boundary limiter direction"
                                if bool(pair_only_objective)
                                else "standard failed-boundary limiter direction"
                            ),
                        )
                        limiter_target_dots = torch.stack([
                            _gradient_tuple_dot(
                                limiter_descent,
                                target_grads,
                                loss,
                            )
                            for target_grads in all_pair_target_grads
                        ])
                        if not bool(torch.all(
                                limiter_target_dots > 0.0).item()):
                            _rollback_failed_pair_transaction()
                            raise RuntimeError(
                                "failed-boundary limiter direction lost an "
                                "exact-score or member-boundary constraint"
                            )
                        limiter_group_dots = [
                            _gradient_tuple_dot(
                                limiter_descent,
                                group_constraint,
                                loss,
                            )
                            for group_constraint in (
                                boundary_identity_group_constraints
                            )
                        ]
                        limiter_direction_norm_sq = _gradient_tuple_dot(
                            limiter_descent, limiter_descent, loss
                        )
                        limiter_cosine = _gradient_tuple_dot(
                            limiter_descent, full_direction, loss
                        ) / torch.sqrt(
                            limiter_direction_norm_sq
                            * full_direction_norm_sq
                        )
                        limiter_record = {
                            "direction_kind": direction_kind,
                            "progress_floor_fraction": 1.0,
                            "safe_descent": limiter_descent,
                            "projection_info": limiter_projection_info,
                            "all_target_dots": limiter_target_dots,
                            "group_dots": limiter_group_dots,
                            "displacements": tuple(
                                None if descent is None else -descent
                                for descent in limiter_descent
                            ),
                            "direction_norm": float(torch.sqrt(
                                limiter_direction_norm_sq
                            ).detach().cpu().item()),
                            "cosine_vs_full": float(
                                limiter_cosine.detach().cpu().item()
                            ),
                            "progress_seed_member_ordinals": tuple(
                                limiter_member_ordinals
                            ),
                            "progress_seed_member_count": len(
                                limiter_member_ordinals
                            ),
                            "progress_seed_zero_budget_excluded_ordinals": (
                                tuple(progress_seed_info[
                                    "excluded_zero_budget_member_ordinals"
                                ])
                            ),
                            "progress_seed_zero_budget_excluded_count": len(
                                progress_seed_info[
                                    "excluded_zero_budget_member_ordinals"
                                ]
                            ),
                            "progress_seed_raw_norm": float(
                                limiter_seed_info["seed_raw_norm"]
                            ),
                            "adam_reference_norm": float(
                                limiter_seed_info["reference_norm"]
                            ),
                            "progress_component_norm": float(
                                limiter_seed_info["seed_component_norm"]
                            ),
                        }
                        direction_records.append(limiter_record)
                        limiter_displacements = limiter_record["displacements"]

                        def _evaluate_limiter_scale(
                                trial_scale,
                                fixed_displacements=limiter_displacements):
                            return _evaluate_joint_pair_exact_scale(
                                trial_scale,
                                trial_displacements=fixed_displacements,
                            )

                        direction_evaluators.append((
                            limiter_record["progress_floor_fraction"],
                            _evaluate_limiter_scale,
                        ))
                # run163 then reached the complementary float32 case: all
                # seven independent/variant rays satisfied their strict
                # positive Jacobian dots, while every exact boundary change at
                # scales 1, 1/2, ... was still signed zero.  Halving cannot
                # make a quantized output visible.  Derive a finite upper bound
                # from the actual selected/competitor log-probability write
                # resolution and the best complete boundary-dot vector, then
                # let the unchanged exact forward search above one.  No
                # boundary, candidate, lifecycle, or score tolerance changes.
                progress_search_scale_limit = (
                    _boundary_write_resolution_scale_limit(
                        selected_logp=pair_boundary_before["selected_logp"],
                        competitor_logp=(
                            pair_boundary_before["competitor_logp"]
                        ),
                        actionable_pair_mask=actionable_pair_mask,
                        direction_boundary_dots=[
                            record["all_target_dots"][boundary_start:]
                            for record in direction_records
                        ],
                        reference=loss,
                        current_scale_limit=progress_search_scale_limit,
                        maximum_scale=2.0 ** 20,
                    )
                )
                boundary_scale_info = (
                    _select_joint_exact_progress_direction(
                        direction_evaluators=direction_evaluators,
                        max_halvings=20,
                        refinement_steps=12,
                        maximum_scale=progress_search_scale_limit,
                        max_expansions=20,
                    )
                )
            if not bool(boundary_scale_info["valid"]):
                # No candidate survived the bounded dyadic search.  Do not
                # conflate that with proof that the transaction origin was
                # already illegal or that no positive safe island exists
                # between sampled scales.  Persist every exact production
                # trial and a separate scale-zero preservation classification,
                # then retain the existing fail-loud/atomic rollback behavior.
                origin_info = direction_evaluators[0][1](0.0)
                origin_preservation = (
                    _joint_exact_origin_preservation_acceptance(
                        info=origin_info,
                        reference=loss,
                    )
                )
                target_locations = torch.nonzero(
                    actionable_pair_mask
                ).detach().cpu().tolist()
                target_candidate_indices = []
                target_identities = []
                target_signs = []
                target_transition_indices = []
                target_factor_indices = []
                target_selected_episode_ordinals = []
                target_episode_transition_steps = []
                target_pair_evidence_flags = []
                for raw_location in target_locations:
                    location = tuple(int(item) for item in raw_location)
                    transition_index = int(location[0])
                    factor_index = int(location[1])
                    candidate_index = int(round(float(
                        pair_boundary_before["selected_candidate_index"][
                            location
                        ].detach().cpu().item()
                    )))
                    _order, canonical_identity = (
                        _canonical_candidate_identity(
                            candidate_index=candidate_index,
                            num_agents=self.num_agents,
                            highest_orders=self.adj_network.highest_orders,
                        )
                    )
                    target_candidate_indices.append(candidate_index)
                    target_identities.append(canonical_identity)
                    target_signs.append(int(torch.sign(
                        pair_pursuit_local_delta[location]
                    ).detach().cpu().item()))
                    target_transition_indices.append(transition_index)
                    target_factor_indices.append(factor_index)
                    provenance = replay_population_provenance[
                        transition_index
                    ]
                    target_selected_episode_ordinals.append(int(round(float(
                        provenance[1].detach().cpu().item()
                    ))))
                    target_episode_transition_steps.append(int(round(float(
                        provenance[2].detach().cpu().item()
                    ))))
                    target_pair_evidence_flags.append(int(round(float(
                        provenance[0].detach().cpu().item()
                    ))))

                # The production selector now includes a bounded three-level
                # interior lattice in its authoritative search.  Reuse those
                # exact-forward traces for failure classification instead of
                # running a second, diagnostic-only search whose feasible
                # points could not be committed.
                failure_probe_traces = []
                failure_probe_valid_count = 0
                for candidate_result in boundary_scale_info[
                        "candidate_results"]:
                    candidate_probes = [
                        trial for trial in candidate_result["trial_trace"]
                        if trial["evaluation_kind"] in (
                            "resolution_interval_subdivision_search",
                            "interval_subdivision_search",
                            "interval_island_refinement",
                            "interval_island_selection",
                        )
                    ]
                    failure_probe_valid_count += sum(
                        int(bool(trial["valid"]))
                        for trial in candidate_probes
                    )
                    failure_probe_traces.append(tuple(candidate_probes))

                shared_failure_fields = {
                    "diagnostic_version": int(
                        STRICT_PAIR_EXACT_FAILURE_DIAGNOSTIC_VERSION
                    ),
                    "optimizer_kind": (
                        "pair_pending_adam"
                        if bool(pair_only_objective)
                        else "standard_adam"
                    ),
                    "target_count": int(len(target_locations)),
                    "target_candidate_indices": "|".join(
                        str(value) for value in target_candidate_indices
                    ),
                    "target_canonical_identities": "|".join(
                        str(value) for value in target_identities
                    ),
                    "target_signs": "|".join(
                        str(value) for value in target_signs
                    ),
                    "target_transition_indices": "|".join(
                        str(value) for value in target_transition_indices
                    ),
                    "target_factor_indices": "|".join(
                        str(value) for value in target_factor_indices
                    ),
                    "target_selected_episode_ordinals": "|".join(
                        str(value)
                        for value in target_selected_episode_ordinals
                    ),
                    "target_episode_transition_steps": "|".join(
                        str(value)
                        for value in target_episode_transition_steps
                    ),
                    "target_pair_evidence_flags": "|".join(
                        str(value) for value in target_pair_evidence_flags
                    ),
                    "candidate_count": int(
                        boundary_scale_info["candidate_count"]
                    ),
                    "diagnostic_probe_valid_count": int(
                        failure_probe_valid_count
                    ),
                    # Reaching this block means every configured nonlinear
                    # direction, scale lattice, interval probe, and limiter
                    # tangent fallback has completed without a commit.  This
                    # explicit v5 bit prevents an older/coarser search from
                    # ever being treated as a safe bounded deferral.
                    "bounded_search_exhaustive": 1,
                    "failure_classification": (
                        "transaction_origin_preservation_invalid"
                        if not bool(origin_preservation["valid"])
                        else (
                            "search_missed_feasible_midpoint"
                            if failure_probe_valid_count > 0
                            else "sampled_grid_has_no_feasible_point"
                        )
                    ),
                    "origin_preservation_valid": int(bool(
                        origin_preservation["valid"]
                    )),
                    "origin_preservation_tolerance": float(
                        origin_preservation["tolerance"]
                    ),
                }
                failure_rows = []
                origin_row = dict(shared_failure_fields)
                origin_row.update({
                    "candidate_ordinal": -1,
                    "direction_kind": "origin_preservation",
                    "progress_floor_fraction": None,
                    "direction_norm": 0.0,
                    "cosine_vs_full": None,
                    "evaluation_ordinal": -1,
                    "evaluation_kind": "origin_preservation",
                    "scale": 0.0,
                    "parameter_displacement_norm": 0.0,
                    "predicted_exact_score_min_change": 0.0,
                    "predicted_boundary_min_change": 0.0,
                    "valid": int(bool(origin_info["valid"])),
                    "boundary_valid": int(bool(
                        origin_info["boundary_valid"]
                    )),
                    "boundary_min_signed_change": float(
                        origin_info["boundary_min_signed_change"]
                        .detach().cpu().item()
                    ),
                    "exact_score_valid": int(bool(
                        origin_info["exact_score_valid"]
                    )),
                    "exact_score_min_signed_change": (
                        None
                        if origin_info[
                            "exact_score_min_signed_change"
                        ] is None
                        else float(origin_info[
                            "exact_score_min_signed_change"
                        ].detach().cpu().item())
                    ),
                    "candidate_valid": int(bool(
                        origin_info["candidate_valid"]
                    )),
                    "candidate_loss_change": (
                        None
                        if origin_info["candidate_loss_change"] is None
                        else float(origin_info["candidate_loss_change"]
                                   .detach().cpu().item())
                    ),
                    "lifecycle_valid": int(bool(
                        origin_info["lifecycle_valid"]
                    )),
                    "lifecycle_violation_count": int(
                        origin_info["lifecycle_violation_count"]
                    ),
                    "lifecycle_min_signed_gap": (
                        None
                        if origin_info["lifecycle_min_signed_gap"] is None
                        else float(origin_info["lifecycle_min_signed_gap"]
                                   .detach().cpu().item())
                    ),
                    "limiting_constraint_type": (
                        origin_info["limiting_constraint_type"]
                    ),
                    "limiting_target_ordinal": int(
                        origin_info["limiting_target_ordinal"]
                    ),
                    "boundary_failed_target_ordinals": "",
                    "progress_target_present": int(bool(
                        origin_info["progress_target_present"]
                    )),
                    "progress_worst_actual": (
                        origin_info["progress_worst_actual"]
                    ),
                    "progress_worst_required": (
                        origin_info["progress_worst_required"]
                    ),
                    "progress_min_completion": float(
                        origin_info["progress_min_completion"]
                    ),
                    "progress_mean_completion": float(
                        origin_info["progress_mean_completion"]
                    ),
                    "competitor_candidate_indices": "|".join(
                        str(value) for value in origin_info[
                            "competitor_candidate_indices"
                        ]
                    ),
                    "target_ranks": "|".join(
                        str(value) for value in origin_info["target_ranks"]
                    ),
                    "target_active": "|".join(
                        str(value) for value in origin_info["target_active"]
                    ),
                })
                failure_rows.append(origin_row)
                limiting_types = []
                for candidate_ordinal, candidate_result in enumerate(
                        boundary_scale_info["candidate_results"]):
                    direction = direction_records[candidate_ordinal]
                    last_info = candidate_result["info"]
                    limiting_types.append(str(
                        last_info.get("limiting_constraint_type", "unknown")
                    ))
                    candidate_trials = tuple(candidate_result["trial_trace"])
                    for trial in candidate_trials:
                        row = dict(shared_failure_fields)
                        row.update(trial)
                        row.update({
                            "candidate_ordinal": int(candidate_ordinal),
                            "direction_kind": str(
                                direction["direction_kind"]
                            ),
                            "progress_floor_fraction": float(
                                direction["progress_floor_fraction"]
                            ),
                            "direction_norm": float(
                                direction["direction_norm"]
                            ),
                            "cosine_vs_full": float(
                                direction["cosine_vs_full"]
                            ),
                            "parameter_displacement_norm": (
                                float(trial["scale"])
                                * float(direction["direction_norm"])
                            ),
                            "predicted_exact_score_min_change": (
                                float(trial["scale"])
                                * float(direction["all_target_dots"][
                                    :len(pair_target_score_grads)
                                ].min().detach().cpu().item())
                            ),
                            "predicted_boundary_min_change": (
                                float(trial["scale"])
                                * float(direction["all_target_dots"][
                                    len(pair_target_score_grads):
                                    len(pair_target_score_grads)
                                    + len(pair_boundary_target_grads)
                                ].min().detach().cpu().item())
                            ),
                            "valid": int(bool(trial["valid"])),
                            "boundary_valid": (
                                None
                                if trial["boundary_valid"] is None
                                else int(bool(trial["boundary_valid"]))
                            ),
                            "exact_score_valid": (
                                None
                                if trial["exact_score_valid"] is None
                                else int(bool(trial["exact_score_valid"]))
                            ),
                            "candidate_valid": (
                                None
                                if trial["candidate_valid"] is None
                                else int(bool(trial["candidate_valid"]))
                            ),
                            "lifecycle_valid": (
                                None
                                if trial["lifecycle_valid"] is None
                                else int(bool(trial["lifecycle_valid"]))
                            ),
                            "progress_target_present": (
                                None
                                if trial["progress_target_present"] is None
                                else int(bool(
                                    trial["progress_target_present"]
                                ))
                            ),
                            "boundary_failed_target_ordinals": "|".join(
                                str(int(value)) for value in trial[
                                    "boundary_failed_target_ordinals"
                                ]
                            ),
                            "competitor_candidate_indices": "|".join(
                                str(value) for value in trial[
                                    "competitor_candidate_indices"
                                ]
                            ),
                            "target_ranks": "|".join(
                                str(value) for value in trial["target_ranks"]
                            ),
                            "target_active": "|".join(
                                str(value) for value in trial["target_active"]
                            ),
                        })
                        failure_rows.append(row)
                error = StrictPairExactInfeasibleError(
                    "strict-pair exact revalidation remained infeasible after "
                    "bounded backtracking across sampled nonlinear progress "
                    "directions: classification={}, "
                    "path={}, targets={}, origin_preservation_valid={}, "
                    "candidate_count={}, probe_valid_count={}, "
                    "origin_contracts=boundary:{}|exact:{}|candidate:{}|"
                    "lifecycle:{}, origin_minima=boundary:{}|exact:{}|"
                    "candidate:{}|lifecycle:{}, "
                    "last_limiters={}".format(
                        shared_failure_fields["failure_classification"],
                        shared_failure_fields["optimizer_kind"],
                        shared_failure_fields[
                            "target_canonical_identities"
                        ],
                        shared_failure_fields[
                            "origin_preservation_valid"
                        ],
                        shared_failure_fields["candidate_count"],
                        shared_failure_fields[
                            "diagnostic_probe_valid_count"
                        ],
                        int(bool(origin_preservation["boundary_valid"])),
                        int(bool(origin_preservation["exact_score_valid"])),
                        int(bool(origin_preservation["candidate_valid"])),
                        int(bool(origin_preservation["lifecycle_valid"])),
                        float(
                            origin_info["boundary_min_signed_change"]
                            .detach().cpu().item()
                        ),
                        (
                            None
                            if origin_info[
                                "exact_score_min_signed_change"
                            ] is None
                            else float(origin_info[
                                "exact_score_min_signed_change"
                            ].detach().cpu().item())
                        ),
                        (
                            None
                            if origin_info["candidate_loss_change"] is None
                            else float(origin_info["candidate_loss_change"]
                                       .detach().cpu().item())
                        ),
                        (
                            None
                            if origin_info["lifecycle_min_signed_gap"] is None
                            else float(origin_info[
                                "lifecycle_min_signed_gap"
                            ].detach().cpu().item())
                        ),
                        "|".join(limiting_types),
                    ),
                    diagnostic_rows=failure_rows,
                )
                _rollback_failed_pair_transaction()
                raise error
            selected_direction_ordinal = int(
                boundary_scale_info["selected_candidate_ordinal"]
            )
            selected_direction = direction_records[
                selected_direction_ordinal
            ]
            target_safe_descent = selected_direction["safe_descent"]
            target_actual_projection_info = selected_direction[
                "projection_info"
            ]
            actual_all_target_dots_after = selected_direction[
                "all_target_dots"
            ]
            boundary_identity_group_dots_after = selected_direction[
                "group_dots"
            ]
            boundary_final_displacements = list(
                selected_direction["displacements"]
            )
            pair_target_actual_min_descent_dot_after = (
                actual_all_target_dots_after[
                    :len(pair_target_score_grads)
                ].min()
            )
            pair_boundary_actual_min_descent_dot_after = (
                actual_all_target_dots_after[
                    len(pair_target_score_grads):
                ].min()
            )
            boundary_backtrack_scale = float(
                boundary_scale_info["final_scale"]
            )
            joint_exact_info = boundary_scale_info["info"]
            joint_exact_valid = bool(boundary_scale_info["valid"])
            pair_boundary_joint_candidate_exact_valid = loss.new_tensor(
                float(joint_exact_info["candidate_valid"])
            )
            pair_boundary_joint_lifecycle_exact_valid = loss.new_tensor(
                float(joint_exact_info["lifecycle_valid"])
            )
            pair_boundary_nonlinear_backtrack_count = loss.new_tensor(
                float(boundary_scale_info["halving_count"])
            )
            pair_boundary_nonlinear_refinement_count = loss.new_tensor(
                float(boundary_scale_info["refinement_count"])
            )
            pair_boundary_nonlinear_backtrack_final_scale = loss.new_tensor(
                boundary_backtrack_scale
            )
            pair_boundary_nonlinear_invalid_upper_scale = loss.new_tensor(
                float(boundary_scale_info["invalid_upper_scale"])
            )
            pair_boundary_direction_candidate_count = loss.new_tensor(
                float(boundary_scale_info["candidate_count"])
            )
            pair_boundary_direction_valid_candidate_count = loss.new_tensor(
                float(boundary_scale_info["valid_candidate_count"])
            )
            pair_boundary_selected_progress_floor_fraction = loss.new_tensor(
                float(selected_direction["progress_floor_fraction"])
            )
            pair_boundary_progress_min_completion = loss.new_tensor(
                float(joint_exact_info["progress_min_completion"])
            )
            pair_boundary_progress_mean_completion = loss.new_tensor(
                float(joint_exact_info["progress_mean_completion"])
            )
            invalid_upper_info = boundary_scale_info.get(
                "invalid_upper_info"
            )
            if invalid_upper_info is not None:
                limiting_type_code = {
                    "none": 0.0,
                    "boundary": 1.0,
                    "exact_score": 2.0,
                    "candidate": 3.0,
                    "lifecycle": 4.0,
                    "selected_factor_inactive": 5.0,
                    "candidate_catalog_changed": 6.0,
                    "lifecycle_catalog_changed": 7.0,
                }.get(
                    invalid_upper_info["limiting_constraint_type"],
                    -1.0,
                )
                pair_boundary_limiting_constraint_code = loss.new_tensor(
                    limiting_type_code
                )
                pair_boundary_limiting_target_ordinal = loss.new_tensor(
                    float(invalid_upper_info["limiting_target_ordinal"])
                )
            candidate_results = tuple(
                boundary_scale_info.get("candidate_results", ())
            )
            if len(candidate_results) != len(direction_records):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "progress direction candidate diagnostics are incomplete"
                )
            limiting_codes = {
                "none": 0,
                "boundary": 1,
                "exact_score": 2,
                "candidate": 3,
                "lifecycle": 4,
                "selected_factor_inactive": 5,
                "candidate_catalog_changed": 6,
                "lifecycle_catalog_changed": 7,
            }
            for record, candidate_result in zip(
                    direction_records, candidate_results):
                direction_kind = (
                    validate_pair_direction_candidate_seed_contract(
                        direction_kind=record["direction_kind"],
                        seed_member_ordinals=record[
                            "progress_seed_member_ordinals"
                        ],
                        zero_budget_excluded_ordinals=record[
                            "progress_seed_zero_budget_excluded_ordinals"
                        ],
                    )
                )
                candidate_info = candidate_result["info"]
                candidate_invalid_info = candidate_result.get(
                    "invalid_upper_info"
                )
                limiter_info = (
                    candidate_invalid_info
                    if candidate_invalid_info is not None
                    else candidate_info
                )
                projection_info = record["projection_info"]
                pair_direction_candidate_trace_rows.append({
                    "diagnostic_version": int(
                        PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION
                    ),
                    "candidate_ordinal": int(
                        candidate_result["candidate_ordinal"]
                    ),
                    "direction_kind": direction_kind,
                    "progress_floor_fraction": float(
                        record["progress_floor_fraction"]
                    ),
                    "direction_norm": float(record["direction_norm"]),
                    "cosine_vs_full": float(record["cosine_vs_full"]),
                    "progress_seed_member_count": int(
                        record["progress_seed_member_count"]
                    ),
                    "progress_seed_member_ordinals": "|".join(
                        str(int(index)) for index in record[
                            "progress_seed_member_ordinals"
                        ]
                    ),
                    "progress_seed_zero_budget_excluded_count": int(
                        record[
                            "progress_seed_zero_budget_excluded_count"
                        ]
                    ),
                    "progress_seed_zero_budget_excluded_ordinals": "|".join(
                        str(int(index)) for index in record[
                            "progress_seed_zero_budget_excluded_ordinals"
                        ]
                    ),
                    "progress_seed_raw_norm": float(
                        record["progress_seed_raw_norm"]
                    ),
                    "adam_reference_norm": float(
                        record["adam_reference_norm"]
                    ),
                    "progress_component_norm": float(
                        record["progress_component_norm"]
                    ),
                    "active_constraint_count": int(round(float(
                        projection_info["active_constraint_count"]
                    ))),
                    "active_constraint_ordinals": "|".join(
                        str(int(index))
                        for index in projection_info[
                            "active_constraint_indices"
                        ]
                    ),
                    "valid": int(bool(candidate_result["valid"])),
                    "halving_count": int(
                        candidate_result["halving_count"]
                    ),
                    "expansion_count": int(
                        candidate_result["expansion_count"]
                    ),
                    "refinement_count": int(
                        candidate_result["refinement_count"]
                    ),
                    "safe_lower_scale": float(
                        candidate_result["final_scale"]
                    ),
                    "safe_frontier_scale": float(
                        candidate_result["safe_frontier_scale"]
                    ),
                    "unsafe_upper_scale": float(
                        candidate_result["invalid_upper_scale"]
                    ),
                    "unsafe_upper_present": int(bool(
                        candidate_result["unsafe_upper_present"]
                    )),
                    "scale_limit": float(
                        candidate_result["scale_limit"]
                    ),
                    "progress_min_completion": float(
                        candidate_info["progress_min_completion"]
                    ),
                    "progress_mean_completion": float(
                        candidate_info["progress_mean_completion"]
                    ),
                    "progress_target_present": int(bool(
                        candidate_info["progress_target_present"]
                    )),
                    "progress_worst_actual": (
                        None if candidate_info["progress_worst_actual"] is None
                        else float(candidate_info["progress_worst_actual"])
                    ),
                    "progress_worst_required": (
                        None
                        if candidate_info["progress_worst_required"] is None
                        else float(candidate_info["progress_worst_required"])
                    ),
                    "limiting_constraint_code": int(limiting_codes.get(
                        limiter_info.get(
                            "limiting_constraint_type", "none"
                        ),
                        -1,
                    )),
                    "limiting_target_ordinal": int(limiter_info.get(
                        "limiting_target_ordinal", -1
                    )),
                    "selected": int(
                        int(candidate_result["candidate_ordinal"])
                        == selected_direction_ordinal
                    ),
                })
            pair_boundary_linearized_allocation_info[
                "selected_progress_floor_fraction"
            ] = float(selected_direction["progress_floor_fraction"])
            pair_boundary_linearized_allocation_info[
                "progress_min_completion"
            ] = float(joint_exact_info["progress_min_completion"])
            pair_boundary_linearized_allocation_info[
                "progress_mean_completion"
            ] = float(joint_exact_info["progress_mean_completion"])
            pair_boundary_linearized_allocation_info[
                "limiting_constraint_code"
            ] = float(pair_boundary_limiting_constraint_code.item())
            pair_boundary_linearized_allocation_info[
                "limiting_target_ordinal"
            ] = float(pair_boundary_limiting_target_ordinal.item())
            if not joint_exact_valid:
                boundary_min = float(
                    joint_exact_info["boundary_min_signed_change"]
                    .detach().cpu().item()
                )
                exact_score_min = joint_exact_info[
                    "exact_score_min_signed_change"
                ]
                exact_score_min = (
                    float(exact_score_min.detach().cpu().item())
                    if exact_score_min is not None else float("nan")
                )
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "{} strict-pair boundary/candidate/lifecycle exact "
                    "revalidation remained infeasible after bounded "
                    "backtracking: scale={:.9g}, boundary_valid={}, "
                    "boundary_min={:.9g}, exact_score_valid={}, "
                    "exact_score_min={:.9g}, candidate_valid={}, "
                    "lifecycle_valid={}, lifecycle_violations={}".format(
                        "pending"
                        if bool(pair_only_objective)
                        else "standard",
                        boundary_backtrack_scale,
                        int(bool(joint_exact_info["boundary_valid"])),
                        boundary_min,
                        int(bool(joint_exact_info["exact_score_valid"])),
                        exact_score_min,
                        int(bool(joint_exact_info["candidate_valid"])),
                        int(bool(joint_exact_info["lifecycle_valid"])),
                        int(joint_exact_info["lifecycle_violation_count"]),
                    )
                )
            pair_correction_applied = bool(
                (pair_actual_update_direction_guard_applied > 0.0).item()
                or (
                    pair_target_actual_direction_guard_applied > 0.0
                ).item()
                or bool(
                    (pair_boundary_nonlinear_backtrack_count > 0.0).item()
                )
                or abs(boundary_backtrack_scale - 1.0) > 1.0e-12
                or int(boundary_scale_info["selected_candidate_ordinal"]) != 0
            )
            if pair_correction_applied:
                if pair_exact_raw_parameter_deltas is None:
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "pair Adam guard is missing complete raw displacement"
                    )
                if not bool(pair_only_objective):
                    if pair_exact_optimizer_state_after_raw_step is None:
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "standard pair guard is missing complete raw Adam "
                            "state"
                        )
                    for parameter_ordinal, (parameter, raw_state) in enumerate(zip(
                            self.adj_parameters,
                            pair_exact_optimizer_state_after_raw_step)):
                        if raw_state is None:
                            origin_parameter = (
                                pair_transaction_all_parameters_before_step[
                                    parameter_ordinal
                                ]
                            )
                            if not torch.equal(
                                    parameter.detach(), origin_parameter):
                                _rollback_failed_pair_transaction()
                                raise RuntimeError(
                                    "strict-pair exact projection changed a "
                                    "parameter without initialized Adam state"
                                )
                            continue
                        state = active_adj_optimizer.state[parameter]
                        state.clear()
                        state.update(copy.deepcopy(raw_state))
                if bool(pair_only_objective):
                    exact_optimizer_parameter_before_step = (
                        pair_parameter_before_step
                    )
                    exact_optimizer_raw_parameter_deltas = (
                        pair_guard_raw_parameter_deltas
                    )
                else:
                    exact_optimizer_parameter_before_step = [
                        (
                            before if raw_state is not None else None
                        )
                        for before, raw_state in zip(
                            pair_transaction_all_parameters_before_step,
                            pair_exact_optimizer_state_after_raw_step,
                        )
                    ]
                    exact_optimizer_raw_parameter_deltas = [
                        (
                            raw_delta if before is not None else None
                        )
                        for before, raw_delta in zip(
                            exact_optimizer_parameter_before_step,
                            pair_exact_raw_parameter_deltas,
                        )
                    ]
                pair_sync_info = _sync_adam_first_moment_to_executed_update(
                    optimizer=active_adj_optimizer,
                    parameters=self.adj_parameters,
                    parameter_before_step=(
                        exact_optimizer_parameter_before_step
                    ),
                    raw_parameter_deltas=(
                        exact_optimizer_raw_parameter_deltas
                    ),
                )
                if (
                        pair_sync_info["raw_reconstruction_error_ratio"] > 1.0
                        or pair_sync_info[
                            "safe_reconstruction_error_ratio"
                        ] > 1.0):
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "pair Adam state does not reconstruct raw and guarded "
                        "updates"
                    )
                pair_optimizer_state_sync_applied = loss.new_tensor(1.0)
                if bool(
                        (
                            pair_target_actual_direction_guard_applied > 0.0
                        ).item()):
                    pair_target_optimizer_state_sync_applied = (
                        loss.new_tensor(1.0)
                    )
            final_pair_update_dot = loss.new_tensor(0.0)
            for parameter, before, pair_grad in zip(
                    self.adj_parameters,
                    pair_parameter_before_step,
                    pair_grads):
                if before is not None and pair_grad is not None:
                    final_pair_update_dot = (
                        final_pair_update_dot
                        + (
                            (parameter.detach() - before) * pair_grad
                        ).sum()
                    )
            final_pair_descent_dot = -final_pair_update_dot
            if not bool(torch.isfinite(final_pair_descent_dot).item()):
                _rollback_failed_pair_transaction()
                raise FloatingPointError(
                    "non-finite final pair Adam descent"
                )
            if bool((final_pair_descent_dot == 0.0).item()):
                _raise_recoverable_pair_optimizer_noop(
                    "zero_final_pair_descent",
                    {
                        "pair_update_dot": float(
                            final_pair_update_dot.detach().cpu().item()
                        ),
                        "pair_update_norm_sq": float(
                            pair_update_norm_sq.detach().cpu().item()
                        ),
                        "pair_gradient_norm_sq": float(
                            pair_gradient_norm_sq.detach().cpu().item()
                        ),
                        "clipped_pair_dot": float(
                            clipped_pair_dot.detach().cpu().item()
                        ),
                        "clipped_gradient_norm_sq": float(
                            clipped_gradient_norm_sq.detach().cpu().item()
                        ),
                    },
                )
            if bool((final_pair_descent_dot < 0.0).item()):
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "pair Adam update reversed the strict descent direction"
                )
            if candidate_objective_active:
                final_candidate_update_dot = loss.new_tensor(0.0)
                for parameter, before, candidate_grad in zip(
                        self.adj_parameters,
                        candidate_parameter_before_step,
                        candidate_grads):
                    if before is not None and candidate_grad is not None:
                        final_candidate_update_dot = (
                            final_candidate_update_dot
                            + (
                                (parameter.detach() - before)
                                * candidate_grad
                            ).sum()
                        )
                candidate_actual_update_descent_dot_after = (
                    -final_candidate_update_dot
                )
                if not bool(torch.isfinite(
                        candidate_actual_update_descent_dot_after).item()):
                    _rollback_failed_pair_transaction()
                    raise FloatingPointError(
                        "non-finite final pair/candidate descent"
                    )
                if bool((
                        candidate_actual_update_descent_dot_after == 0.0
                ).item()):
                    _raise_recoverable_pair_optimizer_noop(
                        "zero_final_candidate_descent",
                        {
                            "pair_update_dot": float(
                                final_candidate_update_dot
                                .detach().cpu().item()
                            ),
                            "pair_update_norm_sq": float(
                                pair_update_norm_sq.detach().cpu().item()
                            ),
                            "pair_gradient_norm_sq": float(
                                candidate_norm_sq.detach().cpu().item()
                            ),
                            "clipped_pair_dot": float(
                                candidate_clipped_gradient_dot
                                .detach().cpu().item()
                            ),
                            "clipped_gradient_norm_sq": float(
                                clipped_combined_norm_sq
                                .detach().cpu().item()
                            ),
                        },
                    )
                if bool((
                        candidate_actual_update_descent_dot_after < 0.0
                ).item()):
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "pair Adam guard conflicts with current candidate "
                        "descent"
                    )
            if lifecycle_target_present:
                final_effective_gradient = tuple(
                    (
                        None
                        if before is None
                        else before - parameter.detach()
                    )
                    for parameter, before in zip(
                        self.adj_parameters,
                        pair_parameter_before_step,
                    )
                )
                lifecycle_final_linear_dots = []
                lifecycle_final_linear_tolerances = []
                for lifecycle_constraint_grad in lifecycle_constraint_grads:
                    lifecycle_final_validation = (
                        _nonnegative_gradient_dot_with_tolerance(
                            proposed_grads=final_effective_gradient,
                            constraint_grads=lifecycle_constraint_grad,
                            reference=loss,
                        )
                    )
                    lifecycle_final_linear_dots.append(
                        lifecycle_final_validation["dot"]
                    )
                    lifecycle_final_linear_tolerances.append(
                        lifecycle_final_validation["tolerance"]
                    )
                    if not bool(
                            lifecycle_final_validation["valid"].item()):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "pair Adam guard conflicts with an accepted "
                            "candidate lifecycle constraint: dot={:.9g}, "
                            "tolerance={:.9g}".format(
                                float(
                                    lifecycle_final_validation["dot"]
                                    .detach().cpu().item()
                                ),
                                float(
                                    lifecycle_final_validation["tolerance"]
                                    .detach().cpu().item()
                                ),
                            )
                        )
                if lifecycle_final_linear_dots:
                    lifecycle_final_linear_dot_tensor = torch.stack(
                        lifecycle_final_linear_dots
                    )
                    lifecycle_final_linear_tolerance_tensor = torch.stack(
                        lifecycle_final_linear_tolerances
                    )
                    lifecycle_final_linear_min_dot = (
                        lifecycle_final_linear_dot_tensor.min()
                    )
                    lifecycle_final_linear_max_tolerance = (
                        lifecycle_final_linear_tolerance_tensor.max()
                    )
                    lifecycle_final_linear_rounding_residual_count = (
                        (
                            (lifecycle_final_linear_dot_tensor < 0.0)
                            & (
                                lifecycle_final_linear_dot_tensor
                                >= -lifecycle_final_linear_tolerance_tensor
                            )
                        ).float().sum()
                    )
                    lifecycle_final_linear_normalized = torch.where(
                        lifecycle_final_linear_tolerance_tensor > 0.0,
                        (
                            -lifecycle_final_linear_dot_tensor
                            / lifecycle_final_linear_tolerance_tensor
                        ).clamp_min(0.0),
                        torch.zeros_like(
                            lifecycle_final_linear_tolerance_tensor
                        ),
                    )
                    lifecycle_final_linear_max_normalized_violation = (
                        lifecycle_final_linear_normalized.max()
                    )
                # The norm-scaled tolerance applies only to the float32
                # first-order dot. Re-evaluate the real signed lifecycle
                # margins after the pair guard so a material nonlinear
                # regression can never be accepted as writeback noise.
                with torch.no_grad():
                    (
                        pair_final_lifecycle_margins,
                        pair_final_lifecycle_valid,
                    ) = (
                        self.adj_network
                        .evaluate_candidate_identity_active_competitor_margins(
                            rnn_obs=lifecycle_rnn_obs,
                            dones=lifecycle_dones.bool(),
                            adj=lifecycle_adj,
                        )
                    )
                    if not bool(torch.equal(
                            pair_final_lifecycle_valid > 0.0,
                            lifecycle_valid_mask > 0.0,
                    )):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "candidate lifecycle catalog changed after pair "
                            "Adam guard"
                        )
                    pair_final_lifecycle_tolerance = (
                        16.0
                        * lifecycle_dtype_epsilon
                        * (
                            lifecycle_signed_floor.abs()
                            + (
                                torch.sign(lifecycle_delta)
                                * pair_final_lifecycle_margins
                            ).abs()
                        ).clamp_min(1.0e-12)
                    )
                    pair_final_lifecycle_violation = (
                        lifecycle_target_mask
                        & (
                            torch.sign(lifecycle_delta)
                            * pair_final_lifecycle_margins
                            < (
                                lifecycle_signed_floor
                                - pair_final_lifecycle_tolerance
                            )
                        )
                    )
                    lifecycle_final_exact_signed_gap = torch.where(
                        lifecycle_target_mask,
                        (
                            torch.sign(lifecycle_delta)
                            * pair_final_lifecycle_margins
                            - lifecycle_signed_floor
                        ),
                        torch.full_like(
                            pair_final_lifecycle_margins,
                            float("inf"),
                        ),
                    )
                    lifecycle_final_exact_min_signed_gap = (
                        lifecycle_final_exact_signed_gap.min()
                    )
                    lifecycle_final_exact_max_tolerance = (
                        pair_final_lifecycle_tolerance[
                            lifecycle_target_mask
                        ].max()
                    )
                    pair_final_lifecycle_violation_count = (
                        pair_final_lifecycle_violation.any(dim=1)
                        .float().sum()
                    )
                    if bool(
                            (
                                pair_final_lifecycle_violation_count > 0.0
                            ).item()):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "pair Adam guard violates accepted exact candidate "
                            "lifecycle margins: transition_count={:.0f}"
                            .format(float(
                                pair_final_lifecycle_violation_count
                                .detach().cpu().item()
                            ))
                        )
                    lifecycle_final_exact_revalidation_valid = (
                        loss.new_tensor(1.0)
                    )

        if pair_boundary_retention_constraint_grads:
            if pair_transaction_all_parameters_before_step is None:
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "selection-boundary retention has no atomic parameter "
                    "snapshot"
                )
            try:
                retention_parameter_before_step = (
                    pair_transaction_all_parameters_before_step
                )
                retention_unprotected_parameter_deltas = [
                    (parameter.detach() - before).detach().clone()
                    for parameter, before in zip(
                        self.adj_parameters,
                        retention_parameter_before_step,
                    )
                ]
                retention_optimizer_state_after_unprotected_step = [
                    copy.deepcopy(
                        active_adj_optimizer.state.get(parameter, {})
                    )
                    for parameter in self.adj_parameters
                ]
                retention_actual_descent = [
                    -delta for delta in retention_unprotected_parameter_deltas
                ]
                retention_priority_grads = None
                retention_additional_priority_grads = ()
                if candidate_objective_active:
                    retention_priority_grads = candidate_grads
                    if pair_diagnostic_target_present:
                        retention_additional_priority_grads = (
                            (pair_grads,)
                            + tuple(pair_target_score_grads)
                            + tuple(pair_boundary_target_grads)
                        )
                elif pair_diagnostic_target_present:
                    retention_priority_grads = pair_grads
                    retention_additional_priority_grads = (
                        tuple(pair_target_score_grads)
                        + tuple(pair_boundary_target_grads)
                    )
                if retention_priority_grads is not None:
                    (
                        retention_safe_descent,
                        retention_actual_enforced_indices,
                        retention_actual_superseded_indices,
                        _retention_actual_projection_info,
                    ) = _project_with_current_candidate_priority(
                        proposed_grads=retention_actual_descent,
                        candidate_grads=retention_priority_grads,
                        lifecycle_constraint_grads=(
                            pair_boundary_retention_constraint_grads
                        ),
                        reference=loss,
                        additional_priority_grads=(
                            retention_additional_priority_grads
                        ),
                    )
                else:
                    (
                        retention_safe_descent,
                        _retention_actual_projection_info,
                    ) = _project_gradients_onto_nonincreasing_halfspaces(
                        proposed_grads=retention_actual_descent,
                        constraint_grads=(
                            pair_boundary_retention_constraint_grads
                        ),
                        reference=loss,
                    )
                    retention_actual_enforced_indices = list(range(len(
                        pair_boundary_retention_constraint_grads
                    )))
                    retention_actual_superseded_indices = []
                for index in retention_actual_superseded_indices:
                    pair_boundary_retention_superseded_keys.add(
                        pair_boundary_retention_keyed_entries[index][0]
                    )
                pair_boundary_retention_keyed_entries = [
                    pair_boundary_retention_keyed_entries[index]
                    for index in retention_actual_enforced_indices
                ]
                pair_boundary_retention_constraint_grads = [
                    pair_boundary_retention_constraint_grads[index]
                    for index in retention_actual_enforced_indices
                ]
                pair_boundary_retention_protected_target_count = (
                    loss.new_tensor(float(len(
                        pair_boundary_retention_constraint_grads
                    )))
                )
                retention_projected_parameter_deltas = [
                    -descent if descent is not None else None
                    for descent in retention_safe_descent
                ]

                def _nonpair_current_objectives_exact_valid():
                    candidate_valid = True
                    lifecycle_valid = True
                    with torch.no_grad():
                        if candidate_objective_active:
                            candidate_trial_margins, candidate_trial_valid = (
                                self.adj_network
                                .evaluate_candidate_identity_active_competitor_margins(
                                    rnn_obs=rnn_obs,
                                    dones=dones.bool(),
                                    adj=adj,
                                )
                            )
                            if not bool(torch.equal(
                                    candidate_trial_valid > 0.0,
                                    candidate_identity_valid_mask > 0.0,
                            )):
                                raise RuntimeError(
                                    "candidate catalog changed during "
                                    "selection-boundary retention"
                                )
                            candidate_trial_info = (
                                compute_capture_candidate_identity_active_competitor_loss(
                                    candidate_competitor_margins=(
                                        candidate_trial_margins
                                    ),
                                    candidate_reference_margins=(
                                        behavior_candidate_margin
                                    ),
                                    candidate_identity_delta=(
                                        candidate_identity_delta[..., 0]
                                    ),
                                    candidate_valid_mask=candidate_trial_valid,
                                    transition_mask=transition_mask,
                                )
                            )
                            candidate_valid = bool((
                                candidate_trial_info["loss"]
                                < candidate_identity_loss_info[
                                    "loss"
                                ].detach()
                            ).item())
                        if lifecycle_target_present:
                            lifecycle_trial_margins, lifecycle_trial_valid = (
                                self.adj_network
                                .evaluate_candidate_identity_active_competitor_margins(
                                    rnn_obs=lifecycle_rnn_obs,
                                    dones=lifecycle_dones.bool(),
                                    adj=lifecycle_adj,
                                )
                            )
                            if not bool(torch.equal(
                                    lifecycle_trial_valid > 0.0,
                                    lifecycle_valid_mask > 0.0,
                            )):
                                raise RuntimeError(
                                    "candidate lifecycle catalog changed "
                                    "during selection-boundary retention"
                                )
                            lifecycle_trial_signed_margin = (
                                torch.sign(lifecycle_delta)
                                * lifecycle_trial_margins
                            )
                            lifecycle_trial_tolerance = (
                                16.0
                                * lifecycle_dtype_epsilon
                                * (
                                    lifecycle_signed_floor.abs()
                                    + lifecycle_trial_signed_margin.abs()
                                ).clamp_min(1.0e-12)
                            )
                            lifecycle_trial_violation = (
                                lifecycle_target_mask
                                & (
                                    lifecycle_trial_signed_margin
                                    < lifecycle_signed_floor
                                    - lifecycle_trial_tolerance
                                )
                            )
                            lifecycle_valid = not bool(
                                torch.any(lifecycle_trial_violation).item()
                            )
                    return candidate_valid and lifecycle_valid

                accepted_scale = None
                accepted_retention = None
                for backtrack_index in range(25):
                    trial_scale = 0.5 ** backtrack_index
                    if pair_diagnostic_target_present:
                        current_objective_acceptance = (
                            _evaluate_joint_pair_exact_scale(
                                boundary_trial_scale=trial_scale,
                                trial_displacements=(
                                    retention_projected_parameter_deltas
                                ),
                                trial_parameter_before=(
                                    retention_parameter_before_step
                                ),
                            )
                        )
                        current_objectives_valid = bool(
                            current_objective_acceptance["valid"]
                        )
                    else:
                        with torch.no_grad():
                            for parameter, before, trial_delta in zip(
                                    self.adj_parameters,
                                    retention_parameter_before_step,
                                    retention_projected_parameter_deltas):
                                if trial_delta is not None:
                                    parameter.copy_(
                                        before + trial_scale * trial_delta
                                    )
                        current_objectives_valid = (
                            _nonpair_current_objectives_exact_valid()
                        )
                    retention_acceptance = (
                        self
                        ._pair_selection_boundary_retention_exact_acceptance(
                            pair_boundary_retention_keyed_entries
                        )
                    )
                    pair_boundary_retention_stop_keys.update(
                        retention_acceptance["context_invalid_keys"]
                    )
                    if (
                            current_objectives_valid
                            and bool(retention_acceptance["valid"])):
                        accepted_scale = trial_scale
                        accepted_retention = retention_acceptance
                        pair_boundary_retention_nonlinear_backtrack_count = (
                            loss.new_tensor(float(backtrack_index))
                        )
                        break
                if accepted_scale is None or accepted_retention is None:
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "selection-boundary retention has no finite exact "
                        "writeback compatible with current objectives"
                    )
                pair_boundary_retention_final_scale = loss.new_tensor(
                    float(accepted_scale)
                )
                if accepted_retention["min_signed_gap"] is not None:
                    pair_boundary_retention_final_exact_min_signed_gap = (
                        accepted_retention["min_signed_gap"].detach()
                    )
                    pair_boundary_retention_final_exact_max_tolerance = (
                        accepted_retention["max_tolerance"].detach()
                    )
                retention_correction_norm_sq = loss.new_tensor(0.0)
                for parameter, before, raw_delta in zip(
                        self.adj_parameters,
                        retention_parameter_before_step,
                        retention_unprotected_parameter_deltas):
                    executed_delta = parameter.detach() - before
                    retention_correction_norm_sq = (
                        retention_correction_norm_sq
                        + (executed_delta - raw_delta).pow(2).sum()
                    )
                if bool((retention_correction_norm_sq > 0.0).item()):
                    pair_boundary_retention_actual_projection_corrected = (
                        loss.new_tensor(1.0)
                    )
                    for parameter, raw_state in zip(
                            self.adj_parameters,
                            retention_optimizer_state_after_unprotected_step):
                        state = active_adj_optimizer.state[parameter]
                        state.clear()
                        state.update(copy.deepcopy(raw_state))
                    retention_sync_info = (
                        _sync_adam_first_moment_to_executed_update(
                            optimizer=active_adj_optimizer,
                            parameters=self.adj_parameters,
                            parameter_before_step=(
                                retention_parameter_before_step
                            ),
                            raw_parameter_deltas=(
                                retention_unprotected_parameter_deltas
                            ),
                        )
                    )
                    if (
                            retention_sync_info[
                                "raw_reconstruction_error_ratio"
                            ] > 1.0
                            or retention_sync_info[
                                "safe_reconstruction_error_ratio"
                            ] > 1.0):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "selection-boundary retention Adam state does "
                            "not reconstruct raw and retained updates"
                        )
                    pair_boundary_retention_optimizer_state_sync_applied = (
                        loss.new_tensor(1.0)
                    )
            except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError):
                _rollback_failed_pair_transaction()
                raise

        # The exact acceptance above belongs to the retention correction
        # sub-stage.  Run130 demonstrated that accepting that intermediate
        # writeback is not a sufficient commit contract: its first ordinary
        # transaction (seq694) returned successfully, yet both exact saved
        # contexts were already below their floors in the age-1 rows.  Keep an
        # immutable list of the entries seen at transaction start and validate
        # them again after every parameter-writing guard has completed.  Only
        # explicitly invalidated or current-evidence-superseded entries may be
        # absent from this final postcondition.
        pair_boundary_retention_final_required_entries = [
            keyed_entry
            for keyed_entry in pair_boundary_retention_initial_keyed_entries
            if (
                keyed_entry[0] not in pair_boundary_retention_stop_keys
                and keyed_entry[0]
                not in pair_boundary_retention_superseded_keys
            )
        ]
        if pair_boundary_retention_final_required_entries:
            pair_boundary_retention_final_postcondition_entered = (
                loss.new_tensor(1.0)
            )
            pair_boundary_retention_final_postcondition_target_count = (
                loss.new_tensor(float(len(
                    pair_boundary_retention_final_required_entries
                )))
            )
            if pair_transaction_all_parameters_before_step is None:
                _rollback_failed_pair_transaction()
                raise RuntimeError(
                    "selection-boundary retention final postcondition has no "
                    "atomic parameter snapshot"
                )
            try:
                final_retention_acceptance = (
                    self
                    ._pair_selection_boundary_retention_exact_acceptance(
                        pair_boundary_retention_final_required_entries
                    )
                )
                if final_retention_acceptance["min_signed_gap"] is not None:
                    pair_boundary_retention_final_exact_min_signed_gap = (
                        final_retention_acceptance["min_signed_gap"].detach()
                    )
                    pair_boundary_retention_final_exact_max_tolerance = (
                        final_retention_acceptance["max_tolerance"].detach()
                    )
                pair_boundary_retention_stop_keys.update(
                    final_retention_acceptance["context_invalid_keys"]
                )
                pair_boundary_retention_final_required_entries = [
                    keyed_entry
                    for keyed_entry
                    in pair_boundary_retention_final_required_entries
                    if keyed_entry[0]
                    not in pair_boundary_retention_stop_keys
                ]
                if (
                        pair_boundary_retention_final_required_entries
                        and not bool(final_retention_acceptance["valid"])):
                    # Reconstruct the complete final transaction displacement,
                    # not an earlier Adam or pair-guard intermediate.  Scaling
                    # this displacement toward the pre-step snapshot preserves
                    # the transaction's direction while finding the largest
                    # exact writeback that also retains every surviving floor.
                    late_unprotected_parameter_deltas = [
                        (parameter.detach() - before).detach().clone()
                        for parameter, before in zip(
                            self.adj_parameters,
                            pair_transaction_all_parameters_before_step,
                        )
                    ]
                    late_optimizer_state_after_unprotected_step = [
                        copy.deepcopy(
                            active_adj_optimizer.state.get(parameter, {})
                        )
                        for parameter in self.adj_parameters
                    ]
                    late_accepted_scale = None
                    late_accepted_retention = None
                    for late_backtrack_index in range(25):
                        late_trial_scale = 0.5 ** late_backtrack_index
                        if pair_diagnostic_target_present:
                            late_current_objective_acceptance = (
                                _evaluate_joint_pair_exact_scale(
                                    boundary_trial_scale=late_trial_scale,
                                    trial_displacements=(
                                        late_unprotected_parameter_deltas
                                    ),
                                    trial_parameter_before=(
                                        pair_transaction_all_parameters_before_step
                                    ),
                                )
                            )
                            late_current_objectives_valid = bool(
                                late_current_objective_acceptance["valid"]
                            )
                        else:
                            with torch.no_grad():
                                for parameter, before, late_delta in zip(
                                        self.adj_parameters,
                                        pair_transaction_all_parameters_before_step,
                                        late_unprotected_parameter_deltas):
                                    parameter.copy_(
                                        before
                                        + late_trial_scale * late_delta
                                    )
                            late_current_objectives_valid = (
                                _nonpair_current_objectives_exact_valid()
                            )
                        late_retention_acceptance = (
                            self
                            ._pair_selection_boundary_retention_exact_acceptance(
                                pair_boundary_retention_final_required_entries
                            )
                        )
                        if (
                                late_current_objectives_valid
                                and bool(late_retention_acceptance["valid"])):
                            late_accepted_scale = late_trial_scale
                            late_accepted_retention = (
                                late_retention_acceptance
                            )
                            pair_boundary_retention_stop_keys.update(
                                late_retention_acceptance[
                                    "context_invalid_keys"
                                ]
                            )
                            pair_boundary_retention_nonlinear_backtrack_count = (
                                pair_boundary_retention_nonlinear_backtrack_count
                                + loss.new_tensor(float(
                                    late_backtrack_index
                                ))
                            )
                            break
                    if (
                            late_accepted_scale is None
                            or late_accepted_retention is None):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "selection-boundary retention final transaction "
                            "postcondition has no compatible exact writeback"
                        )
                    pair_boundary_retention_final_scale = (
                        pair_boundary_retention_final_scale
                        * loss.new_tensor(float(late_accepted_scale))
                    )
                    pair_boundary_retention_actual_projection_corrected = (
                        loss.new_tensor(1.0)
                    )
                    for parameter, raw_state in zip(
                            self.adj_parameters,
                            late_optimizer_state_after_unprotected_step):
                        state = active_adj_optimizer.state[parameter]
                        state.clear()
                        state.update(copy.deepcopy(raw_state))
                    late_sync_info = (
                        _sync_adam_first_moment_to_executed_update(
                            optimizer=active_adj_optimizer,
                            parameters=self.adj_parameters,
                            parameter_before_step=(
                                pair_transaction_all_parameters_before_step
                            ),
                            raw_parameter_deltas=(
                                late_unprotected_parameter_deltas
                            ),
                        )
                    )
                    if (
                            late_sync_info[
                                "raw_reconstruction_error_ratio"
                            ] > 1.0
                            or late_sync_info[
                                "safe_reconstruction_error_ratio"
                            ] > 1.0):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "selection-boundary retention final transaction "
                            "Adam state does not reconstruct the exact writeback"
                        )
                    pair_boundary_retention_optimizer_state_sync_applied = (
                        loss.new_tensor(1.0)
                    )
                    if late_accepted_retention["min_signed_gap"] is not None:
                        pair_boundary_retention_final_exact_min_signed_gap = (
                            late_accepted_retention["min_signed_gap"].detach()
                        )
                        pair_boundary_retention_final_exact_max_tolerance = (
                            late_accepted_retention["max_tolerance"].detach()
                        )

                pair_boundary_retention_final_required_entries = [
                    keyed_entry
                    for keyed_entry
                    in pair_boundary_retention_final_required_entries
                    if keyed_entry[0]
                    not in pair_boundary_retention_stop_keys
                ]
                final_retention_acceptance = (
                    self
                    ._pair_selection_boundary_retention_exact_acceptance(
                        pair_boundary_retention_final_required_entries
                    )
                )
                if final_retention_acceptance["min_signed_gap"] is not None:
                    pair_boundary_retention_final_exact_min_signed_gap = (
                        final_retention_acceptance["min_signed_gap"].detach()
                    )
                    pair_boundary_retention_final_exact_max_tolerance = (
                        final_retention_acceptance["max_tolerance"].detach()
                    )
                if (
                        final_retention_acceptance["context_invalid_keys"]
                        or not bool(final_retention_acceptance["valid"])):
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "selection-boundary retention floor failed the final "
                        "adjacency transaction postcondition"
                    )
            except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError):
                _rollback_failed_pair_transaction()
                raise
        for retention_key in (
                pair_boundary_retention_stop_keys
                | pair_boundary_retention_superseded_keys):
            retention_entry = (
                self._pair_selection_boundary_retention_observations.get(
                    retention_key
                )
            )
            if retention_entry is None:
                continue
            retention_entry["protection_stopped"] = True
            retention_entry["protection_stop_reason"] = (
                "incompatible_current_actionable_evidence"
                if retention_key in pair_boundary_retention_superseded_keys
                else "context_invalid"
            )
            retention_entry["protection_stop_clock"] = int(
                self._pair_selection_boundary_ordinary_update_clock
                + (0 if bool(pair_only_objective) else 1)
            )

        if candidate_objective_active and lifecycle_target_present:
            # Lifecycle rollback may have undone the current candidate step.
            # Recompute causal diagnostics from the final committed parameters,
            # not from the attempted pre-rollback state.
            candidate_final_actual_update_dot = loss.new_tensor(0.0)
            for parameter, before, candidate_grad in zip(
                    self.adj_parameters,
                    candidate_parameter_before_step,
                    candidate_grads):
                if before is not None and candidate_grad is not None:
                    candidate_final_actual_update_dot = (
                        candidate_final_actual_update_dot
                        + ((parameter.detach() - before) * candidate_grad).sum()
                    )
            candidate_actual_update_descent_dot_after = (
                -candidate_final_actual_update_dot
            )
            with torch.no_grad():
                final_candidate_margins, final_candidate_valid = (
                    self.adj_network.evaluate_candidate_identity_active_competitor_margins(
                        rnn_obs=rnn_obs,
                        dones=dones.bool(),
                        adj=adj,
                    )
                )
                if not bool(torch.equal(
                        final_candidate_valid > 0.0,
                        candidate_identity_valid_mask > 0.0,
                )):
                    raise RuntimeError(
                        "candidate catalog changed across lifecycle transaction"
                    )
                final_candidate_info = (
                    compute_capture_candidate_identity_active_competitor_loss(
                    candidate_competitor_margins=final_candidate_margins,
                    candidate_reference_margins=behavior_candidate_margin,
                    candidate_identity_delta=candidate_identity_delta[..., 0],
                    candidate_valid_mask=final_candidate_valid,
                    transition_mask=transition_mask,
                ))
                post_margin = final_candidate_info["margin"]
                post_rank = self.adj_network.canonical_candidate_ranks(
                    final_candidate_margins,
                    final_candidate_valid,
                ).to(final_candidate_margins.dtype)
                candidate_loss_optimizer_change = (
                    final_candidate_info["loss"]
                    - candidate_identity_loss_info["loss"].detach()
                )
                final_signed_margin_change = (
                    torch.sign(candidate_identity_delta[..., 0])
                    * (post_margin - current_margin.detach())
                )
                candidate_positive_optimizer_signed_margin_change = (
                    (
                        final_signed_margin_change
                        * candidate_optimized_positive_target
                    ).sum()
                    / candidate_optimized_positive_denominator
                )
                candidate_negative_optimizer_signed_margin_change = (
                    (
                        final_signed_margin_change
                        * candidate_optimized_negative_target
                    ).sum()
                    / candidate_optimized_negative_denominator
                )
                candidate_positive_optimizer_rank_improved_fraction = (
                    (
                        (post_rank < candidate_identity_current_rank).float()
                        * candidate_optimized_positive_target
                    ).sum()
                    / candidate_optimized_positive_denominator
                )
                candidate_negative_optimizer_rank_reduced_fraction = (
                    (
                        (post_rank > candidate_identity_current_rank).float()
                        * candidate_optimized_negative_target
                    ).sum()
                    / candidate_optimized_negative_denominator
                )

        candidate_sensitive_update_committed = not bool(
            lifecycle_update_rejected.item()
        )
        if candidate_objective_active and candidate_sensitive_update_committed:
            committed_candidate_info = (
                _select_candidate_lifecycle_committed_info(
                    post_candidate_info=post_candidate_info,
                    final_candidate_info=final_candidate_info,
                    lifecycle_target_present=lifecycle_target_present,
                )
            )
            candidate_lifecycle_behavioral_progress_mask = (
                _candidate_lifecycle_behavioral_progress_mask(
                    identity_delta=candidate_effective_delta,
                    pre_unsatisfied_mask=candidate_identity_loss_info[
                        "unsatisfied_target_mask"
                    ],
                    post_unsatisfied_mask=committed_candidate_info[
                        "unsatisfied_target_mask"
                    ],
                    pre_rank=candidate_identity_current_rank,
                    post_rank=post_rank,
                )
            )
        residual_inactive_parameter_count = loss.new_tensor(0.0)
        residual_inactive_parameter_update_norm = loss.new_tensor(0.0)
        if residual_inactive_parameter_before_step is not None:
            residual_inactive_update_sq = loss.new_tensor(0.0)
            for parameter, before in zip(
                    self.adj_parameters,
                    residual_inactive_parameter_before_step):
                if before is None:
                    continue
                residual_inactive_parameter_count = (
                    residual_inactive_parameter_count + 1.0
                )
                residual_inactive_update_sq = (
                    residual_inactive_update_sq
                    + (parameter.detach() - before).pow(2).sum()
                )
            residual_inactive_parameter_update_norm = torch.sqrt(
                residual_inactive_update_sq
            )
            if bool(
                    (residual_inactive_parameter_update_norm > 0.0).item()):
                raise RuntimeError(
                    "candidate residual moved an inactive graph parameter"
                )
        if candidate_objective_active and candidate_sensitive_update_committed:
            lifecycle_behavioral_progress_transition_count = float(
                (
                    candidate_lifecycle_behavioral_progress_mask
                    .abs()
                    .sum(dim=1)
                    > 0.0
                ).float().sum().detach().cpu().item()
            )
            lifecycle_new_count = self._register_candidate_identity_lifecycle(
                rnn_obs=rnn_obs,
                dones=dones,
                adj=adj,
                previous_adj=previous_adj,
                identity_delta=candidate_effective_delta,
                transition_mask=transition_mask,
                behavioral_progress_mask=(
                    candidate_lifecycle_behavioral_progress_mask
                ),
                behavior_policy_version=behavior_candidate_version,
                reference_margin=post_margin,
                reference_rank=post_rank,
                lifecycle_clock=current_candidate_lifecycle_clock,
            )
            lifecycle_duplicate_prevented_count = max(
                0.0,
                lifecycle_behavioral_progress_transition_count
                - float(lifecycle_new_count),
            )
            lifecycle_no_progress_skipped_transition_count = max(
                0.0,
                float(
                    candidate_identity_loss_info[
                        "unsatisfied_target_transition_count"
                    ].detach().cpu().item()
                )
                - lifecycle_behavioral_progress_transition_count,
            )
            lifecycle_cache_size = len(self._candidate_identity_lifecycle)
            lifecycle_observation_archive_size = len(
                self._candidate_identity_lifecycle_observations
            )
        if candidate_sensitive_update_committed:
            self.adj_network.candidate_policy_version = (
                current_candidate_policy_version + 1
            )
            lifecycle_policy_version_advanced = loss.new_tensor(1.0)
        else:
            self.adj_network.candidate_policy_version = (
                current_candidate_policy_version
            )
        # TTL uses optimizer-attempt units, not accepted graph-policy versions.
        # A rejected candidate-sensitive transaction must not claim a new
        # behavior policy, but it must still consume one finite lifecycle step
        # so a repeatedly conflicting cache cannot live forever.
        self.adj_network.candidate_lifecycle_clock = (
            current_candidate_lifecycle_clock
            if explicit_adj_update_round
            else current_candidate_lifecycle_clock + 1
        )
        if transaction_diagnostics_enabled:
            final_displacement_info = (
                _parameter_displacement_direction_diagnostics(
                    parameters=self.adj_parameters,
                    parameter_before_step=(
                        transaction_parameter_before_standard_adam
                    ),
                    pair_grads=pair_grads,
                    reference=loss,
                )
            )
            final_displacement_norm = final_displacement_info[
                "displacement_norm"
            ]
            final_pair_dot = final_displacement_info["pair_dot"]
            final_pair_descent_dot = final_displacement_info[
                "pair_descent_dot"
            ]
            final_pair_descent_cosine = final_displacement_info[
                "pair_descent_cosine"
            ]
            adam_to_final_info = _displacement_delta_diagnostics(
                parameters=self.adj_parameters,
                parameter_before_step=(
                    transaction_parameter_before_standard_adam
                ),
                raw_displacements=transaction_adam_raw_displacements,
                reference=loss,
            )
            adam_to_final_displacement_delta_norm = adam_to_final_info[
                "delta_norm"
            ]
            final_parameters_equal_raw_adam = adam_to_final_info[
                "exact_equal"
            ]
        if pair_diagnostic_target_present:
            pair_update_info = _pair_realized_displacement_diagnostics(
                parameters=self.adj_parameters,
                parameter_before_step=pair_parameter_before_step,
                pair_grads=pair_grads,
                reference=loss,
            )
            pair_actual_update_norm = pair_update_info["update_norm"]
            pair_actual_update_descent_dot = pair_update_info["descent_dot"]
            pair_actual_update_descent_cosine = pair_update_info[
                "descent_cosine"
            ]
            with torch.no_grad():
                # Revalidate the selection boundary and the exact PPO behavior
                # score from one shared production catalog.  Before v39 the
                # nonlinear search preserved the pure temperature-1 selected
                # policy log-probability while this postcondition checked the
                # scheduled/persistent behavior likelihood.  A candidate could
                # therefore pass every search probe and still reverse one real
                # exact target at commit (run: correct=7, reverse=1).  The
                # shared replay field makes search and commit authoritative on
                # the same score and also removes a duplicate GAT forward.
                post_pair_boundary = (
                    self.adj_network
                    .evaluate_selected_factor_replay_boundaries(
                        rnn_obs=rnn_obs,
                        dones=dones.bool(),
                        adj=adj,
                        previous_adj=previous_adj,
                        include_behavior_logp=True,
                    )
                )
                post_pair_factor_logp = post_pair_boundary[
                    "behavior_selected_logp"
                ]
                if transaction_diagnostics_enabled:
                    _validate_exact_score_join_fields(
                        pre_join_fields=pair_score_join_before,
                        post_join_fields={
                            "transition_factor_adjacency": adj.detach(),
                            "active_mask": active_masks.detach(),
                            "valid_factor_mask": valid_factor_masks.detach(),
                            "valid_adjacency": valid_adj.detach(),
                            "factor_denominator": factor_den.detach(),
                            "pair_local_delta": (
                                pair_pursuit_local_delta.detach()
                            ),
                        },
                    )
                pair_score_info = _pair_target_score_change_diagnostics(
                    pre_factor_logp=target_factor_logp.detach(),
                    post_factor_logp=post_pair_factor_logp,
                    pair_local_delta=pair_pursuit_local_delta.detach(),
                    zero_tolerance=(
                        joint_exact_preservation_tolerance
                        if joint_exact_preservation_tolerance is not None
                        else 1.0e-12
                    ),
                )
                pair_target_score_signed_change_mean = pair_score_info[
                    "signed_change_mean"
                ]
                pair_target_score_positive_change_mean = pair_score_info[
                    "positive_change_mean"
                ]
                pair_target_score_negative_signed_change_mean = (
                    pair_score_info["negative_signed_change_mean"]
                )
                pair_target_score_positive_target_count = pair_score_info[
                    "positive_target_count"
                ]
                pair_target_score_negative_target_count = pair_score_info[
                    "negative_target_count"
                ]
                pair_target_score_correct_direction_count = pair_score_info[
                    "correct_direction_count"
                ]
                pair_target_score_reverse_direction_count = pair_score_info[
                    "reverse_direction_count"
                ]
                pair_target_score_approximately_zero_count = pair_score_info[
                    "approximately_zero_count"
                ]
                pair_target_score_before_after_join_valid = pair_score_info[
                    "before_after_join_valid"
                ]
                pair_target_score_zero_tolerance = pair_score_info[
                    "zero_tolerance"
                ]
                pair_target_count = pair_score_info["target_count"]
                pair_score_postcondition_valid = (
                    _pair_target_score_nonregression_postcondition(
                        pair_score_info
                    )
                )
                if not pair_score_postcondition_valid:
                    raise RuntimeError(
                        "{} exact-pair target score guard violated its "
                        "{} contract: correct={:.0f}, reverse={:.0f}, "
                        "zero={:.0f}, total={:.0f}".format(
                            "pending"
                            if bool(pair_only_objective)
                            else "standard",
                            "all-member dtype-tolerant non-regression",
                            float(pair_score_info[
                                "correct_direction_count"
                            ].detach().cpu().item()),
                            float(pair_score_info[
                                "reverse_direction_count"
                            ].detach().cpu().item()),
                            float(pair_score_info[
                                "approximately_zero_count"
                            ].detach().cpu().item()),
                            float(pair_target_count.detach().cpu().item()),
                        )
                    )
                pair_boundary_info = (
                    _pair_selection_boundary_change_diagnostics(
                        pre_boundary=(
                            pair_boundary_before["selected_margin"].detach()
                        ),
                        post_boundary=(
                            post_pair_boundary["selected_margin"].detach()
                        ),
                        pre_target_logp=(
                            pair_boundary_before["selected_logp"].detach()
                        ),
                        post_target_logp=(
                            post_pair_boundary["selected_logp"].detach()
                        ),
                        pre_competitor_logp=(
                            pair_boundary_before["competitor_logp"].detach()
                        ),
                        post_competitor_logp=(
                            post_pair_boundary["competitor_logp"].detach()
                        ),
                        pre_rank=pair_boundary_before["selected_rank"].detach(),
                        post_rank=post_pair_boundary["selected_rank"].detach(),
                        pre_selected_index=(
                            pair_boundary_before["selected_candidate_index"]
                            .detach()
                        ),
                        post_selected_index=(
                            post_pair_boundary["selected_candidate_index"]
                            .detach()
                        ),
                        pre_competitor_index=(
                            pair_boundary_before["competitor_candidate_index"]
                            .detach()
                        ),
                        post_competitor_index=(
                            post_pair_boundary["competitor_candidate_index"]
                            .detach()
                        ),
                        pre_valid=pair_boundary_before["valid"].detach(),
                        post_valid=post_pair_boundary["valid"].detach(),
                        pre_forced=pair_boundary_before["forced"].detach(),
                        post_forced=post_pair_boundary["forced"].detach(),
                        pair_local_delta=pair_pursuit_local_delta.detach(),
                        num_agents=self.num_agents,
                        highest_orders=self.adj_network.highest_orders,
                        linearized_required_improvement=(
                            pair_boundary_linearized_required_improvement
                        ),
                        linearized_crossing_affordable=(
                            pair_boundary_linearized_crossing_affordable
                        ),
                        linearized_allocation_info=(
                            pair_boundary_linearized_allocation_info
                        ),
                        zero_tolerance=1.0e-12,
                    )
                )
                pair_boundary_trace_rows = pair_boundary_info["rows"]
                pair_boundary_target_count = pair_boundary_info[
                    "target_count"
                ]
                pair_boundary_correct_direction_count = pair_boundary_info[
                    "correct_count"
                ]
                pair_boundary_reverse_direction_count = pair_boundary_info[
                    "reverse_count"
                ]
                pair_boundary_approximately_zero_count = pair_boundary_info[
                    "zero_count"
                ]
                pair_boundary_signed_margin_change_mean = pair_boundary_info[
                    "signed_change_mean"
                ]
                pair_boundary_signed_margin_change_median = pair_boundary_info[
                    "signed_change_median"
                ]
                pair_boundary_signed_margin_change_worst = pair_boundary_info[
                    "signed_change_worst"
                ]
                pair_boundary_rank_crossing_count = pair_boundary_info[
                    "crossing_count"
                ]
                pair_boundary_positive_promotion_count = pair_boundary_info[
                    "promotion_count"
                ]
                pair_boundary_negative_eviction_count = pair_boundary_info[
                    "eviction_count"
                ]
                pair_boundary_postcondition_valid = bool((
                    pair_boundary_correct_direction_count
                    == pair_boundary_target_count
                ).item())
                if not pair_boundary_postcondition_valid:
                    raise RuntimeError(
                        "{} strict-pair selection-boundary guard did not "
                        "satisfy its all-member strict-progress contract: "
                        "correct={:.0f}, "
                        "reverse={:.0f}, zero={:.0f}, total={:.0f}".format(
                            "pending"
                            if bool(pair_only_objective)
                            else "standard",
                            float(
                                pair_boundary_correct_direction_count
                                .detach().cpu().item()
                            ),
                            float(
                                pair_boundary_reverse_direction_count
                                .detach().cpu().item()
                            ),
                            float(
                                pair_boundary_approximately_zero_count
                                .detach().cpu().item()
                            ),
                            float(
                                pair_boundary_target_count
                                .detach().cpu().item()
                            ),
                        )
                    )

        with torch.no_grad():
            valid_transition_ratio = float((1.0 - bad_transitions_mask).mean().detach().cpu().item())
            valid_agent_ratio = float(valid_agent_masks.mean().detach().cpu().item())

            valid_factor_stats_mask = (
                factor_mask
                * (1.0 - bad_transitions_mask).view(-1, 1, 1)
            )
            valid_factor_stats_count = valid_factor_stats_mask.sum().clamp_min(1.0)
            valid_graph_stats_mask = transition_mask > 0.5
            clip_numerator = (
                (
                    (graph_imp_weights > 1.0 + self.clip_param)
                    | (graph_imp_weights < 1.0 - self.clip_param)
                ).float()
                * transition_mask
            ).sum()
            clip_denominator = transition_mask.sum()
            clip_fraction = (
                clip_numerator / clip_denominator.clamp_min(1.0)
            )
            factor_clip_numerator = (
                (
                    (factor_imp_weights > 1.0 + self.clip_param)
                    | (factor_imp_weights < 1.0 - self.clip_param)
                ).float()
                * factor_training_mask
            ).sum()
            factor_clip_denominator = factor_training_mask.sum()
            factor_clip_fraction = (
                factor_clip_numerator
                / factor_clip_denominator.clamp_min(1.0)
            )
            trusted_clip_numerator = (
                (
                    (
                        (graph_imp_weights > 1.0 + self.clip_param)
                        | (graph_imp_weights < 1.0 - self.clip_param)
                    ).float()
                    * graph_loss_mask
                ).sum()
            )
            trusted_clip_denominator = graph_loss_mask.sum()
            trusted_clip_fraction = (
                trusted_clip_numerator
                / trusted_clip_denominator.clamp_min(1.0)
            )
            trusted_factor_clip_numerator = (
                (
                    (
                        (factor_imp_weights > 1.0 + self.clip_param)
                        | (factor_imp_weights < 1.0 - self.clip_param)
                    ).float()
                    * factor_loss_mask
                ).sum()
            )
            trusted_factor_clip_denominator = factor_loss_mask.sum()
            trusted_factor_clip_fraction = (
                trusted_factor_clip_numerator
                / trusted_factor_clip_denominator.clamp_min(1.0)
            )
            graph_trust_weight_mean = (
                (graph_trust_weights * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            factor_trust_weight_mean = (
                (factor_trust_weights * factor_training_mask).sum()
                / factor_training_mask.sum().clamp_min(1.0)
            )
            graph_stale_ratio = (
                (
                    (
                        (graph_imp_weights.detach() - 1.0).abs()
                        > float(self.adj_ppo_stale_trust_clip)
                    ).float()
                    * transition_mask
                ).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            factor_stale_ratio = (
                (
                    (
                        (factor_imp_weights.detach() - 1.0).abs()
                        > float(self.adj_ppo_stale_trust_clip)
                    ).float()
                    * factor_training_mask
                ).sum()
                / factor_training_mask.sum().clamp_min(1.0)
            )
            order2_factor_mask = (
                (factor_order_2d == 2.0).float()
                * factor_training_mask
            )
            order3_factor_mask = (
                (factor_order_2d == 3.0).float()
                * factor_training_mask
            )
            order2_factor_count = order2_factor_mask.sum()
            order3_factor_count = order3_factor_mask.sum()
            capture_factor_order_mask_float = (
                order2_factor_mask + order3_factor_mask
            ).clamp(max=1.0)
            capture_factor_order_count = (
                capture_factor_order_mask_float.sum()
            )
            selected_order_factor_count = (
                order2_factor_count + order3_factor_count
            ).clamp_min(1.0)
            order3_factor_fraction = (
                order3_factor_count / selected_order_factor_count
            )
            order2_factor_advantage_abs_mean = (
                (raw_local_factor_advantage.abs() * order2_factor_mask).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_factor_advantage_abs_mean = (
                (raw_local_factor_advantage.abs() * order3_factor_mask).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            weighted_order3_factor_advantage_abs_mean = (
                (local_factor_advantage.abs() * order3_factor_mask).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            credit_order2_factor_advantage_abs_mean = (
                (
                    credit_local_factor_advantage.abs()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            credit_order3_factor_advantage_abs_mean = (
                (
                    credit_local_factor_advantage.abs()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_credit_sum = (
                triplet_graph_return_credit
                * order3_factor_mask
            ).sum()
            triplet_graph_return_credit_mean = (
                triplet_graph_return_credit_sum
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_credit_active_fraction = (
                (
                    (triplet_graph_return_credit > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_credit_gate_mean = (
                (
                    triplet_graph_return_credit_gate
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_2d = delayed_triplet_credit.squeeze(-1)
            delayed_triplet_credit_mean = (
                (
                    delayed_triplet_credit_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_active_fraction = (
                (
                    (delayed_triplet_credit_2d.abs() > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_positive_fraction = (
                (
                    (delayed_triplet_credit_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_negative_fraction = (
                (
                    (delayed_triplet_credit_2d < 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_2d = (
                delayed_triplet_success_gate.squeeze(-1)
            )
            delayed_triplet_success_gate_mean = (
                (
                    delayed_triplet_success_gate_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_active_fraction = (
                (
                    (delayed_triplet_success_gate_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_selective_2d = (
                triplet_success_selective_gate
            )
            delayed_triplet_success_gate_selective_mean = (
                (
                    delayed_triplet_success_gate_selective_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_selective_fraction = (
                (
                    (
                        delayed_triplet_success_gate_selective_2d > 0.0
                    ).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_success_gate_2d = (
                triplet_graph_return_success_gate
            )
            triplet_graph_return_success_gate_mean = (
                (
                    triplet_graph_return_success_gate_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_success_gate_active_fraction = (
                (
                    (triplet_graph_return_success_gate_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_match_2d = (
                delayed_triplet_future_match.squeeze(-1)
            )
            delayed_triplet_future_exact_2d = (
                delayed_triplet_future_exact.squeeze(-1)
            )
            delayed_triplet_future_partial_2d = (
                delayed_triplet_future_partial.squeeze(-1)
            )
            delayed_triplet_future_match_weight_mean = (
                (
                    delayed_triplet_future_match_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_matched_fraction = (
                (
                    (delayed_triplet_future_match_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_exact_fraction = (
                (
                    (delayed_triplet_future_exact_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_partial_fraction = (
                (
                    (delayed_triplet_future_partial_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_2d = (
                capture_to_win_triplet_credit.squeeze(-1)
            )
            capture_to_win_quality_gate_2d = (
                capture_to_win_quality_gate.squeeze(-1)
            )
            capture_to_win_triplet_credit_mean = (
                (
                    capture_to_win_triplet_credit_2d
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_abs_mean = (
                (
                    capture_to_win_triplet_credit_2d.abs()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_active_fraction = (
                (
                    (
                        capture_to_win_triplet_credit_2d.abs() > 0.0
                    ).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_positive_fraction = (
                (
                    (capture_to_win_triplet_credit_2d > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_negative_fraction = (
                (
                    (capture_to_win_triplet_credit_2d < 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_positive_mass = (
                (
                    torch.clamp(
                        capture_to_win_triplet_credit_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_negative_mass = (
                (
                    torch.clamp(
                        -capture_to_win_triplet_credit_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_outcome_local_delta_positive_mass = (
                (
                    torch.clamp(
                        capture_outcome_weighted_delta,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_outcome_local_delta_negative_mass = (
                (
                    torch.clamp(
                        -capture_outcome_weighted_delta,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_outcome_local_delta_active_fraction = (
                (
                    (capture_outcome_weighted_delta.abs() > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_identity_target_mask_float = (
                (capture_to_win_triplet_credit_2d.abs() > 0.0).float()
                * capture_factor_order_mask_float
            )
            capture_identity_target_count = (
                capture_identity_target_mask_float.sum()
            )
            capture_outcome_identity_order2_delta_abs_mass = (
                capture_outcome_weighted_delta.abs() * order2_factor_mask
            ).sum()
            capture_outcome_identity_order3_delta_abs_mass = (
                capture_outcome_weighted_delta.abs() * order3_factor_mask
            ).sum()
            capture_identity_target_order2_fraction = (
                (capture_identity_target_mask_float * order2_factor_mask).sum()
                / capture_identity_target_count.clamp_min(1.0)
            )
            capture_identity_target_order3_fraction = (
                (capture_identity_target_mask_float * order3_factor_mask).sum()
                / capture_identity_target_count.clamp_min(1.0)
            )
            capture_outcome_non_target_delta_abs_max = (
                capture_outcome_weighted_delta.abs()
                * (capture_identity_target_mask_float <= 0.0).float()
                * capture_factor_order_mask_float
            ).max()
            capture_to_win_quality_gate_mean = (
                (
                    capture_to_win_quality_gate_2d
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_abs_mean = (
                (
                    capture_to_win_quality_gate_2d.abs()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_active_fraction = (
                (
                    (capture_to_win_quality_gate_2d.abs() > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_positive_fraction = (
                (
                    (capture_to_win_quality_gate_2d > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_negative_fraction = (
                (
                    (capture_to_win_quality_gate_2d < 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_positive_mass = (
                (
                    torch.clamp(
                        capture_to_win_quality_gate_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_negative_mass = (
                (
                    torch.clamp(
                        -capture_to_win_quality_gate_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            pair_pursuit_credit_2d = pair_pursuit_credit.squeeze(-1)
            pair_pursuit_quality_2d = pair_pursuit_quality.squeeze(-1)
            pair_to_triplet_transition_score_2d = (
                pair_to_triplet_transition_score.squeeze(-1)
            )
            triplet_capture_quality_2d = triplet_capture_quality.squeeze(-1)
            pair_pursuit_credit_mean = (
                (
                    pair_pursuit_credit_2d
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_credit_abs_mean = (
                (
                    pair_pursuit_credit_2d.abs()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_credit_positive_mass = (
                (
                    torch.clamp(pair_pursuit_credit_2d, min=0.0)
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_credit_negative_mass = (
                (
                    torch.clamp(-pair_pursuit_credit_2d, min=0.0)
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_credit_active_fraction = (
                (
                    (pair_pursuit_credit_2d.abs() > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            valid_pair_credit_values = pair_pursuit_credit_2d[
                order2_factor_mask > 0.0
            ].abs()
            if valid_pair_credit_values.numel() > 0:
                pair_pursuit_credit_std = valid_pair_credit_values.std(
                    unbiased=False
                )
                pair_pursuit_credit_max = valid_pair_credit_values.max()
                pair_pursuit_credit_nonzero_count = (
                    valid_pair_credit_values > 0.0
                ).float().sum()
                nonzero_pair_credit_values = valid_pair_credit_values[
                    valid_pair_credit_values > 0.0
                ]
                pair_credit_top1_count = max(
                    1,
                    int(np.ceil(
                        float(nonzero_pair_credit_values.numel()) * 0.01
                    )),
                )
                if nonzero_pair_credit_values.numel() > 0:
                    pair_credit_top1_mass_fraction = (
                        torch.topk(
                            nonzero_pair_credit_values,
                            k=pair_credit_top1_count,
                        )[0].sum()
                        / nonzero_pair_credit_values.sum().clamp_min(1e-8)
                    )
                else:
                    pair_credit_top1_mass_fraction = torch.zeros(
                        (), device=self.device, dtype=f_advts.dtype
                    )
            else:
                pair_pursuit_credit_std = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                pair_pursuit_credit_max = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                pair_pursuit_credit_nonzero_count = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                pair_credit_top1_mass_fraction = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
            pair_pursuit_quality_mean = (
                (
                    pair_pursuit_quality_2d
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_quality_active_fraction = (
                (
                    (pair_pursuit_quality_2d > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_to_triplet_transition_score_mean = (
                (
                    pair_to_triplet_transition_score_2d
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_to_triplet_transition_active_fraction = (
                (
                    (
                        pair_to_triplet_transition_score_2d > 0.0
                    ).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            triplet_capture_quality_mean = (
                (
                    triplet_capture_quality_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_capture_quality_active_fraction = (
                (
                    (triplet_capture_quality_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            pair_transition_delay_2d = pair_transition_delay.squeeze(-1)
            pair_transition_delay_mask = (
                (pair_transition_delay_2d > 0.0).float()
                * order2_factor_mask
            )
            pair_transition_delay_count = (
                pair_transition_delay_mask.sum()
            )
            transition_delay_mean = (
                (
                    pair_transition_delay_2d
                    * pair_transition_delay_mask
                ).sum()
                / pair_transition_delay_count.clamp_min(1.0)
            )
            if bool((pair_transition_delay_count > 0.0).item()):
                valid_transition_delays = pair_transition_delay_2d[
                    pair_transition_delay_mask > 0.0
                ]
                transition_delay_min = valid_transition_delays.min()
                transition_delay_max = valid_transition_delays.max()
            else:
                transition_delay_min = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                transition_delay_max = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )

            capture_counts_safe = torch.clamp(capture_counts, min=0.0)
            capture_matched_count_nonnegative = torch.clamp(
                capture_matched_count, min=0.0
            )
            # Keep compatibility with the older server PyTorch, which does not
            # provide torch.minimum. This is elementwise min(matched, captures)
            # and also prevents a malformed diagnostic count from exceeding the
            # number of real capture events.
            capture_matched_count_safe = torch.where(
                capture_matched_count_nonnegative <= capture_counts_safe,
                capture_matched_count_nonnegative,
                capture_counts_safe,
            )
            capture_count_total = capture_counts_safe.sum()
            capture_matched_count_value = capture_matched_count_safe.sum()
            unmatched_capture_count_value = (
                capture_count_total - capture_matched_count_value
            ).clamp_min(0.0)
            capture_matched_fraction = (
                capture_matched_count_value
                / capture_count_total.clamp_min(1.0)
            )
            unmatched_capture_fraction = (
                unmatched_capture_count_value
                / capture_count_total.clamp_min(1.0)
            )
            failed_episode_capture_count_value = (
                torch.clamp(
                    failed_episode_capture_count,
                    min=0.0,
                ).sum()
            )
            failed_episode_capture_fraction = (
                failed_episode_capture_count_value
                / capture_count_total.clamp_min(1.0)
            )
            capture_to_win_capture_success_fraction = (
                (
                    capture_count_total - failed_episode_capture_count_value
                ).clamp_min(0.0)
                / capture_count_total.clamp_min(1.0)
            )
            capture_to_win_episode_success_fraction = (
                (
                    capture_to_win_episode_success_gate
                    * transition_mask.reshape(batch_size, -1)
                ).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            outcome_episode_mask = torch.clamp(
                capture_outcome_diagnostics[:, 10],
                min=0.0,
                max=1.0,
            )
            outcome_global_mask = (
                transition_mask.reshape(batch_size, -1)
                .max(dim=1)
                .values
            )
            def _outcome_episode_mean(column, mask=outcome_episode_mask):
                return (
                    capture_outcome_diagnostics[:, column] * mask
                ).sum() / mask.sum().clamp_min(1.0)

            def _outcome_global_mean(column):
                return (
                    capture_outcome_diagnostics[:, column]
                    * outcome_global_mask
                ).sum() / outcome_global_mask.sum().clamp_min(1.0)

            capture_outcome_baseline_mean = _outcome_global_mean(0)
            capture_outcome_capture_episode_count_mean = (
                _outcome_global_mean(1)
            )
            capture_outcome_success_episode_count_mean = (
                _outcome_global_mean(2)
            )
            capture_outcome_failure_episode_count_mean = (
                _outcome_global_mean(3)
            )
            capture_outcome_mixed_window_fraction = _outcome_global_mean(4)
            capture_outcome_single_success_window_fraction = (
                _outcome_global_mean(5)
            )
            capture_outcome_single_failure_window_fraction = (
                _outcome_global_mean(6)
            )
            capture_outcome_no_capture_window_fraction = (
                _outcome_global_mean(7)
            )
            sampled_capture_episode_mask = (
                outcome_episode_mask
                * (capture_outcome_diagnostics[:, 8] > 0.0).float()
            )
            capture_outcome_triplet_labels_per_episode_mean = (
                _outcome_episode_mean(
                    8,
                    mask=sampled_capture_episode_mask,
                )
            )
            if bool((sampled_capture_episode_mask.sum() > 0.0).item()):
                capture_outcome_triplet_labels_per_episode_max = (
                    capture_outcome_diagnostics[
                        sampled_capture_episode_mask > 0.0,
                        8,
                    ].max()
                )
            else:
                capture_outcome_triplet_labels_per_episode_max = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
            capture_outcome_raw_episode_advantage_mean = (
                _outcome_episode_mean(
                    9,
                    mask=sampled_capture_episode_mask,
                )
            )
            capture_outcome_raw_episode_advantage_abs_mean = (
                (
                    capture_outcome_diagnostics[:, 9].abs()
                    * sampled_capture_episode_mask
                ).sum()
                / sampled_capture_episode_mask.sum().clamp_min(1.0)
            )
            capture_outcome_window_raw_centered_mean = (
                _outcome_global_mean(11)
            )
            capture_outcome_window_expanded_gate_sum = (
                _outcome_global_mean(12)
            )
            capture_outcome_window_expanded_gate_abs_sum = (
                _outcome_global_mean(13)
            )
            capture_outcome_window_center_error_ratio = (
                capture_outcome_window_expanded_gate_sum.abs()
                / capture_outcome_window_expanded_gate_abs_sum.clamp_min(1e-8)
            )
            capture_to_win_credit_preclip_mean = _outcome_global_mean(14)
            capture_to_win_credit_preclip_std = _outcome_global_mean(15)
            capture_to_win_credit_preclip_max = _outcome_global_mean(16)
            capture_to_win_credit_preclip_min = _outcome_global_mean(17)
            capture_to_win_credit_positive_clip_fraction = (
                _outcome_global_mean(18)
            )
            capture_to_win_credit_negative_clip_fraction = (
                _outcome_global_mean(19)
            )
            capture_identity_event_count_mean = _outcome_episode_mean(20)
            capture_identity_matched_event_count_mean = (
                _outcome_episode_mean(21)
            )
            capture_identity_unmatched_event_count_mean = (
                _outcome_episode_mean(22)
            )
            capture_identity_candidate_factor_count_mean = (
                _outcome_episode_mean(23)
            )
            capture_identity_match_fraction = (
                capture_identity_matched_event_count_mean
                / capture_identity_event_count_mean.clamp_min(1e-8)
            )
            capture_identity_candidates_per_matched_event = (
                capture_identity_candidate_factor_count_mean
                / capture_identity_matched_event_count_mean.clamp_min(1e-8)
            )
            capture_outcome_label_gate_correlation = _outcome_global_mean(24)
            capture_outcome_label_gate_correlation_valid = (
                _outcome_global_mean(25)
            )
            capture_outcome_success_labels_mean = _outcome_global_mean(26)
            capture_outcome_failure_labels_mean = _outcome_global_mean(27)
            capture_outcome_success_gate_total_mean = _outcome_global_mean(28)
            capture_outcome_failure_gate_total_mean = _outcome_global_mean(29)
            positive_reward_step_count_value = positive_reward_step.sum()
            positive_reward_without_capture_count_value = (
                positive_reward_without_capture.sum()
            )
            positive_reward_without_capture_fraction = (
                positive_reward_without_capture_count_value
                / positive_reward_step_count_value.clamp_min(1.0)
            )
            positive_reward_step_fraction = (
                positive_reward_step_count_value
                / transition_mask.sum().clamp_min(1.0)
            )
            offset0_candidate_count_value = (
                offset0_candidate_count.sum()
            )
            offset0_candidate_fraction = (
                (
                    (offset0_candidate_count > 0.0).float()
                    * transition_mask.reshape(batch_size, -1)
                ).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            graph_return_credit_strength_mean = (
                (graph_return_credit_strength * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            order2_factor_loss_mask = order2_factor_mask * factor_trust_weights
            order3_factor_loss_mask = order3_factor_mask * factor_trust_weights
            order2_factor_loss_count = order2_factor_loss_mask.sum()
            order3_factor_loss_count = order3_factor_loss_mask.sum()
            order2_factor_rl_loss = -(
                factor_min_surr * order2_factor_loss_mask
            ).sum() / order2_factor_loss_count.clamp_min(1.0)
            order3_factor_rl_loss = -(
                factor_min_surr * order3_factor_loss_mask
            ).sum() / order3_factor_loss_count.clamp_min(1.0)
            raw_order2_factor_rl_loss = -(
                raw_factor_min_surr * order2_factor_loss_mask
            ).sum() / order2_factor_loss_count.clamp_min(1.0)
            raw_order3_factor_rl_loss = -(
                raw_factor_min_surr * order3_factor_loss_mask
            ).sum() / order3_factor_loss_count.clamp_min(1.0)
            credit_order2_factor_rl_loss = -(
                credit_factor_min_surr * order2_factor_loss_mask
            ).sum() / order2_factor_loss_count.clamp_min(1.0)
            credit_order3_factor_rl_loss = -(
                credit_factor_min_surr * order3_factor_loss_mask
            ).sum() / order3_factor_loss_count.clamp_min(1.0)
            order2_positive_adv_fraction = (
                (
                    (raw_local_factor_advantage > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_positive_adv_fraction = (
                (
                    (raw_local_factor_advantage > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            credit_order2_positive_adv_fraction = (
                (
                    (credit_local_factor_advantage > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            credit_order3_positive_adv_fraction = (
                (
                    (credit_local_factor_advantage > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            order2_promoted_adv_fraction = (
                (
                    promoted_positive_adv_mask.float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_promoted_adv_fraction = (
                (
                    promoted_positive_adv_mask.float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            order2_promotion_strength_mean = (
                (
                    graph_promotion_strength
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_promotion_strength_mean = (
                (
                    graph_promotion_strength
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            order3_promoted_adv_weighted_fraction = (
                (
                    graph_promotion_strength
                    * positive_residual_mask.float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            pair_boundary_retention_late_selection_state_before = None
            if (
                    not bool(pair_only_objective)
                    and (
                        pair_diagnostic_target_present
                        or pair_boundary_retention_final_required_entries
                    )):
                pair_boundary_retention_late_selection_state_before = (
                    self
                    ._pair_selection_boundary_retention_selection_state_snapshot()
                )
            advantage_triplet_credit_info = {
                "adv_factor_credit_memory_observation_consumed": 0.0,
                "adv_triplet_credit_pair_updates": 0.0,
                "adv_triplet_credit_triplet_updates": 0.0,
                "adv_triplet_credit_pair_state_updates": 0.0,
                "adv_triplet_credit_triplet_state_updates": 0.0,
            }
            if (
                    bool(consume_factor_credit_observations)
                    and not bool(pair_only_objective)
                    and not bool(candidate_residual_only)
                    and bool(getattr(
                        self.adj_network,
                        "use_advantage_triplet_scorer",
                        False,
                    ))
                    and hasattr(
                        self.adj_network,
                        "update_factor_credit_memory",
                    )):
                advantage_triplet_credit_info = (
                    self.adj_network.update_factor_credit_memory(
                        valid_adj.detach(),
                        credit_local_factor_advantage.detach(),
                        graph_advantage.detach(),
                        factor_training_mask.detach(),
                    )
                )
                advantage_triplet_credit_info[
                    "adv_factor_credit_memory_observation_consumed"
                ] = 1.0

            valid_imp_weights = graph_imp_weights[
                valid_graph_stats_mask
            ]
            valid_factor_imp_weights = factor_imp_weights[
                factor_training_mask > 0.5
            ]
            valid_target_probs = torch.exp(target_factor_logp)[
                valid_factor_stats_mask.squeeze(-1) > 0.5
            ]
            valid_behavior_probs = torch.exp(behavior_factor_logp)[
                valid_factor_stats_mask.squeeze(-1) > 0.5
            ]
            if valid_imp_weights.numel() == 0:
                imp_weight_mean = torch.tensor(1.0, device=self.device)
                imp_weight_max = torch.tensor(1.0, device=self.device)
                imp_weight_std = torch.tensor(0.0, device=self.device)
                target_prob_mean = torch.tensor(0.0, device=self.device)
                behavior_prob_mean = torch.tensor(0.0, device=self.device)
                positive_adv_fraction = torch.tensor(0.0, device=self.device)
            else:
                imp_weight_mean = valid_imp_weights.mean()
                imp_weight_max = valid_imp_weights.max()
                imp_weight_std = (
                    valid_imp_weights.std()
                    if valid_imp_weights.numel() > 1
                    else torch.tensor(0.0, device=self.device)
                )
                target_prob_mean = valid_target_probs.mean()
                behavior_prob_mean = valid_behavior_probs.mean()
                positive_adv_fraction = (
                    (graph_advantage > 0.0).float()
                    * transition_mask
                ).sum() / transition_mask.sum().clamp_min(1.0)

            if valid_factor_imp_weights.numel() == 0:
                factor_imp_weight_mean = torch.tensor(
                    1.0, device=self.device
                )
                factor_imp_weight_max = torch.tensor(
                    1.0, device=self.device
                )
            else:
                factor_imp_weight_mean = valid_factor_imp_weights.mean()
                factor_imp_weight_max = valid_factor_imp_weights.max()

            current_adj_lr = active_adj_optimizer.param_groups[0]["lr"]

            order3_credit_gate = 1.0
            if (
                    not bool(pair_only_objective)
                    and hasattr(
                        self.adj_network, "update_order3_credit_gate")):
                order3_credit_gate = self.adj_network.update_order3_credit_gate(
                    _to_float(raw_order3_factor_rl_loss),
                    _to_float(raw_order2_factor_rl_loss),
                )
            order3_credit_loss_ema = float(
                getattr(self.adj_network, "order3_credit_loss_ema", 0.0)
            )
            order3_credit_margin_ema = float(
                getattr(self.adj_network, "order3_credit_margin_ema", 0.0)
            )

            # ``update_factor_credit_memory`` and
            # ``update_order3_credit_gate`` are not optimizer steps, but both
            # mutate production selection scores.  Run131 proved that v26's
            # parameter-only "final" gate could pass before these writes and
            # then expose a lost floor to the next transaction.  Treat the
            # complete selection-state write as part of the same atomic
            # transaction and retain the largest exact-safe fraction of it.
            if (
                    pair_boundary_retention_late_selection_state_before
                    is not None):
                try:
                    late_selection_state_after = (
                        self
                        ._pair_selection_boundary_retention_selection_state_snapshot()
                    )
                    seen_count_delta = 0.0
                    for seen_name in (
                            "pair_credit_seen",
                            "triplet_credit_seen"):
                        seen_before = (
                            pair_boundary_retention_late_selection_state_before[
                                "tensors"
                            ][seen_name]
                        )
                        seen_after = late_selection_state_after[
                            "tensors"
                        ][seen_name]
                        seen_delta = seen_after - seen_before
                        seen_count_delta += float(
                            seen_delta.sum().detach().cpu().item()
                        )
                        if (
                                not bool(torch.isfinite(seen_delta).all().item())
                                or float(
                                    seen_delta.min().detach().cpu().item()
                                ) < 0.0
                                or float(
                                    (seen_delta - seen_delta.round()).abs()
                                    .max().detach().cpu().item()
                                ) > 1e-6):
                            pair_boundary_retention_selection_state_seen_count_integral_valid = (
                                loss.new_tensor(0.0)
                            )
                            raise RuntimeError(
                                "selection-boundary retention observed-count "
                                "delta is not nonnegative and integral for {}"
                                .format(seen_name)
                            )
                    pair_boundary_retention_selection_state_seen_count_delta = (
                        loss.new_tensor(seen_count_delta)
                    )
                    late_state_required_entries = [
                        keyed_entry
                        for keyed_entry
                        in pair_boundary_retention_final_required_entries
                        if keyed_entry[0]
                        not in pair_boundary_retention_stop_keys
                    ]
                    late_state_acceptance = (
                        self
                        ._pair_selection_boundary_retention_exact_acceptance(
                            late_state_required_entries
                        )
                    )
                    late_invalid_keys = set(
                        late_state_acceptance["context_invalid_keys"]
                    )
                    pair_boundary_retention_stop_keys.update(late_invalid_keys)
                    late_state_required_entries = [
                        keyed_entry
                        for keyed_entry in late_state_required_entries
                        if keyed_entry[0] not in late_invalid_keys
                    ]

                    # Factor-credit EMAs and the order-3 gate participate in
                    # the same production catalog as the strict-pair score and
                    # boundary.  They are written after the adjacency Adam
                    # step.  Before v40, the late-state guard was entered only
                    # when an *older* retained crossing existed and it tested
                    # the current strict-pair objectives only while repairing
                    # an old-floor violation.  A transaction with no old floor
                    # could therefore pass its parameter postcondition, lose a
                    # newly-created boundary crossing in this late write, and
                    # archive the stale intermediate margin.  The following
                    # transaction then failed before Adam with "floor was
                    # already lost".  Evaluate the current objective at the
                    # full proposed production state regardless of archive
                    # population, so every state write belongs to one atomic
                    # strict-pair transaction.
                    late_current_objective_acceptance = None
                    if pair_diagnostic_target_present:
                        current_parameter_deltas = [
                            (parameter.detach() - before).detach().clone()
                            for parameter, before in zip(
                                self.adj_parameters,
                                pair_transaction_all_parameters_before_step,
                            )
                        ]
                        late_current_objective_acceptance = (
                            _evaluate_joint_pair_exact_scale(
                                boundary_trial_scale=1.0,
                                trial_displacements=current_parameter_deltas,
                                trial_parameter_before=(
                                    pair_transaction_all_parameters_before_step
                                ),
                            )
                        )
                        late_current_objectives_valid = bool(
                            late_current_objective_acceptance["valid"]
                        )
                    else:
                        late_current_objectives_valid = bool(
                            _nonpair_current_objectives_exact_valid()
                        )

                    if (
                            not late_current_objectives_valid
                            or (
                                late_state_required_entries
                                and not bool(late_state_acceptance["valid"])
                            )):
                        late_selection_state_backtracked = True
                        accepted_state_scale = None
                        accepted_state_acceptance = None

                        def _evaluate_late_selection_state_trial(state_scale):
                            self._interpolate_pair_selection_boundary_retention_selection_state(
                                before=(
                                    pair_boundary_retention_late_selection_state_before
                                ),
                                after=late_selection_state_after,
                                scale=state_scale,
                            )
                            if pair_diagnostic_target_present:
                                current_parameter_deltas = [
                                    (parameter.detach() - before).detach().clone()
                                    for parameter, before in zip(
                                        self.adj_parameters,
                                        pair_transaction_all_parameters_before_step,
                                    )
                                ]
                                current_objective_acceptance = (
                                    _evaluate_joint_pair_exact_scale(
                                        boundary_trial_scale=1.0,
                                        trial_displacements=(
                                            current_parameter_deltas
                                        ),
                                        trial_parameter_before=(
                                            pair_transaction_all_parameters_before_step
                                        ),
                                    )
                                )
                                current_objectives_valid = bool(
                                    current_objective_acceptance["valid"]
                                )
                            else:
                                current_objectives_valid = (
                                    _nonpair_current_objectives_exact_valid()
                                )
                            trial_state_acceptance = (
                                self
                                ._pair_selection_boundary_retention_exact_acceptance(
                                    late_state_required_entries
                                )
                            )
                            trial_safe = bool(
                                    current_objectives_valid
                                    and not trial_state_acceptance[
                                        "context_invalid_keys"
                                    ]
                                    and bool(trial_state_acceptance["valid"])
                            )
                            return trial_safe, trial_state_acceptance

                        unsafe_state_scale = 1.0
                        state_backtrack_trials = 0
                        state_scales = [
                            0.5 ** backtrack_index
                            for backtrack_index in range(1, 26)
                        ] + [0.0]
                        for state_scale in state_scales:
                            state_backtrack_trials += 1
                            trial_safe, trial_state_acceptance = (
                                _evaluate_late_selection_state_trial(
                                    state_scale
                                )
                            )
                            if trial_safe:
                                accepted_state_scale = float(state_scale)
                                accepted_state_acceptance = (
                                    trial_state_acceptance
                                )
                                pair_boundary_retention_selection_state_backtrack_count = (
                                    loss.new_tensor(float(
                                        state_backtrack_trials
                                    ))
                                )
                                break
                            unsafe_state_scale = float(state_scale)
                        if (
                                accepted_state_scale is None
                                or accepted_state_acceptance is None):
                            # The immutable transaction origin was validated
                            # before Adam.  Reaching scale zero here means the
                            # complete late production-state proposal remains
                            # incompatible because discrete observed counts
                            # must be committed at their exact endpoint.  This
                            # is a finite proposal with no committable atomic
                            # writeback, not corrupted learner state.  Reject
                            # the whole transaction through the same verified
                            # rollback/no-consumption path used by other
                            # finite optimizer no-ops.  Non-finite state and a
                            # failed rollback remain fail-loud in their
                            # existing checks.
                            late_noop_diagnostics = {
                                "pair_update_dot": 0.0,
                                "pair_update_norm_sq": 0.0,
                                "pair_gradient_norm_sq": 0.0,
                                "clipped_pair_dot": 0.0,
                                "clipped_gradient_norm_sq": 0.0,
                                "selection_state_trial_count": float(
                                    state_backtrack_trials
                                ),
                                "selection_state_seen_count_delta": float(
                                    seen_count_delta
                                ),
                            }
                            if pair_diagnostic_target_present:
                                for diagnostic_name, diagnostic_value in (
                                        pair_adam_guard_scalars.items()):
                                    late_noop_diagnostics[
                                        diagnostic_name
                                    ] = float(
                                        diagnostic_value.detach().cpu().item()
                                    )
                            _raise_recoverable_pair_optimizer_noop(
                                "selection_state_no_compatible_exact_writeback",
                                late_noop_diagnostics,
                            )

                        # Halving establishes a real exact-safe lower bound
                        # and a real unsafe upper bound.  Run132 showed that
                        # stopping at the first dyadic safe point can discard
                        # nearly all credit-state learning.  Refine the same
                        # immutable continuous-state segment deterministically
                        # while the discrete observed counts remain fully
                        # committed on every trial.
                        refinement_count = 12
                        for _refinement_index in range(refinement_count):
                            trial_scale = 0.5 * (
                                accepted_state_scale + unsafe_state_scale
                            )
                            trial_safe, trial_state_acceptance = (
                                _evaluate_late_selection_state_trial(
                                    trial_scale
                                )
                            )
                            if trial_safe:
                                accepted_state_scale = float(trial_scale)
                                accepted_state_acceptance = (
                                    trial_state_acceptance
                                )
                            else:
                                unsafe_state_scale = float(trial_scale)
                        self._interpolate_pair_selection_boundary_retention_selection_state(
                            before=(
                                pair_boundary_retention_late_selection_state_before
                            ),
                            after=late_selection_state_after,
                            scale=accepted_state_scale,
                        )
                        pair_boundary_retention_selection_state_refinement_count = (
                            loss.new_tensor(float(refinement_count))
                        )
                        pair_boundary_retention_selection_state_final_scale = (
                            loss.new_tensor(accepted_state_scale)
                        )
                        pair_boundary_retention_selection_state_unsafe_upper_scale = (
                            loss.new_tensor(unsafe_state_scale)
                        )
                        late_state_acceptance = accepted_state_acceptance

                        # Run133 proved that an exact frontier can still retain
                        # far below one percent of the continuous credit write.
                        # Attribute the conflict without changing the selected
                        # transaction state: start every component replay from
                        # continuous-before + fully committed discrete counts,
                        # apply exactly one full continuous component, and use
                        # the same production resolver and objective contract.
                        selected_late_state = (
                            self
                            ._pair_selection_boundary_retention_selection_state_snapshot()
                        )
                        try:
                            self._interpolate_pair_selection_boundary_retention_selection_state(
                                before=(
                                    pair_boundary_retention_late_selection_state_before
                                ),
                                after=late_selection_state_after,
                                scale=0.0,
                            )
                            baseline_component_forward = (
                                self._pair_selection_boundary_retention_forward(
                                    late_state_required_entries
                                )
                            )
                            component_specs = (
                                ("pair_credit_ema", "tensor"),
                                ("triplet_credit_ema", "tensor"),
                                ("order3_credit_loss_ema", "scalar"),
                                ("order3_credit_margin_ema", "scalar"),
                                ("current_order3_credit_gate", "scalar"),
                                ("all_continuous", "all"),
                            )
                            for component_name, component_kind in component_specs:
                                self._interpolate_pair_selection_boundary_retention_selection_state(
                                    before=(
                                        pair_boundary_retention_late_selection_state_before
                                    ),
                                    after=late_selection_state_after,
                                    scale=0.0,
                                )
                                if component_kind == "all":
                                    self._interpolate_pair_selection_boundary_retention_selection_state(
                                        before=(
                                            pair_boundary_retention_late_selection_state_before
                                        ),
                                        after=late_selection_state_after,
                                        scale=1.0,
                                    )
                                    component_delta_squared = 0.0
                                    for tensor_name in (
                                            "pair_credit_ema",
                                            "triplet_credit_ema"):
                                        tensor_delta = (
                                            late_selection_state_after["tensors"][
                                                tensor_name
                                            ]
                                            - pair_boundary_retention_late_selection_state_before[
                                                "tensors"
                                            ][tensor_name]
                                        )
                                        component_delta_squared += float(
                                            tensor_delta.pow(2).sum()
                                            .detach().cpu().item()
                                        )
                                    for scalar_name in (
                                            "order3_credit_loss_ema",
                                            "order3_credit_margin_ema",
                                            "current_order3_credit_gate"):
                                        scalar_delta = (
                                            float(late_selection_state_after[
                                                "scalars"
                                            ][scalar_name])
                                            - float(
                                                pair_boundary_retention_late_selection_state_before[
                                                    "scalars"
                                                ][scalar_name]
                                            )
                                        )
                                        component_delta_squared += scalar_delta ** 2
                                    component_delta_norm = math.sqrt(
                                        component_delta_squared
                                    )
                                elif component_kind == "tensor":
                                    component_before = (
                                        pair_boundary_retention_late_selection_state_before[
                                            "tensors"
                                        ][component_name]
                                    )
                                    component_after = late_selection_state_after[
                                        "tensors"
                                    ][component_name]
                                    component_delta_norm = float(
                                        (component_after - component_before)
                                        .norm().detach().cpu().item()
                                    )
                                    with torch.no_grad():
                                        getattr(
                                            self.adj_network, component_name
                                        ).copy_(component_after)
                                else:
                                    component_before = float(
                                        pair_boundary_retention_late_selection_state_before[
                                            "scalars"
                                        ][component_name]
                                    )
                                    component_after = float(
                                        late_selection_state_after["scalars"][
                                            component_name
                                        ]
                                    )
                                    component_delta_norm = abs(
                                        component_after - component_before
                                    )
                                    setattr(
                                        self.adj_network,
                                        component_name,
                                        component_after,
                                    )
                                if pair_diagnostic_target_present:
                                    current_parameter_deltas = [
                                        (
                                            parameter.detach() - before
                                        ).detach().clone()
                                        for parameter, before in zip(
                                            self.adj_parameters,
                                            pair_transaction_all_parameters_before_step,
                                        )
                                    ]
                                    component_objectives_valid = bool(
                                        _evaluate_joint_pair_exact_scale(
                                            boundary_trial_scale=1.0,
                                            trial_displacements=(
                                                current_parameter_deltas
                                            ),
                                            trial_parameter_before=(
                                                pair_transaction_all_parameters_before_step
                                            ),
                                        )["valid"]
                                    )
                                else:
                                    component_objectives_valid = bool(
                                        _nonpair_current_objectives_exact_valid()
                                    )
                                component_forward = (
                                    self
                                    ._pair_selection_boundary_retention_forward(
                                        late_state_required_entries
                                    )
                                )
                                pair_boundary_retention_selection_state_component_rows.extend(
                                    self
                                    ._pair_selection_boundary_retention_component_rows(
                                        keyed_entries=(
                                            late_state_required_entries
                                        ),
                                        baseline=baseline_component_forward,
                                        component=component_forward,
                                        component_name=component_name,
                                        component_delta_norm=(
                                            component_delta_norm
                                        ),
                                        joint_objectives_valid=(
                                            component_objectives_valid
                                        ),
                                    )
                                )
                        finally:
                            self._restore_pair_selection_boundary_retention_selection_state(
                                selected_late_state
                            )

                    else:
                        late_selection_state_backtracked = False

                    final_late_state_acceptance = (
                        self
                        ._pair_selection_boundary_retention_exact_acceptance(
                            late_state_required_entries
                        )
                    )
                    if pair_diagnostic_target_present:
                        if late_selection_state_backtracked:
                            final_parameter_deltas = [
                                (parameter.detach() - before).detach().clone()
                                for parameter, before in zip(
                                    self.adj_parameters,
                                    pair_transaction_all_parameters_before_step,
                                )
                            ]
                            final_current_objective_acceptance = (
                                _evaluate_joint_pair_exact_scale(
                                    boundary_trial_scale=1.0,
                                    trial_displacements=final_parameter_deltas,
                                    trial_parameter_before=(
                                        pair_transaction_all_parameters_before_step
                                    ),
                                )
                            )
                        else:
                            final_current_objective_acceptance = (
                                late_current_objective_acceptance
                            )
                        if final_current_objective_acceptance is None:
                            _rollback_failed_pair_transaction()
                            raise RuntimeError(
                                "strict-pair late-state acceptance is missing"
                            )
                        final_current_objectives_valid = bool(
                            final_current_objective_acceptance["valid"]
                        )
                        pair_complete_production_boundary = (
                            final_current_objective_acceptance[
                                "boundary_revalidation"
                            ]
                        )
                    else:
                        final_current_objectives_valid = bool(
                            _nonpair_current_objectives_exact_valid()
                        )
                    if (
                            not final_current_objectives_valid
                            or
                            final_late_state_acceptance["context_invalid_keys"]
                            or not bool(final_late_state_acceptance["valid"])):
                        _rollback_failed_pair_transaction()
                        raise RuntimeError(
                            "selection-boundary retention floor failed the "
                            "final production-state postcondition"
                        )
                    if final_late_state_acceptance[
                            "min_signed_gap"] is not None:
                        pair_boundary_retention_final_exact_min_signed_gap = (
                            final_late_state_acceptance[
                                "min_signed_gap"
                            ].detach()
                        )
                        pair_boundary_retention_final_exact_max_tolerance = (
                            final_late_state_acceptance[
                                "max_tolerance"
                            ].detach()
                        )
                    for retention_key in late_invalid_keys:
                        retention_entry = (
                            self
                            ._pair_selection_boundary_retention_observations
                            .get(retention_key)
                        )
                        if retention_entry is None:
                            continue
                        retention_entry["protection_stopped"] = True
                        retention_entry["protection_stop_reason"] = (
                            "context_invalid"
                        )
                        retention_entry["protection_stop_clock"] = int(
                            self._pair_selection_boundary_ordinary_update_clock
                            + 1
                        )
                    order3_credit_gate = float(
                        self.adj_network.current_order3_credit_gate
                    )
                    order3_credit_loss_ema = float(
                        self.adj_network.order3_credit_loss_ema
                    )
                    order3_credit_margin_ema = float(
                        self.adj_network.order3_credit_margin_ema
                    )
                    if advantage_triplet_credit_info:
                        advantage_triplet_credit_info[
                            "adv_triplet_credit_seen_ratio"
                        ] = float(
                            (self.adj_network.triplet_credit_seen > 0.0)
                            .float().mean().detach().cpu().item()
                        )
                        advantage_triplet_credit_info[
                            "adv_pair_credit_seen_ratio"
                        ] = float(
                            (self.adj_network.pair_credit_seen > 0.0)
                            .float().mean().detach().cpu().item()
                        )
                except PairOptimizerRecoverableNoOpError:
                    raise
                except (
                        RuntimeError,
                        ValueError,
                        KeyError,
                        IndexError,
                        FloatingPointError):
                    _rollback_failed_pair_transaction()
                    raise

        if pair_diagnostic_target_present:
            # Build the persisted score/boundary trace from the *complete*
            # production transaction, after both parameter writeback and the
            # late factor-credit/gate state write.  The old trace was captured
            # before the latter and could archive a larger intermediate floor
            # than the state which actually returned to the runner.  That
            # stale floor was exactly the observation_id=9 failure: the next
            # update correctly replayed the final state but compared it with a
            # non-final commit margin.  Reusing the authoritative final replay
            # also makes every exported crossing/rank/competitor describe the
            # state that will be checkpointed.
            with torch.no_grad():
                final_pair_boundary = (
                    pair_complete_production_boundary
                    if pair_complete_production_boundary is not None
                    else post_pair_boundary
                )
                if final_pair_boundary is None:
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "complete strict-pair production replay is missing"
                    )
                final_pair_score_info = (
                    _pair_target_score_change_diagnostics(
                        pre_factor_logp=target_factor_logp.detach(),
                        post_factor_logp=final_pair_boundary[
                            "behavior_selected_logp"
                        ],
                        pair_local_delta=pair_pursuit_local_delta.detach(),
                        zero_tolerance=(
                            joint_exact_preservation_tolerance
                            if joint_exact_preservation_tolerance is not None
                            else 1.0e-12
                        ),
                    )
                )
                if not _pair_target_score_nonregression_postcondition(
                        final_pair_score_info):
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "complete strict-pair production state reversed an "
                        "exact target score"
                    )
                pair_target_score_signed_change_mean = (
                    final_pair_score_info["signed_change_mean"]
                )
                pair_target_score_positive_change_mean = (
                    final_pair_score_info["positive_change_mean"]
                )
                pair_target_score_negative_signed_change_mean = (
                    final_pair_score_info["negative_signed_change_mean"]
                )
                pair_target_score_positive_target_count = (
                    final_pair_score_info["positive_target_count"]
                )
                pair_target_score_negative_target_count = (
                    final_pair_score_info["negative_target_count"]
                )
                pair_target_score_correct_direction_count = (
                    final_pair_score_info["correct_direction_count"]
                )
                pair_target_score_reverse_direction_count = (
                    final_pair_score_info["reverse_direction_count"]
                )
                pair_target_score_approximately_zero_count = (
                    final_pair_score_info["approximately_zero_count"]
                )
                pair_target_score_before_after_join_valid = (
                    final_pair_score_info["before_after_join_valid"]
                )
                pair_target_score_zero_tolerance = (
                    final_pair_score_info["zero_tolerance"]
                )
                pair_target_count = final_pair_score_info["target_count"]
                pair_score_postcondition_valid = True
                final_pair_boundary_info = (
                    _pair_selection_boundary_change_diagnostics(
                        pre_boundary=(
                            pair_boundary_before["selected_margin"].detach()
                        ),
                        post_boundary=(
                            final_pair_boundary["selected_margin"].detach()
                        ),
                        pre_target_logp=(
                            pair_boundary_before["selected_logp"].detach()
                        ),
                        post_target_logp=(
                            final_pair_boundary["selected_logp"].detach()
                        ),
                        pre_competitor_logp=(
                            pair_boundary_before["competitor_logp"].detach()
                        ),
                        post_competitor_logp=(
                            final_pair_boundary["competitor_logp"].detach()
                        ),
                        pre_rank=pair_boundary_before["selected_rank"].detach(),
                        post_rank=final_pair_boundary["selected_rank"].detach(),
                        pre_selected_index=(
                            pair_boundary_before["selected_candidate_index"]
                            .detach()
                        ),
                        post_selected_index=(
                            final_pair_boundary["selected_candidate_index"]
                            .detach()
                        ),
                        pre_competitor_index=(
                            pair_boundary_before[
                                "competitor_candidate_index"
                            ].detach()
                        ),
                        post_competitor_index=(
                            final_pair_boundary[
                                "competitor_candidate_index"
                            ].detach()
                        ),
                        pre_valid=pair_boundary_before["valid"].detach(),
                        post_valid=final_pair_boundary["valid"].detach(),
                        pre_forced=pair_boundary_before["forced"].detach(),
                        post_forced=final_pair_boundary["forced"].detach(),
                        pair_local_delta=pair_pursuit_local_delta.detach(),
                        num_agents=self.num_agents,
                        highest_orders=self.adj_network.highest_orders,
                        linearized_required_improvement=(
                            pair_boundary_linearized_required_improvement
                        ),
                        linearized_crossing_affordable=(
                            pair_boundary_linearized_crossing_affordable
                        ),
                        linearized_allocation_info=(
                            pair_boundary_linearized_allocation_info
                        ),
                        zero_tolerance=1.0e-12,
                    )
                )
                if not bool((
                        final_pair_boundary_info["correct_count"]
                        == final_pair_boundary_info["target_count"]
                ).item()):
                    _rollback_failed_pair_transaction()
                    raise RuntimeError(
                        "complete strict-pair production state lost its "
                        "selection-boundary progress contract"
                    )
                pair_boundary_trace_rows = final_pair_boundary_info["rows"]
                pair_boundary_target_count = final_pair_boundary_info[
                    "target_count"
                ]
                pair_boundary_correct_direction_count = (
                    final_pair_boundary_info["correct_count"]
                )
                pair_boundary_reverse_direction_count = (
                    final_pair_boundary_info["reverse_count"]
                )
                pair_boundary_approximately_zero_count = (
                    final_pair_boundary_info["zero_count"]
                )
                pair_boundary_signed_margin_change_mean = (
                    final_pair_boundary_info["signed_change_mean"]
                )
                pair_boundary_signed_margin_change_median = (
                    final_pair_boundary_info["signed_change_median"]
                )
                pair_boundary_signed_margin_change_worst = (
                    final_pair_boundary_info["signed_change_worst"]
                )
                pair_boundary_rank_crossing_count = (
                    final_pair_boundary_info["crossing_count"]
                )
                pair_boundary_positive_promotion_count = (
                    final_pair_boundary_info["promotion_count"]
                )
                pair_boundary_negative_eviction_count = (
                    final_pair_boundary_info["eviction_count"]
                )
                pair_boundary_postcondition_valid = True

        candidate_identity_transaction_rows = []
        if (
                transaction_diagnostics_enabled
                and not bool(candidate_residual_only)
                and candidate_objective_active):
            candidate_identity_transaction_rows = (
                _candidate_identity_transaction_trace_rows(
                    candidate_identity_delta=(
                        candidate_identity_delta[..., 0]
                    ),
                    candidate_effective_delta=candidate_effective_delta,
                    candidate_capture_context=(
                        candidate_capture_context[..., 0]
                    ),
                    episode_success_gate=(
                        capture_to_win_episode_success_gate
                    ),
                    behavior_candidate_margin=behavior_candidate_margin,
                    behavior_candidate_rank=behavior_candidate_rank,
                    behavior_candidate_valid=behavior_candidate_valid,
                    behavior_candidate_version=behavior_candidate_version,
                    current_candidate_policy_version=(
                        current_candidate_policy_version
                    ),
                    pre_candidate_margin=current_margin,
                    pre_candidate_rank=candidate_identity_current_rank,
                    pre_candidate_valid=candidate_identity_valid_mask,
                    post_candidate_margin=post_margin,
                    post_candidate_rank=post_rank,
                    post_candidate_valid=candidate_identity_valid_mask,
                    replay_population_provenance=(
                        replay_population_provenance
                    ),
                    candidate_lifecycle_progress_mask=(
                        candidate_lifecycle_behavioral_progress_mask
                    ),
                    num_agents=self.num_agents,
                    highest_orders=self.highest_orders,
                )
            )
            expected_candidate_trace_rows = int(round(_to_float(
                candidate_identity_loss_info[
                    "unsatisfied_target_count"
                ]
            )))
            if (
                    len(candidate_identity_transaction_rows)
                    != expected_candidate_trace_rows):
                raise RuntimeError(
                    "candidate identity trace row count does not match the "
                    "optimized target population"
                )

        candidate_evidence_consumption_rows = []
        if (
                transaction_diagnostics_enabled
                and not bool(candidate_residual_only)
                and candidate_target_present):
            candidate_evidence_consumption_rows = (
                _candidate_evidence_consumption_trace_rows(
                    candidate_identity_delta=(
                        candidate_identity_delta[..., 0]
                    ),
                    behavior_candidate_version=(
                        behavior_candidate_version
                    ),
                    replay_population_provenance=(
                        replay_population_provenance
                    ),
                )
            )
            expected_evidence_rows = int(round(_to_float(
                candidate_identity_loss_info["target_count"]
            )))
            if (
                    len(candidate_evidence_consumption_rows)
                    != expected_evidence_rows):
                raise RuntimeError(
                    "candidate evidence consumption row count does not match "
                    "the target population"
                )

        train_info = {}
        pair_only_scope_contract = bool(
            not pair_only_objective
            or (
                abs(_to_float(graph_rl_loss)) <= 1e-12
                and abs(_to_float(base_factor_rl_loss)) <= 1e-12
                and abs(_to_float(
                    capture_outcome_factor_loss_contribution
                )) <= 1e-12
                and abs(_to_float(
                    capture_candidate_identity_loss_contribution
                )) <= 1e-12
                and abs(_to_float(entropy_loss)) <= 1e-12
                and _to_float(
                    pair_pursuit_factor_loss_target_transition_count
                ) > 0.0
                and _to_float(pair_gradient_norm) > 0.0
                and _to_float(
                    pair_target_control_raw_denominator
                ) > 0.0
                and _to_float(
                    pair_target_control_trusted_denominator
                ) > 0.0
            )
        )
        if not pair_only_scope_contract:
            raise RuntimeError(
                "pending pair-only objective scope contract failed"
            )
        train_info['pair_pending_objective_scope_version'] = 1.0
        train_info['pair_pending_objective_scope_pair_only'] = float(
            bool(pair_only_objective)
        )
        train_info['pair_pending_objective_scope_contract_valid'] = float(
            pair_only_scope_contract
        )
        train_info['pair_pending_control_scope_version'] = 2.0
        train_info[
            'pair_pending_standard_ppo_early_stop_applicable'
        ] = float(not bool(pair_only_objective))
        train_info[
            'pair_pending_all_configured_epochs_required'
        ] = float(bool(pair_only_objective))
        train_info[
            'pair_pending_pair_target_clip_numerator'
        ] = _to_float(pair_target_control_raw_numerator)
        train_info[
            'pair_pending_pair_target_clip_denominator'
        ] = _to_float(pair_target_control_raw_denominator)
        train_info[
            'pair_pending_pair_target_clip_ratio'
        ] = _to_float(pair_target_control_raw_ratio)
        train_info[
            'pair_pending_pair_target_trusted_clip_numerator'
        ] = _to_float(pair_target_control_trusted_numerator)
        train_info[
            'pair_pending_pair_target_trusted_clip_denominator'
        ] = _to_float(pair_target_control_trusted_denominator)
        train_info[
            'pair_pending_pair_target_trusted_clip_ratio'
        ] = _to_float(pair_target_control_trusted_ratio)
        train_info['pair_pending_graph_loss'] = _to_float(graph_rl_loss)
        train_info['pair_pending_base_factor_loss'] = _to_float(
            base_factor_rl_loss
        )
        train_info['pair_pending_capture_outcome_loss'] = _to_float(
            capture_outcome_factor_loss_contribution
        )
        train_info['pair_pending_candidate_loss'] = _to_float(
            capture_candidate_identity_loss_contribution
        )
        train_info['pair_pending_entropy_loss'] = _to_float(entropy_loss)
        train_info['advantage'] = _to_float(f_advts.mean())
        train_info['f_advts_abs_mean'] = _to_float(f_advts.abs().mean())
        train_info['graph_advantage_abs_mean'] = _to_float(
            (graph_advantage.abs() * transition_mask).sum()
            / transition_mask.sum().clamp_min(1.0)
        )
        train_info['graph_advantage_source_version'] = 2.0
        train_info['graph_advantage_legacy_mean_abs'] = _to_float(
            (legacy_graph_advantage.abs() * transition_mask).sum()
            / transition_mask.sum().clamp_min(1.0)
        )
        train_info['graph_advantage_contamination_mean'] = _to_float(
            (graph_advantage_contamination * transition_mask).sum()
            / transition_mask.sum().clamp_min(1.0)
        )
        train_info['graph_advantage_contamination_abs_mean'] = _to_float(
            (graph_advantage_contamination.abs() * transition_mask).sum()
            / transition_mask.sum().clamp_min(1.0)
        )
        train_info['graph_advantage_contamination_abs_max'] = _to_float(
            (graph_advantage_contamination.abs() * transition_mask).max()
        )
        train_info['clamp_ratio'] = _to_float(clip_fraction)
        train_info['factor_clamp_ratio'] = _to_float(
            factor_clip_fraction
        )
        train_info['trusted_clamp_ratio'] = _to_float(
            trusted_clip_fraction
        )
        train_info['trusted_factor_clamp_ratio'] = _to_float(
            trusted_factor_clip_fraction
        )
        # Internal composable control mass. Runner control must aggregate these
        # totals, never average mini-batch ratios with unequal valid/trusted
        # denominators. Internal keys are deliberately excluded from generic
        # scalar logging; Runner emits audited epoch/update totals below.
        train_info['_adj_control_raw_graph_numerator'] = _to_float(
            clip_numerator
        )
        train_info['_adj_control_raw_graph_denominator'] = _to_float(
            clip_denominator
        )
        train_info['_adj_control_raw_factor_numerator'] = _to_float(
            factor_clip_numerator
        )
        train_info['_adj_control_raw_factor_denominator'] = _to_float(
            factor_clip_denominator
        )
        train_info['_adj_control_trusted_graph_numerator'] = _to_float(
            trusted_clip_numerator
        )
        train_info['_adj_control_trusted_graph_denominator'] = _to_float(
            trusted_clip_denominator
        )
        train_info['_adj_control_trusted_factor_numerator'] = _to_float(
            trusted_factor_clip_numerator
        )
        train_info['_adj_control_trusted_factor_denominator'] = _to_float(
            trusted_factor_clip_denominator
        )
        train_info['adj_graph_trust_weight_mean'] = _to_float(
            graph_trust_weight_mean
        )
        train_info['adj_factor_trust_weight_mean'] = _to_float(
            factor_trust_weight_mean
        )
        train_info['adj_graph_stale_ratio'] = _to_float(graph_stale_ratio)
        train_info['adj_factor_stale_ratio'] = _to_float(factor_stale_ratio)
        train_info['use_adj_ppo_stale_trust'] = float(
            self.use_adj_ppo_stale_trust
        )
        train_info['adj_ppo_stale_trust_clip'] = float(
            self.adj_ppo_stale_trust_clip
        )
        train_info['adj_ppo_stale_trust_scale'] = float(
            self.adj_ppo_stale_trust_scale
        )
        train_info['adj_ppo_stale_trust_min_weight'] = float(
            self.adj_ppo_stale_trust_min_weight
        )
        train_info['imp_weight_mean'] = _to_float(imp_weight_mean)
        train_info['imp_weight_max'] = _to_float(imp_weight_max)
        train_info['imp_weight_std'] = _to_float(imp_weight_std)
        train_info['factor_imp_weight_mean'] = _to_float(
            factor_imp_weight_mean
        )
        train_info['factor_imp_weight_max'] = _to_float(
            factor_imp_weight_max
        )
        train_info['target_factor_prob_mean'] = _to_float(target_prob_mean)
        train_info['behavior_factor_prob_mean'] = _to_float(behavior_prob_mean)
        train_info['positive_adv_fraction'] = _to_float(positive_adv_fraction)
        train_info['valid_transition_ratio'] = valid_transition_ratio
        train_info['valid_agent_ratio'] = valid_agent_ratio
        train_info['rl_loss'] = _to_float(rl_loss)
        train_info['graph_rl_loss'] = _to_float(graph_rl_loss)
        train_info['factor_rl_loss'] = _to_float(factor_rl_loss)
        train_info['factor_advantage_abs_mean'] = _to_float(
            (local_factor_advantage.abs() * factor_training_mask).sum()
            / factor_training_mask.sum().clamp_min(1.0)
        )
        train_info['raw_factor_advantage_abs_mean'] = _to_float(
            (raw_local_factor_advantage.abs() * factor_training_mask).sum()
            / factor_training_mask.sum().clamp_min(1.0)
        )
        train_info['order3_factor_fraction'] = _to_float(
            order3_factor_fraction
        )
        train_info['order2_factor_advantage_abs_mean'] = _to_float(
            order2_factor_advantage_abs_mean
        )
        train_info['order3_factor_advantage_abs_mean'] = _to_float(
            order3_factor_advantage_abs_mean
        )
        train_info['weighted_order3_factor_advantage_abs_mean'] = _to_float(
            weighted_order3_factor_advantage_abs_mean
        )
        train_info['credit_order2_factor_advantage_abs_mean'] = _to_float(
            credit_order2_factor_advantage_abs_mean
        )
        train_info['credit_order3_factor_advantage_abs_mean'] = _to_float(
            credit_order3_factor_advantage_abs_mean
        )
        train_info['triplet_graph_return_credit_mean'] = _to_float(
            triplet_graph_return_credit_mean
        )
        train_info['triplet_graph_return_credit_active_fraction'] = _to_float(
            triplet_graph_return_credit_active_fraction
        )
        train_info['triplet_graph_return_credit_gate_mean'] = _to_float(
            triplet_graph_return_credit_gate_mean
        )
        train_info['delayed_triplet_credit_mean'] = _to_float(
            delayed_triplet_credit_mean
        )
        train_info['delayed_triplet_credit_active_fraction'] = _to_float(
            delayed_triplet_credit_active_fraction
        )
        train_info['delayed_triplet_credit_positive_fraction'] = _to_float(
            delayed_triplet_credit_positive_fraction
        )
        train_info['delayed_triplet_credit_negative_fraction'] = _to_float(
            delayed_triplet_credit_negative_fraction
        )
        train_info['delayed_triplet_success_gate_mean'] = _to_float(
            delayed_triplet_success_gate_mean
        )
        train_info['delayed_triplet_success_gate_active_fraction'] = _to_float(
            delayed_triplet_success_gate_active_fraction
        )
        train_info['delayed_triplet_success_gate_selective_mean'] = _to_float(
            delayed_triplet_success_gate_selective_mean
        )
        train_info[
            'delayed_triplet_success_gate_selective_fraction'
        ] = _to_float(delayed_triplet_success_gate_selective_fraction)
        train_info['triplet_graph_return_success_gate_mean'] = _to_float(
            triplet_graph_return_success_gate_mean
        )
        train_info[
            'triplet_graph_return_success_gate_active_fraction'
        ] = _to_float(triplet_graph_return_success_gate_active_fraction)
        train_info['delayed_triplet_future_match_weight_mean'] = _to_float(
            delayed_triplet_future_match_weight_mean
        )
        train_info['delayed_triplet_future_matched_fraction'] = _to_float(
            delayed_triplet_future_matched_fraction
        )
        train_info['delayed_triplet_future_exact_fraction'] = _to_float(
            delayed_triplet_future_exact_fraction
        )
        train_info['delayed_triplet_future_partial_fraction'] = _to_float(
            delayed_triplet_future_partial_fraction
        )
        train_info['capture_to_win_triplet_credit_mean'] = _to_float(
            capture_to_win_triplet_credit_mean
        )
        train_info['capture_to_win_triplet_credit_abs_mean'] = _to_float(
            capture_to_win_triplet_credit_abs_mean
        )
        train_info['capture_to_win_triplet_credit_active_fraction'] = _to_float(
            capture_to_win_triplet_credit_active_fraction
        )
        train_info['capture_to_win_triplet_credit_positive_fraction'] = _to_float(
            capture_to_win_triplet_credit_positive_fraction
        )
        train_info['capture_to_win_triplet_credit_negative_fraction'] = _to_float(
            capture_to_win_triplet_credit_negative_fraction
        )
        train_info['capture_to_win_triplet_credit_positive_mass'] = _to_float(
            capture_to_win_triplet_credit_positive_mass
        )
        train_info['capture_to_win_triplet_credit_negative_mass'] = _to_float(
            capture_to_win_triplet_credit_negative_mass
        )
        train_info['capture_outcome_local_delta_positive_mass'] = _to_float(
            capture_outcome_local_delta_positive_mass
        )
        train_info['capture_outcome_local_delta_negative_mass'] = _to_float(
            capture_outcome_local_delta_negative_mass
        )
        train_info['capture_outcome_local_delta_active_fraction'] = _to_float(
            capture_outcome_local_delta_active_fraction
        )
        train_info['capture_outcome_factor_loss_contribution'] = _to_float(
            capture_outcome_factor_loss_contribution
        )
        train_info[
            'capture_outcome_positive_factor_loss_contribution'
        ] = _to_float(capture_outcome_positive_factor_loss_contribution)
        train_info[
            'capture_outcome_negative_factor_loss_contribution'
        ] = _to_float(capture_outcome_negative_factor_loss_contribution)
        train_info[
            'capture_outcome_factor_loss_target_count'
        ] = _to_float(capture_outcome_factor_loss_target_count)
        train_info[
            'capture_outcome_factor_loss_valid_transition_count'
        ] = _to_float(capture_outcome_factor_loss_valid_transition_count)
        train_info[
            'capture_outcome_factor_loss_factors_per_transition'
        ] = _to_float(capture_outcome_factor_loss_factors_per_transition)
        train_info[
            'capture_outcome_factor_loss_normalization_version'
        ] = float(capture_outcome_factor_loss_normalization_version)
        train_info[
            'base_factor_pair_evidence_valid_transition_count'
        ] = _to_float(
            base_factor_population_info["pair_valid_transition_count"]
        )
        train_info[
            'base_factor_non_pair_valid_transition_count'
        ] = _to_float(
            base_factor_population_info["non_pair_valid_transition_count"]
        )
        train_info[
            'base_factor_pair_evidence_valid_factor_mask_count'
        ] = _to_float(
            base_factor_population_info["pair_valid_factor_mask_count"]
        )
        train_info[
            'base_factor_non_pair_valid_factor_mask_count'
        ] = _to_float(
            base_factor_population_info["non_pair_valid_factor_mask_count"]
        )
        train_info[
            'base_factor_pair_evidence_weighted_denominator'
        ] = _to_float(
            base_factor_population_info["pair_weighted_denominator"]
        )
        train_info[
            'base_factor_non_pair_weighted_denominator'
        ] = _to_float(
            base_factor_population_info["non_pair_weighted_denominator"]
        )
        train_info[
            'base_factor_pair_evidence_numerator'
        ] = _to_float(base_factor_population_info["pair_numerator"])
        train_info[
            'base_factor_non_pair_numerator'
        ] = _to_float(base_factor_population_info["non_pair_numerator"])
        for population_name in ("pair", "non_pair"):
            train_info[
                'base_factor_{}_episode_count'.format(population_name)
            ] = _to_float(
                base_factor_population_info[
                    "{}_episode_count".format(population_name)
                ]
            )
            for statistic_name in ("mean", "min", "max"):
                train_info[
                    'base_factor_{}_episode_weight_{}'.format(
                        population_name,
                        statistic_name,
                    )
                ] = _to_float(
                    base_factor_population_info[
                        "{}_episode_weight_{}".format(
                            population_name,
                            statistic_name,
                        )
                    ]
                )
        train_info[
            'base_factor_population_split_contract_valid'
        ] = float(base_factor_population_info["contract_valid"])
        train_info['pair_pursuit_factor_loss_contribution'] = _to_float(
            pair_pursuit_factor_loss_contribution
        )
        train_info[
            'pair_pursuit_positive_factor_loss_contribution'
        ] = _to_float(pair_pursuit_positive_factor_loss_contribution)
        train_info[
            'pair_pursuit_negative_factor_loss_contribution'
        ] = _to_float(pair_pursuit_negative_factor_loss_contribution)
        train_info['pair_pursuit_factor_loss_target_count'] = _to_float(
            pair_pursuit_factor_loss_target_count
        )
        train_info[
            'pair_pursuit_factor_loss_target_transition_count'
        ] = _to_float(pair_pursuit_factor_loss_target_transition_count)
        train_info[
            'pair_pursuit_factor_loss_valid_transition_count'
        ] = _to_float(pair_pursuit_factor_loss_valid_transition_count)
        train_info[
            'pair_pursuit_factor_loss_factors_per_transition'
        ] = _to_float(pair_pursuit_factor_loss_factors_per_transition)
        train_info[
            'pair_pursuit_factor_loss_normalization_denominator'
        ] = _to_float(pair_pursuit_factor_loss_normalization_denominator)
        train_info[
            'pair_pursuit_factor_loss_normalization_version'
        ] = float(pair_pursuit_factor_loss_normalization_version)
        train_info['pair_pursuit_factor_local_objective'] = 1.0
        train_info['pair_optimizer_positive_mass'] = _to_float(
            pair_optimizer_positive_mass
        )
        train_info['pair_optimizer_negative_mass'] = _to_float(
            pair_optimizer_negative_mass
        )
        train_info['pair_pending_effective_positive_mass'] = _to_float(
            pair_effective_positive_mass
        )
        train_info['pair_pending_effective_negative_mass'] = _to_float(
            pair_effective_negative_mass
        )
        train_info['pair_pending_positive_stale_trust_mean'] = _to_float(
            pair_positive_trust_mean
        )
        train_info['pair_pending_negative_stale_trust_mean'] = _to_float(
            pair_negative_trust_mean
        )
        train_info['pair_optimizer_centered_error'] = _to_float(
            pair_optimizer_centered_error
        )
        train_info['pair_optimizer_class_complete'] = float(
            pair_optimizer_class_complete
        )
        train_info['pair_optimizer_contract_valid'] = float(
            pair_optimizer_contract_valid
        )
        train_info['pair_gradient_diagnostic_version'] = 2.0
        train_info['pair_gradient_target_bearing_update'] = float(
            pair_diagnostic_target_present
        )
        train_info['pair_gradient_norm'] = _to_float(pair_gradient_norm)
        train_info['pair_raw_gradient_norm'] = _to_float(
            pair_raw_gradient_norm
        )
        train_info['pair_raw_gradient_zero'] = _to_float(
            pair_raw_gradient_zero
        )
        train_info['pair_zero_aggregate_recovered'] = _to_float(
            pair_zero_aggregate_recovered
        )
        train_info['pair_zero_aggregate_recovered_norm'] = _to_float(
            pair_zero_aggregate_recovered_norm
        )
        train_info['pair_zero_aggregate_recovered_min_dot'] = _to_float(
            pair_zero_aggregate_recovered_min_dot
        )
        train_info['pair_base_factor_gradient_norm'] = _to_float(
            pair_base_factor_gradient_norm
        )
        train_info['pair_base_factor_gradient_dot'] = _to_float(
            pair_base_factor_gradient_dot
        )
        train_info['pair_base_factor_gradient_cosine'] = _to_float(
            pair_base_factor_gradient_cosine
        )
        train_info['pair_actual_update_norm'] = _to_float(
            pair_actual_update_norm
        )
        train_info['pair_actual_update_descent_dot'] = _to_float(
            pair_actual_update_descent_dot
        )
        train_info['pair_actual_update_descent_cosine'] = _to_float(
            pair_actual_update_descent_cosine
        )
        train_info['pair_target_score_signed_change_mean'] = _to_float(
            pair_target_score_signed_change_mean
        )
        train_info['pair_target_score_positive_change_mean'] = _to_float(
            pair_target_score_positive_change_mean
        )
        train_info[
            'pair_target_score_negative_signed_change_mean'
        ] = _to_float(pair_target_score_negative_signed_change_mean)
        train_info[
            'pair_optimizer_transaction_diagnostic_version'
        ] = float(PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION)
        train_info['pair_optimizer_transaction_support_version'] = 6.0
        train_info['pair_optimizer_transaction_enabled'] = float(
            transaction_diagnostics_enabled
        )
        train_info['pair_optimizer_transaction_nonzero_pair'] = float(
            pair_diagnostic_target_present
        )
        train_info['pair_optimizer_transaction_pair_scalar_loss'] = _to_float(
            pair_pursuit_factor_loss_contribution
        )
        train_info[
            'pair_optimizer_transaction_base_factor_scalar_loss'
        ] = _to_float(base_factor_rl_loss)
        train_info[
            'pair_optimizer_transaction_total_adjacency_loss'
        ] = _to_float(loss)
        for objective_name in PAIR_OPTIMIZER_OBJECTIVE_NAMES:
            objective_info = objective_gradient_diagnostics[objective_name]
            objective_prefix = (
                'pair_optimizer_transaction_objective_{}_'
                .format(objective_name)
            )
            train_info[objective_prefix + 'active'] = _to_float(
                objective_info["active"]
            )
            train_info[objective_prefix + 'scalar_loss'] = _to_float(
                objective_info["scalar_loss"]
            )
            train_info[objective_prefix + 'grad_norm'] = _to_float(
                objective_info["grad_norm"]
            )
            train_info[objective_prefix + 'pair_grad_dot'] = _to_float(
                objective_info["pair_dot"]
            )
            train_info[objective_prefix + 'pair_grad_cosine'] = _to_float(
                objective_info["pair_cosine"]
            )
            train_info[
                objective_prefix + 'pair_descent_component'
            ] = _to_float(objective_info["pair_descent_component"])
        train_info[
            'pair_optimizer_transaction_objective_scalar_reconstruction_error'
        ] = _to_float(objective_scalar_reconstruction_error)
        train_info[
            'pair_optimizer_transaction_objective_scalar_reconstruction_valid'
        ] = _to_float(objective_scalar_reconstruction_valid)
        train_info[
            'pair_optimizer_transaction_all_objectives_independent_grad_sum_norm'
        ] = _to_float(all_objectives_independent_grad_sum_norm)
        train_info[
            'pair_optimizer_transaction_pair_independent_grad_sum_dot'
        ] = _to_float(pair_independent_grad_sum_dot)
        train_info[
            'pair_optimizer_transaction_pair_independent_grad_sum_cosine'
        ] = _to_float(pair_independent_grad_sum_cosine)
        train_info[
            'pair_optimizer_transaction_raw_combined_grad_norm_from_backward'
        ] = _to_float(raw_combined_grad_norm_from_backward)
        train_info[
            'pair_optimizer_transaction_independent_sum_vs_raw_combined_delta_norm'
        ] = _to_float(independent_sum_vs_raw_combined_delta_norm)
        train_info[
            'pair_optimizer_transaction_independent_sum_vs_raw_combined_relative_error'
        ] = _to_float(independent_sum_vs_raw_combined_relative_error)
        train_info[
            'pair_optimizer_transaction_independent_sum_reconstruction_valid'
        ] = _to_float(independent_sum_reconstruction_valid)
        train_info[
            'pair_optimizer_transaction_pre_projection_combined_grad_norm'
        ] = _to_float(pre_projection_combined_grad_norm)
        train_info[
            'pair_optimizer_transaction_post_projection_combined_grad_norm'
        ] = _to_float(post_projection_combined_grad_norm)
        train_info[
            'pair_optimizer_transaction_pair_pre_projection_dot'
        ] = _to_float(pair_pre_projection_dot)
        train_info[
            'pair_optimizer_transaction_pair_pre_projection_cosine'
        ] = _to_float(pair_pre_projection_cosine)
        train_info[
            'pair_optimizer_transaction_pair_post_projection_dot'
        ] = _to_float(pair_post_projection_dot)
        train_info[
            'pair_optimizer_transaction_pair_post_projection_cosine'
        ] = _to_float(pair_post_projection_cosine)
        train_info[
            'pair_optimizer_transaction_projection_delta_norm'
        ] = _to_float(projection_delta_norm)
        train_info[
            'pair_optimizer_transaction_pair_projection_delta_dot'
        ] = _to_float(pair_projection_delta_dot)
        train_info[
            'pair_optimizer_transaction_pair_projection_delta_cosine'
        ] = _to_float(pair_projection_delta_cosine)
        train_info[
            'pair_optimizer_transaction_gradient_projection_intervened'
        ] = _to_float(gradient_projection_intervened)
        train_info[
            'pair_optimizer_transaction_standard_pair_gradient_projection_intervened'
        ] = _to_float(standard_pair_gradient_projection_intervened)
        train_info[
            'pair_optimizer_transaction_pair_target_gradient_constraint_count'
        ] = _to_float(pair_target_gradient_constraint_count)
        train_info[
            'pair_optimizer_transaction_pair_target_gradient_projection_intervened'
        ] = _to_float(pair_target_gradient_projection_intervened)
        train_info[
            'pair_optimizer_transaction_pair_target_gradient_min_dot_before'
        ] = _to_float(pair_target_gradient_min_dot_before)
        train_info[
            'pair_optimizer_transaction_pair_target_gradient_min_dot_after'
        ] = _to_float(pair_target_gradient_min_dot_after)
        train_info[
            'pair_optimizer_transaction_pair_target_gradient_projection_delta_norm'
        ] = _to_float(pair_target_gradient_projection_delta_norm)
        train_info['pair_optimizer_transaction_pair_grad_norm'] = _to_float(
            pair_gradient_norm
        )
        train_info['pair_optimizer_transaction_pair_grad_finite'] = _to_float(
            pair_gradient_finite
        )
        train_info['pair_optimizer_transaction_pair_grad_zero'] = _to_float(
            pair_gradient_zero
        )
        train_info[
            'pair_optimizer_transaction_pair_raw_grad_norm'
        ] = _to_float(pair_raw_gradient_norm)
        train_info[
            'pair_optimizer_transaction_pair_raw_grad_zero'
        ] = _to_float(pair_raw_gradient_zero)
        train_info[
            'pair_optimizer_transaction_pair_zero_aggregate_recovered'
        ] = _to_float(pair_zero_aggregate_recovered)
        train_info[
            'pair_optimizer_transaction_pair_zero_aggregate_recovered_norm'
        ] = _to_float(pair_zero_aggregate_recovered_norm)
        train_info[
            'pair_optimizer_transaction_pair_zero_aggregate_recovered_min_dot'
        ] = _to_float(pair_zero_aggregate_recovered_min_dot)
        train_info[
            'pair_optimizer_transaction_base_factor_grad_norm'
        ] = _to_float(pair_base_factor_gradient_norm)
        train_info['pair_optimizer_transaction_pair_base_grad_dot'] = _to_float(
            pair_base_factor_gradient_dot
        )
        train_info[
            'pair_optimizer_transaction_pair_base_grad_cosine'
        ] = _to_float(pair_base_factor_gradient_cosine)
        train_info[
            'pair_optimizer_transaction_combined_grad_norm_preclip'
        ] = _to_float(combined_grad_norm_preclip)
        train_info[
            'pair_optimizer_transaction_pair_combined_grad_dot_preclip'
        ] = _to_float(pair_combined_grad_dot_preclip)
        train_info[
            'pair_optimizer_transaction_pair_combined_grad_cosine_preclip'
        ] = _to_float(pair_combined_grad_cosine_preclip)
        train_info[
            'pair_optimizer_transaction_pair_combined_descent_component_preclip'
        ] = _to_float(pair_combined_descent_component_preclip)
        train_info[
            'pair_optimizer_transaction_combined_grad_norm_postclip'
        ] = _to_float(combined_grad_norm_postclip)
        train_info[
            'pair_optimizer_transaction_pair_combined_grad_dot_postclip'
        ] = _to_float(pair_combined_grad_dot_postclip)
        train_info[
            'pair_optimizer_transaction_pair_combined_grad_cosine_postclip'
        ] = _to_float(pair_combined_grad_cosine_postclip)
        train_info[
            'pair_optimizer_transaction_pair_combined_descent_component_postclip'
        ] = _to_float(pair_combined_descent_component_postclip)
        train_info[
            'pair_optimizer_transaction_gradient_clip_reported_preclip_norm'
        ] = _to_float(grad_norm)
        train_info[
            'pair_optimizer_transaction_gradient_clip_applied'
        ] = _to_float(gradient_clip_applied)
        train_info['pair_optimizer_transaction_gradient_clip_scale'] = _to_float(
            gradient_clip_scale
        )
        train_info['pair_optimizer_transaction_adam_exp_avg_norm'] = _to_float(
            adam_exp_avg_norm
        )
        train_info[
            'pair_optimizer_transaction_adam_exp_avg_sq_norm'
        ] = _to_float(adam_exp_avg_sq_norm)
        train_info[
            'pair_optimizer_transaction_adam_exp_avg_sq_sqrt_sum'
        ] = _to_float(adam_exp_avg_sq_sqrt_sum)
        train_info[
            'pair_optimizer_transaction_adam_exp_avg_pair_dot'
        ] = _to_float(adam_exp_avg_pair_dot)
        train_info[
            'pair_optimizer_transaction_adam_exp_avg_pair_cosine'
        ] = _to_float(adam_exp_avg_pair_cosine)
        train_info[
            'pair_optimizer_transaction_optimizer_step_before'
        ] = float(optimizer_step_before)
        train_info[
            'pair_optimizer_transaction_optimizer_step_before_min'
        ] = float(optimizer_step_before_min)
        train_info[
            'pair_optimizer_transaction_optimizer_step_before_max'
        ] = float(optimizer_step_before_max)
        train_info[
            'pair_optimizer_transaction_optimizer_step_after'
        ] = float(optimizer_step_after)
        train_info[
            'pair_optimizer_transaction_optimizer_step_after_min'
        ] = float(optimizer_step_after_min)
        train_info[
            'pair_optimizer_transaction_optimizer_step_after_max'
        ] = float(optimizer_step_after_max)
        train_info['pair_optimizer_transaction_learning_rate'] = float(
            transaction_learning_rate
        )
        train_info['pair_optimizer_transaction_adam_beta1'] = float(
            transaction_adam_beta1
        )
        train_info['pair_optimizer_transaction_adam_beta2'] = float(
            transaction_adam_beta2
        )
        train_info['pair_optimizer_transaction_adam_eps'] = float(
            transaction_adam_eps
        )
        train_info['pair_optimizer_transaction_adam_weight_decay'] = float(
            transaction_adam_weight_decay
        )
        train_info['pair_optimizer_transaction_adam_amsgrad'] = float(
            transaction_adam_amsgrad
        )
        train_info[
            'pair_optimizer_transaction_adam_raw_displacement_norm'
        ] = _to_float(adam_raw_displacement_norm)
        train_info['pair_optimizer_transaction_adam_raw_pair_dot'] = _to_float(
            adam_raw_pair_dot
        )
        train_info[
            'pair_optimizer_transaction_adam_raw_pair_descent_dot'
        ] = _to_float(adam_raw_pair_descent_dot)
        train_info[
            'pair_optimizer_transaction_adam_raw_pair_descent_cosine'
        ] = _to_float(adam_raw_pair_descent_cosine)
        train_info[
            'pair_optimizer_transaction_final_displacement_norm'
        ] = _to_float(final_displacement_norm)
        train_info['pair_optimizer_transaction_final_pair_dot'] = _to_float(
            final_pair_dot
        )
        train_info[
            'pair_optimizer_transaction_final_pair_descent_dot'
        ] = _to_float(final_pair_descent_dot)
        train_info[
            'pair_optimizer_transaction_final_pair_descent_cosine'
        ] = _to_float(final_pair_descent_cosine)
        train_info[
            'pair_optimizer_transaction_adam_to_final_displacement_delta_norm'
        ] = _to_float(adam_to_final_displacement_delta_norm)
        train_info[
            'pair_optimizer_transaction_final_parameters_equal_raw_adam'
        ] = _to_float(final_parameters_equal_raw_adam)
        train_info[
            'pair_optimizer_transaction_pair_optimizer_isolated'
        ] = _to_float(pair_optimizer_isolated)
        train_info[
            'pair_optimizer_transaction_pair_actual_update_direction_guard_applied'
        ] = _to_float(pair_actual_update_direction_guard_applied)
        train_info[
            'pair_optimizer_transaction_pair_optimizer_state_sync_applied'
        ] = _to_float(pair_optimizer_state_sync_applied)
        train_info[
            'pair_optimizer_transaction_pair_target_actual_direction_guard_applied'
        ] = _to_float(pair_target_actual_direction_guard_applied)
        train_info[
            'pair_optimizer_transaction_pair_target_actual_min_descent_dot_before'
        ] = _to_float(pair_target_actual_min_descent_dot_before)
        train_info[
            'pair_optimizer_transaction_pair_target_actual_min_descent_dot_after'
        ] = _to_float(pair_target_actual_min_descent_dot_after)
        train_info[
            'pair_optimizer_transaction_pair_target_optimizer_state_sync_applied'
        ] = _to_float(pair_target_optimizer_state_sync_applied)
        train_info[
            'pair_optimizer_transaction_pair_boundary_diagnostic_version'
        ] = float(PAIR_SELECTION_BOUNDARY_DIAGNOSTIC_VERSION)
        train_info[
            'pair_optimizer_transaction_pair_boundary_gradient_constraint_count'
        ] = _to_float(pair_boundary_gradient_constraint_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_gradient_projection_intervened'
        ] = _to_float(pair_boundary_gradient_projection_intervened)
        train_info[
            'pair_optimizer_transaction_pair_boundary_gradient_min_dot_before'
        ] = _to_float(pair_boundary_gradient_min_dot_before)
        train_info[
            'pair_optimizer_transaction_pair_boundary_gradient_min_dot_after'
        ] = _to_float(pair_boundary_gradient_min_dot_after)
        train_info[
            'pair_optimizer_transaction_pair_boundary_actual_direction_guard_applied'
        ] = _to_float(pair_boundary_actual_direction_guard_applied)
        train_info[
            'pair_optimizer_transaction_pair_boundary_actual_min_descent_dot_before'
        ] = _to_float(pair_boundary_actual_min_descent_dot_before)
        train_info[
            'pair_optimizer_transaction_pair_boundary_actual_min_descent_dot_after'
        ] = _to_float(pair_boundary_actual_min_descent_dot_after)
        train_info[
            'pair_optimizer_transaction_pair_boundary_target_count'
        ] = _to_float(pair_boundary_target_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_correct_direction_count'
        ] = _to_float(pair_boundary_correct_direction_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_reverse_direction_count'
        ] = _to_float(pair_boundary_reverse_direction_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_approximately_zero_count'
        ] = _to_float(pair_boundary_approximately_zero_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_signed_margin_change_mean'
        ] = _to_float(pair_boundary_signed_margin_change_mean)
        train_info[
            'pair_optimizer_transaction_pair_boundary_signed_margin_change_median'
        ] = _to_float(pair_boundary_signed_margin_change_median)
        train_info[
            'pair_optimizer_transaction_pair_boundary_signed_margin_change_worst'
        ] = _to_float(pair_boundary_signed_margin_change_worst)
        train_info[
            'pair_optimizer_transaction_pair_boundary_rank_crossing_count'
        ] = _to_float(pair_boundary_rank_crossing_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_positive_promotion_count'
        ] = _to_float(pair_boundary_positive_promotion_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_negative_eviction_count'
        ] = _to_float(pair_boundary_negative_eviction_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_nonlinear_backtrack_count'
        ] = _to_float(pair_boundary_nonlinear_backtrack_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_nonlinear_backtrack_final_scale'
        ] = _to_float(pair_boundary_nonlinear_backtrack_final_scale)
        train_info[
            'pair_optimizer_transaction_pair_boundary_nonlinear_refinement_count'
        ] = _to_float(pair_boundary_nonlinear_refinement_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_nonlinear_invalid_upper_scale'
        ] = _to_float(pair_boundary_nonlinear_invalid_upper_scale)
        train_info[
            'pair_optimizer_transaction_pair_boundary_direction_candidate_count'
        ] = _to_float(pair_boundary_direction_candidate_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_direction_valid_candidate_count'
        ] = _to_float(pair_boundary_direction_valid_candidate_count)
        train_info[
            'pair_optimizer_transaction_pair_boundary_selected_progress_floor_fraction'
        ] = _to_float(
            pair_boundary_selected_progress_floor_fraction
        )
        train_info[
            'pair_optimizer_transaction_pair_boundary_progress_min_completion'
        ] = _to_float(pair_boundary_progress_min_completion)
        train_info[
            'pair_optimizer_transaction_pair_boundary_progress_mean_completion'
        ] = _to_float(pair_boundary_progress_mean_completion)
        train_info[
            'pair_optimizer_transaction_pair_boundary_limiting_constraint_code'
        ] = _to_float(pair_boundary_limiting_constraint_code)
        train_info[
            'pair_optimizer_transaction_pair_boundary_limiting_target_ordinal'
        ] = _to_float(pair_boundary_limiting_target_ordinal)
        train_info[
            'pair_optimizer_transaction_pair_boundary_joint_candidate_exact_valid'
        ] = _to_float(pair_boundary_joint_candidate_exact_valid)
        train_info[
            'pair_optimizer_transaction_pair_boundary_joint_lifecycle_exact_valid'
        ] = _to_float(pair_boundary_joint_lifecycle_exact_valid)
        train_info[
            'pair_optimizer_transaction_candidate_gradient_projection_intervened'
        ] = _to_float(candidate_gradient_conflict)
        train_info[
            'pair_optimizer_transaction_candidate_actual_update_correction_intervened'
        ] = _to_float(candidate_actual_update_corrected)
        train_info[
            'pair_optimizer_transaction_lifecycle_gradient_projection_intervened'
        ] = _to_float(lifecycle_gradient_conflict)
        train_info[
            'pair_optimizer_transaction_lifecycle_actual_update_correction_intervened'
        ] = _to_float(lifecycle_actual_projection_corrected)
        train_info[
            'pair_optimizer_transaction_lifecycle_current_priority_repair_intervened'
        ] = _to_float(lifecycle_current_priority_repair_intervened)
        train_info[
            'pair_optimizer_transaction_lifecycle_current_priority_min_dot_before'
        ] = _to_float(lifecycle_current_priority_min_dot_before)
        train_info[
            'pair_optimizer_transaction_lifecycle_current_priority_min_dot_after'
        ] = _to_float(lifecycle_current_priority_min_dot_after)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_linear_min_dot'
        ] = _to_float(lifecycle_final_linear_min_dot)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_linear_max_tolerance'
        ] = _to_float(lifecycle_final_linear_max_tolerance)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_linear_max_normalized_violation'
        ] = _to_float(lifecycle_final_linear_max_normalized_violation)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_linear_rounding_residual_count'
        ] = _to_float(lifecycle_final_linear_rounding_residual_count)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_exact_revalidation_valid'
        ] = _to_float(lifecycle_final_exact_revalidation_valid)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_exact_min_signed_gap'
        ] = _to_float(lifecycle_final_exact_min_signed_gap)
        train_info[
            'pair_optimizer_transaction_lifecycle_final_exact_max_tolerance'
        ] = _to_float(lifecycle_final_exact_max_tolerance)
        train_info[
            'pair_optimizer_transaction_lifecycle_backtrack_count'
        ] = _to_float(lifecycle_nonlinear_backtrack_count)
        train_info[
            'pair_optimizer_transaction_lifecycle_reject_occurred'
        ] = _to_float(lifecycle_update_rejected)
        train_info['pair_optimizer_transaction_rollback_occurred'] = _to_float(
            lifecycle_update_rejected
        )
        train_info[
            'pair_optimizer_transaction_score_signed_change_mean'
        ] = _to_float(pair_target_score_signed_change_mean)
        train_info[
            'pair_optimizer_transaction_positive_score_change_mean'
        ] = _to_float(pair_target_score_positive_change_mean)
        train_info[
            'pair_optimizer_transaction_negative_signed_score_change_mean'
        ] = _to_float(pair_target_score_negative_signed_change_mean)
        train_info[
            'pair_optimizer_transaction_positive_target_count'
        ] = _to_float(pair_target_score_positive_target_count)
        train_info[
            'pair_optimizer_transaction_negative_target_count'
        ] = _to_float(pair_target_score_negative_target_count)
        train_info[
            'pair_optimizer_transaction_score_correct_direction_target_count'
        ] = _to_float(pair_target_score_correct_direction_count)
        train_info[
            'pair_optimizer_transaction_score_reverse_direction_target_count'
        ] = _to_float(pair_target_score_reverse_direction_count)
        train_info[
            'pair_optimizer_transaction_score_approximately_zero_target_count'
        ] = _to_float(pair_target_score_approximately_zero_count)
        train_info[
            'pair_optimizer_transaction_score_before_after_join_valid'
        ] = _to_float(pair_target_score_before_after_join_valid)
        train_info[
            'pair_optimizer_transaction_score_zero_tolerance'
        ] = _to_float(pair_target_score_zero_tolerance)
        train_info['capture_candidate_identity_loss_contribution'] = _to_float(
            capture_candidate_identity_loss_contribution
        )
        train_info[
            'capture_candidate_identity_positive_loss_contribution'
        ] = _to_float(
            capture_candidate_identity_positive_loss_contribution
        )
        train_info[
            'capture_candidate_identity_negative_loss_contribution'
        ] = _to_float(
            capture_candidate_identity_negative_loss_contribution
        )
        train_info['capture_candidate_identity_target_count'] = _to_float(
            capture_candidate_identity_target_count
        )
        train_info[
            'capture_candidate_identity_valid_transition_count'
        ] = _to_float(capture_candidate_identity_valid_transition_count)
        train_info[
            'capture_candidate_identity_target_transition_count'
        ] = _to_float(capture_candidate_identity_target_transition_count)
        train_info[
            'capture_candidate_identity_unsatisfied_target_count'
        ] = _to_float(capture_candidate_identity_unsatisfied_target_count)
        train_info[
            'capture_candidate_identity_positive_unsatisfied_target_count'
        ] = _to_float(
            capture_candidate_identity_positive_unsatisfied_target_count
        )
        train_info[
            'capture_candidate_identity_negative_unsatisfied_target_count'
        ] = _to_float(
            capture_candidate_identity_negative_unsatisfied_target_count
        )
        train_info[
            'capture_candidate_identity_target_transition_fraction'
        ] = _to_float(capture_candidate_identity_target_transition_fraction)
        train_info['capture_candidate_identity_positive_mass'] = _to_float(
            capture_candidate_identity_positive_mass
        )
        train_info['capture_candidate_identity_negative_mass'] = _to_float(
            capture_candidate_identity_negative_mass
        )
        train_info['capture_candidate_identity_positive_margin_mean'] = _to_float(
            capture_candidate_identity_positive_margin_mean
        )
        train_info['capture_candidate_identity_negative_margin_mean'] = _to_float(
            capture_candidate_identity_negative_margin_mean
        )
        train_info[
            'capture_candidate_identity_positive_signed_margin_mean'
        ] = _to_float(capture_candidate_identity_positive_signed_margin_mean)
        train_info[
            'capture_candidate_identity_negative_signed_margin_mean'
        ] = _to_float(capture_candidate_identity_negative_signed_margin_mean)
        # Version 10 uses the first reachable slot's hardest legal competitor
        # and stops auxiliary optimization after the signed goal is achieved.
        train_info['capture_candidate_identity_loss_definition_version'] = 12.0
        train_info[
            'capture_candidate_identity_loss_normalization_version'
        ] = 3.0
        train_info[
            'capture_candidate_identity_gradient_projection_version'
        ] = 1.0
        train_info[
            'capture_candidate_identity_actual_update_guard_version'
        ] = 2.0
        train_info[
            'capture_candidate_identity_optimizer_state_sync_version'
        ] = 3.0
        train_info[
            'capture_candidate_identity_gradient_norm'
        ] = _to_float(candidate_gradient_norm)
        train_info[
            'capture_candidate_identity_base_gradient_norm'
        ] = _to_float(candidate_base_gradient_norm)
        train_info[
            'capture_candidate_identity_base_gradient_cosine'
        ] = _to_float(candidate_base_gradient_cosine)
        train_info[
            'capture_candidate_identity_gradient_conflict'
        ] = _to_float(candidate_gradient_conflict)
        train_info[
            'capture_candidate_identity_projected_gradient_dot'
        ] = _to_float(candidate_projected_gradient_dot)
        train_info[
            'capture_candidate_identity_total_gradient_norm_ratio'
        ] = _to_float(candidate_total_gradient_norm_ratio)
        train_info[
            'capture_candidate_identity_base_gradient_removed_norm_fraction'
        ] = _to_float(candidate_base_gradient_removed_norm_fraction)
        train_info[
            'capture_candidate_identity_clipped_gradient_dot'
        ] = _to_float(candidate_clipped_gradient_dot)
        train_info[
            'capture_candidate_identity_actual_update_descent_dot_before'
        ] = _to_float(candidate_actual_update_descent_dot_before)
        train_info[
            'capture_candidate_identity_actual_update_descent_dot_after'
        ] = _to_float(candidate_actual_update_descent_dot_after)
        train_info[
            'capture_candidate_identity_actual_update_corrected'
        ] = _to_float(candidate_actual_update_corrected)
        train_info[
            'capture_candidate_identity_actual_update_norm'
        ] = _to_float(candidate_actual_update_norm)
        train_info[
            'capture_candidate_identity_actual_update_correction_norm'
        ] = _to_float(candidate_actual_update_correction_norm)
        train_info[
            'capture_candidate_identity_actual_update_correction_norm_ratio'
        ] = _to_float(candidate_actual_update_correction_norm_ratio)
        train_info[
            'capture_candidate_identity_optimizer_state_sync_applied'
        ] = _to_float(candidate_optimizer_state_sync_applied)
        train_info[
            'capture_candidate_identity_optimizer_state_sync_parameter_count'
        ] = _to_float(candidate_optimizer_state_sync_parameter_count)
        train_info[
            'capture_candidate_identity_optimizer_state_update_equation_version'
        ] = _to_float(candidate_optimizer_state_update_equation_version)
        train_info[
            'capture_candidate_identity_optimizer_state_raw_reconstruction_error'
        ] = _to_float(candidate_optimizer_state_raw_reconstruction_error)
        train_info[
            'capture_candidate_identity_optimizer_state_safe_reconstruction_error'
        ] = _to_float(candidate_optimizer_state_safe_reconstruction_error)
        train_info[
            'capture_candidate_identity_optimizer_state_raw_reconstruction_error_ratio'
        ] = _to_float(
            candidate_optimizer_state_raw_reconstruction_error_ratio
        )
        train_info[
            'capture_candidate_identity_optimizer_state_safe_reconstruction_error_ratio'
        ] = _to_float(
            candidate_optimizer_state_safe_reconstruction_error_ratio
        )
        train_info[
            'capture_candidate_identity_optimizer_state_reconstruction_tolerance'
        ] = _to_float(candidate_optimizer_state_reconstruction_tolerance)
        train_info[
            'capture_candidate_identity_optimizer_state_exp_avg_change_norm'
        ] = _to_float(candidate_optimizer_state_exp_avg_change_norm)
        train_info[
            'capture_candidate_identity_loss_optimizer_change'
        ] = _to_float(candidate_loss_optimizer_change)
        train_info['capture_candidate_identity_lifecycle_version'] = 10.0
        train_info['candidate_residual_only'] = float(
            bool(candidate_residual_only)
        )
        train_info['candidate_residual_optimizer_isolated'] = float(
            bool(candidate_residual_only)
        )
        train_info[
            'candidate_residual_inactive_parameter_count'
        ] = _to_float(residual_inactive_parameter_count)
        train_info[
            'candidate_residual_inactive_parameter_update_norm'
        ] = _to_float(residual_inactive_parameter_update_norm)
        train_info['candidate_residual_skipped_satisfied'] = 0.0
        train_info[
            'capture_candidate_identity_lifecycle_horizon'
        ] = float(self.candidate_identity_lifecycle_horizon)
        train_info[
            'capture_candidate_identity_lifecycle_cache_size'
        ] = float(lifecycle_cache_size)
        train_info[
            'capture_candidate_identity_lifecycle_observation_archive_size'
        ] = float(lifecycle_observation_archive_size)
        train_info[
            'capture_candidate_identity_lifecycle_new_count'
        ] = float(lifecycle_new_count)
        train_info[
            'capture_candidate_identity_lifecycle_behavioral_progress_transition_count'
        ] = float(lifecycle_behavioral_progress_transition_count)
        train_info[
            'capture_candidate_identity_lifecycle_no_progress_skipped_transition_count'
        ] = float(lifecycle_no_progress_skipped_transition_count)
        train_info[
            'capture_candidate_identity_lifecycle_duplicate_prevented_count'
        ] = float(lifecycle_duplicate_prevented_count)
        train_info[
            'capture_candidate_identity_lifecycle_expired_count'
        ] = float(lifecycle_expired_count)
        train_info[
            'capture_candidate_identity_lifecycle_protected_target_count'
        ] = _to_float(lifecycle_protected_target_count)
        train_info[
            'capture_candidate_identity_lifecycle_gradient_norm'
        ] = _to_float(lifecycle_gradient_norm)
        train_info[
            'capture_candidate_identity_lifecycle_base_gradient_cosine'
        ] = _to_float(lifecycle_base_gradient_cosine)
        train_info[
            'capture_candidate_identity_lifecycle_gradient_conflict'
        ] = _to_float(lifecycle_gradient_conflict)
        train_info[
            'capture_candidate_identity_lifecycle_projected_gradient_dot'
        ] = _to_float(lifecycle_projected_gradient_dot)
        train_info[
            'capture_candidate_identity_lifecycle_actual_update_descent_dot_before'
        ] = _to_float(lifecycle_actual_update_descent_dot_before)
        train_info[
            'capture_candidate_identity_lifecycle_actual_update_descent_dot_after'
        ] = _to_float(lifecycle_actual_update_descent_dot_after)
        train_info[
            'capture_candidate_identity_lifecycle_actual_update_corrected'
        ] = _to_float(lifecycle_actual_update_corrected)
        train_info[
            'capture_candidate_identity_lifecycle_update_rejected'
        ] = _to_float(lifecycle_update_rejected)
        train_info[
            'capture_candidate_identity_lifecycle_violation_count'
        ] = _to_float(lifecycle_violation_count)
        train_info[
            'capture_candidate_identity_lifecycle_attempted_loss_optimizer_change'
        ] = _to_float(lifecycle_attempted_loss_optimizer_change)
        train_info[
            'capture_candidate_identity_lifecycle_loss_optimizer_change'
        ] = _to_float(lifecycle_loss_optimizer_change)
        train_info[
            'capture_candidate_identity_lifecycle_state_sync_applied'
        ] = _to_float(lifecycle_state_sync_applied)
        train_info[
            'capture_candidate_identity_lifecycle_age_mean'
        ] = _to_float(lifecycle_age_mean)
        train_info[
            'capture_candidate_identity_lifecycle_constraint_count'
        ] = _to_float(lifecycle_constraint_count)
        train_info[
            'capture_candidate_identity_lifecycle_active_constraint_count'
        ] = _to_float(lifecycle_active_constraint_count)
        train_info[
            'capture_candidate_identity_lifecycle_min_constraint_dot_before'
        ] = _to_float(lifecycle_min_constraint_dot_before)
        train_info[
            'capture_candidate_identity_lifecycle_min_constraint_dot_after'
        ] = _to_float(lifecycle_min_constraint_dot_after)
        train_info[
            'capture_candidate_identity_lifecycle_projection_fallback'
        ] = _to_float(lifecycle_projection_fallback)
        train_info[
            'capture_candidate_identity_lifecycle_superseded_constraint_count'
        ] = _to_float(lifecycle_superseded_constraint_count)
        train_info[
            'capture_candidate_identity_lifecycle_actual_min_constraint_dot_before'
        ] = _to_float(lifecycle_actual_min_constraint_dot_before)
        train_info[
            'capture_candidate_identity_lifecycle_actual_min_constraint_dot_after'
        ] = _to_float(lifecycle_actual_min_constraint_dot_after)
        train_info[
            'capture_candidate_identity_lifecycle_actual_negative_constraint_count_before'
        ] = _to_float(lifecycle_actual_negative_constraint_count_before)
        train_info[
            'capture_candidate_identity_lifecycle_actual_negative_constraint_count_after'
        ] = _to_float(lifecycle_actual_negative_constraint_count_after)
        train_info[
            'capture_candidate_identity_lifecycle_actual_projection_corrected'
        ] = _to_float(lifecycle_actual_projection_corrected)
        train_info[
            'capture_candidate_identity_lifecycle_actual_projection_correction_norm_ratio'
        ] = _to_float(lifecycle_actual_projection_correction_norm_ratio)
        train_info[
            'capture_candidate_identity_lifecycle_nonlinear_backtrack_count'
        ] = _to_float(lifecycle_nonlinear_backtrack_count)
        train_info[
            'capture_candidate_identity_lifecycle_current_candidate_nonlinear_violation'
        ] = _to_float(lifecycle_current_candidate_nonlinear_violation)
        train_info[
            'capture_candidate_identity_lifecycle_target_bearing_update'
        ] = _to_float(lifecycle_target_bearing_update)
        train_info[
            'capture_candidate_identity_lifecycle_policy_version_advanced'
        ] = _to_float(lifecycle_policy_version_advanced)
        train_info[
            'capture_candidate_identity_lifecycle_clock'
        ] = float(current_candidate_lifecycle_clock)
        for retention_age, retention_count, margin_fraction, rank_fraction in (
                (
                    1,
                    lifecycle_retention_1_count,
                    lifecycle_retention_1_fraction,
                    lifecycle_rank_retention_1_fraction,
                ),
                (
                    5,
                    lifecycle_retention_5_count,
                    lifecycle_retention_5_fraction,
                    lifecycle_rank_retention_5_fraction,
                ),
                (
                    10,
                    lifecycle_retention_10_count,
                    lifecycle_retention_10_fraction,
                    lifecycle_rank_retention_10_fraction,
                )):
            train_info[
                'capture_candidate_identity_lifecycle_{}_update_retention_count'
                .format(retention_age)
            ] = _to_float(retention_count)
            train_info[
                'capture_candidate_identity_lifecycle_{}_update_signed_margin_retention_fraction'
                .format(retention_age)
            ] = _to_float(margin_fraction)
            train_info[
                'capture_candidate_identity_lifecycle_{}_update_rank_retention_fraction'
                .format(retention_age)
            ] = _to_float(rank_fraction)
            held_counts = lifecycle_retention_held_counts[retention_age]
            for count_name, count_value in held_counts.items():
                train_info[
                    'capture_candidate_identity_lifecycle_{}_update_{}'
                    .format(retention_age, count_name)
                ] = _to_float(count_value)
        train_info[
            'capture_candidate_identity_positive_optimizer_signed_margin_change_mean'
        ] = _to_float(
            candidate_positive_optimizer_signed_margin_change
        )
        train_info[
            'capture_candidate_identity_negative_optimizer_signed_margin_change_mean'
        ] = _to_float(
            candidate_negative_optimizer_signed_margin_change
        )
        train_info[
            'capture_candidate_identity_positive_optimizer_rank_improved_fraction'
        ] = _to_float(
            candidate_positive_optimizer_rank_improved_fraction
        )
        train_info[
            'capture_candidate_identity_negative_optimizer_rank_reduced_fraction'
        ] = _to_float(
            candidate_negative_optimizer_rank_reduced_fraction
        )
        for diagnostic_name, diagnostic_value in (
                successful_candidate_capture_boundary.items()):
            train_info[
                'successful_candidate_capture_boundary_{}'.format(
                    diagnostic_name
                )
            ] = _to_float(diagnostic_value)
        train_info['capture_candidate_identity_score_semantics_version'] = 8.0
        train_info[
            'capture_candidate_identity_valid_margin_mean'
        ] = _to_float(capture_candidate_identity_valid_margin_mean)
        train_info[
            'capture_candidate_identity_valid_margin_min'
        ] = _to_float(capture_candidate_identity_valid_margin_min)
        train_info[
            'capture_candidate_identity_valid_margin_max'
        ] = _to_float(capture_candidate_identity_valid_margin_max)
        train_info[
            'capture_candidate_identity_behavior_margin_mean'
        ] = _to_float(
            capture_candidate_identity_behavior_margin_mean
        )
        train_info[
            'capture_candidate_identity_behavior_rank_mean'
        ] = _to_float(
            capture_candidate_identity_behavior_rank_mean
        )
        train_info[
            'capture_candidate_identity_positive_behavior_margin_mean'
        ] = _to_float(
            capture_candidate_identity_positive_behavior_margin_mean
        )
        train_info[
            'capture_candidate_identity_negative_behavior_margin_mean'
        ] = _to_float(
            capture_candidate_identity_negative_behavior_margin_mean
        )
        train_info[
            'capture_candidate_identity_positive_behavior_rank_mean'
        ] = _to_float(
            capture_candidate_identity_positive_behavior_rank_mean
        )
        train_info[
            'capture_candidate_identity_negative_behavior_rank_mean'
        ] = _to_float(
            capture_candidate_identity_negative_behavior_rank_mean
        )
        train_info[
            'capture_candidate_identity_positive_current_rank_mean'
        ] = _to_float(
            capture_candidate_identity_positive_current_rank_mean
        )
        train_info[
            'capture_candidate_identity_negative_current_rank_mean'
        ] = _to_float(
            capture_candidate_identity_negative_current_rank_mean
        )
        train_info[
            'capture_candidate_identity_positive_margin_change_mean'
        ] = _to_float(
            capture_candidate_identity_positive_margin_change_mean
        )
        train_info[
            'capture_candidate_identity_negative_margin_change_mean'
        ] = _to_float(
            capture_candidate_identity_negative_margin_change_mean
        )
        train_info[
            'capture_candidate_identity_positive_signed_margin_change_mean'
        ] = _to_float(
            capture_candidate_identity_positive_signed_margin_change_mean
        )
        train_info[
            'capture_candidate_identity_negative_signed_margin_change_mean'
        ] = _to_float(
            capture_candidate_identity_negative_signed_margin_change_mean
        )
        train_info[
            'capture_candidate_identity_positive_margin_improved_fraction'
        ] = _to_float(
            capture_candidate_identity_positive_margin_improved_fraction
        )
        train_info[
            'capture_candidate_identity_negative_margin_reduced_fraction'
        ] = _to_float(
            capture_candidate_identity_negative_margin_reduced_fraction
        )
        train_info[
            'capture_candidate_identity_positive_boundary_crossed_fraction'
        ] = _to_float(
            capture_candidate_identity_positive_boundary_crossed_fraction
        )
        train_info[
            'capture_candidate_identity_negative_boundary_respected_fraction'
        ] = _to_float(
            capture_candidate_identity_negative_boundary_respected_fraction
        )
        train_info[
            'capture_candidate_identity_positive_rank_improved_fraction'
        ] = _to_float(
            capture_candidate_identity_positive_rank_improved_fraction
        )
        train_info[
            'capture_candidate_identity_negative_rank_reduced_fraction'
        ] = _to_float(
            capture_candidate_identity_negative_rank_reduced_fraction
        )
        train_info[
            'capture_candidate_identity_behavior_valid_fraction'
        ] = _to_float(capture_candidate_identity_behavior_valid_fraction)
        train_info['capture_candidate_identity_policy_age_mean'] = _to_float(
            capture_candidate_identity_policy_age_mean
        )
        train_info['capture_candidate_identity_policy_age_max'] = _to_float(
            capture_candidate_identity_policy_age_max
        )
        train_info['capture_candidate_identity_policy_version'] = float(
            current_candidate_policy_version
        )
        train_info['capture_identity_factor_credit_active_fraction'] = _to_float(
            capture_to_win_triplet_credit_active_fraction
        )
        train_info['capture_identity_factor_credit_positive_mass'] = _to_float(
            capture_to_win_triplet_credit_positive_mass
        )
        train_info['capture_identity_factor_credit_negative_mass'] = _to_float(
            capture_to_win_triplet_credit_negative_mass
        )
        train_info['capture_outcome_identity_order2_delta_abs_mass'] = _to_float(
            capture_outcome_identity_order2_delta_abs_mass
        )
        train_info['capture_outcome_identity_order3_delta_abs_mass'] = _to_float(
            capture_outcome_identity_order3_delta_abs_mass
        )
        train_info['capture_identity_target_order2_fraction'] = _to_float(
            capture_identity_target_order2_fraction
        )
        train_info['capture_identity_target_order3_fraction'] = _to_float(
            capture_identity_target_order3_fraction
        )
        train_info['capture_outcome_non_target_delta_abs_max'] = _to_float(
            capture_outcome_non_target_delta_abs_max
        )
        train_info['capture_to_win_quality_gate_mean'] = _to_float(
            capture_to_win_quality_gate_mean
        )
        train_info['capture_to_win_quality_gate_abs_mean'] = _to_float(
            capture_to_win_quality_gate_abs_mean
        )
        train_info['capture_to_win_quality_gate_active_fraction'] = _to_float(
            capture_to_win_quality_gate_active_fraction
        )
        train_info['capture_to_win_quality_gate_positive_fraction'] = _to_float(
            capture_to_win_quality_gate_positive_fraction
        )
        train_info['capture_to_win_quality_gate_negative_fraction'] = _to_float(
            capture_to_win_quality_gate_negative_fraction
        )
        train_info['capture_to_win_quality_gate_positive_mass'] = _to_float(
            capture_to_win_quality_gate_positive_mass
        )
        train_info['capture_to_win_quality_gate_negative_mass'] = _to_float(
            capture_to_win_quality_gate_negative_mass
        )
        train_info['capture_to_win_outcome_contrastive'] = 1.0
        train_info['capture_to_win_quality_gate_definition_version'] = 5.0
        train_info['capture_outcome_baseline_mean'] = _to_float(
            capture_outcome_baseline_mean
        )
        train_info['capture_outcome_capture_episode_count_mean'] = _to_float(
            capture_outcome_capture_episode_count_mean
        )
        train_info['capture_outcome_success_episode_count_mean'] = _to_float(
            capture_outcome_success_episode_count_mean
        )
        train_info['capture_outcome_failure_episode_count_mean'] = _to_float(
            capture_outcome_failure_episode_count_mean
        )
        train_info['capture_outcome_mixed_window_fraction'] = _to_float(
            capture_outcome_mixed_window_fraction
        )
        train_info['capture_outcome_single_success_window_fraction'] = _to_float(
            capture_outcome_single_success_window_fraction
        )
        train_info['capture_outcome_single_failure_window_fraction'] = _to_float(
            capture_outcome_single_failure_window_fraction
        )
        train_info['capture_outcome_no_capture_window_fraction'] = _to_float(
            capture_outcome_no_capture_window_fraction
        )
        train_info['capture_outcome_triplet_labels_per_episode_mean'] = _to_float(
            capture_outcome_triplet_labels_per_episode_mean
        )
        train_info['capture_outcome_triplet_labels_per_episode_max'] = _to_float(
            capture_outcome_triplet_labels_per_episode_max
        )
        train_info['capture_outcome_raw_episode_advantage_mean'] = _to_float(
            capture_outcome_raw_episode_advantage_mean
        )
        train_info['capture_outcome_raw_episode_advantage_abs_mean'] = _to_float(
            capture_outcome_raw_episode_advantage_abs_mean
        )
        train_info['capture_outcome_window_raw_centered_mean'] = _to_float(
            capture_outcome_window_raw_centered_mean
        )
        train_info['capture_outcome_window_expanded_gate_sum'] = _to_float(
            capture_outcome_window_expanded_gate_sum
        )
        train_info['capture_outcome_window_expanded_gate_abs_sum'] = _to_float(
            capture_outcome_window_expanded_gate_abs_sum
        )
        train_info['capture_outcome_window_center_error_ratio'] = _to_float(
            capture_outcome_window_center_error_ratio
        )
        train_info['capture_to_win_credit_preclip_mean'] = _to_float(
            capture_to_win_credit_preclip_mean
        )
        train_info['capture_to_win_credit_preclip_std'] = _to_float(
            capture_to_win_credit_preclip_std
        )
        train_info['capture_to_win_credit_preclip_max'] = _to_float(
            capture_to_win_credit_preclip_max
        )
        train_info['capture_to_win_credit_preclip_min'] = _to_float(
            capture_to_win_credit_preclip_min
        )
        train_info['capture_to_win_credit_positive_clip_fraction'] = _to_float(
            capture_to_win_credit_positive_clip_fraction
        )
        train_info['capture_to_win_credit_negative_clip_fraction'] = _to_float(
            capture_to_win_credit_negative_clip_fraction
        )
        train_info['capture_identity_event_count_mean'] = _to_float(
            capture_identity_event_count_mean
        )
        train_info['capture_identity_matched_event_count_mean'] = _to_float(
            capture_identity_matched_event_count_mean
        )
        train_info['capture_identity_unmatched_event_count_mean'] = _to_float(
            capture_identity_unmatched_event_count_mean
        )
        train_info['capture_identity_candidate_factor_count_mean'] = _to_float(
            capture_identity_candidate_factor_count_mean
        )
        train_info['capture_identity_match_fraction'] = _to_float(
            capture_identity_match_fraction
        )
        train_info['capture_identity_candidates_per_matched_event'] = _to_float(
            capture_identity_candidates_per_matched_event
        )
        train_info['capture_outcome_label_gate_correlation'] = _to_float(
            capture_outcome_label_gate_correlation
        )
        train_info['capture_outcome_label_gate_correlation_valid'] = _to_float(
            capture_outcome_label_gate_correlation_valid
        )
        train_info['capture_outcome_success_labels_mean'] = _to_float(
            capture_outcome_success_labels_mean
        )
        train_info['capture_outcome_failure_labels_mean'] = _to_float(
            capture_outcome_failure_labels_mean
        )
        train_info['capture_outcome_success_gate_total_mean'] = _to_float(
            capture_outcome_success_gate_total_mean
        )
        train_info['capture_outcome_failure_gate_total_mean'] = _to_float(
            capture_outcome_failure_gate_total_mean
        )
        train_info['pair_pursuit_credit_mean'] = _to_float(
            pair_pursuit_credit_mean
        )
        train_info['pair_pursuit_credit_abs_mean'] = _to_float(
            pair_pursuit_credit_abs_mean
        )
        train_info['pair_pursuit_credit_positive_mass'] = _to_float(
            pair_pursuit_credit_positive_mass
        )
        train_info['pair_pursuit_credit_negative_mass'] = _to_float(
            pair_pursuit_credit_negative_mass
        )
        train_info['pair_pursuit_credit_outcome_conditioned'] = 1.0
        train_info['pair_pursuit_credit_active_fraction'] = _to_float(
            pair_pursuit_credit_active_fraction
        )
        train_info['pair_credit_active_fraction'] = _to_float(
            pair_pursuit_credit_active_fraction
        )
        train_info['pair_pursuit_credit_std'] = _to_float(
            pair_pursuit_credit_std
        )
        train_info['pair_pursuit_credit_max'] = _to_float(
            pair_pursuit_credit_max
        )
        train_info['pair_pursuit_credit_nonzero_count'] = _to_float(
            pair_pursuit_credit_nonzero_count
        )
        train_info['pair_credit_top1_mass_fraction'] = _to_float(
            pair_credit_top1_mass_fraction
        )
        train_info['pair_pursuit_quality_mean'] = _to_float(
            pair_pursuit_quality_mean
        )
        train_info['pair_pursuit_quality_active_fraction'] = _to_float(
            pair_pursuit_quality_active_fraction
        )
        train_info['pair_to_triplet_transition_score_mean'] = _to_float(
            pair_to_triplet_transition_score_mean
        )
        train_info[
            'pair_to_triplet_transition_active_fraction'
        ] = _to_float(pair_to_triplet_transition_active_fraction)
        train_info['triplet_capture_quality_mean'] = _to_float(
            triplet_capture_quality_mean
        )
        train_info['triplet_capture_quality_active_fraction'] = _to_float(
            triplet_capture_quality_active_fraction
        )
        train_info['transition_delay_mean'] = _to_float(
            transition_delay_mean
        )
        train_info['transition_delay_min'] = _to_float(
            transition_delay_min
        )
        train_info['transition_delay_max'] = _to_float(
            transition_delay_max
        )
        train_info['capture_event_count'] = _to_float(capture_count_total)
        train_info['capture_matched_count'] = _to_float(
            capture_matched_count_value
        )
        train_info['unmatched_capture_count'] = _to_float(
            unmatched_capture_count_value
        )
        train_info['capture_matched_fraction'] = _to_float(
            capture_matched_fraction
        )
        train_info['unmatched_capture_fraction'] = _to_float(
            unmatched_capture_fraction
        )
        train_info['failed_episode_capture_count'] = _to_float(
            failed_episode_capture_count_value
        )
        train_info['failed_episode_capture_fraction'] = _to_float(
            failed_episode_capture_fraction
        )
        train_info['capture_to_win_capture_success_fraction'] = _to_float(
            capture_to_win_capture_success_fraction
        )
        train_info['capture_to_win_episode_success_fraction'] = _to_float(
            capture_to_win_episode_success_fraction
        )
        train_info['positive_reward_without_capture_fraction'] = _to_float(
            positive_reward_without_capture_fraction
        )
        train_info['positive_reward_step_count'] = _to_float(
            positive_reward_step_count_value
        )
        train_info['positive_reward_without_capture_count'] = _to_float(
            positive_reward_without_capture_count_value
        )
        train_info['positive_reward_step_fraction'] = _to_float(
            positive_reward_step_fraction
        )
        train_info['offset0_candidate_count'] = _to_float(
            offset0_candidate_count_value
        )
        train_info['offset0_candidate_fraction'] = _to_float(
            offset0_candidate_fraction
        )
        train_info['graph_return_credit_strength_mean'] = _to_float(
            graph_return_credit_strength_mean
        )
        train_info['order2_factor_rl_loss'] = _to_float(
            order2_factor_rl_loss
        )
        train_info['order3_factor_rl_loss'] = _to_float(
            order3_factor_rl_loss
        )
        train_info['raw_order2_factor_rl_loss'] = _to_float(
            raw_order2_factor_rl_loss
        )
        train_info['raw_order3_factor_rl_loss'] = _to_float(
            raw_order3_factor_rl_loss
        )
        train_info['raw_o3_minus_o2_factor_rl_loss'] = _to_float(
            raw_order3_factor_rl_loss - raw_order2_factor_rl_loss
        )
        train_info['credit_order2_factor_rl_loss'] = _to_float(
            credit_order2_factor_rl_loss
        )
        train_info['credit_order3_factor_rl_loss'] = _to_float(
            credit_order3_factor_rl_loss
        )
        train_info['credit_o3_minus_o2_factor_rl_loss'] = _to_float(
            credit_order3_factor_rl_loss - credit_order2_factor_rl_loss
        )
        train_info['order2_positive_adv_fraction'] = _to_float(
            order2_positive_adv_fraction
        )
        train_info['order3_positive_adv_fraction'] = _to_float(
            order3_positive_adv_fraction
        )
        train_info['credit_order2_positive_adv_fraction'] = _to_float(
            credit_order2_positive_adv_fraction
        )
        train_info['credit_order3_positive_adv_fraction'] = _to_float(
            credit_order3_positive_adv_fraction
        )
        train_info['order2_promoted_adv_fraction'] = _to_float(
            order2_promoted_adv_fraction
        )
        train_info['order3_promoted_adv_fraction'] = _to_float(
            order3_promoted_adv_fraction
        )
        train_info['order2_promotion_strength_mean'] = _to_float(
            order2_promotion_strength_mean
        )
        train_info['order3_promotion_strength_mean'] = _to_float(
            order3_promotion_strength_mean
        )
        train_info['order3_promoted_adv_weighted_fraction'] = _to_float(
            order3_promoted_adv_weighted_fraction
        )
        train_info['adj_order_adv_coef'] = float(self.adj_order_adv_coef)
        train_info['adj_order_adv_positive_only'] = float(
            self.adj_order_adv_positive_only
        )
        train_info['adj_order_adv_negative_coef'] = float(
            self.adj_order_adv_negative_coef
        )
        train_info['adj_order_adv_require_positive_graph_adv'] = float(
            self.adj_order_adv_require_positive_graph_adv
        )
        train_info['adj_order_adv_graph_gate_soft'] = float(
            self.adj_order_adv_graph_gate_mode == "soft"
        )
        train_info['adj_order_adv_graph_gate_scale'] = float(
            self.adj_order_adv_graph_gate_scale
        )
        train_info['use_adj_triplet_graph_return_credit'] = float(
            self.use_adj_triplet_graph_return_credit
        )
        train_info['adj_triplet_graph_return_credit_coef'] = float(
            self.adj_triplet_graph_return_credit_coef
        )
        train_info['adj_triplet_graph_return_credit_cap'] = float(
            self.adj_triplet_graph_return_credit_cap
        )
        train_info['adj_triplet_graph_return_credit_min_graph_adv'] = float(
            self.adj_triplet_graph_return_credit_min_graph_adv
        )
        train_info['adj_triplet_graph_return_credit_raw_gate_scale'] = float(
            self.adj_triplet_graph_return_credit_raw_gate_scale
        )
        train_info[
            'adj_triplet_graph_return_credit_require_delayed_gate'
        ] = float(
            self.adj_triplet_graph_return_credit_require_delayed_gate
        )
        train_info['use_adj_delayed_triplet_credit'] = float(
            self.use_adj_delayed_triplet_credit
        )
        train_info['adj_delayed_triplet_credit_coef'] = float(
            self.adj_delayed_triplet_credit_coef
        )
        train_info['adj_delayed_triplet_credit_window'] = float(
            self.adj_delayed_triplet_credit_window
        )
        train_info['adj_delayed_triplet_credit_cap'] = float(
            self.adj_delayed_triplet_credit_cap
        )
        train_info['adj_delayed_triplet_credit_min_reward'] = float(
            self.adj_delayed_triplet_credit_min_reward
        )
        train_info['adj_delayed_triplet_credit_positive_only'] = float(
            self.adj_delayed_triplet_credit_positive_only
        )
        train_info['adj_delayed_triplet_credit_min_adv'] = float(
            self.adj_delayed_triplet_credit_min_adv
        )
        train_info['adj_delayed_triplet_credit_require_future_match'] = float(
            self.adj_delayed_triplet_credit_require_future_match
        )
        train_info['use_adj_delayed_triplet_success_gate'] = float(
            self.use_adj_delayed_triplet_success_gate
        )
        train_info['adj_delayed_triplet_success_gate_min_adv'] = float(
            self.adj_delayed_triplet_success_gate_min_adv
        )
        train_info['adj_delayed_triplet_success_gate_scale'] = float(
            self.adj_delayed_triplet_success_gate_scale
        )
        train_info['adj_delayed_triplet_success_gate_floor'] = float(
            self.adj_delayed_triplet_success_gate_floor
        )
        train_info['adj_delayed_triplet_future_overlap_min_nodes'] = float(
            self.adj_delayed_triplet_future_overlap_min_nodes
        )
        train_info['adj_delayed_triplet_partial_match_weight'] = float(
            self.adj_delayed_triplet_partial_match_weight
        )
        train_info['use_adj_capture_to_win_credit'] = float(
            self.use_adj_capture_to_win_credit
        )
        train_info['adj_capture_to_win_credit_coef'] = float(
            self.adj_capture_to_win_credit_coef
        )
        train_info['adj_capture_to_win_credit_min_outcome_adv'] = float(
            self.adj_capture_to_win_credit_min_outcome_adv
        )
        train_info['adj_capture_to_win_credit_scale'] = float(
            self.adj_capture_to_win_credit_scale
        )
        train_info['adj_capture_to_win_credit_cap'] = float(
            self.adj_capture_to_win_credit_cap
        )
        train_info['adj_capture_to_win_credit_require_future_match'] = float(
            self.adj_capture_to_win_credit_require_future_match
        )
        train_info['use_adj_pair_triplet_complementary_credit'] = float(
            self.use_adj_pair_triplet_complementary_credit
        )
        train_info['adj_pair_pursuit_credit_coef'] = float(
            self.adj_pair_pursuit_credit_coef
        )
        train_info['adj_pair_pursuit_credit_window'] = float(
            self.adj_pair_pursuit_credit_window
        )
        train_info['adj_pair_pursuit_credit_cap'] = float(
            self.adj_pair_pursuit_credit_cap
        )
        train_info['adj_pair_pursuit_credit_min_reward'] = float(
            self.adj_pair_pursuit_credit_min_reward
        )
        for credit_key, credit_value in advantage_triplet_credit_info.items():
            train_info[credit_key] = float(credit_value)
        train_info['use_adj_advantage_triplet_scorer'] = float(
            bool(
                getattr(
                    self.adj_network,
                    "use_advantage_triplet_scorer",
                    False,
                )
            )
        )
        train_info['use_adj_triplet_credit_direct_rank'] = float(
            bool(
                getattr(
                    self.adj_network,
                    "use_triplet_credit_direct_rank",
                    False,
                )
            )
        )
        train_info['adj_triplet_credit_rank_coef'] = float(
            getattr(self.adj_network, "triplet_credit_rank_coef", 0.0)
        )
        train_info['adj_triplet_credit_min_multiplier'] = float(
            getattr(self.adj_network, "triplet_credit_min_multiplier", 1.0)
        )
        train_info['adj_triplet_credit_max_multiplier'] = float(
            getattr(self.adj_network, "triplet_credit_max_multiplier", 1.0)
        )
        train_info['adj_triplet_credit_negative_rank_scale'] = float(
            getattr(
                self.adj_network,
                "triplet_credit_negative_rank_scale",
                1.0,
            )
        )
        train_info['adj_triplet_credit_min_positive_fraction'] = float(
            getattr(
                self.adj_network,
                "triplet_credit_min_positive_fraction",
                0.0,
            )
        )
        train_info['adv_triplet_score_multiplier_mean'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_multiplier_mean",
                1.0,
            )
        )
        train_info['adv_triplet_score_multiplier_min'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_multiplier_min",
                1.0,
            )
        )
        train_info['adv_triplet_score_multiplier_max'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_multiplier_max",
                1.0,
            )
        )
        train_info['adv_triplet_score_marginal_mean'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_marginal_mean",
                0.0,
            )
        )
        train_info['adv_triplet_score_positive_fraction'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_positive_fraction",
                0.0,
            )
        )
        train_info['adv_triplet_negative_scaled_fraction'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_negative_scaled_fraction",
                0.0,
            )
        )
        train_info['adj_order3_credit_gate_current'] = float(
            order3_credit_gate
        )
        train_info['adj_order3_credit_loss_ema'] = float(
            order3_credit_loss_ema
        )
        train_info['adj_order3_credit_margin_ema'] = float(
            order3_credit_margin_ema
        )
        train_info['adj_order3_relative_credit_gate'] = float(
            bool(
                getattr(
                    self.adj_network,
                    "use_relative_order3_credit_gate",
                    False,
                )
            )
        )
        train_info['entropy_loss'] = _to_float(entropy_loss)
        train_info['adj_entropy_coef_initial'] = float(self.entropy_coef)
        train_info['adj_entropy_coef'] = float(self.adj_entropy_coef)
        train_info['grad_norm'] = _to_float(grad_norm)
        train_info['adj_lr'] = float(current_adj_lr)
        pair_boundary_retention_trace_rows = []
        pair_boundary_policy_response_rows = []
        pair_boundary_retention_new_count = 0
        if (
                transaction_diagnostics_enabled
                and not bool(candidate_residual_only)):
            try:
                crossing_retention_registration_required = bool(
                    pair_diagnostic_target_present
                    and any(
                        int(row["boundary_crossing"]) == 1
                        for row in pair_boundary_trace_rows
                    )
                )
                if crossing_retention_registration_required and (
                        adj_update_round is None
                        or diagnostic_policy_id is None
                        or diagnostic_transaction_sequence_index is None):
                    raise RuntimeError(
                        "crossing retention transaction identity is "
                        "incomplete"
                    )
                if not bool(pair_only_objective):
                    self._pair_selection_boundary_ordinary_update_clock += 1
                    pair_boundary_retention_trace_rows = (
                        self._pair_selection_boundary_retention_diagnostics()
                    )
                if pair_diagnostic_target_present:
                    pair_boundary_policy_response_rows = (
                        self
                        ._pair_selection_boundary_policy_response_diagnostics(
                            trace_rows=pair_boundary_trace_rows,
                            obs=obs,
                            rnn_obs=rnn_obs,
                            dones=dones,
                            adj=adj,
                            available_actions=available_actions,
                            policy_id=diagnostic_policy_id,
                        )
                    )
                    pair_boundary_retention_new_count = (
                        self
                        ._register_pair_selection_boundary_retention_observations(
                            trace_rows=pair_boundary_trace_rows,
                            rnn_obs=rnn_obs,
                            dones=dones,
                            adj=adj,
                            replay_population_provenance=(
                                replay_population_provenance
                            ),
                            adjacency_update_round=adj_update_round,
                            policy_id=diagnostic_policy_id,
                            transaction_sequence_index=(
                                diagnostic_transaction_sequence_index
                            ),
                        )
                    )
            except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError):
                if pair_transaction_atomic_state_required:
                    _rollback_failed_pair_transaction()
                raise
        train_info[
            'pair_selection_boundary_retention_diagnostic_version'
        ] = float(PAIR_SELECTION_BOUNDARY_RETENTION_DIAGNOSTIC_VERSION)
        train_info[
            'pair_selection_boundary_retention_new_count'
        ] = float(pair_boundary_retention_new_count)
        train_info[
            'pair_selection_boundary_retention_archive_size'
        ] = float(len(
            self._pair_selection_boundary_retention_observations
        ))
        train_info[
            'pair_selection_boundary_ordinary_update_clock'
        ] = float(self._pair_selection_boundary_ordinary_update_clock)
        train_info[
            'pair_selection_boundary_retention_protected_target_count'
        ] = _to_float(pair_boundary_retention_protected_target_count)
        train_info[
            'pair_selection_boundary_retention_context_invalidated_count'
        ] = float(len(pair_boundary_retention_stop_keys))
        train_info[
            'pair_selection_boundary_retention_superseded_count'
        ] = float(len(pair_boundary_retention_superseded_keys))
        train_info[
            'pair_selection_boundary_retention_gradient_projection_intervened'
        ] = _to_float(
            pair_boundary_retention_gradient_projection_intervened
        )
        train_info[
            'pair_selection_boundary_retention_actual_projection_corrected'
        ] = _to_float(
            pair_boundary_retention_actual_projection_corrected
        )
        train_info[
            'pair_selection_boundary_retention_nonlinear_backtrack_count'
        ] = _to_float(
            pair_boundary_retention_nonlinear_backtrack_count
        )
        train_info[
            'pair_selection_boundary_retention_final_scale'
        ] = _to_float(pair_boundary_retention_final_scale)
        train_info[
            'pair_selection_boundary_retention_optimizer_state_sync_applied'
        ] = _to_float(
            pair_boundary_retention_optimizer_state_sync_applied
        )
        train_info[
            'pair_selection_boundary_retention_final_exact_min_signed_gap'
        ] = _to_float(
            pair_boundary_retention_final_exact_min_signed_gap
        )
        train_info[
            'pair_selection_boundary_retention_final_exact_max_tolerance'
        ] = _to_float(
            pair_boundary_retention_final_exact_max_tolerance
        )
        train_info[
            'pair_selection_boundary_retention_final_postcondition_entered'
        ] = _to_float(
            pair_boundary_retention_final_postcondition_entered
        )
        train_info[
            'pair_selection_boundary_retention_final_postcondition_target_count'
        ] = _to_float(
            pair_boundary_retention_final_postcondition_target_count
        )
        train_info[
            'pair_selection_boundary_retention_selection_state_backtrack_count'
        ] = _to_float(
            pair_boundary_retention_selection_state_backtrack_count
        )
        train_info[
            'pair_selection_boundary_retention_selection_state_refinement_count'
        ] = _to_float(
            pair_boundary_retention_selection_state_refinement_count
        )
        train_info[
            'pair_selection_boundary_retention_selection_state_final_scale'
        ] = _to_float(
            pair_boundary_retention_selection_state_final_scale
        )
        train_info[
            'pair_selection_boundary_retention_selection_state_unsafe_upper_scale'
        ] = _to_float(
            pair_boundary_retention_selection_state_unsafe_upper_scale
        )
        train_info[
            'pair_selection_boundary_retention_selection_state_seen_count_delta'
        ] = _to_float(
            pair_boundary_retention_selection_state_seen_count_delta
        )
        train_info[
            'pair_selection_boundary_retention_selection_state_seen_count_integral_valid'
        ] = _to_float(
            pair_boundary_retention_selection_state_seen_count_integral_valid
        )
        train_info[
            '_candidate_identity_transaction_rows'
        ] = candidate_identity_transaction_rows
        train_info[
            '_candidate_evidence_consumption_rows'
        ] = candidate_evidence_consumption_rows
        train_info[
            '_pair_selection_boundary_rows'
        ] = pair_boundary_trace_rows
        train_info[
            '_pair_direction_candidate_rows'
        ] = pair_direction_candidate_trace_rows
        train_info[
            '_pair_selection_boundary_retention_rows'
        ] = pair_boundary_retention_trace_rows
        train_info[
            '_pair_selection_boundary_retention_component_rows'
        ] = pair_boundary_retention_selection_state_component_rows
        train_info[
            '_pair_selection_boundary_policy_response_rows'
        ] = pair_boundary_policy_response_rows

        return train_info, None, None

    def hard_target_updates(self):
        """Hard update the target networks."""
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(
                self.policies[policy_id])

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(
                self.target_policies[policy_id], self.policies[policy_id], self.tau)
            if self.use_vfunction:
                target_policy = self.target_policies[policy_id]
                source_policy = self.policies[policy_id]
                soft_update(
                    target_policy.rnn_critic_network,
                    source_policy.rnn_critic_network,
                    self.tau,
                )
                soft_update(
                    target_policy.vtot_network,
                    source_policy.vtot_network,
                    self.tau,
                )
                for order in range(1, self.highest_orders + 1):
                    soft_update(
                        target_policy.v_network[order],
                        source_policy.v_network[order],
                        self.tau,
                    )

    def prep_training(self):
        """See parent class."""
        self.adj_network.train()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.train()
            self.policies[p_id].rnn_critic_network.train()
            self.target_policies[p_id].rnn_network.train()
            self.target_policies[p_id].rnn_critic_network.train()
            if self.use_vfunction:
                self.policies[p_id].vtot_network.train()
                self.target_policies[p_id].vtot_network.train()

            for num_orders in range(1, self.highest_orders + 1):
                self.policies[p_id].q_network[num_orders].train()
                self.target_policies[p_id].q_network[num_orders].train()
                if self.use_vfunction:
                    self.policies[p_id].v_network[num_orders].train()
                    self.target_policies[p_id].v_network[num_orders].train()

    def prep_rollout(self):
        """See parent class."""

        self.adj_network.eval()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.eval()
            self.policies[p_id].rnn_critic_network.eval()
            self.target_policies[p_id].rnn_network.eval()
            self.target_policies[p_id].rnn_critic_network.eval()
            if self.use_vfunction:
                self.policies[p_id].vtot_network.eval()
                self.target_policies[p_id].vtot_network.eval()

            for num_orders in range(1, self.highest_orders + 1):
                self.policies[p_id].q_network[num_orders].eval()
                self.target_policies[p_id].q_network[num_orders].eval()
                if self.use_vfunction:
                    self.policies[p_id].v_network[num_orders].eval()
                    self.target_policies[p_id].v_network[num_orders].eval()
