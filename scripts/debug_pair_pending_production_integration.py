"""Production integration tests for bounded strict-pair pending.

Unlike the NumPy-only foundation tests, this module executes the real
AdjPolicyBuffer snapshot/preparation path and the real R_SDDFG adjacency
forward, pair-local loss, Adam step, multi-epoch outer transaction, rollback,
and checkpoint restore.
"""

from __future__ import annotations

import copy
import os
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the repository's real trainer fixture. It also installs harmless
# stubs for optional online logging packages when the test environment does
# not provide them.
from debug_pair_optimizer_transaction_diagnostics import (  # noqa: E402
    _MinimalPolicy,
    _full_transaction_args,
    _make_full_transaction_batch,
)
from algorithms.sddfg.algorithm.adj_generator import Adj_Generator  # noqa: E402
from algorithms.sddfg.r_sddfg import (  # noqa: E402
    R_SDDFG,
    StrictPairExactInfeasibleError,
    _clear_parameter_gradients_to_none,
)
import algorithms.sddfg.r_sddfg as r_sddfg_module  # noqa: E402
from runner.base_runner import (  # noqa: E402
    RecRunner,
    _STRICT_PAIR_EXACT_FAILURE_FIELDS,
)
import runner.base_runner as base_runner_module  # noqa: E402
from utils.adj_buffer import AdjPolicyBuffer  # noqa: E402
from utils.pair_pending import (  # noqa: E402
    PAIR_EVIDENCE_COMMITTED,
    PAIR_EVIDENCE_PENDING,
    PairPendingEvidenceStore,
    PairPendingZeroGradientError,
    PairOptimizerRecoverableNoOpError,
)


def _assert_raises(error_type, fn, contains=None):
    try:
        fn()
    except error_type as error:
        if contains is not None and contains not in str(error):
            raise AssertionError(
                "expected {!r} in {!r}".format(contains, str(error))
            )
        return
    raise AssertionError("expected {}".format(error_type.__name__))


def _assert_nested_equal(left, right, path="root"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert left.dtype == right.dtype, path
        assert tuple(left.shape) == tuple(right.shape), path
        assert bool(torch.equal(left.cpu(), right.cpu())), path
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert np.array_equal(left, right), path
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), (
            "{} keys differ: left_only={!r}, right_only={!r}".format(
                path,
                sorted(set(left) - set(right), key=str),
                sorted(set(right) - set(left), key=str),
            )
        )
        for key in left:
            _assert_nested_equal(
                left[key],
                right[key],
                "{}[{!r}]".format(path, key),
            )
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right), path
        assert len(left) == len(right), path
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _assert_nested_equal(
                left_value,
                right_value,
                "{}[{}]".format(path, index),
            )
        return
    assert left == right, path


def _assert_optimizer_semantically_equal(left, right, path):
    """Compare optimizer state by parameter position, not serialized IDs.

    Older supported PyTorch releases may serialize optimizer ``state`` keys
    using identifiers that are local to an optimizer instance.  Two
    independently constructed but equivalent optimizers can therefore have
    different raw key sets.  The live parameter-group order is the stable
    semantic mapping used by ``Optimizer.step`` and ``load_state_dict``.
    """
    assert type(left) is type(right), path
    assert len(left.param_groups) == len(right.param_groups), path
    for group_index, (left_group, right_group) in enumerate(zip(
            left.param_groups, right.param_groups)):
        group_path = "{}.param_groups[{}]".format(path, group_index)
        assert set(left_group) == set(right_group), group_path
        left_meta = {
            key: value for key, value in left_group.items() if key != "params"
        }
        right_meta = {
            key: value for key, value in right_group.items() if key != "params"
        }
        _assert_nested_equal(left_meta, right_meta, group_path + ".metadata")
        left_params = tuple(left_group["params"])
        right_params = tuple(right_group["params"])
        assert len(left_params) == len(right_params), group_path
        for param_index, (left_param, right_param) in enumerate(zip(
                left_params, right_params)):
            param_path = "{}.params[{}]".format(group_path, param_index)
            assert tuple(left_param.shape) == tuple(right_param.shape), param_path
            _assert_nested_equal(
                dict(left.state.get(left_param, {})),
                dict(right.state.get(right_param, {})),
                param_path + ".state",
            )


def _trainer(device, seed=611):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    args = _full_transaction_args()
    args.use_adj_ppo_stale_trust = True
    args.adj_ppo_stale_trust_clip = 0.20
    args.adj_ppo_stale_trust_scale = 0.25
    args.adj_ppo_stale_trust_min_weight = 0.25
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
    trainer.adj_network.train()
    return args, graph, trainer


def _one_episode_fields(graph, device, generation, pair_factor_index=0):
    fields = list(_make_full_transaction_batch(graph, device))
    # The production replay provenance schema is [selected, episode ordinal,
    # generation]. The historical fixture predates the generation column.
    fields[33] = np.zeros((1, 2, 3), dtype=np.float32)
    fields[33][..., 0] = 1.0
    fields[33][..., 1] = 0.0
    fields[33][..., 2] = float(generation)
    # Snapshot field 17 is a non-negative strict structural score. Signed
    # centered target field 15 is rebuilt only when a class-complete cohort is
    # prepared.
    fields[17] = np.zeros((1, 2, 4, 1), dtype=np.float32)
    fields[17][0, 0, int(pair_factor_index), 0] = 0.99
    fields[15] = np.zeros((1, 2, 4, 1), dtype=np.float32)
    return tuple(np.array(value, copy=True) for value in fields)


def _production_policy_buffer(graph, device, horizon=4):
    policy_buffer = AdjPolicyBuffer.__new__(AdjPolicyBuffer)
    policy_buffer.pair_pending_enabled = True
    policy_buffer.pair_pending_max_adj_updates = int(horizon)
    policy_buffer.pair_pending_store = PairPendingEvidenceStore(
        enabled=True,
        max_adj_updates=horizon,
    )
    policy_buffer.pair_pending_current_adj_update = 1
    policy_buffer.pair_pending_behavior_policy_version = int(
        graph.candidate_policy_version
    )
    policy_buffer.pair_pending_new_snapshot_count = 0
    policy_buffer.pair_pending_expired_ttl_count = 0
    policy_buffer.pair_pending_payload_contract_valid = 1.0
    policy_buffer.pair_pending_counter_update = -1
    policy_buffer.pair_pending_prepared_count = 0
    policy_buffer.pair_pending_aborted_count = 0
    policy_buffer.pair_pending_rolled_back_count = 0
    policy_buffer.pair_pending_committed_count = 0
    policy_buffer.pair_pending_class_complete_count = 0
    policy_buffer.pair_pending_pair_only_transaction_count = 0
    policy_buffer.pair_pending_zero_target_abort_count = 0
    policy_buffer.pair_pending_zero_gradient_abort_count = 0
    policy_buffer.pair_pending_early_stop_abort_count = 0
    policy_buffer.pair_pending_expired_provenance_count = 0
    policy_buffer.pair_pending_expired_population_mismatch_count = 0
    policy_buffer.pair_pending_stale_contract_valid = 1.0
    policy_buffer.pair_pending_mass_contract_valid = 1.0
    policy_buffer.pair_pending_objective_scope_contract_valid = 1.0
    policy_buffer.pair_pending_atomic_rollback_contract_valid = 1.0
    policy_buffer.pair_pending_checkpoint_contract_valid = 1.0
    policy_buffer.pair_pending_last_positive_stale_trust = 0.0
    policy_buffer.pair_pending_last_negative_stale_trust = 0.0
    policy_buffer.pair_pending_last_raw_positive_mass = 0.0
    policy_buffer.pair_pending_last_raw_negative_mass = 0.0
    policy_buffer.pair_pending_last_effective_positive_mass = 0.0
    policy_buffer.pair_pending_last_effective_negative_mass = 0.0
    policy_buffer.policy_id = "policy_0"
    policy_buffer.episode_length = 2
    policy_buffer.num_factor = 4
    policy_buffer.adj_pair_pursuit_credit_coef = 0.10
    policy_buffer.adj_pair_pursuit_credit_cap = 0.20
    policy_buffer.episode_generation = np.asarray([660], dtype=np.int64)
    policy_buffer.environment_episode_id = np.asarray(
        [1660], dtype=np.int64
    )
    policy_buffer.episode_behavior_policy_version = np.asarray(
        [int(graph.candidate_policy_version)], dtype=np.int64
    )
    policy_buffer.strict_pair_event_provenance = [(
        {
            "environment_episode_id": 1660,
            "event_id": 60,
            "target_id": 3,
            "participant_slots": (0, 1),
            "pair_transition_step": 0,
            "capture_step": 1,
            "delay": 1,
            "pair_factor_index": 0,
            "pair_identity": "0-1",
            "pair_order": 2,
            "capture_factor_index": 0,
            "capture_factor_identity": "0-1",
            "capture_factor_order": 2,
            "raw_transition_quality": 0.99,
            "identity_event_weight": 1.0,
            "factor_slot_weight": 1.0,
        },
    )]
    policy_buffer.pair_to_triplet_transition_score = np.zeros(
        (2, 1, 4, 1), dtype=np.float32
    )
    policy_buffer.pair_to_triplet_transition_score[0, 0, 0, 0] = 0.99
    policy_buffer.success_now = np.zeros((2, 1, 1), dtype=np.float32)
    policy_buffer.dones_env = np.zeros((2, 1, 1), dtype=np.float32)
    policy_buffer.dones_env[1, 0, 0] = 1.0
    return policy_buffer


def _flat_episode(fields):
    return tuple(
        np.asarray(value).reshape(
            (int(value.shape[0]) * int(value.shape[1]),)
            + tuple(value.shape[2:])
        )
        for value in fields
    )


def _snapshot_generation(
        policy_buffer,
        graph,
        device,
        generation,
        sign,
        update_index,
        event_id,
        environment_episode_id,
        pair_factor_index=None,
        pair_identity=None):
    if pair_factor_index is None:
        pair_factor_index = 1 if int(sign) > 0 else 0
    if pair_identity is None:
        pair_identity = "0-2" if int(pair_factor_index) == 1 else "0-1"
    policy_buffer.pair_pending_current_adj_update = int(update_index)
    policy_buffer.pair_pending_behavior_policy_version = int(
        graph.candidate_policy_version
    )
    policy_buffer.episode_generation[0] = int(generation)
    policy_buffer.environment_episode_id[0] = int(environment_episode_id)
    policy_buffer.episode_behavior_policy_version[0] = int(
        graph.candidate_policy_version
    )
    record = dict(policy_buffer.strict_pair_event_provenance[0][0])
    record["environment_episode_id"] = int(environment_episode_id)
    record["event_id"] = int(event_id)
    record["pair_factor_index"] = int(pair_factor_index)
    record["pair_identity"] = str(pair_identity)
    record["capture_factor_index"] = int(pair_factor_index)
    record["capture_factor_identity"] = str(pair_identity)
    record["participant_slots"] = tuple(
        int(slot) for slot in str(pair_identity).split("-")
    )
    policy_buffer.strict_pair_event_provenance[0] = (record,)
    policy_buffer.pair_to_triplet_transition_score.fill(0.0)
    policy_buffer.pair_to_triplet_transition_score[
        0, 0, int(pair_factor_index), 0
    ] = 0.99
    policy_buffer.success_now.fill(0.0)
    if sign > 0:
        policy_buffer.success_now[1, 0, 0] = 1.0
    fields = _one_episode_fields(
        graph,
        device,
        generation,
        pair_factor_index=pair_factor_index,
    )
    immutable_before = tuple(np.array(value, copy=True) for value in fields)
    policy_buffer._capture_pair_pending_snapshots(
        _flat_episode(fields),
        episode_indices=np.asarray([0], dtype=np.int64),
        selected_pair_evidence=np.asarray([True]),
        base_episode_indices=np.asarray([0], dtype=np.int64),
        data_chunk_length=2,
    )
    generation_keys = policy_buffer.pair_pending_store.keys_for_generation(
        "policy_0", int(generation)
    )
    assert len(generation_keys) == 1
    key = generation_keys[0]
    frozen = policy_buffer.pair_pending_store.entries[key]["batch"]
    # Simulate complete circular-buffer reuse. Immutable pending storage must
    # remain byte-for-byte independent.
    for value in fields:
        value[...] = -123.0
    for expected, actual in zip(immutable_before, frozen):
        assert np.array_equal(expected, actual)
        assert not actual.flags.writeable
    return key


def _prepared_run107_equivalent(device, horizon=4):
    args, graph, trainer = _trainer(device)
    policy_buffer = _production_policy_buffer(
        graph, device, horizon=horizon
    )
    negative_key = _snapshot_generation(
        policy_buffer,
        graph,
        device,
        generation=660,
        sign=-1,
        update_index=1,
        event_id=60,
        environment_episode_id=1660,
    )
    policy_buffer.pair_pending_store.mark_pending(
        negative_key, current_adj_update=1
    )
    policy_buffer.pair_pending_current_adj_update = 5
    positive_key = _snapshot_generation(
        policy_buffer,
        graph,
        device,
        generation=689,
        sign=1,
        update_index=5,
        event_id=89,
        environment_episode_id=1689,
        pair_factor_index=1,
        pair_identity="0-2",
    )
    prepared = policy_buffer.prepare_pair_pending_training_batch(
        expected_ppo_epochs=2
    )
    assert prepared is not None
    assert set(prepared["keys"]) == {negative_key, positive_key}
    assert prepared["episode_count"] == 2
    assert prepared["chunk_count"] == 2
    assert prepared["raw_positive_mass"] > 0.0
    assert prepared["raw_negative_mass"] > 0.0
    assert prepared["mass_contract_valid"] == 1.0
    return args, graph, trainer, policy_buffer, prepared


class _FakeRunner(object):
    _run_pair_pending_outer_transaction = (
        RecRunner._run_pair_pending_outer_transaction
    )
    _capture_pair_pending_rng_state = staticmethod(
        RecRunner._capture_pair_pending_rng_state
    )
    _restore_pair_pending_rng_state = staticmethod(
        RecRunner._restore_pair_pending_rng_state
    )
    _record_strict_pair_exact_failure = (
        RecRunner._record_strict_pair_exact_failure
    )

    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.adj_train_epochs = 2
        self.use_adj_init = False
        self._adj_transaction_sequence_index = 0
        self.total_env_steps = 144800
        self.run_dir = os.path.join(
            tempfile.gettempdir(), "run107_pair_pending_integration"
        )
        self.logged_rows = []

    def _append_fixed_rows_csv(self, _basename, rows):
        self.logged_rows.extend(copy.deepcopy(rows))

    def _record_adj_transaction(self, **kwargs):
        row = {
            "transaction_sequence_index": int(
                self._adj_transaction_sequence_index
            ),
            "ppo_epoch_index": int(kwargs["ppo_epoch_index"]),
        }
        self._adj_transaction_sequence_index += 1
        return row


class _FailOnSecondEpoch(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train_adj_on_batch(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("injected epoch-1 failure")
        return self.trainer.train_adj_on_batch(*args, **kwargs)


class _ZeroGradientOnFirstEpoch(object):
    """Inject the learner's bounded no-action result at the runner boundary."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train_adj_on_batch(self, *args, **kwargs):
        self.calls += 1
        raise PairPendingZeroGradientError(
            "pending pair-only target produced zero adjacency gradient"
        )


class _ExactInfeasibleOnFirstEpoch(object):
    """Inject the typed exhaustive no-feasible result at the runner edge."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train_adj_on_batch(self, *args, **kwargs):
        self.calls += 1
        runtime_fields = {
            "run_id",
            "env_step",
            "adjacency_update_round",
            "ppo_epoch_index",
            "policy_id",
            "partition_index",
            "transaction_sequence_index",
        }
        payload = {
            field: 0
            for field in _STRICT_PAIR_EXACT_FAILURE_FIELDS
            if field not in runtime_fields
        }
        payload.update({
            "diagnostic_version": int(
                r_sddfg_module.STRICT_PAIR_EXACT_FAILURE_DIAGNOSTIC_VERSION
            ),
            "optimizer_kind": "pair_pending_adam",
            "failure_classification": "sampled_grid_has_no_feasible_point",
            "origin_preservation_valid": 1,
            "diagnostic_probe_valid_count": 0,
            "bounded_search_exhaustive": 1,
            "target_count": 2,
        })
        raise StrictPairExactInfeasibleError(
            "fixture exhaustive pair-pending infeasibility",
            diagnostic_rows=(payload,),
        )


class _NoUsableUpdateOnFirstEpoch(object):
    """Inject a verified inner atomic no-op at the pending runner edge."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train_adj_on_batch(self, *args, **kwargs):
        self.calls += 1
        raise PairOptimizerRecoverableNoOpError(
            reason="zero_adam_displacement",
            diagnostics={
                "pair_update_dot": 0.0,
                "pair_update_norm_sq": 0.0,
                "pair_gradient_norm_sq": 1.0,
                "clipped_pair_dot": 1.0,
                "clipped_gradient_norm_sq": 1.0,
            },
            target_count=2,
        )


class _CaptureDiagnosticSequence(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.diagnostic_sequences = []

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train_adj_on_batch(self, *args, **kwargs):
        self.diagnostic_sequences.append(int(
            kwargs["diagnostic_transaction_sequence_index"]
        ))
        return self.trainer.train_adj_on_batch(*args, **kwargs)


class _NonfiniteOnSecondEpoch(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train_adj_on_batch(self, *args, **kwargs):
        self.calls += 1
        result = self.trainer.train_adj_on_batch(*args, **kwargs)
        if self.calls == 2:
            info, priorities, idxes = result
            info = dict(info)
            info[
                "pair_optimizer_transaction_final_pair_descent_dot"
            ] = float("nan")
            return info, priorities, idxes
        return result


def _assert_pair_selection_boundary_contract(info, target_count):
    assert info[
        "pair_optimizer_transaction_pair_boundary_diagnostic_version"
    ] == 8.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_gradient_constraint_count"
    ] == target_count
    assert info[
        "pair_optimizer_transaction_pair_boundary_gradient_min_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_actual_min_descent_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_target_count"
    ] == target_count
    assert info[
        "pair_optimizer_transaction_pair_boundary_correct_direction_count"
    ] == target_count
    assert info[
        "pair_optimizer_transaction_pair_boundary_reverse_direction_count"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_approximately_zero_count"
    ] == 0.0
    rows = info["_pair_selection_boundary_rows"]
    assert len(rows) == int(target_count)
    assert all(
        row["valid"] == 1
        and row["margin_direction_correct"] == 1
        and row["margin_direction_reverse"] == 0
        and row["margin_direction_zero"] == 0
        and row["pre_competitor_candidate_index"] >= 0
        and row["post_competitor_candidate_index"] >= 0
        and row["linearized_required_margin_improvement"] > 0.0
        and row["boundary_deficit_reduction_fraction_valid"] in (0, 1)
        and row["linearized_crossing_affordable"] in (0, 1)
        and row["linearized_original_boundary_budget"] > 0.0
        and row["linearized_strict_floor_budget"] > 0.0
        and row["linearized_allocated_boundary_budget"] > 0.0
        and row["linearized_allocated_boundary_budget"]
        <= row["linearized_original_boundary_budget"] + 1.0e-6
        and row["linearized_zero_deficit_reclaimed_budget"] >= 0.0
        and row["linearized_deficit_target_count"] >= 0
        and row["linearized_affordable_crossing_count"] >= 0
        and row["linearized_budget_conservation_valid"] == 1
        and abs(
            row["pre_margin"]
            - (
                row["pre_target_logp"]
                - row["pre_competitor_logp"]
            )
        ) <= 1e-6
        and abs(
            row["post_margin"]
            - (
                row["post_target_logp"]
                - row["post_competitor_logp"]
            )
        ) <= 1e-6
        for row in rows
    )


def test_real_pair_only_forward(device):
    _args, _graph, trainer, _policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    info, _priorities, _idxes = trainer.train_adj_on_batch(
        prepared["batch"],
        use_adj_init=False,
        adj_update_round=5,
        pair_only_objective=True,
        enable_optimizer_transaction_diagnostics=True,
    )
    assert info["pair_pending_objective_scope_contract_valid"] == 1.0
    assert info["pair_pending_graph_loss"] == 0.0
    assert info["pair_pending_base_factor_loss"] == 0.0
    assert info["pair_pending_capture_outcome_loss"] == 0.0
    assert info["pair_pending_candidate_loss"] == 0.0
    assert info["pair_pending_entropy_loss"] == 0.0
    assert info["pair_gradient_norm"] > 0.0
    assert info["pair_pending_control_scope_version"] == 2.0
    assert info["pair_pending_standard_ppo_early_stop_applicable"] == 0.0
    assert info["pair_pending_all_configured_epochs_required"] == 1.0
    assert info["pair_pending_pair_target_clip_denominator"] > 0.0
    assert info["pair_pending_pair_target_trusted_clip_denominator"] > 0.0
    assert (
        info["pair_pending_pair_target_clip_denominator"]
        == info["pair_pursuit_factor_loss_target_count"]
    )
    assert info["pair_pending_effective_positive_mass"] > 0.0
    assert info["pair_pending_effective_negative_mass"] > 0.0
    assert 0.25 <= info["pair_pending_positive_stale_trust_mean"] <= 1.0
    assert 0.25 <= info["pair_pending_negative_stale_trust_mean"] <= 1.0
    assert info[
        "pair_optimizer_transaction_pair_optimizer_isolated"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_final_pair_descent_dot"
    ] > 0.0
    exact_target_count = (
        info["pair_optimizer_transaction_positive_target_count"]
        + info["pair_optimizer_transaction_negative_target_count"]
    )
    assert exact_target_count > 0.0
    assert info[
        "pair_optimizer_transaction_pair_target_gradient_constraint_count"
    ] == exact_target_count
    assert info[
        "pair_optimizer_transaction_pair_target_gradient_min_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_score_correct_direction_target_count"
    ] == exact_target_count
    assert info[
        "pair_optimizer_transaction_score_reverse_direction_target_count"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] == 0.0
    _assert_pair_selection_boundary_contract(info, exact_target_count)
    assert info[
        "pair_optimizer_transaction_optimizer_step_before"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_optimizer_step_after"
    ] == 1.0
    assert not trainer.adj_optimizer.state
    assert trainer.pair_pending_optimizer.state


def test_pair_only_does_not_consume_poisoned_standard_adam_state(device):
    _args, _graph, trainer, _policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    parameter_before = [
        parameter.detach().clone() for parameter in trainer.adj_parameters
    ]
    trainer.adj_optimizer.zero_grad()
    _clear_parameter_gradients_to_none(trainer.adj_parameters)
    for parameter in trainer.adj_parameters:
        parameter.grad = torch.full_like(parameter, -100.0)
    trainer.adj_optimizer.step()
    with torch.no_grad():
        for parameter, before in zip(
                trainer.adj_parameters, parameter_before):
            parameter.copy_(before)
    _clear_parameter_gradients_to_none(trainer.adj_parameters)
    poisoned_standard_before = copy.deepcopy(
        trainer.adj_optimizer.state_dict()
    )

    info, _priorities, _idxes = trainer.train_adj_on_batch(
        prepared["batch"],
        use_adj_init=False,
        adj_update_round=5,
        pair_only_objective=True,
        enable_optimizer_transaction_diagnostics=True,
    )
    _assert_nested_equal(
        poisoned_standard_before,
        trainer.adj_optimizer.state_dict(),
        "poisoned_standard_optimizer",
    )
    assert info[
        "pair_optimizer_transaction_pair_optimizer_isolated"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_final_pair_descent_dot"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_optimizer_step_before"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_optimizer_step_after"
    ] == 1.0


def test_pair_direction_guard_corrects_conflicting_pair_moment(device):
    _args, _graph, trainer, _policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    trainer.train_adj_on_batch(
        prepared["batch"],
        use_adj_init=False,
        adj_update_round=5,
        pair_only_objective=True,
        enable_optimizer_transaction_diagnostics=True,
    )
    for state in trainer.pair_pending_optimizer.state.values():
        if "exp_avg" in state:
            state["exp_avg"].mul_(-100.0)
    info, _priorities, _idxes = trainer.train_adj_on_batch(
        prepared["batch"],
        use_adj_init=False,
        adj_update_round=5,
        pair_only_objective=True,
        enable_optimizer_transaction_diagnostics=True,
    )
    assert info[
        "pair_optimizer_transaction_adam_raw_pair_descent_dot"
    ] < 0.0
    assert info[
        "pair_optimizer_transaction_pair_actual_update_direction_guard_applied"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_pair_optimizer_state_sync_applied"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_final_pair_descent_dot"
    ] > 0.0


def test_standard_pair_direction_guard_corrects_conflicting_moment(device):
    _args, graph, trainer = _trainer(device, seed=733)
    batch = _make_full_transaction_batch(
        graph,
        device,
        candidate_target=True,
        candidate_target_sign=-1.0,
    )
    first_info, _priorities, _idxes = trainer.train_adj_on_batch(
        batch,
        use_adj_init=False,
        adj_update_round=1,
        pair_only_objective=False,
        enable_optimizer_transaction_diagnostics=True,
    )
    assert first_info[
        "pair_optimizer_transaction_pair_optimizer_isolated"
    ] == 0.0
    assert first_info[
        "pair_optimizer_transaction_final_pair_descent_dot"
    ] > 0.0
    first_target_count = (
        first_info["pair_optimizer_transaction_positive_target_count"]
        + first_info["pair_optimizer_transaction_negative_target_count"]
    )
    assert first_info[
        "pair_optimizer_transaction_pair_target_gradient_constraint_count"
    ] == first_target_count
    assert first_info[
        "pair_optimizer_transaction_pair_target_gradient_min_dot_after"
    ] > 0.0
    assert first_info[
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_after"
    ] > 0.0
    assert first_info[
        "pair_optimizer_transaction_score_correct_direction_target_count"
    ] == first_target_count
    assert first_info[
        "pair_optimizer_transaction_score_reverse_direction_target_count"
    ] == 0.0
    assert first_info[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] == 0.0
    _assert_pair_selection_boundary_contract(first_info, first_target_count)
    assert first_info[
        "capture_candidate_identity_actual_update_descent_dot_after"
    ] > 0.0
    for state in trainer.adj_optimizer.state.values():
        if "exp_avg" in state:
            state["exp_avg"].mul_(-100.0)
    pair_optimizer_before = copy.deepcopy(
        trainer.pair_pending_optimizer.state_dict()
    )
    info, _priorities, _idxes = trainer.train_adj_on_batch(
        batch,
        use_adj_init=False,
        adj_update_round=2,
        pair_only_objective=False,
        enable_optimizer_transaction_diagnostics=True,
    )
    assert info[
        "pair_optimizer_transaction_adam_raw_pair_descent_dot"
    ] < 0.0
    assert info[
        "pair_optimizer_transaction_pair_actual_update_direction_guard_applied"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_pair_optimizer_state_sync_applied"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_final_pair_descent_dot"
    ] > 0.0
    target_count = (
        info["pair_optimizer_transaction_positive_target_count"]
        + info["pair_optimizer_transaction_negative_target_count"]
    )
    assert info[
        "pair_optimizer_transaction_pair_target_gradient_constraint_count"
    ] == target_count
    assert info[
        "pair_optimizer_transaction_pair_target_gradient_min_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_score_correct_direction_target_count"
    ] == target_count
    assert info[
        "pair_optimizer_transaction_score_reverse_direction_target_count"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] == 0.0
    _assert_pair_selection_boundary_contract(info, target_count)
    assert info[
        "capture_candidate_identity_actual_update_descent_dot_after"
    ] > 0.0
    assert info[
        "pair_optimizer_transaction_pair_optimizer_isolated"
    ] == 0.0
    _assert_nested_equal(
        pair_optimizer_before,
        trainer.pair_pending_optimizer.state_dict(),
        "unused_pair_optimizer",
    )
    followup_info, _priorities, _idxes = trainer.train_adj_on_batch(
        batch,
        use_adj_init=False,
        adj_update_round=3,
        pair_only_objective=False,
        enable_optimizer_transaction_diagnostics=True,
    )
    # A bounded super-unit exact search can move far enough that the next
    # transaction's freshly recomputed pair Jacobian is no longer aligned with
    # the reconstructed previous moment.  State sync guarantees exact
    # reconstruction of the committed displacement; it cannot guarantee a
    # future gradient direction across a nonlinear competitor change.  The
    # follow-up transaction must therefore repair any new conflict and pass all
    # current exact gates rather than relying on a stale raw-dot sign.
    assert followup_info[
        "pair_optimizer_transaction_final_pair_descent_dot"
    ] > 0.0
    assert followup_info[
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_after"
    ] > 0.0
    assert followup_info[
        "pair_optimizer_transaction_pair_optimizer_state_sync_applied"
    ] == 1.0
    followup_target_count = (
        followup_info["pair_optimizer_transaction_positive_target_count"]
        + followup_info["pair_optimizer_transaction_negative_target_count"]
    )
    assert followup_info[
        "pair_optimizer_transaction_score_correct_direction_target_count"
    ] + followup_info[
        "pair_optimizer_transaction_score_approximately_zero_target_count"
    ] == followup_target_count
    assert followup_info[
        "pair_optimizer_transaction_score_reverse_direction_target_count"
    ] == 0.0
    _assert_pair_selection_boundary_contract(
        followup_info, followup_target_count
    )


def _seed_exact_candidate_lifecycle_from_batch(trainer, graph, batch, device):
    num_agents = int(trainer.num_agents)
    num_factor = int(trainer.num_factor)
    hidden_size = int(batch[19].shape[-1])
    rnn_obs = torch.as_tensor(batch[19], device=device).reshape(
        -1, num_agents, hidden_size
    )
    dones = torch.as_tensor(batch[2], device=device).reshape(
        -1, num_agents, 1
    ).bool()
    adj = torch.as_tensor(batch[4], device=device).reshape(
        -1, num_agents, num_factor
    )
    previous_adj = torch.as_tensor(batch[26], device=device).reshape_as(adj)
    identity_delta = torch.as_tensor(
        batch[30], device=device
    ).reshape(rnn_obs.shape[0], -1)
    behavior_version = torch.as_tensor(
        batch[32], device=device
    ).reshape(rnn_obs.shape[0], -1, 4)[..., 3]
    transition_mask = torch.ones(
        rnn_obs.shape[0], 1, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        margins, valid = (
            graph.evaluate_candidate_identity_active_competitor_margins(
                rnn_obs=rnn_obs,
                dones=dones,
                adj=adj,
            )
        )
        ranks = graph.canonical_candidate_ranks(
            margins, valid
        ).to(margins.dtype)
    new_count = trainer._register_candidate_identity_lifecycle(
        rnn_obs=rnn_obs,
        dones=dones,
        adj=adj,
        previous_adj=previous_adj,
        identity_delta=identity_delta,
        transition_mask=transition_mask,
        behavioral_progress_mask=identity_delta.abs(),
        behavior_policy_version=behavior_version,
        reference_margin=margins,
        reference_rank=ranks,
        lifecycle_clock=0,
    )
    assert new_count > 0
    assert trainer._candidate_identity_lifecycle


def test_run118_joint_exact_backtracking_uses_production_transaction(device):
    _args, graph, trainer = _trainer(device, seed=11813)
    batch = _make_full_transaction_batch(
        graph,
        device,
        candidate_target=True,
        candidate_target_sign=-1.0,
    )
    _seed_exact_candidate_lifecycle_from_batch(
        trainer, graph, batch, device
    )
    original_acceptance = (
        r_sddfg_module._joint_exact_constraint_acceptance
    )
    calls = {"count": 0}

    def reject_run118_full_step_once(*args, **kwargs):
        result = original_acceptance(*args, **kwargs)
        if kwargs.get("lifecycle_signed_margin") is not None:
            calls["count"] += 1
            if calls["count"] == 1:
                result = dict(result)
                result["valid"] = False
                result["lifecycle_valid"] = False
                result["lifecycle_violation_count"] = 1
        return result

    r_sddfg_module._joint_exact_constraint_acceptance = (
        reject_run118_full_step_once
    )
    try:
        info, _priorities, _idxes = trainer.train_adj_on_batch(
            batch,
            use_adj_init=False,
            adj_update_round=1,
            pair_only_objective=False,
            enable_optimizer_transaction_diagnostics=True,
        )
    finally:
        r_sddfg_module._joint_exact_constraint_acceptance = (
            original_acceptance
        )
    assert calls["count"] >= 2
    assert info[
        "pair_optimizer_transaction_pair_boundary_nonlinear_backtrack_count"
    ] == 1.0
    # The injected full step is rejected once, while every smaller exact
    # revalidation is safe.  v19 must recover almost the whole interval rather
    # than commit the old coarse 0.5 scale.
    assert info[
        "pair_optimizer_transaction_pair_boundary_nonlinear_backtrack_final_scale"
    ] > 0.999
    assert info[
        "pair_optimizer_transaction_pair_boundary_nonlinear_refinement_count"
    ] == 12.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_nonlinear_invalid_upper_scale"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_joint_candidate_exact_valid"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_joint_lifecycle_exact_valid"
    ] == 1.0
    assert info[
        "pair_optimizer_transaction_lifecycle_final_exact_revalidation_valid"
    ] == 1.0


def test_run118_late_exact_failure_rolls_back_production_transaction(device):
    _args, graph, trainer = _trainer(device, seed=11814)
    batch = _make_full_transaction_batch(
        graph,
        device,
        candidate_target=True,
        candidate_target_sign=-1.0,
    )
    _seed_exact_candidate_lifecycle_from_batch(
        trainer, graph, batch, device
    )
    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.adj_parameters
    ]
    standard_before = copy.deepcopy(trainer.adj_optimizer.state_dict())
    pair_before = copy.deepcopy(
        trainer.pair_pending_optimizer.state_dict()
    )
    residual_before = copy.deepcopy(
        trainer.candidate_residual_optimizer.state_dict()
    )
    lifecycle_before = copy.deepcopy(
        trainer._candidate_identity_lifecycle
    )
    observations_before = copy.deepcopy(
        trainer._candidate_identity_lifecycle_observations
    )
    rng_before = RecRunner._capture_pair_pending_rng_state()
    original_acceptance = (
        r_sddfg_module._joint_exact_constraint_acceptance
    )

    def reject_every_joint_exact_trial(*args, **kwargs):
        result = dict(original_acceptance(*args, **kwargs))
        result["valid"] = False
        result["lifecycle_valid"] = False
        result["lifecycle_violation_count"] = 1
        return result

    r_sddfg_module._joint_exact_constraint_acceptance = (
        reject_every_joint_exact_trial
    )
    try:
        _assert_raises(
            RuntimeError,
            lambda: trainer.train_adj_on_batch(
                batch,
                use_adj_init=False,
                adj_update_round=1,
                pair_only_objective=False,
                enable_optimizer_transaction_diagnostics=True,
            ),
            contains="remained infeasible after bounded backtracking",
        )
    finally:
        r_sddfg_module._joint_exact_constraint_acceptance = (
            original_acceptance
        )
    for parameter, before in zip(
            trainer.adj_parameters, parameters_before):
        assert torch.equal(parameter.detach(), before)
    _assert_nested_equal(
        standard_before,
        trainer.adj_optimizer.state_dict(),
        "run118_standard_optimizer_rollback",
    )
    _assert_nested_equal(
        pair_before,
        trainer.pair_pending_optimizer.state_dict(),
        "run118_pair_optimizer_rollback",
    )
    _assert_nested_equal(
        residual_before,
        trainer.candidate_residual_optimizer.state_dict(),
        "run118_residual_optimizer_rollback",
    )
    _assert_nested_equal(
        lifecycle_before,
        trainer._candidate_identity_lifecycle,
        "run118_lifecycle_rollback",
    )
    _assert_nested_equal(
        observations_before,
        trainer._candidate_identity_lifecycle_observations,
        "run118_lifecycle_observation_rollback",
    )
    _assert_nested_equal(
        rng_before,
        RecRunner._capture_pair_pending_rng_state(),
        "run118_rng_rollback",
    )


def test_v15_no_actionable_pair_path_does_not_enter_joint_guard(device):
    _args, graph, trainer = _trainer(device, seed=11815)
    batch = list(_make_full_transaction_batch(graph, device))
    batch[15] = np.zeros_like(batch[15])
    batch[16] = np.zeros_like(batch[16])
    batch = tuple(batch)
    original_acceptance = (
        r_sddfg_module._joint_exact_constraint_acceptance
    )

    def forbidden_joint_guard(*_args, **_kwargs):
        raise AssertionError(
            "no-target transaction entered pair joint exact guard"
        )

    r_sddfg_module._joint_exact_constraint_acceptance = (
        forbidden_joint_guard
    )
    try:
        info, _priorities, _idxes = trainer.train_adj_on_batch(
            batch,
            use_adj_init=False,
            adj_update_round=1,
            pair_only_objective=False,
            enable_optimizer_transaction_diagnostics=True,
        )
    finally:
        r_sddfg_module._joint_exact_constraint_acceptance = (
            original_acceptance
        )
    assert info["pair_optimizer_transaction_nonzero_pair"] == 0.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_target_count"
    ] == 0.0
    assert not info["_pair_selection_boundary_rows"]


def test_run122_nearest_member_identity_uses_production_transaction(device):
    """Exercise nearest-member identity progress through train_adj_on_batch."""
    _args, graph, trainer = _trainer(device, seed=12117)
    batch = list(_make_full_transaction_batch(graph, device))
    pair_credit = np.zeros_like(batch[15])
    pair_quality = np.zeros_like(batch[16])
    pair_credit[0, :, 0, 0] = 0.1
    pair_quality[0, :, 0, 0] = 1.0
    pair_credit[0, :, 1, 0] = -0.1
    pair_quality[0, :, 1, 0] = -1.0
    batch[15] = pair_credit
    batch[16] = pair_quality
    info, _priorities, _idxes = trainer.train_adj_on_batch(
        tuple(batch),
        use_adj_init=False,
        adj_update_round=1,
        pair_only_objective=True,
        enable_optimizer_transaction_diagnostics=True,
    )
    rows = info["_pair_selection_boundary_rows"]
    assert len(rows) == 4
    assert all(row["diagnostic_version"] == 8 for row in rows)
    direction_rows = info["_pair_direction_candidate_rows"]
    assert len(direction_rows) == 8
    assert sum(row["selected"] for row in direction_rows) == 1
    assert sum(
        row["direction_kind"] == "deficit_progress_seed"
        for row in direction_rows
    ) == 3
    assert all(np.isfinite(row["cosine_vs_full"]) for row in direction_rows)
    assert any(
        abs(row["cosine_vs_full"] - 1.0) > 1.0e-6
        for row in direction_rows
        if row["direction_kind"] == "deficit_progress_seed"
    )
    assert all(row["progress_target_present"] == 1 for row in direction_rows)
    assert all(row["progress_seed_member_count"] == 1 for row in direction_rows)
    assert all(
        row["progress_seed_zero_budget_excluded_count"] == 1
        for row in direction_rows
    )
    assert all(
        set(row["progress_seed_member_ordinals"].split("|")).isdisjoint(
            row[
                "progress_seed_zero_budget_excluded_ordinals"
            ].split("|")
        )
        for row in direction_rows
    )
    assert all(
        np.isclose(
            row["progress_worst_actual"]
            / row["progress_worst_required"],
            row["progress_min_completion"],
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        for row in direction_rows
    )
    assert all(row["identity_group_count"] == 2 for row in rows)
    assert all(row["identity_group_exposure_count"] == 2 for row in rows)
    for group_ordinal in (0, 1):
        group_rows = [
            row for row in rows
            if row["identity_group_ordinal"] == group_ordinal
        ]
        assert sum(
            row["identity_group_progress_member"] for row in group_rows
        ) == 1
        progress_row = next(
            row for row in group_rows
            if row["identity_group_progress_member"] == 1
        )
        assert progress_row["identity_group_progress_required"] > 0.0
        assert progress_row[
            "identity_group_actual_signed_margin_change"
        ] == progress_row["signed_margin_change"]
        assert min(
            row["signed_margin_change"] for row in group_rows
        ) == progress_row[
            "identity_group_worst_member_signed_margin_change"
        ]
    assert all(
        row["identity_group_actual_signed_margin_change"] > 0.0
        and row["identity_group_actual_completion_ratio"] > 0.0
        for row in rows
    )
    assert info[
        "pair_optimizer_transaction_pair_boundary_reverse_direction_count"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_approximately_zero_count"
    ] == 0.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_direction_candidate_count"
    ] == 8.0
    assert info[
        "pair_optimizer_transaction_pair_boundary_direction_valid_candidate_count"
    ] >= 1.0
    assert all(
        0.0 < row["selected_progress_floor_fraction"] <= 1.0
        and row["progress_min_completion"] > 0.0
        for row in rows
    )


def test_horizon_and_atomic_commit(device):
    # Horizon 3 cannot retain generation 660 through update 5.
    _args, graph, _trainer_value = _trainer(device)
    policy_buffer = _production_policy_buffer(graph, device, horizon=3)
    key = _snapshot_generation(
        policy_buffer, graph, device, 660, -1, 1, 60, 1660
    )
    policy_buffer.pair_pending_store.mark_pending(key, 1)
    policy_buffer.pair_pending_current_adj_update = 5
    policy_buffer.pair_pending_store.expire_out_of_horizon(5)
    _snapshot_generation(
        policy_buffer, graph, device, 689, 1, 5, 89, 1689
    )
    assert policy_buffer.prepare_pair_pending_training_batch(2) is None

    # Horizon 4 executes both real PPO epochs and commits each generation once.
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device, horizon=4)
    )
    sequence_trainer = _CaptureDiagnosticSequence(trainer)
    runner = _FakeRunner(sequence_trainer, args)
    rows = runner._run_pair_pending_outer_transaction(
        policy_id="policy_0",
        policy_buffer=policy_buffer,
        prepared=prepared,
        adjacency_update_round=5,
        graph_clip_stop_ratio=0.0,
        factor_clip_stop_ratio=0.0,
        min_ppo_epochs=2,
    )
    assert len(rows) == 2
    assert runner._adj_transaction_sequence_index == 2
    assert sequence_trainer.diagnostic_sequences == [0, 1]
    for key in prepared["keys"]:
        assert (
            policy_buffer.pair_pending_store.entries[key]["state"]
            == PAIR_EVIDENCE_COMMITTED
        )
    _assert_raises(
        RuntimeError,
        lambda: policy_buffer.pair_pending_store.prepare_class_complete(
            prepared["keys"], 6, 6, 2
        ),
        "cannot prepare again",
    )


def test_epoch_failure_rolls_back(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    sentinel_rnn_obs = torch.zeros(
        1, 4, 16, dtype=torch.float32, device=device
    )
    sentinel_dones = torch.zeros(
        1, 4, 1, dtype=torch.bool, device=device
    )
    sentinel_adj = torch.zeros(
        1, 4, 4, dtype=torch.float32, device=device
    )
    trainer._pair_selection_boundary_retention_observations = {
        77: {
            "observation_id": 77,
            "source_policy_id": "policy_0",
            "source_transaction_sequence_index": 12,
            "source_adjacency_update_round": 4,
            "source_episode_ordinal": 0,
            "source_episode_step": 700,
            "selection_context_sha256": (
                trainer
                ._pair_selection_boundary_retention_context_digest(
                    rnn_obs=sentinel_rnn_obs,
                    dones=sentinel_dones,
                    adj=sentinel_adj,
                    factor_index=0,
                    target_candidate_index=0,
                    target_sign=1.0,
                )
            ),
            "rnn_obs": sentinel_rnn_obs,
            "dones": sentinel_dones,
            "adj": sentinel_adj,
            "factor_index": 0,
            "target_candidate_index": 0,
            "target_canonical_identity": "0-1",
            "target_sign": 1.0,
            "commit_competitor_candidate_index": 1,
            "commit_competitor_canonical_identity": "0-2",
            "pre_signed_margin": -0.25,
            "commit_signed_margin": 0.25,
            "commit_rank": 1,
            "commit_active": 1,
            "created_at": 19,
            "protection_stopped": False,
            "protection_stop_reason": "",
            "protection_stop_clock": -1,
        }
    }
    trainer._pair_selection_boundary_retention_next_id = 78
    trainer._pair_selection_boundary_ordinary_update_clock = 19
    runner = _FakeRunner(_FailOnSecondEpoch(trainer), args)
    trainer_before = trainer.pair_pending_outer_transaction_state()
    store_before = copy.deepcopy(prepared["store_state_before_prepare"])
    rng_before = runner._capture_pair_pending_rng_state()
    _assert_raises(
        RuntimeError,
        lambda: runner._run_pair_pending_outer_transaction(
            "policy_0",
            policy_buffer,
            prepared,
            5,
            0.0,
            0.0,
            2,
        ),
        "injected epoch-1 failure",
    )
    _assert_nested_equal(
        trainer_before,
        trainer.pair_pending_outer_transaction_state(),
    )
    _assert_nested_equal(
        store_before,
        policy_buffer.pair_pending_store.state_dict(),
    )
    _assert_nested_equal(
        rng_before,
        runner._capture_pair_pending_rng_state(),
    )
    assert runner._adj_transaction_sequence_index == 0


def test_nonfinite_epoch_rolls_back(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    runner = _FakeRunner(_NonfiniteOnSecondEpoch(trainer), args)
    trainer_before = trainer.pair_pending_outer_transaction_state()
    store_before = copy.deepcopy(prepared["store_state_before_prepare"])
    rng_before = runner._capture_pair_pending_rng_state()
    _assert_raises(
        RuntimeError,
        lambda: runner._run_pair_pending_outer_transaction(
            "policy_0",
            policy_buffer,
            prepared,
            5,
            0.0,
            0.0,
            2,
        ),
        "strict pair descent",
    )
    _assert_nested_equal(
        trainer_before,
        trainer.pair_pending_outer_transaction_state(),
    )
    _assert_nested_equal(
        store_before,
        policy_buffer.pair_pending_store.state_dict(),
    )
    _assert_nested_equal(
        rng_before,
        runner._capture_pair_pending_rng_state(),
    )
    assert runner._adj_transaction_sequence_index == 0


def test_contradictory_pair_targets_remain_fail_loud(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    fields = [np.array(value, copy=True) for value in prepared["batch"]]
    pair_credit = np.asarray(fields[15])
    reduction_axes = tuple(range(1, pair_credit.ndim))
    negative_chunks = np.flatnonzero(np.any(
        pair_credit < 0.0, axis=reduction_axes
    ))
    positive_chunks = np.flatnonzero(np.any(
        pair_credit > 0.0, axis=reduction_axes
    ))
    assert negative_chunks.size == 1
    assert positive_chunks.size == 1
    negative_chunk = int(negative_chunks[0])
    positive_chunk = int(positive_chunks[0])

    # Equal and opposite exact labels on the same policy input and strict-pair
    # identity are not a bounded zero-gradient result.  They are contradictory
    # constraints and must remain fail-loud rather than being deferred as if
    # later evidence could make the current transaction feasible.
    for field_index in range(len(fields)):
        if field_index in (15, 33):
            continue
        fields[field_index][positive_chunk] = fields[field_index][
            negative_chunk
        ]
    negative_credit = np.array(
        fields[15][negative_chunk], copy=True
    )
    fields[15][positive_chunk].fill(0.0)
    fields[15][positive_chunk] = -negative_credit
    contradictory_prepared = dict(prepared)
    contradictory_prepared["batch"] = tuple(fields)

    runner = _FakeRunner(trainer, args)
    trainer_before = trainer.pair_pending_outer_transaction_state()
    store_before = copy.deepcopy(prepared["store_state_before_prepare"])
    rng_before = runner._capture_pair_pending_rng_state()
    _assert_raises(
        RuntimeError,
        lambda: runner._run_pair_pending_outer_transaction(
            "policy_0",
            policy_buffer,
            contradictory_prepared,
            5,
            0.0,
            0.0,
            2,
        ),
        "pending exact-pair target gradient constraints are jointly infeasible",
    )
    _assert_nested_equal(
        trainer_before,
        trainer.pair_pending_outer_transaction_state(),
    )
    _assert_nested_equal(
        store_before,
        policy_buffer.pair_pending_store.state_dict(),
    )
    _assert_nested_equal(
        rng_before,
        runner._capture_pair_pending_rng_state(),
    )
    assert runner._adj_transaction_sequence_index == 0
    assert policy_buffer.pair_pending_aborted_count == 1
    assert policy_buffer.pair_pending_rolled_back_count == 1
    assert policy_buffer.pair_pending_zero_gradient_abort_count == 0
    assert policy_buffer.pair_pending_committed_count == 0
    cohort_rows = [
        row for row in runner.logged_rows if "abort_reason" in row
    ]
    assert len(cohort_rows) == 1
    assert cohort_rows[0]["abort_reason"] == "EXCEPTION"
    assert cohort_rows[0]["committed"] == 0
    assert cohort_rows[0]["rolled_back"] == 1


def test_typed_zero_gradient_defers_without_consuming(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    wrapped_trainer = _ZeroGradientOnFirstEpoch(trainer)
    runner = _FakeRunner(wrapped_trainer, args)
    trainer_before = trainer.pair_pending_outer_transaction_state()
    store_before = copy.deepcopy(prepared["store_state_before_prepare"])
    rng_before = runner._capture_pair_pending_rng_state()
    rows = runner._run_pair_pending_outer_transaction(
        "policy_0",
        policy_buffer,
        prepared,
        5,
        0.0,
        0.0,
        2,
    )
    assert rows == []
    assert wrapped_trainer.calls == 1
    _assert_nested_equal(
        trainer_before,
        trainer.pair_pending_outer_transaction_state(),
    )
    _assert_nested_equal(
        store_before,
        policy_buffer.pair_pending_store.state_dict(),
    )
    _assert_nested_equal(
        rng_before,
        runner._capture_pair_pending_rng_state(),
    )
    assert runner._adj_transaction_sequence_index == 0
    assert policy_buffer.pair_pending_aborted_count == 1
    assert policy_buffer.pair_pending_rolled_back_count == 1
    assert policy_buffer.pair_pending_zero_gradient_abort_count == 1
    assert policy_buffer.pair_pending_committed_count == 0
    assert policy_buffer.pair_pending_objective_scope_contract_valid == 1.0
    cohort_rows = [
        row for row in runner.logged_rows if "abort_reason" in row
    ]
    assert len(cohort_rows) == 1
    assert cohort_rows[0]["abort_reason"] == "ZERO_GRADIENT"
    assert cohort_rows[0]["committed"] == 0
    assert cohort_rows[0]["rolled_back"] == 1


def test_typed_no_usable_update_defers_without_consuming(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    wrapped_trainer = _NoUsableUpdateOnFirstEpoch(trainer)
    runner = _FakeRunner(wrapped_trainer, args)
    trainer_before = trainer.pair_pending_outer_transaction_state()
    store_before = copy.deepcopy(prepared["store_state_before_prepare"])
    rng_before = runner._capture_pair_pending_rng_state()
    rows = runner._run_pair_pending_outer_transaction(
        "policy_0",
        policy_buffer,
        prepared,
        5,
        0.0,
        0.0,
        2,
    )
    assert rows == []
    assert wrapped_trainer.calls == 1
    _assert_nested_equal(
        trainer_before,
        trainer.pair_pending_outer_transaction_state(),
    )
    _assert_nested_equal(
        store_before,
        policy_buffer.pair_pending_store.state_dict(),
    )
    _assert_nested_equal(
        rng_before,
        runner._capture_pair_pending_rng_state(),
    )
    assert runner._adj_transaction_sequence_index == 0
    assert policy_buffer.pair_pending_aborted_count == 1
    assert policy_buffer.pair_pending_rolled_back_count == 1
    assert policy_buffer.pair_pending_committed_count == 0
    cohort_rows = [
        row for row in runner.logged_rows if "abort_reason" in row
    ]
    assert len(cohort_rows) == 1
    assert cohort_rows[0]["abort_reason"] == "NO_USABLE_UPDATE"
    assert cohort_rows[0]["committed"] == 0
    assert cohort_rows[0]["rolled_back"] == 1


def test_typed_exact_infeasible_defers_without_consuming(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    wrapped_trainer = _ExactInfeasibleOnFirstEpoch(trainer)
    runner = _FakeRunner(wrapped_trainer, args)
    trainer_before = trainer.pair_pending_outer_transaction_state()
    store_before = copy.deepcopy(prepared["store_state_before_prepare"])
    rng_before = runner._capture_pair_pending_rng_state()
    rows = runner._run_pair_pending_outer_transaction(
        "policy_0",
        policy_buffer,
        prepared,
        5,
        0.0,
        0.0,
        2,
    )
    assert rows == []
    assert wrapped_trainer.calls == 1
    _assert_nested_equal(
        trainer_before,
        trainer.pair_pending_outer_transaction_state(),
    )
    _assert_nested_equal(
        store_before,
        policy_buffer.pair_pending_store.state_dict(),
    )
    _assert_nested_equal(
        rng_before,
        runner._capture_pair_pending_rng_state(),
    )
    assert runner._adj_transaction_sequence_index == 0
    assert policy_buffer.pair_pending_aborted_count == 1
    assert policy_buffer.pair_pending_rolled_back_count == 1
    assert policy_buffer.pair_pending_committed_count == 0
    cohort_rows = [
        row for row in runner.logged_rows if "abort_reason" in row
    ]
    assert len(cohort_rows) == 1
    assert cohort_rows[0]["abort_reason"] == "EXACT_INFEASIBLE"
    failure_rows = [
        row for row in runner.logged_rows
        if row.get("failure_classification")
        == "sampled_grid_has_no_feasible_point"
    ]
    assert len(failure_rows) == 1


def test_standard_ppo_early_stop_is_not_applied_to_pair_only(device):
    args, _graph, trainer, policy_buffer, prepared = (
        _prepared_run107_equivalent(device)
    )
    runner = _FakeRunner(trainer, args)
    original = base_runner_module.should_stop_adj_ppo
    calls = []

    def _unexpected_standard_early_stop(**kwargs):
        calls.append(dict(kwargs))
        return True

    base_runner_module.should_stop_adj_ppo = _unexpected_standard_early_stop
    try:
        rows = runner._run_pair_pending_outer_transaction(
            "policy_0",
            policy_buffer,
            prepared,
            5,
            0.1,
            0.1,
            1,
        )
    finally:
        base_runner_module.should_stop_adj_ppo = original
    assert calls == []
    assert len(rows) == 2
    assert runner._adj_transaction_sequence_index == 2
    assert all(
        policy_buffer.pair_pending_store.entries[key]["state"]
        == PAIR_EVIDENCE_COMMITTED
        for key in prepared["keys"]
    )
    cohort_rows = [
        row for row in runner.logged_rows if "completed_ppo_epochs" in row
    ]
    assert len(cohort_rows) == 1
    assert cohort_rows[0]["committed"] == 1
    assert cohort_rows[0]["completed_ppo_epochs"] == 2
    assert cohort_rows[0]["standard_ppo_early_stop_applicable"] == 0
    assert cohort_rows[0]["all_configured_epochs_required"] == 1


def test_checkpoint_resume(device):
    args, graph, trainer = _trainer(device)
    policy_buffer = _production_policy_buffer(graph, device, horizon=4)
    key = _snapshot_generation(
        policy_buffer, graph, device, 660, -1, 1, 60, 1660
    )
    policy_buffer.pair_pending_store.mark_pending(key, 1)
    checkpoint = policy_buffer.pair_pending_state_dict()

    resumed = _production_policy_buffer(graph, device, horizon=4)
    resumed.load_pair_pending_state_dict(checkpoint)
    assert resumed.pair_pending_store.entries[key]["state"] == (
        PAIR_EVIDENCE_PENDING
    )
    resumed.pair_pending_current_adj_update = 5
    _snapshot_generation(
        resumed, graph, device, 689, 1, 5, 89, 1689
    )
    prepared = resumed.prepare_pair_pending_training_batch(2)
    runner = _FakeRunner(trainer, args)
    rows = runner._run_pair_pending_outer_transaction(
        "policy_0", resumed, prepared, 5, 0.0, 0.0, 2
    )
    assert len(rows) == 2
    assert all(
        resumed.pair_pending_store.entries[value]["state"]
        == PAIR_EVIDENCE_COMMITTED
        for value in prepared["keys"]
    )


def test_standard_pair_transaction_prevents_pending_reuse(device):
    _args, graph, _trainer_value = _trainer(device)
    policy_buffer = _production_policy_buffer(graph, device, horizon=4)
    negative = _snapshot_generation(
        policy_buffer, graph, device, 660, -1, 1, 60, 1660
    )
    positive = _snapshot_generation(
        policy_buffer, graph, device, 689, 1, 1, 89, 1689
    )
    committed = (
        policy_buffer.pair_pending_store
        .commit_available_from_standard_transaction(
            (negative, positive),
            committed_adj_update=1,
        )
    )
    assert set(committed) == {negative, positive}
    assert all(
        policy_buffer.pair_pending_store.entries[key]["state"]
        == PAIR_EVIDENCE_COMMITTED
        for key in committed
    )
    assert policy_buffer.prepare_pair_pending_training_batch(2) is None

    # A later standard transaction may legitimately contain one generation
    # already mirrored above plus a newly available opposite-sign generation.
    # The complete transaction remains class-complete, while only the new
    # member should change state.
    policy_buffer.last_sample_pair_evidence_episode_rows = [
        {
            "selected_for_training": 1,
            "pair_evidence_episode": 1,
            "episode_generation": 660,
        },
        {
            "selected_for_training": 1,
            "pair_evidence_episode": 1,
            "episode_generation": 689,
        },
    ]
    replayed_keys = (
        policy_buffer.standard_pair_transaction_generation_keys()
    )
    assert set(replayed_keys) == {negative, positive}
    assert policy_buffer.commit_standard_pair_transaction(
        replayed_keys,
        adjacency_update_index=2,
    ) == tuple()

    new_positive = _snapshot_generation(
        policy_buffer, graph, device, 690, 1, 2, 90, 1690
    )
    policy_buffer.last_sample_pair_evidence_episode_rows = [
        {
            "selected_for_training": 1,
            "pair_evidence_episode": 1,
            "episode_generation": 660,
        },
        {
            "selected_for_training": 1,
            "pair_evidence_episode": 1,
            "episode_generation": 690,
        },
    ]
    mixed_keys = policy_buffer.standard_pair_transaction_generation_keys()
    assert set(mixed_keys) == {negative, new_positive}
    newly_committed = policy_buffer.commit_standard_pair_transaction(
        mixed_keys,
        adjacency_update_index=2,
    )
    assert newly_committed == (new_positive,)
    assert (
        policy_buffer.pair_pending_store.entries[negative][
            "committed_adj_update"
        ] == 1
    )
    assert (
        policy_buffer.pair_pending_store.entries[new_positive]["state"]
        == PAIR_EVIDENCE_COMMITTED
    )
    assert policy_buffer.prepare_pair_pending_training_batch(2) is None


def test_multiple_capture_events_merge_one_generation_population(device):
    _args, graph, _trainer_value = _trainer(device)
    policy_buffer = _production_policy_buffer(graph, device, horizon=4)
    shared_record = dict(policy_buffer.strict_pair_event_provenance[0][0])
    shared_record.update({
        "event_id": 62,
        "target_id": 1,
    })
    second_record = dict(policy_buffer.strict_pair_event_provenance[0][0])
    second_record.update({
        "event_id": 61,
        "target_id": 2,
        "participant_slots": (0, 2),
        "pair_factor_index": 1,
        "pair_identity": "0-2",
        "capture_factor_index": 1,
        "capture_factor_identity": "0-2",
        "raw_transition_quality": 0.77,
    })
    policy_buffer.strict_pair_event_provenance[0] = (
        policy_buffer.strict_pair_event_provenance[0][0],
        shared_record,
        second_record,
    )
    policy_buffer.pair_to_triplet_transition_score[0, 0, 1, 0] = 0.77
    fields = list(_one_episode_fields(graph, device, 660))
    fields[17][0, 0, 1, 0] = 0.77
    policy_buffer._capture_pair_pending_snapshots(
        _flat_episode(tuple(fields)),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        np.asarray([0], dtype=np.int64),
        2,
    )
    negative_keys = policy_buffer.pair_pending_store.keys_for_generation(
        "policy_0", 660
    )
    assert len(negative_keys) == 3
    event_masks = [
        np.asarray(
            policy_buffer.pair_pending_store.entries[key]["batch"][17]
        )
        for key in negative_keys
    ]
    shared_masks = [
        mask for mask in event_masks
        if np.any(mask[..., 0, :] > 0.0)
    ]
    assert len(shared_masks) == 2
    shared_overlap = (
        (shared_masks[0] > 0.0) & (shared_masks[1] > 0.0)
    )
    assert np.any(shared_overlap)
    assert np.array_equal(
        shared_masks[0][shared_overlap],
        shared_masks[1][shared_overlap],
    )
    for key in negative_keys:
        policy_buffer.pair_pending_store.mark_pending(key, 1)
    policy_buffer.pair_pending_current_adj_update = 5
    positive_key = _snapshot_generation(
        policy_buffer, graph, device, 689, 1, 5, 89, 1689
    )
    prepared = policy_buffer.prepare_pair_pending_training_batch(2)
    assert prepared is not None
    assert set(prepared["keys"]) == set(negative_keys) | {positive_key}
    # The three event-local negative entries merge back into one immutable
    # replay population. Two events share one target transition, which is
    # trained exactly once; the other event contributes one distinct target.
    assert prepared["episode_count"] == 2
    assert prepared["chunk_count"] == 2
    negative_target_count = int(np.sum(
        np.asarray(prepared["batch"][15])[:, :, :2, 0] < 0.0
    ))
    assert negative_target_count == 2


def test_default_off_snapshot_is_noop(device):
    _args, graph, _trainer_value = _trainer(device)
    policy_buffer = _production_policy_buffer(graph, device, horizon=4)
    policy_buffer.pair_pending_enabled = False
    fields = _one_episode_fields(graph, device, 660)
    before = tuple(np.array(value, copy=True) for value in fields)
    policy_buffer._capture_pair_pending_snapshots(
        _flat_episode(fields),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        np.asarray([0], dtype=np.int64),
        2,
    )
    assert len(policy_buffer.pair_pending_store) == 0
    for left, right in zip(before, fields):
        assert np.array_equal(left, right)


def test_default_off_full_training_is_identical(device):
    _args_a, graph_a, trainer_a = _trainer(device, seed=977)
    _args_b, graph_b, trainer_b = _trainer(device, seed=977)
    batch_a = _make_full_transaction_batch(graph_a, device)
    batch_b = tuple(np.array(value, copy=True) for value in batch_a)
    policy_buffer = _production_policy_buffer(graph_b, device, horizon=4)
    policy_buffer.pair_pending_enabled = False

    random.seed(119)
    np.random.seed(119)
    torch.manual_seed(119)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(119)
    rng_before = RecRunner._capture_pair_pending_rng_state()
    info_a, _, _ = trainer_a.train_adj_on_batch(
        batch_a,
        False,
        adj_update_round=1,
        pair_only_objective=False,
        enable_optimizer_transaction_diagnostics=True,
    )
    rng_after_a = RecRunner._capture_pair_pending_rng_state()

    RecRunner._restore_pair_pending_rng_state(rng_before)
    snapshot_fields = _one_episode_fields(graph_b, device, 660)
    snapshot_before = tuple(
        np.array(value, copy=True) for value in snapshot_fields
    )
    policy_buffer._capture_pair_pending_snapshots(
        _flat_episode(snapshot_fields),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        np.asarray([0], dtype=np.int64),
        2,
    )
    assert len(policy_buffer.pair_pending_store) == 0
    assert all(
        np.array_equal(left, right)
        for left, right in zip(snapshot_before, snapshot_fields)
    )
    info_b, _, _ = trainer_b.train_adj_on_batch(
        batch_b,
        False,
        adj_update_round=1,
        pair_only_objective=False,
        enable_optimizer_transaction_diagnostics=True,
    )
    rng_after_b = RecRunner._capture_pair_pending_rng_state()

    for name in (
            "rl_loss",
            "pair_pursuit_factor_loss_contribution",
            "pair_gradient_norm",
            "pair_optimizer_transaction_combined_grad_norm_preclip",
            "pair_optimizer_transaction_combined_grad_norm_postclip",
            "pair_optimizer_transaction_optimizer_step_after"):
        assert np.isclose(
            float(info_a[name]), float(info_b[name]),
            rtol=0.0, atol=0.0,
        )
    state_a = trainer_a.pair_pending_outer_transaction_state()
    state_b = trainer_b.pair_pending_outer_transaction_state()
    optimizer_state_a = state_a.pop("adj_optimizers")
    optimizer_state_b = state_b.pop("adj_optimizers")
    # Keep the serialized optimizer checkpoint contract covered without
    # comparing instance-local serialized parameter identifiers.
    assert set(optimizer_state_a) == set(optimizer_state_b)
    assert optimizer_state_a["version"] == optimizer_state_b["version"]
    _assert_nested_equal(state_a, state_b, "default_off_trainer_state")
    _assert_optimizer_semantically_equal(
        trainer_a.adj_optimizer,
        trainer_b.adj_optimizer,
        "default_off_standard_optimizer",
    )
    _assert_optimizer_semantically_equal(
        trainer_a.candidate_residual_optimizer,
        trainer_b.candidate_residual_optimizer,
        "default_off_residual_optimizer",
    )
    _assert_optimizer_semantically_equal(
        trainer_a.pair_pending_optimizer,
        trainer_b.pair_pending_optimizer,
        "default_off_pair_optimizer",
    )
    _assert_nested_equal(rng_after_a, rng_after_b)


def run_device(device):
    test_real_pair_only_forward(device)
    test_pair_only_does_not_consume_poisoned_standard_adam_state(device)
    test_pair_direction_guard_corrects_conflicting_pair_moment(device)
    test_standard_pair_direction_guard_corrects_conflicting_moment(device)
    test_run118_joint_exact_backtracking_uses_production_transaction(device)
    test_run118_late_exact_failure_rolls_back_production_transaction(device)
    test_v15_no_actionable_pair_path_does_not_enter_joint_guard(device)
    test_run122_nearest_member_identity_uses_production_transaction(device)
    test_horizon_and_atomic_commit(device)
    test_epoch_failure_rolls_back(device)
    test_nonfinite_epoch_rolls_back(device)
    test_contradictory_pair_targets_remain_fail_loud(device)
    test_typed_zero_gradient_defers_without_consuming(device)
    test_typed_no_usable_update_defers_without_consuming(device)
    test_typed_exact_infeasible_defers_without_consuming(device)
    test_standard_ppo_early_stop_is_not_applied_to_pair_only(device)
    test_checkpoint_resume(device)
    test_standard_pair_transaction_prevents_pending_reuse(device)
    test_multiple_capture_events_merge_one_generation_population(device)
    test_default_off_snapshot_is_noop(device)
    test_default_off_full_training_is_identical(device)
    print("PASS pair pending production integration on {}".format(device))


def main():
    run_device(torch.device("cpu"))
    if torch.cuda.is_available():
        run_device(torch.device("cuda:0"))
    else:
        print("SKIP CUDA pair pending production integration: unavailable")
    print("PyTorch version={}".format(torch.__version__))
    print("All pair pending production integration tests passed")


if __name__ == "__main__":
    main()
