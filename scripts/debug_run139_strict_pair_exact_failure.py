#!/usr/bin/env python
"""Run139 failure-path contract and exact-search diagnostic fixture.

The run139 tensors were not checkpointed at the failing transaction.  This
fixture therefore keeps the distinction explicit: transaction location comes
from the real run139 CSV, while the constraint geometry below is synthetic and
tests the production acceptance/search/CSV plumbing added for the next exact
failure.  It must never be reported as a replay of run139's missing tensors.
"""

import csv
import os
import sys
import tempfile
import copy

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import (  # noqa: E402
    StrictPairExactInfeasibleError,
    _joint_exact_constraint_acceptance,
    _joint_exact_origin_preservation_acceptance,
    _select_joint_exact_progress_direction,
)
from runner.base_runner import (  # noqa: E402
    RecRunner,
    _STRICT_PAIR_EXACT_FAILURE_FIELDS,
)
import algorithms.sddfg.r_sddfg as r_sddfg_module  # noqa: E402
import scripts.debug_pair_pending_production_integration as production  # noqa: E402


RUN139_ROOT = os.path.join(
    REPO_ROOT,
    "scripts",
    "results",
    "wolfpack",
    "sddfg",
    "sddfg_intra_ep_4to6_r2_j1_rec30_seed1",
    "run139",
)


def _real_run139_location():
    path = os.path.join(
        RUN139_ROOT,
        "run139_progress_train_adj_transaction.csv",
    )
    with open(path, "r", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError("run139 transaction CSV is empty")
    last = rows[-1]
    last_step = int(float(last["env_step"]))
    last_round = int(float(last["adjacency_update_round"]))
    last_sequence = int(float(last["transaction_sequence_index"]))
    if (last_step, last_round, last_sequence) != (173600, 209, 829):
        raise RuntimeError(
            "run139 last committed transaction changed: {}".format(
                (last_step, last_round, last_sequence)
            )
        )
    unique_steps = sorted(set(int(float(row["env_step"])) for row in rows))
    interval = unique_steps[-1] - unique_steps[-2]
    if interval != 800:
        raise RuntimeError("run139 adjacency interval is not 800")
    return {
        "last_step": last_step,
        "last_round": last_round,
        "last_sequence": last_sequence,
        "failed_step": last_step + interval,
        "failed_round": last_round + 1,
        "failed_sequence": last_sequence + 1,
        "failed_epoch": 0,
        "failed_partition": 0,
    }


def _synthetic_evaluator(limiter):
    reference = torch.tensor(0.0, dtype=torch.float32)

    def evaluate(scale):
        scale_tensor = reference.new_tensor(float(scale))
        signed_boundary = torch.stack((
            0.02 * scale_tensor,
            (
                -0.01 * scale_tensor
                if limiter == "boundary"
                else 0.01 * scale_tensor
            ),
        )).reshape(1, 2)
        signed_exact = torch.stack((
            0.02 * scale_tensor,
            (
                -0.01 * scale_tensor
                if limiter == "exact_score"
                else 0.01 * scale_tensor
            ),
        )).reshape(1, 2)
        candidate_change = (
            0.01 * scale_tensor
            if limiter == "candidate"
            else -0.01 * scale_tensor
        )
        lifecycle_floor = reference.new_tensor([[0.25]])
        lifecycle_margin = lifecycle_floor + (
            torch.where(
                scale_tensor > 0.0,
                scale_tensor.new_tensor(-0.01),
                scale_tensor.new_tensor(0.0),
            )
            if limiter == "lifecycle"
            else 0.01 * scale_tensor
        )
        info = _joint_exact_constraint_acceptance(
            signed_boundary_change=signed_boundary,
            actionable_pair_mask=torch.ones_like(signed_boundary),
            signed_exact_score_change=signed_exact,
            candidate_loss_change=candidate_change,
            lifecycle_signed_margin=lifecycle_margin,
            lifecycle_signed_floor=lifecycle_floor,
            lifecycle_target_mask=torch.ones_like(lifecycle_floor),
            lifecycle_tolerance=torch.zeros_like(lifecycle_floor),
        )
        info.update({
            "progress_target_present": True,
            "progress_worst_actual": float(
                signed_boundary.reshape(-1)[0].item()
            ),
            "progress_worst_required": 0.02,
            "progress_min_completion": float(scale),
            "progress_mean_completion": float(scale),
            "competitor_candidate_indices": (7, 8),
            "target_ranks": (2, 3),
            "target_active": (0, 0),
        })
        return info

    return evaluate, reference


def _exact_search_trace():
    limiters = (
        "boundary",
        "candidate",
        "lifecycle",
        "exact_score",
        "boundary",
        "candidate",
        "lifecycle",
        "exact_score",
    )
    evaluators = []
    for ordinal, limiter in enumerate(limiters):
        evaluate, _reference = _synthetic_evaluator(limiter)
        evaluators.append(("direction_{}".format(ordinal), evaluate))
    result = _select_joint_exact_progress_direction(
        direction_evaluators=evaluators,
        max_halvings=20,
        refinement_steps=12,
        maximum_scale=1.0,
    )
    if result["valid"]:
        raise RuntimeError("synthetic all-limiter fixture unexpectedly passed")
    if result["candidate_count"] != 8:
        raise RuntimeError("synthetic direction count changed")
    expected_scales = [1.0] + [0.5 ** index for index in range(1, 21)]
    for candidate in result["candidate_results"]:
        observed = [row["scale"] for row in candidate["trial_trace"]]
        if observed[:len(expected_scales)] != expected_scales:
            raise RuntimeError("failed exact search lost its base dyadic scales")
        if len(observed) <= len(expected_scales):
            raise RuntimeError("failed exact search omitted its expanded lattice")

    origin_info = evaluators[0][1](0.0)
    origin = _joint_exact_origin_preservation_acceptance(
        origin_info,
        torch.tensor(0.0, dtype=torch.float32),
    )
    if not origin["valid"]:
        raise RuntimeError(
            "scale-zero preservation was conflated with strict progress"
        )
    if origin_info["valid"]:
        raise RuntimeError("scale zero unexpectedly satisfied strict progress")
    return result, origin


def _payload_template():
    runtime_fields = {
        "run_id",
        "env_step",
        "adjacency_update_round",
        "ppo_epoch_index",
        "policy_id",
        "partition_index",
        "transaction_sequence_index",
    }
    payload = {}
    for field in _STRICT_PAIR_EXACT_FAILURE_FIELDS:
        if field not in runtime_fields:
            payload[field] = None
    payload.update({
        "diagnostic_version": 1,
        "optimizer_kind": "standard_adam",
        "target_count": 2,
        "target_candidate_indices": "4|5",
        "target_canonical_identities": "0-1|1-2",
        "target_signs": "1|-1",
        "target_transition_indices": "10|20",
        "target_factor_indices": "0|1",
        "target_selected_episode_ordinals": "0|1",
        "target_episode_transition_steps": "10|20",
        "target_pair_evidence_flags": "1|1",
        "candidate_count": 8,
        "diagnostic_probe_valid_count": 0,
        "failure_classification": "sampled_grid_has_no_feasible_point",
        "origin_preservation_valid": 1,
        "origin_preservation_tolerance": 128.0 * torch.finfo(
            torch.float32
        ).eps,
        "candidate_ordinal": 0,
        "direction_kind": "adam_projection",
        "progress_floor_fraction": 1.0,
        "direction_norm": 0.5,
        "cosine_vs_full": 1.0,
        "evaluation_ordinal": 0,
        "evaluation_kind": "search",
        "scale": 1.0,
        "parameter_displacement_norm": 0.5,
        "predicted_exact_score_min_change": 0.01,
        "predicted_boundary_min_change": 0.01,
        "valid": 0,
        "boundary_valid": 0,
        "boundary_min_signed_change": -0.01,
        "exact_score_valid": 1,
        "exact_score_min_signed_change": 0.01,
        "candidate_valid": 1,
        "candidate_loss_change": -0.01,
        "lifecycle_valid": 1,
        "lifecycle_violation_count": 0,
        "lifecycle_min_signed_gap": 0.01,
        "limiting_constraint_type": "boundary",
        "limiting_target_ordinal": 1,
        "progress_target_present": 1,
        "progress_worst_actual": 0.02,
        "progress_worst_required": 0.02,
        "progress_min_completion": 1.0,
        "progress_mean_completion": 1.0,
        "competitor_candidate_indices": "7|8",
        "target_ranks": "2|3",
        "target_active": "0|0",
    })
    return payload


def _runner_csv_contract(location):
    with tempfile.TemporaryDirectory(prefix="run139_exact_failure_") as temp:
        run_dir = os.path.join(temp, "run_fixture")
        os.makedirs(run_dir)
        runner = RecRunner.__new__(RecRunner)
        runner.run_dir = run_dir
        runner.total_env_steps = location["failed_step"]
        runner._adj_transaction_sequence_index = location["failed_sequence"]
        error = StrictPairExactInfeasibleError(
            "fixture failure",
            diagnostic_rows=(_payload_template(),),
        )

        class FailingTrainer(object):
            def train_adj_on_batch(self, *args, **kwargs):
                raise error

        runner.trainer = FailingTrainer()
        try:
            runner._train_adj_on_batch_with_exact_failure_logging(
                sample=object(),
                use_adj_init=False,
                adjacency_update_round=location["failed_round"],
                consume_factor_credit_observations=True,
                policy_id="policy_0",
                transaction_sequence_index=location["failed_sequence"],
                ppo_epoch_index=location["failed_epoch"],
                partition_index=location["failed_partition"],
            )
        except StrictPairExactInfeasibleError as caught:
            if caught is not error:
                raise RuntimeError("runner replaced the exact failure")
        else:
            raise RuntimeError("runner swallowed the exact failure")
        path = os.path.join(
            run_dir,
            "run_fixture_progress_train_strict_pair_exact_failure.csv",
        )
        with open(path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            if tuple(reader.fieldnames or ()) != (
                    _STRICT_PAIR_EXACT_FAILURE_FIELDS):
                raise RuntimeError("strict-pair failure CSV header drifted")
        if len(rows) != 1:
            raise RuntimeError("strict-pair failure CSV row was not persisted")
        if int(rows[0]["transaction_sequence_index"]) != 830:
            raise RuntimeError("strict-pair failure sequence was not preserved")


def _bounded_deferral_contract(location):
    """Only a v5 exhaustive, origin-valid no-feasible result may defer."""
    with tempfile.TemporaryDirectory(prefix="strict_exact_deferral_") as temp:
        run_dir = os.path.join(temp, "run_fixture")
        os.makedirs(run_dir)
        runner = RecRunner.__new__(RecRunner)
        runner.run_dir = run_dir
        runner.total_env_steps = location["failed_step"]
        runner._adj_transaction_sequence_index = location["failed_sequence"]
        payload = _payload_template()
        payload.update({
            "diagnostic_version": int(
                r_sddfg_module.STRICT_PAIR_EXACT_FAILURE_DIAGNOSTIC_VERSION
            ),
            "bounded_search_exhaustive": 1,
            "failure_classification": "sampled_grid_has_no_feasible_point",
            "origin_preservation_valid": 1,
            "diagnostic_probe_valid_count": 0,
        })
        error = StrictPairExactInfeasibleError(
            "exhaustive fixture failure",
            diagnostic_rows=(payload,),
        )
        if not error.strict_pair_exact_bounded_deferral_safe:
            raise RuntimeError("exhaustive exact failure was not deferrable")

        class FailingTrainer(object):
            def train_adj_on_batch(self, *args, **kwargs):
                raise error

        runner.trainer = FailingTrainer()
        train_info, priorities, idxes = (
            runner._train_adj_on_batch_with_exact_failure_logging(
                sample=object(),
                use_adj_init=False,
                adjacency_update_round=location["failed_round"],
                consume_factor_credit_observations=True,
                policy_id="policy_0",
                transaction_sequence_index=location["failed_sequence"],
                ppo_epoch_index=location["failed_epoch"],
                partition_index=location["failed_partition"],
            )
        )
        if float(train_info.get(
                "_strict_pair_exact_bounded_deferred", 0.0)) != 1.0:
            raise RuntimeError("bounded exact failure omitted deferral sentinel")
        if priorities is not None or idxes is not None:
            raise RuntimeError("bounded exact deferral returned replay updates")
        if runner._adj_transaction_sequence_index != location["failed_sequence"]:
            raise RuntimeError("bounded exact deferral advanced the sequence")

        missed_payload = dict(payload)
        missed_payload["failure_classification"] = (
            "search_missed_feasible_midpoint"
        )
        missed_payload["diagnostic_probe_valid_count"] = 1
        missed = StrictPairExactInfeasibleError(
            "missed-search fixture failure",
            diagnostic_rows=(missed_payload,),
        )
        if missed.strict_pair_exact_bounded_deferral_safe:
            raise RuntimeError("missed feasible search became deferrable")


def _production_failure_contract(location):
    device = torch.device("cpu")
    _args, graph, trainer = production._trainer(device, seed=13930)
    batch = production._make_full_transaction_batch(
        graph,
        device,
        candidate_target=True,
        candidate_target_sign=-1.0,
    )
    production._seed_exact_candidate_lifecycle_from_batch(
        trainer, graph, batch, device
    )
    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.adj_parameters
    ]
    optimizer_before = copy.deepcopy(trainer.adj_optimizer.state_dict())
    original_acceptance = r_sddfg_module._joint_exact_constraint_acceptance

    def reject_every_trial(*args, **kwargs):
        result = dict(original_acceptance(*args, **kwargs))
        result["valid"] = False
        result["lifecycle_valid"] = False
        result["lifecycle_violation_count"] = 1
        result["limiting_constraint_type"] = "lifecycle"
        return result

    with tempfile.TemporaryDirectory(prefix="run139_production_failure_") as temp:
        run_dir = os.path.join(temp, "run_fixture")
        os.makedirs(run_dir)
        runner = RecRunner.__new__(RecRunner)
        runner.run_dir = run_dir
        runner.total_env_steps = location["failed_step"]
        runner._adj_transaction_sequence_index = location["failed_sequence"]
        runner.trainer = trainer
        r_sddfg_module._joint_exact_constraint_acceptance = reject_every_trial
        try:
            try:
                runner._train_adj_on_batch_with_exact_failure_logging(
                    sample=batch,
                    use_adj_init=False,
                    adjacency_update_round=location["failed_round"],
                    consume_factor_credit_observations=True,
                    policy_id="policy_0",
                    transaction_sequence_index=location["failed_sequence"],
                    ppo_epoch_index=location["failed_epoch"],
                    partition_index=location["failed_partition"],
                )
            except StrictPairExactInfeasibleError as error:
                diagnostic_rows = error.strict_pair_exact_failure_rows
            else:
                raise RuntimeError(
                    "production exact failure was unexpectedly swallowed"
                )
        finally:
            r_sddfg_module._joint_exact_constraint_acceptance = (
                original_acceptance
            )
        if not diagnostic_rows:
            raise RuntimeError("production exact failure rows are empty")
        runtime_fields = {
            "run_id",
            "env_step",
            "adjacency_update_round",
            "ppo_epoch_index",
            "policy_id",
            "partition_index",
            "transaction_sequence_index",
        }
        expected_payload = set(_STRICT_PAIR_EXACT_FAILURE_FIELDS) - (
            runtime_fields
        )
        if any(set(row.keys()) != expected_payload for row in diagnostic_rows):
            raise RuntimeError("production exact failure payload schema drifted")
        path = os.path.join(
            run_dir,
            "run_fixture_progress_train_strict_pair_exact_failure.csv",
        )
        with open(path, "r", newline="") as csv_file:
            persisted = list(csv.DictReader(csv_file))
        if len(persisted) != len(diagnostic_rows):
            raise RuntimeError("production exact failure rows were truncated")
    for parameter, before in zip(trainer.adj_parameters, parameters_before):
        if not torch.equal(parameter.detach(), before):
            raise RuntimeError("production exact failure parameter rollback failed")
    production._assert_nested_equal(
        optimizer_before,
        trainer.adj_optimizer.state_dict(),
        "run139_production_optimizer_rollback",
    )
    return len(diagnostic_rows)


def main():
    location = _real_run139_location()
    result, origin = _exact_search_trace()
    _runner_csv_contract(location)
    _bounded_deferral_contract(location)
    production_failure_rows = _production_failure_contract(location)
    limiter_counts = {}
    for candidate in result["candidate_results"]:
        limiter = candidate["info"]["limiting_constraint_type"]
        limiter_counts[limiter] = limiter_counts.get(limiter, 0) + 1
    print("run139 strict-pair failure diagnostic fixture PASS")
    print("real_location={}".format(location))
    print("synthetic_origin_preservation_valid={}".format(origin["valid"]))
    print("synthetic_trial_rows={}".format(sum(
        len(candidate["trial_trace"])
        for candidate in result["candidate_results"]
    )))
    print("synthetic_last_limiter_counts={}".format(limiter_counts))
    print("production_failure_rows={}".format(production_failure_rows))


if __name__ == "__main__":
    main()
