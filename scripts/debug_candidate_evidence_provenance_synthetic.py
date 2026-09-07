#!/usr/bin/env python
"""Fail-loud, trajectory-neutral candidate evidence provenance checks."""

from __future__ import print_function

import csv
import copy
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import wandb  # noqa: F401
except ImportError:
    sys.modules["wandb"] = types.ModuleType("wandb")

from algorithms.sddfg.r_sddfg import (  # noqa: E402
    _candidate_evidence_consumption_trace_rows,
)
from runner.base_runner import (  # noqa: E402
    RecRunner,
    _build_candidate_evidence_provenance_csv_rows,
    _candidate_event_by_target_key,
    _join_candidate_evidence_consumption_rows,
)
from scripts.debug_candidate_score_to_rank_counterfactual import (  # noqa: E402
    _collapse_candidate_evidence_by_generation,
    _independent_generation_counterfactual,
    _join_candidate_evidence_provenance,
    _validate_candidate_evidence_quality_contract,
)
from utils.adj_buffer import (  # noqa: E402
    build_candidate_evidence_provenance_rows,
)
from utils.pair_credit import (  # noqa: E402
    build_capture_identity_factor_weights,
    canonical_capture_factor_catalog,
)


def _draft_case(
        generation=7,
        sign=1,
        multi_identity=False,
        first_identity="0-1",
        first_order=2,
        first_candidate_index=0,
        participant_slots=(0, 1, 2),
        static_dynamic_class="dynamic"):
    time_count = 4
    candidate_count = 10
    event_records = [{
        "event_id": 41 + generation,
        "target_id": 81 + generation,
        "participant_slots": tuple(participant_slots),
        "candidate_index": int(first_candidate_index),
        "candidate_identity": str(first_identity),
        "candidate_order": int(first_order),
        "identity_event_weight": 0.5 if multi_identity else 1.0,
        "capture_step": 1,
        "static_dynamic_class": str(static_dynamic_class),
    }]
    if multi_identity:
        event_records.append({
            "event_id": 41 + generation,
            "target_id": 81 + generation,
            "participant_slots": (0, 1, 2),
            "candidate_index": 1,
            "candidate_identity": "0-2",
            "candidate_order": 2,
            "identity_event_weight": 0.5,
            "capture_step": 1,
            "static_dynamic_class": "dynamic",
        })
    coefficient = 0.15
    delta = np.zeros((time_count, 1, candidate_count), dtype=np.float32)
    for record in event_records:
        delta[
            record["capture_step"],
            0,
            record["candidate_index"],
        ] = (
            float(sign)
            * coefficient
            * float(record["identity_event_weight"])
        )
    behavior = np.zeros(delta.shape + (4,), dtype=np.float32)
    behavior[..., 3] = 4.0
    rows = build_candidate_evidence_provenance_rows(
        event_records_by_episode=[event_records],
        replay_slot_indices=np.asarray([2], dtype=np.int64),
        replay_generations=np.asarray([generation], dtype=np.int64),
        environment_episode_ids=np.asarray(
            [1000 + generation],
            dtype=np.int64,
        ),
        base_selected=np.asarray([1], dtype=np.int64),
        support_selected=np.asarray([0], dtype=np.int64),
        outcome_success=np.asarray([sign > 0], dtype=np.int64),
        episode_outcome_advantage=np.asarray(
            [float(sign)],
            dtype=np.float32,
        ),
        capture_event_mass_per_episode=np.asarray([1.0], dtype=np.float32),
        candidate_coefficient=coefficient,
        candidate_identity_delta=delta,
        candidate_behavior=behavior,
        terminal_steps=np.asarray([3], dtype=np.int64),
    )
    return rows, delta, behavior


def _consumption_rows(delta, behavior):
    flat_delta = torch.as_tensor(delta.reshape(-1, delta.shape[-1]))
    flat_behavior_version = torch.as_tensor(
        behavior[..., 3].reshape(-1, delta.shape[-1])
    )
    provenance = []
    for step in range(delta.shape[0]):
        for episode_ordinal in range(delta.shape[1]):
            provenance.append([
                0.0,
                float(episode_ordinal),
                float(step),
            ])
    return _candidate_evidence_consumption_trace_rows(
        candidate_identity_delta=flat_delta,
        behavior_candidate_version=flat_behavior_version,
        replay_population_provenance=torch.tensor(
            provenance,
            dtype=torch.float32,
        ),
    )


def _final_rows(generation=7, sign=1, multi_identity=False):
    drafts, delta, behavior = _draft_case(
        generation=generation,
        sign=sign,
        multi_identity=multi_identity,
    )
    joined = _join_candidate_evidence_consumption_rows(
        _consumption_rows(delta, behavior),
        drafts,
    )
    rows = _build_candidate_evidence_provenance_csv_rows(
        joined_evidence_rows=joined,
        run_id="run_provenance_synthetic",
        env_step=8000 + generation,
        policy_id="policy_0",
        current_candidate_policy_version=6,
    )
    return rows, joined


def _transaction_row(provenance, epoch, sequence, signed_change):
    return {
        "policy_id": provenance["policy_id"],
        "episode_generation": provenance["replay_generation"],
        "environment_episode_id": provenance["environment_episode_id"],
        "capture_event_id": provenance["capture_event_id"],
        "capture_prey_id": provenance["prey_id"],
        "canonical_identity": provenance["candidate_identity"],
        "target_sign": provenance["target_sign"],
        "transaction_sequence_index": sequence,
        "target_row_sequence_within_transaction": 0,
        "env_step": provenance["env_step"],
        "ppo_epoch_index": epoch,
        "partition_index": 0,
        "factor_order": provenance["candidate_order"],
        "candidate_participant_slots": provenance["participant_slots"],
        "candidate_static_dynamic_class": provenance[
            "static_dynamic_class"
        ],
        "target_weight": provenance["final_target_mass"],
        "signed_margin_change": signed_change,
        "pre_next_better_candidate_index": 4,
        "pre_next_better_margin_gap": 0.08,
        "pre_boundary_deficit": 1.5,
        "pre_valid_population_count": 30,
    }


def test_real_capture_event_records_are_identity_local():
    catalog = canonical_capture_factor_catalog(4, 3)
    pair_index = catalog.index((0, 1))
    adjacency = np.zeros((1, 4, 2), dtype=np.int64)
    adjacency[0, [0, 2], 0] = 1
    adjacency[0, [1, 3], 1] = 1
    result = build_capture_identity_factor_weights(
        current_adj=adjacency,
        capture_events_by_episode=[[
            {
                "event_id": 3,
                "target_id": 9,
                "participant_slots": [0, 1],
            },
        ]],
        expected_capture_counts=np.asarray([1.0], dtype=np.float32),
    )
    records = result["candidate_only_event_records"][0]
    assert len(records) == 1
    assert records[0]["candidate_index"] == pair_index
    assert records[0]["candidate_identity"] == "0-1"
    assert records[0]["participant_slots"] == (0, 1)
    assert np.isclose(records[0]["identity_event_weight"], 1.0)


def test_single_event_single_identity_and_multi_identity_conserve_quality():
    single, _ = _final_rows(multi_identity=False)
    assert len(single) == 1
    assert np.isclose(single[0]["final_target_mass"], 0.15)
    assert single[0]["quality_contract_valid"] == 1

    multi, _ = _final_rows(multi_identity=True)
    assert len(multi) == 2
    assert np.isclose(
        sum(row["final_target_mass"] for row in multi),
        0.15,
    )
    assert all(
        np.isclose(row["identity_event_weight"], 0.5)
        for row in multi
    )


def test_positive_negative_static_dynamic_and_order_contracts():
    positive, _ = _final_rows(generation=10, sign=1)
    negative, _ = _final_rows(generation=11, sign=-1)
    assert positive[0]["target_sign"] == 1
    assert positive[0]["outcome_success"] == 1
    assert negative[0]["target_sign"] == -1
    assert negative[0]["outcome_success"] == 0
    assert positive[0]["static_dynamic_class"] == "dynamic"
    assert positive[0]["candidate_order"] == 2
    drafts, delta, behavior = _draft_case(
        generation=12,
        first_identity="0-1-2",
        first_order=3,
        first_candidate_index=6,
        static_dynamic_class="static",
    )
    joined = _join_candidate_evidence_consumption_rows(
        _consumption_rows(delta, behavior),
        drafts,
    )
    triplet = _build_candidate_evidence_provenance_csv_rows(
        joined,
        "run_provenance_synthetic",
        8012,
        "policy_0",
        6,
    )[0]
    assert triplet["candidate_order"] == 3
    assert triplet["static_dynamic_class"] == "static"


def test_missing_and_corrupt_grouped_event_provenance_fail_loud():
    drafts, delta, behavior = _draft_case()
    for missing in (
            "environment_episode_id",
            "capture_event_id",
            "prey_id"):
        bad = dict(drafts[0])
        del bad[missing]
        try:
            _join_candidate_evidence_consumption_rows(
                _consumption_rows(delta, behavior),
                [bad],
            )
        except RuntimeError as error:
            assert "schema" in str(error)
        else:
            raise AssertionError(
                "missing {} provenance did not fail".format(missing)
            )

    duplicate = dict(drafts[0])
    duplicate["capture_event_id"] += 1
    try:
        _join_candidate_evidence_consumption_rows(
            _consumption_rows(delta, behavior),
            [drafts[0], duplicate],
        )
    except RuntimeError as error:
        assert "mass diverges" in str(error)
    else:
        raise AssertionError("corrupt grouped event provenance did not fail")


def test_same_target_multiple_real_events_join_as_one_group():
    coefficient = 0.15
    event_records = [
        {
            "event_id": 44,
            "target_id": 81,
            "participant_slots": (0, 1),
            "candidate_index": 2,
            "candidate_identity": "0-3",
            "candidate_order": 2,
            "identity_event_weight": 1.0,
            "capture_step": 27,
            "static_dynamic_class": "dynamic",
        },
        {
            "event_id": 45,
            "target_id": 82,
            "participant_slots": (0, 1, 3),
            "candidate_index": 2,
            "candidate_identity": "0-3",
            "candidate_order": 2,
            "identity_event_weight": 1.0,
            "capture_step": 27,
            "static_dynamic_class": "dynamic",
        },
    ]
    delta = np.zeros((30, 4, 10), dtype=np.float32)
    delta[27, 3, 2] = coefficient
    behavior = np.zeros(delta.shape + (4,), dtype=np.float32)
    behavior[..., 3] = 4.0
    drafts = build_candidate_evidence_provenance_rows(
        event_records_by_episode=[[], [], [], event_records],
        replay_slot_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        replay_generations=np.asarray([24, 25, 26, 27], dtype=np.int64),
        environment_episode_ids=np.asarray(
            [1024, 1025, 1026, 1027],
            dtype=np.int64,
        ),
        base_selected=np.asarray([0, 0, 0, 1], dtype=np.int64),
        support_selected=np.asarray([0, 0, 0, 0], dtype=np.int64),
        outcome_success=np.asarray([0, 0, 0, 1], dtype=np.int64),
        episode_outcome_advantage=np.asarray(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        capture_event_mass_per_episode=np.asarray(
            [0.0, 0.0, 0.0, 2.0],
            dtype=np.float32,
        ),
        candidate_coefficient=coefficient,
        candidate_identity_delta=delta,
        candidate_behavior=behavior,
        terminal_steps=np.asarray([29, 29, 29, 29], dtype=np.int64),
    )
    joined = _join_candidate_evidence_consumption_rows(
        _consumption_rows(delta, behavior),
        drafts,
    )
    assert len(joined) == 1
    assert len(joined[0]["drafts"]) == 2
    event_map = _candidate_event_by_target_key(joined)
    assert len(event_map[(3, 27, 2, 1)]) == 2
    rows = _build_candidate_evidence_provenance_csv_rows(
        joined_evidence_rows=joined,
        run_id="run_multi_event",
        env_step=8800,
        policy_id="policy_0",
        current_candidate_policy_version=6,
    )
    assert [row["capture_event_id"] for row in rows] == [44, 45]
    assert np.isclose(
        sum(float(row["final_target_mass"]) for row in rows),
        coefficient,
    )
    transaction = _transaction_row(
        rows[0],
        epoch=0,
        sequence=0,
        signed_change=0.06,
    )
    transaction.update({
        "capture_event_id": -1,
        "capture_prey_id": -1,
        "candidate_participant_slots": "",
        "candidate_static_dynamic_class": "grouped",
        "target_weight": coefficient,
        "candidate_event_group_size": 2,
        "capture_event_ids": "44|45",
        "capture_prey_ids": "81|82",
        "candidate_event_target_masses": "0.075|0.075",
        "candidate_participant_slot_groups": "0-1|0-1-3",
        "candidate_static_dynamic_classes": "dynamic|dynamic",
    })
    normalized = _join_candidate_evidence_provenance(
        provenance_rows=rows,
        transaction_rows=[transaction],
    )
    assert len(normalized) == 2
    assert np.isclose(
        sum(row["signed_margin_change"] for row in normalized),
        0.06,
    )
    assert np.isclose(
        sum(row["target_weight"] for row in normalized),
        coefficient,
    )


def test_generation_and_event_are_persisted_once_across_ppo_epochs():
    rows, _ = _final_rows()
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = object.__new__(RecRunner)
        runner.run_dir = temp_dir
        runner._append_candidate_evidence_provenance_csv(rows)
        replayed = [dict(row) for row in rows]
        replayed[0]["env_step"] += 1
        replayed[0]["first_consumed_update"] += 1
        replayed[0]["base_selected"] = 0
        replayed[0]["support_selected"] = 1
        runner._append_candidate_evidence_provenance_csv(replayed)
        files = [
            name for name in os.listdir(temp_dir)
            if name.endswith(
                "progress_train_candidate_evidence_provenance.csv"
            )
        ]
        assert len(files) == 1
        with open(
                os.path.join(temp_dir, files[0]),
                "r",
                newline="") as handle:
            persisted = list(csv.DictReader(handle))
        assert len(persisted) == 1
        assert int(persisted[0]["consumed_generation_once"]) == 1
        assert int(persisted[0]["duplicate_generation"]) == 0
        assert int(persisted[0]["duplicate_event"]) == 0

        sign_conflict = [dict(row) for row in rows]
        sign_conflict[0]["target_sign"] = -1
        try:
            runner._append_candidate_evidence_provenance_csv(sign_conflict)
        except RuntimeError as error:
            assert "sign changed" in str(error)
        else:
            raise AssertionError("event sign conflict did not fail")


def test_provenance_transaction_join_counts_generations_not_epochs():
    provenance = []
    transactions = []
    for offset, generation in enumerate((20, 21)):
        rows, _ = _final_rows(generation=generation)
        row = rows[0]
        provenance.append(row)
        transactions.append(_transaction_row(
            row,
            epoch=0,
            sequence=offset * 2,
            signed_change=0.01,
        ))
        transactions.append(_transaction_row(
            row,
            epoch=1,
            sequence=offset * 2 + 1,
            signed_change=0.02,
        ))
    quality = _validate_candidate_evidence_quality_contract(provenance)
    assert quality["unique_event_count"] == 2
    normalized = _join_candidate_evidence_provenance(
        provenance_rows=provenance,
        transaction_rows=transactions,
    )
    assert len(normalized) == 2
    assert all(
        row["joined_ppo_transaction_count"] == 2
        for row in normalized
    )
    collapsed = _collapse_candidate_evidence_by_generation(normalized)
    summary = _independent_generation_counterfactual(collapsed)
    assert summary["independent_generation_count"] == 2
    assert summary["unique_event_count"] == 2
    assert np.isclose(summary["cumulative_signed_margin_change"], 0.06)
    assert np.isclose(summary["cumulative_target_weight"], 0.30)
    assert summary["quality_contract_valid"]


def test_replay_exposure_mass_drift_does_not_corrupt_first_consumption_join():
    rows, _ = _final_rows(generation=42)
    provenance = rows[0]
    first_epoch_zero = _transaction_row(
        provenance,
        epoch=0,
        sequence=0,
        signed_change=0.01,
    )
    first_epoch_one = _transaction_row(
        provenance,
        epoch=1,
        sequence=1,
        signed_change=0.02,
    )
    replay_epoch_zero = _transaction_row(
        provenance,
        epoch=0,
        sequence=2,
        signed_change=9.0,
    )
    replay_epoch_zero["env_step"] = (
        int(provenance["first_consumed_update"]) + 800
    )
    replay_epoch_zero["target_weight"] *= 2.0
    replay_epoch_one = dict(replay_epoch_zero)
    replay_epoch_one["transaction_sequence_index"] = 3
    replay_epoch_one["ppo_epoch_index"] = 1

    normalized = _join_candidate_evidence_provenance(
        provenance_rows=[provenance],
        transaction_rows=[
            first_epoch_zero,
            first_epoch_one,
            replay_epoch_zero,
            replay_epoch_one,
        ],
    )
    assert len(normalized) == 1
    assert normalized[0]["joined_ppo_transaction_count"] == 2
    assert normalized[0]["later_replay_transaction_row_count"] == 2
    assert normalized[0]["first_consumption_only"]
    assert np.isclose(normalized[0]["signed_margin_change"], 0.03)
    assert np.isclose(
        normalized[0]["target_weight"],
        provenance["final_target_mass"],
    )


def test_offline_cli_reads_provenance_and_transaction_csv_directly():
    provenance = []
    transactions = []
    for offset, generation in enumerate((50, 51)):
        rows, _ = _final_rows(generation=generation)
        row = rows[0]
        provenance.append(row)
        transactions.extend([
            _transaction_row(row, 0, offset * 2, 0.01),
            _transaction_row(row, 1, offset * 2 + 1, 0.02),
        ])
    with tempfile.TemporaryDirectory() as temp_dir:
        provenance_path = os.path.join(temp_dir, "provenance.csv")
        transaction_path = os.path.join(temp_dir, "transaction.csv")
        with open(provenance_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(provenance[0].keys()),
            )
            writer.writeheader()
            writer.writerows(provenance)
        with open(transaction_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(transactions[0].keys()),
            )
            writer.writeheader()
            writer.writerows(transactions)
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    REPO_ROOT,
                    "scripts",
                    "debug_candidate_score_to_rank_counterfactual.py",
                ),
                "--enable",
                "--independent-generation-accumulation",
                "--successful-overlap-only",
                "--identity",
                "0-1",
                "--csv",
                transaction_path,
                "--provenance-csv",
                provenance_path,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["accepted"]
        assert payload["independent_generation_count"] == 2
        assert payload["unique_event_count"] == 2
        assert payload["joined_transaction_row_count"] == 4


def test_duplicate_generation_event_and_join_conflicts_fail():
    row, _ = _final_rows(generation=30)
    row = row[0]
    transaction = _transaction_row(row, 0, 0, 0.01)
    try:
        _validate_candidate_evidence_quality_contract([row, dict(row)])
    except RuntimeError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate event row did not fail")

    bad_participants = dict(transaction)
    bad_participants["candidate_participant_slots"] = "0-3"
    try:
        _join_candidate_evidence_provenance(
            provenance_rows=[row],
            transaction_rows=[bad_participants],
        )
    except RuntimeError as error:
        assert "participants" in str(error)
    else:
        raise AssertionError("participant join conflict did not fail")

    bad_prey = dict(transaction)
    bad_prey["capture_prey_id"] += 1
    try:
        _join_candidate_evidence_provenance(
            provenance_rows=[row],
            transaction_rows=[bad_prey],
        )
    except RuntimeError as error:
        assert "no optimizer transaction" in str(error)
    else:
        raise AssertionError("prey join conflict did not fail")


def test_quality_corruption_and_distinct_events_are_explicit():
    first, _ = _final_rows(generation=40)
    second, _ = _final_rows(generation=41)
    # Different generation/environment/event/prey keys are independent.
    quality = _validate_candidate_evidence_quality_contract(first + second)
    assert quality["unique_event_count"] == 2
    corrupted = [dict(first[0])]
    corrupted[0]["final_target_mass"] = 0.14
    try:
        _validate_candidate_evidence_quality_contract(corrupted)
    except RuntimeError as error:
        assert "does not reconstruct" in str(error)
    else:
        raise AssertionError("corrupted event quality did not fail")


def _trajectory_case(device, enabled):
    device = torch.device(device)
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(456)
    parameter = torch.nn.Parameter(
        torch.tensor([0.2, -0.4], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    loss = (parameter.pow(2) * torch.tensor(
        [1.0, 3.0],
        dtype=torch.float32,
        device=device,
    )).sum()
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    if enabled:
        drafts, delta, behavior = _draft_case()
        drafts_before = copy.deepcopy(drafts)
        delta_before = delta.copy()
        behavior_before = behavior.copy()
        consumption = _candidate_evidence_consumption_trace_rows(
            candidate_identity_delta=torch.as_tensor(
                delta.reshape(-1, delta.shape[-1]),
                device=device,
            ),
            behavior_candidate_version=torch.as_tensor(
                behavior[..., 3].reshape(-1, delta.shape[-1]),
                device=device,
            ),
            replay_population_provenance=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0],
                ],
                dtype=torch.float32,
                device=device,
            ),
        )
        joined = _join_candidate_evidence_consumption_rows(
            consumption,
            drafts,
        )
        assert len(_candidate_event_by_target_key(joined)) == 1
        assert drafts == drafts_before
        assert np.array_equal(delta, delta_before)
        assert np.array_equal(behavior, behavior_before)
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    if device.type == "cuda":
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
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
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda"
            else None
        ),
    }


def _assert_trajectory_equal(left, right):
    for name in (
            "loss",
            "gradient",
            "parameter",
            "exp_avg",
            "exp_avg_sq"):
        assert torch.equal(left[name], right[name]), name
    assert left["step"] == right["step"]
    assert torch.equal(left["cpu_rng"], right["cpu_rng"])
    if left["cuda_rng"] is not None:
        assert torch.equal(left["cuda_rng"], right["cuda_rng"])


def test_cpu_and_cuda_trajectory_neutrality():
    _assert_trajectory_equal(
        _trajectory_case("cpu", False),
        _trajectory_case("cpu", True),
    )
    if torch.cuda.is_available():
        _assert_trajectory_equal(
            _trajectory_case("cuda", False),
            _trajectory_case("cuda", True),
        )


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} candidate evidence provenance tests".format(
        len(tests)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
