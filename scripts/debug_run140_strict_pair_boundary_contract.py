#!/usr/bin/env python3
"""Replay the run140 seq830 exact-preservation contract from its real CSV.

The CSV contains production exact-forward extrema and direction metadata, not
the original parameter tensors.  This fixture therefore makes two distinctions
explicit:

* ``replay_run140_trace`` uses only real recorded trial values.
* ``exercise_production_acceptance`` builds the smallest tensor population that
  preserves those recorded extrema and calls the production acceptance helper.

It does not claim to reconstruct an unlogged parameter-space Gram matrix.
"""

from __future__ import print_function

import csv
import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import _joint_exact_constraint_acceptance


RUN140_FAILURE_CSV = os.path.join(
    REPO_ROOT,
    "scripts",
    "results",
    "wolfpack",
    "sddfg",
    "sddfg_intra_ep_4to6_r2_j1_rec30_seed1",
    "run140",
    "run140_progress_train_strict_pair_exact_failure.csv",
)
FLOAT32_PRESERVATION_TOLERANCE = 128.0 * 1.1920928955078125e-7


def _read_rows():
    if not os.path.isfile(RUN140_FAILURE_CSV):
        raise RuntimeError("run140 strict-pair failure CSV is missing")
    with open(RUN140_FAILURE_CSV, "r") as handle:
        return list(csv.DictReader(handle))


def _trace_row_preserves_contract(row):
    return bool(
        float(row["boundary_min_signed_change"])
        > 1.0e-12
        and float(row["exact_score_min_signed_change"])
        >= -FLOAT32_PRESERVATION_TOLERANCE
        and int(row["candidate_valid"]) == 1
        and int(row["lifecycle_valid"]) == 1
        and float(row["progress_worst_actual"]) > 1.0e-12
    )


def replay_run140_trace(rows):
    assert len(rows) == 337
    origin = [row for row in rows if row["evaluation_kind"] == "origin_preservation"]
    search = [row for row in rows if row["evaluation_kind"] == "search"]
    probes = [
        row for row in rows
        if row["evaluation_kind"] == "failure_midpoint_probe"
    ]
    assert len(origin) == 1
    assert len(search) == 168
    assert len(probes) == 168
    assert int(origin[0]["origin_preservation_valid"]) == 1
    assert int(origin[0]["target_count"]) == 52
    assert int(origin[0]["candidate_count"]) == 8
    assert int(origin[0]["diagnostic_probe_valid_count"]) == 0
    assert origin[0]["env_step"] == "174400"
    assert origin[0]["transaction_sequence_index"] == "830"

    identities = origin[0]["target_canonical_identities"].split("|")
    signs = origin[0]["target_signs"].split("|")
    identity_counts = {}
    identity_sign_counts = {}
    for identity, sign in zip(identities, signs):
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
        key = (identity, int(sign))
        identity_sign_counts[key] = identity_sign_counts.get(key, 0) + 1
    assert identity_counts == {"0-1": 20, "3-5": 1, "0-2": 23, "1-2": 8}
    assert identity_sign_counts[("0-2", 1)] == 15
    assert identity_sign_counts[("0-2", -1)] == 8

    by_candidate = {}
    for row in search:
        by_candidate.setdefault(int(row["candidate_ordinal"]), []).append(row)
    assert sorted(by_candidate) == list(range(8))
    expected_largest_safe = {
        0: 0.0009765625,
        1: 0.0009765625,
        2: 0.001953125,
        3: 0.00390625,
        4: 0.0078125,
        5: 0.0009765625,
        6: 0.001953125,
        7: 0.00390625,
    }
    largest_safe = {}
    for candidate_ordinal, candidate_rows in sorted(by_candidate.items()):
        assert len(candidate_rows) == 21
        # The old strict-positive exact-score gate rejected every production
        # trial even though each ray contains a positive-progress preservation
        # point under the documented all-member non-regression contract.
        assert not any(int(row["exact_score_valid"]) == 1 for row in candidate_rows)
        assert any(int(row["boundary_valid"]) == 1 for row in candidate_rows)
        preserving = [row for row in candidate_rows if _trace_row_preserves_contract(row)]
        assert preserving
        largest = max(preserving, key=lambda row: float(row["scale"]))
        largest_safe[candidate_ordinal] = float(largest["scale"])
        assert abs(
            largest_safe[candidate_ordinal]
            - expected_largest_safe[candidate_ordinal]
        ) <= 1.0e-15

    direction_rows = [
        min(candidate_rows, key=lambda row: int(row["evaluation_ordinal"]))
        for _, candidate_rows in sorted(by_candidate.items())
    ]
    cosines = [float(row["cosine_vs_full"]) for row in direction_rows]
    norms = [float(row["direction_norm"]) for row in direction_rows]
    assert min(cosines) > 0.99
    max_angle_degrees = max(
        math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        for cosine in cosines
    )
    return {
        "largest_safe": largest_safe,
        "cosine_min": min(cosines),
        "cosine_median": sorted(cosines)[len(cosines) // 2],
        "norm_min": min(norms),
        "norm_max": max(norms),
        "max_angle_degrees": max_angle_degrees,
    }


def exercise_production_acceptance(rows):
    # Candidate zero at scale 2^-10 is the real run140 production trial with
    # the largest preservation-safe dyadic scale on the full Adam ray.
    row = next(
        row for row in rows
        if row["evaluation_kind"] == "search"
        and row["candidate_ordinal"] == "0"
        and row["scale"] == "0.0009765625"
    )
    target_count = int(row["target_count"])
    actionable = torch.ones((1, target_count), dtype=torch.float32)
    boundary_change = torch.full_like(actionable, 1.0e-4)
    exact_change = torch.full_like(actionable, 1.0e-4)
    boundary_change[0, 0] = float(row["progress_worst_actual"])
    boundary_change[0, 31] = float(row["boundary_min_signed_change"])
    exact_change[0, int(row["limiting_target_ordinal"])] = float(
        row["exact_score_min_signed_change"]
    )
    candidate_loss_change = torch.tensor(
        float(row["candidate_loss_change"]), dtype=torch.float32
    )

    legacy = _joint_exact_constraint_acceptance(
        signed_boundary_change=boundary_change,
        actionable_pair_mask=actionable,
        signed_exact_score_change=exact_change,
        candidate_loss_change=candidate_loss_change,
    )
    assert not legacy["valid"]
    assert legacy["boundary_valid"]
    assert not legacy["exact_score_valid"]

    corrected = _joint_exact_constraint_acceptance(
        signed_boundary_change=boundary_change,
        actionable_pair_mask=actionable,
        signed_exact_score_change=exact_change,
        candidate_loss_change=candidate_loss_change,
        preservation_tolerance=FLOAT32_PRESERVATION_TOLERANCE,
    )
    assert corrected["valid"]
    assert corrected["boundary_valid"]
    assert corrected["exact_score_valid"]
    assert corrected["candidate_valid"]

    materially_regressed = boundary_change.clone()
    materially_regressed[0, 1] = -2.0 * FLOAT32_PRESERVATION_TOLERANCE
    rejected_regression = _joint_exact_constraint_acceptance(
        signed_boundary_change=materially_regressed,
        actionable_pair_mask=actionable,
        signed_exact_score_change=exact_change,
        candidate_loss_change=candidate_loss_change,
        preservation_tolerance=FLOAT32_PRESERVATION_TOLERANCE,
    )
    assert not rejected_regression["valid"]
    assert not rejected_regression["boundary_valid"]

    materially_regressed_exact = exact_change.clone()
    materially_regressed_exact[0, 2] = (
        -2.0 * FLOAT32_PRESERVATION_TOLERANCE
    )
    rejected_exact_regression = _joint_exact_constraint_acceptance(
        signed_boundary_change=boundary_change,
        actionable_pair_mask=actionable,
        signed_exact_score_change=materially_regressed_exact,
        candidate_loss_change=candidate_loss_change,
        preservation_tolerance=FLOAT32_PRESERVATION_TOLERANCE,
    )
    assert not rejected_exact_regression["valid"]
    assert not rejected_exact_regression["exact_score_valid"]

    zero_progress = boundary_change.clone()
    zero_progress[0, 0] = 0.0
    rejected_zero_progress = _joint_exact_constraint_acceptance(
        signed_boundary_change=zero_progress,
        actionable_pair_mask=actionable,
        signed_exact_score_change=exact_change,
        candidate_loss_change=candidate_loss_change,
        preservation_tolerance=FLOAT32_PRESERVATION_TOLERANCE,
    )
    assert not rejected_zero_progress["valid"]
    assert not rejected_zero_progress["boundary_valid"]

    rejected_candidate_regression = _joint_exact_constraint_acceptance(
        signed_boundary_change=boundary_change,
        actionable_pair_mask=actionable,
        signed_exact_score_change=exact_change,
        candidate_loss_change=torch.tensor(1.0e-7, dtype=torch.float32),
        preservation_tolerance=FLOAT32_PRESERVATION_TOLERANCE,
    )
    assert not rejected_candidate_regression["valid"]
    assert not rejected_candidate_regression["candidate_valid"]
    return {
        "scale": float(row["scale"]),
        "boundary_min": float(row["boundary_min_signed_change"]),
        "exact_min": float(row["exact_score_min_signed_change"]),
        "progress": float(row["progress_worst_actual"]),
        "completion": float(row["progress_min_completion"]),
    }


def main():
    rows = _read_rows()
    trace = replay_run140_trace(rows)
    acceptance = exercise_production_acceptance(rows)
    print("run140 strict-pair boundary contract fixture PASS")
    print("trace={}".format(trace))
    print("production_acceptance={}".format(acceptance))


if __name__ == "__main__":
    main()
