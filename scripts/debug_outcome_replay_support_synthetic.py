#!/usr/bin/env python
"""Synthetic checks for signed outcome replay-support completion."""

from __future__ import print_function

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.graph_sampling import select_outcome_contrast_complete_episodes
from utils.adj_buffer import (
    build_pair_evidence_episode_diagnostic_rows,
    summarize_pair_evidence_episode_funnel,
    summarize_successful_candidate_capture_context,
)


def select(
        base,
        positive,
        negative,
        recency,
        positive_eligible=None,
        negative_eligible=None):
    return select_outcome_contrast_complete_episodes(
        np.asarray(base, dtype=np.int64),
        np.asarray(positive, dtype=bool),
        np.asarray(negative, dtype=bool),
        np.asarray(recency, dtype=np.int64),
        positive_eligible_mask=(
            None if positive_eligible is None
            else np.asarray(positive_eligible, dtype=bool)
        ),
        negative_eligible_mask=(
            None if negative_eligible is None
            else np.asarray(negative_eligible, dtype=bool)
        ),
    )


def main():
    # run103 reached a real successful capture episode but had only failed
    # pair-evidence episodes.  The funnel must distinguish "no success" from
    # "success/capture exists but strict pair evidence is absent" without
    # changing any input mask.
    occupied = np.asarray([True, True, True, False])
    successful = np.asarray([True, False, False, False])
    active_capture = np.asarray([False, False, False, False])
    candidate_capture = np.asarray([True, False, False, False])
    pair_evidence = np.asarray([False, True, False, True])
    input_snapshots = [
        values.copy() for values in (
            occupied,
            successful,
            active_capture,
            candidate_capture,
            pair_evidence,
        )
    ]
    funnel = summarize_pair_evidence_episode_funnel(
        occupied,
        successful,
        active_capture,
        candidate_capture,
        pair_evidence,
    )
    assert funnel["occupied_episode_count"] == 3
    assert funnel["successful_episode_count"] == 1
    assert funnel["successful_capture_episode_count"] == 1
    assert funnel["successful_candidate_capture_episode_count"] == 1
    assert funnel["pair_evidence_episode_count"] == 1
    assert funnel["pair_positive_episode_count"] == 0
    assert funnel["pair_negative_episode_count"] == 1
    assert (
        funnel[
            "successful_capture_without_pair_evidence_episode_count"
        ] == 1
    )
    assert funnel["pair_evidence_without_capture_episode_count"] == 1
    assert funnel["version"] == 2.0
    assert (
        funnel[
            "successful_capture_gap_candidate_only_not_active_episode_count"
        ] == 1
    )
    assert (
        funnel[
            "successful_capture_gap_active_without_strict_pair_episode_count"
        ] == 0
    )
    assert funnel[
        "successful_capture_gap_reject_reason_contract_valid"
    ] == 1.0
    assert funnel["contract_valid"] == 1.0
    for current, snapshot in zip(
            (
                occupied,
                successful,
                active_capture,
                candidate_capture,
                pair_evidence,
            ),
            input_snapshots):
        assert np.array_equal(current, snapshot)
    print("PASS run103 missing-positive pair evidence funnel is explicit")

    # run104-equivalent reject reason: the only successful capture identity is
    # a valid candidate far below the active boundary.  The diagnostic must
    # preserve its exact behavior margin/rank and terminal timing without
    # manufacturing strict pair evidence.
    candidate_weights = np.zeros((4, 4, 3), dtype=np.float32)
    candidate_weights[2, 0, 1] = 1.0
    candidate_behavior = np.zeros((4, 4, 3, 4), dtype=np.float32)
    candidate_behavior[..., 2] = 1.0
    candidate_behavior[..., 1] = 1.0
    candidate_behavior[2, 0, 1] = [-1.1, 12.0, 1.0, 7.0]
    capture_transition = np.zeros((4, 4), dtype=bool)
    capture_transition[2, 0] = True
    terminal_transition = np.zeros((4, 4), dtype=bool)
    terminal_transition[3, 0] = True
    context = summarize_successful_candidate_capture_context(
        successful_episode_mask=successful,
        pair_evidence_episode_mask=pair_evidence,
        active_capture_episode_mask=active_capture,
        candidate_capture_weights=candidate_weights,
        candidate_behavior=candidate_behavior,
        capture_transition_mask=capture_transition,
        terminal_transition_mask=terminal_transition,
    )
    assert context["successful_candidate_gap_episode_count"] == 1
    assert context["successful_candidate_gap_identity_count"] == 1
    assert np.isclose(
        context["successful_candidate_gap_behavior_margin_mean"], -1.1
    )
    assert context["successful_candidate_gap_behavior_rank_mean"] == 12.0
    assert (
        context[
            "successful_candidate_gap_behavior_boundary_crossed_fraction"
        ] == 0.0
    )
    assert (
        context[
            "successful_capture_gap_terminal_capture_episode_count"
        ] == 0
    )
    assert (
        context[
            "successful_capture_gap_last_capture_to_terminal_step_mean"
        ] == 1.0
    )
    assert context["successful_candidate_gap_context_contract_valid"] == 1.0
    assert not pair_evidence[0]
    print("PASS run104 candidate-only reject reason and timing are explicit")

    terminal_transition[2, 0] = True
    terminal_transition[3, 0] = False
    context = summarize_successful_candidate_capture_context(
        successful,
        pair_evidence,
        active_capture,
        candidate_weights,
        candidate_behavior,
        capture_transition,
        terminal_transition,
    )
    assert (
        context[
            "successful_capture_gap_terminal_capture_episode_count"
        ] == 1
    )
    assert (
        context[
            "successful_capture_gap_last_capture_to_terminal_step_mean"
        ] == 0.0
    )
    print("PASS terminal successful candidate capture timing is exact")

    active_gap = active_capture.copy()
    active_gap[0] = True
    active_funnel = summarize_pair_evidence_episode_funnel(
        occupied,
        successful,
        active_gap,
        candidate_capture,
        pair_evidence,
    )
    assert (
        active_funnel[
            "successful_capture_gap_candidate_only_not_active_episode_count"
        ] == 0
    )
    assert (
        active_funnel[
            "successful_capture_gap_active_without_strict_pair_episode_count"
        ] == 1
    )
    print("PASS active-capture strict-pair reject reason is distinct")

    # Emit the same run104-equivalent failure at episode resolution. A
    # candidate-only success is a source-backed reject reason, not a
    # terminal/future-window guess, and replay selection must remain exact.
    diagnostic_candidate_weights = np.zeros(
        (4, 4, 4),
        dtype=np.float32,
    )
    diagnostic_candidate_weights[2, 0, 1] = 1.0
    diagnostic_behavior = np.zeros((4, 4, 4, 4), dtype=np.float32)
    diagnostic_behavior[..., 1] = 1.0
    diagnostic_behavior[..., 2] = 1.0
    diagnostic_behavior[2, 0, 1] = [-1.1, 12.0, 1.0, 7.0]
    pair_transition = np.zeros((4, 4), dtype=bool)
    pair_transition[1, 1] = True
    diagnostic_capture = np.zeros((4, 4), dtype=bool)
    diagnostic_capture[2, 0] = True
    diagnostic_capture[1, 1] = True
    diagnostic_terminal = np.zeros((4, 4), dtype=bool)
    diagnostic_terminal[3, 0] = True
    diagnostic_terminal[3, 1] = True
    diagnostic_inputs = (
        diagnostic_candidate_weights,
        diagnostic_behavior,
        pair_transition,
        diagnostic_capture,
        diagnostic_terminal,
    )
    diagnostic_snapshots = [
        values.copy() for values in diagnostic_inputs
    ]
    rows = build_pair_evidence_episode_diagnostic_rows(
        episode_generation=np.asarray([370, 369, 368, -1]),
        selected_episode_indices=np.asarray([0, 1]),
        base_episode_indices=np.asarray([1]),
        supplemented_episode_indices=np.asarray([0]),
        outcome_support_used=np.asarray([True, False, False, False]),
        successful_episode_mask=successful,
        active_capture_episode_mask=active_capture,
        candidate_capture_weights=diagnostic_candidate_weights,
        candidate_behavior=diagnostic_behavior,
        pair_evidence_transition_mask=pair_transition,
        capture_transition_mask=diagnostic_capture,
        terminal_transition_mask=diagnostic_terminal,
        capture_identity_candidate_count=np.zeros(
            (4, 4),
            dtype=np.float32,
        ),
        recency_age=np.asarray([0, 1, 2, -1]),
        num_agents=3,
        configured_pair_window=20,
        outcome_class_complete=True,
        pair_class_complete=False,
    )
    assert len(rows) == 3
    row_by_generation = {
        row["episode_generation"]: row for row in rows
    }
    success_row = row_by_generation[370]
    assert success_row["reject_reason"] == "CANDIDATE_ONLY_NOT_ACTIVE"
    assert success_row["selected_for_training"] == 1
    assert success_row["support_selected"] == 1
    assert success_row["selected_episode_ordinal"] == 0
    assert success_row["candidate_identity_indices"] == "1"
    assert success_row["candidate_factor_identities"] == "0-2"
    assert success_row["candidate_factor_order_min"] == 2
    assert success_row["candidate_factor_order_max"] == 2
    assert success_row["participant_slots"] == "0-2"
    assert np.isclose(
        success_row["candidate_behavior_margin_mean"],
        -1.1,
    )
    assert success_row["candidate_behavior_rank_mean"] == 12.0
    assert success_row["last_capture_to_terminal_step"] == 1
    assert success_row["terminal_capture"] == 0
    assert success_row["pair_evidence_episode"] == 0
    assert row_by_generation[369]["pair_evidence_sign"] == -1
    assert (
        row_by_generation[369]["reject_reason"]
        == "NOT_A_SUCCESSFUL_CAPTURE_GAP"
    )
    assert rows[0]["environment_episode_id_available"] == 0
    assert rows[0]["capture_event_id_available"] == 0
    assert rows[0]["capture_prey_id_available"] == 0
    assert rows[0]["static_dynamic_identity_available"] == 0
    assert rows[0]["future_evidence_transition_step_available"] == 0
    for values, snapshot in zip(
            diagnostic_inputs,
            diagnostic_snapshots):
        assert np.array_equal(values, snapshot)
    print("PASS episode-level reject reason and replay provenance are exact")

    active_rows = build_pair_evidence_episode_diagnostic_rows(
        episode_generation=np.asarray([370, 369, 368, -1]),
        selected_episode_indices=np.asarray([0, 1]),
        base_episode_indices=np.asarray([1]),
        supplemented_episode_indices=np.asarray([0]),
        outcome_support_used=np.asarray([True, False, False, False]),
        successful_episode_mask=successful,
        active_capture_episode_mask=active_gap,
        candidate_capture_weights=diagnostic_candidate_weights,
        candidate_behavior=diagnostic_behavior,
        pair_evidence_transition_mask=pair_transition,
        capture_transition_mask=diagnostic_capture,
        terminal_transition_mask=diagnostic_terminal,
        capture_identity_candidate_count=np.zeros(
            (4, 4),
            dtype=np.float32,
        ),
        recency_age=np.asarray([0, 1, 2, -1]),
        num_agents=3,
        configured_pair_window=20,
        outcome_class_complete=True,
        pair_class_complete=False,
        active_capture_event_provenance=[
            ({
                "environment_episode_id": 901,
                "event_id": 17,
                "target_id": 4,
                "participant_slots": (0, 1),
                "factor_index": 2,
                "factor_identity": "0-1",
                "factor_order": 2,
                "identity_event_weight": 1.0,
                "factor_slot_weight": 1.0,
                "capture_step": 2,
                "static_dynamic_class": "dynamic",
            },),
            tuple(),
            tuple(),
            tuple(),
        ],
    )
    assert (
        active_rows[0]["reject_reason"]
        == "ACTIVE_CAPTURE_NO_STRICT_PRIOR_PAIR"
    )
    assert active_rows[0]["environment_episode_id_available"] == 1
    assert active_rows[0]["environment_episode_id"] == 901
    assert active_rows[0]["capture_event_id_available"] == 1
    assert active_rows[0]["capture_event_id"] == 17
    assert active_rows[0]["capture_prey_id_available"] == 1
    assert active_rows[0]["capture_prey_id"] == 4
    assert active_rows[0]["matched_active_factor_identity_available"] == 1
    assert active_rows[0]["matched_active_factor_identity"] == "0-1"
    assert active_rows[0]["static_dynamic_identity_available"] == 1
    assert active_rows[0]["static_dynamic_identity"] == "0-1:dynamic"
    print("PASS episode-level active reject reason is mutually exclusive")

    observer_off_rows = build_pair_evidence_episode_diagnostic_rows(
        episode_generation=np.asarray([370, 369, 368, -1]),
        selected_episode_indices=np.asarray([0, 1]),
        base_episode_indices=np.asarray([1]),
        supplemented_episode_indices=np.asarray([0]),
        outcome_support_used=np.asarray([True, False, False, False]),
        successful_episode_mask=successful,
        active_capture_episode_mask=active_gap,
        candidate_capture_weights=diagnostic_candidate_weights,
        candidate_behavior=diagnostic_behavior,
        pair_evidence_transition_mask=pair_transition,
        capture_transition_mask=diagnostic_capture,
        terminal_transition_mask=diagnostic_terminal,
        capture_identity_candidate_count=np.zeros(
            (4, 4),
            dtype=np.float32,
        ),
        recency_age=np.asarray([0, 1, 2, -1]),
        num_agents=3,
        configured_pair_window=20,
        outcome_class_complete=True,
        pair_class_complete=False,
    )
    assert (
        observer_off_rows[0]["matched_active_factor_identity_available"]
        == 0
    )
    assert observer_off_rows[0]["matched_active_factor_identity"] == ""
    print("PASS observer-off active capture path retains explicit NA semantics")

    multi_capture = diagnostic_capture.copy()
    multi_capture[1, 0] = True
    multi_event_rows = build_pair_evidence_episode_diagnostic_rows(
        episode_generation=np.asarray([370, 369, 368, -1]),
        selected_episode_indices=np.asarray([0, 1]),
        base_episode_indices=np.asarray([1]),
        supplemented_episode_indices=np.asarray([0]),
        outcome_support_used=np.asarray([True, False, False, False]),
        successful_episode_mask=successful,
        active_capture_episode_mask=active_gap,
        candidate_capture_weights=diagnostic_candidate_weights,
        candidate_behavior=diagnostic_behavior,
        pair_evidence_transition_mask=pair_transition,
        capture_transition_mask=multi_capture,
        terminal_transition_mask=diagnostic_terminal,
        capture_identity_candidate_count=np.zeros(
            (4, 4),
            dtype=np.float32,
        ),
        recency_age=np.asarray([0, 1, 2, -1]),
        num_agents=3,
        configured_pair_window=20,
        outcome_class_complete=True,
        pair_class_complete=False,
        active_capture_event_provenance=[
            (
                {
                    "environment_episode_id": 901,
                    "event_id": 16,
                    "target_id": 3,
                    "participant_slots": (1, 2),
                    "factor_index": 1,
                    "factor_identity": "1-2",
                    "factor_order": 2,
                    "identity_event_weight": 1.0,
                    "factor_slot_weight": 1.0,
                    "capture_step": 1,
                    "static_dynamic_class": "static",
                },
                {
                    "environment_episode_id": 901,
                    "event_id": 17,
                    "target_id": 4,
                    "participant_slots": (0, 1),
                    "factor_index": 2,
                    "factor_identity": "0-1",
                    "factor_order": 2,
                    "identity_event_weight": 1.0,
                    "factor_slot_weight": 1.0,
                    "capture_step": 2,
                    "static_dynamic_class": "dynamic",
                },
            ),
            tuple(),
            tuple(),
            tuple(),
        ],
        require_active_capture_event_provenance=True,
    )
    multi_row = multi_event_rows[0]
    assert multi_row["capture_event_id_available"] == 0
    assert multi_row["capture_event_id"] == -1
    assert multi_row["capture_prey_id_available"] == 0
    assert multi_row["matched_active_factor_identity"] == "1-2;0-1"
    assert (
        multi_row["static_dynamic_identity"]
        == "1-2:static;0-1:dynamic"
    )
    print("PASS multiple active captures retain exact identity membership")

    try:
        build_pair_evidence_episode_diagnostic_rows(
            episode_generation=np.asarray([370, 369, 368, -1]),
            selected_episode_indices=np.asarray([0, 1]),
            base_episode_indices=np.asarray([1]),
            supplemented_episode_indices=np.asarray([0]),
            outcome_support_used=np.asarray([True, False, False, False]),
            successful_episode_mask=successful,
            active_capture_episode_mask=active_gap,
            candidate_capture_weights=diagnostic_candidate_weights,
            candidate_behavior=diagnostic_behavior,
            pair_evidence_transition_mask=pair_transition,
            capture_transition_mask=diagnostic_capture,
            terminal_transition_mask=diagnostic_terminal,
            capture_identity_candidate_count=np.zeros(
                (4, 4),
                dtype=np.float32,
            ),
            recency_age=np.asarray([0, 1, 2, -1]),
            num_agents=3,
            configured_pair_window=20,
            outcome_class_complete=True,
            pair_class_complete=False,
            active_capture_event_provenance=[
                tuple(), tuple(), tuple(), tuple()
            ],
            require_active_capture_event_provenance=True,
        )
        raise AssertionError("missing active event provenance did not fail")
    except RuntimeError as exc:
        assert "missing exact event provenance" in str(exc)
    print("PASS required active capture identity fails loudly when absent")

    # Once the same successful capture carries strict pair evidence, the
    # signed classes and total pair-evidence population must reconstruct.
    pair_evidence_with_positive = pair_evidence.copy()
    pair_evidence_with_positive[0] = True
    funnel = summarize_pair_evidence_episode_funnel(
        occupied,
        successful,
        active_capture,
        candidate_capture,
        pair_evidence_with_positive,
    )
    assert funnel["pair_evidence_episode_count"] == 2
    assert funnel["pair_positive_episode_count"] == 1
    assert funnel["pair_negative_episode_count"] == 1
    assert (
        funnel[
            "successful_capture_without_pair_evidence_episode_count"
        ] == 0
    )
    print("PASS complete signed pair evidence funnel reconstructs")

    try:
        summarize_pair_evidence_episode_funnel(
            occupied,
            successful[:-1],
            active_capture,
            candidate_capture,
            pair_evidence,
        )
    except ValueError as exc:
        assert "different shapes" in str(exc)
    else:
        raise AssertionError("misaligned pair evidence provenance must fail")
    print("PASS pair evidence funnel provenance mismatch fails loudly")

    # A one-episode recent window containing a failed capture must receive the
    # newest real successful-capture episode when both classes exist.
    result = select(
        [3],
        [False, True, False, False],
        [False, False, False, True],
        [3, 2, 1, 0],
    )
    assert np.array_equal(result["episode_indices"], [3, 1])
    assert result["augmented_count"] == 1
    assert result["positive_selected_count"] == 1
    assert result["negative_selected_count"] == 1
    assert result["class_complete"] == 1.0
    print("PASS negative-only recent cohort receives real positive support")

    # Symmetric positive-only path.
    result = select(
        [4],
        [False, False, False, False, True],
        [False, True, False, False, False],
        [4, 3, 2, 1, 0],
    )
    assert np.array_equal(result["episode_indices"], [4, 1])
    assert result["class_complete"] == 1.0
    print("PASS positive-only recent cohort receives real negative support")

    # Already complete cohorts remain unchanged and are never duplicated.
    result = select(
        [1, 3],
        [False, True, False, False],
        [False, False, False, True],
        [3, 2, 1, 0],
    )
    assert np.array_equal(result["episode_indices"], [1, 3])
    assert result["augmented_count"] == 0
    print("PASS complete cohort is unchanged")

    # Single-outcome protection is preserved: a missing class is never made up.
    result = select(
        [2],
        [False, False, True],
        [False, False, False],
        [2, 1, 0],
    )
    assert np.array_equal(result["episode_indices"], [2])
    assert result["augmented_count"] == 0
    assert result["negative_available"] == 0.0
    assert result["class_complete"] == 0.0
    assert result["outcome_credit_enabled"] == 0.0
    print("PASS single-outcome population remains unmodified")

    # A zero-credit recent episode requires both real classes; newest eligible
    # slots win even when ring-buffer slot order is non-monotonic.
    result = select(
        [0],
        [False, False, True, False, True],
        [False, True, False, True, False],
        [0, 4, 3, 2, 1],
    )
    assert np.array_equal(result["episode_indices"], [0, 4, 3])
    assert result["augmented_count"] == 2
    print("PASS circular recency order selects newest real class support")

    # Supplemental support rotates across eligible resident episodes instead
    # of repeatedly concentrating every update on the newest rare class.
    positive = [True, False, True, False]
    negative = [False, False, False, True]
    first = select(
        [3], positive, negative, [3, 2, 1, 0],
        positive_eligible=[True, False, True, False],
        negative_eligible=negative,
    )
    assert np.array_equal(first["supplemented_episode_indices"], [2])
    second = select(
        [3], positive, negative, [3, 2, 1, 0],
        positive_eligible=[True, False, False, False],
        negative_eligible=negative,
    )
    assert np.array_equal(second["supplemented_episode_indices"], [0])
    assert first["outcome_credit_enabled"] == 1.0
    assert second["outcome_credit_enabled"] == 1.0
    print("PASS supplemental support rotates without cross-update reuse")

    exhausted = select(
        [3], positive, negative, [3, 2, 1, 0],
        positive_eligible=[False, False, False, False],
        negative_eligible=negative,
    )
    assert np.array_equal(exhausted["episode_indices"], [3])
    assert exhausted["class_complete"] == 0.0
    assert exhausted["support_exhausted"] == 1.0
    assert exhausted["outcome_credit_enabled"] == 0.0
    print("PASS exhausted support disables one-sided outcome training")

    atomic_exhaustion = select(
        [1],
        [True, False, False, False],
        [False, False, True, False],
        [3, 2, 1, 0],
        positive_eligible=[False, False, False, False],
        negative_eligible=[False, False, True, False],
    )
    assert np.array_equal(atomic_exhaustion["episode_indices"], [1])
    assert atomic_exhaustion["augmented_count"] == 0
    assert atomic_exhaustion["outcome_credit_enabled"] == 0.0
    print("PASS incomplete support is not partially consumed")

    try:
        select(
            [0],
            [True, False],
            [True, False],
            [1, 0],
        )
    except ValueError as exc:
        assert "both signed outcome classes" in str(exc)
    else:
        raise AssertionError("overlapping signed episode labels must fail")
    print("PASS contradictory episode signs fail explicitly")

    try:
        select(
            [0],
            [False, True, False],
            [False, False, False],
            [0, 2],
        )
    except ValueError as exc:
        assert "unoccupied replay slot" in str(exc)
    else:
        raise AssertionError("credit in an unoccupied replay slot must fail")
    print("PASS unoccupied replay credit fails explicitly")

    print("PASS all outcome replay-support synthetic tests")


if __name__ == "__main__":
    main()
