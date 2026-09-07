#!/usr/bin/env python3
"""Targeted tests for the offline independent-generation counterfactual."""

import copy
import importlib.util
import math
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parent
    / "debug_candidate_score_to_rank_counterfactual.py"
)
SPEC = importlib.util.spec_from_file_location(
    "candidate_counterfactual",
    str(SCRIPT_PATH),
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
        generation,
        event_id,
        margin_change,
        next_gap,
        boundary_deficit,
        identity="0-4",
        target_sign=1.0,
        factor_order=2,
        static_dynamic="static"):
    return {
        "episode_generation": str(generation),
        "environment_episode_id": "episode-{}".format(generation),
        "capture_event_id": str(event_id),
        "capture_prey_id": "prey-1",
        "canonical_identity": identity,
        "participant_slots": identity,
        "static_dynamic_identity": static_dynamic,
        "factor_order": str(factor_order),
        "target_sign": str(target_sign),
        "target_weight": "0.0375",
        "signed_margin_change": str(margin_change),
        "pre_next_better_candidate_index": "7",
        "pre_next_better_margin_gap": str(next_gap),
        "pre_boundary_deficit": str(boundary_deficit),
        "pre_valid_population_count": "29",
    }


def _expect_error(rows, message_fragment):
    try:
        MODULE._independent_generation_counterfactual(rows)
    except RuntimeError as error:
        assert message_fragment in str(error), (message_fragment, str(error))
        return
    raise AssertionError("expected RuntimeError containing {!r}".format(
        message_fragment
    ))


def test_distinct_generations_accumulate_and_cross():
    result = MODULE._independent_generation_counterfactual([
        _row(10, 100, 0.04, 0.06, 0.15),
        _row(11, 101, 0.05, 0.06, 0.15),
        _row(12, 102, 0.08, 0.06, 0.15),
    ])
    assert result["independent_generation_count"] == 3
    assert result["unique_event_count"] == 3
    assert math.isclose(
        result["cumulative_signed_margin_change"],
        0.17,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["cumulative_target_weight"],
        0.1125,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result["first_counterfactual_next_rank_crossing_count"] == 2
    assert result["first_counterfactual_boundary_reach_count"] == 3
    assert result["quality_contract_valid"]
    assert not result["training_path_imported"]
    assert result["trajectory_neutral"]


def test_duplicate_generation_and_event_are_rejected():
    duplicate_generation = [
        _row(10, 100, 0.01, 0.1, 1.0),
        _row(10, 101, 0.01, 0.1, 1.0),
    ]
    _expect_error(duplicate_generation, "duplicate candidate evidence generation")

    duplicate_event = [
        _row(10, 100, 0.01, 0.1, 1.0),
        _row(11, 100, 0.01, 0.1, 1.0),
    ]
    duplicate_event[1]["environment_episode_id"] = (
        duplicate_event[0]["environment_episode_id"]
    )
    duplicate_event[1]["capture_prey_id"] = "prey-2"
    _expect_error(duplicate_event, "duplicate candidate evidence event")


def test_identity_sign_order_and_static_dynamic_conflicts_are_rejected():
    base = _row(10, 100, 0.01, 0.1, 1.0)
    cases = (
        ("canonical_identity", "2-5", "identities differ"),
        ("target_sign", "-1.0", "target signs conflict"),
        ("factor_order", "3", "factor orders conflict"),
        ("participant_slots", "2-5", "participant identities conflict"),
        (
            "static_dynamic_identity",
            "dynamic",
            "static/dynamic identities conflict",
        ),
    )
    for field, value, message in cases:
        other = _row(11, 101, 0.01, 0.1, 1.0)
        other[field] = value
        _expect_error([copy.deepcopy(base), other], message)


def test_static_dynamic_and_order2_order3_each_have_legal_closed_series():
    for factor_order, static_dynamic in ((2, "static"), (3, "dynamic")):
        rows = [
            _row(
                20,
                200,
                0.01,
                0.1,
                1.0,
                factor_order=factor_order,
                static_dynamic=static_dynamic,
            ),
            _row(
                21,
                201,
                0.01,
                0.1,
                1.0,
                factor_order=factor_order,
                static_dynamic=static_dynamic,
            ),
        ]
        if factor_order == 3:
            for row in rows:
                row["canonical_identity"] = "0-2-5"
                row["participant_slots"] = "0-2-5"
        result = MODULE._independent_generation_counterfactual(rows)
        assert result["factor_order"] == factor_order
        assert result["static_dynamic_identity"] == static_dynamic
        assert result["quality_contract_valid"]


def test_missing_provenance_and_single_generation_are_rejected():
    missing = _row(10, 100, 0.01, 0.1, 1.0)
    del missing["capture_event_id"]
    _expect_error(
        [missing, _row(11, 101, 0.01, 0.1, 1.0)],
        "missing required provenance field capture_event_id",
    )
    _expect_error(
        [_row(10, 100, 0.01, 0.1, 1.0)],
        "requires at least two distinct generations",
    )


def test_direction_cancellation_and_population_change_are_visible():
    first = _row(10, 100, 0.03, 0.2, 1.0)
    second = _row(11, 101, -0.01, 0.25, 1.1)
    second["pre_next_better_candidate_index"] = "9"
    second["pre_valid_population_count"] = "31"
    result = MODULE._independent_generation_counterfactual([first, second])
    assert result["direction_cancellation_count"] == 1
    assert result["population_context_changed"]
    assert result["steps"][1]["population_or_competitor_changed"]
    assert math.isclose(
        result["cumulative_signed_margin_change"],
        0.02,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_run106_candidate_scalar_reconstructs_target_transition_mean():
    weight = 0.03750000149011612
    violations = (2.234462022781372, 2.2853779792785645)
    target_transition_count = 2.0
    reconstructed = weight * sum(violations) / target_transition_count
    assert math.isclose(
        reconstructed,
        0.08474700152873993,
        rel_tol=0.0,
        abs_tol=2e-8,
    )
    assert math.isclose(
        2.0 * weight,
        0.15 * 0.5,
        rel_tol=0.0,
        abs_tol=5e-9,
    )


def main():
    tests = (
        test_distinct_generations_accumulate_and_cross,
        test_duplicate_generation_and_event_are_rejected,
        test_identity_sign_order_and_static_dynamic_conflicts_are_rejected,
        test_static_dynamic_and_order2_order3_each_have_legal_closed_series,
        test_missing_provenance_and_single_generation_are_rejected,
        test_direction_cancellation_and_population_change_are_visible,
        test_run106_candidate_scalar_reconstructs_target_transition_mean,
    )
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} independent evidence tests".format(len(tests)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
