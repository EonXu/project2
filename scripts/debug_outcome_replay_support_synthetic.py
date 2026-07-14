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
