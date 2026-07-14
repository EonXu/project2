#!/usr/bin/env python
"""Synthetic checks for optimizer-cohort outcome centering.

These tests are NumPy-only.  The full AdjBuffer/batch path is exercised by
``validate_sddfg_dynamic_graph.py`` when the server environment has torch.
"""

import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.pair_credit import (  # noqa: E402
    compute_capture_to_win_triplet_outcome_advantage,
)


def _outcome(success, quality):
    return compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.asarray(success, dtype=bool),
        triplet_capture_quality=np.asarray(quality, dtype=np.float32),
    )


def main():
    # Full replay population: 3 successes / 1 failure.  The actual sampled
    # cohort is one success plus one failure, so its own baseline is 0.5 and
    # its expanded factor gate must sum to zero.
    full_quality = np.zeros((2, 4, 3), dtype=np.float32)
    full_quality[0, :, 0] = 1.0
    full = _outcome([True, True, True, False], full_quality)
    assert np.isclose(full["capture_episode_success_rate"], 0.75)

    cohort_quality = full_quality[:, [0, 3], :]
    cohort = _outcome([True, False], cohort_quality)
    assert np.isclose(cohort["capture_episode_success_rate"], 0.5)
    episode_totals = cohort["triplet_outcome_advantage"].sum(axis=(0, 2))
    assert np.allclose(episode_totals, [0.5, -0.5], atol=1e-6)
    assert abs(float(episode_totals.sum())) <= 1e-6
    print("PASS final optimizer cohort is centered independently of full buffer")

    # One episode can expose several identity-matched capture factors.  Its
    # total gate remains one centered episode label rather than scaling with
    # factor or capture count.
    repeated = np.zeros((3, 2, 4), dtype=np.float32)
    repeated[0, 0, 0] = 1.0
    repeated[1, 0, 1] = 1.0
    repeated[2, 0, 2] = 1.0
    repeated[0, 1, 0] = 1.0
    repeated_outcome = _outcome([True, False], repeated)
    repeated_totals = repeated_outcome[
        "triplet_outcome_advantage"
    ].sum(axis=(0, 2))
    assert np.allclose(repeated_totals, [0.5, -0.5], atol=1e-6)
    print("PASS episode total is invariant to capture-factor multiplicity")

    # Single-outcome protection and zero/padding factors remain untouched.
    one_sided = _outcome([True, True], full_quality[:, :2, :])
    assert np.count_nonzero(one_sided["triplet_outcome_advantage"]) == 0
    padded = np.zeros((2, 3, 5), dtype=np.float32)
    padded[0, 0, 0] = 1.0
    padded[0, 1, 0] = 1.0
    padded_outcome = _outcome([True, False, True], padded)
    assert np.count_nonzero(
        padded_outcome["triplet_outcome_advantage"][:, 2]
    ) == 0
    assert abs(float(
        padded_outcome["triplet_outcome_advantage"].sum()
    )) <= 1e-6
    print("PASS single-outcome and padding protections are preserved")

    print("All 3 optimizer-cohort centering synthetic checks passed.")


if __name__ == "__main__":
    main()
