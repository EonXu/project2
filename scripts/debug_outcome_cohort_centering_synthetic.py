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
    compute_outcome_conditioned_pair_credit,
    partition_pair_contrast_optimizer_chunks,
    scale_optimizer_cohort_pair_credit,
)
from utils.graph_sampling import (  # noqa: E402
    select_outcome_contrast_complete_episodes,
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

    # A full-buffer pair contrast can be centered while a sampled slice is
    # one-sided.  Consuming the stored slice was the run93 bug: normalization
    # v2 then amplified the positive-only branch.  The final optimizer cohort
    # must recompute its own contrast and therefore return strict zero here.
    full_pair_score = np.zeros((2, 3, 2), dtype=np.float32)
    full_pair_score[0, 0, 0] = 1.0
    full_pair_score[0, 1, 0] = 1.0
    full_pair = compute_outcome_conditioned_pair_credit(
        pair_transition_score=full_pair_score,
        episode_success=np.asarray([True, False, True], dtype=bool),
    )["credit"]
    assert np.any(full_pair[:, 0] > 0.0)
    stored_positive_slice = full_pair[:, [0], :]
    assert np.count_nonzero(stored_positive_slice) > 0
    selected_positive_only = scale_optimizer_cohort_pair_credit(
        pair_transition_score=full_pair_score[:, [0], :],
        episode_success=np.asarray([True], dtype=bool),
        coefficient=1.0,
        credit_cap=0.0,
    )
    assert selected_positive_only["class_complete"] == 0.0
    assert np.count_nonzero(selected_positive_only["credit"]) == 0
    print("PASS one-sided sampled pair cohort cannot consume stored mass")

    selected_pair = scale_optimizer_cohort_pair_credit(
        pair_transition_score=full_pair_score[:, :2, :],
        episode_success=np.asarray([True, False], dtype=bool),
        coefficient=0.1,
        credit_cap=0.025,
    )
    selected_episode_mass = selected_pair["credit"].sum(axis=(0, 2))
    assert selected_pair["class_complete"] == 1.0
    assert selected_episode_mass[0] > 0.0
    assert selected_episode_mass[1] < 0.0
    assert abs(float(selected_episode_mass.sum())) <= 1e-6
    assert np.isclose(
        selected_pair["postclip_positive_mass"],
        selected_pair["postclip_negative_mass"],
        atol=1e-6,
    )
    print("PASS final pair optimizer cohort remains centered after common cap")

    # A generic failed-capture support episode may have no strict-future pair
    # evidence. The pair population needs its own support completion.
    base = np.asarray([0], dtype=np.int64)
    recency = np.asarray([0, 1, 2, 3], dtype=np.int64)
    capture_positive = np.asarray([True, False, False, False])
    capture_negative = np.asarray([False, True, True, False])
    pair_positive = np.asarray([True, False, False, False])
    pair_negative = np.asarray([False, False, True, False])
    capture_support = select_outcome_contrast_complete_episodes(
        base,
        capture_positive,
        capture_negative,
        recency,
    )
    assert np.array_equal(
        capture_support["episode_indices"],
        np.asarray([0, 1], dtype=np.int64),
    )
    pair_support = select_outcome_contrast_complete_episodes(
        capture_support["episode_indices"],
        pair_positive,
        pair_negative,
        recency,
    )
    assert np.array_equal(
        pair_support["episode_indices"],
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    assert pair_support["class_complete"] == 1.0
    print("PASS replay support is completed on the true pair population")

    # All selected chunks must reach the same Adam step when pair evidence is
    # class complete. Splitting even an equal number of chunks into independent
    # masked means is not a population-total base-PPO objective.
    atomic = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.asarray([3, 0, 5, 2, 1, 4], dtype=np.int64),
        chunk_episode_membership=np.eye(6, dtype=bool),
        pair_evidence_episode_mask=np.asarray(
            [True, False, False, True, False, False],
            dtype=bool,
        ),
        episode_success=np.asarray(
            [True, True, True, False, False, False],
            dtype=bool,
        ),
        num_mini_batch=2,
    )
    assert atomic["class_complete"] == 1.0
    assert len(atomic["partitions"]) == 1
    pair_partition = atomic["partitions"][0]
    assert np.array_equal(
        pair_partition,
        np.asarray([3, 0, 5, 2, 1, 4], dtype=np.int64),
    )
    assert atomic["pair_zero_credit_filler_chunk_count"] == 4
    assert atomic["pair_partition_chunk_count"] == 6
    assert atomic["pair_partition_slot"] == -1
    assert atomic["partition_size_min"] == 6
    assert atomic["partition_size_max"] == 6
    assert atomic["partition_size_imbalance"] == 0
    covered = np.concatenate(atomic["partitions"])
    assert np.array_equal(
        np.sort(covered),
        np.arange(6, dtype=np.int64),
    )
    print("PASS mixed pair transaction uses one full-population Adam step")

    # Reproduce run96's first trajectory-changing shape: five 200-step
    # episodes become 100 ten-step chunks, with two pair-evidence episodes and
    # three non-pair episodes. Support v5 changed 40/60 into two 50-chunk Adam
    # steps; the population-total contract requires one 100-chunk Adam step.
    production_membership = np.repeat(
        np.eye(5, dtype=bool),
        20,
        axis=0,
    )
    production_atomic = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.random.RandomState(7).permutation(100),
        chunk_episode_membership=production_membership,
        pair_evidence_episode_mask=np.asarray(
            [True, True, False, False, False],
            dtype=bool,
        ),
        episode_success=np.asarray(
            [True, False, True, False, False],
            dtype=bool,
        ),
        num_mini_batch=2,
    )
    assert len(production_atomic["partitions"]) == 1
    assert production_atomic["pair_partition_chunk_count"] == 100
    assert production_atomic["pair_zero_credit_filler_chunk_count"] == 60
    assert production_atomic["partition_size_min"] == 100
    assert production_atomic["partition_size_max"] == 100
    assert production_atomic["partition_size_imbalance"] == 0
    assert np.array_equal(
        np.sort(production_atomic["partitions"][0]),
        np.arange(100, dtype=np.int64),
    )
    print("PASS run96 40/60 counterexample uses one 100-chunk transaction")

    # The run96 lower-bound counterexample had more pair chunks than one
    # ordinary balanced slot. It must not fall back to a 3/1 split whose
    # per-transition base weight depends on pair identity.
    lower_bound_atomic = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.asarray([2, 0, 3, 1], dtype=np.int64),
        chunk_episode_membership=np.eye(4, dtype=bool),
        pair_evidence_episode_mask=np.asarray(
            [True, True, True, False],
            dtype=bool,
        ),
        episode_success=np.asarray([True, False, True, False], dtype=bool),
        num_mini_batch=2,
    )
    assert len(lower_bound_atomic["partitions"]) == 1
    assert lower_bound_atomic["pair_partition_chunk_count"] == 4
    assert lower_bound_atomic["pair_zero_credit_filler_chunk_count"] == 1
    assert lower_bound_atomic["pair_partition_slot"] == -1
    assert lower_bound_atomic["partition_size_imbalance"] == 0
    assert np.array_equal(
        lower_bound_atomic["partitions"][0],
        np.asarray([2, 0, 3, 1], dtype=np.int64),
    )
    print("PASS atomic lower bound cannot create identity-weighted base PPO")

    # Equal chunk counts are still insufficient when their valid factor-mask
    # populations differ. Two independent masked means assign equal optimizer
    # weight to populations of size 1 and 3; one full-population masked mean
    # assigns every valid element equal weight.
    factor_values = np.asarray(
        [[1.0, 0.0, 0.0], [3.0, 5.0, 7.0]],
        dtype=np.float32,
    )
    factor_mask = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    first_partition = np.asarray([0], dtype=np.int64)
    second_partition = np.asarray([1], dtype=np.int64)
    separate_mean = 0.5 * (
        float(
            (factor_values[first_partition] * factor_mask[first_partition]).sum()
            / factor_mask[first_partition].sum()
        )
        + float(
            (
                factor_values[second_partition]
                * factor_mask[second_partition]
            ).sum()
            / factor_mask[second_partition].sum()
        )
    )
    population_total_mean = float(
        (factor_values * factor_mask).sum() / factor_mask.sum()
    )
    assert not np.isclose(separate_mean, population_total_mean)
    assert np.isclose(population_total_mean, 4.0)
    print("PASS factor-mask counterexample requires population-total reduction")

    odd_atomic = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.asarray([4, 0, 3, 1, 2], dtype=np.int64),
        chunk_episode_membership=np.eye(5, dtype=bool),
        pair_evidence_episode_mask=np.asarray(
            [True, False, False, False, True],
            dtype=bool,
        ),
        episode_success=np.asarray(
            [True, True, False, False, False],
            dtype=bool,
        ),
        num_mini_batch=2,
    )
    assert len(odd_atomic["partitions"]) == 1
    assert odd_atomic["pair_zero_credit_filler_chunk_count"] == 3
    assert odd_atomic["partition_size_min"] == 5
    assert odd_atomic["partition_size_max"] == 5
    assert odd_atomic["partition_size_imbalance"] == 0
    assert odd_atomic["pair_partition_slot"] == -1
    assert np.array_equal(
        np.sort(np.concatenate(odd_atomic["partitions"])),
        np.arange(5, dtype=np.int64),
    )
    odd_small_pair_partition = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.asarray([4, 0, 3, 1, 2], dtype=np.int64),
        chunk_episode_membership=np.eye(5, dtype=bool),
        pair_evidence_episode_mask=np.asarray(
            [True, False, False, False, True],
            dtype=bool,
        ),
        episode_success=np.asarray(
            [True, True, False, False, False],
            dtype=bool,
        ),
        num_mini_batch=2,
        pair_partition_slot=1,
    )
    assert odd_small_pair_partition[
        "pair_zero_credit_filler_chunk_count"
    ] == 3
    assert odd_small_pair_partition["pair_partition_chunk_count"] == 5
    assert odd_small_pair_partition["partition_size_min"] == 5
    assert odd_small_pair_partition["partition_size_max"] == 5
    assert odd_small_pair_partition["pair_partition_slot"] == -1
    assert np.array_equal(
        odd_atomic["partitions"][0],
        odd_small_pair_partition["partitions"][0],
    )
    print("PASS full-population transaction consumes no seeded size slot")

    # One-sided evidence keeps ordinary complete array_split semantics because
    # its signed pair loss is strictly zero.
    one_sided_partition = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.asarray([2, 0, 1], dtype=np.int64),
        chunk_episode_membership=np.eye(3, dtype=bool),
        pair_evidence_episode_mask=np.asarray(
            [True, False, False],
            dtype=bool,
        ),
        episode_success=np.asarray([True, False, True], dtype=bool),
        num_mini_batch=2,
    )
    assert one_sided_partition["class_complete"] == 0.0
    one_sided_covered = np.concatenate(one_sided_partition["partitions"])
    assert np.array_equal(
        np.sort(one_sided_covered),
        np.arange(3, dtype=np.int64),
    )
    print("PASS one-sided zero-credit population preserves chunk coverage")

    print("All 12 optimizer-cohort centering synthetic checks passed.")


if __name__ == "__main__":
    main()
