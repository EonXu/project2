#!/usr/bin/env python
"""Fast server-side checks for the SDDFG dynamic graph training path."""

from __future__ import print_function

import os
import sys
import copy
from types import SimpleNamespace

# Match the training entrypoint before importing torch.  Strict deterministic
# CUDA linear layers require this CuBLAS workspace mode on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from gym.spaces import Box, Discrete

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.algorithm.adj_generator import Adj_Generator
from algorithms.sddfg.algorithm.agent_q_function import AgentQFunction
from algorithms.sddfg.algorithm.rSDDFGPolicy import (
    R_SDDFGPolicy,
    scatter_add,
)
from algorithms.sddfg.r_sddfg import (
    R_SDDFG,
    compute_capture_candidate_identity_loss,
    compute_capture_outcome_factor_ppo_loss,
)
from utils.adj_buffer import AdjPolicyBuffer
from utils.graph_sampling import select_outcome_contrast_complete_episodes
from utils.pair_credit import (
    CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
    canonical_capture_factor_catalog,
)


def _rng_states_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def validate_outcome_replay_support():
    """A recent cohort must preserve real signed contrast at optimization."""
    result = select_outcome_contrast_complete_episodes(
        np.asarray([3], dtype=np.int64),
        np.asarray([False, True, False, False]),
        np.asarray([False, False, False, True]),
        np.asarray([3, 2, 1, 0], dtype=np.int64),
    )
    assert np.array_equal(result["episode_indices"], [3, 1])
    assert result["positive_selected_count"] == 1
    assert result["negative_selected_count"] == 1
    assert result["class_complete"] == 1.0

    single_outcome = select_outcome_contrast_complete_episodes(
        np.asarray([2], dtype=np.int64),
        np.asarray([False, False, True]),
        np.asarray([False, False, False]),
        np.asarray([2, 1, 0], dtype=np.int64),
    )
    assert np.array_equal(single_outcome["episode_indices"], [2])
    assert single_outcome["augmented_count"] == 0
    assert single_outcome["class_complete"] == 0.0
    assert single_outcome["outcome_credit_enabled"] == 0.0

    exhausted = select_outcome_contrast_complete_episodes(
        np.asarray([3], dtype=np.int64),
        np.asarray([True, False, True, False]),
        np.asarray([False, False, False, True]),
        np.asarray([3, 2, 1, 0], dtype=np.int64),
        positive_eligible_mask=np.zeros(4, dtype=bool),
        negative_eligible_mask=np.asarray([False, False, False, True]),
    )
    assert exhausted["support_exhausted"] == 1.0
    assert exhausted["outcome_credit_enabled"] == 0.0


def validate_capture_outcome_factor_loss():
    """Outcome mass must not shrink when unrelated factor slots are added."""
    ratio_small = torch.ones((2, 2), requires_grad=True)
    delta_small = torch.tensor([[0.25, 0.0], [-0.25, 0.0]])
    small = compute_capture_outcome_factor_ppo_loss(
        ratio_small,
        torch.clamp(ratio_small, 0.8, 1.2),
        delta_small,
        torch.ones_like(delta_small),
        torch.ones((2, 1)),
    )
    ratio_large = torch.ones((2, 6), requires_grad=True)
    delta_large = torch.zeros((2, 6))
    delta_large[:, 0] = delta_small[:, 0]
    large = compute_capture_outcome_factor_ppo_loss(
        ratio_large,
        torch.clamp(ratio_large, 0.8, 1.2),
        delta_large,
        torch.ones_like(delta_large),
        torch.ones((2, 1)),
    )
    for key in ("loss", "positive_loss", "negative_loss"):
        assert torch.allclose(small[key], large[key], atol=1e-7)
    assert small["positive_loss"].item() < 0.0
    assert small["negative_loss"].item() > 0.0
    large["positive_loss"].backward(retain_graph=True)
    assert ratio_large.grad[0, 0].item() < 0.0
    # Keep the preflight compatible with the older PyTorch used by the
    # training server, where torch.count_nonzero is not available.  These
    # entries must be exactly zero because none of the unrelated factors
    # participates in the local outcome objective.
    assert torch.abs(ratio_large.grad[:, 1:]).sum().item() == 0.0


def validate_capture_candidate_identity_loss():
    """Candidate-only supervision must be exact, signed and target-local."""
    scores = torch.tensor(
        [[0.5, 0.7, 0.9], [0.5, 0.7, 0.9]],
        requires_grad=True,
    )
    delta = torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    result = compute_capture_candidate_identity_loss(
        scores,
        delta,
        torch.ones_like(scores),
        torch.ones((2, 1)),
    )
    assert result["positive_loss"].item() > 0.0
    assert result["negative_loss"].item() < 0.0
    result["positive_loss"].backward(retain_graph=True)
    assert scores.grad[0, 0].item() < 0.0
    assert torch.all(scores.grad[0, 1:] > 0.0).item()
    assert torch.abs(scores.grad[1]).sum().item() == 0.0
    scores.grad.zero_()
    result["negative_loss"].backward()
    assert scores.grad[1, 0].item() > 0.0
    assert torch.all(scores.grad[1, 1:] < 0.0).item()
    assert torch.abs(scores.grad[0]).sum().item() == 0.0


def validate_adj_buffer():
    """Exercise the exact single-episode axis case that previously crashed."""
    episode_length = 8
    num_agents = 6
    num_factor = 6
    active_counts = [4, 4, 2, 3, 5, 3, 4, 6, 6]

    for num_episodes in (1, 4):
        buffer = AdjPolicyBuffer(
            buffer_size=4,
            episode_length=episode_length,
            num_agents=num_agents,
            num_factor=num_factor,
            obs_space=Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            share_obs_space=Box(
                low=-1.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32,
            ),
            act_space=Discrete(7),
            use_same_share_obs=True,
            use_avail_acts=False,
            gamma=0.97,
            adj_return_adv_coef=1.0,
            adj_factor_adv_coef=0.25,
            use_adj_delayed_triplet_credit=True,
            adj_delayed_triplet_credit_coef=0.25,
            adj_delayed_triplet_credit_window=3,
            adj_delayed_triplet_credit_positive_only=True,
            adj_delayed_triplet_credit_min_adv=0.25,
            adj_delayed_triplet_credit_require_future_match=True,
            use_adj_delayed_triplet_success_gate=True,
            adj_delayed_triplet_success_gate_min_adv=0.50,
            adj_delayed_triplet_success_gate_scale=0.75,
            adj_delayed_triplet_success_gate_floor=0.25,
            adj_delayed_triplet_future_overlap_min_nodes=2,
            adj_delayed_triplet_partial_match_weight=0.50,
            use_adj_capture_to_win_credit=True,
            adj_capture_to_win_credit_coef=0.15,
            adj_capture_to_win_credit_min_outcome_adv=0.50,
            adj_capture_to_win_credit_scale=0.75,
            adj_capture_to_win_credit_cap=0.25,
            adj_capture_to_win_credit_require_future_match=True,
            use_adj_pair_triplet_complementary_credit=True,
            adj_pair_pursuit_credit_coef=0.10,
            adj_pair_pursuit_credit_window=3,
            adj_pair_pursuit_credit_cap=0.20,
            adj_pair_pursuit_credit_min_reward=0.0,
            seed=11,
        )
        buffer.filled_i = num_episodes
        buffer.current_i = num_episodes % buffer.buffer_size
        buffer.episode_generation[:num_episodes] = np.arange(
            num_episodes,
            dtype=np.int64,
        )
        buffer._next_episode_generation = num_episodes
        buffer.obs[:, :num_episodes] = -1.0
        # AdjPolicyBuffer deliberately initializes every unused transition as
        # terminal.  A populated episode must explicitly mark its non-final
        # transitions as live, exactly as runner insertion does.
        buffer.dones_env[:, :num_episodes, 0] = 0.0

        for episode_idx in range(num_episodes):
            for step, active_count in enumerate(active_counts):
                buffer.obs[step, episode_idx, :active_count] = (
                    0.01 * float(episode_idx + step + 1)
                )

            for step in range(episode_length):
                active_count = active_counts[step]
                buffer.rewards[step, episode_idx, :, 0] = (
                    0.01 * float(episode_idx + step + 1)
                )
                factor_slot = 0
                for first in range(0, active_count, 2):
                    second = first + 1
                    if second >= active_count:
                        second = 0
                    buffer.adj[
                        step,
                        episode_idx,
                        [first, second],
                        factor_slot,
                    ] = 1
                    factor_slot += 1
                # Exercise factor-local credit even when the single-episode
                # return control variate is zero.
                buffer.f_q[
                    step,
                    episode_idx,
                    :factor_slot,
                    0,
                ] = np.arange(factor_slot, dtype=np.float32)
                if step == 3:
                    buffer.adj[
                        step,
                        episode_idx,
                        [0, 1, 2],
                        factor_slot,
                    ] = 1
                if step == episode_length - 1:
                    # done[t] is post-action. A real capture produced by the
                    # terminal action must retain its graph/factor identity;
                    # only the nonexistent t+1 transition is invalid.
                    buffer.capture_counts[step, episode_idx, 0] = 1.0
                    buffer.capture_factor_matches[
                        step,
                        episode_idx,
                        0,
                    ] = 1.0
                    buffer.capture_identity_candidates[
                        step,
                        episode_idx,
                        0,
                    ] = 1.0
                    if num_episodes == 1 or episode_idx % 2 == 0:
                        buffer.success_now[step, episode_idx, 0] = 1.0
            buffer.dones_env[-1, episode_idx, 0] = 1.0

        buffer.compute_advantage(np.arange(num_episodes))
        expected_graph_ready = np.any(
            buffer.adj[
                :-1, :num_episodes, :, :buffer.num_factor
            ].sum(axis=2) > 0,
            axis=2,
        )
        assert np.array_equal(
            buffer.graph_advantage_ready[:, :num_episodes, 0],
            expected_graph_ready,
        ), "computed graph-return advantage ready mask was not persisted"
        if num_episodes > 1:
            assert np.any(
                np.abs(buffer.advantage[:, :num_episodes, 0]) > 0.0
            ), "computed graph-return advantage was not persisted"
        computed = buffer.f_advt[:, :num_episodes]
        assert computed.shape == (
            episode_length,
            num_episodes,
            num_factor,
            1,
        )
        assert np.isfinite(computed).all()
        assert np.abs(computed).sum() > 0.0
        assert buffer.pair_pursuit_credit[:, :num_episodes].sum() > 0.0
        assert buffer.pair_transition_delay[:, :num_episodes].max() >= 1.0
        assert buffer.triplet_capture_quality[:, :num_episodes].sum() > 0.0
        assert np.all(
            buffer.triplet_capture_quality[
                episode_length - 1, :num_episodes, 0, 0
            ] > 0.0
        ), "terminal capture identity was dropped by post-action done masking"
        outcome_gate = buffer.capture_to_win_quality_gate[
            :, :num_episodes
        ]
        assert np.isfinite(outcome_gate).all()
        if num_episodes == 1:
            # A single outcome class has no counterfactual baseline.
            assert np.count_nonzero(outcome_gate) == 0
        else:
            assert np.any(outcome_gate > 0.0)
            assert np.any(outcome_gate < 0.0)
            # capture_to_win_quality_gate keeps the singleton value axis:
            # [time, episode, factor, 1].  Summing only time/factor leaves
            # [episode, 1], which NumPy would silently broadcast against the
            # [episode] expectation and produce a false 2-D comparison.
            episode_gate_sum = outcome_gate.transpose(1, 0, 2, 3).reshape(
                num_episodes,
                -1,
            ).sum(axis=1)
            assert episode_gate_sum.shape == (num_episodes,)
            assert np.allclose(
                episode_gate_sum,
                np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32),
                atol=1e-6,
            )
            assert abs(float(episode_gate_sum.sum())) < 1e-6
            assert np.any(
                buffer.capture_to_win_triplet_credit[
                    :, :num_episodes
                ] < 0.0
            )
        # sample_inds() deliberately recomputes signed outcome credit from the
        # final cohort. Keep graph confidence positive so this axis/batch test
        # exercises both signs instead of testing the confidence gate.
        buffer.advantage[:, :num_episodes, 0] = 1.0
        recent_samples = list(
            buffer.sample_inds(
                data_chunk_length=4,
                num_mini_batch=1,
                recent_episode_window=2,
            )
        )
        assert len(recent_samples) == 1
        expected_recent = min(num_episodes, 2)
        assert buffer.last_sample_episode_count == expected_recent
        assert buffer.last_sample_episode_indices.size == expected_recent
        assert recent_samples[0][0].shape[0] >= 1
        assert len(recent_samples[0]) >= 30
        assert np.count_nonzero(recent_samples[0][3]) == 0, (
            "post-action terminal done was not shifted before graph training"
        )
        if num_episodes > 1:
            assert np.any(recent_samples[0][13] < 0.0), (
                "negative capture outcome credit was lost before the "
                "adjacency batch"
            )
            assert np.any(recent_samples[0][14] < 0.0), (
                "negative capture outcome gate was lost before the "
                "adjacency batch"
            )
            # Isolate replay-support rotation from outcome-credit scaling.
            # A successful episode may legitimately have zero final credit
            # when its graph advantage is non-positive, so the computed
            # trajectory above cannot guarantee two replay-positive episodes.
            # Build a deterministic signed replay fixture here: even episode
            # slots are positive, odd slots negative, and factor 0 at step 0
            # is valid for every episode in this validation setup.
            buffer.capture_to_win_triplet_credit[:, :num_episodes] = 0.0
            buffer.capture_to_win_quality_gate[:, :num_episodes] = 0.0
            # Cohort recentering rescales credit from the selected graph
            # advantage instead of trusting stored full-buffer credit.
            for replay_episode_idx in range(num_episodes):
                replay_sign = (
                    1.0 if replay_episode_idx % 2 == 0 else -1.0
                )
                buffer.capture_to_win_triplet_credit[
                    0, replay_episode_idx, 0, 0
                ] = replay_sign * 0.1
                buffer.capture_to_win_quality_gate[
                    0, replay_episode_idx, 0, 0
                ] = replay_sign * 0.5
            buffer.outcome_support_used[:num_episodes] = False
            buffer._cached_outcome_support_round = None
            buffer._cached_outcome_support_signature = None
            buffer._cached_outcome_support_selection = None
            contrast_complete_samples = list(
                buffer.sample_inds(
                    data_chunk_length=episode_length * 2,
                    num_mini_batch=1,
                    recent_episode_window=1,
                    outcome_support_round=100,
                )
            )
            assert buffer.last_sample_base_episode_count == 1
            assert buffer.last_sample_episode_count == 2
            assert buffer.last_sample_outcome_contrast_augmented_count == 1
            assert buffer.last_sample_outcome_positive_episode_count == 1
            assert buffer.last_sample_outcome_negative_episode_count == 1
            assert buffer.last_sample_outcome_class_complete == 1.0
            assert buffer.last_sample_outcome_positive_support_age == 1.0
            assert np.any(contrast_complete_samples[0][13] > 0.0), (
                "positive capture outcome credit was absent from the "
                "optimizer cohort"
            )
            assert np.any(contrast_complete_samples[0][13] < 0.0), (
                "negative capture outcome credit was absent from the "
                "optimizer cohort"
            )
            first_support_indices = (
                buffer.last_sample_episode_indices.copy()
            )
            cached_epoch_samples = list(
                buffer.sample_inds(
                    data_chunk_length=episode_length * 2,
                    num_mini_batch=1,
                    recent_episode_window=1,
                    outcome_support_round=100,
                )
            )
            assert np.array_equal(
                buffer.last_sample_episode_indices,
                first_support_indices,
            )
            assert buffer.last_sample_outcome_cached_selection_reused == 1.0
            assert np.any(cached_epoch_samples[0][13] > 0.0)
            assert np.any(cached_epoch_samples[0][13] < 0.0)

            rotated_samples = list(
                buffer.sample_inds(
                    data_chunk_length=episode_length * 2,
                    num_mini_batch=1,
                    recent_episode_window=1,
                    outcome_support_round=101,
                )
            )
            assert buffer.last_sample_outcome_cross_update_reuse_count == 0
            assert not np.array_equal(
                buffer.last_sample_episode_indices,
                first_support_indices,
            )
            assert buffer.last_sample_outcome_positive_support_age == 3.0
            assert np.any(rotated_samples[0][13] > 0.0)
            assert np.any(rotated_samples[0][13] < 0.0)

            exhausted_samples = list(
                buffer.sample_inds(
                    data_chunk_length=episode_length,
                    num_mini_batch=1,
                    recent_episode_window=1,
                    outcome_support_round=102,
                )
            )
            assert buffer.last_sample_outcome_support_exhausted == 1.0
            assert buffer.last_sample_outcome_credit_enabled == 0.0
            assert np.count_nonzero(exhausted_samples[0][13]) == 0
            assert np.count_nonzero(exhausted_samples[0][14]) == 0

            # Regression: a 3-success/1-failure full buffer has p=0.75, while
            # the actual optimizer cohort below contains one episode of each
            # class and must be recentered at p=0.5.  Stored full-buffer gates
            # deliberately retain their earlier values to prove sample_inds
            # recomputes the final cohort instead of forwarding them.
            buffer.success_now[:, :num_episodes, 0] = 0.0
            buffer.success_now[-1, [0, 2, 3], 0] = 1.0
            buffer.outcome_support_used[:num_episodes] = False
            buffer._cached_outcome_support_round = None
            buffer._cached_outcome_support_signature = None
            buffer._cached_outcome_support_selection = None
            # Exercise the real compute_advantage -> persisted sequence ->
            # sample_inds path.  Do not hand-write buffer.advantage here: that
            # exact shortcut allowed the old unwritten-storage bug to pass the
            # preflight while failing in training.  Signed graph-direction
            # behavior is covered independently by the NumPy scaling tests.
            recentered_samples = list(
                buffer.sample_inds(
                    data_chunk_length=episode_length * 2,
                    num_mini_batch=1,
                    recent_episode_window=1,
                    outcome_support_round=200,
                )
            )
            assert buffer.last_sample_outcome_credit_enabled == 1.0
            assert np.isclose(
                buffer.last_sample_outcome_full_buffer_baseline,
                0.75,
            )
            assert np.isclose(
                buffer.last_sample_outcome_trained_cohort_baseline,
                0.5,
            )
            assert np.isclose(
                buffer.last_sample_outcome_full_trained_baseline_gap,
                0.25,
            )
            assert buffer.last_sample_outcome_cohort_center_valid == 1.0
            assert buffer.last_sample_outcome_cohort_center_error <= 1e-6
            assert abs(
                buffer.last_sample_outcome_cohort_centered_sum
            ) <= 1e-6
            assert np.any(recentered_samples[0][13] > 0.0)
            assert np.any(recentered_samples[0][13] < 0.0)
            assert buffer.last_sample_outcome_signed_scaling_version == 3.0
            assert (
                buffer.last_sample_outcome_graph_advantage_source_ready_fraction
                == 1.0
            )
            assert buffer.last_sample_outcome_positive_credit_episode_count == 1
            assert buffer.last_sample_outcome_negative_credit_episode_count == 1
            assert buffer.last_sample_outcome_graph_confidence_mean > 0.0
            assert (
                buffer.last_sample_outcome_positive_zero_confidence_fraction
                == 0.0
            )
            assert (
                buffer.last_sample_outcome_negative_zero_confidence_fraction
                == 0.0
            )
            assert buffer.last_sample_outcome_gate_to_credit_drop_fraction == 0.0
        assert np.isfinite(recent_samples[0][13]).all()
        assert np.isfinite(recent_samples[0][14]).all()
        assert np.isfinite(recent_samples[0][15]).all()
        assert np.isfinite(recent_samples[0][16]).all()
        assert np.isfinite(recent_samples[0][17]).all()
        assert np.isfinite(recent_samples[0][18]).all()
        assert np.isfinite(recent_samples[0][20]).all()
        assert np.isfinite(recent_samples[0][21]).all()
        assert np.isfinite(recent_samples[0][22]).all()
        assert np.isfinite(recent_samples[0][23]).all()
        assert np.isfinite(recent_samples[0][24]).all()
        assert np.isfinite(recent_samples[0][25]).all()
        assert np.isfinite(recent_samples[0][26]).all()
        assert np.isfinite(recent_samples[0][27]).all()
        assert np.isfinite(recent_samples[0][28]).all()
        assert np.isfinite(recent_samples[0][29]).all()
        assert (
            recent_samples[0][29].shape[-1]
            == CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH
        )
        assert recent_samples[0][29][..., 10].sum() == expected_recent
        assert recent_samples[0][29][..., 20].sum() == expected_recent
        assert recent_samples[0][29][..., 21].sum() == expected_recent
        assert recent_samples[0][29][..., 22].sum() == 0.0
        assert buffer.capture_to_win_episode_success_gate[
            :, :num_episodes
        ].sum() > 0.0
        expected_failed_captures = 0 if num_episodes == 1 else 2
        assert buffer.failed_episode_capture_count[
            :, :num_episodes
        ].sum() == expected_failed_captures
        split_samples = list(
            buffer.sample_inds(
                data_chunk_length=4,
                num_mini_batch=2,
                recent_episode_window=2,
            )
        )
        assert len(split_samples) == 2
        for split_sample in split_samples:
            diagnostics = split_sample[29]
            # Window-class diagnostics are global to the current buffer and
            # must be present in every sampled chunk, not only in the chunk
            # that happens to contain the per-episode marker at t=0. Padding
            # after an early terminal keeps diagnostics at zero; the terminal
            # action itself remains valid because done is post-action.
            class_sum = diagnostics[..., 4:8].sum(axis=-1)
            diagnostic_valid = class_sum > 0.0
            assert np.any(diagnostic_valid)
            assert np.allclose(
                class_sum[diagnostic_valid],
                1.0,
                atol=1e-6,
            )


def validate_candidate_only_replay_path():
    """A real unselected pair must reach the candidate-only batch field."""
    episode_length = 4
    num_agents = 4
    buffer = AdjPolicyBuffer(
        buffer_size=2,
        episode_length=episode_length,
        num_agents=num_agents,
        num_factor=2,
        obs_space=Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        share_obs_space=Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
        act_space=Discrete(7),
        use_same_share_obs=True,
        use_avail_acts=False,
        use_adj_capture_to_win_credit=True,
        adj_capture_to_win_credit_coef=0.15,
        adj_capture_to_win_credit_cap=0.25,
        seed=19,
    )
    buffer.filled_i = 2
    buffer.current_i = 0
    buffer.episode_generation[:] = np.asarray([0, 1], dtype=np.int64)
    buffer._next_episode_generation = 2
    buffer.obs[:, :, :, :] = 0.1
    buffer.dones_env[:, :, 0] = 0.0
    buffer.dones_env[-1, :, 0] = 1.0
    for step in range(episode_length):
        # Exact capture pair (0, 1) is deliberately not active.
        buffer.adj[step, :, [0, 2], 0] = 1
        buffer.adj[step, :, [1, 3], 1] = 1
    pair_index = canonical_capture_factor_catalog(num_agents, 3).index((0, 1))
    buffer.capture_counts[1, :, 0] = 1.0
    buffer.capture_candidate_only_matches[1, :, pair_index] = 1.0
    buffer.capture_candidate_behavior[1, :, pair_index, :] = np.asarray(
        [0.4, 1.0, 1.0, 0.0],
        dtype=np.float32,
    )
    buffer.success_now[2, 0, 0] = 1.0
    buffer.compute_advantage(np.asarray([0, 1], dtype=np.int64))
    sample = list(buffer.sample_inds(
        data_chunk_length=4,
        num_mini_batch=1,
        recent_episode_window=2,
    ))[0]
    assert len(sample) >= 32
    assert np.count_nonzero(sample[13]) == 0, (
        "candidate-only event leaked into active factor outcome credit"
    )
    candidate_delta = sample[30]
    candidate_behavior = sample[31]
    assert np.any(candidate_delta > 0.0)
    assert np.any(candidate_delta < 0.0)
    nonzero_candidates = np.any(
        np.abs(candidate_delta) > 0.0,
        axis=(0, 1, 3),
    )
    assert int(np.sum(nonzero_candidates)) == 1
    assert bool(nonzero_candidates[pair_index])
    pair_target = np.abs(candidate_delta[:, :, pair_index, 0]) > 0.0
    assert np.all(candidate_behavior[:, :, pair_index, 1][pair_target] == 1.0)
    assert np.all(candidate_behavior[:, :, pair_index, 2][pair_target] == 1.0)


def validate_scatter_gradient():
    src = torch.randn(7, 5, requires_grad=True)
    index = torch.tensor([0, 1, 0, 2, 1, 2, 0], dtype=torch.long)
    output = scatter_add(src, index, dim=0, dim_size=3)
    assert tuple(output.shape) == (3, 5)
    expected = torch.zeros_like(output)
    for src_idx, dst_idx in enumerate(index.tolist()):
        expected[dst_idx] = expected[dst_idx] + src[src_idx].detach()
    assert bool(torch.allclose(output.detach(), expected, atol=1e-6))
    output.sum().backward()
    assert src.grad is not None
    assert bool(torch.isfinite(src.grad).all().item())


def validate_deterministic_cuda_scatter():
    """Exercise the non-atomic fallback used by deterministic CUDA training."""
    if not torch.cuda.is_available():
        return False

    has_strict_api = (
        hasattr(torch, "use_deterministic_algorithms")
        and hasattr(torch, "are_deterministic_algorithms_enabled")
    )
    was_enabled = (
        bool(torch.are_deterministic_algorithms_enabled())
        if has_strict_api else False
    )
    was_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    try:
        # Old server PyTorch has no strict API.  rSDDFGPolicy deliberately
        # treats this cuDNN flag as the compatibility signal for its
        # deterministic scatter/gather implementations.
        torch.backends.cudnn.deterministic = True
        if has_strict_api:
            torch.use_deterministic_algorithms(True)
        src = torch.randn(11, 5, device="cuda", requires_grad=True)
        index = torch.tensor(
            [0, 2, 1, 2, 0, 3, 1, 3, 2, 0, 3],
            dtype=torch.long,
            device="cuda",
        )
        output = scatter_add(src, index, dim=0, dim_size=4)
        expected = torch.zeros(4, 5, dtype=src.dtype)
        for src_idx, dst_idx in enumerate(index.cpu().tolist()):
            expected[dst_idx] += src[src_idx].detach().cpu()
        assert bool(
            torch.allclose(
                output.detach().cpu(),
                expected,
                atol=1e-6,
            )
        )
        output.sum().backward()
        assert src.grad is not None
        assert bool(torch.isfinite(src.grad).all().item())
    finally:
        if has_strict_api:
            torch.use_deterministic_algorithms(was_enabled)
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
    return True


def validate_factor_q_gradient():
    args = SimpleNamespace(
        use_ReLU=True,
        use_orthogonal=True,
        gain=0.01,
    )
    for order in (1, 2, 3):
        action_output = 7 if order == 1 else 3 * 7
        network = AgentQFunction(
            args=args,
            obs_dim=4,
            input_dim=64,
            num_orders=order,
            act_dim=action_output,
            device=torch.device("cpu"),
        )
        obs = torch.randn(5, order * 4)
        hidden = torch.randn(5, order * 64)
        output = network(obs, hidden, no_sequence=False)
        assert bool(torch.isfinite(output).all().item())
        output.pow(2).mean().backward()
        grads = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.grad is not None
        ]
        assert grads
        assert all(bool(torch.isfinite(grad).all().item()) for grad in grads)
        assert sum(float(grad.abs().sum().item()) for grad in grads) > 0.0


def validate_trainer_without_critics():
    class MinimalPolicy(object):
        def __init__(self):
            self.act_dim = 7
            self.module = torch.nn.Linear(4, 7)

        def parameters(self):
            return list(self.module.parameters())

        def critic_fv_parameters(self):
            return []

        def critic_vtot_parameters(self):
            return []

    args = SimpleNamespace(
        use_popart=False,
        use_value_active_masks=False,
        use_per=False,
        per_eps=1e-6,
        use_huber_loss=False,
        huber_delta=10.0,
        clip_param=0.2,
        use_vfunction=False,
        lr=5e-4,
        critic_lr=5e-4,
        adj_lr=3e-4,
        tau=0.005,
        opti_eps=1e-5,
        weight_decay=0.0,
        highest_orders=3,
        use_dyn_graph=True,
        num_factor=6,
        entropy_coef=0.01,
        adj_entropy_coef=0.002,
        use_valuenorm=False,
        adj_max_grad_norm=0.5,
        adj_lr_anneal_steps=200000,
        adj_lr_decay_floor=2e-5,
        adj_entropy_coef_final=0.0,
        adj_entropy_anneal_steps=200000,
    )
    trainer = R_SDDFG(
        args=args,
        num_agents=6,
        policies={"policy_0": MinimalPolicy()},
        adj_network=torch.nn.Linear(4, 4),
        policy_mapping_fn=lambda _agent_id: "policy_0",
        device=torch.device("cpu"),
        episode_length=8,
    )
    assert trainer.critic_fv_optimizer is None
    assert trainer.critic_vtot_optimizer is None
    initial_adj_lr = trainer.adj_optimizer.param_groups[0]["lr"]
    midpoint_adj_lr = trainer.adj_lr_decay(100000)
    midpoint_entropy_coef = trainer.adj_entropy_coef
    final_adj_lr = trainer.adj_lr_decay(200000)
    final_entropy_coef = trainer.adj_entropy_coef
    assert abs(initial_adj_lr - 3e-4) < 1e-12
    assert abs(midpoint_adj_lr - 1.5e-4) < 1e-12
    assert abs(final_adj_lr - 2e-5) < 1e-12
    assert abs(midpoint_entropy_coef - 0.001) < 1e-12
    assert abs(final_entropy_coef) < 1e-12


def validate_full_policy_gradient(device=None):
    if device is None:
        device = torch.device("cpu")
    args = SimpleNamespace(
        hidden_size=64,
        lamda=0.0,
        num_rank=3,
        num_factor=6,
        prev_act_inp=False,
        highest_orders=3,
        use_vfunction=False,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=500000,
        use_ReLU=True,
        use_orthogonal=True,
        use_feature_normalization=True,
        gain=0.01,
        seed=13,
        msg_iterations=4,
        msg_normalized=True,
        msg_anytime=True,
    )
    num_agents = 6
    batch_size = 3
    obs_dim = 10
    act_dim = 7
    policy = R_SDDFGPolicy(
        config={
            "args": args,
            "device": device,
            "num_agents": num_agents,
        },
        policy_config={
            "obs_space": Box(
                low=-1.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "act_space": Discrete(act_dim),
            "cent_obs_dim": 20,
        },
        train=True,
    )

    obs = torch.randn(
        batch_size,
        num_agents,
        obs_dim,
        device=device,
    )
    initial_hidden = torch.zeros(
        batch_size * num_agents,
        args.hidden_size,
        device=device,
    )
    _, hidden, _ = policy.get_hidden_states(
        obs.reshape(batch_size * num_agents, obs_dim),
        prev_actions=None,
        rnn_states=initial_hidden,
        dones=torch.zeros(
            batch_size * num_agents,
            1,
            device=device,
        ),
    )
    hidden = hidden.reshape(batch_size, num_agents, args.hidden_size)

    total_factors = args.num_factor + num_agents
    adj = torch.zeros(
        batch_size,
        num_agents,
        total_factors,
        dtype=torch.long,
    )
    dynamic_factors = (
        (0, 1),
        (2, 3),
        (4, 5),
        (0, 2, 4),
        (1, 3),
        (2, 4, 5),
    )
    for factor_idx, nodes in enumerate(dynamic_factors):
        adj[:, list(nodes), factor_idx] = 1
    for agent_idx in range(num_agents):
        adj[:, agent_idx, args.num_factor + agent_idx] = 1
    # Build the small static fixture on CPU.  PyTorch 1.8 on Windows has an
    # intermittent CUDA advanced-assignment assertion when two validation
    # processes share a device; the policy itself only consumes this tensor.
    adj = adj.to(device)

    action_indices = torch.randint(
        low=0,
        high=act_dim,
        size=(batch_size, num_agents, 1),
        device=device,
    )
    q_tot = policy.get_q_values(
        obs_batch=obs,
        rnn_q_states_batch=hidden,
        action_batch=action_indices,
        adj_input=adj,
        no_sequence=False,
        dones=torch.zeros(
            batch_size,
            num_agents,
            1,
            device=device,
        ),
    )
    assert tuple(q_tot.shape) == (batch_size,)
    assert bool(torch.isfinite(q_tot).all().item())
    q_tot.pow(2).mean().backward()

    rnn_grad = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in policy.rnn_network.parameters()
        if parameter.grad is not None
    )
    q_grad = 0.0
    for order in range(1, args.highest_orders + 1):
        q_grad += sum(
            float(parameter.grad.abs().sum().item())
            for parameter in policy.q_network[order].parameters()
            if parameter.grad is not None
        )
    assert rnn_grad > 0.0
    assert q_grad > 0.0


def validate_deterministic_cuda_full_policy_gradient():
    if not torch.cuda.is_available():
        return False

    has_strict_api = (
        hasattr(torch, "use_deterministic_algorithms")
        and hasattr(torch, "are_deterministic_algorithms_enabled")
    )
    was_enabled = (
        bool(torch.are_deterministic_algorithms_enabled())
        if has_strict_api else False
    )
    was_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    try:
        torch.backends.cudnn.deterministic = True
        if has_strict_api:
            torch.use_deterministic_algorithms(True)
        validate_full_policy_gradient(torch.device("cuda"))
    finally:
        if has_strict_api:
            torch.use_deterministic_algorithms(was_enabled)
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
    return True


def main():
    validate_capture_outcome_factor_loss()
    validate_capture_candidate_identity_loss()
    validate_candidate_only_replay_path()
    torch.manual_seed(7)
    np.random.seed(7)

    args = SimpleNamespace(
        max_player_num=6,
        num_factor=6,
        highest_orders=3,
        sparsity=0.3,
        seed=7,
        use_adj_topology_persistence=True,
        hidden_size=64,
        gat_heads=4,
        gat_negative_slope=0.2,
        gat_hyperedge_hidden=64,
        adj_order3_bonus=1.35,
        adj_order3_bonus_start=1.0,
        adj_order3_bonus_anneal_steps=100,
        adj_sampling_temperature_start=1.0,
        adj_sampling_temperature_final=0.35,
        adj_sampling_temperature_anneal_steps=100,
        adj_min_order3_ratio_start=0.5,
        adj_min_order3_ratio_final=0.72,
        adj_min_order3_ratio_anneal_steps=100,
        adj_max_order3_ratio_start=0.82,
        adj_max_order3_ratio_final=0.82,
        adj_max_order3_ratio_anneal_steps=0,
        adj_greedy_sample_prob_start=0.0,
        adj_greedy_sample_prob_final=0.75,
        adj_greedy_sample_prob_anneal_steps=100,
        adj_greedy_sample_prob_cap=0.50,
        adj_order3_quota_mode="soft",
        adj_order3_soft_quota_coef=2.5,
        adj_triplet_feature_mode="synergy",
        adj_triplet_balance_coef=0.75,
        use_adj_advantage_triplet_scorer=True,
        adj_triplet_credit_ema_alpha=0.05,
        adj_triplet_credit_score_coef=0.50,
        adj_triplet_credit_score_scale=0.05,
        use_adj_triplet_credit_direct_rank=True,
        adj_triplet_credit_rank_coef=1.25,
        adj_triplet_credit_min_multiplier=0.70,
        adj_triplet_credit_max_multiplier=2.50,
        adj_triplet_credit_negative_rank_scale=0.25,
        adj_triplet_credit_min_positive_fraction=0.45,
        adj_triplet_negative_graph_penalty=0.50,
        adj_order3_quota_score_floor=0.45,
        adj_min_pair_ratio=0.20,
        adj_order_adv_coef=0.75,
        adj_order_adv_positive_only=True,
        adj_order_adv_negative_coef=0.20,
        adj_order_adv_require_positive_graph_adv=True,
        use_adj_triplet_graph_return_credit=True,
        adj_triplet_graph_return_credit_coef=0.25,
        adj_triplet_graph_return_credit_cap=0.35,
        adj_triplet_graph_return_credit_min_graph_adv=0.0,
        adj_triplet_graph_return_credit_raw_gate_scale=0.75,
        adj_triplet_graph_return_credit_require_delayed_gate=True,
        use_adj_delayed_triplet_credit=True,
        adj_delayed_triplet_credit_coef=0.25,
        adj_delayed_triplet_credit_window=20,
        adj_delayed_triplet_credit_cap=0.75,
        adj_delayed_triplet_credit_min_reward=0.0,
        adj_delayed_triplet_credit_positive_only=True,
        adj_delayed_triplet_credit_min_adv=0.25,
        adj_delayed_triplet_credit_require_future_match=True,
        use_adj_delayed_triplet_success_gate=True,
        adj_delayed_triplet_success_gate_min_adv=0.50,
        adj_delayed_triplet_success_gate_scale=0.75,
        adj_delayed_triplet_success_gate_floor=0.10,
        adj_delayed_triplet_future_overlap_min_nodes=2,
        adj_delayed_triplet_partial_match_weight=0.35,
        use_adj_capture_to_win_credit=True,
        adj_capture_to_win_credit_coef=0.15,
        adj_capture_to_win_credit_min_outcome_adv=0.50,
        adj_capture_to_win_credit_scale=0.75,
        adj_capture_to_win_credit_cap=0.25,
        adj_capture_to_win_credit_require_future_match=True,
        use_adj_pair_triplet_complementary_credit=True,
        adj_pair_pursuit_credit_coef=0.10,
        adj_pair_pursuit_credit_window=20,
        adj_pair_pursuit_credit_cap=0.20,
        adj_pair_pursuit_credit_min_reward=0.0,
        use_adj_order3_credit_gate=True,
        use_adj_order3_relative_credit_gate=True,
        adj_order3_credit_gate_loss_scale=0.004,
        adj_order3_credit_gate_margin=0.0,
        adj_order3_credit_gate_min_scale=0.70,
        adj_order3_credit_gate_ema_alpha=0.10,
        adj_order3_credit_gate_max_delta=0.05,
        adj_ppo_clip_stop_ratio=0.35,
        adj_ppo_factor_clip_stop_ratio=0.35,
        adj_ppo_min_epochs=1,
        use_adj_ppo_stale_trust=True,
        adj_ppo_stale_trust_clip=0.20,
        adj_ppo_stale_trust_scale=0.25,
        adj_ppo_stale_trust_min_weight=0.25,
        adj_recent_episode_window=4,
        use_adj_dynamic_recent_window=True,
        adj_recent_episode_window_min=1,
        adj_recent_window_stale_threshold=0.35,
        adj_recent_window_factor_stale_threshold=0.30,
        adj_recent_window_shrink_patience=1,
        adj_recent_window_recover_patience=2,
        adj_recent_window_recover_stale_threshold=0.28,
        adj_recent_window_recover_factor_stale_threshold=0.24,
        adj_recent_window_severe_margin=0.20,
        adj_recent_episode_window_emergency=1,
        adj_recent_window_emergency_stale_threshold=0.40,
        adj_recent_window_emergency_factor_stale_threshold=0.25,
        adj_hidden_dim=64,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        adj_anneal_time=500000,
        require_connected_adj=True,
    )
    device = torch.device("cpu")
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )

    active_counts = [2, 3, 4, 5, 6]
    batch_size = len(active_counts)
    rnn_obs = torch.randn(
        batch_size,
        args.max_player_num,
        args.hidden_size,
        device=device,
    )
    dones = torch.ones(
        batch_size,
        args.max_player_num,
        1,
        dtype=torch.bool,
        device=device,
    )
    for batch_idx, active_count in enumerate(active_counts):
        dones[batch_idx, :active_count] = False

    # Train-consistent eval graphs must use a dedicated RNG.  Otherwise every
    # evaluation changes the following training topology sequence.
    train_rng_before = graph.rng.get_state()
    eval_rng_before = graph.eval_rng.get_state()
    graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
        use_eval_rng=True,
    )
    assert _rng_states_equal(train_rng_before, graph.rng.get_state())
    assert not _rng_states_equal(eval_rng_before, graph.eval_rng.get_state())

    behavior_prob, adj, entropy = graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    assert abs(graph.current_order3_bonus - 1.35) < 1e-12
    assert abs(graph.current_sampling_temperature - 0.35) < 1e-12
    assert abs(graph.current_min_order3_ratio - 0.72) < 1e-12
    assert abs(graph.current_max_order3_ratio - 0.82) < 1e-12
    assert abs(
        graph.current_greedy_sample_prob
        - args.adj_greedy_sample_prob_cap
    ) < 1e-12
    assert abs(graph.current_order3_credit_gate - 1.0) < 1e-12
    target_prob, target_entropy = graph.evaluate_prob(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        adj=adj,
    )

    assert tuple(adj.shape) == (
        batch_size,
        args.max_player_num,
        args.num_factor,
    )
    for tensor in (
        behavior_prob,
        target_prob,
        entropy,
        target_entropy,
    ):
        assert bool(torch.isfinite(tensor).all().item())

    graph.zero_grad()
    candidate_scores, candidate_valid = (
        graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    candidate_delta = torch.zeros_like(candidate_scores)
    # Use the three-active-agent row.  With exactly two active agents each GAT
    # row has only one legal neighbor, so softmax is identically one and the
    # exact pair score is mathematically parameter-independent (zero gradient).
    # The candidate objective must be checked where graph selection has a real
    # alternative.
    candidate_delta[1, 0] = 0.2  # canonical pair (0, 1)
    candidate_loss = compute_capture_candidate_identity_loss(
        candidate_scores,
        candidate_delta,
        candidate_valid,
        torch.ones((batch_size, 1)),
    )
    candidate_loss["loss"].backward()
    candidate_grad = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in graph.gat.parameters()
        if parameter.grad is not None
    )
    assert candidate_grad > 0.0
    graph.zero_grad()

    # The production current-score path must move the exact pair weight in the
    # signed direction after a real optimizer step.  Stored behavior metadata
    # is detached and cannot satisfy this check.
    candidate_graph = copy.deepcopy(graph)
    target_row = 1
    target_col = 0
    positive_before_scores, positive_before_valid = (
        candidate_graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    positive_before = float((
        positive_before_scores[target_row, target_col]
        / (
            positive_before_scores[target_row]
            * positive_before_valid[target_row]
        ).sum()
    ).detach().cpu().item())
    positive_optimizer = torch.optim.SGD(
        candidate_graph.gat.parameters(), lr=1e-3
    )
    positive_optimizer.zero_grad()
    positive_scores, positive_valid = (
        candidate_graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    positive_delta = torch.zeros_like(positive_scores)
    positive_delta[target_row, target_col] = 0.2
    positive_loss = compute_capture_candidate_identity_loss(
        positive_scores,
        positive_delta,
        positive_valid,
        torch.ones((batch_size, 1)),
    )["loss"]
    positive_loss.backward()
    positive_optimizer.step()
    positive_after_scores, positive_after_valid = (
        candidate_graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    positive_after = float((
        positive_after_scores[target_row, target_col]
        / (
            positive_after_scores[target_row]
            * positive_after_valid[target_row]
        ).sum()
    ).detach().cpu().item())
    assert positive_after > positive_before

    candidate_graph = copy.deepcopy(graph)
    negative_before_scores, negative_before_valid = (
        candidate_graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    negative_before = float((
        negative_before_scores[target_row, target_col]
        / (
            negative_before_scores[target_row]
            * negative_before_valid[target_row]
        ).sum()
    ).detach().cpu().item())
    negative_optimizer = torch.optim.SGD(
        candidate_graph.gat.parameters(), lr=1e-3
    )
    negative_optimizer.zero_grad()
    negative_scores, negative_valid = (
        candidate_graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    negative_delta = torch.zeros_like(negative_scores)
    negative_delta[target_row, target_col] = -0.2
    negative_loss = compute_capture_candidate_identity_loss(
        negative_scores,
        negative_delta,
        negative_valid,
        torch.ones((batch_size, 1)),
    )["loss"]
    negative_loss.backward()
    negative_optimizer.step()
    negative_after_scores, negative_after_valid = (
        candidate_graph.evaluate_candidate_identity_scores(rnn_obs, dones)
    )
    negative_after = float((
        negative_after_scores[target_row, target_col]
        / (
            negative_after_scores[target_row]
            * negative_after_valid[target_row]
        ).sum()
    ).detach().cpu().item())
    assert negative_after < negative_before
    graph.zero_grad()

    # With persistence mass set to one in this isolated fixture, every
    # still-eligible same-slot factor must be retained.  evaluate_prob must
    # reconstruct exactly the behavior probability using replayed previous_adj.
    saved_greedy_schedule = (
        graph.greedy_sample_prob_start,
        graph.greedy_sample_prob_final,
        graph.greedy_sample_prob_cap,
    )
    graph.greedy_sample_prob_start = 1.0
    graph.greedy_sample_prob_final = 1.0
    graph.greedy_sample_prob_cap = 1.0
    _, persistence_seed_adj, _ = graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    persistence_behavior, persistence_adj, _ = graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
        previous_adj=persistence_seed_adj,
    )
    persistence_target, _ = graph.evaluate_prob(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        adj=persistence_adj,
        previous_adj=persistence_seed_adj,
    )
    assert bool(torch.equal(persistence_seed_adj, persistence_adj))
    persistence_selected = persistence_adj.bool()
    assert bool(
        torch.allclose(
            persistence_behavior[persistence_selected],
            persistence_target[persistence_selected],
            atol=1e-7,
        )
    )
    assert graph.last_topology_persistence_candidate_fraction > 0.0
    assert graph.last_topology_persistence_selected_fraction > 0.0
    persistence_diagnostics_before_terminal = (
        graph.last_topology_persistence_candidate_fraction,
        graph.last_topology_persistence_selected_fraction,
    )
    graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=torch.ones_like(dones),
        explore=True,
        t_env=500000,
        previous_adj=persistence_adj,
    )
    assert persistence_diagnostics_before_terminal == (
        graph.last_topology_persistence_candidate_fraction,
        graph.last_topology_persistence_selected_fraction,
    )
    (
        graph.greedy_sample_prob_start,
        graph.greedy_sample_prob_final,
        graph.greedy_sample_prob_cap,
    ) = saved_greedy_schedule
    graph._update_graph_schedules(500000)

    order_counts = {2: 0, 3: 0}
    for batch_idx, active_count in enumerate(active_counts):
        active = torch.zeros(args.max_player_num, dtype=torch.bool)
        active[:active_count] = True
        selected = adj[batch_idx].bool()
        assert not bool(selected[~active].any().item())
        degree = selected.long().sum(dim=1)
        assert bool((degree[active] > 0).all().item())

        # Coverage alone is insufficient for global message passing. Verify
        # that all active nodes belong to one factor-graph component.
        co_membership = (
            selected.float()
            @ selected.float().transpose(0, 1)
        ) > 0.0
        reachability = co_membership.clone()
        reachability = reachability | torch.eye(
            args.max_player_num,
            dtype=torch.bool,
        )
        for _ in range(args.max_player_num):
            reachability = reachability | (
                reachability.float() @ reachability.float() > 0.0
            )
        active_idx = torch.where(active)[0]
        assert bool(
            reachability[active_idx][:, active_idx].all().item()
        )

        orders = selected.long().sum(dim=0)
        valid_orders = orders[orders > 0]
        assert bool(
            ((valid_orders == 2) | (valid_orders == 3)).all().item()
        )
        for order in (2, 3):
            order_counts[order] += int(
                (valid_orders == order).long().sum().item()
            )
        if active_count >= 3:
            expected_triplets = int(
                np.ceil(
                    graph.current_min_order3_ratio
                    * float(valid_orders.numel())
                    - 1e-8
                )
            )
            # The lower band cannot require more triplets than exist for the
            # active roster (e.g. only one unique triplet exists for 3 agents).
            max_feasible_triplets = 0
            if active_count >= 3:
                max_feasible_triplets = int(
                    active_count
                    * (active_count - 1)
                    * (active_count - 2)
                    // 6
                )
            expected_triplets = min(
                expected_triplets,
                max_feasible_triplets,
            )
            max_triplets = int(
                np.floor(
                    graph.current_max_order3_ratio
                    * float(valid_orders.numel())
                    + 1e-8
                )
            )
            max_triplets = max(expected_triplets, max_triplets)
            actual_triplets = int(
                (valid_orders == 3).long().sum().item()
            )
            if graph.order3_quota_mode == "hard":
                assert actual_triplets >= expected_triplets
            assert actual_triplets <= max_triplets
        assert bool((target_prob[batch_idx][selected] > 0.0).all().item())

    factor_mask_for_credit = (
        adj.float().sum(dim=1) > 1.5
    ).float()
    synthetic_local_credit = torch.linspace(
        -0.2,
        0.2,
        steps=batch_size * args.num_factor,
        device=device,
    ).reshape(batch_size, args.num_factor)
    synthetic_graph_credit = torch.ones(
        batch_size,
        1,
        device=device,
    )
    credit_info = graph.update_factor_credit_memory(
        adj.float(),
        synthetic_local_credit,
        synthetic_graph_credit,
        factor_mask_for_credit,
    )
    assert credit_info["adv_triplet_credit_triplet_updates"] > 0
    assert credit_info["adv_triplet_credit_seen_ratio"] > 0.0
    assert "adv_triplet_marginal_positive_fraction" in credit_info
    graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    assert np.isfinite(graph.last_adv_triplet_score_multiplier_mean)
    assert graph.last_adv_triplet_score_multiplier_mean > 0.0
    assert np.isfinite(graph.last_adv_triplet_score_multiplier_min)
    assert np.isfinite(graph.last_adv_triplet_score_multiplier_max)
    assert graph.last_adv_triplet_score_multiplier_min > 0.0
    assert graph.last_adv_triplet_score_multiplier_max > 0.0
    assert np.isfinite(graph.last_adv_triplet_score_positive_fraction)
    assert np.isfinite(graph.last_adv_triplet_negative_scaled_fraction)
    assert bool(args.use_adj_triplet_graph_return_credit)
    assert args.adj_triplet_graph_return_credit_coef > 0.0
    assert args.adj_triplet_graph_return_credit_cap > 0.0
    assert args.adj_triplet_graph_return_credit_raw_gate_scale > 0.0
    assert bool(args.adj_triplet_graph_return_credit_require_delayed_gate)
    assert bool(args.use_adj_delayed_triplet_credit)
    assert args.adj_delayed_triplet_credit_coef > 0.0
    assert args.adj_delayed_triplet_credit_window > 0
    assert bool(args.adj_delayed_triplet_credit_positive_only)
    assert args.adj_delayed_triplet_credit_min_adv > 0.0
    assert bool(args.adj_delayed_triplet_credit_require_future_match)
    assert bool(args.use_adj_delayed_triplet_success_gate)
    assert args.adj_delayed_triplet_success_gate_min_adv > 0.0
    assert args.adj_delayed_triplet_success_gate_scale > 0.0
    assert args.adj_delayed_triplet_success_gate_floor > 0.0
    assert args.adj_delayed_triplet_future_overlap_min_nodes == 2
    assert args.adj_delayed_triplet_partial_match_weight > 0.0
    assert bool(args.use_adj_capture_to_win_credit)
    assert args.adj_capture_to_win_credit_coef > 0.0
    assert args.adj_capture_to_win_credit_min_outcome_adv > 0.0
    assert args.adj_capture_to_win_credit_scale > 0.0
    assert args.adj_capture_to_win_credit_cap > 0.0
    assert bool(args.adj_capture_to_win_credit_require_future_match)
    assert bool(args.use_adj_pair_triplet_complementary_credit)
    assert args.adj_pair_pursuit_credit_coef > 0.0
    assert args.adj_pair_pursuit_credit_window > 0
    assert args.adj_pair_pursuit_credit_cap > 0.0
    assert args.adj_recent_episode_window_min == 1
    assert args.adj_recent_episode_window_emergency == 1
    assert args.adj_recent_window_emergency_stale_threshold > 0.0
    assert args.adj_recent_window_emergency_factor_stale_threshold > 0.0

    selected_float = adj.float()
    loss = -(
        torch.log(target_prob.clamp_min(1e-8)) * selected_float
    ).sum() / selected_float.sum().clamp_min(1.0)
    loss.backward()

    finite_grad_count = 0
    for parameter in graph.parameters():
        if parameter.grad is None:
            continue
        assert bool(torch.isfinite(parameter.grad).all().item())
        finite_grad_count += 1
    assert finite_grad_count > 0
    gate_after_bad_credit = graph.update_order3_credit_gate(
        0.004,
        0.0,
    )
    assert gate_after_bad_credit < 1.0
    graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    assert graph.current_min_order3_ratio < 0.72
    assert graph.current_greedy_sample_prob < 0.75

    validate_outcome_replay_support()
    validate_adj_buffer()
    validate_scatter_gradient()
    deterministic_cuda_scatter = validate_deterministic_cuda_scatter()
    validate_factor_q_gradient()
    validate_trainer_without_critics()
    validate_full_policy_gradient()
    deterministic_cuda_full_policy = (
        validate_deterministic_cuda_full_policy_gradient()
    )

    print("SDDFG dynamic graph validation passed")
    print("active_counts={}".format(active_counts))
    print("order2_factors={}".format(order_counts[2]))
    print("order3_factors={}".format(order_counts[3]))
    print(
        "current_min_order3_ratio={}".format(
            graph.current_min_order3_ratio
        )
    )
    print(
        "current_max_order3_ratio={}".format(
            graph.current_max_order3_ratio
        )
    )
    print(
        "current_greedy_sample_prob={}".format(
            graph.current_greedy_sample_prob
        )
    )
    print(
        "greedy_sample_prob_cap={}".format(
            graph.greedy_sample_prob_cap
        )
    )
    print("order3_quota_mode={}".format(graph.order3_quota_mode))
    print(
        "order3_soft_quota_coef={}".format(
            graph.order3_soft_quota_coef
        )
    )
    print(
        "triplet_balance_coef={}".format(
            graph.triplet_balance_coef
        )
    )
    print("triplet_feature_mode={}".format(graph.triplet_feature_mode))
    print(
        "use_advantage_triplet_scorer={}".format(
            graph.use_advantage_triplet_scorer
        )
    )
    print(
        "triplet_credit_score_coef={}".format(
            graph.triplet_credit_score_coef
        )
    )
    print(
        "triplet_credit_score_scale={}".format(
            graph.triplet_credit_score_scale
        )
    )
    print(
        "use_triplet_credit_direct_rank={}".format(
            graph.use_triplet_credit_direct_rank
        )
    )
    print(
        "triplet_credit_rank_coef={}".format(
            graph.triplet_credit_rank_coef
        )
    )
    print(
        "triplet_credit_multiplier_bounds=[{},{}]".format(
            graph.triplet_credit_min_multiplier,
            graph.triplet_credit_max_multiplier,
        )
    )
    print(
        "triplet_credit_negative_rank_scale={}".format(
            graph.triplet_credit_negative_rank_scale
        )
    )
    print(
        "triplet_credit_min_positive_fraction={}".format(
            graph.triplet_credit_min_positive_fraction
        )
    )
    print(
        "adv_triplet_credit_seen_ratio={}".format(
            credit_info["adv_triplet_credit_seen_ratio"]
        )
    )
    print(
        "adv_triplet_marginal_positive_fraction={}".format(
            credit_info["adv_triplet_marginal_positive_fraction"]
        )
    )
    print(
        "adv_triplet_score_multiplier_mean={}".format(
            graph.last_adv_triplet_score_multiplier_mean
        )
    )
    print(
        "adv_triplet_score_multiplier_min={}".format(
            graph.last_adv_triplet_score_multiplier_min
        )
    )
    print(
        "adv_triplet_score_multiplier_max={}".format(
            graph.last_adv_triplet_score_multiplier_max
        )
    )
    print(
        "adv_triplet_score_positive_fraction={}".format(
            graph.last_adv_triplet_score_positive_fraction
        )
    )
    print(
        "adv_triplet_negative_scaled_fraction={}".format(
            graph.last_adv_triplet_negative_scaled_fraction
        )
    )
    print(
        "adj_order_adv_positive_only={}".format(
            args.adj_order_adv_positive_only
        )
    )
    print(
        "adj_order_adv_negative_coef={}".format(
            args.adj_order_adv_negative_coef
        )
    )
    print(
        "adj_order_adv_require_positive_graph_adv={}".format(
            args.adj_order_adv_require_positive_graph_adv
        )
    )
    print(
        "use_adj_triplet_graph_return_credit={}".format(
            args.use_adj_triplet_graph_return_credit
        )
    )
    print(
        "adj_triplet_graph_return_credit_coef={}".format(
            args.adj_triplet_graph_return_credit_coef
        )
    )
    print(
        "adj_triplet_graph_return_credit_cap={}".format(
            args.adj_triplet_graph_return_credit_cap
        )
    )
    print(
        "adj_triplet_graph_return_credit_min_graph_adv={}".format(
            args.adj_triplet_graph_return_credit_min_graph_adv
        )
    )
    print(
        "adj_triplet_graph_return_credit_raw_gate_scale={}".format(
            args.adj_triplet_graph_return_credit_raw_gate_scale
        )
    )
    print(
        "adj_triplet_graph_return_credit_require_delayed_gate={}".format(
            args.adj_triplet_graph_return_credit_require_delayed_gate
        )
    )
    print(
        "use_adj_delayed_triplet_credit={}".format(
            args.use_adj_delayed_triplet_credit
        )
    )
    print(
        "adj_delayed_triplet_credit_coef={}".format(
            args.adj_delayed_triplet_credit_coef
        )
    )
    print(
        "adj_delayed_triplet_credit_window={}".format(
            args.adj_delayed_triplet_credit_window
        )
    )
    print(
        "adj_delayed_triplet_credit_cap={}".format(
            args.adj_delayed_triplet_credit_cap
        )
    )
    print(
        "adj_delayed_triplet_credit_min_reward={}".format(
            args.adj_delayed_triplet_credit_min_reward
        )
    )
    print(
        "adj_delayed_triplet_credit_positive_only={}".format(
            args.adj_delayed_triplet_credit_positive_only
        )
    )
    print(
        "adj_delayed_triplet_credit_min_adv={}".format(
            args.adj_delayed_triplet_credit_min_adv
        )
    )
    print(
        "adj_delayed_triplet_credit_require_future_match={}".format(
            args.adj_delayed_triplet_credit_require_future_match
        )
    )
    print(
        "use_adj_delayed_triplet_success_gate={}".format(
            args.use_adj_delayed_triplet_success_gate
        )
    )
    print(
        "adj_delayed_triplet_success_gate_min_adv={}".format(
            args.adj_delayed_triplet_success_gate_min_adv
        )
    )
    print(
        "adj_delayed_triplet_success_gate_scale={}".format(
            args.adj_delayed_triplet_success_gate_scale
        )
    )
    print(
        "adj_delayed_triplet_success_gate_floor={}".format(
            args.adj_delayed_triplet_success_gate_floor
        )
    )
    print(
        "adj_delayed_triplet_future_overlap_min_nodes={}".format(
            args.adj_delayed_triplet_future_overlap_min_nodes
        )
    )
    print(
        "adj_delayed_triplet_partial_match_weight={}".format(
            args.adj_delayed_triplet_partial_match_weight
        )
    )
    print(
        "use_adj_capture_to_win_credit={}".format(
            args.use_adj_capture_to_win_credit
        )
    )
    print(
        "adj_capture_to_win_credit_coef={}".format(
            args.adj_capture_to_win_credit_coef
        )
    )
    print(
        "adj_capture_to_win_credit_min_outcome_adv={}".format(
            args.adj_capture_to_win_credit_min_outcome_adv
        )
    )
    print(
        "adj_capture_to_win_credit_scale={}".format(
            args.adj_capture_to_win_credit_scale
        )
    )
    print(
        "adj_capture_to_win_credit_cap={}".format(
            args.adj_capture_to_win_credit_cap
        )
    )
    print(
        "adj_capture_to_win_credit_require_future_match={}".format(
            args.adj_capture_to_win_credit_require_future_match
        )
    )
    print(
        "use_adj_pair_triplet_complementary_credit={}".format(
            args.use_adj_pair_triplet_complementary_credit
        )
    )
    print(
        "adj_pair_pursuit_credit_coef={}".format(
            args.adj_pair_pursuit_credit_coef
        )
    )
    print(
        "adj_pair_pursuit_credit_window={}".format(
            args.adj_pair_pursuit_credit_window
        )
    )
    print(
        "adj_pair_pursuit_credit_cap={}".format(
            args.adj_pair_pursuit_credit_cap
        )
    )
    print(
        "adj_pair_pursuit_credit_min_reward={}".format(
            args.adj_pair_pursuit_credit_min_reward
        )
    )
    print(
        "current_order3_credit_gate={}".format(
            graph.current_order3_credit_gate
        )
    )
    print(
        "order3_credit_loss_ema={}".format(
            graph.order3_credit_loss_ema
        )
    )
    print(
        "order3_credit_margin_ema={}".format(
            graph.order3_credit_margin_ema
        )
    )
    print(
        "relative_order3_credit_gate={}".format(
            graph.use_relative_order3_credit_gate
        )
    )
    print(
        "order3_credit_gate_max_delta={}".format(
            graph.order3_credit_gate_max_delta
        )
    )
    print("adj_ppo_clip_stop_ratio={}".format(args.adj_ppo_clip_stop_ratio))
    print(
        "adj_ppo_factor_clip_stop_ratio={}".format(
            args.adj_ppo_factor_clip_stop_ratio
        )
    )
    print("adj_ppo_min_epochs={}".format(args.adj_ppo_min_epochs))
    print("use_adj_ppo_stale_trust={}".format(args.use_adj_ppo_stale_trust))
    print(
        "adj_ppo_stale_trust_clip={}".format(
            args.adj_ppo_stale_trust_clip
        )
    )
    print(
        "adj_ppo_stale_trust_scale={}".format(
            args.adj_ppo_stale_trust_scale
        )
    )
    print(
        "adj_ppo_stale_trust_min_weight={}".format(
            args.adj_ppo_stale_trust_min_weight
        )
    )
    print(
        "adj_recent_episode_window={}".format(
            args.adj_recent_episode_window
        )
    )
    print(
        "use_adj_dynamic_recent_window={}".format(
            args.use_adj_dynamic_recent_window
        )
    )
    print(
        "adj_recent_episode_window_min={}".format(
            args.adj_recent_episode_window_min
        )
    )
    print(
        "adj_recent_window_stale_threshold={}".format(
            args.adj_recent_window_stale_threshold
        )
    )
    print(
        "adj_recent_window_factor_stale_threshold={}".format(
            args.adj_recent_window_factor_stale_threshold
        )
    )
    print(
        "adj_recent_window_shrink_patience={}".format(
            args.adj_recent_window_shrink_patience
        )
    )
    print(
        "adj_recent_window_recover_patience={}".format(
            args.adj_recent_window_recover_patience
        )
    )
    print(
        "adj_recent_window_recover_stale_threshold={}".format(
            args.adj_recent_window_recover_stale_threshold
        )
    )
    print(
        "adj_recent_window_recover_factor_stale_threshold={}".format(
            args.adj_recent_window_recover_factor_stale_threshold
        )
    )
    print(
        "adj_recent_window_severe_margin={}".format(
            args.adj_recent_window_severe_margin
        )
    )
    print(
        "adj_recent_episode_window_emergency={}".format(
            args.adj_recent_episode_window_emergency
        )
    )
    print(
        "adj_recent_window_emergency_stale_threshold={}".format(
            args.adj_recent_window_emergency_stale_threshold
        )
    )
    print(
        "adj_recent_window_emergency_factor_stale_threshold={}".format(
            args.adj_recent_window_emergency_factor_stale_threshold
        )
    )
    print("finite_grad_tensors={}".format(finite_grad_count))
    print("adj_buffer_axis_and_return_test=passed")
    print("outcome_replay_support_test=passed")
    print("capture_outcome_factor_loss_test=passed")
    print("capture_candidate_identity_loss_test=passed")
    print("candidate_only_replay_path_test=passed")
    print("isolated_eval_graph_rng_test=passed")
    print("exact_markov_topology_persistence_test=passed")
    print("scatter_gradient_test=passed")
    print(
        "deterministic_cuda_scatter_test={}".format(
            "passed" if deterministic_cuda_scatter else "skipped"
        )
    )
    print("factor_q_gradient_test=passed")
    print("trainer_without_critics_test=passed")
    print("full_policy_gradient_test=passed")
    print(
        "deterministic_cuda_full_policy_gradient_test={}".format(
            "passed" if deterministic_cuda_full_policy else "skipped"
        )
    )


if __name__ == "__main__":
    main()
