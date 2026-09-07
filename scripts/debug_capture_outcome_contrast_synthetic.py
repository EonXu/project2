#!/usr/bin/env python
"""Synthetic checks for signed capture-to-win outcome credit and timing."""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pair_credit import (
    compute_capture_anchored_pair_credit,
    compute_outcome_conditioned_pair_credit,
    compute_capture_to_win_outcome_gate,
    compute_capture_to_win_triplet_outcome_advantage,
    scale_capture_to_win_outcome_credit,
)


def _quality(time_steps=4, episodes=4, factors=3):
    return np.zeros((time_steps, episodes, factors), dtype=np.float32)


def test_success_and_failure_capture_episodes_are_centered():
    quality = _quality()
    for episode in range(4):
        quality[1, episode, episode % 3] = 1.0
    result = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([1, 0, 0, 1], dtype=bool),
        triplet_capture_quality=quality,
    )
    advantage = result["triplet_outcome_advantage"]
    assert np.isclose(result["capture_episode_success_rate"], 0.5)
    assert np.isclose(advantage[1, 0, 0], 0.5)
    assert np.isclose(advantage[1, 1, 1], -0.5)
    assert np.isclose(advantage[1, 2, 2], -0.5)
    assert np.isclose(advantage[1, 3, 0], 0.5)
    assert np.isclose(advantage.sum(), 0.0)


def test_single_outcome_class_does_not_invent_contrast():
    quality = _quality(episodes=3)
    quality[1, :, 0] = 1.0
    failed = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.zeros(3, dtype=bool),
        triplet_capture_quality=quality,
    )
    successful = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.ones(3, dtype=bool),
        triplet_capture_quality=quality,
    )
    assert failed["triplet_outcome_advantage"].sum() == 0.0
    assert successful["triplet_outcome_advantage"].sum() == 0.0


def test_non_capture_success_and_padding_do_not_change_baseline():
    quality = _quality(episodes=4)
    quality[1, 0, 0] = 1.0
    quality[1, 1, 0] = 1.0
    quality[3, 3, 2] = np.nan
    result = compute_capture_to_win_triplet_outcome_advantage(
        # Episode 2 succeeds but has no capture triplet and must not enter the
        # capture-episode baseline. Episode 3 is padding/non-finite input.
        episode_success=np.array([1, 0, 1, 0], dtype=bool),
        triplet_capture_quality=quality,
    )
    advantage = result["triplet_outcome_advantage"]
    assert np.isclose(result["capture_episode_success_rate"], 0.5)
    assert np.isclose(advantage[1, 0, 0], 0.5)
    assert np.isclose(advantage[1, 1, 0], -0.5)
    assert np.count_nonzero(advantage[:, 2:]) == 0
    assert np.isfinite(advantage).all()


def _timeline_inputs(time_steps=4, episodes=2):
    adj = np.zeros((time_steps, episodes, 4, 3), dtype=np.int64)
    rewards = np.zeros((time_steps, episodes), dtype=np.float32)
    captures = np.zeros_like(rewards)
    dones = np.zeros_like(rewards)
    return adj, rewards, captures, dones


def _set_factor(adj, step, episode, factor, nodes):
    adj[step, episode, :, factor] = 0
    adj[step, episode, list(nodes), factor] = 1


def _pair_credit(adj, rewards, captures, dones):
    factor_size = adj.sum(axis=2)
    valid_factor = factor_size > 0
    return compute_capture_anchored_pair_credit(
        current_adj=adj,
        valid_factor=valid_factor,
        factor_size=factor_size,
        active_agent_count=np.full(captures.shape, 4, dtype=np.int64),
        selected_factor_behavior_probability=np.where(
            valid_factor,
            np.float32(0.5),
            np.float32(0.0),
        ),
        capture_counts=captures,
        capture_factor_match=(
            (factor_size == 3).astype(np.float32)
            * captures[:, :, None]
        ),
        valid_graph_transition=valid_factor.any(axis=-1),
        dones_env=dones,
        team_rewards=rewards,
        window=3,
        gamma=0.97,
    )


def test_post_action_event_is_stored_on_same_transition_index():
    # Transition t=0: graph/action uses pair (0,1), then env.step returns
    # reward[0] and no capture[0]. Transition t=1 uses a triplet and env.step
    # returns capture[1]. The strict-future pair transition delay is exactly 1.
    adj, rewards, captures, dones = _timeline_inputs()
    for episode in range(2):
        _set_factor(adj, 0, episode, 0, (0, 1))
        _set_factor(adj, 1, episode, 1, (0, 1, 2))
        rewards[1, episode] = 2.0
    captures[1, 0] = 1.0
    result = _pair_credit(adj, rewards, captures, dones)
    assert result["pair_transition_delay"][0, 0, 0] == 1.0
    assert result["pair_to_triplet_transition_score"][0, 0, 0] > 0.0
    # The other environment has identical reward/graphs but no capture event.
    assert result["pair_to_triplet_transition_score"][0, 1, 0] == 0.0
    assert result["positive_reward_without_capture"][1, 1] == 1.0


def test_terminal_boundary_blocks_future_capture():
    adj, rewards, captures, dones = _timeline_inputs(episodes=2)
    _set_factor(adj, 0, 0, 0, (0, 1))
    _set_factor(adj, 1, 0, 1, (0, 1, 2))
    dones[0, 0] = 1.0
    captures[1, 0] = 1.0
    result = _pair_credit(adj, rewards, captures, dones)
    assert result["pair_to_triplet_transition_score"][0, 0, 0] == 0.0
    assert result["capture_matched_count"].sum() == 0.0


def test_success_event_mask_is_episode_local_and_padding_safe():
    valid = np.array(
        [[1, 1], [1, 1], [0, 1], [0, 0]],
        dtype=bool,
    )
    success = np.zeros((4, 2), dtype=np.float32)
    captures = np.zeros_like(success)
    success[2, 0] = 1.0  # invalid padded step: must be ignored
    success[2, 1] = 1.0
    captures[1, 0] = 1.0
    captures[1, 1] = 1.0
    result = compute_capture_to_win_outcome_gate(
        success_now=success,
        capture_counts=captures,
        valid_graph_transition=valid,
    )
    assert not result["episode_success"][0]
    assert result["episode_success"][1]
    assert result["failed_episode_capture_count"][:, 0].sum() == 1.0
    assert result["failed_episode_capture_count"][:, 1].sum() == 0.0


def test_signed_log_denominators_use_capture_triplets_only():
    quality = _quality(time_steps=2, episodes=2, factors=3)
    quality[0, 0, 0] = 1.0
    quality[0, 1, 1] = 1.0
    result = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([1, 0], dtype=bool),
        triplet_capture_quality=quality,
    )["triplet_outcome_advantage"]
    valid_capture_triplets = quality > 0.0
    denominator = int(valid_capture_triplets.sum())
    positive_fraction = int((result > 0.0).sum()) / denominator
    negative_fraction = int((result < 0.0).sum()) / denominator
    active_fraction = int((np.abs(result) > 0.0).sum()) / denominator
    assert positive_fraction == 0.5
    assert negative_fraction == 0.5
    assert active_fraction == 1.0


def test_signed_credit_scaling_preserves_failed_capture_branch():
    outcome = np.zeros((2, 2, 2), dtype=np.float32)
    outcome[0, 0, 0] = 0.5
    outcome[0, 1, 1] = -0.5
    graph_advantage = np.ones((2, 2), dtype=np.float32)
    credit = scale_capture_to_win_outcome_credit(
        triplet_outcome_advantage=outcome,
        graph_advantage=graph_advantage,
        valid_graph_transition=np.ones((2, 2), dtype=bool),
        coefficient=0.15,
        cap=0.25,
    )
    assert credit[0, 0, 0] > 0.0
    assert credit[0, 1, 1] < 0.0
    assert np.isclose(abs(credit[0, 0, 0]), abs(credit[0, 1, 1]))


def test_episode_outcome_is_normalized_across_all_capture_triplets():
    quality = _quality(time_steps=4, episodes=2, factors=3)
    # The successful episode has two captures and five selected triplet
    # labels; the failed episode has one capture and one label.
    quality[0, 0, 0:3] = 1.0
    quality[2, 0, 0:2] = 1.0
    quality[1, 1, 2] = 1.0
    result = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([1, 0], dtype=bool),
        triplet_capture_quality=quality,
    )
    advantage = result["triplet_outcome_advantage"]
    assert np.array_equal(
        result["capture_triplet_count_per_episode"],
        np.array([5.0, 1.0], dtype=np.float32),
    )
    assert np.isclose(advantage[:, 0, :].sum(), 0.5)
    assert np.isclose(advantage[:, 1, :].sum(), -0.5)
    assert np.isclose(advantage.sum(), 0.0)
    assert np.isclose(np.abs(advantage[:, 0, :]).sum(), 0.5)
    assert np.isclose(np.abs(advantage[:, 1, :]).sum(), 0.5)


def test_pair_credit_is_outcome_centered_and_episode_normalized():
    score = np.zeros((3, 2, 3), dtype=np.float32)
    # Successful episode: five strict-future pair labels with uneven scores.
    score[0, 0, 0:3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    score[2, 0, 0:2] = np.array([4.0, 5.0], dtype=np.float32)
    # Failed episode: one eligible label. Each episode must still contribute
    # one centered terminal-outcome label rather than one label per factor.
    score[1, 1, 2] = 2.0
    result = compute_outcome_conditioned_pair_credit(
        pair_transition_score=score,
        episode_success=np.array([1, 0], dtype=bool),
    )
    credit = result["credit"]
    assert np.isclose(credit[:, 0, :].sum(), 0.5)
    assert np.isclose(credit[:, 1, :].sum(), -0.5)
    assert np.isclose(credit.sum(), 0.0)
    assert np.all(credit[score == 0.0] == 0.0)
    assert result["evidence_episode_count"] == 2


def test_pair_credit_single_outcome_class_is_zero():
    score = np.zeros((2, 3, 2), dtype=np.float32)
    score[0, :, 0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for episode_success in (
            np.zeros(3, dtype=bool),
            np.ones(3, dtype=bool)):
        result = compute_outcome_conditioned_pair_credit(
            pair_transition_score=score,
            episode_success=episode_success,
        )
        assert np.count_nonzero(result["credit"]) == 0
        assert result["centered_sum"] == 0.0


def test_pair_credit_rejects_invalid_support_scores():
    score = np.zeros((2, 2, 2), dtype=np.float32)
    score[0, 0, 0] = -1.0
    try:
        compute_outcome_conditioned_pair_credit(
            pair_transition_score=score,
            episode_success=np.array([1, 0], dtype=bool),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative pair evidence must fail loudly")


def main():
    tests = [
        test_success_and_failure_capture_episodes_are_centered,
        test_single_outcome_class_does_not_invent_contrast,
        test_non_capture_success_and_padding_do_not_change_baseline,
        test_post_action_event_is_stored_on_same_transition_index,
        test_terminal_boundary_blocks_future_capture,
        test_success_event_mask_is_episode_local_and_padding_safe,
        test_signed_log_denominators_use_capture_triplets_only,
        test_signed_credit_scaling_preserves_failed_capture_branch,
        test_episode_outcome_is_normalized_across_all_capture_triplets,
        test_pair_credit_is_outcome_centered_and_episode_normalized,
        test_pair_credit_single_outcome_class_is_zero,
        test_pair_credit_rejects_invalid_support_scores,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} capture-outcome contrast tests".format(len(tests)))


if __name__ == "__main__":
    main()
