#!/usr/bin/env python
"""Minimal synthetic checks for capture-anchored pair credit."""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pair_credit import (
    compute_capture_anchored_pair_credit,
    compute_capture_to_win_outcome_gate,
    compute_capture_to_win_triplet_gate,
)


def _trajectory(time_steps=5, num_agents=4, num_factors=3):
    adj = np.zeros(
        (time_steps, 1, num_agents, num_factors),
        dtype=np.int64,
    )
    rewards = np.zeros((time_steps, 1), dtype=np.float32)
    captures = np.zeros((time_steps, 1), dtype=np.float32)
    dones = np.zeros((time_steps, 1), dtype=np.float32)
    return adj, rewards, captures, dones


def _set_factor(adj, step, factor, nodes):
    adj[step, 0, :, factor] = 0
    adj[step, 0, list(nodes), factor] = 1


def _compute(adj, rewards, captures, dones, window=4):
    factor_size = adj.sum(axis=2)
    valid_factor = factor_size > 0
    valid_graph = valid_factor.any(axis=-1)
    return compute_capture_anchored_pair_credit(
        current_adj=adj,
        valid_factor=valid_factor,
        factor_size=factor_size,
        capture_counts=captures,
        capture_factor_match=(
            (factor_size == 3).astype(np.float32)
            * captures[:, :, None]
        ),
        valid_graph_transition=valid_graph,
        dones_env=dones,
        team_rewards=rewards,
        window=window,
        gamma=0.97,
    )


def test_positive_shaping_without_capture_is_zero():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    rewards[2, 0] = 0.05
    result = _compute(adj, rewards, captures, dones)
    assert result["pair_to_triplet_transition_score"].sum() == 0.0
    assert result["pair_pursuit_quality"][0, 0, 0] > 0.0
    assert result["positive_reward_without_capture"].sum() == 1.0


def test_offset0_pair_triplet_capture_is_excluded():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 0, 1, (0, 1, 2))
    captures[0, 0] = 1.0
    rewards[0, 0] = 2.0
    result = _compute(adj, rewards, captures, dones)
    assert result["pair_to_triplet_transition_score"].sum() == 0.0
    assert result["offset0_candidate_count"].sum() == 1.0


def test_strict_future_matching_capture_triplet_is_positive():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    captures[2, 0] = 1.0
    rewards[2, 0] = 2.0
    result = _compute(adj, rewards, captures, dones)
    score = result["pair_to_triplet_transition_score"][0, 0, 0]
    delay = result["pair_transition_delay"][0, 0, 0]
    assert score > 0.0
    assert delay == 2.0
    assert result["pair_pursuit_quality"][0, 0, 0] != score
    assert result["triplet_capture_quality"][2, 0, 1] == 1.0
    assert result["capture_matched_count"].sum() == 1.0


def test_capture_triplet_without_pair_is_zero():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 2, 3))
    captures[2, 0] = 1.0
    rewards[2, 0] = 2.0
    result = _compute(adj, rewards, captures, dones)
    assert result["pair_to_triplet_transition_score"].sum() == 0.0
    assert result["capture_matched_count"].sum() == 0.0


def test_only_nearest_valid_transition_is_credited():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    _set_factor(adj, 3, 1, (0, 1, 3))
    captures[2, 0] = 1.0
    captures[3, 0] = 1.0
    rewards[2:4, 0] = 2.0
    result = _compute(adj, rewards, captures, dones)
    score = result["pair_to_triplet_transition_score"][0, 0, 0]
    assert np.isclose(score, 0.97 ** 2)
    assert result["pair_transition_delay"][0, 0, 0] == 2.0
    assert result["capture_matched_count"][2, 0] == 1.0
    assert result["capture_matched_count"][3, 0] == 0.0


def test_failed_high_return_capture_does_not_pass_outcome_gate():
    # Rewards are deliberately omitted: no shaped-return value is allowed to
    # turn a failed episode into a successful capture-to-win outcome.
    valid = np.ones((4, 2), dtype=bool)
    captures = np.zeros((4, 2), dtype=np.float32)
    success = np.zeros((4, 2), dtype=np.float32)
    captures[1, 0] = 2.0
    captures[2, 1] = 1.0
    success[3, 1] = 1.0
    result = compute_capture_to_win_outcome_gate(
        success_now=success,
        capture_counts=captures,
        valid_graph_transition=valid,
    )
    assert not result["episode_success"][0]
    assert result["episode_success"][1]
    assert result["episode_success_gate"][:, 0].sum() == 0.0
    assert result["episode_success_gate"][:, 1].sum() == 4.0
    assert result["failed_episode_capture_count"][:, 0].sum() == 2.0
    assert result["failed_episode_capture_count"][:, 1].sum() == 0.0
    triplet_quality = np.zeros((4, 2, 3), dtype=np.float32)
    triplet_quality[1, 0, 0] = 1.0
    triplet_quality[2, 1, 1] = 1.0
    triplet_gate = compute_capture_to_win_triplet_gate(
        episode_success=result["episode_success"],
        triplet_capture_quality=triplet_quality,
    )
    assert triplet_gate[1, 0, 0] == 0.0
    assert triplet_gate[2, 1, 1] == 1.0
    assert triplet_gate.sum() == 1.0


def main():
    tests = [
        test_positive_shaping_without_capture_is_zero,
        test_offset0_pair_triplet_capture_is_excluded,
        test_strict_future_matching_capture_triplet_is_positive,
        test_capture_triplet_without_pair_is_zero,
        test_only_nearest_valid_transition_is_credited,
        test_failed_high_return_capture_does_not_pass_outcome_gate,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} pair-credit synthetic tests".format(len(tests)))


if __name__ == "__main__":
    main()
