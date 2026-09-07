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


def _compute(
        adj,
        rewards,
        captures,
        dones,
        window=4,
        event_provenance=None,
        active_agent_count=None,
        selected_factor_behavior_probability=None):
    factor_size = adj.sum(axis=2)
    valid_factor = factor_size > 0
    valid_graph = valid_factor.any(axis=-1)
    if active_agent_count is None:
        active_agent_count = np.full(
            captures.shape,
            adj.shape[2],
            dtype=np.int64,
        )
    if selected_factor_behavior_probability is None:
        selected_factor_behavior_probability = np.where(
            valid_factor,
            np.float32(0.5),
            np.float32(0.0),
        )
    return compute_capture_anchored_pair_credit(
        current_adj=adj,
        valid_factor=valid_factor,
        factor_size=factor_size,
        active_agent_count=active_agent_count,
        selected_factor_behavior_probability=(
            selected_factor_behavior_probability
        ),
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
        capture_event_provenance_by_episode=event_provenance,
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


def test_terminal_future_capture_is_not_lost_off_by_one():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 1, 0, (0, 1))
    _set_factor(adj, 4, 1, (0, 1, 2))
    captures[4, 0] = 1.0
    dones[4, 0] = 1.0
    result = _compute(adj, rewards, captures, dones, window=4)
    assert result["pair_to_triplet_transition_score"][1, 0, 0] > 0.0
    assert result["pair_transition_delay"][1, 0, 0] == 3.0
    assert result["capture_matched_count"][4, 0] == 1.0

    # Ending the episode before the capture transition must still block it.
    dones[3, 0] = 1.0
    result = _compute(adj, rewards, captures, dones, window=4)
    assert result["pair_to_triplet_transition_score"].sum() == 0.0


def test_strict_pair_provenance_joins_exact_capture_event():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    captures[2, 0] = 1.0
    provenance = [[{
        "environment_episode_id": 91,
        "event_id": 7,
        "target_id": 3,
        "participant_slots": (0, 1, 2),
        "factor_index": 1,
        "factor_identity": "0-1-2",
        "factor_order": 3,
        "identity_event_weight": 1.0,
        "factor_slot_weight": 1.0,
        "capture_step": 2,
        "static_dynamic_class": "dynamic",
    }]]
    result = _compute(
        adj,
        rewards,
        captures,
        dones,
        event_provenance=provenance,
    )
    rows = result["strict_pair_event_provenance"][0]
    assert len(rows) == 1
    row = rows[0]
    assert row["environment_episode_id"] == 91
    assert row["event_id"] == 7
    assert row["target_id"] == 3
    assert row["participant_slots"] == (0, 1, 2)
    assert row["pair_transition_step"] == 0
    assert row["capture_step"] == 2
    assert row["pair_identity"] == "0-1"
    assert row["capture_factor_identity"] == "0-1-2"


def test_strict_pair_provenance_keeps_simultaneous_capture_events_local():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    captures[2, 0] = 2.0
    provenance = [[
        {
            "environment_episode_id": 91,
            "event_id": event_id,
            "target_id": target_id,
            "participant_slots": (0, 1, 2),
            "factor_index": 1,
            "factor_identity": "0-1-2",
            "factor_order": 3,
            "identity_event_weight": 1.0,
            "factor_slot_weight": 1.0,
            "capture_step": 2,
            "static_dynamic_class": "static",
        }
        for event_id, target_id in ((10, 0), (11, 1))
    ]]
    result = _compute(
        adj,
        rewards,
        captures,
        dones,
        event_provenance=provenance,
    )
    rows = result["strict_pair_event_provenance"][0]
    assert len(rows) == 2
    assert [row["event_id"] for row in rows] == [10, 11]
    assert [row["target_id"] for row in rows] == [0, 1]
    assert all(row["pair_transition_step"] == 0 for row in rows)
    assert all(row["capture_step"] == 2 for row in rows)
    # Dense pair credit is still written once for the transition; only the
    # immutable event-local provenance population has two real events.
    assert np.count_nonzero(
        result["pair_to_triplet_transition_score"]
    ) == 1
    assert result["capture_matched_count"][2, 0] == 2.0


def test_strict_pair_provenance_rejects_one_event_identity_drift():
    adj, rewards, captures, dones = _trajectory()
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    captures[2, 0] = 1.0
    provenance = [[
        {
            "environment_episode_id": 91,
            "event_id": 7,
            "target_id": 3,
            "participant_slots": participants,
            "factor_index": 1,
            "factor_identity": "0-1-2",
            "factor_order": 3,
            "identity_event_weight": 1.0,
            "factor_slot_weight": 0.5,
            "capture_step": 2,
            "static_dynamic_class": "static",
        }
        for participants in ((0, 1, 2), (0, 1, 2, 3))
    ]]
    try:
        _compute(
            adj,
            rewards,
            captures,
            dones,
            event_provenance=provenance,
        )
    except RuntimeError as error:
        assert "within one event identity" in str(error)
    else:
        raise AssertionError("one capture event changed identity silently")


def test_forced_two_agent_pair_is_not_graph_selection_evidence():
    adj, rewards, captures, dones = _trajectory(num_agents=4)
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    captures[2, 0] = 1.0
    rewards[2, 0] = 2.0
    active_agent_count = np.full(captures.shape, 3, dtype=np.int64)
    active_agent_count[0, 0] = 2
    result = _compute(
        adj,
        rewards,
        captures,
        dones,
        active_agent_count=active_agent_count,
    )
    assert result["structural_pair_only_mask"][0, 0, 0]
    assert result["forced_pair_non_actionable_mask"][0, 0, 0]
    assert not result["pair_only_mask"][0, 0, 0]
    assert not result["pair_selection_actionable_transition"][0, 0]
    assert result["pair_to_triplet_transition_score"].sum() == 0.0
    assert result["pair_pursuit_quality"].sum() == 0.0
    assert result["strict_pair_event_provenance"][0] == ()
    # The real capture remains present in the behavior diagnostics; only the
    # impossible graph-selection supervision is excluded.
    assert result["capture_factor_quality"][2, 0, 1] == 1.0


def test_constraint_forced_pair_is_not_graph_selection_evidence():
    adj, rewards, captures, dones = _trajectory(num_agents=4)
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 2, 1, (0, 1, 2))
    captures[2, 0] = 1.0
    factor_size = adj.sum(axis=2)
    behavior_probability = np.where(
        factor_size > 0,
        np.float32(0.5),
        np.float32(0.0),
    )
    # Four agents are alive, but the real sequential selector reports that
    # coverage/quota/order constraints left exactly one reachable factor.
    behavior_probability[0, 0, 0] = 1.0
    result = _compute(
        adj,
        rewards,
        captures,
        dones,
        selected_factor_behavior_probability=behavior_probability,
    )
    assert result["structural_pair_only_mask"][0, 0, 0]
    assert result["forced_pair_non_actionable_mask"][0, 0, 0]
    assert not result["pair_only_mask"][0, 0, 0]
    assert not result["pair_selection_actionable_factor"][0, 0, 0]
    assert result["pair_to_triplet_transition_score"].sum() == 0.0


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
        test_terminal_future_capture_is_not_lost_off_by_one,
        test_strict_pair_provenance_joins_exact_capture_event,
        test_strict_pair_provenance_keeps_simultaneous_capture_events_local,
        test_strict_pair_provenance_rejects_one_event_identity_drift,
        test_forced_two_agent_pair_is_not_graph_selection_evidence,
        test_constraint_forced_pair_is_not_graph_selection_evidence,
        test_failed_high_return_capture_does_not_pass_outcome_gate,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} pair-credit synthetic tests".format(len(tests)))


if __name__ == "__main__":
    main()
