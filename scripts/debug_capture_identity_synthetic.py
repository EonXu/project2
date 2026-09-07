"""Synthetic checks for real capture-participant factor attribution."""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pair_credit import (
    build_capture_identity_factor_weights,
    canonical_capture_factor_catalog,
    compute_capture_anchored_pair_credit,
    compute_capture_to_win_triplet_outcome_advantage,
    scale_capture_to_win_outcome_credit,
)


def _set_factor(adj, episode, factor, nodes):
    adj[episode, :, factor] = 0
    adj[episode, list(nodes), factor] = 1


def _event(event_id, target_id, participants):
    return {
        "event_id": event_id,
        "target_id": target_id,
        "participant_slots": list(participants),
    }


def test_only_real_participant_triplet_is_matched():
    adj = np.zeros((1, 6, 4), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1, 2))
    _set_factor(adj, 0, 1, (0, 1, 3))
    _set_factor(adj, 0, 2, (3, 4, 5))
    _set_factor(adj, 0, 3, (0, 1))
    result = build_capture_identity_factor_weights(
        adj,
        [[_event(7, 0, (0, 1, 2))]],
        expected_capture_counts=np.array([1.0], dtype=np.float32),
    )
    assert np.allclose(result["factor_weights"], [[1.0, 0.0, 0.0, 0.0]])
    assert result["matched_event_count"][0] == 1.0
    assert result["unmatched_event_count"][0] == 0.0


def test_two_participant_capture_matches_only_exact_pair():
    adj = np.zeros((1, 5, 4), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 0, 1, (0, 2))
    _set_factor(adj, 0, 2, (0, 1, 2))
    _set_factor(adj, 0, 3, (0, 1, 3))
    result = build_capture_identity_factor_weights(
        adj,
        [[_event(1, 0, (0, 1))]],
        expected_capture_counts=np.array([1.0], dtype=np.float32),
    )
    assert np.allclose(result["factor_weights"], [[1.0, 0.0, 0.0, 0.0]])
    assert result["matched_event_count"][0] == 1.0
    assert result["candidate_factor_count"][0] == 1.0
    assert result["matched_factor_order_sum"][0] == 2.0


def test_duplicate_factor_slots_do_not_multiply_event_mass():
    adj = np.zeros((1, 5, 4), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1))
    _set_factor(adj, 0, 1, (1, 0))
    _set_factor(adj, 0, 2, (0, 2))
    result = build_capture_identity_factor_weights(
        adj,
        [[_event(4, 0, (0, 1))]],
        expected_capture_counts=np.array([1.0], dtype=np.float32),
    )
    assert np.allclose(result["factor_weights"], [[0.5, 0.5, 0.0, 0.0]])
    assert np.isclose(result["factor_weights"].sum(), 1.0)
    assert result["raw_candidate_factor_count"][0] == 2.0
    assert result["candidate_factor_count"][0] == 1.0
    assert result["duplicate_candidate_factor_count"][0] == 1.0


def test_multi_event_and_multi_environment_are_isolated():
    adj = np.zeros((2, 6, 4), dtype=np.int64)
    for episode in range(2):
        _set_factor(adj, episode, 0, (0, 1, 2))
        _set_factor(adj, episode, 1, (0, 1, 3))
        _set_factor(adj, episode, 2, (3, 4, 5))
    result = build_capture_identity_factor_weights(
        adj,
        [
            [_event(1, 0, (0, 1, 2)), _event(2, 1, (3, 4, 5))],
            [_event(3, 0, (0, 1, 2, 3))],
        ],
        expected_capture_counts=np.array([2.0, 1.0], dtype=np.float32),
    )
    assert np.allclose(result["factor_weights"][0], [1.0, 0.0, 1.0, 0.0])
    assert np.allclose(result["factor_weights"][1], [0.5, 0.5, 0.0, 0.0])
    assert np.allclose(result["factor_weights"].sum(axis=1), [2.0, 1.0])
    assert np.allclose(result["matched_event_count"], [2.0, 1.0])


def test_two_participant_capture_without_selected_pair_is_unmatched():
    adj = np.zeros((1, 5, 2), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1, 2))
    _set_factor(adj, 0, 1, (0, 1, 3))
    result = build_capture_identity_factor_weights(
        adj,
        [[_event(1, 0, (0, 1))]],
        expected_capture_counts=np.array([1.0], dtype=np.float32),
    )
    assert result["factor_weights"].sum() == 0.0
    assert result["matched_event_count"][0] == 0.0
    assert result["unmatched_event_count"][0] == 1.0
    assert result["candidate_only_event_count"][0] == 1.0
    catalog = canonical_capture_factor_catalog(5, 3)
    pair_index = catalog.index((0, 1))
    assert np.isclose(
        result["candidate_only_factor_weights"][0, pair_index], 1.0
    )
    assert np.isclose(result["candidate_only_factor_weights"].sum(), 1.0)


def test_active_and_candidate_only_identity_are_strictly_separated():
    adj = np.zeros((2, 5, 2), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1))
    result = build_capture_identity_factor_weights(
        adj,
        [
            [_event(1, 0, (0, 1))],
            [_event(2, 0, (0, 1))],
        ],
        expected_capture_counts=np.ones(2, dtype=np.float32),
    )
    assert np.isclose(result["factor_weights"][0].sum(), 1.0)
    assert np.isclose(
        result["candidate_only_factor_weights"][0].sum(), 0.0
    )
    assert np.isclose(result["factor_weights"][1].sum(), 0.0)
    assert np.isclose(
        result["candidate_only_factor_weights"][1].sum(), 1.0
    )
    assert np.allclose(result["candidate_only_event_mass_error"], 0.0)
    assert np.allclose(result["combined_identity_event_mass_error"], 0.0)


def test_missing_or_duplicate_identity_fails_loudly():
    adj = np.zeros((1, 4, 1), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1, 2))
    try:
        build_capture_identity_factor_weights(
            adj,
            [[]],
            expected_capture_counts=np.array([1.0], dtype=np.float32),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("capture_count without identity did not fail")
    try:
        build_capture_identity_factor_weights(
            adj,
            [[_event(1, 0, (0, 1, 2)), _event(1, 1, (0, 1, 2))]],
            expected_capture_counts=np.array([2.0], dtype=np.float32),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate event identity did not fail")


def test_identity_weight_survives_pair_and_outcome_chain():
    time_steps, episodes, agents, factors = 3, 2, 5, 3
    adj = np.zeros((time_steps, episodes, agents, factors), dtype=np.int64)
    for episode in range(episodes):
        adj[0, episode, [0, 1], 0] = 1
        adj[1, episode, [0, 1, 2], 1] = 1
        adj[1, episode, [0, 1, 3], 2] = 1
    step_match = build_capture_identity_factor_weights(
        adj[1],
        [
            [_event(1, 0, (0, 1, 2))],
            [_event(2, 0, (0, 1, 2))],
        ],
        expected_capture_counts=np.ones(episodes, dtype=np.float32),
    )
    matches = np.zeros((time_steps, episodes, factors), dtype=np.float32)
    matches[1] = step_match["factor_weights"]
    captures = np.zeros((time_steps, episodes), dtype=np.float32)
    captures[1] = 1.0
    factor_size = adj.sum(axis=2)
    pair = compute_capture_anchored_pair_credit(
        current_adj=adj,
        valid_factor=factor_size > 0,
        factor_size=factor_size,
        active_agent_count=np.full(captures.shape, 4, dtype=np.int64),
        selected_factor_behavior_probability=np.where(
            factor_size > 0,
            np.float32(0.5),
            np.float32(0.0),
        ),
        capture_counts=captures,
        capture_factor_match=matches,
        valid_graph_transition=(factor_size > 0).any(axis=2),
        dones_env=np.zeros_like(captures),
        team_rewards=np.zeros_like(captures),
        window=2,
        gamma=0.97,
    )
    assert pair["triplet_capture_quality"][1, :, 1].sum() == 2.0
    assert pair["triplet_capture_quality"][1, :, 2].sum() == 0.0
    assert np.all(pair["pair_to_triplet_transition_score"][0, :, 0] > 0.0)
    outcome = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([True, False]),
        triplet_capture_quality=pair["triplet_capture_quality"],
    )["triplet_outcome_advantage"]
    assert np.isclose(outcome[1, 0, 1], 0.5)
    assert np.isclose(outcome[1, 1, 1], -0.5)
    assert np.count_nonzero(outcome[:, :, 2]) == 0
    assert np.isclose(outcome.sum(), 0.0)


def test_exact_pair_capture_survives_strict_future_chain():
    time_steps, episodes, agents, factors = 3, 2, 5, 3
    adj = np.zeros((time_steps, episodes, agents, factors), dtype=np.int64)
    for episode in range(episodes):
        adj[0, episode, [0, 1], 0] = 1
        adj[2, episode, [0, 1], 1] = 1
        adj[2, episode, [0, 1, 2], 2] = 1
    step_match = build_capture_identity_factor_weights(
        adj[2],
        [
            [_event(1, 0, (0, 1))],
            [_event(2, 0, (0, 1))],
        ],
        expected_capture_counts=np.ones(episodes, dtype=np.float32),
    )
    assert np.allclose(step_match["factor_weights"][:, 1], 1.0)
    assert np.count_nonzero(step_match["factor_weights"][:, 2]) == 0
    matches = np.zeros((time_steps, episodes, factors), dtype=np.float32)
    matches[2] = step_match["factor_weights"]
    captures = np.zeros((time_steps, episodes), dtype=np.float32)
    captures[2] = 1.0
    factor_size = adj.sum(axis=2)
    pair = compute_capture_anchored_pair_credit(
        current_adj=adj,
        valid_factor=factor_size > 0,
        factor_size=factor_size,
        active_agent_count=np.full(captures.shape, 4, dtype=np.int64),
        selected_factor_behavior_probability=np.where(
            factor_size > 0,
            np.float32(0.5),
            np.float32(0.0),
        ),
        capture_counts=captures,
        capture_factor_match=matches,
        valid_graph_transition=(factor_size > 0).any(axis=2),
        dones_env=np.zeros_like(captures),
        team_rewards=np.zeros_like(captures),
        window=3,
        gamma=0.97,
    )
    assert np.all(pair["pair_to_triplet_transition_score"][0, :, 0] > 0.0)
    assert np.all(pair["pair_transition_delay"][0, :, 0] == 2.0)
    assert np.all(pair["capture_factor_quality"][2, :, 1] == 1.0)
    assert np.count_nonzero(pair["capture_factor_quality"][:, :, 2]) == 0
    assert pair["offset0_candidate_count"].sum() == 2.0


def test_multi_event_episode_preserves_episode_total():
    time_steps, episodes, agents, factors = 3, 2, 5, 3
    quality = np.zeros((time_steps, episodes, factors), dtype=np.float32)
    # Successful episode has two identity-matched events; failed episode one.
    quality[0, 0, 0] = 1.0
    quality[1, 0, 1] = 1.0
    quality[1, 1, 2] = 1.0
    result = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([True, False]),
        triplet_capture_quality=quality,
    )
    gate = result["triplet_outcome_advantage"]
    assert np.isclose(gate[:, 0].sum(), 0.5)
    assert np.isclose(gate[:, 1].sum(), -0.5)
    assert np.isclose(gate.sum(), 0.0)
    assert np.isclose(gate[0, 0, 0], 0.25)
    assert np.isclose(gate[1, 0, 1], 0.25)


def test_signed_identity_delta_is_target_local():
    quality = np.zeros((2, 2, 4), dtype=np.float32)
    quality[0, 0, 0] = 1.0
    quality[0, 1, 2] = 1.0
    gate = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([True, False]),
        triplet_capture_quality=quality,
    )["triplet_outcome_advantage"]
    scaled = scale_capture_to_win_outcome_credit(
        triplet_outcome_advantage=gate,
        graph_advantage=np.ones((2, 2), dtype=np.float32),
        valid_graph_transition=np.ones((2, 2), dtype=bool),
        coefficient=0.15,
        cap=0.25,
    )
    target_mask = quality > 0.0
    assert scaled[0, 0, 0] > 0.0
    assert scaled[0, 1, 2] < 0.0
    assert np.count_nonzero(scaled[~target_mask]) == 0
    # Local injection is additive after shared-mean removal. It must not be
    # centered again, so every non-target factor remains bitwise unchanged.
    centered_factor_advantage = np.zeros_like(scaled)
    injected = centered_factor_advantage + scaled
    assert np.count_nonzero(injected[~target_mask]) == 0


def test_padding_and_invalid_participants_cannot_match():
    adj = np.zeros((1, 4, 3), dtype=np.int64)
    _set_factor(adj, 0, 0, (0, 1))
    # Factors 1 and 2 are padding (empty) and cannot become candidates.
    result = build_capture_identity_factor_weights(
        adj,
        [[_event(8, 0, (0, 1))]],
        expected_capture_counts=np.array([1.0], dtype=np.float32),
    )
    assert np.allclose(result["factor_weights"], [[1.0, 0.0, 0.0]])
    try:
        build_capture_identity_factor_weights(
            adj,
            [[_event(9, 0, (0, 4))]],
            expected_capture_counts=np.array([1.0], dtype=np.float32),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-range participant did not fail")


def main():
    tests = [
        test_only_real_participant_triplet_is_matched,
        test_two_participant_capture_matches_only_exact_pair,
        test_duplicate_factor_slots_do_not_multiply_event_mass,
        test_multi_event_and_multi_environment_are_isolated,
        test_two_participant_capture_without_selected_pair_is_unmatched,
        test_active_and_candidate_only_identity_are_strictly_separated,
        test_missing_or_duplicate_identity_fails_loudly,
        test_identity_weight_survives_pair_and_outcome_chain,
        test_exact_pair_capture_survives_strict_future_chain,
        test_multi_event_episode_preserves_episode_total,
        test_signed_identity_delta_is_target_local,
        test_padding_and_invalid_participants_cannot_match,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} capture-identity tests".format(len(tests)))


if __name__ == "__main__":
    main()
