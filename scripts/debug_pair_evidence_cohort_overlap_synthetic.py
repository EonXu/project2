"""Synthetic tests for the offline pair-evidence overlap reconstruction."""

from __future__ import print_function

import copy

from debug_pair_evidence_cohort_overlap import (
    reconstruct_pair_evidence_overlap,
)


def _episode(
        step,
        generation,
        sign,
        age,
        base=0,
        support=0,
        used=0):
    return {
        "env_step": str(step),
        "policy_id": "policy_0",
        "episode_generation": str(generation),
        "pair_evidence_sign": str(sign),
        "pair_evidence_episode": "1",
        "episode_recency_age": str(age),
        "selected_for_training": str(int(bool(base or support))),
        "base_selected": str(base),
        "support_selected": str(support),
        "outcome_support_used": str(used),
        "outcome_success": str(int(sign > 0)),
        "environment_episode_id": str(generation),
        "capture_event_id": str(1000 + generation),
        "capture_prey_id": "1",
        "participant_slots_available": "1",
        "participant_slots": "0-1",
        "capture_event_id_available": "1",
        "row_contract_valid": "1",
    }


def _adj(
        step,
        pair_complete=0,
        pair_nonzero=0,
        pair_augmented=0,
        outcome_complete=0,
        outcome_augmented=0):
    return {
        "step": str(step),
        "adj_sample_episode_count": "4",
        "adj_sample_selected_chunk_count": "80",
        "adj_ppo_stale_trust_min_weight": "0.25",
        "adj_sample_pair_class_complete": str(pair_complete),
        "adj_sample_pair_augmented_count": str(pair_augmented),
        "pair_pursuit_credit_nonzero_count": str(pair_nonzero),
        "pair_gradient_target_bearing_update": str(int(pair_nonzero > 0)),
        "pair_pursuit_credit_positive_mass": (
            "0.25" if pair_nonzero else "0.0"
        ),
        "pair_pursuit_credit_negative_mass": (
            "0.25" if pair_nonzero else "0.0"
        ),
        "adj_sample_outcome_class_complete": str(outcome_complete),
        "adj_sample_outcome_augmented_positive_count": str(
            outcome_augmented
        ),
        "adj_sample_outcome_augmented_negative_count": "0",
    }


def test_natural_expiry_does_not_invent_deferred_opportunity():
    episodes = [
        _episode(100, 1, -1, 0, base=1),
        _episode(200, 1, -1, 4),
        _episode(400, 2, 1, 0, base=1),
        _episode(500, 2, 1, 4),
    ]
    result = reconstruct_pair_evidence_overlap(
        episodes,
        [_adj(100), _adj(200), _adj(400), _adj(500)],
    )
    opportunity = result["positive_opportunities"][0]
    assert opportunity["negative_present_count"] == 0
    assert opportunity["negative_naturally_expired_count"] == 1
    assert not opportunity[
        "class_complete_possible_if_shared_used_were_deferred"
    ]
    assert result["positive_blocked_only_by_shared_used_state_count"] == 0
    assert opportunity[
        "minimum_replay_validity_extension_updates_for_overlap"
    ] == 2
    assert result["pair_evidence_pending_positive_count"] == 1
    assert result["pair_evidence_pending_negative_count"] == 1
    print("PASS natural expiry is not a deferred-consumption opportunity")


def test_shared_used_zero_pair_can_be_identified_without_calling_it_pair_commit():
    episodes = [
        _episode(100, 1, -1, 0, support=1, used=1),
        _episode(200, 1, -1, 4, used=1),
        _episode(200, 2, 1, 0, base=1),
    ]
    result = reconstruct_pair_evidence_overlap(
        episodes,
        [
            _adj(100, outcome_complete=1, outcome_augmented=1),
            _adj(200),
        ],
    )
    negative = result["evidence_lifecycles"][0]
    assert negative["shared_support_used_reason"] == (
        "GENERIC_CAPTURE_SUPPORT_ONLY"
    )
    assert not negative["pair_nonzero_consumed"]
    assert result["pair_evidence_zero_target_used_commit_count"] == 0
    assert result[
        "pair_evidence_shared_support_used_without_pair_target_count"
    ] == 1
    opportunity = result["positive_opportunities"][0]
    assert opportunity["negative_present_count"] == 1
    assert opportunity[
        "negative_shared_used_without_pair_commit_count"
    ] == 1
    assert opportunity["blocked_only_by_shared_used_state"]
    assert opportunity[
        "class_complete_possible_if_shared_used_were_deferred"
    ]
    print("PASS shared generic support used is distinct from pair commit")


def test_positive_first_and_negative_later_overlap_is_detected():
    episodes = [
        _episode(100, 10, 1, 0, base=1),
        _episode(200, 10, 1, 4),
        _episode(200, 11, -1, 0, base=1),
    ]
    result = reconstruct_pair_evidence_overlap(
        episodes,
        [_adj(100), _adj(200, pair_complete=1, pair_nonzero=2)],
    )
    assert result["opposite_sign_overlap_pair_count"] == 1
    # Positive opportunities are evaluated at the positive's first availability.
    assert result["positive_opportunities"][0]["next_negative_generation"] == 11
    assert result["positive_opportunities"][0][
        "negative_overlap_anytime_count"
    ] == 1
    print("PASS later opposite sign is reported without backdating overlap")


def test_recorded_same_step_overlap_is_class_complete_possible():
    episodes = [
        _episode(100, 20, -1, 0, base=1),
        _episode(100, 21, 1, 0, base=1),
    ]
    result = reconstruct_pair_evidence_overlap(
        episodes,
        [_adj(100, pair_complete=1, pair_nonzero=2)],
    )
    opportunity = result["positive_opportunities"][0]
    assert opportunity["negative_present_count"] == 1
    assert opportunity["negative_base_selected_count"] == 1
    assert opportunity["class_complete_possible_in_recorded_replay"]
    assert result["pair_evidence_nonzero_target_commit_count"] == 2
    print("PASS same-step signed overlap is reconstructed")


def test_duplicate_exposure_and_sign_conflict_fail_loud():
    duplicate = [
        _episode(100, 30, -1, 0),
        _episode(100, 30, -1, 0),
    ]
    try:
        reconstruct_pair_evidence_overlap(duplicate, [_adj(100)])
    except ValueError as exc:
        assert "duplicate pair-evidence exposure" in str(exc)
    else:
        raise AssertionError("duplicate exposure did not fail")

    conflict = [
        _episode(100, 31, -1, 0),
        _episode(200, 31, 1, 4),
    ]
    try:
        reconstruct_pair_evidence_overlap(
            conflict, [_adj(100), _adj(200)]
        )
    except ValueError as exc:
        assert "changes sign" in str(exc)
    else:
        raise AssertionError("generation sign conflict did not fail")
    print("PASS duplicate exposure and sign conflict fail loud")


def test_base_support_overlap_and_failed_row_contract_fail_loud():
    overlap = _episode(100, 40, -1, 0, base=1, support=1)
    try:
        reconstruct_pair_evidence_overlap([overlap], [_adj(100)])
    except ValueError as exc:
        assert "base- and support-selected" in str(exc)
    else:
        raise AssertionError("base/support overlap did not fail")

    invalid = _episode(100, 41, -1, 0)
    invalid["row_contract_valid"] = "0"
    try:
        reconstruct_pair_evidence_overlap([invalid], [_adj(100)])
    except ValueError as exc:
        assert "row contract failed" in str(exc)
    else:
        raise AssertionError("invalid row contract did not fail")
    print("PASS selection and row contracts fail loud")


def test_input_rows_are_trajectory_neutral():
    episodes = [
        _episode(100, 50, -1, 0, base=1),
        _episode(200, 51, 1, 0, base=1),
    ]
    adjacency = [_adj(100), _adj(200)]
    before_episode = copy.deepcopy(episodes)
    before_adjacency = copy.deepcopy(adjacency)
    reconstruct_pair_evidence_overlap(episodes, adjacency)
    assert episodes == before_episode
    assert adjacency == before_adjacency
    print("PASS offline reconstruction does not mutate inputs")


def test_horizon_sweep_finds_exact_first_and_second_structural_cohorts():
    episodes = [
        _episode(100, 60, -1, 0, base=1),
        _episode(200, 60, -1, 4),
        _episode(400, 61, 1, 0, base=1),
        _episode(500, 61, 1, 4),
        _episode(700, 62, -1, 0, base=1),
    ]
    adjacency = [
        _adj(100),
        _adj(200),
        _adj(300),
        _adj(400),
        _adj(500),
        _adj(600),
        _adj(700),
    ]
    for row in adjacency:
        row.update({
            "adj_sample_episode_count": "4",
            "adj_sample_selected_chunk_count": "80",
            "adj_ppo_stale_trust_min_weight": "0.25",
        })
    result = reconstruct_pair_evidence_overlap(episodes, adjacency)
    assert result[
        "minimum_pending_updates_for_first_structural_class_complete"
    ] == 2
    # The first cohort consumes generation 60 and 61. No second positive exists.
    assert result[
        "minimum_pending_updates_for_two_structural_class_complete"
    ] is None
    cohort = result["horizon_sweep"][2]["cohorts"][0]
    assert cohort["selected_episode_count"] == 5
    assert cohort["selected_chunk_count"] == 100
    assert cohort["partition_count"] == 1
    assert cohort["partition_slot"] == -1
    assert not cohort["support_v6_full_training_contract_valid"]
    assert not result["bounded_pending_training_safe"]
    print("PASS horizon sweep is structural and fails closed on missing payload")


def test_run107_equivalent_horizon_sweep_is_four_then_seven_updates():
    episodes = []
    for step, age in (
            (139200, 0), (140000, 1), (140800, 2), (141600, 3)):
        episodes.append(_episode(step, 660, -1, age))
    for step, age in (
            (144800, 0), (145600, 1), (146400, 2), (147200, 3)):
        episodes.append(_episode(
            step,
            689,
            1,
            age,
            base=int(step == 144800),
        ))
    for step, age in (
            (148800, 0), (149600, 1), (150400, 2), (151200, 3)):
        episodes.append(_episode(
            step,
            709,
            1,
            age,
            base=int(step == 148800),
        ))
    for step, age in (
            (156800, 0), (157600, 1), (158400, 2), (159200, 3)):
        episodes.append(_episode(
            step,
            748,
            -1,
            age,
            base=int(step == 156800),
        ))
    adjacency = [
        _adj(step)
        for step in range(139200, 159201, 800)
    ]

    result = reconstruct_pair_evidence_overlap(episodes, adjacency)
    assert result["adjacency_update_interval"] == 800
    assert result[
        "minimum_pending_updates_for_first_structural_class_complete"
    ] == 4
    assert result[
        "minimum_pending_updates_for_two_structural_class_complete"
    ] == 7
    first = result["horizon_sweep"][4]["cohorts"]
    assert len(first) == 1
    assert [
        item["replay_generation"] for item in first[0]["evidence"]
    ] == [689, 660]
    second = result["horizon_sweep"][7]["cohorts"]
    assert len(second) == 2
    assert [
        item["replay_generation"] for item in second[1]["evidence"]
    ] == [748, 709]
    assert all(
        cohort["selected_episode_count"] == 5
        and cohort["selected_chunk_count"] == 100
        and cohort["yielded_chunk_count"] == 100
        and cohort["trained_chunk_count"] == 100
        for cohort in second
    )
    assert result["existing_safe_max_pending_updates"] == 0
    assert result[
        "minimum_pending_updates_for_first_fully_trainable_class_complete"
    ] is None
    assert not result["bounded_pending_training_safe"]
    print("PASS run107-equivalent sweep is 4/7 and training fails closed")


def main():
    tests = [
        test_natural_expiry_does_not_invent_deferred_opportunity,
        test_shared_used_zero_pair_can_be_identified_without_calling_it_pair_commit,
        test_positive_first_and_negative_later_overlap_is_detected,
        test_recorded_same_step_overlap_is_class_complete_possible,
        test_duplicate_exposure_and_sign_conflict_fail_loud,
        test_base_support_overlap_and_failed_row_contract_fail_loud,
        test_input_rows_are_trajectory_neutral,
        test_horizon_sweep_finds_exact_first_and_second_structural_cohorts,
        test_run107_equivalent_horizon_sweep_is_four_then_seven_updates,
    ]
    for test in tests:
        test()
    print("PASS {} pair evidence cohort-overlap tests".format(len(tests)))


if __name__ == "__main__":
    main()
