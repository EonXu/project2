"""NumPy-only tests for the fail-closed bounded pair-pending foundation."""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.pair_pending import (  # noqa: E402
    PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
    PAIR_EVIDENCE_COMMITTED,
    PAIR_EVIDENCE_PENDING,
    PAIR_EVIDENCE_PREPARED,
    PairPendingEvidenceStore,
    freeze_pair_pending_batch,
    load_pair_pending_batch_state_dict,
    merge_generation_event_pair_scores,
    pair_pending_batch_state_dict,
    recompute_pair_stale_trust,
    reconstruct_pending_pair_mass,
)


def _batch(seed=0):
    rng = np.random.RandomState(seed)
    fields = []
    for index in range(35):
        fields.append(
            rng.normal(size=(2, 5, (index % 3) + 1)).astype(np.float32)
        )
    # Raw strict-pair score field.
    fields[17] = np.zeros((2, 5, 3), dtype=np.float32)
    fields[17][0, 1, 0] = 2.0
    # Population provenance field.
    fields[33] = np.zeros((2, 5, 3), dtype=np.float32)
    fields[33][..., 0] = 1.0
    return tuple(fields)


def _metadata(
        generation,
        sign,
        first_seen=10,
        last_valid=12,
        event_id=None,
        policy_version=3):
    if event_id is None:
        event_id = generation
    return {
        "policy_id": "policy_0",
        "replay_generation": generation,
        "environment_episode_id": 1000 + generation,
        "episode_ordinal": generation % 5,
        "capture_event_id": event_id,
        "prey_id": generation % 4,
        "participant_slots": (0, 1),
        "canonical_pair_identities": ((0, 1),),
        "factor_order": 2,
        "sign": sign,
        "outcome_success": sign > 0,
        "first_seen_adj_update": first_seen,
        "original_last_valid_adj_update": last_valid,
        "behavior_policy_version": policy_version,
        "source_class": "base",
        "target_bearing_transition_count": 1,
        "raw_event_quality": 1.0,
        "event_provenance_available": True,
    }


def _assert_raises(error_type, fn, contains=None):
    try:
        fn()
    except error_type as error:
        if contains is not None and contains not in str(error):
            raise AssertionError(
                "expected {!r} in {!r}".format(contains, str(error))
            )
        return
    raise AssertionError("expected {}".format(error_type.__name__))


def test_immutable_payload_survives_source_overwrite():
    source = list(_batch(1))
    original = source[0].copy()
    frozen = freeze_pair_pending_batch(source)
    source[0][...] = 999.0
    assert np.array_equal(frozen[0], original)
    assert not frozen[0].flags.writeable
    _assert_raises(ValueError, lambda: frozen[0].__setitem__(0, 0.0))
    restored = load_pair_pending_batch_state_dict(
        pair_pending_batch_state_dict(frozen)
    )
    assert np.array_equal(restored[0], original)
    assert not restored[0].flags.writeable


def test_negative_first_positive_later_horizon_four():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    negative = store.add_available(_metadata(660, -1), _batch(2))
    store.mark_pending(negative, 12)
    assert store.pending_age(negative, 16) == 4
    positive = store.add_available(
        _metadata(689, 1, first_seen=16, last_valid=18),
        _batch(3),
    )
    cohort = store.prepare_class_complete(
        (negative, positive),
        current_adj_update=16,
        current_policy_version=4,
        expected_ppo_epochs=2,
    )
    assert store.entries[negative]["state"] == PAIR_EVIDENCE_PREPARED
    store.commit_prepared(
        cohort_id=cohort,
        committed_adj_update=16,
        completed_ppo_epochs=2,
        optimizer_transaction_count=2,
        positive_effective_mass=0.5,
        negative_effective_mass=0.5,
        target_bearing_transition_count=2,
    )
    assert store.entries[negative]["state"] == PAIR_EVIDENCE_COMMITTED
    assert store.entries[positive]["state"] == PAIR_EVIDENCE_COMMITTED
    _assert_raises(
        RuntimeError,
        lambda: store.add_available(_metadata(660, -1), _batch(2)),
        "cannot be reused",
    )


def test_positive_first_negative_later_and_exact_ttl():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    positive = store.add_available(_metadata(10, 1), _batch(4))
    store.mark_pending(positive, 12)
    assert store.expire_out_of_horizon(16) == ()
    assert store.entries[positive]["state"] == PAIR_EVIDENCE_PENDING
    expired = store.expire_out_of_horizon(17)
    assert expired == (positive,)


def test_horizon_three_cannot_prepare_run107_equivalent():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=3)
    negative = store.add_available(_metadata(660, -1), _batch(5))
    store.mark_pending(negative, 12)
    store.expire_out_of_horizon(16)
    positive = store.add_available(
        _metadata(689, 1, first_seen=16, last_valid=18),
        _batch(6),
    )
    _assert_raises(
        RuntimeError,
        lambda: store.prepare_class_complete(
            (negative, positive), 16, 4, 2
        ),
        "only replay-available or pending",
    )


def test_transaction_failure_never_commits():
    failure_fields = (
        {"completed_ppo_epochs": 1, "optimizer_transaction_count": 1},
        {"completed_ppo_epochs": 2, "optimizer_transaction_count": 1},
        {"completed_ppo_epochs": 2, "optimizer_transaction_count": 2,
         "positive_effective_mass": 0.0},
        {"completed_ppo_epochs": 2, "optimizer_transaction_count": 2,
         "negative_effective_mass": 0.0},
        {"completed_ppo_epochs": 2, "optimizer_transaction_count": 2,
         "target_bearing_transition_count": 0},
        {"completed_ppo_epochs": 2, "optimizer_transaction_count": 2,
         "rollback": True},
        {"completed_ppo_epochs": 2, "optimizer_transaction_count": 2,
         "rejected": True},
    )
    for offset, overrides in enumerate(failure_fields):
        store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
        negative = store.add_available(
            _metadata(100 + offset * 2, -1), _batch(10 + offset)
        )
        positive = store.add_available(
            _metadata(101 + offset * 2, 1), _batch(30 + offset)
        )
        cohort = store.prepare_class_complete(
            (negative, positive), 12, 4, 2
        )
        kwargs = {
            "cohort_id": cohort,
            "committed_adj_update": 12,
            "completed_ppo_epochs": 2,
            "optimizer_transaction_count": 2,
            "positive_effective_mass": 0.5,
            "negative_effective_mass": 0.5,
            "target_bearing_transition_count": 2,
        }
        kwargs.update(overrides)
        _assert_raises(
            RuntimeError,
            lambda kwargs=kwargs, store=store: store.commit_prepared(
                **kwargs
            ),
            "cannot commit",
        )
        assert store.entries[negative]["state"] == PAIR_EVIDENCE_PREPARED
        store.release_prepared(cohort)
        assert store.entries[negative]["state"] in (
            "AVAILABLE_IN_REPLAY", PAIR_EVIDENCE_PENDING
        )


def test_duplicate_generation_event_sign_and_identity_fail_loud():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    store.add_available(_metadata(20, 1, event_id=5), _batch(50))
    conflicting_sign = _metadata(20, -1, event_id=5)
    _assert_raises(
        RuntimeError,
        lambda: store.add_available(conflicting_sign, _batch(50)),
        "conflicting provenance",
    )
    duplicate_event = _metadata(21, 1, event_id=5)
    duplicate_event["environment_episode_id"] = (
        _metadata(20, 1, event_id=5)["environment_episode_id"]
    )
    duplicate_event["prey_id"] = _metadata(20, 1, event_id=5)["prey_id"]
    _assert_raises(
        RuntimeError,
        lambda: store.add_available(duplicate_event, _batch(51)),
        "multiple pair generations",
    )
    bad_identity = _metadata(22, 1)
    bad_identity["canonical_pair_identities"] = ((0, 2),)
    _assert_raises(
        ValueError,
        lambda: store.add_available(bad_identity, _batch(52)),
        "not contained",
    )


def test_multiple_events_share_one_generation_and_commit_atomically():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    first_metadata = _metadata(25, -1, event_id=50)
    second_metadata = _metadata(25, -1, event_id=51)
    first = store.add_available(first_metadata, _batch(55))
    second = store.add_available(second_metadata, _batch(56))
    assert first != second
    assert set(store.keys_for_generation("policy_0", 25)) == {
        first, second
    }
    positive = store.add_available(_metadata(26, 1), _batch(57))
    _assert_raises(
        RuntimeError,
        lambda: store.prepare_class_complete(
            (first, positive), 12, 4, 2
        ),
        "every live event",
    )
    cohort = store.prepare_class_complete(
        (first, second, positive), 12, 4, 2
    )
    store.commit_prepared(cohort, 12, 2, 2, 0.5, 0.5, 3)
    assert all(
        store.entries[key]["state"] == PAIR_EVIDENCE_COMMITTED
        for key in (first, second, positive)
    )
    _assert_raises(
        RuntimeError,
        lambda: store.add_available(first_metadata, _batch(55)),
        "cannot be reused",
    )


def test_generation_event_score_merge_deduplicates_shared_transition():
    first = np.zeros((4, 3, 1), dtype=np.float32)
    second = np.zeros_like(first)
    third = np.zeros_like(first)
    first[1, 0, 0] = 0.75
    second[1, 0, 0] = 0.75
    third[2, 1, 0] = 0.50
    merged = merge_generation_event_pair_scores((first, second, third))
    assert merged.dtype == np.float32
    assert merged.shape == first.shape
    assert float(merged.sum()) == 1.25
    assert int(np.sum(np.any(merged > 0.0, axis=-1))) == 2
    assert np.array_equal(merged[first > 0.0], first[first > 0.0])
    assert np.array_equal(merged[third > 0.0], third[third > 0.0])


def test_generation_event_score_merge_rejects_conflicting_shared_value():
    first = np.zeros((4, 3, 1), dtype=np.float32)
    second = np.zeros_like(first)
    first[1, 0, 0] = 0.75
    second[1, 0, 0] = 0.50
    _assert_raises(
        RuntimeError,
        lambda: merge_generation_event_pair_scores((first, second)),
        "disagree on a shared strict pair target transition",
    )


def test_incomplete_event_provenance_cannot_prepare_training():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    negative_metadata = _metadata(23, -1)
    negative_metadata.update({
        "capture_event_id": -1,
        "prey_id": -1,
        "event_provenance_available": False,
    })
    negative = store.add_available(negative_metadata, _batch(53))
    positive = store.add_available(_metadata(24, 1), _batch(54))
    _assert_raises(
        RuntimeError,
        lambda: store.prepare_class_complete(
            (negative, positive), 12, 4, 2
        ),
        "complete episode/event/prey provenance",
    )


def test_checkpoint_resume_preserves_single_use_and_payload():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    negative = store.add_available(_metadata(30, -1), _batch(60))
    store.mark_pending(negative, 12)
    state = copy.deepcopy(store.state_dict())
    restored = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    restored.load_state_dict(state)
    assert np.array_equal(
        restored.entries[negative]["batch"][0],
        store.entries[negative]["batch"][0],
    )
    positive = restored.add_available(_metadata(31, 1), _batch(61))
    cohort = restored.prepare_class_complete(
        (negative, positive), 14, 5, 2
    )
    restored.commit_prepared(
        cohort, 14, 2, 2, 0.5, 0.5, 2
    )
    committed = restored.state_dict()
    second_restore = PairPendingEvidenceStore(
        enabled=True, max_adj_updates=4
    )
    second_restore.load_state_dict(committed)
    assert second_restore.entries[negative]["state"] == PAIR_EVIDENCE_COMMITTED
    _assert_raises(
        RuntimeError,
        lambda: second_restore.add_available(
            _metadata(30, -1), _batch(60)
        ),
        "cannot be reused",
    )
    _assert_raises(
        RuntimeError,
        lambda: second_restore.load_state_dict(
            committed, require_fresh_when_enabled=True
        ),
        "requires a fresh run",
    )


def test_stale_trust_recompute_matches_formula_and_policy_age():
    old = np.log(np.asarray([0.5, 0.25], dtype=np.float64))
    current = np.log(np.asarray([0.55, 0.50], dtype=np.float64))
    result = recompute_pair_stale_trust(
        old, current, trust_clip=0.2, trust_scale=0.25,
        minimum_weight=0.25
    )
    ratio = np.exp(current - old)
    expected = np.clip(
        np.exp(-np.maximum(np.abs(ratio - 1.0) - 0.2, 0.0) / 0.25),
        0.25,
        1.0,
    )
    assert np.allclose(result, expected.astype(np.float32), atol=1e-7)
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    key = store.add_available(_metadata(40, 1, policy_version=3), _batch(70))
    assert store.policy_age(key, 7) == 4
    _assert_raises(RuntimeError, lambda: store.policy_age(key, 2))


def test_signed_mass_reconstruction_and_nonuniform_trust():
    positive = np.zeros((5, 3), dtype=np.float32)
    negative = np.zeros((5, 3), dtype=np.float32)
    positive[1, 0] = 2.0
    negative[2, 1] = 4.0
    trust_positive = np.ones_like(positive)
    trust_negative = np.full_like(negative, 0.5)
    result = reconstruct_pending_pair_mass(
        pair_transition_scores=(positive, negative),
        episode_success=(True, False),
        coefficient=0.5,
        credit_cap=1.0,
        stale_trust_weights=(trust_positive, trust_negative),
    )
    assert result["contract_valid"] == 1.0
    assert abs(result["raw_positive_mass"] - 0.25) < 1e-7
    assert abs(result["raw_negative_mass"] - 0.25) < 1e-7
    assert abs(result["effective_positive_mass"] - 0.25) < 1e-7
    assert abs(result["effective_negative_mass"] - 0.125) < 1e-7
    assert abs(result["raw_centered_error"]) < 1e-7


def test_standard_nonzero_transaction_mirrors_single_use():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    negative = store.add_available(_metadata(51, -1), _batch(51))
    positive = store.add_available(_metadata(52, 1), _batch(52))
    committed = store.commit_available_from_standard_transaction(
        (negative, positive),
        committed_adj_update=12,
    )
    assert set(committed) == {negative, positive}
    assert all(
        store.entries[key]["state"] == PAIR_EVIDENCE_COMMITTED
        for key in committed
    )
    # Replaying a complete ordinary transaction must not re-open pending
    # evidence or overwrite the original commit metadata.
    original_updates = {
        key: store.entries[key]["committed_adj_update"]
        for key in committed
    }
    assert store.commit_available_from_standard_transaction(
        committed,
        committed_adj_update=13,
    ) == tuple()
    assert {
        key: store.entries[key]["committed_adj_update"]
        for key in committed
    } == original_updates


def test_standard_transaction_mirrors_new_side_with_committed_counterclass():
    store = PairPendingEvidenceStore(enabled=True, max_adj_updates=4)
    old_negative = store.add_available(_metadata(61, -1), _batch(61))
    old_positive = store.add_available(_metadata(62, 1), _batch(62))
    store.commit_available_from_standard_transaction(
        (old_negative, old_positive),
        committed_adj_update=12,
    )

    new_positive = store.add_available(_metadata(63, 1), _batch(63))
    mirrored = store.commit_available_from_standard_transaction(
        (old_negative, new_positive),
        committed_adj_update=13,
    )
    assert mirrored == (new_positive,)
    assert store.entries[old_negative]["committed_adj_update"] == 12
    assert store.entries[new_positive]["state"] == PAIR_EVIDENCE_COMMITTED
    assert store.entries[new_positive]["committed_adj_update"] == 13

    new_negative = store.add_available(_metadata(64, -1), _batch(64))
    _assert_raises(
        RuntimeError,
        lambda: store.commit_available_from_standard_transaction(
            (new_negative,),
            committed_adj_update=14,
        ),
        "not class-complete",
    )
    assert store.entries[new_negative]["state"] == (
        PAIR_EVIDENCE_AVAILABLE_IN_REPLAY
    )


def test_disabled_store_is_default_off_and_side_effect_free():
    batch = _batch(80)
    before = tuple(value.copy() for value in batch)
    store = PairPendingEvidenceStore()
    _assert_raises(
        RuntimeError,
        lambda: store.add_available(_metadata(50, 1), batch),
        "disabled",
    )
    assert len(store) == 0
    assert all(np.array_equal(a, b) for a, b in zip(batch, before))


def test_run107_v2_structural_horizons_are_not_training_payloads():
    path = os.path.join(
        REPO_ROOT,
        "scripts",
        "results",
        "wolfpack",
        "sddfg",
        "sddfg_intra_ep_4to6_r2_j1_rec30_seed1",
        "run107",
        "run107_pair_evidence_pending_horizon_sweep_v2.json",
    )
    with open(path, "r") as stream:
        report = json.load(stream)
    assert report["diagnostic_version"] == 2
    assert report["minimum_pending_updates_for_first_structural_class_complete"] == 4
    assert report["minimum_pending_updates_for_two_structural_class_complete"] == 7
    assert report["minimum_pending_updates_for_first_fully_trainable_class_complete"] is None
    assert report["bounded_pending_training_safe"] is False

    horizon_four = next(
        item for item in report["horizon_sweep"]
        if int(item["extra_pending_updates"]) == 4
    )
    assert horizon_four["structural_class_complete_cohort_count"] == 1
    assert horizon_four["fully_trainable_class_complete_cohort_count"] == 0
    cohort = horizon_four["cohorts"][0]
    generations = {
        int(item["replay_generation"]): int(item["sign"])
        for item in cohort["evidence"]
    }
    assert generations == {660: -1, 689: 1}
    assert cohort["cohort_step"] == 144800
    assert cohort["selected_episode_count"] == 5
    assert cohort["selected_chunk_count"] == 100
    assert cohort["yielded_chunk_count"] == 100
    assert cohort["trained_chunk_count"] == 100
    assert cohort["immutable_payload_available"] is False
    assert cohort["stale_trust_contract_evaluable"] is False
    assert cohort["signed_mass_contract_evaluable"] is False

    horizon_seven = next(
        item for item in report["horizon_sweep"]
        if int(item["extra_pending_updates"]) == 7
    )
    assert horizon_seven["structural_class_complete_cohort_count"] == 2
    assert horizon_seven["fully_trainable_class_complete_cohort_count"] == 0
    second_generations = {
        int(item["replay_generation"]): int(item["sign"])
        for item in horizon_seven["cohorts"][1]["evidence"]
    }
    assert second_generations == {709: 1, 748: -1}
    assert horizon_seven["cohorts"][1]["cohort_step"] == 156800


def main():
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("All {} pair pending foundation tests passed.".format(len(tests)))


if __name__ == "__main__":
    main()
