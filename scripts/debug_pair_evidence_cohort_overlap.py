"""Reconstruct signed pair-evidence overlap from trajectory-neutral CSV logs.

This tool is deliberately offline.  It does not import the training path and
does not infer that a support-used episode received pair-local supervision:
``outcome_support_used`` is shared by generic capture support and pair support.
The adjacency CSV is therefore required to distinguish a real non-zero pair
commit from a generic support selection whose pair target stayed zero.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
from collections import OrderedDict
from functools import reduce


DIAGNOSTIC_VERSION = 2


def _read_csv(path):
    with open(path, "r") as handle:
        return list(csv.DictReader(handle))


def _number(row, name, default=None):
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        if default is not None:
            return float(default)
        raise ValueError("missing numeric field {!r}".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite field {!r}: {!r}".format(name, value))
    return result


def _integer(row, name, default=None):
    return int(round(_number(row, name, default=default)))


def _flag(row, name, default=0):
    value = _number(row, name, default=default)
    if value not in (0.0, 1.0):
        raise ValueError("field {!r} is not binary: {!r}".format(name, value))
    return bool(value)


def _first_step(rows, predicate):
    matching = [int(row["env_step"]) for row in rows if predicate(row)]
    return min(matching) if matching else None


def _step_metrics(adjacency_by_step, step):
    row = adjacency_by_step.get(int(step))
    if row is None:
        return {
            "pair_class_complete": False,
            "pair_nonzero_target": False,
            "pair_positive_mass": 0.0,
            "pair_negative_mass": 0.0,
            "pair_augmented_count": 0,
            "outcome_class_complete": False,
            "outcome_augmented_count": 0,
        }
    positive_mass = _number(
        row, "pair_pursuit_credit_positive_mass", default=0.0
    )
    negative_mass = _number(
        row, "pair_pursuit_credit_negative_mass", default=0.0
    )
    target_count = _number(
        row, "pair_pursuit_credit_nonzero_count", default=0.0
    )
    gradient_target = _number(
        row, "pair_gradient_target_bearing_update", default=0.0
    )
    return {
        "pair_class_complete": _flag(
            row, "adj_sample_pair_class_complete", default=0
        ),
        "pair_nonzero_target": bool(
            target_count > 0.0
            or gradient_target > 0.0
            or positive_mass > 0.0
            or negative_mass > 0.0
        ),
        "pair_positive_mass": positive_mass,
        "pair_negative_mass": negative_mass,
        "pair_augmented_count": _integer(
            row, "adj_sample_pair_augmented_count", default=0
        ),
        "outcome_class_complete": _flag(
            row, "adj_sample_outcome_class_complete", default=0
        ),
        "outcome_augmented_count": (
            _integer(
                row, "adj_sample_outcome_augmented_positive_count", default=0
            )
            + _integer(
                row, "adj_sample_outcome_augmented_negative_count", default=0
            )
        ),
    }


def _support_reason(metrics):
    if metrics["pair_nonzero_target"]:
        return "NONZERO_PAIR_TRANSACTION"
    if (
            metrics["outcome_class_complete"]
            and metrics["outcome_augmented_count"] > 0):
        return "GENERIC_CAPTURE_SUPPORT_ONLY"
    return "SHARED_SUPPORT_USED_WITHOUT_NONZERO_PAIR_TARGET"


def _selected_at_step(group_rows, step):
    return any(
        _integer(row, "env_step") == int(step)
        and _flag(row, "selected_for_training", default=0)
        for row in group_rows
    )


def _source_at_step(group_rows, step):
    matching = [
        row for row in group_rows
        if _integer(row, "env_step") == int(step)
    ]
    if not matching:
        return "PAIR_PENDING_SNAPSHOT"
    row = matching[0]
    if _flag(row, "base_selected", default=0):
        return "BASE_REPLAY"
    if _flag(row, "support_selected", default=0):
        return "CURRENT_REPLAY_SUPPORT"
    return "CURRENT_REPLAY_AVAILABLE"


def _adjacency_population(row):
    episode_count = _integer(row, "adj_sample_episode_count")
    chunk_count = _integer(row, "adj_sample_selected_chunk_count")
    if episode_count <= 0 or chunk_count <= 0:
        raise ValueError("adjacency selected population must be positive")
    if chunk_count % episode_count != 0:
        raise ValueError(
            "adjacency chunks do not divide evenly across selected episodes"
        )
    return episode_count, chunk_count, chunk_count // episode_count


def _run_pending_horizon(
        horizon_updates,
        lifecycles,
        groups,
        adjacency_by_step,
        adjacency_update_interval):
    """Dry-run support-v6 signed selection with immutable payload unavailable.

    The state machine and population are exact at generation/episode/chunk
    level. Target mass and stale trust deliberately remain unevaluable because
    neither the episode CSV nor the terminal checkpoint stores evicted behavior
    logits, masks, or pair-quality tensors.
    """
    horizon_updates = int(horizon_updates)
    consumed = set()
    cohorts = []
    ever_pending_positive = set()
    ever_pending_negative = set()
    max_pending_positive = 0
    max_pending_negative = 0
    overlap_step_count = 0
    blocked_reuse = set()
    all_steps = sorted(adjacency_by_step)
    by_key = {
        (item["policy_id"], item["replay_generation"]): item
        for item in lifecycles
    }
    for step in all_steps:
        available = []
        pending_now = []
        for key, item in by_key.items():
            if step < item["first_available_step"]:
                continue
            extended_last = (
                item["last_available_step"]
                + horizon_updates * adjacency_update_interval
            )
            if key in consumed:
                if step <= extended_last:
                    blocked_reuse.add(key)
                continue
            if step <= extended_last:
                available.append(item)
                if step > item["last_available_step"]:
                    pending_now.append(item)
                    if item["sign"] > 0:
                        ever_pending_positive.add(key)
                    else:
                        ever_pending_negative.add(key)
        max_pending_positive = max(
            max_pending_positive,
            sum(int(item["sign"] > 0) for item in pending_now),
        )
        max_pending_negative = max(
            max_pending_negative,
            sum(int(item["sign"] < 0) for item in pending_now),
        )
        positive = [item for item in available if item["sign"] > 0]
        negative = [item for item in available if item["sign"] < 0]
        if not positive or not negative:
            continue
        overlap_step_count += 1

        selected = [
            item for item in available
            if _selected_at_step(
                groups[(item["policy_id"], item["replay_generation"])],
                step,
            )
        ]
        selected_keys = set(
            (item["policy_id"], item["replay_generation"])
            for item in selected
        )
        has_positive = any(item["sign"] > 0 for item in selected)
        has_negative = any(item["sign"] < 0 for item in selected)
        supplements = []
        if not has_positive:
            supplements.append(max(
                (
                    item for item in positive
                    if (item["policy_id"], item["replay_generation"])
                    not in selected_keys
                ),
                key=lambda item: item["replay_generation"],
            ))
        if not has_negative:
            supplements.append(max(
                (
                    item for item in negative
                    if (item["policy_id"], item["replay_generation"])
                    not in selected_keys
                ),
                key=lambda item: item["replay_generation"],
            ))
        pair_population = [
            item for item in selected + supplements
            if item["sign"] != 0
        ]
        if not (
                any(item["sign"] > 0 for item in pair_population)
                and any(item["sign"] < 0 for item in pair_population)):
            continue

        adj_row = adjacency_by_step[step]
        base_episodes, base_chunks, chunks_per_episode = (
            _adjacency_population(adj_row)
        )
        additional_episode_keys = set(
            (item["policy_id"], item["replay_generation"])
            for item in supplements
            if not _selected_at_step(
                groups[(item["policy_id"], item["replay_generation"])],
                step,
            )
        )
        selected_episode_count = base_episodes + len(additional_episode_keys)
        selected_chunk_count = (
            base_chunks + len(additional_episode_keys) * chunks_per_episode
        )
        chosen = []
        for item in pair_population:
            key = (item["policy_id"], item["replay_generation"])
            if key in consumed:
                raise RuntimeError(
                    "dry-run attempted to reuse committed generation {}".format(
                        key
                    )
                )
            pending_age = max(
                0,
                int(round(
                    (step - item["last_available_step"])
                    / float(adjacency_update_interval)
                )),
            )
            chosen.append(OrderedDict([
                ("policy_id", item["policy_id"]),
                ("replay_generation", item["replay_generation"]),
                ("sign", item["sign"]),
                ("source", _source_at_step(
                    groups[key],
                    step,
                )),
                ("first_seen_step", item["first_available_step"]),
                ("original_replay_last_valid_step", item["last_available_step"]),
                ("prepared_step", step),
                ("pending_age_updates", pending_age),
                ("behavior_policy_version", None),
                ("policy_age", None),
                ("stale_trust", None),
                ("effective_target_mass", None),
            ]))
        logical = OrderedDict([
            ("cohort_step", step),
            ("horizon_updates", horizon_updates),
            ("evidence", chosen),
            ("positive_generation_count", sum(
                int(item["sign"] > 0) for item in pair_population
            )),
            ("negative_generation_count", sum(
                int(item["sign"] < 0) for item in pair_population
            )),
            ("selected_episode_count", selected_episode_count),
            ("selected_chunk_count", selected_chunk_count),
            ("yielded_chunk_count", selected_chunk_count),
            ("trained_chunk_count", selected_chunk_count),
            ("partition_count", 1),
            ("partition_slot", -1),
            ("partition_imbalance", 0),
            ("duplicate_generation_count", 0),
            ("dropped_generation_count", 0),
            ("positive_effective_mass", None),
            ("negative_effective_mass", None),
            ("signed_mass_contract_evaluable", False),
            ("stale_trust_contract_evaluable", False),
            ("immutable_payload_available", False),
            ("support_v6_structural_contract_valid", True),
            ("support_v6_full_training_contract_valid", False),
            (
                "training_rejection_reason",
                "evicted pair evidence lacks immutable behavior logits, masks, "
                "quality tensors, and a hard stale-trust eligibility bound",
            ),
        ])
        cohorts.append(logical)
        for item in pair_population:
            consumed.add((item["policy_id"], item["replay_generation"]))

    last_step = max(adjacency_by_step)
    expired_unconsumed = [
        item for item in lifecycles
        if (item["policy_id"], item["replay_generation"]) not in consumed
        and (
            item["last_available_step"]
            + horizon_updates * adjacency_update_interval
        ) < last_step
    ]
    return OrderedDict([
        ("extra_pending_updates", horizon_updates),
        (
            "pending_positive_generation_count",
            len(ever_pending_positive),
        ),
        (
            "pending_negative_generation_count",
            len(ever_pending_negative),
        ),
        ("max_concurrent_pending_positive_count", max_pending_positive),
        ("max_concurrent_pending_negative_count", max_pending_negative),
        ("opposite_sign_overlap_step_count", overlap_step_count),
        ("structural_class_complete_cohort_count", len(cohorts)),
        ("fully_trainable_class_complete_cohort_count", 0),
        ("consumed_generation_count", len(consumed)),
        ("prevented_reuse_generation_count", len(blocked_reuse)),
        ("reused_generation_count", 0),
        (
            "expired_unconsumed_generation_count",
            len(expired_unconsumed),
        ),
        ("stale_ineligible_generation_count", None),
        ("support_v6_structural_contract_valid", True),
        ("support_v6_full_training_contract_valid", False),
        ("cohorts", cohorts),
    ])


def reconstruct_pair_evidence_overlap(
        episode_rows,
        adjacency_rows,
        max_pending_updates=None):
    adjacency_by_step = {}
    for row in adjacency_rows:
        step = _integer(row, "step")
        if step in adjacency_by_step:
            raise ValueError(
                "adjacency CSV contains duplicate step {}".format(step)
            )
        adjacency_by_step[step] = row

    groups = OrderedDict()
    seen_row_keys = set()
    for raw in episode_rows:
        if not _flag(raw, "row_contract_valid", default=1):
            raise ValueError("pair-evidence episode row contract failed")
        if not _flag(raw, "pair_evidence_episode", default=0):
            continue
        policy_id = str(raw.get("policy_id", ""))
        generation = _integer(raw, "episode_generation")
        sign = _integer(raw, "pair_evidence_sign")
        if sign not in (-1, 1):
            raise ValueError(
                "pair evidence sign must be -1 or 1, got {}".format(sign)
            )
        step = _integer(raw, "env_step")
        row_key = (policy_id, generation, step)
        if row_key in seen_row_keys:
            raise ValueError(
                "duplicate pair-evidence exposure row {}".format(row_key)
            )
        seen_row_keys.add(row_key)
        key = (policy_id, generation)
        groups.setdefault(key, []).append(dict(raw))

    lifecycles = []
    for key, rows in groups.items():
        rows.sort(key=lambda item: _integer(item, "env_step"))
        signs = set(_integer(row, "pair_evidence_sign") for row in rows)
        if len(signs) != 1:
            raise ValueError(
                "pair evidence generation changes sign: {}".format(key)
            )
        if any(
                _flag(row, "base_selected", default=0)
                and _flag(row, "support_selected", default=0)
                for row in rows):
            raise ValueError(
                "one episode cannot be base- and support-selected together"
            )
        sign = next(iter(signs))
        first_step = _integer(rows[0], "env_step")
        last_step = _integer(rows[-1], "env_step")
        first_selected = _first_step(
            rows, lambda row: _flag(row, "selected_for_training", default=0)
        )
        first_base = _first_step(
            rows, lambda row: _flag(row, "base_selected", default=0)
        )
        first_support = _first_step(
            rows, lambda row: _flag(row, "support_selected", default=0)
        )
        first_used = _first_step(
            rows, lambda row: _flag(row, "outcome_support_used", default=0)
        )
        support_metrics = (
            _step_metrics(adjacency_by_step, first_support)
            if first_support is not None else None
        )
        used_reason = (
            _support_reason(support_metrics)
            if first_support is not None else "NOT_SUPPORT_USED"
        )
        nonzero_commit_steps = []
        for row in rows:
            step = _integer(row, "env_step")
            metrics = _step_metrics(adjacency_by_step, step)
            if metrics["pair_nonzero_target"]:
                nonzero_commit_steps.append(step)
        if first_used is not None and first_support is None:
            used_reason = "USED_BEFORE_OBSERVED_SUPPORT_SELECTION"
        lifecycle = OrderedDict([
            ("policy_id", key[0]),
            ("replay_generation", key[1]),
            ("sign", sign),
            ("outcome_success", _integer(rows[0], "outcome_success")),
            ("environment_episode_id", rows[0].get("environment_episode_id", "")),
            ("capture_event_id", rows[0].get("capture_event_id", "")),
            ("prey_id", rows[0].get("capture_prey_id", "")),
            ("participant_slots", rows[0].get("participant_slots", "")),
            ("factor_order", 2),
            ("first_available_step", first_step),
            ("pair_evidence_first_seen_step", first_step),
            ("last_available_step", last_step),
            ("exposure_count", len(rows)),
            ("first_recency_age", _integer(rows[0], "episode_recency_age")),
            ("last_recency_age", _integer(rows[-1], "episode_recency_age")),
            ("first_selected_step", first_selected),
            ("first_base_selected_step", first_base),
            ("first_support_selected_step", first_support),
            ("first_shared_support_used_step", first_used),
            ("shared_support_used_reason", used_reason),
            ("pair_nonzero_commit_steps", sorted(set(nonzero_commit_steps))),
            (
                "pair_evidence_committed_step",
                min(nonzero_commit_steps) if nonzero_commit_steps else None,
            ),
            (
                "pair_evidence_pending_age",
                _integer(rows[-1], "episode_recency_age"),
            ),
            ("pair_nonzero_consumed", bool(nonzero_commit_steps)),
            ("expired_without_pair_commit", not bool(nonzero_commit_steps)),
            ("identity_event_contract_observable", bool(
                rows[0].get("participant_slots_available", "0") == "1"
                and rows[0].get("capture_event_id_available", "0") == "1"
            )),
        ])
        lifecycles.append(lifecycle)

    lifecycles.sort(
        key=lambda item: (
            item["first_available_step"],
            item["replay_generation"],
        )
    )
    positives = [item for item in lifecycles if item["sign"] > 0]
    negatives = [item for item in lifecycles if item["sign"] < 0]
    adjacency_steps = sorted(adjacency_by_step)
    adjacency_step_differences = [
        right - left
        for left, right in zip(adjacency_steps[:-1], adjacency_steps[1:])
        if right > left
    ]
    adjacency_update_interval = (
        reduce(math.gcd, adjacency_step_differences)
        if adjacency_step_differences else 0
    )
    opportunities = []
    overlap_pairs = [
        (positive, negative)
        for positive in positives
        for negative in negatives
        if max(
            positive["first_available_step"],
            negative["first_available_step"],
        ) <= min(
            positive["last_available_step"],
            negative["last_available_step"],
        )
    ]
    overlap_pair_count = len(overlap_pairs)
    used_only_block_count = 0
    for positive in positives:
        step = positive["first_available_step"]
        present = [
            negative for negative in negatives
            if negative["first_available_step"] <= step
            <= negative["last_available_step"]
        ]
        expired = [
            negative for negative in negatives
            if negative["last_available_step"] < step
        ]
        future = [
            negative for negative in negatives
            if negative["first_available_step"] > step
        ]
        any_overlap = [
            negative for negative in negatives
            if max(
                positive["first_available_step"],
                negative["first_available_step"],
            ) <= min(
                positive["last_available_step"],
                negative["last_available_step"],
            )
        ]
        present_base = [
            item for item in present
            if item["first_base_selected_step"] == step
        ]
        present_unconsumed_support = [
            item for item in present
            if not item["pair_nonzero_consumed"]
            and (
                item["first_shared_support_used_step"] is None
                or item["first_shared_support_used_step"] > step
            )
        ]
        present_shared_used_zero_pair = [
            item for item in present
            if not item["pair_nonzero_consumed"]
            and item["first_shared_support_used_step"] is not None
            and item["first_shared_support_used_step"] <= step
        ]
        present_nonzero_consumed = [
            item for item in present if item["pair_nonzero_consumed"]
        ]
        if (
                present
                and not present_base
                and not present_unconsumed_support
                and present_shared_used_zero_pair):
            used_only_block_count += 1
        nearest_before = (
            max(expired, key=lambda item: item["last_available_step"])
            if expired else None
        )
        nearest_after = (
            min(future, key=lambda item: item["first_available_step"])
            if future else None
        )
        nearest_before_gap = (
            step - nearest_before["last_available_step"]
            if nearest_before is not None else None
        )
        nearest_after_gap = (
            nearest_after["first_available_step"]
            - positive["last_available_step"]
            if nearest_after is not None else None
        )
        before_extension_updates = (
            int(math.ceil(
                nearest_before_gap / float(adjacency_update_interval)
            ))
            if (
                nearest_before_gap is not None
                and adjacency_update_interval > 0
            ) else None
        )
        after_extension_updates = (
            int(math.ceil(
                nearest_after_gap / float(adjacency_update_interval)
            ))
            if (
                nearest_after_gap is not None
                and adjacency_update_interval > 0
            ) else None
        )
        extension_candidates = [
            value for value in (
                before_extension_updates,
                after_extension_updates,
            )
            if value is not None
        ]
        opportunities.append(OrderedDict([
            ("positive_generation", positive["replay_generation"]),
            ("positive_first_available_step", step),
            ("negative_present_count", len(present)),
            ("negative_overlap_anytime_count", len(any_overlap)),
            ("negative_base_selected_count", len(present_base)),
            (
                "negative_unconsumed_support_eligible_count",
                len(present_unconsumed_support),
            ),
            (
                "negative_shared_used_without_pair_commit_count",
                len(present_shared_used_zero_pair),
            ),
            (
                "negative_nonzero_pair_consumed_count",
                len(present_nonzero_consumed),
            ),
            ("negative_naturally_expired_count", len(expired)),
            ("negative_future_count", len(future)),
            (
                "present_negative_generations",
                [item["replay_generation"] for item in present],
            ),
            (
                "overlapping_negative_generations",
                [item["replay_generation"] for item in any_overlap],
            ),
            (
                "nearest_expired_negative_generation",
                (
                    nearest_before["replay_generation"]
                    if nearest_before is not None else None
                ),
            ),
            (
                "nearest_expired_negative_last_step",
                (
                    nearest_before["last_available_step"]
                    if nearest_before is not None else None
                ),
            ),
            (
                "nearest_expired_negative_gap_env_steps",
                nearest_before_gap,
            ),
            (
                "negative_backward_extension_updates_for_overlap",
                before_extension_updates,
            ),
            (
                "next_negative_generation",
                (
                    nearest_after["replay_generation"]
                    if nearest_after is not None else None
                ),
            ),
            (
                "next_negative_first_step",
                (
                    nearest_after["first_available_step"]
                    if nearest_after is not None else None
                ),
            ),
            (
                "positive_forward_extension_updates_for_overlap",
                after_extension_updates,
            ),
            (
                "minimum_replay_validity_extension_updates_for_overlap",
                min(extension_candidates) if extension_candidates else None,
            ),
            (
                "class_complete_possible_in_recorded_replay",
                bool(present_base or present_unconsumed_support),
            ),
            (
                "class_complete_possible_if_shared_used_were_deferred",
                bool(present_base or present_unconsumed_support
                     or present_shared_used_zero_pair),
            ),
            (
                "blocked_only_by_shared_used_state",
                bool(
                    present
                    and not present_base
                    and not present_unconsumed_support
                    and present_shared_used_zero_pair
                ),
            ),
        ]))

    shared_zero_pair = [
        item for item in lifecycles
        if item["first_shared_support_used_step"] is not None
        and not item["pair_nonzero_consumed"]
    ]
    nonzero_committed = [
        item for item in lifecycles if item["pair_nonzero_consumed"]
    ]
    reused_after_commit_count = 0
    for item in nonzero_committed:
        commit_step = min(item["pair_nonzero_commit_steps"])
        if item["last_available_step"] > commit_step:
            reused_after_commit_count += sum(
                1 for row in groups[
                    (item["policy_id"], item["replay_generation"])
                ]
                if _integer(row, "env_step") > commit_step
                and _flag(row, "support_selected", default=0)
            )

    contract_valid = bool(
        reused_after_commit_count == 0
        and all(
            item["first_available_step"] <= item["last_available_step"]
            for item in lifecycles
        )
    )
    opportunity_extension_values = [
        item["minimum_replay_validity_extension_updates_for_overlap"]
        for item in opportunities
        if item["minimum_replay_validity_extension_updates_for_overlap"]
        is not None
    ]
    automatic_sweep_max = (
        max(opportunity_extension_values)
        if opportunity_extension_values else 0
    )
    if max_pending_updates is not None:
        max_pending_updates = int(max_pending_updates)
        if max_pending_updates < 0:
            raise ValueError("max_pending_updates must be non-negative")
        automatic_sweep_max = max_pending_updates
    sweep_update_interval = max(1, adjacency_update_interval)
    horizon_sweep = [
        _run_pending_horizon(
            horizon_updates=horizon,
            lifecycles=lifecycles,
            groups=groups,
            adjacency_by_step=adjacency_by_step,
            adjacency_update_interval=sweep_update_interval,
        )
        for horizon in range(automatic_sweep_max + 1)
    ]
    first_structural = next(
        (
            item["extra_pending_updates"]
            for item in horizon_sweep
            if item["structural_class_complete_cohort_count"] >= 1
        ),
        None,
    )
    second_structural = next(
        (
            item["extra_pending_updates"]
            for item in horizon_sweep
            if item["structural_class_complete_cohort_count"] >= 2
        ),
        None,
    )
    stale_min_weights = set(
        _number(
            row,
            "adj_ppo_stale_trust_min_weight",
            default=0.0,
        )
        for row in adjacency_rows
    )
    if len(stale_min_weights) != 1:
        raise ValueError(
            "run changes adj_ppo_stale_trust_min_weight during training"
        )
    stale_min_weight = next(iter(stale_min_weights))
    # A positive or zero floor is not itself a hard age cutoff: the
    # exponential rule has no finite age at which eligibility is revoked.
    stale_trust_hard_cutoff_present = False
    # Zero is the only extension already covered by the current replay and
    # checkpoint contracts.  Any post-replay lifetime needs a new, explicit
    # safety contract rather than being inferred from the trust floor.
    safe_max_pending_updates = 0
    bounded_pending_training_safe = False
    return OrderedDict([
        ("diagnostic_version", DIAGNOSTIC_VERSION),
        ("enabled", True),
        ("mode", "pair_evidence_existing_replay_overlap"),
        ("horizon_contract", "support_v6_structural_fail_closed"),
        ("training_path_imported", False),
        ("trajectory_neutral", True),
        ("pair_evidence_consumption_version", 1),
        ("generation_event_single_use_audited", True),
        ("evidence_generation_count", len(lifecycles)),
        ("positive_generation_count", len(positives)),
        ("negative_generation_count", len(negatives)),
        (
            "pair_evidence_pending_positive_count",
            sum(int(not item["pair_nonzero_consumed"]) for item in positives),
        ),
        (
            "pair_evidence_pending_negative_count",
            sum(int(not item["pair_nonzero_consumed"]) for item in negatives),
        ),
        ("adjacency_update_interval", adjacency_update_interval),
        ("opposite_sign_overlap_pair_count", overlap_pair_count),
        ("positive_opportunity_count", len(opportunities)),
        (
            "positive_with_recorded_class_complete_opportunity_count",
            sum(
                int(item["class_complete_possible_in_recorded_replay"])
                for item in opportunities
            ),
        ),
        (
            "positive_blocked_only_by_shared_used_state_count",
            used_only_block_count,
        ),
        (
            "pair_evidence_one_sided_zero_count",
            sum(
                int(not item["pair_nonzero_consumed"])
                for item in lifecycles
            ),
        ),
        (
            "pair_evidence_zero_target_used_commit_count",
            0,
        ),
        (
            "pair_evidence_shared_support_used_without_pair_target_count",
            len(shared_zero_pair),
        ),
        (
            "pair_evidence_nonzero_target_commit_count",
            len(nonzero_committed),
        ),
        (
            "pair_evidence_deferred_count",
            sum(
                item["negative_shared_used_without_pair_commit_count"]
                for item in opportunities
            ),
        ),
        (
            "pair_evidence_expired_unconsumed_count",
            sum(
                int(item["expired_without_pair_commit"])
                for item in lifecycles
            ),
        ),
        (
            "pair_evidence_reused_after_commit_count",
            reused_after_commit_count,
        ),
        ("pair_evidence_commit_contract_valid", contract_valid),
        ("horizon_sweep_max_updates", automatic_sweep_max),
        (
            "minimum_pending_updates_for_first_structural_class_complete",
            first_structural,
        ),
        (
            "minimum_pending_updates_for_two_structural_class_complete",
            second_structural,
        ),
        (
            "minimum_pending_updates_for_first_fully_trainable_class_complete",
            None,
        ),
        (
            "minimum_pending_updates_for_two_fully_trainable_class_complete",
            None,
        ),
        ("stale_trust_min_weight", stale_min_weight),
        (
            "stale_trust_hard_cutoff_present",
            stale_trust_hard_cutoff_present,
        ),
        ("stale_trust_reaches_zero", False),
        ("pair_behavior_policy_age_logged", False),
        ("immutable_evicted_pair_payload_available", False),
        ("pending_checkpoint_state_available", False),
        (
            "full_training_missing_contracts",
            [
                "immutable_evicted_pair_payload",
                "pair_behavior_policy_version_and_age",
                "finite_hard_stale_eligibility_bound",
                "checkpointed_pending_state",
                "counterfactual_signed_pair_mass",
            ],
        ),
        ("existing_safe_max_pending_updates", safe_max_pending_updates),
        ("bounded_pending_training_safe", bounded_pending_training_safe),
        (
            "bounded_pending_training_rejection_reason",
            "the current trust rule floors sample weight above zero and has "
            "no hard policy-age cutoff; run107 also lacks immutable evicted "
            "pair payload and checkpointed pending state",
        ),
        ("evidence_lifecycles", lifecycles),
        ("positive_opportunities", opportunities),
        ("horizon_sweep", horizon_sweep),
    ])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-csv",
        required=True,
        help="runXXX_progress_train_pair_evidence_episode.csv",
    )
    parser.add_argument(
        "--adj-csv",
        required=True,
        help="runXXX_progress_train_adj.csv",
    )
    parser.add_argument(
        "--max-pending-updates",
        type=int,
        default=None,
        help=(
            "Explicit inclusive structural horizon sweep limit. By default "
            "the tool sweeps through the largest nearest-opposite-sign gap."
        ),
    )
    args = parser.parse_args(argv)
    result = reconstruct_pair_evidence_overlap(
        _read_csv(args.episode_csv),
        _read_csv(args.adj_csv),
        max_pending_updates=args.max_pending_updates,
    )
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
