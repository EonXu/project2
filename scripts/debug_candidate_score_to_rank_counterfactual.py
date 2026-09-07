#!/usr/bin/env python3
"""Opt-in, read-only counterfactuals for candidate rank and active boundary.

The default mode analyzes one optimizer transaction at a time.  The optional
independent-generation mode is deliberately stricter: it only accepts one row
per real capture generation and refuses incomplete or repeated provenance.
Neither mode imports the training path.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable",
        action="store_true",
        default=False,
        help="Run the read-only counterfactual. The default is disabled.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Per-target candidate identity transaction CSV.",
    )
    parser.add_argument(
        "--provenance-csv",
        type=Path,
        help=(
            "Event-level candidate evidence provenance CSV. Required by "
            "--independent-generation-accumulation."
        ),
    )
    parser.add_argument(
        "--successful-overlap-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--independent-generation-accumulation",
        action="store_true",
        default=False,
        help=(
            "Accumulate distinct capture generations for one exact identity. "
            "This mode requires one provenance-complete row per generation."
        ),
    )
    parser.add_argument(
        "--identity",
        help=(
            "Optional exact canonical identity filter. It is useful when an "
            "input file contains several identities."
        ),
    )
    return parser


def _finite_float(row, name):
    if name not in row or row[name] in (None, ""):
        raise RuntimeError(
            "candidate counterfactual is missing required field {}".format(
                name
            )
        )
    value = float(row[name])
    if not math.isfinite(value):
        raise RuntimeError(
            "candidate counterfactual field {} is non-finite".format(name)
        )
    return value


def _required_text(row, name):
    if name not in row or row[name] is None or not str(row[name]).strip():
        raise RuntimeError(
            "candidate counterfactual is missing required provenance field "
            "{}".format(name)
        )
    return str(row[name]).strip()


def _positive_row_counterfactual(row):
    target_sign = _finite_float(row, "target_sign")
    if target_sign <= 0.0:
        raise RuntimeError(
            "run106 CSV summary supports positive rank sensitivity only; "
            "use the full-population helper for a negative target"
        )
    pre_margin = _finite_float(row, "pre_margin")
    post_margin = _finite_float(row, "post_margin")
    pre_gap = _finite_float(row, "pre_next_better_margin_gap")
    post_gap = _finite_float(row, "post_next_better_margin_gap")
    improvement = _finite_float(row, "signed_margin_change")
    boundary_deficit = _finite_float(row, "pre_boundary_deficit")
    pre_rank = int(row["pre_rank"])
    post_rank = int(row["post_rank"])
    if row["same_population_rank_reconstruction_valid"] != "1":
        raise RuntimeError("candidate rank population did not reconstruct")
    if pre_gap < 0.0 or post_gap < 0.0 or boundary_deficit < 0.0:
        raise RuntimeError("candidate counterfactual has a negative gap")

    next_margin_before = pre_margin + pre_gap
    next_margin_after = post_margin + post_gap
    next_margin_change = next_margin_after - next_margin_before
    gap_shrink = pre_gap - post_gap
    rank_stall_reason = "NO_DIRECTIONAL_IMPROVEMENT"
    if (
            improvement > 0.0
            and pre_gap > improvement
            and post_gap > 0.0
            and post_rank == pre_rank):
        rank_stall_reason = "IMPROVEMENT_BELOW_NEXT_BETTER_GAP"
    if (
            improvement > 0.0
            and next_margin_change >= improvement
            and post_gap >= pre_gap):
        rank_stall_reason = "NEXT_BETTER_MOVED_AT_LEAST_AS_FAST"
    if post_rank != pre_rank:
        rank_stall_reason = "RANK_CHANGED"

    def _steps(required):
        if required <= 0.0:
            return 0
        if improvement <= 0.0:
            return None
        return int(math.ceil(required / improvement))

    return {
        "env_step": int(row["env_step"]),
        "ppo_epoch_index": int(row["ppo_epoch_index"]),
        "transaction_sequence_index": int(
            row["transaction_sequence_index"]
        ),
        "episode_generation": int(row["episode_generation"]),
        "canonical_identity": row["canonical_identity"],
        "candidate_index": int(row["candidate_index"]),
        "next_better_candidate_index": int(
            row["pre_next_better_candidate_index"]
        ),
        "pre_rank": pre_rank,
        "post_rank": post_rank,
        "valid_population_count": int(
            row["pre_valid_population_count"]
        ),
        "target_margin_change": improvement,
        "next_better_margin_change": next_margin_change,
        "pre_next_better_gap": pre_gap,
        "post_next_better_gap": post_gap,
        "relative_gap_shrink": gap_shrink,
        "observed_fraction_of_next_gap": (
            improvement / pre_gap if pre_gap > 0.0 else None
        ),
        "observed_updates_to_next_rank_step": _steps(pre_gap),
        "pre_boundary_deficit": boundary_deficit,
        "observed_fraction_of_boundary_deficit": (
            improvement / boundary_deficit
            if boundary_deficit > 0.0 else None
        ),
        "observed_updates_to_active_boundary": _steps(boundary_deficit),
        "rank_stall_reason": rank_stall_reason,
        "read_only_contract_valid": True,
    }


def _candidate_evidence_join_key(row, provenance):
    target_sign = _finite_float(row, "target_sign")
    if target_sign == 0.0:
        raise RuntimeError(
            "candidate evidence join target_sign must be nonzero"
        )
    target_sign = 1 if target_sign > 0.0 else -1
    if provenance:
        return (
            _required_text(row, "policy_id"),
            int(_finite_float(row, "replay_generation")),
            int(_finite_float(row, "environment_episode_id")),
            int(_finite_float(row, "capture_event_id")),
            int(_finite_float(row, "prey_id")),
            _required_text(row, "candidate_identity"),
            target_sign,
        )
    return (
        _required_text(row, "policy_id"),
        int(_finite_float(row, "episode_generation")),
        int(_finite_float(row, "environment_episode_id")),
        int(_finite_float(row, "capture_event_id")),
        int(_finite_float(row, "capture_prey_id")),
        _required_text(row, "canonical_identity"),
        target_sign,
    )


def _expand_candidate_transaction_event_members(row):
    """Expand one target-level transaction into mass-conserving event views."""
    if "candidate_event_group_size" not in row:
        legacy = dict(row)
        legacy["_event_attribution_fraction"] = 1.0
        return [legacy]
    group_size = int(_finite_float(row, "candidate_event_group_size"))
    if group_size <= 0:
        raise RuntimeError("candidate transaction event group is empty")
    grouped_fields = (
        ("capture_event_ids", int),
        ("capture_prey_ids", int),
        ("candidate_event_target_masses", float),
        ("candidate_participant_slot_groups", str),
        ("candidate_static_dynamic_classes", str),
    )
    parsed = {}
    for field, converter in grouped_fields:
        values = str(row.get(field, "")).split("|")
        if len(values) != group_size or any(value == "" for value in values):
            raise RuntimeError(
                "candidate transaction event group field {} has invalid "
                "cardinality".format(field)
            )
        parsed[field] = [converter(value) for value in values]
    event_keys = list(zip(
        parsed["capture_event_ids"],
        parsed["capture_prey_ids"],
    ))
    if len(set(event_keys)) != group_size:
        raise RuntimeError(
            "candidate transaction event group duplicates a real event"
        )
    masses = parsed["candidate_event_target_masses"]
    if any(not math.isfinite(mass) or mass <= 0.0 for mass in masses):
        raise RuntimeError(
            "candidate transaction event group contains invalid target mass"
        )
    grouped_mass = sum(masses)
    target_weight = _finite_float(row, "target_weight")
    if not math.isclose(
            grouped_mass,
            target_weight,
            rel_tol=0.0,
            abs_tol=1e-6):
        raise RuntimeError(
            "candidate transaction event group does not reconstruct target "
            "weight"
        )
    expanded = []
    for index in range(group_size):
        member = dict(row)
        member["capture_event_id"] = parsed["capture_event_ids"][index]
        member["capture_prey_id"] = parsed["capture_prey_ids"][index]
        member["target_weight"] = masses[index]
        member["candidate_participant_slots"] = parsed[
            "candidate_participant_slot_groups"
        ][index]
        member["candidate_static_dynamic_class"] = parsed[
            "candidate_static_dynamic_classes"
        ][index]
        member["_event_attribution_fraction"] = (
            masses[index] / grouped_mass
        )
        expanded.append(member)
    return expanded


def _validate_candidate_evidence_quality_contract(provenance_rows):
    """Reconstruct event quality before any identity filtering."""
    exact_keys = set()
    event_groups = {}
    for row in provenance_rows:
        for field in (
                "provenance_complete",
                "identity_contract_valid",
                "quality_contract_valid",
                "consumed_generation_once"):
            if int(_finite_float(row, field)) != 1:
                raise RuntimeError(
                    "candidate evidence provenance contract {} is invalid"
                    .format(field)
                )
        if (
                int(_finite_float(row, "duplicate_generation")) != 0
                or int(_finite_float(row, "duplicate_event")) != 0):
            raise RuntimeError(
                "candidate evidence provenance CSV contains a duplicate row"
            )
        key = _candidate_evidence_join_key(row, provenance=True)
        if key in exact_keys:
            raise RuntimeError(
                "duplicate candidate evidence provenance key {}".format(key)
            )
        exact_keys.add(key)
        event_key = key[:5]
        event_groups.setdefault(event_key, []).append(row)

    for event_key, event_rows in event_groups.items():
        raw_quality = _finite_float(
            event_rows[0],
            "raw_event_quality",
        )
        coefficient = _finite_float(
            event_rows[0],
            "candidate_coefficient",
        )
        if raw_quality < 0.0 or coefficient < 0.0:
            raise RuntimeError(
                "candidate event quality/coefficient must be non-negative"
            )
        expected = raw_quality * coefficient
        actual = sum(
            _finite_float(row, "final_target_mass")
            for row in event_rows
        )
        identity_weight_sum = sum(
            _finite_float(row, "identity_event_weight")
            for row in event_rows
        )
        if not math.isclose(
                identity_weight_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6):
            raise RuntimeError(
                "candidate event identity weights do not sum to one for "
                "event {}".format(event_key)
            )
        for row in event_rows:
            identity_weight = _finite_float(
                row,
                "identity_event_weight",
            )
            allocated = _finite_float(
                row,
                "identity_allocated_quality",
            )
            final_mass = _finite_float(row, "final_target_mass")
            if identity_weight <= 0.0 or final_mass <= 0.0:
                raise RuntimeError(
                    "candidate event identity weight/mass must be positive"
                )
            if (
                    not math.isclose(
                        allocated,
                        raw_quality * identity_weight,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        final_mass,
                        allocated * coefficient,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )):
                raise RuntimeError(
                    "candidate event identity quality does not reconstruct "
                    "for event {}".format(event_key)
                )
        for row in event_rows[1:]:
            if (
                    not math.isclose(
                        _finite_float(row, "raw_event_quality"),
                        _finite_float(
                            event_rows[0],
                            "raw_event_quality",
                        ),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        _finite_float(row, "candidate_coefficient"),
                        _finite_float(
                            event_rows[0],
                            "candidate_coefficient",
                        ),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )):
                raise RuntimeError(
                    "candidate event quality/coefficient differs across "
                    "identities for event {}".format(event_key)
                )
        if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-6):
            raise RuntimeError(
                "candidate event quality does not reconstruct for event {}: "
                "{} versus {}".format(event_key, actual, expected)
            )
    return {
        "unique_event_count": len(event_groups),
        "unique_evidence_row_count": len(exact_keys),
        "quality_contract_valid": True,
    }


def _join_candidate_evidence_provenance(
        provenance_rows,
        transaction_rows):
    """Join real events to all PPO transactions without counting epochs twice."""
    transaction_by_key = {}
    transaction_sequences = set()
    for row in transaction_rows:
        sequence = int(_finite_float(row, "transaction_sequence_index"))
        target_sequence = int(_finite_float(
            row,
            "target_row_sequence_within_transaction",
        ))
        unique_transaction_target = (sequence, target_sequence)
        if unique_transaction_target in transaction_sequences:
            raise RuntimeError(
                "candidate transaction target row is duplicated"
            )
        transaction_sequences.add(unique_transaction_target)
        for event_member in _expand_candidate_transaction_event_members(row):
            key = _candidate_evidence_join_key(
                event_member,
                provenance=False,
            )
            transaction_by_key.setdefault(key, []).append(event_member)

    normalized = []
    for provenance_row in provenance_rows:
        key = _candidate_evidence_join_key(
            provenance_row,
            provenance=True,
        )
        all_matches = transaction_by_key.get(key, ())
        first_consumed_update = int(_finite_float(
            provenance_row,
            "first_consumed_update",
        ))
        matches = [
            match for match in all_matches
            if int(_finite_float(match, "env_step")) == first_consumed_update
        ]
        if not matches:
            raise RuntimeError(
                "candidate evidence provenance has no optimizer transaction "
                "join at its first consumed update {} for key {}"
                .format(first_consumed_update, key)
            )
        seen_epochs = set()
        for match in matches:
            epoch_key = (
                int(_finite_float(match, "env_step")),
                int(_finite_float(match, "ppo_epoch_index")),
                int(_finite_float(match, "partition_index")),
            )
            if epoch_key in seen_epochs:
                raise RuntimeError(
                    "two transaction rows expose the same PPO epoch for one "
                    "real candidate event"
                )
            seen_epochs.add(epoch_key)
            if int(_finite_float(match, "factor_order")) != int(
                    _finite_float(provenance_row, "candidate_order")):
                raise RuntimeError(
                    "candidate evidence order conflicts with transaction"
                )
            if (
                    _required_text(
                        match,
                        "candidate_participant_slots",
                    )
                    != _required_text(
                        provenance_row,
                        "participant_slots",
                    )):
                raise RuntimeError(
                    "candidate evidence participants conflict with transaction"
                )
            if (
                    _required_text(
                        match,
                        "candidate_static_dynamic_class",
                    )
                    != _required_text(
                        provenance_row,
                        "static_dynamic_class",
                    )):
                raise RuntimeError(
                    "candidate evidence static/dynamic class conflicts with "
                    "transaction"
                )
            if not math.isclose(
                    _finite_float(match, "target_weight"),
                    _finite_float(provenance_row, "final_target_mass"),
                    rel_tol=0.0,
                    abs_tol=1e-6):
                raise RuntimeError(
                    "candidate evidence target mass conflicts with transaction"
                )

        matches = sorted(
            matches,
            key=lambda row: (
                int(_finite_float(row, "env_step")),
                int(_finite_float(row, "ppo_epoch_index")),
                int(_finite_float(row, "partition_index")),
            ),
        )
        first = matches[0]
        normalized.append({
            "episode_generation": int(_finite_float(
                provenance_row,
                "replay_generation",
            )),
            "environment_episode_id": str(int(_finite_float(
                provenance_row,
                "environment_episode_id",
            ))),
            "capture_event_id": str(int(_finite_float(
                provenance_row,
                "capture_event_id",
            ))),
            "capture_prey_id": str(int(_finite_float(
                provenance_row,
                "prey_id",
            ))),
            "canonical_identity": _required_text(
                provenance_row,
                "candidate_identity",
            ),
            "participant_slots": _required_text(
                provenance_row,
                "participant_slots",
            ),
            "static_dynamic_identity": _required_text(
                provenance_row,
                "static_dynamic_class",
            ),
            "factor_order": int(_finite_float(
                provenance_row,
                "candidate_order",
            )),
            "target_sign": int(math.copysign(
                1,
                _finite_float(provenance_row, "target_sign"),
            )),
            "target_weight": _finite_float(
                provenance_row,
                "final_target_mass",
            ),
            "first_consumed_update": first_consumed_update,
            "signed_margin_change": sum(
                _finite_float(match, "signed_margin_change")
                * float(match.get("_event_attribution_fraction", 1.0))
                for match in matches
            ),
            "pre_next_better_candidate_index": int(_finite_float(
                first,
                "pre_next_better_candidate_index",
            )),
            "pre_next_better_margin_gap": _finite_float(
                first,
                "pre_next_better_margin_gap",
            ),
            "pre_boundary_deficit": _finite_float(
                first,
                "pre_boundary_deficit",
            ),
            "pre_valid_population_count": int(_finite_float(
                first,
                "pre_valid_population_count",
            )),
            "joined_ppo_transaction_count": len(matches),
            "later_replay_transaction_row_count": (
                len(all_matches) - len(matches)
            ),
            "first_consumption_only": True,
            "quality_contract_valid": 1,
        })
    return normalized


def _collapse_candidate_evidence_by_generation(rows):
    """Count one generation once while retaining distinct real event mass."""
    grouped = {}
    for row in rows:
        key = (
            int(row["episode_generation"]),
            str(row["canonical_identity"]),
            int(row["target_sign"]),
        )
        grouped.setdefault(key, []).append(row)
    collapsed = []
    for key, generation_rows in grouped.items():
        first = generation_rows[0]
        context_fields = (
            "pre_next_better_candidate_index",
            "pre_next_better_margin_gap",
            "pre_boundary_deficit",
            "pre_valid_population_count",
            "participant_slots",
            "static_dynamic_identity",
            "factor_order",
        )
        for row in generation_rows[1:]:
            for field in context_fields:
                left = first[field]
                right = row[field]
                if isinstance(left, float):
                    matches = math.isclose(
                        left,
                        right,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                else:
                    matches = left == right
                if not matches:
                    raise RuntimeError(
                        "one generation contains multiple candidate evidence "
                        "events with incompatible rank contexts; cannot assign "
                        "one generation-level reachability step"
                    )
        collapsed_row = dict(first)
        collapsed_row["capture_event_id"] = ";".join(
            str(row["capture_event_id"]) for row in generation_rows
        )
        collapsed_row["capture_prey_id"] = ";".join(
            str(row["capture_prey_id"]) for row in generation_rows
        )
        collapsed_row["target_weight"] = sum(
            float(row["target_weight"]) for row in generation_rows
        )
        collapsed_row["signed_margin_change"] = sum(
            float(row["signed_margin_change"]) for row in generation_rows
        )
        collapsed_row["joined_ppo_transaction_count"] = sum(
            int(row["joined_ppo_transaction_count"])
            for row in generation_rows
        )
        collapsed_row["later_replay_transaction_row_count"] = sum(
            int(row["later_replay_transaction_row_count"])
            for row in generation_rows
        )
        collapsed.append(collapsed_row)
    return collapsed


def _independent_generation_counterfactual(rows):
    """Accumulate real, distinct capture generations for one exact identity.

    This is a scalar reachability counterfactual.  It does not assume that the
    competing population stays fixed: every step reports the population size,
    nearest competitor, and boundary that were observed for that generation.
    The cumulative signed margin is therefore an isolation diagnostic, not a
    synthetic training target.
    """
    if not rows:
        raise RuntimeError(
            "independent candidate evidence counterfactual selected no rows"
        )
    input_generations = [
        int(_finite_float(row, "episode_generation"))
        for row in rows
    ]
    if len(set(input_generations)) != len(input_generations):
        raise RuntimeError(
            "duplicate candidate evidence generation; PPO epochs from one "
            "generation cannot be treated as independent evidence"
        )
    if len(input_generations) < 2:
        raise RuntimeError(
            "independent candidate evidence accumulation requires at least "
            "two distinct generations; replaying one generation is not "
            "independent evidence"
        )

    required_text_fields = (
        "environment_episode_id",
        "capture_event_id",
        "capture_prey_id",
        "canonical_identity",
        "participant_slots",
        "static_dynamic_identity",
    )
    required_float_fields = (
        "episode_generation",
        "factor_order",
        "target_sign",
        "target_weight",
        "signed_margin_change",
        "pre_next_better_candidate_index",
        "pre_next_better_margin_gap",
        "pre_boundary_deficit",
        "pre_valid_population_count",
    )

    parsed = []
    generations = set()
    events = set()
    reference_identity = None
    reference_sign = None
    reference_order = None
    reference_participants = None
    reference_static_dynamic = None
    for row in rows:
        text = {
            name: _required_text(row, name)
            for name in required_text_fields
        }
        numeric = {
            name: _finite_float(row, name)
            for name in required_float_fields
        }
        generation = int(numeric["episode_generation"])
        factor_order = int(numeric["factor_order"])
        target_sign = float(numeric["target_sign"])
        if generation < 0:
            raise RuntimeError("candidate evidence generation must be non-negative")
        if factor_order not in (2, 3):
            raise RuntimeError("candidate evidence factor_order must be 2 or 3")
        if target_sign == 0.0:
            raise RuntimeError("candidate evidence target_sign must be nonzero")
        target_sign = 1.0 if target_sign > 0.0 else -1.0
        if numeric["target_weight"] <= 0.0:
            raise RuntimeError("candidate evidence target_weight must be positive")
        if (
                numeric["pre_next_better_margin_gap"] < 0.0
                or numeric["pre_boundary_deficit"] < 0.0):
            raise RuntimeError("candidate evidence has a negative reachability gap")
        if numeric["pre_valid_population_count"] < 1.0:
            raise RuntimeError(
                "candidate evidence has no valid candidate population"
            )

        if generation in generations:
            raise RuntimeError(
                "duplicate candidate evidence generation {}".format(generation)
            )
        generations.add(generation)
        event_key = (
            text["environment_episode_id"],
            text["capture_event_id"],
        )
        if event_key in events:
            raise RuntimeError(
                "duplicate candidate evidence event {}".format(event_key)
            )
        events.add(event_key)

        identity = text["canonical_identity"]
        participants = text["participant_slots"]
        static_dynamic = text["static_dynamic_identity"]
        if reference_identity is None:
            reference_identity = identity
            reference_sign = target_sign
            reference_order = factor_order
            reference_participants = participants
            reference_static_dynamic = static_dynamic
        elif identity != reference_identity:
            raise RuntimeError(
                "independent evidence identities differ: {} versus {}".format(
                    reference_identity,
                    identity,
                )
            )
        elif target_sign != reference_sign:
            raise RuntimeError("independent evidence target signs conflict")
        elif factor_order != reference_order:
            raise RuntimeError("independent evidence factor orders conflict")
        elif participants != reference_participants:
            raise RuntimeError(
                "independent evidence participant identities conflict"
            )
        elif static_dynamic != reference_static_dynamic:
            raise RuntimeError(
                "independent evidence static/dynamic identities conflict"
            )

        parsed.append({
            "generation": generation,
            "environment_episode_id": text["environment_episode_id"],
            "capture_event_id": text["capture_event_id"],
            "capture_prey_id": text["capture_prey_id"],
            "canonical_identity": identity,
            "participant_slots": participants,
            "static_dynamic_identity": static_dynamic,
            "factor_order": factor_order,
            "target_sign": target_sign,
            "target_weight": float(numeric["target_weight"]),
            "signed_margin_change": float(
                numeric["signed_margin_change"]
            ),
            "next_better_candidate_index": int(
                numeric["pre_next_better_candidate_index"]
            ),
            "next_better_gap": float(
                numeric["pre_next_better_margin_gap"]
            ),
            "boundary_deficit": float(
                numeric["pre_boundary_deficit"]
            ),
            "valid_population_count": int(
                numeric["pre_valid_population_count"]
            ),
            "later_replay_transaction_row_count": int(float(
                row.get("later_replay_transaction_row_count", 0)
            )),
        })

    parsed.sort(key=lambda item: item["generation"])

    cumulative_change = 0.0
    cumulative_mass = 0.0
    direction_cancellation_count = 0
    first_next_crossing_count = None
    first_boundary_reach_count = None
    previous_population_count = None
    previous_competitor = None
    steps = []
    for index, item in enumerate(parsed, start=1):
        signed_change = item["signed_margin_change"]
        cumulative_change += signed_change
        cumulative_mass += item["target_weight"]
        if signed_change <= 0.0:
            direction_cancellation_count += 1
        next_gap = item["next_better_gap"]
        boundary_deficit = item["boundary_deficit"]
        next_reachable = cumulative_change > next_gap
        boundary_reachable = cumulative_change > boundary_deficit
        if next_reachable and first_next_crossing_count is None:
            first_next_crossing_count = index
        if boundary_reachable and first_boundary_reach_count is None:
            first_boundary_reach_count = index
        population_changed = (
            previous_population_count is not None
            and (
                item["valid_population_count"] != previous_population_count
                or item["next_better_candidate_index"] != previous_competitor
            )
        )
        steps.append({
            "independent_generation_count": index,
            "episode_generation": item["generation"],
            "environment_episode_id": item["environment_episode_id"],
            "capture_event_id": item["capture_event_id"],
            "capture_prey_id": item["capture_prey_id"],
            "signed_margin_change": signed_change,
            "cumulative_signed_margin_change": cumulative_change,
            "target_weight": item["target_weight"],
            "cumulative_target_weight": cumulative_mass,
            "valid_population_count": item["valid_population_count"],
            "next_better_candidate_index": (
                item["next_better_candidate_index"]
            ),
            "next_better_gap": next_gap,
            "boundary_deficit": boundary_deficit,
            "cumulative_fraction_of_next_gap": (
                cumulative_change / next_gap if next_gap > 0.0 else None
            ),
            "cumulative_fraction_of_boundary_deficit": (
                cumulative_change / boundary_deficit
                if boundary_deficit > 0.0 else None
            ),
            "counterfactual_next_rank_reachable": next_reachable,
            "counterfactual_boundary_reachable": boundary_reachable,
            "population_or_competitor_changed": population_changed,
        })
        previous_population_count = item["valid_population_count"]
        previous_competitor = item["next_better_candidate_index"]

    return {
        "diagnostic_version": 3,
        "mode": "independent_generation_accumulation",
        "enabled": True,
        "accepted": True,
        "training_path_imported": False,
        "trajectory_neutral": True,
        "canonical_identity": reference_identity,
        "factor_order": reference_order,
        "participant_slots": reference_participants,
        "static_dynamic_identity": reference_static_dynamic,
        "target_sign": reference_sign,
        "identity_contract_valid": True,
        "sign_contract_valid": True,
        "order_contract_valid": True,
        "participant_contract_valid": True,
        "provenance_join_contract_valid": True,
        "first_consumption_only": True,
        "later_replay_transaction_row_count": sum(
            int(item["later_replay_transaction_row_count"])
            for item in parsed
        ),
        "independent_generation_count": len(parsed),
        "unique_event_count": len(events),
        "cumulative_signed_margin_change": cumulative_change,
        "cumulative_target_weight": cumulative_mass,
        "direction_cancellation_count": direction_cancellation_count,
        "first_counterfactual_next_rank_crossing_count": (
            first_next_crossing_count
        ),
        "first_counterfactual_boundary_reach_count": (
            first_boundary_reach_count
        ),
        "population_context_changed": any(
            step["population_or_competitor_changed"] for step in steps
        ),
        "scalar_transfer_isolation_only": True,
        "quality_contract_valid": (
            len(generations) == len(parsed)
            and len(events) == len(parsed)
            and math.isclose(
                cumulative_mass,
                sum(item["target_weight"] for item in parsed),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "steps": steps,
    }


def main():
    args = _parser().parse_args()
    if not args.enable:
        print(
            "candidate score-to-rank counterfactual version=1; enabled=false; "
            "training_path_imported=false; trajectory_neutral=true"
        )
        return 0
    if args.csv is None:
        raise SystemExit("--csv is required when --enable is set")

    with args.csv.open("r", newline="", encoding="utf-8-sig") as handle:
        transaction_rows = list(csv.DictReader(handle))

    if args.independent_generation_accumulation:
        try:
            if args.provenance_csv is None:
                raise RuntimeError(
                    "--provenance-csv is required for independent-generation "
                    "accumulation"
                )
            with args.provenance_csv.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig") as handle:
                provenance_rows = list(csv.DictReader(handle))
            quality_summary = _validate_candidate_evidence_quality_contract(
                provenance_rows
            )
            if args.successful_overlap_only:
                provenance_rows = [
                    row for row in provenance_rows
                    if int(_finite_float(row, "outcome_success")) == 1
                ]
            if args.identity:
                provenance_rows = [
                    row for row in provenance_rows
                    if row.get("candidate_identity") == args.identity
                ]
            if not provenance_rows:
                raise RuntimeError(
                    "independent counterfactual selected no provenance rows"
                )
            normalized_rows = _join_candidate_evidence_provenance(
                provenance_rows=provenance_rows,
                transaction_rows=transaction_rows,
            )
            normalized_rows = _collapse_candidate_evidence_by_generation(
                normalized_rows
            )
            summary = _independent_generation_counterfactual(normalized_rows)
            summary["provenance_quality_contract"] = quality_summary
            summary["joined_provenance_row_count"] = len(provenance_rows)
            summary["joined_transaction_row_count"] = sum(
                int(row["joined_ppo_transaction_count"])
                for row in normalized_rows
            )
        except RuntimeError as error:
            print(json.dumps({
                "diagnostic_version": 3,
                "mode": "independent_generation_accumulation",
                "enabled": True,
                "accepted": False,
                "training_path_imported": False,
                "trajectory_neutral": True,
                "rejection_reason": str(error),
            }, indent=2, sort_keys=True))
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    rows = transaction_rows
    if args.successful_overlap_only:
        rows = [
            row for row in rows
            if _finite_float(
                row,
                "successful_candidate_capture_overlap",
            ) > 0.0
        ]
    if args.identity:
        rows = [
            row for row in rows
            if row.get("canonical_identity") == args.identity
        ]
    if not rows:
        raise RuntimeError("candidate counterfactual selected no rows")
    results = [_positive_row_counterfactual(row) for row in rows]
    summary = {
        "diagnostic_version": 1,
        "enabled": True,
        "training_path_imported": False,
        "trajectory_neutral": True,
        "row_count": len(results),
        "unique_generations": len(set(
            row["episode_generation"] for row in results
        )),
        "unique_identities": len(set(
            row["canonical_identity"] for row in results
        )),
        "rank_stall_reasons": {},
        "rows": results,
    }
    for row in results:
        reason = row["rank_stall_reason"]
        summary["rank_stall_reasons"][reason] = (
            summary["rank_stall_reasons"].get(reason, 0) + 1
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
