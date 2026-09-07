"""Offline run168/run169 frontier-rerank analysis.

Reads completed CSV artifacts only.  It does not import the Wolfpack
environment, load a checkpoint, run a policy forward, or consume RNG.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


ACTION_DELTA = np.asarray([
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1],
    [0, 0],
    [0, 0],
    [0, 0],
], dtype=np.int64)
GRID_SIZE = 20
SIGHT_RADIUS = 8


def parts(value, separator=";"):
    if pd.isna(value) or str(value) == "":
        return []
    return str(value).split(separator)


def ints(value, separator=";"):
    return [int(float(item)) for item in parts(value, separator) if item != ""]


def positions(value):
    result = []
    for item in parts(value):
        if item == "":
            result.append(None)
        else:
            x, y = item.split(":")
            result.append((int(x), int(y)))
    return result


def slot_sets(value):
    return [set(ints(item, "|")) for item in parts(value)]


def l1(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def legal_actions(position):
    legal = np.ones(7, dtype=bool)
    for action in range(4):
        candidate = np.asarray(position) + ACTION_DELTA[action]
        legal[action] = bool(
            0 <= candidate[0] < GRID_SIZE
            and 0 <= candidate[1] < GRID_SIZE
        )
    return legal


def moved(position, action):
    candidate = np.asarray(position) + ACTION_DELTA[int(action)]
    if not (
            0 <= candidate[0] < GRID_SIZE
            and 0 <= candidate[1] < GRID_SIZE):
        return tuple(position)
    return int(candidate[0]), int(candidate[1])


def phase(step):
    step = int(step)
    if step <= 20000:
        return "0-20k"
    if step <= 40000:
        return "20-40k"
    if step <= 60000:
        return "40-60k"
    return "60-80k"


def run_file(run_dir, suffix):
    return run_dir / (run_dir.name + "_" + suffix)


def strict_ids(run_dir):
    post = pd.read_csv(str(run_file(
        run_dir, "progress_train_post_capture_24step.csv"
    )))
    result = set()
    for row in post.itertuples(index=False):
        targets = ints(row.capture_target_ids)
        if any(int(target) != int(row.first_capture_target_id)
               for target in targets):
            result.add(int(row.environment_episode_id))
    return result


def episode_table(run_dir, prefix):
    joint = pd.read_csv(str(run_file(
        run_dir, "progress_train_joint_exploration_episode.csv"
    )))
    joint["environment_episode_id"] = np.arange(len(joint), dtype=np.int64)
    joint["phase"] = joint["step"].map(phase)
    first = prefix.loc[prefix["offset_to_first_capture"] == 0].copy()
    first["remaining_distance"] = first.apply(
        lambda row: sorted(
            l1(position, positions(row["food_positions"])[
                1 - int(row["first_capture_target_id"])
            ])
            for position in positions(row["player_slot_positions"])
            if position is not None
        )[0],
        axis=1,
    )
    first["time_to_first"] = pd.to_numeric(
        first["episode_step"], errors="coerce"
    )
    first["capture_participants"] = first[
        "first_capture_participant_slots"
    ].map(lambda value: len(ints(value)))
    first = first[[
        "environment_episode_id", "remaining_distance", "time_to_first",
        "capture_participants",
    ]]
    result = joint.merge(first, on="environment_episode_id", how="left")
    result["first_capture"] = result["remaining_distance"].notna()
    result["strict"] = result["environment_episode_id"].isin(
        strict_ids(run_dir)
    )
    result["far"] = result["remaining_distance"] >= 8
    return result


def action_position(start, actual, action, executed_action):
    if actual is not None and int(action) == int(executed_action):
        return actual
    return moved(start, action)


def robust_action_info(start, before_foods, relevant_foods):
    legal = legal_actions(start)
    scores = np.full(7, -1, dtype=np.int64)
    per_food_safe = {}
    for action in np.flatnonzero(legal):
        candidate = moved(start, action)
        safe = {
            food_id for food_id in relevant_foods
            if l1(candidate, before_foods[food_id]) + 1 <= SIGHT_RADIUS
        }
        per_food_safe[int(action)] = safe
        scores[action] = len(safe)
    best_score = int(scores.max())
    best = set(np.flatnonzero(legal & (scores == best_score)).tolist())
    return scores, best, per_food_safe, best_score


def analyze_prefix(run_dir):
    prefix = pd.read_csv(str(run_file(
        run_dir, "progress_train_pre_capture_prefix.csv"
    )))
    episodes = episode_table(run_dir, prefix).set_index(
        "environment_episode_id"
    )
    transitions = []
    reranks = []
    geometry = []
    visibility = []

    for episode_id, group in prefix.groupby("environment_episode_id"):
        episode_id = int(episode_id)
        group = group.sort_values("episode_step").reset_index(drop=True)
        meta = episodes.loc[episode_id]
        target = int(group["first_capture_target_id"].iloc[0])
        remaining = 1 - target

        counts = np.asarray([
            len(slot_sets(value)[remaining])
            for value in group["food_visible_player_slots"]
        ], dtype=np.int64)
        offsets = group["offset_to_first_capture"].astype(int).values
        last_two = (
            int(offsets[np.flatnonzero(counts >= 2)[-1]])
            if np.any(counts >= 2) else np.nan
        )
        last_one = (
            int(offsets[np.flatnonzero(counts >= 1)[-1]])
            if np.any(counts >= 1) else np.nan
        )
        zero_after = np.flatnonzero(
            (offsets > last_one) & (counts == 0)
        ) if np.isfinite(last_one) else np.flatnonzero(counts == 0)
        first_zero = (
            int(offsets[zero_after[0]]) if zero_after.size else np.nan
        )
        visibility.append({
            "episode": episode_id,
            "strict": bool(meta["strict"]),
            "far": bool(meta["far"]),
            "last_two": last_two,
            "last_one": last_one,
            "first_zero": first_zero,
        })

        for row in group.itertuples(index=False):
            player_positions = positions(row.player_slot_positions)
            remaining_position = positions(row.food_positions)[remaining]
            distances = sorted(
                l1(position, remaining_position)
                for position in player_positions if position is not None
            )
            geometry.append({
                "episode": episode_id,
                "strict": bool(meta["strict"]),
                "far": bool(meta["far"]),
                "offset": int(row.offset_to_first_capture),
                "nearest": distances[0],
                "second_nearest": distances[1],
                "observer_count": len(
                    slot_sets(row.food_visible_player_slots)[remaining]
                ),
            })

        for index in range(1, len(group)):
            before = group.iloc[index - 1]
            after = group.iloc[index]
            before_foods = positions(before["food_positions"])
            after_foods = positions(after["food_positions"])
            before_players = positions(before["player_slot_positions"])
            after_players = positions(after["player_slot_positions"])
            before_visible_all = slot_sets(before["food_visible_player_slots"])
            after_visible_all = slot_sets(after["food_visible_player_slots"])
            before_visible = before_visible_all[remaining]
            after_visible = after_visible_all[remaining]
            if len(before_visible) != 2:
                continue

            reranked_actions = ints(after["greedy_actions"])
            executed_actions = ints(after["selected_actions"])
            if "unconstrained_greedy_actions" in after.index:
                raw_actions = ints(after["unconstrained_greedy_actions"])
            else:
                raw_actions = list(reranked_actions)
            random_slots = set(ints(after["random_replacement_slots"], "|"))
            eligible_slots = set(ints(after.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots",
                np.nan,
            ), "|"))
            constrained_slots = set(ints(after.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots",
                np.nan,
            ), "|"))
            conflict_slots = set(ints(after.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots",
                np.nan,
            ), "|"))
            reranked_slots = set(ints(after.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots",
                np.nan,
            ), "|"))

            retained = {"raw": [], "reranked": [], "executed": []}
            progress = {"raw": [], "reranked": [], "executed": []}
            target_progress = {"raw": [], "reranked": [], "executed": []}
            robust_remaining = {}
            robust_set_sizes = {}
            relevant_by_slot = {}
            for slot in sorted(before_visible):
                exact_foods = {
                    food_id for food_id, observers in enumerate(before_visible_all)
                    if len(observers) == 2 and slot in observers
                }
                relevant_by_slot[slot] = exact_foods
                _, best_actions, per_food_safe, _ = robust_action_info(
                    before_players[slot], before_foods, exact_foods
                )
                robust_remaining[slot] = {
                    action for action in best_actions
                    if remaining in per_food_safe[action]
                }
                robust_set_sizes[slot] = len(best_actions)
                for layer, action in (
                        ("raw", raw_actions[slot]),
                        ("reranked", reranked_actions[slot]),
                        ("executed", executed_actions[slot])):
                    candidate = (
                        after_players[slot]
                        if layer == "executed"
                        else action_position(
                            before_players[slot], after_players[slot], action,
                            executed_actions[slot],
                        )
                    )
                    if candidate is not None and l1(
                            candidate, after_foods[remaining]) <= SIGHT_RADIUS:
                        retained[layer].append(slot)
                    progress[layer].append(
                        np.nan if candidate is None else (
                            l1(before_players[slot], before_foods[remaining])
                            - l1(candidate, after_foods[remaining])
                        )
                    )
                    target_progress[layer].append(
                        np.nan if candidate is None else (
                            l1(before_players[slot], before_foods[target])
                            - l1(candidate, after_foods[target])
                        )
                    )

                if slot in reranked_slots:
                    reranks.append({
                        "episode": episode_id,
                        "strict": bool(meta["strict"]),
                        "far": bool(meta["far"]),
                        "step": int(after["training_env_step"]),
                        "phase": phase(after["training_env_step"]),
                        "roster": sum(x is not None for x in before_players),
                        "offset": int(after["offset_to_first_capture"]),
                        "slot": slot,
                        "raw_action": raw_actions[slot],
                        "reranked_action": reranked_actions[slot],
                        "executed_action": executed_actions[slot],
                        "raw_progress": progress["raw"][-1],
                        "reranked_progress": progress["reranked"][-1],
                        "executed_progress": progress["executed"][-1],
                        "raw_target_progress": target_progress["raw"][-1],
                        "reranked_target_progress": (
                            target_progress["reranked"][-1]
                        ),
                        "executed_target_progress": (
                            target_progress["executed"][-1]
                        ),
                        "raw_stay_turn": raw_actions[slot] >= 4,
                        "reranked_stay_turn": reranked_actions[slot] >= 4,
                        "robust_set_size": robust_set_sizes[slot],
                        "remaining_robust_action_exists": bool(
                            robust_remaining[slot]
                        ),
                        "rerank_remaining_robust": (
                            reranked_actions[slot] in robust_remaining[slot]
                        ),
                        "conflict": slot in conflict_slots,
                    })

            raw_preserve = len(retained["raw"]) >= 2
            reranked_preserve = len(retained["reranked"]) >= 2
            executed_preserve = len(retained["executed"]) >= 2
            actual_loss = len(after_visible) < 2
            roster_loss = any(after_players[slot] is None for slot in before_visible)
            exploration_changed = any(
                reranked_actions[slot] != executed_actions[slot]
                for slot in before_visible
            )
            relevant_conflict = bool(before_visible & conflict_slots)
            intended_positions = []
            execution_mismatch_slots = set()
            for slot, start in enumerate(before_players):
                if start is None:
                    intended_positions.append(None)
                    continue
                intended = moved(start, executed_actions[slot])
                intended_positions.append(intended)
                if after_players[slot] is not None and after_players[slot] != intended:
                    execution_mismatch_slots.add(slot)
            duplicate_intended_slots = set()
            for slot, intended in enumerate(intended_positions):
                if intended is None:
                    continue
                matches = [
                    other for other, other_position in enumerate(intended_positions)
                    if other_position == intended
                ]
                if len(matches) > 1:
                    duplicate_intended_slots.update(matches)
            observer_collision = bool(
                before_visible & execution_mismatch_slots
                & duplicate_intended_slots
            )
            if not raw_preserve:
                first_layer = "raw_greedy_first"
            elif not reranked_preserve:
                first_layer = "rerank_first"
            elif not executed_preserve and exploration_changed:
                first_layer = "exploration_first"
            elif not executed_preserve:
                first_layer = "execution_or_environment"
            else:
                first_layer = "preserved"

            if actual_loss:
                if roster_loss:
                    loss_cause = "roster_event"
                elif observer_collision:
                    loss_cause = "wolf_collision_rollback"
                elif not reranked_preserve and relevant_conflict:
                    loss_cause = "multi_prey_conflict"
                elif not reranked_preserve:
                    loss_cause = "reranked_not_robust"
                elif exploration_changed and not executed_preserve:
                    loss_cause = "exploration"
                elif executed_preserve and len(after_visible) < 2:
                    loss_cause = "diagnostic_alignment_or_visibility"
                else:
                    loss_cause = "environment_or_execution"
            else:
                loss_cause = "none"

            transitions.append({
                "episode": episode_id,
                "strict": bool(meta["strict"]),
                "far": bool(meta["far"]),
                "win": bool(meta["win"]),
                "step": int(after["training_env_step"]),
                "phase": phase(after["training_env_step"]),
                "roster": sum(x is not None for x in before_players),
                "remaining_food": int(remaining),
                "offset": int(after["offset_to_first_capture"]),
                "next_observers": len(after_visible),
                "actual_loss": actual_loss,
                "raw_preserve": raw_preserve,
                "reranked_preserve": reranked_preserve,
                "executed_preserve": executed_preserve,
                "first_layer": first_layer,
                "loss_cause": loss_cause,
                "frontier_applied": bool(reranked_slots),
                "eligible_observers": len(before_visible & eligible_slots),
                "constrained_observers": len(before_visible & constrained_slots),
                "reranked_observers": len(before_visible & reranked_slots),
                "conflict_observers": len(before_visible & conflict_slots),
                "execution_mismatch_observers": len(
                    before_visible & execution_mismatch_slots
                ),
                "collision_observers": len(
                    before_visible & execution_mismatch_slots
                    & duplicate_intended_slots
                ),
                "random_observers": len(before_visible & random_slots),
                "raw_progress_mean": finite_mean(progress["raw"]),
                "reranked_progress_mean": finite_mean(progress["reranked"]),
                "executed_progress_mean": finite_mean(progress["executed"]),
                "raw_target_progress_mean": finite_mean(
                    target_progress["raw"]
                ),
                "reranked_target_progress_mean": finite_mean(
                    target_progress["reranked"]
                ),
                "executed_target_progress_mean": finite_mean(
                    target_progress["executed"]
                ),
                "raw_target_over_remaining_count": int(sum(
                    target_value > remaining_value
                    for target_value, remaining_value in zip(
                        target_progress["raw"], progress["raw"]
                    )
                    if np.isfinite(target_value)
                    and np.isfinite(remaining_value)
                )),
                "reranked_target_over_remaining_count": int(sum(
                    target_value > remaining_value
                    for target_value, remaining_value in zip(
                        target_progress["reranked"], progress["reranked"]
                    )
                    if np.isfinite(target_value)
                    and np.isfinite(remaining_value)
                )),
                "executed_target_over_remaining_count": int(sum(
                    target_value > remaining_value
                    for target_value, remaining_value in zip(
                        target_progress["executed"], progress["executed"]
                    )
                    if np.isfinite(target_value)
                    and np.isfinite(remaining_value)
                )),
                "progress_observer_count": int(sum(
                    np.isfinite(value) for value in progress["executed"]
                )),
                "raw_both_positive": all(x > 0 for x in progress["raw"]),
                "reranked_both_positive": all(
                    x > 0 for x in progress["reranked"]
                ),
                "executed_both_positive": all(
                    x > 0 for x in progress["executed"]
                ),
                "remaining_robust_exists_both": all(
                    bool(robust_remaining[slot]) for slot in before_visible
                ),
                "rerank_remaining_robust_both": all(
                    reranked_actions[slot] in robust_remaining[slot]
                    for slot in before_visible
                ),
                "robust_set_size_mean": float(np.mean([
                    robust_set_sizes[slot] for slot in before_visible
                ])),
                "raw_actions": "|".join(
                    str(raw_actions[slot]) for slot in sorted(before_visible)
                ),
                "reranked_actions": "|".join(
                    str(reranked_actions[slot]) for slot in sorted(before_visible)
                ),
                "executed_actions": "|".join(
                    str(executed_actions[slot]) for slot in sorted(before_visible)
                ),
                "visible_slots": "|".join(map(str, sorted(before_visible))),
                "before_positions": before["player_slot_positions"],
                "after_positions": after["player_slot_positions"],
                "before_foods": before["food_positions"],
                "after_foods": after["food_positions"],
                "intended_positions": ";".join(
                    "" if position is None else "{}:{}".format(*position)
                    for position in intended_positions
                ),
            })

    return {
        "prefix": prefix,
        "episodes": episodes.reset_index(),
        "transitions": pd.DataFrame(transitions),
        "reranks": pd.DataFrame(reranks),
        "geometry": pd.DataFrame(geometry),
        "visibility": pd.DataFrame(visibility),
    }


def describe_boolean_transitions(transitions, label):
    cohort = transitions
    if label == "far_failure":
        cohort = cohort.loc[cohort["far"] & ~cohort["strict"]]
    elif label == "strict":
        cohort = cohort.loc[cohort["strict"]]
    losses = cohort.loc[cohort["actual_loss"]]
    t2_to_t1 = cohort.loc[cohort["next_observers"] == 1]
    return {
        "n_exact_two": int(len(cohort)),
        "raw_persistence": float(cohort["raw_preserve"].mean()),
        "reranked_persistence": float(cohort["reranked_preserve"].mean()),
        "executed_persistence": float(cohort["executed_preserve"].mean()),
        "raw_both_positive": float(cohort["raw_both_positive"].mean()),
        "reranked_both_positive": float(
            cohort["reranked_both_positive"].mean()
        ),
        "executed_both_positive": float(
            cohort["executed_both_positive"].mean()
        ),
        "raw_progress_mean": float(cohort["raw_progress_mean"].mean()),
        "reranked_progress_mean": float(
            cohort["reranked_progress_mean"].mean()
        ),
        "executed_progress_mean": float(
            cohort["executed_progress_mean"].mean()
        ),
        "t2_loss_count": int(len(losses)),
        "loss_first_layer": losses["first_layer"].value_counts().to_dict(),
        "loss_cause": losses["loss_cause"].value_counts().to_dict(),
        "t2_to_t1_count": int(len(t2_to_t1)),
        "t2_to_t1_first_layer": t2_to_t1[
            "first_layer"
        ].value_counts().to_dict(),
        "t2_to_t1_cause": t2_to_t1[
            "loss_cause"
        ].value_counts().to_dict(),
    }


def visibility_table(visibility):
    rows = []
    for label, mask in (
            ("far_failure", visibility["far"] & ~visibility["strict"]),
            ("strict", visibility["strict"]),
            ("all", np.ones(len(visibility), dtype=bool))):
        cohort = visibility.loc[mask]
        for field in ("last_two", "last_one", "first_zero"):
            values = cohort[field].dropna()
            rows.append({
                "cohort": label,
                "field": field,
                "n": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "p25": values.quantile(.25) if len(values) else np.nan,
                "p75": values.quantile(.75) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def geometry_table(geometry):
    rows = []
    offsets = (-32, -16, -8, -4, 0)
    for label, mask in (
            ("far_failure", geometry["far"] & ~geometry["strict"]),
            ("strict", geometry["strict"]),
            ("all", np.ones(len(geometry), dtype=bool))):
        for offset in offsets:
            cohort = geometry.loc[mask & (geometry["offset"] == offset)]
            rows.append({
                "cohort": label,
                "offset": offset,
                "n": len(cohort),
                "nearest_mean": cohort["nearest"].mean(),
                "second_nearest_mean": cohort["second_nearest"].mean(),
                "observer_mean": cohort["observer_count"].mean(),
                "observer_ge2": (cohort["observer_count"] >= 2).mean(),
            })
    return pd.DataFrame(rows)


def exposure(run_dir):
    joint = pd.read_csv(str(run_file(
        run_dir, "progress_train_joint_exploration_episode.csv"
    )))
    joint["phase"] = joint["step"].map(phase)
    fields = {
        "decisions": "joint_decision_count",
        "applied_decisions": (
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied_count"
        ),
        "eligible_slots": (
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slot_count_sum"
        ),
        "constrained_slots": (
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slot_count_sum"
        ),
        "conflict_slots": (
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slot_count_sum"
        ),
        "reranked_slots": (
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slot_count_sum"
        ),
    }
    total = {
        output: int(pd.to_numeric(joint[column], errors="coerce").sum())
        for output, column in fields.items()
    }
    total["episodes"] = len(joint)
    total["episodes_with_applied"] = int((joint[
        fields["applied_decisions"]
    ] > 0).sum())
    total["episodes_with_eligible"] = int((joint[
        fields["eligible_slots"]
    ] > 0).sum())
    staged = joint.groupby("phase", sort=False).agg(**{
        output: (column, "sum") for output, column in fields.items()
    }).reset_index()
    return total, staged


def collision_exposure(prefix):
    records = []
    for _, group in prefix.groupby("environment_episode_id"):
        group = group.sort_values("episode_step").reset_index(drop=True)
        for index in range(1, len(group)):
            before = group.iloc[index - 1]
            after = group.iloc[index]
            if "unconstrained_greedy_actions" not in after.index:
                continue
            starts = positions(before["player_slot_positions"])
            actual = positions(after["player_slot_positions"])
            raw = ints(after["unconstrained_greedy_actions"])
            reranked = ints(after["greedy_actions"])
            reranked_slots = set(ints(after.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots",
                np.nan,
            ), "|"))
            if not reranked_slots:
                continue
            raw_intended = [
                None if start is None else moved(start, raw[slot])
                for slot, start in enumerate(starts)
            ]
            reranked_intended = [
                None if start is None else moved(start, reranked[slot])
                for slot, start in enumerate(starts)
            ]

            def duplicate_slots(destinations):
                duplicates = set()
                for slot, destination in enumerate(destinations):
                    if destination is None:
                        continue
                    matches = [
                        other for other, value in enumerate(destinations)
                        if value == destination
                    ]
                    if len(matches) > 1:
                        duplicates.update(matches)
                return duplicates

            raw_duplicates = duplicate_slots(raw_intended)
            reranked_duplicates = duplicate_slots(reranked_intended)
            changed_collision_slots = reranked_slots & reranked_duplicates
            execution_mismatch = {
                slot for slot in reranked_slots
                if actual[slot] is not None
                and actual[slot] != reranked_intended[slot]
            }
            enters_occupied = set()
            occupied_vacated = set()
            for slot in reranked_slots:
                destination = reranked_intended[slot]
                occupants = {
                    other for other, start in enumerate(starts)
                    if other != slot and start is not None and start == destination
                }
                if not occupants:
                    continue
                enters_occupied.add(slot)
                if all(reranked_intended[other] != destination for other in occupants):
                    occupied_vacated.add(slot)
            records.append({
                "step": int(after["training_env_step"]),
                "phase": phase(after["training_env_step"]),
                "roster": sum(x is not None for x in starts),
                "reranked_slots": len(reranked_slots),
                "raw_collision": bool(raw_duplicates),
                "reranked_collision": bool(reranked_duplicates),
                "rerank_created_collision": bool(
                    reranked_duplicates and not raw_duplicates
                ),
                "rerank_resolved_collision": bool(
                    raw_duplicates and not reranked_duplicates
                ),
                "reranked_collision_slots": len(changed_collision_slots),
                "reranked_execution_mismatch_slots": len(execution_mismatch),
                "reranked_enters_occupied_slots": len(enters_occupied),
                "reranked_enters_vacated_slots": len(occupied_vacated),
            })
    return pd.DataFrame(records)


def rerank_table(reranks):
    if reranks.empty:
        return {}
    return {
        "n": int(len(reranks)),
        "raw_progress_mean": float(reranks["raw_progress"].mean()),
        "reranked_progress_mean": float(reranks["reranked_progress"].mean()),
        "executed_progress_mean": float(reranks["executed_progress"].mean()),
        "raw_stay_turn": float(reranks["raw_stay_turn"].mean()),
        "reranked_stay_turn": float(reranks["reranked_stay_turn"].mean()),
        "remaining_robust_exists": float(
            reranks["remaining_robust_action_exists"].mean()
        ),
        "rerank_remaining_robust": float(
            reranks["rerank_remaining_robust"].mean()
        ),
        "conflict": float(reranks["conflict"].mean()),
        "robust_set_size": reranks[
            "robust_set_size"
        ].value_counts().sort_index().to_dict(),
        "raw_actions": reranks[
            "raw_action"
        ].value_counts().sort_index().to_dict(),
        "reranked_actions": reranks[
            "reranked_action"
        ].value_counts().sort_index().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--runs", nargs="+", type=int, default=[168, 169])
    parser.add_argument("--write-run169-fixtures", type=Path)
    args = parser.parse_args()

    for run in args.runs:
        run_dir = args.run_root / ("run{}".format(run))
        analysis = analyze_prefix(run_dir)
        total, staged = exposure(run_dir) if run >= 169 else ({}, pd.DataFrame())
        print("\n===== RUN {} =====".format(run))
        if total:
            print("EXPOSURE", json.dumps(total, sort_keys=True))
            print("EXPOSURE_STAGES\n{}".format(staged.to_string(index=False)))
        for label in ("all", "far_failure", "strict"):
            print("EXACT_TWO_{} {}".format(
                label.upper(), json.dumps(describe_boolean_transitions(
                    analysis["transitions"], label
                ), sort_keys=True)
            ))
        far_stage = analysis["transitions"].loc[
            analysis["transitions"]["far"]
            & ~analysis["transitions"]["strict"]
        ].groupby("phase").agg(
            n=("episode", "size"),
            raw_persistence=("raw_preserve", "mean"),
            reranked_persistence=("reranked_preserve", "mean"),
            executed_persistence=("executed_preserve", "mean"),
            raw_progress=("raw_progress_mean", "mean"),
            reranked_progress=("reranked_progress_mean", "mean"),
            executed_progress=("executed_progress_mean", "mean"),
            raw_both_positive=("raw_both_positive", "mean"),
            reranked_both_positive=("reranked_both_positive", "mean"),
        ).reset_index()
        print("FAR_EXACT_TWO_STAGES\n{}".format(
            far_stage.to_string(index=False)
        ))
        print("VISIBILITY\n{}".format(
            visibility_table(analysis["visibility"]).to_string(index=False)
        ))
        print("GEOMETRY\n{}".format(
            geometry_table(analysis["geometry"]).to_string(index=False)
        ))
        print("RERANK", json.dumps(
            rerank_table(analysis["reranks"]), sort_keys=True
        ))
        collisions = collision_exposure(analysis["prefix"])
        if not collisions.empty:
            print("COLLISION_EXPOSURE", json.dumps({
                "decisions": int(len(collisions)),
                "raw_collision_decisions": int(collisions[
                    "raw_collision"
                ].sum()),
                "reranked_collision_decisions": int(collisions[
                    "reranked_collision"
                ].sum()),
                "rerank_created_collision_decisions": int(collisions[
                    "rerank_created_collision"
                ].sum()),
                "rerank_resolved_collision_decisions": int(collisions[
                    "rerank_resolved_collision"
                ].sum()),
                "reranked_collision_slots": int(collisions[
                    "reranked_collision_slots"
                ].sum()),
                "reranked_execution_mismatch_slots": int(collisions[
                    "reranked_execution_mismatch_slots"
                ].sum()),
                "reranked_enters_occupied_slots": int(collisions[
                    "reranked_enters_occupied_slots"
                ].sum()),
                "reranked_enters_vacated_slots": int(collisions[
                    "reranked_enters_vacated_slots"
                ].sum()),
            }, sort_keys=True))
        if not analysis["reranks"].empty:
            print("RERANK_PHASE\n{}".format(
                analysis["reranks"].groupby("phase").agg(
                    n=("slot", "size"),
                    raw_progress=("raw_progress", "mean"),
                    reranked_progress=("reranked_progress", "mean"),
                    stay_turn=("reranked_stay_turn", "mean"),
                    conflict=("conflict", "mean"),
                ).reset_index().to_string(index=False)
            ))
            print("RERANK_ROSTER\n{}".format(
                analysis["reranks"].groupby("roster").agg(
                    n=("slot", "size"),
                    raw_progress=("raw_progress", "mean"),
                    reranked_progress=("reranked_progress", "mean"),
                    stay_turn=("reranked_stay_turn", "mean"),
                    conflict=("conflict", "mean"),
                ).reset_index().to_string(index=False)
            ))
        far_losses = analysis["transitions"].loc[
            analysis["transitions"]["far"]
            & ~analysis["transitions"]["strict"]
            & analysis["transitions"]["actual_loss"]
        ].sort_values(["step", "episode"])
        if len(far_losses):
            fields = [
                "episode", "step", "offset", "roster", "visible_slots",
                "first_layer", "loss_cause", "raw_actions",
                "reranked_actions", "executed_actions", "raw_progress_mean",
                "reranked_progress_mean", "executed_progress_mean",
                "frontier_applied", "conflict_observers",
                "execution_mismatch_observers", "collision_observers",
                "remaining_robust_exists_both",
                "rerank_remaining_robust_both", "before_positions",
                "after_positions", "before_foods", "after_foods",
                "intended_positions",
            ]
            print("FAR_LOSS_CASES\n{}".format(
                far_losses[fields].head(12).to_string(index=False)
            ))
        if run == 169 and args.write_run169_fixtures:
            fixture_rows = json.loads(
                far_losses.head(5).to_json(orient="records")
            )
            alignment_frame = analysis["transitions"].loc[
                analysis["transitions"]["far"]
                & ~analysis["transitions"]["strict"]
                & ~analysis["transitions"]["raw_preserve"]
                & analysis["transitions"]["rerank_remaining_robust_both"]
                & analysis["transitions"]["executed_preserve"]
                & analysis["transitions"]["frontier_applied"]
            ].sort_values(["step", "episode"]).head(3)
            alignment_rows = json.loads(
                alignment_frame.to_json(orient="records")
            )
            args.write_run169_fixtures.write_text(
                json.dumps({
                    "schema_version": 2,
                    "source_run": 169,
                    "definition": (
                        "first five chronological far-failure exact-two "
                        "observer-loss transitions"
                    ),
                    "transitions": fixture_rows,
                    "target_alignment_transitions": alignment_rows,
                }, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
