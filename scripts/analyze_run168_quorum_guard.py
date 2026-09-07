"""Offline analysis for the run167 -> run168 pre-capture quorum guard.

This script reads completed CSV artifacts only.  It does not import the
Wolfpack environment, load a model, or consume any training RNG.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.wolfpack_reward import (
    capture_quorum_balanced_alive_prey_coverage_cost,
)


ACTION_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (0, 0),
    5: (0, 0),
    6: (0, 0),
}


def _parts(value, separator=";"):
    if pd.isna(value) or str(value) == "":
        return []
    return str(value).split(separator)


def _ints(value, separator=";"):
    return [int(float(part)) for part in _parts(value, separator) if part != ""]


def _floats(value, separator=";"):
    return [float(part) for part in _parts(value, separator) if part != ""]


def _positions(value):
    positions = []
    for part in _parts(value):
        if part == "":
            positions.append(None)
        else:
            x, y = part.split(":")
            positions.append((int(x), int(y)))
    return positions


def _slot_sets(value):
    return [set(_ints(part, "|")) for part in _parts(value)]


def _factors(value):
    result = []
    for part in _parts(value):
        _, members = part.split(":", 1)
        result.append(set(_ints(members, "|")))
    return result


def _l1(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _move(position, action):
    dx, dy = ACTION_DELTA[int(action)]
    return int(position[0]) + dx, int(position[1]) + dy


def _phase(step):
    step = int(step)
    if step <= 20000:
        return "0-20k"
    if step <= 40000:
        return "20-40k"
    if step <= 60000:
        return "40-60k"
    return "60-80k"


def _run_file(run_dir, suffix):
    run_id = run_dir.name
    return run_dir / (run_id + "_" + suffix)


def _strict_episode_ids(run_dir):
    post = pd.read_csv(str(_run_file(
        run_dir, "progress_train_post_capture_24step.csv"
    )))
    strict = set()
    for row in post.itertuples(index=False):
        targets = _ints(row.capture_target_ids)
        if any(target != int(row.first_capture_target_id) for target in targets):
            strict.add(int(row.environment_episode_id))
    return strict


def load_run(run_dir):
    joint = pd.read_csv(str(_run_file(
        run_dir, "progress_train_joint_exploration_episode.csv"
    )))
    prefix = pd.read_csv(str(_run_file(
        run_dir, "progress_train_pre_capture_prefix.csv"
    )))
    train = pd.read_csv(str(_run_file(run_dir, "progress_train.csv")))
    evaluation = pd.read_csv(str(_run_file(run_dir, "progress_eval.csv")))
    strict = _strict_episode_ids(run_dir)
    wins = set(joint.loc[joint["win"] > 0, "episode_index"].astype(int))
    return {
        "joint": joint,
        "prefix": prefix,
        "train": train,
        "eval": evaluation,
        "strict": strict,
        "wins": wins,
    }


def episode_table(data):
    joint = data["joint"].copy()
    # The trajectory diagnostics number formal (non-warmup) episodes from
    # zero, whereas episode_index includes the 33 collection episodes that
    # preceded the first logged formal episode at env step 6600.
    joint["environment_episode_id"] = np.arange(len(joint), dtype=np.int64)
    joint["phase"] = joint["step"].map(_phase)
    first = data["prefix"].loc[
        data["prefix"]["offset_to_first_capture"] == 0
    ].copy()
    first["remaining_distance"] = first.apply(
        lambda row: _floats(row["min_alive_player_distance_to_food"])[
            1 - int(row["first_capture_target_id"])
        ],
        axis=1,
    )
    first = first[[
        "environment_episode_id",
        "training_env_step",
        "remaining_distance",
        "first_capture_target_id",
    ]]
    episodes = joint.merge(first, on="environment_episode_id", how="left")
    episodes["first_capture"] = episodes["remaining_distance"].notna()
    episodes["strict"] = episodes["environment_episode_id"].isin(
        data["strict"]
    )
    episodes["far"] = episodes["remaining_distance"] >= 8
    return episodes


def macro(data):
    episodes = episode_table(data)
    captures = int(episodes["capture_events"].sum())
    first = int(episodes["first_capture"].sum())
    strict = int(episodes["strict"].sum())
    result = {
        "formal": int(len(episodes)),
        "first_captures": first,
        "total_captures": captures,
        "captures_per_formal": captures / float(len(episodes)),
        "strict_24": strict,
        "first_to_strict": strict / float(first) if first else np.nan,
        "wins": int(episodes["win"].sum()),
        "far_count": int(episodes["far"].sum()),
        "far_share": float(episodes.loc[
            episodes["first_capture"], "far"
        ].mean()),
    }
    return result, episodes


def stage_table(episodes):
    rows = []
    for phase in ("0-20k", "20-40k", "40-60k", "60-80k"):
        cohort = episodes.loc[episodes["phase"] == phase]
        first = int(cohort["first_capture"].sum())
        strict = int(cohort["strict"].sum())
        rows.append({
            "phase": phase,
            "formal": len(cohort),
            "first": first,
            "captures": int(cohort["capture_events"].sum()),
            "strict": strict,
            "first_to_strict": strict / float(first) if first else np.nan,
            "wins": int(cohort["win"].sum()),
            "far": int(cohort["far"].sum()),
            "far_share": float(cohort.loc[
                cohort["first_capture"], "far"
            ].mean()) if first else np.nan,
        })
    return pd.DataFrame(rows)


def far_bands(episodes):
    captured = episodes.loc[episodes["first_capture"]].copy()
    captured["band"] = pd.cut(
        captured["remaining_distance"],
        bins=[-np.inf, 2, 4, 7, np.inf],
        labels=["0-2", "3-4", "5-7", "8+"],
    )
    rows = []
    for band in ("0-2", "3-4", "5-7", "8+"):
        cohort = captured.loc[captured["band"] == band]
        strict = int(cohort["strict"].sum())
        rows.append({
            "band": band,
            "count": len(cohort),
            "share": len(cohort) / float(len(captured)),
            "strict": strict,
            "conversion": strict / float(len(cohort)) if len(cohort) else np.nan,
            "wins": int(cohort["win"].sum()),
        })
    return pd.DataFrame(rows)


def guard_exposure(data):
    joint = data["joint"]
    if "pre_capture_visible_prey_quorum_guard_applied_count" not in joint:
        return {"applied": 0, "protected": 0, "episodes": 0}, pd.DataFrame()
    result = {
        "applied": int(joint[
            "pre_capture_visible_prey_quorum_guard_applied_count"
        ].sum()),
        "protected": int(joint[
            "pre_capture_visible_prey_quorum_protected_slot_count_sum"
        ].sum()),
        "episodes": int((joint[
            "pre_capture_visible_prey_quorum_guard_applied_count"
        ] > 0).sum()),
    }
    staged = joint.assign(phase=joint["step"].map(_phase)).groupby(
        "phase", sort=False
    ).agg(
        episodes=("episode_index", "size"),
        applied=("pre_capture_visible_prey_quorum_guard_applied_count", "sum"),
        protected=(
            "pre_capture_visible_prey_quorum_protected_slot_count_sum", "sum"
        ),
    ).reset_index()
    return result, staged


def visibility_records(data):
    prefix = data["prefix"]
    episodes = episode_table(data).set_index("environment_episode_id")
    summaries = []
    losses = []
    quorum_transitions = []
    for episode_id, group in prefix.groupby("environment_episode_id"):
        episode_id = int(episode_id)
        group = group.sort_values("episode_step").reset_index(drop=True)
        target = int(group["first_capture_target_id"].iloc[0])
        remaining = 1 - target
        counts = np.asarray([
            _ints(value)[remaining]
            for value in group["food_observer_counts"]
        ], dtype=np.int64)
        offsets = group["offset_to_first_capture"].astype(int).values
        last_two = int(offsets[np.flatnonzero(counts >= 2)[-1]]) \
            if np.any(counts >= 2) else np.nan
        last_one = int(offsets[np.flatnonzero(counts >= 1)[-1]]) \
            if np.any(counts >= 1) else np.nan
        if np.isfinite(last_one):
            zero_after = np.flatnonzero(
                (offsets > int(last_one)) & (counts == 0)
            )
            first_zero = int(offsets[zero_after[0]]) \
                if zero_after.size else np.nan
        else:
            zero = np.flatnonzero(counts == 0)
            first_zero = int(offsets[zero[0]]) if zero.size else np.nan
        meta = episodes.loc[episode_id]
        summaries.append({
            "episode": episode_id,
            "strict": bool(meta["strict"]),
            "far": bool(meta["far"]),
            "win": bool(meta["win"]),
            "last_two": last_two,
            "last_one": last_one,
            "first_zero": first_zero,
        })

        for idx in range(1, len(group)):
            before = group.iloc[idx - 1]
            after = group.iloc[idx]
            before_visible = _slot_sets(before["food_visible_player_slots"])[remaining]
            after_visible = _slot_sets(after["food_visible_player_slots"])[remaining]
            if len(before_visible) != 2:
                continue
            actions_g = _ints(after["greedy_actions"])
            actions_e = _ints(after["selected_actions"])
            random_slots = set(_ints(after["random_replacement_slots"], "|"))
            protected = set(_ints(
                after.get("pre_capture_visible_prey_quorum_protected_slots", np.nan),
                "|",
            ))
            start_positions = _positions(before["player_slot_positions"])
            actual_positions = _positions(after["player_slot_positions"])
            prey_before = _positions(before["food_positions"])[remaining]
            prey_after = _positions(after["food_positions"])[remaining]
            target_after = _positions(after["food_positions"])[target]
            factors_before = _factors(before["active_factor_agent_slots"])
            factors_after = _factors(after["active_factor_agent_slots"])
            factor_q = _floats(after["greedy_factor_q_values"])
            margins = _floats(after["greedy_message_action_margins"])
            greedy_retained = []
            executed_retained = []
            for observer_slot in sorted(before_visible):
                if (
                        actual_positions[observer_slot] is not None
                        and actions_g[observer_slot] == actions_e[observer_slot]):
                    greedy_position = actual_positions[observer_slot]
                else:
                    greedy_position = _move(
                        start_positions[observer_slot],
                        actions_g[observer_slot],
                    )
                if _l1(greedy_position, prey_after) <= 8:
                    greedy_retained.append(observer_slot)
                if (
                        actual_positions[observer_slot] is not None
                        and _l1(actual_positions[observer_slot], prey_after) <= 8):
                    executed_retained.append(observer_slot)
            action_pair_factor = any(
                before_visible.issubset(factor) for factor in factors_after
            )
            quorum_transitions.append({
                "episode": episode_id,
                "strict": bool(meta["strict"]),
                "far": bool(meta["far"]),
                "win": bool(meta["win"]),
                "training_step": int(after["training_env_step"]),
                "offset": int(after["offset_to_first_capture"]),
                "joint_explore": int(after["joint_explore"]),
                "guard_applied": int(after.get(
                    "pre_capture_visible_prey_quorum_guard_applied", 0
                )),
                "next_observer_count": len(after_visible),
                "greedy_retained_count": len(greedy_retained),
                "executed_retained_count": len(executed_retained),
                "greedy_preserves_quorum": len(greedy_retained) >= 2,
                "executed_preserves_quorum": len(executed_retained) >= 2,
                "action_pair_factor": action_pair_factor,
                "mean_observer_margin": float(np.nanmean([
                    margins[slot] for slot in before_visible
                    if slot < len(margins)
                ])),
            })
            if len(after_visible) != 1:
                continue
            leaving = sorted(before_visible - after_visible)
            if not leaving:
                continue
            for slot in leaving:
                if (
                        actual_positions[slot] is not None
                        and actions_g[slot] == actions_e[slot]):
                    greedy_position = actual_positions[slot]
                else:
                    greedy_position = _move(
                        start_positions[slot], actions_g[slot]
                    )
                greedy_visible = _l1(greedy_position, prey_after) <= 8
                executed_visible = bool(
                    actual_positions[slot] is not None
                    and _l1(actual_positions[slot], prey_after) <= 8
                )
                selected_differs = actions_g[slot] != actions_e[slot]
                if actual_positions[slot] is None:
                    category = "other"
                elif greedy_visible and selected_differs and not executed_visible:
                    category = "exploration_first"
                elif not greedy_visible:
                    category = "greedy_first"
                else:
                    category = "other"
                pair_before = any(before_visible.issubset(factor) for factor in factors_before)
                pair_after = any(before_visible.issubset(factor) for factor in factors_after)
                pair_factor_indices = [
                    factor_idx for factor_idx, factor in enumerate(factors_after)
                    if before_visible.issubset(factor)
                ]
                learned_factor_count = len(factors_after)
                pair_factor_q = sum(
                    factor_q[factor_idx]
                    for factor_idx in pair_factor_indices
                    if factor_idx < len(factor_q)
                )
                learned_q_total = sum(factor_q[:learned_factor_count])
                exact_quorum_prey = [
                    food_id for food_id, observers in enumerate(
                        _slot_sets(before["food_visible_player_slots"])
                    )
                    if len(observers) == 2 and slot in observers
                ]
                robust_scores = {}
                remaining_safe = {}
                for action in range(7):
                    candidate_position = _move(start_positions[slot], action)
                    legal = (
                        0 <= candidate_position[0] < 20
                        and 0 <= candidate_position[1] < 20
                    )
                    if not legal:
                        continue
                    safe_food = [
                        food_id for food_id in exact_quorum_prey
                        if _l1(
                            candidate_position,
                            _positions(before["food_positions"])[food_id],
                        ) + 1 <= 8
                    ]
                    robust_scores[action] = len(safe_food)
                    remaining_safe[action] = remaining in safe_food
                max_robust_score = max(robust_scores.values())
                max_score_actions = [
                    action for action, score in robust_scores.items()
                    if score == max_robust_score
                ]
                greedy_robust_score = robust_scores[actions_g[slot]]
                fixable_by_frontier_constraint = bool(
                    max_robust_score > greedy_robust_score
                    and any(remaining_safe[action] for action in max_score_actions)
                )
                remaining_safe_for_all_best = bool(
                    max_score_actions
                    and all(remaining_safe[action] for action in max_score_actions)
                )
                live_before = [
                    position for position in start_positions
                    if position is not None
                ]
                live_after = [
                    position for position in actual_positions
                    if position is not None
                ]
                if len(live_before) == len(live_after):
                    coverage_before = (
                        capture_quorum_balanced_alive_prey_coverage_cost(
                            live_before, _positions(before["food_positions"])
                        )
                    )
                    coverage_after = (
                        capture_quorum_balanced_alive_prey_coverage_cost(
                            live_after, _positions(after["food_positions"])
                        )
                    )
                    coverage_delta = 0.01 * (
                        coverage_before - coverage_after
                    )
                else:
                    coverage_delta = np.nan
                losses.append({
                    "episode": episode_id,
                    "strict": bool(meta["strict"]),
                    "far": bool(meta["far"]),
                    "win": bool(meta["win"]),
                    "training_step": int(after["training_env_step"]),
                    "offset": int(after["offset_to_first_capture"]),
                    "slot": int(slot),
                    "category": category,
                    "joint_explore": int(after["joint_explore"]),
                    "random_replaced": slot in random_slots,
                    "guard_applied": int(after.get(
                        "pre_capture_visible_prey_quorum_guard_applied", 0
                    )),
                    "protected": slot in protected,
                    "greedy_action": int(actions_g[slot]),
                    "selected_action": int(actions_e[slot]),
                    "start_distance": _l1(start_positions[slot], prey_before),
                    "greedy_end_distance": _l1(greedy_position, prey_after),
                    "executed_end_distance": (
                        _l1(actual_positions[slot], prey_after)
                        if actual_positions[slot] is not None else np.nan
                    ),
                    "greedy_progress": (
                        _l1(start_positions[slot], prey_before)
                        - _l1(greedy_position, prey_after)
                    ),
                    "executed_progress": (
                        _l1(start_positions[slot], prey_before)
                        - _l1(actual_positions[slot], prey_after)
                        if actual_positions[slot] is not None else np.nan
                    ),
                    "greedy_target_distance": _l1(greedy_position, target_after),
                    "greedy_target_progress": (
                        _l1(start_positions[slot], _positions(
                            before["food_positions"]
                        )[target])
                        - _l1(greedy_position, target_after)
                    ),
                    "greedy_visible": greedy_visible,
                    "executed_visible": executed_visible,
                    "message_margin": margins[slot] if slot < len(margins) else np.nan,
                    "pair_factor_before": pair_before,
                    "pair_factor_after": pair_after,
                    "factor_lost": pair_before and not pair_after,
                    "action_pair_factor_q": pair_factor_q,
                    "learned_factor_q_total": learned_q_total,
                    "coverage_shaping_delta": coverage_delta,
                    "frontier_constraint_fixable": (
                        fixable_by_frontier_constraint
                    ),
                    "remaining_safe_for_all_best": (
                        remaining_safe_for_all_best
                    ),
                    "frontier_best_actions": "|".join(
                        str(action) for action in max_score_actions
                    ),
                    "frontier_best_score": max_robust_score,
                    "frontier_greedy_score": greedy_robust_score,
                    "before_visible": "|".join(map(str, sorted(before_visible))),
                    "after_visible": "|".join(map(str, sorted(after_visible))),
                    "protected_slots": "|".join(map(str, sorted(protected))),
                })
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(losses),
        pd.DataFrame(quorum_transitions),
    )


def visibility_summary(summaries):
    rows = []
    cohorts = {
        "strict": summaries["strict"],
        "far_failure": summaries["far"] & ~summaries["strict"],
        "all_capture": np.ones(len(summaries), dtype=bool),
    }
    for name, mask in cohorts.items():
        cohort = summaries.loc[mask]
        for field in ("last_two", "last_one", "first_zero"):
            values = cohort[field].dropna()
            rows.append({
                "cohort": name,
                "field": field,
                "n": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "p25": values.quantile(.25) if len(values) else np.nan,
                "p75": values.quantile(.75) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def loss_summary(losses):
    if losses.empty:
        return pd.DataFrame()
    rows = []
    cohorts = {
        "all": np.ones(len(losses), dtype=bool),
        "far_failure": losses["far"] & ~losses["strict"],
        "strict": losses["strict"],
    }
    for cohort_name, mask in cohorts.items():
        cohort = losses.loc[mask]
        for category in ("exploration_first", "greedy_first", "other"):
            count = int((cohort["category"] == category).sum())
            rows.append({
                "cohort": cohort_name,
                "category": category,
                "n": count,
                "share": count / float(len(cohort)) if len(cohort) else np.nan,
                "total": len(cohort),
            })
    return pd.DataFrame(rows)


def quorum_transition_summary(transitions):
    if transitions.empty:
        return pd.DataFrame()
    rows = []
    cohorts = {
        "all": np.ones(len(transitions), dtype=bool),
        "far_failure": transitions["far"] & ~transitions["strict"],
        "strict": transitions["strict"],
    }
    for name, mask in cohorts.items():
        cohort = transitions.loc[mask]
        rows.append({
            "cohort": name,
            "n": len(cohort),
            "guard_rate": cohort["guard_applied"].mean(),
            "greedy_quorum_preserve": cohort[
                "greedy_preserves_quorum"
            ].mean(),
            "executed_quorum_preserve": cohort[
                "executed_preserves_quorum"
            ].mean(),
            "next_exact_one": (cohort["next_observer_count"] == 1).mean(),
            "next_zero": (cohort["next_observer_count"] == 0).mean(),
            "action_pair_factor": cohort["action_pair_factor"].mean(),
            "margin_median": cohort["mean_observer_margin"].median(),
        })
    return pd.DataFrame(rows)


def quorum_transition_stages(transitions):
    if transitions.empty:
        return pd.DataFrame()
    staged = transitions.copy()
    staged["phase"] = staged["training_step"].map(_phase)
    rows = []
    for phase in ("0-20k", "20-40k", "40-60k", "60-80k"):
        for name, mask in (
                ("all", np.ones(len(staged), dtype=bool)),
                ("far_failure", staged["far"] & ~staged["strict"]),
        ):
            cohort = staged.loc[(staged["phase"] == phase) & mask]
            rows.append({
                "phase": phase,
                "cohort": name,
                "n": len(cohort),
                "greedy_preserve": cohort[
                    "greedy_preserves_quorum"
                ].mean(),
                "executed_preserve": cohort[
                    "executed_preserves_quorum"
                ].mean(),
                "guard_rate": cohort["guard_applied"].mean(),
                "margin_median": cohort["mean_observer_margin"].median(),
            })
    return pd.DataFrame(rows)


def q_stability(data):
    train = data["train"]
    fields = [
        "q_target_mean", "q_target_std", "q_tot_mean", "td_abs_mean",
        "td_abs_max", "q_network_grad_norm", "policy_grad_norm",
        "reward_normalization_mean", "reward_normalization_std",
    ]
    output = {}
    for field in fields:
        values = pd.to_numeric(train[field], errors="coerce")
        output[field] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "finite": bool(np.isfinite(values.dropna()).all()),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--runs", nargs="+", type=int, default=[167, 168])
    args = parser.parse_args()

    for run in args.runs:
        run_dir = args.run_root / ("run{}".format(run))
        data = load_run(run_dir)
        overall, episodes = macro(data)
        exposure, exposure_stage = guard_exposure(data)
        vis, losses, quorum_transitions = visibility_records(data)
        print("\n===== RUN {} =====".format(run))
        print("MACRO", json.dumps(overall, sort_keys=True))
        print("STAGES\n{}".format(stage_table(episodes).to_string(index=False)))
        print("FAR_BANDS\n{}".format(far_bands(episodes).to_string(index=False)))
        print("EVAL\n{}".format(data["eval"][[
            "step", "eval_capture_events", "eval_win_rate",
            "eval_average_episode_rewards",
        ]].to_string(index=False)))
        print("GUARD", json.dumps(exposure, sort_keys=True))
        if not exposure_stage.empty:
            print("GUARD_STAGES\n{}".format(exposure_stage.to_string(index=False)))
        print("VISIBILITY\n{}".format(
            visibility_summary(vis).to_string(index=False)
        ))
        print("T2_TO_T1\n{}".format(loss_summary(losses).to_string(index=False)))
        print("EXACT_TWO_TRANSITIONS\n{}".format(
            quorum_transition_summary(quorum_transitions).to_string(index=False)
        ))
        print("EXACT_TWO_STAGES\n{}".format(
            quorum_transition_stages(quorum_transitions).to_string(index=False)
        ))
        if not losses.empty:
            detail = losses.loc[
                losses["far"] & ~losses["strict"]
            ].sort_values(["training_step", "episode"]).copy()
            print("FAR_FAILURE_T2_TO_T1_AGG", json.dumps({
                "n": int(len(detail)),
                "guard_applied": int(detail["guard_applied"].sum()),
                "protected": int(detail["protected"].sum()),
                "random_replaced": int(detail["random_replaced"].sum()),
                "greedy_progress_mean": float(detail["greedy_progress"].mean()),
                "executed_progress_mean": float(detail["executed_progress"].mean()),
                "pair_factor_before_share": float(detail[
                    "pair_factor_before"
                ].mean()),
                "factor_lost_share": float(detail["factor_lost"].mean()),
                "message_margin_median": float(detail[
                    "message_margin"
                ].median()),
                "greedy_target_over_remaining_share": float((
                    detail["greedy_target_progress"]
                    > detail["greedy_progress"]
                ).mean()),
                "turn_or_stay_action_share": float(detail[
                    "greedy_action"
                ].isin([4, 5, 6]).mean()),
                "positive_or_zero_coverage_reward_share": float((
                    detail["coverage_shaping_delta"] >= 0.0
                ).mean()),
                "positive_coverage_reward_share": float((
                    detail["coverage_shaping_delta"] > 0.0
                ).mean()),
                "action_pair_factor_share": float(detail[
                    "pair_factor_after"
                ].mean()),
                "action_pair_factor_q_mean": float(detail[
                    "action_pair_factor_q"
                ].mean()),
                "frontier_constraint_fixable_share": float(detail[
                    "frontier_constraint_fixable"
                ].mean()),
                "remaining_safe_for_all_best_share": float(detail[
                    "remaining_safe_for_all_best"
                ].mean()),
            }, sort_keys=True))
            print("FAR_FAILURE_CASES\n{}".format(detail.head(12).to_string(index=False)))
        print("Q_STABILITY", json.dumps(q_stability(data), sort_keys=True))


if __name__ == "__main__":
    main()
