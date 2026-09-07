#!/usr/bin/env python
"""Regression fixture for run164's pre-capture reward conflict."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runner.base_runner import (  # noqa: E402
    _build_wolfpack_reward_shaping_checkpoint_contract,
    _validate_wolfpack_reward_shaping_checkpoint_contract,
)
from envs.Wolfpack.wolfpack_penalty_open import (  # noqa: E402
    WolfpackPenaltyOpen,
)
from utils.wolfpack_reward import (  # noqa: E402
    WOLFPACK_CAPTURE_QUORUM,
    WOLFPACK_DISTANCE_SHAPING_SCALE,
    WOLFPACK_MULTI_PREY_COVERAGE_MODE,
    balanced_alive_prey_coverage_cost,
    capture_quorum_balanced_alive_prey_coverage_cost,
    coverage_potential_rewards,
)


def _legacy_nearest_cost(players, foods):
    return float(sum(
        min(
            abs(player[0] - food[0]) + abs(player[1] - food[1])
            for food in foods
        )
        for player in players
    ))


def _expect_runtime_error(fn, message_fragment):
    try:
        fn()
    except RuntimeError as error:
        assert message_fragment in str(error), str(error)
        return
    raise AssertionError("expected RuntimeError containing {!r}".format(
        message_fragment
    ))


def _build_reward_only_env(fixture, coverage_enabled):
    env = WolfpackPenaltyOpen.__new__(WolfpackPenaltyOpen)
    env.max_food_num = 2
    env.max_player_num = 6
    env.num_players = 3
    env.valid_indices = [0, -1, 1, -1, 2, -1]
    env.player_positions = [
        tuple(position) for position in fixture["current_player_positions"]
    ]
    env.food_positions = [
        tuple(position) for position in fixture["current_food_positions"]
    ]
    env.food_alive_statuses = [True, True]
    env.food_frozen_time = [0, 0]
    env.prev_dist_to_food = [3.0, None, 5.0, None, 2.0, None]
    env.prev_team_coverage_cost = fixture["expected_coverage_previous_cost"]
    env.use_multi_prey_coverage_shaping = coverage_enabled
    env.coopRadius = 1
    env.close_penalty = 0.1
    env.groupMultiplier = 2.0
    env.obs_type = "vector"
    env.pads = 0
    env.grid = np.ones((20, 20), dtype=np.int64)
    env.RGB_padded_grid = np.zeros((20, 20, 3), dtype=np.int64)
    return env


def _move(position, action):
    deltas = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
        4: (0, 0),
        5: (0, 0),
        6: (0, 0),
    }
    dx, dy = deltas[int(action)]
    return (position[0] + dx, position[1] + dy)


def _action_costs(cost_fn, players, foods, slot):
    costs = []
    for action in range(7):
        candidate_players = [tuple(position) for position in players]
        candidate_players[slot] = _move(candidate_players[slot], action)
        costs.append(cost_fn(candidate_players, foods))
    return costs


def main():
    fixture_path = os.path.join(
        PROJECT_ROOT,
        "scripts",
        "fixtures",
        "run164_pre_capture_reward_conflict.json",
    )
    with open(fixture_path, "r", encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)

    previous_players = fixture["previous_player_positions"]
    current_players = fixture["current_player_positions"]
    previous_foods = fixture["previous_food_positions"]
    current_foods = fixture["current_food_positions"]

    legacy_previous = _legacy_nearest_cost(previous_players, previous_foods)
    legacy_current = _legacy_nearest_cost(current_players, current_foods)
    legacy_balanced_previous = balanced_alive_prey_coverage_cost(
        previous_players,
        previous_foods,
    )
    legacy_balanced_current = balanced_alive_prey_coverage_cost(
        current_players,
        current_foods,
    )
    coverage_previous = capture_quorum_balanced_alive_prey_coverage_cost(
        previous_players,
        previous_foods,
    )
    coverage_current = capture_quorum_balanced_alive_prey_coverage_cost(
        current_players,
        current_foods,
    )
    assert legacy_previous == fixture["expected_legacy_previous_cost"]
    assert legacy_current == fixture["expected_legacy_current_cost"]
    assert legacy_balanced_previous == fixture["expected_coverage_previous_cost"]
    assert legacy_balanced_current == fixture["expected_coverage_current_cost"]
    # With three wolves, both objectives resolve to the capture-feasible 2+1
    # assignment.  This preserves run164's original conflict correction.
    assert coverage_previous == legacy_balanced_previous
    assert coverage_current == legacy_balanced_current

    legacy_team_shaping = WOLFPACK_DISTANCE_SHAPING_SCALE * (
        legacy_previous - legacy_current
    )
    coverage_team_shaping = WOLFPACK_DISTANCE_SHAPING_SCALE * (
        coverage_previous - coverage_current
    )
    assert np.isclose(
        legacy_team_shaping,
        fixture["expected_legacy_team_shaping"],
    )
    assert np.isclose(
        coverage_team_shaping,
        fixture["expected_coverage_team_shaping"],
    )
    assert legacy_team_shaping > 0.0
    assert coverage_team_shaping < 0.0

    reward_slots = coverage_potential_rewards(
        previous_cost=coverage_previous,
        current_cost=coverage_current,
        active_slots=fixture["active_slots"],
        max_player_num=6,
    )
    assert np.isfinite(reward_slots).all()
    assert np.isclose(reward_slots.sum(), coverage_team_shaping)
    assert np.all(reward_slots[[1, 3, 5]] == 0.0)

    legacy_env = _build_reward_only_env(fixture, coverage_enabled=False)
    legacy_env.update_food_status()
    assert np.isclose(sum(legacy_env.player_points), legacy_team_shaping)
    coverage_env = _build_reward_only_env(fixture, coverage_enabled=True)
    coverage_env.update_food_status()
    assert np.isclose(sum(coverage_env.player_points), coverage_team_shaping)
    assert coverage_env.last_capture_count == 0

    rng_before = np.random.get_state()
    balanced_alive_prey_coverage_cost(previous_players, previous_foods)
    capture_quorum_balanced_alive_prey_coverage_cost(
        previous_players,
        previous_foods,
    )
    coverage_potential_rewards(
        previous_cost=coverage_previous,
        current_cost=coverage_current,
        active_slots=fixture["active_slots"],
        max_player_num=6,
    )
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    assert np.array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]

    args = SimpleNamespace(
        env_name="wolfpack",
        use_multi_prey_coverage_shaping=True,
    )
    checkpoint = _build_wolfpack_reward_shaping_checkpoint_contract(args)
    assert checkpoint["wolfpack_reward_shaping_mode"] == (
        WOLFPACK_MULTI_PREY_COVERAGE_MODE
    )
    _validate_wolfpack_reward_shaping_checkpoint_contract(checkpoint, args)
    _expect_runtime_error(
        lambda: _validate_wolfpack_reward_shaping_checkpoint_contract({}, args),
        "start a fresh run",
    )
    _expect_runtime_error(
        lambda: _validate_wolfpack_reward_shaping_checkpoint_contract(
            {
                "wolfpack_reward_shaping_checkpoint_version": 1,
                "wolfpack_reward_shaping_mode": (
                    "balanced_alive_prey_coverage"
                ),
                "wolfpack_reward_shaping_scale": (
                    WOLFPACK_DISTANCE_SHAPING_SCALE
                ),
            },
            args,
        ),
        "start a fresh run",
    )
    legacy_args = SimpleNamespace(
        env_name="wolfpack",
        use_multi_prey_coverage_shaping=False,
    )
    _expect_runtime_error(
        lambda: _validate_wolfpack_reward_shaping_checkpoint_contract(
            checkpoint,
            legacy_args,
        ),
        "does not match",
    )

    launcher_path = os.path.join(
        PROJECT_ROOT,
        "scripts",
        "train_wolfpack_sddfg_intra_episode_dynamic.sh",
    )
    with open(launcher_path, "r", encoding="utf-8") as launcher_file:
        launcher_source = launcher_file.read()
    assert launcher_source.count("--use_multi_prey_coverage_shaping") == 1

    run165_fixture_path = os.path.join(
        PROJECT_ROOT,
        "scripts",
        "fixtures",
        "run165_capture_quorum_reward_conflict.json",
    )
    with open(run165_fixture_path, "r", encoding="utf-8") as fixture_file:
        run165_fixture = json.load(fixture_file)
    run165_players = run165_fixture["player_positions"]
    run165_foods = run165_fixture["food_positions"]
    remaining_slot = run165_fixture["remaining_side_slot"]
    legacy_balanced_action_costs = _action_costs(
        balanced_alive_prey_coverage_cost,
        run165_players,
        run165_foods,
        remaining_slot,
    )
    quorum_action_costs = _action_costs(
        capture_quorum_balanced_alive_prey_coverage_cost,
        run165_players,
        run165_foods,
        remaining_slot,
    )
    assert np.allclose(
        legacy_balanced_action_costs,
        run165_fixture["legacy_balanced_action_costs"],
    )
    assert np.allclose(
        quorum_action_costs,
        run165_fixture["capture_quorum_action_costs"],
    )
    greedy_action = run165_fixture["greedy_action"]
    assert legacy_balanced_action_costs[greedy_action] == min(
        legacy_balanced_action_costs
    )
    assert quorum_action_costs[greedy_action] > min(quorum_action_costs)
    for action in np.flatnonzero(
            np.isclose(quorum_action_costs, min(quorum_action_costs))):
        assert run165_fixture["remaining_prey_progress_by_action"][action] > 0

    # The old 1-per-prey constraint splits a two-wolf team 1+1, although a
    # capture requires both wolves.  The production objective must choose a
    # capture-feasible 2+0 assignment instead.
    two_wolf_players = [(0, 0), (0, 1)]
    two_prey_positions = [(0, 3), (10, 10)]
    assert balanced_alive_prey_coverage_cost(
        two_wolf_players,
        two_prey_positions,
    ) == 22.0
    assert capture_quorum_balanced_alive_prey_coverage_cost(
        two_wolf_players,
        two_prey_positions,
    ) == 5.0
    assert WOLFPACK_CAPTURE_QUORUM == 2

    print(
        "PASS run164 pre-capture reward conflict: "
        "legacy={:.6f}, coverage={:.6f}, mode={}, quorum={}, "
        "run165_greedy_before={:.1f}, run165_greedy_after={:.1f}, "
        "rng_unchanged=true, checkpoint_fail_loud=true".format(
            legacy_team_shaping,
            coverage_team_shaping,
            WOLFPACK_MULTI_PREY_COVERAGE_MODE,
            WOLFPACK_CAPTURE_QUORUM,
            legacy_balanced_action_costs[greedy_action],
            quorum_action_costs[greedy_action],
        )
    )


if __name__ == "__main__":
    main()
