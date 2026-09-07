from __future__ import division

import itertools
import numpy as np


WOLFPACK_DISTANCE_SHAPING_SCALE = 0.01
WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION = 2
WOLFPACK_CAPTURE_QUORUM = 2
WOLFPACK_MULTI_PREY_COVERAGE_MODE = (
    "capture_quorum_balanced_alive_prey_coverage"
)
WOLFPACK_LEGACY_DISTANCE_MODE = "independent_nearest_alive_prey"


def balanced_alive_prey_coverage_cost(player_positions, food_positions):
    """Minimum team distance while assigning every alive prey a wolf.

    The legacy shaping independently sends every wolf to its nearest prey and
    therefore permits all wolves to select the same target.  This potential
    keeps the same sum-of-wolf-distances scale, but requires every alive prey
    to receive at least one assignment whenever enough wolves are available.
    It is used only by the training reward; it never enters policy inputs.
    """
    players = [tuple(position) for position in player_positions]
    foods = [tuple(position) for position in food_positions]
    if not players or not foods:
        return 0.0

    for position in players + foods:
        if len(position) != 2 or not np.isfinite(position).all():
            raise ValueError("Wolfpack coverage positions must be finite 2D coordinates")

    distances = np.asarray([
        [
            abs(float(player[0]) - float(food[0]))
            + abs(float(player[1]) - float(food[1]))
            for food in foods
        ]
        for player in players
    ], dtype=np.float64)

    if len(foods) == 1:
        return float(distances[:, 0].sum())

    # Production has at least two active wolves for two prey.  Define the
    # smaller-team fallback as an injective assignment so the helper remains
    # total for fixtures and non-production Wolfpack configurations.
    best = float("inf")
    if len(players) >= len(foods):
        for assignment in itertools.product(
                range(len(foods)), repeat=len(players)):
            if len(set(assignment)) != len(foods):
                continue
            cost = sum(
                distances[player_idx, food_idx]
                for player_idx, food_idx in enumerate(assignment)
            )
            best = min(best, float(cost))
    else:
        for assigned_foods in itertools.permutations(
                range(len(foods)), len(players)):
            cost = sum(
                distances[player_idx, food_idx]
                for player_idx, food_idx in enumerate(assigned_foods)
            )
            best = min(best, float(cost))

    if not np.isfinite(best):
        raise RuntimeError("Wolfpack coverage assignment has no finite solution")
    return best


def capture_quorum_balanced_alive_prey_coverage_cost(
        player_positions,
        food_positions,
        capture_quorum=WOLFPACK_CAPTURE_QUORUM):
    """Minimum distance under capture-feasible multi-prey assignments.

    A one-wolf-per-prey constraint is not capture-feasible in Wolfpack: a
    prey is captured only when at least ``capture_quorum`` wolves are close.
    Assignment feasibility is therefore chosen lexicographically:

    1. maximize the number of prey receiving a capture-ready quorum;
    2. among those assignments, maximize the number of prey with coverage;
    3. among equally feasible assignments, minimize total wolf distance.

    With two prey and quorum two this produces 2+0 for two wolves, 2+1 for
    three wolves, 2+2 for four wolves, and at least 2+2 for larger teams.
    The helper remains reward-only and consumes no RNG.
    """
    players = [tuple(position) for position in player_positions]
    foods = [tuple(position) for position in food_positions]
    capture_quorum = int(capture_quorum)
    if capture_quorum <= 0:
        raise ValueError("Wolfpack capture quorum must be positive")
    if not players or not foods:
        return 0.0

    for position in players + foods:
        if len(position) != 2 or not np.isfinite(position).all():
            raise ValueError(
                "Wolfpack coverage positions must be finite 2D coordinates"
            )

    distances = np.asarray([
        [
            abs(float(player[0]) - float(food[0]))
            + abs(float(player[1]) - float(food[1]))
            for food in foods
        ]
        for player in players
    ], dtype=np.float64)

    candidates = []
    best_feasibility = None
    for assignment in itertools.product(
            range(len(foods)), repeat=len(players)):
        counts = np.bincount(assignment, minlength=len(foods))
        feasibility = (
            int(np.count_nonzero(counts >= capture_quorum)),
            int(np.count_nonzero(counts >= 1)),
        )
        cost = float(sum(
            distances[player_idx, food_idx]
            for player_idx, food_idx in enumerate(assignment)
        ))
        if best_feasibility is None or feasibility > best_feasibility:
            best_feasibility = feasibility
            candidates = [cost]
        elif feasibility == best_feasibility:
            candidates.append(cost)

    if not candidates:
        raise RuntimeError(
            "Wolfpack capture-quorum coverage has no finite assignment"
        )
    best = float(min(candidates))
    if not np.isfinite(best):
        raise RuntimeError(
            "Wolfpack capture-quorum coverage has no finite solution"
        )
    return best


def coverage_potential_rewards(
        previous_cost,
        current_cost,
        active_slots,
        max_player_num,
        scale=WOLFPACK_DISTANCE_SHAPING_SCALE):
    """Distribute a team coverage-potential delta over active slots."""
    previous_cost = float(previous_cost)
    current_cost = float(current_cost)
    scale = float(scale)
    if not all(np.isfinite(value) for value in (
            previous_cost, current_cost, scale)):
        raise ValueError("Wolfpack coverage shaping inputs must be finite")
    if scale < 0.0:
        raise ValueError("Wolfpack coverage shaping scale must be non-negative")
    max_player_num = int(max_player_num)
    slots = sorted({int(slot) for slot in active_slots})
    if max_player_num <= 0:
        raise ValueError("Wolfpack max_player_num must be positive")
    if any(slot < 0 or slot >= max_player_num for slot in slots):
        raise ValueError("Wolfpack coverage shaping active slot is out of range")

    rewards = np.zeros(max_player_num, dtype=np.float64)
    if not slots:
        return rewards
    team_delta = scale * (previous_cost - current_cost)
    rewards[slots] = team_delta / float(len(slots))
    if not np.isfinite(rewards).all():
        raise RuntimeError("Wolfpack coverage shaping produced non-finite rewards")
    return rewards


def terminal_win_reward_components(
        individual_rewards,
        success_now,
        episode_success_before,
        reward_win):
    reward_win = float(reward_win)
    if not np.isfinite(reward_win) or reward_win < 0.0:
        raise ValueError("reward_win must be finite and non-negative")
    base_team_reward = float(np.sum(individual_rewards))
    if not np.isfinite(base_team_reward):
        raise RuntimeError("base Wolfpack team reward must be finite")
    first_success_now = bool(success_now and not episode_success_before)
    terminal_win_reward = reward_win if first_success_now else 0.0
    return (
        base_team_reward,
        terminal_win_reward,
        base_team_reward + terminal_win_reward,
        first_success_now,
    )


def terminal_win_diagnostic_fields(event):
    required = (
        "first_success_now",
        "base_team_reward",
        "terminal_win_reward",
        "team_reward",
    )
    missing = [key for key in required if key not in event]
    if missing:
        raise RuntimeError(
            "Wolfpack terminal-win diagnostic is missing required field(s): "
            + ",".join(missing)
        )
    first_success_now = bool(event["first_success_now"])
    base = float(event["base_team_reward"])
    terminal = float(event["terminal_win_reward"])
    team = float(event["team_reward"])
    if not all(np.isfinite(value) for value in (base, terminal, team)):
        raise RuntimeError("Wolfpack terminal-win diagnostic must be finite")
    tolerance = 64.0 * float(np.finfo(np.float32).eps) * max(
        1.0, abs(base), abs(terminal), abs(team)
    )
    if abs((base + terminal) - team) > tolerance:
        raise RuntimeError(
            "Wolfpack team reward does not equal base plus terminal win reward"
        )
    if (not first_success_now) and abs(terminal) > tolerance:
        raise RuntimeError(
            "Wolfpack terminal win reward repeated outside first success"
        )
    return {
        "first_success_now": int(first_success_now),
        "base_team_reward": base,
        "terminal_win_reward": terminal,
    }
