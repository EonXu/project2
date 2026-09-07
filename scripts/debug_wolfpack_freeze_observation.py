"""Focused fixture for Wolfpack simultaneous-prey state observability."""

import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.Wolfpack.wolfpack_penalty_open import WolfpackPenaltyOpen  # noqa: E402
from runner.wolfpack_runner import WolfpackRunner  # noqa: E402


def _fixture(freeze_remaining):
    env = WolfpackPenaltyOpen.__new__(WolfpackPenaltyOpen)
    env.max_player_num = 2
    env.max_food_num = 2
    env.unit_obs_dim = 2 + 4 + 2 * (env.max_player_num - 1) + 7 * env.max_food_num + 4 + 1
    env.state_dim = 7 * env.max_player_num + 8 * env.max_food_num + 1
    env.valid_indices = [0, 1]
    env.player_positions = [(0, 0), (0, 1)]
    env.player_orientation = [0, 0]
    env.food_positions = [(9, 9), (2, 2)]
    env.food_orientation = [0, 0]
    env.food_alive_statuses = [False, True]
    env.food_frozen_time = [freeze_remaining, 0]
    env.food_freeze_rate = 25
    env.sight_radius = 8
    env._possibleCoordinates_set = {
        (x, y) for x in range(20) for y in range(20)
    }
    env.remaining_timesteps = 100
    env.max_time_steps = 200
    return env


def main():
    early = _fixture(24)
    late = _fixture(1)
    early_obs = early._observation_player_vector_partial(0)
    late_obs = late._observation_player_vector_partial(0)
    early_state = early.get_state()
    late_state = late.get_state()

    assert early_obs.shape == (early.unit_obs_dim,)
    assert early_state.shape == (early.state_dim,)
    assert not np.array_equal(early_obs, late_obs)
    assert not np.array_equal(early_state, late_state)
    first_prey_countdown_index = 2 + 4 + 2 * (early.max_player_num - 1) + 6
    assert np.isclose(early_obs[first_prey_countdown_index], 24.0 / 25.0)
    assert np.isclose(late_obs[first_prey_countdown_index], 1.0 / 25.0)
    first_prey_state_countdown_index = 7 * early.max_player_num + 7
    assert np.isclose(
        early_state[first_prey_state_countdown_index], 24.0 / 25.0
    )
    assert np.isclose(
        late_state[first_prey_state_countdown_index], 1.0 / 25.0
    )

    alive = _fixture(24)
    alive.food_alive_statuses[0] = True
    assert alive._normalized_food_freeze_remaining(0) == 0.0

    zero_window = _fixture(0)
    zero_window.food_freeze_rate = 0
    assert zero_window._normalized_food_freeze_remaining(0) == 0.0

    original_shape = early_obs.shape
    early.valid_indices[0] = -1
    padded = early._observation_player_vector_partial(0)
    assert padded.shape == original_shape
    assert np.all(padded == -1.0)

    post_capture_row = WolfpackRunner._build_post_capture_row(
        environment_episode_id=7,
        training_env_step=1234,
        episode_step=48,
        first_capture_step=47,
        first_capture_target_id=0,
        first_capture_participant_slots=[0, 1],
        step_info={
            "capture_events": [],
            "success_now": False,
            "food_alive_statuses": [False, True],
            "food_freeze_remaining": [24.0 / 25.0, 0.0],
            "food_positions": [[2, 2], [9, 9]],
            "player_slot_positions": [[2, 2], [3, 2], None],
        },
        action_diagnostic={
            "joint_explore": 0,
            "epsilon": 0.75,
            "greedy_actions": [1, 2, 4],
            "selected_actions": [1, 2, 4],
        },
    )
    assert post_capture_row["offset_from_first_capture"] == 1
    assert post_capture_row["food_freeze_remaining"] == "0.96;0"
    assert post_capture_row["min_alive_player_distance_to_food"] == "0;13"
    assert post_capture_row["min_first_participant_distance_to_food"] == "0;13"
    assert post_capture_row["greedy_actions"] == "1;2;4"
    assert post_capture_row["selected_actions"] == "1;2;4"
    print("wolfpack freeze-observation fixture: PASS")


if __name__ == "__main__":
    main()
