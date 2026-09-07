"""Small dependency-free helpers for SDDFG terminal-credit replay."""

import numpy as np


def terminal_win_replay_lane_indices(
        base_indices,
        terminal_episode_indices,
        cursor):
    """Append one deterministic terminal episode when uniform replay missed it.

    The uniform indices remain an exact prefix and this helper consumes no RNG.
    The returned lane mask identifies only the appended auxiliary episode; the
    learner uses it exclusively for terminal-gated transitions.
    """
    base_indices = np.asarray(base_indices, dtype=np.int64).reshape(-1)
    terminal_episode_indices = np.asarray(
        terminal_episode_indices, dtype=np.int64
    ).reshape(-1)
    lane_mask = np.zeros(base_indices.shape[0], dtype=np.float32)
    cursor = int(cursor)
    if terminal_episode_indices.size == 0:
        return base_indices.copy(), lane_mask, cursor, False
    if np.isin(base_indices, terminal_episode_indices).any():
        return base_indices.copy(), lane_mask, cursor, False

    selected = terminal_episode_indices[
        cursor % terminal_episode_indices.size
    ]
    combined = np.concatenate((base_indices, np.asarray([selected])))
    lane_mask = np.concatenate((lane_mask, np.ones(1, dtype=np.float32)))
    return combined, lane_mask, cursor + 1, True
