#!/usr/bin/env python
"""Synthetic replay-alignment checks for Markov topology persistence."""

from __future__ import print_function

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.graph_sampling import (
    build_previous_adjacency_sequence,
    build_previous_done_sequence,
    select_outcome_contrast_complete_episodes,
)


def main():
    adjacency = np.zeros((4, 2, 3, 2), dtype=np.int64)
    adjacency[0, 0, [0, 1], 0] = 1
    adjacency[1, 0, [1, 2], 0] = 1
    adjacency[2, 0, [0, 2], 1] = 1
    adjacency[0, 1, [0, 2], 0] = 1
    adjacency[1, 1, [0, 1, 2], 1] = 1

    previous = build_previous_adjacency_sequence(adjacency)
    assert previous.shape == adjacency.shape
    assert not previous[0].any(), "episode starts must not inherit a graph"
    assert np.array_equal(previous[1], adjacency[0])
    assert np.array_equal(previous[2], adjacency[1])
    assert np.array_equal(previous[3], adjacency[2])
    assert np.array_equal(previous[1, 1], adjacency[0, 1])
    assert not np.array_equal(previous[1, 1], adjacency[-1, 0])
    print("PASS previous-adjacency shift is episode-local")

    empty = np.zeros((0, 2, 3, 2), dtype=np.int64)
    assert build_previous_adjacency_sequence(empty).shape == empty.shape
    print("PASS empty sequence")

    # done[t] is emitted after action_t: terminal t remains valid, while t+1
    # is masked. Two episode columns must remain isolated.
    post_action_done = np.zeros((4, 2, 1), dtype=np.float32)
    post_action_done[1, 0, 0] = 1.0
    post_action_done[2, 1, 0] = 1.0
    previous_done = build_previous_done_sequence(post_action_done)
    assert not previous_done[0].any()
    assert previous_done[1, 0, 0] == 0.0
    assert previous_done[2, 0, 0] == 1.0
    assert previous_done[2, 1, 0] == 0.0
    assert previous_done[3, 1, 0] == 1.0
    print("PASS post-action terminal mask is shifted and episode-local")

    # Episode-major flattening must preserve each episode's independent time
    # sequence; this is the layout consumed by recurrent data chunks.
    flattened = previous_done.transpose(1, 0, 2).reshape(-1, 1)
    assert np.array_equal(flattened[:4], previous_done[:, 0])
    assert np.array_equal(flattened[4:], previous_done[:, 1])
    print("PASS multi-episode done mask flattening")

    # The replay cohort used by the optimizer must retain both real outcome
    # signs whenever the occupied buffer supports a centered contrast.
    contrast = select_outcome_contrast_complete_episodes(
        np.asarray([3], dtype=np.int64),
        np.asarray([False, True, False, False]),
        np.asarray([False, False, False, True]),
        np.asarray([3, 2, 1, 0], dtype=np.int64),
    )
    assert np.array_equal(contrast["episode_indices"], [3, 1])
    assert contrast["class_complete"] == 1.0
    print("PASS centered outcome replay support survives recent-window sampling")
    print("PASS all topology-persistence synthetic tests")


if __name__ == "__main__":
    main()
