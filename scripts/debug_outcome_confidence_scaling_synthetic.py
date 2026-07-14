#!/usr/bin/env python
"""Synthetic checks for sign-independent capture-outcome confidence scaling."""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pair_credit import (  # noqa: E402
    compute_capture_to_win_triplet_outcome_advantage,
    scale_capture_to_win_outcome_credit,
)
from utils.graph_sampling import (  # noqa: E402
    require_ready_graph_advantage,
    write_graph_advantage_sequence,
)


def _scale(outcome, graph, valid=None, coefficient=0.15, cap=0.25):
    graph = np.asarray(graph, dtype=np.float32)
    if valid is None:
        valid = np.ones_like(graph, dtype=bool)
    return scale_capture_to_win_outcome_credit(
        triplet_outcome_advantage=np.asarray(outcome, dtype=np.float32),
        graph_advantage=graph,
        valid_graph_transition=np.asarray(valid, dtype=bool),
        coefficient=coefficient,
        cap=cap,
        return_diagnostics=True,
    )


def test_run62_negative_graph_dead_zone_is_removed():
    quality = np.zeros((2, 2, 2), dtype=np.float32)
    quality[0, 0, 0] = 1.0
    quality[0, 1, 1] = 1.0
    outcome = compute_capture_to_win_triplet_outcome_advantage(
        episode_success=np.array([True, False]),
        triplet_capture_quality=quality,
    )["triplet_outcome_advantage"]
    scaled = _scale(outcome, graph=[[-2.0, -3.0], [0.0, 0.0]])
    assert scaled["credit"][0, 0, 0] > 0.0
    assert scaled["credit"][0, 1, 1] < 0.0
    assert scaled["gate_to_credit_drop_fraction"] == 0.0


def test_graph_sign_cannot_flip_or_erase_outcome_sign():
    outcome = np.zeros((1, 2, 1), dtype=np.float32)
    outcome[0, 0, 0] = 0.5
    outcome[0, 1, 0] = -0.5
    positive_graph = _scale(outcome, graph=[[2.0, 2.0]])["credit"]
    negative_graph = _scale(outcome, graph=[[-2.0, -2.0]])["credit"]
    assert np.allclose(positive_graph, negative_graph, atol=1e-7)
    assert positive_graph[0, 0, 0] > 0.0
    assert positive_graph[0, 1, 0] < 0.0


def test_zero_confidence_is_explicitly_diagnosed():
    outcome = np.array([[[0.5], [-0.5]]], dtype=np.float32)
    scaled = _scale(outcome, graph=[[0.0, 0.0]])
    assert np.count_nonzero(scaled["credit"]) == 0
    assert scaled["positive_zero_confidence_fraction"] == 1.0
    assert scaled["negative_zero_confidence_fraction"] == 1.0
    assert scaled["gate_to_credit_drop_fraction"] == 1.0


def test_invalid_transition_cannot_receive_credit():
    outcome = np.array([[[0.5], [-0.5]]], dtype=np.float32)
    scaled = _scale(
        outcome,
        graph=[[-2.0, -2.0]],
        valid=[[True, False]],
    )
    assert scaled["credit"][0, 0, 0] > 0.0
    assert scaled["credit"][0, 1, 0] == 0.0


def test_symmetric_clip_preserves_both_branches():
    outcome = np.array([[[0.5], [-0.5]]], dtype=np.float32)
    scaled = _scale(
        outcome,
        graph=[[-10.0, -10.0]],
        coefficient=1.0,
        cap=0.1,
    )
    assert np.allclose(scaled["credit"].reshape(-1), [1.0, -1.0])
    assert scaled["positive_clip_fraction"] == 1.0
    assert scaled["negative_clip_fraction"] == 1.0
    assert np.isclose(
        scaled["postclip_positive_mass"],
        scaled["postclip_negative_mass"],
    )


def test_nonfinite_inputs_are_finite_and_local():
    outcome = np.zeros((2, 2, 2), dtype=np.float32)
    outcome[0, 0, 0] = np.nan
    outcome[1, 1, 1] = -0.5
    scaled = _scale(outcome, graph=[[np.nan, 1.0], [1.0, -2.0]])
    assert np.isfinite(scaled["credit"]).all()
    assert scaled["credit"][1, 1, 1] < 0.0
    assert np.count_nonzero(scaled["credit"]) == 1


def test_computed_graph_advantage_is_persisted_with_ready_state():
    storage = np.full((3, 4, 1), 99.0, dtype=np.float32)
    ready = np.ones_like(storage, dtype=bool)
    graph_advantage = np.array(
        [[-2.0, 3.0], [0.0, 4.0], [5.0, -6.0]],
        dtype=np.float32,
    )
    valid = np.array(
        [[True, True], [True, False], [False, True]],
        dtype=bool,
    )
    write_graph_advantage_sequence(
        storage,
        ready,
        episode_indices=np.array([1, 3]),
        graph_advantage=graph_advantage,
        valid_transition=valid,
    )
    assert np.array_equal(
        storage[:, [1, 3], 0],
        np.where(valid, graph_advantage, 0.0),
    )
    assert np.array_equal(ready[:, [1, 3], 0], valid)
    assert np.all(storage[:, [0, 2], 0] == 99.0)


def test_unwritten_graph_advantage_fails_before_credit_scaling():
    graph_advantage = np.zeros((2, 2), dtype=np.float32)
    ready = np.array([[True, False], [True, True]], dtype=bool)
    labelled = np.array([[False, True], [True, False]], dtype=bool)
    try:
        require_ready_graph_advantage(graph_advantage, ready, labelled)
    except RuntimeError as exc:
        assert "unwritten graph advantage" in str(exc)
    else:
        raise AssertionError("unwritten confidence source was silently accepted")


def test_genuine_zero_graph_advantage_remains_valid_and_explicit():
    graph_advantage = np.array([[0.0, -2.0]], dtype=np.float32)
    ready = np.ones_like(graph_advantage, dtype=bool)
    labelled = np.ones_like(graph_advantage, dtype=bool)
    selected = require_ready_graph_advantage(
        graph_advantage,
        ready,
        labelled,
    )
    assert selected[0, 0] == 0.0
    assert selected[0, 1] == -2.0


def test_persisted_graph_advantage_reaches_signed_credit_end_to_end():
    storage = np.zeros((1, 2, 1), dtype=np.float32)
    ready = np.zeros_like(storage, dtype=bool)
    write_graph_advantage_sequence(
        storage,
        ready,
        episode_indices=np.array([0, 1]),
        graph_advantage=np.array([[-2.0, 3.0]], dtype=np.float32),
        valid_transition=np.array([[True, True]], dtype=bool),
    )
    gate = np.array([[[0.5], [-0.5]]], dtype=np.float32)
    source = require_ready_graph_advantage(
        storage[..., 0],
        ready[..., 0],
        np.any(np.abs(gate) > 0.0, axis=2),
    )
    scaled = _scale(gate, source)
    assert scaled["credit"][0, 0, 0] > 0.0
    assert scaled["credit"][0, 1, 0] < 0.0
    assert scaled["labelled_graph_advantage_negative_fraction"] == 0.5
    assert scaled["labelled_graph_advantage_positive_fraction"] == 0.5


def main():
    tests = [
        test_run62_negative_graph_dead_zone_is_removed,
        test_graph_sign_cannot_flip_or_erase_outcome_sign,
        test_zero_confidence_is_explicitly_diagnosed,
        test_invalid_transition_cannot_receive_credit,
        test_symmetric_clip_preserves_both_branches,
        test_nonfinite_inputs_are_finite_and_local,
        test_computed_graph_advantage_is_persisted_with_ready_state,
        test_unwritten_graph_advantage_fails_before_credit_scaling,
        test_genuine_zero_graph_advantage_remains_valid_and_explicit,
        test_persisted_graph_advantage_reaches_signed_credit_end_to_end,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} outcome-confidence scaling tests".format(len(tests)))


if __name__ == "__main__":
    main()
