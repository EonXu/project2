#!/usr/bin/env python
"""Regression tests for adjacency stale-trust control metric selection."""

import math
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from utils.adj_training_control import (
    aggregate_adj_control_populations,
    advance_recent_episode_window,
    select_adj_control_population,
    select_adj_control_ratios,
    should_stop_adj_ppo,
    validate_adj_control_application,
)
from utils.pair_credit import partition_pair_contrast_optimizer_chunks


def _control_metrics(
        raw_graph=(52.0, 100.0),
        raw_factor=(41.0, 100.0),
        trusted_graph=(18.0, 100.0),
        trusted_factor=(9.0, 100.0)):
    return {
        "clamp_ratio": raw_graph[0] / raw_graph[1],
        "factor_clamp_ratio": raw_factor[0] / raw_factor[1],
        "trusted_clamp_ratio": trusted_graph[0] / trusted_graph[1],
        "trusted_factor_clamp_ratio": (
            trusted_factor[0] / trusted_factor[1]
        ),
        "_adj_control_raw_graph_numerator": raw_graph[0],
        "_adj_control_raw_graph_denominator": raw_graph[1],
        "_adj_control_raw_factor_numerator": raw_factor[0],
        "_adj_control_raw_factor_denominator": raw_factor[1],
        "_adj_control_trusted_graph_numerator": trusted_graph[0],
        "_adj_control_trusted_graph_denominator": trusted_graph[1],
        "_adj_control_trusted_factor_numerator": trusted_factor[0],
        "_adj_control_trusted_factor_denominator": trusted_factor[1],
    }


def test_trusted_control_uses_the_effective_loss_population():
    graph_ratio, factor_ratio = select_adj_control_ratios(
        _control_metrics(),
        use_stale_trust=True,
    )
    assert graph_ratio == 0.18
    assert factor_ratio == 0.09


def test_raw_control_is_preserved_when_stale_trust_is_disabled():
    graph_ratio, factor_ratio = select_adj_control_ratios(
        _control_metrics(),
        use_stale_trust=False,
    )
    assert graph_ratio == 0.52
    assert factor_ratio == 0.41


def test_enabled_stale_trust_fails_loud_without_trusted_metrics():
    try:
        select_adj_control_ratios(
            {
                "clamp_ratio": 0.52,
                "factor_clamp_ratio": 0.41,
            },
            use_stale_trust=True,
        )
    except RuntimeError as exc:
        assert "trusted" in str(exc)
    else:
        raise AssertionError("missing trusted metrics did not fail loud")


def test_non_finite_control_metric_fails_loud():
    metrics = _control_metrics()
    metrics["trusted_clamp_ratio"] = math.nan
    try:
        select_adj_control_ratios(
            metrics,
            use_stale_trust=True,
        )
    except RuntimeError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite trusted metric did not fail loud")


def test_control_population_uses_total_mass_not_mean_of_ratios():
    first = select_adj_control_population(
        _control_metrics(
            trusted_graph=(10.0, 100.0),
            trusted_factor=(5.0, 50.0),
        ),
        use_stale_trust=True,
    )
    second = select_adj_control_population(
        _control_metrics(
            trusted_graph=(1.0, 1.0),
            trusted_factor=(1.0, 2.0),
        ),
        use_stale_trust=True,
    )
    aggregate = aggregate_adj_control_populations([first, second])
    assert abs(aggregate["graph_ratio"] - 11.0 / 101.0) < 1e-12
    assert abs(aggregate["factor_ratio"] - 6.0 / 52.0) < 1e-12
    assert abs(aggregate["graph_ratio"] - 0.55) > 0.4
    assert aggregate["graph_denominator"] == 101.0
    assert aggregate["factor_denominator"] == 52.0


def test_reported_ratio_must_match_population_totals():
    metrics = _control_metrics()
    metrics["trusted_clamp_ratio"] = 0.19
    try:
        select_adj_control_population(metrics, use_stale_trust=True)
    except RuntimeError as exc:
        assert "numerator/denominator" in str(exc)
    else:
        raise AssertionError("inconsistent control population was accepted")


def test_odd_replay_population_is_partitioned_without_drop():
    import numpy as np

    parts = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.random.RandomState(7).permutation(5),
        chunk_episode_membership=np.eye(5, dtype=bool),
        pair_evidence_episode_mask=np.zeros(5, dtype=bool),
        episode_success=np.zeros(5, dtype=bool),
        num_mini_batch=2,
    )["partitions"]
    assert sorted(len(part) for part in parts) == [2, 3]
    flattened = np.concatenate(parts)
    assert sorted(flattened.tolist()) == [0, 1, 2, 3, 4]
    assert len(np.unique(flattened)) == 5


def test_single_replay_chunk_has_one_nonempty_partition():
    import numpy as np

    parts = partition_pair_contrast_optimizer_chunks(
        chunk_permutation=np.asarray([0], dtype=np.int64),
        chunk_episode_membership=np.ones((1, 1), dtype=bool),
        pair_evidence_episode_mask=np.zeros(1, dtype=bool),
        episode_success=np.zeros(1, dtype=bool),
        num_mini_batch=2,
    )["partitions"]
    assert len(parts) == 1
    assert parts[0].tolist() == [0]


def test_final_configured_epoch_is_not_reported_as_early_stop():
    assert not should_stop_adj_ppo(
        epochs_ran=2,
        configured_epochs=2,
        graph_ratio=0.9,
        factor_ratio=0.9,
        graph_stop_ratio=0.35,
        factor_stop_ratio=0.35,
        min_epochs=1,
    )


def test_trusted_violation_stops_only_when_an_epoch_is_missing():
    assert should_stop_adj_ppo(
        epochs_ran=1,
        configured_epochs=2,
        graph_ratio=0.4,
        factor_ratio=0.1,
        graph_stop_ratio=0.35,
        factor_stop_ratio=0.35,
        min_epochs=1,
    )


def test_trusted_recent_window_recovers_from_raw_only_staleness():
    state = {
        "window": 1,
        "high_count": 0,
        "low_count": 0,
    }
    for _ in range(2):
        state = advance_recent_episode_window(
            current_window=state["window"],
            configured_window=4,
            min_window=1,
            previous_graph_ratio=0.24,
            previous_factor_ratio=0.10,
            graph_stale_threshold=0.35,
            factor_stale_threshold=0.30,
            recover_graph_threshold=0.28,
            recover_factor_threshold=0.24,
            shrink_patience=1,
            recover_patience=2,
            severe_margin=0.20,
            emergency_window=1,
            emergency_graph_threshold=0.40,
            emergency_factor_threshold=0.25,
            high_count=state["high_count"],
            low_count=state["low_count"],
        )
    assert state["window"] == 2
    assert state["recovered"] == 1.0
    assert state["emergency_shrunk"] == 0.0


def test_recent_window_initial_state_keeps_configured_diversity():
    state = advance_recent_episode_window(
        current_window=4,
        configured_window=4,
        min_window=1,
        previous_graph_ratio=math.nan,
        previous_factor_ratio=math.nan,
        graph_stale_threshold=0.35,
        factor_stale_threshold=0.30,
        recover_graph_threshold=0.28,
        recover_factor_threshold=0.24,
        shrink_patience=1,
        recover_patience=2,
        severe_margin=0.20,
        emergency_window=1,
        emergency_graph_threshold=0.40,
        emergency_factor_threshold=0.25,
        high_count=0,
        low_count=0,
    )
    assert state["window"] == 4
    assert state["shrunk"] == 0.0


def test_recent_window_shrink_is_an_event_not_a_state():
    state = advance_recent_episode_window(
        current_window=4,
        configured_window=4,
        min_window=1,
        previous_graph_ratio=0.36,
        previous_factor_ratio=0.10,
        graph_stale_threshold=0.35,
        factor_stale_threshold=0.30,
        recover_graph_threshold=0.28,
        recover_factor_threshold=0.24,
        shrink_patience=1,
        recover_patience=2,
        severe_margin=0.20,
        emergency_window=1,
        emergency_graph_threshold=0.80,
        emergency_factor_threshold=0.80,
        high_count=0,
        low_count=0,
    )
    assert state["window"] == 3
    assert state["shrunk"] == 1.0
    assert state["emergency_shrunk"] == 0.0

    unchanged = advance_recent_episode_window(
        current_window=1,
        configured_window=4,
        min_window=1,
        previous_graph_ratio=0.36,
        previous_factor_ratio=0.10,
        graph_stale_threshold=0.35,
        factor_stale_threshold=0.30,
        recover_graph_threshold=0.28,
        recover_factor_threshold=0.24,
        shrink_patience=1,
        recover_patience=2,
        severe_margin=0.20,
        emergency_window=1,
        emergency_graph_threshold=0.80,
        emergency_factor_threshold=0.80,
        high_count=0,
        low_count=0,
    )
    assert unchanged["window"] == 1
    assert unchanged["shrunk"] == 0.0
    assert unchanged["emergency_shrunk"] == 0.0


def test_emergency_shrink_is_reported_only_when_window_changes():
    changed = advance_recent_episode_window(
        current_window=4,
        configured_window=4,
        min_window=1,
        previous_graph_ratio=0.90,
        previous_factor_ratio=0.10,
        graph_stale_threshold=0.35,
        factor_stale_threshold=0.30,
        recover_graph_threshold=0.28,
        recover_factor_threshold=0.24,
        shrink_patience=3,
        recover_patience=2,
        severe_margin=0.20,
        emergency_window=1,
        emergency_graph_threshold=0.80,
        emergency_factor_threshold=0.80,
        high_count=0,
        low_count=0,
    )
    assert changed["window"] == 1
    assert changed["shrunk"] == 1.0
    assert changed["emergency_shrunk"] == 1.0

    unchanged = advance_recent_episode_window(
        current_window=1,
        configured_window=4,
        min_window=1,
        previous_graph_ratio=0.90,
        previous_factor_ratio=0.10,
        graph_stale_threshold=0.35,
        factor_stale_threshold=0.30,
        recover_graph_threshold=0.28,
        recover_factor_threshold=0.24,
        shrink_patience=3,
        recover_patience=2,
        severe_margin=0.20,
        emergency_window=1,
        emergency_graph_threshold=0.80,
        emergency_factor_threshold=0.80,
        high_count=0,
        low_count=0,
    )
    assert unchanged["window"] == 1
    assert unchanged["shrunk"] == 0.0
    assert unchanged["emergency_shrunk"] == 0.0


def test_runtime_contract_rejects_raw_control_under_stale_trust():
    try:
        validate_adj_control_application(
            use_stale_trust=True,
            raw_graph_ratio=0.52,
            raw_factor_ratio=0.41,
            trusted_graph_ratio=0.18,
            trusted_factor_ratio=0.09,
            control_graph_ratio=0.52,
            control_factor_ratio=0.41,
            epochs_ran=1,
            configured_epochs=2,
            early_stop_triggered=True,
            graph_stop_ratio=0.35,
            factor_stop_ratio=0.35,
            min_epochs=1,
        )
    except RuntimeError as exc:
        assert "trusted" in str(exc)
    else:
        raise AssertionError("raw control was accepted under stale trust")


def test_runtime_contract_rejects_last_epoch_false_early_stop():
    try:
        validate_adj_control_application(
            use_stale_trust=True,
            raw_graph_ratio=0.60,
            raw_factor_ratio=0.10,
            trusted_graph_ratio=0.40,
            trusted_factor_ratio=0.10,
            control_graph_ratio=0.40,
            control_factor_ratio=0.10,
            epochs_ran=2,
            configured_epochs=2,
            early_stop_triggered=True,
            graph_stop_ratio=0.35,
            factor_stop_ratio=0.35,
            min_epochs=1,
        )
    except RuntimeError as exc:
        assert "actual truncation" in str(exc)
    else:
        raise AssertionError("completed update was reported as early-stop")


def test_runner_uses_the_audited_control_path():
    runner_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runner",
        "base_runner.py",
    )
    with open(runner_path, "r", encoding="utf-8") as runner_file:
        source = runner_file.read()
    assert "validate_adj_control_application(" in source
    assert "_last_adj_control_graph_ratio" in source
    assert "_last_adj_control_factor_ratio" in source
    assert "_last_adj_graph_stale_ratio" not in source
    assert "_last_adj_factor_stale_ratio" not in source
    assert "np.mean(epoch_control_clip_ratios)" not in source
    assert "np.mean(update_control_clip_ratios)" not in source


def main():
    tests = [
        test_trusted_control_uses_the_effective_loss_population,
        test_raw_control_is_preserved_when_stale_trust_is_disabled,
        test_enabled_stale_trust_fails_loud_without_trusted_metrics,
        test_non_finite_control_metric_fails_loud,
        test_control_population_uses_total_mass_not_mean_of_ratios,
        test_reported_ratio_must_match_population_totals,
        test_odd_replay_population_is_partitioned_without_drop,
        test_single_replay_chunk_has_one_nonempty_partition,
        test_final_configured_epoch_is_not_reported_as_early_stop,
        test_trusted_violation_stops_only_when_an_epoch_is_missing,
        test_trusted_recent_window_recovers_from_raw_only_staleness,
        test_recent_window_initial_state_keeps_configured_diversity,
        test_recent_window_shrink_is_an_event_not_a_state,
        test_emergency_shrink_is_reported_only_when_window_changes,
        test_runtime_contract_rejects_raw_control_under_stale_trust,
        test_runtime_contract_rejects_last_epoch_false_early_stop,
        test_runner_uses_the_audited_control_path,
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} adjacency stale-trust control tests".format(len(tests)))


if __name__ == "__main__":
    main()
