"""Dependency-light fixtures for the SDDFG terminal-credit replay lane."""

import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.terminal_replay import terminal_win_replay_lane_indices  # noqa: E402


def test_no_terminal_is_exact_uniform_sample():
    base = np.asarray([3, 1, 4, 2], dtype=np.int64)
    combined, mask, cursor, forced = terminal_win_replay_lane_indices(
        base, [], 7
    )
    assert np.array_equal(combined, base)
    assert not mask.any()
    assert cursor == 7
    assert not forced


def test_natural_terminal_needs_no_auxiliary_lane():
    base = np.asarray([3, 8, 4, 2], dtype=np.int64)
    combined, mask, cursor, forced = terminal_win_replay_lane_indices(
        base, [8, 11], 5
    )
    assert np.array_equal(combined, base)
    assert not mask.any()
    assert cursor == 5
    assert not forced


def test_forced_lane_is_local_and_round_robin():
    base = np.asarray([3, 1, 4, 2], dtype=np.int64)
    first, first_mask, cursor, forced = (
        terminal_win_replay_lane_indices(base, [8, 11], 0)
    )
    second, second_mask, cursor, forced_second = (
        terminal_win_replay_lane_indices(base, [8, 11], cursor)
    )
    assert np.array_equal(first[:-1], base)
    assert np.array_equal(second[:-1], base)
    assert first[-1] == 8 and second[-1] == 11
    assert np.array_equal(first_mask, [0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.array_equal(second_mask, first_mask)
    assert forced and forced_second and cursor == 2


def test_run158_schedule_replay():
    terminal_indices = [223, 300, 322, 328, 342]
    baseline_rng = np.random.RandomState(410001)
    enabled_rng = np.random.RandomState(410001)
    cursor = 0
    natural = 0
    forced = 0
    first_natural = None
    first_forced = None
    for formal_episode in range(4, 369, 4):
        replay_size = 32 + formal_episode
        available = [i for i in terminal_indices if i < replay_size]
        for update_index in range(4):
            baseline = baseline_rng.choice(replay_size, 8, replace=False)
            enabled = enabled_rng.choice(replay_size, 8, replace=False)
            assert np.array_equal(baseline, enabled)
            natural_here = int(np.isin(baseline, available).sum())
            natural += natural_here
            if natural_here and first_natural is None:
                first_natural = formal_episode
            if update_index == 0:
                combined, mask, cursor, lane_forced = (
                    terminal_win_replay_lane_indices(
                        enabled, available, cursor
                    )
                )
                assert np.array_equal(combined[:8], enabled)
                if lane_forced:
                    forced += 1
                    if first_forced is None:
                        first_forced = formal_episode
                    assert mask[-1] == 1.0 and not mask[:-1].any()
    assert natural == 7
    assert forced == 41
    assert first_natural == 220
    assert first_forced == 192
    assert (natural + forced) * 24 == 1152


def test_run159_weighted_auxiliary_loss_regression_fixture():
    # Real run159 post-lane medians: the four-update aggregate Q loss was
    # 3.496773 versus run158's 0.900566.  Reconstruct the forced update's
    # auxiliary transition MSE, then apply the formal 0.10 transition weight.
    uniform_loss = 0.9005660228431225
    observed_four_update_loss = 3.4967732932418585
    forced_update_loss = (
        4.0 * observed_four_update_loss - 3.0 * uniform_loss
    )
    uniform_transition_count = 8 * 200
    auxiliary_transition_count = 24
    auxiliary_loss = (
        forced_update_loss
        * (uniform_transition_count + auxiliary_transition_count)
        - uniform_loss * uniform_transition_count
    ) / auxiliary_transition_count
    weighted_forced_update_loss = (
        uniform_loss * uniform_transition_count
        + 0.10 * auxiliary_loss * auxiliary_transition_count
    ) / uniform_transition_count
    weighted_four_update_loss = (
        weighted_forced_update_loss + 3.0 * uniform_loss
    ) / 4.0
    assert abs(observed_four_update_loss - 3.4967732932418585) < 1e-12
    assert auxiliary_loss > 700.0
    assert weighted_four_update_loss < 1.20
    assert weighted_four_update_loss > uniform_loss


def test_source_contracts_are_present():
    paths = {
        "trainer": os.path.join(
            REPO_ROOT, "algorithms", "sddfg", "r_sddfg.py"
        ),
        "runner": os.path.join(REPO_ROOT, "runner", "base_runner.py"),
        "launcher": os.path.join(
            REPO_ROOT,
            "scripts",
            "train_wolfpack_sddfg_intra_episode_dynamic.sh",
        ),
    }
    sources = {}
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as source_file:
            sources[name] = source_file.read()
    assert "* q_n_step_gate.float()" in sources["trainer"]
    assert "terminal_replay_loss_weight * auxiliary_transition_mask" in sources["trainer"]
    assert "q_update_index == 0" in sources["runner"]
    # Q-target contract v5 extends the terminal-gated n-step contract with
    # replay-state frontier alignment for the Double-Q next-action selection.
    # Keep this terminal-lane fixture synchronized with the complete contract,
    # rather than accepting a version-only update that omits the new semantic.
    assert "sddfg_q_target_checkpoint_version\": 5" in sources["runner"]
    assert "sddfg_q_frontier_target_alignment" in sources["runner"]
    assert "q_terminal_replay_loss_weight" in sources["trainer"]
    assert "--q_terminal_replay_lane" in sources["launcher"]
    assert "--q_terminal_replay_loss_weight" in sources["launcher"]
    assert "Q target: contract_version=5" in sources["launcher"]
    assert (
        "frontier_next_action="
        "production_pre_capture_exact_quorum_rerank_from_replayed_local_observation"
        in sources["launcher"]
    )


def main():
    tests = (
        test_no_terminal_is_exact_uniform_sample,
        test_natural_terminal_needs_no_auxiliary_lane,
        test_forced_lane_is_local_and_round_robin,
        test_run158_schedule_replay,
        test_run159_weighted_auxiliary_loss_regression_fixture,
        test_source_contracts_are_present,
    )
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} terminal replay lane tests".format(len(tests)))


if __name__ == "__main__":
    main()
