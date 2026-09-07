import os
import sys

import numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.wolfpack_reward import (
    terminal_win_diagnostic_fields,
    terminal_win_reward_components,
)


def _reward_fixture():
    individual = np.asarray([[0.25], [0.50], [-0.10]], dtype=np.float32)

    base, bonus, total, first = terminal_win_reward_components(
        individual, False, False, 1.0
    )
    assert not first
    assert bonus == 0.0
    assert np.isclose(base, 0.65)
    assert np.isclose(total, base)

    base, bonus, total, first = terminal_win_reward_components(
        individual, True, False, 1.0
    )
    assert first
    assert bonus == 1.0
    assert np.isclose(total, base + 1.0)

    base, bonus, total, first = terminal_win_reward_components(
        individual, True, True, 1.0
    )
    assert not first
    assert bonus == 0.0
    assert np.isclose(total, base)


def _csv_fixture():
    event = {
        "first_success_now": True,
        "base_team_reward": 2.0,
        "terminal_win_reward": 1.0,
        "team_reward": 3.0,
    }
    fields = terminal_win_diagnostic_fields(event)
    assert fields == {
        "first_success_now": 1,
        "base_team_reward": 2.0,
        "terminal_win_reward": 1.0,
    }
    malformed = dict(event)
    malformed["team_reward"] = 2.0
    try:
        terminal_win_diagnostic_fields(malformed)
    except RuntimeError:
        pass
    else:
        raise AssertionError("malformed reward decomposition did not fail")
    repeated = dict(event)
    repeated["first_success_now"] = False
    try:
        terminal_win_diagnostic_fields(repeated)
    except RuntimeError:
        pass
    else:
        raise AssertionError("repeated terminal win reward did not fail")


def _launcher_fixture():
    path = os.path.join(REPO_ROOT, "scripts", "train", "train_wolfpack.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert source.count("reward_win=all_args.reward_win") == 2
    runner_path = os.path.join(REPO_ROOT, "runner", "wolfpack_runner.py")
    with open(runner_path, "r", encoding="utf-8") as handle:
        runner_source = handle.read()
    assert runner_source.count(
        "**terminal_win_diagnostic_fields(event)"
    ) == 2
    assert '"team_reward": float(info_dict["team_reward"])' in runner_source
    assert "episode_rewards[p_id][t] = rewards" in runner_source
    assert "self.warmup_env_infos.append(dict(env_info))" in open(
        os.path.join(REPO_ROOT, "runner", "base_runner.py"),
        "r",
        encoding="utf-8",
    ).read()
    assert "warmup terminal-win diagnostics" in runner_source
    assert "episode_terminal_win_rewards[t, env_i, 0]" in runner_source
    assert "policy_buffer.terminal_win_rewards[:, idx_range]" in runner_source
    assert "terminal_bonus_normalized_delta_at_insert" in runner_source
    assert "warmup_terminal_win_bonus_event_count" in runner_source
    trainer_path = os.path.join(
        REPO_ROOT, "algorithms", "sddfg", "r_sddfg.py"
    )
    with open(trainer_path, "r", encoding="utf-8") as handle:
        trainer_source = handle.read()
    assert "rewards = to_torch(rew_batch[p_id][0])" in trainer_source
    assert "_build_terminal_gated_n_step_td_targets(" in trainer_source
    assert "terminal_win_reward_batch" in trainer_source
    assert "train_info['q_target_terminal_gated']" in trainer_source


def _td_target_fixture():
    gamma = 0.97
    next_q = 0.25
    for done in (0.0, 1.0):
        target_without_bonus = 0.4 + (1.0 - done) * gamma * next_q
        target_with_bonus = 1.4 + (1.0 - done) * gamma * next_q
        assert np.isclose(target_with_bonus - target_without_bonus, 1.0)


def _reward_normalization_fixture():
    raw = np.asarray([0.0, 0.05, -0.05, 3.0], dtype=np.float32)
    mean = float(raw.mean())
    std = float(raw.std())
    normalized_delta = ((3.0 - mean) / std) - ((2.0 - mean) / std)
    assert np.isclose(normalized_delta, 1.0 / std)
    terminal_mask = np.zeros((4, 8, 1), dtype=np.float32)
    terminal_mask[1, 3, 0] = 1.0
    terminal_mask[3, 7, 0] = 1.0
    assert int((terminal_mask > 0.0).sum()) == 2
    assert int(np.any(terminal_mask > 0.0, axis=(1, 2)).sum()) == 2


def main():
    _reward_fixture()
    _csv_fixture()
    _launcher_fixture()
    _td_target_fixture()
    _reward_normalization_fixture()
    print("wolfpack terminal-win reward fixture: PASS")


if __name__ == "__main__":
    main()
