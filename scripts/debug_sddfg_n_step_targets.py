"""Focused fixtures for SDDFG finite-horizon recurrent Q targets."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.r_sddfg import (  # noqa: E402
    _build_n_step_td_targets,
    _build_terminal_replay_loss_mask,
    _build_terminal_replay_loss_population,
    _build_terminal_gated_n_step_td_targets,
)
from utils.terminal_replay import terminal_win_replay_lane_indices  # noqa: E402


def _assert_close(actual, expected, tolerance=1e-6):
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(
            "expected {}, got {}".format(float(expected), float(actual))
        )


def test_one_step_is_exact_legacy_expression():
    torch.manual_seed(7)
    rewards = torch.randn(3, 11, 1)
    dones = torch.zeros(3, 11, 1)
    dones[0, 4] = 1.0
    dones[2, 10] = 1.0
    next_values = torch.randn(3, 11, 1)
    gamma = 0.97
    expected = rewards + (1.0 - dones) * gamma * next_values
    actual = _build_n_step_td_targets(
        rewards, dones, next_values, gamma, 1
    )
    if not torch.equal(actual, expected):
        raise AssertionError("q_n_step=1 changed the legacy TD expression")


def test_terminal_reward_reaches_the_24_step_window():
    rewards = torch.zeros(1, 30, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 123.0)
    rewards[0, 23, 0] = 1.0
    dones[0, 23, 0] = 1.0
    targets = _build_n_step_td_targets(
        rewards, dones, next_values, 0.97, 24
    )
    _assert_close(targets[0, 0, 0], 0.97 ** 23)
    _assert_close(targets[0, 1, 0], 0.97 ** 22)
    if not torch.isfinite(targets).all():
        raise AssertionError("24-step target is non-finite")


def test_run156_terminal_scale_reaches_predecessor_without_a_spike():
    # run156's observed sample-time terminal normalized-delta median was
    # 8.81339721316139 (range 6.6977..20.7900).  Replaying that scale at the
    # end of a 24-transition window makes the old one-step predecessor target
    # exactly zero, while the finite return supplies a discounted 4.3741.
    terminal_delta = 8.81339721316139
    rewards = torch.zeros(1, 24, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    rewards[0, 23, 0] = terminal_delta
    dones[0, 23, 0] = 1.0
    legacy = _build_n_step_td_targets(
        rewards, dones, next_values, 0.97, 1
    )
    finite_return = _build_n_step_td_targets(
        rewards, dones, next_values, 0.97, 24
    )
    _assert_close(legacy[0, 0, 0], 0.0)
    _assert_close(
        finite_return[0, 0, 0], terminal_delta * (0.97 ** 23)
    )
    _assert_close(finite_return[0, 0, 0], 4.374145569035753)


def test_run157_no_terminal_trace_stays_exactly_one_step():
    # This is the first 24 rewards of run157's real step-6600 player-0 trace.
    # At the first logged optimizer update (step 9600), replay used the
    # production mean/std below.  Unconditional n=24 turns its ordinary dense
    # rewards into a large target even though no win marker exists.
    raw_rewards = torch.tensor([
        0.01, -0.08, -0.10, -0.10, -0.10, -0.10,
        -0.01, 0.00, 0.00, 0.00, 0.01, -0.01,
        0.00, 0.00, 0.01, 0.00, -0.01, -0.01,
        0.00, 0.00, 0.00, 0.00, -0.01, 0.00,
    ], dtype=torch.float32).view(1, 24, 1)
    rewards = (
        raw_rewards - (-0.006331250071525574)
    ) / 0.043747838686568834
    dones = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    terminal_win_rewards = torch.zeros_like(rewards)

    one_step = _build_n_step_td_targets(
        rewards, dones, next_values, 0.97, 1
    )
    unconditional = _build_n_step_td_targets(
        rewards, dones, next_values, 0.97, 24
    )
    gated, gate, gated_one_step = _build_terminal_gated_n_step_td_targets(
        rewards,
        dones,
        next_values,
        terminal_win_rewards,
        0.97,
        24,
    )
    if float((unconditional - one_step).abs().max()) <= 4.0:
        raise AssertionError("run157 trace did not reproduce dense-return inflation")
    if not torch.equal(gated, one_step):
        raise AssertionError("no-terminal production trace did not stay one-step")
    if not torch.equal(gated_one_step, one_step):
        raise AssertionError("gated helper changed its one-step reference")
    if bool(gate.any().item()):
        raise AssertionError("no-terminal production trace activated n-step")


def test_terminal_gate_updates_exactly_the_preceding_24_transitions():
    rewards = torch.zeros(1, 30, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 7.0)
    terminal_win_rewards = torch.zeros_like(rewards)
    terminal_delta = 8.81339721316139
    rewards[0, 23, 0] = terminal_delta
    terminal_win_rewards[0, 23, 0] = 1.0
    dones[0, 23, 0] = 1.0

    actual, gate, one_step = _build_terminal_gated_n_step_td_targets(
        rewards,
        dones,
        next_values,
        terminal_win_rewards,
        0.97,
        24,
    )
    full = _build_n_step_td_targets(
        rewards, dones, next_values, 0.97, 24
    )
    if int(gate.sum().item()) != 24:
        raise AssertionError("terminal gate did not cover exactly 24 transitions")
    if not torch.equal(actual[:, :24], full[:, :24]):
        raise AssertionError("terminal-bearing window did not use exact n-step")
    if not torch.equal(actual[:, 24:], one_step[:, 24:]):
        raise AssertionError("non-terminal window did not remain one-step")
    _assert_close(actual[0, 0, 0], 4.374145569035753)


def test_terminal_gate_does_not_cross_an_earlier_episode_boundary():
    rewards = torch.zeros(1, 12, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    terminal_win_rewards = torch.zeros_like(rewards)
    dones[0, 2, 0] = 1.0
    rewards[0, 5, 0] = 9.0
    terminal_win_rewards[0, 5, 0] = 1.0
    dones[0, 5, 0] = 1.0

    actual, gate, one_step = _build_terminal_gated_n_step_td_targets(
        rewards,
        dones,
        next_values,
        terminal_win_rewards,
        0.97,
        8,
    )
    if bool(gate[0, :3].any().item()):
        raise AssertionError("terminal gate crossed an earlier episode boundary")
    if not bool(gate[0, 3:6].all().item()):
        raise AssertionError("reachable terminal predecessors were not gated")
    if not torch.equal(actual[0, :3], one_step[0, :3]):
        raise AssertionError("pre-boundary targets did not remain one-step")


def test_continue_after_success_stops_return_at_the_win_marker():
    rewards = torch.zeros(1, 12, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 1000.0)
    terminal_win_rewards = torch.zeros_like(rewards)
    rewards[0, 4, 0] = 8.0
    terminal_win_rewards[0, 4, 0] = 1.0
    # The formal launcher deliberately keeps done=false after success.  These
    # later values must not enter the task-completion return.
    rewards[0, 5:, 0] = 500.0

    actual, gate, one_step = _build_terminal_gated_n_step_td_targets(
        rewards,
        dones,
        next_values,
        terminal_win_rewards,
        0.97,
        8,
    )
    if int(gate.sum().item()) != 5:
        raise AssertionError("continue-after-success gate covered wrong window")
    _assert_close(actual[0, 0, 0], 8.0 * (0.97 ** 4))
    if not torch.equal(actual[0, 5:], one_step[0, 5:]):
        raise AssertionError("post-success continuation stopped being one-step")


def test_done_mask_blocks_rewards_and_bootstrap_after_terminal():
    rewards = torch.zeros(1, 12, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 17.0)
    rewards[0, 0, 0] = 1.0
    rewards[0, 1, 0] = 2.0
    rewards[0, 2, 0] = 3.0
    rewards[0, 5, 0] = 1000.0
    dones[0, 2, 0] = 1.0
    targets = _build_n_step_td_targets(
        rewards, dones, next_values, 0.9, 8
    )
    expected = 1.0 + 0.9 * 2.0 + (0.9 ** 2) * 3.0
    _assert_close(targets[0, 0, 0], expected)


def test_full_horizon_bootstraps_from_aligned_future_state():
    rewards = torch.zeros(1, 10, 1)
    dones = torch.zeros_like(rewards)
    next_values = torch.arange(1, 11, dtype=torch.float32).view(1, 10, 1)
    targets = _build_n_step_td_targets(
        rewards, dones, next_values, 0.5, 4
    )
    _assert_close(targets[0, 0, 0], (0.5 ** 4) * 4.0)
    _assert_close(targets[0, 3, 0], (0.5 ** 4) * 7.0)
    # A tail transition has no complete four-step horizon in the stored
    # episode, so it must not bootstrap beyond that episode.
    _assert_close(targets[0, 9, 0], 0.0)


def test_q_target_checkpoint_contract_is_fail_loud():
    try:
        from runner.base_runner import (
            _build_sddfg_q_target_checkpoint_contract,
            _validate_sddfg_q_target_checkpoint_contract,
        )
    except ImportError:
        # Minimal local analysis environments may have torch but omit the
        # runner's TensorBoard dependency.  The production preflight imports
        # and executes this branch on the training server.
        return
    args = SimpleNamespace(
        algorithm_name="sddfg",
        q_n_step=24,
        episode_length=200,
        q_terminal_replay_lane=True,
        q_terminal_replay_loss_weight=0.10,
        pre_capture_visible_prey_quorum_greedy_frontier_guard=True,
    )
    contract = _build_sddfg_q_target_checkpoint_contract(args)
    assert contract == {
        "sddfg_q_target_checkpoint_version": 5,
        "sddfg_q_n_step": 24,
        "sddfg_q_n_step_mode": "terminal_gated",
        "sddfg_q_terminal_replay_lane": True,
        "sddfg_q_terminal_replay_loss_weight": 0.10,
        "sddfg_q_frontier_target_alignment": True,
    }
    _validate_sddfg_q_target_checkpoint_contract(contract, args)

    mismatched = dict(contract)
    mismatched["sddfg_q_frontier_target_alignment"] = False
    try:
        _validate_sddfg_q_target_checkpoint_contract(mismatched, args)
        raise AssertionError(
            "frontier target-alignment mismatch did not fail loud"
        )
    except RuntimeError:
        pass
    mismatched = dict(contract)
    mismatched["sddfg_q_n_step"] = 8
    try:
        _validate_sddfg_q_target_checkpoint_contract(mismatched, args)
        raise AssertionError("q_n_step mismatch did not fail loud")
    except RuntimeError:
        pass
    mismatched = dict(contract)
    mismatched["sddfg_q_terminal_replay_loss_weight"] = 1.0
    try:
        _validate_sddfg_q_target_checkpoint_contract(mismatched, args)
        raise AssertionError(
            "terminal replay loss weight mismatch did not fail loud"
        )
    except RuntimeError:
        pass
    mismatched = dict(contract)
    mismatched["sddfg_q_terminal_replay_lane"] = False
    try:
        _validate_sddfg_q_target_checkpoint_contract(mismatched, args)
        raise AssertionError("terminal replay lane mismatch did not fail loud")
    except RuntimeError:
        pass
    mismatched = dict(contract)
    mismatched["sddfg_q_n_step_mode"] = "unconditional"
    try:
        _validate_sddfg_q_target_checkpoint_contract(mismatched, args)
        raise AssertionError("q-target mode mismatch did not fail loud")
    except RuntimeError:
        pass
    try:
        _validate_sddfg_q_target_checkpoint_contract({}, args)
        raise AssertionError("legacy SDDFG checkpoint did not fail loud")
    except RuntimeError:
        pass
    assert _build_sddfg_q_target_checkpoint_contract(None) == {
        "sddfg_q_target_checkpoint_version": 0
    }


def test_replay_sample_attaches_exact_terminal_provenance_without_resampling():
    try:
        from runner.base_runner import (
            _append_sddfg_terminal_win_provenance_to_sample,
        )
    except ImportError:
        return
    marker = np.zeros((2, 24, 1), dtype=np.float32)
    marker[1, 23, 0] = 1.0
    lane_mask = np.asarray([0.0, 1.0], dtype=np.float32)
    policy_buffer = SimpleNamespace(
        last_reward_sample_diagnostics={
            "terminal_win_rewards": marker,
            "terminal_replay_lane_episode_mask": lane_mask,
        }
    )
    buffer = SimpleNamespace(
        policy_buffers={"policy_0": policy_buffer}
    )
    sample = tuple(range(10))
    augmented = _append_sddfg_terminal_win_provenance_to_sample(
        buffer, ["policy_0"], sample
    )
    assert augmented[:10] == sample
    assert len(augmented) == 12
    attached = augmented[10]["policy_0"]
    if not np.array_equal(attached, marker):
        raise AssertionError("sampled terminal provenance changed in transit")
    marker[1, 23, 0] = 0.0
    if attached[1, 23, 0] != 1.0:
        raise AssertionError("terminal provenance was not copied transactionally")
    if not np.array_equal(augmented[11]["policy_0"], lane_mask):
        raise AssertionError("terminal replay lane mask changed in transit")


def test_run158_replay_lane_preserves_uniform_rng_and_closes_sampling_gap():
    # Exact run158 production schedule: replay RNG seed=seed+410000,
    # 32 warmup episodes, four uniform batches of eight every four formal
    # episodes, and terminal episodes at these replay indices.  Uniform replay
    # sampled them only seven times (168 gated transitions) through 80k and
    # did not first sample the 44.8k win until 50.4k.
    terminal_indices = [223, 300, 322, 328, 342]
    baseline_rng = np.random.RandomState(410001)
    lane_rng = np.random.RandomState(410001)
    cursor = 0
    natural_terminal_episodes = 0
    lane_terminal_episodes = 0
    forced_lanes = 0
    first_natural_formal_episode = None
    first_lane_formal_episode = None
    for formal_episode in range(4, 369, 4):
        replay_size = 32 + formal_episode
        available_terminals = [
            index for index in terminal_indices if index < replay_size
        ]
        for update_index in range(4):
            baseline = baseline_rng.choice(
                replay_size, 8, replace=False
            )
            base_with_lane_enabled = lane_rng.choice(
                replay_size, 8, replace=False
            )
            if not np.array_equal(baseline, base_with_lane_enabled):
                raise AssertionError("terminal lane changed uniform replay RNG")
            natural = int(np.isin(
                baseline, available_terminals
            ).sum())
            natural_terminal_episodes += natural
            lane_terminal_episodes += natural
            if natural and first_natural_formal_episode is None:
                first_natural_formal_episode = formal_episode
            if update_index == 0:
                combined, lane_mask, cursor, forced = (
                    terminal_win_replay_lane_indices(
                        base_with_lane_enabled,
                        available_terminals,
                        cursor,
                    )
                )
                if not np.array_equal(
                        combined[:8], base_with_lane_enabled):
                    raise AssertionError("uniform sample stopped being a prefix")
                if forced:
                    forced_lanes += 1
                    lane_terminal_episodes += 1
                    if first_lane_formal_episode is None:
                        first_lane_formal_episode = formal_episode
                    if lane_mask[-1] != 1.0 or lane_mask[:-1].any():
                        raise AssertionError("auxiliary lane mask is not local")
    assert natural_terminal_episodes == 7
    assert first_natural_formal_episode == 220
    assert forced_lanes == 41
    assert lane_terminal_episodes == 48
    assert first_lane_formal_episode == 192
    assert lane_terminal_episodes * 24 == 1152


def test_terminal_replay_lane_trains_only_completion_window():
    valid = torch.ones(3, 30, 1)
    gate = torch.zeros_like(valid, dtype=torch.bool)
    gate[2, :24] = True
    lane_episodes = torch.tensor([0.0, 0.0, 1.0])
    actual, lane_count = _build_terminal_replay_loss_mask(
        valid, gate, lane_episodes
    )
    if not torch.equal(actual[:2], valid[:2]):
        raise AssertionError("terminal lane changed the uniform batch loss")
    if not bool(actual[2, :24].eq(1.0).all().item()):
        raise AssertionError("terminal completion window was not trained")
    if not bool(actual[2, 24:].eq(0.0).all().item()):
        raise AssertionError("auxiliary ordinary transitions entered loss")
    _assert_close(lane_count, 1.0)


def test_run159_auxiliary_loss_weight_preserves_uniform_objective():
    # One run159 forced update contained eight ordinary 200-step episodes and
    # one 24-step completion window.  The repeated full-weight auxiliary MSE
    # drove aggregated post-lane loss from run158's 0.90 median to 3.50.  This
    # fixture gives the completion window a deliberately much larger residual
    # and verifies that the formal 0.10 weight reduces only that population.
    valid = torch.ones(9, 200, 1)
    gate = torch.zeros_like(valid, dtype=torch.bool)
    gate[8, :24] = True
    lane_episodes = torch.tensor([0.0] * 8 + [1.0])
    loss_mask, loss_weight, uniform_mask, auxiliary_mask, lane_count = (
        _build_terminal_replay_loss_population(
            valid,
            gate,
            lane_episodes,
            0.10,
        )
    )
    _assert_close(lane_count, 1.0)
    _assert_close(uniform_mask.sum(), 1600.0)
    _assert_close(auxiliary_mask.sum(), 24.0)
    _assert_close(loss_mask.sum(), 1624.0)
    _assert_close(loss_weight.sum(), 1602.4, tolerance=1e-4)

    squared_error = torch.ones_like(valid)
    squared_error[8, :24] = 400.0
    before = (squared_error * loss_mask).sum() / loss_mask.sum()
    # Production divides by the unchanged 1,600-transition uniform
    # population; the lane is an additive weighted correction.
    after = (squared_error * loss_weight).sum() / uniform_mask.sum()
    _assert_close(after, 1.6, tolerance=1e-6)
    if not float(after) < float(before) * 0.25:
        raise AssertionError("run159 auxiliary MSE was not bounded")
    if not float(after) > 1.0:
        raise AssertionError("terminal auxiliary credit disappeared")
    if not torch.equal(loss_weight[:8], valid[:8]):
        raise AssertionError("uniform replay objective changed")
    if not bool(loss_weight[8, 24:].eq(0.0).all().item()):
        raise AssertionError("auxiliary ordinary transitions entered loss")


def test_launcher_records_and_enables_the_24_step_target():
    launcher_path = os.path.join(
        REPO_ROOT, "scripts", "train_wolfpack_sddfg_intra_episode_dynamic.sh"
    )
    with open(launcher_path, "r", encoding="utf-8") as launcher_file:
        launcher = launcher_file.read()
    assert 'num_env_steps="${1:-${NUM_ENV_STEPS:-60000}}"' in launcher
    assert 'q_n_step="${Q_N_STEP:-24}"' in launcher
    assert '--q_n_step "${q_n_step}"' in launcher
    assert '--q_terminal_replay_lane' in launcher
    assert '--q_terminal_replay_loss_weight' in launcher
    assert 'Q_TERMINAL_REPLAY_LOSS_WEIGHT:-0.10' in launcher
    assert 'mode=terminal_gated' in launcher
    assert 'Q target: contract_version=5' in launcher
    assert 'frontier_next_action=production_pre_capture_exact_quorum_rerank' in launcher
    assert 'debug_sddfg_n_step_targets.py' in launcher


def main():
    tests = (
        test_one_step_is_exact_legacy_expression,
        test_terminal_reward_reaches_the_24_step_window,
        test_run156_terminal_scale_reaches_predecessor_without_a_spike,
        test_run157_no_terminal_trace_stays_exactly_one_step,
        test_terminal_gate_updates_exactly_the_preceding_24_transitions,
        test_terminal_gate_does_not_cross_an_earlier_episode_boundary,
        test_continue_after_success_stops_return_at_the_win_marker,
        test_done_mask_blocks_rewards_and_bootstrap_after_terminal,
        test_full_horizon_bootstraps_from_aligned_future_state,
        test_q_target_checkpoint_contract_is_fail_loud,
        test_replay_sample_attaches_exact_terminal_provenance_without_resampling,
        test_run158_replay_lane_preserves_uniform_rng_and_closes_sampling_gap,
        test_terminal_replay_lane_trains_only_completion_window,
        test_run159_auxiliary_loss_weight_preserves_uniform_objective,
        test_launcher_records_and_enables_the_24_step_target,
    )
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} SDDFG n-step target tests".format(len(tests)))


if __name__ == "__main__":
    main()
