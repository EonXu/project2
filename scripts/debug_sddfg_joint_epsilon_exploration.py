"""Focused production-semantic checks for coordinated epsilon exploration."""

import copy
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.joint_exploration import (  # noqa: E402
    build_batched_wolfpack_frontier_action_mask_from_local_obs,
    build_exact_visible_prey_quorum_frontier_action_mask,
    bound_joint_random_replacements,
    epsilon_random_mask,
    protect_exact_visible_prey_quorum_random_replacements,
    sample_action_indices,
)
try:
    from utils.util import DecayThenFlatSchedule  # noqa: E402
except ModuleNotFoundError as import_error:  # dependency-light local fixture
    if import_error.name != "gym":
        raise

    class DecayThenFlatSchedule:
        def __init__(self, start, finish, time_length, decay="linear"):
            if decay != "linear":
                raise ValueError("fixture fallback supports linear decay only")
            self.start = float(start)
            self.finish = float(finish)
            self.time_length = int(time_length)

        def eval(self, time):
            fraction = min(max(float(time), 0.0) / self.time_length, 1.0)
            return self.start + fraction * (self.finish - self.start)


POLICY_EPSILON_START = 1.0
POLICY_EPSILON_FINISH = 0.05
POLICY_EPSILON_ANNEAL_TIME = 228000
POST_CAPTURE_JOINT_GREEDY_FLOOR = 0.25
POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS = 1
PRE_CAPTURE_VISIBLE_PREY_QUORUM_GUARD = True
PRE_CAPTURE_VISIBLE_PREY_QUORUM_GREEDY_FRONTIER_GUARD = True


def _wolfpack_vector_obs_row(
        num_agents,
        max_food_num,
        slot,
        alive,
        visible_offsets,
        freeze_values):
    """Create only the documented local-vector fields used by the mask."""
    obs_dim = 2 + 4 + 2 * (num_agents - 1) + 7 * max_food_num + 4 + 1
    if not alive:
        return np.full(obs_dim, -1.0, dtype=np.float32)
    row = np.zeros(obs_dim, dtype=np.float32)
    row[2] = 1.0
    teammate_start = 2 + 4
    row[teammate_start:teammate_start + 2 * (num_agents - 1)] = -1.0
    prey_start = teammate_start + 2 * (num_agents - 1)
    for food_id in range(max_food_num):
        block = prey_start + 7 * food_id
        offset = visible_offsets.get((slot, food_id))
        if offset is None:
            row[block:block + 2] = -1.0
        else:
            row[block:block + 2] = np.asarray(offset, dtype=np.float32)
            row[block + 2] = 1.0
        row[block + 6] = float(freeze_values[food_id])
    row[-1] = 1.0
    return row


def _check_q_target_frontier_mask_contract():
    """Replay local observations must reconstruct the rollout-safe mask."""
    num_agents = 6
    max_food_num = 2
    alive = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
    visible_offsets = {
        (0, 0): (8, 0),
        (1, 0): (7, 0),
    }
    pre_capture = np.stack([
        _wolfpack_vector_obs_row(
            num_agents,
            max_food_num,
            slot,
            bool(alive[slot]),
            visible_offsets,
            (0.0, 0.0),
        )
        for slot in range(num_agents)
    ])
    post_capture = np.stack([
        _wolfpack_vector_obs_row(
            num_agents,
            max_food_num,
            slot,
            bool(alive[slot]),
            visible_offsets,
            (0.0, 0.5),
        )
        for slot in range(num_agents)
    ])
    terminal = np.full_like(pre_capture, -1.0)
    obs = np.stack([pre_capture, post_capture, terminal])
    dones = np.stack([~alive, ~alive, np.ones(num_agents, dtype=bool)])
    available = np.ones((3, num_agents, 7), dtype=np.float32)
    available[:, ~alive] = 0.0
    available[:, ~alive, 4] = 1.0
    available[2] = 0.0
    available[2, :, 4] = 1.0
    obs_before = obs.copy()
    available_before = available.copy()

    mask, eligible, constrained, conflicts, pre_rows = (
        build_batched_wolfpack_frontier_action_mask_from_local_obs(
            obs,
            dones,
            available,
            num_agents=num_agents,
            max_food_num=max_food_num,
            sight_radius=8,
            prey_max_step=1,
            capture_quorum=2,
        )
    )
    assert mask.shape == (3, num_agents, 7)
    np.testing.assert_array_equal(pre_rows, [1, 0, 0])
    np.testing.assert_array_equal(eligible[0], [1, 1, 0, 0, 0, 0])
    assert int(constrained[0, 0]) == 1
    assert int(conflicts.sum()) == 0
    # Slot 0 is at L1 distance 8. Moving north (action 0) loses the
    # worst-case frontier, while moving south (action 2) remains robust.
    assert int(mask[0, 0, 0]) == 0
    assert int(mask[0, 0, 2]) == 1
    np.testing.assert_array_equal(mask[1], available[1])
    np.testing.assert_array_equal(mask[2], available[2])
    np.testing.assert_array_equal(obs, obs_before)
    np.testing.assert_array_equal(available, available_before)
    assert np.all(mask <= available)
    assert np.all(mask.sum(axis=2) > 0)


def _check_run169_real_q_target_counterfactual():
    """Reconstruct a real reranked failure state from saved run169 data."""
    fixture_path = os.path.join(
        REPO_ROOT, "scripts", "fixtures", "run169_frontier_failures.json"
    )
    with open(fixture_path, "r", encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    transition = fixture["target_alignment_transitions"][0]
    assert not transition["raw_preserve"]
    assert transition["reranked_preserve"]
    assert transition["executed_preserve"]

    def parse_positions(value):
        parsed = []
        for item in str(value).split(";"):
            if item == "":
                parsed.append(None)
            else:
                x, y = item.split(":")
                parsed.append((int(x), int(y)))
        return parsed

    players = parse_positions(transition["before_positions"])
    foods = parse_positions(transition["before_foods"])
    alive = np.asarray(
        [position is not None for position in players], dtype=bool
    )
    visible_offsets = {}
    for slot, player in enumerate(players):
        if player is None:
            continue
        for food_id, food in enumerate(foods):
            offset = (food[0] - player[0], food[1] - player[1])
            if abs(offset[0]) + abs(offset[1]) <= 8:
                visible_offsets[(slot, food_id)] = offset
    obs = np.stack([
        _wolfpack_vector_obs_row(
            6, 2, slot, bool(alive[slot]), visible_offsets, (0.0, 0.0)
        )
        for slot in range(6)
    ])[None, ...]
    available = np.ones((1, 6, 7), dtype=np.float32)
    for slot, player in enumerate(players):
        if player is None:
            available[0, slot] = 0.0
            available[0, slot, 4] = 1.0
            continue
        x, y = player
        available[0, slot, 0] = float(x > 0)
        available[0, slot, 1] = float(y < 19)
        available[0, slot, 2] = float(x < 19)
        available[0, slot, 3] = float(y > 0)

    mask, eligible, constrained, conflicts, pre_rows = (
        build_batched_wolfpack_frontier_action_mask_from_local_obs(
            obs,
            (~alive)[None, :],
            available,
            num_agents=6,
            max_food_num=2,
            sight_radius=8,
            prey_max_step=1,
            capture_quorum=2,
        )
    )
    observer_slots = [int(x) for x in transition["visible_slots"].split("|")]
    raw_actions = [int(x) for x in transition["raw_actions"].split("|")]
    reranked_actions = [
        int(x) for x in transition["reranked_actions"].split("|")
    ]
    np.testing.assert_array_equal(pre_rows, [1])
    assert int(eligible[0, observer_slots].sum()) == int(
        transition["eligible_observers"]
    )
    assert int(constrained[0, observer_slots].sum()) == int(
        transition["constrained_observers"]
    )
    assert int(conflicts[0, observer_slots].sum()) == int(
        transition["conflict_observers"]
    )
    for slot, raw_action, reranked_action in zip(
            observer_slots, raw_actions, reranked_actions):
        assert int(mask[0, slot, reranked_action]) == 1
        if raw_action != reranked_action:
            assert int(mask[0, slot, raw_action]) == 0
            # A minimal utility counterfactual: the unaligned bootstrap takes
            # the higher-valued raw action; applying the production mask
            # selects the logged robust alternative on the same state.
            utility = np.zeros(7, dtype=np.float64)
            utility[raw_action] = 2.0
            utility[reranked_action] = 1.0
            assert int(np.argmax(utility)) == raw_action
            aligned = np.where(mask[0, slot] > 0, utility, -np.inf)
            assert int(np.argmax(aligned)) == reranked_action


def _check_policy_epsilon_schedule_contract():
    schedule = DecayThenFlatSchedule(
        POLICY_EPSILON_START,
        POLICY_EPSILON_FINISH,
        POLICY_EPSILON_ANNEAL_TIME,
        decay="linear",
    )
    expected = {
        0: 1.0,
        20000: 11.0 / 12.0,
        40000: 5.0 / 6.0,
        60000: 0.75,
        100000: 7.0 / 12.0,
        180000: 0.25,
        228000: 0.05,
        300000: 0.05,
    }
    actual = []
    for step, expected_epsilon in expected.items():
        epsilon = float(schedule.eval(step))
        assert abs(epsilon - expected_epsilon) < 1e-12
        assert POLICY_EPSILON_FINISH <= epsilon <= POLICY_EPSILON_START
        actual.append(epsilon)
    assert all(
        actual[index] >= actual[index + 1]
        for index in range(len(actual) - 1)
    )

    launcher = os.path.join(
        REPO_ROOT,
        "scripts",
        "train_wolfpack_sddfg_intra_episode_dynamic.sh",
    )
    with open(launcher, "r", encoding="utf-8") as launcher_file:
        launcher_text = launcher_file.read()
    assert "--epsilon_start 1.0" in launcher_text
    assert "--epsilon_finish 0.05" in launcher_text
    assert "--epsilon_anneal_time 228000" in launcher_text
    assert "--use_joint_epsilon_exploration" in launcher_text
    assert "--post_capture_joint_greedy_floor 0.25" in launcher_text
    assert (
        '--post_capture_explore_max_random_agents '
        '"${post_capture_explore_max_random_agents}"'
    ) in launcher_text
    assert "--pre_capture_visible_prey_quorum_guard" in launcher_text
    assert (
        "--pre_capture_visible_prey_quorum_greedy_frontier_guard"
        in launcher_text
    )


def _check_checkpoint_schedule_contract():
    try:
        from runner.base_runner import (
            _build_policy_exploration_checkpoint_contract,
            _validate_policy_exploration_checkpoint_contract,
        )
    except ImportError:
        return

    args = SimpleNamespace(
        use_joint_epsilon_exploration=True,
        epsilon_start=POLICY_EPSILON_START,
        epsilon_finish=POLICY_EPSILON_FINISH,
        epsilon_anneal_time=POLICY_EPSILON_ANNEAL_TIME,
        post_capture_joint_greedy_floor=POST_CAPTURE_JOINT_GREEDY_FLOOR,
        post_capture_explore_max_random_agents=(
            POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS
        ),
        pre_capture_visible_prey_quorum_guard=(
            PRE_CAPTURE_VISIBLE_PREY_QUORUM_GUARD
        ),
        pre_capture_visible_prey_quorum_greedy_frontier_guard=(
            PRE_CAPTURE_VISIBLE_PREY_QUORUM_GREEDY_FRONTIER_GUARD
        ),
    )
    checkpoint = {
        "policy_exploration_checkpoint_version": 6,
        "joint_epsilon_exploration_enabled": True,
        "policy_epsilon_start": POLICY_EPSILON_START,
        "policy_epsilon_finish": POLICY_EPSILON_FINISH,
        "policy_epsilon_anneal_time": POLICY_EPSILON_ANNEAL_TIME,
        "post_capture_joint_greedy_floor": (
            POST_CAPTURE_JOINT_GREEDY_FLOOR
        ),
        "post_capture_explore_max_random_agents": (
            POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS
        ),
        "pre_capture_visible_prey_quorum_guard": (
            PRE_CAPTURE_VISIBLE_PREY_QUORUM_GUARD
        ),
        "pre_capture_visible_prey_quorum_greedy_frontier_guard": (
            PRE_CAPTURE_VISIBLE_PREY_QUORUM_GREEDY_FRONTIER_GUARD
        ),
        "policy_rng_states": {"policy_0": ("fixture",)},
    }
    built_checkpoint = _build_policy_exploration_checkpoint_contract(
        args,
        {
            "policy_0": SimpleNamespace(
                rng=np.random.RandomState(7),
            ),
        },
        ["policy_0"],
    )
    assert built_checkpoint["policy_exploration_checkpoint_version"] == 6
    assert built_checkpoint["joint_epsilon_exploration_enabled"] is True
    assert built_checkpoint["policy_epsilon_start"] == POLICY_EPSILON_START
    assert built_checkpoint["policy_epsilon_finish"] == POLICY_EPSILON_FINISH
    assert (
        built_checkpoint["policy_epsilon_anneal_time"]
        == POLICY_EPSILON_ANNEAL_TIME
    )
    assert set(built_checkpoint["policy_rng_states"]) == {"policy_0"}
    assert (
        built_checkpoint["post_capture_joint_greedy_floor"]
        == POST_CAPTURE_JOINT_GREEDY_FLOOR
    )
    assert (
        built_checkpoint["post_capture_explore_max_random_agents"]
        == POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS
    )
    assert built_checkpoint[
        "pre_capture_visible_prey_quorum_guard"
    ] is True
    assert built_checkpoint[
        "pre_capture_visible_prey_quorum_greedy_frontier_guard"
    ] is True

    assert _build_policy_exploration_checkpoint_contract(
        None,
        {},
        [],
    ) == {"policy_exploration_checkpoint_version": 0}
    try:
        _build_policy_exploration_checkpoint_contract(
            None,
            {"policy_0": SimpleNamespace(rng=np.random.RandomState(8))},
            ["policy_0"],
        )
        raise AssertionError("policy checkpoint without runner args did not fail loud")
    except RuntimeError:
        pass

    assert _validate_policy_exploration_checkpoint_contract(
        checkpoint,
        args,
    ) == checkpoint["policy_rng_states"]

    for changed_key, changed_value in (
            ("policy_epsilon_start", 0.9),
            ("policy_epsilon_finish", 0.1),
            ("policy_epsilon_anneal_time", 228001),
            ("post_capture_joint_greedy_floor", 0.3),
            ("post_capture_explore_max_random_agents", 2),
            ("pre_capture_visible_prey_quorum_guard", False),
            ("pre_capture_visible_prey_quorum_greedy_frontier_guard", False)):
        malformed = dict(checkpoint)
        malformed[changed_key] = changed_value
        try:
            _validate_policy_exploration_checkpoint_contract(malformed, args)
            raise AssertionError("schedule mismatch did not fail loud")
        except RuntimeError:
            pass

    legacy_joint_checkpoint = dict(checkpoint)
    legacy_joint_checkpoint["policy_exploration_checkpoint_version"] = 2
    legacy_joint_checkpoint.pop("post_capture_joint_greedy_floor")
    try:
        _validate_policy_exploration_checkpoint_contract(
            legacy_joint_checkpoint,
            args,
        )
        raise AssertionError(
            "version-2 checkpoint accepted a nonzero post-capture floor"
        )
    except RuntimeError:
        pass

    legacy_floor_checkpoint = dict(checkpoint)
    legacy_floor_checkpoint["policy_exploration_checkpoint_version"] = 3
    legacy_floor_checkpoint.pop("post_capture_explore_max_random_agents")
    try:
        _validate_policy_exploration_checkpoint_contract(
            legacy_floor_checkpoint,
            args,
        )
        raise AssertionError(
            "version-3 checkpoint accepted bounded post-capture exploration"
        )
    except RuntimeError:
        pass

    old_checkpoint = dict(checkpoint)
    old_checkpoint["policy_exploration_checkpoint_version"] = 1
    try:
        _validate_policy_exploration_checkpoint_contract(old_checkpoint, args)
        raise AssertionError("old schedule-less checkpoint did not fail loud")
    except RuntimeError:
        pass

    legacy_bounded_checkpoint = dict(checkpoint)
    legacy_bounded_checkpoint["policy_exploration_checkpoint_version"] = 4
    legacy_bounded_checkpoint.pop(
        "pre_capture_visible_prey_quorum_guard"
    )
    try:
        _validate_policy_exploration_checkpoint_contract(
            legacy_bounded_checkpoint,
            args,
        )
        raise AssertionError(
            "version-4 checkpoint accepted the pre-capture quorum guard"
        )
    except RuntimeError:
        pass


def _check_post_capture_stage_contract():
    try:
        from runner.wolfpack_runner import _is_post_capture_pursuit_active
    except ImportError:
        return

    assert not _is_post_capture_pursuit_active([True, True])
    assert _is_post_capture_pursuit_active([False, True])
    assert _is_post_capture_pursuit_active([True, False])
    assert not _is_post_capture_pursuit_active([False, False])
    try:
        _is_post_capture_pursuit_active([])
        raise AssertionError("empty prey status did not fail loud")
    except RuntimeError:
        pass


def _check_mask_contract():
    n_agents = 6
    epsilon = 0.886
    trials = 200000

    legacy_rng = np.random.RandomState(11)
    legacy = np.stack([
        epsilon_random_mask(
            legacy_rng, epsilon, n_agents, joint_decision=False
        )
        for _ in range(trials)
    ])
    joint_rng = np.random.RandomState(11)
    joint = np.stack([
        epsilon_random_mask(
            joint_rng, epsilon, n_agents, joint_decision=True
        )
        for _ in range(trials)
    ])

    assert abs(float(legacy.mean()) - epsilon) < 0.005
    assert abs(float(joint.mean()) - epsilon) < 0.005
    assert np.any((legacy.sum(axis=1) > 0) & (legacy.sum(axis=1) < n_agents))
    assert not np.any((joint.sum(axis=1) > 0) & (joint.sum(axis=1) < n_agents))

    legacy_greedy = float(np.mean(legacy.sum(axis=1) == 0))
    joint_greedy = float(np.mean(joint.sum(axis=1) == 0))
    assert abs(legacy_greedy - (1.0 - epsilon) ** n_agents) < 5e-5
    assert abs(joint_greedy - (1.0 - epsilon)) < 0.005

    assert not epsilon_random_mask(
        np.random.RandomState(3), 0.0, n_agents, True
    ).any()
    assert epsilon_random_mask(
        np.random.RandomState(3), 1.0, n_agents, True
    ).all()
    try:
        epsilon_random_mask(np.random.RandomState(3), float("nan"), n_agents, True)
        raise AssertionError("nonfinite epsilon did not fail loud")
    except ValueError:
        pass
    try:
        epsilon_random_mask(np.random.RandomState(3), 0.5, 0, True)
        raise AssertionError("empty joint decision did not fail loud")
    except ValueError:
        pass
    return legacy_greedy, joint_greedy


def _check_bounded_post_capture_mask_contract():
    alive = np.asarray([1, 1, 0, 1, 1, 0], dtype=bool)
    all_random = np.ones(6, dtype=np.int64)
    all_greedy = np.zeros(6, dtype=np.int64)

    # Disabled and non-explore paths are exact no-RNG legacy paths.
    for joint_mask, limit in ((all_random, 0), (all_greedy, 1)):
        rng = np.random.RandomState(61)
        expected = np.random.RandomState(61)
        actual = bound_joint_random_replacements(
            rng, joint_mask, alive, limit
        )
        np.testing.assert_array_equal(
            actual,
            joint_mask if limit == 0 else all_greedy,
        )
        np.testing.assert_array_equal(
            rng.randint(0, 1000000, size=32),
            expected.randint(0, 1000000, size=32),
        )

    rng = np.random.RandomState(67)
    counts = np.zeros(6, dtype=np.int64)
    for _ in range(12000):
        bounded = bound_joint_random_replacements(
            rng, all_random, alive, 1
        )
        assert int(bounded.sum()) == 1
        assert not np.any(bounded[~alive])
        counts += bounded
    assert np.all(counts[alive] > 2800)
    assert np.all(counts[alive] < 3200)
    assert not np.any(counts[~alive])

    first_rng = np.random.RandomState(71)
    second_rng = np.random.RandomState(71)
    np.testing.assert_array_equal(
        bound_joint_random_replacements(
            first_rng, all_random, alive, 1
        ),
        bound_joint_random_replacements(
            second_rng, all_random, alive, 1
        ),
    )
    np.testing.assert_array_equal(
        first_rng.randint(0, 1000000, size=32),
        second_rng.randint(0, 1000000, size=32),
    )

    try:
        bound_joint_random_replacements(
            np.random.RandomState(1), [1, 0], [1, 1], 1
        )
        raise AssertionError("non-joint mask did not fail loud")
    except ValueError:
        pass

    # Production regression from run161 episode 7, first post-capture action:
    # the legacy branch replaced three alive slots simultaneously and the
    # remaining-prey nearest distance moved 1 -> 3.  Reusing those already
    # legal actions under the bounded mask changes at most one slot while
    # leaving dead-slot stays untouched.
    run161_greedy = np.asarray([2, 1, 3, 3, 4, 4], dtype=np.int64)
    run161_legacy_final = np.asarray([4, 5, 6, 3, 4, 4], dtype=np.int64)
    run161_alive = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
    assert int(np.count_nonzero(
        run161_greedy != run161_legacy_final
    )) == 3
    run161_mask = bound_joint_random_replacements(
        np.random.RandomState(0),
        np.ones(6, dtype=np.int64),
        run161_alive,
        1,
    )
    run161_bounded_final = run161_greedy.copy()
    run161_bounded_final[run161_mask.astype(bool)] = (
        run161_legacy_final[run161_mask.astype(bool)]
    )
    assert int(np.count_nonzero(
        run161_greedy != run161_bounded_final
    )) == 1
    np.testing.assert_array_equal(
        run161_bounded_final[~run161_alive],
        run161_greedy[~run161_alive],
    )


def _check_pre_capture_visible_prey_quorum_guard_contract():
    all_random = np.ones(6, dtype=np.int64)
    alive = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)

    # run167 episode 319 equivalent: the two remaining-prey observers are
    # protected, while both non-observers retain the same joint-random branch.
    visible = np.asarray([
        [1, 0],
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ], dtype=np.int64)
    effective, protected = (
        protect_exact_visible_prey_quorum_random_replacements(
            all_random,
            alive,
            visible,
            capture_quorum=2,
        )
    )
    np.testing.assert_array_equal(protected, [1, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(effective, [0, 1, 0, 1, 0, 0])

    # A three-observer prey has redundancy and is intentionally not guarded.
    redundant = visible.copy()
    redundant[1, 0] = 1
    redundant[:, 1] = 0
    effective, protected = (
        protect_exact_visible_prey_quorum_random_replacements(
            all_random,
            alive,
            redundant,
            capture_quorum=2,
        )
    )
    np.testing.assert_array_equal(protected, np.zeros(6, dtype=np.int64))
    np.testing.assert_array_equal(effective, alive.astype(np.int64))

    # No-explore is an exact no-op and the helper cannot consume RNG because
    # its contract accepts no RNG object.
    effective, protected = (
        protect_exact_visible_prey_quorum_random_replacements(
            np.zeros(6, dtype=np.int64),
            alive,
            visible,
            capture_quorum=2,
        )
    )
    np.testing.assert_array_equal(effective, np.zeros(6, dtype=np.int64))
    np.testing.assert_array_equal(protected, np.zeros(6, dtype=np.int64))

    invalid = visible.copy()
    invalid[4, 0] = 1
    try:
        protect_exact_visible_prey_quorum_random_replacements(
            all_random,
            alive,
            invalid,
            capture_quorum=2,
        )
        raise AssertionError("dead observer did not fail loud")
    except ValueError:
        pass


def _check_run167_real_quorum_loss_counterfactual():
    action_delta = {
        0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),
        4: (0, 0), 5: (0, 0), 6: (0, 0),
    }

    def visible_count(positions, actions, prey_next, alive):
        count = 0
        for slot, position in enumerate(positions):
            if not alive[slot]:
                continue
            dx, dy = action_delta[int(actions[slot])]
            moved = (position[0] + dx, position[1] + dy)
            if abs(moved[0] - prey_next[0]) + abs(
                    moved[1] - prey_next[1]) <= 8:
                count += 1
        return count

    fixtures = [
        {
            "name": "run167_far_episode_319_terminal_T2_loss",
            "alive": [1, 1, 1, 1, 0, 0],
            "visible": [
                [1, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0],
            ],
            "positions": [(3, 15), (3, 8), (7, 12), (17, 7), (0, 0), (0, 0)],
            "prey_next": (9, 17),
            "greedy": [4, 0, 0, 1, 4, 4],
            "selected": [0, 5, 1, 2, 4, 4],
            "expected_protected": [0, 2],
            "before_count": 1,
            "after_count": 2,
        },
        {
            "name": "run167_strict_episode_140_recoverable_T2_loss",
            "alive": [1, 1, 1, 1, 0, 0],
            "visible": [
                [1, 0], [0, 1], [1, 0], [0, 1], [0, 0], [0, 0],
            ],
            "positions": [(3, 11), (13, 3), (2, 10), (11, 16), (0, 0), (0, 0)],
            "prey_next": (11, 10),
            "greedy": [3, 1, 3, 3, 4, 4],
            "selected": [0, 3, 6, 2, 4, 4],
            "expected_protected": [0, 1, 2, 3],
            "before_count": 1,
            "after_count": 3,
        },
    ]
    for fixture in fixtures:
        alive = np.asarray(fixture["alive"], dtype=bool)
        random_mask = np.ones(6, dtype=np.int64)
        effective, protected = (
            protect_exact_visible_prey_quorum_random_replacements(
                random_mask,
                alive,
                np.asarray(fixture["visible"], dtype=np.int64),
                capture_quorum=2,
            )
        )
        final_actions = np.asarray(fixture["greedy"], dtype=np.int64)
        selected = np.asarray(fixture["selected"], dtype=np.int64)
        final_actions[effective.astype(bool)] = selected[
            effective.astype(bool)
        ]
        assert np.flatnonzero(protected).tolist() == fixture[
            "expected_protected"
        ]
        assert visible_count(
            fixture["positions"],
            fixture["selected"],
            fixture["prey_next"],
            alive,
        ) == fixture["before_count"], fixture["name"]
        assert visible_count(
            fixture["positions"],
            final_actions,
            fixture["prey_next"],
            alive,
        ) == fixture["after_count"], fixture["name"]


def _check_run168_real_greedy_frontier_counterfactual():
    fixture_path = os.path.join(
        REPO_ROOT,
        "scripts",
        "fixtures",
        "run168_greedy_frontier_failures.json",
    )
    with open(fixture_path, "r", encoding="utf-8") as fixture_file:
        fixtures = json.load(fixture_file)
    assert len(fixtures) >= 2

    for fixture in fixtures:
        alive = np.asarray(fixture["alive"], dtype=bool)
        visible = np.asarray(fixture["visible"], dtype=np.int64)
        offsets = np.asarray(fixture["offsets"], dtype=np.float64)
        available = np.ones((6, 7), dtype=np.float32)
        available[~alive] = 0.0
        available[~alive, 4] = 1.0
        rng = np.random.RandomState(1000 + int(fixture["episode"]))
        rng_before = copy.deepcopy(rng.get_state())
        mask, eligible, constrained, conflicts = (
            build_exact_visible_prey_quorum_frontier_action_mask(
                alive_mask=alive,
                visible_prey_mask=visible,
                visible_prey_offsets=offsets,
                available_actions=available,
                sight_radius=8,
                prey_max_step=1,
                capture_quorum=2,
            )
        )
        leaving_slot = int(fixture["leaving_slot"])
        safe_actions = np.flatnonzero(mask[leaving_slot]).tolist()
        assert safe_actions == fixture["expected_safe_actions"], fixture["name"]
        assert int(eligible[leaving_slot]) == 1, fixture["name"]
        assert int(constrained[leaving_slot]) == 1, fixture["name"]
        assert int(conflicts[leaving_slot]) == 0, fixture["name"]
        assert int(fixture["raw_greedy"][leaving_slot]) not in safe_actions
        np.testing.assert_array_equal(mask[~alive, 4], 1)
        np.testing.assert_array_equal(mask[~alive].sum(axis=1), 1)
        replay = np.random.RandomState(0)
        replay.set_state(rng_before)
        np.testing.assert_array_equal(
            rng.randint(0, 1000000, size=32),
            replay.randint(0, 1000000, size=32),
        )

    # If one observer is the exact-quorum member for prey on opposing
    # frontiers, the helper exposes the unavoidable conflict instead of
    # silently claiming both can be preserved.
    alive = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    visible = np.asarray([
        [1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0],
    ], dtype=np.int64)
    offsets = np.full((6, 2, 2), np.nan, dtype=np.float64)
    offsets[0, 0] = [0, 8]
    offsets[1, 0] = [0, 4]
    offsets[0, 1] = [0, -8]
    offsets[2, 1] = [0, -4]
    available = np.ones((6, 7), dtype=np.float32)
    available[~alive] = 0.0
    available[~alive, 4] = 1.0
    mask, _, constrained, conflicts = (
        build_exact_visible_prey_quorum_frontier_action_mask(
            alive, visible, offsets, available, 8, 1, 2
        )
    )
    assert int(constrained[0]) == 1
    assert int(conflicts[0]) == 1
    assert set(np.flatnonzero(mask[0]).tolist()) == {1, 3}


def _check_available_actions_and_rng_roundtrip():
    available = np.asarray([
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0],
    ], dtype=np.float32)

    rng = np.random.RandomState(23)
    before = copy.deepcopy(rng.get_state())
    first_mask = epsilon_random_mask(rng, 0.73, 6, True)
    first_actions = sample_action_indices(rng, available, 6, 7)
    for agent, action in enumerate(first_actions):
        assert available[agent, int(action)] > 0.5

    replay = np.random.RandomState(999)
    replay.set_state(copy.deepcopy(before))
    second_mask = epsilon_random_mask(replay, 0.73, 6, True)
    second_actions = sample_action_indices(replay, available, 6, 7)
    np.testing.assert_array_equal(first_mask, second_mask)
    np.testing.assert_array_equal(first_actions, second_actions)

    samples = np.stack([
        sample_action_indices(rng, available, 6, 7)
        for _ in range(200)
    ])
    assert len(set(tuple(row) for row in samples)) > 20


def _check_production_policy_path():
    try:
        import torch
        from algorithms.sddfg.algorithm.rSDDFGPolicy import (
            R_SDDFGPolicy,
            _finite_message_action_margins,
        )
    except ImportError:
        return

    message_margins = _finite_message_action_margins(torch.as_tensor([[
        [1.0, 3.0, 2.0],
        [5.0, -float("inf"), -float("inf")],
        [float("nan"), 0.0, 1.0],
    ]]))
    np.testing.assert_allclose(
        message_margins.detach().cpu().numpy()[0, [0, 2]],
        np.asarray([1.0, 1.0]),
    )
    assert np.isnan(float(message_margins[0, 1]))

    # Exercise the production max-sum rerank itself (not only the numpy mask
    # builder): a run168 episode-30 equivalent frontier mask changes slot 1
    # from the raw utility tie-break action 0 to its sole robust action 1.
    greedy_policy = R_SDDFGPolicy.__new__(R_SDDFGPolicy)
    greedy_policy.lamda = 0.5
    greedy_policy.num_factor = 1
    greedy_policy.n_agents = 6
    greedy_policy.highest_orders = 3
    greedy_policy.act_dim = 7
    greedy_policy.device = torch.device("cpu")
    greedy_policy.tpdv = {
        "dtype": torch.float32,
        "device": greedy_policy.device,
    }
    greedy_policy.args = SimpleNamespace(
        msg_iterations=0,
        msg_anytime=False,
        msg_normalized=True,
    )
    greedy_available = torch.ones((6, 7), dtype=torch.float32)
    greedy_available[4:] = 0.0
    greedy_available[4:, 4] = 1.0
    greedy_mask = np.ones((6, 7), dtype=np.int64)
    greedy_mask[1] = 0
    greedy_mask[1, 1] = 1
    greedy_mask[4:] = 0
    greedy_mask[4:, 4] = 1
    reranked_actions, reranked_q, _, reranked_factor_q = greedy_policy.greedy(
        adj=torch.zeros((1, 6, 1), dtype=torch.long),
        q_batch=[
            torch.empty((0, 7)),
            torch.empty((0, 49)),
            torch.empty((0, 343)),
        ],
        idx_node_order=[[], [], []],
        available_actions=greedy_available,
        num_edges=0,
        batch_size=1,
        greedy_action_mask=greedy_mask,
        greedy_value_diagnostic_slot_mask=np.asarray(
            [0, 1, 0, 0, 0, 0], dtype=np.int64
        ),
    )
    assert int(reranked_actions.reshape(-1)[1]) == 1
    raw_tie_action = int(greedy_policy.last_greedy_frontier_diagnostic[
        "unconstrained_actions"
    ][1])
    # With no factors every utility is tied.  PyTorch 1.3.1 and newer
    # releases do not promise the same argmax index for that synthetic tie;
    # the production contract is the constrained result, not the legacy tie
    # index selected before applying the mask.
    assert 0 <= raw_tie_action < 7
    assert greedy_policy.last_greedy_frontier_diagnostic[
        "constrained_actions"
    ][1] == 1
    assert torch.isfinite(reranked_q).all()
    assert torch.isfinite(reranked_factor_q).all()
    value_diagnostic = greedy_policy.last_greedy_value_diagnostic
    assert value_diagnostic["schema_version"] == 1
    assert value_diagnostic["sampled"] == 1
    assert value_diagnostic["slot_ids"] == [1]
    assert value_diagnostic["action_dim"] == 7
    assert value_diagnostic["factor_count"] == 1
    assert np.asarray(
        value_diagnostic["message_action_utilities"]
    ).shape == (1, 7)
    assert np.asarray(
        value_diagnostic["coordinate_joint_q_values"]
    ).shape == (1, 7)
    assert np.asarray(
        value_diagnostic["coordinate_factor_q_values"]
    ).shape == (1, 7, 1)
    assert np.asarray(
        value_diagnostic["frontier_action_mask"]
    )[0].tolist() == greedy_mask[1].tolist()

    # The same production greedy implementation must keep learner batches on
    # device and must not materialize rollout-only diagnostics.  The size is
    # the exact 9600-action server failure shape: 1600 decisions x 6 slots.
    production_learner_batch_size = 1600
    learner_greedy_actions, learner_greedy_q, _, _ = greedy_policy.greedy(
        adj=torch.zeros(
            (production_learner_batch_size, 6, 1), dtype=torch.long
        ),
        q_batch=[
            torch.empty((0, 7)),
            torch.empty((0, 49)),
            torch.empty((0, 343)),
        ],
        idx_node_order=[[], [], []],
        available_actions=torch.ones(
            (production_learner_batch_size, 6, 7), dtype=torch.float32
        ),
        num_edges=0,
        batch_size=production_learner_batch_size,
        greedy_action_mask=None,
    )
    assert learner_greedy_actions.shape == (
        production_learner_batch_size, 6, 1
    )
    assert learner_greedy_q.shape == (production_learner_batch_size,)
    assert greedy_policy.last_greedy_frontier_diagnostic is None
    assert greedy_policy.last_greedy_value_diagnostic is None

    learner_frontier_mask = np.ones(
        (production_learner_batch_size, 6, 7), dtype=np.int64
    )
    learner_frontier_mask[:, 1] = 0
    learner_frontier_mask[:, 1, 2] = 1
    masked_learner_actions, masked_learner_q, _, _ = greedy_policy.greedy(
        adj=torch.zeros(
            (production_learner_batch_size, 6, 1), dtype=torch.long
        ),
        q_batch=[
            torch.empty((0, 7)),
            torch.empty((0, 49)),
            torch.empty((0, 343)),
        ],
        idx_node_order=[[], [], []],
        available_actions=torch.ones(
            (production_learner_batch_size, 6, 7), dtype=torch.float32
        ),
        num_edges=0,
        batch_size=production_learner_batch_size,
        greedy_action_mask=learner_frontier_mask,
        record_greedy_frontier_actions=False,
    )
    assert masked_learner_actions.shape == (
        production_learner_batch_size, 6, 1
    )
    assert bool(torch.all(masked_learner_actions[:, 1, 0] == 2).item())
    assert torch.isfinite(masked_learner_q).all()
    assert greedy_policy.last_greedy_frontier_diagnostic is None
    assert greedy_policy.last_greedy_frontier_batch_diagnostic == {
        "state_count": production_learner_batch_size,
        "reranked_state_count": production_learner_batch_size,
        "reranked_slot_count": production_learner_batch_size,
    }

    class _Schedule:
        def __init__(self, epsilon):
            self.epsilon = float(epsilon)

        def eval(self, _):
            return self.epsilon

    available = np.asarray([
        [1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
    ], dtype=np.float32)
    greedy = np.zeros(6, dtype=np.int64)

    def build_policy(
            joint,
            epsilon,
            seed,
            pre_capture_guard=False,
            frontier_guard=False):
        policy = R_SDDFGPolicy.__new__(R_SDDFGPolicy)
        policy.tpdv = {"dtype": torch.float32, "device": torch.device("cpu")}
        policy.n_agents = 6
        policy.act_dim = 7
        policy.multidiscrete = False
        policy.use_joint_epsilon_exploration = bool(joint)
        policy.pre_capture_visible_prey_quorum_guard = bool(
            pre_capture_guard
        )
        policy.pre_capture_visible_prey_quorum_greedy_frontier_guard = bool(
            frontier_guard
        )
        policy.exploration = _Schedule(epsilon)
        policy.rng = np.random.RandomState(seed)
        policy.last_greedy_frontier_diagnostic = None
        policy.last_greedy_frontier_batch_diagnostic = None
        policy.get_rnn_batch = lambda *args, **kwargs: (None, None, None, 0)
        def fixture_greedy(*args, **kwargs):
            fixture_batch_size = int(args[5])
            raw = np.repeat(
                greedy.reshape(1, 6), fixture_batch_size, axis=0
            )
            fixture_available = args[3] if len(args) > 3 else None
            if fixture_available is not None:
                if torch.is_tensor(fixture_available):
                    fixture_available = (
                        fixture_available.detach().cpu().numpy()
                    )
                fixture_available = np.asarray(
                    fixture_available
                ).reshape(fixture_batch_size, 6, 7)
                for batch_index in range(fixture_batch_size):
                    for slot in range(6):
                        legal = np.flatnonzero(
                            fixture_available[batch_index, slot] > 0.5
                        )
                        if legal.size == 1:
                            raw[batch_index, slot] = int(legal[0])
            constrained = raw.copy()
            action_mask = kwargs.get("greedy_action_mask")
            if action_mask is not None:
                action_mask = np.asarray(action_mask, dtype=bool).reshape(
                    fixture_batch_size, 6, 7
                )
                for batch_index in range(fixture_batch_size):
                    for slot in range(6):
                        if not action_mask[
                                batch_index, slot,
                                constrained[batch_index, slot]]:
                            constrained[batch_index, slot] = int(
                                np.flatnonzero(
                                    action_mask[batch_index, slot]
                                )[0]
                            )
                changed = constrained != raw
                policy.last_greedy_frontier_batch_diagnostic = {
                    "state_count": fixture_batch_size,
                    "reranked_state_count": int(
                        changed.any(axis=1).sum()
                    ),
                    "reranked_slot_count": int(changed.sum()),
                }
                if kwargs.get("record_greedy_frontier_actions", True):
                    assert fixture_batch_size == 1
                    policy.last_greedy_frontier_diagnostic = {
                        "unconstrained_actions": raw.reshape(-1).tolist(),
                        "constrained_actions": constrained.reshape(-1).tolist(),
                    }
                else:
                    policy.last_greedy_frontier_diagnostic = None
            else:
                policy.last_greedy_frontier_diagnostic = None
                policy.last_greedy_frontier_batch_diagnostic = None
            return (
                torch.as_tensor(
                    constrained.reshape(fixture_batch_size, 6, 1),
                    dtype=torch.long,
                ),
                torch.zeros(fixture_batch_size),
                None,
                None,
            )
        policy.greedy = fixture_greedy
        return policy

    obs = np.zeros((1, 6, 1), dtype=np.float32)
    states = torch.zeros((1, 6, 1), dtype=torch.float32)

    greedy_actions = build_policy(True, 0.0, 7).get_actions(
        obs, states, torch.as_tensor(available), t_env=60000, explore=True
    )[0]
    np.testing.assert_array_equal(np.argmax(greedy_actions, axis=-1), greedy)

    # Regression for the run169-at-Q-training failure: enabling the rollout
    # feature on the policy must not make learner/Q-target calls require
    # per-transition visibility geometry.  Only the runner's explicit apply
    # flag may activate the frontier mask.
    learner_policy = build_policy(
        True, 0.0, 11, frontier_guard=True
    )
    learner_actions = learner_policy.get_actions(
        obs,
        states,
        torch.as_tensor(available),
        t_env=None,
        explore=False,
        dones=torch.zeros((6, 1), dtype=torch.bool),
    )[0]
    np.testing.assert_array_equal(
        np.argmax(learner_actions, axis=-1), greedy
    )

    # Reproduce the server failure shape exactly: (step-1)*batch = 1600
    # independent next-state joint decisions, each with six policy slots.
    learner_batch_size = 1600
    learner_batch_policy = build_policy(
        True, 0.0, 12, frontier_guard=True
    )
    learner_batch_actions = learner_batch_policy.get_actions(
        np.zeros((learner_batch_size, 6, 1), dtype=np.float32),
        torch.zeros((learner_batch_size, 6, 1), dtype=torch.float32),
        torch.as_tensor(np.repeat(
            available.reshape(1, 6, 7), learner_batch_size, axis=0
        )),
        t_env=None,
        explore=False,
        dones=torch.zeros(
            (learner_batch_size, 6, 1), dtype=torch.bool
        ),
    )[0]
    assert learner_batch_actions.shape == (learner_batch_size, 6, 7)
    np.testing.assert_array_equal(
        np.argmax(learner_batch_actions, axis=-1),
        np.repeat(greedy.reshape(1, 6), learner_batch_size, axis=0),
    )

    # The replay-target path applies a [batch, slots, actions] mask without
    # asking for rollout geometry or flattening the 9600 slot actions into a
    # six-slot diagnostic. This directly guards the two server failures that
    # previously appeared when rollout-only logic leaked into Q training.
    target_available = np.repeat(
        available.reshape(1, 6, 7), learner_batch_size, axis=0
    )
    # A frontier mask is a restriction of environment legality, never a new
    # action-availability source.  Starting from all ones would incorrectly
    # re-enable the sparse unavailable actions in this fixture.
    target_mask = target_available.astype(np.int64, copy=True)
    target_mask[:, 1] = 0
    target_mask[:, 1, 2] = 1
    assert not np.any(
        (target_mask > 0) & ~(target_available > 0.5)
    )
    learner_target_policy = build_policy(
        True, 0.0, 14, frontier_guard=True
    )
    learner_target_actions = learner_target_policy.get_actions(
        np.zeros((learner_batch_size, 6, 1), dtype=np.float32),
        torch.zeros((learner_batch_size, 6, 1), dtype=torch.float32),
        torch.as_tensor(target_available),
        t_env=None,
        explore=False,
        dones=torch.zeros(
            (learner_batch_size, 6, 1), dtype=torch.bool
        ),
        precomputed_greedy_frontier_action_mask=target_mask,
    )[0]
    expected_target = np.repeat(
        greedy.reshape(1, 6), learner_batch_size, axis=0
    )
    expected_target[:, 1] = 2
    assert learner_target_actions.shape == (learner_batch_size, 6, 7)
    np.testing.assert_array_equal(
        np.argmax(learner_target_actions, axis=-1), expected_target
    )
    assert learner_target_policy.last_greedy_frontier_diagnostic is None
    assert learner_target_policy.last_greedy_frontier_batch_diagnostic == {
        "state_count": learner_batch_size,
        "reranked_state_count": learner_batch_size,
        "reranked_slot_count": learner_batch_size,
    }

    frontier_visible = np.asarray([
        [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0],
    ], dtype=np.int64)
    frontier_offsets = np.full((6, 2, 2), np.nan, dtype=np.float32)
    frontier_offsets[1, 1] = [0, 8]
    frontier_offsets[2, 1] = [0, 4]
    frontier_available = np.ones((6, 7), dtype=np.float32)
    frontier_dones = np.asarray([0, 0, 0, 0, 1, 1], dtype=bool)
    frontier_available[frontier_dones] = 0.0
    frontier_available[frontier_dones, 4] = 1.0
    frontier_policy = build_policy(
        True, 0.0, 13, frontier_guard=True
    )
    frontier_actions = frontier_policy.get_actions(
        obs,
        states,
        torch.as_tensor(frontier_available),
        t_env=6400,
        explore=True,
        dones=torch.as_tensor(frontier_dones.reshape(6, 1)),
        pre_capture_visible_prey_mask=frontier_visible,
        pre_capture_visible_prey_offsets=frontier_offsets,
        pre_capture_visibility_radius=8,
        pre_capture_prey_max_step=1,
        apply_pre_capture_visible_prey_quorum_greedy_frontier_guard=True,
    )[0]
    assert int(np.argmax(frontier_actions, axis=-1)[1]) == 1
    frontier_diag = frontier_policy.last_action_exploration_diagnostic
    assert frontier_diag[
        "pre_capture_visible_prey_quorum_greedy_frontier_guard_enabled"
    ] == 1
    assert frontier_diag[
        "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots"
    ] == [1]

    missing_geometry_policy = build_policy(
        True, 0.0, 17, frontier_guard=True
    )
    try:
        missing_geometry_policy.get_actions(
            obs,
            states,
            torch.as_tensor(available),
            t_env=6400,
            explore=False,
            apply_pre_capture_visible_prey_quorum_greedy_frontier_guard=True,
        )
        raise AssertionError(
            "explicit frontier apply without geometry did not fail loud"
        )
    except RuntimeError:
        pass

    seed = 19
    expected_rng = np.random.RandomState(seed)
    expected_mask = epsilon_random_mask(expected_rng, 1.0, 6, True)
    expected_random = sample_action_indices(expected_rng, available, 6, 7)
    expected = (1 - expected_mask) * greedy + expected_mask * expected_random
    explored = build_policy(True, 1.0, seed).get_actions(
        obs, states, torch.as_tensor(available), t_env=60000, explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
    )[0]
    np.testing.assert_array_equal(np.argmax(explored, axis=-1), expected)
    for agent, action in enumerate(expected):
        assert available[agent, int(action)] > 0.5
    production_diag = build_policy(True, 1.0, seed)
    production_diag.get_actions(
        obs, states, torch.as_tensor(available), t_env=60000, explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
    )
    diag = production_diag.last_action_exploration_diagnostic
    assert diag["schema_version"] == 4
    assert diag["joint_explore"] == 1
    assert diag["random_replacement_slot_count"] == 6
    assert diag["invalid_available_action_count"] == 0
    assert diag["dead_slot_non_stay_violation_count"] == 0
    assert diag["non_explore_greedy_mismatch"] == 0
    assert sum(diag["action_histogram"]) == 6

    # run167 production failure equivalent: the branch/action RNG draws are
    # unchanged, but exact-quorum observer slots 0 and 2 stay greedy. Other
    # slots keep their already-sampled legal random actions.
    guard_seed = 79
    guard_visible = np.asarray([
        [1, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0],
    ], dtype=np.int64)
    guard_expected_rng = np.random.RandomState(guard_seed)
    guard_joint_mask = epsilon_random_mask(
        guard_expected_rng, 1.0, 6, True
    )
    guard_mask, guard_protected = (
        protect_exact_visible_prey_quorum_random_replacements(
            guard_joint_mask,
            np.ones(6, dtype=bool),
            guard_visible,
            capture_quorum=2,
        )
    )
    guard_random = sample_action_indices(
        guard_expected_rng, available, 6, 7
    )
    guard_expected = (
        (1 - guard_mask) * greedy + guard_mask * guard_random
    )
    guard_policy = build_policy(True, 1.0, guard_seed, True)
    guard_actions = guard_policy.get_actions(
        obs,
        states,
        torch.as_tensor(available),
        t_env=60000,
        explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
        pre_capture_visible_prey_mask=guard_visible,
    )[0]
    np.testing.assert_array_equal(
        np.argmax(guard_actions, axis=-1), guard_expected
    )
    np.testing.assert_array_equal(guard_protected, [1, 0, 1, 0, 0, 0])
    guard_diag = guard_policy.last_action_exploration_diagnostic
    assert guard_diag[
        "pre_capture_visible_prey_quorum_guard_applied"
    ] == 1
    assert guard_diag[
        "pre_capture_visible_prey_quorum_protected_slots"
    ] == [0, 2]
    assert guard_diag["random_replacement_slots"] == [1, 3, 4, 5]
    assert guard_diag["random_replacement_slot_count"] == 4
    np.testing.assert_array_equal(
        guard_policy.rng.randint(0, 1000000, size=32),
        guard_expected_rng.randint(0, 1000000, size=32),
    )

    # During the simultaneous-prey completion window the override changes
    # only the one joint Bernoulli probability. Random actions remain legal
    # and the production diagnostic exposes the exact effective epsilon.
    floor_policy = build_policy(True, 0.95, 47)
    floor_policy.get_actions(
        obs,
        states,
        torch.as_tensor(available),
        t_env=20000,
        explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
        post_capture_joint_greedy_floor=0.25,
    )
    floor_diag = floor_policy.last_action_exploration_diagnostic
    assert abs(float(floor_diag["base_epsilon"]) - 0.95) < 1e-12
    assert abs(float(floor_diag["epsilon"]) - 0.75) < 1e-12
    assert floor_diag["post_capture_greedy_floor_applied"] == 1
    assert floor_diag["invalid_available_action_count"] == 0
    assert floor_diag["post_capture_explore_bounded_applied"] == 0

    # Regression for the run168-at-step-6400 failure: post-capture rollout
    # deliberately omits the pre-capture visibility mask.  A non-explore
    # joint decision must stay in the post-capture lifecycle instead of
    # falling through to the pre-capture quorum guard.
    post_no_explore_policy = build_policy(True, 0.0, 83, True)
    post_no_explore_actions = post_no_explore_policy.get_actions(
        obs,
        states,
        torch.as_tensor(available),
        t_env=6400,
        explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
        post_capture_joint_greedy_floor=0.25,
        post_capture_explore_max_random_agents=1,
    )[0]
    np.testing.assert_array_equal(
        np.argmax(post_no_explore_actions, axis=-1), greedy
    )
    post_no_explore_diag = (
        post_no_explore_policy.last_action_exploration_diagnostic
    )
    assert post_no_explore_diag["joint_explore"] == 0
    assert post_no_explore_diag[
        "pre_capture_visible_prey_quorum_guard_applied"
    ] == 0
    assert post_no_explore_diag[
        "pre_capture_visible_prey_quorum_protected_slots"
    ] == []

    # The same lifecycle separation must hold when the post-capture branch
    # explores: bounded replacement is applied and the pre-capture guard is
    # not evaluated without a mask.
    post_explore_policy = build_policy(True, 1.0, 89, True)
    post_explore_policy.get_actions(
        obs,
        states,
        torch.as_tensor(available),
        t_env=6400,
        explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
        post_capture_joint_greedy_floor=0.25,
        post_capture_explore_max_random_agents=1,
    )
    post_explore_diag = post_explore_policy.last_action_exploration_diagnostic
    assert post_explore_diag["joint_explore"] == 1
    assert post_explore_diag["post_capture_explore_bounded_applied"] == 1
    assert post_explore_diag[
        "pre_capture_visible_prey_quorum_guard_applied"
    ] == 0
    assert post_explore_diag[
        "pre_capture_visible_prey_quorum_protected_slots"
    ] == []

    # Formal run161 follow-up semantics: retain the same one joint Bernoulli,
    # then replace at most one uniformly chosen alive slot.  The expected RNG
    # reconstruction proves the exact order: branch, slot, legal actions.
    bounded_seed = 1
    bounded_dones = np.asarray([0, 0, 0, 0, 1, 0], dtype=bool)
    bounded_expected_rng = np.random.RandomState(bounded_seed)
    bounded_joint_mask = epsilon_random_mask(
        bounded_expected_rng, 0.75, 6, True
    )
    bounded_mask = bound_joint_random_replacements(
        bounded_expected_rng,
        bounded_joint_mask,
        ~bounded_dones,
        POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS,
    )
    bounded_random = sample_action_indices(
        bounded_expected_rng, available, 6, 7
    )
    bounded_expected = (
        (1 - bounded_mask) * greedy + bounded_mask * bounded_random
    )
    bounded_policy = build_policy(True, 1.0, bounded_seed)
    bounded_actions = bounded_policy.get_actions(
        obs,
        states,
        torch.as_tensor(available),
        t_env=20000,
        explore=True,
        dones=torch.as_tensor(bounded_dones.reshape(6, 1)),
        post_capture_joint_greedy_floor=0.25,
        post_capture_explore_max_random_agents=(
            POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS
        ),
    )[0]
    np.testing.assert_array_equal(
        np.argmax(bounded_actions, axis=-1), bounded_expected
    )
    bounded_diag = bounded_policy.last_action_exploration_diagnostic
    assert bounded_diag["joint_explore"] == 1
    assert bounded_diag["post_capture_explore_bounded_applied"] == 1
    assert bounded_diag["random_replacement_slot_count"] == 1
    assert bounded_diag["random_replacement_slots"] == [3]
    assert bounded_diag["invalid_available_action_count"] == 0
    assert bounded_diag["dead_slot_non_stay_violation_count"] == 0
    assert bounded_diag["selected_actions"][4] == 0
    assert np.count_nonzero(
        np.asarray(bounded_diag["selected_actions"])
        != np.asarray(bounded_diag["greedy_actions"])
    ) <= 1

    production_greedy_diag = build_policy(True, 0.0, 31)
    production_greedy_diag.get_actions(
        obs, states, torch.as_tensor(available), t_env=60000, explore=True,
        dones=torch.zeros((6, 1), dtype=torch.bool),
    )
    diag = production_greedy_diag.last_action_exploration_diagnostic
    assert diag["joint_explore"] == 0
    assert diag["final_equals_greedy"] == 1
    assert diag["non_explore_greedy_mismatch"] == 0
    value_diag = diag["frontier_value_ranking_diagnostic"]
    assert value_diag["sampled"] == 0
    assert value_diag["factor_count"] == 0

    checkpoint_rng = np.random.RandomState(41)
    checkpoint_rng.rand(17)
    checkpoint = {
        "policy_exploration_checkpoint_version": 6,
        "joint_epsilon_exploration_enabled": True,
        "policy_epsilon_start": POLICY_EPSILON_START,
        "policy_epsilon_finish": POLICY_EPSILON_FINISH,
        "policy_epsilon_anneal_time": POLICY_EPSILON_ANNEAL_TIME,
        "post_capture_joint_greedy_floor": (
            POST_CAPTURE_JOINT_GREEDY_FLOOR
        ),
        "post_capture_explore_max_random_agents": (
            POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS
        ),
        "pre_capture_visible_prey_quorum_guard": True,
        "pre_capture_visible_prey_quorum_greedy_frontier_guard": True,
        "policy_rng_states": {"policy_0": checkpoint_rng.get_state()},
    }
    payload = io.BytesIO()
    torch.save(checkpoint, payload)
    payload.seek(0)
    restored = torch.load(payload)
    resumed_rng = np.random.RandomState(999)
    resumed_rng.set_state(restored["policy_rng_states"]["policy_0"])
    np.testing.assert_array_equal(
        checkpoint_rng.randint(0, 1000000, size=64),
        resumed_rng.randint(0, 1000000, size=64),
    )

    # Default-off compatibility keeps the legacy draw order: mask, then action.
    seed = 29
    expected_rng = np.random.RandomState(seed)
    expected_mask = epsilon_random_mask(expected_rng, 0.886, 6, False)
    expected_random = sample_action_indices(expected_rng, available, 6, 7)
    expected = (1 - expected_mask) * greedy + expected_mask * expected_random
    legacy = build_policy(False, 0.886, seed).get_actions(
        obs, states, torch.as_tensor(available), t_env=60000, explore=True
    )[0]
    np.testing.assert_array_equal(np.argmax(legacy, axis=-1), expected)


def _check_episode_diagnostic_contract():
    try:
        from runner.wolfpack_runner import WolfpackRunner
    except ImportError:
        return

    obs_dim = 2 + 4 + 2 * (6 - 1) + 7 * 2 + 4 + 1
    local_obs = np.zeros((6, obs_dim), dtype=np.float32)
    prey_start = 2 + 4 + 2 * (6 - 1)
    local_obs[0, prey_start + 2] = 1.0
    local_obs[2, prey_start + 3] = 1.0
    local_obs[1, prey_start + 7 + 4] = 1.0
    local_obs[5] = -1.0
    visible_mask = WolfpackRunner._visible_prey_mask_from_local_vector_obs(
        local_obs,
        num_agents=6,
        max_food_num=2,
    )
    np.testing.assert_array_equal(
        visible_mask,
        np.asarray([
            [1, 0], [0, 1], [1, 0], [0, 0], [0, 0], [0, 0],
        ], dtype=np.int64),
    )
    malformed_obs = local_obs.copy()
    malformed_obs[0, prey_start + 3] = 1.0
    try:
        WolfpackRunner._visible_prey_mask_from_local_vector_obs(
            malformed_obs,
            num_agents=6,
            max_food_num=2,
        )
        raise AssertionError("invalid local prey one-hot did not fail loud")
    except RuntimeError:
        pass

    diagnostics = []
    flags = [1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    for index, explored in enumerate(flags):
        diagnostics.append({
            "schema_version": 4,
            "joint_explore": explored,
            "epsilon": 0.6,
            "final_equals_greedy": 1 - explored,
            "explore_final_equals_greedy": 0,
            "non_explore_greedy_mismatch": 0,
            "invalid_available_action_count": 0,
            "dead_slot_non_stay_violation_count": 0,
            "alive_slot_count": 4 + (index % 3),
            "unique_alive_action_count": 3,
            "action_histogram": [1, 1, 1, 1, 0, 0, 0],
            "selected_actions": [0, 1, 2, 3, 4, 4],
            "post_capture_explore_bounded_applied": int(explored),
            "random_replacement_slot_count": int(explored),
            "pre_capture_visible_prey_quorum_guard_applied": int(
                explored and index == 0
            ),
            "pre_capture_visible_prey_quorum_protected_slots": (
                [0, 1] if explored and index == 0 else []
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied": int(
                index == 1
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots": (
                [0, 1] if index == 1 else []
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots": (
                [0] if index == 1 else []
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots": [],
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots": (
                [0] if index == 1 else []
            ),
        })

    with tempfile.TemporaryDirectory() as directory:
        runner = WolfpackRunner.__new__(WolfpackRunner)
        runner.run_dir = directory
        runner.last_episode_step_info = {
            "topology_events": [
                {"capture_identity_matched_event_count": 1},
                {"capture_identity_matched_event_count": 0},
            ]
        }
        runner._dump_joint_exploration_episode_diagnostic(
            train_step=20000,
            episode_index=17,
            diagnostics=diagnostics,
            env_info={
                "capture_events": 2.0,
                "win_rate": 1.0,
                "terminal_win_bonus_event_count": 1.0,
            },
        )
        import pandas as pd
        row = pd.read_csv(os.path.join(
            directory, "progress_train_joint_exploration_episode.csv"
        )).iloc[0]
        assert int(row["joint_decision_count"]) == 10
        assert str(row["joint_explore_flag_bits"]).zfill(10) == "1100100001"
        assert str(row["final_equals_greedy_flag_bits"]).zfill(10) == "0011011110"
        assert int(row["explore_decision_count"]) == 4
        assert int(row["non_explore_decision_count"]) == 6
        assert int(row["non_explore_streak_max"]) == 4
        assert int(row["non_explore_streak_ge2_count"]) == 2
        assert int(row["non_explore_streak_ge4_count"]) == 1
        assert int(row["invalid_available_action_count"]) == 0
        assert int(row["exact_matched_capture_events"]) == 1
        assert int(row["diagnostic_schema_version"]) == 4
        assert int(row["post_capture_explore_bounded_applied_count"]) == 4
        assert int(row[
            "pre_capture_visible_prey_quorum_guard_applied_count"
        ]) == 1
        assert int(row[
            "pre_capture_visible_prey_quorum_protected_slot_count_sum"
        ]) == 2
        assert int(row[
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied_count"
        ]) == 1
        assert int(row[
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slot_count_sum"
        ]) == 1
        assert int(row["random_replacement_slot_count_sum"]) == 4
        assert int(row["random_replacement_slot_count_max"]) == 1

        post_capture = WolfpackRunner._build_post_capture_row(
            environment_episode_id=5,
            training_env_step=24000,
            episode_step=11,
            first_capture_step=10,
            first_capture_target_id=0,
            first_capture_participant_slots=(0, 1),
            step_info={
                "food_alive_statuses": [False, True],
                "food_freeze_remaining": [0.96, 0.0],
                "food_positions": [(2, 2), (8, 8)],
                "player_slot_positions": [
                    (1, 2), (3, 2), None, (8, 7), (9, 8), None,
                ],
                "capture_events": [],
                "success_now": False,
            },
            action_diagnostic={
                "joint_explore": 1,
                "epsilon": 0.75,
                "greedy_actions": [0, 0, 0, 0, 0, 0],
                "selected_actions": [0, 0, 0, 1, 0, 0],
                "post_capture_explore_max_random_agents": 1,
                "post_capture_explore_bounded_applied": 1,
                "random_replacement_slots": [3],
            },
        )
        assert post_capture["nearest_alive_player_slots_to_food"] == "0|1;3|4"
        assert post_capture["random_replacement_slots"] == "3"
        assert post_capture["post_capture_explore_bounded_applied"] == 1

        adjacency = np.zeros((1, 6, 2), dtype=np.int64)
        adjacency[0, [0, 1], 0] = 1
        adjacency[0, [3, 4, 5], 1] = 1
        pre_capture = WolfpackRunner._build_pre_capture_transition_row(
            environment_episode_id=5,
            training_env_step=24000,
            episode_step=40,
            step_info={
                "food_alive_statuses": [False, True],
                "food_freeze_remaining": [1.0, 0.0],
                "food_positions": [(2, 2), (8, 8)],
                "player_slot_positions": [
                    (1, 2), (3, 2), None, (8, 7), (9, 8), None,
                ],
                "food_visible_player_slots": [[0, 1], [3, 4]],
                "capture_events": [{"target_id": 0}],
                "success_now": False,
            },
            action_diagnostic={
                "joint_explore": 0,
                "epsilon": 0.8,
                "greedy_actions": [0, 1, 2, 3, 4, 0],
                "selected_actions": [0, 1, 2, 3, 4, 0],
                "random_replacement_slots": [],
                "pre_capture_visible_prey_quorum_guard_applied": 1,
                "pre_capture_visible_prey_quorum_protected_slots": [0, 1],
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied": 1,
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots": [0, 1],
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots": [0],
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots": [],
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots": [0],
                "unconstrained_greedy_actions": [3, 1, 2, 3, 4, 0],
                "frontier_value_ranking_diagnostic": {
                    "schema_version": 1,
                    "sampled": 1,
                    "slot_ids": [0],
                    "action_dim": 7,
                    "factor_count": 2,
                    "legal_action_mask": [[1, 1, 1, 1, 1, 1, 1]],
                    "frontier_action_mask": [[1, 0, 0, 0, 0, 0, 0]],
                    "message_action_utilities": [[
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                    ]],
                    "coordinate_joint_q_values": [[
                        1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                    ]],
                    "coordinate_factor_q_values": [[
                        [0.7, 1.0], [0.6, 1.0], [0.5, 1.0],
                        [0.4, 1.0], [0.3, 1.0], [0.2, 1.0],
                        [0.1, 1.0],
                    ]],
                },
            },
            greedy_joint_q_value=np.asarray([2.5], dtype=np.float32),
            greedy_message_action_margins=np.asarray(
                [[0.1, 0.2, 0.3, 0.4, np.nan, np.nan]],
                dtype=np.float32,
            ),
            greedy_factor_q_values=np.asarray(
                [[[1.25], [1.75]]], dtype=np.float32
            ),
            adjacency=adjacency,
            learned_factor_count=2,
        )
        assert pre_capture["diagnostic_schema_version"] == 4
        assert pre_capture["state_action_alignment"] == (
            "action_s_t__info_s_t_plus_1"
        )
        assert pre_capture["active_factor_agent_slots"] == "f0:0|1;f1:3|4|5"
        assert pre_capture["active_factor_orders"] == "2;3"
        assert pre_capture["nearest_alive_player_slots_to_food"] == "0|1;3|4"
        assert pre_capture["food_visible_player_slots"] == "0|1;3|4"
        assert pre_capture["food_observer_counts"] == "2;2"
        assert pre_capture[
            "pre_capture_visible_prey_quorum_guard_applied"
        ] == 1
        assert pre_capture[
            "pre_capture_visible_prey_quorum_protected_slots"
        ] == "0|1"
        assert pre_capture[
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied"
        ] == 1
        assert pre_capture[
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots"
        ] == "0"
        assert pre_capture["unconstrained_greedy_actions"].split(";")[0] == "3"
        assert pre_capture["frontier_value_ranking_schema_version"] == 1
        assert pre_capture["frontier_value_ranking_sampled"] == 1
        assert pre_capture["frontier_value_ranking_slot_ids"] == "0"
        assert pre_capture["frontier_value_ranking_action_dim"] == 7
        assert pre_capture["frontier_value_ranking_factor_count"] == 2
        assert len(pre_capture[
            "frontier_value_ranking_coordinate_factor_q"
        ].split(";")) == 14
        assert abs(float(pre_capture["greedy_joint_q_value"]) - 2.5) < 1e-12

        snapshots = []
        for episode_step in range(1, 41):
            snapshot = dict(pre_capture)
            snapshot["episode_step"] = episode_step
            snapshots.append(snapshot)
        finalized = WolfpackRunner._finalize_pre_capture_window(
            snapshots,
            first_capture_step=40,
            first_capture_target_id=0,
            first_capture_participant_slots=(0, 1),
        )
        assert len(finalized) == 33
        assert finalized[0]["offset_to_first_capture"] == -32
        assert finalized[-1]["offset_to_first_capture"] == 0
        assert finalized[-1]["first_capture_target_id"] == 0
        assert finalized[-1]["first_capture_participant_slots"] == "0;1"
        prefix = WolfpackRunner._finalize_pre_capture_window(
            snapshots,
            first_capture_step=40,
            first_capture_target_id=0,
            first_capture_participant_slots=(0, 1),
            history_steps=None,
        )
        assert len(prefix) == 40
        assert prefix[0]["offset_to_first_capture"] == -39
        assert prefix[-1]["offset_to_first_capture"] == 0


def main():
    _check_policy_epsilon_schedule_contract()
    _check_checkpoint_schedule_contract()
    _check_post_capture_stage_contract()
    legacy_greedy, joint_greedy = _check_mask_contract()
    _check_bounded_post_capture_mask_contract()
    _check_pre_capture_visible_prey_quorum_guard_contract()
    _check_run167_real_quorum_loss_counterfactual()
    _check_run168_real_greedy_frontier_counterfactual()
    _check_q_target_frontier_mask_contract()
    _check_run169_real_q_target_counterfactual()
    _check_available_actions_and_rng_roundtrip()
    _check_production_policy_path()
    _check_episode_diagnostic_contract()
    print(
        "joint epsilon exploration fixture passed: "
        "legacy_full_greedy={:.8f}, joint_full_greedy={:.6f}".format(
            legacy_greedy,
            joint_greedy,
        )
    )


if __name__ == '__main__':
    main()
