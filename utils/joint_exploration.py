"""Deterministic NumPy helpers for multi-agent epsilon exploration."""

import numpy as np


def epsilon_random_mask(rng, epsilon, num_agents, joint_decision=False):
    """Draw epsilon-greedy replacements at agent or joint-decision scope."""
    epsilon = float(epsilon)
    num_agents = int(num_agents)
    if not np.isfinite(epsilon) or epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if joint_decision:
        take_random = bool(float(rng.rand()) < epsilon)
        return np.full(num_agents, take_random, dtype=np.int64)
    return (rng.rand(num_agents) < epsilon).astype(np.int64)


def bound_joint_random_replacements(
        rng,
        joint_random_mask,
        alive_mask,
        max_random_agents):
    """Limit an active joint-random branch to uniformly chosen alive slots.

    ``max_random_agents == 0`` is the exact legacy contract.  Positive limits
    are intended for short task-critical windows where replacing every alive
    action at once destroys an already useful joint configuration.  The joint
    Bernoulli is still drawn once; this helper only chooses which alive slots
    receive the already-defined uniform-random replacement.
    """
    joint_random_mask = np.asarray(joint_random_mask, dtype=np.int64).reshape(-1)
    alive_mask = np.asarray(alive_mask, dtype=bool).reshape(-1)
    max_random_agents = int(max_random_agents)
    if joint_random_mask.shape != alive_mask.shape:
        raise ValueError("joint random and alive masks must have equal shape")
    if max_random_agents < 0:
        raise ValueError("max_random_agents must be non-negative")
    if joint_random_mask.size == 0:
        raise ValueError("joint random mask must not be empty")
    if not np.all(joint_random_mask == joint_random_mask[0]):
        raise ValueError("joint random mask must contain one shared branch flag")
    if max_random_agents == 0 or not bool(joint_random_mask[0]):
        return joint_random_mask.copy()

    alive_indices = np.flatnonzero(alive_mask)
    if alive_indices.size <= max_random_agents:
        return alive_mask.astype(np.int64)
    selected = rng.choice(
        alive_indices,
        size=max_random_agents,
        replace=False,
    )
    bounded = np.zeros_like(joint_random_mask)
    bounded[selected] = 1
    return bounded


def protect_exact_visible_prey_quorum_random_replacements(
        joint_random_mask,
        alive_mask,
        visible_prey_mask,
        capture_quorum=2):
    """Keep an already-observed exact prey quorum on the greedy joint action.

    ``visible_prey_mask[a, p]`` must be derived from agent ``a``'s current
    local observation, never from hidden environment state.  When exactly
    ``capture_quorum`` alive agents currently observe prey ``p``, those slots
    are removed from the already-drawn joint-random replacement mask.  Other
    alive slots retain their sampled random actions.  This helper is fully
    deterministic and deliberately accepts no RNG, so the policy RNG stream
    and the one-Bernoulli epsilon contract are unchanged.
    """
    joint_random_mask = np.asarray(
        joint_random_mask, dtype=np.int64
    ).reshape(-1)
    alive_mask = np.asarray(alive_mask, dtype=bool).reshape(-1)
    visible_prey_mask = np.asarray(visible_prey_mask)
    capture_quorum = int(capture_quorum)

    if joint_random_mask.shape != alive_mask.shape:
        raise ValueError("joint random and alive masks must have equal shape")
    if joint_random_mask.size == 0:
        raise ValueError("joint random mask must not be empty")
    if not np.all(joint_random_mask == joint_random_mask[0]):
        raise ValueError("joint random mask must contain one shared branch flag")
    if capture_quorum <= 0:
        raise ValueError("capture quorum must be positive")
    if visible_prey_mask.ndim != 2:
        raise ValueError("visible prey mask must be [agents, prey]")
    if visible_prey_mask.shape[0] != joint_random_mask.size:
        raise ValueError("visible prey mask agent axis does not match policy")
    if visible_prey_mask.shape[1] <= 0:
        raise ValueError("visible prey mask must contain at least one prey")
    if not np.isin(visible_prey_mask, (0, 1, False, True)).all():
        raise ValueError("visible prey mask must be boolean")

    visible_prey_mask = visible_prey_mask.astype(bool, copy=False)
    if np.any(visible_prey_mask[~alive_mask]):
        raise ValueError("dead slots cannot report locally visible prey")
    protected = np.zeros_like(alive_mask)
    if bool(joint_random_mask[0]):
        alive_visible = visible_prey_mask & alive_mask[:, None]
        exact_quorum_prey = (
            alive_visible.sum(axis=0) == capture_quorum
        )
        if np.any(exact_quorum_prey):
            protected = np.any(
                alive_visible[:, exact_quorum_prey], axis=1
            )

    effective = joint_random_mask.copy()
    effective[~alive_mask] = 0
    effective[protected] = 0
    return effective, protected.astype(np.int64)


def build_exact_visible_prey_quorum_frontier_action_mask(
        alive_mask,
        visible_prey_mask,
        visible_prey_offsets,
        available_actions,
        sight_radius,
        prey_max_step=1,
        capture_quorum=2):
    """Restrict only locally observed exact-quorum frontier actions.

    The mask uses the current actor observation only.  For every prey seen by
    exactly ``capture_quorum`` alive agents, each observer's legal actions are
    scored by how many such prey are guaranteed to remain inside the L1 sight
    radius after an adversarial legal one-step prey move.  Only actions with
    the maximum guarantee are retained.  Interior observers are unchanged,
    because every legal action is equally safe there.

    The Wolfpack vector-action contract is fixed here: actions 0..3 move in
    absolute N/E/S/W directions and actions 4..6 do not move.  The helper is
    deterministic, receives no RNG, and never reads hidden environment state.
    """
    alive_mask = np.asarray(alive_mask, dtype=bool).reshape(-1)
    visible_prey_mask = np.asarray(visible_prey_mask)
    visible_prey_offsets = np.asarray(
        visible_prey_offsets, dtype=np.float64
    )
    available_actions = np.asarray(available_actions)
    sight_radius = int(sight_radius)
    prey_max_step = int(prey_max_step)
    capture_quorum = int(capture_quorum)

    if alive_mask.size == 0:
        raise ValueError("frontier action mask requires at least one agent")
    if visible_prey_mask.ndim != 2:
        raise ValueError("visible prey mask must be [agents, prey]")
    if visible_prey_mask.shape[0] != alive_mask.size:
        raise ValueError("visible prey mask agent axis does not match policy")
    if visible_prey_mask.shape[1] <= 0:
        raise ValueError("visible prey mask must contain at least one prey")
    if not np.isin(visible_prey_mask, (0, 1, False, True)).all():
        raise ValueError("visible prey mask must be boolean")
    visible_prey_mask = visible_prey_mask.astype(bool, copy=False)
    expected_offset_shape = visible_prey_mask.shape + (2,)
    if visible_prey_offsets.shape != expected_offset_shape:
        raise ValueError(
            "visible prey offsets must have shape {}".format(
                expected_offset_shape
            )
        )
    if available_actions.ndim != 2:
        raise ValueError("available actions must be [agents, actions]")
    if available_actions.shape[0] != alive_mask.size:
        raise ValueError("available actions agent axis does not match policy")
    if available_actions.shape[1] != 7:
        raise ValueError(
            "Wolfpack frontier action mask requires seven actions"
        )
    if sight_radius <= 0 or prey_max_step < 0:
        raise ValueError("frontier sight/motion bounds are invalid")
    if capture_quorum <= 0:
        raise ValueError("capture quorum must be positive")
    if np.any(visible_prey_mask[~alive_mask]):
        raise ValueError("dead slots cannot report locally visible prey")
    if not np.isfinite(
            visible_prey_offsets[visible_prey_mask]
    ).all():
        raise ValueError("visible prey offsets must be finite")
    visible_distances = np.abs(
        visible_prey_offsets[visible_prey_mask]
    ).sum(axis=1)
    if np.any(visible_distances > sight_radius):
        raise ValueError(
            "visible prey offsets exceed the policy sight radius"
        )

    legal = available_actions > 0.5
    if np.any(legal.sum(axis=1) <= 0):
        raise ValueError("every policy slot requires a legal action")
    if np.any(legal[~alive_mask].sum(axis=1) != 1):
        raise ValueError("dead slots must expose exactly one legal action")

    exact_quorum_prey = (
        (visible_prey_mask & alive_mask[:, None]).sum(axis=0)
        == capture_quorum
    )
    action_deltas = np.asarray([
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
        [0, 0],
        [0, 0],
        [0, 0],
    ], dtype=np.float64)

    frontier_mask = legal.copy()
    eligible = np.zeros(alive_mask.size, dtype=np.int64)
    constrained = np.zeros(alive_mask.size, dtype=np.int64)
    conflicts = np.zeros(alive_mask.size, dtype=np.int64)
    for agent in np.flatnonzero(alive_mask):
        relevant = np.flatnonzero(
            exact_quorum_prey & visible_prey_mask[agent]
        )
        if relevant.size == 0:
            continue
        eligible[agent] = 1
        scores = np.full(7, -1, dtype=np.int64)
        for action in np.flatnonzero(legal[agent]):
            post_action_offsets = (
                visible_prey_offsets[agent, relevant]
                - action_deltas[action]
            )
            post_action_distances = np.abs(
                post_action_offsets
            ).sum(axis=1)
            scores[action] = int(np.count_nonzero(
                post_action_distances + prey_max_step <= sight_radius
            ))
        best_score = int(scores.max())
        if best_score < 0:
            raise RuntimeError(
                "frontier action mask found no legal action"
            )
        best_actions = legal[agent] & (scores == best_score)
        if not np.any(best_actions):
            raise RuntimeError(
                "frontier action mask produced an empty action set"
            )
        frontier_mask[agent] = best_actions
        constrained[agent] = int(not np.array_equal(
            best_actions, legal[agent]
        ))
        conflicts[agent] = int(best_score < int(relevant.size))

    if np.any(frontier_mask & ~legal):
        raise RuntimeError("frontier action mask enabled an illegal action")
    if np.any(frontier_mask.sum(axis=1) <= 0):
        raise RuntimeError("frontier action mask removed every legal action")
    return frontier_mask.astype(np.int64), eligible, constrained, conflicts


def build_batched_wolfpack_frontier_action_mask_from_local_obs(
        obs_batch,
        dones_batch,
        available_actions,
        num_agents,
        max_food_num,
        sight_radius,
        prey_max_step=1,
        capture_quorum=2):
    """Build the rollout frontier mask for replay next-state actions.

    The deployed Wolfpack policy derives its exact-quorum constraint from the
    current local vector observations.  Double-Q next-action selection must
    use the same constrained policy; otherwise it bootstraps the value of raw
    actions that production execution will replace.  This batched helper uses
    only replayed local observations, replayed dead-slot masks, and replayed
    environment action legality.  It consumes no RNG and reads no central or
    future state.

    A positive freeze-remaining field means the state is in the post-capture
    lifecycle, where the existing post-capture policy remains authoritative.
    Those rows retain the environment legality mask unchanged.
    """
    obs = np.asarray(obs_batch, dtype=np.float32)
    dones = np.asarray(dones_batch).astype(bool)
    available = np.asarray(available_actions)
    num_agents = int(num_agents)
    max_food_num = int(max_food_num)
    sight_radius = int(sight_radius)

    if obs.ndim != 3 or obs.shape[1] != num_agents:
        raise ValueError(
            "batched frontier observations must be [batch, agents, obs]"
        )
    if dones.ndim == 3 and dones.shape[-1] == 1:
        dones = dones[..., 0]
    if dones.shape != obs.shape[:2]:
        raise ValueError("batched frontier dones do not match observations")
    if available.ndim != 3 or available.shape[:2] != obs.shape[:2]:
        raise ValueError(
            "batched frontier available actions do not match observations"
        )
    if available.shape[2] != 7:
        raise ValueError("Wolfpack batched frontier requires seven actions")
    if max_food_num <= 0:
        raise ValueError("Wolfpack batched frontier requires prey slots")
    expected_obs_dim = (
        2 + 4 + 2 * (num_agents - 1) + 7 * max_food_num + 4 + 1
    )
    if obs.shape[2] != expected_obs_dim:
        raise ValueError(
            "Wolfpack batched frontier observation width is {}, expected {}"
            .format(obs.shape[2], expected_obs_dim)
        )
    if not np.isfinite(obs).all():
        raise FloatingPointError(
            "batched frontier observations contain non-finite values"
        )
    if not np.isfinite(available).all():
        raise FloatingPointError(
            "batched frontier action legality contains non-finite values"
        )

    legal = available > 0.5
    if np.any(legal.sum(axis=2) <= 0):
        raise ValueError("every replay policy slot requires a legal action")
    output = legal.copy()
    eligible = np.zeros(obs.shape[:2], dtype=np.int64)
    constrained = np.zeros(obs.shape[:2], dtype=np.int64)
    conflicts = np.zeros(obs.shape[:2], dtype=np.int64)
    pre_capture_rows = np.zeros(obs.shape[0], dtype=np.int64)
    prey_start = 2 + 4 + 2 * (num_agents - 1)

    for batch_index in range(obs.shape[0]):
        alive = ~dones[batch_index]
        if not np.any(alive):
            # Terminal rows do not contribute a bootstrap value.  Preserve
            # their replayed legality instead of imposing dead-slot encoding
            # requirements on the environment terminal observation.
            continue
        if np.any(np.all(obs[batch_index, alive] == -1.0, axis=1)):
            raise RuntimeError(
                "alive replay slot has an inactive Wolfpack observation"
            )
        if np.any(~np.all(obs[batch_index, ~alive] == -1.0, axis=1)):
            raise RuntimeError(
                "inactive replay slot has a live Wolfpack observation"
            )
        if np.any(legal[batch_index, ~alive].sum(axis=1) != 1):
            raise RuntimeError(
                "inactive replay slot must expose exactly one legal action"
            )
        if np.any(~legal[batch_index, ~alive, 4]):
            raise RuntimeError(
                "inactive replay slot must expose stay as its legal action"
            )

        visible = np.zeros((num_agents, max_food_num), dtype=np.int64)
        offsets = np.full(
            (num_agents, max_food_num, 2), np.nan, dtype=np.float32
        )
        freeze_values = []
        for food_id in range(max_food_num):
            block_start = prey_start + 7 * food_id
            relative = obs[
                batch_index, :, block_start:block_start + 2
            ]
            orientation = obs[
                batch_index, :, block_start + 2:block_start + 6
            ]
            if np.any(
                    alive[:, None]
                    & (orientation != 0.0)
                    & (orientation != 1.0)):
                raise RuntimeError(
                    "replay Wolfpack prey orientation is non-binary"
                )
            orientation_sum = orientation.sum(axis=1)
            if np.any(
                    alive
                    & (orientation_sum != 0.0)
                    & (orientation_sum != 1.0)):
                raise RuntimeError(
                    "replay Wolfpack prey orientation is not one-hot"
                )
            food_visible = alive & (orientation_sum == 1.0)
            visible[:, food_id] = food_visible.astype(np.int64)
            offsets[food_visible, food_id] = relative[food_visible]
            alive_freeze = obs[
                batch_index, alive, block_start + 6
            ]
            if not np.allclose(
                    alive_freeze,
                    alive_freeze[0],
                    rtol=0.0,
                    atol=1e-6):
                raise RuntimeError(
                    "replay Wolfpack freeze status differs across agents"
                )
            freeze_values.append(float(alive_freeze[0]))

        if any(value < 0.0 or value > 1.0 for value in freeze_values):
            raise RuntimeError("replay Wolfpack freeze status is invalid")
        if any(value > 0.0 for value in freeze_values):
            continue
        pre_capture_rows[batch_index] = 1
        (
            row_mask,
            row_eligible,
            row_constrained,
            row_conflicts,
        ) = build_exact_visible_prey_quorum_frontier_action_mask(
            alive_mask=alive,
            visible_prey_mask=visible,
            visible_prey_offsets=offsets,
            available_actions=available[batch_index],
            sight_radius=sight_radius,
            prey_max_step=prey_max_step,
            capture_quorum=capture_quorum,
        )
        output[batch_index] = row_mask.astype(bool)
        eligible[batch_index] = row_eligible
        constrained[batch_index] = row_constrained
        conflicts[batch_index] = row_conflicts

    if np.any(output & ~legal):
        raise RuntimeError("batched frontier enabled an illegal action")
    if np.any(output.sum(axis=2) <= 0):
        raise RuntimeError("batched frontier removed every legal action")
    return (
        output.astype(np.int64),
        eligible,
        constrained,
        conflicts,
        pre_capture_rows,
    )


def sample_action_indices(rng, available_actions, batch_size, act_dim):
    """Uniformly sample one legal action per row using the supplied RNG."""
    batch_size = int(batch_size)
    act_dim = int(act_dim)
    if available_actions is None:
        return rng.randint(0, act_dim, size=batch_size).astype(np.int64)

    aa = np.asarray(available_actions).reshape(batch_size, act_dim)
    out = np.zeros(batch_size, dtype=np.int64)
    for row in range(batch_size):
        valid = np.flatnonzero(aa[row] > 0.5)
        if valid.size == 0:
            valid = np.arange(act_dim, dtype=np.int64)
        out[row] = int(rng.choice(valid))
    return out
