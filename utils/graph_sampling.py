"""Graph sampling-mode helpers shared by rollout code and lightweight tests."""

import numpy as np


def write_graph_advantage_sequence(
        storage,
        ready_storage,
        episode_indices,
        graph_advantage,
        valid_transition):
    """Persist computed graph-return advantage for replay confidence.

    ``PolicyAdjBuffer`` historically allocated ``self.advantage`` and sampled
    it as the capture-outcome confidence source, but never wrote the computed
    graph advantage into it. Keeping the write in this NumPy-only helper makes
    the time/episode axes and overwrite semantics directly testable without a
    torch runtime.
    """
    values = np.asarray(storage)
    ready = np.asarray(ready_storage)
    indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    advantage = np.asarray(graph_advantage, dtype=np.float32)
    valid = np.asarray(valid_transition, dtype=bool)

    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError(
            "graph advantage storage must have shape [time, buffer, 1]"
        )
    if ready.shape != values.shape:
        raise ValueError("graph advantage ready storage must match value storage")
    expected = (values.shape[0], indices.size)
    if advantage.shape != expected or valid.shape != expected:
        raise ValueError(
            "graph advantage and valid mask must have shape {}, got {} and {}"
            .format(expected, advantage.shape, valid.shape)
        )
    if np.unique(indices).size != indices.size:
        raise ValueError("graph advantage episode indices must be unique")
    if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= values.shape[1]):
        raise IndexError("graph advantage episode index is outside replay storage")
    if np.any(~np.isfinite(advantage[valid])):
        raise FloatingPointError(
            "computed graph advantage is non-finite on a valid transition"
        )

    stored = np.where(valid, advantage, 0.0).astype(values.dtype, copy=False)
    values[:, indices, 0] = stored
    ready[:, indices, 0] = valid
    return stored


def require_ready_graph_advantage(
        graph_advantage,
        ready_mask,
        labelled_transition):
    """Validate and return the detached replay confidence source.

    A genuinely zero computed advantage is valid. An unwritten zero is not:
    the explicit ready mask distinguishes those states and fails before a
    class-complete outcome cohort can be silently erased.
    """
    advantage = np.asarray(graph_advantage, dtype=np.float32)
    ready = np.asarray(ready_mask, dtype=bool)
    labelled = np.asarray(labelled_transition, dtype=bool)
    if advantage.shape != ready.shape or advantage.shape != labelled.shape:
        raise ValueError(
            "graph advantage, ready mask, and labelled mask must share shape"
        )
    missing = labelled & ~ready
    if np.any(missing):
        raise RuntimeError(
            "capture outcome cohort references an unwritten graph advantage"
        )
    if np.any(~np.isfinite(advantage[labelled])):
        raise FloatingPointError(
            "capture outcome cohort has non-finite graph confidence input"
        )
    return np.where(ready, advantage, 0.0).astype(np.float32, copy=False)


def build_previous_adjacency_sequence(adjacency_sequence):
    """Shift ``[time, episode, agent, factor]`` graphs within each episode."""
    adjacency = np.asarray(adjacency_sequence)
    if adjacency.ndim != 4:
        raise ValueError(
            "adjacency_sequence must be 4-D [time, episode, agent, factor]"
        )
    previous = np.zeros_like(adjacency)
    if adjacency.shape[0] > 1:
        previous[1:] = adjacency[:-1]
    return previous


def build_previous_done_sequence(done_sequence):
    """Shift post-action done flags to action-state validity masks.

    ``done_sequence[t, e]`` describes the outcome *after* action ``t``. The
    graph/action at ``t`` is therefore valid even when that value is true; only
    transition ``t + 1`` must be masked. The zero prefix also prevents one
    episode/environment from inheriting another one's terminal flag.
    """
    dones = np.asarray(done_sequence)
    if dones.ndim < 2:
        raise ValueError(
            "done_sequence must have [time, episode, ...] axes"
        )
    previous = np.zeros_like(dones)
    if dones.shape[0] > 1:
        previous[1:] = dones[:-1]
    return previous


def select_outcome_contrast_complete_episodes(
        base_episode_indices,
        positive_episode_mask,
        negative_episode_mask,
        recency_order,
        positive_eligible_mask=None,
        negative_eligible_mask=None):
    """Complete a replay cohort with real missing outcome classes.

    The centered capture-outcome baseline is defined over completed capture
    episodes.  Sampling only one recent episode can otherwise turn that
    centered population into a one-sided optimizer batch.  When (and only
    when) both signed-credit classes exist in the occupied buffer, append the
    newest episode of each class missing from ``base_episode_indices``.

    No label is synthesized and no episode is duplicated.  Optional eligible
    masks allow the replay buffer to prevent one old episode from being used
    as supplemental support in every later optimizer update. ``recency_order``
    must contain every occupied replay slot once, newest first.
    """
    base = np.asarray(base_episode_indices, dtype=np.int64).reshape(-1)
    positive = np.asarray(positive_episode_mask, dtype=bool).reshape(-1)
    negative = np.asarray(negative_episode_mask, dtype=bool).reshape(-1)
    recency = np.asarray(recency_order, dtype=np.int64).reshape(-1)
    positive_eligible = (
        positive.copy()
        if positive_eligible_mask is None
        else np.asarray(positive_eligible_mask, dtype=bool).reshape(-1)
    )
    negative_eligible = (
        negative.copy()
        if negative_eligible_mask is None
        else np.asarray(negative_eligible_mask, dtype=bool).reshape(-1)
    )

    if positive.shape != negative.shape:
        raise ValueError("positive and negative episode masks must have equal shape")
    if positive_eligible.shape != positive.shape or negative_eligible.shape != negative.shape:
        raise ValueError("outcome eligibility masks must match class-mask shape")
    if np.any(positive_eligible & ~positive):
        raise ValueError("positive eligibility includes a non-positive episode")
    if np.any(negative_eligible & ~negative):
        raise ValueError("negative eligibility includes a non-negative episode")
    if np.any(positive & negative):
        raise ValueError(
            "one episode cannot carry both signed outcome classes"
        )
    if np.unique(base).size != base.size:
        raise ValueError("base episode indices must be unique")
    if np.unique(recency).size != recency.size:
        raise ValueError("recency_order must contain unique episode indices")
    if recency.size and (
            np.min(recency) < 0 or np.max(recency) >= positive.size):
        raise IndexError("recency_order references an invalid replay slot")
    if base.size and (np.min(base) < 0 or np.max(base) >= positive.size):
        raise IndexError("base episode indices reference an invalid replay slot")

    occupied = np.zeros_like(positive, dtype=bool)
    occupied[recency] = True
    if np.any((positive | negative) & ~occupied):
        raise ValueError("outcome class masks include an unoccupied replay slot")

    positive_available = bool(np.any(positive & occupied))
    negative_available = bool(np.any(negative & occupied))
    selected = list(base.tolist())
    supplemented = []

    if positive_available and negative_available:
        selected_set = set(selected)
        has_positive = bool(base.size and np.any(positive[base]))
        has_negative = bool(base.size and np.any(negative[base]))
        pending_supplements = []
        support_missing = False
        for class_mask, eligible_mask, class_present in (
                (positive, positive_eligible, has_positive),
                (negative, negative_eligible, has_negative)):
            if class_present:
                continue
            supplement = next(
                (int(idx) for idx in recency
                 if class_mask[idx]
                 and eligible_mask[idx]
                 and int(idx) not in selected_set),
                None,
            )
            if supplement is None:
                support_missing = True
                break
            pending_supplements.append(supplement)
            selected_set.add(supplement)
        # Cohort completion is atomic. Do not consume the still-available sign
        # when another missing sign is exhausted, because no outcome update can
        # be performed from that one-sided cohort.
        if not support_missing:
            selected.extend(pending_supplements)
            supplemented.extend(pending_supplements)

    result = np.asarray(selected, dtype=np.int64)
    result_has_positive = bool(result.size and np.any(positive[result]))
    result_has_negative = bool(result.size and np.any(negative[result]))
    contrast_available = positive_available and negative_available
    # ``class_complete`` means that the optimizer cohort really contains both
    # outcome classes.  A no-capture or single-class population is protected
    # (credit disabled), but must not be logged as a complete contrast.
    class_complete = bool(
        contrast_available
        and result_has_positive
        and result_has_negative
    )
    supplemented = np.asarray(supplemented, dtype=np.int64)
    return {
        "episode_indices": result,
        "supplemented_episode_indices": supplemented,
        "augmented_count": int(result.size - base.size),
        "positive_available": float(positive_available),
        "negative_available": float(negative_available),
        "positive_available_count": int(np.sum(positive & occupied)),
        "negative_available_count": int(np.sum(negative & occupied)),
        "base_positive_count": int(np.sum(positive[base])),
        "base_negative_count": int(np.sum(negative[base])),
        "augmented_positive_count": int(np.sum(positive[supplemented])),
        "augmented_negative_count": int(np.sum(negative[supplemented])),
        "positive_selected_count": int(np.sum(positive[result])),
        "negative_selected_count": int(np.sum(negative[result])),
        "class_complete": float(class_complete),
        "support_exhausted": float(contrast_available and not class_complete),
        "outcome_credit_enabled": float(contrast_available and class_complete),
    }


def resolve_graph_sampling_mode(
        action_explore,
        training_episode,
        warmup,
        use_train_consistent_eval_graph):
    """Return ``(graph_explore, use_eval_rng)`` for one rollout.

    Policy actions remain controlled by ``action_explore``.  When
    train-consistent evaluation is enabled, only the graph follows the
    current training behavior distribution; a dedicated RNG keeps evaluation
    from perturbing subsequent training topology samples.
    """
    if bool(training_episode) or bool(warmup):
        return bool(action_explore), False
    if bool(use_train_consistent_eval_graph):
        return True, True
    return False, False
