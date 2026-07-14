"""Capture-anchored pair-to-triplet credit for dynamic factor graphs.

This module is deliberately NumPy-only so the structural credit rule can be
tested without importing the training stack or PyTorch.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


# Replay diagnostic columns. AdjBuffer writes exactly one marked row per
# completed episode so mini-batch logging remains episode-weighted rather than
# episode-length-weighted.
CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH = 30


def canonical_capture_factor_catalog(num_agents, highest_order=3):
    """Return the fixed pair-then-triplet identity catalog used by SDDFG."""
    num_agents = int(num_agents)
    highest_order = int(highest_order)
    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    catalog = list(combinations(range(num_agents), 2))
    if highest_order >= 3:
        catalog.extend(combinations(range(num_agents), 3))
    return tuple(tuple(int(node) for node in factor) for factor in catalog)


def _transition_matrix(values, name, shape, dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(name, shape, array.shape)
        )
    return array


def build_capture_identity_factor_weights(
        current_adj,
        capture_events_by_episode,
        expected_capture_counts=None):
    """Match real capture participants to selected representable factors.

    Let ``P_c`` be the exact participant set of capture event ``c`` and let
    ``d_c = min(len(P_c), 3)`` be the highest factor order represented by this
    graph implementation. Eligible identities satisfy ``|nodes(f)| = d_c``
    and ``nodes(f) subset P_c``. Thus a two-wolf capture is attributed only to
    its exact pair (never to an invented triplet), a three-wolf capture to its
    exact triplet, and a larger capture to genuine participant-only triplet
    substructures.

    Every matched event contributes total mass one. Candidate factor slots are
    canonicalized by their node set before this mass is divided, so duplicate
    slots cannot multiply an event's supervision. If one canonical identity is
    present in multiple slots, that identity's share is divided across those
    slots as well. Downstream episode normalization can therefore preserve both
    event and episode mass without depending on factor duplication.

    ``current_adj`` has shape ``[episode, agent, factor]`` (or ``[agent,
    factor]`` for one episode). ``capture_events_by_episode`` is a sequence of
    event lists. Every event must contain a unique ``event_id``, ``target_id``
    and the exact ``participant_slots`` emitted by the environment.
    """
    adj = np.asarray(current_adj)
    squeeze_episode = False
    if adj.ndim == 2:
        adj = adj[None, ...]
        squeeze_episode = True
    if adj.ndim != 3:
        raise ValueError(
            "current_adj must have shape [episode, agent, factor], got {}"
            .format(adj.shape)
        )
    num_episodes, num_agents, num_factors = adj.shape
    if not isinstance(capture_events_by_episode, (list, tuple)):
        raise TypeError("capture_events_by_episode must be a list or tuple")
    if squeeze_episode and (
            len(capture_events_by_episode) == 0
            or isinstance(capture_events_by_episode[0], dict)):
        capture_events_by_episode = [capture_events_by_episode]
    if len(capture_events_by_episode) != num_episodes:
        raise ValueError(
            "capture event episode axis must be {}, got {}".format(
                num_episodes, len(capture_events_by_episode)
            )
        )
    if expected_capture_counts is not None:
        expected = np.asarray(expected_capture_counts, dtype=np.float32)
        expected = expected.reshape(-1)
        if expected.shape != (num_episodes,):
            raise ValueError(
                "expected_capture_counts must have shape {}, got {}".format(
                    (num_episodes,), expected.shape
                )
            )
        if not np.isfinite(expected).all() or np.any(expected < 0.0):
            raise ValueError(
                "expected_capture_counts must be finite and non-negative"
            )
    else:
        expected = None

    factor_weights = np.zeros(
        (num_episodes, num_factors), dtype=np.float32
    )
    matched_event_count = np.zeros(num_episodes, dtype=np.float32)
    unmatched_event_count = np.zeros(num_episodes, dtype=np.float32)
    candidate_factor_count = np.zeros(num_episodes, dtype=np.float32)
    raw_candidate_factor_count = np.zeros(num_episodes, dtype=np.float32)
    duplicate_candidate_factor_count = np.zeros(num_episodes, dtype=np.float32)
    candidate_only_event_count = np.zeros(num_episodes, dtype=np.float32)
    participant_count = np.zeros(num_episodes, dtype=np.float32)
    matched_factor_order_sum = np.zeros(num_episodes, dtype=np.float32)
    canonical_catalog = canonical_capture_factor_catalog(num_agents, 3)
    canonical_index = {
        frozenset(nodes): candidate
        for candidate, nodes in enumerate(canonical_catalog)
    }
    # This tensor is deliberately separate from selected factor slots.  It
    # marks only exact real-event identities which were representable in the
    # fixed candidate catalog but absent from the active graph.  Downstream it
    # may supervise candidate scoring, never active-factor PPO.
    candidate_only_factor_weights = np.zeros(
        (num_episodes, len(canonical_catalog)), dtype=np.float32
    )

    factor_nodes = []
    for episode in range(num_episodes):
        episode_nodes = []
        for factor in range(num_factors):
            nodes = frozenset(
                np.flatnonzero(adj[episode, :, factor] > 0).tolist()
            )
            episode_nodes.append(nodes)
        factor_nodes.append(episode_nodes)

    for episode, events in enumerate(capture_events_by_episode):
        if events is None:
            raise RuntimeError(
                "capture identity is missing for episode {}".format(episode)
            )
        if not isinstance(events, (list, tuple)):
            raise TypeError("each episode capture event collection must be a list")
        if expected is not None and len(events) != int(round(expected[episode])):
            raise RuntimeError(
                "capture_count/event identity mismatch for episode {}: "
                "count={}, events={}".format(
                    episode, expected[episode], len(events)
                )
            )
        seen_event_ids = set()
        for event in events:
            if not isinstance(event, dict):
                raise TypeError("capture event must be a dict")
            if "event_id" not in event or "target_id" not in event:
                raise RuntimeError(
                    "capture event requires event_id and target_id"
                )
            event_id = int(event["event_id"])
            if event_id in seen_event_ids:
                raise RuntimeError(
                    "duplicate capture event_id {} in one transition".format(
                        event_id
                    )
                )
            seen_event_ids.add(event_id)
            participants = tuple(
                sorted(int(slot) for slot in event.get("participant_slots", []))
            )
            if len(participants) < 2 or len(set(participants)) != len(participants):
                raise RuntimeError(
                    "capture participants must contain at least two unique slots"
                )
            if participants[0] < 0 or participants[-1] >= num_agents:
                raise RuntimeError(
                    "capture participant slot is outside [0, {})".format(
                        num_agents
                    )
                )
            participant_set = frozenset(participants)
            participant_count[episode] += float(len(participant_set))
            attributable_order = min(len(participant_set), 3)
            event_candidate_identities = tuple(
                frozenset(nodes)
                for nodes in combinations(participants, attributable_order)
            )
            event_candidate_indices = []
            for identity in event_candidate_identities:
                candidate_index = canonical_index.get(identity)
                if candidate_index is None:
                    raise RuntimeError(
                        "capture identity is absent from the canonical "
                        "candidate catalog: {}".format(sorted(identity))
                    )
                event_candidate_indices.append(int(candidate_index))
            candidates_by_identity = {}
            for factor, nodes in enumerate(factor_nodes[episode]):
                if (
                        len(nodes) == attributable_order
                        and nodes.issubset(participant_set)):
                    candidates_by_identity.setdefault(nodes, []).append(factor)
            raw_candidate_count = sum(
                len(slots) for slots in candidates_by_identity.values()
            )
            canonical_candidate_count = len(candidates_by_identity)
            raw_candidate_factor_count[episode] += float(raw_candidate_count)
            candidate_factor_count[episode] += float(canonical_candidate_count)
            duplicate_candidate_factor_count[episode] += float(
                raw_candidate_count - canonical_candidate_count
            )
            if not candidates_by_identity:
                # Adj_Generator enumerates every canonical pair/triplet over
                # active slots. A valid, exactly representable capture with no
                # selected identity factor is therefore candidate-only.
                if len(event_candidate_indices) == 0:
                    raise RuntimeError(
                        "representable capture has no canonical candidate"
                    )
                candidate_mass_before = float(
                    candidate_only_factor_weights[episode].sum()
                )
                candidate_weight = 1.0 / float(len(event_candidate_indices))
                for candidate_index in event_candidate_indices:
                    candidate_only_factor_weights[
                        episode, candidate_index
                    ] += candidate_weight
                candidate_mass_added = float(
                    candidate_only_factor_weights[episode].sum()
                    - candidate_mass_before
                )
                if not np.isclose(candidate_mass_added, 1.0, atol=1e-6):
                    raise RuntimeError(
                        "candidate-only capture weights do not conserve unit "
                        "event mass: event_id={}, mass={}".format(
                            event_id, candidate_mass_added
                        )
                    )
                candidate_only_event_count[episode] += 1.0
                unmatched_event_count[episode] += 1.0
                continue
            identity_weight = 1.0 / float(canonical_candidate_count)
            event_mass_before = float(factor_weights[episode].sum())
            for factor_slots in candidates_by_identity.values():
                slot_weight = identity_weight / float(len(factor_slots))
                for factor in factor_slots:
                    factor_weights[episode, factor] += slot_weight
            event_mass_added = (
                float(factor_weights[episode].sum()) - event_mass_before
            )
            if not np.isclose(event_mass_added, 1.0, atol=1e-6):
                raise RuntimeError(
                    "capture event factor weights do not conserve unit mass: "
                    "event_id={}, mass={}".format(event_id, event_mass_added)
                )
            matched_event_count[episode] += 1.0
            matched_factor_order_sum[episode] += float(attributable_order)

    matched_event_weight_sum = factor_weights.sum(axis=1).astype(
        np.float32, copy=False
    )
    candidate_only_event_weight_sum = candidate_only_factor_weights.sum(
        axis=1
    ).astype(np.float32, copy=False)
    event_mass_error = np.abs(
        matched_event_weight_sum - matched_event_count
    ).astype(np.float32, copy=False)
    candidate_only_event_mass_error = np.abs(
        candidate_only_event_weight_sum - candidate_only_event_count
    ).astype(np.float32, copy=False)
    combined_identity_event_mass_error = np.abs(
        matched_event_weight_sum
        + candidate_only_event_weight_sum
        - matched_event_count
        - candidate_only_event_count
    ).astype(np.float32, copy=False)
    if np.any(candidate_only_event_mass_error > 1e-6):
        raise RuntimeError(
            "candidate-only capture identity weights do not conserve event mass"
        )
    if np.any(combined_identity_event_mass_error > 1e-6):
        raise RuntimeError(
            "active/candidate-only capture identity weights do not conserve "
            "combined event mass"
        )
    if not np.array_equal(unmatched_event_count, candidate_only_event_count):
        raise RuntimeError(
            "capture identity unmatched-reason partition is inconsistent"
        )
    result = {
        "factor_weights": factor_weights,
        "matched_event_count": matched_event_count,
        "unmatched_event_count": unmatched_event_count,
        "candidate_factor_count": candidate_factor_count,
        "raw_candidate_factor_count": raw_candidate_factor_count,
        "duplicate_candidate_factor_count": duplicate_candidate_factor_count,
        "candidate_only_event_count": candidate_only_event_count,
        "candidate_only_factor_weights": candidate_only_factor_weights,
        "participant_count": participant_count,
        "matched_factor_order_sum": matched_factor_order_sum,
        "matched_event_weight_sum": matched_event_weight_sum,
        "candidate_only_event_weight_sum": candidate_only_event_weight_sum,
        "event_mass_error": event_mass_error,
        "candidate_only_event_mass_error": candidate_only_event_mass_error,
        "combined_identity_event_mass_error": (
            combined_identity_event_mass_error
        ),
    }
    if squeeze_episode:
        return {key: value[0] for key, value in result.items()}
    return result


def compute_capture_to_win_outcome_gate(
        success_now,
        capture_counts,
        valid_graph_transition):
    """Return an episode-outcome gate anchored to real environment success.

    ``success_now`` is the environment event emitted by the rollout runner.  A
    capture is attributed to a successful outcome only when that same episode
    contains a real success event.  Shaped rewards and episode return are
    intentionally absent from this definition.
    """
    valid_graph_transition = np.asarray(
        valid_graph_transition, dtype=bool
    )
    if valid_graph_transition.ndim != 2:
        raise ValueError(
            "valid_graph_transition must have shape [time, episode], got {}"
            .format(valid_graph_transition.shape)
        )
    shape = valid_graph_transition.shape
    success_now = _transition_matrix(
        success_now,
        "success_now",
        shape,
    )
    capture_counts = _transition_matrix(
        capture_counts,
        "capture_counts",
        shape,
    )
    success_event = (
        np.isfinite(success_now)
        & (success_now > 0.0)
        & valid_graph_transition
    )
    valid_episode = valid_graph_transition.any(axis=0)
    episode_success = success_event.any(axis=0) & valid_episode
    episode_success_gate = (
        valid_graph_transition.astype(np.float32)
        * episode_success[None, :].astype(np.float32)
    )
    safe_capture_counts = np.where(
        np.isfinite(capture_counts),
        np.maximum(capture_counts, 0.0),
        0.0,
    ).astype(np.float32, copy=False)
    failed_episode_capture_count = (
        safe_capture_counts
        * (~episode_success)[None, :].astype(np.float32)
    ).astype(np.float32, copy=False)
    return {
        "episode_success": episode_success,
        "episode_success_gate": episode_success_gate,
        "failed_episode_capture_count": failed_episode_capture_count,
    }


def compute_capture_to_win_triplet_gate(
        episode_success,
        triplet_capture_quality):
    """Select real capture triplets from episodes with a real win event."""
    quality = np.asarray(triplet_capture_quality, dtype=np.float32)
    if quality.ndim != 3:
        raise ValueError(
            "triplet_capture_quality must have shape [time, episode, factor], "
            "got {}".format(quality.shape)
        )
    episode_success = np.asarray(episode_success, dtype=bool)
    if episode_success.shape != (quality.shape[1],):
        raise ValueError(
            "episode_success must have shape {}, got {}".format(
                (quality.shape[1],), episode_success.shape
            )
        )
    quality = np.where(
        np.isfinite(quality),
        np.maximum(quality, 0.0),
        0.0,
    ).astype(np.float32, copy=False)
    return (
        quality * episode_success[None, :, None].astype(np.float32)
    ).astype(np.float32, copy=False)


def compute_capture_to_win_triplet_outcome_advantage(
        episode_success,
        triplet_capture_quality):
    """Build a centered capture-outcome advantage across capture episodes.

    Positive-only outcome gates cannot distinguish a failed capture from an
    unobserved outcome: both receive zero.  This helper uses capture episodes
    in the current adjacency buffer as the comparison set.  Successful
    capture episodes receive ``1 - success_rate`` and failed capture episodes
    receive ``-success_rate``.  The signal is therefore zero-mean by episode,
    and remains exactly zero until both outcome classes have been observed.

    The returned tensor is nonzero only at real capture triplets.  No reward,
    return, pair transition, padding step, or non-capture factor can activate
    it.
    """
    quality = np.asarray(triplet_capture_quality, dtype=np.float32)
    if quality.ndim != 3:
        raise ValueError(
            "triplet_capture_quality must have shape [time, episode, factor], "
            "got {}".format(quality.shape)
        )
    episode_success = np.asarray(episode_success, dtype=bool)
    if episode_success.shape != (quality.shape[1],):
        raise ValueError(
            "episode_success must have shape {}, got {}".format(
                (quality.shape[1],), episode_success.shape
            )
        )
    quality = np.where(
        np.isfinite(quality),
        np.maximum(quality, 0.0),
        0.0,
    ).astype(np.float32, copy=False)
    capture_episode = quality.sum(axis=(0, 2)) > 0.0
    capture_episode_count = int(capture_episode.sum())
    if capture_episode_count == 0:
        success_rate = 0.0
    else:
        success_rate = float(
            episode_success[capture_episode].astype(np.float32).mean()
        )
    outcome_advantage = np.zeros(
        episode_success.shape,
        dtype=np.float32,
    )
    outcome_advantage[capture_episode] = (
        episode_success[capture_episode].astype(np.float32)
        - np.float32(success_rate)
    )
    capture_triplet_count = quality.sum(axis=(0, 2)).astype(
        np.float32,
        copy=False,
    )
    # One episode contributes one outcome label.  Without this normalization,
    # the same label is copied once per capture and once per selected triplet
    # at that capture step.  Successful Wolfpack episodes naturally contain
    # more captures, so the expanded factor tensor would no longer be centered
    # even though the episode-level outcome advantage is exactly zero-mean.
    normalized_quality = np.zeros_like(quality, dtype=np.float32)
    nonzero_capture_count = capture_triplet_count > 0.0
    if np.any(nonzero_capture_count):
        normalized_quality[:, nonzero_capture_count, :] = (
            quality[:, nonzero_capture_count, :]
            / capture_triplet_count[
                nonzero_capture_count
            ][None, :, None]
        )
    triplet_outcome_advantage = (
        normalized_quality * outcome_advantage[None, :, None]
    ).astype(np.float32, copy=False)
    return {
        "triplet_outcome_advantage": triplet_outcome_advantage,
        "capture_episode_mask": capture_episode,
        "capture_episode_success_rate": np.float32(success_rate),
        "capture_episode_count": np.int64(capture_episode_count),
        "successful_capture_episode_count": np.int64(
            (capture_episode & episode_success).sum()
        ),
        "failed_capture_episode_count": np.int64(
            (capture_episode & ~episode_success).sum()
        ),
        "episode_outcome_advantage": outcome_advantage,
        "capture_triplet_count_per_episode": capture_triplet_count,
    }


def scale_capture_to_win_outcome_credit(
        triplet_outcome_advantage,
        graph_advantage,
        valid_graph_transition,
        coefficient,
        cap,
        return_diagnostics=False):
    """Scale signed outcome evidence with sign-independent graph confidence.

    The episode outcome is the only source of the credit sign.  Graph
    advantage contributes detached reliability through its magnitude.  Using
    only ``max(graph_advantage, 0)`` is not a reliability estimate: it silently
    drops every correctly labelled success *and* failure whose capture graph
    happens to have non-positive advantage.  That produced class-complete,
    centered optimizer cohorts with zero training credit in run62.
    """
    outcome = np.asarray(triplet_outcome_advantage, dtype=np.float32)
    graph = np.asarray(graph_advantage, dtype=np.float32)
    valid = np.asarray(valid_graph_transition, dtype=bool)
    if outcome.ndim != 3:
        raise ValueError(
            "triplet_outcome_advantage must have shape [time, episode, factor], "
            "got {}".format(outcome.shape)
        )
    if graph.shape != outcome.shape[:2]:
        raise ValueError(
            "graph_advantage must have shape {}, got {}".format(
                outcome.shape[:2], graph.shape
            )
        )
    if valid.shape != graph.shape:
        raise ValueError(
            "valid_graph_transition must have shape {}, got {}".format(
                graph.shape, valid.shape
            )
        )
    coefficient = float(coefficient)
    cap = float(cap)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError("cap must be finite and non-negative")
    safe_outcome = np.where(np.isfinite(outcome), outcome, 0.0)
    safe_graph = np.where(np.isfinite(graph), graph, 0.0)
    safe_outcome = np.where(valid[:, :, None], safe_outcome, 0.0)
    graph_confidence = np.abs(safe_graph).astype(np.float32, copy=False)
    preclip_credit = (
        safe_outcome
        * graph_confidence[:, :, None]
        * coefficient
    ).astype(np.float32, copy=False)
    credit = preclip_credit.copy()
    credit_cap = np.inf
    if cap > 0.0:
        valid_abs_graph = np.abs(safe_graph[valid])
        graph_abs_scale = (
            float(valid_abs_graph.mean())
            if valid_abs_graph.size > 0
            else 1.0
        )
        if not np.isfinite(graph_abs_scale) or graph_abs_scale < 1e-6:
            graph_abs_scale = 1.0
        credit_cap = graph_abs_scale * cap
        credit = np.clip(credit, -credit_cap, credit_cap)
    credit = credit.astype(np.float32, copy=False)
    if not return_diagnostics:
        return credit
    labelled_outcome = np.abs(safe_outcome) > 0.0
    positive_outcome = safe_outcome > 0.0
    negative_outcome = safe_outcome < 0.0
    broadcast_confidence = np.broadcast_to(
        graph_confidence[:, :, None], safe_outcome.shape
    )
    broadcast_graph = np.broadcast_to(
        safe_graph[:, :, None], safe_outcome.shape
    )
    nonzero_preclip = np.abs(preclip_credit) > 0.0
    positive_preclip = preclip_credit > 0.0
    negative_preclip = preclip_credit < 0.0
    if np.isfinite(credit_cap):
        positive_clipped = preclip_credit > credit_cap
        negative_clipped = preclip_credit < -credit_cap
    else:
        positive_clipped = np.zeros_like(positive_preclip, dtype=bool)
        negative_clipped = np.zeros_like(negative_preclip, dtype=bool)
    valid_preclip_values = preclip_credit[nonzero_preclip]

    def _masked_mean(values, mask):
        selected = np.asarray(values)[mask]
        return np.float32(selected.mean() if selected.size > 0 else 0.0)

    def _masked_fraction(mask, denominator_mask):
        denominator = int(np.sum(denominator_mask))
        return np.float32(
            int(np.sum(mask & denominator_mask)) / float(max(denominator, 1))
        )

    def _masked_percentile(values, mask, percentile):
        selected = np.asarray(values)[mask]
        return np.float32(
            np.percentile(selected, percentile) if selected.size > 0 else 0.0
        )

    zero_confidence = broadcast_confidence <= 1e-12
    zero_credit = np.abs(credit) <= 0.0
    return {
        "credit": credit,
        "graph_confidence_mean": _masked_mean(
            broadcast_confidence, labelled_outcome
        ),
        "graph_confidence_std": np.float32(
            broadcast_confidence[labelled_outcome].std()
            if np.any(labelled_outcome) else 0.0
        ),
        "graph_confidence_p50": _masked_percentile(
            broadcast_confidence, labelled_outcome, 50.0
        ),
        "graph_confidence_p95": _masked_percentile(
            broadcast_confidence, labelled_outcome, 95.0
        ),
        "graph_confidence_max": np.float32(
            broadcast_confidence[labelled_outcome].max()
            if np.any(labelled_outcome) else 0.0
        ),
        "positive_graph_confidence_mean": _masked_mean(
            broadcast_confidence, positive_outcome
        ),
        "positive_graph_confidence_max": np.float32(
            broadcast_confidence[positive_outcome].max()
            if np.any(positive_outcome) else 0.0
        ),
        "negative_graph_confidence_mean": _masked_mean(
            broadcast_confidence, negative_outcome
        ),
        "negative_graph_confidence_max": np.float32(
            broadcast_confidence[negative_outcome].max()
            if np.any(negative_outcome) else 0.0
        ),
        "labelled_graph_advantage_positive_fraction": _masked_fraction(
            broadcast_graph > 0.0, labelled_outcome
        ),
        "labelled_graph_advantage_negative_fraction": _masked_fraction(
            broadcast_graph < 0.0, labelled_outcome
        ),
        "labelled_graph_advantage_zero_fraction": _masked_fraction(
            broadcast_graph == 0.0, labelled_outcome
        ),
        "positive_zero_confidence_fraction": _masked_fraction(
            zero_confidence, positive_outcome
        ),
        "negative_zero_confidence_fraction": _masked_fraction(
            zero_confidence, negative_outcome
        ),
        "gate_to_credit_drop_fraction": _masked_fraction(
            zero_credit, labelled_outcome
        ),
        "preclip_positive_mass": np.float32(
            np.maximum(preclip_credit, 0.0).sum()
        ),
        "preclip_negative_mass": np.float32(
            np.maximum(-preclip_credit, 0.0).sum()
        ),
        "postclip_positive_mass": np.float32(
            np.maximum(credit, 0.0).sum()
        ),
        "postclip_negative_mass": np.float32(
            np.maximum(-credit, 0.0).sum()
        ),
        "preclip_mean": np.float32(
            valid_preclip_values.mean()
            if valid_preclip_values.size > 0 else 0.0
        ),
        "preclip_std": np.float32(
            valid_preclip_values.std()
            if valid_preclip_values.size > 0 else 0.0
        ),
        "preclip_max": np.float32(
            valid_preclip_values.max()
            if valid_preclip_values.size > 0 else 0.0
        ),
        "preclip_min": np.float32(
            valid_preclip_values.min()
            if valid_preclip_values.size > 0 else 0.0
        ),
        "positive_clip_fraction": np.float32(
            positive_clipped.sum() / max(int(positive_preclip.sum()), 1)
        ),
        "negative_clip_fraction": np.float32(
            negative_clipped.sum() / max(int(negative_preclip.sum()), 1)
        ),
    }


def compute_capture_anchored_pair_credit(
        current_adj,
        valid_factor,
        factor_size,
        capture_counts,
        capture_factor_match,
        valid_graph_transition,
        dones_env,
        team_rewards,
        window,
        gamma):
    """Build strictly-future pair-to-identity-capture-factor evidence.

    The current factor must be a pair-only backbone. A future real capture
    factor must contain the same two agents and must occur at ``offset > 0``.
    For the Wolfpack two-participant event this future factor is the exact
    pair; for a three-or-more-participant event it can be a genuine triplet
    containing that pair. Ordinary positive reward is diagnostic only and is
    never a transition-credit condition.

    A pair factor at ``t`` is eligible only when no triplet selected at the
    same step contains both pair members (a pair-only backbone).  It receives
    transition evidence only from the nearest step ``t + delay``, with
    ``delay >= 1``, that has a real capture event and a selected identity
    capture factor containing the same two agents. Positive rewards never
    activate credit.

    ``pair_pursuit_quality`` is an independent structural diagnostic: the
    consecutive duration of the same pair-only backbone, normalized by the
    configured search window.  It is not copied from, and does not gate, the
    pair-to-capture-factor transition score (legacy field name retained).
    """
    adj = np.asarray(current_adj)
    if adj.ndim != 4:
        raise ValueError(
            "current_adj must have shape [time, episode, agent, factor], "
            "got {}".format(adj.shape)
        )
    time_steps, num_episodes, _num_agents, num_factors = adj.shape
    factor_shape = (time_steps, num_episodes, num_factors)
    transition_shape = (time_steps, num_episodes)

    valid_factor = np.asarray(valid_factor, dtype=bool)
    factor_size = np.asarray(factor_size)
    if valid_factor.shape != factor_shape:
        raise ValueError(
            "valid_factor must have shape {}, got {}".format(
                factor_shape, valid_factor.shape
            )
        )
    if factor_size.shape != factor_shape:
        raise ValueError(
            "factor_size must have shape {}, got {}".format(
                factor_shape, factor_size.shape
            )
        )

    capture_counts = _transition_matrix(
        capture_counts,
        "capture_counts",
        transition_shape,
    )
    capture_counts = np.where(
        np.isfinite(capture_counts),
        np.maximum(capture_counts, 0.0),
        0.0,
    ).astype(np.float32, copy=False)
    capture_factor_match = np.asarray(
        capture_factor_match, dtype=np.float32
    )
    if capture_factor_match.shape != factor_shape:
        raise ValueError(
            "capture_factor_match must have shape {}, got {}".format(
                factor_shape, capture_factor_match.shape
            )
        )
    if not np.isfinite(capture_factor_match).all():
        raise ValueError("capture_factor_match must be finite")
    capture_factor_match = np.maximum(
        capture_factor_match, 0.0
    ).astype(np.float32, copy=False)
    valid_graph_transition = _transition_matrix(
        valid_graph_transition,
        "valid_graph_transition",
        transition_shape,
        dtype=bool,
    ).astype(bool, copy=False)
    dones_env = _transition_matrix(
        dones_env,
        "dones_env",
        transition_shape,
    )
    team_rewards = _transition_matrix(
        team_rewards,
        "team_rewards",
        transition_shape,
    )

    window = max(1, int(window))
    gamma = float(gamma)
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    gamma = min(1.0, max(0.0, gamma))

    pair_mask = valid_factor & (factor_size == 2)
    triplet_mask = valid_factor & (factor_size == 3)
    factor_nodes = [
        [dict() for _ in range(num_episodes)]
        for _ in range(time_steps)
    ]
    for step in range(time_steps):
        for episode in range(num_episodes):
            for factor in np.flatnonzero(valid_factor[step, episode]):
                nodes = frozenset(
                    np.flatnonzero(adj[step, episode, :, factor] > 0).tolist()
                )
                if len(nodes) in (2, 3):
                    factor_nodes[step][episode][int(factor)] = nodes

    pair_only_mask = np.zeros(factor_shape, dtype=bool)
    pair_pursuit_quality = np.zeros(factor_shape, dtype=np.float32)
    for step in range(time_steps):
        for episode in range(num_episodes):
            current_triplets = [
                factor_nodes[step][episode].get(int(factor), frozenset())
                for factor in np.flatnonzero(triplet_mask[step, episode])
            ]
            for factor in np.flatnonzero(pair_mask[step, episode]):
                pair_nodes = factor_nodes[step][episode].get(
                    int(factor), frozenset()
                )
                if len(pair_nodes) != 2:
                    continue
                if any(pair_nodes.issubset(nodes) for nodes in current_triplets):
                    continue
                pair_only_mask[step, episode, factor] = True

                streak = 1
                for previous_step in range(step - 1, -1, -1):
                    if streak >= window:
                        break
                    if dones_env[previous_step, episode] >= 0.5:
                        break
                    previous_match = False
                    for previous_factor in np.flatnonzero(
                            pair_only_mask[previous_step, episode]):
                        if factor_nodes[previous_step][episode].get(
                                int(previous_factor), frozenset()) == pair_nodes:
                            previous_match = True
                            break
                    if not previous_match:
                        break
                    streak += 1
                pair_pursuit_quality[step, episode, factor] = (
                    float(streak) / float(window)
                )

    capture_event = capture_counts > 0.0
    capture_factor_mask = pair_mask | triplet_mask
    invalid_identity_match = (
        (capture_factor_match > 0.0) & ~capture_factor_mask
    )
    if np.any(invalid_identity_match):
        first = np.argwhere(invalid_identity_match)[0]
        bad_step, bad_episode, bad_factor = (
            int(first[0]), int(first[1]), int(first[2])
        )
        bad_nodes = np.flatnonzero(
            adj[bad_step, bad_episode, :, bad_factor] > 0
        ).tolist()
        raise RuntimeError(
            "capture identity matched a non-pair/non-triplet or invalid "
            "factor: step={}, episode={}, factor={}, nodes={}, "
            "factor_size={}, valid_factor={}, match_weight={}, "
            "valid_graph_transition={}".format(
                bad_step,
                bad_episode,
                bad_factor,
                bad_nodes,
                factor_size[bad_step, bad_episode, bad_factor],
                bool(valid_factor[bad_step, bad_episode, bad_factor]),
                capture_factor_match[bad_step, bad_episode, bad_factor],
                bool(valid_graph_transition[bad_step, bad_episode]),
            )
        )
    capture_factor_quality = (
        capture_factor_match * capture_factor_mask.astype(np.float32)
    )
    transition_score = np.zeros(factor_shape, dtype=np.float32)
    transition_delay = np.zeros(factor_shape, dtype=np.float32)
    capture_matched_gate = np.zeros(transition_shape, dtype=np.float32)
    offset0_candidate_count = np.zeros(transition_shape, dtype=np.float32)

    for step in range(time_steps):
        for episode in range(num_episodes):
            if capture_event[step, episode]:
                current_capture_factors = [
                    factor_nodes[step][episode].get(
                        int(factor), frozenset()
                    )
                    for factor in np.flatnonzero(
                        capture_factor_quality[step, episode] > 0.0
                    )
                ]
                for factor in np.flatnonzero(pair_mask[step, episode]):
                    pair_nodes = factor_nodes[step][episode].get(
                        int(factor), frozenset()
                    )
                    if any(
                            pair_nodes.issubset(nodes)
                            for nodes in current_capture_factors):
                        offset0_candidate_count[step, episode] += 1.0

            for factor in np.flatnonzero(pair_only_mask[step, episode]):
                pair_nodes = factor_nodes[step][episode].get(
                    int(factor), frozenset()
                )
                for delay in range(1, window + 1):
                    future_step = step + delay
                    if future_step >= time_steps:
                        break
                    # dones_env[k] means the episode ended after transition k;
                    # therefore t+delay is unreachable when t+delay-1 is done.
                    if dones_env[future_step - 1, episode] >= 0.5:
                        break
                    if not capture_event[future_step, episode]:
                        continue
                    matching_capture_factor = False
                    for capture_factor in np.flatnonzero(
                            capture_factor_quality[
                                future_step, episode
                            ] > 0.0):
                        capture_nodes = factor_nodes[future_step][episode].get(
                            int(capture_factor), frozenset()
                        )
                        if pair_nodes.issubset(capture_nodes):
                            matching_capture_factor = True
                            break
                    if not matching_capture_factor:
                        continue

                    transition_score[step, episode, factor] = float(
                        gamma ** delay
                    )
                    transition_delay[step, episode, factor] = float(delay)
                    capture_matched_gate[future_step, episode] = 1.0
                    # The nearest valid transition is the only one credited.
                    break

    identity_matched_capture_count = np.minimum(
        capture_counts,
        capture_factor_quality.sum(axis=2),
    )
    capture_matched_count = (
        identity_matched_capture_count * capture_matched_gate
    ).astype(np.float32, copy=False)
    positive_reward_step = (
        (team_rewards > 0.0) & valid_graph_transition
    ).astype(np.float32)
    positive_reward_without_capture = (
        (positive_reward_step > 0.0) & ~capture_event
    ).astype(np.float32)

    return {
        "pair_only_mask": pair_only_mask,
        "pair_pursuit_quality": pair_pursuit_quality,
        "pair_to_triplet_transition_score": transition_score,
        "pair_transition_delay": transition_delay,
        # Keep the legacy key for batch/checkpoint compatibility. From
        # definition version 5 onward it means an exact identity-matched
        # capture factor and can be order 2 or 3.
        "triplet_capture_quality": capture_factor_quality,
        "capture_factor_quality": capture_factor_quality,
        "capture_matched_count": capture_matched_count,
        "positive_reward_step": positive_reward_step,
        "positive_reward_without_capture": positive_reward_without_capture,
        "offset0_candidate_count": offset0_candidate_count,
    }
