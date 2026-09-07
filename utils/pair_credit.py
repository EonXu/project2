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
    event lists. Every event must contain ``event_id``, ``target_id`` and the
    exact ``participant_slots`` emitted by the environment. This helper
    enforces uniqueness within the current transition; the rollout runner
    additionally enforces one stable event contract across the whole episode.
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
    # Event provenance is kept before the dense tensors intentionally lose
    # event identity through aggregation. Candidate-only rows remain
    # diagnostic/supervision provenance. Matched rows additionally make the
    # strict future-pair join capture/event/prey local; they do not change the
    # dense factor weights or any loss.
    candidate_only_event_records = [
        [] for _ in range(num_episodes)
    ]
    matched_event_records = [
        [] for _ in range(num_episodes)
    ]

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
                    candidate_identity = canonical_catalog[candidate_index]
                    candidate_only_event_records[episode].append({
                        "event_id": event_id,
                        "target_id": int(event["target_id"]),
                        "participant_slots": tuple(participants),
                        "candidate_index": int(candidate_index),
                        "candidate_identity": "-".join(
                            str(int(node)) for node in candidate_identity
                        ),
                        "candidate_order": int(len(candidate_identity)),
                        "identity_event_weight": float(candidate_weight),
                    })
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
            for identity, factor_slots in sorted(
                    candidates_by_identity.items(),
                    key=lambda item: tuple(sorted(item[0]))):
                slot_weight = identity_weight / float(len(factor_slots))
                for factor in factor_slots:
                    factor_weights[episode, factor] += slot_weight
                    identity_tuple = tuple(sorted(int(node) for node in identity))
                    matched_event_records[episode].append({
                        "event_id": event_id,
                        "target_id": int(event["target_id"]),
                        "participant_slots": tuple(participants),
                        "factor_index": int(factor),
                        "factor_identity": "-".join(
                            str(node) for node in identity_tuple
                        ),
                        "factor_order": int(len(identity_tuple)),
                        "identity_event_weight": float(identity_weight),
                        "factor_slot_weight": float(slot_weight),
                    })
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
        "candidate_only_event_records": tuple(
            tuple(records) for records in candidate_only_event_records
        ),
        "matched_event_records": tuple(
            tuple(records) for records in matched_event_records
        ),
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


def compute_outcome_conditioned_pair_credit(
        pair_transition_score,
        episode_success):
    """Turn strict-future pair-to-capture evidence into win-aligned credit.

    A real capture is only an intermediate event in Wolfpack.  The old pair
    objective assigned positive credit to every pair backbone that preceded a
    matching capture, including captures from episodes that ultimately failed.
    That objective conflicts with the centered capture-to-win objective.

    Let ``rho[t,e,f]`` be the non-negative, strict-future, exact-identity pair
    transition score and let ``C`` be episodes with non-zero pair evidence.
    This helper computes

        p = mean(y[e] for e in C)
        a[e] = y[e] - p
        rho_hat[t,e,f] = rho[t,e,f] / sum_{t,f} rho[t,e,f]
        credit[t,e,f] = rho_hat[t,e,f] * a[e]

    Therefore every evidence-bearing episode contributes exactly one centered
    outcome label regardless of episode length or the number of eligible pair
    transitions.  A single outcome class produces exactly zero credit, and no
    non-capture, offset-zero, wrong-identity, or padded transition can acquire
    mass here because this function cannot create support outside ``rho``.
    """
    score = np.asarray(pair_transition_score, dtype=np.float32)
    if score.ndim != 3:
        raise ValueError(
            "pair_transition_score must have shape [time, episode, factor], "
            "got {}".format(score.shape)
        )
    episode_success = np.asarray(episode_success, dtype=bool)
    if episode_success.shape != (score.shape[1],):
        raise ValueError(
            "episode_success must have shape {}, got {}".format(
                (score.shape[1],), episode_success.shape
            )
        )
    if not np.isfinite(score).all() or np.any(score < 0.0):
        raise ValueError(
            "pair_transition_score must be finite and non-negative"
        )

    episode_mass = score.sum(axis=(0, 2)).astype(np.float32, copy=False)
    evidence_episode = episode_mass > 0.0
    if np.any(evidence_episode):
        success_rate = float(
            episode_success[evidence_episode].astype(np.float32).mean()
        )
    else:
        success_rate = 0.0

    episode_outcome_advantage = np.zeros(
        episode_success.shape,
        dtype=np.float32,
    )
    episode_outcome_advantage[evidence_episode] = (
        episode_success[evidence_episode].astype(np.float32)
        - np.float32(success_rate)
    )
    normalized_score = np.zeros_like(score, dtype=np.float32)
    if np.any(evidence_episode):
        normalized_score[:, evidence_episode, :] = (
            score[:, evidence_episode, :]
            / episode_mass[evidence_episode][None, :, None]
        )
    signed_credit = (
        normalized_score
        * episode_outcome_advantage[None, :, None]
    ).astype(np.float32, copy=False)

    signed_episode_mass = signed_credit.sum(axis=(0, 2)).astype(
        np.float32,
        copy=False,
    )
    episode_mass_error = np.abs(
        signed_episode_mass - episode_outcome_advantage
    ).astype(np.float32, copy=False)
    centered_sum = float(signed_episode_mass[evidence_episode].sum())
    if np.any(episode_mass_error > 1e-6):
        raise RuntimeError(
            "outcome-conditioned pair credit does not conserve one centered "
            "outcome label per evidence episode"
        )
    if abs(centered_sum) > 1e-5:
        raise RuntimeError(
            "outcome-conditioned pair credit is not cohort-centered: {}"
            .format(centered_sum)
        )
    return {
        "credit": signed_credit,
        "normalized_pair_transition_score": normalized_score,
        "evidence_episode_mask": evidence_episode,
        "evidence_episode_count": np.int64(evidence_episode.sum()),
        "successful_evidence_episode_count": np.int64(
            (evidence_episode & episode_success).sum()
        ),
        "failed_evidence_episode_count": np.int64(
            (evidence_episode & ~episode_success).sum()
        ),
        "evidence_episode_success_rate": np.float32(success_rate),
        "episode_outcome_advantage": episode_outcome_advantage,
        "episode_pair_evidence_mass": episode_mass,
        "signed_episode_mass": signed_episode_mass,
        "episode_mass_error": episode_mass_error,
        "centered_sum": np.float32(centered_sum),
    }


def scale_optimizer_cohort_pair_credit(
        pair_transition_score,
        episode_success,
        coefficient,
        credit_cap):
    """Build signed pair credit for the episodes that actually train.

    Pair outcome contrast is an episode-cohort statistic.  A credit tensor
    centered over the full circular buffer cannot be sliced into an arbitrary
    replay cohort: the selected subset may contain only its positive or only
    its negative branch even though the stored full-buffer sum was zero.

    This helper therefore recomputes the outcome baseline on the final
    optimizer cohort, applies one common scale/cap, and verifies conservation
    after scaling.  A cohort whose *pair-evidence* episodes contain only one
    terminal outcome class receives exactly zero credit.
    """
    coefficient = float(coefficient)
    credit_cap = float(credit_cap)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("pair credit coefficient must be finite and non-negative")
    if not np.isfinite(credit_cap) or credit_cap < 0.0:
        raise ValueError("pair credit cap must be finite and non-negative")

    result = compute_outcome_conditioned_pair_credit(
        pair_transition_score=pair_transition_score,
        episode_success=episode_success,
    )
    credit = (
        result["credit"] * np.float32(coefficient)
    ).astype(np.float32, copy=False)
    preclip_positive_mass = float(np.maximum(credit, 0.0).sum())
    preclip_negative_mass = float(np.maximum(-credit, 0.0).sum())
    common_scale = 1.0
    max_abs_credit = float(np.abs(credit).max()) if credit.size else 0.0
    if credit_cap > 0.0 and max_abs_credit > credit_cap:
        common_scale = credit_cap / max_abs_credit
        credit = (credit * np.float32(common_scale)).astype(
            np.float32,
            copy=False,
        )

    centered_sum = float(credit.sum())
    mass_scale = max(1.0, float(np.abs(credit).sum()))
    if abs(centered_sum) > 1e-5 * mass_scale:
        raise RuntimeError(
            "optimizer-cohort pair credit lost signed mass conservation: {}"
            .format(centered_sum)
        )
    successful_count = int(result["successful_evidence_episode_count"])
    failed_count = int(result["failed_evidence_episode_count"])
    class_complete = successful_count > 0 and failed_count > 0
    if not class_complete and np.count_nonzero(credit) != 0:
        raise RuntimeError(
            "single-outcome optimizer pair cohort produced non-zero credit"
        )
    return {
        "credit": credit,
        "evidence_episode_count": int(result["evidence_episode_count"]),
        "successful_evidence_episode_count": successful_count,
        "failed_evidence_episode_count": failed_count,
        "class_complete": float(class_complete),
        "centered_sum": centered_sum,
        "preclip_positive_mass": preclip_positive_mass,
        "preclip_negative_mass": preclip_negative_mass,
        "postclip_positive_mass": float(np.maximum(credit, 0.0).sum()),
        "postclip_negative_mass": float(np.maximum(-credit, 0.0).sum()),
        "common_scale": float(common_scale),
    }


def partition_pair_contrast_optimizer_chunks(
        chunk_permutation,
        chunk_episode_membership,
        pair_evidence_episode_mask,
        episode_success,
        num_mini_batch,
        pair_partition_slot=0):
    """Keep a class-complete pair cohort and its base population together.

    Pair credit is centered across episodes, so splitting its positive and
    negative branches across separate Adam steps does not preserve the
    objective even when their sum over the whole adjacency update is zero. A
    second, subtler problem remains if only the pair-bearing chunks are kept
    atomic: every yielded partition computes its own masked mean and executes
    an independent Adam step. Equal chunk counts therefore do not imply equal
    valid-transition or valid-factor-mask populations, and sequential Adam
    steps are not equivalent to one population-total objective.

    Whenever both outcome classes exist, return one partition containing the
    complete selected replay population. Pair-bearing and zero-pair chunks then
    share one graph/base-factor normalization and one Adam transaction, while
    the identity-local pair loss still ignores zero-pair chunks. The legacy
    ``pair_partition_slot`` argument remains validated for call compatibility
    but cannot affect a single full-population partition.

    No chunk is duplicated or dropped. A one-sided/no-evidence population uses
    the ordinary balanced partition because its pair credit is exactly zero.
    """
    permutation = np.asarray(chunk_permutation, dtype=np.int64).reshape(-1)
    chunk_episode_membership = np.asarray(
        chunk_episode_membership,
        dtype=bool,
    )
    evidence = np.asarray(
        pair_evidence_episode_mask,
        dtype=bool,
    ).reshape(-1)
    success = np.asarray(episode_success, dtype=bool).reshape(-1)
    if evidence.shape != success.shape:
        raise ValueError(
            "pair evidence and episode outcome masks must share shape"
        )
    if (
            chunk_episode_membership.ndim != 2
            or chunk_episode_membership.shape
            != (permutation.size, evidence.size)):
        raise ValueError(
            "chunk_episode_membership must have shape [chunk, episode]"
        )
    if permutation.size == 0:
        raise ValueError("pair optimizer partition requires at least one chunk")
    if (
            np.unique(permutation).size != permutation.size
            or not np.array_equal(
                np.sort(permutation),
                np.arange(permutation.size, dtype=np.int64),
            )):
        raise ValueError(
            "chunk permutation must cover every chunk exactly once"
        )
    if np.any(chunk_episode_membership.sum(axis=0) <= 0):
        raise RuntimeError(
            "every selected episode must belong to at least one replay chunk"
        )
    if np.any(chunk_episode_membership.sum(axis=1) <= 0):
        raise RuntimeError(
            "every replay chunk must contain at least one selected episode"
        )

    num_mini_batch = max(
        1,
        min(int(num_mini_batch), int(permutation.size)),
    )
    pair_partition_slot = int(pair_partition_slot)
    if pair_partition_slot < 0 or pair_partition_slot >= num_mini_batch:
        raise ValueError(
            "pair_partition_slot must index a configured mini-batch"
        )
    successful_evidence = evidence & success
    failed_evidence = evidence & ~success
    class_complete = bool(
        np.any(successful_evidence) and np.any(failed_evidence)
    )
    if not class_complete:
        partitions = [
            np.asarray(partition, dtype=np.int64)
            for partition in np.array_split(permutation, num_mini_batch)
            if len(partition) > 0
        ]
        pair_partition_index = -1
        effective_pair_partition_slot = -1
    else:
        pair_chunk_mask = np.any(
            chunk_episode_membership[permutation]
            & evidence[None, :],
            axis=1,
        )
        pair_chunks = permutation[pair_chunk_mask]
        non_pair_chunks = permutation[~pair_chunk_mask]
        if pair_chunks.size == 0:
            raise RuntimeError(
                "class-complete pair evidence produced no replay chunks"
            )
        # Preserve the replay permutation but do not split the selected cohort.
        # ``train_adj_on_batch`` reduces graph and factor losses over this whole
        # sample and performs exactly one Adam step, which is the only exact
        # population-total implementation for stateful Adam.
        partitions = [permutation.copy()]
        pair_zero_credit_fillers = non_pair_chunks
        pair_partition_index = 0
        effective_pair_partition_slot = -1

    if not class_complete:
        pair_zero_credit_fillers = np.empty((0,), dtype=np.int64)

    covered = np.concatenate(partitions)
    if (
            covered.size != permutation.size
            or np.unique(covered).size != permutation.size
            or not np.array_equal(
                np.sort(covered),
                np.arange(permutation.size, dtype=np.int64),
            )):
        raise RuntimeError(
            "pair-aware replay partitions must cover every chunk exactly once"
        )
    if class_complete:
        pair_partition = partitions[pair_partition_index]
        if (
                len(partitions) != 1
                or pair_partition.size != permutation.size):
            raise RuntimeError(
                "class-complete pair replay must use one full-population "
                "optimizer transaction"
            )
        pair_partition_episode_offsets = np.flatnonzero(
            np.any(
                chunk_episode_membership[pair_partition],
                axis=0,
            )
        )
        all_pair_episode_offsets = np.flatnonzero(evidence)
        if not np.all(
                np.isin(
                    all_pair_episode_offsets,
                    pair_partition_episode_offsets,
                )):
            raise RuntimeError(
                "pair evidence episodes were split across optimizer steps"
            )

    partition_sizes = np.asarray(
        [int(partition.size) for partition in partitions],
        dtype=np.int64,
    )
    return {
        "partitions": partitions,
        "class_complete": float(class_complete),
        "pair_partition_index": int(pair_partition_index),
        "pair_partition_slot": int(effective_pair_partition_slot),
        "pair_evidence_episode_count": int(evidence.sum()),
        "successful_evidence_episode_count": int(
            successful_evidence.sum()
        ),
        "failed_evidence_episode_count": int(failed_evidence.sum()),
        "pair_zero_credit_filler_chunk_count": int(
            pair_zero_credit_fillers.size
        ),
        "pair_partition_chunk_count": int(
            partitions[pair_partition_index].size
            if pair_partition_index >= 0 else 0
        ),
        "partition_size_min": int(partition_sizes.min()),
        "partition_size_max": int(partition_sizes.max()),
        "partition_size_imbalance": int(
            partition_sizes.max() - partition_sizes.min()
        ),
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
        active_agent_count,
        selected_factor_behavior_probability,
        capture_counts,
        capture_factor_match,
        valid_graph_transition,
        dones_env,
        team_rewards,
        window,
        gamma,
        capture_event_provenance_by_episode=None):
    """Build strictly-future pair-to-identity-capture-factor evidence.

    The current factor must be a pair-only backbone selected from a real
    choice. A factor whose stored conditional behavior probability is exactly
    one has no selection competitor and therefore no adjacency-policy
    Jacobian. This includes the two-active-agent case and can also occur in a
    later sequential slot because coverage, quota, or order-band constraints
    leave one reachable candidate. Such forced actions remain valid behavior
    diagnostics but are explicitly ineligible for graph-selection
    supervision. A future real capture factor must contain the same two agents
    and must occur at ``offset > 0``.
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
    active_agent_count = _transition_matrix(
        active_agent_count,
        "active_agent_count",
        transition_shape,
    )
    if not np.isfinite(active_agent_count).all():
        raise ValueError("active_agent_count must be finite")
    rounded_active_agent_count = np.rint(active_agent_count)
    if not np.allclose(
            active_agent_count, rounded_active_agent_count, atol=1e-6):
        raise ValueError("active_agent_count must contain integer counts")
    if np.any(
            (rounded_active_agent_count < 0)
            | (rounded_active_agent_count > _num_agents)):
        raise ValueError(
            "active_agent_count must be in [0, {}]".format(_num_agents)
        )
    active_agent_count = rounded_active_agent_count.astype(
        np.int64, copy=False
    )
    selected_factor_behavior_probability = np.asarray(
        selected_factor_behavior_probability,
        dtype=np.float32,
    )
    if selected_factor_behavior_probability.shape != factor_shape:
        raise ValueError(
            "selected_factor_behavior_probability must have shape {}, got {}"
            .format(
                factor_shape,
                selected_factor_behavior_probability.shape,
            )
        )
    if not np.isfinite(selected_factor_behavior_probability).all():
        raise ValueError(
            "selected_factor_behavior_probability must be finite"
        )
    if np.any(
            (selected_factor_behavior_probability < 0.0)
            | (selected_factor_behavior_probability > 1.0)):
        raise ValueError(
            "selected_factor_behavior_probability must be in [0, 1]"
        )
    invalid_selected_probability = (
        valid_factor & (selected_factor_behavior_probability <= 0.0)
    )
    if np.any(invalid_selected_probability):
        first = tuple(
            int(item)
            for item in np.argwhere(invalid_selected_probability)[0]
        )
        raise RuntimeError(
            "selected valid factor has no behavior probability: index={}"
            .format(first)
        )
    pair_selection_actionable_factor = (
        (active_agent_count[..., None] >= 3)
        & (selected_factor_behavior_probability < 1.0)
    )
    pair_selection_actionable_transition = np.any(
        pair_selection_actionable_factor & valid_factor,
        axis=2,
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

    event_records_by_step_factor = None
    if capture_event_provenance_by_episode is not None:
        if (
                not isinstance(
                    capture_event_provenance_by_episode, (list, tuple))
                or len(capture_event_provenance_by_episode) != num_episodes):
            raise ValueError(
                "capture_event_provenance_by_episode must contain one event "
                "record collection per episode"
            )
        event_records_by_step_factor = [
            {} for _ in range(num_episodes)
        ]
        for episode, records in enumerate(
                capture_event_provenance_by_episode):
            if not isinstance(records, (list, tuple)):
                raise TypeError(
                    "capture event provenance for one episode must be a "
                    "list or tuple"
                )
            logical_keys = set()
            event_mass = {}
            event_contracts = {}
            for record in records:
                if not isinstance(record, dict):
                    raise TypeError(
                        "capture event provenance record must be a dict"
                    )
                required = (
                    "environment_episode_id",
                    "event_id",
                    "target_id",
                    "participant_slots",
                    "factor_index",
                    "factor_identity",
                    "factor_order",
                    "identity_event_weight",
                    "factor_slot_weight",
                    "capture_step",
                )
                missing = [name for name in required if name not in record]
                if missing:
                    raise RuntimeError(
                        "matched capture provenance is missing {}".format(
                            ",".join(missing)
                        )
                    )
                capture_step = int(record["capture_step"])
                factor_index = int(record["factor_index"])
                event_id = int(record["event_id"])
                prey_id = int(record["target_id"])
                participants = tuple(
                    sorted(int(slot)
                           for slot in record["participant_slots"])
                )
                factor_order = int(record["factor_order"])
                identity_nodes = tuple(
                    int(node)
                    for node in str(record["factor_identity"]).split("-")
                    if str(node) != ""
                )
                identity_weight = float(record["identity_event_weight"])
                factor_slot_weight = float(record["factor_slot_weight"])
                environment_episode_id = int(
                    record["environment_episode_id"]
                )
                if (
                        capture_step < 0
                        or capture_step >= time_steps
                        or factor_index < 0
                        or factor_index >= num_factors
                        or event_id < 0
                        or prey_id < 0
                        or factor_order not in (2, 3)
                        or len(identity_nodes) != factor_order
                        or tuple(sorted(identity_nodes)) != identity_nodes
                        or not set(identity_nodes).issubset(participants)
                        or not np.isfinite(identity_weight)
                        or identity_weight <= 0.0
                        or not np.isfinite(factor_slot_weight)
                        or factor_slot_weight <= 0.0):
                    raise RuntimeError(
                        "matched capture provenance contains an invalid value"
                    )
                factor_identity = tuple(sorted(
                    np.flatnonzero(
                        adj[capture_step, episode, :, factor_index] > 0
                    ).tolist()
                ))
                if (
                        factor_identity != identity_nodes
                        or factor_order != len(factor_identity)):
                    raise RuntimeError(
                        "matched capture provenance factor identity is "
                        "misaligned with the rollout graph"
                    )
                event_contract_key = (
                    environment_episode_id,
                    event_id,
                )
                event_contract = (
                    prey_id,
                    capture_step,
                    participants,
                )
                prior_event_contract = event_contracts.get(
                    event_contract_key
                )
                if (
                        prior_event_contract is not None
                        and prior_event_contract != event_contract):
                    raise RuntimeError(
                        "matched capture provenance changes target, step, or "
                        "participants within one event identity"
                    )
                event_contracts[event_contract_key] = event_contract
                logical_key = (
                    environment_episode_id,
                    event_id,
                    prey_id,
                    participants,
                    factor_index,
                )
                if logical_key in logical_keys:
                    raise RuntimeError(
                        "matched capture provenance duplicates one event "
                        "factor"
                    )
                logical_keys.add(logical_key)
                event_key = (
                    environment_episode_id,
                    event_id,
                    prey_id,
                    capture_step,
                )
                event_mass[event_key] = (
                    event_mass.get(event_key, 0.0) + factor_slot_weight
                )
                normalized = dict(record)
                normalized["participant_slots"] = participants
                normalized["factor_identity_nodes"] = identity_nodes
                normalized["capture_step"] = capture_step
                normalized["factor_index"] = factor_index
                event_records_by_step_factor[episode].setdefault(
                    (capture_step, factor_index), []
                ).append(normalized)
            if any(
                    abs(float(mass) - 1.0) > 1e-6
                    for mass in event_mass.values()):
                raise RuntimeError(
                    "matched capture event provenance does not conserve unit "
                    "event mass across active factor slots"
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

    structural_pair_only_mask = np.zeros(factor_shape, dtype=bool)
    pair_only_mask = np.zeros(factor_shape, dtype=bool)
    forced_pair_non_actionable_mask = np.zeros(factor_shape, dtype=bool)
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
                structural_pair_only_mask[step, episode, factor] = True
                if not pair_selection_actionable_factor[
                        step, episode, factor]:
                    forced_pair_non_actionable_mask[
                        step, episode, factor
                    ] = True
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
    strict_pair_event_provenance = [
        [] for _ in range(num_episodes)
    ]

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
                    matching_capture_factors = []
                    for capture_factor in np.flatnonzero(
                            capture_factor_quality[
                                future_step, episode
                            ] > 0.0):
                        capture_nodes = factor_nodes[future_step][episode].get(
                            int(capture_factor), frozenset()
                        )
                        if pair_nodes.issubset(capture_nodes):
                            matching_capture_factors.append(
                                int(capture_factor)
                            )
                    if not matching_capture_factors:
                        continue

                    matched_event_records = []
                    if event_records_by_step_factor is not None:
                        matching_records = []
                        for capture_factor in matching_capture_factors:
                            matching_records.extend(
                                event_records_by_step_factor[episode].get(
                                    (future_step, capture_factor), ()
                                )
                            )
                        records_by_event = {}
                        for record in matching_records:
                            event_key = (
                                int(record["environment_episode_id"]),
                                int(record["event_id"]),
                                int(record["target_id"]),
                                tuple(record["participant_slots"]),
                            )
                            records_by_event.setdefault(
                                event_key, []
                            ).append(record)
                        if not records_by_event:
                            raise RuntimeError(
                                "strict pair transition could not join a "
                                "capture event at the nearest future step; "
                                "episode={}, pair_step={}, capture_step={}, "
                                "event_count={}".format(
                                    episode,
                                    step,
                                    future_step,
                                    0,
                                )
                            )
                        # A simultaneous multi-prey capture may contain more
                        # than one real event at the same nearest future step.
                        # Emit one deterministic provenance row per event;
                        # duplicate factor slots within an event still collapse
                        # to one row and cannot multiply that event's mass.
                        for event_key in sorted(records_by_event):
                            matched_event_records.append(sorted(
                                records_by_event[event_key],
                                key=lambda record: (
                                    int(record["factor_index"]),
                                    str(record["factor_identity"]),
                                ),
                            )[0])

                    transition_score[step, episode, factor] = float(
                        gamma ** delay
                    )
                    transition_delay[step, episode, factor] = float(delay)
                    capture_matched_gate[future_step, episode] = 1.0
                    for matched_event_record in matched_event_records:
                        pair_identity = tuple(sorted(pair_nodes))
                        if (
                                future_step < step
                                or not set(pair_identity).issubset(
                                    set(matched_event_record[
                                        "factor_identity_nodes"
                                    ]))):
                            raise RuntimeError(
                                "strict pair event provenance join contract "
                                "failed"
                            )
                        strict_pair_event_provenance[episode].append({
                            "environment_episode_id": int(
                                matched_event_record[
                                    "environment_episode_id"
                                ]
                            ),
                            "event_id": int(
                                matched_event_record["event_id"]
                            ),
                            "target_id": int(
                                matched_event_record["target_id"]
                            ),
                            "participant_slots": tuple(
                                matched_event_record["participant_slots"]
                            ),
                            "pair_transition_step": int(step),
                            "capture_step": int(future_step),
                            "delay": int(delay),
                            "pair_factor_index": int(factor),
                            "pair_identity": "-".join(
                                str(int(node)) for node in pair_identity
                            ),
                            "pair_order": 2,
                            "capture_factor_index": int(
                                matched_event_record["factor_index"]
                            ),
                            "capture_factor_identity": str(
                                matched_event_record["factor_identity"]
                            ),
                            "capture_factor_order": int(
                                matched_event_record["factor_order"]
                            ),
                            "raw_transition_quality": float(
                                gamma ** delay
                            ),
                            "identity_event_weight": float(
                                matched_event_record[
                                    "identity_event_weight"
                                ]
                            ),
                            "factor_slot_weight": float(
                                matched_event_record["factor_slot_weight"]
                            ),
                        })
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
        "structural_pair_only_mask": structural_pair_only_mask,
        "pair_only_mask": pair_only_mask,
        "pair_selection_actionable_transition": (
            pair_selection_actionable_transition
        ),
        "pair_selection_actionable_factor": (
            pair_selection_actionable_factor
        ),
        "forced_pair_non_actionable_mask": forced_pair_non_actionable_mask,
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
        "strict_pair_event_provenance": tuple(
            tuple(records) for records in strict_pair_event_provenance
        ),
    }
