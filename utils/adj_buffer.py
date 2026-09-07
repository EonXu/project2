import numpy as np
from .util import get_dim_from_space
from .segment_tree import SumSegmentTree, MinSegmentTree
import torch

from .pair_credit import (
    CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
    canonical_capture_factor_catalog,
    compute_capture_anchored_pair_credit,
    partition_pair_contrast_optimizer_chunks,
    scale_optimizer_cohort_pair_credit,
    compute_capture_to_win_outcome_gate,
    compute_capture_to_win_triplet_outcome_advantage,
    scale_capture_to_win_outcome_credit,
)
from .graph_sampling import (
    build_previous_adjacency_sequence,
    build_previous_done_sequence,
    require_ready_graph_advantage,
    select_outcome_contrast_complete_episodes,
    write_graph_advantage_sequence,
)
from .pair_pending import (
    PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
    PAIR_EVIDENCE_COMMITTED,
    PAIR_EVIDENCE_PENDING,
    PairPendingEvidenceStore,
    merge_generation_event_pair_scores,
    pair_pending_entry_key,
    reconstruct_pending_pair_mass,
)


def _cast(x):
    return x.transpose(2, 0, 1, 3)


def summarize_pair_evidence_episode_funnel(
        occupied_episode_mask,
        successful_episode_mask,
        active_capture_episode_mask,
        candidate_capture_episode_mask,
        pair_evidence_episode_mask):
    """Count the replay episode funnel before signed pair support selection.

    These counts are deliberately computed from boolean episode provenance,
    before support eligibility, replay augmentation, or optimizer sampling.
    They are read-only diagnostics: in particular, they must never be used to
    relax support reuse or to manufacture a missing signed class.
    """
    named_masks = (
        ("occupied", occupied_episode_mask),
        ("successful", successful_episode_mask),
        ("active capture", active_capture_episode_mask),
        ("candidate capture", candidate_capture_episode_mask),
        ("pair evidence", pair_evidence_episode_mask),
    )
    normalized = {}
    reference_shape = None
    for name, mask in named_masks:
        values = np.asarray(mask)
        if values.ndim != 1:
            raise ValueError(
                "{} episode mask must be one-dimensional".format(name)
            )
        if reference_shape is None:
            reference_shape = values.shape
        elif values.shape != reference_shape:
            raise ValueError(
                "pair evidence funnel episode masks have different shapes"
            )
        normalized[name] = values.astype(bool, copy=False)

    occupied = normalized["occupied"]
    successful = normalized["successful"] & occupied
    active_capture = normalized["active capture"] & occupied
    candidate_capture = normalized["candidate capture"] & occupied
    capture = active_capture | candidate_capture
    pair_evidence = normalized["pair evidence"] & occupied
    pair_positive = pair_evidence & successful
    pair_negative = pair_evidence & ~successful & occupied
    successful_capture = capture & successful
    successful_capture_without_pair = (
        successful_capture & ~pair_evidence
    )
    pair_without_capture = pair_evidence & ~capture
    # A successful capture without strict pair evidence must fall into one of
    # two source-grounded branches.  An episode with any exact active capture
    # identity is attributed to the active branch; otherwise a representable
    # candidate-only capture is the earliest missing link.  This precedence
    # makes the reject-reason partition mutually exclusive even when one
    # episode contains multiple capture events from both branches.
    gap_active_without_strict_pair = (
        successful_capture_without_pair & active_capture
    )
    gap_candidate_only_not_active = (
        successful_capture_without_pair
        & candidate_capture
        & ~active_capture
    )
    gap_unclassified = (
        successful_capture_without_pair
        & ~gap_active_without_strict_pair
        & ~gap_candidate_only_not_active
    )

    occupied_count = int(np.sum(occupied))
    pair_evidence_count = int(np.sum(pair_evidence))
    pair_positive_count = int(np.sum(pair_positive))
    pair_negative_count = int(np.sum(pair_negative))
    if pair_positive_count + pair_negative_count != pair_evidence_count:
        raise RuntimeError(
            "signed pair evidence episode counts do not reconstruct total"
        )
    if pair_evidence_count > occupied_count:
        raise RuntimeError(
            "pair evidence episode count exceeds occupied replay population"
        )
    gap_count = int(np.sum(successful_capture_without_pair))
    gap_partition_count = int(
        np.sum(gap_active_without_strict_pair)
        + np.sum(gap_candidate_only_not_active)
        + np.sum(gap_unclassified)
    )
    if gap_partition_count != gap_count:
        raise RuntimeError(
            "successful-capture pair-evidence reject reasons do not "
            "reconstruct the gap population"
        )
    if np.any(gap_unclassified):
        raise RuntimeError(
            "successful capture without pair evidence has neither an active "
            "nor a candidate-only exact identity"
        )

    return {
        "version": 2.0,
        "occupied_episode_count": occupied_count,
        "successful_episode_count": int(np.sum(successful)),
        "capture_episode_count": int(np.sum(capture)),
        "successful_capture_episode_count": int(
            np.sum(successful_capture)
        ),
        "successful_active_capture_episode_count": int(
            np.sum(successful & active_capture)
        ),
        "successful_candidate_capture_episode_count": int(
            np.sum(successful & candidate_capture)
        ),
        "pair_evidence_episode_count": pair_evidence_count,
        "pair_positive_episode_count": pair_positive_count,
        "pair_negative_episode_count": pair_negative_count,
        "successful_capture_without_pair_evidence_episode_count": int(
            np.sum(successful_capture_without_pair)
        ),
        "pair_evidence_without_capture_episode_count": int(
            np.sum(pair_without_capture)
        ),
        "successful_capture_gap_candidate_only_not_active_episode_count": int(
            np.sum(gap_candidate_only_not_active)
        ),
        "successful_capture_gap_active_without_strict_pair_episode_count": int(
            np.sum(gap_active_without_strict_pair)
        ),
        "successful_capture_gap_unclassified_episode_count": int(
            np.sum(gap_unclassified)
        ),
        "successful_capture_gap_reject_reason_contract_valid": 1.0,
        "contract_valid": 1.0,
    }


def summarize_successful_candidate_capture_context(
        successful_episode_mask,
        pair_evidence_episode_mask,
        active_capture_episode_mask,
        candidate_capture_weights,
        candidate_behavior,
        capture_transition_mask,
        terminal_transition_mask):
    """Summarize exact successful candidate-only identities without mutation.

    This diagnostic closes the earliest run104 funnel gap at behavior time.  It
    reports the stored first-reachable competitor margin/rank for exact
    candidate-only capture identities and the distance from the last capture to
    episode termination.  It neither changes candidate targets nor treats a
    candidate-only event as strict pair evidence.
    """
    successful = np.asarray(successful_episode_mask, dtype=bool)
    pair_evidence = np.asarray(pair_evidence_episode_mask, dtype=bool)
    active_capture = np.asarray(active_capture_episode_mask, dtype=bool)
    weights = np.asarray(candidate_capture_weights, dtype=np.float32)
    behavior = np.asarray(candidate_behavior, dtype=np.float32)
    capture_transition = np.asarray(capture_transition_mask, dtype=bool)
    terminal_transition = np.asarray(terminal_transition_mask, dtype=bool)
    if successful.ndim != 1:
        raise ValueError("successful episode mask must be one-dimensional")
    if pair_evidence.shape != successful.shape:
        raise ValueError("pair evidence episode mask shape mismatch")
    if active_capture.shape != successful.shape:
        raise ValueError("active capture episode mask shape mismatch")
    if weights.ndim != 3:
        raise ValueError(
            "candidate capture weights must have shape [time, episode, identity]"
        )
    if behavior.shape != weights.shape + (4,):
        raise ValueError(
            "candidate behavior shape {} does not match capture weights {}"
            .format(behavior.shape, weights.shape)
        )
    if capture_transition.shape != weights.shape[:2]:
        raise ValueError("capture transition mask shape mismatch")
    if terminal_transition.shape != weights.shape[:2]:
        raise ValueError("terminal transition mask shape mismatch")
    if successful.shape != (weights.shape[1],):
        raise ValueError("episode masks do not match candidate replay axis")
    if (
            not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.isfinite(behavior).all()):
        raise ValueError(
            "candidate capture context must be finite and non-negative"
        )

    candidate_episode = np.any(weights > 0.0, axis=(0, 2))
    successful_gap = successful & candidate_episode & ~pair_evidence
    candidate_only_gap = successful_gap & ~active_capture
    target_mask = (
        (weights > 0.0)
        & candidate_only_gap[None, :, None]
    )
    target_weight = np.where(target_mask, weights, 0.0)
    identity_mass = float(target_weight.sum())
    identity_count = int(np.sum(target_mask))
    behavior_margin = behavior[..., 0]
    behavior_rank = behavior[..., 1]
    behavior_valid = behavior[..., 2]
    if np.any(target_mask & (behavior_valid <= 0.0)):
        raise RuntimeError(
            "successful candidate-only capture targets invalid behavior metadata"
        )
    if np.any(target_mask & (behavior_rank < 1.0)):
        raise RuntimeError(
            "successful candidate-only capture has no canonical behavior rank"
        )

    def _weighted_mean(values):
        if identity_mass <= 0.0:
            return 0.0
        return float((values * target_weight).sum() / identity_mass)

    def _masked_min(values):
        return float(values[target_mask].min()) if identity_count > 0 else 0.0

    def _masked_max(values):
        return float(values[target_mask].max()) if identity_count > 0 else 0.0

    terminal_capture_count = 0
    last_capture_to_terminal = []
    for episode in np.flatnonzero(successful_gap):
        capture_steps = np.flatnonzero(capture_transition[:, episode])
        terminal_steps = np.flatnonzero(terminal_transition[:, episode])
        if capture_steps.size == 0:
            raise RuntimeError(
                "successful capture gap episode has no capture transition"
            )
        last_capture = int(capture_steps[-1])
        reachable_terminal = terminal_steps[terminal_steps >= last_capture]
        if reachable_terminal.size == 0:
            raise RuntimeError(
                "completed successful capture episode has no terminal "
                "transition at or after its last capture"
            )
        terminal_step = int(reachable_terminal[0])
        distance = terminal_step - last_capture
        last_capture_to_terminal.append(float(distance))
        if distance == 0:
            terminal_capture_count += 1
    distances = np.asarray(last_capture_to_terminal, dtype=np.float32)

    return {
        "successful_candidate_gap_episode_count": int(
            np.sum(candidate_only_gap)
        ),
        "successful_candidate_gap_identity_count": identity_count,
        "successful_candidate_gap_identity_mass": identity_mass,
        "successful_candidate_gap_behavior_margin_mean": _weighted_mean(
            behavior_margin
        ),
        "successful_candidate_gap_behavior_margin_min": _masked_min(
            behavior_margin
        ),
        "successful_candidate_gap_behavior_margin_max": _masked_max(
            behavior_margin
        ),
        "successful_candidate_gap_behavior_rank_mean": _weighted_mean(
            behavior_rank
        ),
        "successful_candidate_gap_behavior_rank_min": _masked_min(
            behavior_rank
        ),
        "successful_candidate_gap_behavior_rank_max": _masked_max(
            behavior_rank
        ),
        "successful_candidate_gap_behavior_boundary_crossed_fraction": (
            _weighted_mean((behavior_margin > 0.0).astype(np.float32))
        ),
        "successful_candidate_gap_behavior_rank1_fraction": (
            _weighted_mean((behavior_rank <= 1.0).astype(np.float32))
        ),
        "successful_capture_gap_terminal_capture_episode_count": int(
            terminal_capture_count
        ),
        "successful_capture_gap_last_capture_to_terminal_step_mean": (
            float(distances.mean()) if distances.size > 0 else 0.0
        ),
        "successful_capture_gap_last_capture_to_terminal_step_min": (
            float(distances.min()) if distances.size > 0 else 0.0
        ),
        "successful_capture_gap_last_capture_to_terminal_step_max": (
            float(distances.max()) if distances.size > 0 else 0.0
        ),
        "successful_candidate_gap_context_contract_valid": 1.0,
    }


def build_pair_evidence_episode_diagnostic_rows(
        episode_generation,
        selected_episode_indices,
        base_episode_indices,
        supplemented_episode_indices,
        outcome_support_used,
        successful_episode_mask,
        active_capture_episode_mask,
        candidate_capture_weights,
        candidate_behavior,
        pair_evidence_transition_mask,
        capture_transition_mask,
        terminal_transition_mask,
        capture_identity_candidate_count,
        recency_age,
        num_agents,
        configured_pair_window,
        outcome_class_complete,
        pair_class_complete,
        active_capture_event_provenance=None,
        require_active_capture_event_provenance=False):
    """Build read-only per-episode provenance rows for the replay funnel.

    Every occupied replay generation is emitted exactly once per adjacency
    update.  The reject reason describes the earliest source-backed gap only;
    it never changes selection, targets, or support eligibility.
    """
    generations = np.asarray(episode_generation, dtype=np.int64)
    selected = np.asarray(selected_episode_indices, dtype=np.int64).reshape(-1)
    base = np.asarray(base_episode_indices, dtype=np.int64).reshape(-1)
    supplemented = np.asarray(
        supplemented_episode_indices,
        dtype=np.int64,
    ).reshape(-1)
    support_used = np.asarray(outcome_support_used, dtype=bool)
    successful = np.asarray(successful_episode_mask, dtype=bool)
    active_capture = np.asarray(active_capture_episode_mask, dtype=bool)
    candidate_weights = np.asarray(
        candidate_capture_weights,
        dtype=np.float32,
    )
    behavior = np.asarray(candidate_behavior, dtype=np.float32)
    pair_transition = np.asarray(
        pair_evidence_transition_mask,
        dtype=bool,
    )
    capture_transition = np.asarray(capture_transition_mask, dtype=bool)
    terminal_transition = np.asarray(terminal_transition_mask, dtype=bool)
    identity_candidate_count = np.asarray(
        capture_identity_candidate_count,
        dtype=np.float32,
    )
    ages = np.asarray(recency_age, dtype=np.int64)

    if active_capture_event_provenance is None:
        if (
                bool(require_active_capture_event_provenance)
                and np.any(active_capture & (generations >= 0))):
            raise RuntimeError(
                "active capture episode diagnostic is missing exact event "
                "provenance"
            )
        active_event_provenance = [
            tuple() for _ in range(int(generations.size))
        ]
    else:
        if (
                not isinstance(active_capture_event_provenance, (list, tuple))
                or len(active_capture_event_provenance)
                != int(generations.size)):
            raise ValueError(
                "active capture event provenance does not match replay "
                "episode axis"
            )
        active_event_provenance = active_capture_event_provenance

    if generations.ndim != 1:
        raise ValueError("episode generation must be one-dimensional")
    episode_count = int(generations.size)
    for name, values in (
            ("outcome support used", support_used),
            ("successful", successful),
            ("active capture", active_capture),
            ("recency age", ages)):
        if values.shape != (episode_count,):
            raise ValueError(
                "{} episode diagnostic shape mismatch".format(name)
            )
    if candidate_weights.ndim != 3:
        raise ValueError(
            "candidate capture weights must be [time, episode, identity]"
        )
    if candidate_weights.shape[1] != episode_count:
        raise ValueError("candidate capture replay episode axis mismatch")
    if behavior.shape != candidate_weights.shape + (4,):
        raise ValueError("candidate behavior replay shape mismatch")
    transition_shape = candidate_weights.shape[:2]
    for name, values in (
            ("pair evidence", pair_transition),
            ("capture", capture_transition),
            ("terminal", terminal_transition),
            ("capture identity candidate", identity_candidate_count)):
        if values.shape != transition_shape:
            raise ValueError(
                "{} transition diagnostic shape mismatch".format(name)
            )
    if (
            not np.isfinite(candidate_weights).all()
            or np.any(candidate_weights < 0.0)
            or not np.isfinite(behavior).all()
            or not np.isfinite(identity_candidate_count).all()
            or np.any(identity_candidate_count < 0.0)):
        raise ValueError("episode funnel provenance must be finite")
    for name, indices in (
            ("selected", selected),
            ("base", base),
            ("supplemented", supplemented)):
        if (
                indices.size
                and (
                    np.any(indices < 0)
                    or np.any(indices >= episode_count)
                    or np.unique(indices).size != indices.size)):
            raise ValueError(
                "{} episode indices are invalid or duplicated".format(name)
            )
    if np.intersect1d(base, supplemented).size:
        raise RuntimeError(
            "base and supplemented episode provenance overlap"
        )
    if set(selected.tolist()) != set(
            np.concatenate([base, supplemented]).tolist()
    ):
        raise RuntimeError(
            "selected episodes do not reconstruct base plus support"
        )

    selected_ordinal = {
        int(slot): int(ordinal)
        for ordinal, slot in enumerate(selected.tolist())
    }
    base_set = set(int(slot) for slot in base.tolist())
    supplemented_set = set(int(slot) for slot in supplemented.tolist())
    candidate_catalog = canonical_capture_factor_catalog(
        int(num_agents),
        3,
    )
    if len(candidate_catalog) != candidate_weights.shape[2]:
        raise RuntimeError(
            "candidate identity catalog does not match replay width"
        )

    rows = []
    occupied_slots = np.flatnonzero(generations >= 0)
    for slot_value in occupied_slots:
        slot = int(slot_value)
        pair_steps = np.flatnonzero(pair_transition[:, slot])
        capture_steps = np.flatnonzero(capture_transition[:, slot])
        terminal_steps = np.flatnonzero(terminal_transition[:, slot])
        candidate_mask = candidate_weights[:, slot, :] > 0.0
        candidate_indices = np.flatnonzero(
            np.any(candidate_mask, axis=0)
        ).astype(np.int64, copy=False)
        candidate_episode = bool(candidate_indices.size)
        active_episode = bool(active_capture[slot])
        active_records = active_event_provenance[slot]
        if not isinstance(active_records, (list, tuple)):
            raise TypeError(
                "active capture event provenance for one episode must be a "
                "list or tuple"
            )
        if bool(active_records) and not active_episode:
            raise RuntimeError(
                "active capture episode flag does not match exact event "
                "provenance"
            )
        if (
                active_episode
                and not bool(active_records)
                and bool(require_active_capture_event_provenance)):
            raise RuntimeError(
                "active capture episode diagnostic is missing exact event "
                "provenance"
            )
        normalized_active_records = []
        active_logical_keys = set()
        for record in active_records:
            if not isinstance(record, dict):
                raise TypeError(
                    "active capture event provenance record must be a dict"
                )
            required_fields = (
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
                "static_dynamic_class",
            )
            missing = [
                field for field in required_fields if field not in record
            ]
            if missing:
                raise RuntimeError(
                    "active capture event provenance is missing {}".format(
                        ",".join(missing)
                    )
                )
            environment_episode_id = int(record["environment_episode_id"])
            event_id = int(record["event_id"])
            target_id = int(record["target_id"])
            factor_index = int(record["factor_index"])
            factor_order = int(record["factor_order"])
            capture_step = int(record["capture_step"])
            factor_identity = str(record["factor_identity"])
            identity_nodes = tuple(
                int(node) for node in factor_identity.split("-") if node != ""
            )
            participant_slots = tuple(
                sorted(int(node) for node in record["participant_slots"])
            )
            static_dynamic_class = str(record["static_dynamic_class"])
            identity_event_weight = float(record["identity_event_weight"])
            factor_slot_weight = float(record["factor_slot_weight"])
            if (
                    environment_episode_id < 0
                    or event_id < 0
                    or target_id < 0
                    or factor_index < 0
                    or factor_order not in (2, 3)
                    or len(identity_nodes) != factor_order
                    or tuple(sorted(identity_nodes)) != identity_nodes
                    or factor_identity != "-".join(
                        str(node) for node in identity_nodes
                    )
                    or not set(identity_nodes).issubset(participant_slots)
                    or capture_step < 0
                    or capture_step >= int(capture_transition.shape[0])
                    or not bool(capture_transition[capture_step, slot])
                    or static_dynamic_class not in ("static", "dynamic")
                    or not np.isfinite(identity_event_weight)
                    or identity_event_weight <= 0.0
                    or not np.isfinite(factor_slot_weight)
                    or factor_slot_weight <= 0.0):
                raise RuntimeError(
                    "active capture event provenance contains an invalid "
                    "episode diagnostic value"
                )
            logical_key = (
                environment_episode_id,
                event_id,
                target_id,
                capture_step,
                factor_index,
            )
            if logical_key in active_logical_keys:
                raise RuntimeError(
                    "active capture event provenance duplicates one event "
                    "factor"
                )
            active_logical_keys.add(logical_key)
            normalized_active_records.append({
                "environment_episode_id": environment_episode_id,
                "event_id": event_id,
                "target_id": target_id,
                "factor_identity": factor_identity,
                "factor_index": factor_index,
                "capture_step": capture_step,
                "static_dynamic_class": static_dynamic_class,
            })
        normalized_active_records.sort(key=lambda record: (
            record["capture_step"],
            record["event_id"],
            record["factor_index"],
            record["factor_identity"],
        ))
        active_environment_ids = sorted(set(
            record["environment_episode_id"]
            for record in normalized_active_records
        ))
        active_event_ids = sorted(set(
            record["event_id"] for record in normalized_active_records
        ))
        active_target_ids = sorted(set(
            record["target_id"] for record in normalized_active_records
        ))
        active_factor_identities = []
        active_identity_classes = []
        for record in normalized_active_records:
            identity = record["factor_identity"]
            identity_class = "{}:{}".format(
                identity,
                record["static_dynamic_class"],
            )
            if identity not in active_factor_identities:
                active_factor_identities.append(identity)
            if identity_class not in active_identity_classes:
                active_identity_classes.append(identity_class)
        if len(active_environment_ids) > 1:
            raise RuntimeError(
                "one replay episode contains multiple environment episode "
                "identities"
            )
        capture_episode = bool(capture_steps.size)
        success_episode = bool(successful[slot])
        pair_episode = bool(pair_steps.size)
        successful_capture_gap = bool(
            success_episode and capture_episode and not pair_episode
        )
        if successful_capture_gap and active_episode:
            reject_reason = "ACTIVE_CAPTURE_NO_STRICT_PRIOR_PAIR"
        elif (
                successful_capture_gap
                and candidate_episode
                and not active_episode):
            reject_reason = "CANDIDATE_ONLY_NOT_ACTIVE"
        elif successful_capture_gap:
            raise RuntimeError(
                "successful capture gap has no source-backed reject reason"
            )
        else:
            reject_reason = "NOT_A_SUCCESSFUL_CAPTURE_GAP"

        first_capture = int(capture_steps[0]) if capture_steps.size else -1
        last_capture = int(capture_steps[-1]) if capture_steps.size else -1
        terminal_step = int(terminal_steps[0]) if terminal_steps.size else -1
        capture_to_terminal = (
            terminal_step - last_capture
            if last_capture >= 0 and terminal_step >= last_capture
            else -1
        )
        if successful_capture_gap and capture_to_terminal < 0:
            raise RuntimeError(
                "completed successful capture gap has invalid terminal timing"
            )

        target_mask = candidate_mask
        target_weight = np.where(
            target_mask,
            candidate_weights[:, slot, :],
            0.0,
        )
        target_mass = float(target_weight.sum())
        target_count = int(np.sum(target_mask))
        margin = behavior[:, slot, :, 0]
        rank = behavior[:, slot, :, 1]
        valid = behavior[:, slot, :, 2]
        if np.any(target_mask & (valid <= 0.0)):
            raise RuntimeError(
                "episode diagnostic found invalid candidate metadata"
            )
        if np.any(target_mask & (rank < 1.0)):
            raise RuntimeError(
                "episode diagnostic found invalid candidate rank"
            )
        weighted_margin = (
            float((margin * target_weight).sum() / target_mass)
            if target_mass > 0.0 else 0.0
        )
        weighted_rank = (
            float((rank * target_weight).sum() / target_mass)
            if target_mass > 0.0 else 0.0
        )
        identity_strings = [
            "-".join(str(int(node)) for node in candidate_catalog[index])
            for index in candidate_indices.tolist()
        ]
        identity_orders = [
            len(candidate_catalog[index])
            for index in candidate_indices.tolist()
        ]
        rows.append({
            "diagnostic_version": 2,
            "replay_slot_index": slot,
            "episode_generation": int(generations[slot]),
            "episode_recency_age": int(ages[slot]),
            "selected_for_training": int(slot in selected_ordinal),
            "selected_episode_ordinal": int(
                selected_ordinal.get(slot, -1)
            ),
            "base_selected": int(slot in base_set),
            "support_selected": int(slot in supplemented_set),
            "outcome_support_used": int(support_used[slot]),
            "outcome_class_complete": int(bool(outcome_class_complete)),
            "pair_class_complete": int(bool(pair_class_complete)),
            "outcome_success": int(success_episode),
            "capture_episode": int(capture_episode),
            "active_capture_episode": int(active_episode),
            "candidate_capture_episode": int(candidate_episode),
            "successful_capture_without_pair_evidence": int(
                successful_capture_gap
            ),
            "pair_evidence_episode": int(pair_episode),
            "pair_evidence_sign": int(
                1 if pair_episode and success_episode
                else -1 if pair_episode else 0
            ),
            "reject_reason": reject_reason,
            "capture_transition_count": int(capture_steps.size),
            "first_capture_step": first_capture,
            "last_capture_step": last_capture,
            "terminal_step": terminal_step,
            "last_capture_to_terminal_step": int(capture_to_terminal),
            "terminal_capture": int(
                capture_to_terminal == 0 and capture_episode
            ),
            "strict_pair_evidence_anchor_count": int(pair_steps.size),
            "first_strict_pair_evidence_anchor_step": int(
                pair_steps[0] if pair_steps.size else -1
            ),
            "last_strict_pair_evidence_anchor_step": int(
                pair_steps[-1] if pair_steps.size else -1
            ),
            "configured_strict_pair_window": int(configured_pair_window),
            "capture_identity_candidate_count": float(
                identity_candidate_count[:, slot].sum()
            ),
            "candidate_identity_count": int(candidate_indices.size),
            "candidate_target_transition_count": target_count,
            "candidate_identity_mass": target_mass,
            "candidate_identity_indices": ";".join(
                str(int(index)) for index in candidate_indices.tolist()
            ),
            "candidate_factor_identities": ";".join(identity_strings),
            "candidate_factor_order_min": int(
                min(identity_orders) if identity_orders else 0
            ),
            "candidate_factor_order_max": int(
                max(identity_orders) if identity_orders else 0
            ),
            "participant_slots_available": int(bool(identity_strings)),
            "participant_slots": ";".join(identity_strings),
            "candidate_behavior_margin_mean": weighted_margin,
            "candidate_behavior_rank_mean": weighted_rank,
            # Event-level active-capture provenance is already part of the
            # replay's immutable single-use evidence state.  Expose it here
            # read-only so a retained promotion can be joined to later exact
            # behavior.  Scalar event/prey fields remain unavailable when an
            # episode contains more than one distinct capture event.
            "environment_episode_id_available": int(
                len(active_environment_ids) == 1
            ),
            "environment_episode_id": int(
                active_environment_ids[0]
                if len(active_environment_ids) == 1 else -1
            ),
            "capture_event_id_available": int(len(active_event_ids) == 1),
            "capture_event_id": int(
                active_event_ids[0] if len(active_event_ids) == 1 else -1
            ),
            "capture_prey_id_available": int(len(active_target_ids) == 1),
            "capture_prey_id": int(
                active_target_ids[0] if len(active_target_ids) == 1 else -1
            ),
            "matched_active_factor_identity_available": int(
                bool(active_factor_identities)
            ),
            "matched_active_factor_identity": ";".join(
                active_factor_identities
            ),
            "static_dynamic_identity_available": int(
                bool(active_identity_classes)
            ),
            "static_dynamic_identity": ";".join(active_identity_classes),
            "future_evidence_transition_step_available": 0,
            "future_evidence_transition_step": -1,
            "row_contract_valid": 1,
        })
    if len(rows) != int(occupied_slots.size):
        raise RuntimeError(
            "episode funnel rows do not cover occupied replay generations"
        )
    if len({
            int(row["episode_generation"]) for row in rows
    }) != len(rows):
        raise RuntimeError(
            "episode funnel rows contain duplicate replay generations"
        )
    return rows


CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION = 1
CANDIDATE_EVIDENCE_PROVENANCE_DRAFT_FIELDS = (
    "diagnostic_version",
    "replay_slot_index",
    "replay_generation",
    "environment_episode_id",
    "episode_ordinal",
    "capture_event_id",
    "prey_id",
    "capture_step",
    "terminal_step",
    "capture_to_terminal_distance",
    "outcome_success",
    "candidate_index",
    "candidate_identity",
    "candidate_order",
    "participant_slots",
    "participant_count",
    "static_dynamic_class",
    "target_sign",
    "raw_event_quality",
    "identity_event_weight",
    "identity_allocated_quality",
    "candidate_coefficient",
    "final_target_mass",
    "target_bearing_transition_count",
    "base_selected",
    "support_selected",
    "behavior_policy_version",
    "quality_reconstruction_error",
    "provenance_complete",
    "identity_contract_valid",
    "quality_contract_valid",
)


def build_candidate_evidence_provenance_rows(
        event_records_by_episode,
        replay_slot_indices,
        replay_generations,
        environment_episode_ids,
        base_selected,
        support_selected,
        outcome_success,
        episode_outcome_advantage,
        capture_event_mass_per_episode,
        candidate_coefficient,
        candidate_identity_delta,
        candidate_behavior,
        terminal_steps):
    """Build detached event-level rows that reconstruct candidate targets.

    One row is one real capture event times one canonical candidate identity.
    The helper consumes already-materialized NumPy diagnostics only; it never
    mutates replay or participates in target/loss construction.
    """
    episode_count = len(event_records_by_episode)
    vector_inputs = (
        ("replay_slot_indices", replay_slot_indices),
        ("replay_generations", replay_generations),
        ("environment_episode_ids", environment_episode_ids),
        ("base_selected", base_selected),
        ("support_selected", support_selected),
        ("outcome_success", outcome_success),
        ("episode_outcome_advantage", episode_outcome_advantage),
        ("capture_event_mass_per_episode", capture_event_mass_per_episode),
        ("terminal_steps", terminal_steps),
    )
    vectors = {}
    for name, values in vector_inputs:
        array = np.asarray(values).reshape(-1)
        if array.shape != (episode_count,):
            raise ValueError(
                "{} must have shape {}, got {}".format(
                    name,
                    (episode_count,),
                    array.shape,
                )
            )
        vectors[name] = array
    delta = np.asarray(candidate_identity_delta, dtype=np.float32)
    behavior = np.asarray(candidate_behavior, dtype=np.float32)
    if delta.ndim != 3:
        raise ValueError(
            "candidate_identity_delta must have shape "
            "[time, episode, candidate]"
        )
    if (
            behavior.shape != delta.shape + (4,)
            or delta.shape[1] != episode_count):
        raise ValueError(
            "candidate behavior/provenance populations do not match"
        )
    if (
            not np.isfinite(delta).all()
            or not np.isfinite(behavior).all()):
        raise FloatingPointError(
            "candidate evidence provenance inputs must be finite"
        )
    coefficient = float(candidate_coefficient)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(
            "candidate evidence coefficient must be finite and non-negative"
        )

    rows = []
    reconstructed_delta = np.zeros_like(delta, dtype=np.float32)
    event_row_groups = {}
    for episode_ordinal, event_records in enumerate(
            event_records_by_episode):
        if not isinstance(event_records, (list, tuple)):
            raise TypeError(
                "candidate event records must be grouped by episode"
            )
        event_mass = float(
            vectors["capture_event_mass_per_episode"][episode_ordinal]
        )
        outcome_advantage = float(
            vectors["episode_outcome_advantage"][episode_ordinal]
        )
        if event_mass < 0.0 or not np.isfinite(event_mass):
            raise ValueError("candidate event mass must be finite and non-negative")
        if not np.isfinite(outcome_advantage):
            raise ValueError("candidate outcome advantage must be finite")
        if not event_records or outcome_advantage == 0.0:
            continue
        if event_mass <= 0.0:
            raise RuntimeError(
                "candidate evidence event has no episode event-mass "
                "denominator"
            )
        raw_event_quality = abs(outcome_advantage) / event_mass
        target_sign = 1 if outcome_advantage > 0.0 else -1
        terminal_step = int(vectors["terminal_steps"][episode_ordinal])
        replay_slot = int(
            vectors["replay_slot_indices"][episode_ordinal]
        )
        replay_generation = int(
            vectors["replay_generations"][episode_ordinal]
        )
        environment_episode_id = int(
            vectors["environment_episode_ids"][episode_ordinal]
        )
        if (
                replay_slot < 0
                or replay_generation < 0
                or environment_episode_id < 0
                or terminal_step < 0):
            raise RuntimeError(
                "candidate evidence has incomplete episode provenance"
            )
        for record in event_records:
            capture_step = int(record["capture_step"])
            candidate_index = int(record["candidate_index"])
            identity_weight = float(record["identity_event_weight"])
            if (
                    capture_step < 0
                    or capture_step >= delta.shape[0]
                    or candidate_index < 0
                    or candidate_index >= delta.shape[2]
                    or terminal_step < capture_step):
                raise RuntimeError(
                    "candidate evidence capture provenance is invalid"
                )
            identity_allocated_quality = (
                raw_event_quality * identity_weight
            )
            final_target_mass = (
                identity_allocated_quality * coefficient
            )
            if final_target_mass <= 0.0:
                continue
            reconstructed_delta[
                capture_step,
                episode_ordinal,
                candidate_index,
            ] += float(target_sign) * final_target_mass
            participants = tuple(
                int(slot) for slot in record["participant_slots"]
            )
            event_key = (
                environment_episode_id,
                int(record["event_id"]),
                int(record["target_id"]),
            )
            row = {
                "diagnostic_version": int(
                    CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION
                ),
                "replay_slot_index": replay_slot,
                "replay_generation": replay_generation,
                "environment_episode_id": environment_episode_id,
                "episode_ordinal": int(episode_ordinal),
                "capture_event_id": int(record["event_id"]),
                "prey_id": int(record["target_id"]),
                "capture_step": capture_step,
                "terminal_step": terminal_step,
                "capture_to_terminal_distance": int(
                    terminal_step - capture_step
                ),
                "outcome_success": int(bool(
                    vectors["outcome_success"][episode_ordinal]
                )),
                "candidate_index": candidate_index,
                "candidate_identity": str(
                    record["candidate_identity"]
                ),
                "candidate_order": int(record["candidate_order"]),
                "participant_slots": "-".join(
                    str(slot) for slot in participants
                ),
                "participant_count": int(len(participants)),
                "static_dynamic_class": str(
                    record["static_dynamic_class"]
                ),
                "target_sign": int(target_sign),
                "raw_event_quality": float(raw_event_quality),
                "identity_event_weight": float(identity_weight),
                "identity_allocated_quality": float(
                    identity_allocated_quality
                ),
                "candidate_coefficient": coefficient,
                "final_target_mass": float(final_target_mass),
                "target_bearing_transition_count": 1,
                "base_selected": int(bool(
                    vectors["base_selected"][episode_ordinal]
                )),
                "support_selected": int(bool(
                    vectors["support_selected"][episode_ordinal]
                )),
                "behavior_policy_version": float(
                    behavior[
                        capture_step,
                        episode_ordinal,
                        candidate_index,
                        3,
                    ]
                ),
                "quality_reconstruction_error": 0.0,
                "provenance_complete": 1,
                "identity_contract_valid": 1,
                "quality_contract_valid": 1,
            }
            if tuple(row.keys()) != (
                    CANDIDATE_EVIDENCE_PROVENANCE_DRAFT_FIELDS):
                raise RuntimeError(
                    "candidate evidence provenance draft schema diverged"
                )
            event_row_groups.setdefault(event_key, []).append(row)
            rows.append(row)

    for event_key, event_rows in event_row_groups.items():
        identity_mass = sum(
            float(row["final_target_mass"]) for row in event_rows
        )
        expected_mass = (
            float(event_rows[0]["raw_event_quality"]) * coefficient
        )
        reconstruction_error = abs(identity_mass - expected_mass)
        valid = reconstruction_error <= 1e-6
        if not valid:
            raise RuntimeError(
                "candidate evidence event quality does not reconstruct: "
                "event={}, error={}".format(
                    event_key,
                    reconstruction_error,
                )
            )
        for row in event_rows:
            row["quality_reconstruction_error"] = reconstruction_error
            row["quality_contract_valid"] = int(valid)

    if not np.allclose(
            reconstructed_delta,
            delta,
            rtol=0.0,
            atol=1e-6):
        raise RuntimeError(
            "candidate evidence provenance rows do not reconstruct the final "
            "candidate target tensor"
        )
    return rows


class AdjBuffer(object):
    def __init__(self, policy_info, policy_agents, num_factor, buffer_size, episode_length, use_same_share_obs,
                 use_avail_acts, use_reward_normalization=False, gamma=0.97, gae_lambda=0.95, hidden_size=64,
                 adj_return_adv_coef=1.0, adj_factor_adv_coef=0.0, seed=0,
                 use_adj_delayed_triplet_credit=False,
                 adj_delayed_triplet_credit_coef=0.0,
                 adj_delayed_triplet_credit_window=0,
                 adj_delayed_triplet_credit_cap=0.0,
                 adj_delayed_triplet_credit_min_reward=0.0,
                 adj_delayed_triplet_credit_positive_only=False,
                 adj_delayed_triplet_credit_min_adv=0.0,
                 adj_delayed_triplet_credit_require_future_match=False,
                 use_adj_delayed_triplet_success_gate=False,
                 adj_delayed_triplet_success_gate_min_adv=0.0,
                 adj_delayed_triplet_success_gate_scale=1.0,
                 adj_delayed_triplet_success_gate_floor=0.0,
                 adj_delayed_triplet_future_overlap_min_nodes=3,
                 adj_delayed_triplet_partial_match_weight=0.5,
                 use_adj_capture_to_win_credit=False,
                 adj_capture_to_win_credit_coef=0.0,
                 adj_capture_to_win_credit_min_outcome_adv=0.5,
                 adj_capture_to_win_credit_scale=0.75,
                 adj_capture_to_win_credit_cap=0.35,
                 adj_capture_to_win_credit_require_future_match=False,
                 use_adj_pair_triplet_complementary_credit=False,
                 adj_pair_pursuit_credit_coef=0.0,
                 adj_pair_pursuit_credit_window=20,
                 adj_pair_pursuit_credit_cap=0.20,
                 adj_pair_pursuit_credit_min_reward=0.0,
                 pair_bounded_pending_evidence=False,
                 pair_pending_max_adj_updates=0):
        """
        Replay buffer class for training RNN policies. Stores entire episodes rather than single transitions.

        :param policy_info: (dict) maps policy id to a dict containing information about corresponding policy.
        :param policy_agents: (dict) maps policy id to list of agents controled by corresponding policy.
        :param buffer_size: (int) max number of transitions to store in the buffer.
        :param use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :param use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.policy_info = policy_info
        self.rng = np.random.RandomState(int(seed))

        self.policy_buffers = {
            p_id: AdjPolicyBuffer(
                buffer_size,
                episode_length,
                len(policy_agents[p_id]),
                num_factor,
                self.policy_info[p_id]['obs_space'],
                self.policy_info[p_id]['share_obs_space'],
                self.policy_info[p_id]['act_space'],
                use_same_share_obs,
                use_avail_acts,
                use_reward_normalization,
                gamma,
                gae_lambda,
                hidden_size,
                adj_return_adv_coef,
                adj_factor_adv_coef,
                seed=int(seed) + 1000 + i,
                use_adj_delayed_triplet_credit=use_adj_delayed_triplet_credit,
                adj_delayed_triplet_credit_coef=adj_delayed_triplet_credit_coef,
                adj_delayed_triplet_credit_window=adj_delayed_triplet_credit_window,
                adj_delayed_triplet_credit_cap=adj_delayed_triplet_credit_cap,
                adj_delayed_triplet_credit_min_reward=adj_delayed_triplet_credit_min_reward,
                adj_delayed_triplet_credit_positive_only=adj_delayed_triplet_credit_positive_only,
                adj_delayed_triplet_credit_min_adv=adj_delayed_triplet_credit_min_adv,
                adj_delayed_triplet_credit_require_future_match=adj_delayed_triplet_credit_require_future_match,
                use_adj_delayed_triplet_success_gate=use_adj_delayed_triplet_success_gate,
                adj_delayed_triplet_success_gate_min_adv=adj_delayed_triplet_success_gate_min_adv,
                adj_delayed_triplet_success_gate_scale=adj_delayed_triplet_success_gate_scale,
                adj_delayed_triplet_success_gate_floor=adj_delayed_triplet_success_gate_floor,
                adj_delayed_triplet_future_overlap_min_nodes=adj_delayed_triplet_future_overlap_min_nodes,
                adj_delayed_triplet_partial_match_weight=adj_delayed_triplet_partial_match_weight,
                use_adj_capture_to_win_credit=use_adj_capture_to_win_credit,
                adj_capture_to_win_credit_coef=adj_capture_to_win_credit_coef,
                adj_capture_to_win_credit_min_outcome_adv=adj_capture_to_win_credit_min_outcome_adv,
                adj_capture_to_win_credit_scale=adj_capture_to_win_credit_scale,
                adj_capture_to_win_credit_cap=adj_capture_to_win_credit_cap,
                adj_capture_to_win_credit_require_future_match=adj_capture_to_win_credit_require_future_match,
                use_adj_pair_triplet_complementary_credit=use_adj_pair_triplet_complementary_credit,
                adj_pair_pursuit_credit_coef=adj_pair_pursuit_credit_coef,
                adj_pair_pursuit_credit_window=adj_pair_pursuit_credit_window,
                adj_pair_pursuit_credit_cap=adj_pair_pursuit_credit_cap,
                adj_pair_pursuit_credit_min_reward=adj_pair_pursuit_credit_min_reward,
                pair_bounded_pending_evidence=(
                    pair_bounded_pending_evidence
                ),
                pair_pending_max_adj_updates=(
                    pair_pending_max_adj_updates
                ),
                policy_id=p_id,
            )
            for i, p_id in enumerate(self.policy_info.keys())
        }

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,
               prob_adj=None, q_tot=None, f_v=None, f_q=None, rnn_states=None,
               capture_counts=None, success_now=None,
               capture_factor_matches=None,
               capture_candidate_only_matches=None,
               capture_candidate_behavior=None,
               capture_identity_candidates=None,
               capture_candidate_event_provenance=None,
               capture_active_event_provenance=None,
               environment_episode_ids=None,
               behavior_policy_versions=None):
        """
        Insert a set of episodes into buffer. If the buffer size overflows, old episodes are dropped.

        :param num_insert_episodes: (int) number of episodes to be added to buffer
        :param obs: (dict) maps policy id to numpy array of observations of agents corresponding to that policy
        :param share_obs: (dict) maps policy id to numpy array of centralized observation corresponding to that policy
        :param acts: (dict) maps policy id to numpy array of actions of agents corresponding to that policy
        :param rewards: (dict) maps policy id to numpy array of rewards of agents corresponding to that policy
        :param dones: (dict) maps policy id to numpy array of terminal status of agents corresponding to that policy
        :param dones_env: (dict) maps policy id to numpy array of terminal status of env
        :param valid_transition: (dict) maps policy id to numpy array of whether the corresponding transition is valid of agents corresponding to that policy
        :param avail_acts: (dict) maps policy id to numpy array of available actions of agents corresponding to that policy

        :return: (np.ndarray) indexes in which the new transitions were placed.
        """
        for p_id in self.policy_info.keys():
            policy_dones_env = np.array(dones_env[p_id])
            policy_capture_counts = (
                capture_counts.get(p_id)
                if isinstance(capture_counts, dict)
                else capture_counts
            )
            policy_success_now = (
                success_now.get(p_id)
                if isinstance(success_now, dict)
                else success_now
            )
            policy_capture_factor_matches = (
                capture_factor_matches.get(p_id)
                if isinstance(capture_factor_matches, dict)
                else capture_factor_matches
            )
            policy_capture_identity_candidates = (
                capture_identity_candidates.get(p_id)
                if isinstance(capture_identity_candidates, dict)
                else capture_identity_candidates
            )
            policy_capture_candidate_only_matches = (
                capture_candidate_only_matches.get(p_id)
                if isinstance(capture_candidate_only_matches, dict)
                else capture_candidate_only_matches
            )
            policy_capture_candidate_behavior = (
                capture_candidate_behavior.get(p_id)
                if isinstance(capture_candidate_behavior, dict)
                else capture_candidate_behavior
            )
            policy_candidate_event_provenance = (
                capture_candidate_event_provenance.get(p_id)
                if isinstance(capture_candidate_event_provenance, dict)
                else capture_candidate_event_provenance
            )
            policy_active_event_provenance = (
                capture_active_event_provenance.get(p_id)
                if isinstance(capture_active_event_provenance, dict)
                else capture_active_event_provenance
            )
            policy_behavior_versions = (
                behavior_policy_versions.get(p_id)
                if isinstance(behavior_policy_versions, dict)
                else behavior_policy_versions
            )
            if policy_capture_counts is None:
                policy_capture_counts = np.zeros_like(
                    policy_dones_env,
                    dtype=np.float32,
                )
            if policy_success_now is None:
                policy_success_now = np.zeros_like(
                    policy_dones_env,
                    dtype=np.float32,
                )
            idx_range = self.policy_buffers[p_id].insert(num_insert_episodes, np.array(obs[p_id]),
                                                         np.array(share_obs[p_id]), np.array(acts[p_id]),
                                                         np.array(rewards[p_id]), np.array(dones[p_id]),
                                                         policy_dones_env, np.array(avail_acts[p_id]),
                                                         np.array(adj[p_id]), np.array(prob_adj[p_id]),
                                                         np.array(q_tot[p_id]),
                                                         np.array(f_v[p_id]), np.array(f_q[p_id]),
                                                         np.array(rnn_states[p_id]),
                                                         np.asarray(policy_capture_counts),
                                                         np.asarray(policy_success_now),
                                                          policy_capture_factor_matches,
                                                          policy_capture_candidate_only_matches,
                                                          policy_capture_candidate_behavior,
                                                          policy_capture_identity_candidates,
                                                          policy_candidate_event_provenance,
                                                          policy_active_event_provenance,
                                                          environment_episode_ids,
                                                          policy_behavior_versions)
        return idx_range

    '''def sample(self, batch_size,data_chunk_length,num_mini_batch):
        """
        Sample a set of episodes from buffer, uniformly at random.
        :param batch_size: (int) number of episodes to sample from buffer.

        :return: obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: share_obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: acts: (dict) maps policy id to sampled actions corresponding to that policy
        :return: rewards: (dict) maps policy id to sampled rewards corresponding to that policy
        :return: dones: (dict) maps policy id to sampled terminal status of agents corresponding to that policy
        :return: dones_env: (dict) maps policy id to sampled environment terminal status corresponding to that policy
        :return: valid_transition: (dict) maps policy_id to whether each sampled transition is valid or not (invalid if corresponding agent is dead)
        :return: avail_acts: (dict) maps policy_id to available actions corresponding to that policy
        """
        replace = self.__len__() < batch_size
        inds = self.rng.choice(self.__len__(), batch_size, replace=replace)
        obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs_batch[p_id], share_obs_batch[p_id], dones_batch[p_id], dones_env_batch[p_id], adj_batch[p_id], prob_adj_batch[p_id], advantages_batch[p_id], f_advts_batch[p_id], rnn_obs_batch[p_id] = self.policy_buffers[p_id].sample_inds(inds,data_chunk_length,num_mini_batch)

        return obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch'''

    def compute_advantage(self, idx, value_normalizer=None):

        for p_id in self.policy_info.keys():
            self.policy_buffers[p_id].compute_advantage(idx)
        return idx

    def set_pair_pending_clock(
            self,
            adjacency_update_index,
            behavior_policy_version):
        for policy_buffer in self.policy_buffers.values():
            policy_buffer.set_pair_pending_clock(
                adjacency_update_index,
                behavior_policy_version,
            )

    def prepare_pair_pending_training_batches(self, expected_ppo_epochs):
        return {
            policy_id: policy_buffer.prepare_pair_pending_training_batch(
                expected_ppo_epochs
            )
            for policy_id, policy_buffer in self.policy_buffers.items()
        }

    def pair_pending_state_dict(self):
        return {
            "version": 1,
            "policies": {
                policy_id: policy_buffer.pair_pending_state_dict()
                for policy_id, policy_buffer in self.policy_buffers.items()
            },
        }

    def load_pair_pending_state_dict(self, state):
        if int(state.get("version", -1)) != 1:
            raise RuntimeError(
                "unsupported adjacency pair pending checkpoint version"
            )
        policies = state.get("policies", {})
        if set(policies) != set(self.policy_buffers):
            raise RuntimeError(
                "pair pending checkpoint policy set mismatch"
            )
        for policy_id, policy_buffer in self.policy_buffers.items():
            policy_buffer.load_pair_pending_state_dict(
                policies[policy_id]
            )

    def pair_pending_update_diagnostics(self):
        return {
            policy_id: policy_buffer.pair_pending_update_diagnostics()
            for policy_id, policy_buffer in self.policy_buffers.items()
        }


class AdjPolicyBuffer(object):
    def __init__(self, buffer_size, episode_length, num_agents, num_factor, obs_space, share_obs_space, act_space,
                 use_same_share_obs, use_avail_acts, use_reward_normalization=False, gamma=0.97, gae_lambda=0.95,
                 hidden_size=64, adj_return_adv_coef=1.0, adj_factor_adv_coef=0.0, seed=0,
                 use_adj_delayed_triplet_credit=False,
                 adj_delayed_triplet_credit_coef=0.0,
                 adj_delayed_triplet_credit_window=0,
                 adj_delayed_triplet_credit_cap=0.0,
                 adj_delayed_triplet_credit_min_reward=0.0,
                 adj_delayed_triplet_credit_positive_only=False,
                 adj_delayed_triplet_credit_min_adv=0.0,
                 adj_delayed_triplet_credit_require_future_match=False,
                 use_adj_delayed_triplet_success_gate=False,
                 adj_delayed_triplet_success_gate_min_adv=0.0,
                 adj_delayed_triplet_success_gate_scale=1.0,
                 adj_delayed_triplet_success_gate_floor=0.0,
                 adj_delayed_triplet_future_overlap_min_nodes=3,
                 adj_delayed_triplet_partial_match_weight=0.5,
                 use_adj_capture_to_win_credit=False,
                 adj_capture_to_win_credit_coef=0.0,
                 adj_capture_to_win_credit_min_outcome_adv=0.5,
                 adj_capture_to_win_credit_scale=0.75,
                 adj_capture_to_win_credit_cap=0.35,
                 adj_capture_to_win_credit_require_future_match=False,
                 use_adj_pair_triplet_complementary_credit=False,
                 adj_pair_pursuit_credit_coef=0.0,
                 adj_pair_pursuit_credit_window=20,
                 adj_pair_pursuit_credit_cap=0.20,
                 adj_pair_pursuit_credit_min_reward=0.0,
                 pair_bounded_pending_evidence=False,
                 pair_pending_max_adj_updates=0,
                 policy_id="policy_0"):
        """
        Buffer class containing buffer data corresponding to a single policy.

        :param buffer_size: (int) max number of episodes to store in buffer.
        :param episode_length: (int) max length of an episode.
        :param num_agents: (int) number of agents controlled by the policy.
        :param obs_space: (gym.Space) observation space of the environment.
        :param share_obs_space: (gym.Space) centralized observation space of the environment.
        :param act_space: (gym.Space) action space of the environment.
        :use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.num_factor = num_factor
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0
        # Stable slot generations distinguish a resident episode from a later
        # episode that overwrites the same circular-buffer slot. Outcome
        # contrast support is consumable across adjacency updates: a sparse
        # historical episode may supplement one update, but must not be
        # repeatedly replayed as the missing class in every later update.
        self.episode_generation = np.full(
            self.buffer_size,
            -1,
            dtype=np.int64,
        )
        self.environment_episode_id = np.full(
            self.buffer_size,
            -1,
            dtype=np.int64,
        )
        self.episode_behavior_policy_version = np.full(
            self.buffer_size,
            -1,
            dtype=np.int64,
        )
        self.capture_candidate_event_provenance = [
            tuple() for _ in range(self.buffer_size)
        ]
        self.capture_active_event_provenance = [
            tuple() for _ in range(self.buffer_size)
        ]
        self.strict_pair_event_provenance = [
            tuple() for _ in range(self.buffer_size)
        ]
        self.outcome_support_used = np.zeros(
            self.buffer_size,
            dtype=bool,
        )
        self._next_episode_generation = 0
        self._next_outcome_support_round = 0
        self._cached_outcome_support_round = None
        self._cached_outcome_support_signature = None
        self._cached_outcome_support_selection = None
        self.outcome_generation_update_count = 0
        self.outcome_slot_overwrite_count = 0
        self.outcome_generation_conflict_count = 0
        self.outcome_invalid_used_state_count = 0
        self.last_sample_candidate_evidence_provenance_rows = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_size = hidden_size
        self.adj_return_adv_coef = float(adj_return_adv_coef)
        self.adj_factor_adv_coef = float(adj_factor_adv_coef)
        self.use_adj_delayed_triplet_credit = bool(
            use_adj_delayed_triplet_credit
        )
        self.adj_delayed_triplet_credit_coef = max(
            0.0,
            float(adj_delayed_triplet_credit_coef),
        )
        self.adj_delayed_triplet_credit_window = max(
            0,
            int(adj_delayed_triplet_credit_window),
        )
        self.adj_delayed_triplet_credit_cap = max(
            0.0,
            float(adj_delayed_triplet_credit_cap),
        )
        self.adj_delayed_triplet_credit_min_reward = float(
            adj_delayed_triplet_credit_min_reward
        )
        self.adj_delayed_triplet_credit_positive_only = bool(
            adj_delayed_triplet_credit_positive_only
        )
        self.adj_delayed_triplet_credit_min_adv = max(
            0.0,
            float(adj_delayed_triplet_credit_min_adv),
        )
        self.adj_delayed_triplet_credit_require_future_match = bool(
            adj_delayed_triplet_credit_require_future_match
        )
        self.use_adj_delayed_triplet_success_gate = bool(
            use_adj_delayed_triplet_success_gate
        )
        self.adj_delayed_triplet_success_gate_min_adv = float(
            adj_delayed_triplet_success_gate_min_adv
        )
        self.adj_delayed_triplet_success_gate_scale = max(
            1e-6,
            float(adj_delayed_triplet_success_gate_scale),
        )
        self.adj_delayed_triplet_success_gate_floor = min(
            1.0,
            max(0.0, float(adj_delayed_triplet_success_gate_floor)),
        )
        self.adj_delayed_triplet_future_overlap_min_nodes = min(
            3,
            max(1, int(adj_delayed_triplet_future_overlap_min_nodes)),
        )
        self.adj_delayed_triplet_partial_match_weight = min(
            1.0,
            max(0.0, float(adj_delayed_triplet_partial_match_weight)),
        )
        self.use_adj_capture_to_win_credit = bool(
            use_adj_capture_to_win_credit
        )
        self.adj_capture_to_win_credit_coef = max(
            0.0,
            float(adj_capture_to_win_credit_coef),
        )
        self.adj_capture_to_win_credit_min_outcome_adv = float(
            adj_capture_to_win_credit_min_outcome_adv
        )
        self.adj_capture_to_win_credit_scale = max(
            1e-6,
            float(adj_capture_to_win_credit_scale),
        )
        self.adj_capture_to_win_credit_cap = max(
            0.0,
            float(adj_capture_to_win_credit_cap),
        )
        self.adj_capture_to_win_credit_require_future_match = bool(
            adj_capture_to_win_credit_require_future_match
        )
        self.use_adj_pair_triplet_complementary_credit = bool(
            use_adj_pair_triplet_complementary_credit
        )
        self.adj_pair_pursuit_credit_coef = max(
            0.0,
            float(adj_pair_pursuit_credit_coef),
        )
        self.adj_pair_pursuit_credit_window = max(
            1,
            int(adj_pair_pursuit_credit_window),
        )
        self.adj_pair_pursuit_credit_cap = max(
            0.0,
            float(adj_pair_pursuit_credit_cap),
        )
        self.adj_pair_pursuit_credit_min_reward = float(
            adj_pair_pursuit_credit_min_reward
        )
        self.pair_pending_store = PairPendingEvidenceStore(
            enabled=bool(pair_bounded_pending_evidence),
            max_adj_updates=int(pair_pending_max_adj_updates),
        )
        self.pair_pending_enabled = bool(pair_bounded_pending_evidence)
        self.pair_pending_max_adj_updates = int(
            pair_pending_max_adj_updates
        )
        self.pair_pending_current_adj_update = 0
        self.pair_pending_behavior_policy_version = 0
        self.policy_id = str(policy_id)
        self.pair_pending_new_snapshot_count = 0
        self.pair_pending_expired_ttl_count = 0
        self.pair_pending_payload_contract_valid = 1.0
        self.pair_pending_counter_update = -1
        self.pair_pending_prepared_count = 0
        self.pair_pending_aborted_count = 0
        self.pair_pending_rolled_back_count = 0
        self.pair_pending_committed_count = 0
        self.pair_pending_class_complete_count = 0
        self.pair_pending_pair_only_transaction_count = 0
        self.pair_pending_zero_target_abort_count = 0
        self.pair_pending_zero_gradient_abort_count = 0
        self.pair_pending_early_stop_abort_count = 0
        self.pair_pending_expired_provenance_count = 0
        self.pair_pending_expired_population_mismatch_count = 0
        self.pair_pending_stale_contract_valid = 1.0
        self.pair_pending_mass_contract_valid = 1.0
        self.pair_pending_objective_scope_contract_valid = 1.0
        self.pair_pending_atomic_rollback_contract_valid = 1.0
        self.pair_pending_checkpoint_contract_valid = 1.0
        self.pair_pending_last_positive_stale_trust = 0.0
        self.pair_pending_last_negative_stale_trust = 0.0
        self.pair_pending_last_raw_positive_mass = 0.0
        self.pair_pending_last_raw_negative_mass = 0.0
        self.pair_pending_last_effective_positive_mass = 0.0
        self.pair_pending_last_effective_negative_mass = 0.0

        self.rng = np.random.RandomState(int(seed))
        # obs
        if obs_space.__class__.__name__ == 'Box':
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == 'list':
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros((self.episode_length + 1, self.buffer_size,
                             self.num_agents, obs_shape[0]), dtype=np.float32)

        if self.use_same_share_obs:
            self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        else:
            self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, share_obs_shape[0]),
                                      dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros((self.episode_length, self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones((self.episode_length + 1, self.buffer_size, self.num_agents, act_dim),
                                      dtype=np.float32)

        # rewards
        self.rewards = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.capture_counts = np.zeros_like(self.dones_env, dtype=np.float32)
        self.success_now = np.zeros_like(self.dones_env, dtype=np.float32)
        self.capture_factor_matches = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                self.num_factor,
            ),
            dtype=np.float32,
        )
        self.num_candidate_factor = len(
            canonical_capture_factor_catalog(self.num_agents, 3)
        )
        self.capture_candidate_only_matches = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                self.num_candidate_factor,
            ),
            dtype=np.float32,
        )
        self.capture_candidate_behavior = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                self.num_candidate_factor,
                4,
            ),
            dtype=np.float32,
        )
        self.capture_identity_candidates = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_factor),
                            dtype=np.int64)
        self.prob_adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_factor),
                                 dtype=np.float32)
        self.qtot = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.margin_q = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.f_q = np.zeros((self.episode_length, self.buffer_size, self.num_factor + self.num_agents, 1),
                            dtype=np.float32)
        self.advantage = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        # A computed graph advantage may legitimately be zero. Track whether
        # each value was actually written so an unwritten replay default cannot
        # silently erase a class-complete outcome cohort.
        self.graph_advantage_ready = np.zeros_like(
            self.advantage,
            dtype=bool,
        )
        self.f_v = np.zeros((self.episode_length, self.buffer_size, self.num_factor + self.num_agents, 1),
                            dtype=np.float32)
        self.f_advt = np.zeros((self.episode_length, self.buffer_size, self.num_factor, 1), dtype=np.float32)
        self.delayed_triplet_credit = np.zeros_like(self.f_advt)
        self.delayed_triplet_success_gate = np.zeros_like(self.f_advt)
        self.delayed_triplet_future_match = np.zeros_like(self.f_advt)
        self.delayed_triplet_future_exact = np.zeros_like(self.f_advt)
        self.delayed_triplet_future_partial = np.zeros_like(self.f_advt)
        self.capture_to_win_triplet_credit = np.zeros_like(self.f_advt)
        self.capture_to_win_quality_gate = np.zeros_like(self.f_advt)
        self.pair_pursuit_credit = np.zeros_like(self.f_advt)
        self.pair_pursuit_quality = np.zeros_like(self.f_advt)
        self.pair_to_triplet_transition_score = np.zeros_like(self.f_advt)
        self.pair_transition_delay = np.zeros_like(self.f_advt)
        self.triplet_capture_quality = np.zeros_like(self.f_advt)
        self.capture_matched_count = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.capture_to_win_episode_success_gate = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.failed_episode_capture_count = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.capture_outcome_diagnostics = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
            ),
            dtype=np.float32,
        )
        self.positive_reward_step = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.positive_reward_without_capture = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.offset0_candidate_count = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.rnn_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.hidden_size),
                                dtype=np.float32)

    def __len__(self):
        return self.filled_i

    def compute_advantage(self, idx, value_normalizer=None):
        """
        Build graph-policy credit from the trajectory that the sampled graph
        actually produced.

        The old implementation recursively accumulated ``factor_q-factor_v``
        only when exactly the same node set appeared at the next timestep.
        That is not a temporal-difference residual, and it drops the credit
        chain exactly at join/leave/recover events where the topology changes.

        The primary signal below is a discounted team return with a
        timestep-and-roster baseline over the recent adjacency buffer.  It is
        valid across topology changes and credits the sampled graph using real
        rewards.  A centered factor Q-V term is retained only as a lower-weight
        auxiliary signal to distinguish factors selected at the same step.
        """
        requested_idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if requested_idx.size == 0 or self.filled_i <= 0:
            return idx

        # Every occupied slot from 0..filled_i-1 is valid, including after the
        # circular buffer has become full. Recompute all occupied episodes so
        # their return baselines stay consistent as the recent buffer changes.
        reference_idx = np.arange(self.filled_i, dtype=np.int64)
        episode_idx = reference_idx
        self.f_advt[:, episode_idx] = 0.0
        self.delayed_triplet_credit[:, episode_idx] = 0.0
        self.delayed_triplet_success_gate[:, episode_idx] = 0.0
        self.delayed_triplet_future_match[:, episode_idx] = 0.0
        self.delayed_triplet_future_exact[:, episode_idx] = 0.0
        self.delayed_triplet_future_partial[:, episode_idx] = 0.0
        self.capture_to_win_triplet_credit[:, episode_idx] = 0.0
        self.capture_to_win_quality_gate[:, episode_idx] = 0.0
        self.pair_pursuit_credit[:, episode_idx] = 0.0
        self.pair_pursuit_quality[:, episode_idx] = 0.0
        self.pair_to_triplet_transition_score[:, episode_idx] = 0.0
        self.pair_transition_delay[:, episode_idx] = 0.0
        self.triplet_capture_quality[:, episode_idx] = 0.0
        self.capture_matched_count[:, episode_idx] = 0.0
        self.capture_to_win_episode_success_gate[:, episode_idx] = 0.0
        self.failed_episode_capture_count[:, episode_idx] = 0.0
        self.capture_outcome_diagnostics[:, episode_idx] = 0.0
        self.positive_reward_step[:, episode_idx] = 0.0
        self.positive_reward_without_capture[:, episode_idx] = 0.0
        self.offset0_candidate_count[:, episode_idx] = 0.0
        obs_ref = np.take(
            self.obs[:-1],
            reference_idx,
            axis=1,
        )
        active_ref = ~np.all(
            obs_ref <= -0.999,
            axis=-1,
        )
        active_count_ref = active_ref.sum(axis=-1)

        rewards_ref = np.take(
            self.rewards,
            reference_idx,
            axis=1,
        )[..., 0]
        dones_env_ref = np.take(
            self.dones_env,
            reference_idx,
            axis=1,
        )[..., 0]
        active_den = np.maximum(active_count_ref, 1).astype(np.float32)
        team_rewards_ref = (
            rewards_ref * active_ref.astype(np.float32)
        ).sum(axis=-1) / active_den
        expected_transition_shape = (
            self.episode_length,
            reference_idx.size,
        )
        if team_rewards_ref.shape != expected_transition_shape:
            raise RuntimeError(
                "AdjBuffer transition axes are not [time, episode]: "
                "got {}, expected {}".format(
                    team_rewards_ref.shape,
                    expected_transition_shape,
                )
            )

        returns_ref = np.zeros_like(team_rewards_ref, dtype=np.float32)
        running_return = np.zeros(
            (reference_idx.size,),
            dtype=np.float32,
        )
        for step in reversed(range(self.episode_length)):
            continuation = (
                1.0 - dones_env_ref[step]
            ).astype(np.float32)
            running_return = (
                team_rewards_ref[step]
                + self.gamma * continuation * running_return
            )
            returns_ref[step] = running_return

        # Use recent episodes at the same timestep and roster size as a
        # control variate.  The shock schedule is deterministic in the current
        # experiment, while graph choices and outcomes vary between episodes.
        return_adv_ref = np.zeros_like(returns_ref, dtype=np.float32)
        for step in range(self.episode_length):
            step_rosters = active_count_ref[step]
            for roster_size in np.unique(step_rosters):
                roster_mask = step_rosters == roster_size
                roster_values = returns_ref[step, roster_mask]
                if roster_values.size > 1:
                    baseline = float(roster_values.mean())
                else:
                    # Before the adjacency buffer is populated, do not invent
                    # a return advantage from a single sample.
                    baseline = float(roster_values[0])
                return_adv_ref[step, roster_mask] = (
                    roster_values - baseline
                )

        graph_adv = np.take(
            return_adv_ref,
            episode_idx,
            axis=1,
        )

        current_adj = np.take(
            self.adj[:-1, :, :, :self.num_factor],
            episode_idx,
            axis=1,
        ).astype(np.int64, copy=False)
        current_prob_adj = np.take(
            self.prob_adj[:-1, :, :, :self.num_factor],
            episode_idx,
            axis=1,
        ).astype(np.float32, copy=False)
        current_obs = np.take(
            self.obs[:-1],
            episode_idx,
            axis=1,
        )
        current_active = ~np.all(
            current_obs <= -0.999,
            axis=-1,
        )
        factor_size = current_adj.sum(axis=2)
        selected_factor_behavior_probability = (
            (
                current_prob_adj
                * current_adj.astype(np.float32, copy=False)
            ).sum(axis=2, dtype=np.float32)
            / np.maximum(factor_size, 1).astype(np.float32, copy=False)
        ).astype(np.float32, copy=False)
        factor_alive_count = (
            current_adj * current_active[:, :, :, None]
        ).sum(axis=2)
        # dones_env[t] is post-action termination. The graph/action at t is a
        # valid training transition; only t+1 (padding after an early terminal)
        # must be excluded. This must use the same previous-done convention as
        # the recurrent replay generator below.
        previous_done_ref = build_previous_done_sequence(dones_env_ref)
        valid_factor = (
            (factor_size > 0)
            & (factor_alive_count == factor_size)
            & (previous_done_ref[..., None] < 0.5)
        )

        if self.adj_factor_adv_coef != 0.0:
            local_adv = (
                np.take(
                    self.f_q[:, :, :self.num_factor],
                    episode_idx,
                    axis=1,
                )[..., 0]
                - np.take(
                    self.f_v[:, :, :self.num_factor],
                    episode_idx,
                    axis=1,
                )[..., 0]
            )
            # Factor critics are order-specific networks.  Comparing a pair
            # factor directly with a triplet factor mixes critic scales and in
            # run22 biased the learned graph toward low-order pairs.  Keep the
            # auxiliary credit local to alternatives of the same factor order.
            centered_local_adv = np.zeros_like(local_adv, dtype=np.float32)
            for order in np.unique(factor_size[valid_factor]):
                order_mask = valid_factor & (factor_size == order)
                valid_count = np.maximum(
                    order_mask.sum(axis=-1, keepdims=True),
                    1,
                ).astype(np.float32)
                order_mean = (
                    local_adv * order_mask.astype(np.float32)
                ).sum(axis=-1, keepdims=True) / valid_count
                centered_local_adv += (
                    (local_adv - order_mean)
                    * order_mask.astype(np.float32)
                )
            local_adv = centered_local_adv
        else:
            local_adv = np.zeros_like(valid_factor, dtype=np.float32)

        def _standardize_valid(values, mask):
            valid_values = values[mask]
            if valid_values.size <= 1:
                return np.zeros_like(values, dtype=np.float32)
            mean = float(valid_values.mean())
            std = float(valid_values.std())
            if not np.isfinite(std) or std < 1e-5:
                std = 1.0
            standardized = (values - mean) / (std + 1e-5)
            standardized[~mask] = 0.0
            return standardized.astype(np.float32)

        def _standardize_valid_by_order(values, mask, orders):
            standardized = np.zeros_like(values, dtype=np.float32)
            if not mask.any():
                return standardized
            for order in np.unique(orders[mask]):
                order_mask = mask & (orders == order)
                order_standardized = _standardize_valid(values, order_mask)
                standardized[order_mask] = order_standardized[order_mask]
            return standardized.astype(np.float32)

        # A graph is one structured action per transition. Normalize its
        # return advantage once per transition rather than once per selected
        # factor; otherwise large rosters receive more statistical weight only
        # because they use more factor slots.
        valid_graph_transition = valid_factor.any(axis=-1)
        graph_adv = _standardize_valid(
            graph_adv,
            valid_graph_transition,
        )
        stored_graph_advantage = write_graph_advantage_sequence(
            storage=self.advantage,
            ready_storage=self.graph_advantage_ready,
            episode_indices=episode_idx,
            graph_advantage=graph_adv,
            valid_transition=valid_graph_transition,
        )
        graph_advantage_readback = self.advantage[:, episode_idx, 0]
        graph_ready_readback = self.graph_advantage_ready[:, episode_idx, 0]
        if not np.array_equal(
                graph_ready_readback, valid_graph_transition):
            raise RuntimeError(
                "graph advantage ready mask does not match valid transitions"
            )
        storage_error = np.abs(
            graph_advantage_readback - stored_graph_advantage
        )
        if np.any(storage_error > 0.0):
            raise RuntimeError(
                "graph advantage replay write/readback mismatch: max_error={}"
                .format(float(storage_error.max()))
            )
        graph_factor_adv = np.repeat(
            graph_adv[:, :, None],
            self.num_factor,
            axis=2,
        ) * valid_factor.astype(np.float32)
        if self.adj_factor_adv_coef != 0.0:
            local_adv = _standardize_valid_by_order(
                local_adv,
                valid_factor,
                factor_size,
            )
        else:
            local_adv = np.zeros_like(local_adv, dtype=np.float32)

        pair_factor_mask = valid_factor & (factor_size == 2)
        triplet_factor_mask = valid_factor & (factor_size == 3)
        graph_success_gate = np.ones_like(team_rewards_ref, dtype=np.float32)

        delayed_triplet_credit = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_success_gate = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_future_match = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_future_exact = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_future_partial = np.zeros_like(local_adv, dtype=np.float32)
        if (
            self.use_adj_delayed_triplet_credit
            and self.adj_delayed_triplet_credit_coef > 0.0
            and self.adj_delayed_triplet_credit_window > 0
        ):
            reward_signal = (
                team_rewards_ref
                - float(self.adj_delayed_triplet_credit_min_reward)
            ).astype(np.float32, copy=False)
            reward_signal = np.maximum(reward_signal, 0.0)
            window = int(self.adj_delayed_triplet_credit_window)
            success_gate = np.ones_like(reward_signal, dtype=np.float32)
            if self.use_adj_delayed_triplet_success_gate:
                graph_delayed_signal = np.zeros_like(
                    reward_signal,
                    dtype=np.float32,
                )
                for step in range(self.episode_length):
                    discount = 1.0
                    alive = np.ones(reference_idx.size, dtype=np.float32)
                    for offset in range(window):
                        future_step = step + offset
                        if future_step >= self.episode_length:
                            break
                        graph_delayed_signal[step] += (
                            discount
                            * alive
                            * reward_signal[future_step]
                        )
                        alive *= (
                            1.0
                            - dones_env_ref[future_step].astype(np.float32)
                        )
                        discount *= float(self.gamma)

                graph_delayed_adv = np.zeros_like(
                    graph_delayed_signal,
                    dtype=np.float32,
                )
                for step in range(self.episode_length):
                    step_rosters = active_count_ref[step]
                    for roster_size in np.unique(step_rosters):
                        roster_mask = step_rosters == roster_size
                        roster_values = graph_delayed_signal[
                            step,
                            roster_mask,
                        ]
                        if roster_values.size > 1:
                            baseline = float(roster_values.mean())
                        else:
                            baseline = float(roster_values[0])
                        graph_delayed_adv[step, roster_mask] = (
                            roster_values - baseline
                        )
                graph_delayed_adv = _standardize_valid(
                    graph_delayed_adv,
                    valid_graph_transition,
                )
                success_gate = (
                    (
                        graph_delayed_adv
                        - float(self.adj_delayed_triplet_success_gate_min_adv)
                    )
                    / float(self.adj_delayed_triplet_success_gate_scale)
                )
                success_gate = np.clip(success_gate, 0.0, 1.0).astype(
                    np.float32,
                    copy=False,
                )
                if self.adj_delayed_triplet_success_gate_floor > 0.0:
                    gate_floor = float(
                        self.adj_delayed_triplet_success_gate_floor
                    )
                    success_gate = (
                        gate_floor
                        + (1.0 - gate_floor) * success_gate
                    ).astype(np.float32, copy=False)
                success_gate[~valid_graph_transition] = 0.0
            graph_success_gate = success_gate.copy()
            if self.adj_delayed_triplet_credit_require_future_match:
                delayed_signal = np.zeros_like(local_adv, dtype=np.float32)
                triplet_sets = []
                for step in range(self.episode_length):
                    step_sets = []
                    for ep_pos in range(reference_idx.size):
                        factor_sets = []
                        for factor_idx in range(self.num_factor):
                            if not triplet_factor_mask[
                                step,
                                ep_pos,
                                factor_idx,
                            ]:
                                continue
                            nodes = tuple(
                                np.flatnonzero(
                                    current_adj[
                                        step,
                                        ep_pos,
                                        :,
                                        factor_idx,
                                    ] > 0
                                ).tolist()
                            )
                            if len(nodes) == 3:
                                factor_sets.append(nodes)
                        step_sets.append(factor_sets)
                    triplet_sets.append(step_sets)

                min_overlap = int(
                    self.adj_delayed_triplet_future_overlap_min_nodes
                )
                partial_match_weight = float(
                    self.adj_delayed_triplet_partial_match_weight
                )
                for step in range(self.episode_length):
                    for ep_pos in range(reference_idx.size):
                        current_triplets = {}
                        for factor_idx in range(self.num_factor):
                            if not triplet_factor_mask[
                                step,
                                ep_pos,
                                factor_idx,
                            ]:
                                continue
                            nodes = tuple(
                                np.flatnonzero(
                                    current_adj[
                                        step,
                                        ep_pos,
                                        :,
                                        factor_idx,
                                    ] > 0
                                ).tolist()
                            )
                            if len(nodes) == 3:
                                current_triplets[factor_idx] = nodes
                        if not current_triplets:
                            continue
                        discount = 1.0
                        for offset in range(window):
                            future_step = step + offset
                            if future_step >= self.episode_length:
                                break
                            future_reward = float(
                                reward_signal[future_step, ep_pos]
                            )
                            if future_reward > 0.0:
                                future_sets = triplet_sets[
                                    future_step
                                ][ep_pos]
                                for factor_idx, nodes in (
                                    current_triplets.items()
                                ):
                                    match_weight = 0.0
                                    node_set = set(nodes)
                                    for future_nodes in future_sets:
                                        overlap = len(
                                            node_set.intersection(future_nodes)
                                        )
                                        if overlap < min_overlap:
                                            continue
                                        if overlap >= 3:
                                            match_weight = 1.0
                                            break
                                        match_weight = max(
                                            match_weight,
                                            partial_match_weight,
                                        )
                                    if match_weight > 0.0:
                                        delayed_triplet_future_match[
                                            step,
                                            ep_pos,
                                            factor_idx,
                                        ] = max(
                                            delayed_triplet_future_match[
                                                step,
                                                ep_pos,
                                                factor_idx,
                                            ],
                                            match_weight,
                                        )
                                        if match_weight >= 1.0:
                                            delayed_triplet_future_exact[
                                                step,
                                                ep_pos,
                                                factor_idx,
                                            ] = 1.0
                                        else:
                                            delayed_triplet_future_partial[
                                                step,
                                                ep_pos,
                                                factor_idx,
                                            ] = 1.0
                                        delayed_signal[
                                            step,
                                            ep_pos,
                                            factor_idx,
                                        ] += (
                                            discount
                                            * future_reward
                                            * match_weight
                                        )
                            if dones_env_ref[future_step, ep_pos] >= 0.5:
                                break
                            discount *= float(self.gamma)

                delayed_adv = np.zeros_like(delayed_signal, dtype=np.float32)
                for step in range(self.episode_length):
                    step_rosters = active_count_ref[step]
                    for roster_size in np.unique(step_rosters):
                        roster_mask = step_rosters == roster_size
                        factor_mask = (
                            triplet_factor_mask[step]
                            & roster_mask[:, None]
                        )
                        factor_values = delayed_signal[step][factor_mask]
                        if factor_values.size == 0:
                            continue
                        if factor_values.size > 1:
                            baseline = float(factor_values.mean())
                        else:
                            baseline = float(factor_values[0])
                        delayed_adv[step][factor_mask] = (
                            delayed_signal[step][factor_mask]
                            - baseline
                        )
                delayed_adv = _standardize_valid(
                    delayed_adv,
                    triplet_factor_mask,
                )
            else:
                delayed_signal = np.zeros_like(reward_signal, dtype=np.float32)
                for step in range(self.episode_length):
                    discount = 1.0
                    alive = np.ones(reference_idx.size, dtype=np.float32)
                    for offset in range(window):
                        future_step = step + offset
                        if future_step >= self.episode_length:
                            break
                        delayed_signal[step] += (
                            discount
                            * alive
                            * reward_signal[future_step]
                        )
                        alive *= (
                            1.0
                            - dones_env_ref[future_step].astype(np.float32)
                        )
                        discount *= float(self.gamma)
                delayed_adv = np.zeros_like(delayed_signal, dtype=np.float32)
                for step in range(self.episode_length):
                    step_rosters = active_count_ref[step]
                    for roster_size in np.unique(step_rosters):
                        roster_mask = step_rosters == roster_size
                        roster_values = delayed_signal[step, roster_mask]
                        if roster_values.size > 1:
                            baseline = float(roster_values.mean())
                        else:
                            baseline = float(roster_values[0])
                        delayed_adv[step, roster_mask] = (
                            roster_values - baseline
                        )
                delayed_adv = _standardize_valid(
                    delayed_adv,
                    valid_graph_transition,
                )
            min_delayed_adv = float(self.adj_delayed_triplet_credit_min_adv)
            if self.adj_delayed_triplet_credit_positive_only:
                delayed_adv = np.maximum(
                    delayed_adv - min_delayed_adv,
                    0.0,
                )
            elif min_delayed_adv > 0.0:
                delayed_adv = np.where(
                    np.abs(delayed_adv) >= min_delayed_adv,
                    delayed_adv,
                    0.0,
                )
            delayed_triplet_credit = (
                delayed_adv
                if self.adj_delayed_triplet_credit_require_future_match
                else delayed_adv[:, :, None]
            )
            delayed_triplet_success_gate = (
                success_gate[:, :, None]
                * triplet_factor_mask.astype(np.float32)
            )
            delayed_triplet_credit = (
                delayed_triplet_credit
                * delayed_triplet_success_gate
            )
            delayed_triplet_credit = (
                delayed_triplet_credit
                * triplet_factor_mask.astype(np.float32)
                * float(self.adj_delayed_triplet_credit_coef)
            )
            if self.adj_delayed_triplet_credit_cap > 0.0:
                delayed_triplet_credit = np.clip(
                    delayed_triplet_credit,
                    -float(self.adj_delayed_triplet_credit_cap),
                    float(self.adj_delayed_triplet_credit_cap),
                )

        capture_counts_ref = np.take(
            self.capture_counts,
            episode_idx,
            axis=1,
        )[..., 0]
        success_now_ref = np.take(
            self.success_now,
            episode_idx,
            axis=1,
        )[..., 0]
        capture_factor_matches_ref = np.take(
            self.capture_factor_matches,
            episode_idx,
            axis=1,
        )
        capture_candidate_only_matches_ref = np.take(
            self.capture_candidate_only_matches,
            episode_idx,
            axis=1,
        )
        capture_identity_candidates_ref = np.take(
            self.capture_identity_candidates,
            episode_idx,
            axis=1,
        )[..., 0]
        if capture_counts_ref.shape != team_rewards_ref.shape:
            raise RuntimeError(
                "AdjBuffer capture event axes are not [time, episode]: "
                "got {}, expected {}".format(
                    capture_counts_ref.shape,
                    team_rewards_ref.shape,
                )
            )
        if success_now_ref.shape != team_rewards_ref.shape:
            raise RuntimeError(
                "AdjBuffer success event axes are not [time, episode]: "
                "got {}, expected {}".format(
                    success_now_ref.shape,
                    team_rewards_ref.shape,
                )
            )

        outcome_diagnostics = compute_capture_to_win_outcome_gate(
            success_now=success_now_ref,
            capture_counts=capture_counts_ref,
            valid_graph_transition=valid_graph_transition,
        )
        episode_success = outcome_diagnostics["episode_success"]
        capture_to_win_episode_success_gate = outcome_diagnostics[
            "episode_success_gate"
        ]
        failed_episode_capture_count = outcome_diagnostics[
            "failed_episode_capture_count"
        ]

        pair_credit_diagnostics = compute_capture_anchored_pair_credit(
            current_adj=current_adj,
            valid_factor=valid_factor,
            factor_size=factor_size,
            active_agent_count=active_count_ref,
            selected_factor_behavior_probability=(
                selected_factor_behavior_probability
            ),
            capture_counts=capture_counts_ref,
            capture_factor_match=capture_factor_matches_ref,
            valid_graph_transition=valid_graph_transition,
            dones_env=dones_env_ref,
            team_rewards=team_rewards_ref,
            window=self.adj_pair_pursuit_credit_window,
            gamma=self.gamma,
            capture_event_provenance_by_episode=(
                [
                    self.capture_active_event_provenance[int(replay_slot)]
                    for replay_slot in reference_idx.tolist()
                ]
                if self.pair_pending_enabled
                else None
            ),
        )
        strict_pair_event_provenance = pair_credit_diagnostics[
            "strict_pair_event_provenance"
        ]
        if len(strict_pair_event_provenance) != reference_idx.size:
            raise RuntimeError(
                "strict pair event provenance episode axis is misaligned"
            )
        for episode_offset, replay_slot in enumerate(
                reference_idx.tolist()):
            self.strict_pair_event_provenance[int(replay_slot)] = tuple(
                dict(record)
                for record in strict_pair_event_provenance[episode_offset]
            )
        pair_pursuit_credit = np.zeros_like(local_adv, dtype=np.float32)
        pair_pursuit_quality = pair_credit_diagnostics[
            "pair_pursuit_quality"
        ].astype(np.float32, copy=False)
        pair_to_triplet_transition_score = pair_credit_diagnostics[
            "pair_to_triplet_transition_score"
        ].astype(np.float32, copy=False)
        pair_transition_delay = pair_credit_diagnostics[
            "pair_transition_delay"
        ].astype(np.float32, copy=False)
        # Legacy storage/batch field name is retained for compatibility. The
        # value is definition-v5 identity-matched capture-factor quality and
        # may target an exact order-2 pair or an order-3 participant subgraph.
        triplet_capture_quality = pair_credit_diagnostics[
            "capture_factor_quality"
        ].astype(np.float32, copy=False)
        capture_matched_count = pair_credit_diagnostics[
            "capture_matched_count"
        ].astype(np.float32, copy=False)
        positive_reward_step = pair_credit_diagnostics[
            "positive_reward_step"
        ].astype(np.float32, copy=False)
        positive_reward_without_capture = pair_credit_diagnostics[
            "positive_reward_without_capture"
        ].astype(np.float32, copy=False)
        offset0_candidate_count = pair_credit_diagnostics[
            "offset0_candidate_count"
        ].astype(np.float32, copy=False)
        pair_to_triplet_transition_score *= pair_factor_mask.astype(
            np.float32
        )
        if (
            self.use_adj_pair_triplet_complementary_credit
            and self.adj_pair_pursuit_credit_coef > 0.0
        ):
            credit_cap = 0.0
            if self.adj_pair_pursuit_credit_cap > 0.0:
                valid_abs_graph = np.abs(graph_adv[valid_graph_transition])
                graph_abs_scale = (
                    float(valid_abs_graph.mean())
                    if valid_abs_graph.size > 0
                    else 1.0
                )
                if not np.isfinite(graph_abs_scale) or graph_abs_scale < 1e-6:
                    graph_abs_scale = 1.0
                credit_cap = float(
                    graph_abs_scale
                    * float(self.adj_pair_pursuit_credit_cap)
                )
            pair_pursuit_credit = scale_optimizer_cohort_pair_credit(
                pair_transition_score=pair_to_triplet_transition_score,
                episode_success=episode_success,
                coefficient=self.adj_pair_pursuit_credit_coef,
                credit_cap=credit_cap,
            )["credit"]

        capture_to_win_triplet_credit = np.zeros_like(
            local_adv,
            dtype=np.float32,
        )
        capture_to_win_quality_gate = np.zeros_like(
            local_adv,
            dtype=np.float32,
        )
        capture_outcome_diagnostics = np.zeros(
            (
                self.episode_length,
                reference_idx.size,
                CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
            ),
            dtype=np.float32,
        )
        if (
            self.use_adj_capture_to_win_credit
            and self.adj_capture_to_win_credit_coef > 0.0
        ):
            # A capture-heavy failed episode can have a high shaped return.
            # Capture-to-win credit must therefore use the real environment
            # success event, never standardized team reward.
            # Keep pair->capture and capture->win as separate causal stages.
            # The old gate reused delayed positive-return/future-overlap
            # evidence, so capture-heavy failed episodes could still be
            # labelled as capture-to-win.  Compare successful and failed real
            # capture episodes directly so failures are not silently treated
            # as missing data.  Legacy future-match settings remain readable
            # but no longer alter this outcome definition.
            all_identity_capture_quality = np.concatenate(
                [
                    triplet_capture_quality,
                    capture_candidate_only_matches_ref,
                ],
                axis=2,
            )
            contrastive_outcome = (
                compute_capture_to_win_triplet_outcome_advantage(
                    episode_success=episode_success,
                    triplet_capture_quality=all_identity_capture_quality,
                )
            )
            all_identity_outcome_gate = contrastive_outcome[
                "triplet_outcome_advantage"
            ]
            capture_to_win_quality_gate = all_identity_outcome_gate[
                :, :, :self.num_factor
            ]
            capture_episode_count = int(
                contrastive_outcome["capture_episode_count"]
            )
            successful_capture_episode_count = int(
                contrastive_outcome[
                    "successful_capture_episode_count"
                ]
            )
            failed_capture_episode_count = int(
                contrastive_outcome["failed_capture_episode_count"]
            )
            mixed_outcome = float(
                successful_capture_episode_count > 0
                and failed_capture_episode_count > 0
            )
            single_success_outcome = float(
                successful_capture_episode_count > 0
                and failed_capture_episode_count == 0
            )
            single_failure_outcome = float(
                failed_capture_episode_count > 0
                and successful_capture_episode_count == 0
            )
            no_capture_window = float(capture_episode_count == 0)
            global_diagnostics = np.asarray(
                [
                    contrastive_outcome[
                        "capture_episode_success_rate"
                    ],
                    capture_episode_count,
                    successful_capture_episode_count,
                    failed_capture_episode_count,
                    mixed_outcome,
                    single_success_outcome,
                    single_failure_outcome,
                    no_capture_window,
                ],
                dtype=np.float32,
            )
            capture_identity_event_count = capture_counts_ref.sum(axis=0)
            capture_identity_matched_event_count = (
                triplet_capture_quality.sum(axis=(0, 2))
            )
            if np.any(
                    capture_identity_matched_event_count
                    > capture_identity_event_count + 1e-5):
                raise RuntimeError(
                    "identity-matched capture mass exceeds real capture count"
                )
            capture_identity_unmatched_event_count = np.maximum(
                capture_identity_event_count
                - capture_identity_matched_event_count,
                0.0,
            )
            capture_identity_candidate_factor_count = (
                capture_identity_candidates_ref.sum(axis=0)
            )
            episode_label_count = contrastive_outcome[
                "capture_triplet_count_per_episode"
            ]
            episode_gate_total = all_identity_outcome_gate.sum(axis=(0, 2))
            capture_episode_mask = contrastive_outcome[
                "capture_episode_mask"
            ]
            label_gate_correlation = 0.0
            label_gate_correlation_valid = 0.0
            if int(capture_episode_mask.sum()) >= 2:
                corr_labels = episode_label_count[capture_episode_mask]
                corr_gates = episode_gate_total[capture_episode_mask]
                if (
                    float(corr_labels.std()) > 1e-8
                    and float(corr_gates.std()) > 1e-8
                ):
                    label_gate_correlation = float(
                        np.corrcoef(corr_labels, corr_gates)[0, 1]
                    )
                    label_gate_correlation_valid = 1.0
            successful_capture_mask = capture_episode_mask & episode_success
            failed_capture_mask = capture_episode_mask & ~episode_success

            def _masked_mean(values, mask):
                return float(values[mask].mean()) if np.any(mask) else 0.0

            attribution_diagnostics = np.asarray(
                [
                    label_gate_correlation,
                    label_gate_correlation_valid,
                    _masked_mean(episode_label_count, successful_capture_mask),
                    _masked_mean(episode_label_count, failed_capture_mask),
                    _masked_mean(episode_gate_total, successful_capture_mask),
                    _masked_mean(episode_gate_total, failed_capture_mask),
                ],
                dtype=np.float32,
            )
            # Store one diagnostic row per completed episode, never one row
            # per transition. This keeps p_success and class counts truly
            # episode-weighted even when episode lengths differ.
            for ep_pos in range(reference_idx.size):
                valid_steps = np.flatnonzero(
                    valid_graph_transition[:, ep_pos]
                )
                if valid_steps.size == 0:
                    continue
                diagnostic_step = int(valid_steps[0])
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, :8
                ] = global_diagnostics
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 8
                ] = contrastive_outcome[
                    "capture_triplet_count_per_episode"
                ][ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 9
                ] = contrastive_outcome[
                    "episode_outcome_advantage"
                ][ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 10
                ] = 1.0
                capture_episode_mask = contrastive_outcome[
                    "capture_episode_mask"
                ]
                if np.any(capture_episode_mask):
                    window_raw_outcome_mean = float(
                        contrastive_outcome[
                            "episode_outcome_advantage"
                        ][capture_episode_mask].mean()
                    )
                else:
                    window_raw_outcome_mean = 0.0
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 11
                ] = window_raw_outcome_mean
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 12
                ] = float(all_identity_outcome_gate.sum())
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 13
                ] = float(np.abs(all_identity_outcome_gate).sum())
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 24:30
                ] = attribution_diagnostics
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 20
                ] = capture_identity_event_count[ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 21
                ] = capture_identity_matched_event_count[ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 22
                ] = capture_identity_unmatched_event_count[ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 23
                ] = capture_identity_candidate_factor_count[ep_pos]

            scaled_outcome_credit = scale_capture_to_win_outcome_credit(
                triplet_outcome_advantage=capture_to_win_quality_gate,
                graph_advantage=graph_adv,
                valid_graph_transition=valid_graph_transition,
                coefficient=self.adj_capture_to_win_credit_coef,
                cap=self.adj_capture_to_win_credit_cap,
                return_diagnostics=True,
            )
            capture_to_win_triplet_credit = scaled_outcome_credit["credit"]
            capture_outcome_diagnostics[..., 14][valid_graph_transition] = (
                scaled_outcome_credit["preclip_mean"]
            )
            capture_outcome_diagnostics[..., 15][valid_graph_transition] = (
                scaled_outcome_credit["preclip_std"]
            )
            capture_outcome_diagnostics[..., 16][valid_graph_transition] = (
                scaled_outcome_credit["preclip_max"]
            )
            capture_outcome_diagnostics[..., 17][valid_graph_transition] = (
                scaled_outcome_credit["preclip_min"]
            )
            capture_outcome_diagnostics[..., 18][valid_graph_transition] = (
                scaled_outcome_credit["positive_clip_fraction"]
            )
            capture_outcome_diagnostics[..., 19][valid_graph_transition] = (
                scaled_outcome_credit["negative_clip_fraction"]
            )

        combined_adv = (
            self.adj_return_adv_coef * graph_factor_adv
            + self.adj_factor_adv_coef * local_adv
            + delayed_triplet_credit
        )
        combined_adv[~valid_factor] = 0.0
        f_advt_values = self.f_advt[..., 0]
        f_advt_values[:, episode_idx, :] = combined_adv
        delayed_triplet_values = self.delayed_triplet_credit[..., 0]
        delayed_triplet_values[:, episode_idx, :] = delayed_triplet_credit
        delayed_triplet_success_gate_values = (
            self.delayed_triplet_success_gate[..., 0]
        )
        delayed_triplet_success_gate_values[:, episode_idx, :] = (
            delayed_triplet_success_gate
        )
        delayed_triplet_future_match_values = (
            self.delayed_triplet_future_match[..., 0]
        )
        delayed_triplet_future_match_values[:, episode_idx, :] = (
            delayed_triplet_future_match
        )
        delayed_triplet_future_exact_values = (
            self.delayed_triplet_future_exact[..., 0]
        )
        delayed_triplet_future_exact_values[:, episode_idx, :] = (
            delayed_triplet_future_exact
        )
        delayed_triplet_future_partial_values = (
            self.delayed_triplet_future_partial[..., 0]
        )
        delayed_triplet_future_partial_values[:, episode_idx, :] = (
            delayed_triplet_future_partial
        )
        capture_to_win_credit_values = (
            self.capture_to_win_triplet_credit[..., 0]
        )
        capture_to_win_credit_values[:, episode_idx, :] = (
            capture_to_win_triplet_credit
        )
        capture_to_win_quality_gate_values = (
            self.capture_to_win_quality_gate[..., 0]
        )
        capture_to_win_quality_gate_values[:, episode_idx, :] = (
            capture_to_win_quality_gate
        )
        self.capture_outcome_diagnostics[
            :, episode_idx, :
        ] = capture_outcome_diagnostics
        pair_pursuit_credit_values = self.pair_pursuit_credit[..., 0]
        pair_pursuit_credit_values[:, episode_idx, :] = pair_pursuit_credit
        pair_pursuit_quality_values = self.pair_pursuit_quality[..., 0]
        pair_pursuit_quality_values[:, episode_idx, :] = pair_pursuit_quality
        pair_to_triplet_transition_values = (
            self.pair_to_triplet_transition_score[..., 0]
        )
        pair_to_triplet_transition_values[:, episode_idx, :] = (
            pair_to_triplet_transition_score
        )
        pair_transition_delay_values = self.pair_transition_delay[..., 0]
        pair_transition_delay_values[:, episode_idx, :] = pair_transition_delay
        triplet_capture_quality_values = self.triplet_capture_quality[..., 0]
        triplet_capture_quality_values[:, episode_idx, :] = (
            triplet_capture_quality
        )
        capture_matched_count_values = self.capture_matched_count[..., 0]
        capture_matched_count_values[:, episode_idx] = capture_matched_count
        episode_success_gate_values = (
            self.capture_to_win_episode_success_gate[..., 0]
        )
        episode_success_gate_values[:, episode_idx] = (
            capture_to_win_episode_success_gate
        )
        failed_episode_capture_values = (
            self.failed_episode_capture_count[..., 0]
        )
        failed_episode_capture_values[:, episode_idx] = (
            failed_episode_capture_count
        )
        positive_reward_step_values = self.positive_reward_step[..., 0]
        positive_reward_step_values[:, episode_idx] = positive_reward_step
        positive_reward_without_capture_values = (
            self.positive_reward_without_capture[..., 0]
        )
        positive_reward_without_capture_values[:, episode_idx] = (
            positive_reward_without_capture
        )
        offset0_candidate_count_values = self.offset0_candidate_count[..., 0]
        offset0_candidate_count_values[:, episode_idx] = (
            offset0_candidate_count
        )

        return idx

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,
               prob_adj=None, qtot=None, f_v=None, f_q=None, rnn_obs=None,
               capture_counts=None, success_now=None,
               capture_factor_matches=None,
               capture_candidate_only_matches=None,
               capture_candidate_behavior=None,
               capture_identity_candidates=None,
               capture_candidate_event_provenance=None,
               capture_active_event_provenance=None,
               environment_episode_ids=None,
               behavior_policy_versions=None):
        """
        Insert a set of episodes corresponding to this policy into buffer. If the buffer size overflows, old transitions are dropped.

        :param num_insert_steps: (int) number of transitions to be added to buffer
        :param obs: (np.ndarray) observations of agents corresponding to this policy.
        :param share_obs: (np.ndarray) centralized observations of agents corresponding to this policy.
        :param acts: (np.ndarray) actions of agents corresponding to this policy.
        :param rewards: (np.ndarray) rewards of agents corresponding to this policy.
        :param dones: (np.ndarray) terminal status of agents corresponding to this policy.
        :param dones_env: (np.ndarray) environment terminal status.
        :param valid_transition: (np.ndarray) whether each transition is valid or not (invalid if agent was dead during transition)
        :param avail_acts: (np.ndarray) available actions of agents corresponding to this policy.

        :return: (np.ndarray) indexes of the buffer the new transitions were placed in.
        """

        # obs: [step, episode, agent, dim]

        episode_length = acts.shape[0]
        assert episode_length == self.episode_length, ("different dimension!")

        if behavior_policy_versions is None:
            behavior_policy_versions = np.zeros(
                int(num_insert_episodes), dtype=np.int64
            )
        else:
            behavior_policy_versions = np.asarray(
                behavior_policy_versions, dtype=np.int64
            )
            if behavior_policy_versions.ndim == 0:
                behavior_policy_versions = np.full(
                    int(num_insert_episodes),
                    int(behavior_policy_versions),
                    dtype=np.int64,
                )
            else:
                behavior_policy_versions = (
                    behavior_policy_versions.reshape(-1)
                )
        if (
                behavior_policy_versions.shape
                != (int(num_insert_episodes),)
                or np.any(behavior_policy_versions < 0)):
            raise ValueError(
                "behavior policy versions must contain one non-negative "
                "version per inserted episode"
            )

        expected_event_shape = (
            self.episode_length,
            num_insert_episodes,
            1,
        )
        if capture_counts is None:
            capture_counts = np.zeros(expected_event_shape, dtype=np.float32)
        if success_now is None:
            success_now = np.zeros(expected_event_shape, dtype=np.float32)
        capture_counts = np.asarray(capture_counts, dtype=np.float32)
        success_now = np.asarray(success_now, dtype=np.float32)
        if capture_counts.ndim == 2:
            capture_counts = capture_counts[..., None]
        if success_now.ndim == 2:
            success_now = success_now[..., None]
        if capture_counts.shape != expected_event_shape:
            raise ValueError(
                "capture_counts must have shape {}, got {}".format(
                    expected_event_shape,
                    capture_counts.shape,
                )
            )
        if success_now.shape != expected_event_shape:
            raise ValueError(
                "success_now must have shape {}, got {}".format(
                    expected_event_shape,
                    success_now.shape,
                )
            )
        capture_counts = np.where(
            np.isfinite(capture_counts),
            np.maximum(capture_counts, 0.0),
            0.0,
        ).astype(np.float32, copy=False)
        success_now = (
            np.isfinite(success_now) & (success_now > 0.0)
        ).astype(np.float32)
        expected_factor_match_shape = (
            self.episode_length,
            num_insert_episodes,
            self.num_factor,
        )
        if capture_factor_matches is None:
            if self.use_adj_capture_to_win_credit:
                raise RuntimeError(
                    "capture-to-win credit requires identity-matched "
                    "capture_factor_matches"
                )
            capture_factor_matches = np.zeros(
                expected_factor_match_shape,
                dtype=np.float32,
            )
        capture_factor_matches = np.asarray(
            capture_factor_matches,
            dtype=np.float32,
        )
        if capture_factor_matches.shape != expected_factor_match_shape:
            raise ValueError(
                "capture_factor_matches must have shape {}, got {}".format(
                    expected_factor_match_shape,
                    capture_factor_matches.shape,
                )
            )
        if not np.isfinite(capture_factor_matches).all() or np.any(
                capture_factor_matches < 0.0):
            raise ValueError(
                "capture_factor_matches must be finite and non-negative"
            )
        expected_candidate_match_shape = (
            self.episode_length,
            num_insert_episodes,
            self.num_candidate_factor,
        )
        if capture_candidate_only_matches is None:
            if self.use_adj_capture_to_win_credit:
                raise RuntimeError(
                    "capture-to-win credit requires exact candidate-only "
                    "identity matches"
                )
            capture_candidate_only_matches = np.zeros(
                expected_candidate_match_shape,
                dtype=np.float32,
            )
        capture_candidate_only_matches = np.asarray(
            capture_candidate_only_matches,
            dtype=np.float32,
        )
        if capture_candidate_only_matches.shape != expected_candidate_match_shape:
            raise ValueError(
                "capture_candidate_only_matches must have shape {}, got {}"
                .format(
                    expected_candidate_match_shape,
                    capture_candidate_only_matches.shape,
                )
            )
        if (
                not np.isfinite(capture_candidate_only_matches).all()
                or np.any(capture_candidate_only_matches < 0.0)):
            raise ValueError(
                "capture_candidate_only_matches must be finite and "
                "non-negative"
            )
        expected_candidate_behavior_shape = (
            self.episode_length,
            num_insert_episodes,
            self.num_candidate_factor,
            4,
        )
        if capture_candidate_behavior is None:
            if self.use_adj_capture_to_win_credit:
                raise RuntimeError(
                    "capture-to-win candidate supervision requires rollout "
                    "candidate active-replacement-logit/rank/mask/policy-version "
                    "metadata"
                )
            capture_candidate_behavior = np.zeros(
                expected_candidate_behavior_shape,
                dtype=np.float32,
            )
        capture_candidate_behavior = np.asarray(
            capture_candidate_behavior,
            dtype=np.float32,
        )
        if capture_candidate_behavior.shape != expected_candidate_behavior_shape:
            raise ValueError(
                "capture_candidate_behavior must have shape {}, got {}"
                .format(
                    expected_candidate_behavior_shape,
                    capture_candidate_behavior.shape,
                )
            )
        if not np.isfinite(capture_candidate_behavior).all():
            raise ValueError("capture_candidate_behavior must be finite")
        behavior_rank = capture_candidate_behavior[..., 1]
        behavior_valid = capture_candidate_behavior[..., 2]
        behavior_version = capture_candidate_behavior[..., 3]
        if (
                np.any(behavior_rank < 0.0)
                or np.any(behavior_version < 0.0)
                or np.any((behavior_valid != 0.0) & (behavior_valid != 1.0))):
            raise ValueError(
                "candidate behavior rank/version must be non-negative and "
                "valid_mask must be binary"
            )
        candidate_target = capture_candidate_only_matches > 0.0
        if np.any(candidate_target & (behavior_valid <= 0.0)):
            raise RuntimeError(
                "candidate-only capture identity targets an invalid rollout "
                "candidate"
            )
        if np.any(candidate_target & (behavior_rank < 1.0)):
            raise RuntimeError(
                "candidate-only capture identity has no positive canonical "
                "behavior rank"
            )
        if capture_identity_candidates is None:
            capture_identity_candidates = np.zeros(
                expected_event_shape,
                dtype=np.float32,
            )
        capture_identity_candidates = np.asarray(
            capture_identity_candidates,
            dtype=np.float32,
        )
        if capture_identity_candidates.ndim == 2:
            capture_identity_candidates = (
                capture_identity_candidates[..., None]
            )
        if capture_identity_candidates.shape != expected_event_shape:
            raise ValueError(
                "capture_identity_candidates must have shape {}, got {}"
                .format(
                    expected_event_shape,
                    capture_identity_candidates.shape,
                )
            )
        if not np.isfinite(capture_identity_candidates).all() or np.any(
                capture_identity_candidates < 0.0):
            raise ValueError(
                "capture_identity_candidates must be finite and non-negative"
            )
        has_candidate_capture = bool(np.any(
            capture_candidate_only_matches > 0.0
        ))
        if capture_candidate_event_provenance is None:
            if has_candidate_capture:
                raise RuntimeError(
                    "candidate-only capture targets require event-level "
                    "provenance"
                )
            capture_candidate_event_provenance = [
                tuple() for _ in range(num_insert_episodes)
            ]
        if (
                not isinstance(capture_candidate_event_provenance, (list, tuple))
                or len(capture_candidate_event_provenance)
                != int(num_insert_episodes)):
            raise ValueError(
                "candidate event provenance must contain one event list per "
                "inserted episode"
            )
        if environment_episode_ids is None:
            if has_candidate_capture:
                raise RuntimeError(
                    "candidate-only capture targets require environment "
                    "episode identities"
                )
            environment_episode_ids = np.full(
                num_insert_episodes,
                -1,
                dtype=np.int64,
            )
        environment_episode_ids = np.asarray(
            environment_episode_ids,
            dtype=np.int64,
        ).reshape(-1)
        if environment_episode_ids.shape != (int(num_insert_episodes),):
            raise ValueError(
                "environment_episode_ids must have shape {}, got {}".format(
                    (int(num_insert_episodes),),
                    environment_episode_ids.shape,
                )
            )
        if (
                has_candidate_capture
                and (
                    np.any(environment_episode_ids < 0)
                    or np.unique(environment_episode_ids).size
                    != environment_episode_ids.size
                )):
            raise RuntimeError(
                "candidate event provenance requires unique non-negative "
                "environment episode identities"
            )
        candidate_catalog = canonical_capture_factor_catalog(
            self.num_agents,
            3,
        )
        normalized_event_provenance = []
        for episode_offset, event_records in enumerate(
                capture_candidate_event_provenance):
            if not isinstance(event_records, (list, tuple)):
                raise TypeError(
                    "candidate event provenance for one episode must be a "
                    "list or tuple"
                )
            normalized_records = []
            event_identity_keys = set()
            for record in event_records:
                if not isinstance(record, dict):
                    raise TypeError(
                        "candidate event provenance record must be a dict"
                    )
                required_fields = (
                    "environment_episode_id",
                    "event_id",
                    "target_id",
                    "participant_slots",
                    "candidate_index",
                    "candidate_identity",
                    "candidate_order",
                    "identity_event_weight",
                    "capture_step",
                    "static_dynamic_class",
                )
                missing = [
                    field for field in required_fields
                    if field not in record
                ]
                if missing:
                    raise RuntimeError(
                        "candidate event provenance is missing {}".format(
                            ",".join(missing)
                        )
                    )
                environment_episode_id = int(
                    record["environment_episode_id"]
                )
                if environment_episode_id != int(
                        environment_episode_ids[episode_offset]):
                    raise RuntimeError(
                        "candidate event provenance environment episode "
                        "identity is misaligned"
                    )
                event_id = int(record["event_id"])
                prey_id = int(record["target_id"])
                capture_step = int(record["capture_step"])
                candidate_index = int(record["candidate_index"])
                candidate_order = int(record["candidate_order"])
                participants = tuple(
                    int(slot) for slot in record["participant_slots"]
                )
                identity = str(record["candidate_identity"])
                static_dynamic_class = str(
                    record["static_dynamic_class"]
                )
                identity_weight = float(record["identity_event_weight"])
                if (
                        event_id < 0
                        or prey_id < 0
                        or capture_step < 0
                        or capture_step >= self.episode_length
                        or candidate_index < 0
                        or candidate_index >= len(candidate_catalog)
                        or candidate_order not in (2, 3)
                        or static_dynamic_class not in ("static", "dynamic")
                        or not np.isfinite(identity_weight)
                        or identity_weight <= 0.0):
                    raise RuntimeError(
                        "candidate event provenance contains an invalid value"
                    )
                expected_identity_nodes = candidate_catalog[candidate_index]
                expected_identity = "-".join(
                    str(int(node)) for node in expected_identity_nodes
                )
                if (
                        identity != expected_identity
                        or candidate_order != len(expected_identity_nodes)
                        or participants != tuple(sorted(set(participants)))
                        or not set(expected_identity_nodes).issubset(
                            set(participants)
                        )):
                    raise RuntimeError(
                        "candidate event provenance identity contract failed"
                    )
                event_identity_key = (
                    environment_episode_id,
                    event_id,
                    prey_id,
                    candidate_index,
                )
                if event_identity_key in event_identity_keys:
                    raise RuntimeError(
                        "candidate event provenance duplicates one event "
                        "identity"
                    )
                event_identity_keys.add(event_identity_key)
                normalized_records.append({
                    "environment_episode_id": environment_episode_id,
                    "event_id": event_id,
                    "target_id": prey_id,
                    "participant_slots": participants,
                    "candidate_index": candidate_index,
                    "candidate_identity": identity,
                    "candidate_order": candidate_order,
                    "identity_event_weight": identity_weight,
                    "capture_step": capture_step,
                    "static_dynamic_class": static_dynamic_class,
                })
            normalized_event_provenance.append(tuple(normalized_records))
        if has_candidate_capture and not any(normalized_event_provenance):
            raise RuntimeError(
                "candidate-only capture tensor has no event provenance rows"
            )
        reconstructed_candidate_matches = np.zeros_like(
            capture_candidate_only_matches,
            dtype=np.float32,
        )
        for episode_offset, records in enumerate(
                normalized_event_provenance):
            event_mass = {}
            for record in records:
                event_key = (
                    record["environment_episode_id"],
                    record["event_id"],
                    record["target_id"],
                    record["capture_step"],
                )
                event_mass[event_key] = (
                    event_mass.get(event_key, 0.0)
                    + float(record["identity_event_weight"])
                )
                reconstructed_candidate_matches[
                    record["capture_step"],
                    episode_offset,
                    record["candidate_index"],
                ] += float(record["identity_event_weight"])
            if any(
                    abs(float(mass) - 1.0) > 1e-6
                    for mass in event_mass.values()):
                raise RuntimeError(
                    "candidate event provenance does not conserve unit event "
                    "quality across identities"
                )
        if not np.allclose(
                reconstructed_candidate_matches,
                capture_candidate_only_matches,
                rtol=0.0,
                atol=1e-6):
            raise RuntimeError(
                "candidate event provenance does not reconstruct the "
                "candidate-only capture tensor"
            )

        has_active_capture = bool(np.any(capture_factor_matches > 0.0))
        if capture_active_event_provenance is None:
            if has_active_capture and self.pair_pending_enabled:
                raise RuntimeError(
                    "active identity-matched captures require event-level "
                    "provenance"
                )
            capture_active_event_provenance = [
                tuple() for _ in range(num_insert_episodes)
            ]
        if (
                not isinstance(capture_active_event_provenance, (list, tuple))
                or len(capture_active_event_provenance)
                != int(num_insert_episodes)):
            raise ValueError(
                "active event provenance must contain one event list per "
                "inserted episode"
            )
        normalized_active_event_provenance = []
        reconstructed_active_matches = np.zeros_like(
            capture_factor_matches,
            dtype=np.float32,
        )
        for episode_offset, event_records in enumerate(
                capture_active_event_provenance):
            if not isinstance(event_records, (list, tuple)):
                raise TypeError(
                    "active event provenance for one episode must be a list "
                    "or tuple"
                )
            normalized_records = []
            logical_keys = set()
            event_mass = {}
            for record in event_records:
                if not isinstance(record, dict):
                    raise TypeError(
                        "active event provenance record must be a dict"
                    )
                required_fields = (
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
                    "static_dynamic_class",
                )
                missing = [
                    field for field in required_fields
                    if field not in record
                ]
                if missing:
                    raise RuntimeError(
                        "active event provenance is missing {}".format(
                            ",".join(missing)
                        )
                    )
                environment_episode_id = int(
                    record["environment_episode_id"]
                )
                if environment_episode_id != int(
                        environment_episode_ids[episode_offset]):
                    raise RuntimeError(
                        "active event provenance environment episode "
                        "identity is misaligned"
                    )
                event_id = int(record["event_id"])
                prey_id = int(record["target_id"])
                capture_step = int(record["capture_step"])
                factor_index = int(record["factor_index"])
                factor_order = int(record["factor_order"])
                participants = tuple(
                    sorted(int(slot)
                           for slot in record["participant_slots"])
                )
                factor_identity = str(record["factor_identity"])
                identity_nodes = tuple(
                    int(node)
                    for node in factor_identity.split("-")
                    if node != ""
                )
                identity_weight = float(record["identity_event_weight"])
                factor_slot_weight = float(record["factor_slot_weight"])
                static_dynamic_class = str(
                    record["static_dynamic_class"]
                )
                if (
                        event_id < 0
                        or prey_id < 0
                        or capture_step < 0
                        or capture_step >= self.episode_length
                        or factor_index < 0
                        or factor_index >= self.num_factor
                        or factor_order not in (2, 3)
                        or len(identity_nodes) != factor_order
                        or tuple(sorted(identity_nodes)) != identity_nodes
                        or not set(identity_nodes).issubset(participants)
                        or static_dynamic_class not in ("static", "dynamic")
                        or not np.isfinite(identity_weight)
                        or identity_weight <= 0.0
                        or not np.isfinite(factor_slot_weight)
                        or factor_slot_weight <= 0.0):
                    raise RuntimeError(
                        "active event provenance contains an invalid value"
                    )
                rollout_nodes = tuple(sorted(
                    np.flatnonzero(
                        adj[
                            capture_step,
                            episode_offset,
                            :,
                            factor_index,
                        ] > 0
                    ).tolist()
                ))
                if rollout_nodes != identity_nodes:
                    raise RuntimeError(
                        "active event provenance identity does not match the "
                        "rollout graph"
                    )
                logical_key = (
                    environment_episode_id,
                    event_id,
                    prey_id,
                    factor_index,
                )
                if logical_key in logical_keys:
                    raise RuntimeError(
                        "active event provenance duplicates one event factor"
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
                reconstructed_active_matches[
                    capture_step,
                    episode_offset,
                    factor_index,
                ] += factor_slot_weight
                normalized = dict(record)
                normalized["participant_slots"] = participants
                normalized["factor_identity_nodes"] = identity_nodes
                normalized_records.append(normalized)
            if any(
                    abs(float(mass) - 1.0) > 1e-6
                    for mass in event_mass.values()):
                raise RuntimeError(
                    "active event provenance does not conserve unit event "
                    "quality"
                )
            normalized_active_event_provenance.append(
                tuple(normalized_records)
            )
        if (
                (self.pair_pending_enabled
                 or any(normalized_active_event_provenance))
                and not np.allclose(
                reconstructed_active_matches,
                capture_factor_matches,
                rtol=0.0,
                atol=1e-6)):
            raise RuntimeError(
                "active event provenance does not reconstruct the identity-"
                "matched capture tensor"
            )

        if self.current_i + num_insert_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_episodes)
        else:
            num_left_episodes = self.current_i + num_insert_episodes - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_episodes)))

        if self.use_same_share_obs:
            # remove agent dimension since all agents share centralized observation
            share_obs = share_obs[:, :, 0]

        overwritten_mask = self.episode_generation[idx_range] >= 0
        if self.pair_pending_enabled:
            for replay_slot in idx_range[overwritten_mask].tolist():
                old_generation = int(self.episode_generation[replay_slot])
                for pending_key in (
                        self.pair_pending_store.keys_for_generation(
                            self.policy_id,
                            old_generation,
                        )):
                    entry = self.pair_pending_store.entries[pending_key]
                    if entry["state"] == PAIR_EVIDENCE_AVAILABLE_IN_REPLAY:
                        # TTL starts when the circular replay actually evicts
                        # the episode, not when that episode happened to be
                        # sampled most recently.
                        self.pair_pending_store.refresh_available(
                            pending_key,
                            self.pair_pending_current_adj_update,
                            self.pair_pending_behavior_policy_version,
                        )
                        self.pair_pending_store.mark_pending(
                            pending_key,
                            self.pair_pending_current_adj_update,
                        )
        self.outcome_slot_overwrite_count += int(np.sum(overwritten_mask))
        if np.any(
                self.outcome_support_used[idx_range]
                & ~overwritten_mask):
            self.outcome_invalid_used_state_count += int(np.sum(
                self.outcome_support_used[idx_range] & ~overwritten_mask
            ))
            raise RuntimeError(
                "unused adjacency replay slot carries outcome support state"
            )
        generations = np.arange(
            self._next_episode_generation,
            self._next_episode_generation + len(idx_range),
            dtype=np.int64,
        )
        self._next_episode_generation += len(idx_range)
        self.outcome_generation_update_count += len(idx_range)
        self.episode_generation[idx_range] = generations
        self.environment_episode_id[idx_range] = environment_episode_ids
        self.episode_behavior_policy_version[idx_range] = (
            behavior_policy_versions
        )
        for episode_offset, replay_slot in enumerate(idx_range.tolist()):
            self.capture_candidate_event_provenance[replay_slot] = tuple(
                dict(record)
                for record in normalized_event_provenance[episode_offset]
            )
            self.capture_active_event_provenance[replay_slot] = tuple(
                dict(record)
                for record in
                normalized_active_event_provenance[episode_offset]
            )
            self.strict_pair_event_provenance[replay_slot] = tuple()
        self.outcome_support_used[idx_range] = False
        # A circular-buffer overwrite cannot inherit the prior episode's graph
        # confidence. compute_advantage() repopulates both arrays before replay.
        self.advantage[:, idx_range, 0] = 0.0
        self.graph_advantage_ready[:, idx_range, 0] = False
        occupied_generations = self.episode_generation[
            self.episode_generation >= 0
        ]
        if np.unique(occupied_generations).size != occupied_generations.size:
            self.outcome_generation_conflict_count += 1
            raise RuntimeError(
                "occupied adjacency replay slots share an episode generation"
            )
        # Insertion between PPO epochs would invalidate the cohort identity;
        # make such a state visible rather than silently reusing stale slots.
        self._cached_outcome_support_round = None
        self._cached_outcome_support_signature = None
        self._cached_outcome_support_selection = None

        self.obs[:, idx_range] = obs.copy()
        self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()
        self.capture_counts[:, idx_range] = capture_counts.copy()
        self.success_now[:, idx_range] = success_now.copy()
        self.capture_factor_matches[:, idx_range] = (
            capture_factor_matches.copy()
        )
        self.capture_candidate_only_matches[:, idx_range] = (
            capture_candidate_only_matches.copy()
        )
        self.capture_candidate_behavior[:, idx_range] = (
            capture_candidate_behavior.copy()
        )
        self.capture_identity_candidates[:, idx_range] = (
            capture_identity_candidates.copy()
        )
        self.adj[:, idx_range] = adj.copy()
        self.prob_adj[:, idx_range] = prob_adj.copy()
        self.qtot[:, idx_range] = qtot.copy()
        self.f_v[:, idx_range] = f_v.copy()
        self.f_q[:, idx_range] = f_q.copy()
        self.rnn_obs[:, idx_range] = rnn_obs.copy()

        if self.use_avail_acts:
            self.avail_acts[:, idx_range] = avail_acts.copy()

        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    # def sample_inds(self, data_chunk_length, num_mini_batch):
    #     """
    #     Sample a set of transitions from buffer from the specified indices.
    #     :param sample_inds: (np.ndarray) indices of samples to return from buffer.
    #
    #     :return: obs: (np.ndarray) sampled observations corresponding to that policy
    #     :return: share_obs: (np.ndarray) sampled observations corresponding to that policy
    #     :return: acts: (np.ndarray) sampled actions corresponding to that policy
    #     :return: rewards: (np.ndarray) sampled rewards corresponding to that policy
    #     :return: dones: (np.ndarray) sampled terminal status of agents corresponding to that policy
    #     :return: dones_env: (np.ndarray) sampled environment terminal status corresponding to that policy
    #     :return: valid_transition: (np.ndarray) whether each sampled transition in episodes are valid or not (invalid if corresponding agent is dead)
    #     :return: avail_acts: (np.ndarray) sampled available actions corresponding to that policy
    #     """
    #     batch_size = self.episode_length * self.buffer_size
    #     data_chunks = batch_size // data_chunk_length
    #     mini_batch_size = data_chunks // num_mini_batch
    #     rand = torch.randperm(data_chunks).numpy()
    #     sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
    #
    #     obs = self.obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     #np.stack(self.dones_env[:-1].transpose(1,0,2,3),axis=0)
    #     #acts = self.acts.reshape(batch_size,self.num_agents,-1)
    #     dones = np.concatenate((np.zeros((1, self.buffer_size, self.num_agents,1), dtype=np.float32),self.dones[:-1]))
    #     dones_env = np.concatenate((np.zeros((1, self.buffer_size,1), dtype=np.float32),self.dones_env[:-1]))
    #     adj = self.adj[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     prob_adj = self.prob_adj[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     advantage = self.advantage
    #
    #     rnn_obs = self.rnn_obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #
    #     '''advantage_copy = advantage.copy()
    #     advantage_copy[dones_env == 1.0] = np.nan
    #     mean_advantage = np.nanmean(advantage_copy[1:])
    #     std_advantage = np.nanstd(advantage_copy[1:])
    #     advantage[1:] = (advantage[1:] - mean_advantage) / (std_advantage + 1e-10)'''
    #     advantages = advantage.transpose(1,0,2).reshape(batch_size,-1)
    #
    #     '''margin_advt = self.margin_advt
    #     margin_advt_copy = margin_advt.copy()
    #     margin_advt_copy[dones == 1.0] = np.nan
    #     mean_advt = np.nanmean(margin_advt_copy[1:])
    #     std_advt = np.nanstd(margin_advt_copy[1:])
    #     margin_advt[1:] = (margin_advt[1:] - mean_advt) / (std_advt + 1e-10)
    #     margin_advts = margin_advt.transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)'''
    #
    #     f_advt = self.f_advt
    #     f_advt_copy = f_advt.copy()
    #     f_advt_copy[np.tile(dones_env,self.num_factor) == 1.0] = np.nan
    #     mean_advt_f = np.nanmean(f_advt_copy)
    #     std_advt_f = np.nanstd(f_advt_copy)
    #     '''f_advt_copy = f_advt.copy()
    #     f_advt_copy[np.tile(dones_env,self.num_factor) == 1.0] = np.nan
    #     mean_advt_f = np.nanmean(f_advt_copy.reshape(batch_size,self.num_factor,-1),axis=0)
    #     std_advt_f = np.nanstd(f_advt_copy.reshape(batch_size,self.num_factor,-1),axis=0)'''
    #     f_advt = (f_advt - mean_advt_f) / (std_advt_f + 1e-5)
    #     f_advts = f_advt.transpose(1,0,2,3).reshape(batch_size,self.num_factor,-1)
    #     #rewards = self.rewards.reshape(batch_size,self.num_agents,-1)
    #     dones = dones.transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     dones_env = dones_env.transpose(1,0,2).reshape(batch_size,-1)
    #     if self.use_same_share_obs:
    #         share_obs = self.share_obs[:-1].transpose(1,0,2).reshape(batch_size,-1)
    #     else:
    #         share_obs = self.share_obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #
    #     '''if self.use_avail_acts:
    #         avail_acts = self.avail_acts[:-1].reshape(batch_size,self.num_agents,-1)
    #     else:
    #         avail_acts = None'''
    #     for indices in sampler:
    #         obs_batch = []
    #         share_obs_batch = []
    #         dones_batch = []
    #         dones_env_batch = []
    #         adj_batch = []
    #         prob_adj_batch = []
    #         advantages_batch = []
    #         #margin_advts_batch = []
    #         f_advts_batch = []
    #         rnn_obs_batch = []
    #         for i in indices:
    #             ind = i * data_chunk_length
    #             obs_batch.append(obs[ind:ind+data_chunk_length])
    #             share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
    #             dones_batch.append(dones[ind:ind+data_chunk_length])
    #             dones_env_batch.append(dones_env[ind:ind+data_chunk_length])
    #             adj_batch.append(adj[ind:ind+data_chunk_length])
    #             prob_adj_batch.append(prob_adj[ind:ind+data_chunk_length])
    #             advantages_batch.append(advantages[ind:ind+data_chunk_length])
    #             #margin_advts_batch.append(margin_advts[ind:ind+data_chunk_length])
    #             f_advts_batch.append(f_advts[ind:ind+data_chunk_length])
    #             rnn_obs_batch.append(rnn_obs[ind:ind+data_chunk_length])
    #         obs_batch = np.stack(obs_batch,axis=0)
    #         share_obs_batch = np.stack(share_obs_batch,axis=0)
    #         dones_batch = np.stack(dones_batch,axis=0)
    #         dones_env_batch = np.stack(dones_env_batch,axis=0)
    #         adj_batch = np.stack(adj_batch,axis=0)
    #         prob_adj_batch = np.stack(prob_adj_batch,axis=0)
    #         advantages_batch = np.stack(advantages_batch,axis=0)
    #         #margin_advts_batch = np.stack(margin_advts_batch,axis=0)
    #         f_advts_batch = np.stack(f_advts_batch,axis=0)
    #         rnn_obs_batch = np.stack(rnn_obs_batch,axis=0)
    #
    #         yield obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch
    def _recent_episode_indices(self, recent_episode_window=0):
        valid_episodes = int(self.filled_i)
        if valid_episodes <= 0:
            raise RuntimeError(
                "AdjPolicyBuffer.sample_inds called before any episode is inserted."
            )
        recent_episode_window = int(recent_episode_window or 0)
        if recent_episode_window <= 0 or recent_episode_window >= valid_episodes:
            return np.arange(valid_episodes, dtype=np.int64)

        end = int(self.current_i) % int(self.buffer_size)
        start = (end - recent_episode_window) % int(self.buffer_size)
        return (
            np.arange(
                start,
                start + recent_episode_window,
                dtype=np.int64,
            )
            % int(self.buffer_size)
        )

    def set_pair_pending_clock(
            self,
            adjacency_update_index,
            behavior_policy_version):
        next_update = int(adjacency_update_index)
        if next_update != self.pair_pending_current_adj_update:
            self.pair_pending_new_snapshot_count = 0
            self.pair_pending_expired_ttl_count = 0
            self.pair_pending_prepared_count = 0
            self.pair_pending_aborted_count = 0
            self.pair_pending_rolled_back_count = 0
            self.pair_pending_committed_count = 0
            self.pair_pending_class_complete_count = 0
            self.pair_pending_pair_only_transaction_count = 0
            self.pair_pending_zero_target_abort_count = 0
            self.pair_pending_zero_gradient_abort_count = 0
            self.pair_pending_early_stop_abort_count = 0
            self.pair_pending_expired_provenance_count = 0
            self.pair_pending_expired_population_mismatch_count = 0
            self.pair_pending_stale_contract_valid = 1.0
            self.pair_pending_mass_contract_valid = 1.0
            self.pair_pending_objective_scope_contract_valid = 1.0
            self.pair_pending_atomic_rollback_contract_valid = 1.0
            self.pair_pending_checkpoint_contract_valid = 1.0
            self.pair_pending_last_positive_stale_trust = 0.0
            self.pair_pending_last_negative_stale_trust = 0.0
            self.pair_pending_last_raw_positive_mass = 0.0
            self.pair_pending_last_raw_negative_mass = 0.0
            self.pair_pending_last_effective_positive_mass = 0.0
            self.pair_pending_last_effective_negative_mass = 0.0
            self.pair_pending_counter_update = next_update
        self.pair_pending_current_adj_update = next_update
        self.pair_pending_behavior_policy_version = int(
            behavior_policy_version
        )
        if (
                self.pair_pending_current_adj_update < 0
                or self.pair_pending_behavior_policy_version < 0):
            raise ValueError("pair pending clocks must be non-negative")
        if self.pair_pending_enabled:
            expired = self.pair_pending_store.expire_out_of_horizon(
                self.pair_pending_current_adj_update
            )
            self.pair_pending_expired_ttl_count += len(expired)

    def pair_pending_update_diagnostics(self):
        diagnostics = self.pair_pending_store.diagnostics(
            self.pair_pending_current_adj_update,
            self.pair_pending_behavior_policy_version,
        )
        diagnostics.update({
            "policy_id": self.policy_id,
            "adjacency_update_index": int(
                self.pair_pending_current_adj_update
            ),
            "current_policy_version": int(
                self.pair_pending_behavior_policy_version
            ),
            "new_snapshot_count": int(
                self.pair_pending_new_snapshot_count
            ),
            "expired_by_ttl_count": int(
                self.pair_pending_expired_ttl_count
            ),
            "payload_contract_valid": float(
                self.pair_pending_payload_contract_valid
            ),
            "prepared_this_update_count": int(
                self.pair_pending_prepared_count
            ),
            "aborted_this_update_count": int(
                self.pair_pending_aborted_count
            ),
            "rolled_back_this_update_count": int(
                self.pair_pending_rolled_back_count
            ),
            "committed_this_update_count": int(
                self.pair_pending_committed_count
            ),
            "class_complete_from_pending_count": int(
                self.pair_pending_class_complete_count
            ),
            "pair_only_transaction_count": int(
                self.pair_pending_pair_only_transaction_count
            ),
            "zero_target_abort_count": int(
                self.pair_pending_zero_target_abort_count
            ),
            "zero_gradient_abort_count": int(
                self.pair_pending_zero_gradient_abort_count
            ),
            "early_stop_abort_count": int(
                self.pair_pending_early_stop_abort_count
            ),
            "expired_by_provenance_count": int(
                self.pair_pending_expired_provenance_count
            ),
            "expired_by_population_mismatch_count": int(
                self.pair_pending_expired_population_mismatch_count
            ),
            "stale_contract_valid": float(
                self.pair_pending_stale_contract_valid
            ),
            "mass_contract_valid": float(
                self.pair_pending_mass_contract_valid
            ),
            "objective_scope_contract_valid": float(
                self.pair_pending_objective_scope_contract_valid
            ),
            "atomic_rollback_contract_valid": float(
                self.pair_pending_atomic_rollback_contract_valid
            ),
            "checkpoint_contract_valid": float(
                self.pair_pending_checkpoint_contract_valid
            ),
            "positive_stale_trust": float(
                self.pair_pending_last_positive_stale_trust
            ),
            "negative_stale_trust": float(
                self.pair_pending_last_negative_stale_trust
            ),
            "raw_positive_mass": float(
                self.pair_pending_last_raw_positive_mass
            ),
            "raw_negative_mass": float(
                self.pair_pending_last_raw_negative_mass
            ),
            "effective_positive_mass": float(
                self.pair_pending_last_effective_positive_mass
            ),
            "effective_negative_mass": float(
                self.pair_pending_last_effective_negative_mass
            ),
        })
        return diagnostics

    def record_pair_pending_transaction_result(
            self,
            committed=False,
            rolled_back=False,
            abort_reason="",
            objective_scope_contract_valid=True,
            stale_contract_valid=True,
            mass_contract_valid=True,
            atomic_rollback_contract_valid=True,
            checkpoint_contract_valid=True,
            positive_stale_trust=0.0,
            negative_stale_trust=0.0,
            raw_positive_mass=0.0,
            raw_negative_mass=0.0,
            effective_positive_mass=0.0,
            effective_negative_mass=0.0):
        """Record one logical pending cohort result without training effects."""
        if not self.pair_pending_enabled:
            raise RuntimeError(
                "cannot record a pending transaction while pending is disabled"
            )
        committed = bool(committed)
        rolled_back = bool(rolled_back)
        abort_reason = str(abort_reason or "")
        if committed == bool(abort_reason):
            raise RuntimeError(
                "pending result must be exactly one of commit or abort"
            )
        if committed and rolled_back:
            raise RuntimeError(
                "committed pending transaction cannot also be rolled back"
            )
        if committed:
            self.pair_pending_committed_count += 1
        else:
            self.pair_pending_aborted_count += 1
            self.pair_pending_rolled_back_count += int(rolled_back)
            normalized_reason = abort_reason.upper()
            self.pair_pending_zero_target_abort_count += int(
                "ZERO_TARGET" in normalized_reason
                or "ZERO TARGET" in normalized_reason
            )
            self.pair_pending_zero_gradient_abort_count += int(
                "ZERO_GRADIENT" in normalized_reason
                or "ZERO GRADIENT" in normalized_reason
            )
            self.pair_pending_early_stop_abort_count += int(
                "EARLY_STOP" in normalized_reason
            )
        self.pair_pending_objective_scope_contract_valid = min(
            self.pair_pending_objective_scope_contract_valid,
            float(bool(objective_scope_contract_valid)),
        )
        self.pair_pending_stale_contract_valid = min(
            self.pair_pending_stale_contract_valid,
            float(bool(stale_contract_valid)),
        )
        self.pair_pending_mass_contract_valid = min(
            self.pair_pending_mass_contract_valid,
            float(bool(mass_contract_valid)),
        )
        self.pair_pending_atomic_rollback_contract_valid = min(
            self.pair_pending_atomic_rollback_contract_valid,
            float(bool(atomic_rollback_contract_valid)),
        )
        self.pair_pending_checkpoint_contract_valid = min(
            self.pair_pending_checkpoint_contract_valid,
            float(bool(checkpoint_contract_valid)),
        )
        named_values = (
            ("positive_stale_trust", positive_stale_trust),
            ("negative_stale_trust", negative_stale_trust),
            ("raw_positive_mass", raw_positive_mass),
            ("raw_negative_mass", raw_negative_mass),
            ("effective_positive_mass", effective_positive_mass),
            ("effective_negative_mass", effective_negative_mass),
        )
        normalized_values = {}
        for name, value in named_values:
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    "pending transaction {} is invalid".format(name)
                )
            normalized_values[name] = value
        self.pair_pending_last_positive_stale_trust = normalized_values[
            "positive_stale_trust"
        ]
        self.pair_pending_last_negative_stale_trust = normalized_values[
            "negative_stale_trust"
        ]
        self.pair_pending_last_raw_positive_mass = normalized_values[
            "raw_positive_mass"
        ]
        self.pair_pending_last_raw_negative_mass = normalized_values[
            "raw_negative_mass"
        ]
        self.pair_pending_last_effective_positive_mass = normalized_values[
            "effective_positive_mass"
        ]
        self.pair_pending_last_effective_negative_mass = normalized_values[
            "effective_negative_mass"
        ]

    def pair_pending_state_dict(self):
        return {
            "policy_id": self.policy_id,
            "current_adj_update": int(
                self.pair_pending_current_adj_update
            ),
            "behavior_policy_version": int(
                self.pair_pending_behavior_policy_version
            ),
            "store": self.pair_pending_store.state_dict(),
        }

    def load_pair_pending_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError(
                "pair pending policy checkpoint must be a dictionary"
            )
        if str(state.get("policy_id")) != self.policy_id:
            raise RuntimeError(
                "pair pending checkpoint policy identity mismatch"
            )
        self.pair_pending_current_adj_update = int(
            state.get("current_adj_update", 0)
        )
        self.pair_pending_behavior_policy_version = int(
            state.get("behavior_policy_version", 0)
        )
        self.pair_pending_store.load_state_dict(state["store"])

    def _capture_pair_pending_snapshots(
            self,
            flat_fields,
            episode_indices,
            selected_pair_evidence,
            base_episode_indices,
            data_chunk_length):
        if not self.pair_pending_enabled:
            return
        if len(flat_fields) != 35:
            raise RuntimeError(
                "production pair pending snapshot requires the real 35-field "
                "adjacency batch"
            )
        episode_indices = np.asarray(
            episode_indices, dtype=np.int64
        ).reshape(-1)
        selected_pair_evidence = np.asarray(
            selected_pair_evidence, dtype=bool
        ).reshape(-1)
        if selected_pair_evidence.shape != episode_indices.shape:
            raise RuntimeError(
                "pair pending evidence and selected episode axes differ"
            )
        if self.episode_length % int(data_chunk_length) != 0:
            raise RuntimeError(
                "pair pending snapshot cannot split the episode into "
                "complete recurrent chunks"
            )
        valid_episodes = int(episode_indices.size)
        chunks_per_episode = (
            int(self.episode_length) // int(data_chunk_length)
        )
        base_slots = set(
            int(slot)
            for slot in np.asarray(
                base_episode_indices, dtype=np.int64
            ).reshape(-1).tolist()
        )
        reshaped_fields = []
        for field_index, field in enumerate(flat_fields):
            array = np.asarray(field)
            if array.shape[0] != valid_episodes * self.episode_length:
                raise RuntimeError(
                    "pair pending production field {} has transition axis "
                    "{}, expected {}".format(
                        field_index,
                        array.shape[0],
                        valid_episodes * self.episode_length,
                    )
                )
            reshaped_fields.append(array.reshape(
                (valid_episodes, self.episode_length) + array.shape[1:]
            ))

        for episode_ordinal in np.flatnonzero(
                selected_pair_evidence).tolist():
            replay_slot = int(episode_indices[episode_ordinal])
            generation = int(self.episode_generation[replay_slot])
            records = tuple(
                self.strict_pair_event_provenance[replay_slot]
            )
            if not records:
                raise RuntimeError(
                    "strict pair evidence cannot be snapshotted without an "
                    "event-local pair-to-capture join"
                )
            grouped_records = {}
            for record in records:
                event_key = (
                    int(record["environment_episode_id"]),
                    int(record["event_id"]),
                    int(record["target_id"]),
                    tuple(record["participant_slots"]),
                )
                grouped_records.setdefault(event_key, []).append(record)
                transition_step = int(record["pair_transition_step"])
                capture_step = int(record["capture_step"])
                pair_factor = int(record["pair_factor_index"])
                if (
                        transition_step < 0
                        or transition_step >= self.episode_length
                        or capture_step <= transition_step
                        or capture_step >= self.episode_length
                        or pair_factor < 0
                        or pair_factor >= self.num_factor
                        or self.pair_to_triplet_transition_score[
                            transition_step,
                            replay_slot,
                            pair_factor,
                            0,
                        ] <= 0.0):
                    raise RuntimeError(
                        "strict pair event provenance is not aligned with "
                        "the target-bearing transition tensor"
                    )

            base_episode_batch = []
            for field in reshaped_fields:
                episode_values = np.array(
                    field[episode_ordinal], copy=True, order="C"
                )
                base_episode_batch.append(episode_values.reshape(
                    (chunks_per_episode, int(data_chunk_length))
                    + episode_values.shape[1:]
                ))
            success = bool(np.any(
                self.success_now[:, replay_slot, 0] > 0.0
            ))
            terminal_steps = np.flatnonzero(
                self.dones_env[:, replay_slot, 0] > 0.0
            )
            terminal_step = (
                int(terminal_steps[0])
                if terminal_steps.size > 0
                else self.episode_length - 1
            )
            for event_key, event_records in sorted(
                    grouped_records.items()):
                (
                    environment_episode_id,
                    event_id,
                    prey_id,
                    participants,
                ) = event_key
                if environment_episode_id != int(
                        self.environment_episode_id[replay_slot]):
                    raise RuntimeError(
                        "strict pair event and replay episode identities differ"
                    )
                capture_steps = {
                    int(record["capture_step"]) for record in event_records
                }
                if len(capture_steps) != 1:
                    raise RuntimeError(
                        "one capture event maps to multiple capture steps"
                    )
                capture_step = next(iter(capture_steps))
                if capture_step > terminal_step:
                    raise RuntimeError(
                        "capture event occurs after the episode terminal step"
                    )
                pair_identities = tuple(sorted({
                    tuple(
                        int(node)
                        for node in str(record["pair_identity"]).split("-")
                    )
                    for record in event_records
                }))
                episode_batch = [
                    np.array(value, copy=True, order="C")
                    for value in base_episode_batch
                ]
                full_raw_pair_score = np.asarray(base_episode_batch[17])
                event_raw_pair_score = np.zeros_like(full_raw_pair_score)
                for record in event_records:
                    transition_step = int(record["pair_transition_step"])
                    pair_factor = int(record["pair_factor_index"])
                    chunk_index = (
                        transition_step // int(data_chunk_length)
                    )
                    chunk_offset = (
                        transition_step % int(data_chunk_length)
                    )
                    source_value = full_raw_pair_score[
                        chunk_index, chunk_offset, pair_factor
                    ]
                    if not np.any(source_value > 0.0):
                        raise RuntimeError(
                            "event-local pair snapshot lost its target value"
                        )
                    event_raw_pair_score[
                        chunk_index, chunk_offset, pair_factor
                    ] = source_value
                episode_batch[17] = event_raw_pair_score
                target_count = int(np.sum(np.any(
                    event_raw_pair_score > 0.0, axis=-1
                )))
                if target_count <= 0:
                    raise RuntimeError(
                        "strict pair snapshot has no target-bearing factor"
                    )
                metadata = {
                    "policy_id": self.policy_id,
                    "replay_generation": generation,
                    "environment_episode_id": environment_episode_id,
                    "episode_ordinal": int(episode_ordinal),
                    "capture_event_id": event_id,
                    "prey_id": prey_id,
                    "participant_slots": participants,
                    "canonical_pair_identities": pair_identities,
                    "factor_order": 2,
                    "sign": 1 if success else -1,
                    "outcome_success": success,
                    "first_seen_adj_update": int(
                        self.pair_pending_current_adj_update
                    ),
                    "original_last_valid_adj_update": int(
                        self.pair_pending_current_adj_update
                    ),
                    "behavior_policy_version": int(
                        self.episode_behavior_policy_version[replay_slot]
                    ),
                    "source_class": (
                        "base" if replay_slot in base_slots else "support"
                    ),
                    "target_bearing_transition_count": target_count,
                    "raw_event_quality": float(1.0),
                    "event_provenance_available": True,
                    "capture_step": capture_step,
                    "terminal_step": terminal_step,
                    "capture_to_terminal_distance": (
                        terminal_step - capture_step
                    ),
                }
                key = pair_pending_entry_key(metadata)
                existing = self.pair_pending_store.entries.get(key)
                if existing is not None:
                    self.pair_pending_store.refresh_available(
                        key,
                        self.pair_pending_current_adj_update,
                        self.pair_pending_behavior_policy_version,
                    )
                    continue
                self.pair_pending_store.add_available(
                    metadata,
                    tuple(episode_batch),
                )
                self.pair_pending_new_snapshot_count += 1

    def prepare_pair_pending_training_batch(self, expected_ppo_epochs):
        """Build one deterministic pair-only class-complete transaction."""
        if not self.pair_pending_enabled:
            return None
        self.pair_pending_store.expire_out_of_horizon(
            self.pair_pending_current_adj_update
        )
        selected = []
        has_pending = False
        signs = set()
        for key, entry in sorted(self.pair_pending_store.entries.items()):
            if entry["state"] not in (
                    PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
                    PAIR_EVIDENCE_PENDING):
                continue
            if (
                    entry["state"] == PAIR_EVIDENCE_PENDING
                    and self.pair_pending_store.pending_age(
                        key,
                        self.pair_pending_current_adj_update,
                    ) > self.pair_pending_max_adj_updates):
                continue
            selected.append((key, entry))
            signs.add(int(entry["metadata"]["sign"]))
            has_pending = has_pending or (
                entry["state"] == PAIR_EVIDENCE_PENDING
            )
        # Current replay overlap remains owned by support v6. The production
        # pending transaction exists only to bridge an evicted strict-evidence
        # side to a live opposite sign.
        if not has_pending or signs != {-1, 1}:
            return None

        keys = tuple(key for key, _entry in selected)
        source_states = tuple(
            str(entry["state"]) for _key, entry in selected
        )
        store_state_before_prepare = self.pair_pending_store.state_dict()
        cohort_id = self.pair_pending_store.prepare_class_complete(
            keys=keys,
            current_adj_update=self.pair_pending_current_adj_update,
            current_policy_version=(
                self.pair_pending_behavior_policy_version
            ),
            expected_ppo_epochs=expected_ppo_epochs,
        )
        self.pair_pending_prepared_count += 1
        self.pair_pending_class_complete_count += 1
        self.pair_pending_pair_only_transaction_count += 1
        # One generation is one replay/training population even when the
        # episode contains several capture events. Merge the event-local raw
        # pair masks before constructing the batch so the old episode is never
        # duplicated once per event.
        grouped_selected = []
        grouped_by_generation = {}
        for key, entry in selected:
            generation_key = (
                str(entry["metadata"]["policy_id"]),
                int(entry["metadata"]["replay_generation"]),
            )
            if generation_key not in grouped_by_generation:
                group = {
                    "generation_key": generation_key,
                    "items": [],
                }
                grouped_by_generation[generation_key] = group
                grouped_selected.append(group)
            grouped_by_generation[generation_key]["items"].append(
                (key, entry)
            )
        mutable_batches = []
        grouped_metadata = []
        for group in grouped_selected:
            items = group["items"]
            representative = [
                np.array(value, copy=True) for value in items[0][1]["batch"]
            ]
            event_raw_scores = []
            reference_metadata = items[0][1]["metadata"]
            for _key, entry in items:
                metadata = entry["metadata"]
                if (
                        int(metadata["environment_episode_id"])
                        != int(reference_metadata["environment_episode_id"])
                        or int(metadata["sign"])
                        != int(reference_metadata["sign"])
                        or bool(metadata["outcome_success"])
                        != bool(reference_metadata["outcome_success"])):
                    self.pair_pending_store.release_prepared(cohort_id)
                    raise RuntimeError(
                        "events from one replay generation disagree on "
                        "episode identity or outcome"
                    )
                for field_index, (reference_value, event_value) in enumerate(
                        zip(representative, entry["batch"])):
                    if field_index == 17:
                        continue
                    if (
                            reference_value.dtype != event_value.dtype
                            or reference_value.shape != event_value.shape
                            or not np.array_equal(
                                reference_value, event_value
                            )):
                        self.pair_pending_store.release_prepared(cohort_id)
                        raise RuntimeError(
                            "event-local snapshots from one generation do "
                            "not share an identical training population"
                        )
                event_raw_score = np.asarray(entry["batch"][17])
                event_raw_scores.append(event_raw_score)
            try:
                # Distinct capture events remain distinct evidence keys, but
                # their common immutable replay generation is optimized once.
                # Shared causal transitions therefore form a set union rather
                # than receiving duplicate pair-credit mass.
                merged_raw_score = merge_generation_event_pair_scores(
                    event_raw_scores
                )
            except Exception:
                self.pair_pending_store.release_prepared(cohort_id)
                raise
            representative[17] = merged_raw_score
            mutable_batches.append(representative)
            grouped_metadata.append(dict(reference_metadata))
        episode_scores = []
        episode_success = []
        for metadata, fields in zip(grouped_metadata, mutable_batches):
            raw_score = np.asarray(fields[17], dtype=np.float32)
            raw_score = raw_score.reshape(
                self.episode_length,
                self.num_factor,
                -1,
            )
            if raw_score.shape[-1] != 1:
                self.pair_pending_store.release_prepared(cohort_id)
                raise RuntimeError(
                    "pending raw pair score has an invalid final axis"
                )
            episode_scores.append(raw_score[..., 0])
            episode_success.append(
                bool(metadata["outcome_success"])
            )
        try:
            mass = reconstruct_pending_pair_mass(
                pair_transition_scores=episode_scores,
                episode_success=episode_success,
                coefficient=self.adj_pair_pursuit_credit_coef,
                credit_cap=self.adj_pair_pursuit_credit_cap,
            )
        except Exception:
            self.pair_pending_store.release_prepared(cohort_id)
            raise
        self.pair_pending_mass_contract_valid = min(
            self.pair_pending_mass_contract_valid,
            float(mass["contract_valid"]),
        )

        credit = np.asarray(mass["credit"], dtype=np.float32)
        if credit.shape != (
                self.episode_length,
                len(grouped_selected),
                self.num_factor):
            self.pair_pending_store.release_prepared(cohort_id)
            raise RuntimeError(
                "pending pair credit reconstruction has an invalid shape"
            )
        concatenated_fields = []
        try:
            for field_index in range(35):
                field_values = []
                for episode_ordinal, fields in enumerate(mutable_batches):
                    value = np.asarray(fields[field_index])
                    if field_index == 15:
                        episode_credit = (
                            credit[:, episode_ordinal, :, None]
                            .reshape(value.shape)
                        )
                        value = episode_credit.astype(
                            value.dtype, copy=False
                        )
                    elif field_index == 33:
                        value = np.array(value, copy=True)
                        value[..., 0] = 1.0
                        value[..., 1] = float(episode_ordinal)
                    field_values.append(np.array(value, copy=True))
                concatenated_fields.append(np.concatenate(
                    field_values, axis=0
                ))
            batch = tuple(concatenated_fields)
            if len(batch) != 35:
                raise RuntimeError(
                    "pending pair-only batch schema is incomplete"
                )
            target_count = int(np.sum(np.any(
                np.asarray(batch[15]) != 0.0,
                axis=-1,
            )))
            if target_count <= 0:
                raise RuntimeError(
                    "pending pair-only batch has zero target mass"
                )
        except Exception:
            self.pair_pending_store.release_prepared(cohort_id)
            raise
        return {
            "cohort_id": cohort_id,
            "keys": keys,
            "batch": batch,
            "episode_count": len(grouped_selected),
            "chunk_count": int(batch[0].shape[0]),
            "target_bearing_transition_count": target_count,
            "raw_positive_mass": float(mass["raw_positive_mass"]),
            "raw_negative_mass": float(mass["raw_negative_mass"]),
            "centered_mass_error": float(mass["raw_centered_error"]),
            "source_states": source_states,
            "entry_metadata": tuple(
                dict(entry["metadata"]) for _key, entry in selected
            ),
            "pending_ages": tuple(
                int(self.pair_pending_store.pending_age(
                    key,
                    self.pair_pending_current_adj_update,
                ))
                for key, _entry in selected
            ),
            "policy_ages": tuple(
                int(self.pair_pending_store.policy_age(
                    key,
                    self.pair_pending_behavior_policy_version,
                ))
                for key, _entry in selected
            ),
            "current_policy_version": int(
                self.pair_pending_behavior_policy_version
            ),
            "payload_contract_valid": 1.0,
            "mass_contract_valid": float(mass["contract_valid"]),
            "store_state_before_prepare": store_state_before_prepare,
        }

    def standard_pair_transaction_generation_keys(self):
        if not self.pair_pending_enabled:
            return tuple()
        keys = []
        for row in getattr(
                self, "last_sample_pair_evidence_episode_rows", ()):
            if (
                    int(row.get("selected_for_training", 0)) != 1
                    or int(row.get("pair_evidence_episode", 0)) != 1):
                continue
            for key in self.pair_pending_store.keys_for_generation(
                    self.policy_id,
                    int(row["episode_generation"])):
                entry = self.pair_pending_store.entries.get(key)
                if entry is None:
                    continue
                if entry["state"] not in (
                        PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
                        PAIR_EVIDENCE_COMMITTED):
                    raise RuntimeError(
                        "standard pair transaction selected evidence with "
                        "an incompatible pending state"
                    )
                # Keep the complete trained class, including generations
                # already mirrored as COMMITTED. The store validates the
                # complete signed transaction and changes only newly
                # AVAILABLE members, making the mirror operation idempotent.
                keys.append(key)
        return tuple(sorted(set(keys)))

    def commit_standard_pair_transaction(
            self,
            keys,
            adjacency_update_index):
        if not self.pair_pending_enabled:
            return tuple()
        return (
            self.pair_pending_store
            .commit_available_from_standard_transaction(
                keys,
                adjacency_update_index,
            )
        )

    def sample_inds(
            self,
            data_chunk_length,
            num_mini_batch,
            recent_episode_window=0,
            outcome_support_round=None):
        """
        只从 filled_i 范围内采样，避免未填充 episode 污染邻接训练。
        """
        base_episode_indices = self._recent_episode_indices(
            recent_episode_window
        )
        self.last_sample_candidate_evidence_provenance_rows = []
        valid_slot_count = int(self.filled_i)
        end = int(self.current_i) % int(self.buffer_size)
        recency_order = (
            (end - 1 - np.arange(valid_slot_count, dtype=np.int64))
            % int(self.buffer_size)
        )
        if np.any(self.episode_generation[recency_order] < 0):
            raise RuntimeError(
                "occupied adjacency replay slots are missing episode "
                "generation identities"
            )
        # Outcome support is a property of completed, identity-matched capture
        # episodes.  Do not infer the class from the already scaled credit:
        # graph confidence can legitimately turn a labelled episode's final
        # credit into zero, which would make sampling and the centered
        # baseline use different populations.
        active_capture_episode_mask = np.any(
            self.triplet_capture_quality[..., 0] > 0.0,
            axis=(0, 2),
        )
        candidate_capture_episode_mask = np.any(
            self.capture_candidate_only_matches > 0.0,
            axis=(0, 2),
        )
        capture_episode_mask = (
            active_capture_episode_mask | candidate_capture_episode_mask
        )
        successful_episode_mask = np.any(
            self.success_now[..., 0] > 0.0,
            axis=0,
        )
        pair_evidence_episode_mask = np.any(
            self.pair_to_triplet_transition_score[..., 0] > 0.0,
            axis=(0, 2),
        )
        positive_episode_mask = (
            capture_episode_mask & successful_episode_mask
        )
        negative_episode_mask = (
            capture_episode_mask & ~successful_episode_mask
        )
        pair_positive_episode_mask = (
            pair_evidence_episode_mask & successful_episode_mask
        )
        pair_negative_episode_mask = (
            pair_evidence_episode_mask & ~successful_episode_mask
        )
        pair_evidence_funnel = summarize_pair_evidence_episode_funnel(
            occupied_episode_mask=(self.episode_generation >= 0),
            successful_episode_mask=successful_episode_mask,
            active_capture_episode_mask=active_capture_episode_mask,
            candidate_capture_episode_mask=candidate_capture_episode_mask,
            pair_evidence_episode_mask=pair_evidence_episode_mask,
        )
        pair_evidence_funnel.update(
            summarize_successful_candidate_capture_context(
                successful_episode_mask=successful_episode_mask,
                pair_evidence_episode_mask=pair_evidence_episode_mask,
                active_capture_episode_mask=active_capture_episode_mask,
                candidate_capture_weights=(
                    self.capture_candidate_only_matches
                ),
                candidate_behavior=self.capture_candidate_behavior,
                capture_transition_mask=(
                    np.any(
                        self.triplet_capture_quality[..., 0] > 0.0,
                        axis=2,
                    )
                    | np.any(
                        self.capture_candidate_only_matches > 0.0,
                        axis=2,
                    )
                ),
                terminal_transition_mask=(
                    self.dones_env[..., 0] > 0.0
                ),
            )
        )
        if outcome_support_round is None:
            outcome_support_round = self._next_outcome_support_round
            self._next_outcome_support_round += 1
        outcome_support_round = int(outcome_support_round)
        support_signature = (
            tuple(base_episode_indices.tolist()),
            tuple(recency_order.tolist()),
            tuple(self.episode_generation[recency_order].tolist()),
            positive_episode_mask.tobytes(),
            negative_episode_mask.tobytes(),
            pair_positive_episode_mask.tobytes(),
            pair_negative_episode_mask.tobytes(),
        )
        cached_selection_reused = 0.0
        if self._cached_outcome_support_round == outcome_support_round:
            if self._cached_outcome_support_signature != support_signature:
                raise RuntimeError(
                    "adjacency replay content changed within one outcome "
                    "support round"
                )
            contrast_selection = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in
                self._cached_outcome_support_selection.items()
            }
            cached_selection_reused = 1.0
        else:
            contrast_selection = select_outcome_contrast_complete_episodes(
                base_episode_indices,
                positive_episode_mask,
                negative_episode_mask,
                recency_order,
                positive_eligible_mask=(
                    positive_episode_mask & ~self.outcome_support_used
                ),
                negative_eligible_mask=(
                    negative_episode_mask & ~self.outcome_support_used
                ),
            )
            # Generic capture support is not sufficient for the pair-local
            # objective: a capture episode can have no strict-future pair
            # evidence. Complete the selected population a second time using
            # the exact population that can receive signed pair credit. Keep
            # the pair-specific augmentation out of the disabled ablation.
            pair_credit_enabled = bool(
                self.use_adj_pair_triplet_complementary_credit
                and self.adj_pair_pursuit_credit_coef > 0.0
            )
            if pair_credit_enabled:
                pair_contrast_selection = (
                    select_outcome_contrast_complete_episodes(
                        contrast_selection["episode_indices"],
                        pair_positive_episode_mask,
                        pair_negative_episode_mask,
                        recency_order,
                        positive_eligible_mask=(
                            pair_positive_episode_mask
                            & ~self.outcome_support_used
                        ),
                        negative_eligible_mask=(
                            pair_negative_episode_mask
                            & ~self.outcome_support_used
                        ),
                    )
                )
            else:
                selected_indices = contrast_selection["episode_indices"]
                pair_positive_selected = int(
                    np.sum(pair_positive_episode_mask[selected_indices])
                )
                pair_negative_selected = int(
                    np.sum(pair_negative_episode_mask[selected_indices])
                )
                pair_contrast_available = bool(
                    np.any(pair_positive_episode_mask)
                    and np.any(pair_negative_episode_mask)
                )
                pair_contrast_selection = {
                    "episode_indices": selected_indices,
                    "supplemented_episode_indices": np.empty(
                        (0,),
                        dtype=np.int64,
                    ),
                    "positive_available": float(
                        np.any(pair_positive_episode_mask)
                    ),
                    "negative_available": float(
                        np.any(pair_negative_episode_mask)
                    ),
                    "positive_selected_count": pair_positive_selected,
                    "negative_selected_count": pair_negative_selected,
                    "class_complete": float(
                        pair_positive_selected > 0
                        and pair_negative_selected > 0
                    ),
                    "support_exhausted": float(
                        pair_contrast_available
                        and not (
                            pair_positive_selected > 0
                            and pair_negative_selected > 0
                        )
                    ),
                }
            capture_supplemented = contrast_selection[
                "supplemented_episode_indices"
            ]
            pair_supplemented = pair_contrast_selection[
                "supplemented_episode_indices"
            ]
            supplemented = np.concatenate(
                [capture_supplemented, pair_supplemented]
            ).astype(np.int64, copy=False)
            if np.unique(supplemented).size != supplemented.size:
                raise RuntimeError(
                    "capture and pair replay support selected a duplicate "
                    "episode"
                )
            final_episode_indices = pair_contrast_selection[
                "episode_indices"
            ]
            contrast_selection["episode_indices"] = final_episode_indices
            contrast_selection["supplemented_episode_indices"] = supplemented
            contrast_selection["augmented_count"] = int(
                final_episode_indices.size - base_episode_indices.size
            )
            contrast_selection["positive_selected_count"] = int(
                np.sum(positive_episode_mask[final_episode_indices])
            )
            contrast_selection["negative_selected_count"] = int(
                np.sum(negative_episode_mask[final_episode_indices])
            )
            contrast_selection["augmented_positive_count"] = int(
                np.sum(positive_episode_mask[supplemented])
            )
            contrast_selection["augmented_negative_count"] = int(
                np.sum(negative_episode_mask[supplemented])
            )
            outcome_contrast_available = bool(
                contrast_selection["positive_available"]
                and contrast_selection["negative_available"]
            )
            outcome_class_complete = bool(
                outcome_contrast_available
                and contrast_selection["positive_selected_count"] > 0
                and contrast_selection["negative_selected_count"] > 0
            )
            contrast_selection["class_complete"] = float(
                outcome_class_complete
            )
            contrast_selection["support_exhausted"] = float(
                outcome_contrast_available and not outcome_class_complete
            )
            contrast_selection["outcome_credit_enabled"] = float(
                outcome_class_complete
            )
            contrast_selection["pair_positive_available"] = float(
                pair_contrast_selection["positive_available"]
            )
            contrast_selection["pair_negative_available"] = float(
                pair_contrast_selection["negative_available"]
            )
            contrast_selection["pair_positive_selected_count"] = int(
                pair_contrast_selection["positive_selected_count"]
            )
            contrast_selection["pair_negative_selected_count"] = int(
                pair_contrast_selection["negative_selected_count"]
            )
            contrast_selection["pair_class_complete"] = float(
                pair_contrast_selection["class_complete"]
            )
            contrast_selection["pair_support_exhausted"] = float(
                pair_contrast_selection["support_exhausted"]
            )
            contrast_selection["pair_augmented_count"] = int(
                pair_supplemented.size
            )
            if supplemented.size:
                if np.any(self.outcome_support_used[supplemented]):
                    raise RuntimeError(
                        "outcome support episode was reused across adjacency "
                        "updates"
                    )
                self.outcome_support_used[supplemented] = True
            self._cached_outcome_support_round = outcome_support_round
            self._cached_outcome_support_signature = support_signature
            self._cached_outcome_support_selection = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in contrast_selection.items()
            }
        episode_indices = contrast_selection["episode_indices"]
        valid_episodes = int(episode_indices.size)
        self.last_sample_base_episode_count = int(base_episode_indices.size)
        self.last_sample_episode_count = valid_episodes
        self.last_sample_recent_window = int(recent_episode_window or 0)
        self.last_sample_episode_indices = episode_indices.copy()
        self.last_sample_outcome_contrast_augmented_count = int(
            contrast_selection["augmented_count"]
        )
        self.last_sample_outcome_positive_available = float(
            contrast_selection["positive_available"]
        )
        self.last_sample_outcome_negative_available = float(
            contrast_selection["negative_available"]
        )
        self.last_sample_outcome_positive_episode_count = int(
            contrast_selection["positive_selected_count"]
        )
        self.last_sample_outcome_negative_episode_count = int(
            contrast_selection["negative_selected_count"]
        )
        self.last_sample_outcome_class_complete = float(
            contrast_selection["class_complete"]
        )
        self.last_sample_outcome_support_exhausted = float(
            contrast_selection["support_exhausted"]
        )
        self.last_sample_outcome_credit_enabled = float(
            contrast_selection["outcome_credit_enabled"]
        )
        self.last_sample_outcome_cached_selection_reused = float(
            cached_selection_reused
        )
        self.last_sample_outcome_support_round = int(outcome_support_round)
        self.last_sample_outcome_cross_update_reuse_count = 0
        self.last_sample_outcome_positive_available_count = int(
            contrast_selection["positive_available_count"]
        )
        self.last_sample_outcome_negative_available_count = int(
            contrast_selection["negative_available_count"]
        )
        self.last_sample_outcome_base_positive_count = int(
            contrast_selection["base_positive_count"]
        )
        self.last_sample_outcome_base_negative_count = int(
            contrast_selection["base_negative_count"]
        )
        self.last_sample_outcome_augmented_positive_count = int(
            contrast_selection["augmented_positive_count"]
        )
        self.last_sample_outcome_augmented_negative_count = int(
            contrast_selection["augmented_negative_count"]
        )
        self.last_sample_pair_positive_available = float(
            contrast_selection["pair_positive_available"]
        )
        self.last_sample_pair_negative_available = float(
            contrast_selection["pair_negative_available"]
        )
        self.last_sample_pair_positive_episode_count = int(
            contrast_selection["pair_positive_selected_count"]
        )
        self.last_sample_pair_negative_episode_count = int(
            contrast_selection["pair_negative_selected_count"]
        )
        self.last_sample_pair_class_complete = float(
            contrast_selection["pair_class_complete"]
        )
        self.last_sample_pair_support_exhausted = float(
            contrast_selection["pair_support_exhausted"]
        )
        self.last_sample_pair_augmented_count = int(
            contrast_selection["pair_augmented_count"]
        )
        self.last_sample_pair_evidence_funnel_version = float(
            pair_evidence_funnel["version"]
        )
        for funnel_name in (
                "occupied_episode_count",
                "successful_episode_count",
                "capture_episode_count",
                "successful_capture_episode_count",
                "successful_active_capture_episode_count",
                "successful_candidate_capture_episode_count",
                "pair_evidence_episode_count",
                "pair_positive_episode_count",
                "pair_negative_episode_count",
                "successful_capture_without_pair_evidence_episode_count",
                "pair_evidence_without_capture_episode_count",
                "successful_capture_gap_candidate_only_not_active_episode_count",
                "successful_capture_gap_active_without_strict_pair_episode_count",
                "successful_capture_gap_unclassified_episode_count",
                "successful_capture_gap_reject_reason_contract_valid",
                "successful_candidate_gap_episode_count",
                "successful_candidate_gap_identity_count",
                "successful_candidate_gap_identity_mass",
                "successful_candidate_gap_behavior_margin_mean",
                "successful_candidate_gap_behavior_margin_min",
                "successful_candidate_gap_behavior_margin_max",
                "successful_candidate_gap_behavior_rank_mean",
                "successful_candidate_gap_behavior_rank_min",
                "successful_candidate_gap_behavior_rank_max",
                "successful_candidate_gap_behavior_boundary_crossed_fraction",
                "successful_candidate_gap_behavior_rank1_fraction",
                "successful_capture_gap_terminal_capture_episode_count",
                "successful_capture_gap_last_capture_to_terminal_step_mean",
                "successful_capture_gap_last_capture_to_terminal_step_min",
                "successful_capture_gap_last_capture_to_terminal_step_max",
                "successful_candidate_gap_context_contract_valid",
                "contract_valid"):
            setattr(
                self,
                "last_sample_pair_evidence_funnel_{}".format(funnel_name),
                float(pair_evidence_funnel[funnel_name]),
            )
        recency_age = np.full(self.buffer_size, -1, dtype=np.int64)
        recency_age[recency_order] = np.arange(
            recency_order.size,
            dtype=np.int64,
        )
        supplemented = contrast_selection["supplemented_episode_indices"]
        base_ages = recency_age[base_episode_indices]
        supplemented_ages = recency_age[supplemented]
        self.last_sample_pair_evidence_episode_rows = (
            build_pair_evidence_episode_diagnostic_rows(
                episode_generation=self.episode_generation,
                selected_episode_indices=episode_indices,
                base_episode_indices=base_episode_indices,
                supplemented_episode_indices=supplemented,
                outcome_support_used=self.outcome_support_used,
                successful_episode_mask=successful_episode_mask,
                active_capture_episode_mask=active_capture_episode_mask,
                candidate_capture_weights=(
                    self.capture_candidate_only_matches
                ),
                candidate_behavior=self.capture_candidate_behavior,
                pair_evidence_transition_mask=(
                    np.any(
                        self.pair_to_triplet_transition_score[..., 0] > 0.0,
                        axis=2,
                    )
                ),
                capture_transition_mask=(
                    np.any(
                        self.triplet_capture_quality[..., 0] > 0.0,
                        axis=2,
                    )
                    | np.any(
                        self.capture_candidate_only_matches > 0.0,
                        axis=2,
                    )
                ),
                terminal_transition_mask=(
                    self.dones_env[..., 0] > 0.0
                ),
                capture_identity_candidate_count=(
                    self.capture_identity_candidates[..., 0]
                ),
                recency_age=recency_age,
                num_agents=self.num_agents,
                configured_pair_window=self.adj_pair_pursuit_credit_window,
                outcome_class_complete=contrast_selection[
                    "class_complete"
                ],
                pair_class_complete=contrast_selection[
                    "pair_class_complete"
                ],
                active_capture_event_provenance=(
                    self.capture_active_event_provenance
                ),
                require_active_capture_event_provenance=(
                    self.pair_pending_enabled
                ),
            )
        )
        self.last_sample_outcome_base_age_mean = float(
            np.mean(base_ages) if base_ages.size else np.nan
        )
        self.last_sample_outcome_base_age_max = float(
            np.max(base_ages) if base_ages.size else np.nan
        )
        self.last_sample_outcome_augmented_age_mean = float(
            np.mean(supplemented_ages)
            if supplemented_ages.size else np.nan
        )
        self.last_sample_outcome_augmented_age_max = float(
            np.max(supplemented_ages)
            if supplemented_ages.size else np.nan
        )
        positive_supplement = supplemented[
            positive_episode_mask[supplemented]
        ]
        negative_supplement = supplemented[
            negative_episode_mask[supplemented]
        ]
        self.last_sample_outcome_positive_support_generation = float(
            self.episode_generation[positive_supplement[0]]
            if positive_supplement.size else -1
        )
        self.last_sample_outcome_negative_support_generation = float(
            self.episode_generation[negative_supplement[0]]
            if negative_supplement.size else -1
        )
        self.last_sample_outcome_positive_support_age = float(
            recency_age[positive_supplement[0]]
            if positive_supplement.size else np.nan
        )
        self.last_sample_outcome_negative_support_age = float(
            recency_age[negative_supplement[0]]
            if negative_supplement.size else np.nan
        )
        occupied_outcome_mask = (
            positive_episode_mask | negative_episode_mask
        )
        occupied_outcome_count = int(np.sum(occupied_outcome_mask))
        used_outcome_count = int(np.sum(
            self.outcome_support_used & occupied_outcome_mask
        ))
        self.last_sample_outcome_support_used_count = used_outcome_count
        self.last_sample_outcome_support_used_fraction = float(
            used_outcome_count / float(max(occupied_outcome_count, 1))
        )
        self.last_sample_outcome_generation_update_count = int(
            self.outcome_generation_update_count
        )
        self.last_sample_outcome_slot_overwrite_count = int(
            self.outcome_slot_overwrite_count
        )
        self.last_sample_outcome_generation_conflict_count = int(
            self.outcome_generation_conflict_count
        )
        self.last_sample_outcome_invalid_used_state_count = int(
            self.outcome_invalid_used_state_count
        )

        def _cohort_success_rate(indices):
            indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            positive_count = int(np.sum(positive_episode_mask[indices]))
            negative_count = int(np.sum(negative_episode_mask[indices]))
            labelled_count = positive_count + negative_count
            return (
                float(positive_count / float(labelled_count))
                if labelled_count > 0 else np.nan
            )

        full_buffer_baseline = _cohort_success_rate(recency_order)
        base_cohort_baseline = _cohort_success_rate(base_episode_indices)
        trained_cohort_baseline = _cohort_success_rate(episode_indices)
        self.last_sample_outcome_full_buffer_baseline = full_buffer_baseline
        self.last_sample_outcome_base_cohort_baseline = base_cohort_baseline
        self.last_sample_outcome_trained_cohort_baseline = (
            trained_cohort_baseline
        )
        self.last_sample_outcome_full_trained_baseline_gap = float(
            abs(trained_cohort_baseline - full_buffer_baseline)
            if np.isfinite(trained_cohort_baseline)
            and np.isfinite(full_buffer_baseline)
            else np.nan
        )
        self.last_sample_outcome_trained_capture_episode_count = int(
            np.sum(positive_episode_mask[episode_indices])
            + np.sum(negative_episode_mask[episode_indices])
        )
        self.last_sample_outcome_cohort_centered_sum = 0.0
        self.last_sample_outcome_cohort_center_error = 0.0
        self.last_sample_outcome_cohort_center_valid = 0.0
        self.last_sample_outcome_positive_gate_episode_count = 0
        self.last_sample_outcome_negative_gate_episode_count = 0
        self.last_sample_outcome_positive_credit_episode_count = 0
        self.last_sample_outcome_negative_credit_episode_count = 0
        self.last_sample_outcome_signed_scaling_version = 3.0
        self.last_sample_outcome_graph_advantage_source_ready_fraction = 0.0
        self.last_sample_outcome_graph_confidence_mean = 0.0
        self.last_sample_outcome_graph_confidence_std = 0.0
        self.last_sample_outcome_graph_confidence_p50 = 0.0
        self.last_sample_outcome_graph_confidence_p95 = 0.0
        self.last_sample_outcome_graph_confidence_max = 0.0
        self.last_sample_outcome_positive_graph_confidence_mean = 0.0
        self.last_sample_outcome_positive_graph_confidence_max = 0.0
        self.last_sample_outcome_negative_graph_confidence_mean = 0.0
        self.last_sample_outcome_negative_graph_confidence_max = 0.0
        self.last_sample_outcome_graph_advantage_positive_fraction = 0.0
        self.last_sample_outcome_graph_advantage_negative_fraction = 0.0
        self.last_sample_outcome_graph_advantage_zero_fraction = 0.0
        self.last_sample_outcome_positive_zero_confidence_fraction = 0.0
        self.last_sample_outcome_negative_zero_confidence_fraction = 0.0
        self.last_sample_outcome_gate_to_credit_drop_fraction = 0.0
        self.last_sample_outcome_preclip_positive_mass = 0.0
        self.last_sample_outcome_preclip_negative_mass = 0.0
        self.last_sample_outcome_postclip_positive_mass = 0.0
        self.last_sample_outcome_postclip_negative_mass = 0.0
        self.last_sample_outcome_positive_clip_fraction = 0.0
        self.last_sample_outcome_negative_clip_fraction = 0.0

        batch_size = self.episode_length * valid_episodes
        data_chunk_length = min(int(data_chunk_length), batch_size)
        if batch_size % data_chunk_length != 0:
            raise RuntimeError(
                "adjacency replay chunks must cover the selected transition "
                "population without a truncated tail"
            )

        data_chunks = max(1, batch_size // data_chunk_length)
        num_mini_batch = max(1, min(int(num_mini_batch), data_chunks))
        chunk_starts = (
            np.arange(data_chunks, dtype=np.int64) * data_chunk_length
        )
        chunk_ends = chunk_starts + data_chunk_length - 1
        chunk_episode_membership = np.zeros(
            (data_chunks, valid_episodes),
            dtype=bool,
        )
        for chunk_index in range(data_chunks):
            transition_start = int(chunk_starts[chunk_index])
            transition_end = int(chunk_ends[chunk_index])
            episode_start = transition_start // self.episode_length
            episode_end_inclusive = transition_end // self.episode_length
            if (
                    episode_start != episode_end_inclusive
                    and (
                        transition_start % self.episode_length != 0
                        or (transition_end + 1) % self.episode_length != 0
                    )):
                raise RuntimeError(
                    "adjacency replay chunk crosses a partial episode boundary"
                )
            chunk_episode_membership[
                chunk_index,
                episode_start:episode_end_inclusive + 1,
            ] = True
        pair_credit_partition_enabled = bool(
            self.use_adj_pair_triplet_complementary_credit
            and self.adj_pair_pursuit_credit_coef > 0.0
        )
        selected_pair_evidence = pair_evidence_episode_mask[episode_indices]
        if not pair_credit_partition_enabled:
            selected_pair_evidence = np.zeros_like(
                selected_pair_evidence,
                dtype=bool,
            )
        selected_success = successful_episode_mask[episode_indices]
        pair_partition = partition_pair_contrast_optimizer_chunks(
            chunk_permutation=self.rng.permutation(data_chunks),
            chunk_episode_membership=chunk_episode_membership,
            pair_evidence_episode_mask=selected_pair_evidence,
            episode_success=selected_success,
            num_mini_batch=num_mini_batch,
            pair_partition_slot=0,
        )
        sampler = pair_partition["partitions"]
        self.last_sample_pair_optimizer_atomic_partition = float(
            pair_partition["class_complete"]
        )
        self.last_sample_pair_optimizer_evidence_episode_count = int(
            pair_partition["pair_evidence_episode_count"]
        )
        self.last_sample_pair_optimizer_positive_episode_count = int(
            pair_partition["successful_evidence_episode_count"]
        )
        self.last_sample_pair_optimizer_negative_episode_count = int(
            pair_partition["failed_evidence_episode_count"]
        )
        self.last_sample_pair_optimizer_zero_credit_filler_chunk_count = int(
            pair_partition["pair_zero_credit_filler_chunk_count"]
        )
        self.last_sample_pair_optimizer_pair_partition_chunk_count = int(
            pair_partition["pair_partition_chunk_count"]
        )
        self.last_sample_pair_optimizer_partition_slot = int(
            pair_partition["pair_partition_slot"]
        )
        self.last_sample_pair_optimizer_partition_size_min = int(
            pair_partition["partition_size_min"]
        )
        self.last_sample_pair_optimizer_partition_size_max = int(
            pair_partition["partition_size_max"]
        )
        self.last_sample_pair_optimizer_partition_size_imbalance = int(
            pair_partition["partition_size_imbalance"]
        )
        covered_chunks = np.concatenate(sampler)
        covered_transition_indices = np.concatenate(
            [
                np.arange(
                    int(chunk_index) * data_chunk_length,
                    (int(chunk_index) + 1) * data_chunk_length,
                    dtype=np.int64,
                )
                for chunk_index in covered_chunks
            ]
        )
        if (
                covered_transition_indices.size != batch_size
                or np.unique(covered_transition_indices).size != batch_size):
            raise RuntimeError(
                "adjacency replay mini-batches must cover every selected "
                "transition exactly once"
            )
        covered_episode_offsets = np.unique(
            covered_transition_indices // self.episode_length
        )
        if covered_episode_offsets.size != valid_episodes:
            raise RuntimeError(
                "adjacency replay mini-batches did not train every selected "
                "episode"
            )
        covered_generations = self.episode_generation[
            episode_indices[covered_episode_offsets]
        ]
        if np.unique(covered_generations).size != valid_episodes:
            raise RuntimeError(
                "adjacency replay selected duplicate episode generations"
            )
        self.last_sample_trained_episode_count = int(
            covered_episode_offsets.size
        )
        self.last_sample_dropped_episode_count = int(
            valid_episodes - covered_episode_offsets.size
        )
        self.last_sample_unique_generation_count = int(
            np.unique(covered_generations).size
        )
        self.last_sample_selected_chunk_count = int(data_chunks)
        self.last_sample_yielded_chunk_count = 0
        self.last_sample_dropped_chunk_count = 0
        self.last_sample_duplicate_chunk_count = 0
        self.last_sample_remainder_chunk_count = int(
            data_chunks % num_mini_batch
        )
        self.last_sample_partition_valid = 1.0

        obs_seq = np.take(self.obs[:-1], episode_indices, axis=1)
        obs = obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)

        # Intra-episode dynamic fix:
        # Adj/GAT training needs the alive mask of the *current* graph state.
        # The replay buffer's self.dones stores next-state masks, and a zero
        # prefix makes initially empty capacity slots look alive at s0.  Use
        # Wolfpack's padded all--1 observations to reconstruct current inactive
        # slots for every sampled transition, including s0.
        dones_seq = np.all(
            obs_seq <= -0.999, axis=-1, keepdims=True
        ).astype(np.float32)
        dones = (
            dones_seq.transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_agents, 1)
        )

        # Keep environment termination semantics from the original shifted
        # dones_env path; only per-slot activity is reconstructed from obs.
        dones_env_seq = np.take(
            self.dones_env,
            episode_indices,
            axis=1,
        )
        previous_dones_env_seq = build_previous_done_sequence(dones_env_seq)
        dones_env = (
            previous_dones_env_seq.transpose(1, 0, 2)
            .reshape(batch_size, 1)
        )

        adj_seq_full = np.take(self.adj[:-1], episode_indices, axis=1)
        previous_adj_seq_full = build_previous_adjacency_sequence(
            adj_seq_full
        )
        prob_adj_seq = np.take(
            self.prob_adj[:-1],
            episode_indices,
            axis=1,
        )
        rnn_obs_seq = np.take(self.rnn_obs[:-1], episode_indices, axis=1)
        adj = adj_seq_full.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        previous_adj = (
            previous_adj_seq_full
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_agents, -1)
        )
        prob_adj = prob_adj_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        rnn_obs = rnn_obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        if self.use_avail_acts:
            available_actions_seq = np.take(
                self.avail_acts[:-1],
                episode_indices,
                axis=1,
            )
        else:
            available_actions_seq = np.ones(
                (
                    self.episode_length,
                    valid_episodes,
                    self.num_agents,
                    self.acts.shape[-1],
                ),
                dtype=np.float32,
            )
        available_actions = (
            available_actions_seq
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_agents, -1)
        )
        if (
                not np.all(np.isfinite(available_actions))
                or np.any(available_actions < 0.0)
                or np.any(available_actions > 1.0)
                or np.any(np.sum(available_actions, axis=-1) < 1.0)):
            raise RuntimeError(
                "adjacency replay contains an invalid available-action mask"
            )

        advantage = np.take(self.advantage, episode_indices, axis=1)
        graph_advantage_ready = np.take(
            self.graph_advantage_ready,
            episode_indices,
            axis=1,
        )
        advantages = advantage.transpose(1, 0, 2).reshape(batch_size, -1)

        f_advt = np.take(self.f_advt, episode_indices, axis=1)
        delayed_triplet_credit = np.take(
            self.delayed_triplet_credit,
            episode_indices,
            axis=1,
        )
        delayed_triplet_success_gate = np.take(
            self.delayed_triplet_success_gate,
            episode_indices,
            axis=1,
        )
        delayed_triplet_future_match = np.take(
            self.delayed_triplet_future_match,
            episode_indices,
            axis=1,
        )
        delayed_triplet_future_exact = np.take(
            self.delayed_triplet_future_exact,
            episode_indices,
            axis=1,
        )
        delayed_triplet_future_partial = np.take(
            self.delayed_triplet_future_partial,
            episode_indices,
            axis=1,
        )
        capture_to_win_triplet_credit = np.take(
            self.capture_to_win_triplet_credit,
            episode_indices,
            axis=1,
        )
        capture_to_win_quality_gate = np.take(
            self.capture_to_win_quality_gate,
            episode_indices,
            axis=1,
        )
        pair_pursuit_credit = np.take(
            self.pair_pursuit_credit,
            episode_indices,
            axis=1,
        )
        pair_pursuit_quality = np.take(
            self.pair_pursuit_quality,
            episode_indices,
            axis=1,
        )
        pair_to_triplet_transition_score = np.take(
            self.pair_to_triplet_transition_score,
            episode_indices,
            axis=1,
        )
        triplet_capture_quality = np.take(
            self.triplet_capture_quality,
            episode_indices,
            axis=1,
        )
        candidate_only_capture_quality = np.take(
            self.capture_candidate_only_matches,
            episode_indices,
            axis=1,
        ).astype(np.float32, copy=False)
        candidate_behavior = np.take(
            self.capture_candidate_behavior,
            episode_indices,
            axis=1,
        ).astype(np.float32, copy=False)
        candidate_identity_delta = np.zeros_like(
            candidate_only_capture_quality,
            dtype=np.float32,
        )
        pair_transition_delay = np.take(
            self.pair_transition_delay,
            episode_indices,
            axis=1,
        )
        capture_counts = np.take(
            self.capture_counts,
            episode_indices,
            axis=1,
        )
        capture_matched_count = np.take(
            self.capture_matched_count,
            episode_indices,
            axis=1,
        )
        capture_to_win_episode_success_gate = np.take(
            self.capture_to_win_episode_success_gate,
            episode_indices,
            axis=1,
        )
        failed_episode_capture_count = np.take(
            self.failed_episode_capture_count,
            episode_indices,
            axis=1,
        )
        capture_outcome_diagnostics = np.take(
            self.capture_outcome_diagnostics,
            episode_indices,
            axis=1,
        )
        positive_reward_step = np.take(
            self.positive_reward_step,
            episode_indices,
            axis=1,
        )
        positive_reward_without_capture = np.take(
            self.positive_reward_without_capture,
            episode_indices,
            axis=1,
        )
        offset0_candidate_count = np.take(
            self.offset0_candidate_count,
            episode_indices,
            axis=1,
        )

        # Identify valid selected factors. Advantages were already normalized
        # once per structured graph action in compute_advantage().
        active_seq = ~np.all(obs_seq <= -0.999, axis=-1)
        adj_seq = adj_seq_full[:, :, :, :self.num_factor]
        factor_size = adj_seq.sum(axis=2)
        factor_alive_count = (
            adj_seq * active_seq[:, :, :, None]
        ).sum(axis=2)
        valid_factor = (
            (factor_size > 0)
            & (factor_alive_count == factor_size)
            & (previous_dones_env_seq[..., 0, None] < 0.5)
        )

        selected_success = np.any(
            np.take(self.success_now, episode_indices, axis=1)[..., 0] > 0.0,
            axis=0,
        )
        # Stored pair credit is centered over the complete circular buffer.
        # Reusing a slice of it can expose only one signed branch to the
        # optimizer (71% of run93 pair-target updates), even though the stored
        # full-buffer tensor conserves mass.  Rebuild the exact pair contrast
        # over the final selected/trained episode cohort instead.
        selected_pair_score = (
            pair_to_triplet_transition_score[..., 0]
            * valid_factor.astype(np.float32)
        ).astype(np.float32, copy=False)
        if (
            self.use_adj_pair_triplet_complementary_credit
            and self.adj_pair_pursuit_credit_coef > 0.0
        ):
            selected_pair_credit_cap = 0.0
            if self.adj_pair_pursuit_credit_cap > 0.0:
                selected_valid_graph = np.any(valid_factor, axis=2)
                selected_graph_values = advantage[..., 0][
                    selected_valid_graph
                ]
                if (
                    selected_graph_values.size > 0
                    and not np.isfinite(selected_graph_values).all()
                ):
                    raise RuntimeError(
                        "non-finite graph advantage reached pair cohort scale"
                    )
                selected_graph_abs_scale = (
                    float(np.abs(selected_graph_values).mean())
                    if selected_graph_values.size > 0 else 1.0
                )
                if selected_graph_abs_scale < 1e-6:
                    selected_graph_abs_scale = 1.0
                selected_pair_credit_cap = float(
                    selected_graph_abs_scale
                    * float(self.adj_pair_pursuit_credit_cap)
                )
            selected_pair_credit = scale_optimizer_cohort_pair_credit(
                pair_transition_score=selected_pair_score,
                episode_success=selected_success,
                coefficient=self.adj_pair_pursuit_credit_coef,
                credit_cap=selected_pair_credit_cap,
            )
            pair_pursuit_credit = selected_pair_credit["credit"][..., None]
        else:
            pair_pursuit_credit.fill(0.0)

        # The centered outcome baseline must be defined over the episodes that
        # actually reach this optimizer update.  compute_advantage() maintains
        # a full-buffer diagnostic population, whereas replay support may add
        # or remove episodes before training.  Reusing the stored full-buffer
        # gate here would make a nominally complete 1-positive/1-negative
        # cohort carry a non-zero raw centered sum whenever the full-buffer
        # success rate differs from 0.5.
        if bool(contrast_selection["outcome_credit_enabled"]):
            selected_active_capture_quality = (
                triplet_capture_quality[..., 0].astype(
                    np.float32,
                    copy=False,
                )
            )
            # Every real event appears in exactly one branch: an exact active
            # factor when selected, otherwise its exact canonical candidate.
            # Concatenating before centering gives both branches one common
            # final-optimizer-cohort baseline without duplicating event mass.
            selected_capture_quality = np.concatenate(
                [
                    selected_active_capture_quality,
                    candidate_only_capture_quality,
                ],
                axis=2,
            )
            selected_contrast = (
                compute_capture_to_win_triplet_outcome_advantage(
                    episode_success=selected_success,
                    triplet_capture_quality=selected_capture_quality,
                )
            )
            selected_positive_count = int(
                selected_contrast["successful_capture_episode_count"]
            )
            selected_negative_count = int(
                selected_contrast["failed_capture_episode_count"]
            )
            if selected_positive_count <= 0 or selected_negative_count <= 0:
                raise RuntimeError(
                    "outcome support marked a cohort complete without both "
                    "real completed capture outcomes"
                )
            cohort_gate_all = selected_contrast[
                "triplet_outcome_advantage"
            ].astype(np.float32, copy=False)
            cohort_gate = cohort_gate_all[:, :, :self.num_factor]
            cohort_candidate_gate = cohort_gate_all[:, :, self.num_factor:]
            cohort_episode_total = cohort_gate_all.sum(axis=(0, 2))
            cohort_capture_mask = selected_contrast["capture_episode_mask"]
            cohort_centered_sum = float(
                cohort_episode_total[cohort_capture_mask].sum()
            )
            cohort_center_error = abs(cohort_centered_sum)
            if cohort_center_error > 1e-5:
                raise RuntimeError(
                    "sampled capture-outcome cohort is not centered: {}"
                    .format(cohort_centered_sum)
                )
            valid_graph_transition = np.any(valid_factor, axis=2)
            labelled_graph_transition = np.any(
                np.abs(cohort_gate) > 0.0,
                axis=2,
            )
            cohort_graph_advantage = require_ready_graph_advantage(
                graph_advantage=advantage[..., 0],
                ready_mask=graph_advantage_ready[..., 0],
                labelled_transition=labelled_graph_transition,
            )
            labelled_count = int(np.sum(labelled_graph_transition))
            self.last_sample_outcome_graph_advantage_source_ready_fraction = (
                float(np.sum(
                    graph_advantage_ready[..., 0]
                    & labelled_graph_transition
                )) / float(max(labelled_count, 1))
            )
            scaled_cohort_credit = scale_capture_to_win_outcome_credit(
                triplet_outcome_advantage=cohort_gate,
                graph_advantage=cohort_graph_advantage,
                valid_graph_transition=valid_graph_transition,
                coefficient=self.adj_capture_to_win_credit_coef,
                cap=self.adj_capture_to_win_credit_cap,
                return_diagnostics=True,
            )
            candidate_identity_delta = (
                cohort_candidate_gate
                * float(self.adj_capture_to_win_credit_coef)
            ).astype(np.float32, copy=False)
            if self.adj_capture_to_win_credit_cap > 0.0:
                candidate_identity_delta = np.clip(
                    candidate_identity_delta,
                    -float(self.adj_capture_to_win_credit_cap),
                    float(self.adj_capture_to_win_credit_cap),
                )
            terminal_steps = []
            for replay_slot in episode_indices.tolist():
                terminal_locations = np.flatnonzero(
                    self.dones_env[:, replay_slot, 0] > 0.0
                )
                if terminal_locations.size == 0:
                    raise RuntimeError(
                        "candidate evidence episode has no terminal step"
                    )
                terminal_steps.append(int(terminal_locations[0]))
            base_episode_set = {
                int(index) for index in base_episode_indices.tolist()
            }
            supplemented_episode_set = {
                int(index) for index in supplemented.tolist()
            }
            self.last_sample_candidate_evidence_provenance_rows = (
                build_candidate_evidence_provenance_rows(
                    event_records_by_episode=[
                        self.capture_candidate_event_provenance[int(index)]
                        for index in episode_indices.tolist()
                    ],
                    replay_slot_indices=episode_indices,
                    replay_generations=self.episode_generation[
                        episode_indices
                    ],
                    environment_episode_ids=self.environment_episode_id[
                        episode_indices
                    ],
                    base_selected=np.asarray([
                        int(index) in base_episode_set
                        for index in episode_indices.tolist()
                    ], dtype=np.int64),
                    support_selected=np.asarray([
                        int(index) in supplemented_episode_set
                        for index in episode_indices.tolist()
                    ], dtype=np.int64),
                    outcome_success=selected_success,
                    episode_outcome_advantage=selected_contrast[
                        "episode_outcome_advantage"
                    ],
                    capture_event_mass_per_episode=selected_contrast[
                        "capture_triplet_count_per_episode"
                    ],
                    candidate_coefficient=(
                        self.adj_capture_to_win_credit_coef
                    ),
                    candidate_identity_delta=candidate_identity_delta,
                    candidate_behavior=candidate_behavior,
                    terminal_steps=np.asarray(
                        terminal_steps,
                        dtype=np.int64,
                    ),
                )
            )
            capture_to_win_quality_gate = cohort_gate[..., None]
            capture_to_win_triplet_credit = scaled_cohort_credit[
                "credit"
            ][..., None]
            self.last_sample_outcome_cohort_centered_sum = (
                cohort_centered_sum
            )
            self.last_sample_outcome_cohort_center_error = (
                cohort_center_error
            )
            self.last_sample_outcome_cohort_center_valid = 1.0
            self.last_sample_outcome_positive_gate_episode_count = int(
                np.sum(np.any(cohort_gate_all > 0.0, axis=(0, 2)))
            )
            self.last_sample_outcome_negative_gate_episode_count = int(
                np.sum(np.any(cohort_gate_all < 0.0, axis=(0, 2)))
            )
            self.last_sample_outcome_positive_credit_episode_count = int(
                np.sum(
                    np.any(
                        scaled_cohort_credit["credit"] > 0.0,
                        axis=(0, 2),
                    )
                    | np.any(candidate_identity_delta > 0.0, axis=(0, 2))
                )
            )
            self.last_sample_outcome_negative_credit_episode_count = int(
                np.sum(
                    np.any(
                        scaled_cohort_credit["credit"] < 0.0,
                        axis=(0, 2),
                    )
                    | np.any(candidate_identity_delta < 0.0, axis=(0, 2))
                )
            )
            self.last_sample_outcome_graph_confidence_mean = float(
                scaled_cohort_credit["graph_confidence_mean"]
            )
            self.last_sample_outcome_graph_confidence_std = float(
                scaled_cohort_credit["graph_confidence_std"]
            )
            self.last_sample_outcome_graph_confidence_p50 = float(
                scaled_cohort_credit["graph_confidence_p50"]
            )
            self.last_sample_outcome_graph_confidence_p95 = float(
                scaled_cohort_credit["graph_confidence_p95"]
            )
            self.last_sample_outcome_graph_confidence_max = float(
                scaled_cohort_credit["graph_confidence_max"]
            )
            self.last_sample_outcome_positive_graph_confidence_mean = float(
                scaled_cohort_credit["positive_graph_confidence_mean"]
            )
            self.last_sample_outcome_positive_graph_confidence_max = float(
                scaled_cohort_credit["positive_graph_confidence_max"]
            )
            self.last_sample_outcome_negative_graph_confidence_mean = float(
                scaled_cohort_credit["negative_graph_confidence_mean"]
            )
            self.last_sample_outcome_negative_graph_confidence_max = float(
                scaled_cohort_credit["negative_graph_confidence_max"]
            )
            self.last_sample_outcome_graph_advantage_positive_fraction = float(
                scaled_cohort_credit[
                    "labelled_graph_advantage_positive_fraction"
                ]
            )
            self.last_sample_outcome_graph_advantage_negative_fraction = float(
                scaled_cohort_credit[
                    "labelled_graph_advantage_negative_fraction"
                ]
            )
            self.last_sample_outcome_graph_advantage_zero_fraction = float(
                scaled_cohort_credit["labelled_graph_advantage_zero_fraction"]
            )
            self.last_sample_outcome_positive_zero_confidence_fraction = float(
                scaled_cohort_credit["positive_zero_confidence_fraction"]
            )
            self.last_sample_outcome_negative_zero_confidence_fraction = float(
                scaled_cohort_credit["negative_zero_confidence_fraction"]
            )
            self.last_sample_outcome_gate_to_credit_drop_fraction = float(
                scaled_cohort_credit["gate_to_credit_drop_fraction"]
            )
            self.last_sample_outcome_preclip_positive_mass = float(
                scaled_cohort_credit["preclip_positive_mass"]
            )
            self.last_sample_outcome_preclip_negative_mass = float(
                scaled_cohort_credit["preclip_negative_mass"]
            )
            self.last_sample_outcome_postclip_positive_mass = float(
                scaled_cohort_credit["postclip_positive_mass"]
            )
            self.last_sample_outcome_postclip_negative_mass = float(
                scaled_cohort_credit["postclip_negative_mass"]
            )
            self.last_sample_outcome_positive_clip_fraction = float(
                scaled_cohort_credit["positive_clip_fraction"]
            )
            self.last_sample_outcome_negative_clip_fraction = float(
                scaled_cohort_credit["negative_clip_fraction"]
            )
        else:
            # A one-sided optimizer cohort is not a valid contrast. Preserve
            # all other replay fields and graph/pair objectives, but suppress
            # only capture-outcome training inputs for this cohort.
            capture_to_win_triplet_credit.fill(0.0)
            capture_to_win_quality_gate.fill(0.0)
            candidate_identity_delta.fill(0.0)
        # Do not normalize again over factor slots: that would overweight
        # larger rosters solely because they select more factors.
        f_advt = np.where(
            np.isfinite(f_advt),
            f_advt,
            0.0,
        ).astype(np.float32, copy=False)
        f_advt[~valid_factor[..., None]] = 0.0
        f_advt = np.clip(f_advt, -5.0, 5.0)
        f_advts = f_advt.transpose(1, 0, 2, 3).reshape(batch_size, self.num_factor, -1)
        delayed_triplet_credit = np.where(
            np.isfinite(delayed_triplet_credit),
            delayed_triplet_credit,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_credit[~valid_factor[..., None]] = 0.0
        delayed_triplet_credit = np.clip(delayed_triplet_credit, -5.0, 5.0)
        delayed_triplet_credits = (
            delayed_triplet_credit
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_success_gate = np.where(
            np.isfinite(delayed_triplet_success_gate),
            delayed_triplet_success_gate,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_success_gate[~valid_factor[..., None]] = 0.0
        delayed_triplet_success_gate = np.clip(
            delayed_triplet_success_gate,
            0.0,
            1.0,
        )
        delayed_triplet_success_gates = (
            delayed_triplet_success_gate
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_future_match = np.where(
            np.isfinite(delayed_triplet_future_match),
            delayed_triplet_future_match,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_future_match[~valid_factor[..., None]] = 0.0
        delayed_triplet_future_match = np.clip(
            delayed_triplet_future_match,
            0.0,
            1.0,
        )
        delayed_triplet_future_matches = (
            delayed_triplet_future_match
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_future_exact = np.where(
            np.isfinite(delayed_triplet_future_exact),
            delayed_triplet_future_exact,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_future_exact[~valid_factor[..., None]] = 0.0
        delayed_triplet_future_exact = np.clip(
            delayed_triplet_future_exact,
            0.0,
            1.0,
        )
        delayed_triplet_future_exacts = (
            delayed_triplet_future_exact
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_future_partial = np.where(
            np.isfinite(delayed_triplet_future_partial),
            delayed_triplet_future_partial,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_future_partial[~valid_factor[..., None]] = 0.0
        delayed_triplet_future_partial = np.clip(
            delayed_triplet_future_partial,
            0.0,
            1.0,
        )
        delayed_triplet_future_partials = (
            delayed_triplet_future_partial
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        capture_to_win_triplet_credit = np.where(
            np.isfinite(capture_to_win_triplet_credit),
            capture_to_win_triplet_credit,
            0.0,
        ).astype(np.float32, copy=False)
        capture_to_win_triplet_credit[~valid_factor[..., None]] = 0.0
        capture_to_win_triplet_credit = np.clip(
            capture_to_win_triplet_credit,
            -5.0,
            5.0,
        )
        capture_to_win_triplet_credits = (
            capture_to_win_triplet_credit
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        candidate_identity_delta = np.where(
            np.isfinite(candidate_identity_delta),
            candidate_identity_delta,
            0.0,
        ).astype(np.float32, copy=False)
        candidate_identity_deltas = (
            candidate_identity_delta[..., None]
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_candidate_factor, -1)
        )
        candidate_capture_context = np.where(
            np.isfinite(candidate_only_capture_quality),
            np.maximum(candidate_only_capture_quality, 0.0),
            0.0,
        ).astype(np.float32, copy=False)
        candidate_capture_contexts = (
            candidate_capture_context[..., None]
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_candidate_factor, -1)
        )
        candidate_behaviors = (
            candidate_behavior
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_candidate_factor, 4)
        )
        capture_to_win_quality_gate = np.where(
            np.isfinite(capture_to_win_quality_gate),
            capture_to_win_quality_gate,
            0.0,
        ).astype(np.float32, copy=False)
        capture_to_win_quality_gate[~valid_factor[..., None]] = 0.0
        # The gate is a signed, centered outcome advantage.  Clipping it to
        # [0, 1] would silently erase all failed-capture evidence in replay.
        capture_to_win_quality_gate = np.clip(
            capture_to_win_quality_gate,
            -1.0,
            1.0,
        )
        capture_to_win_quality_gates = (
            capture_to_win_quality_gate
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_pursuit_credit = np.where(
            np.isfinite(pair_pursuit_credit),
            pair_pursuit_credit,
            0.0,
        ).astype(np.float32, copy=False)
        pair_pursuit_credit[~valid_factor[..., None]] = 0.0
        pair_pursuit_credit = np.clip(pair_pursuit_credit, -5.0, 5.0)
        pair_pursuit_credits = (
            pair_pursuit_credit
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_evidence_transition = np.repeat(
            selected_pair_evidence.astype(np.float32, copy=False),
            self.episode_length,
        ).reshape(batch_size, 1)
        episode_ordinal_transition = np.repeat(
            np.arange(valid_episodes, dtype=np.float32),
            self.episode_length,
        ).reshape(batch_size, 1)
        episode_step_transition = np.tile(
            np.arange(self.episode_length, dtype=np.float32),
            valid_episodes,
        ).reshape(batch_size, 1)
        replay_population_provenance = np.concatenate(
            [
                pair_evidence_transition,
                episode_ordinal_transition,
                episode_step_transition,
            ],
            axis=1,
        )
        if replay_population_provenance.shape != (batch_size, 3):
            raise RuntimeError(
                "replay population provenance has shape {}, expected {}"
                .format(
                    replay_population_provenance.shape,
                    (batch_size, 3),
                )
            )
        if not np.all(
                (pair_evidence_transition == 0.0)
                | (pair_evidence_transition == 1.0)):
            raise RuntimeError(
                "pair-evidence transition provenance must be binary"
            )
        pair_pursuit_quality = np.where(
            np.isfinite(pair_pursuit_quality),
            pair_pursuit_quality,
            0.0,
        ).astype(np.float32, copy=False)
        pair_pursuit_quality[~valid_factor[..., None]] = 0.0
        pair_pursuit_quality = np.clip(pair_pursuit_quality, 0.0, 1.0)
        pair_pursuit_qualities = (
            pair_pursuit_quality
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_to_triplet_transition_score = np.where(
            np.isfinite(pair_to_triplet_transition_score),
            pair_to_triplet_transition_score,
            0.0,
        ).astype(np.float32, copy=False)
        pair_to_triplet_transition_score[~valid_factor[..., None]] = 0.0
        pair_to_triplet_transition_score = np.clip(
            pair_to_triplet_transition_score,
            0.0,
            1.0,
        )
        pair_to_triplet_transition_scores = (
            pair_to_triplet_transition_score
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        triplet_capture_quality = np.where(
            np.isfinite(triplet_capture_quality),
            triplet_capture_quality,
            0.0,
        ).astype(np.float32, copy=False)
        triplet_capture_quality[~valid_factor[..., None]] = 0.0
        triplet_capture_quality = np.maximum(
            triplet_capture_quality,
            0.0,
        )
        triplet_capture_qualities = (
            triplet_capture_quality
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_transition_delay = np.where(
            np.isfinite(pair_transition_delay),
            np.maximum(pair_transition_delay, 0.0),
            0.0,
        ).astype(np.float32, copy=False)
        pair_transition_delay[~valid_factor[..., None]] = 0.0
        pair_transition_delays = (
            pair_transition_delay
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )

        def _flatten_graph_diagnostic(values):
            values = np.where(
                np.isfinite(values),
                np.maximum(values, 0.0),
                0.0,
            ).astype(np.float32, copy=False)
            return values.transpose(1, 0, 2).reshape(batch_size, -1)

        capture_counts_flat = _flatten_graph_diagnostic(capture_counts)
        capture_matched_count_flat = _flatten_graph_diagnostic(
            capture_matched_count
        )
        capture_to_win_episode_success_gate_flat = (
            _flatten_graph_diagnostic(
                capture_to_win_episode_success_gate
            )
        )
        failed_episode_capture_count_flat = _flatten_graph_diagnostic(
            failed_episode_capture_count
        )
        capture_outcome_diagnostics_flat = np.where(
            np.isfinite(capture_outcome_diagnostics),
            capture_outcome_diagnostics,
            0.0,
        ).astype(np.float32, copy=False)
        capture_outcome_diagnostics_flat = (
            capture_outcome_diagnostics_flat
            .transpose(1, 0, 2)
            .reshape(batch_size, CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH)
        )
        positive_reward_step_flat = _flatten_graph_diagnostic(
            positive_reward_step
        )
        positive_reward_without_capture_flat = _flatten_graph_diagnostic(
            positive_reward_without_capture
        )
        offset0_candidate_count_flat = _flatten_graph_diagnostic(
            offset0_candidate_count
        )

        # ``dones`` and ``dones_env`` were already flattened episode-major
        # above. Re-transposing them here used to be correct when they were
        # still sequence tensors, but now corrupts the axis contract (and for
        # ``dones`` raises "axes don't match array"). Fail loudly if a future
        # edit changes either layout.
        if dones.shape != (batch_size, self.num_agents, 1):
            raise RuntimeError(
                "flattened agent activity mask must have shape {}, got {}"
                .format((batch_size, self.num_agents, 1), dones.shape)
            )
        if dones_env.shape != (batch_size, 1):
            raise RuntimeError(
                "flattened previous-done mask must have shape {}, got {}"
                .format((batch_size, 1), dones_env.shape)
            )

        if self.use_same_share_obs:
            share_obs_seq = np.take(
                self.share_obs[:-1],
                episode_indices,
                axis=1,
            )
            share_obs = share_obs_seq.transpose(1, 0, 2).reshape(batch_size, -1)
        else:
            share_obs_seq = np.take(
                self.share_obs[:-1],
                episode_indices,
                axis=1,
            )
            share_obs = share_obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents,
                                                                    -1)

        production_flat_fields = (
            obs,
            share_obs,
            dones,
            dones_env,
            adj,
            prob_adj,
            advantages,
            f_advts,
            delayed_triplet_credits,
            delayed_triplet_success_gates,
            delayed_triplet_future_matches,
            delayed_triplet_future_exacts,
            delayed_triplet_future_partials,
            capture_to_win_triplet_credits,
            capture_to_win_quality_gates,
            pair_pursuit_credits,
            pair_pursuit_qualities,
            pair_to_triplet_transition_scores,
            triplet_capture_qualities,
            rnn_obs,
            pair_transition_delays,
            capture_counts_flat,
            capture_matched_count_flat,
            positive_reward_without_capture_flat,
            offset0_candidate_count_flat,
            positive_reward_step_flat,
            previous_adj,
            capture_to_win_episode_success_gate_flat,
            failed_episode_capture_count_flat,
            capture_outcome_diagnostics_flat,
            candidate_identity_deltas,
            candidate_capture_contexts,
            candidate_behaviors,
            replay_population_provenance,
            available_actions,
        )
        self._capture_pair_pending_snapshots(
            production_flat_fields,
            episode_indices=episode_indices,
            selected_pair_evidence=selected_pair_evidence,
            base_episode_indices=base_episode_indices,
            data_chunk_length=data_chunk_length,
        )

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            dones_batch = []
            dones_env_batch = []
            adj_batch = []
            prob_adj_batch = []
            advantages_batch = []
            f_advts_batch = []
            delayed_triplet_credit_batch = []
            delayed_triplet_success_gate_batch = []
            delayed_triplet_future_match_batch = []
            delayed_triplet_future_exact_batch = []
            delayed_triplet_future_partial_batch = []
            capture_to_win_triplet_credit_batch = []
            capture_to_win_quality_gate_batch = []
            pair_pursuit_credit_batch = []
            pair_pursuit_quality_batch = []
            pair_to_triplet_transition_score_batch = []
            triplet_capture_quality_batch = []
            rnn_obs_batch = []
            pair_transition_delay_batch = []
            capture_counts_batch = []
            capture_matched_count_batch = []
            positive_reward_step_batch = []
            positive_reward_without_capture_batch = []
            offset0_candidate_count_batch = []
            previous_adj_batch = []
            capture_to_win_episode_success_gate_batch = []
            failed_episode_capture_count_batch = []
            capture_outcome_diagnostics_batch = []
            candidate_identity_delta_batch = []
            candidate_capture_context_batch = []
            candidate_behavior_batch = []
            replay_population_provenance_batch = []
            available_actions_batch = []

            for i in indices:
                ind = i * data_chunk_length
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                dones_batch.append(dones[ind:ind + data_chunk_length])
                dones_env_batch.append(dones_env[ind:ind + data_chunk_length])
                adj_batch.append(adj[ind:ind + data_chunk_length])
                prob_adj_batch.append(prob_adj[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                f_advts_batch.append(f_advts[ind:ind + data_chunk_length])
                delayed_triplet_credit_batch.append(
                    delayed_triplet_credits[ind:ind + data_chunk_length]
                )
                delayed_triplet_success_gate_batch.append(
                    delayed_triplet_success_gates[ind:ind + data_chunk_length]
                )
                delayed_triplet_future_match_batch.append(
                    delayed_triplet_future_matches[ind:ind + data_chunk_length]
                )
                delayed_triplet_future_exact_batch.append(
                    delayed_triplet_future_exacts[ind:ind + data_chunk_length]
                )
                delayed_triplet_future_partial_batch.append(
                    delayed_triplet_future_partials[ind:ind + data_chunk_length]
                )
                capture_to_win_triplet_credit_batch.append(
                    capture_to_win_triplet_credits[ind:ind + data_chunk_length]
                )
                capture_to_win_quality_gate_batch.append(
                    capture_to_win_quality_gates[ind:ind + data_chunk_length]
                )
                pair_pursuit_credit_batch.append(
                    pair_pursuit_credits[ind:ind + data_chunk_length]
                )
                pair_pursuit_quality_batch.append(
                    pair_pursuit_qualities[ind:ind + data_chunk_length]
                )
                pair_to_triplet_transition_score_batch.append(
                    pair_to_triplet_transition_scores[
                        ind:ind + data_chunk_length
                    ]
                )
                triplet_capture_quality_batch.append(
                    triplet_capture_qualities[ind:ind + data_chunk_length]
                )
                rnn_obs_batch.append(rnn_obs[ind:ind + data_chunk_length])
                pair_transition_delay_batch.append(
                    pair_transition_delays[ind:ind + data_chunk_length]
                )
                capture_counts_batch.append(
                    capture_counts_flat[ind:ind + data_chunk_length]
                )
                capture_matched_count_batch.append(
                    capture_matched_count_flat[ind:ind + data_chunk_length]
                )
                positive_reward_step_batch.append(
                    positive_reward_step_flat[ind:ind + data_chunk_length]
                )
                positive_reward_without_capture_batch.append(
                    positive_reward_without_capture_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                offset0_candidate_count_batch.append(
                    offset0_candidate_count_flat[ind:ind + data_chunk_length]
                )
                previous_adj_batch.append(
                    previous_adj[ind:ind + data_chunk_length]
                )
                capture_to_win_episode_success_gate_batch.append(
                    capture_to_win_episode_success_gate_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                failed_episode_capture_count_batch.append(
                    failed_episode_capture_count_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                capture_outcome_diagnostics_batch.append(
                    capture_outcome_diagnostics_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                candidate_identity_delta_batch.append(
                    candidate_identity_deltas[
                        ind:ind + data_chunk_length
                    ]
                )
                candidate_capture_context_batch.append(
                    candidate_capture_contexts[
                        ind:ind + data_chunk_length
                    ]
                )
                candidate_behavior_batch.append(
                    candidate_behaviors[ind:ind + data_chunk_length]
                )
                replay_population_provenance_batch.append(
                    replay_population_provenance[
                        ind:ind + data_chunk_length
                    ]
                )
                available_actions_batch.append(
                    available_actions[ind:ind + data_chunk_length]
                )

            # This is the execution-side count: a partition is considered
            # yielded only when the consumer receives its training sample.
            # The runner independently counts chunks after train_adj_on_batch
            # returns, closing selected -> yielded -> trained accounting.
            self.last_sample_yielded_chunk_count += int(len(indices))
            yield (
                np.stack(obs_batch, axis=0),
                np.stack(share_obs_batch, axis=0),
                np.stack(dones_batch, axis=0),
                np.stack(dones_env_batch, axis=0),
                np.stack(adj_batch, axis=0),
                np.stack(prob_adj_batch, axis=0),
                np.stack(advantages_batch, axis=0),
                np.stack(f_advts_batch, axis=0),
                np.stack(delayed_triplet_credit_batch, axis=0),
                np.stack(delayed_triplet_success_gate_batch, axis=0),
                np.stack(delayed_triplet_future_match_batch, axis=0),
                np.stack(delayed_triplet_future_exact_batch, axis=0),
                np.stack(delayed_triplet_future_partial_batch, axis=0),
                np.stack(capture_to_win_triplet_credit_batch, axis=0),
                np.stack(capture_to_win_quality_gate_batch, axis=0),
                np.stack(pair_pursuit_credit_batch, axis=0),
                np.stack(pair_pursuit_quality_batch, axis=0),
                np.stack(pair_to_triplet_transition_score_batch, axis=0),
                np.stack(triplet_capture_quality_batch, axis=0),
                np.stack(rnn_obs_batch, axis=0),
                np.stack(pair_transition_delay_batch, axis=0),
                np.stack(capture_counts_batch, axis=0),
                np.stack(capture_matched_count_batch, axis=0),
                np.stack(positive_reward_without_capture_batch, axis=0),
                np.stack(offset0_candidate_count_batch, axis=0),
                np.stack(positive_reward_step_batch, axis=0),
                np.stack(previous_adj_batch, axis=0),
                np.stack(capture_to_win_episode_success_gate_batch, axis=0),
                np.stack(failed_episode_capture_count_batch, axis=0),
                np.stack(capture_outcome_diagnostics_batch, axis=0),
                np.stack(candidate_identity_delta_batch, axis=0),
                np.stack(candidate_capture_context_batch, axis=0),
                np.stack(candidate_behavior_batch, axis=0),
                np.stack(replay_population_provenance_batch, axis=0),
                np.stack(available_actions_batch, axis=0),
            )
