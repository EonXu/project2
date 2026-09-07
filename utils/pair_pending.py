"""Fail-closed foundations for bounded strict-pair evidence pending.

This module is intentionally NumPy-only. It owns the state, immutable payload,
bounded-age, stale-trust, mass-reconstruction, and checkpoint contracts used
by the production runner's pair-only outer transaction. Keeping storage and
state transitions independent from the trainer prevents a partially trained
cohort from being silently marked consumed.
"""

from __future__ import annotations

import copy

import numpy as np

from .pair_credit import scale_optimizer_cohort_pair_credit


PAIR_PENDING_DIAGNOSTIC_VERSION = 3
PAIR_PENDING_CHECKPOINT_VERSION = 2
PAIR_PENDING_BATCH_FIELD_COUNT = 35

PAIR_EVIDENCE_AVAILABLE_IN_REPLAY = "AVAILABLE_IN_REPLAY"
PAIR_EVIDENCE_PENDING = "PENDING"
PAIR_EVIDENCE_PREPARED = "PREPARED"
PAIR_EVIDENCE_COMMITTED = "COMMITTED"
PAIR_EVIDENCE_EXPIRED = "EXPIRED"

_LIVE_STATES = (
    PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
    PAIR_EVIDENCE_PENDING,
)
_KNOWN_STATES = _LIVE_STATES + (
    PAIR_EVIDENCE_PREPARED,
    PAIR_EVIDENCE_COMMITTED,
    PAIR_EVIDENCE_EXPIRED,
)


class PairPendingZeroGradientError(RuntimeError):
    """A valid pending cohort has no actionable aggregate direction.

    Signed outcome evidence can be class-complete and non-zero while its
    current positive and negative adjacency Jacobians cancel exactly, or while
    the bounded PPO surrogate is locally flat after clipping. Such a cohort
    must not be committed or discarded, but it is a bounded no-action result
    rather than a process-fatal implementation failure. The runner atomically
    restores training/RNG/store state and leaves the evidence live so a later
    cohort can become actionable before TTL expiry.
    """


PAIR_OPTIMIZER_RECOVERABLE_NOOP_REASONS = {
    "zero_adam_displacement": 1,
    "zero_clipped_gradient": 2,
    "zero_clipped_pair_dot": 3,
    "zero_candidate_adam_displacement": 4,
    "zero_candidate_clipped_gradient": 5,
    "zero_candidate_final_descent": 6,
    "zero_final_pair_descent": 7,
    "zero_final_candidate_descent": 8,
    "selection_state_no_compatible_exact_writeback": 9,
}


class PairOptimizerRecoverableNoOpError(RuntimeError):
    """A finite strict-pair proposal made no commit-capable displacement.

    This typed control outcome is raised only *after* the complete adjacency
    transaction has been restored and verified.  It deliberately differs from
    data-corruption failures: a float32 Adam step may quantize to zero, or a
    finite clipped direction may underflow to a zero pair component.  Neither
    state can satisfy the strict positive-update contract, but neither damages
    parameters, optimizer history, evidence, lifecycle state, or RNG.  The
    runner may therefore count the partition as a recoverable no-op and move
    to the next batch without consuming pair evidence.
    """

    def __init__(self, reason, diagnostics, target_count):
        if reason not in PAIR_OPTIMIZER_RECOVERABLE_NOOP_REASONS:
            raise ValueError(
                "unknown pair optimizer recoverable no-op reason: {}".format(
                    reason
                )
            )
        super().__init__(
            "pair optimizer recoverable no-op: {}".format(reason)
        )
        self.reason = str(reason)
        self.reason_code = int(
            PAIR_OPTIMIZER_RECOVERABLE_NOOP_REASONS[reason]
        )
        self.diagnostics = {
            str(name): float(value)
            for name, value in diagnostics.items()
        }
        self.target_count = float(target_count)
        self.atomic_rollback_complete = True


def _require_int(value, name, minimum=None):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be an integer".format(name))
    if minimum is not None and result < int(minimum):
        raise ValueError(
            "{} must be at least {}, got {}".format(name, minimum, result)
        )
    return result


def _require_finite(value, name, minimum=None):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    if minimum is not None and result < float(minimum):
        raise ValueError(
            "{} must be at least {}, got {}".format(name, minimum, result)
        )
    return result


def _normalize_slots(values):
    slots = tuple(int(value) for value in values)
    if not slots or len(set(slots)) != len(slots) or min(slots) < 0:
        raise ValueError(
            "participant_slots must be a non-empty tuple of unique "
            "non-negative indices"
        )
    return slots


def _normalize_identities(values):
    identities = []
    for value in values:
        identity = tuple(sorted(int(node) for node in value))
        if len(identity) < 2 or len(set(identity)) != len(identity):
            raise ValueError(
                "canonical pair identities must contain unique participants"
            )
        identities.append(identity)
    identities = tuple(sorted(set(identities)))
    if not identities:
        raise ValueError("at least one strict pair identity is required")
    return identities


def normalize_pair_pending_metadata(metadata):
    """Validate and detach one strict-pair evidence provenance record."""
    if not isinstance(metadata, dict):
        raise ValueError("pair pending metadata must be a dictionary")
    required = (
        "policy_id",
        "replay_generation",
        "environment_episode_id",
        "episode_ordinal",
        "capture_event_id",
        "prey_id",
        "participant_slots",
        "canonical_pair_identities",
        "factor_order",
        "sign",
        "outcome_success",
        "first_seen_adj_update",
        "original_last_valid_adj_update",
        "behavior_policy_version",
        "source_class",
        "target_bearing_transition_count",
        "raw_event_quality",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(
            "pair pending metadata is missing fields: {}".format(
                ", ".join(missing)
            )
        )

    sign = _require_int(metadata["sign"], "sign")
    if sign not in (-1, 1):
        raise ValueError("strict pair evidence sign must be -1 or +1")
    outcome_success = bool(metadata["outcome_success"])
    if outcome_success != (sign > 0):
        raise ValueError(
            "strict pair evidence sign disagrees with the terminal outcome"
        )
    participants = _normalize_slots(metadata["participant_slots"])
    identities = _normalize_identities(metadata["canonical_pair_identities"])
    factor_order = _require_int(metadata["factor_order"], "factor_order", 2)
    if any(len(identity) != factor_order for identity in identities):
        raise ValueError(
            "factor_order does not match every canonical pair identity"
        )
    if any(not set(identity).issubset(participants) for identity in identities):
        raise ValueError(
            "canonical pair identity is not contained in participant_slots"
        )

    generation = _require_int(
        metadata["replay_generation"], "replay_generation", 0
    )
    environment_episode_id = _require_int(
        metadata["environment_episode_id"], "environment_episode_id"
    )
    episode_ordinal = _require_int(
        metadata["episode_ordinal"], "episode_ordinal", 0
    )
    capture_event_id = _require_int(
        metadata["capture_event_id"], "capture_event_id"
    )
    prey_id = _require_int(metadata["prey_id"], "prey_id")
    event_provenance_available = bool(
        metadata.get("event_provenance_available", False)
    )
    if event_provenance_available:
        if (
                environment_episode_id < 0
                or capture_event_id < 0
                or prey_id < 0):
            raise ValueError(
                "available pair event provenance requires non-negative "
                "episode, event, and prey identifiers"
            )
    else:
        if capture_event_id >= 0 or prey_id >= 0:
            raise ValueError(
                "unavailable event provenance must use negative event/prey "
                "sentinels"
            )

    first_seen = _require_int(
        metadata["first_seen_adj_update"], "first_seen_adj_update", 0
    )
    last_valid = _require_int(
        metadata["original_last_valid_adj_update"],
        "original_last_valid_adj_update",
        first_seen,
    )
    normalized = {
        "policy_id": str(metadata["policy_id"]),
        "replay_generation": generation,
        "environment_episode_id": environment_episode_id,
        "episode_ordinal": episode_ordinal,
        "capture_event_id": capture_event_id,
        "prey_id": prey_id,
        "participant_slots": participants,
        "participant_count": len(participants),
        "canonical_pair_identities": identities,
        "factor_order": factor_order,
        "sign": sign,
        "outcome_success": outcome_success,
        "first_seen_adj_update": first_seen,
        "original_last_valid_adj_update": last_valid,
        "behavior_policy_version": _require_int(
            metadata["behavior_policy_version"],
            "behavior_policy_version",
            0,
        ),
        "source_class": str(metadata["source_class"]),
        "target_bearing_transition_count": _require_int(
            metadata["target_bearing_transition_count"],
            "target_bearing_transition_count",
            1,
        ),
        "raw_event_quality": _require_finite(
            metadata["raw_event_quality"], "raw_event_quality", 0.0
        ),
        "event_provenance_available": event_provenance_available,
    }
    if not normalized["policy_id"] or not normalized["source_class"]:
        raise ValueError("policy_id and source_class must be non-empty")
    return normalized


def pair_pending_generation_key(metadata):
    normalized = normalize_pair_pending_metadata(metadata)
    return (
        normalized["policy_id"],
        normalized["replay_generation"],
    )


def pair_pending_event_key(metadata):
    normalized = normalize_pair_pending_metadata(metadata)
    if not normalized["event_provenance_available"]:
        return None
    return (
        normalized["policy_id"],
        normalized["environment_episode_id"],
        normalized["capture_event_id"],
        normalized["prey_id"],
        normalized["participant_slots"],
    )


def pair_pending_entry_key(metadata):
    """Return the immutable evidence key, not just the replay generation.

    One completed rollout generation may contain more than one real capture
    event.  Each event is a distinct evidence item, while the generation stays
    the atomic replay/training population.  Keeping both levels in the key
    prevents either silently dropping later events or replaying one episode
    once per event.
    """
    normalized = normalize_pair_pending_metadata(metadata)
    return (
        normalized["policy_id"],
        normalized["replay_generation"],
        normalized["environment_episode_id"],
        normalized["capture_event_id"],
        normalized["prey_id"],
        normalized["participant_slots"],
    )


def freeze_pair_pending_batch(batch):
    """Copy one 35-field adjacency batch and make every array read-only.

    The copy owns its storage.  No returned array can be a view into the
    circular replay, and no autograd object is accepted.
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError("pair pending batch must be a tuple or list")
    if len(batch) != PAIR_PENDING_BATCH_FIELD_COUNT:
        raise ValueError(
            "pair pending batch must contain {} fields, got {}".format(
                PAIR_PENDING_BATCH_FIELD_COUNT, len(batch)
            )
        )
    frozen = []
    leading_shape = None
    for field_index, value in enumerate(batch):
        if hasattr(value, "detach"):
            raise ValueError(
                "pair pending batch field {} is a tensor; detach to a NumPy "
                "array before freezing".format(field_index)
            )
        array = np.asarray(value)
        if array.ndim < 2:
            raise ValueError(
                "pair pending batch field {} must retain chunk and sequence "
                "axes".format(field_index)
            )
        if leading_shape is None:
            leading_shape = tuple(array.shape[:2])
            if leading_shape[0] <= 0 or leading_shape[1] <= 0:
                raise ValueError("pair pending batch cannot be empty")
        elif tuple(array.shape[:2]) != leading_shape:
            raise ValueError(
                "pair pending batch field {} has leading shape {}, expected "
                "{}".format(field_index, tuple(array.shape[:2]), leading_shape)
            )
        if np.issubdtype(array.dtype, np.number):
            if not np.isfinite(array).all():
                raise ValueError(
                    "pair pending batch field {} contains NaN or Inf".format(
                        field_index
                    )
                )
        owned = np.array(array, copy=True, order="C")
        owned.setflags(write=False)
        frozen.append(owned)
    return tuple(frozen)


def thaw_pair_pending_batch(frozen_batch):
    """Return mutable, owning copies suitable for checkpoint restore/use."""
    if len(frozen_batch) != PAIR_PENDING_BATCH_FIELD_COUNT:
        raise ValueError("invalid frozen pair pending batch field count")
    return tuple(np.array(value, copy=True, order="C") for value in frozen_batch)


def pair_pending_batch_state_dict(frozen_batch):
    return [np.array(value, copy=True, order="C") for value in frozen_batch]


def load_pair_pending_batch_state_dict(state):
    return freeze_pair_pending_batch(state)


def merge_generation_event_pair_scores(event_pair_scores):
    """Union event-local strict-pair targets for one replay generation.

    A single environment transition can be causal evidence for two distinct
    capture events (for example, two prey captured on the same environment
    step).  Event identities must remain distinct for provenance and atomic
    consumption, but the immutable replay generation is trained only once.
    Consequently, an identical shared transition is one member of the
    generation objective, not one member per event.

    Overlapping non-zero claims are accepted only when their stored raw score
    is byte-for-byte equal.  A disagreement still fails loudly because it
    means the supposedly immutable generation has conflicting target values.
    """
    scores = [np.asarray(value) for value in event_pair_scores]
    if not scores:
        raise ValueError("at least one event-local pair score is required")
    reference = scores[0]
    if not np.issubdtype(reference.dtype, np.number):
        raise ValueError("event-local pair scores must be numeric")
    if not np.isfinite(reference).all() or np.any(reference < 0.0):
        raise ValueError(
            "event-local pair scores must be finite and non-negative"
        )
    merged = np.zeros_like(reference)
    claimed = np.zeros(reference.shape, dtype=bool)
    for score in scores:
        if score.dtype != reference.dtype or score.shape != reference.shape:
            raise ValueError(
                "event-local pair score dtype or shape differs within one "
                "replay generation"
            )
        if not np.isfinite(score).all() or np.any(score < 0.0):
            raise ValueError(
                "event-local pair scores must be finite and non-negative"
            )
        event_claimed = score > 0.0
        overlap = claimed & event_claimed
        if np.any(overlap) and not np.array_equal(
                merged[overlap], score[overlap]):
            raise RuntimeError(
                "capture events disagree on a shared strict pair target "
                "transition"
            )
        merged[event_claimed] = score[event_claimed]
        claimed |= event_claimed
    return merged


def recompute_pair_stale_trust(
        old_log_probability,
        current_log_probability,
        trust_clip,
        trust_scale,
        minimum_weight):
    """Reproduce the factor stale-trust weighting used by the trainer."""
    old_log_probability = np.asarray(old_log_probability, dtype=np.float64)
    current_log_probability = np.asarray(
        current_log_probability, dtype=np.float64
    )
    if old_log_probability.shape != current_log_probability.shape:
        raise ValueError("old/current pair log-probability shapes differ")
    if not (
            np.isfinite(old_log_probability).all()
            and np.isfinite(current_log_probability).all()):
        raise ValueError("pair stale-trust log-probabilities must be finite")
    trust_clip = _require_finite(trust_clip, "trust_clip", 0.0)
    trust_scale = _require_finite(trust_scale, "trust_scale", 1e-12)
    minimum_weight = _require_finite(
        minimum_weight, "minimum_weight", 0.0
    )
    if minimum_weight > 1.0:
        raise ValueError("minimum_weight cannot exceed one")

    log_ratio = np.clip(
        current_log_probability - old_log_probability,
        -50.0,
        50.0,
    )
    ratio = np.exp(log_ratio)
    deviation = np.abs(ratio - 1.0)
    excess = np.maximum(deviation - trust_clip, 0.0)
    weight = np.exp(-excess / trust_scale)
    return np.clip(weight, minimum_weight, 1.0).astype(
        np.float32, copy=False
    )


def reconstruct_pending_pair_mass(
        pair_transition_scores,
        episode_success,
        coefficient,
        credit_cap,
        stale_trust_weights=None):
    """Rebuild raw centered credit and transaction-time effective masses."""
    scores = [np.asarray(value, dtype=np.float32)
              for value in pair_transition_scores]
    if not scores:
        raise ValueError("at least one pair evidence episode is required")
    reference_shape = scores[0].shape
    if len(reference_shape) != 2:
        raise ValueError(
            "each pair transition score must have shape [time, factor]"
        )
    if any(value.shape != reference_shape for value in scores):
        raise ValueError("pair evidence episode score shapes differ")
    if any(not np.isfinite(value).all() or np.any(value < 0.0)
           for value in scores):
        raise ValueError("pair transition scores must be finite and non-negative")

    score_tensor = np.stack(scores, axis=1)
    success = np.asarray(episode_success, dtype=bool)
    result = scale_optimizer_cohort_pair_credit(
        pair_transition_score=score_tensor,
        episode_success=success,
        coefficient=coefficient,
        credit_cap=credit_cap,
    )
    credit = np.asarray(result["credit"], dtype=np.float32)
    if stale_trust_weights is None:
        trust = np.ones_like(credit, dtype=np.float32)
    else:
        trust_values = [
            np.asarray(value, dtype=np.float32)
            for value in stale_trust_weights
        ]
        if len(trust_values) != len(scores):
            raise ValueError(
                "stale-trust episode count does not match pair evidence"
            )
        if any(value.shape != reference_shape for value in trust_values):
            raise ValueError("stale-trust shapes differ from pair scores")
        trust = np.stack(trust_values, axis=1)
        if (
                not np.isfinite(trust).all()
                or np.any(trust < 0.0)
                or np.any(trust > 1.0)):
            raise ValueError("stale-trust weights must lie in [0, 1]")

    effective_credit = credit * trust
    raw_positive_mass = float(np.maximum(credit, 0.0).sum())
    raw_negative_mass = float(np.maximum(-credit, 0.0).sum())
    effective_positive_mass = float(
        np.maximum(effective_credit, 0.0).sum()
    )
    effective_negative_mass = float(
        np.maximum(-effective_credit, 0.0).sum()
    )
    raw_centered_error = abs(float(credit.sum()))
    raw_scale = max(1.0, float(np.abs(credit).sum()))
    contract_valid = bool(
        result["class_complete"]
        and raw_positive_mass > 0.0
        and raw_negative_mass > 0.0
        and effective_positive_mass > 0.0
        and effective_negative_mass > 0.0
        and raw_centered_error <= 1e-5 * raw_scale
    )
    if not contract_valid:
        raise RuntimeError(
            "pending pair mass reconstruction did not produce a finite, "
            "class-complete, centered non-zero cohort"
        )
    return {
        "credit": credit,
        "effective_credit": effective_credit,
        "raw_positive_mass": raw_positive_mass,
        "raw_negative_mass": raw_negative_mass,
        "effective_positive_mass": effective_positive_mass,
        "effective_negative_mass": effective_negative_mass,
        "raw_centered_error": raw_centered_error,
        "effective_centered_sum": float(effective_credit.sum()),
        "common_scale": float(result["common_scale"]),
        "contract_valid": 1.0,
    }


class PairPendingEvidenceStore(object):
    """Deterministic pair-specific state with two-phase logical commit."""

    def __init__(self, enabled=False, max_adj_updates=0):
        self.enabled = bool(enabled)
        self.max_adj_updates = _require_int(
            max_adj_updates, "max_adj_updates", 0
        )
        if self.enabled and self.max_adj_updates <= 0:
            raise ValueError(
                "enabled pair pending requires a positive adjacency-update "
                "horizon"
            )
        self._entries = {}
        self._event_keys = {}
        self._committed_generation_keys = set()
        self._prepared_cohorts = {}
        self._next_cohort_id = 0

    def __len__(self):
        return len(self._entries)

    @property
    def entries(self):
        return dict(self._entries)

    def keys_for_generation(self, policy_id, replay_generation):
        generation_key = (str(policy_id), int(replay_generation))
        return tuple(
            key for key, entry in sorted(self._entries.items())
            if pair_pending_generation_key(entry["metadata"]) == generation_key
        )

    def refresh_available(
            self,
            key,
            current_adj_update,
            behavior_policy_version=None):
        """Extend replay validity without replacing the immutable payload."""
        entry = self._require_entry(key)
        if entry["state"] != PAIR_EVIDENCE_AVAILABLE_IN_REPLAY:
            return False
        current_adj_update = _require_int(
            current_adj_update, "current_adj_update", 0
        )
        prior = int(
            entry["metadata"]["original_last_valid_adj_update"]
        )
        if current_adj_update < prior:
            raise RuntimeError(
                "pair evidence replay validity cannot move backwards"
            )
        entry["metadata"]["original_last_valid_adj_update"] = (
            current_adj_update
        )
        if behavior_policy_version is not None:
            behavior_policy_version = _require_int(
                behavior_policy_version, "behavior_policy_version", 0
            )
            if behavior_policy_version < int(
                    entry["metadata"]["behavior_policy_version"]):
                raise RuntimeError(
                    "pair behavior policy version cannot move backwards"
                )
        return current_adj_update != prior

    def add_available(self, metadata, batch):
        if not self.enabled:
            raise RuntimeError("pair pending store is disabled")
        normalized = normalize_pair_pending_metadata(metadata)
        key = pair_pending_entry_key(normalized)
        generation_key = pair_pending_generation_key(normalized)
        frozen_batch = freeze_pair_pending_batch(batch)
        raw_pair_score = np.asarray(frozen_batch[17])
        if np.any(raw_pair_score < 0.0):
            raise RuntimeError(
                "immutable pending payload contains negative raw pair evidence"
            )
        target_bearing_count = int(
            np.sum(np.any(raw_pair_score > 0.0, axis=-1))
        )
        if (
                target_bearing_count
                != int(normalized["target_bearing_transition_count"])):
            raise RuntimeError(
                "immutable pending payload target-bearing count does not "
                "match provenance"
            )
        replay_provenance = np.asarray(frozen_batch[33])
        if (
                replay_provenance.shape[-1] != 3
                or not np.all(replay_provenance[..., 0] == 1.0)):
            raise RuntimeError(
                "immutable pending payload is not a strict pair-evidence "
                "episode population"
            )
        if generation_key in self._committed_generation_keys:
            raise RuntimeError(
                "committed pair evidence generation cannot be reused"
            )
        existing = self._entries.get(key)
        if existing is not None:
            if existing["metadata"] != normalized:
                raise RuntimeError(
                    "duplicate pair generation has conflicting provenance"
                )
            for old, new in zip(existing["batch"], frozen_batch):
                if (
                        old.dtype != new.dtype
                        or old.shape != new.shape
                        or not np.array_equal(old, new)):
                    raise RuntimeError(
                        "duplicate pair generation has conflicting payload"
                    )
            return key

        event_key = pair_pending_event_key(normalized)
        if event_key is not None:
            prior = self._event_keys.get(event_key)
            if prior is not None and prior != key:
                raise RuntimeError(
                    "one capture event appears in multiple pair generations"
                )
            self._event_keys[event_key] = key
        self._entries[key] = {
            "metadata": normalized,
            "batch": frozen_batch,
            "state": PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
            "prepared_cohort_id": None,
            "committed_adj_update": None,
            "expired_adj_update": None,
            "expiry_reason": "",
        }
        return key

    def mark_pending(self, key, current_adj_update):
        entry = self._require_entry(key)
        current_adj_update = _require_int(
            current_adj_update, "current_adj_update", 0
        )
        if entry["state"] == PAIR_EVIDENCE_COMMITTED:
            raise RuntimeError("committed pair evidence cannot become pending")
        if entry["state"] == PAIR_EVIDENCE_EXPIRED:
            raise RuntimeError("expired pair evidence cannot become pending")
        if entry["state"] == PAIR_EVIDENCE_PREPARED:
            raise RuntimeError("prepared pair evidence cannot be evicted")
        if current_adj_update < int(
                entry["metadata"]["original_last_valid_adj_update"]):
            raise ValueError(
                "pair evidence cannot pend before its replay validity ends"
            )
        entry["state"] = PAIR_EVIDENCE_PENDING
        return self.pending_age(key, current_adj_update)

    def pending_age(self, key, current_adj_update):
        entry = self._require_entry(key)
        current_adj_update = _require_int(
            current_adj_update, "current_adj_update", 0
        )
        return max(
            0,
            current_adj_update
            - int(entry["metadata"]["original_last_valid_adj_update"]),
        )

    def policy_age(self, key, current_policy_version):
        entry = self._require_entry(key)
        current_policy_version = _require_int(
            current_policy_version, "current_policy_version", 0
        )
        age = (
            current_policy_version
            - int(entry["metadata"]["behavior_policy_version"])
        )
        if age < 0:
            raise RuntimeError(
                "current pair policy version predates behavior policy"
            )
        return age

    def expire_out_of_horizon(self, current_adj_update):
        current_adj_update = _require_int(
            current_adj_update, "current_adj_update", 0
        )
        expired = []
        for key in sorted(self._entries):
            entry = self._entries[key]
            if entry["state"] != PAIR_EVIDENCE_PENDING:
                continue
            if self.pending_age(key, current_adj_update) > self.max_adj_updates:
                entry["state"] = PAIR_EVIDENCE_EXPIRED
                entry["expired_adj_update"] = current_adj_update
                entry["expiry_reason"] = "PENDING_HORIZON"
                expired.append(key)
        return tuple(expired)

    def prepare_class_complete(
            self,
            keys,
            current_adj_update,
            current_policy_version,
            expected_ppo_epochs):
        if not self.enabled:
            raise RuntimeError("pair pending store is disabled")
        current_adj_update = _require_int(
            current_adj_update, "current_adj_update", 0
        )
        current_policy_version = _require_int(
            current_policy_version, "current_policy_version", 0
        )
        expected_ppo_epochs = _require_int(
            expected_ppo_epochs, "expected_ppo_epochs", 1
        )
        normalized_keys = tuple(tuple(key) for key in keys)
        if len(normalized_keys) < 2 or len(set(normalized_keys)) != len(
                normalized_keys):
            raise ValueError(
                "class-complete preparation requires unique evidence entries"
            )
        signs = set()
        selected_generation_keys = set()
        for key in normalized_keys:
            entry = self._require_entry(key)
            generation_key = pair_pending_generation_key(entry["metadata"])
            if generation_key in self._committed_generation_keys:
                raise RuntimeError(
                    "committed pair evidence generation cannot prepare again"
                )
            selected_generation_keys.add(generation_key)
            if entry["state"] not in _LIVE_STATES:
                raise RuntimeError(
                    "only replay-available or pending evidence can prepare"
                )
            if (
                    entry["state"] == PAIR_EVIDENCE_PENDING
                    and self.pending_age(key, current_adj_update)
                    > self.max_adj_updates):
                raise RuntimeError("expired pending evidence cannot prepare")
            if not bool(
                    entry["metadata"]["event_provenance_available"]):
                raise RuntimeError(
                    "fully trainable pending evidence requires complete "
                    "episode/event/prey provenance"
                )
            self.policy_age(key, current_policy_version)
            signs.add(int(entry["metadata"]["sign"]))
        for generation_key in selected_generation_keys:
            live_generation_keys = set(
                key for key in self.keys_for_generation(*generation_key)
                if self._entries[key]["state"] in _LIVE_STATES
            )
            selected_for_generation = set(
                key for key in normalized_keys
                if pair_pending_generation_key(
                    self._entries[key]["metadata"]
                ) == generation_key
            )
            if live_generation_keys != selected_for_generation:
                raise RuntimeError(
                    "class-complete preparation must include every live "
                    "event from a selected replay generation"
                )
        if signs != {-1, 1}:
            raise RuntimeError(
                "pair evidence preparation is not class-complete"
            )

        cohort_id = int(self._next_cohort_id)
        self._next_cohort_id += 1
        for key in normalized_keys:
            entry = self._entries[key]
            entry["state"] = PAIR_EVIDENCE_PREPARED
            entry["prepared_cohort_id"] = cohort_id
        self._prepared_cohorts[cohort_id] = {
            "keys": normalized_keys,
            "prepared_adj_update": current_adj_update,
            "current_policy_version": current_policy_version,
            "expected_ppo_epochs": expected_ppo_epochs,
        }
        return cohort_id

    def release_prepared(self, cohort_id):
        cohort = self._require_prepared_cohort(cohort_id)
        for key in cohort["keys"]:
            entry = self._entries[key]
            if entry["state"] != PAIR_EVIDENCE_PREPARED:
                raise RuntimeError("prepared cohort entry state diverged")
            entry["state"] = (
                PAIR_EVIDENCE_PENDING
                if self.pending_age(key, cohort["prepared_adj_update"]) > 0
                else PAIR_EVIDENCE_AVAILABLE_IN_REPLAY
            )
            entry["prepared_cohort_id"] = None
        del self._prepared_cohorts[int(cohort_id)]

    def commit_prepared(
            self,
            cohort_id,
            committed_adj_update,
            completed_ppo_epochs,
            optimizer_transaction_count,
            positive_effective_mass,
            negative_effective_mass,
            target_bearing_transition_count,
            rollback=False,
            rejected=False):
        cohort = self._require_prepared_cohort(cohort_id)
        committed_adj_update = _require_int(
            committed_adj_update, "committed_adj_update", 0
        )
        completed_ppo_epochs = _require_int(
            completed_ppo_epochs, "completed_ppo_epochs", 0
        )
        optimizer_transaction_count = _require_int(
            optimizer_transaction_count, "optimizer_transaction_count", 0
        )
        target_bearing_transition_count = _require_int(
            target_bearing_transition_count,
            "target_bearing_transition_count",
            0,
        )
        positive_effective_mass = _require_finite(
            positive_effective_mass, "positive_effective_mass", 0.0
        )
        negative_effective_mass = _require_finite(
            negative_effective_mass, "negative_effective_mass", 0.0
        )
        expected_epochs = int(cohort["expected_ppo_epochs"])
        contract_valid = (
            completed_ppo_epochs == expected_epochs
            and optimizer_transaction_count == expected_epochs
            and target_bearing_transition_count > 0
            and positive_effective_mass > 0.0
            and negative_effective_mass > 0.0
            and not bool(rollback)
            and not bool(rejected)
        )
        if not contract_valid:
            raise RuntimeError(
                "pair pending logical transaction cannot commit: all PPO "
                "epochs, non-zero signed mass, target support, and successful "
                "optimizer transactions are required"
            )
        for key in cohort["keys"]:
            entry = self._entries[key]
            if entry["state"] != PAIR_EVIDENCE_PREPARED:
                raise RuntimeError("prepared pair entry state diverged")
            generation_key = pair_pending_generation_key(entry["metadata"])
            if generation_key in self._committed_generation_keys:
                raise RuntimeError(
                    "pair evidence generation was already committed"
                )
        for key in cohort["keys"]:
            entry = self._entries[key]
            entry["state"] = PAIR_EVIDENCE_COMMITTED
            entry["prepared_cohort_id"] = None
            entry["committed_adj_update"] = committed_adj_update
        self._committed_generation_keys.update(
            pair_pending_generation_key(self._entries[key]["metadata"])
            for key in cohort["keys"]
        )
        del self._prepared_cohorts[int(cohort_id)]

    def commit_available_from_standard_transaction(
            self,
            keys,
            committed_adj_update):
        """Mirror a confirmed standard pair transaction into single-use state.

        The ordinary support-v6 path remains responsible for its own optimizer
        semantics. The complete trained signed class must be supplied here,
        including members already mirrored as COMMITTED. Those members are
        validated idempotently; only newly AVAILABLE generations transition to
        COMMITTED so they cannot later enter the pending path.
        """
        committed_adj_update = _require_int(
            committed_adj_update, "committed_adj_update", 0
        )
        normalized_keys = tuple(sorted(set(tuple(key) for key in keys)))
        if not normalized_keys:
            return tuple()
        signs = set()
        selected_generation_keys = set()
        newly_available_keys = []
        generation_states = {}
        for key in normalized_keys:
            entry = self._require_entry(key)
            state = str(entry["state"])
            if state not in (
                    PAIR_EVIDENCE_AVAILABLE_IN_REPLAY,
                    PAIR_EVIDENCE_COMMITTED):
                raise RuntimeError(
                    "standard pair transaction can mirror only available or "
                    "already committed evidence"
                )
            generation_key = pair_pending_generation_key(entry["metadata"])
            generation_committed = (
                generation_key in self._committed_generation_keys
            )
            if (
                    state == PAIR_EVIDENCE_AVAILABLE_IN_REPLAY
                    and generation_committed):
                raise RuntimeError(
                    "available standard pair evidence has an already committed "
                    "generation identity"
                )
            if state == PAIR_EVIDENCE_COMMITTED:
                if (
                        not generation_committed
                        or entry["committed_adj_update"] is None):
                    raise RuntimeError(
                        "committed standard pair evidence is missing its "
                        "generation-level commit record"
                    )
            else:
                newly_available_keys.append(key)
            selected_generation_keys.add(generation_key)
            generation_states.setdefault(generation_key, set()).add(state)
            signs.add(int(entry["metadata"]["sign"]))
        for generation_key in selected_generation_keys:
            stored_generation_keys = set(
                self.keys_for_generation(*generation_key)
            )
            selected_for_generation = set(
                key for key in normalized_keys
                if pair_pending_generation_key(
                    self._entries[key]["metadata"]
                ) == generation_key
            )
            if stored_generation_keys != selected_for_generation:
                raise RuntimeError(
                    "standard pair transaction must mirror every stored "
                    "event from a selected replay generation"
                )
            if len(generation_states[generation_key]) != 1:
                raise RuntimeError(
                    "standard pair transaction found mixed event states "
                    "within one replay generation"
                )
        if signs != {-1, 1}:
            raise RuntimeError(
                "standard pair transaction commit is not class-complete"
            )
        for key in newly_available_keys:
            entry = self._entries[key]
            entry["state"] = PAIR_EVIDENCE_COMMITTED
            entry["committed_adj_update"] = committed_adj_update
        self._committed_generation_keys.update(selected_generation_keys)
        return tuple(newly_available_keys)

    def state_dict(self):
        entries = []
        for key in sorted(self._entries):
            entry = self._entries[key]
            entries.append({
                "key": tuple(key),
                "metadata": copy.deepcopy(entry["metadata"]),
                "batch": pair_pending_batch_state_dict(entry["batch"]),
                "state": str(entry["state"]),
                "prepared_cohort_id": entry["prepared_cohort_id"],
                "committed_adj_update": entry["committed_adj_update"],
                "expired_adj_update": entry["expired_adj_update"],
                "expiry_reason": str(entry["expiry_reason"]),
            })
        return {
            "version": PAIR_PENDING_CHECKPOINT_VERSION,
            "enabled": self.enabled,
            "max_adj_updates": self.max_adj_updates,
            "entries": entries,
            "committed_generation_keys": [
                tuple(key) for key in sorted(
                    self._committed_generation_keys
                )
            ],
            "prepared_cohorts": copy.deepcopy(self._prepared_cohorts),
            "next_cohort_id": int(self._next_cohort_id),
        }

    def load_state_dict(self, state, require_fresh_when_enabled=False):
        if not isinstance(state, dict):
            raise ValueError("pair pending checkpoint must be a dictionary")
        version = int(state.get("version", -1))
        if version != PAIR_PENDING_CHECKPOINT_VERSION:
            raise RuntimeError(
                "unsupported pair pending checkpoint version {}".format(
                    version
                )
            )
        checkpoint_enabled = bool(state.get("enabled", False))
        checkpoint_horizon = int(state.get("max_adj_updates", -1))
        if (
                checkpoint_enabled != self.enabled
                or checkpoint_horizon != self.max_adj_updates):
            raise RuntimeError(
                "pair pending checkpoint configuration does not match runtime"
            )
        if require_fresh_when_enabled and self.enabled:
            raise RuntimeError(
                "bounded pair pending experiment requires a fresh run"
            )

        restored_entries = {}
        restored_event_keys = {}
        for serialized in state.get("entries", ()):
            metadata = normalize_pair_pending_metadata(
                serialized["metadata"]
            )
            key = pair_pending_entry_key(metadata)
            if tuple(serialized["key"]) != key or key in restored_entries:
                raise RuntimeError(
                    "pair pending checkpoint generation key is invalid"
                )
            entry_state = str(serialized["state"])
            if entry_state not in _KNOWN_STATES:
                raise RuntimeError(
                    "pair pending checkpoint contains an unknown state"
                )
            event_key = pair_pending_event_key(metadata)
            if (
                    event_key is not None
                    and event_key in restored_event_keys):
                raise RuntimeError(
                    "pair pending checkpoint duplicates a capture event"
                )
            if event_key is not None:
                restored_event_keys[event_key] = key
            restored_entries[key] = {
                "metadata": metadata,
                "batch": load_pair_pending_batch_state_dict(
                    serialized["batch"]
                ),
                "state": entry_state,
                "prepared_cohort_id": serialized["prepared_cohort_id"],
                "committed_adj_update": serialized[
                    "committed_adj_update"
                ],
                "expired_adj_update": serialized["expired_adj_update"],
                "expiry_reason": str(serialized["expiry_reason"]),
            }

        committed = set(
            tuple(key)
            for key in state.get("committed_generation_keys", ())
        )
        restored_generation_states = {}
        for entry in restored_entries.values():
            generation_key = pair_pending_generation_key(entry["metadata"])
            restored_generation_states.setdefault(
                generation_key, []
            ).append(entry["state"])
        if any(
                generation_key not in restored_generation_states
                or not restored_generation_states[generation_key]
                or any(
                    entry_state != PAIR_EVIDENCE_COMMITTED
                    for entry_state in
                    restored_generation_states[generation_key]
                )
                for generation_key in committed):
            raise RuntimeError(
                "pair pending checkpoint committed-key contract failed"
            )
        prepared = copy.deepcopy(state.get("prepared_cohorts", {}))
        prepared = {int(key): value for key, value in prepared.items()}
        for cohort_id, cohort in prepared.items():
            for key in tuple(cohort["keys"]):
                key = tuple(key)
                if (
                        key not in restored_entries
                        or restored_entries[key]["state"]
                        != PAIR_EVIDENCE_PREPARED
                        or restored_entries[key]["prepared_cohort_id"]
                        != cohort_id):
                    raise RuntimeError(
                        "pair pending checkpoint prepared cohort contract "
                        "failed"
                    )
            cohort["keys"] = tuple(tuple(key) for key in cohort["keys"])

        self._entries = restored_entries
        self._event_keys = restored_event_keys
        self._committed_generation_keys = committed
        self._prepared_cohorts = prepared
        self._next_cohort_id = int(state.get("next_cohort_id", 0))

    def diagnostics(self, current_adj_update, current_policy_version=None):
        current_adj_update = _require_int(
            current_adj_update, "current_adj_update", 0
        )
        if current_policy_version is not None:
            current_policy_version = _require_int(
                current_policy_version, "current_policy_version", 0
            )
        counts = {state: 0 for state in _KNOWN_STATES}
        pending_ages = []
        pending_policy_ages = []
        pending_positive = 0
        pending_negative = 0
        available_positive = 0
        available_negative = 0
        for key, entry in self._entries.items():
            counts[entry["state"]] += 1
            sign = int(entry["metadata"]["sign"])
            if entry["state"] == PAIR_EVIDENCE_AVAILABLE_IN_REPLAY:
                if sign > 0:
                    available_positive += 1
                else:
                    available_negative += 1
            if entry["state"] == PAIR_EVIDENCE_PENDING:
                pending_ages.append(self.pending_age(key, current_adj_update))
                if current_policy_version is not None:
                    pending_policy_ages.append(
                        self.policy_age(key, current_policy_version)
                    )
                if sign > 0:
                    pending_positive += 1
                else:
                    pending_negative += 1
        current_pending_overlap = int(
            (pending_positive > 0 and available_negative > 0)
            or (pending_negative > 0 and available_positive > 0)
            or (pending_positive > 0 and pending_negative > 0)
        )
        return {
            "diagnostic_version": PAIR_PENDING_DIAGNOSTIC_VERSION,
            "available_in_replay_count": counts[
                PAIR_EVIDENCE_AVAILABLE_IN_REPLAY
            ],
            "pending_positive_count": pending_positive,
            "pending_negative_count": pending_negative,
            "available_positive_count": available_positive,
            "available_negative_count": available_negative,
            "current_replay_pending_overlap_count": (
                current_pending_overlap
            ),
            "prepared_count": counts[PAIR_EVIDENCE_PREPARED],
            "committed_count": counts[PAIR_EVIDENCE_COMMITTED],
            "expired_count": counts[PAIR_EVIDENCE_EXPIRED],
            "pending_age_mean": (
                float(np.mean(pending_ages)) if pending_ages else 0.0
            ),
            "pending_age_max": (
                int(max(pending_ages)) if pending_ages else 0
            ),
            "policy_age_mean": (
                float(np.mean(pending_policy_ages))
                if pending_policy_ages else 0.0
            ),
            "policy_age_max": (
                int(max(pending_policy_ages))
                if pending_policy_ages else 0
            ),
            "reused_after_commit_count": 0,
            "zero_target_commit_count": 0,
            "payload_contract_valid": 1.0,
            "checkpoint_contract_valid": 1.0,
        }

    def _require_entry(self, key):
        key = tuple(key)
        if key not in self._entries:
            raise KeyError("unknown pair pending generation {}".format(key))
        return self._entries[key]

    def _require_prepared_cohort(self, cohort_id):
        cohort_id = int(cohort_id)
        if cohort_id not in self._prepared_cohorts:
            raise KeyError(
                "unknown prepared pair cohort {}".format(cohort_id)
            )
        return self._prepared_cohorts[cohort_id]
