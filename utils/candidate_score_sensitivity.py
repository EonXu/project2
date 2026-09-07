"""Read-only score-to-rank and active-boundary counterfactuals.

This module deliberately has no torch, optimizer, replay, or RNG dependency.
It is an offline diagnostic helper and is not imported by the training path.
"""

import math


def _canonical_rank(scores, valid, target_index):
    """Return the deterministic 1-based rank used by SDDFG diagnostics."""
    if len(scores) != len(valid):
        raise ValueError("candidate score and valid populations must match")
    if target_index < 0 or target_index >= len(scores):
        raise ValueError("candidate target index is outside the population")
    if not bool(valid[target_index]):
        raise ValueError("candidate target is not in the valid population")
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("candidate population contains a non-finite score")
    target_score = float(scores[target_index])
    rank = 1
    for index, (score, is_valid) in enumerate(zip(scores, valid)):
        if not bool(is_valid) or index == target_index:
            continue
        score = float(score)
        if score > target_score or (
                score == target_score and index < target_index):
            rank += 1
    return rank


def candidate_score_to_rank_sensitivity(
        scores,
        valid,
        target_index,
        target_sign,
        observed_signed_improvement=0.0):
    """Compute the minimum virtual target-only movement for rank/boundary progress.

    The virtual perturbation changes only the target margin. It does not claim
    that a future optimizer update will leave the remaining population fixed.
    That makes the result a local reachability counterfactual, not a training
    prediction.
    """
    scores = [float(value) for value in scores]
    valid = [bool(value) for value in valid]
    target_index = int(target_index)
    target_sign = 1.0 if float(target_sign) > 0.0 else -1.0
    observed_signed_improvement = float(observed_signed_improvement)
    if not math.isfinite(observed_signed_improvement):
        raise ValueError("observed signed improvement is non-finite")

    pre_rank = _canonical_rank(scores, valid, target_index)
    target_score = scores[target_index]
    scale = max(
        1.0,
        max(abs(score) for score, is_valid in zip(scores, valid) if is_valid),
    )
    # A conservative float32-scale strict inequality. PyTorch 1.3.1 lacks
    # torch.nextafter, and this offline helper must not depend on newer APIs.
    strict_epsilon = 16.0 * 1.1920928955078125e-7 * scale

    rank_step_required = None
    rank_competitor_index = -1
    rank_after_virtual_step = pre_rank
    for index, (score, is_valid) in enumerate(zip(scores, valid)):
        if not is_valid or index == target_index:
            continue
        if target_sign > 0.0:
            currently_blocks = (
                score > target_score
                or (score == target_score and index < target_index)
            )
            threshold = max(score - target_score, 0.0) + strict_epsilon
        else:
            currently_blocks = (
                target_score > score
                or (target_score == score and target_index < index)
            )
            threshold = max(target_score - score, 0.0) + strict_epsilon
        if not currently_blocks:
            continue
        virtual_scores = list(scores)
        virtual_scores[target_index] += target_sign * threshold
        virtual_rank = _canonical_rank(
            virtual_scores,
            valid,
            target_index,
        )
        rank_improved = (
            virtual_rank < pre_rank
            if target_sign > 0.0
            else virtual_rank > pre_rank
        )
        if not rank_improved:
            continue
        if rank_step_required is None or threshold < rank_step_required:
            rank_step_required = threshold
            rank_competitor_index = index
            rank_after_virtual_step = virtual_rank

    pre_signed_boundary = target_sign * target_score
    boundary_step_required = max(
        -pre_signed_boundary + strict_epsilon,
        0.0,
    )
    virtual_boundary_score = (
        target_score + target_sign * boundary_step_required
    )
    virtual_boundary_crossed = (
        target_sign * virtual_boundary_score > 0.0
    )

    def _observed_step_equivalent(required):
        if required is None:
            return None
        if required <= 0.0:
            return 0
        if observed_signed_improvement <= 0.0:
            return None
        return int(math.ceil(required / observed_signed_improvement))

    return {
        "pre_rank": int(pre_rank),
        "valid_population_count": int(sum(valid)),
        "target_score": float(target_score),
        "target_sign": float(target_sign),
        "strict_epsilon": float(strict_epsilon),
        "rank_competitor_index": int(rank_competitor_index),
        "rank_step_required": (
            None if rank_step_required is None
            else float(rank_step_required)
        ),
        "rank_after_virtual_step": int(rank_after_virtual_step),
        "rank_virtual_step_improves": bool(
            rank_after_virtual_step != pre_rank
        ),
        "boundary_step_required": float(boundary_step_required),
        "virtual_boundary_crossed": bool(virtual_boundary_crossed),
        "observed_signed_improvement": float(observed_signed_improvement),
        "observed_updates_to_rank_step": _observed_step_equivalent(
            rank_step_required
        ),
        "observed_updates_to_boundary": _observed_step_equivalent(
            boundary_step_required
        ),
    }
