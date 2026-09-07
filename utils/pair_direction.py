PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION = 7


# This is a production wire contract shared by the exact-search producer and
# the runner CSV consumer.  Keep the values finite and explicit: accepting an
# arbitrary prefix here would hide producer/logger drift instead of detecting
# it at the point where a new search family is introduced.
PAIR_DIRECTION_CANDIDATE_KINDS = (
    "adam_projection",
    "deficit_progress_seed",
    "boundary_limiter_blend",
    "boundary_limiter_tangent",
    "boundary_limiter_bundle_blend",
    "boundary_limiter_bundle_tangent",
)


# ``progress_seed_zero_budget_excluded_*`` describes the deficit-progress
# seed selection which predates the failed-boundary limiter family.  A limiter
# direction has a different contract: a member with no *extra progress budget*
# may still be the exact-forward boundary limiter and is therefore a valid
# tangent seed.  Keep that distinction explicit instead of weakening the
# deficit seed's exclusion rule globally.
PAIR_DIRECTION_ZERO_BUDGET_LIMITER_SEED_KINDS = (
    "boundary_limiter_blend",
    "boundary_limiter_tangent",
    "boundary_limiter_bundle_blend",
    "boundary_limiter_bundle_tangent",
)


def validate_pair_direction_candidate_kind(direction_kind):
    direction_kind = str(direction_kind)
    if direction_kind not in PAIR_DIRECTION_CANDIDATE_KINDS:
        raise RuntimeError(
            "unknown strict-pair direction kind: {}".format(direction_kind)
        )
    return direction_kind


def pair_direction_allows_zero_budget_limiter_seed(direction_kind):
    direction_kind = validate_pair_direction_candidate_kind(direction_kind)
    return direction_kind in PAIR_DIRECTION_ZERO_BUDGET_LIMITER_SEED_KINDS


def validate_pair_direction_candidate_seed_contract(
        direction_kind,
        seed_member_ordinals,
        zero_budget_excluded_ordinals):
    direction_kind = validate_pair_direction_candidate_kind(direction_kind)
    seed_members = set(int(value) for value in seed_member_ordinals)
    excluded_members = set(
        int(value) for value in zero_budget_excluded_ordinals
    )
    if (
            seed_members.intersection(excluded_members)
            and not pair_direction_allows_zero_budget_limiter_seed(
                direction_kind
            )):
        raise RuntimeError(
            "direction-candidate seed includes a zero-budget member"
        )
    return direction_kind
