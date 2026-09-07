import os
import csv
import copy
import random
import numpy as np
import wandb
import torch
from tensorboardX import SummaryWriter
import pandas as pd
from utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from utils.adj_buffer import (
    AdjBuffer,
    CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION,
    CANDIDATE_EVIDENCE_PROVENANCE_DRAFT_FIELDS,
)
from utils.util import  get_dim_from_space
from utils.util import DecayThenFlatSchedule
from utils.normalization import RewardScaling
from utils.adj_training_control import (
    aggregate_adj_control_populations,
    advance_recent_episode_window,
    select_adj_control_population,
    should_stop_adj_ppo,
    validate_adj_control_application,
)
from utils.pair_pending import (
    PairOptimizerRecoverableNoOpError,
    PairPendingZeroGradientError,
)
from utils.pair_direction import (
    PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION,
    validate_pair_direction_candidate_seed_contract,
    validate_pair_direction_candidate_kind,
)
from utils.wolfpack_reward import (
    WOLFPACK_DISTANCE_SHAPING_SCALE,
    WOLFPACK_LEGACY_DISTANCE_MODE,
    WOLFPACK_MULTI_PREY_COVERAGE_MODE,
    WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION,
)


_SDDFG_CHECKPOINT_KINDS = ("periodic", "best", "terminal")
_PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION = 45
PAIR_EXACT_SCORE_RECORDING_CONTRACT_VERSION = 2
_ADJ_TRANSACTION_CSV_BASENAME = "progress_train_adj_transaction.csv"
_PAIR_SELECTION_BOUNDARY_DIAGNOSTIC_VERSION = 8
_PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION = (
    PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION
)
_PAIR_DIRECTION_CANDIDATE_CSV_BASENAME = (
    "progress_train_pair_direction_candidate_transaction.csv"
)
_STRICT_PAIR_EXACT_FAILURE_CSV_BASENAME = (
    "progress_train_strict_pair_exact_failure.csv"
)
_STRICT_PAIR_EXACT_FAILURE_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "diagnostic_version",
    "optimizer_kind",
    "target_count",
    "target_candidate_indices",
    "target_canonical_identities",
    "target_signs",
    "target_transition_indices",
    "target_factor_indices",
    "target_selected_episode_ordinals",
    "target_episode_transition_steps",
    "target_pair_evidence_flags",
    "candidate_count",
    "diagnostic_probe_valid_count",
    "bounded_search_exhaustive",
    "failure_classification",
    "origin_preservation_valid",
    "origin_preservation_tolerance",
    "candidate_ordinal",
    "direction_kind",
    "progress_floor_fraction",
    "direction_norm",
    "cosine_vs_full",
    "evaluation_ordinal",
    "evaluation_kind",
    "scale",
    "parameter_displacement_norm",
    "predicted_exact_score_min_change",
    "predicted_boundary_min_change",
    "valid",
    "boundary_valid",
    "boundary_min_signed_change",
    "exact_score_valid",
    "exact_score_min_signed_change",
    "candidate_valid",
    "candidate_loss_change",
    "lifecycle_valid",
    "lifecycle_violation_count",
    "lifecycle_min_signed_gap",
    "limiting_constraint_type",
    "limiting_target_ordinal",
    "boundary_failed_target_ordinals",
    "progress_target_present",
    "progress_worst_actual",
    "progress_worst_required",
    "progress_min_completion",
    "progress_mean_completion",
    "competitor_candidate_indices",
    "target_ranks",
    "target_active",
)
_PAIR_DIRECTION_CANDIDATE_TRACE_FIELDS = (
    "diagnostic_version",
    "candidate_ordinal",
    "direction_kind",
    "progress_floor_fraction",
    "direction_norm",
    "cosine_vs_full",
    "progress_seed_member_count",
    "progress_seed_member_ordinals",
    "progress_seed_zero_budget_excluded_count",
    "progress_seed_zero_budget_excluded_ordinals",
    "progress_seed_raw_norm",
    "adam_reference_norm",
    "progress_component_norm",
    "active_constraint_count",
    "active_constraint_ordinals",
    "valid",
    "halving_count",
    "expansion_count",
    "refinement_count",
    "safe_lower_scale",
    "safe_frontier_scale",
    "unsafe_upper_scale",
    "unsafe_upper_present",
    "scale_limit",
    "progress_min_completion",
    "progress_mean_completion",
    "progress_target_present",
    "progress_worst_actual",
    "progress_worst_required",
    "limiting_constraint_code",
    "limiting_target_ordinal",
    "selected",
)
_PAIR_DIRECTION_CANDIDATE_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "optimizer_kind",
) + _PAIR_DIRECTION_CANDIDATE_TRACE_FIELDS
_PAIR_SELECTION_BOUNDARY_CSV_BASENAME = (
    "progress_train_pair_selection_boundary_transaction.csv"
)
_PAIR_SELECTION_BOUNDARY_TRACE_FIELDS = (
    "diagnostic_version",
    "target_row_sequence_within_transaction",
    "transition_index_in_partition",
    "factor_index",
    "target_candidate_index",
    "target_canonical_identity",
    "target_sign",
    "target_weight",
    "pre_competitor_candidate_index",
    "pre_competitor_canonical_identity",
    "post_competitor_candidate_index",
    "post_competitor_canonical_identity",
    "pre_margin",
    "post_margin",
    "pre_target_logp",
    "post_target_logp",
    "pre_competitor_logp",
    "post_competitor_logp",
    "pre_signed_margin",
    "post_signed_margin",
    "signed_margin_change",
    "margin_direction_correct",
    "margin_direction_reverse",
    "margin_direction_zero",
    "pre_rank",
    "post_rank",
    "signed_rank_improvement",
    "pre_boundary_deficit",
    "post_boundary_deficit",
    "boundary_deficit_reduction",
    "boundary_deficit_reduction_fraction",
    "boundary_deficit_reduction_fraction_valid",
    "linearized_required_margin_improvement",
    "linearized_crossing_affordable",
    "linearized_original_boundary_budget",
    "linearized_strict_floor_budget",
    "linearized_allocated_boundary_budget",
    "linearized_zero_deficit_reclaimed_budget",
    "linearized_deficit_target_count",
    "linearized_affordable_crossing_count",
    "linearized_budget_conservation_valid",
    "linearized_budget_tolerance",
    "linearized_target_strict_floor",
    "linearized_target_waterfill_allocation",
    "identity_group_ordinal",
    "identity_group_exposure_count",
    "identity_group_strict_budget",
    "identity_group_allocated_budget",
    "identity_group_extra_budget",
    "identity_group_progress_member_ordinal",
    "identity_group_progress_member",
    "identity_group_progress_required",
    "identity_group_actual_signed_margin_change",
    "identity_group_actual_completion_ratio",
    "identity_group_worst_member_signed_margin_change",
    "selected_progress_floor_fraction",
    "progress_min_completion",
    "progress_mean_completion",
    "limiting_constraint_code",
    "limiting_target_ordinal",
    "limiting_target",
    "identity_group_count",
    "multi_exposure_identity_group_count",
    "boundary_crossing",
    "pre_active_at_replay_boundary",
    "post_active_at_replay_boundary",
    "positive_promotion",
    "negative_eviction",
    "valid",
)
_PAIR_SELECTION_BOUNDARY_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "optimizer_kind",
) + _PAIR_SELECTION_BOUNDARY_TRACE_FIELDS
_PAIR_SELECTION_BOUNDARY_RETENTION_DIAGNOSTIC_VERSION = 3
_PAIR_SELECTION_BOUNDARY_RETENTION_CSV_BASENAME = (
    "progress_train_pair_selection_boundary_retention_transaction.csv"
)
_PAIR_SELECTION_BOUNDARY_RETENTION_TRACE_FIELDS = (
    "diagnostic_version",
    "observation_id",
    "source_policy_id",
    "source_transaction_sequence_index",
    "source_adjacency_update_round",
    "ordinary_update_age",
    "source_episode_ordinal",
    "source_episode_step",
    "selection_context_sha256",
    "factor_index",
    "target_candidate_index",
    "target_canonical_identity",
    "target_sign",
    "commit_competitor_candidate_index",
    "commit_competitor_canonical_identity",
    "current_competitor_candidate_index",
    "current_competitor_canonical_identity",
    "pre_signed_margin",
    "commit_signed_margin",
    "current_signed_margin",
    "retained_progress_fraction",
    "commit_rank",
    "current_rank",
    "commit_active",
    "current_active",
    "competitor_changed",
    "context_valid",
    "margin_nonregression",
    "rank_retained",
    "active_retained",
    "protection_stopped",
    "protection_stop_reason",
    "protection_stop_clock",
)
_PAIR_SELECTION_BOUNDARY_RETENTION_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "optimizer_kind",
) + _PAIR_SELECTION_BOUNDARY_RETENTION_TRACE_FIELDS
_PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_DIAGNOSTIC_VERSION = 1
_PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_CSV_BASENAME = (
    "progress_train_pair_selection_boundary_retention_component.csv"
)
_PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_TRACE_FIELDS = (
    "diagnostic_version",
    "observation_id",
    "component",
    "source_kind",
    "component_delta_norm",
    "commit_floor",
    "baseline_competitor_candidate_index",
    "baseline_competitor_canonical_identity",
    "component_competitor_candidate_index",
    "component_competitor_canonical_identity",
    "baseline_signed_margin",
    "component_signed_margin",
    "signed_margin_delta",
    "baseline_rank",
    "component_rank",
    "baseline_active",
    "component_active",
    "competitor_changed",
    "baseline_context_valid",
    "component_context_valid",
    "context_valid",
    "floor_retained",
    "rank_retained",
    "active_retained",
    "joint_objectives_valid",
)
_PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "optimizer_kind",
) + _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_TRACE_FIELDS
_PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_DIAGNOSTIC_VERSION = 1
_PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_CSV_BASENAME = (
    "progress_train_pair_selection_boundary_policy_response.csv"
)
_PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_TRACE_FIELDS = (
    "diagnostic_version",
    "target_row_sequence_within_transaction",
    "transition_index_in_partition",
    "factor_index",
    "target_candidate_index",
    "target_canonical_identity",
    "target_sign",
    "crossing_kind",
    "pre_active_candidate_index",
    "pre_active_canonical_identity",
    "post_active_candidate_index",
    "post_active_canonical_identity",
    "policy_context_sha256",
    "policy_state_sha256",
    "available_action_count",
    "pre_factor_order",
    "post_factor_order",
    "structure_input_diff_norm",
    "observation_input_diff_norm",
    "rnn_state_input_diff_norm",
    "factor_q_comparable",
    "pre_factor_q_norm",
    "post_factor_q_norm",
    "factor_q_diff_norm",
    "pre_best_value",
    "post_best_value",
    "best_value_delta",
    "pre_selected_factor_value",
    "post_selected_factor_value",
    "selected_factor_value_delta",
    "greedy_action_changed_count",
    "greedy_action_changed_fraction",
    "policy_response_nonzero",
    "rng_neutral",
    "state_neutral",
    "valid",
)
_PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "optimizer_kind",
) + _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_TRACE_FIELDS
_CANDIDATE_IDENTITY_TRANSACTION_DIAGNOSTIC_VERSION = 2
_CANDIDATE_IDENTITY_TRANSACTION_CSV_BASENAME = (
    "progress_train_candidate_identity_transaction.csv"
)
_CANDIDATE_IDENTITY_TRANSACTION_TRACE_FIELDS = (
    "diagnostic_version",
    "target_row_sequence_within_transaction",
    "selected_episode_ordinal",
    "episode_transition_step",
    "transition_index_in_partition",
    "candidate_index",
    "canonical_identity",
    "factor_order",
    "target_sign",
    "target_weight",
    "pair_evidence_transition",
    "successful_candidate_capture_overlap",
    "behavior_margin",
    "behavior_rank",
    "behavior_valid",
    "behavior_policy_version",
    "candidate_policy_age",
    "pre_margin",
    "post_margin",
    "signed_margin_change",
    "margin_direction_correct",
    "pre_rank",
    "post_rank",
    "signed_rank_improvement",
    "rank_improved",
    "pre_signed_boundary",
    "post_signed_boundary",
    "pre_boundary_deficit",
    "post_boundary_deficit",
    "boundary_crossing",
    "pre_valid_population_count",
    "post_valid_population_count",
    "pre_strictly_better_count",
    "post_strictly_better_count",
    "pre_tie_precedes_count",
    "post_tie_precedes_count",
    "pre_next_better_candidate_index",
    "post_next_better_candidate_index",
    "pre_next_better_margin_gap",
    "post_next_better_margin_gap",
    "same_population_rank_reconstruction_valid",
    "lifecycle_behavioral_progress",
)
_CANDIDATE_IDENTITY_TRANSACTION_EVENT_FIELDS = (
    "environment_episode_id",
    "capture_event_id",
    "capture_prey_id",
    "capture_step",
    "candidate_participant_slots",
    "candidate_static_dynamic_class",
    "candidate_event_group_size",
    "capture_event_ids",
    "capture_prey_ids",
    "candidate_event_target_masses",
    "candidate_participant_slot_groups",
    "candidate_static_dynamic_classes",
)
_CANDIDATE_IDENTITY_TRANSACTION_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "optimizer_step_before",
    "optimizer_step_after",
    "episode_generation",
    "replay_slot_index",
    "episode_recency_age",
    "base_selected",
    "support_selected",
    "outcome_support_used",
    "candidate_scalar_loss",
    "candidate_gradient_norm",
    "candidate_projected_gradient_dot",
    "candidate_clipped_gradient_dot",
    "candidate_actual_update_descent_dot_before",
    "candidate_actual_update_descent_dot_after",
    "candidate_actual_update_corrected",
    "candidate_loss_optimizer_change",
    "candidate_gradient_projection_intervened",
    "lifecycle_gradient_projection_intervened",
    "lifecycle_backtrack_count",
    "lifecycle_reject_occurred",
    "rollback_occurred",
) + (
    _CANDIDATE_IDENTITY_TRANSACTION_TRACE_FIELDS
    + _CANDIDATE_IDENTITY_TRANSACTION_EVENT_FIELDS
)
_CANDIDATE_EVIDENCE_PROVENANCE_CSV_BASENAME = (
    "progress_train_candidate_evidence_provenance.csv"
)
_CANDIDATE_EVIDENCE_CONSUMPTION_FIELDS = (
    "diagnostic_version",
    "selected_episode_ordinal",
    "episode_transition_step",
    "candidate_index",
    "target_sign",
    "target_weight",
    "behavior_policy_version",
)
_CANDIDATE_EVIDENCE_PROVENANCE_CSV_FIELDS = (
    "run_id",
    "env_step",
    "policy_id",
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
    "candidate_policy_age",
    "first_consumed_update",
    "consumed_generation_once",
    "duplicate_generation",
    "duplicate_event",
    "quality_reconstruction_error",
    "provenance_complete",
    "identity_contract_valid",
    "quality_contract_valid",
)
_ADJ_TRANSACTION_OBJECTIVE_NAMES = (
    "graph",
    "base_factor",
    "capture_outcome",
    "pair",
    "candidate",
    "entropy",
)
_ADJ_TRANSACTION_OBJECTIVE_METRICS = (
    "active",
    "scalar_loss",
    "grad_norm",
    "pair_grad_dot",
    "pair_grad_cosine",
    "pair_descent_component",
)
_ADJ_TRANSACTION_V3_GLOBAL_FIELDS = (
    "objective_scalar_reconstruction_error",
    "objective_scalar_reconstruction_valid",
    "all_objectives_independent_grad_sum_norm",
    "pair_independent_grad_sum_dot",
    "pair_independent_grad_sum_cosine",
    "raw_combined_grad_norm_from_backward",
    "independent_sum_vs_raw_combined_delta_norm",
    "independent_sum_vs_raw_combined_relative_error",
    "independent_sum_reconstruction_valid",
    "pre_projection_combined_grad_norm",
    "post_projection_combined_grad_norm",
    "pair_pre_projection_dot",
    "pair_pre_projection_cosine",
    "pair_post_projection_dot",
    "pair_post_projection_cosine",
    "projection_delta_norm",
    "pair_projection_delta_dot",
    "pair_projection_delta_cosine",
    "gradient_projection_intervened",
)
_ADJ_TRANSACTION_V3_CSV_FIELDS = tuple(
    "objective_{}_{}".format(objective_name, metric_name)
    for objective_name in _ADJ_TRANSACTION_OBJECTIVE_NAMES
    for metric_name in _ADJ_TRANSACTION_OBJECTIVE_METRICS
) + _ADJ_TRANSACTION_V3_GLOBAL_FIELDS
_ADJ_TRANSACTION_CSV_FIELDS = (
    "run_id",
    "env_step",
    "adjacency_update_round",
    "ppo_epoch_index",
    "policy_id",
    "partition_index",
    "transaction_sequence_index",
    "diagnostic_version",
    "support_version",
    "credit_observation_consumed",
    "pair_credit_raw_observation_count",
    "triplet_credit_raw_observation_count",
    "pair_credit_state_update_count",
    "triplet_credit_state_update_count",
    "optimizer_kind",
    "class_complete",
    "nonzero_pair_transaction",
    "selected_episode_count",
    "selected_chunk_count",
    "transaction_chunk_count",
    "target_bearing_transition_count",
    "positive_pair_mass",
    "negative_pair_mass",
    "centered_mass_error",
    "pair_scalar_loss",
    "base_factor_scalar_loss",
    "total_adjacency_loss",
    "optimizer_step_before",
    "optimizer_step_before_min",
    "optimizer_step_before_max",
    "optimizer_step_after",
    "optimizer_step_after_min",
    "optimizer_step_after_max",
    "pair_grad_norm",
    "pair_grad_finite",
    "pair_grad_zero",
    "base_factor_grad_norm",
    "pair_base_grad_dot",
    "pair_base_grad_cosine",
    "combined_grad_norm_preclip",
    "pair_combined_grad_dot_preclip",
    "pair_combined_grad_cosine_preclip",
    "pair_combined_descent_component_preclip",
    "combined_grad_norm_postclip",
    "pair_combined_grad_dot_postclip",
    "pair_combined_grad_cosine_postclip",
    "pair_combined_descent_component_postclip",
    "gradient_clip_reported_preclip_norm",
    "gradient_clip_applied",
    "gradient_clip_scale",
    "adam_exp_avg_norm",
    "adam_exp_avg_sq_norm",
    "adam_exp_avg_sq_sqrt_sum",
    "adam_exp_avg_pair_dot",
    "adam_exp_avg_pair_cosine",
    "adam_current_step_number",
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "adam_weight_decay",
    "adam_amsgrad",
    "adam_raw_displacement_norm",
    "adam_raw_pair_dot",
    "adam_raw_pair_descent_dot",
    "adam_raw_pair_descent_cosine",
    "final_displacement_norm",
    "final_pair_dot",
    "final_pair_descent_dot",
    "final_pair_descent_cosine",
    "adam_to_final_displacement_delta_norm",
    "final_parameters_equal_raw_adam",
    "pair_optimizer_isolated",
    "pair_actual_update_direction_guard_applied",
    "pair_optimizer_state_sync_applied",
    "standard_pair_gradient_projection_intervened",
    "pair_target_gradient_constraint_count",
    "pair_target_gradient_projection_intervened",
    "pair_target_gradient_min_dot_before",
    "pair_target_gradient_min_dot_after",
    "pair_target_gradient_projection_delta_norm",
    "pair_target_actual_direction_guard_applied",
    "pair_target_actual_min_descent_dot_before",
    "pair_target_actual_min_descent_dot_after",
    "pair_target_optimizer_state_sync_applied",
    "pair_boundary_diagnostic_version",
    "pair_boundary_gradient_constraint_count",
    "pair_boundary_gradient_projection_intervened",
    "pair_boundary_gradient_min_dot_before",
    "pair_boundary_gradient_min_dot_after",
    "pair_boundary_actual_direction_guard_applied",
    "pair_boundary_actual_min_descent_dot_before",
    "pair_boundary_actual_min_descent_dot_after",
    "pair_boundary_target_count",
    "pair_boundary_correct_direction_count",
    "pair_boundary_reverse_direction_count",
    "pair_boundary_approximately_zero_count",
    "pair_boundary_signed_margin_change_mean",
    "pair_boundary_signed_margin_change_median",
    "pair_boundary_signed_margin_change_worst",
    "pair_boundary_rank_crossing_count",
    "pair_boundary_positive_promotion_count",
    "pair_boundary_negative_eviction_count",
    "pair_boundary_nonlinear_backtrack_count",
    "pair_boundary_nonlinear_backtrack_final_scale",
    "pair_boundary_nonlinear_refinement_count",
    "pair_boundary_nonlinear_invalid_upper_scale",
    "pair_boundary_direction_candidate_count",
    "pair_boundary_direction_valid_candidate_count",
    "pair_boundary_selected_progress_floor_fraction",
    "pair_boundary_progress_min_completion",
    "pair_boundary_progress_mean_completion",
    "pair_boundary_limiting_constraint_code",
    "pair_boundary_limiting_target_ordinal",
    "pair_boundary_joint_candidate_exact_valid",
    "pair_boundary_joint_lifecycle_exact_valid",
    "candidate_gradient_projection_intervened",
    "candidate_actual_update_correction_intervened",
    "lifecycle_gradient_projection_intervened",
    "lifecycle_actual_update_correction_intervened",
    "lifecycle_current_priority_repair_intervened",
    "lifecycle_current_priority_min_dot_before",
    "lifecycle_current_priority_min_dot_after",
    "lifecycle_final_linear_min_dot",
    "lifecycle_final_linear_max_tolerance",
    "lifecycle_final_linear_max_normalized_violation",
    "lifecycle_final_linear_rounding_residual_count",
    "lifecycle_final_exact_revalidation_valid",
    "lifecycle_final_exact_min_signed_gap",
    "lifecycle_final_exact_max_tolerance",
    "lifecycle_backtrack_count",
    "lifecycle_reject_occurred",
    "retention_protected_target_count",
    "retention_context_invalidated_count",
    "retention_superseded_count",
    "retention_gradient_projection_intervened",
    "retention_actual_projection_corrected",
    "retention_nonlinear_backtrack_count",
    "retention_final_scale",
    "retention_optimizer_state_sync_applied",
    "retention_final_exact_min_signed_gap",
    "retention_final_exact_max_tolerance",
    "retention_final_postcondition_entered",
    "retention_final_postcondition_target_count",
    "retention_selection_state_backtrack_count",
    "retention_selection_state_refinement_count",
    "retention_selection_state_final_scale",
    "retention_selection_state_unsafe_upper_scale",
    "retention_selection_state_seen_count_delta",
    "retention_selection_state_seen_count_integral_valid",
    "rollback_occurred",
    "exact_score_signed_change_mean",
    "positive_target_score_change_mean",
    "negative_target_signed_score_change_mean",
    "positive_target_count",
    "negative_target_count",
    "score_correct_direction_target_count",
    "score_reverse_direction_target_count",
    "score_approximately_zero_target_count",
    "score_before_after_target_join_valid",
    "score_zero_tolerance",
) + _ADJ_TRANSACTION_V3_CSV_FIELDS
_ADJ_TRANSACTION_TRAIN_INFO_FIELDS = {
    "diagnostic_version":
        "pair_optimizer_transaction_diagnostic_version",
    "support_version":
        "pair_optimizer_transaction_support_version",
    "credit_observation_consumed":
        "adv_factor_credit_memory_observation_consumed",
    "pair_credit_raw_observation_count":
        "adv_triplet_credit_pair_updates",
    "triplet_credit_raw_observation_count":
        "adv_triplet_credit_triplet_updates",
    "pair_credit_state_update_count":
        "adv_triplet_credit_pair_state_updates",
    "triplet_credit_state_update_count":
        "adv_triplet_credit_triplet_state_updates",
    "nonzero_pair_transaction":
        "pair_optimizer_transaction_nonzero_pair",
    "pair_scalar_loss":
        "pair_optimizer_transaction_pair_scalar_loss",
    "base_factor_scalar_loss":
        "pair_optimizer_transaction_base_factor_scalar_loss",
    "total_adjacency_loss":
        "pair_optimizer_transaction_total_adjacency_loss",
    "optimizer_step_before":
        "pair_optimizer_transaction_optimizer_step_before",
    "optimizer_step_before_min":
        "pair_optimizer_transaction_optimizer_step_before_min",
    "optimizer_step_before_max":
        "pair_optimizer_transaction_optimizer_step_before_max",
    "optimizer_step_after":
        "pair_optimizer_transaction_optimizer_step_after",
    "optimizer_step_after_min":
        "pair_optimizer_transaction_optimizer_step_after_min",
    "optimizer_step_after_max":
        "pair_optimizer_transaction_optimizer_step_after_max",
    "pair_grad_norm":
        "pair_optimizer_transaction_pair_grad_norm",
    "pair_grad_finite":
        "pair_optimizer_transaction_pair_grad_finite",
    "pair_grad_zero":
        "pair_optimizer_transaction_pair_grad_zero",
    "base_factor_grad_norm":
        "pair_optimizer_transaction_base_factor_grad_norm",
    "pair_base_grad_dot":
        "pair_optimizer_transaction_pair_base_grad_dot",
    "pair_base_grad_cosine":
        "pair_optimizer_transaction_pair_base_grad_cosine",
    "combined_grad_norm_preclip":
        "pair_optimizer_transaction_combined_grad_norm_preclip",
    "pair_combined_grad_dot_preclip":
        "pair_optimizer_transaction_pair_combined_grad_dot_preclip",
    "pair_combined_grad_cosine_preclip":
        "pair_optimizer_transaction_pair_combined_grad_cosine_preclip",
    "pair_combined_descent_component_preclip":
        "pair_optimizer_transaction_pair_combined_descent_component_preclip",
    "combined_grad_norm_postclip":
        "pair_optimizer_transaction_combined_grad_norm_postclip",
    "pair_combined_grad_dot_postclip":
        "pair_optimizer_transaction_pair_combined_grad_dot_postclip",
    "pair_combined_grad_cosine_postclip":
        "pair_optimizer_transaction_pair_combined_grad_cosine_postclip",
    "pair_combined_descent_component_postclip":
        "pair_optimizer_transaction_pair_combined_descent_component_postclip",
    "gradient_clip_reported_preclip_norm":
        "pair_optimizer_transaction_gradient_clip_reported_preclip_norm",
    "gradient_clip_applied":
        "pair_optimizer_transaction_gradient_clip_applied",
    "gradient_clip_scale":
        "pair_optimizer_transaction_gradient_clip_scale",
    "adam_exp_avg_norm":
        "pair_optimizer_transaction_adam_exp_avg_norm",
    "adam_exp_avg_sq_norm":
        "pair_optimizer_transaction_adam_exp_avg_sq_norm",
    "adam_exp_avg_sq_sqrt_sum":
        "pair_optimizer_transaction_adam_exp_avg_sq_sqrt_sum",
    "adam_exp_avg_pair_dot":
        "pair_optimizer_transaction_adam_exp_avg_pair_dot",
    "adam_exp_avg_pair_cosine":
        "pair_optimizer_transaction_adam_exp_avg_pair_cosine",
    "adam_current_step_number":
        "pair_optimizer_transaction_optimizer_step_before",
    "learning_rate":
        "pair_optimizer_transaction_learning_rate",
    "adam_beta1":
        "pair_optimizer_transaction_adam_beta1",
    "adam_beta2":
        "pair_optimizer_transaction_adam_beta2",
    "adam_eps":
        "pair_optimizer_transaction_adam_eps",
    "adam_weight_decay":
        "pair_optimizer_transaction_adam_weight_decay",
    "adam_amsgrad":
        "pair_optimizer_transaction_adam_amsgrad",
    "adam_raw_displacement_norm":
        "pair_optimizer_transaction_adam_raw_displacement_norm",
    "adam_raw_pair_dot":
        "pair_optimizer_transaction_adam_raw_pair_dot",
    "adam_raw_pair_descent_dot":
        "pair_optimizer_transaction_adam_raw_pair_descent_dot",
    "adam_raw_pair_descent_cosine":
        "pair_optimizer_transaction_adam_raw_pair_descent_cosine",
    "final_displacement_norm":
        "pair_optimizer_transaction_final_displacement_norm",
    "final_pair_dot":
        "pair_optimizer_transaction_final_pair_dot",
    "final_pair_descent_dot":
        "pair_optimizer_transaction_final_pair_descent_dot",
    "final_pair_descent_cosine":
        "pair_optimizer_transaction_final_pair_descent_cosine",
    "adam_to_final_displacement_delta_norm":
        "pair_optimizer_transaction_adam_to_final_displacement_delta_norm",
    "final_parameters_equal_raw_adam":
        "pair_optimizer_transaction_final_parameters_equal_raw_adam",
    "pair_optimizer_isolated":
        "pair_optimizer_transaction_pair_optimizer_isolated",
    "pair_actual_update_direction_guard_applied":
        "pair_optimizer_transaction_pair_actual_update_direction_guard_applied",
    "pair_optimizer_state_sync_applied":
        "pair_optimizer_transaction_pair_optimizer_state_sync_applied",
    "standard_pair_gradient_projection_intervened":
        "pair_optimizer_transaction_standard_pair_gradient_projection_intervened",
    "pair_target_gradient_constraint_count":
        "pair_optimizer_transaction_pair_target_gradient_constraint_count",
    "pair_target_gradient_projection_intervened":
        "pair_optimizer_transaction_pair_target_gradient_projection_intervened",
    "pair_target_gradient_min_dot_before":
        "pair_optimizer_transaction_pair_target_gradient_min_dot_before",
    "pair_target_gradient_min_dot_after":
        "pair_optimizer_transaction_pair_target_gradient_min_dot_after",
    "pair_target_gradient_projection_delta_norm":
        "pair_optimizer_transaction_pair_target_gradient_projection_delta_norm",
    "pair_target_actual_direction_guard_applied":
        "pair_optimizer_transaction_pair_target_actual_direction_guard_applied",
    "pair_target_actual_min_descent_dot_before":
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_before",
    "pair_target_actual_min_descent_dot_after":
        "pair_optimizer_transaction_pair_target_actual_min_descent_dot_after",
    "pair_target_optimizer_state_sync_applied":
        "pair_optimizer_transaction_pair_target_optimizer_state_sync_applied",
    "pair_boundary_diagnostic_version":
        "pair_optimizer_transaction_pair_boundary_diagnostic_version",
    "pair_boundary_gradient_constraint_count":
        "pair_optimizer_transaction_pair_boundary_gradient_constraint_count",
    "pair_boundary_gradient_projection_intervened":
        "pair_optimizer_transaction_pair_boundary_gradient_projection_intervened",
    "pair_boundary_gradient_min_dot_before":
        "pair_optimizer_transaction_pair_boundary_gradient_min_dot_before",
    "pair_boundary_gradient_min_dot_after":
        "pair_optimizer_transaction_pair_boundary_gradient_min_dot_after",
    "pair_boundary_actual_direction_guard_applied":
        "pair_optimizer_transaction_pair_boundary_actual_direction_guard_applied",
    "pair_boundary_actual_min_descent_dot_before":
        "pair_optimizer_transaction_pair_boundary_actual_min_descent_dot_before",
    "pair_boundary_actual_min_descent_dot_after":
        "pair_optimizer_transaction_pair_boundary_actual_min_descent_dot_after",
    "pair_boundary_target_count":
        "pair_optimizer_transaction_pair_boundary_target_count",
    "pair_boundary_correct_direction_count":
        "pair_optimizer_transaction_pair_boundary_correct_direction_count",
    "pair_boundary_reverse_direction_count":
        "pair_optimizer_transaction_pair_boundary_reverse_direction_count",
    "pair_boundary_approximately_zero_count":
        "pair_optimizer_transaction_pair_boundary_approximately_zero_count",
    "pair_boundary_signed_margin_change_mean":
        "pair_optimizer_transaction_pair_boundary_signed_margin_change_mean",
    "pair_boundary_signed_margin_change_median":
        "pair_optimizer_transaction_pair_boundary_signed_margin_change_median",
    "pair_boundary_signed_margin_change_worst":
        "pair_optimizer_transaction_pair_boundary_signed_margin_change_worst",
    "pair_boundary_rank_crossing_count":
        "pair_optimizer_transaction_pair_boundary_rank_crossing_count",
    "pair_boundary_positive_promotion_count":
        "pair_optimizer_transaction_pair_boundary_positive_promotion_count",
    "pair_boundary_negative_eviction_count":
        "pair_optimizer_transaction_pair_boundary_negative_eviction_count",
    "pair_boundary_nonlinear_backtrack_count":
        "pair_optimizer_transaction_pair_boundary_nonlinear_backtrack_count",
    "pair_boundary_nonlinear_backtrack_final_scale":
        "pair_optimizer_transaction_pair_boundary_nonlinear_backtrack_final_scale",
    "pair_boundary_nonlinear_refinement_count":
        "pair_optimizer_transaction_pair_boundary_nonlinear_refinement_count",
    "pair_boundary_nonlinear_invalid_upper_scale":
        "pair_optimizer_transaction_pair_boundary_nonlinear_invalid_upper_scale",
    "pair_boundary_direction_candidate_count":
        "pair_optimizer_transaction_pair_boundary_direction_candidate_count",
    "pair_boundary_direction_valid_candidate_count":
        "pair_optimizer_transaction_pair_boundary_direction_valid_candidate_count",
    "pair_boundary_selected_progress_floor_fraction":
        "pair_optimizer_transaction_pair_boundary_selected_progress_floor_fraction",
    "pair_boundary_progress_min_completion":
        "pair_optimizer_transaction_pair_boundary_progress_min_completion",
    "pair_boundary_progress_mean_completion":
        "pair_optimizer_transaction_pair_boundary_progress_mean_completion",
    "pair_boundary_limiting_constraint_code":
        "pair_optimizer_transaction_pair_boundary_limiting_constraint_code",
    "pair_boundary_limiting_target_ordinal":
        "pair_optimizer_transaction_pair_boundary_limiting_target_ordinal",
    "pair_boundary_joint_candidate_exact_valid":
        "pair_optimizer_transaction_pair_boundary_joint_candidate_exact_valid",
    "pair_boundary_joint_lifecycle_exact_valid":
        "pair_optimizer_transaction_pair_boundary_joint_lifecycle_exact_valid",
    "candidate_gradient_projection_intervened":
        "pair_optimizer_transaction_candidate_gradient_projection_intervened",
    "candidate_actual_update_correction_intervened":
        "pair_optimizer_transaction_candidate_actual_update_correction_intervened",
    "lifecycle_gradient_projection_intervened":
        "pair_optimizer_transaction_lifecycle_gradient_projection_intervened",
    "lifecycle_actual_update_correction_intervened":
        "pair_optimizer_transaction_lifecycle_actual_update_correction_intervened",
    "lifecycle_current_priority_repair_intervened":
        "pair_optimizer_transaction_lifecycle_current_priority_repair_intervened",
    "lifecycle_current_priority_min_dot_before":
        "pair_optimizer_transaction_lifecycle_current_priority_min_dot_before",
    "lifecycle_current_priority_min_dot_after":
        "pair_optimizer_transaction_lifecycle_current_priority_min_dot_after",
    "lifecycle_final_linear_min_dot":
        "pair_optimizer_transaction_lifecycle_final_linear_min_dot",
    "lifecycle_final_linear_max_tolerance":
        "pair_optimizer_transaction_lifecycle_final_linear_max_tolerance",
    "lifecycle_final_linear_max_normalized_violation":
        "pair_optimizer_transaction_lifecycle_final_linear_max_normalized_violation",
    "lifecycle_final_linear_rounding_residual_count":
        "pair_optimizer_transaction_lifecycle_final_linear_rounding_residual_count",
    "lifecycle_final_exact_revalidation_valid":
        "pair_optimizer_transaction_lifecycle_final_exact_revalidation_valid",
    "lifecycle_final_exact_min_signed_gap":
        "pair_optimizer_transaction_lifecycle_final_exact_min_signed_gap",
    "lifecycle_final_exact_max_tolerance":
        "pair_optimizer_transaction_lifecycle_final_exact_max_tolerance",
    "retention_protected_target_count":
        "pair_selection_boundary_retention_protected_target_count",
    "retention_context_invalidated_count":
        "pair_selection_boundary_retention_context_invalidated_count",
    "retention_superseded_count":
        "pair_selection_boundary_retention_superseded_count",
    "retention_gradient_projection_intervened":
        "pair_selection_boundary_retention_gradient_projection_intervened",
    "retention_actual_projection_corrected":
        "pair_selection_boundary_retention_actual_projection_corrected",
    "retention_nonlinear_backtrack_count":
        "pair_selection_boundary_retention_nonlinear_backtrack_count",
    "retention_final_scale":
        "pair_selection_boundary_retention_final_scale",
    "retention_optimizer_state_sync_applied":
        "pair_selection_boundary_retention_optimizer_state_sync_applied",
    "retention_final_exact_min_signed_gap":
        "pair_selection_boundary_retention_final_exact_min_signed_gap",
    "retention_final_exact_max_tolerance":
        "pair_selection_boundary_retention_final_exact_max_tolerance",
    "retention_final_postcondition_entered":
        "pair_selection_boundary_retention_final_postcondition_entered",
    "retention_final_postcondition_target_count":
        "pair_selection_boundary_retention_final_postcondition_target_count",
    "retention_selection_state_backtrack_count":
        "pair_selection_boundary_retention_selection_state_backtrack_count",
    "retention_selection_state_refinement_count":
        "pair_selection_boundary_retention_selection_state_refinement_count",
    "retention_selection_state_final_scale":
        "pair_selection_boundary_retention_selection_state_final_scale",
    "retention_selection_state_unsafe_upper_scale":
        "pair_selection_boundary_retention_selection_state_unsafe_upper_scale",
    "retention_selection_state_seen_count_delta":
        "pair_selection_boundary_retention_selection_state_seen_count_delta",
    "retention_selection_state_seen_count_integral_valid":
        "pair_selection_boundary_retention_selection_state_seen_count_integral_valid",
    "lifecycle_backtrack_count":
        "pair_optimizer_transaction_lifecycle_backtrack_count",
    "lifecycle_reject_occurred":
        "pair_optimizer_transaction_lifecycle_reject_occurred",
    "rollback_occurred":
        "pair_optimizer_transaction_rollback_occurred",
    "exact_score_signed_change_mean":
        "pair_optimizer_transaction_score_signed_change_mean",
    "positive_target_score_change_mean":
        "pair_optimizer_transaction_positive_score_change_mean",
    "negative_target_signed_score_change_mean":
        "pair_optimizer_transaction_negative_signed_score_change_mean",
    "positive_target_count":
        "pair_optimizer_transaction_positive_target_count",
    "negative_target_count":
        "pair_optimizer_transaction_negative_target_count",
    "score_correct_direction_target_count":
        "pair_optimizer_transaction_score_correct_direction_target_count",
    "score_reverse_direction_target_count":
        "pair_optimizer_transaction_score_reverse_direction_target_count",
    "score_approximately_zero_target_count":
        "pair_optimizer_transaction_score_approximately_zero_target_count",
    "score_before_after_target_join_valid":
        "pair_optimizer_transaction_score_before_after_join_valid",
    "score_zero_tolerance":
        "pair_optimizer_transaction_score_zero_tolerance",
}
for _objective_name in _ADJ_TRANSACTION_OBJECTIVE_NAMES:
    for _objective_metric in _ADJ_TRANSACTION_OBJECTIVE_METRICS:
        _csv_field = "objective_{}_{}".format(
            _objective_name,
            _objective_metric,
        )
        _ADJ_TRANSACTION_TRAIN_INFO_FIELDS[_csv_field] = (
            "pair_optimizer_transaction_{}".format(_csv_field)
        )
for _global_field in _ADJ_TRANSACTION_V3_GLOBAL_FIELDS:
    _ADJ_TRANSACTION_TRAIN_INFO_FIELDS[_global_field] = (
        "pair_optimizer_transaction_{}".format(_global_field)
    )
del _objective_name
del _objective_metric
del _csv_field
del _global_field


def _expected_adj_transaction_partition_count(
        class_complete,
        selected_chunk_count,
        num_mini_batch):
    """Return the exact number of standard Adam transactions in one epoch."""
    selected_chunk_count = int(selected_chunk_count)
    if selected_chunk_count <= 0:
        raise RuntimeError("adjacency transaction has no selected chunk")
    if bool(class_complete):
        return 1
    return max(1, min(int(num_mini_batch), selected_chunk_count))


def _build_adj_transaction_row(
        run_id,
        env_step,
        adjacency_update_round,
        ppo_epoch_index,
        policy_id,
        partition_index,
        transaction_sequence_index,
        selected_episode_count,
        selected_chunk_count,
        transaction_chunk_count,
        train_adj_info):
    """Build and validate one unaggregated standard-Adam transaction row."""
    missing = [
        train_key
        for train_key in _ADJ_TRANSACTION_TRAIN_INFO_FIELDS.values()
        if train_key not in train_adj_info
    ]
    for required_key in (
            "pair_optimizer_transaction_enabled",
            "pair_optimizer_class_complete",
            "pair_pursuit_factor_loss_target_transition_count",
            "pair_optimizer_positive_mass",
            "pair_optimizer_negative_mass",
            "pair_optimizer_centered_error",
            "pair_pending_objective_scope_pair_only"):
        if required_key not in train_adj_info:
            missing.append(required_key)
    if missing:
        raise RuntimeError(
            "standard adjacency transaction diagnostics are missing: {}"
            .format(", ".join(sorted(set(missing))))
        )
    if float(train_adj_info["pair_optimizer_transaction_enabled"]) != 1.0:
        raise RuntimeError(
            "standard adjacency transaction diagnostics were disabled"
        )
    row = {
        "run_id": str(run_id),
        "env_step": int(env_step),
        "adjacency_update_round": int(adjacency_update_round),
        "ppo_epoch_index": int(ppo_epoch_index),
        "policy_id": str(policy_id),
        "partition_index": int(partition_index),
        "transaction_sequence_index": int(transaction_sequence_index),
        "optimizer_kind": (
            "pair_pending_adam"
            if float(train_adj_info.get(
                "pair_pending_objective_scope_pair_only", 0.0
            )) == 1.0
            else "standard_adam"
        ),
        "class_complete": float(
            train_adj_info["pair_optimizer_class_complete"]
        ),
        "selected_episode_count": float(selected_episode_count),
        "selected_chunk_count": float(selected_chunk_count),
        "transaction_chunk_count": float(transaction_chunk_count),
        "target_bearing_transition_count": float(train_adj_info[
            "pair_pursuit_factor_loss_target_transition_count"
        ]),
        "positive_pair_mass": float(train_adj_info[
            "pair_optimizer_positive_mass"
        ]),
        "negative_pair_mass": float(train_adj_info[
            "pair_optimizer_negative_mass"
        ]),
        "centered_mass_error": float(train_adj_info[
            "pair_optimizer_centered_error"
        ]),
    }
    for csv_key, train_key in _ADJ_TRANSACTION_TRAIN_INFO_FIELDS.items():
        row[csv_key] = float(train_adj_info[train_key])
    if set(row.keys()) != set(_ADJ_TRANSACTION_CSV_FIELDS):
        missing_csv = sorted(set(_ADJ_TRANSACTION_CSV_FIELDS) - set(row.keys()))
        extra_csv = sorted(set(row.keys()) - set(_ADJ_TRANSACTION_CSV_FIELDS))
        raise RuntimeError(
            "adjacency transaction CSV schema mismatch: missing={}, extra={}"
            .format(missing_csv, extra_csv)
        )
    if float(row["diagnostic_version"]) != float(
            _PAIR_OPTIMIZER_TRANSACTION_DIAGNOSTIC_VERSION):
        raise RuntimeError("unexpected pair optimizer diagnostic version")
    if float(row["support_version"]) != 6.0:
        raise RuntimeError("unexpected pair optimizer support version")
    credit_consumed = float(row["credit_observation_consumed"])
    if credit_consumed not in (0.0, 1.0):
        raise RuntimeError(
            "factor-credit observation consumption must be binary"
        )
    credit_count_fields = (
        "pair_credit_raw_observation_count",
        "triplet_credit_raw_observation_count",
        "pair_credit_state_update_count",
        "triplet_credit_state_update_count",
    )
    for field in credit_count_fields:
        value = float(row[field])
        if (
                not np.isfinite(value)
                or value < 0.0
                or value != float(int(value))):
            raise RuntimeError(
                "factor-credit diagnostic count must be a non-negative "
                "integer: {}".format(field)
            )
    if credit_consumed == 0.0 and any(
            float(row[field]) != 0.0 for field in credit_count_fields):
        raise RuntimeError(
            "unconsumed factor-credit observations changed diagnostic counts"
        )
    if credit_consumed == 1.0 and int(row["ppo_epoch_index"]) != 0:
        raise RuntimeError(
            "factor-credit observations were consumed outside fresh epoch zero"
        )
    if (
            float(row["pair_credit_state_update_count"])
            > float(row["pair_credit_raw_observation_count"])
            or float(row["triplet_credit_state_update_count"])
            > float(row["triplet_credit_raw_observation_count"])):
        raise RuntimeError(
            "factor-credit state updates exceed raw observation counts"
        )
    if row["optimizer_kind"] == "pair_pending_adam":
        if credit_consumed != 0.0:
            raise RuntimeError(
                "pending pair-only replay consumed factor-credit observations"
            )
        if row["pair_optimizer_isolated"] != 1.0:
            raise RuntimeError(
                "pair-only transaction did not use its isolated Adam"
            )
        if row["final_pair_descent_dot"] <= 0.0:
            raise RuntimeError(
                "pair-only transaction did not commit a descent update"
            )
    for field in _ADJ_TRANSACTION_CSV_FIELDS:
        if field in ("run_id", "policy_id", "optimizer_kind"):
            continue
        if not np.isfinite(float(row[field])):
            raise FloatingPointError(
                "non-finite adjacency transaction field: {}".format(field)
            )
    if row["optimizer_step_after"] != row["optimizer_step_before"] + 1.0:
        raise RuntimeError(
            "transaction optimizer step after must equal before + 1"
        )
    if row["pair_grad_finite"] != 1.0:
        raise FloatingPointError("pair transaction gradient is non-finite")
    for reconstruction_field in (
            "objective_scalar_reconstruction_valid",
            "independent_sum_reconstruction_valid"):
        if row[reconstruction_field] != 1.0:
            raise RuntimeError(
                "adjacency objective reconstruction contract failed: {}"
                .format(reconstruction_field)
            )
    if not np.isclose(
            row["raw_combined_grad_norm_from_backward"],
            row["pre_projection_combined_grad_norm"],
            rtol=1e-6,
            atol=1e-8):
        raise RuntimeError(
            "raw total gradient and pre-projection gradient norms disagree"
        )
    if not np.isclose(
            row["post_projection_combined_grad_norm"],
            row["combined_grad_norm_preclip"],
            rtol=1e-6,
            atol=1e-8):
        raise RuntimeError(
            "post-projection gradient and optimizer pre-clip norm disagree"
        )
    if (
            row["nonzero_pair_transaction"] > 0.0
            and row["target_bearing_transition_count"] <= 0.0):
        raise RuntimeError(
            "nonzero pair transaction has no target-bearing transition"
        )
    if row["nonzero_pair_transaction"] > 0.0:
        exact_target_count = (
            row["positive_target_count"] + row["negative_target_count"]
        )
        exact_score_classified_count = (
            row["score_correct_direction_target_count"]
            + row["score_reverse_direction_target_count"]
            + row["score_approximately_zero_target_count"]
        )
        if row["pair_target_gradient_constraint_count"] != exact_target_count:
            raise RuntimeError(
                "{} pair target-Jacobian constraint count differs from the "
                "exact target population".format(row["optimizer_kind"])
            )
        if (
                row["pair_target_gradient_min_dot_after"] <= 0.0
                or row[
                    "pair_target_actual_min_descent_dot_after"
                ] <= 0.0):
            raise RuntimeError(
                "{} pair transaction did not preserve every exact target "
                "direction".format(row["optimizer_kind"])
            )
        if exact_score_classified_count != exact_target_count:
            raise RuntimeError(
                "{} pair transaction exact-target score classifications do "
                "not partition the target population"
                .format(row["optimizer_kind"])
            )
        # The production nonlinear progress contract is deliberately stricter
        # for selected-factor boundaries than for the selected exact scores.
        # Boundaries must improve strictly, while float32 writeback may preserve
        # an exact score within the producer-reported tolerance.  The trainer
        # already rejects every score classified as reverse (below -tolerance),
        # so the CSV consumer must accept correct + approximately-zero members
        # and fail only on an actual tolerance-exceeding reversal.
        if (
                row["score_reverse_direction_target_count"] != 0.0
                or (
                    row["score_correct_direction_target_count"]
                    + row["score_approximately_zero_target_count"]
                ) != exact_target_count):
            raise RuntimeError(
                "{} pair transaction reversed an exact target beyond its "
                "audited preservation tolerance"
                .format(row["optimizer_kind"])
            )
        if (
                row["pair_target_actual_direction_guard_applied"] > 0.0
                and row[
                    "pair_target_optimizer_state_sync_applied"
                ] != 1.0):
            raise RuntimeError(
                "standard pair target update guard did not synchronize Adam"
            )
    if row["class_complete"] > 0.0:
        if (
                int(row["partition_index"]) != 0
                or row["transaction_chunk_count"]
                != row["selected_chunk_count"]):
            raise RuntimeError(
                "support v6 class-complete transaction is not one full "
                "selected population"
            )
    return row


def _validate_adj_transaction_update_records(records, sequence_start):
    """Fail on missing, duplicate, or non-unit standard Adam transactions."""
    sequence_start = int(sequence_start)
    expected_sequences = list(range(
        sequence_start,
        sequence_start + len(records),
    ))
    actual_sequences = [
        int(record["transaction_sequence_index"])
        for record in records
    ]
    if actual_sequences != expected_sequences:
        raise RuntimeError(
            "adjacency transaction sequence is duplicated, missing, or "
            "out of order"
        )
    keys = [
        (
            int(record["env_step"]),
            int(record["adjacency_update_round"]),
            int(record["ppo_epoch_index"]),
            str(record["policy_id"]),
            int(record["partition_index"]),
        )
        for record in records
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate adjacency optimizer transaction key")
    step_increment = sum(
        float(record["optimizer_step_after"])
        - float(record["optimizer_step_before"])
        for record in records
    )
    if step_increment != float(len(records)):
        raise RuntimeError(
            "transaction log count does not match standard Adam step increment"
        )
    return True


def _candidate_evidence_target_key(row, draft):
    """Return the exact target location used to join replay and trainer traces."""
    if draft:
        return (
            int(row["episode_ordinal"]),
            int(row["capture_step"]),
            int(row["candidate_index"]),
            int(row["target_sign"]),
        )
    return (
        int(row["selected_episode_ordinal"]),
        int(row["episode_transition_step"]),
        int(row["candidate_index"]),
        int(row["target_sign"]),
    )


def _join_candidate_evidence_consumption_rows(consumption_rows, draft_rows):
    """Join every consumed dense target to its complete real-event group.

    Multiple real captures may occur at the same episode step and contribute
    to the same canonical candidate/sign cell.  The training tensor correctly
    stores their summed mass, so provenance must retain the full event group
    instead of guessing one representative event.  A group is accepted only
    when its distinct event rows reconstruct the consumed target exactly.
    """
    consumption_rows = list(consumption_rows)
    draft_rows = list(draft_rows)
    if not consumption_rows:
        return []
    draft_by_key = {}
    for draft_row in draft_rows:
        if tuple(draft_row.keys()) != (
                CANDIDATE_EVIDENCE_PROVENANCE_DRAFT_FIELDS):
            raise RuntimeError(
                "candidate evidence provenance draft has an incompatible schema"
            )
        if int(draft_row["diagnostic_version"]) != int(
                CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected candidate evidence provenance draft version"
            )
        key = _candidate_evidence_target_key(draft_row, draft=True)
        draft_by_key.setdefault(key, []).append(draft_row)

    joined_rows = []
    seen_consumption_keys = set()
    for consumption_row in consumption_rows:
        if tuple(consumption_row.keys()) != (
                _CANDIDATE_EVIDENCE_CONSUMPTION_FIELDS):
            raise RuntimeError(
                "candidate evidence consumption trace has an incompatible schema"
            )
        if int(consumption_row["diagnostic_version"]) != int(
                CANDIDATE_EVIDENCE_PROVENANCE_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected candidate evidence consumption version"
            )
        key = _candidate_evidence_target_key(
            consumption_row,
            draft=False,
        )
        if key in seen_consumption_keys:
            raise RuntimeError(
                "candidate evidence consumption target is duplicated"
            )
        seen_consumption_keys.add(key)
        matches = list(draft_by_key.get(key, ()))
        if not matches:
            raise RuntimeError(
                "candidate evidence target has no real capture event; "
                "key={}".format(
                    key,
                )
            )
        matches = sorted(
            matches,
            key=lambda row: (
                int(row["environment_episode_id"]),
                int(row["capture_event_id"]),
                int(row["prey_id"]),
                str(row["candidate_identity"]),
            ),
        )
        persisted_event_keys = [
            (
                int(row["environment_episode_id"]),
                int(row["capture_event_id"]),
                int(row["prey_id"]),
                str(row["candidate_identity"]),
                int(row["target_sign"]),
            )
            for row in matches
        ]
        if len(set(persisted_event_keys)) != len(persisted_event_keys):
            raise RuntimeError(
                "candidate evidence target contains a duplicate real event"
            )
        identities = {
            (
                str(row["candidate_identity"]),
                int(row["candidate_order"]),
            )
            for row in matches
        }
        if len(identities) != 1:
            raise RuntimeError(
                "candidate evidence target group contains incompatible "
                "canonical identities"
            )
        target_mass = sum(
            float(row["final_target_mass"]) for row in matches
        )
        if not np.isclose(
                float(consumption_row["target_weight"]),
                target_mass,
                rtol=0.0,
                atol=1e-6):
            raise RuntimeError(
                "candidate evidence target mass diverges from grouped event "
                "provenance"
            )
        for draft_row in matches:
            if not np.isclose(
                    float(consumption_row["behavior_policy_version"]),
                    float(draft_row["behavior_policy_version"]),
                    rtol=0.0,
                    atol=1e-6):
                raise RuntimeError(
                    "candidate evidence behavior policy version does not join"
                )
        joined_rows.append({
            "consumption": consumption_row,
            "drafts": tuple(matches),
        })
    return joined_rows


def _candidate_event_by_target_key(joined_evidence_rows):
    event_by_key = {}
    for joined_row in joined_evidence_rows:
        key = _candidate_evidence_target_key(
            joined_row["consumption"],
            draft=False,
        )
        if key in event_by_key:
            raise RuntimeError(
                "candidate evidence joined target key is duplicated"
            )
        drafts = tuple(joined_row["drafts"])
        if not drafts:
            raise RuntimeError(
                "candidate evidence joined target has an empty event group"
            )
        event_by_key[key] = drafts
    return event_by_key


def _build_candidate_evidence_provenance_csv_rows(
        joined_evidence_rows,
        run_id,
        env_step,
        policy_id,
        current_candidate_policy_version):
    """Finalize event provenance at the first real target consumption."""
    rows = []
    current_policy_version = float(current_candidate_policy_version)
    if (
            not np.isfinite(current_policy_version)
            or current_policy_version < 0.0):
        raise RuntimeError(
            "candidate evidence current policy version is invalid"
        )
    for joined_row in joined_evidence_rows:
        consumption = joined_row["consumption"]
        drafts = tuple(joined_row["drafts"])
        grouped_mass = sum(
            float(draft["final_target_mass"]) for draft in drafts
        )
        if not np.isclose(
                grouped_mass,
                float(consumption["target_weight"]),
                rtol=0.0,
                atol=1e-6):
            raise RuntimeError(
                "candidate evidence finalized event group does not reconstruct"
            )
        for draft in drafts:
            candidate_policy_age = (
                current_policy_version
                - float(draft["behavior_policy_version"])
            )
            if candidate_policy_age < -1e-6:
                raise RuntimeError(
                    "candidate evidence policy age is negative"
                )
            row = {
                "run_id": str(run_id),
                "env_step": int(env_step),
                "policy_id": str(policy_id),
                "replay_generation": int(draft["replay_generation"]),
                "environment_episode_id": int(
                    draft["environment_episode_id"]
                ),
                "episode_ordinal": int(draft["episode_ordinal"]),
                "capture_event_id": int(draft["capture_event_id"]),
                "prey_id": int(draft["prey_id"]),
                "capture_step": int(draft["capture_step"]),
                "terminal_step": int(draft["terminal_step"]),
                "capture_to_terminal_distance": int(
                    draft["capture_to_terminal_distance"]
                ),
                "outcome_success": int(draft["outcome_success"]),
                "candidate_index": int(draft["candidate_index"]),
                "candidate_identity": str(draft["candidate_identity"]),
                "candidate_order": int(draft["candidate_order"]),
                "participant_slots": str(draft["participant_slots"]),
                "participant_count": int(draft["participant_count"]),
                "static_dynamic_class": str(
                    draft["static_dynamic_class"]
                ),
                "target_sign": int(draft["target_sign"]),
                "raw_event_quality": float(draft["raw_event_quality"]),
                "identity_event_weight": float(
                    draft["identity_event_weight"]
                ),
                "identity_allocated_quality": float(
                    draft["identity_allocated_quality"]
                ),
                "candidate_coefficient": float(
                    draft["candidate_coefficient"]
                ),
                "final_target_mass": float(draft["final_target_mass"]),
                "target_bearing_transition_count": int(
                    draft["target_bearing_transition_count"]
                ),
                "base_selected": int(draft["base_selected"]),
                "support_selected": int(draft["support_selected"]),
                "behavior_policy_version": float(
                    draft["behavior_policy_version"]
                ),
                "candidate_policy_age": float(max(
                    candidate_policy_age,
                    0.0,
                )),
                "first_consumed_update": int(env_step),
                "consumed_generation_once": 1,
                "duplicate_generation": 0,
                "duplicate_event": 0,
                "quality_reconstruction_error": float(
                    draft["quality_reconstruction_error"]
                ),
                "provenance_complete": int(draft["provenance_complete"]),
                "identity_contract_valid": int(
                    draft["identity_contract_valid"]
                ),
                "quality_contract_valid": int(
                    draft["quality_contract_valid"]
                ),
            }
            if tuple(row.keys()) != (
                    _CANDIDATE_EVIDENCE_PROVENANCE_CSV_FIELDS):
                raise RuntimeError(
                    "candidate evidence provenance CSV schema diverged"
                )
            rows.append(row)
    return rows


def _candidate_evidence_persisted_key(row):
    return (
        str(row["policy_id"]),
        int(float(row["replay_generation"])),
        int(float(row["environment_episode_id"])),
        int(float(row["capture_event_id"])),
        int(float(row["prey_id"])),
        str(row["candidate_identity"]),
        int(float(row["target_sign"])),
    )


def _candidate_evidence_immutable_payload(row):
    ignored = {
        "env_step",
        "episode_ordinal",
        "raw_event_quality",
        "identity_allocated_quality",
        "final_target_mass",
        "target_bearing_transition_count",
        "base_selected",
        "support_selected",
        "behavior_policy_version",
        "candidate_policy_age",
        "first_consumed_update",
        "consumed_generation_once",
        "duplicate_generation",
        "duplicate_event",
        "quality_reconstruction_error",
        "quality_contract_valid",
    }
    return tuple(
        (field, str(row[field]))
        for field in _CANDIDATE_EVIDENCE_PROVENANCE_CSV_FIELDS
        if field not in ignored
    )


def _build_candidate_identity_transaction_rows(
        transaction_row,
        trace_rows,
        episode_rows,
        train_adj_info,
        candidate_event_by_target):
    """Join detached target traces to one persisted Adam transaction."""
    trace_rows = list(trace_rows)
    if not trace_rows:
        return []
    episode_by_ordinal = {}
    for episode_row in episode_rows:
        selected = int(round(float(
            episode_row.get("selected_for_training", 0)
        )))
        episode_ordinal = int(round(float(
            episode_row.get("selected_episode_ordinal", -1)
        )))
        if selected <= 0:
            continue
        if episode_ordinal < 0:
            raise RuntimeError(
                "selected replay episode has no selected ordinal"
            )
        if episode_ordinal in episode_by_ordinal:
            raise RuntimeError(
                "selected replay episode ordinal is duplicated"
            )
        episode_by_ordinal[episode_ordinal] = episode_row

    expected_trace_sequences = list(range(len(trace_rows)))
    actual_trace_sequences = [
        int(trace_row["target_row_sequence_within_transaction"])
        for trace_row in trace_rows
    ]
    if actual_trace_sequences != expected_trace_sequences:
        raise RuntimeError(
            "candidate target trace rows are missing or out of order"
        )
    output_rows = []
    for trace_row in trace_rows:
        if (
                tuple(trace_row.keys())
                != _CANDIDATE_IDENTITY_TRANSACTION_TRACE_FIELDS):
            raise RuntimeError(
                "candidate target trace has an incompatible schema"
            )
        diagnostic_version = int(trace_row["diagnostic_version"])
        if (
                diagnostic_version
                != _CANDIDATE_IDENTITY_TRANSACTION_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected candidate identity transaction diagnostic version"
            )
        episode_ordinal = int(trace_row["selected_episode_ordinal"])
        if episode_ordinal not in episode_by_ordinal:
            raise RuntimeError(
                "candidate target trace does not join to a selected replay "
                "episode"
            )
        episode_row = episode_by_ordinal[episode_ordinal]
        target_key = _candidate_evidence_target_key(
            trace_row,
            draft=False,
        )
        if target_key not in candidate_event_by_target:
            raise RuntimeError(
                "optimized candidate target has no real event provenance"
            )
        event_rows = tuple(candidate_event_by_target[target_key])
        if not event_rows:
            raise RuntimeError(
                "candidate transaction target joined an empty event group"
            )
        if any(
                str(trace_row["canonical_identity"])
                != str(event_row["candidate_identity"])
                or int(trace_row["factor_order"])
                != int(event_row["candidate_order"])
                for event_row in event_rows):
            raise RuntimeError(
                "candidate transaction identity/order disagrees with event "
                "provenance"
            )
        event_environment_ids = {
            int(event_row["environment_episode_id"])
            for event_row in event_rows
        }
        event_capture_steps = {
            int(event_row["capture_step"])
            for event_row in event_rows
        }
        if len(event_environment_ids) != 1 or len(event_capture_steps) != 1:
            raise RuntimeError(
                "candidate transaction event group is not target-local"
            )
        event_target_mass = sum(
            float(event_row["final_target_mass"])
            for event_row in event_rows
        )
        if not np.isclose(
                event_target_mass,
                float(trace_row["target_weight"]),
                rtol=0.0,
                atol=1e-6):
            raise RuntimeError(
                "candidate transaction event group mass does not reconstruct "
                "the optimized target"
            )
        if float(
                trace_row["successful_candidate_capture_overlap"]) > 0.0:
            identity_indices = {
                int(value)
                for value in str(
                    episode_row.get("candidate_identity_indices", "")
                ).split(";")
                if str(value).strip() != ""
            }
            if int(trace_row["candidate_index"]) not in identity_indices:
                raise RuntimeError(
                    "successful candidate target trace identity does not "
                    "match its episode-level capture identity"
                )
            factor_identities = {
                str(value)
                for value in str(
                    episode_row.get("candidate_factor_identities", "")
                ).split(";")
                if str(value).strip() != ""
            }
            if str(trace_row["canonical_identity"]) not in factor_identities:
                raise RuntimeError(
                    "successful candidate target trace canonical identity "
                    "does not match its episode provenance"
                )
        row = {
            "run_id": transaction_row["run_id"],
            "env_step": transaction_row["env_step"],
            "adjacency_update_round": transaction_row[
                "adjacency_update_round"
            ],
            "ppo_epoch_index": transaction_row["ppo_epoch_index"],
            "policy_id": transaction_row["policy_id"],
            "partition_index": transaction_row["partition_index"],
            "transaction_sequence_index": transaction_row[
                "transaction_sequence_index"
            ],
            "optimizer_step_before": transaction_row[
                "optimizer_step_before"
            ],
            "optimizer_step_after": transaction_row[
                "optimizer_step_after"
            ],
            "episode_generation": int(episode_row["episode_generation"]),
            "replay_slot_index": int(episode_row["replay_slot_index"]),
            "episode_recency_age": int(episode_row["episode_recency_age"]),
            "base_selected": int(round(float(
                episode_row["base_selected"]
            ))),
            "support_selected": int(round(float(
                episode_row["support_selected"]
            ))),
            "outcome_support_used": int(round(float(
                episode_row["outcome_support_used"]
            ))),
            "candidate_scalar_loss": float(train_adj_info[
                "capture_candidate_identity_loss_contribution"
            ]),
            "candidate_gradient_norm": float(train_adj_info[
                "capture_candidate_identity_gradient_norm"
            ]),
            "candidate_projected_gradient_dot": float(train_adj_info[
                "capture_candidate_identity_projected_gradient_dot"
            ]),
            "candidate_clipped_gradient_dot": float(train_adj_info[
                "capture_candidate_identity_clipped_gradient_dot"
            ]),
            "candidate_actual_update_descent_dot_before": float(
                train_adj_info[
                    "capture_candidate_identity_actual_update_descent_dot_before"
                ]
            ),
            "candidate_actual_update_descent_dot_after": float(
                train_adj_info[
                    "capture_candidate_identity_actual_update_descent_dot_after"
                ]
            ),
            "candidate_actual_update_corrected": float(train_adj_info[
                "capture_candidate_identity_actual_update_corrected"
            ]),
            "candidate_loss_optimizer_change": float(train_adj_info[
                "capture_candidate_identity_loss_optimizer_change"
            ]),
            "candidate_gradient_projection_intervened": transaction_row[
                "candidate_gradient_projection_intervened"
            ],
            "lifecycle_gradient_projection_intervened": transaction_row[
                "lifecycle_gradient_projection_intervened"
            ],
            "lifecycle_backtrack_count": transaction_row[
                "lifecycle_backtrack_count"
            ],
            "lifecycle_reject_occurred": transaction_row[
                "lifecycle_reject_occurred"
            ],
            "rollback_occurred": transaction_row["rollback_occurred"],
        }
        row.update(trace_row)
        event_group_size = len(event_rows)
        single_event = event_rows[0] if event_group_size == 1 else None
        row.update({
            "environment_episode_id": next(iter(event_environment_ids)),
            "capture_event_id": (
                int(single_event["capture_event_id"])
                if single_event is not None else -1
            ),
            "capture_prey_id": (
                int(single_event["prey_id"])
                if single_event is not None else -1
            ),
            "capture_step": next(iter(event_capture_steps)),
            "candidate_participant_slots": (
                str(single_event["participant_slots"])
                if single_event is not None else ""
            ),
            "candidate_static_dynamic_class": (
                str(single_event["static_dynamic_class"])
                if single_event is not None else "grouped"
            ),
            "candidate_event_group_size": int(event_group_size),
            "capture_event_ids": "|".join(
                str(int(event_row["capture_event_id"]))
                for event_row in event_rows
            ),
            "capture_prey_ids": "|".join(
                str(int(event_row["prey_id"]))
                for event_row in event_rows
            ),
            "candidate_event_target_masses": "|".join(
                "{:.17g}".format(float(event_row["final_target_mass"]))
                for event_row in event_rows
            ),
            "candidate_participant_slot_groups": "|".join(
                str(event_row["participant_slots"])
                for event_row in event_rows
            ),
            "candidate_static_dynamic_classes": "|".join(
                str(event_row["static_dynamic_class"])
                for event_row in event_rows
            ),
        })
        if tuple(row.keys()) != _CANDIDATE_IDENTITY_TRANSACTION_CSV_FIELDS:
            raise RuntimeError(
                "candidate identity transaction CSV row schema diverged"
            )
        output_rows.append(row)
    return output_rows


def _build_pair_direction_candidate_rows(transaction_row, trace_rows):
    """Persist every independently evaluated nonlinear direction candidate."""
    trace_rows = list(trace_rows)
    expected_count = int(round(float(
        transaction_row["pair_boundary_direction_candidate_count"]
    )))
    if not trace_rows:
        if expected_count != 0:
            raise RuntimeError(
                "strict-pair transaction has no direction-candidate trace"
            )
        return []
    if len(trace_rows) != expected_count:
        raise RuntimeError(
            "direction-candidate trace count differs from transaction count"
        )
    selected_count = sum(int(row["selected"]) for row in trace_rows)
    valid_count = sum(int(row["valid"]) for row in trace_rows)
    expected_valid = int(round(float(
        transaction_row["pair_boundary_direction_valid_candidate_count"]
    )))
    if selected_count != 1 or valid_count != expected_valid:
        raise RuntimeError(
            "direction-candidate selected/valid counts are inconsistent"
        )
    output_rows = []
    for candidate_ordinal, trace_row in enumerate(trace_rows):
        if tuple(trace_row.keys()) != _PAIR_DIRECTION_CANDIDATE_TRACE_FIELDS:
            raise RuntimeError("direction-candidate trace schema diverged")
        if int(trace_row["diagnostic_version"]) != int(
                _PAIR_DIRECTION_CANDIDATE_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected direction-candidate diagnostic version"
            )
        if int(trace_row["candidate_ordinal"]) != candidate_ordinal:
            raise RuntimeError(
                "direction-candidate rows are missing or out of order"
            )
        direction_kind = validate_pair_direction_candidate_kind(
            trace_row["direction_kind"]
        )
        numeric_fields = (
            "progress_floor_fraction",
            "direction_norm",
            "cosine_vs_full",
            "safe_lower_scale",
            "safe_frontier_scale",
            "unsafe_upper_scale",
            "scale_limit",
            "progress_min_completion",
            "progress_mean_completion",
            "progress_seed_raw_norm",
            "adam_reference_norm",
            "progress_component_norm",
        )
        if not all(np.isfinite(float(trace_row[field])) for field in numeric_fields):
            raise RuntimeError("non-finite direction-candidate diagnostic")
        cosine = float(trace_row["cosine_vs_full"])
        if cosine < -1.000001 or cosine > 1.000001:
            raise RuntimeError("invalid direction-candidate cosine")
        unsafe_upper_present = int(trace_row["unsafe_upper_present"])
        if unsafe_upper_present not in (0, 1):
            raise RuntimeError(
                "invalid direction-candidate unsafe-upper flag"
            )
        if int(trace_row["expansion_count"]) < 0:
            raise RuntimeError(
                "invalid direction-candidate expansion count"
            )
        safe_scale = float(trace_row["safe_lower_scale"])
        safe_frontier = float(trace_row["safe_frontier_scale"])
        upper_scale = float(trace_row["unsafe_upper_scale"])
        scale_limit = float(trace_row["scale_limit"])
        if scale_limit < 1.0 or safe_scale <= 0.0:
            raise RuntimeError(
                "invalid direction-candidate bounded scale interval"
            )
        if unsafe_upper_present:
            if int(trace_row["valid"]) == 1 and not (
                    safe_frontier >= safe_scale
                    and upper_scale > safe_frontier):
                raise RuntimeError(
                    "direction-candidate unsafe upper does not bracket safe"
                )
        elif not (
                int(trace_row["valid"]) == 1
                and np.isclose(
                    safe_frontier, scale_limit, rtol=0.0, atol=1.0e-12
                )
                and np.isclose(
                    upper_scale, scale_limit, rtol=0.0, atol=1.0e-12
                )
                and safe_scale <= safe_frontier):
            raise RuntimeError(
                "direction-candidate safe cap lacks an explicit endpoint"
            )
        active_ordinals = [
            value for value in str(
                trace_row["active_constraint_ordinals"]
            ).split("|") if value != ""
        ]
        if len(active_ordinals) != int(trace_row["active_constraint_count"]):
            raise RuntimeError(
                "direction-candidate active-set count is inconsistent"
            )
        seed_ordinals = [
            value for value in str(
                trace_row["progress_seed_member_ordinals"]
            ).split("|") if value != ""
        ]
        excluded_ordinals = [
            value for value in str(
                trace_row[
                    "progress_seed_zero_budget_excluded_ordinals"
                ]
            ).split("|") if value != ""
        ]
        if len(seed_ordinals) != int(
                trace_row["progress_seed_member_count"]):
            raise RuntimeError(
                "direction-candidate progress-seed count is inconsistent"
            )
        if len(excluded_ordinals) != int(
                trace_row[
                    "progress_seed_zero_budget_excluded_count"
                ]):
            raise RuntimeError(
                "direction-candidate excluded-seed count is inconsistent"
            )
        validate_pair_direction_candidate_seed_contract(
            direction_kind=direction_kind,
            seed_member_ordinals=seed_ordinals,
            zero_budget_excluded_ordinals=excluded_ordinals,
        )
        progress_target_present = int(
            trace_row["progress_target_present"]
        )
        if progress_target_present not in (0, 1):
            raise RuntimeError(
                "invalid direction-candidate progress-target flag"
            )
        progress_actual = trace_row["progress_worst_actual"]
        progress_required = trace_row["progress_worst_required"]
        if progress_target_present:
            if not (
                    np.isfinite(float(progress_actual))
                    and np.isfinite(float(progress_required))
                    and float(progress_required) > 0.0):
                raise RuntimeError(
                    "invalid direction-candidate original progress values"
                )
            reconstructed_completion = (
                float(progress_actual) / float(progress_required)
            )
            if not np.isclose(
                    reconstructed_completion,
                    float(trace_row["progress_min_completion"]),
                    rtol=2.0e-6,
                    atol=2.0e-7):
                raise RuntimeError(
                    "direction-candidate completion is not original-required"
                )
        elif progress_actual is not None or progress_required is not None:
            raise RuntimeError(
                "no-progress candidate has fabricated actual/required values"
            )
        if int(trace_row["selected"]) == 1:
            if int(trace_row["valid"]) != 1:
                raise RuntimeError("selected direction candidate is invalid")
            if not np.isclose(
                    float(trace_row["progress_floor_fraction"]),
                    float(transaction_row[
                        "pair_boundary_selected_progress_floor_fraction"
                    ]),
                    rtol=0.0,
                    atol=1.0e-12):
                raise RuntimeError(
                    "selected direction fraction diverges from transaction"
                )
        row = {
            "run_id": transaction_row["run_id"],
            "env_step": transaction_row["env_step"],
            "adjacency_update_round":
                transaction_row["adjacency_update_round"],
            "ppo_epoch_index": transaction_row["ppo_epoch_index"],
            "policy_id": transaction_row["policy_id"],
            "partition_index": transaction_row["partition_index"],
            "transaction_sequence_index":
                transaction_row["transaction_sequence_index"],
            "optimizer_kind": transaction_row["optimizer_kind"],
        }
        row.update(trace_row)
        if tuple(row.keys()) != _PAIR_DIRECTION_CANDIDATE_CSV_FIELDS:
            raise RuntimeError("direction-candidate CSV row schema diverged")
        output_rows.append(row)
    return output_rows


def _build_pair_selection_boundary_rows(transaction_row, trace_rows):
    """Attach real selection-boundary target traces to one Adam transaction."""
    trace_rows = list(trace_rows)
    if not trace_rows:
        if float(transaction_row["nonzero_pair_transaction"]) != 0.0:
            raise RuntimeError(
                "nonzero strict-pair transaction has no boundary trace"
            )
        return []
    expected_count = int(round(
        float(transaction_row["positive_target_count"])
        + float(transaction_row["negative_target_count"])
    ))
    if len(trace_rows) != expected_count:
        raise RuntimeError(
            "pair boundary trace count does not match exact target count: "
            "{} vs {}".format(len(trace_rows), expected_count)
        )
    output_rows = []
    for sequence_index, trace_row in enumerate(trace_rows):
        if tuple(trace_row.keys()) != _PAIR_SELECTION_BOUNDARY_TRACE_FIELDS:
            raise RuntimeError(
                "pair selection-boundary trace schema diverged"
            )
        if int(trace_row["diagnostic_version"]) != int(
                _PAIR_SELECTION_BOUNDARY_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected pair selection-boundary diagnostic version"
            )
        if int(trace_row[
                "target_row_sequence_within_transaction"
        ]) != sequence_index:
            raise RuntimeError(
                "pair selection-boundary rows are missing or out of order"
            )
        if (
                int(trace_row["valid"]) != 1
                or int(trace_row["margin_direction_correct"]) != 1
                or int(trace_row["margin_direction_reverse"]) != 0
                or int(trace_row["margin_direction_zero"]) != 0
                or int(trace_row[
                    "linearized_budget_conservation_valid"
                ]) != 1):
            raise RuntimeError(
                "persisted pair selection-boundary target is not a strict, "
                "budget-conserving improvement"
            )
        row = {
            "run_id": transaction_row["run_id"],
            "env_step": transaction_row["env_step"],
            "adjacency_update_round":
                transaction_row["adjacency_update_round"],
            "ppo_epoch_index": transaction_row["ppo_epoch_index"],
            "policy_id": transaction_row["policy_id"],
            "partition_index": transaction_row["partition_index"],
            "transaction_sequence_index":
                transaction_row["transaction_sequence_index"],
            "optimizer_kind": transaction_row["optimizer_kind"],
        }
        row.update(trace_row)
        if tuple(row.keys()) != _PAIR_SELECTION_BOUNDARY_CSV_FIELDS:
            raise RuntimeError(
                "pair selection-boundary CSV schema diverged"
            )
        output_rows.append(row)
    return output_rows


def _build_pair_selection_boundary_retention_rows(
        transaction_row, trace_rows):
    """Attach exact-context crossing retention evidence to an ordinary step."""
    trace_rows = list(trace_rows)
    if not trace_rows:
        return []
    if str(transaction_row["optimizer_kind"]) != "standard_adam":
        raise RuntimeError(
            "crossing retention replay must follow an ordinary Adam update"
        )
    output_rows = []
    observed_keys = set()
    for trace_row in trace_rows:
        if tuple(trace_row.keys()) != (
                _PAIR_SELECTION_BOUNDARY_RETENTION_TRACE_FIELDS):
            raise RuntimeError(
                "pair selection-boundary retention trace schema diverged"
            )
        if int(trace_row["diagnostic_version"]) != int(
                _PAIR_SELECTION_BOUNDARY_RETENTION_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected selection-boundary retention diagnostic version"
            )
        age = int(trace_row["ordinary_update_age"])
        if age not in (1, 2):
            raise RuntimeError(
                "selection-boundary retention age is not one or two"
            )
        if str(trace_row["source_policy_id"]) != str(
                transaction_row["policy_id"]):
            raise RuntimeError(
                "selection-boundary retention changed policy identity"
            )
        if int(trace_row["source_transaction_sequence_index"]) >= int(
                transaction_row["transaction_sequence_index"]):
            raise RuntimeError(
                "selection-boundary retention source is not earlier"
            )
        context_hash = str(trace_row["selection_context_sha256"])
        if len(context_hash) != 64:
            raise RuntimeError(
                "selection-boundary retention context hash is invalid"
            )
        observation_key = (int(trace_row["observation_id"]), age)
        if observation_key in observed_keys:
            raise RuntimeError(
                "duplicate selection-boundary retention observation"
            )
        observed_keys.add(observation_key)
        context_valid = int(trace_row["context_valid"])
        current_fields = (
            trace_row["current_signed_margin"],
            trace_row["retained_progress_fraction"],
            trace_row["current_rank"],
            trace_row["current_active"],
        )
        if context_valid == 1:
            if any(value is None for value in current_fields):
                raise RuntimeError(
                    "valid retention context has incomplete current values"
                )
            if not all(np.isfinite(float(value)) for value in current_fields):
                raise RuntimeError(
                    "valid retention context has non-finite current values"
                )
        elif context_valid == 0:
            if any(value is not None for value in current_fields):
                raise RuntimeError(
                    "invalid retention context persisted synthetic values"
                )
        else:
            raise RuntimeError(
                "selection-boundary retention context flag is not binary"
            )
        protection_stopped = int(trace_row["protection_stopped"])
        protection_stop_reason = str(trace_row["protection_stop_reason"])
        protection_stop_clock = int(trace_row["protection_stop_clock"])
        if protection_stopped not in (0, 1):
            raise RuntimeError(
                "selection-boundary retention stop flag is not binary"
            )
        if protection_stopped == 0:
            if protection_stop_reason or protection_stop_clock != -1:
                raise RuntimeError(
                    "active selection-boundary retention has stop metadata"
                )
        elif (
                protection_stop_reason not in (
                    "context_invalid",
                    "incompatible_current_actionable_evidence",
                )
                or protection_stop_clock < 0):
            raise RuntimeError(
                "stopped selection-boundary retention has invalid metadata"
            )
        row = {
            "run_id": transaction_row["run_id"],
            "env_step": transaction_row["env_step"],
            "adjacency_update_round":
                transaction_row["adjacency_update_round"],
            "ppo_epoch_index": transaction_row["ppo_epoch_index"],
            "policy_id": transaction_row["policy_id"],
            "partition_index": transaction_row["partition_index"],
            "transaction_sequence_index":
                transaction_row["transaction_sequence_index"],
            "optimizer_kind": transaction_row["optimizer_kind"],
        }
        row.update(trace_row)
        if tuple(row.keys()) != (
                _PAIR_SELECTION_BOUNDARY_RETENTION_CSV_FIELDS):
            raise RuntimeError(
                "pair selection-boundary retention CSV schema diverged"
            )
        output_rows.append(row)
    return output_rows


def _build_pair_selection_boundary_retention_component_rows(
        transaction_row, trace_rows):
    """Persist exact-context, one-component production-state replays."""
    trace_rows = list(trace_rows)
    if not trace_rows:
        return []
    if str(transaction_row["optimizer_kind"]) != "standard_adam":
        raise RuntimeError(
            "selection-state component attribution requires ordinary Adam"
        )
    output_rows = []
    observed_keys = set()
    allowed_components = {
        "pair_credit_ema",
        "triplet_credit_ema",
        "order3_credit_loss_ema",
        "order3_credit_margin_ema",
        "current_order3_credit_gate",
        "all_continuous",
    }
    for trace_row in trace_rows:
        if tuple(trace_row.keys()) != (
                _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_TRACE_FIELDS):
            raise RuntimeError(
                "selection-state component attribution schema diverged"
            )
        if int(trace_row["diagnostic_version"]) != int(
                _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected selection-state component diagnostic version"
            )
        component = str(trace_row["component"])
        if component not in allowed_components:
            raise RuntimeError(
                "selection-state attribution component is unsupported"
            )
        observed_key = (int(trace_row["observation_id"]), component)
        if observed_key in observed_keys:
            raise RuntimeError(
                "duplicate selection-state component attribution row"
            )
        observed_keys.add(observed_key)
        if (
                not np.isfinite(float(trace_row["component_delta_norm"]))
                or float(trace_row["component_delta_norm"]) < 0.0):
            raise RuntimeError(
                "selection-state component delta norm is invalid"
            )
        baseline_context_valid = int(trace_row["baseline_context_valid"])
        component_context_valid = int(trace_row["component_context_valid"])
        context_valid = int(trace_row["context_valid"])
        baseline_fields = (
            trace_row["baseline_signed_margin"],
            trace_row["baseline_rank"],
            trace_row["baseline_active"],
        )
        component_fields = (
            trace_row["component_signed_margin"],
            trace_row["component_rank"],
            trace_row["component_active"],
        )
        for valid_flag, values, label in (
                (baseline_context_valid, baseline_fields, "baseline"),
                (component_context_valid, component_fields, "component")):
            if valid_flag == 1:
                if any(value is None for value in values):
                    raise RuntimeError(
                        "valid {} attribution has missing values".format(
                            label
                        )
                    )
                if not all(np.isfinite(float(value)) for value in values):
                    raise RuntimeError(
                        "valid {} attribution is non-finite".format(label)
                    )
            elif valid_flag == 0:
                if any(value is not None for value in values):
                    raise RuntimeError(
                        "invalid {} context persisted synthetic values".format(
                            label
                        )
                    )
            else:
                raise RuntimeError(
                    "selection-state component context flag is not binary"
                )
        if context_valid != int(
                baseline_context_valid == 1
                and component_context_valid == 1):
            raise RuntimeError(
                "combined selection-state component context flag diverged"
            )
        if context_valid == 1:
            if trace_row["signed_margin_delta"] is None or not np.isfinite(
                    float(trace_row["signed_margin_delta"])):
                raise RuntimeError(
                    "valid component attribution delta is missing or non-finite"
                )
        elif trace_row["signed_margin_delta"] is not None:
            raise RuntimeError(
                "invalid component context persisted a synthetic delta"
            )
        for binary_field in (
                "competitor_changed",
                "baseline_context_valid",
                "component_context_valid",
                "context_valid",
                "floor_retained",
                "rank_retained",
                "active_retained",
                "joint_objectives_valid"):
            if int(trace_row[binary_field]) not in (0, 1):
                raise RuntimeError(
                    "selection-state component binary field is invalid"
                )
        row = {
            "run_id": transaction_row["run_id"],
            "env_step": transaction_row["env_step"],
            "adjacency_update_round":
                transaction_row["adjacency_update_round"],
            "ppo_epoch_index": transaction_row["ppo_epoch_index"],
            "policy_id": transaction_row["policy_id"],
            "partition_index": transaction_row["partition_index"],
            "transaction_sequence_index":
                transaction_row["transaction_sequence_index"],
            "optimizer_kind": transaction_row["optimizer_kind"],
        }
        row.update(trace_row)
        if tuple(row.keys()) != (
                _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_CSV_FIELDS):
            raise RuntimeError(
                "selection-state component CSV schema diverged"
            )
        output_rows.append(row)
    return output_rows


def _build_pair_selection_boundary_policy_response_rows(
        transaction_row, trace_rows):
    """Persist exact-context active-structure policy counterfactuals."""
    trace_rows = list(trace_rows)
    if not trace_rows:
        return []
    crossing_count = int(round(float(
        transaction_row["pair_boundary_rank_crossing_count"]
    )))
    if len(trace_rows) != crossing_count:
        raise RuntimeError(
            "policy counterfactual count does not match boundary crossings"
        )
    output_rows = []
    observed_targets = set()
    for trace_row in trace_rows:
        if tuple(trace_row.keys()) != (
                _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_TRACE_FIELDS):
            raise RuntimeError(
                "selection-boundary policy response schema diverged"
            )
        if int(trace_row["diagnostic_version"]) != int(
                _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_DIAGNOSTIC_VERSION):
            raise RuntimeError(
                "unexpected selection-boundary policy response version"
            )
        target_key = (
            int(trace_row["transition_index_in_partition"]),
            int(trace_row["factor_index"]),
            int(trace_row["target_candidate_index"]),
        )
        if target_key in observed_targets:
            raise RuntimeError(
                "duplicate selection-boundary policy counterfactual"
            )
        observed_targets.add(target_key)
        crossing_kind = str(trace_row["crossing_kind"])
        if crossing_kind not in ("promotion", "eviction"):
            raise RuntimeError(
                "selection-boundary policy crossing kind is invalid"
            )
        for digest_field in (
                "policy_context_sha256", "policy_state_sha256"):
            if len(str(trace_row[digest_field])) != 64:
                raise RuntimeError(
                    "selection-boundary policy digest is invalid"
                )
        for binary_field in (
                "factor_q_comparable",
                "policy_response_nonzero",
                "rng_neutral",
                "state_neutral",
                "valid"):
            if int(trace_row[binary_field]) not in (0, 1):
                raise RuntimeError(
                    "selection-boundary policy binary field is invalid"
                )
        if (
                int(trace_row["valid"]) != 1
                or int(trace_row["rng_neutral"]) != 1
                or int(trace_row["state_neutral"]) != 1):
            raise RuntimeError(
                "selection-boundary policy counterfactual is not neutral"
            )
        if int(trace_row["available_action_count"]) < 1:
            raise RuntimeError(
                "selection-boundary policy context has no available action"
            )
        nonnegative_fields = (
            "structure_input_diff_norm",
            "observation_input_diff_norm",
            "rnn_state_input_diff_norm",
            "pre_factor_q_norm",
            "post_factor_q_norm",
            "greedy_action_changed_count",
            "greedy_action_changed_fraction",
        )
        if any(
                not np.isfinite(float(trace_row[field]))
                or float(trace_row[field]) < 0.0
                for field in nonnegative_fields):
            raise RuntimeError(
                "selection-boundary policy response magnitude is invalid"
            )
        if float(trace_row["structure_input_diff_norm"]) <= 0.0:
            raise RuntimeError(
                "policy counterfactual did not change active structure"
            )
        if (
                float(trace_row["observation_input_diff_norm"]) != 0.0
                or float(trace_row["rnn_state_input_diff_norm"]) != 0.0):
            raise RuntimeError(
                "policy counterfactual changed non-structural input"
            )
        finite_fields = (
            "pre_best_value",
            "post_best_value",
            "best_value_delta",
            "pre_selected_factor_value",
            "post_selected_factor_value",
            "selected_factor_value_delta",
        )
        if not all(np.isfinite(float(trace_row[field]))
                   for field in finite_fields):
            raise RuntimeError(
                "selection-boundary policy response is non-finite"
            )
        comparable = int(trace_row["factor_q_comparable"])
        if comparable == 1:
            if (
                    trace_row["factor_q_diff_norm"] is None
                    or not np.isfinite(float(
                        trace_row["factor_q_diff_norm"]
                    ))
                    or float(trace_row["factor_q_diff_norm"]) < 0.0):
                raise RuntimeError(
                    "comparable factor-Q response has an invalid delta"
                )
        elif trace_row["factor_q_diff_norm"] is not None:
            raise RuntimeError(
                "incomparable factor-Q response persisted a synthetic delta"
            )
        changed_count = int(trace_row["greedy_action_changed_count"])
        changed_fraction = float(
            trace_row["greedy_action_changed_fraction"]
        )
        if (
                changed_count < 0
                or changed_fraction < 0.0
                or changed_fraction > 1.0):
            raise RuntimeError(
                "selection-boundary greedy action delta is invalid"
            )
        row = {
            "run_id": transaction_row["run_id"],
            "env_step": transaction_row["env_step"],
            "adjacency_update_round":
                transaction_row["adjacency_update_round"],
            "ppo_epoch_index": transaction_row["ppo_epoch_index"],
            "policy_id": transaction_row["policy_id"],
            "partition_index": transaction_row["partition_index"],
            "transaction_sequence_index":
                transaction_row["transaction_sequence_index"],
            "optimizer_kind": transaction_row["optimizer_kind"],
        }
        row.update(trace_row)
        if tuple(row.keys()) != (
                _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_CSV_FIELDS):
            raise RuntimeError(
                "selection-boundary policy response CSV schema diverged"
            )
        output_rows.append(row)
    return output_rows


def _build_sddfg_checkpoint_metadata(
        total_env_steps,
        target_env_steps,
        checkpoint_kind):
    """Build fail-loud metadata for one coherent SDDFG checkpoint."""
    total_env_steps = int(total_env_steps)
    target_env_steps = int(target_env_steps)
    checkpoint_kind = str(checkpoint_kind)
    if checkpoint_kind not in _SDDFG_CHECKPOINT_KINDS:
        raise RuntimeError(
            "unsupported SDDFG checkpoint kind: {}".format(
                checkpoint_kind
            )
        )
    if total_env_steps < 0 or target_env_steps <= 0:
        raise RuntimeError("invalid SDDFG checkpoint step metadata")
    training_complete = checkpoint_kind == "terminal"
    if training_complete and total_env_steps < target_env_steps:
        raise RuntimeError(
            "terminal SDDFG checkpoint precedes the target training step"
        )
    return {
        "runner_checkpoint_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "training_complete": training_complete,
        "total_env_steps": total_env_steps,
        "target_env_steps": target_env_steps,
    }


def _validate_sddfg_checkpoint_metadata(checkpoint):
    """Reject partial, ambiguous, or internally inconsistent checkpoints."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError("SDDFG checkpoint must be a dict")
    required = (
        "runner_checkpoint_version",
        "checkpoint_kind",
        "training_complete",
        "total_env_steps",
        "target_env_steps",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(
            "SDDFG checkpoint is missing metadata: {}".format(
                ", ".join(missing)
            )
        )
    if int(checkpoint["runner_checkpoint_version"]) != 1:
        raise RuntimeError("unsupported SDDFG runner checkpoint version")
    expected = _build_sddfg_checkpoint_metadata(
        total_env_steps=checkpoint["total_env_steps"],
        target_env_steps=checkpoint["target_env_steps"],
        checkpoint_kind=checkpoint["checkpoint_kind"],
    )
    if bool(checkpoint["training_complete"]) != bool(
            expected["training_complete"]):
        raise RuntimeError(
            "SDDFG checkpoint completion flag disagrees with its kind"
        )
    return expected


def _build_sddfg_q_target_checkpoint_contract(args):
    """Persist the temporal-credit horizon as part of SDDFG semantics."""
    if args is None:
        return {"sddfg_q_target_checkpoint_version": 0}
    if str(getattr(args, "algorithm_name", "")) != "sddfg":
        return {"sddfg_q_target_checkpoint_version": 0}
    q_n_step = int(getattr(args, "q_n_step", 1))
    episode_length = int(getattr(args, "episode_length", 0))
    if q_n_step <= 0 or episode_length <= 0 or q_n_step > episode_length:
        raise RuntimeError("SDDFG checkpoint has an invalid q_n_step contract")
    terminal_replay_lane = bool(getattr(
        args, "q_terminal_replay_lane", False
    ))
    terminal_replay_loss_weight = float(getattr(
        args, "q_terminal_replay_loss_weight", 0.10
    ))
    if (
            not np.isfinite(terminal_replay_loss_weight)
            or terminal_replay_loss_weight <= 0.0
            or terminal_replay_loss_weight > 1.0):
        raise RuntimeError(
            "SDDFG checkpoint has an invalid terminal replay loss weight"
        )
    return {
        "sddfg_q_target_checkpoint_version": 5,
        "sddfg_q_n_step": q_n_step,
        "sddfg_q_n_step_mode": (
            "one_step" if q_n_step == 1 else "terminal_gated"
        ),
        "sddfg_q_terminal_replay_lane": terminal_replay_lane,
        "sddfg_q_terminal_replay_loss_weight": (
            terminal_replay_loss_weight if terminal_replay_lane else 0.0
        ),
        "sddfg_q_frontier_target_alignment": bool(getattr(
            args,
            "pre_capture_visible_prey_quorum_greedy_frontier_guard",
            False,
        )),
    }


def _validate_sddfg_q_target_checkpoint_contract(checkpoint, args):
    """Fail loud rather than resume with a different TD-return horizon."""
    version = int(checkpoint.get("sddfg_q_target_checkpoint_version", 0))
    is_sddfg = str(getattr(args, "algorithm_name", "")) == "sddfg"
    if is_sddfg and version != 5:
        raise RuntimeError(
            "SDDFG resume requires the version-5 q-target checkpoint "
            "contract; start a fresh run"
        )
    if version not in (0, 5):
        raise RuntimeError("unsupported SDDFG q-target checkpoint version")
    if version == 0:
        return
    stored_n_step = int(checkpoint.get("sddfg_q_n_step", 0))
    current_n_step = int(getattr(args, "q_n_step", 1))
    if stored_n_step <= 0 or stored_n_step != current_n_step:
        raise RuntimeError(
            "checkpoint SDDFG q_n_step does not match the current run"
        )
    stored_mode = str(checkpoint.get("sddfg_q_n_step_mode", ""))
    expected_mode = "one_step" if current_n_step == 1 else "terminal_gated"
    if stored_mode != expected_mode:
        raise RuntimeError(
            "checkpoint SDDFG q-target mode does not match the current run"
        )
    stored_terminal_replay_lane = bool(checkpoint.get(
        "sddfg_q_terminal_replay_lane", False
    ))
    current_terminal_replay_lane = bool(getattr(
        args, "q_terminal_replay_lane", False
    ))
    if stored_terminal_replay_lane != current_terminal_replay_lane:
        raise RuntimeError(
            "checkpoint SDDFG terminal replay lane does not match the "
            "current run"
        )
    stored_terminal_replay_loss_weight = float(checkpoint.get(
        "sddfg_q_terminal_replay_loss_weight", float("nan")
    ))
    current_terminal_replay_loss_weight = float(getattr(
        args, "q_terminal_replay_loss_weight", 0.10
    ))
    expected_terminal_replay_loss_weight = (
        current_terminal_replay_loss_weight
        if current_terminal_replay_lane else 0.0
    )
    if (
            not np.isfinite(stored_terminal_replay_loss_weight)
            or not np.isclose(
                stored_terminal_replay_loss_weight,
                expected_terminal_replay_loss_weight,
                rtol=0.0,
                atol=1e-12,
            )):
        raise RuntimeError(
            "checkpoint SDDFG terminal replay loss weight does not match "
            "the current run"
        )
    stored_frontier_target_alignment = bool(checkpoint.get(
        "sddfg_q_frontier_target_alignment", False
    ))
    current_frontier_target_alignment = bool(getattr(
        args,
        "pre_capture_visible_prey_quorum_greedy_frontier_guard",
        False,
    ))
    if (
            stored_frontier_target_alignment
            != current_frontier_target_alignment):
        raise RuntimeError(
            "checkpoint SDDFG frontier target alignment does not match "
            "the current run"
        )


def _build_wolfpack_reward_shaping_checkpoint_contract(args):
    """Persist the reward potential that defines Wolfpack policy targets."""
    if (
            args is None
            or str(getattr(args, "env_name", "")).strip().lower()
            != "wolfpack"):
        return {"wolfpack_reward_shaping_checkpoint_version": 0}
    coverage_enabled = bool(getattr(
        args,
        "use_multi_prey_coverage_shaping",
        False,
    ))
    return {
        "wolfpack_reward_shaping_checkpoint_version": (
            WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION
        ),
        "wolfpack_reward_shaping_mode": (
            WOLFPACK_MULTI_PREY_COVERAGE_MODE
            if coverage_enabled else WOLFPACK_LEGACY_DISTANCE_MODE
        ),
        "wolfpack_reward_shaping_scale": WOLFPACK_DISTANCE_SHAPING_SCALE,
    }


def _validate_wolfpack_reward_shaping_checkpoint_contract(checkpoint, args):
    """Reject resume when the reward potential would silently change."""
    is_wolfpack = (
        str(getattr(args, "env_name", "")).strip().lower() == "wolfpack"
    )
    coverage_enabled = bool(getattr(
        args,
        "use_multi_prey_coverage_shaping",
        False,
    ))
    version = int(checkpoint.get(
        "wolfpack_reward_shaping_checkpoint_version",
        0,
    ))
    if not is_wolfpack:
        if version not in (0, WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION):
            raise RuntimeError(
                "unsupported Wolfpack reward-shaping checkpoint version"
            )
        return
    if coverage_enabled and version != WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION:
        raise RuntimeError(
            "balanced Wolfpack multi-prey coverage reward requires the "
            "versioned reward-shaping checkpoint contract; start a fresh run"
        )
    if version == 0:
        # Old checkpoints unambiguously used the legacy nearest-prey potential.
        return
    if version != WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION:
        raise RuntimeError(
            "unsupported Wolfpack reward-shaping checkpoint version"
        )
    expected = _build_wolfpack_reward_shaping_checkpoint_contract(args)
    stored_mode = str(checkpoint.get("wolfpack_reward_shaping_mode", ""))
    if stored_mode != expected["wolfpack_reward_shaping_mode"]:
        raise RuntimeError(
            "checkpoint Wolfpack reward-shaping mode does not match the "
            "current run"
        )
    stored_scale = float(checkpoint.get(
        "wolfpack_reward_shaping_scale",
        float("nan"),
    ))
    if (
            not np.isfinite(stored_scale)
            or not np.isclose(
                stored_scale,
                expected["wolfpack_reward_shaping_scale"],
                rtol=0.0,
                atol=1e-12,
            )):
        raise RuntimeError(
            "checkpoint Wolfpack reward-shaping scale does not match the "
            "current run"
        )


def _append_sddfg_terminal_win_provenance_to_sample(
        buffer,
        policy_ids,
        sample):
    """Attach the exact sampled win markers without sampling again."""
    terminal_win_reward_batch = {}
    terminal_replay_lane_episode_mask_batch = {}
    for policy_id in policy_ids:
        policy_buffer = buffer.policy_buffers[policy_id]
        reward_sample = getattr(
            policy_buffer,
            "last_reward_sample_diagnostics",
            None,
        )
        if reward_sample is None:
            raise RuntimeError(
                "SDDFG replay did not expose terminal-win provenance for "
                "Q-target construction"
            )
        terminal_win_rewards = np.asarray(
            reward_sample["terminal_win_rewards"],
            dtype=np.float32,
        )
        if terminal_win_rewards.ndim != 3 or terminal_win_rewards.shape[-1] != 1:
            raise RuntimeError(
                "SDDFG sampled terminal-win provenance has invalid shape"
            )
        if not np.isfinite(terminal_win_rewards).all():
            raise FloatingPointError(
                "non-finite sampled SDDFG terminal-win provenance"
            )
        terminal_win_reward_batch[policy_id] = terminal_win_rewards.copy()
        terminal_replay_lane_episode_mask = np.asarray(
            reward_sample.get(
                "terminal_replay_lane_episode_mask",
                np.zeros(terminal_win_rewards.shape[0], dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        if terminal_replay_lane_episode_mask.shape[0] != terminal_win_rewards.shape[0]:
            raise RuntimeError(
                "SDDFG terminal replay lane mask does not match the sample"
            )
        terminal_replay_lane_episode_mask_batch[policy_id] = (
            terminal_replay_lane_episode_mask.copy()
        )
    return tuple(sample) + (
        terminal_win_reward_batch,
        terminal_replay_lane_episode_mask_batch,
    )


def _validate_policy_exploration_checkpoint_contract(checkpoint, args):
    """Fail loud when resume would change exploration RNG or schedule."""
    exploration_version = int(checkpoint.get(
        "policy_exploration_checkpoint_version",
        0,
    ))
    joint_exploration_enabled = bool(getattr(
        args,
        "use_joint_epsilon_exploration",
        False,
    ))
    post_capture_joint_greedy_floor = float(getattr(
        args,
        "post_capture_joint_greedy_floor",
        0.0,
    ))
    post_capture_explore_max_random_agents = int(getattr(
        args,
        "post_capture_explore_max_random_agents",
        0,
    ))
    pre_capture_visible_prey_quorum_guard = bool(getattr(
        args,
        "pre_capture_visible_prey_quorum_guard",
        False,
    ))
    pre_capture_visible_prey_quorum_greedy_frontier_guard = bool(getattr(
        args,
        "pre_capture_visible_prey_quorum_greedy_frontier_guard",
        False,
    ))
    if (
            joint_exploration_enabled
            and exploration_version not in (2, 3, 4, 5, 6)):
        raise RuntimeError(
            "joint epsilon exploration requires the versioned policy RNG "
            "and epsilon-schedule checkpoint contract; start a fresh run"
        )
    if (
            joint_exploration_enabled
            and post_capture_joint_greedy_floor > 0.0
            and exploration_version not in (3, 4, 5, 6)):
        raise RuntimeError(
            "post-capture joint exploration requires the version-3+ policy "
            "exploration checkpoint contract; start a fresh run"
        )
    if (
            joint_exploration_enabled
            and post_capture_explore_max_random_agents > 0
            and exploration_version not in (4, 5, 6)):
        raise RuntimeError(
            "bounded post-capture exploration requires the version-4 policy "
            "exploration checkpoint contract; start a fresh run"
        )
    if (
            joint_exploration_enabled
            and pre_capture_visible_prey_quorum_guard
            and exploration_version not in (5, 6)):
        raise RuntimeError(
            "pre-capture visible-prey quorum guard requires the version-5 "
            "policy exploration checkpoint contract; start a fresh run"
        )
    if (
            pre_capture_visible_prey_quorum_greedy_frontier_guard
            and exploration_version != 6):
        raise RuntimeError(
            "pre-capture greedy frontier guard requires the version-6 "
            "policy exploration checkpoint contract; start a fresh run"
        )
    if exploration_version not in (0, 1, 2, 3, 4, 5, 6):
        raise RuntimeError(
            "unsupported policy exploration checkpoint version"
        )
    if exploration_version == 0:
        return {}

    stored_joint_exploration = bool(checkpoint.get(
        "joint_epsilon_exploration_enabled",
        False,
    ))
    if stored_joint_exploration != joint_exploration_enabled:
        raise RuntimeError(
            "checkpoint joint epsilon exploration contract does not match "
            "the current run"
        )
    if exploration_version >= 2:
        schedule_contract = (
            ("policy_epsilon_start", "epsilon_start", float),
            ("policy_epsilon_finish", "epsilon_finish", float),
            ("policy_epsilon_anneal_time", "epsilon_anneal_time", int),
        )
        for stored_key, argument_name, converter in schedule_contract:
            if stored_key not in checkpoint:
                raise RuntimeError(
                    "checkpoint policy exploration schedule is missing {}"
                    .format(stored_key)
                )
            stored_value = converter(checkpoint[stored_key])
            current_value = converter(getattr(args, argument_name))
            if stored_value != current_value:
                raise RuntimeError(
                    "checkpoint policy exploration schedule {} does not "
                    "match the current run".format(argument_name)
                )

    if exploration_version >= 3:
        stored_floor = float(checkpoint.get(
            "post_capture_joint_greedy_floor",
            float("nan"),
        ))
        if (
                not np.isfinite(stored_floor)
                or stored_floor != post_capture_joint_greedy_floor):
            raise RuntimeError(
                "checkpoint post-capture joint greedy floor does not match "
                "the current run"
            )
    if exploration_version >= 4:
        stored_max_random_agents = int(checkpoint.get(
            "post_capture_explore_max_random_agents",
            -1,
        ))
        if (
                stored_max_random_agents < 0
                or stored_max_random_agents
                != post_capture_explore_max_random_agents):
            raise RuntimeError(
                "checkpoint bounded post-capture exploration contract does "
                "not match the current run"
            )
    if exploration_version >= 5:
        stored_pre_capture_guard = bool(checkpoint.get(
            "pre_capture_visible_prey_quorum_guard",
            False,
        ))
        if (
                stored_pre_capture_guard
                != pre_capture_visible_prey_quorum_guard):
            raise RuntimeError(
                "checkpoint pre-capture visible-prey quorum guard contract "
                "does not match the current run"
            )
    if exploration_version >= 6:
        stored_frontier_guard = bool(checkpoint.get(
            "pre_capture_visible_prey_quorum_greedy_frontier_guard",
            False,
        ))
        if (
                stored_frontier_guard
                != pre_capture_visible_prey_quorum_greedy_frontier_guard):
            raise RuntimeError(
                "checkpoint pre-capture greedy frontier guard contract "
                "does not match the current run"
            )

    policy_rng_states = checkpoint.get("policy_rng_states", {})
    if not isinstance(policy_rng_states, dict):
        raise RuntimeError(
            "checkpoint policy exploration RNG state must be a dict"
        )
    return policy_rng_states


def _build_policy_exploration_checkpoint_contract(args, policies, policy_ids):
    """Build exploration metadata without weakening production checkpoints."""
    policy_ids = list(policy_ids or ())
    policies = policies or {}
    if args is None:
        if policy_ids or policies:
            raise RuntimeError(
                "policy exploration checkpoint requires runner args when "
                "policies are present"
            )
        # Model-save fixtures without policies have no exploration state.  Keep
        # their checkpoint explicitly legacy-empty instead of inventing a v2
        # epsilon schedule that cannot be validated on restore.
        return {"policy_exploration_checkpoint_version": 0}

    try:
        epsilon_start = float(getattr(args, "epsilon_start"))
        epsilon_finish = float(getattr(args, "epsilon_finish"))
        epsilon_anneal_time = int(getattr(args, "epsilon_anneal_time"))
        post_capture_joint_greedy_floor = float(getattr(
            args,
            "post_capture_joint_greedy_floor",
            0.0,
        ))
        post_capture_explore_max_random_agents = int(getattr(
            args,
            "post_capture_explore_max_random_agents",
            0,
        ))
        pre_capture_visible_prey_quorum_guard = bool(getattr(
            args,
            "pre_capture_visible_prey_quorum_guard",
            False,
        ))
        pre_capture_visible_prey_quorum_greedy_frontier_guard = bool(getattr(
            args,
            "pre_capture_visible_prey_quorum_greedy_frontier_guard",
            False,
        ))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "policy exploration checkpoint requires a valid epsilon schedule"
        ) from error
    if not (
            np.isfinite(epsilon_start)
            and np.isfinite(epsilon_finish)
            and 0.0 <= epsilon_finish <= epsilon_start <= 1.0
            and epsilon_anneal_time > 0
            and np.isfinite(post_capture_joint_greedy_floor)
            and 0.0 <= post_capture_joint_greedy_floor < 1.0
            and post_capture_explore_max_random_agents >= 0):
        raise RuntimeError(
            "policy exploration checkpoint has an invalid epsilon schedule"
        )
    if (
            pre_capture_visible_prey_quorum_guard
            and not bool(getattr(
                args,
                "use_joint_epsilon_exploration",
                False,
            ))):
        raise RuntimeError(
            "pre-capture visible-prey quorum guard checkpoint requires "
            "joint epsilon exploration"
        )

    policy_rng_states = {}
    for pid in policy_ids:
        if pid not in policies:
            raise RuntimeError(
                "policy exploration checkpoint is missing policy {}"
                .format(pid)
            )
        policy_rng = getattr(policies[pid], "rng", None)
        if policy_rng is None or not callable(getattr(policy_rng, "get_state", None)):
            raise RuntimeError(
                "policy exploration checkpoint is missing RNG state for {}"
                .format(pid)
            )
        policy_rng_states[str(pid)] = copy.deepcopy(policy_rng.get_state())

    return {
        "policy_exploration_checkpoint_version": 6,
        "joint_epsilon_exploration_enabled": bool(getattr(
            args,
            "use_joint_epsilon_exploration",
            False,
        )),
        "policy_epsilon_start": epsilon_start,
        "policy_epsilon_finish": epsilon_finish,
        "policy_epsilon_anneal_time": epsilon_anneal_time,
        "post_capture_joint_greedy_floor": (
            post_capture_joint_greedy_floor
        ),
        "post_capture_explore_max_random_agents": (
            post_capture_explore_max_random_agents
        ),
        "pre_capture_visible_prey_quorum_guard": (
            pre_capture_visible_prey_quorum_guard
        ),
        "pre_capture_visible_prey_quorum_greedy_frontier_guard": (
            pre_capture_visible_prey_quorum_greedy_frontier_guard
        ),
        "policy_rng_states": policy_rng_states,
    }


def _require_sddfg_optimizer_checkpoint(
        algorithm_name,
        optimizer_state_path):
    """Return checkpoint presence, but never silently degrade SDDFG restore."""
    exists = os.path.exists(optimizer_state_path)
    if str(algorithm_name) == "sddfg" and not exists:
        raise RuntimeError(
            "SDDFG restore requires adj_optimizer_state.pt with all Adam "
            "states and checkpoint metadata"
        )
    return exists

# ===== 修改点 1：新增 run 前缀工具函数 =====
def _get_run_csv_name(run_dir, filename):
    """
    将 run 目录下的 CSV 文件统一加 run 前缀。

    例如：
      run_dir = ".../run1"
      filename = "progress.csv"
      return = "run1_progress.csv"

    若 filename 已经带 run 前缀，则不重复添加。
    """
    filename = str(filename)

    if not filename.endswith(".csv"):
        return filename

    run_name = os.path.basename(str(run_dir).rstrip(os.sep))

    # 防御：如果 run_dir 不是 run1/run2 这种格式，则保持原名
    if not run_name.startswith("run"):
        return filename

    prefix = run_name + "_"
    if filename.startswith(prefix):
        return filename

    return prefix + filename

class RecRunner(object):
    """Base class for training recurrent policies."""

    # ===== 修改点 2：所有 base_runner scalar CSV 自动带 run 前缀 =====
    def _append_scalar_csv(self, filename, row_dict):
        filename = _get_run_csv_name(self.run_dir, filename)
        progress_filename = os.path.join(self.run_dir, filename)

        df = pd.DataFrame([row_dict])
        if not os.path.exists(progress_filename):
            df.to_csv(progress_filename, index=False)
            return

        # Scalar dictionaries can gain fields after warmup (for example,
        # policy losses do not exist before the first optimizer update). Keep
        # the CSV schema aligned instead of appending wider rows under an old
        # header. Rewriting happens only when a new column first appears.
        existing_columns = list(pd.read_csv(progress_filename, nrows=0).columns)
        new_columns = [column for column in df.columns if column not in existing_columns]
        all_columns = existing_columns + new_columns

        if new_columns:
            existing_df = pd.read_csv(progress_filename)
            existing_df.reindex(columns=all_columns).to_csv(progress_filename, index=False)

        df.reindex(columns=all_columns).to_csv(
            progress_filename,
            mode='a',
            header=False,
            index=False,
        )

    # 检查sddfg参数
    def _append_fixed_rows_csv(self, filename, rows):
        """Append a non-empty batch of fixed-schema diagnostic rows."""
        rows = list(rows)
        if not rows:
            return
        filename = _get_run_csv_name(self.run_dir, filename)
        progress_filename = os.path.join(self.run_dir, filename)
        fieldnames = tuple(rows[0].keys())
        if len(set(fieldnames)) != len(fieldnames):
            raise RuntimeError("diagnostic CSV row has duplicate columns")
        for row in rows:
            if tuple(row.keys()) != fieldnames:
                raise RuntimeError(
                    "fixed diagnostic rows have inconsistent schemas"
                )
        file_exists = os.path.exists(progress_filename)
        if os.path.exists(progress_filename):
            with open(progress_filename, "r", newline="") as csv_file:
                reader = csv.reader(csv_file)
                existing_columns = tuple(next(reader, ()))
            if existing_columns != fieldnames:
                raise RuntimeError(
                    "existing fixed diagnostic CSV has an incompatible schema"
                )
        with open(progress_filename, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _record_strict_pair_exact_failure(
            self,
            diagnostic_rows,
            adjacency_update_round,
            ppo_epoch_index,
            policy_id,
            partition_index):
        """Persist a rolled-back strict-pair failure before re-raising it."""
        diagnostic_rows = tuple(diagnostic_rows)
        if not diagnostic_rows:
            raise RuntimeError(
                "strict-pair exact failure omitted diagnostic rows"
            )
        runtime_fields = (
            "run_id",
            "env_step",
            "adjacency_update_round",
            "ppo_epoch_index",
            "policy_id",
            "partition_index",
            "transaction_sequence_index",
        )
        payload_fields = tuple(
            field for field in _STRICT_PAIR_EXACT_FAILURE_FIELDS
            if field not in runtime_fields
        )
        for row in diagnostic_rows:
            if set(row.keys()) != set(payload_fields):
                missing = sorted(set(payload_fields) - set(row.keys()))
                extra = sorted(set(row.keys()) - set(payload_fields))
                raise RuntimeError(
                    "strict-pair exact failure payload schema mismatch: "
                    "missing={}, extra={}".format(missing, extra)
                )
        persisted_rows = []
        for diagnostic in diagnostic_rows:
            combined = {
                "run_id": os.path.basename(
                    str(self.run_dir).rstrip(os.sep)
                ),
                "env_step": int(self.total_env_steps),
                "adjacency_update_round": int(adjacency_update_round),
                "ppo_epoch_index": int(ppo_epoch_index),
                "policy_id": str(policy_id),
                "partition_index": int(partition_index),
                "transaction_sequence_index": int(
                    self._adj_transaction_sequence_index
                ),
            }
            combined.update(diagnostic)
            ordered = {
                field: combined[field]
                for field in _STRICT_PAIR_EXACT_FAILURE_FIELDS
            }
            persisted_rows.append(ordered)
        self._append_fixed_rows_csv(
            _STRICT_PAIR_EXACT_FAILURE_CSV_BASENAME,
            persisted_rows,
        )

    def _ensure_adj_transaction_log_state(self):
        """Initialize the fixed-schema per-Adam transaction log."""
        if bool(getattr(
                self,
                "_adj_transaction_log_initialized",
                False)):
            return
        filename = _get_run_csv_name(
            self.run_dir,
            _ADJ_TRANSACTION_CSV_BASENAME,
        )
        path = os.path.join(str(self.run_dir), filename)
        existing_records = []
        existing_keys = set()
        if os.path.exists(path):
            with open(path, "r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                if tuple(reader.fieldnames or ()) != _ADJ_TRANSACTION_CSV_FIELDS:
                    raise RuntimeError(
                        "existing adjacency transaction CSV has an "
                        "incompatible schema"
                    )
                existing_records = list(reader)
            _validate_adj_transaction_update_records(
                existing_records,
                sequence_start=0,
            )
            existing_keys = set(
                (
                    int(float(record["env_step"])),
                    int(float(record["adjacency_update_round"])),
                    int(float(record["ppo_epoch_index"])),
                    str(record["policy_id"]),
                    int(float(record["partition_index"])),
                )
                for record in existing_records
            )
        self._adj_transaction_csv_path = path
        self._adj_transaction_sequence_index = len(existing_records)
        self._adj_transaction_keys = existing_keys
        self._adj_transaction_log_initialized = True

    def _append_adj_transaction_csv(self, row):
        """Append one real standard Adam step before any runner averaging."""
        self._ensure_adj_transaction_log_state()
        sequence_index = int(row["transaction_sequence_index"])
        if sequence_index != int(self._adj_transaction_sequence_index):
            raise RuntimeError(
                "adjacency transaction sequence is missing or out of order"
            )
        transaction_key = (
            int(row["env_step"]),
            int(row["adjacency_update_round"]),
            int(row["ppo_epoch_index"]),
            str(row["policy_id"]),
            int(row["partition_index"]),
        )
        if transaction_key in self._adj_transaction_keys:
            raise RuntimeError("duplicate adjacency optimizer transaction")
        file_exists = os.path.exists(self._adj_transaction_csv_path)
        with open(
                self._adj_transaction_csv_path,
                "a",
                newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(_ADJ_TRANSACTION_CSV_FIELDS),
                extrasaction="raise",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                field: row[field]
                for field in _ADJ_TRANSACTION_CSV_FIELDS
            })
        self._adj_transaction_keys.add(transaction_key)
        self._adj_transaction_sequence_index += 1

    def _ensure_candidate_evidence_provenance_log_state(self):
        """Load event-level evidence keys without touching training state."""
        if bool(getattr(
                self,
                "_candidate_evidence_provenance_log_initialized",
                False)):
            return
        filename = _get_run_csv_name(
            self.run_dir,
            _CANDIDATE_EVIDENCE_PROVENANCE_CSV_BASENAME,
        )
        path = os.path.join(str(self.run_dir), filename)
        payload_by_key = {}
        if os.path.exists(path):
            with open(path, "r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                if tuple(reader.fieldnames or ()) != (
                        _CANDIDATE_EVIDENCE_PROVENANCE_CSV_FIELDS):
                    raise RuntimeError(
                        "existing candidate evidence provenance CSV has an "
                        "incompatible schema"
                    )
                for record in reader:
                    key = _candidate_evidence_persisted_key(record)
                    if key in payload_by_key:
                        raise RuntimeError(
                            "candidate evidence provenance CSV contains a "
                            "duplicate event-identity key"
                        )
                    if (
                            int(float(record["consumed_generation_once"])) != 1
                            or int(float(record["duplicate_generation"])) != 0
                            or int(float(record["duplicate_event"])) != 0
                            or int(float(record["provenance_complete"])) != 1
                            or int(float(record["identity_contract_valid"])) != 1
                            or int(float(record["quality_contract_valid"])) != 1):
                        raise RuntimeError(
                            "persisted candidate evidence provenance contract "
                            "is invalid"
                        )
                    payload_by_key[key] = (
                        _candidate_evidence_immutable_payload(record)
                    )
        self._candidate_evidence_provenance_csv_path = path
        self._candidate_evidence_provenance_payload_by_key = payload_by_key
        self._candidate_evidence_provenance_log_initialized = True

    def _append_candidate_evidence_provenance_csv(self, rows):
        """Persist each real generation/event/identity evidence exactly once."""
        rows = list(rows)
        if not rows:
            return
        self._ensure_candidate_evidence_provenance_log_state()
        rows_to_write = []
        call_keys = set()
        for row in rows:
            if tuple(row.keys()) != (
                    _CANDIDATE_EVIDENCE_PROVENANCE_CSV_FIELDS):
                raise RuntimeError(
                    "candidate evidence provenance row schema diverged"
                )
            key = _candidate_evidence_persisted_key(row)
            event_identity_key = key[:-1]
            existing_signs = {
                existing_key[-1]
                for existing_key in (
                    list(
                        self._candidate_evidence_provenance_payload_by_key.keys()
                    )
                    + list(call_keys)
                )
                if existing_key[:-1] == event_identity_key
            }
            if existing_signs and key[-1] not in existing_signs:
                raise RuntimeError(
                    "candidate evidence sign changed for the same real "
                    "generation/event/identity"
                )
            if key in call_keys:
                raise RuntimeError(
                    "candidate evidence event-identity is duplicated inside "
                    "one optimizer transaction"
                )
            call_keys.add(key)
            payload = _candidate_evidence_immutable_payload(row)
            existing_payload = (
                self._candidate_evidence_provenance_payload_by_key.get(key)
            )
            if existing_payload is not None:
                if existing_payload != payload:
                    raise RuntimeError(
                        "replayed candidate evidence provenance changed across "
                        "PPO epochs or replay exposure"
                    )
                # A repeated PPO epoch/replay exposure is visibility of the
                # same evidence, not a second independent event.
                continue
            rows_to_write.append(row)
            self._candidate_evidence_provenance_payload_by_key[key] = payload
        if not rows_to_write:
            return
        file_exists = os.path.exists(
            self._candidate_evidence_provenance_csv_path
        )
        with open(
                self._candidate_evidence_provenance_csv_path,
                "a",
                newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(
                    _CANDIDATE_EVIDENCE_PROVENANCE_CSV_FIELDS
                ),
                extrasaction="raise",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_write)

    def _record_adj_transaction(
            self,
            train_adj_info,
            policy_buffer,
            sample,
            adjacency_update_round,
            ppo_epoch_index,
            policy_id,
            partition_index,
            selected_episode_count_override=None,
            selected_chunk_count_override=None,
            class_complete_override=None):
        """Persist one unaggregated standard adjacency optimizer transaction."""
        self._ensure_adj_transaction_log_state()
        candidate_trace_rows = train_adj_info.pop(
            "_candidate_identity_transaction_rows",
            [],
        )
        pair_selection_boundary_trace_rows = train_adj_info.pop(
            "_pair_selection_boundary_rows",
            [],
        )
        pair_direction_candidate_trace_rows = train_adj_info.pop(
            "_pair_direction_candidate_rows",
            [],
        )
        pair_selection_boundary_retention_trace_rows = train_adj_info.pop(
            "_pair_selection_boundary_retention_rows",
            [],
        )
        pair_selection_boundary_retention_component_trace_rows = (
            train_adj_info.pop(
                "_pair_selection_boundary_retention_component_rows",
                [],
            )
        )
        pair_selection_boundary_policy_response_trace_rows = (
            train_adj_info.pop(
                "_pair_selection_boundary_policy_response_rows",
                [],
            )
        )
        candidate_evidence_consumption_rows = train_adj_info.pop(
            "_candidate_evidence_consumption_rows",
            [],
        )
        candidate_evidence_draft_rows = list(getattr(
            policy_buffer,
            "last_sample_candidate_evidence_provenance_rows",
            (),
        ))
        joined_candidate_evidence_rows = (
            _join_candidate_evidence_consumption_rows(
                consumption_rows=candidate_evidence_consumption_rows,
                draft_rows=candidate_evidence_draft_rows,
            )
        )
        candidate_event_by_target = _candidate_event_by_target_key(
            joined_candidate_evidence_rows
        )
        selected_episode_count = (
            selected_episode_count_override
            if selected_episode_count_override is not None
            else getattr(
                policy_buffer,
                "last_sample_episode_count",
                np.nan,
            )
        )
        selected_chunk_count = (
            selected_chunk_count_override
            if selected_chunk_count_override is not None
            else getattr(
                policy_buffer,
                "last_sample_selected_chunk_count",
                np.nan,
            )
        )
        transaction_chunk_count = int(np.asarray(sample[0]).shape[0])
        row = _build_adj_transaction_row(
            run_id=os.path.basename(str(self.run_dir).rstrip(os.sep)),
            env_step=self.total_env_steps,
            adjacency_update_round=adjacency_update_round,
            ppo_epoch_index=ppo_epoch_index,
            policy_id=policy_id,
            partition_index=partition_index,
            transaction_sequence_index=(
                self._adj_transaction_sequence_index
            ),
            selected_episode_count=selected_episode_count,
            selected_chunk_count=selected_chunk_count,
            transaction_chunk_count=transaction_chunk_count,
            train_adj_info=train_adj_info,
        )
        class_complete_from_buffer = (
            float(class_complete_override)
            if class_complete_override is not None
            else float(getattr(
                policy_buffer,
                "last_sample_pair_optimizer_atomic_partition",
                np.nan,
            ))
        )
        if (
                not np.isfinite(class_complete_from_buffer)
                or class_complete_from_buffer != row["class_complete"]):
            raise RuntimeError(
                "runner/trainer class-complete transaction flags diverged"
            )
        candidate_transaction_rows = (
            _build_candidate_identity_transaction_rows(
                transaction_row=row,
                trace_rows=candidate_trace_rows,
                episode_rows=list(getattr(
                    policy_buffer,
                    "last_sample_pair_evidence_episode_rows",
                    (),
                )),
                train_adj_info=train_adj_info,
                candidate_event_by_target=candidate_event_by_target,
            )
        )
        pair_selection_boundary_rows = (
            _build_pair_selection_boundary_rows(
                transaction_row=row,
                trace_rows=pair_selection_boundary_trace_rows,
            )
        )
        pair_direction_candidate_rows = (
            _build_pair_direction_candidate_rows(
                transaction_row=row,
                trace_rows=pair_direction_candidate_trace_rows,
            )
        )
        pair_selection_boundary_retention_rows = (
            _build_pair_selection_boundary_retention_rows(
                transaction_row=row,
                trace_rows=(
                    pair_selection_boundary_retention_trace_rows
                ),
            )
        )
        pair_selection_boundary_retention_component_rows = (
            _build_pair_selection_boundary_retention_component_rows(
                transaction_row=row,
                trace_rows=(
                    pair_selection_boundary_retention_component_trace_rows
                ),
            )
        )
        pair_selection_boundary_policy_response_rows = (
            _build_pair_selection_boundary_policy_response_rows(
                transaction_row=row,
                trace_rows=(
                    pair_selection_boundary_policy_response_trace_rows
                ),
            )
        )
        candidate_evidence_provenance_rows = (
            _build_candidate_evidence_provenance_csv_rows(
                joined_evidence_rows=joined_candidate_evidence_rows,
                run_id=row["run_id"],
                env_step=row["env_step"],
                policy_id=row["policy_id"],
                current_candidate_policy_version=train_adj_info.get(
                    "capture_candidate_identity_policy_version",
                    0.0,
                ),
            )
        )
        self._append_adj_transaction_csv(row)
        self._append_fixed_rows_csv(
            _CANDIDATE_IDENTITY_TRANSACTION_CSV_BASENAME,
            candidate_transaction_rows,
        )
        self._append_fixed_rows_csv(
            _PAIR_SELECTION_BOUNDARY_CSV_BASENAME,
            pair_selection_boundary_rows,
        )
        self._append_fixed_rows_csv(
            _PAIR_DIRECTION_CANDIDATE_CSV_BASENAME,
            pair_direction_candidate_rows,
        )
        self._append_fixed_rows_csv(
            _PAIR_SELECTION_BOUNDARY_RETENTION_CSV_BASENAME,
            pair_selection_boundary_retention_rows,
        )
        self._append_fixed_rows_csv(
            _PAIR_SELECTION_BOUNDARY_RETENTION_COMPONENT_CSV_BASENAME,
            pair_selection_boundary_retention_component_rows,
        )
        self._append_fixed_rows_csv(
            _PAIR_SELECTION_BOUNDARY_POLICY_RESPONSE_CSV_BASENAME,
            pair_selection_boundary_policy_response_rows,
        )
        self._append_candidate_evidence_provenance_csv(
            candidate_evidence_provenance_rows
        )
        return row

    def _validate_dynamic_graph_args(self):
        if getattr(self.args, "algorithm_name", None) == "sddfg":
            if self.hidden_size % max(1, int(self.args.gat_heads)) != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} must be divisible by gat_heads={self.args.gat_heads} "
                    "to avoid silent GAT head dimension truncation."
                )
            if bool(getattr(self.args, "require_connected_adj", False)):
                max_agents = int(getattr(self.args, "max_player_num", 1))
                max_order = max(2, int(getattr(self.args, "highest_orders", 2)))
                min_connected_factors = (
                    max_agents - 1 + max_order - 2
                ) // (max_order - 1)
                if int(getattr(self.args, "num_factor", 0)) < min_connected_factors:
                    raise ValueError(
                        "require_connected_adj needs at least {} factors for "
                        "max_player_num={} and highest_orders={}, got {}.".format(
                            min_connected_factors,
                            max_agents,
                            max_order,
                            getattr(self.args, "num_factor", 0),
                        )
                    )
            pair_pending_enabled = bool(getattr(
                self.args,
                "pair_bounded_pending_evidence",
                False,
            ))
            pair_pending_horizon = int(getattr(
                self.args,
                "pair_pending_max_adj_updates",
                0,
            ))
            if pair_pending_enabled:
                if pair_pending_horizon <= 0:
                    raise ValueError(
                        "enabled pair bounded pending requires a positive "
                        "adjacency-update horizon"
                    )
                if not bool(getattr(
                        self.args,
                        "use_adj_pair_triplet_complementary_credit",
                        False,
                )):
                    raise ValueError(
                        "pair bounded pending requires strict pair credit"
                    )
                if not bool(getattr(
                        self.args,
                        "use_adj_capture_to_win_credit",
                        False,
                )):
                    raise ValueError(
                        "pair bounded pending requires event-local capture "
                        "identity provenance"
                    )
            elif pair_pending_horizon != 0:
                raise ValueError(
                    "pair_pending_max_adj_updates must be zero while the "
                    "experiment is disabled"
                )

        min_adj_begin_step = int(getattr(self.args, "min_adj_begin_step", 5000))
        adj_begin_step = int(getattr(self.args, "adj_begin_step", 0))
        adj_lr = float(getattr(self.args, "adj_lr", 5e-4))

        if self.use_dyn_graph and adj_begin_step < min_adj_begin_step:
            print(
                f"[WARN] adj_begin_step={adj_begin_step} is very early for dynamic-agent training. "
                f"Recommended >= {min_adj_begin_step}."
            )

        if self.use_dyn_graph and adj_lr > 1e-3:
            print(
                f"[WARN] adj_lr={adj_lr} is high for PPO-style adj/GAT updates. "
                "Recommended 5e-4 ~ 1e-3 for dynamic wolfpack."
            )

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        # 将传入的配置保存到实例变量，方便后续使用
        self.args = config["args"]
        self.device = config["device"]
        self.adj = config["adj"]

        # 一些算法分组（字符串列表），用于后续根据 algorithm_name 决定行为分支
        # q_learning 列表里是基于 Q-learning 的算法（例如 QMix、VDN 等）
        self.q_learning = ["qplex", "qtran", "wqmix", "qmix", "vdn", "ddfg", "sopcg", "casec", "sddfg"]
        # adj_correlation 列表表示需要关联邻接（adjacency）网络的算法
        self.adj_correlation = ["ddfg", "sopcg", "casec", "sddfg"]

        # —— 从args抄出各种常见训练/环境配置，便于后续使用 ——
        self.share_policy = self.args.share_policy  # 是否共享策略（多agent同一套参数）
        self.algorithm_name = self.args.algorithm_name  # 算法名（决定采用何种Policy/Trainer）
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps  # 训练目标的总交互步数
        self.use_wandb = self.args.use_wandb  # 是否用wandb记录日志
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per  # 是否使用优先经验回放
        self.q_terminal_replay_lane = bool(getattr(
            self.args, "q_terminal_replay_lane", False
        ))
        self.q_terminal_replay_loss_weight = float(getattr(
            self.args, "q_terminal_replay_loss_weight", 0.10
        ))
        if (
                not np.isfinite(self.q_terminal_replay_loss_weight)
                or self.q_terminal_replay_loss_weight <= 0.0
                or self.q_terminal_replay_loss_weight > 1.0):
            raise RuntimeError(
                "terminal replay loss weight must be finite in (0, 1]"
            )
        if self.q_terminal_replay_lane:
            if self.algorithm_name != "sddfg":
                raise RuntimeError(
                    "terminal replay lane is supported only by SDDFG"
                )
            if self.use_per:
                raise RuntimeError(
                    "terminal replay lane cannot be combined with PER"
                )
            if int(getattr(self.args, "q_n_step", 1)) <= 1:
                raise RuntimeError(
                    "terminal replay lane requires terminal-gated n-step Q"
                )
        self.q_terminal_replay_lane_forced_count = 0
        self.q_terminal_replay_lane_gated_transition_count = 0.0
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size  # 轨迹回放缓冲区大小
        self.batch_size = self.args.batch_size
        self.adj_buffer_size = self.args.adj_buffer_size  # 邻接图训练的缓冲区大小
        self.hidden_size = self.args.hidden_size  # RNN/网络隐藏维度
        self.highest_orders = self.args.highest_orders  # 因子图最高阶（如1/2/3阶）
        self.use_soft_update = self.args.use_soft_update  # 目标网络软更新
        self.hard_update_interval_episode = self.args.hard_update_interval_episode  # 硬更新的间隔（按episode）
        self.popart_update_interval_step = self.args.popart_update_interval_step  # PopArt更新频率
        self.actor_train_interval_step = self.args.actor_train_interval_step  # Actor更新步频（对AC类）
        self.train_interval_episode = self.args.train_interval_episode  # 每多少个episode进行一次训练
        self.train_adj_episode = self.args.train_adj_episode  # 每多少个episode训练一次邻接图
        self.drop_temperature_episode = self.args.drop_temperature_episode  # 温度/epsilon下降节奏
        self.train_interval = self.args.train_interval  # （备用）训练间隔
        self.use_dyn_graph = self.args.use_dyn_graph  # 是否使用动态图（学习邻接）
        self.equal_vdn = self.args.equal_vdn  # 是否退化成VDN等价形式（特殊实验用）
        self.use_eval = self.args.use_eval  # 是否开启评估
        self.eval_interval = self.args.eval_interval  # 评估间隔（按总步数）
        self.save_interval = self.args.save_interval  # 模型保存间隔（按总步数）
        self.log_interval = self.args.log_interval  # 日志打印/写入间隔（按总步数）
        self.gae_lambda = self.args.gae_lambda  # GAE参数（若使用）
        self.gamma = self.args.gamma  # 折扣因子
        self.use_linear_lr_decay = self.args.use_linear_lr_decay  # 是否线性衰减学习率
        #self.independent_p_q = self.args.independent_p_q  # 一些算法中p/q网络是否独立
        #self.pair_rnn_hidden_dim = self.args.pair_rnn_hidden_dim  # （特定算法用）pair-RNN隐藏维度
        self.epsilon_anneal_time = self.args.epsilon_anneal_time  # epsilon退火时间（探索）

        # 下面是训练过程中的统计量和状态，用于控制训练周期、记录日志、保存等
        self.total_env_steps = 0  # 训练期间与环境交互的总步数（累加）
        self.num_episodes_collected = 0  # 训练期间收集到的总 episode 数
        self.num_adj_episodes_collected = 0  # 已用于邻接训练的 episode 数
        self.total_train_steps = 0  # 已进行的梯度更新次数
        self.last_train_episode = 0  # 上次执行梯度更新时的 episode 计数
        self.last_train_adj_episode = 0  # 上次训练邻接网络时的 episode
        self.last_drop_t_episode = 0  # 上次 temperature 降低时的 episode
        self.last_eval_T = 0  # 上次做 eval 的 step
        self.last_save_T = 0  # 上次保存模型的 step
        self.last_log_T = 0  # 上次记录日志的 step
        self.last_hard_update_episode = 0  # 上次进行硬参数拷贝（target <- live）的 episode
        self.use_vfunction = self.args.use_vfunction  # 是否使用 state/value function v（针对多阶的 value）
        self.use_save = self.args.use_save  # 是否将模型保存到磁盘
        self.pretrain_adj = self.args.pretrain_adj  # 是否预加载（预训练）邻接网络
        self.num_mini_batch = self.args.num_mini_batch  # 用于邻接训练时的 minibatch 数目
        self.adj_train_epochs = max(
            1,
            int(getattr(self.args, "adj_train_epochs", 10)),
        )
        self.use_adj_linear_lr_decay = bool(
            getattr(self.args, "use_adj_linear_lr_decay", False)
        )
        self._adj_lr_decay_warning_emitted = False
        self.adj_begin_step = self.args.adj_begin_step  # 在多少 step 之后开始训练动态邻接
        self.use_adj_init = self.args.use_adj_init  # 是否在训练开始时使用邻接的初始值
        self._validate_dynamic_graph_args() #检查sddfg参数
        self._lr_decay_warning_emitted = False

        # config 中有些字段不是必须的，使用 contains 检查并设置默认值
        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        # 决定 episode_length 与 data_chunk_length（用于 RNN 训练时切分序列）
        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            # 对于 naive recurrent policy，data_chunk_length 直接等于整个 episode 长度
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                # 否则使用 args 中指定的数据块长度（用于截断 BPTT）
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        #import pdb;pdb.set_trace()
        # policy_info 存放了每个 policy 的 observation/action 空间等结构化信息
        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        # 多 agent/因子设置
        self.num_agents = config["num_agents"]
        self.num_factor = self.args.num_factor  # 在 DDFG 中的 factor 数（用于高阶交互建模）
        self.agent_ids = [i for i in range(self.num_agents)]

        # 环境与并行数
        self.env = config["env"]
        self.eval_env = config["eval_env"]
        # no parallel envs
        self.num_envs = self.env.num_envs  # 并行环境的数量（通常是向量化环境里的一维）
        self.num_eval_envs = self.eval_env.num_envs

        self.action_repr_updating = True  # 某些算法（casec）需要更新 action 表示

        # 模型与日志路径设置（支持 wandb 或本地写入）
        # import pdb;pdb.set_trace()
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            # 当使用 wandb 时，保存目录由 wandb 控制
            self.save_dir = str(wandb.run.dir)
            # import pdb;pdb.set_trace()
            self.run_dir = config["run_dir"]
        else:
            # 本地写入日志和模型
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)  # tensorboardX 写入器
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        # 根据算法名选择对应的 Policy/Adj/TrainAlgo 类
        if self.algorithm_name == "ddfg":
            # 当算法为 ddfg 时，导入对应的邻接生成器（Adj）和策略类（Policy），以及训练算法实现（TrainAlgo）
            from algorithms.ddfg.algorithm.adj_generator_new import Adj_Generator as Adj
            from algorithms.ddfg.algorithm.rDDFGPolicy import R_DDFGPolicy as Policy
            from algorithms.ddfg.r_ddfg import R_DDFG as TrainAlgo
        elif self.algorithm_name == "sddfg":
            from algorithms.sddfg.r_sddfg import R_SDDFG as TrainAlgo
            from algorithms.sddfg.algorithm.rSDDFGPolicy import R_SDDFGPolicy as Policy
            from algorithms.sddfg.algorithm.adj_generator import Adj_Generator as Adj

        elif self.algorithm_name == "vdn":
            from algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from algorithms.vdn.vdn import VDN as TrainAlgo

        elif self.algorithm_name == "qmix":
            from algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from algorithms.qmix.qmix import QMix as TrainAlgo

        elif self.algorithm_name == "qplex":
            from algorithms.qplex.algorithm.QPlexPolicy import QPlexPolicy as Policy
            from algorithms.qplex.qplex import QPlex as TrainAlgo

        else:
            # 目前只实现了 sddfg分支，其他算法未实现时抛异常
            raise NotImplementedError

        # 采集函数指向 collect_rollout（子类实现），训练/保存/恢复函数根据算法类别指派
        self.collecter = self.collect_rollout
        if self.algorithm_name in self.adj_correlation:
            self.saver = self.save_q_mdfg_cent
            self.restorer = self.restore_mdfg_cent
        elif self.algorithm_name in self.q_learning:
            self.saver = self.save_q
            self.restorer = self.restore_q
        else:
            self.saver = self.save
            self.restorer = self.restore

        # —— 根据是否Q-learning系，绑定训练方法（train）为 batch_train_q 或 batch_train ——
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train
        self.train_adj = self.batch_train_adj

        # 根据 policy_info 创建每个 policy 的 Policy 实例（例如 actor/critic/network）
        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}

        # 获取 observation/state/action 的维度（从 gym space 中推断）
        self.obs_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["obs_space"])
        self.state_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["share_obs_space"])
        self.act_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["act_space"])

        # initialize trainer class for updating policies
        if self.algorithm_name in self.adj_correlation:
            # 初始化邻接网络，用于预测 agent 之间的结构/相关性
            self.adj_network = Adj(self.args, self.obs_dim, self.state_dim, self.act_dim, self.device)
            # 初始化训练器（TrainAlgo），将 policies、adj_network 等传入
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.adj_network, self.policy_mapping_fn,
                                     device=self.device, episode_length=self.episode_length)
        else:
            # 不需要 adj 的算法直接初始化训练器（TrainAlgo）
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                     device=self.device, episode_length=self.episode_length)
        # 如果提供了 model_dir，则根据是否预训练 adj 决定加载方式
        if self.model_dir is not None:
            if self.pretrain_adj:
                self.load_adj()
            else:
                self.restorer()
        # map policy id to agent ids controlled by that policy
        self.policy_agents = {
            policy_id: sorted(
                [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id])
            for policy_id in self.policies.keys()
        }

        # 保存每个 policy 的 obs/act/central_obs 维度以便后续使用
        self.policy_obs_dim = {policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {policy_id: self.policies[policy_id].central_obs_dim for policy_id in
                                       self.policy_ids}

        # 估计总共会进行多少 train episode，用于 PER 中 beta 的退火调度
        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        # RewardScaling 用于对 reward 做幂次/尺度缩放以稳定训练（实现细节见 utils/normalization.py）
        self.reward_scaling = RewardScaling(shape=1, gamma=self.gamma)

        # 根据是否打开 PER 来选择不同类型的回放缓冲区（普通或优先级）
        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization,
                                                     seed=int(self.args.seed) + 410000,
                                                     )
        else:
            # RecReplayBuffer 用于存储 recurrent（序列）数据，支持按 policy 存储
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.num_factor,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization,
                                          seed=int(self.args.seed) + 410000,
                                          )

        # 邻接缓冲区，用于存储用于训练动态邻接网络的数据（含 gae/gamma/hidden 等）
        self.adj_buffer = AdjBuffer(self.policy_info,
                                    self.policy_agents,
                                    self.num_factor,
                                    self.adj_buffer_size,
                                    self.episode_length,
                                    self.use_same_share_obs,
                                    self.use_avail_acts,
                                    self.use_reward_normalization,
                                    self.gamma,
                                    self.gae_lambda,
                                    self.hidden_size,
                                    adj_return_adv_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_return_adv_coef",
                                            1.0,
                                        )
                                    ),
                                    adj_factor_adv_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_factor_adv_coef",
                                            0.0,
                                        )
                                    ),
                                    seed=int(self.args.seed) + 420000,
                                    use_adj_delayed_triplet_credit=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_delayed_triplet_credit",
                                            False,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_coef",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_window=int(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_window",
                                            0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_cap=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_cap",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_min_reward=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_min_reward",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_positive_only=bool(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_positive_only",
                                            False,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_min_adv=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_min_adv",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_require_future_match=bool(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_require_future_match",
                                            False,
                                        )
                                    ),
                                    use_adj_delayed_triplet_success_gate=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_delayed_triplet_success_gate",
                                            False,
                                        )
                                    ),
                                    adj_delayed_triplet_success_gate_min_adv=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_success_gate_min_adv",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_success_gate_scale=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_success_gate_scale",
                                            1.0,
                                        )
                                    ),
                                    adj_delayed_triplet_success_gate_floor=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_success_gate_floor",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_future_overlap_min_nodes=int(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_future_overlap_min_nodes",
                                            3,
                                        )
                                    ),
                                    adj_delayed_triplet_partial_match_weight=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_partial_match_weight",
                                            0.5,
                                        )
                                    ),
                                    use_adj_capture_to_win_credit=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_capture_to_win_credit",
                                            False,
                                        )
                                    ),
                                    adj_capture_to_win_credit_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_coef",
                                            0.0,
                                        )
                                    ),
                                    adj_capture_to_win_credit_min_outcome_adv=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_min_outcome_adv",
                                            0.5,
                                        )
                                    ),
                                    adj_capture_to_win_credit_scale=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_scale",
                                            0.75,
                                        )
                                    ),
                                    adj_capture_to_win_credit_cap=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_cap",
                                            0.35,
                                        )
                                    ),
                                    adj_capture_to_win_credit_require_future_match=bool(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_require_future_match",
                                            False,
                                        )
                                    ),
                                    use_adj_pair_triplet_complementary_credit=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_pair_triplet_complementary_credit",
                                            False,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_coef",
                                            0.0,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_window=int(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_window",
                                            20,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_cap=float(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_cap",
                                            0.20,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_min_reward=float(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_min_reward",
                                            0.0,
                                        )
                                    ),
                                    pair_bounded_pending_evidence=bool(
                                        getattr(
                                            self.args,
                                            "pair_bounded_pending_evidence",
                                            False,
                                        )
                                    ),
                                    pair_pending_max_adj_updates=int(
                                        getattr(
                                            self.args,
                                            "pair_pending_max_adj_updates",
                                            0,
                                        )
                                    ),
                                    )
        restored_pair_pending_state = getattr(
            self, "_restored_pair_pending_state", None
        )
        if restored_pair_pending_state is not None:
            self.adj_buffer.load_pair_pending_state_dict(
                restored_pair_pending_state
            )
        elif (
                bool(getattr(
                    self.args,
                    "pair_bounded_pending_evidence",
                    False,
                ))
                and self.model_dir is not None):
            raise RuntimeError(
                "bounded pair pending requires a fresh run or a versioned "
                "pending checkpoint"
            )

    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""

        # 在收集 rollouts 之前把 trainer 切换到 rollout 模式（例如关闭梯度、把网络置为 eval 等）
        self.trainer.prep_rollout()

        # 执行一次采集（子类需实现 collect_rollout），并把返回的环境信息统计到 env_infos
        env_info = self.collecter(explore=True, training_episode=True, warmup=False)
        for k, v in env_info.items():
            self.env_infos[k].append(v)

        # train：如果满足训练触发条件（收集足够的 episode）则进行一次训练
        if ((self.num_episodes_collected - self.last_train_episode - self.batch_size) / self.train_interval_episode) >= 1:
            # LR scheduling must follow the policy-update clock.  Its former
            # placement inside adjacency updates decayed SDDFG while value
            # baselines silently kept their initial learning rate.
            if self.use_linear_lr_decay:
                lr_decay = getattr(self.trainer, "lr_decay", None)
                if callable(lr_decay):
                    policy_lr_anneal_steps = int(
                        getattr(self.args, "policy_lr_anneal_steps", 0)
                    )
                    if policy_lr_anneal_steps <= 0:
                        policy_lr_anneal_steps = self.num_env_steps
                    lr_decay(
                        self.total_env_steps,
                        policy_lr_anneal_steps,
                    )
                elif not self._lr_decay_warning_emitted:
                    print(
                        "[WARN] --use_linear_lr_decay has no effect for "
                        "trainer {} because it has no lr_decay() method.".format(
                            type(self.trainer).__name__
                        )
                    )
                    self._lr_decay_warning_emitted = True
            self.train()
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected - self.batch_size

        # 如果使用动态邻接图且满足训练邻接的条件，则训练邻接网络
        if self.use_dyn_graph and self.total_env_steps >= self.adj_begin_step and (
                (self.num_adj_episodes_collected - self.last_train_adj_episode) / self.train_adj_episode) >= 1:
            if not self.pretrain_adj:
                # Keep graph scheduling independent from policy LR decay.
                # Previously the adjacency flag had no effect unless the
                # unrelated global policy decay flag was also enabled.
                if self.use_adj_linear_lr_decay:
                    adj_lr_decay = getattr(self.trainer, "adj_lr_decay", None)
                    if callable(adj_lr_decay):
                        adj_lr_decay(self.total_env_steps)
                    elif not self._adj_lr_decay_warning_emitted:
                        print(
                            "[WARN] --use_adj_linear_lr_decay has no effect "
                            "for trainer {} because it has no "
                            "adj_lr_decay() method.".format(
                                type(self.trainer).__name__
                            )
                        )
                        self._adj_lr_decay_warning_emitted = True
                # 训练邻接网络的具体实现由 batch_train_adj 完成
                self.train_adj()
                self.log_train_adj(self.train_adj_infos)
            # Keep both sides of the scheduling delta in the same counter
            # domain.  Warmup episodes increment num_episodes_collected but
            # are intentionally excluded from num_adj_episodes_collected.
            # Storing the total counter here made the next adjacency update
            # wait dozens of episodes (and progressively longer) instead of
            # the configured train_adj_episode interval.
            self.last_train_adj_episode = (
                self.num_adj_episodes_collected
            )
        # save：按间隔保存模型
        if self.use_save and (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps
        # log：按间隔记录日志
        if self.total_env_steps > 0 and ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # eval：按间隔执行评估
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps

    def warmup(self, num_warmup_episodes):
        """
        预热：在正式训练前先收集一些随机（或带探索）的episode，填满回放缓冲区，让训练更稳。
        :param num_warmup_episodes: 需要的预热episode数量
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        # Preserve detached episode summaries so environment-specific runners
        # can audit rare rewards that entered replay during warmup.  This does
        # not alter collection, replay insertion, RNG, or training state.
        self.warmup_env_infos = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs)):
            # 预热一般用 explore=True，warmup=True，表示只收集不训练
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            self.warmup_env_infos.append(dict(env_info))
            warmup_rewards.append(env_info['average_episode_rewards'])

        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """（非Qmix类）做一次参数更新：对每个policy从buffer取样→train→（必要的）软/硬更新。"""
        self.trainer.prep_training()  # 切换到训练模式（train）
        self.train_infos = []
        update_actor = False

        for p_id in self.policy_ids:
            # PER：带重要性采样权重beta；否则普通采样
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            # 如果obs是集中式共享，则调用“共享中心obs”的训练；否则用独立版本
            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs \
                else self.trainer.cent_train_policy_on_batch

            # 训练并拿到信息（loss等）、优先级（PER）、索引
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = True  # 这里保留开关位（某些算法只在满足条件时更新actor）

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        # 目标网络更新：优先软更新；否则到间隔用硬更新
        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Q-learning系（QMIX/VDN/DDFG等）的批训练：从buffer采样→trainer.train_policy_on_batch。"""
        self.trainer.prep_training()
        self.train_infos = []
        self.policy_reward_sample_diagnostics = []

        if self.algorithm_name == 'casec':
            sample = self.buffer.sample(self.batch_size)
            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample, self.action_repr_updating)
            self.train_infos.append(train_info)

        #elif self.algorithm_name == 'rddfg_cent_rw':
        elif self.algorithm_name in ["ddfg", "sddfg"]:
            # DDFG：每个训练间隔内多次小批次迭代，有助于稳定
            for q_update_index in range(self.train_interval_episode):
                for p_id in self.policy_ids:
                    if self.use_per:
                        beta = self.beta_anneal.eval(self.total_train_steps)
                        sample = self.buffer.sample(self.batch_size, beta, p_id)
                    else:
                        # 均分batch到多次迭代
                        q_batch_size = (
                            self.batch_size // self.train_interval_episode
                        )
                        if (
                                self.algorithm_name == "sddfg"
                                and self.q_terminal_replay_lane
                                and q_update_index == 0):
                            sample = self.buffer.sample_with_terminal_win_lane(
                                q_batch_size
                            )
                        else:
                            sample = self.buffer.sample(q_batch_size)
                    if self.algorithm_name == "sddfg":
                        # Explicitly extend only the SDDFG learner batch.  DDFG
                        # and all other replay consumers retain their existing
                        # tuple contract and RNG/sample sequence.
                        sample = _append_sddfg_terminal_win_provenance_to_sample(
                            self.buffer,
                            self.policy_ids,
                            sample,
                        )
                    train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)
                    policy_buffer = self.buffer.policy_buffers[p_id]
                    reward_sample = getattr(
                        policy_buffer,
                        "last_reward_sample_diagnostics",
                        None,
                    )
                    if reward_sample is None:
                        raise RuntimeError(
                            "policy replay did not expose reward sample diagnostics"
                        )
                    terminal_win_rewards = np.asarray(
                        reward_sample["terminal_win_rewards"]
                    )
                    reward_std = float(reward_sample["std_reward"])
                    terminal_reward_mask = terminal_win_rewards > 0.0
                    terminal_sample_count = int(terminal_reward_mask.sum())
                    terminal_episode_count = int(
                        np.any(terminal_reward_mask, axis=(1, 2)).sum()
                    )
                    train_info["sampled_terminal_reward_transition_count"] = float(
                        terminal_sample_count
                    )
                    train_info["sampled_terminal_reward_episode_count"] = float(
                        terminal_episode_count
                    )
                    train_info["reward_normalization_mean"] = float(
                        reward_sample["mean_reward"]
                    )
                    train_info["reward_normalization_std"] = reward_std
                    train_info["terminal_bonus_normalized_delta"] = float(
                        1.0 / reward_std
                    )
                    terminal_lane_forced = int(float(reward_sample.get(
                        "terminal_replay_lane_forced", 0.0
                    )))
                    terminal_lane_episode_count = int(np.asarray(
                        reward_sample.get(
                            "terminal_replay_lane_episode_mask",
                            np.zeros(terminal_win_rewards.shape[0]),
                        )
                    ).sum())
                    self.q_terminal_replay_lane_forced_count += (
                        terminal_lane_forced
                    )
                    self.q_terminal_replay_lane_gated_transition_count += float(
                        train_info.get(
                            "q_target_n_step_gated_transition_count", 0.0
                        )
                    )
                    train_info["q_terminal_replay_lane_enabled"] = float(
                        self.q_terminal_replay_lane
                    )
                    train_info["q_terminal_replay_lane_forced"] = float(
                        terminal_lane_forced
                    )
                    train_info["q_terminal_replay_lane_episode_count"] = float(
                        terminal_lane_episode_count
                    )
                    train_info[
                        "q_terminal_replay_candidate_episode_count"
                    ] = float(reward_sample.get(
                        "terminal_replay_candidate_episode_count", 0.0
                    ))
                    train_info[
                        "q_terminal_replay_lane_forced_cumulative"
                    ] = float(self.q_terminal_replay_lane_forced_count)
                    train_info[
                        "q_terminal_replay_gated_transition_cumulative"
                    ] = float(
                        self.q_terminal_replay_lane_gated_transition_count
                    )
                    self.policy_reward_sample_diagnostics.append({
                        "sample_indices": np.asarray(
                            reward_sample["sample_indices"], dtype=np.int64
                        ).copy(),
                        "terminal_transition_count": terminal_sample_count,
                        "terminal_episode_count": terminal_episode_count,
                    })
                    if self.use_per:
                        self.buffer.update_priorities(idxes, new_priorities, p_id)
                    self.train_infos.append(train_info)
        else:
            # 其他Q系：每个policy各采一批训练一次
            for p_id in self.policy_ids:
                if self.use_per:
                    beta = self.beta_anneal.eval(self.total_train_steps)
                    sample = self.buffer.sample(self.batch_size, beta, p_id)
                else:
                    sample = self.buffer.sample(self.batch_size)
                train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)
                if self.use_per:
                    self.buffer.update_priorities(idxes, new_priorities, p_id)
                self.train_infos.append(train_info)

        # CASEC：首次更新动作表征后，固定住并做一次硬更新（稳定）
        if self.algorithm_name == 'casec' and self.action_repr_updating:
            self.trainer.update_action_repr()
            self.action_repr_updating = False
            self.trainer.hard_target_updates()
            self.last_hard_update_episode = self.num_episodes_collected

        # 目标网络更新：优先软更新；否则按间隔硬更新
        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    @staticmethod
    def _capture_pair_pending_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None
            ),
        }

    @staticmethod
    def _restore_pair_pending_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _run_pair_pending_outer_transaction(
            self,
            policy_id,
            policy_buffer,
            prepared,
            adjacency_update_round,
            graph_clip_stop_ratio,
            factor_clip_stop_ratio,
            min_ppo_epochs):
        """Run all pair-only PPO epochs, then commit once or roll back all."""
        trainer_state = self.trainer.pair_pending_outer_transaction_state()
        rng_state = self._capture_pair_pending_rng_state()
        sequence_before = int(self._adj_transaction_sequence_index)
        epoch_infos = []
        abort_reason = ""
        completed_epochs = 0
        optimizer_transactions = 0
        rolled_back = False
        checkpoint_contract_valid = False
        try:
            for epoch_index in range(int(self.adj_train_epochs)):
                train_info, _, _ = self.trainer.train_adj_on_batch(
                    prepared["batch"],
                    self.use_adj_init,
                    adj_update_round=adjacency_update_round,
                    pair_only_objective=True,
                    enable_optimizer_transaction_diagnostics=True,
                    diagnostic_policy_id=policy_id,
                    diagnostic_transaction_sequence_index=(
                        sequence_before + epoch_index
                    ),
                )
                if float(train_info.get(
                        "pair_pending_objective_scope_contract_valid",
                        0.0)) != 1.0:
                    raise RuntimeError(
                        "pending pair-only objective scope failed"
                    )
                pair_gradient_norm = float(
                    train_info.get("pair_gradient_norm", 0.0)
                )
                if (
                        not np.isfinite(pair_gradient_norm)
                        or pair_gradient_norm <= 0.0):
                    raise RuntimeError(
                        "pending pair-only transaction has non-finite or zero "
                        "gradient"
                    )
                optimizer_step_after = float(train_info.get(
                    "pair_optimizer_transaction_optimizer_step_after", 0.0
                ))
                optimizer_step_before = float(train_info.get(
                    "pair_optimizer_transaction_optimizer_step_before", 0.0
                ))
                if (
                        not np.isfinite(optimizer_step_before)
                        or not np.isfinite(optimizer_step_after)
                        or optimizer_step_after != optimizer_step_before + 1.0):
                    raise RuntimeError(
                        "pending pair-only Adam step did not advance once"
                    )
                if float(train_info.get(
                        "pair_optimizer_transaction_rollback_occurred",
                        0.0)) != 0.0:
                    raise RuntimeError(
                        "inner adjacency transaction rolled back"
                    )
                if float(train_info.get(
                        "pair_optimizer_transaction_pair_optimizer_isolated",
                        0.0)) != 1.0:
                    raise RuntimeError(
                        "pending pair-only transaction did not use isolated "
                        "Adam state"
                    )
                final_pair_descent_dot = float(train_info.get(
                    "pair_optimizer_transaction_final_pair_descent_dot", 0.0
                ))
                if (
                        not np.isfinite(final_pair_descent_dot)
                        or final_pair_descent_dot <= 0.0):
                    raise RuntimeError(
                        "pending pair-only transaction did not execute a "
                        "strict pair descent step"
                    )
                epoch_infos.append(train_info)
                completed_epochs += 1
                optimizer_transactions += 1
                if float(train_info.get(
                        "pair_pending_standard_ppo_early_stop_applicable",
                        1.0)) != 0.0:
                    raise RuntimeError(
                        "pending pair-only transaction unexpectedly enabled "
                        "standard PPO early-stop"
                    )
                if float(train_info.get(
                        "pair_pending_all_configured_epochs_required",
                        0.0)) != 1.0:
                    raise RuntimeError(
                        "pending pair-only transaction did not require every "
                        "configured PPO epoch"
                    )
                # Standard adjacency early-stop controls the graph/base PPO
                # population. Pending replay deliberately disables those
                # objectives and trains only exact pair targets, so applying
                # that unrelated population gate can roll back a valid pair
                # step solely because the immutable snapshot was already
                # stale before this transaction began (run108 at 144.8k).
                # Pair-only remains bounded by its own clipped surrogate,
                # transaction-time stale trust, finite/nonzero gradient
                # checks, and the outer all-epoch atomic contract.
            effective_positive_mass = min(
                float(info["pair_pending_effective_positive_mass"])
                for info in epoch_infos
            )
            effective_negative_mass = min(
                float(info["pair_pending_effective_negative_mass"])
                for info in epoch_infos
            )
            policy_buffer.pair_pending_store.commit_prepared(
                prepared["cohort_id"],
                committed_adj_update=adjacency_update_round,
                completed_ppo_epochs=completed_epochs,
                optimizer_transaction_count=optimizer_transactions,
                positive_effective_mass=effective_positive_mass,
                negative_effective_mass=effective_negative_mass,
                target_bearing_transition_count=prepared[
                    "target_bearing_transition_count"
                ],
                rollback=False,
                rejected=False,
            )
            serialized_pending_state = (
                policy_buffer.pair_pending_store.state_dict()
            )
            if serialized_pending_state.get("prepared_cohorts"):
                raise RuntimeError(
                    "committed pair pending checkpoint retains a prepared "
                    "cohort"
                )
            committed_keys = {
                tuple(key)
                for key in serialized_pending_state.get(
                    "committed_generation_keys", ()
                )
            }
            expected_committed_generations = {
                (
                    str(metadata["policy_id"]),
                    int(metadata["replay_generation"]),
                )
                for metadata in prepared["entry_metadata"]
            }
            if not expected_committed_generations.issubset(committed_keys):
                raise RuntimeError(
                    "pair pending checkpoint omits committed generations"
                )
            checkpoint_contract_valid = True
        except Exception as error:
            bounded_zero_gradient = isinstance(
                error, PairPendingZeroGradientError
            )
            recoverable_optimizer_noop = isinstance(
                error, PairOptimizerRecoverableNoOpError
            )
            if recoverable_optimizer_noop and not bool(getattr(
                    error, "atomic_rollback_complete", False)):
                raise RuntimeError(
                    "pending pair optimizer no-op has no verified rollback"
                )
            bounded_exact_infeasible = bool(getattr(
                error,
                "strict_pair_exact_bounded_deferral_safe",
                False,
            ))
            exact_failure_rows = getattr(
                error,
                "strict_pair_exact_failure_rows",
                None,
            )
            if not abort_reason:
                error_text = str(error).upper()
                if bounded_zero_gradient:
                    abort_reason = "ZERO_GRADIENT"
                elif recoverable_optimizer_noop:
                    abort_reason = "NO_USABLE_UPDATE"
                elif bounded_exact_infeasible:
                    abort_reason = "EXACT_INFEASIBLE"
                elif (
                        "ZERO TARGET" in error_text
                        or "NO TARGET" in error_text):
                    abort_reason = "ZERO_TARGET"
                else:
                    abort_reason = "EXCEPTION"
            self.trainer.restore_pair_pending_outer_transaction_state(
                trainer_state
            )
            policy_buffer.pair_pending_store.load_state_dict(
                prepared["store_state_before_prepare"]
            )
            self._restore_pair_pending_rng_state(rng_state)
            self._adj_transaction_sequence_index = sequence_before
            rolled_back = True
            if exact_failure_rows is not None:
                self._record_strict_pair_exact_failure(
                    diagnostic_rows=exact_failure_rows,
                    adjacency_update_round=adjacency_update_round,
                    ppo_epoch_index=(
                        int(self.adj_train_epochs) + int(completed_epochs)
                    ),
                    policy_id=policy_id,
                    partition_index=0,
                )
            objective_scope_contract_valid = bool(
                bounded_zero_gradient
                or recoverable_optimizer_noop
                or bounded_exact_infeasible
                or (
                    epoch_infos
                    and all(
                        float(info.get(
                            "pair_pending_objective_scope_contract_valid",
                            0.0,
                        )) == 1.0
                        for info in epoch_infos
                    )
                    and "OBJECTIVE SCOPE" not in str(error).upper()
                )
            )
            policy_buffer.record_pair_pending_transaction_result(
                committed=False,
                rolled_back=True,
                abort_reason=abort_reason,
                objective_scope_contract_valid=(
                    objective_scope_contract_valid
                ),
                stale_contract_valid=True,
                mass_contract_valid=bool(
                    prepared.get("mass_contract_valid", 0.0)
                ),
                atomic_rollback_contract_valid=True,
                checkpoint_contract_valid=True,
                positive_stale_trust=(
                    min(float(info.get(
                        "pair_pending_positive_stale_trust_mean", 0.0
                    )) for info in epoch_infos)
                    if epoch_infos else 0.0
                ),
                negative_stale_trust=(
                    min(float(info.get(
                        "pair_pending_negative_stale_trust_mean", 0.0
                    )) for info in epoch_infos)
                    if epoch_infos else 0.0
                ),
                raw_positive_mass=prepared["raw_positive_mass"],
                raw_negative_mass=prepared["raw_negative_mass"],
                effective_positive_mass=(
                    min(float(info.get(
                        "pair_pending_effective_positive_mass", 0.0
                    )) for info in epoch_infos)
                    if epoch_infos else 0.0
                ),
                effective_negative_mass=(
                    min(float(info.get(
                        "pair_pending_effective_negative_mass", 0.0
                    )) for info in epoch_infos)
                    if epoch_infos else 0.0
                ),
            )
            self._append_fixed_rows_csv(
                "progress_train_pair_pending_cohort.csv",
                [{
                    "run_id": os.path.basename(
                        str(self.run_dir).rstrip(os.sep)
                    ),
                    "env_step": int(self.total_env_steps),
                    "adjacency_update_round": int(
                        adjacency_update_round
                    ),
                    "policy_id": str(policy_id),
                    "cohort_id": int(prepared["cohort_id"]),
                    "episode_count": int(prepared["episode_count"]),
                    "chunk_count": int(prepared["chunk_count"]),
                    "target_bearing_transition_count": int(prepared[
                        "target_bearing_transition_count"
                    ]),
                    "raw_positive_mass": float(
                        prepared["raw_positive_mass"]
                    ),
                    "raw_negative_mass": float(
                        prepared["raw_negative_mass"]
                    ),
                    "generations": "|".join(
                        str(metadata["replay_generation"])
                        for metadata in prepared["entry_metadata"]
                    ),
                    "capture_event_ids": "|".join(
                        str(metadata["capture_event_id"])
                        for metadata in prepared["entry_metadata"]
                    ),
                    "environment_episode_ids": "|".join(
                        str(metadata["environment_episode_id"])
                        for metadata in prepared["entry_metadata"]
                    ),
                    "prey_ids": "|".join(
                        str(metadata["prey_id"])
                        for metadata in prepared["entry_metadata"]
                    ),
                    "participant_slots": "|".join(
                        "-".join(
                            str(slot)
                            for slot in metadata["participant_slots"]
                        )
                        for metadata in prepared["entry_metadata"]
                    ),
                    "pair_identities": "|".join(
                        ",".join(
                            "-".join(str(node) for node in identity)
                            for identity in metadata[
                                "canonical_pair_identities"
                            ]
                        )
                        for metadata in prepared["entry_metadata"]
                    ),
                    "signs": "|".join(
                        str(metadata["sign"])
                        for metadata in prepared["entry_metadata"]
                    ),
                    "sources": "|".join(prepared["source_states"]),
                    "pending_ages": "|".join(
                        str(value) for value in prepared["pending_ages"]
                    ),
                    "policy_ages": "|".join(
                        str(value) for value in prepared["policy_ages"]
                    ),
                    "behavior_policy_versions": "|".join(
                        str(metadata["behavior_policy_version"])
                        for metadata in prepared["entry_metadata"]
                    ),
                    "current_policy_version": int(
                        prepared["current_policy_version"]
                    ),
                    "epoch_pair_losses": "|".join(
                        str(float(info.get(
                            "pair_pursuit_factor_loss_contribution", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_pair_gradient_norms": "|".join(
                        str(float(info.get("pair_gradient_norm", 0.0)))
                        for info in epoch_infos
                    ),
                    "epoch_optimizer_steps_before": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_optimizer_step_before",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_optimizer_steps_after": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_optimizer_step_after",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_positive_stale_trust": "|".join(
                        str(float(info.get(
                            "pair_pending_positive_stale_trust_mean", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_negative_stale_trust": "|".join(
                        str(float(info.get(
                            "pair_pending_negative_stale_trust_mean", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_effective_positive_mass": "|".join(
                        str(float(info.get(
                            "pair_pending_effective_positive_mass", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_effective_negative_mass": "|".join(
                        str(float(info.get(
                            "pair_pending_effective_negative_mass", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_pair_target_clip_ratio": "|".join(
                        str(float(info.get(
                            "pair_pending_pair_target_clip_ratio", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_pair_target_trusted_clip_ratio": "|".join(
                        str(float(info.get(
                            "pair_pending_pair_target_trusted_clip_ratio", 0.0
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_adam_raw_pair_descent_dot": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_adam_raw_pair_descent_dot",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_final_pair_descent_dot": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_final_pair_descent_dot",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_exact_score_signed_change": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_score_signed_change_mean",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_score_correct_target_count": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_"
                            "score_correct_direction_target_count",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "epoch_score_reverse_target_count": "|".join(
                        str(float(info.get(
                            "pair_optimizer_transaction_"
                            "score_reverse_direction_target_count",
                            0.0,
                        )))
                        for info in epoch_infos
                    ),
                    "standard_ppo_early_stop_applicable": 0,
                    "all_configured_epochs_required": 1,
                    "completed_ppo_epochs": int(completed_epochs),
                    "optimizer_transaction_count": int(
                        optimizer_transactions
                    ),
                    "committed": 0,
                    "rolled_back": 1,
                    "abort_reason": str(abort_reason),
                    "objective_scope_contract_valid": int(
                        objective_scope_contract_valid
                    ),
                    "atomic_contract_valid": 1,
                    "checkpoint_contract_valid": 1,
                }],
            )
            if abort_reason in (
                    "EARLY_STOP",
                    "ZERO_GRADIENT",
                    "NO_USABLE_UPDATE",
                    "EXACT_INFEASIBLE"):
                # These conditions are bounded no-action results. The outer
                # transaction has restored every mutable state and has not
                # consumed evidence. A zero-gradient cohort remains live so
                # later real evidence can change its signed Jacobian before
                # the existing TTL expires.
                return []
            raise

        # Persist epoch rows only after the logical cohort is committed. A
        # failed epoch can therefore never leave a durable transaction row.
        transaction_rows = []
        for epoch_index, train_info in enumerate(epoch_infos):
            transaction_row = self._record_adj_transaction(
                train_adj_info=train_info,
                policy_buffer=policy_buffer,
                sample=prepared["batch"],
                adjacency_update_round=adjacency_update_round,
                # Keep existing transaction keys unique without changing the
                # historical schema. The logical CSV retains epoch 0/1.
                ppo_epoch_index=(
                    int(self.adj_train_epochs) + epoch_index
                ),
                policy_id=policy_id,
                partition_index=0,
                selected_episode_count_override=prepared[
                    "episode_count"
                ],
                selected_chunk_count_override=prepared["chunk_count"],
                class_complete_override=1.0,
            )
            transaction_rows.append(transaction_row)
        self._append_fixed_rows_csv(
            "progress_train_pair_pending_cohort.csv",
            [{
                "run_id": os.path.basename(
                    str(self.run_dir).rstrip(os.sep)
                ),
                "env_step": int(self.total_env_steps),
                "adjacency_update_round": int(adjacency_update_round),
                "policy_id": str(policy_id),
                "cohort_id": int(prepared["cohort_id"]),
                "episode_count": int(prepared["episode_count"]),
                "chunk_count": int(prepared["chunk_count"]),
                "target_bearing_transition_count": int(prepared[
                    "target_bearing_transition_count"
                ]),
                "raw_positive_mass": float(
                    prepared["raw_positive_mass"]
                ),
                "raw_negative_mass": float(
                    prepared["raw_negative_mass"]
                ),
                "generations": "|".join(
                    str(metadata["replay_generation"])
                    for metadata in prepared["entry_metadata"]
                ),
                "capture_event_ids": "|".join(
                    str(metadata["capture_event_id"])
                    for metadata in prepared["entry_metadata"]
                ),
                "environment_episode_ids": "|".join(
                    str(metadata["environment_episode_id"])
                    for metadata in prepared["entry_metadata"]
                ),
                "prey_ids": "|".join(
                    str(metadata["prey_id"])
                    for metadata in prepared["entry_metadata"]
                ),
                "participant_slots": "|".join(
                    "-".join(
                        str(slot)
                        for slot in metadata["participant_slots"]
                    )
                    for metadata in prepared["entry_metadata"]
                ),
                "pair_identities": "|".join(
                    ",".join(
                        "-".join(str(node) for node in identity)
                        for identity in metadata[
                            "canonical_pair_identities"
                        ]
                    )
                    for metadata in prepared["entry_metadata"]
                ),
                "signs": "|".join(
                    str(metadata["sign"])
                    for metadata in prepared["entry_metadata"]
                ),
                "sources": "|".join(prepared["source_states"]),
                "pending_ages": "|".join(
                    str(value) for value in prepared["pending_ages"]
                ),
                "policy_ages": "|".join(
                    str(value) for value in prepared["policy_ages"]
                ),
                "behavior_policy_versions": "|".join(
                    str(metadata["behavior_policy_version"])
                    for metadata in prepared["entry_metadata"]
                ),
                "current_policy_version": int(
                    prepared["current_policy_version"]
                ),
                "epoch_pair_losses": "|".join(
                    str(float(info.get(
                        "pair_pursuit_factor_loss_contribution", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_pair_gradient_norms": "|".join(
                    str(float(info.get("pair_gradient_norm", 0.0)))
                    for info in epoch_infos
                ),
                "epoch_optimizer_steps_before": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_optimizer_step_before",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "epoch_optimizer_steps_after": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_optimizer_step_after",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "epoch_positive_stale_trust": "|".join(
                    str(float(info.get(
                        "pair_pending_positive_stale_trust_mean", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_negative_stale_trust": "|".join(
                    str(float(info.get(
                        "pair_pending_negative_stale_trust_mean", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_effective_positive_mass": "|".join(
                    str(float(info.get(
                        "pair_pending_effective_positive_mass", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_effective_negative_mass": "|".join(
                    str(float(info.get(
                        "pair_pending_effective_negative_mass", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_pair_target_clip_ratio": "|".join(
                    str(float(info.get(
                        "pair_pending_pair_target_clip_ratio", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_pair_target_trusted_clip_ratio": "|".join(
                    str(float(info.get(
                        "pair_pending_pair_target_trusted_clip_ratio", 0.0
                    )))
                    for info in epoch_infos
                ),
                "epoch_adam_raw_pair_descent_dot": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_adam_raw_pair_descent_dot",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "epoch_final_pair_descent_dot": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_final_pair_descent_dot",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "epoch_exact_score_signed_change": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_score_signed_change_mean",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "epoch_score_correct_target_count": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_"
                        "score_correct_direction_target_count",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "epoch_score_reverse_target_count": "|".join(
                    str(float(info.get(
                        "pair_optimizer_transaction_"
                        "score_reverse_direction_target_count",
                        0.0,
                    )))
                    for info in epoch_infos
                ),
                "standard_ppo_early_stop_applicable": 0,
                "all_configured_epochs_required": 1,
                "completed_ppo_epochs": int(completed_epochs),
                "optimizer_transaction_count": int(
                    optimizer_transactions
                ),
                "committed": 1,
                "rolled_back": int(rolled_back),
                "abort_reason": "",
                "objective_scope_contract_valid": 1,
                "atomic_contract_valid": 1,
                "checkpoint_contract_valid": int(
                    checkpoint_contract_valid
                ),
            }],
        )
        policy_buffer.record_pair_pending_transaction_result(
            committed=True,
            rolled_back=False,
            abort_reason="",
            objective_scope_contract_valid=True,
            stale_contract_valid=all(
                np.isfinite(float(info.get(
                    "pair_pending_positive_stale_trust_mean", np.nan
                )))
                and np.isfinite(float(info.get(
                    "pair_pending_negative_stale_trust_mean", np.nan
                )))
                for info in epoch_infos
            ),
            mass_contract_valid=bool(
                prepared.get("mass_contract_valid", 0.0)
            ),
            atomic_rollback_contract_valid=True,
            checkpoint_contract_valid=checkpoint_contract_valid,
            positive_stale_trust=min(
                float(info["pair_pending_positive_stale_trust_mean"])
                for info in epoch_infos
            ),
            negative_stale_trust=min(
                float(info["pair_pending_negative_stale_trust_mean"])
                for info in epoch_infos
            ),
            raw_positive_mass=prepared["raw_positive_mass"],
            raw_negative_mass=prepared["raw_negative_mass"],
            effective_positive_mass=effective_positive_mass,
            effective_negative_mass=effective_negative_mass,
        )
        return transaction_rows

    def _train_adj_on_batch_with_exact_failure_logging(
            self,
            sample,
            use_adj_init,
            adjacency_update_round,
            consume_factor_credit_observations,
            policy_id,
            transaction_sequence_index,
            ppo_epoch_index,
            partition_index):
        """Run one real producer call and persist only typed exact failures."""
        try:
            return self.trainer.train_adj_on_batch(
                sample,
                use_adj_init,
                adj_update_round=adjacency_update_round,
                consume_factor_credit_observations=(
                    consume_factor_credit_observations
                ),
                diagnostic_policy_id=policy_id,
                diagnostic_transaction_sequence_index=(
                    transaction_sequence_index
                ),
            )
        except RuntimeError as error:
            if isinstance(error, PairOptimizerRecoverableNoOpError):
                if not bool(getattr(
                        error, "atomic_rollback_complete", False)):
                    raise RuntimeError(
                        "pair optimizer no-op escaped without a verified "
                        "atomic rollback"
                    )
                diagnostics = dict(getattr(error, "diagnostics", {}))
                return ({
                    "_pair_optimizer_recoverable_noop": 1.0,
                    "pair_optimizer_recoverable_noop_transaction_count": 1.0,
                    "pair_optimizer_recoverable_noop_target_count": float(
                        getattr(error, "target_count", 0.0)
                    ),
                    "pair_optimizer_recoverable_noop_reason_code": float(
                        getattr(error, "reason_code", 0.0)
                    ),
                    "pair_optimizer_recoverable_noop_pair_update_dot": float(
                        diagnostics.get("pair_update_dot", 0.0)
                    ),
                    "pair_optimizer_recoverable_noop_pair_update_norm_sq": (
                        float(diagnostics.get("pair_update_norm_sq", 0.0))
                    ),
                    "pair_optimizer_recoverable_noop_pair_gradient_norm_sq": (
                        float(diagnostics.get("pair_gradient_norm_sq", 0.0))
                    ),
                    "pair_optimizer_recoverable_noop_clipped_pair_dot": float(
                        diagnostics.get("clipped_pair_dot", 0.0)
                    ),
                    "pair_optimizer_recoverable_noop_"
                    "clipped_gradient_norm_sq": float(
                        diagnostics.get("clipped_gradient_norm_sq", 0.0)
                    ),
                }, None, None)
            failure_rows = getattr(
                error,
                "strict_pair_exact_failure_rows",
                None,
            )
            if failure_rows is None:
                raise
            self._record_strict_pair_exact_failure(
                diagnostic_rows=failure_rows,
                adjacency_update_round=adjacency_update_round,
                ppo_epoch_index=ppo_epoch_index,
                policy_id=policy_id,
                partition_index=partition_index,
            )
            if bool(getattr(
                    error,
                    "strict_pair_exact_bounded_deferral_safe",
                    False)):
                return ({
                    "_strict_pair_exact_bounded_deferred": 1.0,
                    "strict_pair_exact_bounded_deferred_transaction_count": (
                        1.0
                    ),
                    "strict_pair_exact_bounded_deferred_target_count": float(
                        failure_rows[0].get("target_count", 0.0)
                    ),
                }, None, None)
            raise

    def batch_train_adj(self):
        """邻接图策略训练：从adj_buffer按序列小块采样→多次迭代→PPO剪切更新adj_network。"""
        self.trainer.prep_training()
        # 收集训练过程指标（多次迭代取均值）
        self.train_adj_infos = {}
        adj_transaction_records = []
        pair_evidence_episode_rows = []
        adj_transaction_sequence_start = 0
        if self.algorithm_name == "sddfg":
            self._ensure_adj_transaction_log_state()
            adj_transaction_sequence_start = int(
                self._adj_transaction_sequence_index
            )

        data_chunk_length = 10  # 每次从轨迹中取多少步的连续片段来训练邻接（小BPTT段）

        # The 16-episode window overlaps across graph updates.  run35 showed
        # that continuing to replay stale graph decisions after clamp_ratio
        # has already exceeded 0.7 makes the graph PPO target drift instead of
        # improving reward.  Stop the current adj update early when either the
        # graph-level or factor-level PPO clamp fraction is too high.
        graph_clip_stop_ratio = float(
            getattr(self.args, "adj_ppo_clip_stop_ratio", 0.0)
        )
        factor_clip_stop_ratio = float(
            getattr(self.args, "adj_ppo_factor_clip_stop_ratio", 0.0)
        )
        min_ppo_epochs = max(
            1,
            int(getattr(self.args, "adj_ppo_min_epochs", 1)),
        )
        configured_recent_episode_window = max(
            0,
            int(getattr(self.args, "adj_recent_episode_window", 0)),
        )
        stale_trust_control_enabled = bool(
            getattr(self.args, "use_adj_ppo_stale_trust", False)
        )
        recent_episode_window = configured_recent_episode_window
        dynamic_recent_enabled = bool(
            getattr(self.args, "use_adj_dynamic_recent_window", False)
        )
        recent_window_shrunk = 0.0
        recent_window_recovered = 0.0
        recent_window_emergency_shrunk = 0.0
        recent_window_high_stale_count = 0.0
        recent_window_low_stale_count = 0.0
        if dynamic_recent_enabled and configured_recent_episode_window > 0:
            min_recent_window = max(
                1,
                min(
                    configured_recent_episode_window,
                    int(getattr(self.args, "adj_recent_episode_window_min", 1)),
                ),
            )
            graph_stale_threshold = float(
                getattr(self.args, "adj_recent_window_stale_threshold", 0.35)
            )
            factor_stale_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_factor_stale_threshold",
                    graph_stale_threshold,
                )
            )
            recover_graph_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_recover_stale_threshold",
                    -1.0,
                )
            )
            if recover_graph_threshold < 0.0:
                recover_graph_threshold = 0.8 * graph_stale_threshold
            recover_factor_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_recover_factor_stale_threshold",
                    -1.0,
                )
            )
            if recover_factor_threshold < 0.0:
                recover_factor_threshold = 0.8 * factor_stale_threshold
            shrink_patience = max(
                1,
                int(getattr(self.args, "adj_recent_window_shrink_patience", 1)),
            )
            recover_patience = max(
                1,
                int(getattr(self.args, "adj_recent_window_recover_patience", 2)),
            )
            severe_margin = max(
                0.0,
                float(getattr(self.args, "adj_recent_window_severe_margin", 0.15)),
            )
            emergency_recent_window = max(
                1,
                min(
                    configured_recent_episode_window,
                    int(
                        getattr(
                            self.args,
                            "adj_recent_episode_window_emergency",
                            1,
                        )
                    ),
                ),
            )
            emergency_graph_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_stale_threshold",
                    graph_stale_threshold + severe_margin,
                )
            )
            emergency_factor_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_factor_stale_threshold",
                    factor_stale_threshold + severe_margin,
                )
            )
            control_state = advance_recent_episode_window(
                current_window=getattr(
                    self,
                    "_adj_dynamic_recent_episode_window",
                    configured_recent_episode_window,
                ),
                configured_window=configured_recent_episode_window,
                min_window=min_recent_window,
                previous_graph_ratio=getattr(
                    self,
                    "_last_adj_control_graph_ratio",
                    np.nan,
                ),
                previous_factor_ratio=getattr(
                    self,
                    "_last_adj_control_factor_ratio",
                    np.nan,
                ),
                graph_stale_threshold=graph_stale_threshold,
                factor_stale_threshold=factor_stale_threshold,
                recover_graph_threshold=recover_graph_threshold,
                recover_factor_threshold=recover_factor_threshold,
                shrink_patience=shrink_patience,
                recover_patience=recover_patience,
                severe_margin=severe_margin,
                emergency_window=emergency_recent_window,
                emergency_graph_threshold=emergency_graph_threshold,
                emergency_factor_threshold=emergency_factor_threshold,
                high_count=getattr(
                    self,
                    "_adj_recent_window_high_stale_count",
                    0,
                ),
                low_count=getattr(
                    self,
                    "_adj_recent_window_low_stale_count",
                    0,
                ),
            )
            recent_episode_window = int(control_state["window"])
            recent_window_shrunk = float(control_state["shrunk"])
            recent_window_recovered = float(control_state["recovered"])
            recent_window_emergency_shrunk = float(
                control_state["emergency_shrunk"]
            )
            recent_window_high_stale_count = float(
                control_state["high_count"]
            )
            recent_window_low_stale_count = float(
                control_state["low_count"]
            )
            self._adj_dynamic_recent_episode_window = recent_episode_window
            self._adj_recent_window_high_stale_count = int(
                control_state["high_count"]
            )
            self._adj_recent_window_low_stale_count = int(
                control_state["low_count"]
            )
        early_stop_triggered = 0.0
        candidate_residual_samples = []
        candidate_residual_infos = []
        candidate_residual_epochs_ran = 0
        last_epoch_raw_control_population = None
        last_epoch_trusted_control_population = None
        last_epoch_control_population = None
        update_raw_control_populations = []
        update_trusted_control_populations = []
        update_control_populations = []
        epochs_ran = 0
        sample_episode_count = np.nan
        sample_trained_episode_count = np.nan
        sample_dropped_episode_count = np.nan
        sample_unique_generation_count = np.nan
        sample_selected_chunk_count = np.nan
        sample_yielded_chunk_count = np.nan
        sample_trained_chunk_count = np.nan
        sample_bounded_deferred_chunk_count = np.nan
        sample_recoverable_noop_chunk_count = np.nan
        sample_dropped_chunk_count = np.nan
        sample_duplicate_chunk_count = np.nan
        sample_remainder_chunk_count = np.nan
        sample_partition_valid = np.nan
        sample_recent_fraction = np.nan
        sample_base_episode_count = np.nan
        sample_outcome_contrast_augmented_count = np.nan
        sample_outcome_positive_available = np.nan
        sample_outcome_negative_available = np.nan
        sample_outcome_positive_episode_count = np.nan
        sample_outcome_negative_episode_count = np.nan
        sample_outcome_class_complete = np.nan
        sample_outcome_support_exhausted = np.nan
        sample_outcome_credit_enabled = np.nan
        sample_outcome_cached_selection_reused = np.nan
        sample_outcome_support_round = np.nan
        sample_outcome_cross_update_reuse_count = np.nan
        sample_outcome_positive_available_count = np.nan
        sample_outcome_negative_available_count = np.nan
        sample_outcome_base_positive_count = np.nan
        sample_outcome_base_negative_count = np.nan
        sample_outcome_augmented_positive_count = np.nan
        sample_outcome_augmented_negative_count = np.nan
        sample_outcome_base_age_mean = np.nan
        sample_outcome_base_age_max = np.nan
        sample_outcome_augmented_age_mean = np.nan
        sample_outcome_augmented_age_max = np.nan
        sample_outcome_positive_support_generation = np.nan
        sample_outcome_negative_support_generation = np.nan
        sample_outcome_positive_support_age = np.nan
        sample_outcome_negative_support_age = np.nan
        sample_outcome_support_used_count = np.nan
        sample_outcome_support_used_fraction = np.nan
        sample_outcome_full_buffer_baseline = np.nan
        sample_outcome_base_cohort_baseline = np.nan
        sample_outcome_trained_cohort_baseline = np.nan
        sample_outcome_full_trained_baseline_gap = np.nan
        sample_outcome_trained_capture_episode_count = np.nan
        sample_outcome_cohort_centered_sum = np.nan
        sample_outcome_cohort_center_error = np.nan
        sample_outcome_cohort_center_valid = np.nan
        sample_outcome_positive_gate_episode_count = np.nan
        sample_outcome_negative_gate_episode_count = np.nan
        sample_outcome_positive_credit_episode_count = np.nan
        sample_outcome_negative_credit_episode_count = np.nan
        sample_outcome_signed_scaling_version = np.nan
        sample_outcome_graph_advantage_source_ready_fraction = np.nan
        sample_outcome_graph_confidence_mean = np.nan
        sample_outcome_graph_confidence_std = np.nan
        sample_outcome_graph_confidence_p50 = np.nan
        sample_outcome_graph_confidence_p95 = np.nan
        sample_outcome_graph_confidence_max = np.nan
        sample_outcome_positive_graph_confidence_mean = np.nan
        sample_outcome_positive_graph_confidence_max = np.nan
        sample_outcome_negative_graph_confidence_mean = np.nan
        sample_outcome_negative_graph_confidence_max = np.nan
        sample_outcome_graph_advantage_positive_fraction = np.nan
        sample_outcome_graph_advantage_negative_fraction = np.nan
        sample_outcome_graph_advantage_zero_fraction = np.nan
        sample_outcome_positive_zero_confidence_fraction = np.nan
        sample_outcome_negative_zero_confidence_fraction = np.nan
        sample_outcome_gate_to_credit_drop_fraction = np.nan
        sample_outcome_preclip_positive_mass = np.nan
        sample_outcome_preclip_negative_mass = np.nan
        sample_outcome_postclip_positive_mass = np.nan
        sample_outcome_postclip_negative_mass = np.nan
        sample_outcome_positive_clip_fraction = np.nan
        sample_outcome_negative_clip_fraction = np.nan
        sample_outcome_generation_update_count = np.nan
        sample_outcome_slot_overwrite_count = np.nan
        sample_outcome_generation_conflict_count = np.nan
        sample_outcome_invalid_used_state_count = np.nan

        self._adj_outcome_support_round = int(
            getattr(self, "_adj_outcome_support_round", 0)
        ) + 1
        outcome_support_round = self._adj_outcome_support_round
        if self.algorithm_name == "sddfg":
            self.adj_buffer.set_pair_pending_clock(
                adjacency_update_index=outcome_support_round,
                behavior_policy_version=int(getattr(
                    self.adj_network,
                    "candidate_policy_version",
                    outcome_support_round,
                )),
            )
        standard_pair_trained_generation_keys = {
            str(policy_id): set()
            for policy_id in self.policy_info.keys()
        }
        standard_adj_transaction_commit_count = 0

        for epoch_idx in range(self.adj_train_epochs):
            epoch_raw_control_populations = []
            epoch_trusted_control_populations = []
            epoch_control_populations = []
            for p_id in self.policy_info.keys():
                policy_buffer = self.adj_buffer.policy_buffers[p_id]
                # 从该policy对应的邻接缓冲中，按“连续片段+mini-batch”方式迭代采样
                data_generator = self.adj_buffer.policy_buffers[p_id].sample_inds(data_chunk_length,
                                                                                  self.num_mini_batch,
                                                                                  recent_episode_window=recent_episode_window,
                                                                                  outcome_support_round=outcome_support_round)
                trained_chunk_count = 0
                bounded_deferred_chunk_count = 0
                recoverable_noop_chunk_count = 0
                epoch_policy_transaction_count = 0
                for partition_idx, sample in enumerate(data_generator):
                    # 调用trainer的邻接训练：内部是PPO剪切 + 熵正则
                    train_adj_info, new_priorities, idxes = (
                        self._train_adj_on_batch_with_exact_failure_logging(
                            sample=sample,
                            use_adj_init=self.use_adj_init,
                            adjacency_update_round=outcome_support_round,
                            consume_factor_credit_observations=(
                                epoch_idx == 0
                            ),
                            policy_id=p_id,
                            transaction_sequence_index=(
                                self._adj_transaction_sequence_index
                            ),
                            ppo_epoch_index=epoch_idx,
                            partition_index=partition_idx,
                        )
                    )
                    exact_bounded_deferred = float(train_adj_info.get(
                        "_strict_pair_exact_bounded_deferred", 0.0
                    )) == 1.0
                    optimizer_recoverable_noop = float(train_adj_info.get(
                        "_pair_optimizer_recoverable_noop", 0.0
                    )) == 1.0
                    if exact_bounded_deferred or optimizer_recoverable_noop:
                        if self.algorithm_name != "sddfg":
                            raise RuntimeError(
                                "non-SDDFG adjacency transaction returned a "
                                "recoverable strict-pair no-op"
                            )
                        # The trainer has restored parameters, Adam, lifecycle,
                        # boundary state, and RNG before raising.  Count this
                        # replay partition as explicitly processed, but do not
                        # create a committed transaction row, advance the
                        # transaction sequence, consume generation evidence,
                        # or contribute a clamp-control population.
                        epoch_policy_transaction_count += 1
                        chunk_count = int(np.asarray(sample[0]).shape[0])
                        trained_chunk_count += chunk_count
                        if exact_bounded_deferred:
                            bounded_deferred_chunk_count += chunk_count
                        else:
                            recoverable_noop_chunk_count += chunk_count
                        for metric_name, metric_value in (
                                (name, value)
                                for name, value in train_adj_info.items()
                                if not name.startswith("_")
                        ):
                            self.train_adj_infos.setdefault(
                                metric_name, []
                            ).append(float(metric_value))
                        raw_control_population = {
                            "population": "raw",
                            "graph_numerator": 0.0,
                            "graph_denominator": 0.0,
                            "graph_ratio": 0.0,
                            "graph_valid": 0.0,
                            "factor_numerator": 0.0,
                            "factor_denominator": 0.0,
                            "factor_ratio": 0.0,
                            "factor_valid": 0.0,
                        }
                        trusted_control_population = dict(
                            raw_control_population
                        )
                        trusted_control_population["population"] = "trusted"
                        control_population = (
                            trusted_control_population
                            if stale_trust_control_enabled
                            else raw_control_population
                        )
                        epoch_raw_control_populations.append(
                            raw_control_population
                        )
                        epoch_trusted_control_populations.append(
                            trusted_control_population
                        )
                        epoch_control_populations.append(control_population)
                        update_raw_control_populations.append(
                            raw_control_population
                        )
                        update_trusted_control_populations.append(
                            trusted_control_population
                        )
                        update_control_populations.append(control_population)
                        continue
                    if self.algorithm_name == "sddfg":
                        transaction_row = self._record_adj_transaction(
                            train_adj_info=train_adj_info,
                            policy_buffer=policy_buffer,
                            sample=sample,
                            adjacency_update_round=outcome_support_round,
                            ppo_epoch_index=epoch_idx,
                            policy_id=p_id,
                            partition_index=partition_idx,
                        )
                        adj_transaction_records.append(transaction_row)
                        standard_adj_transaction_commit_count += 1
                        epoch_policy_transaction_count += 1
                        if (
                                bool(getattr(
                                    self.args,
                                    "pair_bounded_pending_evidence",
                                    False,
                                ))
                                and float(train_adj_info.get(
                                    "pair_optimizer_transaction_nonzero_pair",
                                    0.0,
                                )) == 1.0
                                and float(train_adj_info.get(
                                    "pair_optimizer_class_complete",
                                    0.0,
                                )) == 1.0
                                and float(train_adj_info.get(
                                    "pair_optimizer_transaction_"
                                    "rollback_occurred",
                                    0.0,
                                )) == 0.0):
                            standard_pair_trained_generation_keys[
                                str(p_id)
                            ].update(
                                policy_buffer
                                .standard_pair_transaction_generation_keys()
                            )
                    trained_chunk_count += int(np.asarray(sample[0]).shape[0])
                    if (
                            epoch_idx == 0
                            and float(train_adj_info.get(
                                "capture_candidate_identity_unsatisfied_target_count",
                                0.0,
                            )) > 0.0):
                        # Keep the exact current-update sample.  It may be used
                        # only if PPO clipping stops later configured epochs.
                        # This is finite same-update completion, not historical
                        # cross-update outcome replay.
                        candidate_residual_samples.append(sample)
                    # 聚合指标
                    for k, v in train_adj_info.items():
                        if (
                                k.startswith("_adj_control_")
                                or k.startswith(
                                    "pair_optimizer_transaction_"
                                )
                                or k in (
                                    "clamp_ratio",
                                    "factor_clamp_ratio",
                                    "trusted_clamp_ratio",
                                    "trusted_factor_clamp_ratio",
                                )):
                            continue
                        self.train_adj_infos.setdefault(k, []).append(v)
                    raw_control_population = select_adj_control_population(
                        train_adj_info,
                        use_stale_trust=False,
                    )
                    trusted_control_population = select_adj_control_population(
                        train_adj_info,
                        use_stale_trust=True,
                    )
                    control_population = (
                        trusted_control_population
                        if stale_trust_control_enabled
                        else raw_control_population
                    )
                    epoch_raw_control_populations.append(
                        raw_control_population
                    )
                    epoch_trusted_control_populations.append(
                        trusted_control_population
                    )
                    epoch_control_populations.append(control_population)
                    update_raw_control_populations.append(
                        raw_control_population
                    )
                    update_trusted_control_populations.append(
                        trusted_control_population
                    )
                    update_control_populations.append(control_population)
                if self.algorithm_name == "sddfg":
                    selected_chunk_count_for_transactions = int(getattr(
                        policy_buffer,
                        "last_sample_selected_chunk_count",
                        -1,
                    ))
                    class_complete_for_transactions = bool(float(getattr(
                        policy_buffer,
                        "last_sample_pair_optimizer_atomic_partition",
                        np.nan,
                    )))
                    expected_transaction_count = (
                        _expected_adj_transaction_partition_count(
                            class_complete=class_complete_for_transactions,
                            selected_chunk_count=(
                                selected_chunk_count_for_transactions
                            ),
                            num_mini_batch=self.num_mini_batch,
                        )
                    )
                    if (
                            epoch_policy_transaction_count
                            != expected_transaction_count):
                        raise RuntimeError(
                            "per-epoch adjacency transaction count does not "
                            "match replay partitions: expected={}, actual={}, "
                            "class_complete={}, epoch={}".format(
                                expected_transaction_count,
                                epoch_policy_transaction_count,
                                int(class_complete_for_transactions),
                                epoch_idx,
                            )
                        )
                sample_episode_count = float(
                    getattr(policy_buffer, "last_sample_episode_count", np.nan)
                )
                sample_trained_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_trained_episode_count",
                    np.nan,
                ))
                sample_dropped_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_dropped_episode_count",
                    np.nan,
                ))
                sample_unique_generation_count = float(getattr(
                    policy_buffer,
                    "last_sample_unique_generation_count",
                    np.nan,
                ))
                sample_selected_chunk_count = float(getattr(
                    policy_buffer,
                    "last_sample_selected_chunk_count",
                    np.nan,
                ))
                sample_yielded_chunk_count = float(getattr(
                    policy_buffer,
                    "last_sample_yielded_chunk_count",
                    np.nan,
                ))
                sample_trained_chunk_count = float(trained_chunk_count)
                sample_bounded_deferred_chunk_count = float(
                    bounded_deferred_chunk_count
                )
                sample_recoverable_noop_chunk_count = float(
                    recoverable_noop_chunk_count
                )
                sample_dropped_chunk_count = float(getattr(
                    policy_buffer,
                    "last_sample_dropped_chunk_count",
                    np.nan,
                ))
                sample_duplicate_chunk_count = float(getattr(
                    policy_buffer,
                    "last_sample_duplicate_chunk_count",
                    np.nan,
                ))
                sample_remainder_chunk_count = float(getattr(
                    policy_buffer,
                    "last_sample_remainder_chunk_count",
                    np.nan,
                ))
                sample_partition_valid = float(getattr(
                    policy_buffer,
                    "last_sample_partition_valid",
                    np.nan,
                ))
                sample_base_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_base_episode_count",
                    np.nan,
                ))
                sample_outcome_contrast_augmented_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_contrast_augmented_count",
                    np.nan,
                ))
                sample_outcome_positive_available = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_available",
                    np.nan,
                ))
                sample_outcome_negative_available = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_available",
                    np.nan,
                ))
                sample_outcome_positive_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_episode_count",
                    np.nan,
                ))
                sample_outcome_negative_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_episode_count",
                    np.nan,
                ))
                sample_outcome_class_complete = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_class_complete",
                    np.nan,
                ))
                sample_outcome_support_exhausted = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_exhausted",
                    np.nan,
                ))
                sample_outcome_credit_enabled = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_credit_enabled",
                    np.nan,
                ))
                sample_outcome_cached_selection_reused = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cached_selection_reused",
                    np.nan,
                ))
                sample_outcome_support_round = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_round",
                    np.nan,
                ))
                sample_outcome_cross_update_reuse_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cross_update_reuse_count",
                    np.nan,
                ))
                sample_outcome_positive_available_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_available_count",
                    np.nan,
                ))
                sample_outcome_negative_available_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_available_count",
                    np.nan,
                ))
                sample_outcome_base_positive_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_positive_count",
                    np.nan,
                ))
                sample_outcome_base_negative_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_negative_count",
                    np.nan,
                ))
                sample_outcome_augmented_positive_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_positive_count",
                    np.nan,
                ))
                sample_outcome_augmented_negative_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_negative_count",
                    np.nan,
                ))
                sample_pair_positive_available = float(getattr(
                    policy_buffer,
                    "last_sample_pair_positive_available",
                    np.nan,
                ))
                sample_pair_negative_available = float(getattr(
                    policy_buffer,
                    "last_sample_pair_negative_available",
                    np.nan,
                ))
                sample_pair_positive_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_pair_positive_episode_count",
                    np.nan,
                ))
                sample_pair_negative_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_pair_negative_episode_count",
                    np.nan,
                ))
                sample_pair_class_complete = float(getattr(
                    policy_buffer,
                    "last_sample_pair_class_complete",
                    np.nan,
                ))
                sample_pair_support_exhausted = float(getattr(
                    policy_buffer,
                    "last_sample_pair_support_exhausted",
                    np.nan,
                ))
                sample_pair_augmented_count = float(getattr(
                    policy_buffer,
                    "last_sample_pair_augmented_count",
                    np.nan,
                ))
                sample_pair_evidence_funnel_metrics = {}
                for metric_name, attribute_name in (
                        (
                            "adj_sample_pair_evidence_funnel_version",
                            "last_sample_pair_evidence_funnel_version",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_occupied_episode_count",
                            "last_sample_pair_evidence_funnel_occupied_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_episode_count",
                            "last_sample_pair_evidence_funnel_successful_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_capture_episode_count",
                            "last_sample_pair_evidence_funnel_capture_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_episode_count",
                            "last_sample_pair_evidence_funnel_successful_capture_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_active_capture_episode_count",
                            "last_sample_pair_evidence_funnel_successful_active_capture_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_capture_episode_count",
                            "last_sample_pair_evidence_funnel_successful_candidate_capture_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_pair_evidence_episode_count",
                            "last_sample_pair_evidence_funnel_pair_evidence_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_pair_positive_episode_count",
                            "last_sample_pair_evidence_funnel_pair_positive_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_pair_negative_episode_count",
                            "last_sample_pair_evidence_funnel_pair_negative_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_without_pair_evidence_episode_count",
                            "last_sample_pair_evidence_funnel_successful_capture_without_pair_evidence_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_pair_evidence_without_capture_episode_count",
                            "last_sample_pair_evidence_funnel_pair_evidence_without_capture_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_candidate_only_not_active_episode_count",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_candidate_only_not_active_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_active_without_strict_pair_episode_count",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_active_without_strict_pair_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_unclassified_episode_count",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_unclassified_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_reject_reason_contract_valid",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_reject_reason_contract_valid",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_episode_count",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_identity_count",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_identity_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_identity_mass",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_identity_mass",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_margin_mean",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_margin_mean",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_margin_min",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_margin_min",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_margin_max",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_margin_max",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank_mean",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank_mean",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank_min",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank_min",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank_max",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank_max",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_boundary_crossed_fraction",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_boundary_crossed_fraction",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank1_fraction",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_behavior_rank1_fraction",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_terminal_capture_episode_count",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_terminal_capture_episode_count",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_last_capture_to_terminal_step_mean",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_last_capture_to_terminal_step_mean",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_last_capture_to_terminal_step_min",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_last_capture_to_terminal_step_min",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_capture_gap_last_capture_to_terminal_step_max",
                            "last_sample_pair_evidence_funnel_successful_capture_gap_last_capture_to_terminal_step_max",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_successful_candidate_gap_context_contract_valid",
                            "last_sample_pair_evidence_funnel_successful_candidate_gap_context_contract_valid",
                        ),
                        (
                            "adj_sample_pair_evidence_funnel_contract_valid",
                            "last_sample_pair_evidence_funnel_contract_valid",
                        )):
                    sample_pair_evidence_funnel_metrics[metric_name] = float(
                        getattr(policy_buffer, attribute_name, np.nan)
                    )
                if self.algorithm_name == "sddfg" and epoch_idx == 0:
                    buffer_episode_rows = list(getattr(
                        policy_buffer,
                        "last_sample_pair_evidence_episode_rows",
                        (),
                    ))
                    occupied_count = int(round(
                        sample_pair_evidence_funnel_metrics[
                            "adj_sample_pair_evidence_funnel_occupied_episode_count"
                        ]
                    ))
                    if len(buffer_episode_rows) != occupied_count:
                        raise RuntimeError(
                            "episode-level funnel rows do not cover the "
                            "occupied replay population"
                        )
                    for buffer_row in buffer_episode_rows:
                        episode_row = {
                            "run_id": os.path.basename(str(self.run_dir)),
                            "env_step": int(self.total_env_steps),
                            "adjacency_update_round": int(
                                outcome_support_round
                            ),
                            "policy_id": str(p_id),
                            "episode_row_sequence_index": int(
                                len(pair_evidence_episode_rows)
                            ),
                        }
                        episode_row.update(dict(buffer_row))
                        pair_evidence_episode_rows.append(episode_row)
                sample_pair_optimizer_atomic_partition = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_atomic_partition",
                    np.nan,
                ))
                sample_pair_optimizer_evidence_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_evidence_episode_count",
                    np.nan,
                ))
                sample_pair_optimizer_positive_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_positive_episode_count",
                    np.nan,
                ))
                sample_pair_optimizer_negative_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_negative_episode_count",
                    np.nan,
                ))
                sample_pair_optimizer_zero_credit_filler_chunk_count = float(
                    getattr(
                        policy_buffer,
                        "last_sample_pair_optimizer_zero_credit_filler_chunk_count",
                        np.nan,
                    )
                )
                sample_pair_optimizer_pair_partition_chunk_count = float(
                    getattr(
                        policy_buffer,
                        "last_sample_pair_optimizer_pair_partition_chunk_count",
                        np.nan,
                    )
                )
                sample_pair_optimizer_partition_slot = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_partition_slot",
                    np.nan,
                ))
                sample_pair_optimizer_partition_size_min = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_partition_size_min",
                    np.nan,
                ))
                sample_pair_optimizer_partition_size_max = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_partition_size_max",
                    np.nan,
                ))
                sample_pair_optimizer_partition_size_imbalance = float(getattr(
                    policy_buffer,
                    "last_sample_pair_optimizer_partition_size_imbalance",
                    np.nan,
                ))
                sample_outcome_base_age_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_age_mean",
                    np.nan,
                ))
                sample_outcome_base_age_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_age_max",
                    np.nan,
                ))
                sample_outcome_augmented_age_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_age_mean",
                    np.nan,
                ))
                sample_outcome_augmented_age_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_age_max",
                    np.nan,
                ))
                sample_outcome_positive_support_generation = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_support_generation",
                    np.nan,
                ))
                sample_outcome_negative_support_generation = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_support_generation",
                    np.nan,
                ))
                sample_outcome_positive_support_age = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_support_age",
                    np.nan,
                ))
                sample_outcome_negative_support_age = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_support_age",
                    np.nan,
                ))
                sample_outcome_support_used_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_used_count",
                    np.nan,
                ))
                sample_outcome_support_used_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_used_fraction",
                    np.nan,
                ))
                sample_outcome_full_buffer_baseline = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_full_buffer_baseline",
                    np.nan,
                ))
                sample_outcome_base_cohort_baseline = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_cohort_baseline",
                    np.nan,
                ))
                sample_outcome_trained_cohort_baseline = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_trained_cohort_baseline",
                    np.nan,
                ))
                sample_outcome_full_trained_baseline_gap = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_full_trained_baseline_gap",
                    np.nan,
                ))
                sample_outcome_trained_capture_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_trained_capture_episode_count",
                    np.nan,
                ))
                sample_outcome_cohort_centered_sum = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cohort_centered_sum",
                    np.nan,
                ))
                sample_outcome_cohort_center_error = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cohort_center_error",
                    np.nan,
                ))
                sample_outcome_cohort_center_valid = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cohort_center_valid",
                    np.nan,
                ))
                sample_outcome_positive_gate_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_gate_episode_count",
                    np.nan,
                ))
                sample_outcome_negative_gate_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_gate_episode_count",
                    np.nan,
                ))
                sample_outcome_positive_credit_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_credit_episode_count",
                    np.nan,
                ))
                sample_outcome_negative_credit_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_credit_episode_count",
                    np.nan,
                ))
                sample_outcome_signed_scaling_version = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_signed_scaling_version",
                    np.nan,
                ))
                sample_outcome_graph_advantage_source_ready_fraction = float(
                    getattr(
                        policy_buffer,
                        "last_sample_outcome_graph_advantage_source_ready_fraction",
                        np.nan,
                    )
                )
                sample_outcome_graph_confidence_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_mean",
                    np.nan,
                ))
                sample_outcome_graph_confidence_std = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_std",
                    np.nan,
                ))
                sample_outcome_graph_confidence_p50 = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_p50",
                    np.nan,
                ))
                sample_outcome_graph_confidence_p95 = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_p95",
                    np.nan,
                ))
                sample_outcome_graph_confidence_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_max",
                    np.nan,
                ))
                sample_outcome_positive_graph_confidence_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_graph_confidence_mean",
                    np.nan,
                ))
                sample_outcome_positive_graph_confidence_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_graph_confidence_max",
                    np.nan,
                ))
                sample_outcome_negative_graph_confidence_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_graph_confidence_mean",
                    np.nan,
                ))
                sample_outcome_negative_graph_confidence_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_graph_confidence_max",
                    np.nan,
                ))
                sample_outcome_graph_advantage_positive_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_advantage_positive_fraction",
                    np.nan,
                ))
                sample_outcome_graph_advantage_negative_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_advantage_negative_fraction",
                    np.nan,
                ))
                sample_outcome_graph_advantage_zero_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_advantage_zero_fraction",
                    np.nan,
                ))
                sample_outcome_positive_zero_confidence_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_zero_confidence_fraction",
                    np.nan,
                ))
                sample_outcome_negative_zero_confidence_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_zero_confidence_fraction",
                    np.nan,
                ))
                sample_outcome_gate_to_credit_drop_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_gate_to_credit_drop_fraction",
                    np.nan,
                ))
                sample_outcome_preclip_positive_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_preclip_positive_mass",
                    np.nan,
                ))
                sample_outcome_preclip_negative_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_preclip_negative_mass",
                    np.nan,
                ))
                sample_outcome_postclip_positive_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_postclip_positive_mass",
                    np.nan,
                ))
                sample_outcome_postclip_negative_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_postclip_negative_mass",
                    np.nan,
                ))
                sample_outcome_positive_clip_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_clip_fraction",
                    np.nan,
                ))
                sample_outcome_negative_clip_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_clip_fraction",
                    np.nan,
                ))
                sample_outcome_generation_update_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_generation_update_count",
                    np.nan,
                ))
                sample_outcome_slot_overwrite_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_slot_overwrite_count",
                    np.nan,
                ))
                sample_outcome_generation_conflict_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_generation_conflict_count",
                    np.nan,
                ))
                sample_outcome_invalid_used_state_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_invalid_used_state_count",
                    np.nan,
                ))
                filled_episode_count = float(
                    max(1, getattr(policy_buffer, "filled_i", 1))
                )
                if (
                        not np.isfinite(sample_trained_episode_count)
                        or not np.isfinite(sample_dropped_episode_count)
                        or not np.isfinite(sample_unique_generation_count)):
                    raise RuntimeError(
                        "adjacency replay did not report its executed episode "
                        "population"
                    )
                if (
                        sample_trained_episode_count != sample_episode_count
                        or sample_unique_generation_count != sample_episode_count
                        or sample_dropped_episode_count != 0.0):
                    raise RuntimeError(
                        "adjacency replay selected and executed episode "
                        "populations diverged"
                    )
                chunk_contract_values = (
                    sample_selected_chunk_count,
                    sample_yielded_chunk_count,
                    sample_trained_chunk_count,
                    sample_dropped_chunk_count,
                    sample_duplicate_chunk_count,
                    sample_partition_valid,
                )
                if not all(np.isfinite(value) for value in chunk_contract_values):
                    raise RuntimeError(
                        "adjacency replay did not report its executed chunk "
                        "population"
                    )
                if (
                        sample_selected_chunk_count
                        != sample_yielded_chunk_count
                        or sample_selected_chunk_count
                        != sample_trained_chunk_count
                        or sample_dropped_chunk_count != 0.0
                        or sample_duplicate_chunk_count != 0.0
                        or sample_partition_valid != 1.0):
                    raise RuntimeError(
                        "adjacency replay selected, yielded, and trained chunk "
                        "populations diverged"
                    )
                if np.isfinite(sample_episode_count):
                    sample_recent_fraction = (
                        sample_episode_count / filled_episode_count
                    )
            # 首轮训练之后就不再使用“弱化初始化”分支
            if standard_adj_transaction_commit_count > 0:
                self.use_adj_init = False
            epochs_ran = epoch_idx + 1
            last_epoch_raw_control_population = (
                aggregate_adj_control_populations(
                    epoch_raw_control_populations
                )
            )
            last_epoch_trusted_control_population = (
                aggregate_adj_control_populations(
                    epoch_trusted_control_populations
                )
            )
            last_epoch_control_population = (
                aggregate_adj_control_populations(
                    epoch_control_populations
                )
            )
            if should_stop_adj_ppo(
                    epochs_ran=epochs_ran,
                    configured_epochs=self.adj_train_epochs,
                    graph_ratio=last_epoch_control_population["graph_ratio"],
                    factor_ratio=last_epoch_control_population["factor_ratio"],
                    graph_stop_ratio=graph_clip_stop_ratio,
                    factor_stop_ratio=factor_clip_stop_ratio,
                    min_epochs=min_ppo_epochs):
                early_stop_triggered = 1.0
                break

        if (
                self.algorithm_name == "sddfg"
                and bool(getattr(
                    self.args,
                    "pair_bounded_pending_evidence",
                    False,
                ))):
            for policy_id, keys in sorted(
                    standard_pair_trained_generation_keys.items()):
                if keys:
                    self.adj_buffer.policy_buffers[
                        policy_id
                    ].commit_standard_pair_transaction(
                        keys,
                        outcome_support_round,
                    )
            # Normal adjacency epochs above may have advanced the policy
            # version. Pending eligibility and policy age must be evaluated
            # against that current version, not the pre-update snapshot.
            self.adj_buffer.set_pair_pending_clock(
                adjacency_update_index=outcome_support_round,
                behavior_policy_version=int(getattr(
                    self.adj_network,
                    "candidate_policy_version",
                    outcome_support_round,
                )),
            )
            prepared_by_policy = (
                self.adj_buffer.prepare_pair_pending_training_batches(
                    expected_ppo_epochs=self.adj_train_epochs
                )
            )
            for policy_id in sorted(prepared_by_policy):
                prepared = prepared_by_policy[policy_id]
                if prepared is None:
                    continue
                pending_rows = self._run_pair_pending_outer_transaction(
                    policy_id=policy_id,
                    policy_buffer=(
                        self.adj_buffer.policy_buffers[policy_id]
                    ),
                    prepared=prepared,
                    adjacency_update_round=outcome_support_round,
                    graph_clip_stop_ratio=graph_clip_stop_ratio,
                    factor_clip_stop_ratio=factor_clip_stop_ratio,
                    min_ppo_epochs=min_ppo_epochs,
                )
                adj_transaction_records.extend(pending_rows)

        if self.algorithm_name == "sddfg":
            pending_update_rows = []
            for policy_id, diagnostics in sorted(
                    self.adj_buffer.pair_pending_update_diagnostics().items()):
                pending_update_rows.append({
                    "run_id": os.path.basename(
                        str(self.run_dir).rstrip(os.sep)
                    ),
                    "env_step": int(self.total_env_steps),
                    "adjacency_update_round": int(outcome_support_round),
                    "policy_id": str(policy_id),
                    "diagnostic_version": int(
                        diagnostics["diagnostic_version"]
                    ),
                    "enabled": int(bool(getattr(
                        self.args,
                        "pair_bounded_pending_evidence",
                        False,
                    ))),
                    "max_adj_updates": int(getattr(
                        self.args,
                        "pair_pending_max_adj_updates",
                        0,
                    )),
                    "new_snapshot_count": int(
                        diagnostics["new_snapshot_count"]
                    ),
                    "available_in_replay_count": int(
                        diagnostics["available_in_replay_count"]
                    ),
                    "available_positive_count": int(
                        diagnostics["available_positive_count"]
                    ),
                    "available_negative_count": int(
                        diagnostics["available_negative_count"]
                    ),
                    "pending_positive_count": int(
                        diagnostics["pending_positive_count"]
                    ),
                    "pending_negative_count": int(
                        diagnostics["pending_negative_count"]
                    ),
                    "expired_by_ttl_count": int(
                        diagnostics["expired_by_ttl_count"]
                    ),
                    "prepared_count": int(
                        diagnostics["prepared_count"]
                    ),
                    "prepared_this_update_count": int(
                        diagnostics["prepared_this_update_count"]
                    ),
                    "aborted_this_update_count": int(
                        diagnostics["aborted_this_update_count"]
                    ),
                    "rolled_back_this_update_count": int(
                        diagnostics["rolled_back_this_update_count"]
                    ),
                    "committed_count": int(
                        diagnostics["committed_count"]
                    ),
                    "committed_this_update_count": int(
                        diagnostics["committed_this_update_count"]
                    ),
                    "class_complete_from_pending_count": int(
                        diagnostics[
                            "class_complete_from_pending_count"
                        ]
                    ),
                    "pair_only_transaction_count": int(
                        diagnostics["pair_only_transaction_count"]
                    ),
                    "current_replay_pending_overlap_count": int(
                        diagnostics[
                            "current_replay_pending_overlap_count"
                        ]
                    ),
                    "pending_age_mean": float(
                        diagnostics["pending_age_mean"]
                    ),
                    "pending_age_max": int(
                        diagnostics["pending_age_max"]
                    ),
                    "policy_age_mean": float(
                        diagnostics["policy_age_mean"]
                    ),
                    "policy_age_max": int(
                        diagnostics["policy_age_max"]
                    ),
                    "positive_stale_trust": float(
                        diagnostics["positive_stale_trust"]
                    ),
                    "negative_stale_trust": float(
                        diagnostics["negative_stale_trust"]
                    ),
                    "raw_positive_mass": float(
                        diagnostics["raw_positive_mass"]
                    ),
                    "raw_negative_mass": float(
                        diagnostics["raw_negative_mass"]
                    ),
                    "effective_positive_mass": float(
                        diagnostics["effective_positive_mass"]
                    ),
                    "effective_negative_mass": float(
                        diagnostics["effective_negative_mass"]
                    ),
                    "zero_target_commit_count": int(
                        diagnostics["zero_target_commit_count"]
                    ),
                    "zero_target_abort_count": int(
                        diagnostics["zero_target_abort_count"]
                    ),
                    "zero_gradient_abort_count": int(
                        diagnostics["zero_gradient_abort_count"]
                    ),
                    "early_stop_abort_count": int(
                        diagnostics["early_stop_abort_count"]
                    ),
                    "reused_after_commit_count": int(
                        diagnostics["reused_after_commit_count"]
                    ),
                    "expired_by_provenance_count": int(
                        diagnostics["expired_by_provenance_count"]
                    ),
                    "expired_by_population_mismatch_count": int(
                        diagnostics[
                            "expired_by_population_mismatch_count"
                        ]
                    ),
                    "payload_contract_valid": float(
                        diagnostics["payload_contract_valid"]
                    ),
                    "stale_contract_valid": float(
                        diagnostics["stale_contract_valid"]
                    ),
                    "mass_contract_valid": float(
                        diagnostics["mass_contract_valid"]
                    ),
                    "objective_scope_contract_valid": float(
                        diagnostics["objective_scope_contract_valid"]
                    ),
                    "atomic_rollback_contract_valid": float(
                        diagnostics["atomic_rollback_contract_valid"]
                    ),
                    "checkpoint_contract_valid": float(
                        diagnostics["checkpoint_contract_valid"]
                    ),
                })
            self._append_fixed_rows_csv(
                "progress_train_pair_pending_update.csv",
                pending_update_rows,
            )
            episode_row_keys = [
                (
                    int(row["adjacency_update_round"]),
                    str(row["policy_id"]),
                    int(row["episode_generation"]),
                )
                for row in pair_evidence_episode_rows
            ]
            if len(set(episode_row_keys)) != len(episode_row_keys):
                raise RuntimeError(
                    "episode-level pair evidence diagnostics contain a "
                    "duplicate replay generation"
                )
            self._append_fixed_rows_csv(
                "progress_train_pair_evidence_episode.csv",
                pair_evidence_episode_rows,
            )
            _validate_adj_transaction_update_records(
                adj_transaction_records,
                sequence_start=adj_transaction_sequence_start,
            )
            if (
                    int(self._adj_transaction_sequence_index)
                    != adj_transaction_sequence_start
                    + len(adj_transaction_records)):
                raise RuntimeError(
                    "persisted adjacency transaction count does not match "
                    "standard Adam transaction count"
                )

        control_contract_valid = validate_adj_control_application(
            use_stale_trust=stale_trust_control_enabled,
            raw_graph_ratio=last_epoch_raw_control_population["graph_ratio"],
            raw_factor_ratio=last_epoch_raw_control_population["factor_ratio"],
            trusted_graph_ratio=(
                last_epoch_trusted_control_population["graph_ratio"]
            ),
            trusted_factor_ratio=(
                last_epoch_trusted_control_population["factor_ratio"]
            ),
            control_graph_ratio=last_epoch_control_population["graph_ratio"],
            control_factor_ratio=last_epoch_control_population["factor_ratio"],
            epochs_ran=epochs_ran,
            configured_epochs=self.adj_train_epochs,
            early_stop_triggered=early_stop_triggered,
            graph_stop_ratio=graph_clip_stop_ratio,
            factor_stop_ratio=factor_clip_stop_ratio,
            min_epochs=min_ppo_epochs,
        )

        remaining_candidate_epochs = max(
            int(self.adj_train_epochs) - int(epochs_ran),
            0,
        )
        if (
                early_stop_triggered > 0.0
                and remaining_candidate_epochs > 0
                and candidate_residual_samples):
            for _ in range(remaining_candidate_epochs):
                candidate_residual_epochs_ran += 1
                for sample in candidate_residual_samples:
                    residual_info, _, _ = self.trainer.train_adj_on_batch(
                        sample,
                        self.use_adj_init,
                        adj_update_round=outcome_support_round,
                        candidate_residual_only=True,
                    )
                    candidate_residual_infos.append(residual_info)

        self.train_adj_infos.setdefault("adj_ppo_epochs_ran", []).append(
            float(epochs_ran)
        )
        self.train_adj_infos.setdefault(
            "adj_ppo_early_stop_triggered", []
        ).append(float(early_stop_triggered))
        self.train_adj_infos.setdefault(
            "adj_ppo_last_epoch_clip_ratio", []
        ).append(float(last_epoch_raw_control_population["graph_ratio"]))
        self.train_adj_infos.setdefault(
            "adj_ppo_last_epoch_factor_clip_ratio", []
        ).append(float(last_epoch_raw_control_population["factor_ratio"]))
        self.train_adj_infos.setdefault(
            "adj_ppo_last_epoch_control_clip_ratio", []
        ).append(float(last_epoch_control_population["graph_ratio"]))
        self.train_adj_infos.setdefault(
            "adj_ppo_last_epoch_control_factor_clip_ratio", []
        ).append(float(last_epoch_control_population["factor_ratio"]))
        for metric_key, metric_value in (
                (
                    "adj_ppo_last_epoch_control_graph_numerator",
                    last_epoch_control_population["graph_numerator"],
                ),
                (
                    "adj_ppo_last_epoch_control_graph_denominator",
                    last_epoch_control_population["graph_denominator"],
                ),
                (
                    "adj_ppo_last_epoch_control_factor_numerator",
                    last_epoch_control_population["factor_numerator"],
                ),
                (
                    "adj_ppo_last_epoch_control_factor_denominator",
                    last_epoch_control_population["factor_denominator"],
                ),
                (
                    "adj_ppo_last_epoch_control_graph_valid",
                    last_epoch_control_population["graph_valid"],
                ),
                (
                    "adj_ppo_last_epoch_control_factor_valid",
                    last_epoch_control_population["factor_valid"],
                )):
            self.train_adj_infos.setdefault(metric_key, []).append(
                float(metric_value)
            )
        self.train_adj_infos.setdefault(
            "adj_ppo_control_uses_trusted_population", []
        ).append(float(stale_trust_control_enabled))
        self.train_adj_infos.setdefault(
            "adj_control_runtime_contract_valid", []
        ).append(float(control_contract_valid))
        self.train_adj_infos.setdefault(
            "adj_ppo_clip_stop_ratio", []
        ).append(float(graph_clip_stop_ratio))
        self.train_adj_infos.setdefault(
            "adj_ppo_factor_clip_stop_ratio", []
        ).append(float(factor_clip_stop_ratio))
        self.train_adj_infos.setdefault("adj_ppo_min_epochs", []).append(
            float(min_ppo_epochs)
        )
        residual_committed = [
            info for info in candidate_residual_infos
            if float(info.get(
                "capture_candidate_identity_unsatisfied_target_count",
                0.0,
            )) > 0.0
        ]
        residual_skipped = sum(
            float(info.get("candidate_residual_skipped_satisfied", 0.0))
            for info in candidate_residual_infos
        )
        self.train_adj_infos.setdefault(
            "candidate_residual_epochs_ran", []
        ).append(float(candidate_residual_epochs_ran))
        self.train_adj_infos.setdefault(
            "candidate_residual_sample_count", []
        ).append(float(len(candidate_residual_infos)))
        self.train_adj_infos.setdefault(
            "candidate_residual_committed_count", []
        ).append(float(len(residual_committed)))
        self.train_adj_infos.setdefault(
            "candidate_residual_skipped_satisfied_count", []
        ).append(float(residual_skipped))
        for metric_name in (
                "candidate_residual_optimizer_isolated",
                "candidate_residual_inactive_parameter_count",
                "candidate_residual_inactive_parameter_update_norm"):
            values = [
                float(info.get(metric_name, 0.0))
                for info in candidate_residual_infos
            ]
            self.train_adj_infos.setdefault(metric_name, []).append(
                float(np.mean(values)) if values else 0.0
            )
        for metric_name in (
                "capture_candidate_identity_loss_contribution",
                "capture_candidate_identity_gradient_norm",
                "capture_candidate_identity_positive_optimizer_signed_margin_change_mean",
                "capture_candidate_identity_negative_optimizer_signed_margin_change_mean",
                "capture_candidate_identity_positive_optimizer_rank_improved_fraction",
                "capture_candidate_identity_negative_optimizer_rank_reduced_fraction",
                "capture_candidate_identity_positive_boundary_crossed_fraction",
                "capture_candidate_identity_negative_boundary_respected_fraction"):
            values = [
                float(info.get(metric_name, 0.0))
                for info in residual_committed
            ]
            self.train_adj_infos.setdefault(
                "candidate_residual_" + metric_name, []
            ).append(float(np.mean(values)) if values else 0.0)
        self.train_adj_infos.setdefault(
            "adj_recent_episode_window", []
        ).append(float(recent_episode_window))
        self.train_adj_infos.setdefault(
            "adj_recent_episode_window_config", []
        ).append(float(configured_recent_episode_window))
        self.train_adj_infos.setdefault(
            "adj_dynamic_recent_window_enabled", []
        ).append(float(dynamic_recent_enabled))
        self.train_adj_infos.setdefault(
            "adj_recent_window_shrunk", []
        ).append(float(recent_window_shrunk))
        self.train_adj_infos.setdefault(
            "adj_recent_window_recovered", []
        ).append(float(recent_window_recovered))
        self.train_adj_infos.setdefault(
            "adj_recent_window_emergency_shrunk", []
        ).append(float(recent_window_emergency_shrunk))
        self.train_adj_infos.setdefault(
            "adj_recent_window_high_stale_count", []
        ).append(float(recent_window_high_stale_count))
        self.train_adj_infos.setdefault(
            "adj_recent_window_low_stale_count", []
        ).append(float(recent_window_low_stale_count))
        self.train_adj_infos.setdefault(
            "adj_sample_episode_count", []
        ).append(float(sample_episode_count))
        self.train_adj_infos.setdefault(
            "adj_sample_trained_episode_count", []
        ).append(float(sample_trained_episode_count))
        self.train_adj_infos.setdefault(
            "adj_sample_dropped_episode_count", []
        ).append(float(sample_dropped_episode_count))
        self.train_adj_infos.setdefault(
            "adj_sample_unique_generation_count", []
        ).append(float(sample_unique_generation_count))
        for metric_name, metric_value in (
                ("adj_sample_selected_chunk_count", sample_selected_chunk_count),
                ("adj_sample_yielded_chunk_count", sample_yielded_chunk_count),
                ("adj_sample_trained_chunk_count", sample_trained_chunk_count),
                ("adj_sample_dropped_chunk_count", sample_dropped_chunk_count),
                ("adj_sample_duplicate_chunk_count", sample_duplicate_chunk_count),
                ("adj_sample_remainder_chunk_count", sample_remainder_chunk_count),
                ("adj_sample_partition_valid", sample_partition_valid)):
            self.train_adj_infos.setdefault(metric_name, []).append(
                float(metric_value)
            )
        self.train_adj_infos.setdefault(
            "adj_sample_recent_fraction", []
        ).append(float(sample_recent_fraction))
        self.train_adj_infos.setdefault(
            "adj_sample_bounded_deferred_chunk_count", []
        ).append(float(sample_bounded_deferred_chunk_count))
        self.train_adj_infos.setdefault(
            "adj_sample_recoverable_noop_chunk_count", []
        ).append(float(sample_recoverable_noop_chunk_count))
        for metric_name, metric_value in (
                # v5 keeps final-cohort centering and pair-atomic transactions,
                # and makes the atomic partition identity-neutral for base PPO.
                ("adj_outcome_contrast_replay_support_version", 6.0),
                ("adj_sample_base_episode_count", sample_base_episode_count),
                (
                    "adj_sample_outcome_contrast_augmented_count",
                    sample_outcome_contrast_augmented_count,
                ),
                (
                    "adj_sample_outcome_positive_available",
                    sample_outcome_positive_available,
                ),
                (
                    "adj_sample_outcome_negative_available",
                    sample_outcome_negative_available,
                ),
                (
                    "adj_sample_outcome_positive_episode_count",
                    sample_outcome_positive_episode_count,
                ),
                (
                    "adj_sample_outcome_negative_episode_count",
                    sample_outcome_negative_episode_count,
                ),
                (
                    "adj_sample_outcome_class_complete",
                    sample_outcome_class_complete,
                ),
                ("adj_sample_outcome_support_exhausted", sample_outcome_support_exhausted),
                ("adj_sample_outcome_credit_enabled", sample_outcome_credit_enabled),
                ("adj_sample_outcome_cached_selection_reused", sample_outcome_cached_selection_reused),
                ("adj_sample_outcome_support_round", sample_outcome_support_round),
                ("adj_sample_outcome_cross_update_reuse_count", sample_outcome_cross_update_reuse_count),
                ("adj_sample_outcome_positive_available_count", sample_outcome_positive_available_count),
                ("adj_sample_outcome_negative_available_count", sample_outcome_negative_available_count),
                ("adj_sample_outcome_base_positive_count", sample_outcome_base_positive_count),
                ("adj_sample_outcome_base_negative_count", sample_outcome_base_negative_count),
                ("adj_sample_outcome_augmented_positive_count", sample_outcome_augmented_positive_count),
                ("adj_sample_outcome_augmented_negative_count", sample_outcome_augmented_negative_count),
                ("adj_sample_pair_positive_available", sample_pair_positive_available),
                ("adj_sample_pair_negative_available", sample_pair_negative_available),
                ("adj_sample_pair_positive_episode_count", sample_pair_positive_episode_count),
                ("adj_sample_pair_negative_episode_count", sample_pair_negative_episode_count),
                ("adj_sample_pair_class_complete", sample_pair_class_complete),
                ("adj_sample_pair_support_exhausted", sample_pair_support_exhausted),
                ("adj_sample_pair_augmented_count", sample_pair_augmented_count),
                ("adj_sample_pair_optimizer_atomic_partition", sample_pair_optimizer_atomic_partition),
                ("adj_sample_pair_optimizer_evidence_episode_count", sample_pair_optimizer_evidence_episode_count),
                ("adj_sample_pair_optimizer_positive_episode_count", sample_pair_optimizer_positive_episode_count),
                ("adj_sample_pair_optimizer_negative_episode_count", sample_pair_optimizer_negative_episode_count),
                ("adj_sample_pair_optimizer_zero_credit_filler_chunk_count", sample_pair_optimizer_zero_credit_filler_chunk_count),
                ("adj_sample_pair_optimizer_pair_partition_chunk_count", sample_pair_optimizer_pair_partition_chunk_count),
                ("adj_sample_pair_optimizer_partition_slot", sample_pair_optimizer_partition_slot),
                ("adj_sample_pair_optimizer_partition_size_min", sample_pair_optimizer_partition_size_min),
                ("adj_sample_pair_optimizer_partition_size_max", sample_pair_optimizer_partition_size_max),
                ("adj_sample_pair_optimizer_partition_size_imbalance", sample_pair_optimizer_partition_size_imbalance),
                ("adj_sample_outcome_base_age_mean", sample_outcome_base_age_mean),
                ("adj_sample_outcome_base_age_max", sample_outcome_base_age_max),
                ("adj_sample_outcome_augmented_age_mean", sample_outcome_augmented_age_mean),
                ("adj_sample_outcome_augmented_age_max", sample_outcome_augmented_age_max),
                ("adj_sample_outcome_positive_support_generation", sample_outcome_positive_support_generation),
                ("adj_sample_outcome_negative_support_generation", sample_outcome_negative_support_generation),
                ("adj_sample_outcome_positive_support_age", sample_outcome_positive_support_age),
                ("adj_sample_outcome_negative_support_age", sample_outcome_negative_support_age),
                ("adj_sample_outcome_support_used_count", sample_outcome_support_used_count),
                ("adj_sample_outcome_support_used_fraction", sample_outcome_support_used_fraction),
                ("adj_sample_outcome_full_buffer_baseline", sample_outcome_full_buffer_baseline),
                ("adj_sample_outcome_base_cohort_baseline", sample_outcome_base_cohort_baseline),
                ("adj_sample_outcome_trained_cohort_baseline", sample_outcome_trained_cohort_baseline),
                ("adj_sample_outcome_full_trained_baseline_gap", sample_outcome_full_trained_baseline_gap),
                ("adj_sample_outcome_trained_capture_episode_count", sample_outcome_trained_capture_episode_count),
                ("adj_sample_outcome_cohort_centered_sum", sample_outcome_cohort_centered_sum),
                ("adj_sample_outcome_cohort_center_error", sample_outcome_cohort_center_error),
                ("adj_sample_outcome_cohort_center_valid", sample_outcome_cohort_center_valid),
                ("adj_sample_outcome_positive_gate_episode_count", sample_outcome_positive_gate_episode_count),
                ("adj_sample_outcome_negative_gate_episode_count", sample_outcome_negative_gate_episode_count),
                ("adj_sample_outcome_positive_credit_episode_count", sample_outcome_positive_credit_episode_count),
                ("adj_sample_outcome_negative_credit_episode_count", sample_outcome_negative_credit_episode_count),
                ("adj_sample_outcome_signed_scaling_version", sample_outcome_signed_scaling_version),
                (
                    "adj_sample_outcome_graph_advantage_source_ready_fraction",
                    sample_outcome_graph_advantage_source_ready_fraction,
                ),
                ("adj_sample_outcome_graph_confidence_mean", sample_outcome_graph_confidence_mean),
                ("adj_sample_outcome_graph_confidence_std", sample_outcome_graph_confidence_std),
                ("adj_sample_outcome_graph_confidence_p50", sample_outcome_graph_confidence_p50),
                ("adj_sample_outcome_graph_confidence_p95", sample_outcome_graph_confidence_p95),
                ("adj_sample_outcome_graph_confidence_max", sample_outcome_graph_confidence_max),
                ("adj_sample_outcome_positive_graph_confidence_mean", sample_outcome_positive_graph_confidence_mean),
                ("adj_sample_outcome_positive_graph_confidence_max", sample_outcome_positive_graph_confidence_max),
                ("adj_sample_outcome_negative_graph_confidence_mean", sample_outcome_negative_graph_confidence_mean),
                ("adj_sample_outcome_negative_graph_confidence_max", sample_outcome_negative_graph_confidence_max),
                ("adj_sample_outcome_graph_advantage_positive_fraction", sample_outcome_graph_advantage_positive_fraction),
                ("adj_sample_outcome_graph_advantage_negative_fraction", sample_outcome_graph_advantage_negative_fraction),
                ("adj_sample_outcome_graph_advantage_zero_fraction", sample_outcome_graph_advantage_zero_fraction),
                ("adj_sample_outcome_positive_zero_confidence_fraction", sample_outcome_positive_zero_confidence_fraction),
                ("adj_sample_outcome_negative_zero_confidence_fraction", sample_outcome_negative_zero_confidence_fraction),
                ("adj_sample_outcome_gate_to_credit_drop_fraction", sample_outcome_gate_to_credit_drop_fraction),
                ("adj_sample_outcome_preclip_positive_mass", sample_outcome_preclip_positive_mass),
                ("adj_sample_outcome_preclip_negative_mass", sample_outcome_preclip_negative_mass),
                ("adj_sample_outcome_postclip_positive_mass", sample_outcome_postclip_positive_mass),
                ("adj_sample_outcome_postclip_negative_mass", sample_outcome_postclip_negative_mass),
                ("adj_sample_outcome_positive_clip_fraction", sample_outcome_positive_clip_fraction),
                ("adj_sample_outcome_negative_clip_fraction", sample_outcome_negative_clip_fraction),
                ("adj_sample_outcome_generation_update_count", sample_outcome_generation_update_count),
                ("adj_sample_outcome_slot_overwrite_count", sample_outcome_slot_overwrite_count),
                ("adj_sample_outcome_generation_conflict_count", sample_outcome_generation_conflict_count),
                ("adj_sample_outcome_invalid_used_state_count", sample_outcome_invalid_used_state_count)):
            self.train_adj_infos.setdefault(metric_name, []).append(
                float(metric_value)
            )
        for metric_name, metric_value in (
                sample_pair_evidence_funnel_metrics.items()):
            self.train_adj_infos.setdefault(metric_name, []).append(
                float(metric_value)
            )
        self.train_adj_infos.setdefault(
            "adj_recent_episode_window_emergency", []
        ).append(float(getattr(self.args, "adj_recent_episode_window_emergency", 1)))
        self.train_adj_infos.setdefault(
            "adj_recent_window_emergency_stale_threshold", []
        ).append(
            float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_stale_threshold",
                    np.nan,
                )
            )
        )
        self.train_adj_infos.setdefault(
            "adj_recent_window_emergency_factor_stale_threshold", []
        ).append(
            float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_factor_stale_threshold",
                    np.nan,
                )
            )
        )
        update_raw_control_population = aggregate_adj_control_populations(
            update_raw_control_populations
        )
        update_trusted_control_population = (
            aggregate_adj_control_populations(
                update_trusted_control_populations
            )
        )
        update_control_population = aggregate_adj_control_populations(
            update_control_populations
        )
        for metric_key, metric_value in (
                (
                    "clamp_ratio",
                    update_raw_control_population["graph_ratio"],
                ),
                (
                    "factor_clamp_ratio",
                    update_raw_control_population["factor_ratio"],
                ),
                (
                    "trusted_clamp_ratio",
                    update_trusted_control_population["graph_ratio"],
                ),
                (
                    "trusted_factor_clamp_ratio",
                    update_trusted_control_population["factor_ratio"],
                )):
            self.train_adj_infos.setdefault(metric_key, []).append(
                float(metric_value)
            )
        for population_name, population in (
                ("raw", update_raw_control_population),
                ("trusted", update_trusted_control_population)):
            for component_name in (
                    "graph_numerator",
                    "graph_denominator",
                    "factor_numerator",
                    "factor_denominator",
                    "graph_valid",
                    "factor_valid"):
                metric_key = "adj_update_{}_control_{}".format(
                    population_name,
                    component_name,
                )
                self.train_adj_infos.setdefault(metric_key, []).append(
                    float(population[component_name])
                )
        self._last_adj_control_graph_ratio = float(
            update_control_population["graph_ratio"]
        )
        self._last_adj_control_factor_ratio = float(
            update_control_population["factor_ratio"]
        )
        if (
                not np.isfinite(self._last_adj_control_graph_ratio)
                or not np.isfinite(self._last_adj_control_factor_ratio)):
            raise RuntimeError(
                "adjacency update control-population ratios must be finite"
            )
        self.train_adj_infos.setdefault(
            "adj_recent_window_graph_control_ratio",
            [],
        ).append(self._last_adj_control_graph_ratio)
        self.train_adj_infos.setdefault(
            "adj_recent_window_factor_control_ratio",
            [],
        ).append(self._last_adj_control_factor_ratio)
        for metric_key, metric_value in (
                (
                    "adj_recent_window_graph_control_numerator",
                    update_control_population["graph_numerator"],
                ),
                (
                    "adj_recent_window_graph_control_denominator",
                    update_control_population["graph_denominator"],
                ),
                (
                    "adj_recent_window_factor_control_numerator",
                    update_control_population["factor_numerator"],
                ),
                (
                    "adj_recent_window_factor_control_denominator",
                    update_control_population["factor_denominator"],
                ),
                (
                    "adj_recent_window_graph_control_valid",
                    update_control_population["graph_valid"],
                ),
                (
                    "adj_recent_window_factor_control_valid",
                    update_control_population["factor_valid"],
                )):
            self.train_adj_infos.setdefault(metric_key, []).append(
                float(metric_value)
            )
        self.train_adj_infos.setdefault(
            "adj_recent_window_control_uses_trusted_population",
            [],
        ).append(float(stale_trust_control_enabled))

    """（非Q类）保存所有策略的actor/critic权重到 save_dir。"""

    def save(self):
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(), critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(), actor_save_path + '/actor.pt')

    """（QMix/VDN等）保存各policy的Q网络，以及混合器/额外网络。"""
    def save_q(self):
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 算法特定部件的保存
        if self.algorithm_name == "qtran":
            torch.save(self.trainer.eval_joint_q.state_dict(), self.save_dir + '/eval_joint_q.pt')
            torch.save(self.trainer.v.state_dict(), self.save_dir + '/v.pt')
        elif self.algorithm_name == "wqmix":
            for pid in self.policy_ids:
                policy_Q = self.central_policies[pid].q_network
                p_save_path = self.save_dir + '/' + str(pid)
                if not os.path.exists(p_save_path):
                    os.makedirs(p_save_path)
                torch.save(policy_Q.state_dict(), p_save_path + '/central_q_network.pt')
            torch.save(self.trainer.mixer.state_dict(), self.save_dir + '/mixer.pt')
            torch.save(self.trainer.central_mixer.state_dict(), self.save_dir + '/central_mixer.pt')
        else:
            torch.save(self.trainer.mixer.state_dict(), self.save_dir + '/mixer.pt')

    """（带图+中心化）保存：adj_network、各policy的rnn/q/v等多组件。"""
    def save_q_mdfg_cent(self):
        save_path = self.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        optimizer_state_getter = getattr(
            self.trainer,
            "adjacency_optimizer_checkpoint_state",
            None,
        )
        optimizer_checkpoint = None
        optimizer_state_path = None
        if callable(optimizer_state_getter):
            optimizer_checkpoint = optimizer_state_getter()
            optimizer_checkpoint.update(
                _build_sddfg_checkpoint_metadata(
                    total_env_steps=self.total_env_steps,
                    target_env_steps=self.num_env_steps,
                    checkpoint_kind=getattr(
                        self,
                        "_checkpoint_kind",
                        "periodic",
                    ),
                )
            )
            optimizer_checkpoint["pair_pending_checkpoint_version"] = 1
            runner_args = getattr(self, "args", None)
            optimizer_checkpoint["pair_pending_enabled"] = bool(getattr(
                runner_args,
                "pair_bounded_pending_evidence",
                False,
            ))
            optimizer_checkpoint["pair_pending_max_adj_updates"] = int(
                getattr(runner_args, "pair_pending_max_adj_updates", 0)
            )
            optimizer_checkpoint.update(
                _build_policy_exploration_checkpoint_contract(
                    runner_args,
                    self.policies,
                    self.policy_ids,
                )
            )
            optimizer_checkpoint.update(
                _build_sddfg_q_target_checkpoint_contract(runner_args)
            )
            optimizer_checkpoint.update(
                _build_wolfpack_reward_shaping_checkpoint_contract(
                    runner_args
                )
            )
            pending_state_getter = getattr(
                getattr(self, "adj_buffer", None),
                "pair_pending_state_dict",
                None,
            )
            if (
                    optimizer_checkpoint["pair_pending_enabled"]
                    and not callable(pending_state_getter)):
                raise RuntimeError(
                    "enabled pair pending checkpoint has no adjacency buffer "
                    "state provider"
                )
            optimizer_checkpoint["pair_pending_state"] = (
                pending_state_getter()
                if callable(pending_state_getter) else None
            )
            optimizer_state_path = os.path.join(
                save_path,
                "adj_optimizer_state.pt",
            )
            # The optimizer file is the commit marker for the whole model
            # directory. Remove an older marker before overwriting any model
            # component so an interrupted save cannot look coherent.
            if os.path.exists(optimizer_state_path):
                os.remove(optimizer_state_path)

        adj = self.adj_network
        torch.save(adj.state_dict(), save_path + '/adj_network.pt')

        # 每个policy的rnn/q/v（以及casec专有的动作表征等）
        for pid in self.policy_ids:
            p_save_path = save_path + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            rnn_Q = self.policies[pid].rnn_network
            torch.save(rnn_Q.state_dict(), p_save_path + '/rnn_network.pt')

            if self.algorithm_name == 'casec':
                if self.independent_p_q:
                    p_rnn = self.policies[pid].p_rnn_network
                    torch.save(p_rnn.state_dict(), p_save_path + '/p_rnn_network.pt')
                action_encoder = self.policies[pid].action_encoder
                torch.save(action_encoder.state_dict(), p_save_path + '/action_encoder.pt')
                p_action_repr = self.policies[pid].p_action_repr
                torch.save(p_action_repr, p_save_path + '/p_action_repr.pt')
                action_repr = self.policies[pid].action_repr
                torch.save(action_repr, p_save_path + '/action_repr.pt')
            else:
                rnn_v = self.policies[pid].rnn_critic_network
                torch.save(rnn_v.state_dict(), p_save_path + '/rnn_critic_network.pt')

            if self.use_vfunction:
                policy_vtot = self.policies[pid].vtot_network
                torch.save(policy_vtot.state_dict(), p_save_path + '/vtot_network.pt')

            for num_orders in range(1, self.highest_orders + 1):
                policy_Q = self.policies[pid].q_network[num_orders]
                torch.save(policy_Q.state_dict(), p_save_path + f'/q_network_{num_orders}.pt')
                if self.use_vfunction:
                    policy_V = self.policies[pid].v_network[num_orders]
                    torch.save(policy_V.state_dict(), p_save_path + f'/v_network_{num_orders}.pt')

        # Commit all Adam states last. Atomic appearance of this file proves
        # every model component above finished writing for this generation.
        if optimizer_checkpoint is not None:
            temporary_state_path = optimizer_state_path + ".tmp"
            try:
                torch.save(optimizer_checkpoint, temporary_state_path)
                os.replace(temporary_state_path, optimizer_state_path)
            finally:
                if os.path.exists(temporary_state_path):
                    os.remove(temporary_state_path)

    """（非Q类）从 model_dir 恢复：actor/critic 权重到当前policy。"""
    def restore(self):
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')
            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    # ==================== 修改点 A：新增通用指定目录保存函数 ====================
    def save_to_dir(self, save_dir):
        """
        将当前模型保存到指定目录。
        对 ddfg/sddfg，会复用 self.saver，也就是 save_q_mdfg_cent()。
        """
        old_save_dir = self.save_dir
        old_checkpoint_kind = getattr(
            self,
            "_checkpoint_kind",
            "periodic",
        )
        self.save_dir = str(save_dir)
        self._checkpoint_kind = "best"

        try:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            self.saver()
        finally:
            self.save_dir = old_save_dir
            self._checkpoint_kind = old_checkpoint_kind

    def save_terminal_checkpoint(self):
        """Save and mark the exact policy state after the final update."""
        old_checkpoint_kind = getattr(
            self,
            "_checkpoint_kind",
            "periodic",
        )
        self._checkpoint_kind = "terminal"
        try:
            self.saver()
        finally:
            self._checkpoint_kind = old_checkpoint_kind
        print(
            "[CHECKPOINT] terminal checkpoint committed at step={}".format(
                int(self.total_env_steps)
            )
        )

    # ==================== 修改点 B：新增 best checkpoint 保存函数 ====================
    def _safe_float(self, x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _latest_train_metric(self, key, default=np.nan):
        """
        Read a scalar from the latest adjacency or policy training result.

        Adjacency metrics such as clamp_ratio are stored as lists in
        train_adj_infos, whereas policy losses are stored as dictionaries in
        train_infos. Prefer the adjacency source because it is the authoritative
        source for graph-policy tie-break metrics.
        """
        adj_info = getattr(self, "train_adj_infos", None)
        if isinstance(adj_info, dict) and key in adj_info:
            try:
                values = np.asarray(adj_info[key], dtype=np.float64).reshape(-1)
                if values.size > 0 and np.all(np.isfinite(values)):
                    return float(values.mean())
            except Exception:
                pass

        infos = getattr(self, "train_infos", None)
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and key in info:
                    return self._safe_float(info.get(key), default)
        return default

    def _is_better_best_win(self, eval_win, eval_reward, clamp_ratio, step):
        """
        best_win 比较优先级：
        1) eval_win_rate；
        2) eval_average_episode_rewards；
        3) clamp_ratio；
        4) later step。
        """
        win_eps = 1e-12
        reward_eps = float(getattr(self.args, "best_reward_eps", 1e-3))
        clamp_eps = float(getattr(self.args, "best_clamp_eps", 1e-4))

        if not hasattr(self, "best_eval_win_rate"):
            self.best_eval_win_rate = -np.inf
        if not hasattr(self, "best_eval_win_reward"):
            self.best_eval_win_reward = -np.inf
        if not hasattr(self, "best_eval_win_clamp_ratio"):
            self.best_eval_win_clamp_ratio = np.inf
        if not hasattr(self, "best_eval_win_step"):
            self.best_eval_win_step = -1

        # 1. win_rate 更高
        if eval_win > self.best_eval_win_rate + win_eps:
            return True

        # win_rate 更低
        if eval_win < self.best_eval_win_rate - win_eps:
            return False

        # 2. win_rate 相同，reward 更高
        if eval_reward > self.best_eval_win_reward + reward_eps:
            return True

        if eval_reward < self.best_eval_win_reward - reward_eps:
            return False

        # 3. reward 接近，比较 clamp_ratio
        clamp_valid = np.isfinite(clamp_ratio)
        best_clamp_valid = np.isfinite(self.best_eval_win_clamp_ratio)

        if clamp_valid and best_clamp_valid:
            if clamp_ratio < self.best_eval_win_clamp_ratio - clamp_eps:
                return True
            if clamp_ratio > self.best_eval_win_clamp_ratio + clamp_eps:
                return False
        elif clamp_valid and not best_clamp_valid:
            return True

        # 4. 最后比较更晚 step
        return int(step) > int(self.best_eval_win_step)

    def maybe_save_best_eval(self, eval_info):
        """
        根据 eval 指标保存 best checkpoint。
        best_reward: 只按 eval_average_episode_rewards。
        best_win: eval_win_rate -> eval_average_episode_rewards -> clamp_ratio -> later step。
        """
        if eval_info is None:
            return

        eval_reward = float(eval_info.get("eval_average_episode_rewards", -np.inf))
        eval_win = float(eval_info.get("eval_win_rate", -np.inf))
        step = int(self.total_env_steps)

        if not hasattr(self, "best_eval_reward"):
            self.best_eval_reward = -np.inf

        best_root = os.path.join(str(self.run_dir), "best_models")
        if not os.path.exists(best_root):
            os.makedirs(best_root)

        # ---------- best_reward ----------
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.best_eval_reward_step = step
            reward_dir = os.path.join(best_root, "best_reward")
            self.save_to_dir(reward_dir)

            pd.DataFrame([{
                "step": step,
                "best_eval_reward": float(eval_reward),
                "eval_win_rate": float(eval_win),
            }]).to_csv(
                os.path.join(best_root, "best_reward_info.csv"),
                index=False
            )

            print(
                f"[BEST] Save best_reward checkpoint at step={step}, "
                f"eval_reward={eval_reward:.6f}, eval_win_rate={eval_win:.6f}"
            )

        # ---------- best_win ----------
        clamp_ratio = self._latest_train_metric("clamp_ratio", np.nan)

        if self._is_better_best_win(
                eval_win=eval_win,
                eval_reward=eval_reward,
                clamp_ratio=clamp_ratio,
                step=step,
        ):
            self.best_eval_win_rate = eval_win
            self.best_eval_win_reward = eval_reward
            self.best_eval_win_clamp_ratio = clamp_ratio
            self.best_eval_win_step = step

            win_dir = os.path.join(best_root, "best_win")
            self.save_to_dir(win_dir)

            pd.DataFrame([{
                "step": step,
                "best_eval_win_rate": float(eval_win),
                "eval_average_episode_rewards": float(eval_reward),
                "clamp_ratio": float(clamp_ratio) if np.isfinite(clamp_ratio) else np.nan,
            }]).to_csv(
                os.path.join(best_root, "best_win_info.csv"),
                index=False
            )

            clamp_text = f"{clamp_ratio:.6f}" if np.isfinite(clamp_ratio) else "N/A"
            print(
                f"[BEST] Save best_win checkpoint at step={step}, "
                f"eval_win_rate={eval_win:.6f}, "
                f"eval_reward={eval_reward:.6f}, "
                f"clamp_ratio={clamp_text}"
            )

    """（QMix/VDN等）从 model_dir 恢复：各policy的Q网络 & mixer。"""

    def restore_q(self):
        import os
        base_dir = str(self.model_dir) if self.model_dir is not None else ""
        if base_dir and not base_dir.endswith(os.sep):
            base_dir += os.sep

        for pid in self.policy_ids:
            path = os.path.join(base_dir, str(pid))
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(os.path.join(path, "q_network.pt"), map_location=self.device)
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)

        mixer_path = os.path.join(base_dir, "mixer.pt")
        if hasattr(self.trainer, "mixer") and self.trainer.mixer is not None and os.path.exists(mixer_path):
            policy_mixer_state_dict = torch.load(mixer_path, map_location=self.device)
            self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    """只恢复邻接网络（当pretrain_adj=True时用）。"""

    def load_adj(self):
        import os
        path_adj = str(self.model_dir) if self.model_dir is not None else ""
        if path_adj != "" and (not path_adj.endswith(os.sep)):
            path_adj = path_adj + os.sep
        adj_path = os.path.join(path_adj, "adj_network.pt")
        adj_state_dict = torch.load(adj_path, map_location=self.device)
        self.adj_network.load_state_dict(adj_state_dict)

    """（带图+中心化）从 model_dir 恢复 adj/rnn/q/v 等多组件。"""

    def restore_mdfg_cent(self):
        import os

        base_dir = str(self.model_dir) if self.model_dir is not None else ""
        if base_dir != "" and (not base_dir.endswith(os.sep)):
            base_dir = base_dir + os.sep

        optimizer_state_path = os.path.join(
            base_dir,
            "adj_optimizer_state.pt",
        )
        optimizer_checkpoint_present = _require_sddfg_optimizer_checkpoint(
            algorithm_name=self.algorithm_name,
            optimizer_state_path=optimizer_state_path,
        )
        optimizer_checkpoint = None
        if optimizer_checkpoint_present:
            optimizer_checkpoint = torch.load(
                optimizer_state_path,
                map_location=self.device,
            )
            if self.algorithm_name == "sddfg":
                _validate_sddfg_checkpoint_metadata(
                    optimizer_checkpoint
                )
                _validate_sddfg_q_target_checkpoint_contract(
                    optimizer_checkpoint,
                    self.args,
                )
                _validate_wolfpack_reward_shaping_checkpoint_contract(
                    optimizer_checkpoint,
                    self.args,
                )

        # Validate the checkpoint marker before mutating any model state.
        adj_path = os.path.join(base_dir, "adj_network.pt")
        adj_state_dict = torch.load(adj_path, map_location=self.device)
        self.adj_network.load_state_dict(adj_state_dict)
        if optimizer_checkpoint_present:
            optimizer_state_loader = getattr(
                self.trainer,
                "load_adjacency_optimizer_checkpoint_state",
                None,
            )
            if not callable(optimizer_state_loader):
                raise RuntimeError(
                    "checkpoint contains adjacency optimizer state but "
                    "the trainer cannot restore it"
                )
            optimizer_state_loader(optimizer_checkpoint)
            pending_enabled = bool(getattr(
                self.args,
                "pair_bounded_pending_evidence",
                False,
            ))
            pending_state = optimizer_checkpoint.get(
                "pair_pending_state"
            )
            if pending_enabled and pending_state is None:
                raise RuntimeError(
                    "bounded pair pending cannot resume an old checkpoint "
                    "without immutable pending state; start a fresh run"
                )
            self._restored_pair_pending_state = pending_state

            policy_rng_states = (
                _validate_policy_exploration_checkpoint_contract(
                    optimizer_checkpoint,
                    self.args,
                )
            )
            if policy_rng_states:
                missing_policy_rng = [
                    str(pid) for pid in self.policy_ids
                    if str(pid) not in policy_rng_states
                ]
                if missing_policy_rng:
                    raise RuntimeError(
                        "policy exploration checkpoint is missing RNG state "
                        "for {}".format(", ".join(missing_policy_rng))
                    )
                for pid in self.policy_ids:
                    self.policies[pid].rng.set_state(copy.deepcopy(
                        policy_rng_states[str(pid)]
                    ))

        for pid in self.policy_ids:
            policy_dir = os.path.join(base_dir, str(pid))
            print("load the pretrained model from {}".format(policy_dir))

            rnn_state_dict = torch.load(os.path.join(policy_dir, "rnn_network.pt"), map_location=self.device)
            self.policies[pid].rnn_network.load_state_dict(rnn_state_dict)

            rnn_critic_state_dict = torch.load(os.path.join(policy_dir, "rnn_critic_network.pt"),
                                               map_location=self.device)
            self.policies[pid].rnn_critic_network.load_state_dict(rnn_critic_state_dict)

            if self.use_vfunction:
                vtot_dict = torch.load(os.path.join(policy_dir, "vtot_network.pt"), map_location=self.device)
                self.policies[pid].vtot_network.load_state_dict(vtot_dict)

            for num_orders in range(1, self.highest_orders + 1):
                q_path = os.path.join(policy_dir, f"q_network_{num_orders}.pt")
                policy_q_state_dict = torch.load(q_path, map_location=self.device)
                self.policies[pid].q_network[num_orders].load_state_dict(policy_q_state_dict)

                if self.use_vfunction:
                    v_path = os.path.join(policy_dir, f"v_network_{num_orders}.pt")
                    policy_v_state_dict = torch.load(v_path, map_location=self.device)
                    self.policies[pid].v_network[num_orders].load_state_dict(policy_v_state_dict)

    def log(self):
        """（抽象）写训练与采样相关日志。子类中具体实现。"""
        raise NotImplementedError

    def log_clear(self):
        """（抽象）清理日志缓存。子类中具体实现。"""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        打印/写入“环境相关信息”（如平均奖励、成功率等），并保存到CSV中。
        :param env_info: dict，键是指标名，值是一个list（累积的值）
        :param suffix: 在键名后附加的后缀（如 'eval_'）
        """
        row = {"step": self.total_env_steps}

        for k, v in env_info.items():
            if len(v) > 0:
                mean_v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k
                row[suffix_k] = mean_v

                print(suffix_k + " is " + str(mean_v))
                if self.use_wandb:
                    wandb.log({suffix_k: mean_v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalar(suffix_k, mean_v, self.total_env_steps)

        filename = 'progress_eval.csv' if suffix == "eval_" else 'progress.csv'
        self._append_scalar_csv(filename, row)

    def log_train(self, policy_id, train_info):
        """
        写“训练指标”日志（如loss、grad_norm等），每个policy单独一列。
        :param policy_id: 哪个policy
        :param train_info: dict，训练过程的标量
        """
        row = {"step": self.total_env_steps, "policy_id": str(policy_id)}
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            row[k] = v
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalar(policy_k, v, self.total_env_steps)

        self._append_scalar_csv('progress_train.csv', row)

    def log_train_adj(self, train_adj_info):
        """
        写“邻接图训练”的指标（如rl_loss/entropy_loss/grad_norm等）。
        train_adj_info: dict，里面每个key是列表（多次迭代的值），这里取均值再写。
        """
        row = {"step": self.total_env_steps}
        for k, v in train_adj_info.items():
            if len(v) > 0:
                v = float(np.mean(v))
            else:
                v = 0.0
            row[k] = v
            # Missing diagnostics deliberately use NaN so CSV readers can
            # distinguish "not observed" from a real zero. TensorBoard rejects
            # non-finite scalars and emitted thousands of misleading warnings
            # in run62; keep the CSV sentinel but do not write it as an event.
            if np.isfinite(v):
                if self.use_wandb:
                    wandb.log({"adj/" + k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalar(
                        "adj/" + k, v, self.total_env_steps
                    )

        self._append_scalar_csv('progress_train_adj.csv', row)

    def collect_rollout(self):
        """（抽象）与环境交互、采样一整段episode，并写入buffer。子类需实现。"""
        raise NotImplementedError
