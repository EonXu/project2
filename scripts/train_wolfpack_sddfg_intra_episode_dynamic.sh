#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_wolfpack_sddfg_intra_episode_dynamic.sh [seed] [gpu] [num_env_steps]
# Environment overrides:
#   PYTHON_BIN=python CUDA_DEVICE=0 USER_NAME=local
#   ANNEAL_STEPS=2000000  # shared policy/graph LR schedule horizon
#   ADJ_ENTROPY_COEF=0.0002 ADJ_ENTROPY_ANNEAL_STEPS=500000
#   ADJ_ORDER3_BONUS_START=1.0 ADJ_ORDER3_BONUS=1.35 ADJ_ORDER3_BONUS_ANNEAL_STEPS=120000
#   ADJ_SAMPLING_TEMP_START=1.0 ADJ_SAMPLING_TEMP_FINAL=0.35 ADJ_SAMPLING_TEMP_ANNEAL_STEPS=200000
#   ADJ_MIN_ORDER3_RATIO_START=0.50 ADJ_MIN_ORDER3_RATIO_FINAL=0.72 ADJ_MIN_ORDER3_RATIO_ANNEAL_STEPS=200000
#   ADJ_MAX_ORDER3_RATIO_START=0.82 ADJ_MAX_ORDER3_RATIO_FINAL=0.82 ADJ_MAX_ORDER3_RATIO_ANNEAL_STEPS=0
#   ADJ_GREEDY_SAMPLE_PROB_START=0.0 ADJ_GREEDY_SAMPLE_PROB_FINAL=0.75 ADJ_GREEDY_SAMPLE_PROB_ANNEAL_STEPS=200000
#   ADJ_ORDER3_QUOTA_MODE=soft ADJ_ORDER3_SOFT_QUOTA_COEF=2.5
#   ADJ_TRIPLET_FEATURE_MODE=synergy ADJ_TRIPLET_BALANCE_COEF=0.75
#   USE_ADJ_ADVANTAGE_TRIPLET_SCORER=1 ADJ_TRIPLET_CREDIT_SCORE_COEF=0.5
#   USE_ADJ_TRIPLET_CREDIT_DIRECT_RANK=1 ADJ_TRIPLET_CREDIT_RANK_COEF=1.25
#   ADJ_TRIPLET_CREDIT_NEGATIVE_RANK_SCALE=0.25
#   ADJ_ORDER3_QUOTA_SCORE_FLOOR=0.45
#   ADJ_MIN_PAIR_RATIO=0.20 ADJ_ORDER_ADV_COEF=0.75
#   ADJ_ORDER_ADV_POSITIVE_ONLY=1 ADJ_ORDER_ADV_NEGATIVE_COEF=0.20
#   ADJ_ORDER_ADV_REQUIRE_POSITIVE_GRAPH_ADV=1
#   USE_ADJ_ORDER3_CREDIT_GATE=1 USE_ADJ_ORDER3_RELATIVE_CREDIT_GATE=1
#   ADJ_ORDER3_CREDIT_GATE_LOSS_SCALE=0.004 ADJ_ORDER3_CREDIT_GATE_MARGIN=0.0
#   ADJ_PPO_CLIP_STOP_RATIO=0.35 ADJ_PPO_FACTOR_CLIP_STOP_RATIO=0.35
#   USE_ADJ_PPO_STALE_TRUST=1 ADJ_PPO_STALE_TRUST_CLIP=0.20
#   ADJ_RECENT_EPISODE_WINDOW=4
#   USE_ADJ_DYNAMIC_RECENT_WINDOW=1 ADJ_RECENT_EPISODE_WINDOW_MIN=3
#   ADJ_RECENT_WINDOW_SHRINK_PATIENCE=3 ADJ_RECENT_WINDOW_RECOVER_PATIENCE=2
#   SKIP_SDDFG_PREFLIGHT=1  # only after the same revision has passed once

seed="${1:-1}"
gpu="${2:-${CUDA_DEVICE:-0}}"
num_env_steps="${3:-2000000}"
python_bin="${PYTHON_BIN:-python}"
user_name="${USER_NAME:-sddfg_dynamic}"

# Keep 20k/200k validation runs as exact LR-schedule prefixes of the intended
# 2M experiment. Override only for an explicit schedule ablation.
anneal_steps="${ANNEAL_STEPS:-2000000}"
if ! [[ "${anneal_steps}" =~ ^[0-9]+$ ]] || [ "${anneal_steps}" -le 0 ]; then
  echo "ANNEAL_STEPS must be a positive integer, got: ${anneal_steps}" >&2
  exit 2
fi

# run21 showed that the legacy graph entropy coefficient (1e-3 over a 2M
# horizon) kept the learned graph near-uniform at 200k.  Use a graph-specific
# weaker/faster entropy schedule while keeping policy/graph LR schedules on
# the common 2M clock.
adj_entropy_coef="${ADJ_ENTROPY_COEF:-0.0002}"
adj_entropy_anneal_steps="${ADJ_ENTROPY_ANNEAL_STEPS:-500000}"
if ! [[ "${adj_entropy_anneal_steps}" =~ ^[0-9]+$ ]] || [ "${adj_entropy_anneal_steps}" -le 0 ]; then
  echo "ADJ_ENTROPY_ANNEAL_STEPS must be a positive integer, got: ${adj_entropy_anneal_steps}" >&2
  exit 2
fi
adj_order3_bonus_start="${ADJ_ORDER3_BONUS_START:-1.0}"
adj_order3_bonus="${ADJ_ORDER3_BONUS:-1.35}"
adj_order3_bonus_anneal_steps="${ADJ_ORDER3_BONUS_ANNEAL_STEPS:-120000}"
adj_sampling_temp_start="${ADJ_SAMPLING_TEMP_START:-1.0}"
adj_sampling_temp_final="${ADJ_SAMPLING_TEMP_FINAL:-0.35}"
adj_sampling_temp_anneal_steps="${ADJ_SAMPLING_TEMP_ANNEAL_STEPS:-200000}"
adj_min_order3_ratio_start="${ADJ_MIN_ORDER3_RATIO_START:-0.50}"
adj_min_order3_ratio_final="${ADJ_MIN_ORDER3_RATIO_FINAL:-0.72}"
adj_min_order3_ratio_anneal_steps="${ADJ_MIN_ORDER3_RATIO_ANNEAL_STEPS:-200000}"
adj_max_order3_ratio_start="${ADJ_MAX_ORDER3_RATIO_START:-0.82}"
adj_max_order3_ratio_final="${ADJ_MAX_ORDER3_RATIO_FINAL:-0.82}"
adj_max_order3_ratio_anneal_steps="${ADJ_MAX_ORDER3_RATIO_ANNEAL_STEPS:-0}"
adj_greedy_sample_prob_start="${ADJ_GREEDY_SAMPLE_PROB_START:-0.0}"
adj_greedy_sample_prob_final="${ADJ_GREEDY_SAMPLE_PROB_FINAL:-0.75}"
adj_greedy_sample_prob_anneal_steps="${ADJ_GREEDY_SAMPLE_PROB_ANNEAL_STEPS:-200000}"
adj_greedy_sample_prob_cap="${ADJ_GREEDY_SAMPLE_PROB_CAP:-0.50}"
adj_order3_quota_mode="${ADJ_ORDER3_QUOTA_MODE:-soft}"
adj_order3_soft_quota_coef="${ADJ_ORDER3_SOFT_QUOTA_COEF:-2.5}"
adj_triplet_feature_mode="${ADJ_TRIPLET_FEATURE_MODE:-synergy}"
adj_triplet_balance_coef="${ADJ_TRIPLET_BALANCE_COEF:-0.75}"
use_adj_advantage_triplet_scorer="${USE_ADJ_ADVANTAGE_TRIPLET_SCORER:-1}"
adj_triplet_credit_ema_alpha="${ADJ_TRIPLET_CREDIT_EMA_ALPHA:-0.05}"
adj_triplet_credit_score_coef="${ADJ_TRIPLET_CREDIT_SCORE_COEF:-0.50}"
adj_triplet_credit_score_scale="${ADJ_TRIPLET_CREDIT_SCORE_SCALE:-0.05}"
use_adj_triplet_credit_direct_rank="${USE_ADJ_TRIPLET_CREDIT_DIRECT_RANK:-1}"
adj_triplet_credit_rank_coef="${ADJ_TRIPLET_CREDIT_RANK_COEF:-1.25}"
adj_triplet_credit_min_multiplier="${ADJ_TRIPLET_CREDIT_MIN_MULTIPLIER:-0.70}"
adj_triplet_credit_max_multiplier="${ADJ_TRIPLET_CREDIT_MAX_MULTIPLIER:-2.50}"
adj_triplet_credit_negative_rank_scale="${ADJ_TRIPLET_CREDIT_NEGATIVE_RANK_SCALE:-0.25}"
adj_triplet_credit_min_positive_fraction="${ADJ_TRIPLET_CREDIT_MIN_POSITIVE_FRACTION:-0.45}"
adj_triplet_negative_graph_penalty="${ADJ_TRIPLET_NEGATIVE_GRAPH_PENALTY:-0.50}"
adj_order3_quota_score_floor="${ADJ_ORDER3_QUOTA_SCORE_FLOOR:-0.45}"
adj_min_pair_ratio="${ADJ_MIN_PAIR_RATIO:-0.20}"
adj_order_adv_coef="${ADJ_ORDER_ADV_COEF:-0.75}"
adj_order_adv_positive_only="${ADJ_ORDER_ADV_POSITIVE_ONLY:-1}"
adj_order_adv_negative_coef="${ADJ_ORDER_ADV_NEGATIVE_COEF:-0.20}"
adj_order_adv_require_positive_graph_adv="${ADJ_ORDER_ADV_REQUIRE_POSITIVE_GRAPH_ADV:-1}"
use_adj_order3_credit_gate="${USE_ADJ_ORDER3_CREDIT_GATE:-1}"
use_adj_order3_relative_credit_gate="${USE_ADJ_ORDER3_RELATIVE_CREDIT_GATE:-1}"
adj_order3_credit_gate_loss_scale="${ADJ_ORDER3_CREDIT_GATE_LOSS_SCALE:-0.004}"
adj_order3_credit_gate_min_scale="${ADJ_ORDER3_CREDIT_GATE_MIN_SCALE:-0.70}"
adj_order3_credit_gate_ema_alpha="${ADJ_ORDER3_CREDIT_GATE_EMA_ALPHA:-0.10}"
adj_order3_credit_gate_margin="${ADJ_ORDER3_CREDIT_GATE_MARGIN:-0.0}"
adj_order3_credit_gate_max_delta="${ADJ_ORDER3_CREDIT_GATE_MAX_DELTA:-0.05}"
adj_ppo_clip_stop_ratio="${ADJ_PPO_CLIP_STOP_RATIO:-0.35}"
adj_ppo_factor_clip_stop_ratio="${ADJ_PPO_FACTOR_CLIP_STOP_RATIO:-0.35}"
adj_ppo_min_epochs="${ADJ_PPO_MIN_EPOCHS:-1}"
use_adj_ppo_stale_trust="${USE_ADJ_PPO_STALE_TRUST:-1}"
adj_ppo_stale_trust_clip="${ADJ_PPO_STALE_TRUST_CLIP:-0.20}"
adj_ppo_stale_trust_scale="${ADJ_PPO_STALE_TRUST_SCALE:-0.25}"
adj_ppo_stale_trust_min_weight="${ADJ_PPO_STALE_TRUST_MIN_WEIGHT:-0.25}"
adj_recent_episode_window="${ADJ_RECENT_EPISODE_WINDOW:-4}"
use_adj_dynamic_recent_window="${USE_ADJ_DYNAMIC_RECENT_WINDOW:-1}"
adj_recent_episode_window_min="${ADJ_RECENT_EPISODE_WINDOW_MIN:-3}"
adj_recent_window_stale_threshold="${ADJ_RECENT_WINDOW_STALE_THRESHOLD:-0.35}"
adj_recent_window_factor_stale_threshold="${ADJ_RECENT_WINDOW_FACTOR_STALE_THRESHOLD:-0.30}"
adj_recent_window_shrink_patience="${ADJ_RECENT_WINDOW_SHRINK_PATIENCE:-3}"
adj_recent_window_recover_patience="${ADJ_RECENT_WINDOW_RECOVER_PATIENCE:-2}"
adj_recent_window_recover_stale_threshold="${ADJ_RECENT_WINDOW_RECOVER_STALE_THRESHOLD:-0.28}"
adj_recent_window_recover_factor_stale_threshold="${ADJ_RECENT_WINDOW_RECOVER_FACTOR_STALE_THRESHOLD:-0.24}"
adj_recent_window_severe_margin="${ADJ_RECENT_WINDOW_SEVERE_MARGIN:-0.20}"
if ! [[ "${adj_order3_bonus_anneal_steps}" =~ ^[0-9]+$ ]]; then
  echo "ADJ_ORDER3_BONUS_ANNEAL_STEPS must be a non-negative integer, got: ${adj_order3_bonus_anneal_steps}" >&2
  exit 2
fi
if ! [[ "${adj_sampling_temp_anneal_steps}" =~ ^[0-9]+$ ]]; then
  echo "ADJ_SAMPLING_TEMP_ANNEAL_STEPS must be a non-negative integer, got: ${adj_sampling_temp_anneal_steps}" >&2
  exit 2
fi
if ! [[ "${adj_min_order3_ratio_anneal_steps}" =~ ^[0-9]+$ ]]; then
  echo "ADJ_MIN_ORDER3_RATIO_ANNEAL_STEPS must be a non-negative integer, got: ${adj_min_order3_ratio_anneal_steps}" >&2
  exit 2
fi
if ! [[ "${adj_max_order3_ratio_anneal_steps}" =~ ^[0-9]+$ ]]; then
  echo "ADJ_MAX_ORDER3_RATIO_ANNEAL_STEPS must be a non-negative integer, got: ${adj_max_order3_ratio_anneal_steps}" >&2
  exit 2
fi
if ! [[ "${adj_greedy_sample_prob_anneal_steps}" =~ ^[0-9]+$ ]]; then
  echo "ADJ_GREEDY_SAMPLE_PROB_ANNEAL_STEPS must be a non-negative integer, got: ${adj_greedy_sample_prob_anneal_steps}" >&2
  exit 2
fi
if ! [[ "${adj_ppo_min_epochs}" =~ ^[0-9]+$ ]] || [ "${adj_ppo_min_epochs}" -le 0 ]; then
  echo "ADJ_PPO_MIN_EPOCHS must be a positive integer, got: ${adj_ppo_min_epochs}" >&2
  exit 2
fi
if ! [[ "${adj_recent_episode_window}" =~ ^[0-9]+$ ]]; then
  echo "ADJ_RECENT_EPISODE_WINDOW must be a non-negative integer, got: ${adj_recent_episode_window}" >&2
  exit 2
fi
adj_order3_credit_gate_args=()
if [[ "${use_adj_order3_credit_gate}" == "1" ]]; then
  adj_order3_credit_gate_args=(--use_adj_order3_credit_gate)
fi
if [[ "${use_adj_order3_relative_credit_gate}" == "1" ]]; then
  adj_order3_credit_gate_args+=(--use_adj_order3_relative_credit_gate)
fi
adj_advantage_triplet_scorer_args=()
if [[ "${use_adj_advantage_triplet_scorer}" == "1" ]]; then
  adj_advantage_triplet_scorer_args=(--use_adj_advantage_triplet_scorer)
fi
if [[ "${use_adj_triplet_credit_direct_rank}" == "1" ]]; then
  adj_advantage_triplet_scorer_args+=(--use_adj_triplet_credit_direct_rank)
fi
adj_order_adv_args=()
if [[ "${adj_order_adv_positive_only}" == "1" ]]; then
  adj_order_adv_args=(--adj_order_adv_positive_only)
fi
if [[ "${adj_order_adv_require_positive_graph_adv}" == "1" ]]; then
  adj_order_adv_args+=(--adj_order_adv_require_positive_graph_adv)
fi
adj_ppo_stale_trust_args=()
if [[ "${use_adj_ppo_stale_trust}" == "1" ]]; then
  adj_ppo_stale_trust_args=(--use_adj_ppo_stale_trust)
fi
if [[ "${use_adj_dynamic_recent_window}" == "1" ]]; then
  adj_ppo_stale_trust_args+=(--use_adj_dynamic_recent_window)
fi

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${script_dir}"

algorithm="sddfg"
env_name="wolfpack"
wolfpack_id="wolfpack-v0"

# Fixed tensor capacity. Every episode starts with 4 active members; only the
# active mask and graph topology change inside the episode.
num_agents=4
max_player_num=6
max_food_num=2
episode_length=200

# Intra-episode event schedule:
#   t=50:  4 -> 2, t=60: 2 -> 3, t=80: 3 -> 5
#   t=120: 5 -> 3, t=130: 3 -> 4, t=150: 4 -> 6
shock_steps="50,120"
shock_remove_num=2
shock_join_delay=10
shock_join_num=1
shock_recover_delay=30
dynamic_min_agents=2

experiment_name="sddfg_intra_ep_4to6_r${shock_remove_num}_j${shock_join_num}_rec${shock_recover_delay}_seed${seed}"

# Keep stdout/stderr even when the server terminal disconnects. The Python
# runner still creates its normal runN directory; this console log is stored at
# the experiment root and can be matched by its timestamp.
result_root="${script_dir}/results/${env_name}/${algorithm}/${experiment_name}"
mkdir -p "${result_root}/console_logs"
console_log="${result_root}/console_logs/train_$(date +%Y%m%d_%H%M%S).log"
repo_root="$(CDPATH= cd -- "${script_dir}/.." && pwd)"

# Persist source provenance in the same console log as the training output.
# Full SDDFG/baseline comparisons are invalid if their server-side code
# revisions cannot be reconciled afterwards.
git_commit="unavailable"
git_tree_state="unavailable"
if command -v git >/dev/null 2>&1 && git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
  if [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=no)" ]]; then
    git_tree_state="dirty"
  else
    git_tree_state="clean"
  fi
fi

{
  echo "Algorithm: ${algorithm}"
  echo "Seed: ${seed}; GPU: ${gpu}; num_env_steps: ${num_env_steps}"
  echo "Anneal steps: lr=${anneal_steps}; adj_entropy=${adj_entropy_anneal_steps}; adj_entropy_coef=${adj_entropy_coef}; adj_order3_bonus=${adj_order3_bonus_start}->${adj_order3_bonus}/${adj_order3_bonus_anneal_steps}; adj_sampling_temp=${adj_sampling_temp_start}->${adj_sampling_temp_final}/${adj_sampling_temp_anneal_steps}"
  echo "Order-aware graph: min_order3_ratio=${adj_min_order3_ratio_start}->${adj_min_order3_ratio_final}/${adj_min_order3_ratio_anneal_steps}; max_order3_ratio=${adj_max_order3_ratio_start}->${adj_max_order3_ratio_final}/${adj_max_order3_ratio_anneal_steps}; greedy_sample_prob=${adj_greedy_sample_prob_start}->${adj_greedy_sample_prob_final}/${adj_greedy_sample_prob_anneal_steps}; greedy_cap=${adj_greedy_sample_prob_cap}; quota_mode=${adj_order3_quota_mode}; soft_quota_coef=${adj_order3_soft_quota_coef}; triplet_feature_mode=${adj_triplet_feature_mode}; triplet_balance_coef=${adj_triplet_balance_coef}; advantage_triplet_scorer=${use_adj_advantage_triplet_scorer}; credit_alpha=${adj_triplet_credit_ema_alpha}; credit_coef=${adj_triplet_credit_score_coef}; credit_scale=${adj_triplet_credit_score_scale}; direct_rank=${use_adj_triplet_credit_direct_rank}; rank_coef=${adj_triplet_credit_rank_coef}; rank_multiplier=[${adj_triplet_credit_min_multiplier},${adj_triplet_credit_max_multiplier}]; negative_rank_scale=${adj_triplet_credit_negative_rank_scale}; min_positive_fraction=${adj_triplet_credit_min_positive_fraction}; negative_graph_penalty=${adj_triplet_negative_graph_penalty}; quota_score_floor=${adj_order3_quota_score_floor}; min_pair_ratio=${adj_min_pair_ratio}; adj_order_adv_coef=${adj_order_adv_coef}; positive_only=${adj_order_adv_positive_only}; negative_coef=${adj_order_adv_negative_coef}; require_positive_graph_adv=${adj_order_adv_require_positive_graph_adv}"
  echo "Order3 credit gate: enabled=${use_adj_order3_credit_gate}; relative=${use_adj_order3_relative_credit_gate}; loss_scale=${adj_order3_credit_gate_loss_scale}; margin=${adj_order3_credit_gate_margin}; min_scale=${adj_order3_credit_gate_min_scale}; ema_alpha=${adj_order3_credit_gate_ema_alpha}; max_delta=${adj_order3_credit_gate_max_delta}"
  echo "Adj PPO guard: clip_stop=${adj_ppo_clip_stop_ratio}; factor_clip_stop=${adj_ppo_factor_clip_stop_ratio}; min_epochs=${adj_ppo_min_epochs}"
  echo "Adj PPO stale trust: enabled=${use_adj_ppo_stale_trust}; clip=${adj_ppo_stale_trust_clip}; scale=${adj_ppo_stale_trust_scale}; min_weight=${adj_ppo_stale_trust_min_weight}; recent_episode_window=${adj_recent_episode_window}; dynamic_recent=${use_adj_dynamic_recent_window}; min_recent=${adj_recent_episode_window_min}; stale_threshold=${adj_recent_window_stale_threshold}; factor_stale_threshold=${adj_recent_window_factor_stale_threshold}; shrink_patience=${adj_recent_window_shrink_patience}; recover_patience=${adj_recent_window_recover_patience}; recover_threshold=${adj_recent_window_recover_stale_threshold}; recover_factor_threshold=${adj_recent_window_recover_factor_stale_threshold}; severe_margin=${adj_recent_window_severe_margin}"
  echo "Experiment: ${experiment_name}"
  echo "Git commit: ${git_commit}; tracked tree: ${git_tree_state}"
  echo "Console log: ${console_log}"
} | tee "${console_log}"

# Fail before allocating a long training run if the dynamic graph, adjacency
# buffer axes, factor-Q gradients, or critic-free trainer initialization are
# inconsistent with the server's PyTorch/Gym versions.
if [[ "${SKIP_SDDFG_PREFLIGHT:-0}" != "1" ]]; then
  "${python_bin}" "${script_dir}/validate_sddfg_dynamic_graph.py" \
    2>&1 | tee -a "${console_log}"
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${script_dir}/train/train_wolfpack.py" \
  --env_name "${env_name}" \
  --wolfpack_id "${wolfpack_id}" \
  --algorithm_name "${algorithm}" \
  --experiment_name "${experiment_name}" \
  --user_name "${user_name}" \
  --seed "${seed}" \
  --n_training_threads 1 \
  --n_eval_rollout_threads 1 \
  --num_env_steps "${num_env_steps}" \
  --episode_length "${episode_length}" \
  --buffer_size 5000 \
  --batch_size 32 \
  --lr 0.0005 \
  --critic_lr 0.0005 \
  --use_linear_lr_decay \
  --policy_lr_anneal_steps "${anneal_steps}" \
  --adj_lr 0.0003 \
  --use_adj_linear_lr_decay \
  --adj_lr_anneal_steps "${anneal_steps}" \
  --adj_lr_decay_floor 0.00002 \
  --gamma 0.97 \
  --gae_lambda 0.95 \
  --hard_update_interval_episode 200 \
  --use_reward_normalization \
  --use_dyn_graph \
  --require_connected_adj \
  --msg_iterations 4 \
  --highest_orders 3 \
  --num_factor 6 \
  --sparsity 0.3 \
  --gain 0.01 \
  --gat_heads 4 \
  --gat_negative_slope 0.2 \
  --gat_hyperedge_hidden 64 \
  --adj_order3_bonus "${adj_order3_bonus}" \
  --adj_order3_bonus_start "${adj_order3_bonus_start}" \
  --adj_order3_bonus_anneal_steps "${adj_order3_bonus_anneal_steps}" \
  --adj_sampling_temperature_start "${adj_sampling_temp_start}" \
  --adj_sampling_temperature_final "${adj_sampling_temp_final}" \
  --adj_sampling_temperature_anneal_steps "${adj_sampling_temp_anneal_steps}" \
  --adj_min_order3_ratio_start "${adj_min_order3_ratio_start}" \
  --adj_min_order3_ratio_final "${adj_min_order3_ratio_final}" \
  --adj_min_order3_ratio_anneal_steps "${adj_min_order3_ratio_anneal_steps}" \
  --adj_max_order3_ratio_start "${adj_max_order3_ratio_start}" \
  --adj_max_order3_ratio_final "${adj_max_order3_ratio_final}" \
  --adj_max_order3_ratio_anneal_steps "${adj_max_order3_ratio_anneal_steps}" \
  --adj_greedy_sample_prob_start "${adj_greedy_sample_prob_start}" \
  --adj_greedy_sample_prob_final "${adj_greedy_sample_prob_final}" \
  --adj_greedy_sample_prob_anneal_steps "${adj_greedy_sample_prob_anneal_steps}" \
  --adj_greedy_sample_prob_cap "${adj_greedy_sample_prob_cap}" \
  --adj_order3_quota_mode "${adj_order3_quota_mode}" \
  --adj_order3_soft_quota_coef "${adj_order3_soft_quota_coef}" \
  --adj_triplet_feature_mode "${adj_triplet_feature_mode}" \
  --adj_triplet_balance_coef "${adj_triplet_balance_coef}" \
  "${adj_advantage_triplet_scorer_args[@]}" \
  --adj_triplet_credit_ema_alpha "${adj_triplet_credit_ema_alpha}" \
  --adj_triplet_credit_score_coef "${adj_triplet_credit_score_coef}" \
  --adj_triplet_credit_score_scale "${adj_triplet_credit_score_scale}" \
  --adj_triplet_credit_rank_coef "${adj_triplet_credit_rank_coef}" \
  --adj_triplet_credit_min_multiplier "${adj_triplet_credit_min_multiplier}" \
  --adj_triplet_credit_max_multiplier "${adj_triplet_credit_max_multiplier}" \
  --adj_triplet_credit_negative_rank_scale "${adj_triplet_credit_negative_rank_scale}" \
  --adj_triplet_credit_min_positive_fraction "${adj_triplet_credit_min_positive_fraction}" \
  --adj_triplet_negative_graph_penalty "${adj_triplet_negative_graph_penalty}" \
  --adj_order3_quota_score_floor "${adj_order3_quota_score_floor}" \
  --adj_min_pair_ratio "${adj_min_pair_ratio}" \
  --adj_return_adv_coef 1.0 \
  --adj_factor_adv_coef 0.25 \
  --adj_order_adv_coef "${adj_order_adv_coef}" \
  "${adj_order_adv_args[@]}" \
  --adj_order_adv_negative_coef "${adj_order_adv_negative_coef}" \
  "${adj_order3_credit_gate_args[@]}" \
  --adj_order3_credit_gate_loss_scale "${adj_order3_credit_gate_loss_scale}" \
  --adj_order3_credit_gate_margin "${adj_order3_credit_gate_margin}" \
  --adj_order3_credit_gate_min_scale "${adj_order3_credit_gate_min_scale}" \
  --adj_order3_credit_gate_ema_alpha "${adj_order3_credit_gate_ema_alpha}" \
  --adj_order3_credit_gate_max_delta "${adj_order3_credit_gate_max_delta}" \
  --adj_exploration_mix 0.0 \
  --adj_begin_step 5000 \
  --adj_buffer_size 16 \
  --adj_recent_episode_window "${adj_recent_episode_window}" \
  --train_adj_episode 4 \
  --adj_train_epochs 2 \
  --adj_ppo_clip_stop_ratio "${adj_ppo_clip_stop_ratio}" \
  --adj_ppo_factor_clip_stop_ratio "${adj_ppo_factor_clip_stop_ratio}" \
  --adj_ppo_min_epochs "${adj_ppo_min_epochs}" \
  "${adj_ppo_stale_trust_args[@]}" \
  --adj_ppo_stale_trust_clip "${adj_ppo_stale_trust_clip}" \
  --adj_ppo_stale_trust_scale "${adj_ppo_stale_trust_scale}" \
  --adj_ppo_stale_trust_min_weight "${adj_ppo_stale_trust_min_weight}" \
  --adj_recent_episode_window_min "${adj_recent_episode_window_min}" \
  --adj_recent_window_stale_threshold "${adj_recent_window_stale_threshold}" \
  --adj_recent_window_factor_stale_threshold "${adj_recent_window_factor_stale_threshold}" \
  --adj_recent_window_shrink_patience "${adj_recent_window_shrink_patience}" \
  --adj_recent_window_recover_patience "${adj_recent_window_recover_patience}" \
  --adj_recent_window_recover_stale_threshold "${adj_recent_window_recover_stale_threshold}" \
  --adj_recent_window_recover_factor_stale_threshold "${adj_recent_window_recover_factor_stale_threshold}" \
  --adj_recent_window_severe_margin "${adj_recent_window_severe_margin}" \
  --num_mini_batch 2 \
  --clip_param 0.2 \
  --entropy_coef 0.001 \
  --adj_entropy_coef "${adj_entropy_coef}" \
  --adj_entropy_coef_final 0.0 \
  --adj_entropy_anneal_steps "${adj_entropy_anneal_steps}" \
  --max_grad_norm 10 \
  --adj_max_grad_norm 0.5 \
  --num_agents "${num_agents}" \
  --max_player_num "${max_player_num}" \
  --max_food_num "${max_food_num}" \
  --grid_height 20 \
  --grid_width 20 \
  --obs_type vector \
  --sight_sideways 8 \
  --sight_radius 8 \
  --coop_radius 1 \
  --group_multiplier 2.0 \
  --close_penalty 0.1 \
  --food_freeze_rate 25 \
  --add_rate 0.0 \
  --del_rate 0.0 \
  --intra_episode_dynamic \
  --shock_steps "${shock_steps}" \
  --shock_remove_num "${shock_remove_num}" \
  --shock_join_delay "${shock_join_delay}" \
  --shock_join_num "${shock_join_num}" \
  --shock_recover_delay "${shock_recover_delay}" \
  --dynamic_min_agents "${dynamic_min_agents}" \
  --continue_after_success \
  --epsilon_start 1.0 \
  --epsilon_finish 0.05 \
  --epsilon_anneal_time 500000 \
  --num_random_episodes 10 \
  --train_interval_episode 4 \
  --log_interval 3000 \
  --eval_interval 20000 \
  --save_interval 50000 \
  --num_eval_episodes 10 \
  --use_wandb \
  2>&1 | tee -a "${console_log}"

# In this repository --use_wandb is a store_false flag, so passing it disables
# W&B and writes local TensorBoard/results under:
# scripts/results/wolfpack/sddfg/${experiment_name}/run*/
