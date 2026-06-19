#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for episode-internal dynamic Wolfpack value-decomposition
# baselines. Prefer the algorithm-specific wrapper scripts for normal use.
# Usage:
#   bash scripts/train_wolfpack_value_baseline_intra_episode_dynamic.sh \
#     <vdn|qmix|qplex> [seed] [gpu] [num_env_steps]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <vdn|qmix|qplex> [seed] [gpu] [num_env_steps]" >&2
  exit 2
fi

algorithm="$1"
shift

case "${algorithm}" in
  vdn|qmix|qplex) ;;
  *)
    echo "Unsupported algorithm: ${algorithm}; expected vdn, qmix, or qplex" >&2
    exit 2
    ;;
esac

seed="${1:-1}"
gpu="${2:-${CUDA_DEVICE:-0}}"
num_env_steps="${3:-2000000}"
python_bin="${PYTHON_BIN:-python}"
user_name="${USER_NAME:-${algorithm}_dynamic}"

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${script_dir}"

env_name="wolfpack"
wolfpack_id="wolfpack-v0"

# Fixed tensor capacity. The tensor shape never changes during an episode;
# only the active mask, recurrent-state validity, and effective cooperation
# topology change.
num_agents=4
max_player_num=6
max_food_num=2
episode_length=200

# Episode-internal schedule shared exactly with the SDDFG dynamic experiment:
#   step 50:  4 -> 2
#   step 60:  2 -> 3 (one new member)
#   step 80:  3 -> 5 (two original members recover)
#   step 120: 5 -> 3
#   step 130: 3 -> 4 (one new member)
#   step 150: 4 -> 6 (two original members recover)
shock_steps="50,120"
shock_remove_num=2
shock_join_delay=10
shock_join_num=1
shock_recover_delay=30
dynamic_min_agents=2

experiment_name="${algorithm}_intra_ep_4to6_r${shock_remove_num}_j${shock_join_num}_rec${shock_recover_delay}_seed${seed}"
result_root="${script_dir}/results/${env_name}/${algorithm}/${experiment_name}"
mkdir -p "${result_root}/console_logs"
console_log="${result_root}/console_logs/train_$(date +%Y%m%d_%H%M%S).log"

algorithm_args=()
case "${algorithm}" in
  vdn)
    # VDN uses the shared recurrent Q network and additive mixer only.
    ;;
  qmix)
    algorithm_args+=(
      --mixer_hidden_dim 32
      --hypernet_hidden_dim 64
      --hypernet_layers 2
    )
    ;;
  qplex)
    algorithm_args+=(
      --mixer_hidden_dim 32
      --hypernet_hidden_dim 64
      --hypernet_layers 2
      --n_head 4
      --num_kernel 10
      --adv_hypernet_embed 64
      --adv_hypernet_layers 3
      --attend_reg_coef 0.001
      --weighted_head
      --state_bias
      --is_minus_one
    )
    ;;
esac

echo "Algorithm: ${algorithm}"
echo "Seed: ${seed}; GPU: ${gpu}; num_env_steps: ${num_env_steps}"
echo "Experiment: ${experiment_name}"
echo "Console log: ${console_log}"

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" train/train_wolfpack.py \
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
  --gamma 0.97 \
  --hard_update_interval_episode 200 \
  --use_reward_normalization \
  --use_linear_lr_decay \
  --policy_lr_decay_floor 0.00001 \
  --max_grad_norm 10 \
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
  "${algorithm_args[@]}" \
  --use_wandb \
  2>&1 | tee "${console_log}"

