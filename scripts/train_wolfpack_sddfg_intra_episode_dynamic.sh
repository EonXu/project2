#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_wolfpack_sddfg_intra_episode_dynamic.sh [seed] [gpu] [num_env_steps]
# Environment overrides:
#   PYTHON_BIN=python CUDA_DEVICE=0 USER_NAME=local

seed="${1:-1}"
gpu="${2:-${CUDA_DEVICE:-0}}"
num_env_steps="${3:-2000000}"
python_bin="${PYTHON_BIN:-python}"
user_name="${USER_NAME:-sddfg_dynamic}"

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
  --critic_lr 0.0005 \
  --adj_lr 0.0003 \
  --gamma 0.97 \
  --gae_lambda 0.95 \
  --hard_update_interval_episode 200 \
  --use_reward_normalization \
  --use_linear_lr_decay \
  --policy_lr_decay_floor 0.00001 \
  --critic_lr_decay_floor 0.00001 \
  --use_valuenorm \
  --use_vfunction \
  --use_dyn_graph \
  --msg_iterations 4 \
  --highest_orders 3 \
  --num_factor 6 \
  --sparsity 0.3 \
  --gain 0.01 \
  --gat_heads 4 \
  --gat_negative_slope 0.2 \
  --gat_hyperedge_hidden 64 \
  --adj_begin_step 5000 \
  --adj_buffer_size 8 \
  --train_adj_episode 4 \
  --num_mini_batch 1 \
  --adj_anneal_time 500000 \
  --clip_param 0.2 \
  --entropy_coef 0.01 \
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
  --use_wandb

# In this repository --use_wandb is a store_false flag, so passing it disables
# W&B and writes local TensorBoard/results under:
# scripts/results/wolfpack/sddfg/${experiment_name}/run*/
