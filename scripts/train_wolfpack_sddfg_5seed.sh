#!/bin/sh
set -e

seed=$1

if [ -z "$seed" ]; then
  echo "Usage: bash train_sddfg_wolfpack_4-5_seed.sh {seed}"
  exit 1
fi


alg="sddfg"
env="wolfpack"
wolfpack_id="wolfpack-v0"

root_dir="/root/SDDFG/scripts/results/wolfpack/${alg}"

# ==================== common env ====================
num_agents=4
max_player_num=5
max_food_num=2
grid_h=20
grid_w=20
obs_type="vector"
sight_sideways=8
sight_radius=8
coop_radius=1
group_multiplier=2.0
food_freeze_rate=200

# ==================== common SDDFG structure ====================
msg_iterations=4
highest_orders=3
num_factor=3
sparsity=0.3
gain=0.01
gat_heads=4
gat_negative_slope=0.2
gat_hyperedge_hidden=64

adj_begin_step=0
adj_buffer_size=4
train_adj_episode=4
num_mini_batch=1

max_grad_norm=10
adj_max_grad_norm=0.5
clip_param=0.2

gamma=0.97
gae_lambda=0.95

buffer_size=5000
batch_size=32
episode_length=200

train_interval_episode=4
hard_update_interval_episode=200

adj_anneal_time=500000

log_interval=3000
eval_interval=20000
save_interval=50000

latest_run_dir () {
  exp_name=$1
  ls -td ${root_dir}/${exp_name}/run* 2>/dev/null | head -1
}

check_dir () {
  d=$1
  msg=$2
  if [ ! -d "$d" ]; then
    echo "[ERROR] ${msg}: ${d}"
    exit 1
  fi
}

run_sddfg_stage () {
  stage_label=$1
  exp_name=$2
  user_name=$3
  model_dir=$4
  num_env_steps=$5
  add_rate=$6
  del_rate=$7
  close_penalty=$8
  lr=$9
  critic_lr=${10}
  adj_lr=${11}
  entropy_coef=${12}
  epsilon_start=${13}
  epsilon_finish=${14}
  epsilon_anneal_time=${15}
  num_random_episodes=${16}
  num_eval_episodes=${17}
  soft_update_false=${18}

  model_arg=""
  if [ -n "$model_dir" ]; then
    model_arg="--model_dir ${model_dir}"
  fi

  soft_arg=""
  if [ "$soft_update_false" = "1" ]; then
    soft_arg="--use_soft_update"
  fi

  echo "============================================================"
  echo "Running SDDFG ${stage_label}, seed=${seed}"
  echo "experiment_name=${exp_name}"
  echo "user_name=${user_name}"
  echo "model_dir=${model_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py \
    --env_name ${env} \
    --algorithm_name ${alg} \
    --experiment_name ${exp_name} \
    --user_name ${user_name} \
    --seed ${seed} \
    ${model_arg} \
    --wolfpack_id ${wolfpack_id} \
    --num_env_steps ${num_env_steps} \
    --episode_length ${episode_length} \
    --buffer_size ${buffer_size} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --critic_lr ${critic_lr} \
    --adj_lr ${adj_lr} \
    --gamma ${gamma} \
    --gae_lambda ${gae_lambda} \
    ${soft_arg} \
    --hard_update_interval_episode ${hard_update_interval_episode} \
    --use_reward_normalization \
    --use_linear_lr_decay \
    --use_valuenorm \
    --use_vfunction \
    --use_dyn_graph \
    --msg_iterations ${msg_iterations} \
    --highest_orders ${highest_orders} \
    --num_factor ${num_factor} \
    --sparsity ${sparsity} \
    --gain ${gain} \
    --gat_heads ${gat_heads} \
    --gat_negative_slope ${gat_negative_slope} \
    --gat_hyperedge_hidden ${gat_hyperedge_hidden} \
    --adj_begin_step ${adj_begin_step} \
    --adj_buffer_size ${adj_buffer_size} \
    --train_adj_episode ${train_adj_episode} \
    --num_mini_batch ${num_mini_batch} \
    --max_grad_norm ${max_grad_norm} \
    --adj_max_grad_norm ${adj_max_grad_norm} \
    --clip_param ${clip_param} \
    --entropy_coef ${entropy_coef} \
    --adj_anneal_time ${adj_anneal_time} \
    --num_agents ${num_agents} \
    --max_player_num ${max_player_num} \
    --grid_height ${grid_h} \
    --grid_width ${grid_w} \
    --max_food_num ${max_food_num} \
    --obs_type ${obs_type} \
    --sight_sideways ${sight_sideways} \
    --sight_radius ${sight_radius} \
    --coop_radius ${coop_radius} \
    --group_multiplier ${group_multiplier} \
    --close_penalty ${close_penalty} \
    --food_freeze_rate ${food_freeze_rate} \
    --add_rate ${add_rate} \
    --del_rate ${del_rate} \
    --epsilon_start ${epsilon_start} \
    --epsilon_finish ${epsilon_finish} \
    --epsilon_anneal_time ${epsilon_anneal_time} \
    --num_random_episodes ${num_random_episodes} \
    --train_interval_episode ${train_interval_episode} \
    --log_interval ${log_interval} \
    --eval_interval ${eval_interval} \
    --save_interval ${save_interval} \
    --num_eval_episodes ${num_eval_episodes} \
    --use_wandb
}

echo "============================================================"
echo "SDDFG wolfpack 4-5 full pipeline, seed=${seed}"
echo "============================================================"

# ============================================================
# Stage1a
# 对应原 Stage1-A1：
# 从零训练；
# add/del=0.05；
# adj_lr=3e-4；
# entropy_coef=0.015；
# epsilon 1.0 -> 0.05；
# use_soft_update=False，因此传 --use_soft_update。
# ============================================================
exp_stage1a="sddfg426_wolfpack_4-5_stage1a"

run_sddfg_stage \
  "stage1a" \
  "${exp_stage1a}" \
  "stage1a_seed${seed}" \
  "" \
  1000000 \
  0.05 \
  0.05 \
  0.05 \
  0.0005 \
  0.0005 \
  0.0003 \
  0.015 \
  1.0 \
  0.05 \
  1000000 \
  5 \
  1 \
  1

stage1a_dir=$(latest_run_dir "${exp_stage1a}")
check_dir "${stage1a_dir}" "Missing Stage1a run dir"

stage1a_best="${stage1a_dir}/best_models/best_win"
check_dir "${stage1a_best}" "Missing Stage1a best_win"

# ============================================================
# Stage1b
# 对应原 Stage1-A2 / run11：
# model_dir = Stage1a best_win；
# add/del=0.05；
# adj_lr=1e-4；
# entropy_coef=0.015；
# epsilon 固定 0.05；
# num_random_episodes=0；
# num_eval_episodes=1；
# use_soft_update=False，因此传 --use_soft_update。
# ============================================================
exp_stage1b="sddfg426_wolfpack_4-5_stage1b"

run_sddfg_stage \
  "stage1b" \
  "${exp_stage1b}" \
  "stage1b_from_stage1a_bestwin_seed${seed}" \
  "${stage1a_best}" \
  1000000 \
  0.05 \
  0.05 \
  0.05 \
  0.0005 \
  0.0005 \
  0.0001 \
  0.015 \
  0.05 \
  0.05 \
  1000000 \
  0 \
  1 \
  1

stage1b_dir=$(latest_run_dir "${exp_stage1b}")
check_dir "${stage1b_dir}" "Missing Stage1b run dir"

stage1b_final="${stage1b_dir}/models"
check_dir "${stage1b_final}" "Missing Stage1b final models"

# ============================================================
# Stage2a
# 对应原 stage2a_run11：
# model_dir = Stage1b final models；
# add/del=0.10；
# adj_lr=1e-4；
# entropy_coef=0.005；
# epsilon 0.2 -> 0.05；
# use_soft_update=False，因此传 --use_soft_update。
# ============================================================
exp_stage2a="sddfg426_wolfpack_4-5_stage2a"

run_sddfg_stage \
  "stage2a" \
  "${exp_stage2a}" \
  "stage2a_from_stage1b_final_seed${seed}" \
  "${stage1b_final}" \
  1000000 \
  0.10 \
  0.10 \
  0.05 \
  0.0005 \
  0.0005 \
  0.0001 \
  0.005 \
  0.2 \
  0.05 \
  600000 \
  0 \
  5 \
  1

stage2a_dir=$(latest_run_dir "${exp_stage2a}")
check_dir "${stage2a_dir}" "Missing Stage2a run dir"

stage2a_final="${stage2a_dir}/models"
check_dir "${stage2a_final}" "Missing Stage2a final models"

# ============================================================
# Stage2b
# 对应原 stage2b_run12：
# model_dir = Stage2a final models；
# add/del=0.15；
# lr=3e-4；
# critic_lr=3e-4；
# adj_lr=1e-4；
# entropy_coef=0.005；
# use_soft_update=True，因此不传 --use_soft_update。
# ============================================================
exp_stage2b="sddfg426_wolfpack_4-5_stage2b"

run_sddfg_stage \
  "stage2b" \
  "${exp_stage2b}" \
  "stage2b_from_stage2a_final_seed${seed}" \
  "${stage2a_final}" \
  1000000 \
  0.15 \
  0.15 \
  0.05 \
  0.0003 \
  0.0003 \
  0.0001 \
  0.005 \
  0.05 \
  0.05 \
  500000 \
  0 \
  5 \
  0

stage2b_dir=$(latest_run_dir "${exp_stage2b}")
check_dir "${stage2b_dir}" "Missing Stage2b run dir"

stage2b_best="${stage2b_dir}/best_models/best_win"
check_dir "${stage2b_best}" "Missing Stage2b best_win"

# ============================================================
# Stage3
# 对应当前最优 stage3/run4：
# model_dir = Stage2b best_win；
# add/del=0.20；
# close_penalty=0.10；
# lr=3e-4；
# critic_lr=3e-4；
# adj_lr=1e-5；
# entropy_coef=0.001；
# use_soft_update=True，因此不传 --use_soft_update。
# ============================================================
exp_stage3="sddfg426_wolfpack_4-5_stage3"

run_sddfg_stage \
  "stage3" \
  "${exp_stage3}" \
  "stage3_from_stage2b_bestwin_seed${seed}" \
  "${stage2b_best}" \
  1500000 \
  0.20 \
  0.20 \
  0.10 \
  0.0003 \
  0.0003 \
  0.00001 \
  0.001 \
  0.05 \
  0.05 \
  500000 \
  0 \
  5 \
  0

echo "============================================================"
echo "Finished SDDFG wolfpack 4-5 full pipeline for seed=${seed}"
echo "============================================================"