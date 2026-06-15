#!/bin/sh
set -e

alg="vdn"
env="wolfpack"
wolfpack_id="wolfpack-v0"

root_dir="/root/SDDFG/scripts/results/wolfpack/${alg}"

# 公共环境参数：对齐当前 SDDFG / baseline 实验
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

# 公共训练参数
episode_length=200
buffer_size=5000
batch_size=32
lr=5e-4
gamma=0.97
train_interval_episode=4
hard_update_interval_episode=200
log_interval=3000
eval_interval=20000
save_interval=50000
num_eval_episodes=5

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

run_stage () {
  stage_name=$1
  exp_name=$2
  user_name=$3
  seed=$4
  model_dir=$5
  num_env_steps=$6
  add_rate=$7
  del_rate=$8
  close_penalty=$9
  epsilon_start=${10}
  epsilon_finish=${11}
  epsilon_anneal_time=${12}
  num_random_episodes=${13}
  soft_update_false=${14}

  model_arg=""
  if [ -n "$model_dir" ]; then
    model_arg="--model_dir ${model_dir}"
  fi

  soft_arg=""
  if [ "$soft_update_false" = "1" ]; then
    soft_arg="--use_soft_update"
  fi

  echo "Running ${alg} ${stage_name}, seed=${seed}, model_dir=${model_dir}"

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
    --gamma ${gamma} \
    ${soft_arg} \
    --hard_update_interval_episode ${hard_update_interval_episode} \
    --use_reward_normalization \
    --use_linear_lr_decay \
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
    --num_factor 3 \
    --use_wandb
}

for seed in 1 2 3 4 5; do
  echo "=============================="
  echo "VDN pure baseline seed=${seed}"
  echo "=============================="

  # Stage1: 2M steps，对齐 SDDFG Stage1-A1+A2 总步数，但不使用 SDDFG adj schedule
  exp_stage1="vdn_wolfpack_stage1"
  run_stage "stage1" "${exp_stage1}" "vdn_stage1_seed${seed}" "${seed}" "" \
    2000000 0.05 0.05 0.05 1.0 0.05 1000000 5 1

  stage1_dir=$(latest_run_dir "${exp_stage1}")
  check_dir "${stage1_dir}" "Missing Stage1 run dir"
  stage1_model="${stage1_dir}/models"
  check_dir "${stage1_model}" "Missing Stage1 final models"

  # Stage2a: 从 Stage1 final/models 续训，不用 best_win
  exp_stage2a="vdn_wolfpack_stage2a"
  run_stage "stage2a" "${exp_stage2a}" "vdn_stage2a_seed${seed}" "${seed}" "${stage1_model}" \
    1000000 0.10 0.10 0.05 0.2 0.05 600000 0 1

  stage2a_dir=$(latest_run_dir "${exp_stage2a}")
  check_dir "${stage2a_dir}" "Missing Stage2a run dir"
  stage2a_model="${stage2a_dir}/models"
  check_dir "${stage2a_model}" "Missing Stage2a final models"

  # Stage2b: 从 Stage2a final/models 续训
  exp_stage2b="vdn_wolfpack_stage2b"
  run_stage "stage2b" "${exp_stage2b}" "vdn_stage2b_seed${seed}" "${seed}" "${stage2a_model}" \
    1000000 0.15 0.15 0.05 0.05 0.05 500000 0 0

  stage2b_dir=$(latest_run_dir "${exp_stage2b}")
  check_dir "${stage2b_dir}" "Missing Stage2b run dir"
  stage2b_model="${stage2b_dir}/models"
  check_dir "${stage2b_model}" "Missing Stage2b final models"

  # Stage3: 纯 baseline 默认从 Stage2b final/models 续训，不用 best_win
  exp_stage3="vdn_wolfpack_stage3"
  run_stage "stage3" "${exp_stage3}" "vdn_stage3_seed${seed}" "${seed}" "${stage2b_model}" \
    1500000 0.20 0.20 0.10 0.05 0.05 500000 0 0

  echo "Finished VDN seed=${seed}"
done