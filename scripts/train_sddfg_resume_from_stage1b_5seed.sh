#!/bin/sh
set -e

# Usage:
#   bash train_sddfg_resume_from_stage1b_direct_runid_5seed.sh
#   bash train_sddfg_resume_from_stage1b_direct_runid_5seed.sh 3
#   bash train_sddfg_resume_from_stage1b_direct_runid_5seed.sh 1 2 5
#
# If target run already exists, default behavior is to stop.
# If you want to overwrite existing stage2a/stage2b/stage3 run${seed}, use:
#   OVERWRITE_RUN=1 bash train_sddfg_resume_from_stage1b_direct_runid_5seed.sh 2

if [ $# -eq 0 ]; then
  SEEDS="1 2 3 4 5"
else
  SEEDS="$@"
fi

SCRIPT_ROOT="/root/SDDFG/scripts"
cd ${SCRIPT_ROOT}

alg="sddfg"
env="wolfpack"
wolfpack_id="wolfpack-v0"
root_dir="${SCRIPT_ROOT}/results/wolfpack/${alg}"
train_py="train/train_wolfpack.py"

OVERWRITE_RUN=${OVERWRITE_RUN:-0}

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

find_latest_run_by_seed () {
  exp_name=$1
  seed=$2
  python - "$root_dir" "$exp_name" "$seed" <<'PY'
import csv, glob, os, sys

root, exp, target_seed = sys.argv[1], sys.argv[2], str(sys.argv[3])
base = os.path.join(root, exp)
candidates = []

for run_dir in glob.glob(os.path.join(base, "run*")):
    run_name = os.path.basename(run_dir.rstrip(os.sep))

    # 注意：临时目录里通常是 run1_config.csv；
    # 移动改名后，我们会再把它改成 run${seed}_config.csv。
    cfgs = [
        os.path.join(run_dir, f"{run_name}_config.csv"),
        os.path.join(run_dir, "config.csv"),
    ]

    cfg = next((p for p in cfgs if os.path.exists(p)), None)
    if cfg is None:
        continue

    seed_val = None
    try:
        with open(cfg, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("Name", "")).strip() == "seed":
                    seed_val = str(row.get("Value", "")).strip()
                    break
    except Exception:
        continue

    if seed_val == target_seed:
        candidates.append((os.path.getmtime(cfg), run_dir))

if not candidates:
    sys.exit(2)

print(sorted(candidates)[-1][1])
PY
}

check_dir () {
  d=$1
  msg=$2
  if [ ! -d "$d" ]; then
    echo "[ERROR] ${msg}: ${d}"
    exit 1
  fi
}

prepare_target_run_dir () {
  target_run_dir=$1

  if [ -e "${target_run_dir}" ]; then
    if [ "${OVERWRITE_RUN}" = "1" ]; then
      echo "[WARN] Removing existing target run dir: ${target_run_dir}"
      rm -rf "${target_run_dir}"
    else
      echo "[ERROR] Target run dir already exists: ${target_run_dir}"
      echo "If you want to overwrite it, run:"
      echo "  OVERWRITE_RUN=1 bash $0 ${seed}"
      exit 1
    fi
  fi

  mkdir -p "$(dirname "${target_run_dir}")"
}

fix_config_name_after_move () {
  target_run_dir=$1
  seed=$2

  # 训练程序在临时实验目录里大概率生成 run1_config.csv。
  # 移动成 run${seed} 后，把配置文件名也同步改成 run${seed}_config.csv。
  for old_cfg in "${target_run_dir}"/run*_config.csv; do
    if [ -f "${old_cfg}" ]; then
      new_cfg="${target_run_dir}/run${seed}_config.csv"
      if [ "${old_cfg}" != "${new_cfg}" ]; then
        mv "${old_cfg}" "${new_cfg}"
      fi
      break
    fi
  done
}

run_sddfg_stage () {
  stage_label=$1
  exp_name=$2
  user_name=$3
  seed=$4
  model_dir=$5
  num_env_steps=$6
  add_rate=$7
  del_rate=$8
  close_penalty=$9
  lr=${10}
  critic_lr=${11}
  adj_lr=${12}
  entropy_coef=${13}
  epsilon_start=${14}
  epsilon_finish=${15}
  epsilon_anneal_time=${16}
  num_random_episodes=${17}
  num_eval_episodes=${18}
  soft_update_false=${19}

  model_arg=""
  if [ -n "$model_dir" ]; then
    model_arg="--model_dir ${model_dir}"
  fi

  # In this codebase --use_soft_update is action='store_false'.
  # Passing it means use_soft_update=False.
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

  CUDA_VISIBLE_DEVICES=0 python ${train_py} \
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

run_sddfg_stage_fixed_run () {
  stage_label=$1
  exp_name=$2
  user_name=$3
  seed=$4

  shift 4

  target_run_dir="${root_dir}/${exp_name}/run${seed}"
  tmp_exp_name="${exp_name}_tmp_${stage_label}_seed${seed}_pid$$"

  echo "============================================================"
  echo "Fixed-run mode"
  echo "stage=${stage_label}"
  echo "seed=${seed}"
  echo "target_run_dir=${target_run_dir}"
  echo "tmp_exp_name=${tmp_exp_name}"
  echo "============================================================"

  prepare_target_run_dir "${target_run_dir}"

  # 避免同名临时目录残留
  rm -rf "${root_dir}/${tmp_exp_name}"

  # 先跑到临时实验目录
  run_sddfg_stage \
    "${stage_label}" \
    "${tmp_exp_name}" \
    "${user_name}" \
    "${seed}" \
    "$@"

  # 找到临时实验目录里刚训练出来的 run
  tmp_run_dir=$(find_latest_run_by_seed "${tmp_exp_name}" "${seed}")
  check_dir "${tmp_run_dir}" "Missing temporary run dir for ${stage_label}, seed=${seed}"

  echo "Moving temporary run:"
  echo "  from: ${tmp_run_dir}"
  echo "  to:   ${target_run_dir}"

  mv "${tmp_run_dir}" "${target_run_dir}"

  fix_config_name_after_move "${target_run_dir}" "${seed}"

  # 清理空的临时实验目录
  rm -rf "${root_dir}/${tmp_exp_name}"

  echo "Fixed run created: ${target_run_dir}"
}

for seed in ${SEEDS}; do
  echo "============================================================"
  echo "Resume SDDFG wolfpack 4-5 from existing stage1b, seed=${seed}"
  echo "============================================================"

  exp_stage1b="sddfg426_wolfpack_4-5_stage1b"

  # 输入 seed=1 就读取 run1；seed=2 就读取 run2。
  stage1b_dir="${root_dir}/${exp_stage1b}/run${seed}"
  check_dir "${stage1b_dir}" "Missing Stage1b run${seed} dir"

  echo "Using Stage1b checkpoint: ${stage1b_dir}/models"
  stage1b_final="${stage1b_dir}/models"
  check_dir "${stage1b_final}" "Missing Stage1b final models for seed=${seed}"

  exp_stage2a="sddfg426_wolfpack_4-5_stage2a"
  run_sddfg_stage_fixed_run \
    "stage2a" \
    "${exp_stage2a}" \
    "stage2a_from_stage1b_final_seed${seed}" \
    "${seed}" \
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

  stage2a_dir="${root_dir}/${exp_stage2a}/run${seed}"
  stage2a_final="${stage2a_dir}/models"
  check_dir "${stage2a_final}" "Missing Stage2a final models for seed=${seed}"

  exp_stage2b="sddfg426_wolfpack_4-5_stage2b"
  run_sddfg_stage_fixed_run \
    "stage2b" \
    "${exp_stage2b}" \
    "stage2b_from_stage2a_final_seed${seed}" \
    "${seed}" \
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

  stage2b_dir="${root_dir}/${exp_stage2b}/run${seed}"
  stage2b_best="${stage2b_dir}/best_models/best_win"
  check_dir "${stage2b_best}" "Missing Stage2b best_win for seed=${seed}"

  exp_stage3="sddfg426_wolfpack_4-5_stage3"
  run_sddfg_stage_fixed_run \
    "stage3" \
    "${exp_stage3}" \
    "stage3_from_stage2b_bestwin_seed${seed}" \
    "${seed}" \
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

  stage3_dir="${root_dir}/${exp_stage3}/run${seed}"
  check_dir "${stage3_dir}" "Missing Stage3 fixed run dir for seed=${seed}"

  echo "============================================================"
  echo "Finished resume pipeline for seed=${seed}"
  echo "Stage1b source: ${stage1b_dir}"
  echo "Stage2a fixed:  ${stage2a_dir}"
  echo "Stage2b fixed:  ${stage2b_dir}"
  echo "Stage3 fixed:   ${stage3_dir}"
  echo "============================================================"
done