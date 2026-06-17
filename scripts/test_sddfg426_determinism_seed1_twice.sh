#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   cd /root/SDDFG/scripts
#   bash test_sddfg426_determinism_seed1_twice.sh
#
# 可选：
#   CUDA_VISIBLE_DEVICES="" bash test_sddfg426_determinism_seed1_twice.sh   # CPU 测试
#   STEPS=100000 bash test_sddfg426_determinism_seed1_twice.sh
#
# 目标：同一份代码、同一参数、同一 seed 连续跑两遍短训练，然后比较关键 CSV 是否 bitwise 完全一致。

# ===== 必须在 python 进程启动前设置 =====
# 注意：PYTHONHASHSEED 在 Python 运行后再 os.environ 设置已经太晚，所以必须写在 bash 里。
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 保留用户显式设置：
#   未设置 CUDA_VISIBLE_DEVICES -> 默认用 0
#   设置为空 CUDA_VISIBLE_DEVICES="" -> CPU / 无 GPU 可见
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

TRAIN_PY=${TRAIN_PY:-train/train_wolfpack.py}
EXP=${EXP:-sddfg426_repro_patch_stage1a_seed1_twice_50k}
NAME=${NAME:-repro_patch_seed1}
SEED=${SEED:-1}
STEPS=${STEPS:-50000}

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] 找不到 ${TRAIN_PY}。请在包含 train/train_wolfpack.py 的项目目录运行，或设置 TRAIN_PY=/path/to/train_wolfpack.py"
  exit 1
fi

PROJECT_ROOT=$(python - <<PY
from pathlib import Path
p = Path("${TRAIN_PY}").resolve()
print(p.parent.parent)
PY
)

# 训练脚本实际写入目录：PROJECT_ROOT/results/wolfpack/sddfg/$EXP
RESULT_DIR="${PROJECT_ROOT}/results/wolfpack/sddfg/${EXP}"

# ===== 复现性预检查：这里只报警，不自动改代码 =====
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
echo "[INFO] PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "[INFO] CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"

if grep -R "list(set" -n "${PROJECT_ROOT}/envs/Wolfpack" >/tmp/sddfg_repro_list_set.txt 2>/dev/null; then
  echo "[WARN] 仍检测到 list(set(...))，这会导致跨进程顺序不稳定。请优先检查："
  cat /tmp/sddfg_repro_list_set.txt
fi

if grep -R "random\." -n "${PROJECT_ROOT}/train" "${PROJECT_ROOT}/envs" "${PROJECT_ROOT}/algorithms" "${PROJECT_ROOT}/runner" "${PROJECT_ROOT}/utils" 2>/dev/null | grep -v __pycache__ >/tmp/sddfg_repro_random_dot.txt; then
  echo "[WARN] 检测到 random.xxx 调用。若这些调用在训练路径中，会破坏 bitwise 复现："
  cat /tmp/sddfg_repro_random_dot.txt | head -30
fi

if grep -R "np.random\.\|torch.rand\|randperm\|Categorical\|OneHotCategorical" -n "${PROJECT_ROOT}/train" "${PROJECT_ROOT}/envs" "${PROJECT_ROOT}/algorithms" "${PROJECT_ROOT}/runner" "${PROJECT_ROOT}/utils" 2>/dev/null | grep -v __pycache__ >/tmp/sddfg_repro_rng_calls.txt; then
  echo "[INFO] 检测到的 RNG/采样调用，若已改为私有 self.rng 或只在 seed 初始化中使用可忽略："
  cat /tmp/sddfg_repro_rng_calls.txt | head -50
fi

# ===== 清理旧结果 =====
echo "[INFO] 清理旧结果: ${RESULT_DIR}"
rm -rf "${RESULT_DIR}"

# 如果 CUDA_VISIBLE_DEVICES 为空，显式给 train_wolfpack.py 传 --cuda，
# 因为该项目 config.py 里 --cuda 是 action='store_false'，传入 --cuda 表示禁用 CUDA。
CUDA_ARG=()
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_ARG=(--cuda)
fi

COMMON_ARGS=(
  --env_name wolfpack
  --algorithm_name sddfg
  --experiment_name "${EXP}"
  --wolfpack_id wolfpack-v0
  --seed "${SEED}"
  --use_wandb

  --num_env_steps "${STEPS}"
  --episode_length 200
  --buffer_size 5000
  --batch_size 32
  --adj_buffer_size 4

  --lr 5e-4
  --critic_lr 5e-4
  --adj_lr 3e-4
  --gamma 0.97
  --gae_lambda 0.95
  --hard_update_interval_episode 200
  --train_interval_episode 4
  --train_adj_episode 4

  # 与 stage1a 对齐，避免 adj_begin_step=0 在早期引入 GAT/adj 训练噪声。
  --adj_begin_step 30000

  --epsilon_start 1.0
  --epsilon_finish 0.05
  --epsilon_anneal_time 1000000
  --adj_anneal_time 500000
  --num_random_episodes 5

  --log_interval 3000
  --eval_interval 20000
  --save_interval 50000
  --num_eval_episodes 1

  --msg_iterations 4
  --highest_orders 3
  --num_factor 3
  --use_dyn_graph
  --use_vfunction

  --num_agents 4
  --max_player_num 5
  --grid_height 20
  --grid_width 20
  --max_food_num 2
  --obs_type vector
  --sight_sideways 8
  --sight_radius 8
  --coop_radius 1
  --group_multiplier 2.0
  --food_freeze_rate 200
  --add_rate 0.05
  --del_rate 0.05
  --close_penalty 0.05

  --gat_heads 4
  --gat_negative_slope 0.2
  --gat_hyperedge_hidden 64
  --max_grad_norm 10
  --adj_max_grad_norm 0.5
  --clip_param 0.2
  --entropy_coef 0.015
  --num_mini_batch 1

  --user_name "${NAME}"
)

for i in 1 2; do
  echo "========== Repro run ${i}/2: seed=${SEED}, steps=${STEPS}, exp=${EXP} =========="
  python "${TRAIN_PY}" "${COMMON_ARGS[@]}" "${CUDA_ARG[@]}"
done

RUN1="${RESULT_DIR}/run1"
RUN2="${RESULT_DIR}/run2"

if [[ ! -d "${RUN1}" || ! -d "${RUN2}" ]]; then
  echo "[ERROR] 未找到 run1/run2：${RESULT_DIR}"
  find "${RESULT_DIR}" -maxdepth 2 -type f 2>/dev/null || true
  exit 1
fi

compare_one() {
  local stem="$1"
  local f1="${RUN1}/run1_${stem}.csv"
  local f2="${RUN2}/run2_${stem}.csv"

  if [[ ! -f "$f1" || ! -f "$f2" ]]; then
    echo "[SKIP] ${stem}: 文件缺失"
    return 0
  fi

  if cmp -s "$f1" "$f2"; then
    echo "[OK] ${stem}: 完全一致"
    return 0
  fi

  echo "[DIFF] ${stem}: 不一致"
  echo "       $f1"
  echo "       $f2"
  echo "------- first diff, unified diff head -------"
  diff -u "$f1" "$f2" | head -100 || true
  echo "---------------------------------------------"
  return 1
}

compare_plain() {
  local label="$1"
  local f1="$2"
  local f2="$3"

  if [[ ! -f "$f1" || ! -f "$f2" ]]; then
    echo "[SKIP] ${label}: 文件缺失"
    return 0
  fi

  if cmp -s "$f1" "$f2"; then
    echo "[OK] ${label}: 完全一致"
    return 0
  fi

  echo "[DIFF] ${label}: 不一致"
  diff -u "$f1" "$f2" | head -80 || true
  return 1
}

status=0

# config.csv 通常不带 run 前缀，单独比较。
compare_plain config "${RUN1}/config.csv" "${RUN2}/config.csv" || status=1

compare_one progress || status=1
compare_one progress_train || status=1
compare_one progress_train_num_players_traj || status=1
compare_one progress_train_active_masks_traj || status=1
compare_one progress_train_individual_rewards || status=1
compare_one progress_train_adj || status=1
compare_one progress_train_adj_metrics_traj || status=1
compare_one progress_eval || status=1
compare_one progress_eval_num_players_traj || status=1
compare_one progress_eval_individual_rewards || status=1

if [[ "$status" -eq 0 ]]; then
  echo "========== PASS: 两次同 seed 短训练关键 CSV 完全一致 =========="
else
  echo "========== FAIL: 仍存在非确定性 =========="
  echo "优先级建议："
  echo "  1) 若 progress_train_num_players_traj 一致但 individual_rewards 分叉，检查 add_agent 坐标、hunter 动作采样、prey 动作。"
  echo "  2) 若新增 agent 后 slot4/slot新增位置先分叉，优先检查 envs/Wolfpack/wolfpack_penalty_open.py 里的 list(set(...))。"
  echo "  3) 若 CPU PASS 但 GPU FAIL，优先怀疑旧 PyTorch/CUDA 非确定 op。"
fi

exit "$status"