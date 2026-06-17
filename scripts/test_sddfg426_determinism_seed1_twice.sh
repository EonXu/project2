#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   cd /root/SDDFG/scripts   # 或者你的项目中包含 train/train_wolfpack.py 的目录
#   bash test_sddfg426_determinism_seed1_twice.sh
#
# 目标：同一份代码、同一参数、同一 seed 连续跑两遍短训练，然后比较关键 CSV 是否完全一致。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRAIN_PY=${TRAIN_PY:-train/train_wolfpack.py}
EXP=${EXP:-sddfg426_repro_patch_stage1a_seed1_twice_50k}
NAME=${NAME:-repro_patch_seed1}
SEED=${SEED:-1}
STEPS=${STEPS:-50000}

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] 找不到 ${TRAIN_PY}。请在包含 train/train_wolfpack.py 的项目目录运行，或设置 TRAIN_PY=/path/to/train_wolfpack.py"
  exit 1
fi

# 计算训练脚本实际会写入的 results 目录：dirname(dirname(TRAIN_PY))/results/wolfpack/sddfg/$EXP
RESULT_DIR=$(python - <<PY
from pathlib import Path
p = Path("${TRAIN_PY}").resolve()
print(p.parent.parent / "results" / "wolfpack" / "sddfg" / "${EXP}")
PY
)

echo "[INFO] 清理旧结果: ${RESULT_DIR}"
rm -rf "${RESULT_DIR}"

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
  --adj_begin_step 0

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
  --adj_begin_step 30000

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
  python "${TRAIN_PY}" "${COMMON_ARGS[@]}"
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
  else
    echo "[DIFF] ${stem}: 不一致"
    echo "       $f1"
    echo "       $f2"
    diff -u "$f1" "$f2" | head -80 || true
    return 1
  fi
}

status=0
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
  echo "========== FAIL: 仍存在非确定性，优先检查 DQNAgent.act/random.seed/torch 非确定 op =========="
fi
exit "$status"
