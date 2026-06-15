#!/bin/sh
env="wolfpack"
algo="sddfg"
exp="sddfg-stage3-extreme"
name="open_remove_add_sddfg"

# Wolfpack 参数
wolfpack_id="wolfpack-v0"
grid_h=20
grid_w=20

# 观测：vector / partial_obs / full_rgb
obs_type="vector"
sight_sideways=8
sight_radius=8

# 围捕/奖励
coop_radius=1
group_multiplier=2.0
food_freeze_rate=200
close_penalty=0.1
max_food_num=2

add_rate=0.2              # 【核心挑战】高频人员加入
del_rate=0.2             # 【核心挑战】高频人员掉线

adj_lr=1e-5                # 【极低学习率】微调图网络，冻结核心拓扑逻辑
entropy_coef=0.001         # 【极低探索熵】转换为确定性图注意力，追求极致执行力

num_agents=4      # 初始在场数（会动态变化）
max_player_num=5  # 固定槽位数

# SDDFG 超参数
gat_heads=4                # GAT 多头注意力
gat_negative_slope=0.2     # GAT 负斜率

# === SDDFG 算法通用参数 ===
msg_iterations=4
adj_output_dim=64
highest_orders=3
sparsity=0.3
gain=0.01
adj_begin_step=0
gae_lambda=0.95
gamma=0.97
lr=5e-4
critic_lr=5e-4


train_interval_episode=4
train_adj_episode=4
adj_buffer_size=4
hard_update_interval_episode=200
num_env_steps=2000000      # 阶段二跑 2M 步足以验证鲁棒性

buffer_size=5000
batch_size=32
num_mini_batch=1

log_interval=3000
eval_interval=20000

# 【必填：热启动路径】请替换为你刚跑完的 stage2-perfect 的真实模型路径
MODEL_DIR="/root/SDDFG/scripts/results/wolfpack/sddfg/sddfg-stage2-dynamic/run1/models"


seed_min=1
seed_max=1

for seed in $(seq ${seed_min} ${seed_max}); do
  echo "seed is ${seed}, running SDDFG:"
  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp}  --wolfpack_id ${wolfpack_id} \
    --seed ${seed} --buffer_size ${buffer_size} --batch_size ${batch_size} \
    --use_soft_update --hard_update_interval_episode ${hard_update_interval_episode} --num_env_steps ${num_env_steps} \
    --log_interval ${log_interval}  --eval_interval ${eval_interval} --user_name ${name} \
    --msg_iterations ${msg_iterations}  --adj_output_dim ${adj_output_dim} \
    --highest_orders ${highest_orders}  --num_agents ${num_agents} --max_player_num ${max_player_num} \
    --grid_height ${grid_h} --grid_width ${grid_w} --max_food_num ${max_food_num} \
    --obs_type ${obs_type}  --sight_sideways ${sight_sideways} --sight_radius ${sight_radius} \
    --coop_radius ${coop_radius} --group_multiplier ${group_multiplier} --food_freeze_rate ${food_freeze_rate} \
    --close_penalty ${close_penalty} --add_rate ${add_rate} --del_rate ${del_rate} \
    --lr ${lr} --critic_lr ${critic_lr} --adj_lr ${adj_lr} --entropy_coef ${entropy_coef} \
    --use_vfunction --use_wandb --gamma ${gamma} \
    --train_interval_episode ${train_interval_episode} --train_adj_episode ${train_adj_episode} \
    --adj_buffer_size ${adj_buffer_size} --num_mini_batch ${num_mini_batch} \
    --sparsity ${sparsity} --gain ${gain}  --adj_begin_step ${adj_begin_step} --gae_lambda ${gae_lambda} \
    --use_linear_lr_decay --use_dyn_graph \
    --use_reward_normalization --use_valuenorm
done


##!/bin/sh
#env="wolfpack"
#algo="sddfg"
#exp="sddfg-stage2-dynamic"
#name="open_remove_add_sddfg"
#
## Wolfpack 参数
#wolfpack_id="wolfpack-v0"
#grid_h=20
#grid_w=20
#
## 观测：vector / partial_obs / full_rgb
#obs_type="vector"
#sight_sideways=8
#sight_radius=8
#
## 围捕/奖励
#coop_radius=1
#group_multiplier=2.0
#food_freeze_rate=200
#close_penalty=0.05         # 【保持不变】维持黄金基线的进攻欲望
#max_food_num=2
#
#add_rate=0.15              # 【核心挑战】高频人员加入
#del_rate=0.15              # 【核心挑战】高频人员掉线
#
#adj_lr=1e-4                # 【平滑过渡】降低学习率，微调图网络
#entropy_coef=0.005         # 【平滑过渡】降低探索熵，提升构图确定性
#
#num_agents=4      # 初始在场数（会动态变化）
#max_player_num=5  # 固定槽位数
#
## SDDFG 超参数
#gat_heads=4                # GAT 多头注意力
#gat_negative_slope=0.2     # GAT 负斜率
#
## === SDDFG 算法通用参数 ===
#msg_iterations=4
#adj_output_dim=64
#highest_orders=3
#sparsity=0.3
#gain=0.01
#adj_begin_step=0
#gae_lambda=0.95
#gamma=0.97
#lr=5e-4
#critic_lr=5e-4
#
#
#train_interval_episode=4
#train_adj_episode=4
#adj_buffer_size=4
#hard_update_interval_episode=200
#num_env_steps=2000000      # 阶段二跑 2M 步足以验证鲁棒性
#
#buffer_size=5000
#batch_size=32
#num_mini_batch=1
#
#log_interval=3000
#eval_interval=20000
#
## 【必填：热启动路径】请替换为你刚跑完的 stage1-perfect 的真实模型路径
#MODEL_DIR="/root/SDDFG/scripts/results//wolfpack/sddfg/sddfg-stage1-warmup/run3(close_penalty=0.1)/models"
#
#
#seed_min=1
#seed_max=1
#
#for seed in $(seq ${seed_min} ${seed_max}); do
#  echo "seed is ${seed}, running SDDFG:"
#  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py --env_name ${env} \
#    --algorithm_name ${algo} --experiment_name ${exp}  --wolfpack_id ${wolfpack_id} \
#    --seed ${seed} --buffer_size ${buffer_size} --batch_size ${batch_size} \
#    --use_soft_update --hard_update_interval_episode ${hard_update_interval_episode} --num_env_steps ${num_env_steps} \
#    --log_interval ${log_interval}  --eval_interval ${eval_interval} --user_name ${name} \
#    --msg_iterations ${msg_iterations}  --adj_output_dim ${adj_output_dim} \
#    --highest_orders ${highest_orders}  --num_agents ${num_agents} --max_player_num ${max_player_num} \
#    --grid_height ${grid_h} --grid_width ${grid_w} --max_food_num ${max_food_num} \
#    --obs_type ${obs_type}  --sight_sideways ${sight_sideways} --sight_radius ${sight_radius} \
#    --coop_radius ${coop_radius} --group_multiplier ${group_multiplier} --food_freeze_rate ${food_freeze_rate} \
#    --close_penalty ${close_penalty} --add_rate ${add_rate} --del_rate ${del_rate} \
#    --lr ${lr} --critic_lr ${critic_lr} --adj_lr ${adj_lr} --entropy_coef ${entropy_coef} \
#    --use_vfunction --use_wandb --gamma ${gamma} \
#    --train_interval_episode ${train_interval_episode} --train_adj_episode ${train_adj_episode} \
#    --adj_buffer_size ${adj_buffer_size} --num_mini_batch ${num_mini_batch} \
#    --sparsity ${sparsity} --gain ${gain}  --adj_begin_step ${adj_begin_step} --gae_lambda ${gae_lambda} \
#    --use_linear_lr_decay --use_dyn_graph \
#    --use_reward_normalization --use_valuenorm
#done

##!/bin/sh
#env="wolfpack"
#algo="sddfg"
#exp="sddfg-stage1-warmup"  # 明确标明这是第一阶段
#name="open_remove_add_sddfg"
#
## Wolfpack 参数
#wolfpack_id="wolfpack-v0"
#grid_h=20
#grid_w=20
#
## 观测：vector / partial_obs / full_rgb
#obs_type="vector"
#sight_sideways=8
#sight_radius=8
#
## 围捕/奖励
#coop_radius=1
#group_multiplier=2.0
#food_freeze_rate=200
#close_penalty=0.05
#max_food_num=2
#
#add_rate=0.05
#del_rate=0.05
#
#adj_lr=3e-4
#entropy_coef=0.015          # 【核心微调】稍微提升探索，打破挂机局部最优
#
#num_agents=4      # 初始在场数（会动态变化）
#max_player_num=5  # 固定槽位数
#
## SDDFG 超参数
#gat_heads=4                # GAT 多头注意力
#gat_negative_slope=0.2     # GAT 负斜率
#
## === SDDFG 算法通用参数 ===
#msg_iterations=4
#adj_output_dim=64
#highest_orders=3
#sparsity=0.3
#gain=0.01
#adj_begin_step=0
#gae_lambda=0.95
#gamma=0.97
#lr=5e-4
#critic_lr=5e-4
#
#
#train_interval_episode=4
#train_adj_episode=4
#adj_buffer_size=4
#hard_update_interval_episode=200
#num_env_steps=2000000      # 【增加】给足 2M 步让其完美收敛
#
#buffer_size=5000
#batch_size=32
#num_mini_batch=1
#
#log_interval=3000
#eval_interval=20000
#
## 预训练模型路径 (如需从 DDFG 权重热启动可保留，否则设为空或注释掉)
## model_dir="/root/SDDFG/scripts/results/wolfpack/ddfg/..."
#
#seed_min=1
#seed_max=1
#
#for seed in $(seq ${seed_min} ${seed_max}); do
#  echo "seed is ${seed}, running SDDFG:"
#  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py --env_name ${env} \
#    --algorithm_name ${algo} --experiment_name ${exp}  --wolfpack_id ${wolfpack_id} \
#    --seed ${seed} --buffer_size ${buffer_size} --batch_size ${batch_size} \
#    --use_soft_update --hard_update_interval_episode ${hard_update_interval_episode} --num_env_steps ${num_env_steps} \
#    --log_interval ${log_interval}  --eval_interval ${eval_interval} --user_name ${name} \
#    --msg_iterations ${msg_iterations}  --adj_output_dim ${adj_output_dim} \
#    --highest_orders ${highest_orders}  --num_agents ${num_agents} --max_player_num ${max_player_num} \
#    --grid_height ${grid_h} --grid_width ${grid_w} --max_food_num ${max_food_num} \
#    --obs_type ${obs_type}  --sight_sideways ${sight_sideways} --sight_radius ${sight_radius} \
#    --coop_radius ${coop_radius} --group_multiplier ${group_multiplier} --food_freeze_rate ${food_freeze_rate} \
#    --close_penalty ${close_penalty} --add_rate ${add_rate} --del_rate ${del_rate} \
#    --lr ${lr} --critic_lr ${critic_lr} --adj_lr ${adj_lr} --entropy_coef ${entropy_coef} \
#    --use_vfunction --use_wandb --gamma ${gamma} \
#    --train_interval_episode ${train_interval_episode} --train_adj_episode ${train_adj_episode} \
#    --adj_buffer_size ${adj_buffer_size} --num_mini_batch ${num_mini_batch} \
#    --sparsity ${sparsity} --gain ${gain}  --adj_begin_step ${adj_begin_step} --gae_lambda ${gae_lambda} \
#    --use_linear_lr_decay --use_dyn_graph \
#    --use_reward_normalization --use_valuenorm
#done