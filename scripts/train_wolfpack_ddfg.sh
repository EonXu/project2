#!/bin/sh
env="wolfpack"
algo="ddfg"
exp="ddfg-open-wolfpack-remove-only"
name="open_remove_only"
seed_min=1
seed_max=1

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
close_penalty=0
max_food_num=2

# open：回合内增减玩家概率（(建议先用 0/0 跑通，再逐步调大)
add_rate=0.0
del_rate=0.0

num_agents=4      # 初始在场数（会动态变化）
max_player_num=5       # 固定槽位数（DDFG 的 num_agents）

# DDFG/
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
adj_lr=1e-4

train_interval_episode=4
train_adj_episode=4
adj_buffer_size=4
hard_update_interval_episode=200

# 通用训练参数
num_env_steps=2000000

buffer_size=5000
batch_size=32
num_mini_batch=1

log_interval=3000
eval_interval=20000


echo "env=${env}, algo=${algo}, exp=${exp}, seeds=[${seed_min},${seed_max}]"

for seed in $(seq ${seed_min} ${seed_max}); do
  echo "seed is ${seed}:"
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
    --lr ${lr} --critic_lr ${critic_lr} --adj_lr ${adj_lr} --entropy_coef 0.001 \
    --use_vfunction --use_wandb --gamma ${gamma} \
    --train_interval_episode ${train_interval_episode} --train_adj_episode ${train_adj_episode} \
    --adj_buffer_size ${adj_buffer_size} --num_mini_batch ${num_mini_batch} \
    --sparsity ${sparsity} --gain ${gain}  --adj_begin_step ${adj_begin_step} --gae_lambda ${gae_lambda} \
    --use_linear_lr_decay --use_dyn_graph \
    --use_reward_normalization --use_valuenorm
done


