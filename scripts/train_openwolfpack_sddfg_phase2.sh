#!/bin/sh

# ==================== 1. 实验基础参数 ====================
env="wolfpack"
algo="sddfg"
exp="sddfg426_stage2"
name="stage2b_from_stage2a_add015"
seed=1

# ==================== 2. Stage1 A2 best_win 模型路径 ====================
model_dir="/root/SDDFG/scripts/results/wolfpack/sddfg/sddfg426_stage2/run11/models"

# ==================== 3. open-wolfpack 环境基础参数 ====================
wolfpack_id="wolfpack-v0"
grid_h=20
grid_w=20

obs_type="vector"
sight_sideways=8
sight_radius=8

max_food_num=2
food_freeze_rate=200


# ==================== 4. Stage2b 标准动态参数 ====================
num_agents=4
max_player_num=5

add_rate=0.15
del_rate=0.15

coop_radius=1
group_multiplier=2.0
close_penalty=0.05


# ==================== 5. SDDFG 因子图参数 ====================
msg_iterations=4
highest_orders=3
num_factor=3
sparsity=0.3
gain=0.01


# ==================== 6. GAT 图网络参数 ====================
gat_heads=4
gat_negative_slope=0.2
gat_hyperedge_hidden=64


# ==================== 7. 训练基础参数 ====================
num_env_steps=1000000
buffer_size=5000
batch_size=32

lr=3e-4
critic_lr=3e-4
adj_lr=1e-4

gamma=0.97
gae_lambda=0.95

train_interval_episode=4
hard_update_interval_episode=200


# ==================== 8. GAT / 邻接训练参数 ====================
adj_begin_step=0
adj_buffer_size=4
train_adj_episode=4
num_mini_batch=1

max_grad_norm=10
adj_max_grad_norm=0.5

clip_param=0.2
entropy_coef=0.005


# ==================== 9. 探索参数 ====================
epsilon_start=0.05
epsilon_finish=0.05
epsilon_anneal_time=500000

adj_anneal_time=500000
num_random_episodes=0


# ==================== 10. 学习率 floor ====================
policy_lr_decay_floor=1e-5
critic_lr_decay_floor=1e-5


# ==================== 11. 日志 / 评估 / 保存参数 ====================
log_interval=3000
eval_interval=20000
save_interval=50000
num_eval_episodes=5


echo "seed is ${seed}, running Stage2b from Stage2a, add/del=0.15"
echo "model_dir=${model_dir}"


CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py --env_name ${env} \
  --algorithm_name ${algo} --experiment_name ${exp} --wolfpack_id ${wolfpack_id} \
  --seed ${seed} --buffer_size ${buffer_size} --batch_size ${batch_size} \
  --hard_update_interval_episode ${hard_update_interval_episode} \
  --num_env_steps ${num_env_steps} \
  --log_interval ${log_interval} --eval_interval ${eval_interval} --save_interval ${save_interval} \
  --num_eval_episodes ${num_eval_episodes} --user_name ${name} \
  --model_dir ${model_dir} \
  --msg_iterations ${msg_iterations} --highest_orders ${highest_orders} --num_factor ${num_factor} \
  --num_agents ${num_agents} --max_player_num ${max_player_num} \
  --grid_height ${grid_h} --grid_width ${grid_w} --max_food_num ${max_food_num} \
  --obs_type ${obs_type} --sight_sideways ${sight_sideways} --sight_radius ${sight_radius} \
  --coop_radius ${coop_radius} --group_multiplier ${group_multiplier} --food_freeze_rate ${food_freeze_rate} \
  --close_penalty ${close_penalty} --add_rate ${add_rate} --del_rate ${del_rate} \
  --hidden_size 64 --gat_heads ${gat_heads} \
  --gat_negative_slope ${gat_negative_slope} --gat_hyperedge_hidden ${gat_hyperedge_hidden} \
  --lr ${lr} --critic_lr ${critic_lr} --adj_lr ${adj_lr} \
  --max_grad_norm ${max_grad_norm} --adj_max_grad_norm ${adj_max_grad_norm} \
  --entropy_coef ${entropy_coef} --clip_param ${clip_param} \
  --epsilon_start ${epsilon_start} --epsilon_finish ${epsilon_finish} \
  --epsilon_anneal_time ${epsilon_anneal_time} --adj_anneal_time ${adj_anneal_time} \
  --num_random_episodes ${num_random_episodes} \
  --gamma ${gamma} --gae_lambda ${gae_lambda} \
  --train_interval_episode ${train_interval_episode} --train_adj_episode ${train_adj_episode} \
  --adj_buffer_size ${adj_buffer_size} --num_mini_batch ${num_mini_batch} \
  --sparsity ${sparsity} --gain ${gain} --adj_begin_step ${adj_begin_step} \
  --policy_lr_decay_floor ${policy_lr_decay_floor} --critic_lr_decay_floor ${critic_lr_decay_floor} \
  --use_vfunction --use_wandb --use_linear_lr_decay --use_dyn_graph \
  --use_reward_normalization --use_valuenorm