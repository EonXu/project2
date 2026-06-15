#!/bin/sh
env="wolfpack"
algo="sddfg"
exp="sddfg426-wolfpack-open-phase1"
name="phase1_dyn005_c01_g6_f8"

# ==================== open-wolfpack 环境参数 ====================
wolfpack_id="wolfpack-v0"              # gym 注册环境 ID
grid_h=20                              # 地图高度；中等地图，避免奖励过稀疏
grid_w=20                              # 地图宽度；与高度保持一致

obs_type="vector"                      # 向量观测，训练最稳定
sight_sideways=8                       # 横向视野范围；中等局部观测难度
sight_radius=8                         # 前向/纵向视野范围；保证早期可学习性

max_food_num=2                         # 猎物数量；保持协作捕猎任务密度适中
food_freeze_rate=200                   # 猎物冻结步长；降低环境过度非平稳性

# ==================== 协同捕猎奖励与难度参数 ====================
coop_radius=1                          # 协作捕猎半径
group_multiplier=6.0                   # 增强协作捕获奖励，提升 win 信号
close_penalty=0.1                      # 降低单狼惩罚，允许探索接近 prey

# ====================  动态智能体数量参数 ====================
num_agents=3                           # 初始在场智能体数量；协作起点适中
max_player_num=5                       # 固定智能体槽位数量；用于 mask 与 GAT 建图
add_rate=0.005                          # 每步新增智能体概率；中等动态强度
del_rate=0.005                          # 每步删除智能体概率；与新增概率平衡

# ==================== SDDFG 参数 ====================
msg_iterations=4                       # 因子图消息传递轮数；保证结构信息充分传播
highest_orders=3                       # 最高三阶 factor；验证高阶协作建模能力
num_factor=8                           # 减少空 factor，提高结构利用率
sparsity=0.3                           # 图稀疏度参数；保留便于后续容量消融
gain=0.01                              # 网络初始化增益；降低早期 Q 值震荡
gat_heads=4                            # GAT 多头数；与 hidden_size=64 整除
gat_negative_slope=0.2                 # GAT LeakyReLU 负斜率；标准稳定值
gat_hyperedge_hidden=64                # 增强三阶 hyperedge scorer

# ==================== GAT / 邻接训练稳定参数 ====================
adj_begin_step=30000                   # 更晚启动 GAT 训练，避免随机策略污染结构
adj_buffer_size=64                     # 邻接 buffer
train_adj_episode=16                   # 降低 GAT 更新频率
num_mini_batch=4                       # mini-batch 数

max_grad_norm=10                       # 策略网络梯度裁剪
adj_max_grad_norm=0.5                  # GAT 邻接网络梯度裁剪；防止结构学习爆炸

clip_param=0.2                         # 邻接 PPO ratio 裁剪范围
entropy_coef=0.003                     # 增强结构探索，减少过早固定

# ==================== 训练基础参数 ====================
num_env_steps=2000000                  # 总环境交互步数；单 seed 主实验规模
buffer_size=5000                       # 经验池容量；覆盖足够动态轨迹
batch_size=32                          # batch 大小；兼顾稳定性与显存占用

lr=5e-4                                # 策略 / Q 网络学习率
critic_lr=5e-4                         # value / critic 学习率
adj_lr=1e-4                            # GAT 邻接网络学习率；避免邻接 PPO 震荡

gamma=0.99                             # 折扣因子；协作捕猎需要长期信用分配
gae_lambda=0.95                        # GAE 参数；稳定邻接 advantage 估计

train_interval_episode=4               # 策略训练间隔；保持与回放采样节奏匹配
hard_update_interval_episode=200       # target 网络硬更新间隔

# ==================== 探索参数 ====================
epsilon_start=1.0                      # 动作初始探索率
epsilon_finish=0.05                    # 动作最终探索率
epsilon_anneal_time=1500000            # 动作探索退火步数

adj_anneal_time=1000000                # 延长 GAT 邻接探索
num_random_episodes=8                  # 随机预热 episode

# ==================== 日志 / 评估 / 保存参数 ====================
log_interval=3000                      # 日志记录间隔；降低动态日志 I/O
eval_interval=20000                    # 评估间隔；平衡评估开销与趋势观察
save_interval=50000                    # 模型保存间隔；便于回放与恢复
num_eval_episodes=5                    # 每次评估 episode 数；降低动态环境随机方差

seed_min=1
seed_max=1

for seed in $(seq ${seed_min} ${seed_max}); do
  echo "seed is ${seed}, running SDDFG"
  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py --env_name ${env} \
  --algorithm_name ${algo} --experiment_name ${exp} --wolfpack_id ${wolfpack_id} \
  --seed ${seed} --buffer_size ${buffer_size} --batch_size ${batch_size} \
  --use_soft_update --hard_update_interval_episode ${hard_update_interval_episode} \
  --num_env_steps ${num_env_steps} \
  --log_interval ${log_interval} --eval_interval ${eval_interval} --save_interval ${save_interval} \
  --num_eval_episodes ${num_eval_episodes} --user_name ${name} \
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
  --use_vfunction --use_wandb --use_linear_lr_decay --use_dyn_graph \
  --use_reward_normalization --use_valuenorm
done