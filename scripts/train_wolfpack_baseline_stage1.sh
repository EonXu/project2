#!/bin/sh

alg=$1

if [ -z "$alg" ]; then
  echo "Usage: bash train_baseline_stage1.sh {vdn|qmix|qplex}"
  exit 1
fi

env="wolfpack"
exp="${alg}_wolfpack_stage1"
name="${alg}_stage1"

wolfpack_id="wolfpack-v0"

grid_h=20
grid_w=20
obs_type="vector"
sight_sideways=8
sight_radius=8

num_agents=4
max_player_num=5
max_food_num=2

coop_radius=1
group_multiplier=2.0
close_penalty=0.05
food_freeze_rate=200

add_rate=0.05
del_rate=0.05

num_env_steps=2000000
episode_length=200
buffer_size=5000
batch_size=32

lr=5e-4
gamma=0.97

train_interval_episode=4
hard_update_interval_episode=200

epsilon_start=1.0
epsilon_finish=0.05
epsilon_anneal_time=1000000
num_random_episodes=5

log_interval=3000
eval_interval=20000
save_interval=50000
num_eval_episodes=5

mixer_args=""
if [ "$alg" = "qmix" ]; then
  mixer_args="--mixer_hidden_dim 32 --hypernet_hidden_dim 64 --hypernet_layers 2"
fi

if [ "$alg" = "qplex" ]; then
  mixer_args="--mixer_hidden_dim 32 --hypernet_hidden_dim 64 --hypernet_layers 2 --n_head 4 --num_kernel 10 --adv_hypernet_embed 64 --adv_hypernet_layers 3 --attend_reg_coef 0.001 --state_bias --weighted_head --nonlinear"
fi

for seed in 1 2 3 4 5; do
  echo "Running ${alg} Stage1, seed=${seed}"

  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py \
    --env_name ${env} \
    --algorithm_name ${alg} \
    --experiment_name ${exp} \
    --user_name ${name} \
    --seed ${seed} \
    --wolfpack_id ${wolfpack_id} \
    --num_env_steps ${num_env_steps} \
    --episode_length ${episode_length} \
    --buffer_size ${buffer_size} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --gamma ${gamma} \
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
    ${mixer_args} \
    --use_wandb
done