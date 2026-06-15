#!/bin/sh

env="wolfpack"
algo="rddfg_cent_rw"
exp="ddfg-wolfpack-remove-only"
name="open_remove_only"
seed_min=1
seed_max=1

# Wolfpack 参数
num_agents=3
implicit_max_player_num=5
add_rate=0.0
del_rate=0.05
highest_orders=2
num_factor=5

echo "env is ${env}, algo is ${algo}, exp is ${exp}, seeds ${seed_min}..${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python train/train_wolfpack.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} \
    --seed ${seed} --buffer_size 5000 --batch_size 32 \
    --use_soft_update --hard_update_interval_episode 200 --num_env_steps 2000000 \
    --log_interval 3000 --eval_interval 20000 --user_name ${name} \
    --msg_iterations 4 --adj_output_dim 64 --highest_orders ${highest_orders} \
    --num_factor ${num_factor} --num_agents ${num_agents} --implicit_max_player_num ${num_agents} \
    --add_rate ${add_rate} --del_rate ${del_rate} \
    --lr 5e-4 --critic_lr 5e-4 --adj_lr 1e-8 --entropy_coef 0 \
    --use_wandb --use_vfunction --gamma 0.97 --num_rank 1 \
    --train_interval_episode 4 --train_adj_episode 4 --adj_buffer_size 4 --num_mini_batch 1 \
    --sparsity ${sparsity} --gain 0.01 --adj_begin_step 0 --gae_lambda 0.95 --use_linear_lr_decay --use_dyn_graph
done
