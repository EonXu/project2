import math
import sys
import gym
import os
from pathlib import Path
import wandb
import socket
import setproctitle
import numpy as np
import pandas as pd
import torch
from config import get_config
from utils.util import get_cent_act_dim, get_dim_from_space
from runner.wolfpack_runner import WolfpackRunner as Runner

import envs.Wolfpack
from envs.env_wrappers import ShareDummyVecEnv

"""
与 train_*.py 一致的多进程封装：
- 通过 seed + rank * 1000 区分每个并行环境；
- 这里用 gym.make('wolfpack', **kwargs)；wolfpack 在 envs/Wolfpack/__init__.py 注册；
- 注意：WolfpackPenaltyOpen 没有 env.seed() 方法，所以需要通过 kwargs 传 seed；
       若存在 env.seed()，再调用一次也无妨，代码中已用 hasattr 保护。
"""


def make_train_env(all_args):
    """
    用 gym.make('wolfpack', **kwargs)；wolfpack 在 envs/Wolfpack/__init__.py 注册；
    - 注意：WolfpackPenaltyOpen 没有 env.seed() 方法，所以需要通过 kwargs 传 seed；
           若存在 env.seed()，再调用一次也无妨，代码中已用 hasattr 保护。
    """
    def get_env_fn(rank):
        def init_env():
            # 仅支持 wolfpack
            if all_args.env_name != "wolfpack":
                raise NotImplementedError(f"Unsupported env: {all_args.env_name}")

            # 组装 wolfpack 的 kwargs（全部来自 all_args）
            # max_player_num 统一由 num_agents 提供；只减少模式把 add_rate 强制置 0
            wolf_kwargs = dict(
                grid_height=all_args.grid_height,
                grid_width=all_args.grid_width,
                num_players=all_args.num_agents, #num_agent为初始的智能体数量
                max_time_steps=all_args.episode_length,
                coop_radius=all_args.coop_radius,
                groupMultiplier=all_args.group_multiplier,
                food_freeze_rate=all_args.food_freeze_rate,
                add_rate=all_args.add_rate,
                del_rate=all_args.del_rate,
                seed=all_args.seed,
                max_player_num=all_args.implicit_max_player_num,
                implicit_max_player_num=all_args.implicit_max_player_num,
                obs_type=all_args.obs_type,
                with_random_grid=all_args.with_random_grid,
                random_grid_dir=all_args.random_grid_dir,
                prey_with_gpu=all_args.prey_with_gpu,
                close_penalty=all_args.close_penalty,
                sight_sideways=all_args.sight_sideways,
                sight_radius=all_args.sight_radius
            )

            env = gym.make("wolfpack", **wolf_kwargs)
            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])


"""
   评估环境与训练环境相同，只是种子区间拉开（与 train_*.py 一致的做法）：
   使用 seed * 50000 + rank * 10000，避免和训练采样重叠。
   """


# -----------------------------
# 环境构建（评估）
# -----------------------------
def make_eval_env(all_args):
    """
    评估环境与训练环境相同，只是种子区间拉开（与 train_*.py 一致的做法）：
    使用 seed * 1000 避免和训练采样重叠。
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "wolfpack":
                raise NotImplementedError(f"Unsupported env: {all_args.env_name}")

            env_seed = all_args.seed * 1000

            wolf_kwargs = dict(
                grid_height=all_args.grid_height,
                grid_width=all_args.grid_width,
                num_players=all_args.num_agents,  # num_agent为初始的智能体数量
                max_time_steps=all_args.episode_length,
                coop_radius=all_args.coop_radius,
                groupMultiplier=all_args.group_multiplier,
                food_freeze_rate=all_args.food_freeze_rate,
                add_rate=all_args.add_rate,
                del_rate=all_args.del_rate,
                seed=all_args.seed,
                max_player_num=all_args.implicit_max_player_num,
                implicit_max_player_num=all_args.implicit_max_player_num,
                obs_type=all_args.obs_type,
                with_random_grid=all_args.with_random_grid,
                random_grid_dir=all_args.random_grid_dir,
                prey_with_gpu=all_args.prey_with_gpu,
                close_penalty=all_args.close_penalty,
                sight_sideways=all_args.sight_sideways,
                sight_radius=all_args.sight_radius
            )

            env = gym.make("wolfpack", **wolf_kwargs)
            return env
        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])


def parse_args(args, parser):

    # Wolfpack 环境专属（与注册缺省一致，命令行可覆盖）
    parser.add_argument("--grid_height", type=int, default=20,
                        help="网格高度")
    parser.add_argument("--grid_width", type=int, default=20,
                        help="网格宽度")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="回合开始时实际在场的玩家数量 （ < implicit_max_player_num）")
    parser.add_argument("--implicit_max_player_num", type=int, default=5,
                        help="开放队伍规模调度器可用的最大容量（ > num_agents）")

    parser.add_argument("--close_penalty", type=float, default=0.5,
                        help="单狼贴近惩罚")
    parser.add_argument("--coop_radius", type=int, default=1,
                        help="围捕判定的邻近半径（曼哈顿距离）")
    parser.add_argument("--group_multiplier", type=float, default=2.0,
                        help="协作奖励乘子")
    parser.add_argument("--food_freeze_rate", type=int, default=0,
                        help="食物被围捕后冻结步长")

    # 开放调度：为“只减少”模式预留配置
    parser.add_argument("--add_rate", type=float, default=0.0,
                        help="每步添加玩家的概率（只减少时=0）")
    parser.add_argument("--del_rate", type=float, default=0.05,
                        help="每步删除玩家的概率")

    parser.add_argument("--sight_sideways", type=int, default=8,
                        help="部分可观测时的横向视野")
    parser.add_argument("--sight_radius", type=int, default=8,
                        help="部分可观测时的纵向/前向半径")
    parser.add_argument("--obs_type", type=str, default="vector",
                        help="观测类型：vector/partial_obs/full_rgb")
    parser.add_argument("--prey_with_gpu", action="store_true", default=False,
                        help="食物 DQN 是否用 GPU（通常 False）")

    parser.add_argument("--with_random_grid", action="store_true", default=False,
                        help="是否用元胞自动机随机图（默认 False 为全可走）")
    parser.add_argument("--random_grid_dir", type=str, default=None,
                        help="载入预保存随机地图的文件路径（pkl），优先级高于 with_random_grid")


    parser.add_argument('--use_available_actions', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--num_factor', type=int,
                        default=10, help="number of factor")
    parser.add_argument('--highest_orders', type=int,
                        default=3, help="number of agents")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                     1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env = make_train_env(all_args)
    num_agents = all_args.num_agents

    if all_args.use_dyn_graph == False and all_args.equal_vdn == False and all_args.algorithm_name in ["rmdfg",
                                                                                                       "rmdfg_cent",
                                                                                                       "rddfg_cent_rw",
                                                                                                       "rmfg_cent"]:
        num_factor = num_agents * (num_agents - 1) // 2 + num_agents
    else:
        num_factor = int(math.factorial(num_agents) // (math.factorial(all_args.highest_orders) * math.factorial(
            num_agents - all_args.highest_orders)) * all_args.sparsity)
    all_args.num_factor = num_factor
    all_args.num_agents = num_agents
    # create policies and mapping fn
    if all_args.share_policy:
        print(env.share_observation_space[0])
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0],
                         "unit_dim": env.envs[0].unit_dim}
        }

        def policy_mapping_fn(id):
            return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id):
            return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qtran", "wqmix", "qmix", "vdn", "qplex",
                                   "rddfg_cent_rw", "rddfg_low", "rmfg_cent", "sopcg", "casec"]:
        assert all_args.n_rollout_threads == 1, ("only support 1 env in recurrent version.")
        eval_env = make_train_env(all_args)
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn", 'mqtran']:
        # from offpolicy.runner.mlp.smac_runner import SMACRunner as Runner
        # eval_env = make_eval_env(all_args)
        print("尚未实现此部分")
    else:
        raise NotImplementedError

    adj = torch.zeros((num_agents, num_factor), dtype=torch.int64)
    index = 0
    n = 0
    if all_args.use_dyn_graph == False and all_args.equal_vdn == False and all_args.algorithm_name in ["rddfg_cent_rw",
                                                                                                       "rddfg_low",
                                                                                                       "rmfg_cent"]:
        for i in range(num_agents - 1):
            for j in range(i + 1, num_agents):
                adj[i, index] = 1
                adj[j, index] = 1
                index = index + 1
        for i in range(index, num_factor):
            adj[n, i] = 1
            n = n + 1

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "run_dir": run_dir,
              "use_same_share_obs": all_args.use_same_share_obs,
              "use_available_actions": all_args.use_available_actions,
              "adj": adj}

    progress_filename = os.path.join(run_dir, 'config.csv')
    df = pd.DataFrame(list(all_args.__dict__.items()), columns=['Name', 'Value'])
    df.to_csv(progress_filename, index=False)

    progress_filename = os.path.join(run_dir, 'progress.csv')
    df = pd.DataFrame(columns=['step', 'reward'])
    df.to_csv(progress_filename, index=False)

    progress_filename = os.path.join(run_dir, 'progress_eval.csv')
    df = pd.DataFrame(columns=['step', 'reward'])
    df.to_csv(progress_filename, index=False)

    progress_filename_train = os.path.join(run_dir, 'progress_train.csv')
    df = pd.DataFrame(columns=['step', 'loss', 'loss_v', 'loss_fv'])
    df.to_csv(progress_filename_train, index=False)

    progress_filename_train = os.path.join(run_dir, 'progress_train_adj.csv')
    df = pd.DataFrame(columns=['step', 'advantage', 'clamp_ratio', 'rl_loss', 'auto_loss'])
    df.to_csv(progress_filename_train, index=False)

    total_num_steps = 0

    runner = Runner(config=config)

    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])