import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # 静音，避免 ALSA 找声卡
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # 可选：隐藏 pygame 提示

import sys

import math
import argparse
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import torch
import gym

from pathlib import Path

# --------- robust project root detection ----------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

# 向上寻找包含 config.py 的工程根目录（例如 /root/SDDFG）
while True:
    if (PROJECT_ROOT / "config.py").exists():
        break
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        # 找不到就退回到脚本所在目录的上一级（尽量不直接崩）
        PROJECT_ROOT = THIS_FILE.parent.parent
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)
# -----------------------------------------------

from config import get_config

from utils.util import get_cent_act_dim, get_dim_from_space
from runner.wolfpack_runner import WolfpackRunner as Runner

import envs.Wolfpack
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def _parse_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, bool)):
        return x
    s = str(x).strip()
    if s == "":
        return ""
    if s.lower() in ["none", "null"]:
        return None
    if s.lower() in ["true", "false"]:
        return s.lower() == "true"
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def load_args_from_config_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    d = {}
    for _, row in df.iterrows():
        name = str(row["Name"])
        d[name] = _parse_value(row["Value"])
    return d


def apply_dict_to_namespace(ns, d: dict):
    for k, v in d.items():
        try:
            setattr(ns, k, v)
        except Exception:
            pass


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "wolfpack":
                raise NotImplementedError(f"Unsupported env: {all_args.env_name}")

            env_seed = all_args.seed * 1000 + rank

            wolf_kwargs = dict(
                grid_height=all_args.grid_height,
                grid_width=all_args.grid_width,
                num_agents=all_args.num_agents,
                max_food_num=all_args.max_food_num,
                sight_sideways=all_args.sight_sideways,
                sight_radius=all_args.sight_radius,
                max_time_steps=all_args.episode_length,
                coop_radius=all_args.coop_radius,
                food_freeze_rate=all_args.food_freeze_rate,
                seed=env_seed,
                add_rate=all_args.add_rate,
                del_rate=all_args.del_rate,
                close_penalty=all_args.close_penalty,
                prey_with_gpu=all_args.prey_with_gpu,
                obs_type=all_args.obs_type,
                with_random_grid=all_args.with_random_grid,
                random_grid_dir=all_args.random_grid_dir,
            )

            # 兼容不同环境版本的参数名：group_multiplier vs groupMultiplier
            if hasattr(all_args, "group_multiplier"):
                wolf_kwargs["group_multiplier"] = all_args.group_multiplier
            elif hasattr(all_args, "groupMultiplier"):
                wolf_kwargs["groupMultiplier"] = all_args.groupMultiplier

            # 可选：如果你的环境支持 max_player_num，建议同步传入以保证维度一致
            if hasattr(all_args, "max_player_num"):
                wolf_kwargs["max_player_num"] = all_args.max_player_num

            # 创建环境：若遇到参数名不匹配，自动重试转换
            try:
                env = gym.make(all_args.wolfpack_id, **wolf_kwargs)
            except TypeError as e:
                msg = str(e)

                # 若不接受 group_multiplier，则改为 groupMultiplier 重试
                if "unexpected keyword argument 'group_multiplier'" in msg and "group_multiplier" in wolf_kwargs:
                    v = wolf_kwargs.pop("group_multiplier")
                    wolf_kwargs["groupMultiplier"] = v
                    env = gym.make(all_args.wolfpack_id, **wolf_kwargs)

                # 若不接受 groupMultiplier，则改为 group_multiplier 重试
                elif "unexpected keyword argument 'groupMultiplier'" in msg and "groupMultiplier" in wolf_kwargs:
                    v = wolf_kwargs.pop("groupMultiplier")
                    wolf_kwargs["group_multiplier"] = v
                    env = gym.make(all_args.wolfpack_id, **wolf_kwargs)

                else:
                    raise

            return env


        return init_env

    if int(getattr(all_args, "n_eval_rollout_threads", 1)) == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def build_static_adj(all_args, N: int) -> torch.Tensor:
    adj = torch.zeros((N, all_args.num_factor), dtype=torch.int64)
    index = 0
    n = 0

    static_graph_algos = ["ddfg", "sddfg", "rddfg_low", "rmfg_cent", "rmdfg", "rmdfg_cent"]

    if (all_args.use_dyn_graph == False and all_args.equal_vdn == False
            and all_args.algorithm_name in static_graph_algos):
        for i in range(N - 1):
            for j in range(i + 1, N):
                if index >= all_args.num_factor:
                    break
                adj[i, index] = 1
                adj[j, index] = 1
                index += 1

        # 剩余 factor 用一阶 self-like factor 填充，保持原 DDFG 静态图风格
        n = 0
        for i in range(index, all_args.num_factor):
            adj[n % N, i] = 1
            n += 1

    return adj
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--run_dir", type=str, required=True, help="训练输出的 run 目录（里面有 config.csv 和 models/）")
    cli.add_argument("--out", type=str, default="demo.mp4", help="输出 mp4 路径")
    cli.add_argument("--fps", type=float, default=10.0, help="播放与录制帧率")
    cli.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    cli.add_argument("--seed", type=int, default=None, help="可选：覆盖 config.csv 里的 seed")
    cli.add_argument("--no_window", action="store_true", help="无窗口模式（服务器无显示时可用）")
    args = cli.parse_args()

    run_dir = Path(args.run_dir)
    cfg_csv = run_dir / "config.csv"
    if not cfg_csv.exists():
        raise FileNotFoundError(f"找不到 {cfg_csv}，请确认 run_dir 是否正确。")

    cfg_dict = load_args_from_config_csv(str(cfg_csv))

    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    apply_dict_to_namespace(all_args, cfg_dict)

    if args.seed is not None:
        all_args.seed = int(args.seed)

    all_args.use_wandb = False
    all_args.use_eval = True

    all_args.n_training_threads = 1
    all_args.n_eval_rollout_threads = 1

    if str(args.device).lower().startswith("cuda"):
        all_args.cuda = True
    else:
        all_args.cuda = False

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if getattr(all_args, "cuda_deterministic", False):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    models_dir = run_dir / "models"
    all_args.model_dir = str(models_dir)

    all_args.skip_warmup = True
    all_args.render = (not args.no_window)
    all_args.render_fps = float(args.fps)
    all_args.save_video = True
    all_args.video_path = str(Path(args.out))

    env = make_eval_env(all_args)
    eval_env = env

    N = int(all_args.max_player_num)
    K = int(getattr(all_args, "highest_orders", 2))
    sparsity = float(getattr(all_args, "sparsity", 0.5))

    if (all_args.use_dyn_graph == False and all_args.equal_vdn == False
            and all_args.algorithm_name in ["ddfg", "sddfg", "rddfg_low", "rmfg_cent", "rmdfg", "rmdfg_cent"]):
        # 与 build_static_adj 的 pair + self-like factor 保持一致
        all_args.num_factor = N * (N - 1) // 2 + N
    else:
        if getattr(all_args, "num_factor", None) is None:
            num_factor_full = math.factorial(N) // (math.factorial(K) * math.factorial(N - K))
            calculated_num_factor = int(num_factor_full * sparsity)
            all_args.num_factor = max(1, calculated_num_factor)

    if all_args.share_policy:
        policy_info = {
            "policy_0": {
                "cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[0],
                "share_obs_space": env.share_observation_space[0],
                "act_space": env.action_space[0],
            }
        }

        def policy_mapping_fn(_agent_id):
            return "policy_0"
    else:
        policy_info = {
            f"policy_{agent_id}": {
                "cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[agent_id],
                "share_obs_space": env.share_observation_space[agent_id],
                "act_space": env.action_space[agent_id],
            }
            for agent_id in range(N)
        }

        def policy_mapping_fn(agent_id):
            return f"policy_{agent_id}"

    adj = build_static_adj(all_args, N)

    config = {
        "args": all_args,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": env,
        "eval_env": eval_env,
        "num_agents": N,
        "device": device,
        "run_dir": run_dir,
        "use_same_share_obs": all_args.use_same_share_obs,
        "use_available_actions": all_args.use_available_actions,
        "adj": adj,
    }

    runner = Runner(config=config)

    print("\n开始回放：窗口实时显示 + 同步保存 mp4")
    print(f"models_dir: {models_dir}")
    print(f"video_out : {all_args.video_path}")
    print(f"fps       : {all_args.render_fps}\n")

    runner.collect_rollout(explore=False, training_episode=False, warmup=False)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
