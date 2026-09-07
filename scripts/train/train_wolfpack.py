import os, sys
import hashlib
import json
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import random
import gym
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
from runner import base_runner as base_runner_module
from utils.wolfpack_reward import (
    WOLFPACK_DISTANCE_SHAPING_SCALE,
    WOLFPACK_LEGACY_DISTANCE_MODE,
    WOLFPACK_MULTI_PREY_COVERAGE_MODE,
    WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION,
)

import envs.Wolfpack
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


_EXPECTED_PAIR_EXACT_SCORE_RECORDING_CONTRACT_VERSION = 2


def _validate_pair_exact_score_recording_contract():
    actual = getattr(
        base_runner_module,
        "PAIR_EXACT_SCORE_RECORDING_CONTRACT_VERSION",
        None,
    )
    expected = _EXPECTED_PAIR_EXACT_SCORE_RECORDING_CONTRACT_VERSION
    if actual != expected:
        raise RuntimeError(
            "stale runner/base_runner.py exact-score recording contract: "
            "expected={}, actual={}; synchronize runner/base_runner.py and "
            "scripts/debug_pair_optimizer_transaction_diagnostics.py before "
            "starting training".format(expected, actual)
        )

def set_global_seeds(seed: int, cuda_deterministic: bool = True):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # 兼容新旧 PyTorch
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                print(f"[WARN] use_deterministic_algorithms failed: {e}")
        elif hasattr(torch, "set_deterministic"):
            try:
                torch.set_deterministic(True)
            except Exception as e:
                print(f"[WARN] set_deterministic failed: {e}")
        else:
            print("[WARN] Current torch has no strict deterministic API; "
                  "only cudnn.deterministic=True is used.")

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

def get_run_csv_path(run_dir, filename):
    """
    生成带 run 前缀的 CSV 路径。
    例如 run1 + progress.csv -> run1_progress.csv。
    """
    filename = str(filename)
    run_name = os.path.basename(str(run_dir).rstrip(os.sep))

    if filename.endswith(".csv") and run_name.startswith("run"):
        prefix = run_name + "_"
        if not filename.startswith(prefix):
            filename = prefix + filename

    return os.path.join(run_dir, filename)


def persist_source_manifest(run_dir, all_args):
    """Copy and verify the launch-time per-file source manifest."""
    source_path = os.environ.get("SDDFG_SOURCE_MANIFEST_PATH", "")
    expected_sha256 = os.environ.get(
        "SDDFG_SOURCE_MANIFEST_SHA256",
        "",
    )
    all_args.source_manifest_sha256 = expected_sha256 or "unavailable"
    all_args.source_manifest_file = "unavailable"
    if not source_path:
        return
    if not os.path.isfile(source_path):
        raise RuntimeError(
            "declared SDDFG source manifest does not exist: {}".format(
                source_path
            )
        )
    with open(source_path, "rb") as manifest_file:
        manifest_bytes = manifest_file.read()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "SDDFG source manifest hash changed between launch and run setup"
        )
    destination = os.path.join(str(run_dir), "source_manifest.sha256")
    temporary = destination + ".tmp"
    with open(temporary, "wb") as manifest_file:
        manifest_file.write(manifest_bytes)
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(temporary, destination)
    all_args.source_manifest_sha256 = actual_sha256
    all_args.source_manifest_file = os.path.basename(destination)


def _finite_json_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_terminal_summary(runner, all_args, total_num_steps):
    """Atomically write a compact, non-empty terminal run summary."""
    last_eval = {}
    for key, value in getattr(runner, "last_eval_summary", {}).items():
        finite_value = _finite_json_float(value)
        if finite_value is not None:
            last_eval[str(key)] = finite_value
    if bool(getattr(all_args, "use_eval", False)) and not last_eval:
        raise RuntimeError(
            "terminal run summary requires the final evaluation metrics"
        )
    best = {}
    for output_key, attribute_name in (
            ("reward", "best_eval_reward"),
            ("reward_step", "best_eval_reward_step"),
            ("win_rate", "best_eval_win_rate"),
            ("win_reward", "best_eval_win_reward"),
            ("win_step", "best_eval_win_step")):
        finite_value = _finite_json_float(getattr(
            runner,
            attribute_name,
            None,
        ))
        if finite_value is not None:
            best[output_key] = finite_value

    payload = {
        "summary_schema_version": 1,
        "resolved_run_id": str(getattr(
            all_args,
            "resolved_run_id",
            "unavailable",
        )),
        "resolved_run_dir": str(getattr(
            all_args,
            "resolved_run_dir",
            "unavailable",
        )),
        "algorithm": str(all_args.algorithm_name),
        "experiment": str(all_args.experiment_name),
        "seed": int(all_args.seed),
        "actual_env_steps": int(total_num_steps),
        "target_env_steps": int(all_args.num_env_steps),
        "training_complete": int(total_num_steps) >= int(all_args.num_env_steps),
        "checkpoint_kind": "terminal",
        "last_eval_step": int(getattr(
            runner,
            "last_eval_summary_step",
            -1,
        )),
        "last_eval": last_eval,
        "best_eval": best,
        "source_manifest_file": str(getattr(
            all_args,
            "source_manifest_file",
            "unavailable",
        )),
        "source_manifest_sha256": str(getattr(
            all_args,
            "source_manifest_sha256",
            "unavailable",
        )),
    }
    if str(all_args.algorithm_name) == "sddfg":
        payload["algorithm_definitions"] = {
            "candidate_objective_version": 12,
            "candidate_normalization_version": 3,
            "candidate_lifecycle_version": 10,
        }
        payload["policy_exploration"] = {
            "contract_version": 6,
            "joint_epsilon_exploration_enabled": bool(getattr(
                all_args,
                "use_joint_epsilon_exploration",
                False,
            )),
            "epsilon_start": float(all_args.epsilon_start),
            "epsilon_finish": float(all_args.epsilon_finish),
            "epsilon_anneal_time": int(all_args.epsilon_anneal_time),
            "post_capture_joint_greedy_floor": float(getattr(
                all_args,
                "post_capture_joint_greedy_floor",
                0.0,
            )),
            "post_capture_explore_max_random_agents": int(getattr(
                all_args,
                "post_capture_explore_max_random_agents",
                0,
            )),
            "pre_capture_visible_prey_quorum_guard": bool(getattr(
                all_args,
                "pre_capture_visible_prey_quorum_guard",
                False,
            )),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard": bool(
                getattr(
                    all_args,
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard",
                    False,
                )
            ),
        }
        payload["q_target"] = {
            "contract_version": 5,
            "n_step": int(getattr(all_args, "q_n_step", 1)),
            "n_step_mode": (
                "one_step"
                if int(getattr(all_args, "q_n_step", 1)) == 1
                else "terminal_gated"
            ),
            "terminal_replay_lane": bool(getattr(
                all_args,
                "q_terminal_replay_lane",
                False,
            )),
            "terminal_replay_loss_weight": float(getattr(
                all_args,
                "q_terminal_replay_loss_weight",
                0.10,
            )),
            "gamma": float(all_args.gamma),
            "pre_capture_frontier_policy_aligned": bool(getattr(
                all_args,
                "pre_capture_visible_prey_quorum_greedy_frontier_guard",
                False,
            )),
        }
        coverage_enabled = bool(getattr(
            all_args,
            "use_multi_prey_coverage_shaping",
            False,
        ))
        payload["reward_shaping"] = {
            "contract_version": WOLFPACK_REWARD_SHAPING_CONTRACT_VERSION,
            "mode": (
                WOLFPACK_MULTI_PREY_COVERAGE_MODE
                if coverage_enabled else WOLFPACK_LEGACY_DISTANCE_MODE
            ),
            "distance_scale": WOLFPACK_DISTANCE_SHAPING_SCALE,
        }
    summary_path = os.path.join(str(runner.log_dir), "summary.json")
    temporary = summary_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as summary_file:
        json.dump(
            payload,
            summary_file,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        summary_file.write("\n")
        summary_file.flush()
        os.fsync(summary_file.fileno())
    os.replace(temporary, summary_path)

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
            wolf_kwargs = dict(
                grid_height=all_args.grid_height,
                grid_width=all_args.grid_width,
                num_agents=all_args.num_agents,  # 初始在场数（会动态变化）
                max_food_num=all_args.max_food_num,
                sight_sideways=all_args.sight_sideways,
                sight_radius=all_args.sight_radius,
                max_time_steps=all_args.episode_length,
                coop_radius=all_args.coop_radius,
                groupMultiplier=all_args.group_multiplier,
                food_freeze_rate=all_args.food_freeze_rate,
                add_rate=all_args.add_rate,
                del_rate=all_args.del_rate,
                seed=all_args.seed + rank,
                max_player_num=all_args.max_player_num,  # 固定槽位数（DDFG 用它作为 num_agents）
                obs_type=all_args.obs_type,
                with_random_grid=all_args.with_random_grid,
                random_grid_dir=all_args.random_grid_dir,
                prey_with_gpu=all_args.prey_with_gpu,
                close_penalty=all_args.close_penalty,
                intra_episode_dynamic=all_args.intra_episode_dynamic,
                shock_steps=all_args.shock_steps,
                shock_remove_num=all_args.shock_remove_num,
                shock_join_delay=all_args.shock_join_delay,
                shock_join_num=all_args.shock_join_num,
                shock_recover_delay=all_args.shock_recover_delay,
                dynamic_min_agents=all_args.dynamic_min_agents,
                continue_after_success=all_args.continue_after_success,
                reward_win=all_args.reward_win,
                use_multi_prey_coverage_shaping=(
                    all_args.use_multi_prey_coverage_shaping
                ),
            )

            env = gym.make(all_args.wolfpack_id, **wolf_kwargs)
            return env

        return init_env

    if all_args.n_training_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_training_threads)])


"""
   评估环境与训练环境相同，只是种子区间拉开（与 train_*.py 一致的做法）：
   使用 seed * 1000，避免和训练采样重叠。
   """
def make_eval_env(all_args):
    """
    评估环境与训练环境相同，只是种子区间拉开（与 train_*.py 一致的做法）：
    使用 seed * 1000 避免和训练采样重叠。
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "wolfpack":
                raise NotImplementedError(f"Unsupported env: {all_args.env_name}")

            env_seed = all_args.seed * 1000 + rank

            wolf_kwargs = dict(
                grid_height=all_args.grid_height,
                grid_width=all_args.grid_width,
                num_agents=all_args.num_agents,  # 初始在场数（会动态变化）
                max_food_num=all_args.max_food_num,
                sight_sideways=all_args.sight_sideways,
                sight_radius=all_args.sight_radius,
                max_time_steps=all_args.episode_length, #等价于 episode_limit
                coop_radius=all_args.coop_radius,
                groupMultiplier=all_args.group_multiplier,
                food_freeze_rate=all_args.food_freeze_rate,
                add_rate=all_args.add_rate,
                del_rate=all_args.del_rate,
                seed=env_seed,
                max_player_num=all_args.max_player_num,  # 固定槽位数（DDFG 用它作为 num_agents）
                obs_type=all_args.obs_type,
                with_random_grid=all_args.with_random_grid,
                random_grid_dir=all_args.random_grid_dir,
                prey_with_gpu=all_args.prey_with_gpu,
                close_penalty=all_args.close_penalty,
                intra_episode_dynamic=all_args.intra_episode_dynamic,
                shock_steps=all_args.shock_steps,
                shock_remove_num=all_args.shock_remove_num,
                shock_join_delay=all_args.shock_join_delay,
                shock_join_num=all_args.shock_join_num,
                shock_recover_delay=all_args.shock_recover_delay,
                dynamic_min_agents=all_args.dynamic_min_agents,
                continue_after_success=all_args.continue_after_success,
                reward_win=all_args.reward_win,
                use_multi_prey_coverage_shaping=(
                    all_args.use_multi_prey_coverage_shaping
                ),
            )

            env = gym.make(all_args.wolfpack_id, **wolf_kwargs)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """
    解析 wolfpack 相关的环境参数。
    注意：DDFG/Runner 侧的“固定智能体数”应使用 max_player_num（槽位数），
    环境内部的 num_agents 仅表示“初始在场玩家数”，后续可能动态增减。
    """
    # gym 注册的环境 id
    parser.add_argument("--wolfpack_id", type=str, default="wolfpack-v0")

    # 地图/格网
    parser.add_argument("--grid_height", type=int, default=20,
                        help="网格高度")
    parser.add_argument("--grid_width", type=int, default=20,
                        help="网格宽度")
    parser.add_argument("--with_random_grid", action="store_true", default=False,
                        help="是否用元胞自动机随机图（默认 False 为全可走）")
    parser.add_argument("--random_grid_dir", type=str, default=None,
                        help="载入预保存随机地图的文件路径（pkl），优先级高于 with_random_grid")

    # 玩家/猎物
    parser.add_argument("--num_agents", type=int, default=3, help="初始在场玩家数（会动态变化）")
    parser.add_argument("--max_player_num", type=int, default=5, help="固定槽位数（DDFG 的 num_agents）")
    parser.add_argument("--max_food_num", type=int, default=2)
    parser.add_argument('--num_factor', type=int, default=None, help="number of factor")

    # 视野/回合长度
    parser.add_argument("--sight_sideways", type=int, default=8,
                        help="部分可观测时的横向视野")
    parser.add_argument("--sight_radius", type=int, default=8,
                        help="部分可观测时的纵向/前向半径")

    # 围捕/奖励
    parser.add_argument("--coop_radius", type=int, default=1,
                        help="围捕判定的邻近半径（曼哈顿距离）")
    parser.add_argument("--group_multiplier", type=float, default=2.0, help="协作奖励乘子")
    parser.add_argument("--food_freeze_rate", type=int, default=200, help="食物被围捕后冻结步长")
    parser.add_argument("--close_penalty", type=float, default=1.0, help="单狼贴近惩罚")
    parser.add_argument(
        "--use_multi_prey_coverage_shaping",
        action="store_true",
        default=False,
        help=(
            "Use reward-only balanced alive-prey coverage potential instead "
            "of independent nearest-prey distance shaping"
        ),
    )

    # open 环境：回合内增减玩家的几何分布参数
    parser.add_argument("--add_rate", type=float, default=0.0,
                        help="每步添加玩家的概率")
    parser.add_argument("--del_rate", type=float, default=0.1,
                        help="每步删除玩家的概率")

    parser.add_argument(
        "--intra_episode_dynamic",
        action="store_true",
        default=False,
        help="启用单回合内脚本化退出、加入和恢复"
    )
    parser.add_argument(
        "--shock_steps",
        type=str,
        default="",
        help="突发退出时间步，逗号分隔，例如 50,120"
    )
    parser.add_argument(
        "--shock_remove_num",
        type=int,
        default=0,
        help="每次突发退出的智能体数量"
    )
    parser.add_argument(
        "--shock_join_delay",
        type=int,
        default=10,
        help="突发退出后多少步加入新成员"
    )
    parser.add_argument(
        "--shock_join_num",
        type=int,
        default=0,
        help="每次突发事件后加入的新成员数量"
    )
    parser.add_argument(
        "--shock_recover_delay",
        type=int,
        default=30,
        help="退出成员恢复所需步数，必须大于 0"
    )
    parser.add_argument(
        "--dynamic_min_agents",
        type=int,
        default=2,
        help="回合内至少保留的在线智能体数量"
    )
    parser.add_argument(
        "--continue_after_success",
        action="store_true",
        default=False,
        help="全部 prey 暂时被冻结后不提前结束，继续到 episode_length"
    )

    # 观测类型
    parser.add_argument("--obs_type", type=str, default="vector",
                        help="观测类型：vector/partial_obs/full_rgb")

    # 其它
    parser.add_argument("--prey_with_gpu", action="store_true", default=False,
                        help="食物 DQN 是否用 GPU（通常 False）")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_available_actions', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--highest_orders', type=int, default=3, help="number of agents")

    def add_missing_option(option, *option_args, **option_kwargs):
        if option not in parser._option_string_actions:
            parser.add_argument(option, *option_args, **option_kwargs)

    add_missing_option(
        "--use_train_consistent_eval_graph",
        action="store_true",
        default=False,
        help=(
            "Use the current stochastic training graph behavior during "
            "evaluation, with greedy policy actions and an isolated eval RNG."
        ),
    )
    add_missing_option(
        "--use_adj_topology_persistence",
        action="store_true",
        default=False,
        help=(
            "Redirect graph greedy-mixture mass to the previous same-slot "
            "factor when it remains eligible, with exact PPO probability."
        ),
    )

    add_missing_option(
        "--adj_recent_episode_window",
        type=int,
        default=0,
        help="Train adjacency PPO only from the most recent N adj-buffer episodes.",
    )
    add_missing_option(
        "--use_adj_dynamic_recent_window",
        action="store_true",
        default=False,
        help="Adapt the adjacency PPO recent replay window using stale ratios.",
    )
    add_missing_option(
        "--adj_recent_episode_window_min",
        type=int,
        default=1,
        help="Minimum adaptive adjacency PPO recent replay window.",
    )
    add_missing_option(
        "--adj_recent_window_stale_threshold",
        type=float,
        default=0.35,
        help="Graph stale-ratio threshold for adaptive recent replay.",
    )
    add_missing_option(
        "--adj_recent_window_factor_stale_threshold",
        type=float,
        default=0.30,
        help="Factor stale-ratio threshold for adaptive recent replay.",
    )
    add_missing_option(
        "--adj_recent_window_shrink_patience",
        type=int,
        default=1,
        help="Consecutive high-stale updates before shrinking recent replay.",
    )
    add_missing_option(
        "--adj_recent_window_recover_patience",
        type=int,
        default=2,
        help="Consecutive low-stale updates before recovering recent replay.",
    )
    add_missing_option(
        "--adj_recent_window_recover_stale_threshold",
        type=float,
        default=-1.0,
        help="Graph stale threshold for recent-window recovery.",
    )
    add_missing_option(
        "--adj_recent_window_recover_factor_stale_threshold",
        type=float,
        default=-1.0,
        help="Factor stale threshold for recent-window recovery.",
    )
    add_missing_option(
        "--adj_recent_window_severe_margin",
        type=float,
        default=0.15,
        help="Extra stale margin that shrinks recent replay to the minimum.",
    )
    add_missing_option(
        "--adj_recent_episode_window_emergency",
        type=int,
        default=1,
        help="Near-on-policy recent replay window for emergency high-stale updates.",
    )
    add_missing_option(
        "--adj_recent_window_emergency_stale_threshold",
        type=float,
        default=0.45,
        help="Graph stale threshold for immediate emergency recent-window shrink.",
    )
    add_missing_option(
        "--adj_recent_window_emergency_factor_stale_threshold",
        type=float,
        default=0.35,
        help="Factor stale threshold for immediate emergency recent-window shrink.",
    )
    add_missing_option(
        "--use_adj_triplet_credit_direct_rank",
        action="store_true",
        default=False,
        help="Use marginal triplet credit as a direct ranking bias.",
    )
    add_missing_option(
        "--adj_triplet_credit_rank_coef",
        type=float,
        default=0.0,
        help="Direct ranking-bias strength for marginal triplet credit.",
    )
    add_missing_option(
        "--adj_triplet_credit_min_multiplier",
        type=float,
        default=0.25,
        help="Minimum direct triplet-credit ranking multiplier.",
    )
    add_missing_option(
        "--adj_triplet_credit_max_multiplier",
        type=float,
        default=3.0,
        help="Maximum direct triplet-credit ranking multiplier.",
    )
    add_missing_option(
        "--adj_triplet_credit_negative_rank_scale",
        type=float,
        default=1.0,
        help="Scale for negative marginal-credit direct ranking.",
    )
    add_missing_option(
        "--adj_triplet_credit_min_positive_fraction",
        type=float,
        default=0.0,
        help="Candidate positive-fraction floor before strong negative rank.",
    )
    add_missing_option(
        "--adj_triplet_graph_return_credit_require_delayed_gate",
        action="store_true",
        default=False,
        help="Require delayed success-window evidence for triplet graph-return credit.",
    )
    add_missing_option(
        "--adj_delayed_triplet_credit_require_future_match",
        action="store_true",
        default=False,
        help=(
            "Credit delayed triplet rewards only when the same triplet node "
            "set reappears inside the future credit window."
        ),
    )
    add_missing_option(
        "--use_adj_delayed_triplet_success_gate",
        action="store_true",
        default=False,
        help="Gate delayed triplet credit by graph-level future-window reward advantage.",
    )
    add_missing_option(
        "--adj_delayed_triplet_success_gate_min_adv",
        type=float,
        default=0.0,
        help="Minimum graph-level future-window advantage for delayed triplet credit.",
    )
    add_missing_option(
        "--adj_delayed_triplet_success_gate_scale",
        type=float,
        default=1.0,
        help="Soft ramp scale for delayed triplet success gating.",
    )
    add_missing_option(
        "--adj_delayed_triplet_success_gate_floor",
        type=float,
        default=0.0,
        help="Minimum valid-transition success-gate weight.",
    )
    add_missing_option(
        "--adj_delayed_triplet_future_overlap_min_nodes",
        type=int,
        default=3,
        help="Minimum overlap for future triplet pursuit-group credit.",
    )
    add_missing_option(
        "--adj_delayed_triplet_partial_match_weight",
        type=float,
        default=0.5,
        help="Credit multiplier for partial future triplet matches.",
    )
    add_missing_option(
        "--use_adj_capture_to_win_credit",
        action="store_true",
        default=False,
        help="Add conservative triplet credit when capture evidence leads to winning episodes.",
    )
    add_missing_option(
        "--adj_capture_to_win_credit_coef",
        type=float,
        default=0.0,
        help="Coefficient for capture-to-win triplet credit.",
    )
    add_missing_option(
        "--adj_capture_to_win_credit_min_outcome_adv",
        type=float,
        default=0.5,
        help="Minimum episode outcome advantage for capture-to-win credit.",
    )
    add_missing_option(
        "--adj_capture_to_win_credit_scale",
        type=float,
        default=0.75,
        help="Soft ramp scale for capture-to-win credit.",
    )
    add_missing_option(
        "--adj_capture_to_win_credit_cap",
        type=float,
        default=0.35,
        help="Credit cap relative to graph-advantage scale for capture-to-win credit.",
    )
    add_missing_option(
        "--adj_capture_to_win_credit_require_future_match",
        action="store_true",
        default=False,
        help="Require future triplet match evidence for capture-to-win credit.",
    )
    add_missing_option(
        "--use_adj_pair_triplet_complementary_credit",
        action="store_true",
        default=False,
        help=(
            "Credit a pair-only backbone only when it becomes part of a "
            "strictly later capture_count triplet."
        ),
    )
    add_missing_option(
        "--adj_pair_pursuit_credit_coef",
        type=float,
        default=0.0,
        help="Coefficient for strict-future pair-to-capture-triplet credit.",
    )
    add_missing_option(
        "--adj_pair_pursuit_credit_window",
        type=int,
        default=20,
        help="Future window for pair-to-triplet pursuit credit.",
    )
    add_missing_option(
        "--adj_pair_pursuit_credit_cap",
        type=float,
        default=0.20,
        help="Credit cap for pair-to-triplet pursuit credit.",
    )
    add_missing_option(
        "--adj_pair_pursuit_credit_min_reward",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option; capture-anchored pair credit "
            "uses capture_count and ignores team reward."
        ),
    )
    add_missing_option(
        "--pair_bounded_pending_evidence",
        action="store_true",
        default=False,
        help=(
            "Enable the default-off pair-specific immutable bounded pending "
            "mechanism."
        ),
    )
    add_missing_option(
        "--pair_pending_max_adj_updates",
        type=int,
        default=0,
        help=(
            "Maximum adjacency-update age after ordinary replay eviction for "
            "strict pair pending evidence; zero preserves the legacy path."
        ),
    )

    all_args, unknown_args = parser.parse_known_args(args)
    if unknown_args:
        raise ValueError(
            "Unknown command line arguments were ignored by the parser: "
            + " ".join(unknown_args)
        )

    if all_args.intra_episode_dynamic:
        try:
            shock_step_values = sorted({
                int(value.strip())
                for value in all_args.shock_steps.split(",")
                if value.strip()
            })
        except ValueError as exc:
            raise ValueError("--shock_steps must be comma-separated integers") from exc

        if not shock_step_values:
            raise ValueError("--shock_steps cannot be empty in intra-episode dynamic mode")
        if all_args.shock_recover_delay <= 0:
            raise ValueError("--shock_recover_delay must be > 0")
        if all_args.num_agents > all_args.max_player_num:
            raise ValueError("num_agents cannot exceed max_player_num")
        if not 1 <= all_args.dynamic_min_agents <= all_args.num_agents:
            raise ValueError("dynamic_min_agents must be in [1, num_agents]")
        if all_args.shock_remove_num <= 0:
            raise ValueError("shock_remove_num must be > 0")
        if all_args.shock_remove_num > all_args.num_agents - all_args.dynamic_min_agents:
            raise ValueError("shock_remove_num would violate dynamic_min_agents at the first shock")
        if all_args.shock_join_num < 0 or all_args.shock_join_delay < 0:
            raise ValueError("shock_join_num and shock_join_delay must be >= 0")
        if any(step <= 0 or step >= all_args.episode_length for step in shock_step_values):
            raise ValueError("every shock step must be in (0, episode_length)")

        last_required_step = max(
            shock_step_values[-1] + all_args.shock_recover_delay,
            shock_step_values[-1] + all_args.shock_join_delay,
        )
        if last_required_step > all_args.episode_length:
            raise ValueError("the final join/recovery event exceeds episode_length")

        peak_agents = all_args.num_agents + len(shock_step_values) * all_args.shock_join_num
        if peak_agents > all_args.max_player_num:
            raise ValueError(
                "max_player_num is too small to hold all joined and recovered members: "
                f"required={peak_agents}, configured={all_args.max_player_num}"
            )

        # 脚本化事件与随机调度器互斥，保证事件可复现。
        if all_args.add_rate != 0.0 or all_args.del_rate != 0.0:
            raise ValueError(
                "scripted intra-episode dynamics require --add_rate 0 --del_rate 0"
            )

        if all_args.algorithm_name == "sddfg" and not all_args.use_dyn_graph:
            raise ValueError(
                "SDDFG intra-episode dynamics require --use_dyn_graph"
            )

    return all_args

def _init_adj(all_args, num_agents: int, num_factor: int):
    """
    初始化“静态邻接矩阵”adj（形状: [num_agents, num_factor]）。
    仅在 use_dyn_graph=False 且算法需要固定因子图时使用；否则保持全 0。
    """
    adj = torch.zeros((num_agents, num_factor), dtype=torch.int64)

    # 与其他 train 脚本保持一致：只在特定算法且非动态图的情况下做“先验结构”初始化
    alg_need_adj = all_args.algorithm_name in ["rddfg_cent_rw", "rmfg_cent", "sopcg", "casec", "rddfg_low"]
    if (not getattr(all_args, "use_dyn_graph", False)) and (not getattr(all_args, "equal_vdn", False)) and alg_need_adj:
        from itertools import combinations

        k = int(getattr(all_args, "highest_orders", 2))
        combos = list(combinations(range(num_agents), k))
        idx = 0
        for comb in combos:
            if idx >= num_factor:
                break
            for a in comb:
                adj[a, idx] = 1
            idx += 1

        # 如果 num_factor 大于组合数，剩余因子用“单点因子”填充，避免全空
        for j in range(idx, num_factor):
            adj[j % num_agents, j] = 1

    return adj

def main(args):
    _validate_pair_exact_score_recording_contract()
    parser = get_config()
    all_args = parse_args(args, parser)

    # 随机种子
    set_global_seeds(all_args.seed, all_args.cuda_deterministic)

    # 设备选择（与其他 train 一致）
    # 设备选择（与其他 train 一致）
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)

        if all_args.cuda_deterministic:
            # 关键：禁用 cuDNN，避免 GRU/RNN backward 走 cuDNN 非确定实现
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # 防止 Ampere/新卡上 TF32 引入额外数值差异；旧 torch 没这些属性也不报错
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass

            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

            # 旧 torch 可能没有 use_deterministic_algorithms，所以加保护
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                print(f"[WARN] torch.use_deterministic_algorithms unavailable or failed: {repr(e)}", flush=True)

            print("[DETERMINISM] CUDA deterministic mode enabled; cuDNN disabled.", flush=True)

    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 设置文件以输出 TensorBoard、超参数和已保存的模型
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))


    # init wandb
    if all_args.use_wandb:
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

    all_args.resolved_run_id = str(run_dir.name)
    all_args.resolved_run_dir = str(run_dir)
    persist_source_manifest(run_dir, all_args)
    print(
        "[RUN] resolved_run_id={}; resolved_run_dir={}".format(
            all_args.resolved_run_id,
            all_args.resolved_run_dir,
        ),
        flush=True,
    )

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # 创建 env / eval_env
    env = make_train_env(all_args)
    eval_env = make_eval_env(all_args) if all_args.use_eval else env

    N = all_args.max_player_num #（固定槽位数）
    K = int(getattr(all_args, "highest_orders", 2))
    sparsity = float(getattr(all_args, "sparsity", 0.5))

    #静态图+ 特定 DDFG/DFG 类算法
    if all_args.use_dyn_graph == False and all_args.equal_vdn == False and all_args.algorithm_name in ["rmdfg",
                                                                                                       "rmdfg_cent",
                                                                                                       "ddfg",
                                                                                                       "rmfg_cent",
                                                                                                       "sddfg"]:
        all_args.num_factor = N * (N - 1) // 2 + N # 两两配对 + N
    else:#动态图
        if all_args.num_factor is None:
            num_factor_full = math.factorial(N) // (math.factorial(K) * math.factorial(N-K)) #C(n,k)

            # 根据稀疏度计算
            calculated_num_factor = int(num_factor_full * sparsity)

            # [保底机制] 确保至少为 1，防止除 0 错误
            all_args.num_factor = max(1, calculated_num_factor)

            print(f"Auto-calculated num_factor: {all_args.num_factor} (Sparsity: {sparsity})")
        else:
            # 如果用户指定了，就直接使用用户的值，不做任何修改
            print(f"Using specified num_factor: {all_args.num_factor}")


    # create policies and mapping fn
    def get_unit_dim(cent_obs_dim, local_obs_dim):
        """QPLEX mixer 需要可整除的 state unit；其他算法不使用该约束。"""
        if all_args.algorithm_name != "qplex":
            return local_obs_dim
        if all_args.env_name == "wolfpack":
            # Wolfpack state starts with seven features per player slot:
            # (x, y, orientation one-hot, active flag). Food and remaining-time
            # features are a global suffix and must not be split into QPLEX's
            # per-agent attention keys.
            wolfpack_agent_state_dim = 7
            required_agent_prefix = wolfpack_agent_state_dim * all_args.max_player_num
            if cent_obs_dim < required_agent_prefix:
                raise ValueError(
                    "Wolfpack QPLEX state is shorter than its agent-state prefix: "
                    f"cent_obs_dim={cent_obs_dim}, required={required_agent_prefix}"
                )
            return wolfpack_agent_state_dim
        if cent_obs_dim % all_args.max_player_num != 0:
            raise ValueError(
                "QPLEX requires cent_obs_dim divisible by max_player_num, "
                f"got cent_obs_dim={cent_obs_dim}, "
                f"max_player_num={all_args.max_player_num}"
            )
        return cent_obs_dim // all_args.max_player_num

    if all_args.share_policy:
        cent_obs_dim = get_dim_from_space(env.share_observation_space[0])
        obs_dim = get_dim_from_space(env.observation_space[0])
        unit_dim = get_unit_dim(cent_obs_dim, obs_dim)

        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0],
                         "unit_dim": unit_dim,}
        }

        def policy_mapping_fn(id):
            return 'policy_0'
    else:
        policy_info = {}

        for agent_id in range(N):
            cent_obs_dim = get_dim_from_space(env.share_observation_space[agent_id])
            obs_dim = get_dim_from_space(env.observation_space[agent_id])
            unit_dim = get_unit_dim(cent_obs_dim, obs_dim)

            policy_info['policy_' + str(agent_id)] = {
                "cent_obs_dim": cent_obs_dim,
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[agent_id],
                "share_obs_space": env.share_observation_space[agent_id],
                "act_space": env.action_space[agent_id],
                "unit_dim": unit_dim,
            }

        def policy_mapping_fn(agent_id):
            return 'policy_' + str(agent_id)

    #初始化“静态邻接矩阵”adj（形状: [max_player_num, num_factor]）。
    #仅在 use_dyn_graph=False 且算法需要固定因子图时使用；否则保持全 0。
    adj = torch.zeros((N, all_args.num_factor), dtype=torch.int64)
    index = 0
    n = 0
    if all_args.use_dyn_graph == False and all_args.equal_vdn == False and all_args.algorithm_name in ["ddfg",
                                                                                                       "rddfg_low",
                                                                                                       "rmfg_cent","sddfg"]:
        for i in range(N - 1):
            for j in range(i + 1, N):
                adj[i, index] = 1
                adj[j, index] = 1
                index = index + 1
        for i in range(index, all_args.num_factor):
            adj[n, i] = 1
            n = n + 1

    # Runner config（与其他 train 结构保持一致）
    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": all_args.max_player_num,
              "device": device,
              "run_dir": run_dir,
              "use_same_share_obs": all_args.use_same_share_obs,
              "use_available_actions": all_args.use_available_actions,
              "adj": adj}
    runner = Runner(config=config)

    progress_filename = get_run_csv_path(run_dir, 'config.csv')
    df = pd.DataFrame(list(all_args.__dict__.items()), columns=['Name', 'Value'])
    df.to_csv(progress_filename, index=False)

    total_num_steps = 0

    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    # Periodic saving can leave the exported model tens of thousands of
    # environment steps behind the policy used by the final evaluation. Save
    # the exact terminal parameters and all adjacency Adam states after the
    # last training episode.
    if all_args.use_save:
        runner.save_terminal_checkpoint()
        runner.last_save_T = runner.total_env_steps

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.close()
        write_terminal_summary(runner, all_args, total_num_steps)

if __name__ == "__main__":
    main(sys.argv[1:])
