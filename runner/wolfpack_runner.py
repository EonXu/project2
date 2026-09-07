import numpy as np
import torch
import time
from typing import Any, Dict, List, Optional
from runner.base_runner import RecRunner
from utils.graph_sampling import resolve_graph_sampling_mode
from utils.pair_credit import (
    build_capture_identity_factor_weights,
    canonical_capture_factor_catalog,
)
from utils.wolfpack_reward import terminal_win_diagnostic_fields
import os
import pandas as pd


def _is_post_capture_pursuit_active(food_alive_statuses):
    statuses = tuple(bool(value) for value in food_alive_statuses)
    if not statuses:
        raise RuntimeError("post-capture exploration requires prey status")
    return any(statuses) and not all(statuses)


def _get_run_csv_path(path: str) -> str:
    """
    将写入 run 目录下的 CSV 路径改成带 run 前缀的文件名。

    例如：
      ".../run1/progress_train_num_players_traj.csv"
      ->
      ".../run1/run1_progress_train_num_players_traj.csv"
    """
    path = str(path)
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path)

    if not file_name.endswith(".csv"):
        return path

    run_name = os.path.basename(dir_name.rstrip(os.sep))

    # 防御：只对 run1/run2/run3 这类目录生效
    if not run_name.startswith("run"):
        return path

    prefix = run_name + "_"
    if file_name.startswith(prefix):
        return path

    return os.path.join(dir_name, prefix + file_name)

def _append_df_csv(path: str, df: pd.DataFrame):
    """
    追加写入 CSV（如果文件不存在则写入表头；存在则不写表头，直接 append）。
    文件名会自动加 run 前缀，例如 run1_progress_eval_num_players_traj.csv。
    """
    path = _get_run_csv_path(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)


class WolfpackRunner(RecRunner):
    _BATCHED_FACTOR_GRAPH_ALGOS = {"sddfg", "ddfg", "ddfg_low"}

    @staticmethod
    def _prepare_episode_metrics_for_logging(metric_lists):
        """Prepare conditional episode metrics for scalar aggregation.

        ``collecter`` uses ``-1`` when an episode never reaches global
        success. Mixing that sentinel with real steps biases
        ``first_success_step`` downward. Average successful episodes only, and
        retain ``-1`` solely when the complete logging window has no success.
        """
        prepared = dict(metric_lists)
        successful_steps = []
        for value in metric_lists.get("first_success_step", []):
            try:
                step = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(step) and step >= 0.0:
                successful_steps.append(step)

        prepared["first_success_step"] = (
            successful_steps if successful_steps else [-1.0]
        )
        prepared["first_success_step_sample_count"] = [
            float(len(successful_steps))
        ]
        return prepared

    @staticmethod
    def _extract_first_info_dict(infos, env_i=0):
        """
        Robustly extract a representative info dict from env wrappers.

        Common shapes:
        - ShareDummyVecEnv: infos is np.ndarray(dtype=object) with shape (num_envs, max_player_num)
        - Some envs: infos is list[dict] (per-agent)
        - Some wrappers: infos is list[list[dict]] (per-env, per-agent)
        - Rare: infos is dict
        """
        import numpy as np

        if isinstance(infos, dict):
            return infos

        # try per-env entry
        try:
            per_env = infos[env_i]
        except Exception:
            per_env = infos

        if isinstance(per_env, dict):
            return per_env

        # ShareDummyVecEnv: np.ndarray of dicts
        if isinstance(per_env, np.ndarray):
            try:
                if per_env.size > 0 and isinstance(per_env.flat[0], dict):
                    return per_env.flat[0]
            except Exception:
                return {}

        # list/tuple of dicts
        if isinstance(per_env, (list, tuple)):
            try:
                if len(per_env) > 0 and isinstance(per_env[0], dict):
                    return per_env[0]
            except Exception:
                return {}

        return {}

    @staticmethod
    def _infer_active_masks_from_obs(obs: np.ndarray) -> np.ndarray:
        """
        通过 obs 中“全 -1 向量”判断空槽位。
        返回 shape: (num_envs, num_agents, 1)，float32，1=存在，0=不存在
        """
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 2:
            obs_arr = obs_arr[None, ...]  # (1, num_agents, obs_dim)
        is_pad = np.all(obs_arr == -1, axis=-1, keepdims=True)
        return (~is_pad).astype(np.float32)

    @staticmethod
    def _visible_prey_geometry_from_local_vector_obs(
            obs_batch,
            num_agents: int,
            max_food_num: int):
        """Recover only information already present in each actor observation.

        Wolfpack's vector observation stores each visible prey orientation as
        a one-hot vector; an unseen or inactive prey has four zeros.  This
        parser deliberately does not read ``info`` or central state, so the
        exploration guard cannot gain oracle prey positions.
        """
        if torch.is_tensor(obs_batch):
            obs = obs_batch.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        else:
            obs = np.asarray(obs_batch, dtype=np.float32)
        num_agents = int(num_agents)
        max_food_num = int(max_food_num)
        if obs.ndim == 3:
            if obs.shape[0] != 1:
                raise RuntimeError(
                    "visible-prey quorum guard requires one environment"
                )
            obs = obs[0]
        if obs.ndim != 2 or obs.shape[0] != num_agents:
            raise RuntimeError(
                "visible-prey quorum guard received an invalid observation "
                "batch"
            )
        if max_food_num <= 0:
            raise RuntimeError(
                "visible-prey quorum guard requires at least one prey slot"
            )
        expected_dim = (
            2 + 4 + 2 * (num_agents - 1) + 7 * max_food_num + 4 + 1
        )
        if obs.shape[1] != expected_dim:
            raise RuntimeError(
                "visible-prey quorum guard observation contract mismatch: "
                "got {}, expected {}".format(obs.shape[1], expected_dim)
            )
        if not np.isfinite(obs).all():
            raise FloatingPointError(
                "visible-prey quorum guard received non-finite observation"
            )

        alive_slots = ~np.all(obs == -1.0, axis=1)
        prey_start = 2 + 4 + 2 * (num_agents - 1)
        visible = np.zeros((num_agents, max_food_num), dtype=np.int64)
        offsets = np.full(
            (num_agents, max_food_num, 2),
            np.nan,
            dtype=np.float32,
        )
        for food_id in range(max_food_num):
            block_start = prey_start + 7 * food_id
            relative_position = obs[:, block_start:block_start + 2]
            orientation = obs[:, block_start + 2:block_start + 6]
            alive_orientation = orientation[alive_slots]
            if alive_orientation.size and not np.all(
                    (alive_orientation == 0.0)
                    | (alive_orientation == 1.0)):
                raise RuntimeError(
                    "visible-prey quorum guard found a non-binary prey "
                    "orientation"
                )
            orientation_sum = orientation.sum(axis=1)
            if np.any(
                    alive_slots
                    & (orientation_sum != 0.0)
                    & (orientation_sum != 1.0)):
                raise RuntimeError(
                    "visible-prey quorum guard found an invalid prey one-hot"
                )
            visible[:, food_id] = (
                alive_slots & (orientation_sum == 1.0)
            ).astype(np.int64)
            visible_rows = visible[:, food_id].astype(bool)
            offsets[visible_rows, food_id] = relative_position[visible_rows]
        return visible, offsets

    @staticmethod
    def _visible_prey_mask_from_local_vector_obs(
            obs_batch,
            num_agents: int,
            max_food_num: int) -> np.ndarray:
        visible, _ = (
            WolfpackRunner._visible_prey_geometry_from_local_vector_obs(
                obs_batch,
                num_agents=num_agents,
                max_food_num=max_food_num,
            )
        )
        return visible

    @staticmethod
    def _extract_active_masks_batch(infos, num_envs: int, num_agents: int,
                                    fallback_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        优先从 infos 取 active_masks；若没有则 fallback 用 obs 推断。
        """
        active_masks = np.ones((num_envs, num_agents, 1), dtype=np.float32)
        for e in range(num_envs):
            info_dict = WolfpackRunner._extract_first_info_dict(infos, env_i=e)
            am = info_dict.get("active_masks", None)
            if am is not None:
                active_masks[e] = np.asarray(am, dtype=np.float32).reshape(num_agents, 1)
            elif fallback_obs is not None:
                active_masks[e] = WolfpackRunner._infer_active_masks_from_obs(fallback_obs[e])
        return active_masks

    @staticmethod
    def _mask_rnn_states_by_dones(rnn_states, dones):
        """
        将 done/pad 的 slot hidden 强制置 0，避免 hidden 漂移污染后续。

        同时兼容：
        1) SDDFG/DDFG policy 返回的 torch.Tensor hidden；
        2) VDN/QMIX/QPLEX 等原始 baseline 返回的 numpy.ndarray hidden。
        """
        if rnn_states is None:
            return rnn_states

        dones_arr = np.asarray(dones)
        if dones_arr.ndim == 2:
            dones_arr = dones_arr[None, ...]

        alive = (1.0 - dones_arr.astype(np.float32)).reshape(-1, 1)

        if isinstance(rnn_states, torch.Tensor):
            alive_t = torch.as_tensor(
                alive,
                device=rnn_states.device,
                dtype=rnn_states.dtype
            )
            return rnn_states * alive_t

        # VDN / QMIX / QPLEX baseline 的 hidden 可能是 numpy.ndarray
        if isinstance(rnn_states, np.ndarray):
            alive_np = alive.astype(rnn_states.dtype, copy=False)
            return rnn_states * alive_np

        raise TypeError(
            f"Unsupported rnn_states type in _mask_rnn_states_by_dones: {type(rnn_states)}"
        )

    @staticmethod
    def _calc_adj_metrics(adj, dones, num_agents: int,
                          prob_adj=None, prev_adj=None) -> Dict[str, float]:
        """
        统计当前邻接图结构质量。兼容 DDFG 原邻接生成器与 SDDFG/GAT 邻接生成器。

        adj:   [B, N, F] 或 [N, F]
        dones: [B, N, 1] 或 [N, 1]，True=死亡/空槽
        """
        if adj is None:
            return {}

        try:
            adj_t = adj.detach().float() if torch.is_tensor(adj) else torch.as_tensor(adj).float()
            if adj_t.dim() == 2:
                adj_t = adj_t.unsqueeze(0)

            B, N, F = adj_t.shape

            dones_arr = np.asarray(dones)
            if dones_arr.ndim == 2:
                dones_arr = dones_arr[None, ...]
            dones_arr = dones_arr.reshape(B, num_agents, -1)[..., 0]

            alive = torch.as_tensor(
                (~dones_arr.astype(bool)).astype(np.float32),
                device=adj_t.device,
                dtype=torch.float32
            ).unsqueeze(-1)  # [B, N, 1]

            factor_order = adj_t.sum(dim=1)              # [B, F]
            alive_count = (adj_t * alive).sum(dim=1)     # [B, F]
            valid_factor = (factor_order > 0) & (alive_count == factor_order)

            total = float(max(B * F, 1))
            valid_num = float(valid_factor.float().sum().item())
            valid_adj = adj_t * valid_factor.unsqueeze(1).to(adj_t.dtype)
            dynamic_degree = valid_adj.sum(dim=2)  # [B, N], excludes appended self-factors
            alive_agents = alive.squeeze(-1) > 0.5
            active_agent_total = float(alive_agents.float().sum().item())
            covered_agents = alive_agents & (dynamic_degree > 0.0)
            covered_agent_num = float(covered_agents.float().sum().item())

            # Coverage does not imply global communication: factors can cover
            # every agent while forming disconnected components. Compute the
            # transitive closure of factor co-membership over active agents.
            co_membership = torch.bmm(
                valid_adj,
                valid_adj.transpose(1, 2),
            ) > 0.0
            active_pairs = (
                alive_agents.unsqueeze(2)
                & alive_agents.unsqueeze(1)
            )
            reachability = co_membership & active_pairs
            active_eye = (
                torch.eye(N, dtype=torch.bool, device=adj_t.device)
                .unsqueeze(0)
                & active_pairs
            )
            reachability = reachability | active_eye
            for _ in range(N):
                reachability = reachability | (
                    torch.bmm(
                        reachability.float(),
                        reachability.float(),
                    ) > 0.0
                )
            connected_by_batch = (
                reachability | (~active_pairs)
            ).reshape(B, -1).all(dim=1)
            connected_graph_ratio = float(
                connected_by_batch.float().mean().item()
            )
            selected_prob_mean = 0.0
            selected_prob_min = 0.0
            if prob_adj is not None and valid_num > 0:
                prob_t = (
                    prob_adj.detach().float()
                    if torch.is_tensor(prob_adj)
                    else torch.as_tensor(prob_adj).float()
                )
                if prob_t.dim() == 2:
                    prob_t = prob_t.unsqueeze(0)
                prob_t = prob_t.to(adj_t.device)
                factor_prob = (
                    (prob_t * valid_adj).sum(dim=1)
                    / factor_order.clamp_min(1.0)
                )
                valid_factor_prob = factor_prob[valid_factor]
                selected_prob_mean = float(valid_factor_prob.mean().item())
                selected_prob_min = float(valid_factor_prob.min().item())

            factor_retention_ratio = 1.0
            if prev_adj is not None and valid_num > 0:
                prev_t = (
                    prev_adj.detach().float()
                    if torch.is_tensor(prev_adj)
                    else torch.as_tensor(prev_adj).float()
                )
                if prev_t.dim() == 2:
                    prev_t = prev_t.unsqueeze(0)
                prev_t = prev_t.to(adj_t.device)
                prev_factor_order = prev_t.sum(dim=1)
                exact_match = (
                    adj_t.transpose(1, 2).unsqueeze(2)
                    == prev_t.transpose(1, 2).unsqueeze(1)
                ).all(dim=-1)
                retained = (
                    exact_match
                    & valid_factor.unsqueeze(2)
                    & (prev_factor_order > 0).unsqueeze(1)
                ).any(dim=2)
                factor_retention_ratio = float(
                    retained.float().sum().item() / max(valid_num, 1.0)
                )

            return {
                "adj_valid_factor_ratio": valid_num / total,
                "adj_empty_factor_ratio": float((factor_order == 0).float().mean().item()),
                "adj_order1_ratio": float(((factor_order == 1) & valid_factor).float().sum().item() / total),
                "adj_order2_ratio": float(((factor_order == 2) & valid_factor).float().sum().item() / total),
                "adj_order3_ratio": float(((factor_order == 3) & valid_factor).float().sum().item() / total),
                "adj_invalid_factor_ratio": float(((factor_order > 0) & (~valid_factor)).float().sum().item() / total),
                "adj_mean_order": float(factor_order[valid_factor].mean().item()) if valid_num > 0 else 0.0,
                # Dynamic graph coverage is critical for intra-episode joins:
                # a live node with zero dynamic degree only receives its
                # self-factor and cannot exchange cooperative messages.
                "adj_active_agent_coverage": covered_agent_num / max(active_agent_total, 1.0),
                "adj_uncovered_active_agent_ratio": (
                    active_agent_total - covered_agent_num
                ) / max(active_agent_total, 1.0),
                "adj_active_agent_degree": (
                    float(dynamic_degree[alive_agents].mean().item())
                    if active_agent_total > 0.0 else 0.0
                ),
                "adj_connected_graph_ratio": connected_graph_ratio,
                "adj_selected_prob_mean": selected_prob_mean,
                "adj_selected_prob_min": selected_prob_min,
                "adj_factor_retention_ratio": factor_retention_ratio,
            }
        except Exception:
            return {}

    @staticmethod
    def _mask_adj_by_dones(adj: torch.Tensor,
                           prob_adj: torch.Tensor,
                           dones,
                           num_agents: int,
                           num_factor: int):
        """
        DDFG/SDDFG 通用 factor-level 邻接 mask。

        目标：
        1. 死亡/空槽 agent 的行清零；
        2. 只要某个 factor 中包含死亡/空槽节点，整条 factor 清零；
        3. 防止 {alive, dead} 被行级 mask 后退化成伪一阶 factor；
        4. 同时支持 DDFG 原 adj_generator_new 与 SDDFG/GAT adj_generator。

        adj:      [B, N, F] 或 [N, F]
        prob_adj: [B, N, F] 或 [N, F]，可为 None
        dones:    [B, N, 1] / [N, 1]，True=死亡/空槽
        """
        if adj is None:
            return prob_adj, adj

        if not torch.is_tensor(adj):
            adj = torch.as_tensor(adj)

        device = adj.device

        squeeze_batch = False
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
            squeeze_batch = True

        if prob_adj is not None:
            if not torch.is_tensor(prob_adj):
                prob_adj = torch.as_tensor(prob_adj, device=device)
            else:
                prob_adj = prob_adj.to(device)
            if prob_adj.dim() == 2:
                prob_adj = prob_adj.unsqueeze(0)

        B = adj.shape[0]

        dones_arr = np.asarray(dones)
        if dones_arr.ndim == 2:
            dones_arr = dones_arr[None, ...]
        dones_arr = dones_arr.reshape(B, num_agents, -1)[..., 0]

        alive = torch.as_tensor(
            (~dones_arr.astype(bool)).astype(np.float32),
            device=device,
            dtype=torch.float32
        ).unsqueeze(-1)  # [B, N, 1]

        adj_float = adj.float()

        factor_size = adj_float.sum(dim=1, keepdim=True)  # [B, 1, F]
        alive_count = (adj_float * alive).sum(dim=1, keepdim=True)

        factor_valid = ((factor_size > 0.0) & (alive_count == factor_size)).float()

        adj_masked = (adj_float * alive * factor_valid).long()

        if prob_adj is not None:
            prob_adj = prob_adj.float() * alive * factor_valid
            # 未选中/非法位置保持极小概率，避免后续 log(0)
            prob_adj = torch.where(
                adj_masked > 0,
                prob_adj.clamp_min(1e-8),
                torch.zeros_like(prob_adj)
            )

        if squeeze_batch:
            adj_masked = adj_masked.squeeze(0)
            if prob_adj is not None:
                prob_adj = prob_adj.squeeze(0)

        return prob_adj, adj_masked

    @staticmethod
    def _build_alive_eye(num_agents: int, dones, device):
        """
        构造存活 agent 的 identity self-factor。
        输出 [B, N, N]。
        """
        dones_arr = np.asarray(dones)
        if dones_arr.ndim == 2:
            dones_arr = dones_arr[None, ...]

        B = dones_arr.shape[0]
        alive = torch.as_tensor(
            (~dones_arr.reshape(B, num_agents, -1)[..., 0].astype(bool)).astype(np.int64),
            device=device,
            dtype=torch.int64
        ).unsqueeze(-1)  # [B, N, 1]

        eye = torch.eye(num_agents, dtype=torch.int64, device=device).unsqueeze(0).repeat(B, 1, 1)
        eye = eye * alive
        return eye

    def __init__(self, config):
        super(WolfpackRunner, self).__init__(config)

        self.post_capture_joint_greedy_floor = float(getattr(
            self.args,
            "post_capture_joint_greedy_floor",
            0.0,
        ))
        if not np.isfinite(self.post_capture_joint_greedy_floor) or not (
                0.0 <= self.post_capture_joint_greedy_floor < 1.0):
            raise ValueError(
                "post_capture_joint_greedy_floor must be finite in [0, 1)"
            )
        self.post_capture_explore_max_random_agents = int(getattr(
            self.args,
            "post_capture_explore_max_random_agents",
            0,
        ))
        if self.post_capture_explore_max_random_agents < 0:
            raise ValueError(
                "post_capture_explore_max_random_agents must be non-negative"
            )
        if (
                self.post_capture_explore_max_random_agents > 0
                and self.post_capture_joint_greedy_floor <= 0.0):
            raise RuntimeError(
                "bounded post-capture exploration requires a nonzero "
                "post-capture joint greedy floor"
            )
        if (
                self.post_capture_joint_greedy_floor > 0.0
                and not bool(getattr(
                    self.args,
                    "use_joint_epsilon_exploration",
                    False,
                ))):
            raise RuntimeError(
                "post-capture greedy floor requires joint epsilon exploration"
            )
        if self.post_capture_joint_greedy_floor > 0.0 and self.num_envs != 1:
            raise RuntimeError(
                "post-capture joint greedy floor currently requires one "
                "environment decision per policy call"
            )
        self.pre_capture_visible_prey_quorum_guard = bool(getattr(
            self.args,
            "pre_capture_visible_prey_quorum_guard",
            False,
        ))
        if (
                self.pre_capture_visible_prey_quorum_guard
                and self.algorithm_name != "sddfg"):
            raise RuntimeError(
                "pre-capture visible-prey quorum guard is SDDFG-only"
            )
        if (
                self.pre_capture_visible_prey_quorum_guard
                and not bool(getattr(
                    self.args,
                    "use_joint_epsilon_exploration",
                    False,
                ))):
            raise RuntimeError(
                "pre-capture visible-prey quorum guard requires joint "
                "epsilon exploration"
            )
        if self.pre_capture_visible_prey_quorum_guard and self.num_envs != 1:
            raise RuntimeError(
                "pre-capture visible-prey quorum guard currently requires "
                "one environment decision per policy call"
            )
        self.pre_capture_visible_prey_quorum_greedy_frontier_guard = bool(
            getattr(
                self.args,
                "pre_capture_visible_prey_quorum_greedy_frontier_guard",
                False,
            )
        )
        if (
                self.pre_capture_visible_prey_quorum_greedy_frontier_guard
                and self.algorithm_name != "sddfg"):
            raise RuntimeError(
                "pre-capture greedy frontier guard is SDDFG-only"
            )
        if (
                self.pre_capture_visible_prey_quorum_greedy_frontier_guard
                and self.num_envs != 1):
            raise RuntimeError(
                "pre-capture greedy frontier guard currently requires one "
                "environment decision per policy call"
            )

        # 预热：随机策略采样若干回合填充经验池
        # 回放/可视化时可通过 args.skip_warmup=True 跳过预热，加快启动速度
        skip_warmup = bool(getattr(self.args, "skip_warmup", False))
        self.start = time.time()
        if not skip_warmup:
            num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
            self.warmup(num_warmup_episodes)
        warmup_infos = tuple(getattr(self, "warmup_env_infos", ()))
        self.warmup_terminal_win_first_transition_count = int(round(sum(
            float(info.get("terminal_win_first_transition_count", 0.0))
            for info in warmup_infos
        )))
        self.warmup_terminal_win_bonus_event_count = int(round(sum(
            float(info.get("terminal_win_bonus_event_count", 0.0))
            for info in warmup_infos
        )))
        self.warmup_terminal_win_bonus_sum = float(sum(
            float(info.get("terminal_win_bonus_sum", 0.0))
            for info in warmup_infos
        ))
        if (
                float(self.args.reward_win) > 0.0
                and self.warmup_terminal_win_bonus_event_count
                != self.warmup_terminal_win_first_transition_count):
            raise RuntimeError(
                "Wolfpack warmup terminal win bonus count did not match "
                "first-success transition count"
            )
        print(
            "warmup terminal-win diagnostics: episodes={}, first_transitions={}, "
            "bonus_events={}, bonus_sum={:.9g}".format(
                len(warmup_infos),
                self.warmup_terminal_win_first_transition_count,
                self.warmup_terminal_win_bonus_event_count,
                self.warmup_terminal_win_bonus_sum,
            ),
            flush=True,
        )
        end = time.time()

        denom = max((end - self.start), 1e-6)
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n"
              .format(self.env_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / denom)))

        self.log_clear()
        # 防御式初始化：避免在第一次训练更新发生前触发 log() 时崩溃
        self.train_infos = [dict() for _ in getattr(self, "policy_ids", [])]

    def _get_run_dir(self) -> str:
        """获取日志目录（优先 run_dir，其次 log_dir，否则当前目录）。"""
        rd = getattr(self, "run_dir", None)
        if rd is None:
            rd = getattr(self, "log_dir", None)
        if rd is None:
            rd = "."
        return str(rd)

    def _dump_eval_episode_step_tables(self, eval_step: int, eval_ep: int, step_info: dict, eval_id: int = 0):
        """
        将评估阶段步骤轨迹添加到 progress_eval_num_players_traj.csv , progress_eval_individual_rewards.csv
        """
        run_dir = self._get_run_dir()

        # ---------- t -> num_players ----------
        num_players = np.asarray(step_info.get("num_players", []), dtype=np.float32).reshape(-1)
        T = int(num_players.shape[0])
        t_idx = np.arange(T, dtype=np.int32)

        if T > 0:
            data = {
                "eval_id": int(eval_id),
                "step": int(eval_step),
                "eval_ep": int(eval_ep),
                "t": t_idx,
                "num_players": num_players.astype(np.int32),
            }
            team_rewards = np.asarray(step_info.get("team_rewards", []), dtype=np.float32).reshape(-1)
            if team_rewards.size == T:
                data["team_reward"] = team_rewards
            df_np_long = pd.DataFrame(data)
            _append_df_csv(os.path.join(run_dir, "progress_eval_num_players_traj.csv"), df_np_long)

        # ---------- t -> 显式拓扑事件 ----------
        # 不再只看 num_players 差分；同一步退出和加入时人数可能不变。
        event_rows = []
        for event in step_info.get("topology_events", []):
            row = {
                "eval_id": int(eval_id),
                "step": int(eval_step),
                "eval_ep": int(eval_ep),
                "episode_step": int(event.get("episode_step", 0)),
                "topology_changed": int(bool(event.get("topology_changed", False))),
                "left_count": int(event.get("left_count", 0)),
                "joined_count": int(event.get("joined_count", 0)),
                "recovered_count": int(event.get("recovered_count", 0)),
                "pending_recovery_count": int(event.get("pending_recovery_count", 0)),
                "capture_count": int(event.get("capture_count", 0)),
                "capture_event_ids": ";".join(
                    str(item.get("event_id"))
                    for item in event.get("capture_events", [])
                ),
                "capture_target_ids": ";".join(
                    str(item.get("target_id"))
                    for item in event.get("capture_events", [])
                ),
                "capture_participant_slots": ";".join(
                    "|".join(map(str, item.get("participant_slots", [])))
                    for item in event.get("capture_events", [])
                ),
                "capture_identity_matched_event_count": int(
                    event.get("capture_identity_matched_event_count", 0)
                ),
                "capture_identity_unmatched_event_count": int(
                    event.get("capture_identity_unmatched_event_count", 0)
                ),
                "capture_identity_candidate_factor_count": int(
                    event.get("capture_identity_candidate_factor_count", 0)
                ),
                "capture_identity_raw_candidate_factor_count": int(
                    event.get("capture_identity_raw_candidate_factor_count", 0)
                ),
                "capture_identity_duplicate_factor_count": int(
                    event.get("capture_identity_duplicate_factor_count", 0)
                ),
                "capture_identity_candidate_only_event_count": int(
                    event.get("capture_identity_candidate_only_event_count", 0)
                ),
                "capture_identity_participant_count": int(
                    event.get("capture_identity_participant_count", 0)
                ),
                "capture_identity_matched_factor_order_sum": int(
                    event.get("capture_identity_matched_factor_order_sum", 0)
                ),
                "capture_identity_matched_event_weight_sum": float(
                    event.get("capture_identity_matched_event_weight_sum", 0.0)
                ),
                "capture_identity_event_mass_error": float(
                    event.get("capture_identity_event_mass_error", 0.0)
                ),
                "success_now": int(bool(event.get("success_now", False))),
                **terminal_win_diagnostic_fields(event),
                "left_slots": ",".join(map(str, event.get("left_slots", []))),
                "joined_slots": ",".join(map(str, event.get("joined_slots", []))),
                "recovered_slots": ",".join(map(str, event.get("recovered_slots", []))),
                "slot_uids": ",".join(map(str, event.get("slot_uids", []))),
            }
            event_rows.append(row)
        if event_rows:
            _append_df_csv(
                os.path.join(run_dir, "progress_eval_topology_events.csv"),
                pd.DataFrame(event_rows),
            )

        # ---------- t -> individual_rewards (WIDE FORMAT) ----------
        ir_list = step_info.get("individual_rewards", None)
        if ir_list is None:
            return

        rows_long = []
        max_slots_seen = 0

        for t in range(len(ir_list)):
            arr = np.asarray(ir_list[t])
            # expected: (max_player_num, 1) or (max_player_num,)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            arr = arr.reshape(-1)  # (num_slots,)

            num_slots = int(arr.shape[0])
            if num_slots > max_slots_seen:
                max_slots_seen = num_slots

            row = {
                "eval_id": int(eval_id),
                "step": int(eval_step),
                "eval_ep": int(eval_ep),
                "t": int(t),
            }
            # 把每个 slot 的 reward 依次摊平为列："0","1","2",...
            for s in range(num_slots):
                row[str(s)] = float(arr[s])

            rows_long.append(row)

        if len(rows_long) > 0:
            df_ir_wide = pd.DataFrame(rows_long)

            # 确保列顺序固定：eval_id, step, eval_ep, t, 0,1,2,...,max_slots_seen-1
            reward_cols = [str(i) for i in range(max_slots_seen)]
            df_ir_wide = df_ir_wide.reindex(columns=["eval_id", "step", "eval_ep", "t"] + reward_cols)

            _append_df_csv(os.path.join(run_dir, "progress_eval_individual_rewards.csv"), df_ir_wide)

    def _dump_train_episode_step_tables(self, train_step: int, step_info: dict):
        """
        将训练阶段的逐步动态信息写入 CSV。
        记录频率由 collect_rollout 里基于 log_interval 控制，避免每个 episode 都写导致文件过大。
        """
        run_dir = self._get_run_dir()

        # ---------- t -> num_players ----------
        num_players = np.asarray(step_info.get("num_players", []), dtype=np.float32).reshape(-1)
        if num_players.size > 0:
            data = {
                "step": int(train_step),
                "t": np.arange(num_players.size, dtype=np.int32),
                "num_players": num_players.astype(np.int32),
            }
            team_rewards = np.asarray(step_info.get("team_rewards", []), dtype=np.float32).reshape(-1)
            if team_rewards.size == num_players.size:
                data["team_reward"] = team_rewards
            df_np = pd.DataFrame(data)
            _append_df_csv(os.path.join(run_dir, "progress_train_num_players_traj.csv"), df_np)

        # ---------- t -> 显式拓扑事件 ----------
        event_rows = []
        for event in step_info.get("topology_events", []):
            event_rows.append({
                "step": int(train_step),
                "episode_step": int(event.get("episode_step", 0)),
                "topology_changed": int(bool(event.get("topology_changed", False))),
                "left_count": int(event.get("left_count", 0)),
                "joined_count": int(event.get("joined_count", 0)),
                "recovered_count": int(event.get("recovered_count", 0)),
                "pending_recovery_count": int(event.get("pending_recovery_count", 0)),
                "capture_count": int(event.get("capture_count", 0)),
                "capture_event_ids": ";".join(
                    str(item.get("event_id"))
                    for item in event.get("capture_events", [])
                ),
                "capture_target_ids": ";".join(
                    str(item.get("target_id"))
                    for item in event.get("capture_events", [])
                ),
                "capture_participant_slots": ";".join(
                    "|".join(map(str, item.get("participant_slots", [])))
                    for item in event.get("capture_events", [])
                ),
                "capture_identity_matched_event_count": int(
                    event.get("capture_identity_matched_event_count", 0)
                ),
                "capture_identity_unmatched_event_count": int(
                    event.get("capture_identity_unmatched_event_count", 0)
                ),
                "capture_identity_candidate_factor_count": int(
                    event.get("capture_identity_candidate_factor_count", 0)
                ),
                "capture_identity_raw_candidate_factor_count": int(
                    event.get("capture_identity_raw_candidate_factor_count", 0)
                ),
                "capture_identity_duplicate_factor_count": int(
                    event.get("capture_identity_duplicate_factor_count", 0)
                ),
                "capture_identity_candidate_only_event_count": int(
                    event.get("capture_identity_candidate_only_event_count", 0)
                ),
                "capture_identity_participant_count": int(
                    event.get("capture_identity_participant_count", 0)
                ),
                "capture_identity_matched_factor_order_sum": int(
                    event.get("capture_identity_matched_factor_order_sum", 0)
                ),
                "capture_identity_matched_event_weight_sum": float(
                    event.get("capture_identity_matched_event_weight_sum", 0.0)
                ),
                "capture_identity_event_mass_error": float(
                    event.get("capture_identity_event_mass_error", 0.0)
                ),
                "success_now": int(bool(event.get("success_now", False))),
                **terminal_win_diagnostic_fields(event),
                "left_slots": ",".join(map(str, event.get("left_slots", []))),
                "joined_slots": ",".join(map(str, event.get("joined_slots", []))),
                "recovered_slots": ",".join(map(str, event.get("recovered_slots", []))),
                "slot_uids": ",".join(map(str, event.get("slot_uids", []))),
            })
        if event_rows:
            _append_df_csv(
                os.path.join(run_dir, "progress_train_topology_events.csv"),
                pd.DataFrame(event_rows),
            )

        # ---------- t -> active masks ----------
        active_masks = step_info.get("active_masks", [])
        rows_active = []
        for t_i, am in enumerate(active_masks):
            arr = np.asarray(am, dtype=np.float32).reshape(-1)
            row = {
                "step": int(train_step),
                "t": int(t_i),
                "active_ratio": float(arr.mean()) if arr.size > 0 else 0.0,
            }
            for slot_i, v in enumerate(arr):
                row[f"active_{slot_i}"] = float(v)
            rows_active.append(row)

        if len(rows_active) > 0:
            _append_df_csv(
                os.path.join(run_dir, "progress_train_active_masks_traj.csv"),
                pd.DataFrame(rows_active)
            )

        # ---------- t -> individual rewards ----------
        ir_list = step_info.get("individual_rewards", [])
        rows_reward = []
        max_slots_seen = 0

        for t_i, ir in enumerate(ir_list):
            arr = np.asarray(ir, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            arr = arr.reshape(-1)

            max_slots_seen = max(max_slots_seen, int(arr.shape[0]))

            row = {
                "step": int(train_step),
                "t": int(t_i),
            }
            for slot_i, v in enumerate(arr):
                row[str(slot_i)] = float(v)
            rows_reward.append(row)

        if len(rows_reward) > 0:
            df_ir = pd.DataFrame(rows_reward)
            reward_cols = [str(i) for i in range(max_slots_seen)]
            df_ir = df_ir.reindex(columns=["step", "t"] + reward_cols)
            _append_df_csv(os.path.join(run_dir, "progress_train_individual_rewards.csv"), df_ir)

        # ---------- t -> adj metrics ----------
        adj_metrics = step_info.get("adj_metrics", [])
        rows_adj = []
        for t_i, m in enumerate(adj_metrics):
            if not isinstance(m, dict):
                continue
            row = {"step": int(train_step), "t": int(t_i)}
            row.update({k: float(v) for k, v in m.items()})
            rows_adj.append(row)

        if len(rows_adj) > 0:
            _append_df_csv(
                os.path.join(run_dir, "progress_train_adj_metrics_traj.csv"),
                pd.DataFrame(rows_adj)
            )

    @staticmethod
    def _build_post_capture_row(
            environment_episode_id: int,
            training_env_step: int,
            episode_step: int,
            first_capture_step: int,
            first_capture_target_id: int,
            first_capture_participant_slots,
            step_info: Dict[str, Any],
            action_diagnostic: Dict[str, Any]):
        """Build one already-observed post-capture diagnostic row."""
        required_info_fields = (
            "food_alive_statuses",
            "food_freeze_remaining",
            "food_positions",
            "player_slot_positions",
        )
        missing = [
            key for key in required_info_fields if key not in step_info
        ]
        if missing:
            raise RuntimeError(
                "Wolfpack post-capture diagnostics are missing production "
                f"environment fields: {missing}"
            )

        food_positions = list(step_info["food_positions"])
        player_slot_positions = list(step_info["player_slot_positions"])
        all_player_positions = [
            position for position in player_slot_positions
            if position is not None
        ]
        participant_positions = [
            player_slot_positions[int(slot)]
            for slot in first_capture_participant_slots
            if (
                0 <= int(slot) < len(player_slot_positions)
                and player_slot_positions[int(slot)] is not None
            )
        ]

        def _min_distances(players):
            result = []
            for food_position in food_positions:
                if not players:
                    result.append(float("nan"))
                    continue
                result.append(float(min(
                    abs(int(player[0]) - int(food_position[0]))
                    + abs(int(player[1]) - int(food_position[1]))
                    for player in players
                )))
            return result

        def _nearest_player_slots():
            result = []
            for food_position in food_positions:
                distances = []
                for slot, player_position in enumerate(player_slot_positions):
                    if player_position is None:
                        continue
                    distance = (
                        abs(int(player_position[0]) - int(food_position[0]))
                        + abs(int(player_position[1]) - int(food_position[1]))
                    )
                    distances.append((int(slot), int(distance)))
                if not distances:
                    result.append("")
                    continue
                minimum = min(distance for _, distance in distances)
                result.append("|".join(
                    str(slot) for slot, distance in distances
                    if distance == minimum
                ))
            return result

        capture_events = list(step_info.get("capture_events", []))
        return {
            "environment_episode_id": int(environment_episode_id),
            "training_env_step": int(training_env_step),
            "episode_step": int(episode_step),
            "offset_from_first_capture": int(
                int(episode_step) - int(first_capture_step)
            ),
            "first_capture_target_id": int(first_capture_target_id),
            "first_capture_participant_slots": ";".join(
                str(int(slot)) for slot in first_capture_participant_slots
            ),
            "capture_target_ids": ";".join(
                str(int(event["target_id"])) for event in capture_events
            ),
            "success_now": int(bool(step_info.get("success_now", False))),
            "food_alive_statuses": ";".join(
                str(int(bool(value)))
                for value in step_info["food_alive_statuses"]
            ),
            "food_freeze_remaining": ";".join(
                format(float(value), ".8g")
                for value in step_info["food_freeze_remaining"]
            ),
            "min_alive_player_distance_to_food": ";".join(
                format(float(value), ".8g")
                for value in _min_distances(all_player_positions)
            ),
            "min_first_participant_distance_to_food": ";".join(
                format(float(value), ".8g")
                for value in _min_distances(participant_positions)
            ),
            "nearest_alive_player_slots_to_food": ";".join(
                _nearest_player_slots()
            ),
            "joint_explore": int(action_diagnostic.get("joint_explore", -1)),
            "epsilon": float(
                action_diagnostic.get("epsilon", float("nan"))
            ),
            "greedy_actions": ";".join(
                str(int(value))
                for value in action_diagnostic.get("greedy_actions", [])
            ),
            "selected_actions": ";".join(
                str(int(value))
                for value in action_diagnostic.get("selected_actions", [])
            ),
            "post_capture_explore_max_random_agents": int(
                action_diagnostic.get(
                    "post_capture_explore_max_random_agents", 0
                )
            ),
            "post_capture_explore_bounded_applied": int(
                action_diagnostic.get(
                    "post_capture_explore_bounded_applied", 0
                )
            ),
            "random_replacement_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "random_replacement_slots", []
                )
            ),
        }

    @staticmethod
    def _build_pre_capture_transition_row(
            environment_episode_id: int,
            training_env_step: int,
            episode_step: int,
            step_info: Dict[str, Any],
            action_diagnostic: Dict[str, Any],
            greedy_joint_q_value,
            greedy_message_action_margins,
            greedy_factor_q_values,
            adjacency,
            learned_factor_count: int):
        """Build one trajectory-neutral s_t action -> s_(t+1) snapshot."""
        required_info_fields = (
            "food_alive_statuses",
            "food_freeze_remaining",
            "food_positions",
            "player_slot_positions",
            "food_visible_player_slots",
        )
        missing = [
            key for key in required_info_fields if key not in step_info
        ]
        if missing:
            raise RuntimeError(
                "Wolfpack pre-capture diagnostics are missing production "
                f"environment fields: {missing}"
            )

        def _as_numpy(value):
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def _flat_float_text(value):
            if value is None:
                return ""
            return ";".join(
                format(float(item), ".9g")
                for item in _as_numpy(value).reshape(-1)
            )

        def _flat_int_text(value):
            if value is None:
                return ""
            return ";".join(
                str(int(item))
                for item in _as_numpy(value).reshape(-1)
            )

        food_positions = list(step_info["food_positions"])
        player_slot_positions = list(step_info["player_slot_positions"])
        min_distances = []
        nearest_slots = []
        for food_position in food_positions:
            slot_distances = [
                (
                    int(slot),
                    abs(int(position[0]) - int(food_position[0]))
                    + abs(int(position[1]) - int(food_position[1])),
                )
                for slot, position in enumerate(player_slot_positions)
                if position is not None
            ]
            if not slot_distances:
                min_distances.append(float("nan"))
                nearest_slots.append("")
                continue
            minimum = min(distance for _, distance in slot_distances)
            min_distances.append(float(minimum))
            nearest_slots.append("|".join(
                str(slot) for slot, distance in slot_distances
                if distance == minimum
            ))

        adjacency_np = _as_numpy(adjacency)
        if adjacency_np.ndim == 3:
            if int(adjacency_np.shape[0]) != 1:
                raise RuntimeError(
                    "pre-capture adjacency diagnostic requires one environment"
                )
            adjacency_np = adjacency_np[0]
        if adjacency_np.ndim != 2:
            raise RuntimeError(
                "pre-capture adjacency diagnostic expects [agent, factor]"
            )
        learned_factor_count = int(learned_factor_count)
        if not (0 <= learned_factor_count <= int(adjacency_np.shape[1])):
            raise RuntimeError("invalid learned factor count in diagnostic")

        value_diagnostic = action_diagnostic.get(
            "frontier_value_ranking_diagnostic", None
        )
        if not isinstance(value_diagnostic, dict):
            if bool(int(action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_enabled",
                    0))):
                raise RuntimeError(
                    "enabled pre-capture frontier action is missing value-"
                    "ranking diagnostics"
                )
            value_diagnostic = {
                "schema_version": 1,
                "sampled": 0,
                "slot_ids": [],
                "action_dim": 7,
                "factor_count": int(adjacency_np.shape[1]),
                "legal_action_mask": [],
                "frontier_action_mask": [],
                "message_action_utilities": [],
                "coordinate_joint_q_values": [],
                "coordinate_factor_q_values": [],
            }
        if int(value_diagnostic.get("schema_version", 0)) != 1:
            raise RuntimeError(
                "unexpected frontier value-ranking diagnostic schema"
            )
        value_sampled = int(value_diagnostic.get("sampled", -1))
        if value_sampled not in (0, 1):
            raise RuntimeError(
                "frontier value-ranking sampled flag is not binary"
            )
        value_slot_ids = np.asarray(
            value_diagnostic.get("slot_ids", ()), dtype=np.int64
        ).reshape(-1)
        value_action_dim = int(value_diagnostic.get("action_dim", 0))
        value_factor_count = int(value_diagnostic.get("factor_count", 0))
        if value_action_dim != 7:
            raise RuntimeError(
                "frontier value-ranking diagnostic requires seven actions"
            )
        if value_factor_count != int(adjacency_np.shape[1]):
            raise RuntimeError(
                "frontier value-ranking diagnostic factor width does not "
                "match the production graph"
            )
        if (
                np.unique(value_slot_ids).size != value_slot_ids.size
                or np.any(value_slot_ids < 0)
                or np.any(value_slot_ids >= int(adjacency_np.shape[0]))):
            raise RuntimeError(
                "frontier value-ranking diagnostic slot ids are invalid"
            )
        expected_action_shape = (
            int(value_slot_ids.size), value_action_dim
        )
        expected_factor_shape = expected_action_shape + (
            value_factor_count,
        )
        value_legal_mask = np.asarray(
            value_diagnostic.get("legal_action_mask", ()), dtype=np.int64
        )
        value_frontier_mask = np.asarray(
            value_diagnostic.get("frontier_action_mask", ()), dtype=np.int64
        )
        value_message_utilities = np.asarray(
            value_diagnostic.get("message_action_utilities", ()),
            dtype=np.float64,
        )
        value_coordinate_joint_q = np.asarray(
            value_diagnostic.get("coordinate_joint_q_values", ()),
            dtype=np.float64,
        )
        value_coordinate_factor_q = np.asarray(
            value_diagnostic.get("coordinate_factor_q_values", ()),
            dtype=np.float64,
        )
        if value_slot_ids.size == 0:
            for name, value in (
                    ("legal mask", value_legal_mask),
                    ("frontier mask", value_frontier_mask),
                    ("message utility", value_message_utilities),
                    ("coordinate joint-Q", value_coordinate_joint_q),
                    ("coordinate factor-Q", value_coordinate_factor_q)):
                if value.size != 0:
                    raise RuntimeError(
                        "empty frontier value-ranking slot population has "
                        "nonempty {}".format(name)
                    )
            value_legal_mask = value_legal_mask.reshape(
                expected_action_shape
            )
            value_frontier_mask = value_frontier_mask.reshape(
                expected_action_shape
            )
            value_message_utilities = value_message_utilities.reshape(
                expected_action_shape
            )
            value_coordinate_joint_q = value_coordinate_joint_q.reshape(
                expected_action_shape
            )
            value_coordinate_factor_q = value_coordinate_factor_q.reshape(
                expected_factor_shape
            )
        else:
            if value_sampled != 1:
                raise RuntimeError(
                    "unsampled frontier value-ranking diagnostic has slots"
                )
            for name, value, expected_shape in (
                    ("legal mask", value_legal_mask, expected_action_shape),
                    ("frontier mask", value_frontier_mask,
                     expected_action_shape),
                    ("message utility", value_message_utilities,
                     expected_action_shape),
                    ("coordinate joint-Q", value_coordinate_joint_q,
                     expected_action_shape),
                    ("coordinate factor-Q", value_coordinate_factor_q,
                     expected_factor_shape)):
                if tuple(value.shape) != tuple(expected_shape):
                    raise RuntimeError(
                        "frontier value-ranking {} shape mismatch: got {}, "
                        "expected {}".format(
                            name, tuple(value.shape), tuple(expected_shape)
                        )
                    )
            if np.any(
                    (value_legal_mask != 0) & (value_legal_mask != 1)):
                raise RuntimeError(
                    "frontier value-ranking legality mask is not binary"
                )
            if np.any(
                    (value_frontier_mask != 0)
                    & (value_frontier_mask != 1)):
                raise RuntimeError(
                    "frontier value-ranking action mask is not binary"
                )
            if np.any(value_frontier_mask > value_legal_mask):
                raise RuntimeError(
                    "frontier value-ranking mask enabled an illegal action"
                )
            if np.any(value_frontier_mask.sum(axis=1) <= 0):
                raise RuntimeError(
                    "frontier value-ranking mask has an empty action set"
                )
            legal_entries = value_legal_mask.astype(bool)
            if not np.isfinite(value_message_utilities[legal_entries]).all():
                raise FloatingPointError(
                    "frontier value-ranking message utility is non-finite"
                )
            if not np.isfinite(value_coordinate_joint_q[legal_entries]).all():
                raise FloatingPointError(
                    "frontier value-ranking coordinate joint-Q is non-finite"
                )
            factor_legal_entries = np.repeat(
                legal_entries[:, :, None], value_factor_count, axis=2
            )
            if not np.isfinite(
                    value_coordinate_factor_q[factor_legal_entries]).all():
                raise FloatingPointError(
                    "frontier value-ranking coordinate factor-Q is non-finite"
                )
        factor_members = []
        factor_orders = []
        for factor_id in range(learned_factor_count):
            members = np.flatnonzero(
                adjacency_np[:, factor_id] > 0
            ).astype(np.int64).tolist()
            factor_members.append(
                "f{}:{}".format(
                    factor_id,
                    "|".join(str(int(slot)) for slot in members),
                )
            )
            factor_orders.append(str(len(members)))

        capture_events = list(step_info.get("capture_events", []))
        q_values = _as_numpy(greedy_joint_q_value).reshape(-1)
        if q_values.size != 1:
            raise RuntimeError(
                "pre-capture joint-Q diagnostic requires one environment value"
            )
        return {
            "diagnostic_schema_version": 4,
            "environment_episode_id": int(environment_episode_id),
            "training_env_step": int(training_env_step),
            "episode_step": int(episode_step),
            "state_action_alignment": "action_s_t__info_s_t_plus_1",
            "capture_target_ids": ";".join(
                str(int(event["target_id"])) for event in capture_events
            ),
            "success_now": int(bool(step_info.get("success_now", False))),
            "food_alive_statuses": ";".join(
                str(int(bool(value)))
                for value in step_info["food_alive_statuses"]
            ),
            "food_freeze_remaining": ";".join(
                format(float(value), ".8g")
                for value in step_info["food_freeze_remaining"]
            ),
            "food_positions": ";".join(
                "{}:{}".format(int(position[0]), int(position[1]))
                for position in food_positions
            ),
            "player_slot_positions": ";".join(
                "" if position is None else "{}:{}".format(
                    int(position[0]), int(position[1])
                )
                for position in player_slot_positions
            ),
            "min_alive_player_distance_to_food": ";".join(
                format(value, ".8g") for value in min_distances
            ),
            "nearest_alive_player_slots_to_food": ";".join(nearest_slots),
            "food_visible_player_slots": ";".join(
                "|".join(str(int(slot)) for slot in visible_slots)
                for visible_slots in step_info[
                    "food_visible_player_slots"
                ]
            ),
            "food_observer_counts": ";".join(
                str(len(visible_slots))
                for visible_slots in step_info[
                    "food_visible_player_slots"
                ]
            ),
            "joint_explore": int(action_diagnostic.get("joint_explore", -1)),
            "epsilon": float(
                action_diagnostic.get("epsilon", float("nan"))
            ),
            "greedy_actions": ";".join(
                str(int(value))
                for value in action_diagnostic.get("greedy_actions", [])
            ),
            "selected_actions": ";".join(
                str(int(value))
                for value in action_diagnostic.get("selected_actions", [])
            ),
            "random_replacement_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "random_replacement_slots", []
                )
            ),
            "pre_capture_visible_prey_quorum_guard_applied": int(
                action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_guard_applied", 0
                )
            ),
            "pre_capture_visible_prey_quorum_protected_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_protected_slots", []
                )
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied": int(
                action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied",
                    0,
                )
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots",
                    [],
                )
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots",
                    [],
                )
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots",
                    [],
                )
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots": "|".join(
                str(int(value)) for value in action_diagnostic.get(
                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots",
                    [],
                )
            ),
            "unconstrained_greedy_actions": ";".join(
                str(int(value)) for value in action_diagnostic.get(
                    "unconstrained_greedy_actions", []
                )
            ),
            "frontier_value_ranking_schema_version": 1,
            "frontier_value_ranking_sampled": value_sampled,
            "frontier_value_ranking_slot_ids": "|".join(
                str(int(value)) for value in value_slot_ids
            ),
            "frontier_value_ranking_action_dim": value_action_dim,
            "frontier_value_ranking_factor_count": value_factor_count,
            "frontier_value_ranking_legal_action_mask": _flat_int_text(
                value_legal_mask
            ),
            "frontier_value_ranking_action_mask": _flat_int_text(
                value_frontier_mask
            ),
            "frontier_value_ranking_message_utilities": _flat_float_text(
                value_message_utilities
            ),
            "frontier_value_ranking_coordinate_joint_q": _flat_float_text(
                value_coordinate_joint_q
            ),
            "frontier_value_ranking_coordinate_factor_q": _flat_float_text(
                value_coordinate_factor_q
            ),
            "greedy_joint_q_value": float(q_values[0]),
            "greedy_message_action_margins": _flat_float_text(
                greedy_message_action_margins
            ),
            "greedy_factor_q_values": _flat_float_text(
                greedy_factor_q_values
            ),
            "active_factor_agent_slots": ";".join(factor_members),
            "active_factor_orders": ";".join(factor_orders),
        }

    @staticmethod
    def _finalize_pre_capture_window(
            snapshots,
            first_capture_step: int,
            first_capture_target_id: int,
            first_capture_participant_slots,
            history_steps=32):
        """Label an inclusive pre-capture window without recomputation."""
        if not snapshots:
            raise RuntimeError("first capture has no pre-capture snapshots")
        first_capture_step = int(first_capture_step)
        if int(snapshots[-1]["episode_step"]) != first_capture_step:
            raise RuntimeError(
                "pre-capture snapshot/action alignment missed capture step"
            )
        rows = []
        if history_steps is None:
            selected_snapshots = list(snapshots)
        else:
            history_steps = int(history_steps)
            if history_steps < 0:
                raise ValueError("pre-capture history_steps must be non-negative")
            selected_snapshots = list(snapshots)[-(history_steps + 1):]
        for snapshot in selected_snapshots:
            row = dict(snapshot)
            offset = int(row["episode_step"]) - first_capture_step
            if offset > 0 or (
                    history_steps is not None and offset < -history_steps):
                raise RuntimeError("pre-capture diagnostic offset escaped window")
            row["offset_to_first_capture"] = int(offset)
            row["first_capture_target_id"] = int(first_capture_target_id)
            row["first_capture_participant_slots"] = ";".join(
                str(int(slot)) for slot in first_capture_participant_slots
            )
            rows.append(row)
        return rows

    def _dump_train_pre_capture_rows(self, rows: List[Dict[str, Any]]):
        """Persist first-capture t-32..t rows without forwards or RNG draws."""
        if not rows:
            return
        _append_df_csv(
            os.path.join(
                self._get_run_dir(),
                "progress_train_pre_capture_32step.csv",
            ),
            pd.DataFrame(rows),
        )

    def _dump_train_pre_capture_prefix_rows(
            self, rows: List[Dict[str, Any]]):
        """Persist the full episode prefix through first capture.

        Rows reuse already-computed action, Q, factor, and environment info.
        Retaining them changes neither policy/environment forwards nor RNG.
        """
        if not rows:
            return
        _append_df_csv(
            os.path.join(
                self._get_run_dir(),
                "progress_train_pre_capture_prefix.csv",
            ),
            pd.DataFrame(rows),
        )

    def _dump_train_post_capture_rows(self, rows: List[Dict[str, Any]]):
        """Persist only the first-capture 24-step window for each episode.

        The rows are assembled entirely from the already-computed action
        diagnostic and environment ``info`` payload.  This diagnostic performs
        no policy/environment forward pass and consumes no RNG.
        """
        if not rows:
            return
        _append_df_csv(
            os.path.join(
                self._get_run_dir(),
                "progress_train_post_capture_24step.csv",
            ),
            pd.DataFrame(rows),
        )

    def _dump_joint_exploration_episode_diagnostic(
            self, train_step: int, episode_index: int, diagnostics, env_info):
        """Persist one trajectory-neutral row per formal training episode."""
        if not diagnostics:
            return

        schema_versions = {int(item["schema_version"]) for item in diagnostics}
        if schema_versions != {4}:
            raise RuntimeError(
                "unexpected joint exploration diagnostic schema version"
            )
        decision_count = len(diagnostics)
        explore_flags = [int(item["joint_explore"]) for item in diagnostics]
        explore_count = int(sum(explore_flags))
        non_explore_count = int(decision_count - explore_count)

        streak_lengths = []
        streak = 0
        for explored in explore_flags:
            if explored:
                if streak > 0:
                    streak_lengths.append(streak)
                    streak = 0
            else:
                streak += 1
        if streak > 0:
            streak_lengths.append(streak)

        hist = np.sum(
            np.asarray(
                [item["action_histogram"] for item in diagnostics],
                dtype=np.int64,
            ),
            axis=0,
        )
        explore_hist = np.sum(
            np.asarray(
                [
                    item["action_histogram"]
                    for item in diagnostics
                    if int(item["joint_explore"]) == 1
                ],
                dtype=np.int64,
            ),
            axis=0,
        ) if explore_count > 0 else np.zeros_like(hist)

        non_explore_mismatch_count = int(sum(
            int(item["non_explore_greedy_mismatch"])
            for item in diagnostics
        ))
        invalid_action_count = int(sum(
            int(item["invalid_available_action_count"])
            for item in diagnostics
        ))
        dead_slot_violation_count = int(sum(
            int(item["dead_slot_non_stay_violation_count"])
            for item in diagnostics
        ))
        if non_explore_mismatch_count != 0:
            raise RuntimeError(
                "joint epsilon non-explore branch changed the production "
                "greedy joint action"
            )
        if invalid_action_count != 0 or dead_slot_violation_count != 0:
            raise RuntimeError(
                "joint epsilon exploration selected an unavailable action"
            )

        selected_joint_actions = {
            tuple(int(value) for value in item["selected_actions"])
            for item in diagnostics
        }
        explored_joint_actions = {
            tuple(int(value) for value in item["selected_actions"])
            for item in diagnostics
            if int(item["joint_explore"]) == 1
        }
        epsilon_values = np.asarray(
            [float(item["epsilon"]) for item in diagnostics],
            dtype=np.float64,
        )
        alive_counts = np.asarray(
            [int(item["alive_slot_count"]) for item in diagnostics],
            dtype=np.int64,
        )
        random_replacement_counts = np.asarray(
            [
                int(item.get("random_replacement_slot_count", 0))
                for item in diagnostics
            ],
            dtype=np.int64,
        )
        bounded_applied_count = int(sum(
            int(item.get("post_capture_explore_bounded_applied", 0))
            for item in diagnostics
        ))
        pre_capture_quorum_guard_applied_count = int(sum(
            int(item.get(
                "pre_capture_visible_prey_quorum_guard_applied", 0
            ))
            for item in diagnostics
        ))
        pre_capture_quorum_protected_counts = np.asarray(
            [
                len(item.get(
                    "pre_capture_visible_prey_quorum_protected_slots", []
                ))
                for item in diagnostics
            ],
            dtype=np.int64,
        )
        frontier_applied_count = int(sum(
            int(item.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied",
                0,
            ))
            for item in diagnostics
        ))
        frontier_eligible_counts = np.asarray([
            len(item.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots",
                [],
            ))
            for item in diagnostics
        ], dtype=np.int64)
        frontier_constrained_counts = np.asarray([
            len(item.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots",
                [],
            ))
            for item in diagnostics
        ], dtype=np.int64)
        frontier_conflict_counts = np.asarray([
            len(item.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots",
                [],
            ))
            for item in diagnostics
        ], dtype=np.int64)
        frontier_reranked_counts = np.asarray([
            len(item.get(
                "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots",
                [],
            ))
            for item in diagnostics
        ], dtype=np.int64)
        row = {
            "diagnostic_schema_version": 4,
            "step": int(train_step),
            "episode_index": int(episode_index),
            "joint_decision_count": int(decision_count),
            "joint_explore_flag_bits": "".join(
                str(flag) for flag in explore_flags
            ),
            "final_equals_greedy_flag_bits": "".join(
                str(int(item["final_equals_greedy"]))
                for item in diagnostics
            ),
            "alive_slot_count_trace": ";".join(
                str(int(item["alive_slot_count"]))
                for item in diagnostics
            ),
            "explore_decision_count": explore_count,
            "non_explore_decision_count": non_explore_count,
            "empirical_explore_rate": float(explore_count / decision_count),
            "empirical_non_explore_rate": float(non_explore_count / decision_count),
            "epsilon_mean": float(epsilon_values.mean()),
            "epsilon_min": float(epsilon_values.min()),
            "epsilon_max": float(epsilon_values.max()),
            "final_equals_greedy_count": int(sum(
                int(item["final_equals_greedy"])
                for item in diagnostics
            )),
            "explore_final_equals_greedy_count": int(sum(
                int(item["explore_final_equals_greedy"])
                for item in diagnostics
            )),
            "non_explore_greedy_mismatch_count": non_explore_mismatch_count,
            "invalid_available_action_count": invalid_action_count,
            "dead_slot_non_stay_violation_count": dead_slot_violation_count,
            "alive_slot_count_min": int(alive_counts.min()),
            "alive_slot_count_max": int(alive_counts.max()),
            "alive_slot_count_mean": float(alive_counts.mean()),
            "unique_joint_action_count": int(len(selected_joint_actions)),
            "explore_unique_joint_action_count": int(len(explored_joint_actions)),
            "explore_unique_alive_action_count_mean": float(np.mean([
                int(item["unique_alive_action_count"])
                for item in diagnostics
                if int(item["joint_explore"]) == 1
            ])) if explore_count > 0 else 0.0,
            "post_capture_explore_bounded_applied_count": bounded_applied_count,
            "pre_capture_visible_prey_quorum_guard_applied_count": (
                pre_capture_quorum_guard_applied_count
            ),
            "pre_capture_visible_prey_quorum_protected_slot_count_sum": int(
                pre_capture_quorum_protected_counts.sum()
            ),
            "pre_capture_visible_prey_quorum_protected_slot_count_max": int(
                pre_capture_quorum_protected_counts.max()
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_applied_count": (
                frontier_applied_count
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slot_count_sum": int(
                frontier_eligible_counts.sum()
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slot_count_sum": int(
                frontier_constrained_counts.sum()
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slot_count_sum": int(
                frontier_conflict_counts.sum()
            ),
            "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slot_count_sum": int(
                frontier_reranked_counts.sum()
            ),
            "random_replacement_slot_count_sum": int(
                random_replacement_counts.sum()
            ),
            "random_replacement_slot_count_max": int(
                random_replacement_counts.max()
            ),
            "non_explore_streak_count": int(len(streak_lengths)),
            "non_explore_streak_max": int(max(streak_lengths) if streak_lengths else 0),
            "non_explore_streak_ge2_count": int(sum(
                length >= 2 for length in streak_lengths
            )),
            "non_explore_streak_ge4_count": int(sum(
                length >= 4 for length in streak_lengths
            )),
            "non_explore_streak_ge8_count": int(sum(
                length >= 8 for length in streak_lengths
            )),
            "capture_events": float(env_info.get("capture_events", 0.0)),
            "exact_matched_capture_events": float(
                sum(
                    int(event.get("capture_identity_matched_event_count", 0))
                    for event in self.last_episode_step_info.get(
                        "topology_events", []
                    )
                ) if isinstance(self.last_episode_step_info, dict) else 0.0
            ),
            "win": float(env_info.get("win_rate", 0.0)),
            "terminal_win_bonus_event_count": float(
                env_info.get("terminal_win_bonus_event_count", 0.0)
            ),
        }
        for action_index, count in enumerate(hist.tolist()):
            row["selected_action_{}_count".format(action_index)] = int(count)
        for action_index, count in enumerate(explore_hist.tolist()):
            row["explore_action_{}_count".format(action_index)] = int(count)

        _append_df_csv(
            os.path.join(
                self._get_run_dir(),
                "progress_train_joint_exploration_episode.csv",
            ),
            pd.DataFrame([row]),
        )

    def eval(self):
        """评估若干回合并做日志。
         改进：一次评估可能包含 num_eval_episodes 个 episode，本实现会对“每一个评估 episode”
        都导出两张随时间步变化的表（num_players、individual_rewards），并且不会重复 dump。
        """
        self.trainer.prep_rollout()

        # 评估发生时的训练步数（评估本身不推进 total_env_steps）
        eval_step = int(getattr(self, "total_env_steps", 0))

        # 评估调用编号：用于区分不同 eval 调用，避免覆盖 per-episode 文件
        eval_id = int(getattr(self, "_eval_call_id", 0))
        self._eval_call_id = eval_id + 1

        eval_infos: Dict[str, list] = {
            "win_rate": [],
            "average_episode_rewards": [],
            "capture_events": [],
            "first_success_step": [],
            "partial_capture_without_win": [],
            "num_players": [],
            "num_players_mean": [],
            "num_players_min": [],
            "num_players_max": [],
            "num_players_std": [],
            "join_events": [],
            "leave_events": [],
            "recover_events": [],
            "roster_change_events": [],
            "pending_recovery_final": [],
            "recovery_completion_rate": [],
            "activation_state_resets": [],
            "active_ratio_mean": [],
            "active_ratio_min": [],
            "active_ratio_max": [],
            "episode_real_length": [],
            "adj_valid_factor_ratio": [],
            "adj_empty_factor_ratio": [],
            "adj_order1_ratio": [],
            "adj_order2_ratio": [],
            "adj_order3_ratio": [],
            "adj_invalid_factor_ratio": [],
            "adj_mean_order": [],
            "adj_active_agent_coverage": [],
            "adj_uncovered_active_agent_ratio": [],
            "adj_active_agent_degree": [],
            "adj_connected_graph_ratio": [],
            "adj_selected_prob_mean": [],
            "adj_selected_prob_min": [],
            "adj_factor_retention_ratio": [],
            "graph_explore_enabled": [],
            "graph_eval_rng_enabled": [],
            "topology_persistence_enabled": [],
        }

        for ep_i in range(self.args.num_eval_episodes):
            env_info = self.collecter(explore=False, training_episode=False, warmup=False)  # 采样一个评估 episode

            # 先 dump：每个 episode 只 dump 一次
            step_info = getattr(self, "last_episode_step_info", None)
            if isinstance(step_info, dict):
                self._dump_eval_episode_step_tables(eval_step=eval_step, eval_ep=ep_i,
                                                    step_info=step_info, eval_id=eval_id)
            # 再累计 scalar 指标（用于 progress_eval.csv）
            for k, v in env_info.items():
                if k not in eval_infos:
                    eval_infos[k] = []
                eval_infos[k].append(v)

        # first_success_step is conditional on an episode succeeding; exclude
        # the -1 no-success sentinel before scalar aggregation.
        eval_infos_for_log = self._prepare_episode_metrics_for_logging(
            eval_infos
        )
        self.log_env(eval_infos_for_log, suffix="eval_")

        eval_summary = {}
        for k, v in eval_infos_for_log.items():
            # Baseline algorithms do not emit SDDFG-only adjacency metrics.  Do
            # not call np.mean([]): besides producing a meaningless NaN, it
            # raises a RuntimeWarning and makes a healthy baseline run look as
            # if its training values became non-finite.
            if len(v) == 0:
                continue
            try:
                eval_summary["eval_" + k] = float(np.mean(v))
            except Exception:
                pass

        # 保存 best eval checkpoint
        self.maybe_save_best_eval(eval_summary)
        self.last_eval_summary = dict(eval_summary)
        self.last_eval_summary_step = int(self.total_env_steps)

        return eval_summary

    @torch.no_grad()
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        收集一个完整 episode 并入缓冲，所有智能体共享同一个策略。
        新增 exist_mask 的获取与传递。
        :param explore: (bool) 收集此回合时是否使用探索策略。
        :param training_episode: (bool) 此回合用于评估还是训练。
        :param warmup: (bool) 此回合是否在预热阶段收集。
        :return env_info: (dict) 包含有关展开数据的信息（总奖励等）。
        """
        env_info: Dict[str, Any] = {}
        joint_exploration_diagnostics = []
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env
        graph_explore, graph_use_eval_rng = resolve_graph_sampling_mode(
            action_explore=explore,
            training_episode=training_episode,
            warmup=warmup,
            use_train_consistent_eval_graph=getattr(
                self.args,
                "use_train_consistent_eval_graph",
                False,
            ) and self.algorithm_name == "sddfg",
        )
        graph_sample_kwargs = {
            "explore": graph_explore,
            "t_env": self.total_env_steps,
        }
        if (
            hasattr(self, "adj_network")
            and hasattr(self.adj_network, "eval_rng")
        ):
            graph_sample_kwargs["use_eval_rng"] = graph_use_eval_rng

        obs, share_obs, avail_acts = env.reset()

        # 可视化/录制为可选功能；训练默认关闭。
        render_enabled = bool(getattr(self.args, "render", False))
        save_video = bool(getattr(self.args, "save_video", False))
        video_path = getattr(self.args, "video_path", None)
        render_fps = float(getattr(self.args, "render_fps", getattr(self.args, "fps", 10)))

        if video_path is not None and str(video_path).strip() != "":
            save_video = True
        if save_video:
            if video_path is None or str(video_path).strip() == "":
                video_path = os.path.join(str(getattr(self, "run_dir", ".")), "episode.mp4")
            video_path = os.path.abspath(str(video_path))

        # 仅支持 DummyVecEnv 的 env.envs[0] 渲染/抓帧
        base_env = None
        if hasattr(env, "envs") and isinstance(getattr(env, "envs"), list) and len(env.envs) > 0:
            base_env = env.envs[0]

        video_writer = None
        video_size = None

        def _render_and_grab_rgb():
            nonlocal render_enabled

            if base_env is None:
                return None

            try:
                if render_enabled:
                    base_env.render()
            except Exception:
                # 服务器无显示设备时关闭窗口渲染；如需视频则继续走离屏路径。
                render_enabled = False

            if not save_video:
                return None

            if render_enabled:
                try:
                    import pygame
                    if getattr(base_env, "visualizer", None) is not None:
                        screen = getattr(base_env.visualizer, "screen", None)
                        if screen is not None:
                            arr = pygame.surfarray.array3d(screen)
                            frame = np.transpose(arr, (1, 0, 2)).copy()
                            return frame
                except Exception:
                    pass

            # 2) 离屏渲染：不依赖显示设备。直接用 base_env.grid 画 RGB 帧（纯 numpy）
            try:
                g = getattr(base_env, "grid", None)
                if g is None:
                    return None

                # 环境里的 grid 是 list-of-lists：0/1/2/3（见 wolfpack_penalty_open.py）
                # 这里统一转成 ndarray
                if isinstance(g, np.ndarray):
                    grid_arr = g
                else:
                    grid_arr = np.asarray(g, dtype=np.int32)

                if grid_arr.ndim != 2:
                    return None

                H, W = int(grid_arr.shape[0]), int(grid_arr.shape[1])
                cell = int(getattr(self.args, "render_cell", 20))  # 每格像素，匹配 Visualizer 的 20 更直观

                # 按 Visualizer 规则配色：默认蓝；1黑；2白；3红
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
                rgb[:] = (0, 0, 255)  # BLUE
                rgb[grid_arr == 1] = (0, 0, 0)  # BLACK
                rgb[grid_arr == 2] = (255, 255, 255)  # WHITE
                rgb[grid_arr == 3] = (255, 0, 0)  # RED

                # 放大到可视尺寸
                frame = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
                return frame

            except Exception:
                return None

        def _ensure_writer(frame_rgb):
            nonlocal video_writer, video_size
            if frame_rgb is None:
                return

            h, w = frame_rgb.shape[:2]
            if video_writer is None:
                os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
                try:
                    import imageio.v2 as imageio
                except Exception as e:
                    raise RuntimeError(f"[video] imageio 不可用，请安装：pip install imageio imageio-ffmpeg。原因：{e}")

                # 优先 libx264，失败就用 mpeg4（更兼容）
                try:
                    video_writer = imageio.get_writer(video_path, fps=render_fps, codec="libx264")
                except Exception:
                    video_writer = imageio.get_writer(video_path, fps=render_fps, codec="mpeg4")

                video_size = (w, h)

        def _write_frame(frame_rgb):
            if frame_rgb is None:
                return
            _ensure_writer(frame_rgb)
            if video_writer is None:
                return
            h, w = frame_rgb.shape[:2]
            if video_size != (w, h):
                return
            video_writer.append_data(frame_rgb)

        # reset 后先渲染/抓首帧
        if render_enabled or save_video:
            frame0 = _render_and_grab_rgb()
            _write_frame(frame0)
            if render_enabled and render_fps > 0:
                time.sleep(1.0 / render_fps)

        # ========== Wolfpack infos 逐步轨迹（用于评估导出）==========
        # 注意：这里记录的是“每一个 episode”的逐步信息；无论 explore=True(训练) 还是 explore=False(评估)，都会记录。
        num_players_traj: List[int] = []
        individual_rewards_traj: List[np.ndarray] = []  # list of (N,1)
        active_masks_traj: List[np.ndarray] = []  # list of (N,1)
        topology_events_traj: List[Dict[str, Any]] = []
        team_rewards_traj: List[float] = []
        adj_metrics_traj: List[Dict[str, float]] = []  # list of per-step adj structure metrics
        prev_adj_for_metrics = None

        # 显式事件计数避免仅依赖 num_players 差分；同一步退出和加入可能相互抵消。
        explicit_event_info_seen = False
        left_event_count = 0
        join_event_count = 0
        recover_event_count = 0
        topology_change_steps = 0
        pending_recovery_final = 0
        capture_event_count = 0
        capture_event_field_seen = False
        first_success_step = -1
        terminal_win_first_transition_count = 0
        terminal_win_bonus_event_count = 0
        terminal_win_bonus_sum = 0.0
        terminal_win_bonus_max = 0.0
        # Number of 0->1 slot activations whose recurrent state and previous
        # action were reset. This covers both new joins and recoveries.
        activation_state_reset_count = 0
        environment_episode_ids = np.arange(
            int(self.num_adj_episodes_collected),
            int(self.num_adj_episodes_collected) + int(self.num_envs),
            dtype=np.int64,
        )
        episode_dynamic_slots = [
            set() for _ in range(self.num_envs)
        ]
        episode_candidate_event_provenance = [
            [] for _ in range(self.num_envs)
        ]
        episode_active_event_provenance = [
            [] for _ in range(self.num_envs)
        ]
        episode_candidate_event_keys = [
            set() for _ in range(self.num_envs)
        ]
        episode_active_event_keys = [
            set() for _ in range(self.num_envs)
        ]
        episode_capture_event_contracts = [
            {} for _ in range(self.num_envs)
        ]
        first_capture_step_by_env = [None for _ in range(self.num_envs)]
        first_capture_target_by_env = [None for _ in range(self.num_envs)]
        first_capture_participants_by_env = [None for _ in range(self.num_envs)]
        post_capture_pursuit_active_by_env = [False for _ in range(self.num_envs)]
        pre_capture_snapshots_by_env = [
            [] for _ in range(self.num_envs)
        ]
        pre_capture_rows: List[Dict[str, Any]] = []
        pre_capture_prefix_rows: List[Dict[str, Any]] = []
        post_capture_rows: List[Dict[str, Any]] = []

        # 每次 rollout 开始先清空，避免 eval() 读到上一个 episode 的残留
        self.last_episode_step_info = None
        # 缓存本 episode 最近一次 infos（用于 episode 级统计，如 win_rate / episode_limit）
        last_info_dict: Optional[Dict[str, Any]] = None

        # for shared policy, we concatenate obs across envs and agents
        last_acts_batch = np.zeros((self.num_envs * self.num_agents, self.act_dim), dtype=np.float32)
        rnn_states_batch = np.zeros((self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)

        # ------- episode 级缓存（shape 与 SMACRunner/PREYRunner 完全一致） -------
        episode_obs = {
            p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_share_obs = {
            p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim),
                           dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_acts = {
            p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_rewards = {
            p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_dones = {
            p_id: np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_dones_env = {
            p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_avail_acts = {
            p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_adj = {
            p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.num_factor), dtype=np.int64)
            for p_id in self.policy_ids
        }
        episode_prob_adj = {
            p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.num_factor), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_qtot = {
            p_id: np.zeros((self.episode_length, self.num_envs, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        # f_v/f_q 在 PREY/SMAC 里使用了 (num_factor   num_agents) 的槽位；保持一致便于 Trainer 复用
        episode_f_v = {
            p_id: np.zeros((self.episode_length, self.num_envs, self.num_factor + self.num_agents, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_f_q = {
            p_id: np.zeros((self.episode_length, self.num_envs, self.num_factor + self.num_agents, 1), dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_rnn_states = {
            p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.hidden_size),
                           dtype=np.float32)
            for p_id in self.policy_ids
        }
        episode_terminal_win_rewards = np.zeros(
            (self.episode_length, self.num_envs, 1), dtype=np.float32
        )
        # Real environment events used by AdjBuffer credit. These are kept
        # separate from rewards so distance shaping can never masquerade as a
        # capture event.
        episode_capture_counts = np.zeros(
            (self.episode_length, self.num_envs, 1),
            dtype=np.float32,
        )
        episode_success_now = np.zeros_like(
            episode_capture_counts,
            dtype=np.float32,
        )
        episode_capture_factor_matches = np.zeros(
            (
                self.episode_length,
                self.num_envs,
                self.num_factor,
            ),
            dtype=np.float32,
        )
        episode_capture_candidate_only_matches = np.zeros(
            (
                self.episode_length,
                self.num_envs,
                len(canonical_capture_factor_catalog(self.num_agents, 3)),
            ),
            dtype=np.float32,
        )
        # Candidate replay metadata channels are:
        # [first_reachable_competitor_margin, competitor_rank, valid_mask,
        # graph_policy_version].
        episode_capture_candidate_behavior = np.zeros(
            (
                self.episode_length,
                self.num_envs,
                len(canonical_capture_factor_catalog(self.num_agents, 3)),
                4,
            ),
            dtype=np.float32,
        )
        episode_capture_identity_matched = np.zeros_like(
            episode_capture_counts,
            dtype=np.float32,
        )
        episode_capture_identity_unmatched = np.zeros_like(
            episode_capture_counts,
            dtype=np.float32,
        )
        episode_capture_identity_candidates = np.zeros_like(
            episode_capture_counts,
            dtype=np.float32,
        )

        active_masks = self._infer_active_masks_from_obs(obs)
        dones = (active_masks == 0)  # bool：空槽位一开始就视为 done

        # -------- 采集整回合 --------
        t = 0
        while t < self.episode_length:

            # 当前图、动作和 Q 值都对应 s_t，必须保留该时刻的 mask。
            dones_curr = dones.copy()
            active_masks_curr = (~dones_curr).astype(np.float32)

            obs_batch = np.concatenate(obs)
            states_batch = np.concatenate(share_obs)
            avail_acts_batch = np.concatenate(avail_acts)

            # ------ 与 SMACRunner/PREYRunner 同：先更新 RNN hidden ------
            if self.algorithm_name in self.adj_correlation:  # get actions for all agents to step the env

                _, rnn_states_batch, _ = policy.get_hidden_states(
                    obs_batch,
                    last_acts_batch,
                    rnn_states_batch,
                    dones=dones.reshape(-1, 1),
                )
                rnn_states_batch = self._mask_rnn_states_by_dones(rnn_states_batch, dones)

                if self.use_dyn_graph:
                    step_graph_sample_kwargs = dict(graph_sample_kwargs)
                    if hasattr(self.adj_network, "use_topology_persistence"):
                        step_graph_sample_kwargs["previous_adj"] = (
                            prev_adj_for_metrics
                        )
                    prob_adj, adj, _ = self.adj_network.sample(
                        obs_batch[None, :],
                        rnn_states_batch.unsqueeze(0),
                        self.use_adj_init,
                        dones,
                        **step_graph_sample_kwargs
                    )
                    candidate_behavior = getattr(
                        self.adj_network,
                        "last_candidate_behavior_metadata",
                        None,
                    )
                    if candidate_behavior is None:
                        raise RuntimeError(
                            "dynamic graph sampling did not expose canonical "
                            "candidate behavior metadata"
                        )
                    candidate_behavior = (
                        candidate_behavior.detach().cpu().numpy()
                    )
                    expected_candidate_behavior_shape = (
                        self.num_envs,
                        episode_capture_candidate_behavior.shape[2],
                        4,
                    )
                    if candidate_behavior.shape != expected_candidate_behavior_shape:
                        raise RuntimeError(
                            "candidate behavior metadata must have shape {}, "
                            "got {}".format(
                                expected_candidate_behavior_shape,
                                candidate_behavior.shape,
                            )
                        )
                    episode_capture_candidate_behavior[t] = candidate_behavior

                    prob_adj, adj = self._mask_adj_by_dones(
                        adj=adj,
                        prob_adj=prob_adj,
                        dones=dones,
                        num_agents=self.num_agents,
                        num_factor=self.num_factor
                    )

                    eye = self._build_alive_eye(self.num_agents, dones, adj.device)
                    adj_all = torch.cat([adj.cpu().detach(), eye.cpu()], dim=2)

                else:
                    # 静态图同样按当前成员状态过滤失效 factor。
                    adj = self.adj.to(self.device) if torch.is_tensor(self.adj) else torch.as_tensor(self.adj,
                                                                                                     device=self.device)
                    if adj.dim() == 2:
                        adj = adj.unsqueeze(0)

                    prob_adj = torch.zeros_like(adj, dtype=torch.float32, device=adj.device)

                    prob_adj, adj = self._mask_adj_by_dones(
                        adj=adj,
                        prob_adj=prob_adj,
                        dones=dones,
                        num_agents=self.num_agents,
                        num_factor=self.num_factor
                    )

                    eye = self._build_alive_eye(self.num_agents, dones, adj.device)
                    adj_all = torch.cat([adj.cpu().detach(), eye.cpu()], dim=2)

                # 动作选择
                if warmup:
                    acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
                else:
                    if self.algorithm_name == 'casec':
                        acts_batch, qtot, _, f_q = policy.get_actions(rnn_states_batch.unsqueeze(0),
                                                                      torch.tensor(avail_acts_batch),
                                                                      t_env=self.total_env_steps,
                                                                      explore=explore)
                    else:
                        post_capture_floor = 0.0
                        if (
                                self.algorithm_name == "sddfg"
                                and bool(post_capture_pursuit_active_by_env[0])):
                            post_capture_floor = (
                                self.post_capture_joint_greedy_floor
                            )
                        action_kwargs = {
                            "t_env": self.total_env_steps,
                            "explore": explore,
                            "adj_input": adj_all.to(self.device),
                            "no_sequence": False,
                            "dones": torch.tensor(dones).to(self.device),
                        }
                        if self.algorithm_name == "sddfg":
                            action_kwargs[
                                "post_capture_joint_greedy_floor"
                            ] = post_capture_floor
                            action_kwargs[
                                "post_capture_explore_max_random_agents"
                            ] = (
                                self.post_capture_explore_max_random_agents
                                if post_capture_floor > 0.0 else 0
                            )
                            if (
                                    (
                                        self.pre_capture_visible_prey_quorum_guard
                                        or self.pre_capture_visible_prey_quorum_greedy_frontier_guard
                                    )
                                    and post_capture_floor <= 0.0):
                                (
                                    local_visible_prey_mask,
                                    local_visible_prey_offsets,
                                ) = self._visible_prey_geometry_from_local_vector_obs(
                                    obs_batch,
                                    num_agents=self.num_agents,
                                    max_food_num=int(getattr(
                                        self.args,
                                        "max_food_num",
                                        0,
                                    )),
                                )
                                action_kwargs[
                                    "pre_capture_visible_prey_mask"
                                ] = local_visible_prey_mask
                                action_kwargs[
                                    "pre_capture_visible_prey_offsets"
                                ] = local_visible_prey_offsets
                                action_kwargs[
                                    "pre_capture_visibility_radius"
                                ] = int(getattr(self.args, "sight_radius", 0))
                                action_kwargs[
                                    "pre_capture_prey_max_step"
                                ] = 1
                                action_kwargs[
                                    "apply_pre_capture_visible_prey_quorum_greedy_frontier_guard"
                                ] = bool(
                                    self.pre_capture_visible_prey_quorum_greedy_frontier_guard
                                )
                        acts_batch, qtot, action_margins, f_q = policy.get_actions(
                            obs_batch[None, :],
                            rnn_states_batch.unsqueeze(0),
                            torch.tensor(avail_acts_batch).to(self.device),
                            **action_kwargs)
                        if (
                                explore
                                and bool(getattr(
                                    policy,
                                    "use_joint_epsilon_exploration",
                                    False,
                                ))):
                            action_diagnostic = getattr(
                                policy,
                                "last_action_exploration_diagnostic",
                                None,
                            )
                            if not isinstance(action_diagnostic, dict):
                                raise RuntimeError(
                                    "joint epsilon production action did not "
                                    "expose its decision diagnostic"
                                )
                            if int(action_diagnostic.get("batch_size", 0)) != 1:
                                raise RuntimeError(
                                    "Wolfpack joint epsilon diagnostics require "
                                    "one environment decision per policy call"
                                )
                            if int(action_diagnostic.get("schema_version", 0)) != 4:
                                raise RuntimeError(
                                    "Wolfpack joint epsilon production action "
                                    "used an unexpected diagnostic schema"
                                )
                            bounded_expected = bool(
                                post_capture_floor > 0.0
                                and self.post_capture_explore_max_random_agents > 0
                                and int(action_diagnostic.get(
                                    "joint_explore", 0
                                )) == 1
                            )
                            bounded_actual = bool(int(action_diagnostic.get(
                                "post_capture_explore_bounded_applied", 0
                            )))
                            if bounded_actual != bounded_expected:
                                raise RuntimeError(
                                    "post-capture bounded exploration lifecycle "
                                    "did not match the production task state"
                                )
                            guard_expected_enabled = bool(
                                self.pre_capture_visible_prey_quorum_guard
                            )
                            guard_actual_enabled = bool(int(
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_guard_enabled",
                                    0,
                                )
                            ))
                            if guard_actual_enabled != guard_expected_enabled:
                                raise RuntimeError(
                                    "pre-capture visible-prey quorum guard "
                                    "enablement did not match runner config"
                                )
                            frontier_expected_enabled = bool(
                                self.pre_capture_visible_prey_quorum_greedy_frontier_guard
                                and post_capture_floor <= 0.0
                            )
                            frontier_actual_enabled = bool(int(
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_enabled",
                                    0,
                                )
                            ))
                            if frontier_actual_enabled != frontier_expected_enabled:
                                raise RuntimeError(
                                    "pre-capture greedy frontier guard "
                                    "enablement did not match runner config"
                                )
                            frontier_eligible_slots = tuple(int(value) for value in (
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_eligible_slots",
                                    (),
                                )
                            ))
                            frontier_constrained_slots = tuple(int(value) for value in (
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_constrained_slots",
                                    (),
                                )
                            ))
                            frontier_conflict_slots = tuple(int(value) for value in (
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_conflict_slots",
                                    (),
                                )
                            ))
                            frontier_reranked_slots = tuple(int(value) for value in (
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_greedy_frontier_guard_reranked_slots",
                                    (),
                                )
                            ))
                            if not set(frontier_reranked_slots).issubset(
                                    set(frontier_constrained_slots)):
                                raise RuntimeError(
                                    "pre-capture greedy frontier guard reranked "
                                    "an unconstrained slot"
                                )
                            if not set(frontier_conflict_slots).issubset(
                                    set(frontier_eligible_slots)):
                                raise RuntimeError(
                                    "pre-capture greedy frontier conflict escaped "
                                    "the eligible population"
                                )
                            if post_capture_floor > 0.0 and (
                                    frontier_eligible_slots
                                    or frontier_constrained_slots
                                    or frontier_conflict_slots
                                    or frontier_reranked_slots):
                                raise RuntimeError(
                                    "pre-capture greedy frontier guard escaped "
                                    "its lifecycle"
                                )
                            raw_greedy = tuple(int(value) for value in (
                                action_diagnostic.get(
                                    "unconstrained_greedy_actions", ()
                                )
                            ))
                            final_greedy = tuple(int(value) for value in (
                                action_diagnostic.get("greedy_actions", ())
                            ))
                            expected_reranked_slots = tuple(
                                slot for slot, (raw_action, final_action) in enumerate(
                                    zip(raw_greedy, final_greedy)
                                )
                                if raw_action != final_action
                            )
                            if (
                                    len(raw_greedy) != self.num_agents
                                    or len(final_greedy) != self.num_agents
                                    or frontier_reranked_slots
                                    != expected_reranked_slots):
                                raise RuntimeError(
                                    "pre-capture greedy frontier rerank "
                                    "diagnostic is inconsistent"
                                )
                            protected_slots = tuple(int(value) for value in (
                                action_diagnostic.get(
                                    "pre_capture_visible_prey_quorum_protected_slots",
                                    (),
                                )
                            ))
                            guard_applied = bool(int(action_diagnostic.get(
                                "pre_capture_visible_prey_quorum_guard_applied",
                                0,
                            )))
                            if bounded_expected and (
                                    guard_applied or protected_slots):
                                raise RuntimeError(
                                    "pre/post-capture exploration guards "
                                    "were applied simultaneously"
                                )
                            if guard_applied != bool(protected_slots):
                                raise RuntimeError(
                                    "pre-capture visible-prey quorum guard "
                                    "diagnostic is internally inconsistent"
                                )
                            if protected_slots and post_capture_floor > 0.0:
                                raise RuntimeError(
                                    "pre-capture visible-prey quorum guard "
                                    "escaped its lifecycle"
                                )
                            expected_protected_slots = ()
                            local_visible_mask = action_kwargs.get(
                                "pre_capture_visible_prey_mask"
                            )
                            if (
                                    guard_expected_enabled
                                    and post_capture_floor <= 0.0
                                    and int(action_diagnostic.get(
                                        "joint_explore", 0
                                    )) == 1):
                                if local_visible_mask is None:
                                    raise RuntimeError(
                                        "pre-capture visible-prey quorum guard "
                                        "lost its local observation mask"
                                    )
                                visible_counts = np.asarray(
                                    local_visible_mask, dtype=np.int64
                                ).sum(axis=0)
                                exact_quorum_columns = np.flatnonzero(
                                    visible_counts == 2
                                )
                                if exact_quorum_columns.size:
                                    expected_protected_slots = tuple(
                                        np.flatnonzero(np.any(
                                            np.asarray(
                                                local_visible_mask,
                                                dtype=bool,
                                            )[:, exact_quorum_columns],
                                            axis=1,
                                        )).astype(np.int64).tolist()
                                    )
                            if protected_slots != expected_protected_slots:
                                raise RuntimeError(
                                    "pre-capture visible-prey quorum guard "
                                    "protected unexpected slots"
                                )
                            if bounded_expected:
                                expected_random_slots = min(
                                    int(action_diagnostic.get(
                                        "alive_slot_count", 0
                                    )),
                                    self.post_capture_explore_max_random_agents,
                                )
                                if int(action_diagnostic.get(
                                        "random_replacement_slot_count", -1
                                )) != expected_random_slots:
                                    raise RuntimeError(
                                        "post-capture explore branch replaced "
                                        "an unexpected number of alive slots"
                                    )
                            elif (
                                    int(action_diagnostic.get(
                                        "joint_explore", 0
                                    )) == 1
                                    and post_capture_floor <= 0.0
                                    and guard_expected_enabled):
                                expected_random_slots = (
                                    int(action_diagnostic.get(
                                        "alive_slot_count", 0
                                    )) - len(expected_protected_slots)
                                )
                                if int(action_diagnostic.get(
                                        "random_replacement_slot_count", -1
                                )) != expected_random_slots:
                                    raise RuntimeError(
                                        "pre-capture guarded explore branch "
                                        "replaced an unexpected number of "
                                        "alive slots"
                                    )
                            joint_exploration_diagnostics.append(
                                dict(action_diagnostic)
                            )
                    if self.use_vfunction:
                        f_v = policy.get_v_values(rnn_states_batch.unsqueeze(0), states_batch[None, :],
                                                  adj_all.to(self.device), no_sequence=False,
                                                  dones=torch.tensor(dones).to(self.device)
                                                  )
            else:
                # 非 DDFG 算法，保持与 MPERunner/SMACRunner 一致
                if warmup:
                    ## completely random actions in warmup phase
                    #acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
                    # advance rnn state (so shapes/logic remain consistent)
                    # _, rnn_states_batch, _ = policy.get_hidden_states(
                    #     obs_batch, last_acts_batch, rnn_states_batch, dones=dones.reshape(-1, 1)
                    # )

                    # warmup 阶段只采随机动作填充 replay buffer。
                    # RNN hidden 不需要前向推进；按照当前 active mask 清理即可。
                    acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
                    rnn_states_batch = self._mask_rnn_states_by_dones(rnn_states_batch, dones)

                else:
                    acts_batch, rnn_states_batch, _ = policy.get_actions(
                        obs_batch,
                        last_acts_batch,
                        rnn_states_batch,
                        avail_acts_batch,
                        t_env=self.total_env_steps,
                        explore=explore
                    )
                    rnn_states_batch = self._mask_rnn_states_by_dones(rnn_states_batch, dones)

                # 普通 baseline 不使用动态图。仍写入 dummy adj/prob_adj，以复用 RecReplayBuffer 字段。
                prob_adj = torch.zeros((self.num_agents, self.num_factor), dtype=torch.float32)
                adj = torch.zeros((self.num_agents, self.num_factor), dtype=torch.int64)

            # numpy 化 & 缓存
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch,
                                                              np.ndarray) else rnn_states_batch.cpu().detach().numpy()

            # replay 中 t 时刻的 hidden 必须与当前 adj/action 对应。环境事件发生后
            # 会重置 live hidden，因此先保存事件前快照，避免时间对齐被破坏。
            rnn_states_for_transition = np.array(rnn_states_batch, copy=True)

            # 只有 s_t 存活的 slot 才保存动作；空槽位的 previous action 必须为 0。
            active_masks_curr_flat = active_masks_curr.reshape(-1, 1)
            last_acts_batch = acts_batch * active_masks_curr_flat

            env_acts = np.split(acts_batch, self.num_envs)

            # Env step
            next_obs, next_share_obs, rewards, env_dones, infos, next_avail_acts = env.step(env_acts)

            # 可视化/录制：每步渲染并抓帧写入 mp4
            if (render_enabled or save_video) and base_env is not None:
                frame_rgb = _render_and_grab_rgb()
                _write_frame(frame_rgb)
                if render_enabled and render_fps > 0:
                    time.sleep(1.0 / render_fps)

            active_masks_next = self._extract_active_masks_batch(infos, self.num_envs, self.num_agents,
                                                                 fallback_obs=next_obs)

            # 区分“图中不存在”和“RNN 需要重置”。新加入/恢复成员应立即参与
            # s_{t+1} 的图，但不能继承该槽位以前的 hidden/previous action。
            joined_masks = np.logical_and(active_masks_curr == 0, active_masks_next == 1)
            next_dones = np.logical_or(env_dones, active_masks_next == 0)
            rnn_reset_masks = np.logical_or(next_dones, joined_masks)

            rnn_states_batch = self._mask_rnn_states_by_dones(
                rnn_states_batch, rnn_reset_masks
            )
            last_acts_batch = last_acts_batch * (
                1.0 - joined_masks.reshape(-1, 1).astype(np.float32)
            )

            # A newly activated slot must never inherit state from the former
            # occupant. Fail immediately instead of silently contaminating the
            # replay buffer and only discovering it after a long experiment.
            joined_flat = joined_masks.reshape(-1).astype(bool)
            if np.any(joined_flat):
                # Episode statistics below refer to env_i=0, matching the
                # runner's existing Wolfpack info extraction convention.
                activation_state_reset_count += int(joined_masks[0].sum())
                if not np.allclose(rnn_states_batch[joined_flat], 0.0, atol=1e-7):
                    raise RuntimeError(
                        "RNN state reset failed for a joined/recovered Wolfpack slot"
                    )
                if not np.allclose(last_acts_batch[joined_flat], 0.0, atol=1e-7):
                    raise RuntimeError(
                        "Previous-action reset failed for a joined/recovered Wolfpack slot"
                    )

            active_masks = active_masks_next
            dones = next_dones

            # ========== 解析 wolfpack env 返回的 infos，并记录逐步轨迹 ==========
            step_info_dicts = [
                self._extract_first_info_dict(infos, env_i=env_i)
                for env_i in range(self.num_envs)
            ]
            for env_i, step_info in enumerate(step_info_dicts):
                if not isinstance(step_info, dict):
                    continue
                episode_dynamic_slots[env_i].update(
                    int(slot)
                    for slot in (
                        list(step_info.get("joined_slots", []))
                        + list(step_info.get("recovered_slots", []))
                    )
                )
                if "capture_count" in step_info:
                    capture_event_field_seen = True
                try:
                    episode_capture_counts[t, env_i, 0] = max(
                        0.0,
                        float(step_info.get("capture_count", 0.0)),
                    )
                except (TypeError, ValueError):
                    episode_capture_counts[t, env_i, 0] = 0.0
                episode_success_now[t, env_i, 0] = float(
                    bool(step_info.get("success_now", False))
                )

                food_alive_statuses = step_info.get(
                    "food_alive_statuses",
                    None,
                )
                if food_alive_statuses is not None:
                    food_alive_statuses = [
                        bool(value) for value in food_alive_statuses
                    ]
                    post_capture_pursuit_active_by_env[env_i] = bool(
                        _is_post_capture_pursuit_active(food_alive_statuses)
                    )

                capture_events = list(step_info.get("capture_events", []))
                episode_step = int(step_info.get("episode_step", t + 1))
                action_diagnostic = getattr(
                    policy, "last_action_exploration_diagnostic", None
                )
                if not isinstance(action_diagnostic, dict):
                    action_diagnostic = {}

                is_first_capture_transition = bool(
                    first_capture_step_by_env[env_i] is None
                    and capture_events
                )
                if (
                        training_episode
                        and not warmup
                        and self.algorithm_name == "sddfg"
                        and first_capture_step_by_env[env_i] is None):
                    snapshots = pre_capture_snapshots_by_env[env_i]
                    snapshots.append(self._build_pre_capture_transition_row(
                        environment_episode_id=int(
                            environment_episode_ids[env_i]
                        ),
                        training_env_step=int(
                            self.total_env_steps + self.num_envs
                        ),
                        episode_step=episode_step,
                        step_info=step_info,
                        action_diagnostic=action_diagnostic,
                        greedy_joint_q_value=qtot,
                        greedy_message_action_margins=action_margins,
                        greedy_factor_q_values=f_q,
                        adjacency=adj_all,
                        learned_factor_count=self.num_factor,
                    ))
                if first_capture_step_by_env[env_i] is None and capture_events:
                    first_capture_step_by_env[env_i] = episode_step
                    first_capture_target_by_env[env_i] = int(
                        capture_events[0]["target_id"]
                    )
                    first_capture_participants_by_env[env_i] = list(
                        capture_events[0].get("participant_slots", [])
                    )
                    if (
                            training_episode
                            and not warmup
                            and self.algorithm_name == "sddfg"
                            and is_first_capture_transition):
                        pre_capture_rows.extend(
                            self._finalize_pre_capture_window(
                                pre_capture_snapshots_by_env[env_i],
                                first_capture_step=episode_step,
                                first_capture_target_id=(
                                    first_capture_target_by_env[env_i]
                                ),
                                first_capture_participant_slots=(
                                    first_capture_participants_by_env[env_i]
                                    or []
                                ),
                            )
                        )
                        pre_capture_prefix_rows.extend(
                            self._finalize_pre_capture_window(
                                pre_capture_snapshots_by_env[env_i],
                                first_capture_step=episode_step,
                                first_capture_target_id=(
                                    first_capture_target_by_env[env_i]
                                ),
                                first_capture_participant_slots=(
                                    first_capture_participants_by_env[env_i]
                                    or []
                                ),
                                history_steps=None,
                            )
                        )

                first_capture_step = first_capture_step_by_env[env_i]
                if (
                        training_episode
                        and not warmup
                        and first_capture_step is not None
                        and episode_step <= int(first_capture_step) + 24):
                    post_capture_rows.append(self._build_post_capture_row(
                        environment_episode_id=int(
                            environment_episode_ids[env_i]
                        ),
                        training_env_step=int(
                            self.total_env_steps + self.num_envs
                        ),
                        episode_step=episode_step,
                        first_capture_step=int(first_capture_step),
                        first_capture_target_id=int(
                            first_capture_target_by_env[env_i]
                        ),
                        first_capture_participant_slots=(
                            first_capture_participants_by_env[env_i] or []
                        ),
                        step_info=step_info,
                        action_diagnostic=action_diagnostic,
                    ))

            info_dict = step_info_dicts[0] if step_info_dicts else {}
            has_fresh_info = isinstance(info_dict, dict) and bool(info_dict)

            if has_fresh_info:
                last_info_dict = info_dict
            else:
                info_dict = last_info_dict if isinstance(last_info_dict, dict) else {}

            if "num_players" in info_dict:
                try:
                    num_players_traj.append(int(info_dict["num_players"]))
                except Exception:
                    pass

            am = info_dict.get("active_masks", None)

            if am is not None:
                try:
                    am_arr = np.asarray(am, dtype=np.float32)
                    if am_arr.ndim == 1:
                        am_arr = am_arr.reshape(-1, 1)
                    if am_arr.ndim == 2:
                        am_arr = am_arr[:, :1]
                    active_masks_traj.append(am_arr.copy())
                except Exception:
                    pass

            ir = info_dict.get("individual_rewards", None)
            if ir is not None:
                try:
                    ir_arr = np.asarray(ir, dtype=np.float32)
                    if ir_arr.ndim == 1:
                        ir_arr = ir_arr.reshape(-1, 1)
                    if ir_arr.ndim == 2:
                        ir_arr = ir_arr[:, :1]
                    individual_rewards_traj.append(ir_arr.copy())
                except Exception:
                    pass

            if "team_reward" in info_dict:
                try:
                    team_rewards_traj.append(float(info_dict["team_reward"]))
                except Exception:
                    team_rewards_traj.append(0.0)

            if has_fresh_info:
                reward_diagnostic = terminal_win_diagnostic_fields(info_dict)
                first_success_now = int(
                    reward_diagnostic["first_success_now"]
                )
                terminal_win_reward = float(
                    reward_diagnostic["terminal_win_reward"]
                )
                episode_terminal_win_rewards[t, env_i, 0] = terminal_win_reward
                expected_win_reward = float(self.args.reward_win)
                reward_tolerance = 64.0 * float(np.finfo(np.float32).eps) * max(
                    1.0,
                    abs(expected_win_reward),
                    abs(terminal_win_reward),
                )
                if (
                        first_success_now
                        and abs(terminal_win_reward - expected_win_reward)
                        > reward_tolerance):
                    raise RuntimeError(
                        "Wolfpack first success did not emit configured "
                        "terminal win reward"
                    )
                terminal_win_first_transition_count += first_success_now
                if abs(terminal_win_reward) > reward_tolerance:
                    terminal_win_bonus_event_count += 1
                    terminal_win_bonus_sum += terminal_win_reward
                    terminal_win_bonus_max = max(
                        terminal_win_bonus_max,
                        abs(terminal_win_reward),
                    )
                capture_event_count += int(info_dict.get("capture_count", 0))
                if (
                    first_success_step < 0
                    and bool(info_dict.get("success_now", False))
                ):
                    first_success_step = int(
                        info_dict.get("episode_step", t + 1)
                    )

            event_keys = {
                "left_count", "joined_count", "recovered_count",
                "topology_changed", "pending_recovery_count"
            }
            if has_fresh_info and any(key in info_dict for key in event_keys):
                explicit_event_info_seen = True
                left_count = int(info_dict.get("left_count", 0))
                joined_count = int(info_dict.get("joined_count", 0))
                recovered_count = int(info_dict.get("recovered_count", 0))
                activated_count = int(joined_masks[0].sum())
                expected_activated_count = joined_count + recovered_count
                if activated_count != expected_activated_count:
                    raise RuntimeError(
                        "Wolfpack active-mask transition disagrees with topology events: "
                        f"mask activated {activated_count} slot(s), but info reports "
                        f"{joined_count} join(s) and {recovered_count} recovery(s)"
                    )

                left_event_count += left_count
                join_event_count += joined_count
                recover_event_count += recovered_count
                pending_recovery_final = int(info_dict.get("pending_recovery_count", 0))

                topology_changed = bool(info_dict.get(
                    "topology_changed",
                    left_count or joined_count or recovered_count,
                ))
                if topology_changed:
                    topology_change_steps += 1

                topology_events_traj.append({
                    "episode_step": int(info_dict.get("episode_step", t + 1)),
                    "topology_changed": topology_changed,
                    "left_count": left_count,
                    "joined_count": joined_count,
                    "recovered_count": recovered_count,
                    "pending_recovery_count": pending_recovery_final,
                    "capture_count": int(info_dict.get("capture_count", 0)),
                    "capture_events": [
                        {
                            "event_id": int(event["event_id"]),
                            "target_id": int(event["target_id"]),
                            "participant_slots": list(
                                event.get("participant_slots", [])
                            ),
                        }
                        for event in info_dict.get("capture_events", [])
                    ],
                    "success_now": bool(info_dict.get("success_now", False)),
                    "first_success_now": bool(info_dict["first_success_now"]),
                    "base_team_reward": float(info_dict["base_team_reward"]),
                    "terminal_win_reward": float(
                        info_dict["terminal_win_reward"]
                    ),
                    "team_reward": float(info_dict["team_reward"]),
                    "left_slots": list(info_dict.get("left_slots", [])),
                    "joined_slots": list(info_dict.get("joined_slots", [])),
                    "recovered_slots": list(info_dict.get("recovered_slots", [])),
                    "slot_uids": list(info_dict.get("slot_uids", [])),
                })

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            dones_env = np.all(env_dones, axis=1)

            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            '''for k in range(self.num_envs):
                 reward_norm = self.reward_scaling(rewards[k,0])
                 rewards[k] = reward_norm'''

            # --------- 写入 episode 缓存（与 SMAC/PREY 完全一致） ---------
            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = env_acts
            episode_rewards[p_id][t] = rewards
            episode_rnn_states[p_id][t] = rnn_states_for_transition

            # here dones store agent done flag of the next step
            if self.algorithm_name in self.adj_correlation:
                env_adj = (
                    adj.cpu().detach().numpy()[0]
                    if self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS
                    else adj.cpu().detach().numpy()
                )
                env_prob_adj = (
                    prob_adj.cpu().detach().numpy()[0]
                    if self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS
                    else prob_adj.cpu().detach().numpy()
                )
                episode_adj[p_id][t] = env_adj
                episode_prob_adj[p_id][t] = env_prob_adj

                if bool(getattr(
                        self.args,
                        "use_adj_capture_to_win_credit",
                        False,
                )):
                    capture_events_by_env = []
                    expected_capture_counts = []
                    for env_i, step_info in enumerate(step_info_dicts):
                        if "capture_events" not in step_info:
                            raise RuntimeError(
                                "capture-to-win identity credit requires "
                                "environment info field 'capture_events'"
                            )
                        capture_events_by_env.append(
                            step_info.get("capture_events")
                        )
                        expected_capture_counts.append(
                            episode_capture_counts[t, env_i, 0]
                        )
                    identity_match = build_capture_identity_factor_weights(
                        current_adj=episode_adj[p_id][t],
                        capture_events_by_episode=capture_events_by_env,
                        expected_capture_counts=np.asarray(
                            expected_capture_counts,
                            dtype=np.float32,
                        ),
                    )
                    episode_capture_factor_matches[t] = identity_match[
                        "factor_weights"
                    ]
                    episode_capture_candidate_only_matches[t] = (
                        identity_match["candidate_only_factor_weights"]
                    )
                    episode_capture_identity_matched[t, :, 0] = (
                        identity_match["matched_event_count"]
                    )
                    episode_capture_identity_unmatched[t, :, 0] = (
                        identity_match["unmatched_event_count"]
                    )
                    episode_capture_identity_candidates[t, :, 0] = (
                        identity_match["candidate_factor_count"]
                    )
                    event_records_by_env = identity_match[
                        "candidate_only_event_records"
                    ]
                    if len(event_records_by_env) != self.num_envs:
                        raise RuntimeError(
                            "candidate event provenance environment axis "
                            "does not match the rollout"
                        )
                    for env_i, event_records in enumerate(
                            event_records_by_env):
                        for event_record in event_records:
                            event_id = int(event_record["event_id"])
                            participant_slots = tuple(
                                int(slot)
                                for slot in event_record["participant_slots"]
                            )
                            event_contract = (
                                int(event_record["target_id"]),
                                int(t),
                                participant_slots,
                            )
                            prior_contract = (
                                episode_capture_event_contracts[env_i].get(
                                    event_id
                                )
                            )
                            if (
                                    prior_contract is not None
                                    and prior_contract != event_contract):
                                raise RuntimeError(
                                    "capture event_id was reused with "
                                    "conflicting prey, step, or participants "
                                    "inside one environment episode"
                                )
                            episode_capture_event_contracts[env_i][
                                event_id
                            ] = event_contract
                            event_key = (
                                event_id,
                                int(event_record["target_id"]),
                                int(event_record["candidate_index"]),
                            )
                            if event_key in episode_candidate_event_keys[env_i]:
                                raise RuntimeError(
                                    "candidate capture event identity was "
                                    "duplicated inside one environment episode"
                                )
                            episode_candidate_event_keys[env_i].add(event_key)
                            static_dynamic_class = (
                                "dynamic"
                                if any(
                                    slot in episode_dynamic_slots[env_i]
                                    for slot in participant_slots
                                )
                                else "static"
                            )
                            contextual_record = dict(event_record)
                            contextual_record.update({
                                "environment_episode_id": int(
                                    environment_episode_ids[env_i]
                                ),
                                "capture_step": int(t),
                                "static_dynamic_class": (
                                    static_dynamic_class
                                ),
                            })
                            episode_candidate_event_provenance[env_i].append(
                                contextual_record
                            )
                    matched_event_records_by_env = identity_match[
                        "matched_event_records"
                    ]
                    if len(matched_event_records_by_env) != self.num_envs:
                        raise RuntimeError(
                            "matched event provenance environment axis does "
                            "not match the rollout"
                        )
                    for env_i, event_records in enumerate(
                            matched_event_records_by_env):
                        for event_record in event_records:
                            event_id = int(event_record["event_id"])
                            participant_slots = tuple(
                                int(slot)
                                for slot in event_record["participant_slots"]
                            )
                            event_contract = (
                                int(event_record["target_id"]),
                                int(t),
                                participant_slots,
                            )
                            prior_contract = (
                                episode_capture_event_contracts[env_i].get(
                                    event_id
                                )
                            )
                            if (
                                    prior_contract is not None
                                    and prior_contract != event_contract):
                                raise RuntimeError(
                                    "capture event_id was reused with "
                                    "conflicting prey, step, or participants "
                                    "inside one environment episode"
                                )
                            episode_capture_event_contracts[env_i][
                                event_id
                            ] = event_contract
                            event_key = (
                                event_id,
                                int(event_record["target_id"]),
                                int(event_record["factor_index"]),
                            )
                            if event_key in episode_active_event_keys[env_i]:
                                raise RuntimeError(
                                    "active capture event factor was "
                                    "duplicated inside one environment episode"
                                )
                            episode_active_event_keys[env_i].add(event_key)
                            static_dynamic_class = (
                                "dynamic"
                                if any(
                                    slot in episode_dynamic_slots[env_i]
                                    for slot in participant_slots
                                )
                                else "static"
                            )
                            contextual_record = dict(event_record)
                            contextual_record.update({
                                "environment_episode_id": int(
                                    environment_episode_ids[env_i]
                                ),
                                "capture_step": int(t),
                                "static_dynamic_class": (
                                    static_dynamic_class
                                ),
                            })
                            episode_active_event_provenance[env_i].append(
                                contextual_record
                            )
                    if topology_events_traj:
                        topology_events_traj[-1][
                            "capture_identity_matched_event_count"
                        ] = int(identity_match["matched_event_count"][0])
                        topology_events_traj[-1][
                            "capture_identity_unmatched_event_count"
                        ] = int(identity_match["unmatched_event_count"][0])
                        topology_events_traj[-1][
                            "capture_identity_candidate_factor_count"
                        ] = int(identity_match["candidate_factor_count"][0])
                        topology_events_traj[-1][
                            "capture_identity_raw_candidate_factor_count"
                        ] = int(identity_match[
                            "raw_candidate_factor_count"
                        ][0])
                        topology_events_traj[-1][
                            "capture_identity_duplicate_factor_count"
                        ] = int(identity_match[
                            "duplicate_candidate_factor_count"
                        ][0])
                        topology_events_traj[-1][
                            "capture_identity_candidate_only_event_count"
                        ] = int(identity_match[
                            "candidate_only_event_count"
                        ][0])
                        topology_events_traj[-1][
                            "capture_identity_participant_count"
                        ] = int(identity_match["participant_count"][0])
                        topology_events_traj[-1][
                            "capture_identity_matched_factor_order_sum"
                        ] = int(identity_match[
                            "matched_factor_order_sum"
                        ][0])
                        topology_events_traj[-1][
                            "capture_identity_matched_event_weight_sum"
                        ] = float(identity_match[
                            "matched_event_weight_sum"
                        ][0])
                        topology_events_traj[-1][
                            "capture_identity_event_mass_error"
                        ] = float(identity_match["event_mass_error"][0])

                # ===== 日志新增：记录每步邻接图结构质量，兼容 DDFG 与 SDDFG/GAT =====
                try:
                    adj_metrics_traj.append(
                        # adj 是 s_t 的图，必须使用事件发生前的 dones_curr。
                        self._calc_adj_metrics(
                            adj,
                            dones_curr,
                            self.num_agents,
                            prob_adj=prob_adj,
                            prev_adj=prev_adj_for_metrics,
                        )
                    )
                    prev_adj_for_metrics = (
                        adj.detach().clone()
                        if torch.is_tensor(adj)
                        else np.array(adj, copy=True)
                    )
                except Exception:
                    pass

                if self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS and not warmup:
                    episode_qtot[p_id][t] = qtot.cpu().detach().numpy()
                    episode_f_q[p_id][t] = f_q.cpu().detach().numpy()
                    if self.use_vfunction:
                        episode_f_v[p_id][t] = f_v.cpu().detach().numpy()

            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            episode_avail_acts[p_id][t] = avail_acts

            # 时间推进
            t += 1
            obs = next_obs
            share_obs = next_share_obs
            avail_acts = next_avail_acts

            # This codebase assumes a single env thread in recurrent runners.
            assert self.num_envs == 1, "Only one env is supported here."

            if terminate_episodes:
                # robustly parse infos: could be dict (per env), or array/list of per-agent dicts
                for i in range(self.num_envs):
                    info0 = None
                    try:
                        info0 = infos[i][0]
                    except Exception:
                        try:
                            info0 = infos[i]
                        except Exception:
                            info0 = None
                    if isinstance(info0, dict) and ('won' in info0):
                        env_info['win_rate'] = 1 if info0['won'] else 0
                break

        # 末帧写入（与 SMAC/PREY 一致）
        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = share_obs
        episode_avail_acts[p_id][t] = avail_acts

        if self.algorithm_name in self.adj_correlation:
            obs_batch = np.concatenate(obs)

            _, rnn_states_batch, _ = policy.get_hidden_states(
                obs_batch, last_acts_batch, rnn_states_batch, dones=dones.reshape(-1, 1)
            )
            rnn_states_batch = self._mask_rnn_states_by_dones(rnn_states_batch, dones)

            if self.use_dyn_graph:
                step_graph_sample_kwargs = dict(graph_sample_kwargs)
                if hasattr(self.adj_network, "use_topology_persistence"):
                    step_graph_sample_kwargs["previous_adj"] = (
                        prev_adj_for_metrics
                    )
                prob_adj, adj, _ = self.adj_network.sample(
                    obs_batch[None, :],
                    rnn_states_batch.unsqueeze(0),
                    self.use_adj_init,
                    dones,
                    **step_graph_sample_kwargs
                )

                prob_adj, adj = self._mask_adj_by_dones(
                    adj=adj,
                    prob_adj=prob_adj,
                    dones=dones,
                    num_agents=self.num_agents,
                    num_factor=self.num_factor
                )
            else:
                adj = self.adj.to(self.device) if torch.is_tensor(self.adj) else torch.as_tensor(self.adj,
                                                                                                 device=self.device)
                if adj.dim() == 2:
                    adj = adj.unsqueeze(0)

                prob_adj = torch.zeros_like(adj, dtype=torch.float32, device=adj.device)

                prob_adj, adj = self._mask_adj_by_dones(
                    adj=adj,
                    prob_adj=prob_adj,
                    dones=dones,
                    num_agents=self.num_agents,
                    num_factor=self.num_factor
                )

            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch,
                                                              np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            env_adj = (
                adj.cpu().detach().numpy()[0]
                if self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS
                else adj.cpu().detach().numpy()
            )
            env_prob_adj = (
                prob_adj.cpu().detach().numpy()[0]
                if self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS
                else prob_adj.cpu().detach().numpy()
            )
            episode_adj[p_id][t] = env_adj
            episode_prob_adj[p_id][t] = env_prob_adj
            episode_rnn_states[p_id][t] = rnn_states_batch

        # 入缓冲区 & 统计
        if explore:
            self.num_episodes_collected += self.num_envs

            idx_range = self.buffer.insert(
                self.num_envs,  # push all episodes collected in this rollout step to the buffer
                episode_obs,
                episode_share_obs,
                episode_acts,
                episode_rewards,
                episode_dones,
                episode_dones_env,
                episode_avail_acts,
                episode_adj,
                episode_prob_adj,
            )
            policy_buffer = self.buffer.policy_buffers[p_id]
            policy_buffer.terminal_win_rewards[:, idx_range] = (
                episode_terminal_win_rewards.copy()
            )
            replay_reward_stats = policy_buffer.reward_normalization_diagnostics()
            env_info["replay_reward_normalization_mean"] = float(
                replay_reward_stats["mean_reward"]
            )
            env_info["replay_reward_normalization_std"] = float(
                replay_reward_stats["std_reward"]
            )
            env_info["terminal_bonus_normalized_delta_at_insert"] = float(
                float(self.args.reward_win)
                / replay_reward_stats["std_reward"]
            )

            if (
                self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS
                and self.total_env_steps >= self.adj_begin_step
                and not warmup
            ):
                if (
                    bool(getattr(
                        self.args,
                        "use_adj_pair_triplet_complementary_credit",
                        False,
                    ))
                    and not capture_event_field_seen
                ):
                    raise RuntimeError(
                        "capture-anchored pair credit requires the Wolfpack "
                        "environment info field 'capture_count'"
                    )
                self.num_adj_episodes_collected += self.num_envs

                idx = self.adj_buffer.insert(self.num_envs,
                                             episode_obs,
                                             episode_share_obs,
                                             episode_acts,
                                             # Graph-return advantages are
                                             # standardized inside AdjBuffer.
                                             # Keep raw rewards here so old
                                             # and new episodes do not use
                                             # different running-normalizer
                                             # scales.
                                             episode_rewards,
                                             episode_dones,
                                             episode_dones_env,
                                             episode_avail_acts,
                                             episode_adj,
                                             episode_prob_adj,
                                             episode_qtot,
                                             episode_f_v,
                                             episode_f_q,
                                             episode_rnn_states,
                                             capture_counts=episode_capture_counts,
                                             success_now=episode_success_now,
                                             capture_factor_matches=(
                                                 episode_capture_factor_matches
                                             ),
                                             capture_candidate_only_matches=(
                                                 episode_capture_candidate_only_matches
                                             ),
                                             capture_candidate_behavior=(
                                                 episode_capture_candidate_behavior
                                             ),
                                             capture_identity_candidates=(
                                                 episode_capture_identity_candidates
                                             ),
                                             capture_candidate_event_provenance=(
                                                 episode_candidate_event_provenance
                                             ),
                                             capture_active_event_provenance=(
                                                 episode_active_event_provenance
                                             ),
                                             environment_episode_ids=(
                                                 environment_episode_ids
                                             ),
                                             behavior_policy_versions=(
                                                 int(getattr(
                                                     self.adj_network,
                                                     "candidate_policy_version",
                                                     0,
                                                 ))
                                             ))
                self.adj_buffer.compute_advantage(idx)

        # ========== 将逐步 infos 轨迹缓存起来（供 eval() 导出）==========
        self.last_episode_step_info = {
            "num_players": num_players_traj,
            "individual_rewards": individual_rewards_traj,
            "team_rewards": team_rewards_traj,
            "active_masks": active_masks_traj,
            "topology_events": topology_events_traj,
            "adj_metrics": adj_metrics_traj,
        }

        # episode 级标量（用于 progress/progress_eval）
        if len(num_players_traj) > 0:
            np_players = np.asarray(num_players_traj, dtype=np.float32)
            env_info["num_players"] = int(np_players[-1])
            env_info["num_players_mean"] = float(np_players.mean())
            env_info["num_players_min"] = float(np_players.min())
            env_info["num_players_max"] = float(np_players.max())
            env_info["num_players_std"] = float(np_players.std())

            if explicit_event_info_seen:
                # 环境显式事件可区分新成员与原成员恢复，也不会被同一步增减抵消。
                env_info["join_events"] = float(join_event_count)
                env_info["leave_events"] = float(left_event_count)
                env_info["recover_events"] = float(recover_event_count)
                env_info["roster_change_events"] = float(topology_change_steps)
                env_info["pending_recovery_final"] = float(pending_recovery_final)
                env_info["recovery_completion_rate"] = (
                    float(recover_event_count) / max(float(left_event_count), 1.0)
                )
            elif np_players.shape[0] > 1:
                # 兼容没有显式事件字段的旧环境。
                diff = np.diff(np_players)
                env_info["join_events"] = float(np.maximum(diff, 0).sum())
                env_info["leave_events"] = float(np.maximum(-diff, 0).sum())
                env_info["roster_change_events"] = float(np.count_nonzero(diff))
                env_info["recover_events"] = 0.0
                env_info["pending_recovery_final"] = 0.0
                env_info["recovery_completion_rate"] = 0.0
            else:
                env_info["join_events"] = 0.0
                env_info["leave_events"] = 0.0
                env_info["recover_events"] = 0.0
                env_info["roster_change_events"] = 0.0
                env_info["pending_recovery_final"] = 0.0
                env_info["recovery_completion_rate"] = 0.0

        if len(active_masks_traj) > 0:
            try:
                am_stack = np.stack([
                    np.asarray(x, dtype=np.float32).reshape(self.num_agents, 1)
                    for x in active_masks_traj
                ], axis=0)
                active_ratio_by_t = am_stack.mean(axis=(1, 2))
                env_info["active_ratio_mean"] = float(active_ratio_by_t.mean())
                env_info["active_ratio_min"] = float(active_ratio_by_t.min())
                env_info["active_ratio_max"] = float(active_ratio_by_t.max())
            except Exception:
                pass

        if len(adj_metrics_traj) > 0:
            try:
                metric_keys = sorted(set().union(*[m.keys() for m in adj_metrics_traj if isinstance(m, dict)]))
                for k in metric_keys:
                    vals = [float(m.get(k, 0.0)) for m in adj_metrics_traj if isinstance(m, dict)]
                    if len(vals) > 0:
                        env_info[k] = float(np.mean(vals))
            except Exception:
                pass

        env_info["episode_real_length"] = int(t)

        if isinstance(last_info_dict, dict):
            if "won" in last_info_dict:
                try:
                    env_info["win_rate"] = 1 if bool(last_info_dict["won"]) else 0
                except Exception:
                    pass

        env_info.setdefault("win_rate", 0)
        env_info["capture_events"] = float(capture_event_count)
        env_info["first_success_step"] = float(first_success_step)
        if (
                float(self.args.reward_win) > 0.0
                and terminal_win_bonus_event_count
                != terminal_win_first_transition_count):
            raise RuntimeError(
                "Wolfpack terminal win bonus count did not match first-success "
                "transition count"
            )
        env_info["terminal_win_first_transition_count"] = float(
            terminal_win_first_transition_count
        )
        env_info["terminal_win_bonus_event_count"] = float(
            terminal_win_bonus_event_count
        )
        env_info["terminal_win_bonus_sum"] = float(terminal_win_bonus_sum)
        env_info["terminal_win_bonus_max"] = float(terminal_win_bonus_max)
        env_info["warmup_terminal_win_first_transition_count"] = float(
            getattr(self, "warmup_terminal_win_first_transition_count", 0)
        )
        env_info["warmup_terminal_win_bonus_event_count"] = float(
            getattr(self, "warmup_terminal_win_bonus_event_count", 0)
        )
        env_info["warmup_terminal_win_bonus_sum"] = float(
            getattr(self, "warmup_terminal_win_bonus_sum", 0.0)
        )
        env_info["partial_capture_without_win"] = float(
            capture_event_count > 0 and env_info["win_rate"] < 0.5
        )
        env_info.setdefault("num_players", 0)
        env_info.setdefault("num_players_mean", float(env_info["num_players"]))
        env_info.setdefault("num_players_min", float(env_info["num_players"]))
        env_info.setdefault("num_players_max", float(env_info["num_players"]))
        env_info.setdefault("num_players_std", 0.0)
        env_info.setdefault("join_events", 0.0)
        env_info.setdefault("leave_events", 0.0)
        env_info.setdefault("recover_events", 0.0)
        env_info.setdefault("roster_change_events", 0.0)
        env_info.setdefault("pending_recovery_final", 0.0)
        env_info.setdefault("recovery_completion_rate", 0.0)
        env_info["activation_state_resets"] = float(activation_state_reset_count)
        env_info.setdefault("active_ratio_mean", 0.0)
        env_info.setdefault("active_ratio_min", 0.0)
        env_info.setdefault("active_ratio_max", 0.0)
        env_info.setdefault("episode_real_length", int(t))
        env_info["graph_explore_enabled"] = float(graph_explore)
        env_info["graph_eval_rng_enabled"] = float(graph_use_eval_rng)
        env_info["topology_persistence_enabled"] = float(
            bool(
                hasattr(self, "adj_network")
                and getattr(
                    self.adj_network,
                    "use_topology_persistence",
                    False,
                )
            )
        )

        env_info['average_episode_rewards'] = np.sum(episode_rewards[p_id][:, 0, 0, 0])
        if (
                training_episode
                and (not warmup)
                and bool(getattr(
                    policy,
                    "use_joint_epsilon_exploration",
                    False,
                ))):
            if len(joint_exploration_diagnostics) != int(t):
                raise RuntimeError(
                    "joint epsilon diagnostic count did not match formal "
                    "training episode length"
                )
            self._dump_joint_exploration_episode_diagnostic(
                train_step=int(self.total_env_steps),
                episode_index=int(self.num_episodes_collected),
                diagnostics=joint_exploration_diagnostics,
                env_info=env_info,
            )
        # ===== 日志新增：训练阶段也低频导出逐步轨迹 CSV =====
        # eval 阶段已经由 eval() 调用 _dump_eval_episode_step_tables；
        # 这里仅对训练 episode 按 log_interval 导出，避免文件过大。
        if training_episode and (not warmup):
            self._dump_train_pre_capture_rows(pre_capture_rows)
            self._dump_train_pre_capture_prefix_rows(
                pre_capture_prefix_rows
            )
            self._dump_train_post_capture_rows(post_capture_rows)
            try:
                should_dump_train_traj = (
                    self.total_env_steps > 0 and
                    ((self.total_env_steps - getattr(self, "last_log_T", 0)) / max(float(self.log_interval), 1.0)) >= 1
                )
                if should_dump_train_traj and isinstance(self.last_episode_step_info, dict):
                    self._dump_train_episode_step_tables(
                        train_step=int(self.total_env_steps),
                        step_info=self.last_episode_step_info
                    )
            except Exception as e:
                print(f"[log] dump train step tables failed: {repr(e)}", flush=True)

        # 关闭视频写入器，确保 mp4 文件落盘。
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass

        return env_info



    def log(self):
        """See parent class."""
        end = time.time()
        print(
            "\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n".format(
                self.env_name,
                self.algorithm_name,
                self.args.experiment_name,
                self.total_env_steps,
                self.num_env_steps,
                int(self.total_env_steps / (end - self.start)),
            )
        )

        # Aggregate all mini-updates produced by the latest training trigger.
        raw_train_infos = getattr(self, "train_infos", None)
        if not isinstance(raw_train_infos, (list, tuple)):
            raw_train_infos = []

        # DDFG/SDDFG performs train_interval_episode updates per trigger, so
        # train_infos contains K * num_policies dictionaries. Aggregate by
        # policy instead of discarding the metrics when K > 1.
        aggregated_train_infos = []
        num_policies = len(self.policy_ids)
        for policy_index in range(num_policies):
            policy_updates = [
                info for info in raw_train_infos[policy_index::num_policies]
                if isinstance(info, dict) and info
            ]
            keys = sorted(set().union(*(info.keys() for info in policy_updates))) \
                if policy_updates else []
            aggregated = {}
            for key in keys:
                values = [float(info[key]) for info in policy_updates if key in info]
                if values:
                    if not np.all(np.isfinite(values)):
                        raise FloatingPointError(
                            f"non-finite policy training metric {key}: {values}"
                        )
                    aggregated[key] = float(np.mean(values))
            aggregated["updates_aggregated"] = float(len(policy_updates))
            aggregated["total_train_steps"] = float(self.total_train_steps)
            aggregated_train_infos.append(aggregated)

        self.train_infos = aggregated_train_infos
        for p_id, train_info in zip(self.policy_ids, aggregated_train_infos):
            self.log_train(p_id, train_info)

        # ===== 日志新增：记录动作探索 epsilon 与邻接探索 adj_epsilon =====
        schedule_info = {}
        try:
            p0 = self.policies[self.policy_ids[0]]
            if hasattr(p0, "exploration"):
                schedule_info["epsilon"] = float(p0.exploration.eval(self.total_env_steps))
                schedule_info["joint_epsilon_exploration_enabled"] = float(
                    bool(getattr(
                        p0,
                        "use_joint_epsilon_exploration",
                        False,
                    ))
                )
        except Exception:
            pass

        try:
            if hasattr(self.adj_network, "exploration_mix"):
                schedule_info["adj_epsilon"] = float(
                    self.adj_network.exploration_mix
                )
            elif hasattr(self.adj_network, "exploration"):
                schedule_info["adj_epsilon"] = float(self.adj_network.exploration.eval(self.total_env_steps))
        except Exception:
            pass

        try:
            if hasattr(self.adj_network, "current_sampling_temperature"):
                schedule_info["adj_sampling_temperature"] = float(
                    self.adj_network.current_sampling_temperature
                )
            if hasattr(self.adj_network, "current_order3_bonus"):
                schedule_info["adj_order3_bonus_current"] = float(
                    self.adj_network.current_order3_bonus
                )
            if hasattr(self.adj_network, "current_min_order3_ratio"):
                schedule_info["adj_min_order3_ratio_current"] = float(
                    self.adj_network.current_min_order3_ratio
                )
            if hasattr(self.adj_network, "current_max_order3_ratio"):
                schedule_info["adj_max_order3_ratio_current"] = float(
                    self.adj_network.current_max_order3_ratio
                )
            if hasattr(self.adj_network, "current_greedy_sample_prob"):
                schedule_info["adj_greedy_sample_prob_current"] = float(
                    self.adj_network.current_greedy_sample_prob
                )
            if hasattr(self.adj_network, "current_order3_credit_gate"):
                schedule_info["adj_order3_credit_gate_current"] = float(
                    self.adj_network.current_order3_credit_gate
                )
            if hasattr(self.adj_network, "order3_credit_loss_ema"):
                schedule_info["adj_order3_credit_loss_ema"] = float(
                    self.adj_network.order3_credit_loss_ema
                )
            if hasattr(self.adj_network, "order3_credit_margin_ema"):
                schedule_info["adj_order3_credit_margin_ema"] = float(
                    self.adj_network.order3_credit_margin_ema
                )
            if hasattr(self.adj_network, "use_relative_order3_credit_gate"):
                schedule_info["adj_order3_relative_credit_gate"] = float(
                    bool(self.adj_network.use_relative_order3_credit_gate)
                )
            if hasattr(self.adj_network, "greedy_sample_prob_cap"):
                schedule_info["adj_greedy_sample_prob_cap"] = float(
                    self.adj_network.greedy_sample_prob_cap
                )
            if hasattr(self.adj_network, "order3_credit_gate_max_delta"):
                schedule_info["adj_order3_credit_gate_max_delta"] = float(
                    self.adj_network.order3_credit_gate_max_delta
                )
            if hasattr(self.adj_network, "order3_soft_quota_coef"):
                schedule_info["adj_order3_soft_quota_coef"] = float(
                    self.adj_network.order3_soft_quota_coef
                )
            if hasattr(self.adj_network, "triplet_balance_coef"):
                schedule_info["adj_triplet_balance_coef"] = float(
                    self.adj_network.triplet_balance_coef
                )
            if hasattr(self.adj_network, "use_topology_persistence"):
                schedule_info["use_adj_topology_persistence"] = float(
                    bool(self.adj_network.use_topology_persistence)
                )
            if hasattr(
                self.adj_network,
                "last_topology_persistence_candidate_fraction",
            ):
                schedule_info[
                    "adj_topology_persistence_candidate_fraction"
                ] = float(
                    self.adj_network.last_topology_persistence_candidate_fraction
                )
            if hasattr(
                self.adj_network,
                "last_topology_persistence_selected_fraction",
            ):
                schedule_info[
                    "adj_topology_persistence_selected_fraction"
                ] = float(
                    self.adj_network.last_topology_persistence_selected_fraction
                )
            for metric_key in (
                "raw_order2_factor_rl_loss",
                "raw_order3_factor_rl_loss",
                "raw_o3_minus_o2_factor_rl_loss",
                "credit_order2_factor_advantage_abs_mean",
                "credit_order3_factor_advantage_abs_mean",
                "triplet_graph_return_credit_mean",
                "triplet_graph_return_credit_active_fraction",
                "triplet_graph_return_credit_gate_mean",
                "delayed_triplet_credit_mean",
                "delayed_triplet_credit_active_fraction",
                "delayed_triplet_credit_positive_fraction",
                "delayed_triplet_credit_negative_fraction",
                "delayed_triplet_success_gate_mean",
                "delayed_triplet_success_gate_active_fraction",
                "delayed_triplet_success_gate_selective_mean",
                "delayed_triplet_success_gate_selective_fraction",
                "triplet_graph_return_success_gate_mean",
                "triplet_graph_return_success_gate_active_fraction",
                "delayed_triplet_future_match_weight_mean",
                "delayed_triplet_future_matched_fraction",
                "delayed_triplet_future_exact_fraction",
                "delayed_triplet_future_partial_fraction",
                "capture_to_win_triplet_credit_mean",
                "capture_to_win_triplet_credit_abs_mean",
                "capture_to_win_triplet_credit_active_fraction",
                "capture_to_win_triplet_credit_positive_fraction",
                "capture_to_win_triplet_credit_negative_fraction",
                "capture_to_win_triplet_credit_positive_mass",
                "capture_to_win_triplet_credit_negative_mass",
                "capture_outcome_local_delta_positive_mass",
                "capture_outcome_local_delta_negative_mass",
                "capture_outcome_local_delta_active_fraction",
                "capture_outcome_factor_loss_contribution",
                "capture_outcome_positive_factor_loss_contribution",
                "capture_outcome_negative_factor_loss_contribution",
                "capture_identity_factor_credit_active_fraction",
                "capture_identity_factor_credit_positive_mass",
                "capture_identity_factor_credit_negative_mass",
                "capture_outcome_identity_order2_delta_abs_mass",
                "capture_outcome_identity_order3_delta_abs_mass",
                "capture_identity_target_order2_fraction",
                "capture_identity_target_order3_fraction",
                "capture_outcome_non_target_delta_abs_max",
                "capture_to_win_quality_gate_mean",
                "capture_to_win_quality_gate_abs_mean",
                "capture_to_win_quality_gate_active_fraction",
                "capture_to_win_quality_gate_positive_fraction",
                "capture_to_win_quality_gate_negative_fraction",
                "capture_to_win_quality_gate_positive_mass",
                "capture_to_win_quality_gate_negative_mass",
                "capture_to_win_outcome_contrastive",
                "capture_to_win_quality_gate_definition_version",
                "capture_outcome_baseline_mean",
                "capture_outcome_capture_episode_count_mean",
                "capture_outcome_success_episode_count_mean",
                "capture_outcome_failure_episode_count_mean",
                "capture_outcome_mixed_window_fraction",
                "capture_outcome_single_success_window_fraction",
                "capture_outcome_single_failure_window_fraction",
                "capture_outcome_no_capture_window_fraction",
                "capture_outcome_triplet_labels_per_episode_mean",
                "capture_outcome_triplet_labels_per_episode_max",
                "capture_outcome_raw_episode_advantage_mean",
                "capture_outcome_raw_episode_advantage_abs_mean",
                "capture_outcome_window_raw_centered_mean",
                "capture_outcome_window_expanded_gate_sum",
                "capture_outcome_window_expanded_gate_abs_sum",
                "capture_outcome_window_center_error_ratio",
                "capture_to_win_credit_preclip_mean",
                "capture_to_win_credit_preclip_std",
                "capture_to_win_credit_preclip_max",
                "capture_to_win_credit_preclip_min",
                "capture_to_win_credit_positive_clip_fraction",
                "capture_to_win_credit_negative_clip_fraction",
                "capture_identity_event_count_mean",
                "capture_identity_matched_event_count_mean",
                "capture_identity_unmatched_event_count_mean",
                "capture_identity_candidate_factor_count_mean",
                "capture_identity_match_fraction",
                "capture_identity_candidates_per_matched_event",
                "capture_outcome_label_gate_correlation",
                "capture_outcome_label_gate_correlation_valid",
                "capture_outcome_success_labels_mean",
                "capture_outcome_failure_labels_mean",
                "capture_outcome_success_gate_total_mean",
                "capture_outcome_failure_gate_total_mean",
                "pair_pursuit_credit_mean",
                "pair_pursuit_credit_active_fraction",
                "pair_credit_active_fraction",
                "pair_pursuit_credit_std",
                "pair_pursuit_credit_max",
                "pair_pursuit_credit_nonzero_count",
                "pair_credit_top1_mass_fraction",
                "pair_pursuit_quality_mean",
                "pair_pursuit_quality_active_fraction",
                "pair_to_triplet_transition_score_mean",
                "pair_to_triplet_transition_active_fraction",
                "triplet_capture_quality_mean",
                "triplet_capture_quality_active_fraction",
                "transition_delay_mean",
                "transition_delay_min",
                "transition_delay_max",
                "capture_event_count",
                "capture_matched_count",
                "unmatched_capture_count",
                "capture_matched_fraction",
                "unmatched_capture_fraction",
                "failed_episode_capture_count",
                "failed_episode_capture_fraction",
                "capture_to_win_capture_success_fraction",
                "capture_to_win_episode_success_fraction",
                "positive_reward_without_capture_fraction",
                "positive_reward_step_count",
                "positive_reward_without_capture_count",
                "positive_reward_step_fraction",
                "offset0_candidate_count",
                "offset0_candidate_fraction",
                "graph_return_credit_strength_mean",
                "credit_order2_factor_rl_loss",
                "credit_order3_factor_rl_loss",
                "credit_o3_minus_o2_factor_rl_loss",
                "order2_positive_adv_fraction",
                "order3_positive_adv_fraction",
                "credit_order2_positive_adv_fraction",
                "credit_order3_positive_adv_fraction",
                "order2_promoted_adv_fraction",
                "order3_promoted_adv_fraction",
                "adj_order_adv_positive_only",
                "adj_order_adv_negative_coef",
                "adj_order_adv_require_positive_graph_adv",
                "use_adj_triplet_graph_return_credit",
                "adj_triplet_graph_return_credit_coef",
                "adj_triplet_graph_return_credit_cap",
                "adj_triplet_graph_return_credit_min_graph_adv",
                "adj_triplet_graph_return_credit_raw_gate_scale",
                "adj_triplet_graph_return_credit_require_delayed_gate",
                "use_adj_delayed_triplet_credit",
                "adj_delayed_triplet_credit_coef",
                "adj_delayed_triplet_credit_window",
                "adj_delayed_triplet_credit_cap",
                "adj_delayed_triplet_credit_min_reward",
                "adj_delayed_triplet_credit_positive_only",
                "adj_delayed_triplet_credit_min_adv",
                "adj_delayed_triplet_credit_require_future_match",
                "use_adj_delayed_triplet_success_gate",
                "adj_delayed_triplet_success_gate_min_adv",
                "adj_delayed_triplet_success_gate_scale",
                "adj_delayed_triplet_success_gate_floor",
                "adj_delayed_triplet_future_overlap_min_nodes",
                "adj_delayed_triplet_partial_match_weight",
                "use_adj_capture_to_win_credit",
                "adj_capture_to_win_credit_coef",
                "adj_capture_to_win_credit_min_outcome_adv",
                "adj_capture_to_win_credit_scale",
                "adj_capture_to_win_credit_cap",
                "adj_capture_to_win_credit_require_future_match",
                "use_adj_pair_triplet_complementary_credit",
                "adj_pair_pursuit_credit_coef",
                "adj_pair_pursuit_credit_window",
                "adj_pair_pursuit_credit_cap",
                "adj_pair_pursuit_credit_min_reward",
                "use_adj_advantage_triplet_scorer",
                "adv_triplet_credit_pair_updates",
                "adv_triplet_credit_triplet_updates",
                "adv_triplet_credit_seen_ratio",
                "adv_pair_credit_seen_ratio",
                "adv_triplet_credit_mean",
                "adv_triplet_marginal_mean",
                "adv_triplet_marginal_positive_fraction",
                "use_adj_triplet_credit_direct_rank",
                "adj_triplet_credit_rank_coef",
                "adj_triplet_credit_min_multiplier",
                "adj_triplet_credit_max_multiplier",
                "adj_triplet_credit_negative_rank_scale",
                "adj_triplet_credit_min_positive_fraction",
                "adv_triplet_score_multiplier_mean",
                "adv_triplet_score_multiplier_min",
                "adv_triplet_score_multiplier_max",
                "adv_triplet_score_marginal_mean",
                "adv_triplet_score_positive_fraction",
                "adv_triplet_negative_scaled_fraction",
                "adj_ppo_epochs_ran",
                "adj_ppo_early_stop_triggered",
                "adj_ppo_last_epoch_clip_ratio",
                "adj_ppo_last_epoch_factor_clip_ratio",
                "adj_ppo_last_epoch_control_clip_ratio",
                "adj_ppo_last_epoch_control_factor_clip_ratio",
                "adj_ppo_control_uses_trusted_population",
                "adj_ppo_clip_stop_ratio",
                "adj_ppo_factor_clip_stop_ratio",
                "adj_ppo_min_epochs",
                "trusted_clamp_ratio",
                "trusted_factor_clamp_ratio",
                "adj_graph_trust_weight_mean",
                "adj_factor_trust_weight_mean",
                "adj_graph_stale_ratio",
                "adj_factor_stale_ratio",
                "use_adj_ppo_stale_trust",
                "adj_ppo_stale_trust_clip",
                "adj_ppo_stale_trust_scale",
                "adj_ppo_stale_trust_min_weight",
                "adj_recent_episode_window",
                "adj_recent_episode_window_config",
                "adj_dynamic_recent_window_enabled",
                "adj_recent_window_shrunk",
                "adj_recent_window_graph_control_ratio",
                "adj_recent_window_factor_control_ratio",
                "adj_recent_window_control_uses_trusted_population",
                "adj_recent_window_recovered",
                "adj_recent_window_emergency_shrunk",
                "adj_recent_window_high_stale_count",
                "adj_recent_window_low_stale_count",
                "adj_sample_episode_count",
                "adj_sample_recent_fraction",
                "adj_outcome_contrast_replay_support_version",
                "adj_sample_base_episode_count",
                "adj_sample_outcome_contrast_augmented_count",
                "adj_sample_outcome_positive_available",
                "adj_sample_outcome_negative_available",
                "adj_sample_outcome_positive_episode_count",
                "adj_sample_outcome_negative_episode_count",
                "adj_sample_outcome_class_complete",
                "adj_sample_outcome_support_exhausted",
                "adj_sample_outcome_credit_enabled",
                "adj_sample_outcome_cached_selection_reused",
                "adj_sample_outcome_support_round",
                "adj_sample_outcome_cross_update_reuse_count",
                "adj_sample_outcome_positive_available_count",
                "adj_sample_outcome_negative_available_count",
                "adj_sample_outcome_base_positive_count",
                "adj_sample_outcome_base_negative_count",
                "adj_sample_outcome_augmented_positive_count",
                "adj_sample_outcome_augmented_negative_count",
                "adj_sample_outcome_base_age_mean",
                "adj_sample_outcome_base_age_max",
                "adj_sample_outcome_augmented_age_mean",
                "adj_sample_outcome_augmented_age_max",
                "adj_sample_outcome_positive_support_generation",
                "adj_sample_outcome_negative_support_generation",
                "adj_sample_outcome_positive_support_age",
                "adj_sample_outcome_negative_support_age",
                "adj_sample_outcome_support_used_count",
                "adj_sample_outcome_support_used_fraction",
                "adj_sample_outcome_full_buffer_baseline",
                "adj_sample_outcome_base_cohort_baseline",
                "adj_sample_outcome_trained_cohort_baseline",
                "adj_sample_outcome_full_trained_baseline_gap",
                "adj_sample_outcome_trained_capture_episode_count",
                "adj_sample_outcome_cohort_centered_sum",
                "adj_sample_outcome_cohort_center_error",
                "adj_sample_outcome_cohort_center_valid",
                "adj_sample_outcome_positive_gate_episode_count",
                "adj_sample_outcome_negative_gate_episode_count",
                "adj_sample_outcome_positive_credit_episode_count",
                "adj_sample_outcome_negative_credit_episode_count",
                "adj_sample_outcome_signed_scaling_version",
                "adj_sample_outcome_graph_advantage_source_ready_fraction",
                "adj_sample_outcome_graph_confidence_mean",
                "adj_sample_outcome_graph_confidence_std",
                "adj_sample_outcome_graph_confidence_p50",
                "adj_sample_outcome_graph_confidence_p95",
                "adj_sample_outcome_graph_confidence_max",
                "adj_sample_outcome_positive_graph_confidence_mean",
                "adj_sample_outcome_positive_graph_confidence_max",
                "adj_sample_outcome_negative_graph_confidence_mean",
                "adj_sample_outcome_negative_graph_confidence_max",
                "adj_sample_outcome_graph_advantage_positive_fraction",
                "adj_sample_outcome_graph_advantage_negative_fraction",
                "adj_sample_outcome_graph_advantage_zero_fraction",
                "adj_sample_outcome_positive_zero_confidence_fraction",
                "adj_sample_outcome_negative_zero_confidence_fraction",
                "adj_sample_outcome_gate_to_credit_drop_fraction",
                "adj_sample_outcome_preclip_positive_mass",
                "adj_sample_outcome_preclip_negative_mass",
                "adj_sample_outcome_postclip_positive_mass",
                "adj_sample_outcome_postclip_negative_mass",
                "adj_sample_outcome_positive_clip_fraction",
                "adj_sample_outcome_negative_clip_fraction",
                "adj_sample_outcome_generation_update_count",
                "adj_sample_outcome_slot_overwrite_count",
                "adj_sample_outcome_generation_conflict_count",
                "adj_sample_outcome_invalid_used_state_count",
                "adj_recent_episode_window_emergency",
                "adj_recent_window_emergency_stale_threshold",
                "adj_recent_window_emergency_factor_stale_threshold",
            ):
                metric_value = self._latest_train_metric(
                    metric_key,
                    np.nan,
                )
                if np.isfinite(metric_value):
                    schedule_info[metric_key] = float(metric_value)
        except Exception:
            pass

        for k, v in schedule_info.items():
            tag = "schedule/" + k
            if self.use_wandb:
                # 不在 wolfpack_runner.py 里直接 import wandb；
                # 通过 log_env 写入 env_infos 更稳。这里仅写入 TensorBoard 时可用。
                pass
            else:
                self.writter.add_scalar(tag, v, self.total_env_steps)

        # 同时放进 env_infos，让 progress.csv 也能记录 epsilon/adj_epsilon
        for k, v in schedule_info.items():
            if k not in self.env_infos:
                self.env_infos[k] = []
            self.env_infos[k].append(v)

        env_infos_for_log = self._prepare_episode_metrics_for_logging(
            self.env_infos
        )
        self.log_env(env_infos_for_log)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {
            "win_rate": [],
            "average_episode_rewards": [],
            "capture_events": [],
            "first_success_step": [],
            "partial_capture_without_win": [],
            "terminal_win_first_transition_count": [],
            "terminal_win_bonus_event_count": [],
            "terminal_win_bonus_sum": [],
            "terminal_win_bonus_max": [],
            "warmup_terminal_win_first_transition_count": [],
            "warmup_terminal_win_bonus_event_count": [],
            "warmup_terminal_win_bonus_sum": [],
            "replay_reward_normalization_mean": [],
            "replay_reward_normalization_std": [],
            "terminal_bonus_normalized_delta_at_insert": [],

            # episode 末尾人数
            "num_players": [],

            # episode 内动态人数统计
            "num_players_mean": [],
            "num_players_min": [],
            "num_players_max": [],
            "num_players_std": [],
            "join_events": [],
            "leave_events": [],
            "recover_events": [],
            "roster_change_events": [],
            "pending_recovery_final": [],
            "recovery_completion_rate": [],
            "activation_state_resets": [],

            # active mask 覆盖率
            "active_ratio_mean": [],
            "active_ratio_min": [],
            "active_ratio_max": [],

            # episode 长度
            "episode_real_length": [],

            # 邻接图结构质量
            "adj_valid_factor_ratio": [],
            "adj_empty_factor_ratio": [],
            "adj_order1_ratio": [],
            "adj_order2_ratio": [],
            "adj_order3_ratio": [],
            "adj_invalid_factor_ratio": [],
            "adj_mean_order": [],

            # 探索强度
            "adj_active_agent_coverage": [],
            "adj_uncovered_active_agent_ratio": [],
            "adj_active_agent_degree": [],
            "adj_connected_graph_ratio": [],
            "adj_selected_prob_mean": [],
            "adj_selected_prob_min": [],
            "adj_factor_retention_ratio": [],
            "graph_explore_enabled": [],
            "graph_eval_rng_enabled": [],
            "topology_persistence_enabled": [],
            "epsilon": [],
            "adj_epsilon": [],
            "adj_sampling_temperature": [],
            "adj_order3_bonus_current": [],
            "adj_min_order3_ratio_current": [],
            "adj_max_order3_ratio_current": [],
            "adj_greedy_sample_prob_current": [],
            "adj_order3_credit_gate_current": [],
            "adj_order3_credit_loss_ema": [],
            "adj_order3_credit_margin_ema": [],
            "adj_order3_relative_credit_gate": [],
            "adj_greedy_sample_prob_cap": [],
            "adj_order3_credit_gate_max_delta": [],
            "adj_order3_soft_quota_coef": [],
            "adj_triplet_balance_coef": [],
            "use_adj_topology_persistence": [],
            "adj_topology_persistence_candidate_fraction": [],
            "adj_topology_persistence_selected_fraction": [],
            "raw_order2_factor_rl_loss": [],
            "raw_order3_factor_rl_loss": [],
            "raw_o3_minus_o2_factor_rl_loss": [],
            "credit_order2_factor_advantage_abs_mean": [],
            "credit_order3_factor_advantage_abs_mean": [],
            "triplet_graph_return_credit_mean": [],
            "triplet_graph_return_credit_active_fraction": [],
            "triplet_graph_return_credit_gate_mean": [],
            "delayed_triplet_credit_mean": [],
            "delayed_triplet_credit_active_fraction": [],
            "delayed_triplet_credit_positive_fraction": [],
            "delayed_triplet_credit_negative_fraction": [],
            "delayed_triplet_success_gate_mean": [],
            "delayed_triplet_success_gate_active_fraction": [],
            "delayed_triplet_success_gate_selective_mean": [],
            "delayed_triplet_success_gate_selective_fraction": [],
            "triplet_graph_return_success_gate_mean": [],
            "triplet_graph_return_success_gate_active_fraction": [],
            "delayed_triplet_future_match_weight_mean": [],
            "delayed_triplet_future_matched_fraction": [],
            "delayed_triplet_future_exact_fraction": [],
            "delayed_triplet_future_partial_fraction": [],
            "capture_to_win_triplet_credit_mean": [],
            "capture_to_win_triplet_credit_abs_mean": [],
            "capture_to_win_triplet_credit_active_fraction": [],
            "capture_to_win_triplet_credit_positive_fraction": [],
            "capture_to_win_triplet_credit_negative_fraction": [],
            "capture_to_win_triplet_credit_positive_mass": [],
            "capture_to_win_triplet_credit_negative_mass": [],
            "capture_outcome_local_delta_positive_mass": [],
            "capture_outcome_local_delta_negative_mass": [],
            "capture_outcome_local_delta_active_fraction": [],
            "capture_outcome_factor_loss_contribution": [],
            "capture_outcome_positive_factor_loss_contribution": [],
            "capture_outcome_negative_factor_loss_contribution": [],
            "capture_outcome_factor_loss_target_count": [],
            "capture_outcome_factor_loss_valid_transition_count": [],
            "capture_outcome_factor_loss_factors_per_transition": [],
            "capture_outcome_factor_loss_normalization_version": [],
            "capture_candidate_identity_loss_contribution": [],
            "capture_candidate_identity_positive_loss_contribution": [],
            "capture_candidate_identity_negative_loss_contribution": [],
            "capture_candidate_identity_target_count": [],
            "capture_candidate_identity_valid_transition_count": [],
            "capture_candidate_identity_target_transition_count": [],
            "capture_candidate_identity_unsatisfied_target_count": [],
            "capture_candidate_identity_positive_unsatisfied_target_count": [],
            "capture_candidate_identity_negative_unsatisfied_target_count": [],
            "capture_candidate_identity_target_transition_fraction": [],
            "capture_candidate_identity_positive_mass": [],
            "capture_candidate_identity_negative_mass": [],
            "capture_candidate_identity_positive_margin_mean": [],
            "capture_candidate_identity_negative_margin_mean": [],
            "capture_candidate_identity_positive_signed_margin_mean": [],
            "capture_candidate_identity_negative_signed_margin_mean": [],
            "capture_candidate_identity_loss_definition_version": [],
            "capture_candidate_identity_loss_normalization_version": [],
            "capture_candidate_identity_gradient_projection_version": [],
            "capture_candidate_identity_actual_update_guard_version": [],
            "capture_candidate_identity_optimizer_state_sync_version": [],
            "capture_candidate_identity_gradient_norm": [],
            "capture_candidate_identity_base_gradient_norm": [],
            "capture_candidate_identity_base_gradient_cosine": [],
            "capture_candidate_identity_gradient_conflict": [],
            "capture_candidate_identity_projected_gradient_dot": [],
            "capture_candidate_identity_total_gradient_norm_ratio": [],
            "capture_candidate_identity_base_gradient_removed_norm_fraction": [],
            "capture_candidate_identity_clipped_gradient_dot": [],
            "capture_candidate_identity_actual_update_descent_dot_before": [],
            "capture_candidate_identity_actual_update_descent_dot_after": [],
            "capture_candidate_identity_actual_update_corrected": [],
            "capture_candidate_identity_actual_update_norm": [],
            "capture_candidate_identity_actual_update_correction_norm": [],
            "capture_candidate_identity_actual_update_correction_norm_ratio": [],
            "capture_candidate_identity_optimizer_state_sync_applied": [],
            "capture_candidate_identity_optimizer_state_sync_parameter_count": [],
            "capture_candidate_identity_optimizer_state_update_equation_version": [],
            "capture_candidate_identity_optimizer_state_raw_reconstruction_error": [],
            "capture_candidate_identity_optimizer_state_safe_reconstruction_error": [],
            "capture_candidate_identity_optimizer_state_raw_reconstruction_error_ratio": [],
            "capture_candidate_identity_optimizer_state_safe_reconstruction_error_ratio": [],
            "capture_candidate_identity_optimizer_state_reconstruction_tolerance": [],
            "capture_candidate_identity_optimizer_state_exp_avg_change_norm": [],
            "capture_candidate_identity_loss_optimizer_change": [],
            "capture_candidate_identity_lifecycle_version": [],
            "capture_candidate_identity_lifecycle_horizon": [],
            "capture_candidate_identity_lifecycle_cache_size": [],
            "capture_candidate_identity_lifecycle_observation_archive_size": [],
            "capture_candidate_identity_lifecycle_new_count": [],
            "capture_candidate_identity_lifecycle_behavioral_progress_transition_count": [],
            "capture_candidate_identity_lifecycle_no_progress_skipped_transition_count": [],
            "capture_candidate_identity_lifecycle_duplicate_prevented_count": [],
            "capture_candidate_identity_lifecycle_expired_count": [],
            "capture_candidate_identity_lifecycle_protected_target_count": [],
            "capture_candidate_identity_lifecycle_gradient_norm": [],
            "capture_candidate_identity_lifecycle_base_gradient_cosine": [],
            "capture_candidate_identity_lifecycle_gradient_conflict": [],
            "capture_candidate_identity_lifecycle_projected_gradient_dot": [],
            "capture_candidate_identity_lifecycle_actual_update_descent_dot_before": [],
            "capture_candidate_identity_lifecycle_actual_update_descent_dot_after": [],
            "capture_candidate_identity_lifecycle_actual_update_corrected": [],
            "capture_candidate_identity_lifecycle_update_rejected": [],
            "capture_candidate_identity_lifecycle_violation_count": [],
            "capture_candidate_identity_lifecycle_attempted_loss_optimizer_change": [],
            "capture_candidate_identity_lifecycle_loss_optimizer_change": [],
            "capture_candidate_identity_lifecycle_state_sync_applied": [],
            "capture_candidate_identity_lifecycle_age_mean": [],
            "capture_candidate_identity_lifecycle_constraint_count": [],
            "capture_candidate_identity_lifecycle_active_constraint_count": [],
            "capture_candidate_identity_lifecycle_min_constraint_dot_before": [],
            "capture_candidate_identity_lifecycle_min_constraint_dot_after": [],
            "capture_candidate_identity_lifecycle_projection_fallback": [],
            "capture_candidate_identity_lifecycle_superseded_constraint_count": [],
            "capture_candidate_identity_lifecycle_actual_min_constraint_dot_before": [],
            "capture_candidate_identity_lifecycle_actual_min_constraint_dot_after": [],
            "capture_candidate_identity_lifecycle_actual_negative_constraint_count_before": [],
            "capture_candidate_identity_lifecycle_actual_negative_constraint_count_after": [],
            "capture_candidate_identity_lifecycle_actual_projection_corrected": [],
            "capture_candidate_identity_lifecycle_actual_projection_correction_norm_ratio": [],
            "capture_candidate_identity_lifecycle_nonlinear_backtrack_count": [],
            "capture_candidate_identity_lifecycle_current_candidate_nonlinear_violation": [],
            "capture_candidate_identity_lifecycle_target_bearing_update": [],
            "capture_candidate_identity_lifecycle_policy_version_advanced": [],
            "capture_candidate_identity_lifecycle_clock": [],
            "capture_candidate_identity_lifecycle_1_update_retention_count": [],
            "capture_candidate_identity_lifecycle_1_update_signed_margin_retention_fraction": [],
            "capture_candidate_identity_lifecycle_1_update_rank_retention_fraction": [],
            "capture_candidate_identity_lifecycle_1_update_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_1_update_rank_held_count": [],
            "capture_candidate_identity_lifecycle_1_update_positive_eligible_count": [],
            "capture_candidate_identity_lifecycle_1_update_positive_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_1_update_positive_rank_held_count": [],
            "capture_candidate_identity_lifecycle_1_update_negative_eligible_count": [],
            "capture_candidate_identity_lifecycle_1_update_negative_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_1_update_negative_rank_held_count": [],
            "capture_candidate_identity_lifecycle_5_update_retention_count": [],
            "capture_candidate_identity_lifecycle_5_update_signed_margin_retention_fraction": [],
            "capture_candidate_identity_lifecycle_5_update_rank_retention_fraction": [],
            "capture_candidate_identity_lifecycle_5_update_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_5_update_rank_held_count": [],
            "capture_candidate_identity_lifecycle_5_update_positive_eligible_count": [],
            "capture_candidate_identity_lifecycle_5_update_positive_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_5_update_positive_rank_held_count": [],
            "capture_candidate_identity_lifecycle_5_update_negative_eligible_count": [],
            "capture_candidate_identity_lifecycle_5_update_negative_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_5_update_negative_rank_held_count": [],
            "capture_candidate_identity_lifecycle_10_update_retention_count": [],
            "capture_candidate_identity_lifecycle_10_update_signed_margin_retention_fraction": [],
            "capture_candidate_identity_lifecycle_10_update_rank_retention_fraction": [],
            "capture_candidate_identity_lifecycle_10_update_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_10_update_rank_held_count": [],
            "capture_candidate_identity_lifecycle_10_update_positive_eligible_count": [],
            "capture_candidate_identity_lifecycle_10_update_positive_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_10_update_positive_rank_held_count": [],
            "capture_candidate_identity_lifecycle_10_update_negative_eligible_count": [],
            "capture_candidate_identity_lifecycle_10_update_negative_signed_margin_held_count": [],
            "capture_candidate_identity_lifecycle_10_update_negative_rank_held_count": [],
            "capture_candidate_identity_positive_optimizer_signed_margin_change_mean": [],
            "capture_candidate_identity_negative_optimizer_signed_margin_change_mean": [],
            "capture_candidate_identity_positive_optimizer_rank_improved_fraction": [],
            "capture_candidate_identity_negative_optimizer_rank_reduced_fraction": [],
            "capture_candidate_identity_score_semantics_version": [],
            "capture_candidate_identity_valid_margin_mean": [],
            "capture_candidate_identity_valid_margin_min": [],
            "capture_candidate_identity_valid_margin_max": [],
            "capture_candidate_identity_behavior_margin_mean": [],
            "capture_candidate_identity_behavior_rank_mean": [],
            "capture_candidate_identity_positive_behavior_margin_mean": [],
            "capture_candidate_identity_negative_behavior_margin_mean": [],
            "capture_candidate_identity_positive_behavior_rank_mean": [],
            "capture_candidate_identity_negative_behavior_rank_mean": [],
            "capture_candidate_identity_positive_current_rank_mean": [],
            "capture_candidate_identity_negative_current_rank_mean": [],
            "capture_candidate_identity_positive_margin_change_mean": [],
            "capture_candidate_identity_negative_margin_change_mean": [],
            "capture_candidate_identity_positive_signed_margin_change_mean": [],
            "capture_candidate_identity_negative_signed_margin_change_mean": [],
            "capture_candidate_identity_positive_margin_improved_fraction": [],
            "capture_candidate_identity_negative_margin_reduced_fraction": [],
            "capture_candidate_identity_positive_boundary_crossed_fraction": [],
            "capture_candidate_identity_negative_boundary_respected_fraction": [],
            "capture_candidate_identity_positive_rank_improved_fraction": [],
            "capture_candidate_identity_negative_rank_reduced_fraction": [],
            "capture_candidate_identity_behavior_valid_fraction": [],
            "capture_candidate_identity_policy_age_mean": [],
            "capture_candidate_identity_policy_age_max": [],
            "capture_candidate_identity_policy_version": [],
            "capture_identity_factor_credit_active_fraction": [],
            "capture_identity_factor_credit_positive_mass": [],
            "capture_identity_factor_credit_negative_mass": [],
            "capture_outcome_identity_order2_delta_abs_mass": [],
            "capture_outcome_identity_order3_delta_abs_mass": [],
            "capture_identity_target_order2_fraction": [],
            "capture_identity_target_order3_fraction": [],
            "capture_outcome_non_target_delta_abs_max": [],
            "capture_to_win_quality_gate_mean": [],
            "capture_to_win_quality_gate_abs_mean": [],
            "capture_to_win_quality_gate_active_fraction": [],
            "capture_to_win_quality_gate_positive_fraction": [],
            "capture_to_win_quality_gate_negative_fraction": [],
            "capture_to_win_quality_gate_positive_mass": [],
            "capture_to_win_quality_gate_negative_mass": [],
            "capture_to_win_outcome_contrastive": [],
            "capture_to_win_quality_gate_definition_version": [],
            "capture_outcome_baseline_mean": [],
            "capture_outcome_capture_episode_count_mean": [],
            "capture_outcome_success_episode_count_mean": [],
            "capture_outcome_failure_episode_count_mean": [],
            "capture_outcome_mixed_window_fraction": [],
            "capture_outcome_single_success_window_fraction": [],
            "capture_outcome_single_failure_window_fraction": [],
            "capture_outcome_no_capture_window_fraction": [],
            "capture_outcome_triplet_labels_per_episode_mean": [],
            "capture_outcome_triplet_labels_per_episode_max": [],
            "capture_outcome_raw_episode_advantage_mean": [],
            "capture_outcome_raw_episode_advantage_abs_mean": [],
            "capture_outcome_window_raw_centered_mean": [],
            "capture_outcome_window_expanded_gate_sum": [],
            "capture_outcome_window_expanded_gate_abs_sum": [],
            "capture_outcome_window_center_error_ratio": [],
            "capture_to_win_credit_preclip_mean": [],
            "capture_to_win_credit_preclip_std": [],
            "capture_to_win_credit_preclip_max": [],
            "capture_to_win_credit_preclip_min": [],
            "capture_to_win_credit_positive_clip_fraction": [],
            "capture_to_win_credit_negative_clip_fraction": [],
            "capture_identity_event_count_mean": [],
            "capture_identity_matched_event_count_mean": [],
            "capture_identity_unmatched_event_count_mean": [],
            "capture_identity_candidate_factor_count_mean": [],
            "capture_identity_match_fraction": [],
            "capture_identity_candidates_per_matched_event": [],
            "capture_outcome_label_gate_correlation": [],
            "capture_outcome_label_gate_correlation_valid": [],
            "capture_outcome_success_labels_mean": [],
            "capture_outcome_failure_labels_mean": [],
            "capture_outcome_success_gate_total_mean": [],
            "capture_outcome_failure_gate_total_mean": [],
            "pair_pursuit_credit_mean": [],
            "pair_pursuit_credit_active_fraction": [],
            "pair_credit_active_fraction": [],
            "pair_pursuit_credit_std": [],
            "pair_pursuit_credit_max": [],
            "pair_pursuit_credit_nonzero_count": [],
            "pair_credit_top1_mass_fraction": [],
            "pair_pursuit_quality_mean": [],
            "pair_pursuit_quality_active_fraction": [],
            "pair_to_triplet_transition_score_mean": [],
            "pair_to_triplet_transition_active_fraction": [],
            "triplet_capture_quality_mean": [],
            "triplet_capture_quality_active_fraction": [],
            "transition_delay_mean": [],
            "transition_delay_min": [],
            "transition_delay_max": [],
            "capture_event_count": [],
            "capture_matched_count": [],
            "unmatched_capture_count": [],
            "capture_matched_fraction": [],
            "unmatched_capture_fraction": [],
            "failed_episode_capture_count": [],
            "failed_episode_capture_fraction": [],
            "capture_to_win_capture_success_fraction": [],
            "capture_to_win_episode_success_fraction": [],
            "positive_reward_without_capture_fraction": [],
            "positive_reward_step_count": [],
            "positive_reward_without_capture_count": [],
            "positive_reward_step_fraction": [],
            "offset0_candidate_count": [],
            "offset0_candidate_fraction": [],
            "graph_return_credit_strength_mean": [],
            "credit_order2_factor_rl_loss": [],
            "credit_order3_factor_rl_loss": [],
            "credit_o3_minus_o2_factor_rl_loss": [],
            "order2_positive_adv_fraction": [],
            "order3_positive_adv_fraction": [],
            "credit_order2_positive_adv_fraction": [],
            "credit_order3_positive_adv_fraction": [],
            "order2_promoted_adv_fraction": [],
            "order3_promoted_adv_fraction": [],
            "adj_order_adv_positive_only": [],
            "adj_order_adv_negative_coef": [],
            "adj_order_adv_require_positive_graph_adv": [],
            "use_adj_triplet_graph_return_credit": [],
            "adj_triplet_graph_return_credit_coef": [],
            "adj_triplet_graph_return_credit_cap": [],
            "adj_triplet_graph_return_credit_min_graph_adv": [],
            "adj_triplet_graph_return_credit_raw_gate_scale": [],
            "adj_triplet_graph_return_credit_require_delayed_gate": [],
            "use_adj_delayed_triplet_credit": [],
            "adj_delayed_triplet_credit_coef": [],
            "adj_delayed_triplet_credit_window": [],
            "adj_delayed_triplet_credit_cap": [],
            "adj_delayed_triplet_credit_min_reward": [],
            "adj_delayed_triplet_credit_positive_only": [],
            "adj_delayed_triplet_credit_min_adv": [],
            "adj_delayed_triplet_credit_require_future_match": [],
            "use_adj_delayed_triplet_success_gate": [],
            "adj_delayed_triplet_success_gate_min_adv": [],
            "adj_delayed_triplet_success_gate_scale": [],
            "adj_delayed_triplet_success_gate_floor": [],
            "adj_delayed_triplet_future_overlap_min_nodes": [],
            "adj_delayed_triplet_partial_match_weight": [],
            "use_adj_capture_to_win_credit": [],
            "adj_capture_to_win_credit_coef": [],
            "adj_capture_to_win_credit_min_outcome_adv": [],
            "adj_capture_to_win_credit_scale": [],
            "adj_capture_to_win_credit_cap": [],
            "adj_capture_to_win_credit_require_future_match": [],
            "use_adj_pair_triplet_complementary_credit": [],
            "adj_pair_pursuit_credit_coef": [],
            "adj_pair_pursuit_credit_window": [],
            "adj_pair_pursuit_credit_cap": [],
            "adj_pair_pursuit_credit_min_reward": [],
            "use_adj_advantage_triplet_scorer": [],
            "adv_triplet_credit_pair_updates": [],
            "adv_triplet_credit_triplet_updates": [],
            "adv_triplet_credit_seen_ratio": [],
            "adv_pair_credit_seen_ratio": [],
            "adv_triplet_credit_mean": [],
            "adv_triplet_marginal_mean": [],
            "adv_triplet_marginal_positive_fraction": [],
            "use_adj_triplet_credit_direct_rank": [],
            "adj_triplet_credit_rank_coef": [],
            "adj_triplet_credit_min_multiplier": [],
            "adj_triplet_credit_max_multiplier": [],
            "adj_triplet_credit_negative_rank_scale": [],
            "adj_triplet_credit_min_positive_fraction": [],
            "adv_triplet_score_multiplier_mean": [],
            "adv_triplet_score_multiplier_min": [],
            "adv_triplet_score_multiplier_max": [],
            "adv_triplet_score_marginal_mean": [],
            "adv_triplet_score_positive_fraction": [],
            "adv_triplet_negative_scaled_fraction": [],
            "adj_ppo_epochs_ran": [],
            "adj_ppo_early_stop_triggered": [],
            "adj_ppo_last_epoch_clip_ratio": [],
            "adj_ppo_last_epoch_factor_clip_ratio": [],
            "adj_ppo_last_epoch_control_clip_ratio": [],
            "adj_ppo_last_epoch_control_factor_clip_ratio": [],
            "adj_ppo_control_uses_trusted_population": [],
            "adj_control_runtime_contract_valid": [],
            "adj_ppo_clip_stop_ratio": [],
            "adj_ppo_factor_clip_stop_ratio": [],
            "adj_ppo_min_epochs": [],
            "trusted_clamp_ratio": [],
            "trusted_factor_clamp_ratio": [],
            "adj_graph_trust_weight_mean": [],
            "adj_factor_trust_weight_mean": [],
            "adj_graph_stale_ratio": [],
            "adj_factor_stale_ratio": [],
            "use_adj_ppo_stale_trust": [],
            "adj_ppo_stale_trust_clip": [],
            "adj_ppo_stale_trust_scale": [],
            "adj_ppo_stale_trust_min_weight": [],
            "adj_recent_episode_window": [],
            "adj_recent_episode_window_config": [],
            "adj_dynamic_recent_window_enabled": [],
            "adj_recent_window_shrunk": [],
            "adj_recent_window_graph_control_ratio": [],
            "adj_recent_window_factor_control_ratio": [],
            "adj_recent_window_control_uses_trusted_population": [],
            "adj_recent_window_recovered": [],
            "adj_recent_window_emergency_shrunk": [],
            "adj_recent_window_high_stale_count": [],
            "adj_recent_window_low_stale_count": [],
            "adj_sample_episode_count": [],
            "adj_sample_recent_fraction": [],
            "adj_outcome_contrast_replay_support_version": [],
            "adj_sample_base_episode_count": [],
            "adj_sample_outcome_contrast_augmented_count": [],
            "adj_sample_outcome_positive_available": [],
            "adj_sample_outcome_negative_available": [],
            "adj_sample_outcome_positive_episode_count": [],
            "adj_sample_outcome_negative_episode_count": [],
            "adj_sample_outcome_class_complete": [],
            "adj_sample_outcome_support_exhausted": [],
            "adj_sample_outcome_credit_enabled": [],
            "adj_sample_outcome_cached_selection_reused": [],
            "adj_sample_outcome_support_round": [],
            "adj_sample_outcome_cross_update_reuse_count": [],
            "adj_sample_outcome_positive_available_count": [],
            "adj_sample_outcome_negative_available_count": [],
            "adj_sample_outcome_base_positive_count": [],
            "adj_sample_outcome_base_negative_count": [],
            "adj_sample_outcome_augmented_positive_count": [],
            "adj_sample_outcome_augmented_negative_count": [],
            "adj_sample_outcome_base_age_mean": [],
            "adj_sample_outcome_base_age_max": [],
            "adj_sample_outcome_augmented_age_mean": [],
            "adj_sample_outcome_augmented_age_max": [],
            "adj_sample_outcome_positive_support_generation": [],
            "adj_sample_outcome_negative_support_generation": [],
            "adj_sample_outcome_positive_support_age": [],
            "adj_sample_outcome_negative_support_age": [],
            "adj_sample_outcome_support_used_count": [],
            "adj_sample_outcome_support_used_fraction": [],
            "adj_sample_outcome_full_buffer_baseline": [],
            "adj_sample_outcome_base_cohort_baseline": [],
            "adj_sample_outcome_trained_cohort_baseline": [],
            "adj_sample_outcome_full_trained_baseline_gap": [],
            "adj_sample_outcome_trained_capture_episode_count": [],
            "adj_sample_outcome_cohort_centered_sum": [],
            "adj_sample_outcome_cohort_center_error": [],
            "adj_sample_outcome_cohort_center_valid": [],
            "adj_sample_outcome_positive_gate_episode_count": [],
            "adj_sample_outcome_negative_gate_episode_count": [],
            "adj_sample_outcome_positive_credit_episode_count": [],
            "adj_sample_outcome_negative_credit_episode_count": [],
            "adj_sample_outcome_generation_update_count": [],
            "adj_sample_outcome_slot_overwrite_count": [],
            "adj_sample_outcome_generation_conflict_count": [],
            "adj_sample_outcome_invalid_used_state_count": [],
            "adj_recent_episode_window_emergency": [],
            "adj_recent_window_emergency_stale_threshold": [],
            "adj_recent_window_emergency_factor_stale_threshold": [],
        }
