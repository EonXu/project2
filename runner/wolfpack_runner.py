import numpy as np
import torch
import time
from typing import Any, Dict, List, Optional
from runner.base_runner import RecRunner
import os
import pandas as pd

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

        # 预热：随机策略采样若干回合填充经验池
        # 回放/可视化时可通过 args.skip_warmup=True 跳过预热，加快启动速度
        skip_warmup = bool(getattr(self.args, "skip_warmup", False))
        self.start = time.time()
        if not skip_warmup:
            num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
            self.warmup(num_warmup_episodes)
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
                "success_now": int(bool(event.get("success_now", False))),
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
                "success_now": int(bool(event.get("success_now", False))),
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
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

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
        first_success_step = -1
        # Number of 0->1 slot activations whose recurrent state and previous
        # action were reset. This covers both new joins and recoveries.
        activation_state_reset_count = 0

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
                    prob_adj, adj, _ = self.adj_network.sample(
                        obs_batch[None, :],
                        rnn_states_batch.unsqueeze(0),
                        self.use_adj_init,
                        dones,
                        explore,
                        self.total_env_steps
                    )

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
                        acts_batch, qtot, _, f_q = policy.get_actions(
                            obs_batch[None, :],
                            rnn_states_batch.unsqueeze(0),
                            torch.tensor(avail_acts_batch).to(self.device),
                            t_env=self.total_env_steps,
                            explore=explore,
                            adj_input=adj_all.to(self.device),
                            no_sequence=False,
                            dones=torch.tensor(dones).to(self.device))
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
            info_dict = self._extract_first_info_dict(infos, env_i=0)
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
                    "success_now": bool(info_dict.get("success_now", False)),
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
                prob_adj, adj, _ = self.adj_network.sample(
                    obs_batch[None, :],
                    rnn_states_batch.unsqueeze(0),
                    self.use_adj_init,
                    dones,
                    explore,
                    self.total_env_steps
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

            self.buffer.insert(self.num_envs,  # push all episodes collected in this rollout step to the buffer
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

            if (
                self.algorithm_name in self._BATCHED_FACTOR_GRAPH_ALGOS
                and self.total_env_steps >= self.adj_begin_step
                and not warmup
            ):
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
                                             episode_rnn_states)
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

        env_info['average_episode_rewards'] = np.sum(episode_rewards[p_id][:, 0, 0, 0])
        # ===== 日志新增：训练阶段也低频导出逐步轨迹 CSV =====
        # eval 阶段已经由 eval() 调用 _dump_eval_episode_step_tables；
        # 这里仅对训练 episode 按 log_interval 导出，避免文件过大。
        if training_episode and (not warmup):
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
                "capture_to_win_triplet_credit_active_fraction",
                "capture_to_win_quality_gate_mean",
                "capture_to_win_quality_gate_active_fraction",
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
                "adj_recent_window_recovered",
                "adj_recent_window_emergency_shrunk",
                "adj_recent_window_high_stale_count",
                "adj_recent_window_low_stale_count",
                "adj_sample_episode_count",
                "adj_sample_recent_fraction",
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
            "capture_to_win_triplet_credit_active_fraction": [],
            "capture_to_win_quality_gate_mean": [],
            "capture_to_win_quality_gate_active_fraction": [],
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
            "adj_recent_window_recovered": [],
            "adj_recent_window_emergency_shrunk": [],
            "adj_recent_window_high_stale_count": [],
            "adj_recent_window_low_stale_count": [],
            "adj_sample_episode_count": [],
            "adj_sample_recent_fraction": [],
            "adj_recent_episode_window_emergency": [],
            "adj_recent_window_emergency_stale_threshold": [],
            "adj_recent_window_emergency_factor_stale_threshold": [],
        }
