import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from utils.util import DecayThenFlatSchedule, to_torch
import numpy as np


class _GATLayer(nn.Module):
    def __init__(self, in_dim, num_heads=4, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, in_dim // num_heads)
        self.proj = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.negative_slope = negative_slope

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(self, H, exist_mask):
        """
        H: [B, N, hidden]
        exist_mask: [B, N], 1=存活/有效, 0=死亡/空槽
        return:
            A: [B, N, N], directed attention，非法边、自环严格为 0
        """
        B, N, _ = H.shape
        exist_mask = exist_mask.to(device=H.device, dtype=H.dtype)
        node_mask = exist_mask > 0.5

        x = self.proj(H).view(B, N, self.num_heads, self.head_dim)

        xi = x.unsqueeze(2).expand(-1, -1, N, -1, -1)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1, -1)
        xij = torch.cat([xi, xj], dim=-1)

        e = F.leaky_relu(
            torch.einsum("hd,bnmhd->bnmh", self.attn_vec, xij),
            negative_slope=self.negative_slope
        )

        # 合法 pair：两端都存活，且禁止自环
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        diag_mask = torch.eye(N, dtype=torch.bool, device=H.device).view(1, N, N)
        pair_mask = pair_mask & (~diag_mask)
        pair_mask_h = pair_mask.unsqueeze(-1)

        # softmax 前屏蔽，softmax 后二次归零并重归一化，防止全 -1e9 行产生均匀伪概率
        e = e.masked_fill(~pair_mask_h, -1e9)
        alpha = torch.softmax(e, dim=2)
        alpha = alpha * pair_mask_h.to(alpha.dtype)

        denom = alpha.sum(dim=2, keepdim=True).clamp_min(1e-8)
        alpha = alpha / denom

        has_valid_neighbor = pair_mask.any(dim=2, keepdim=True).unsqueeze(-1)
        alpha = torch.where(has_valid_neighbor, alpha, torch.zeros_like(alpha))

        A = alpha.mean(dim=-1)
        A = A * pair_mask.to(A.dtype)
        return A


class Adj_Generator(nn.Module):
    def __init__(self, args, obs_dim, state_dim, act_dim, device):
        super(Adj_Generator, self).__init__()

        self.device = device
        self.args = args
        self.num_variable = args.max_player_num
        self.num_factor = args.num_factor
        self.highest_orders = args.highest_orders
        self.sparsity = max(0.0, min(1.0, float(getattr(args, "sparsity", 1.0))))
        self.require_connected = bool(
            getattr(args, "require_connected_adj", False)
        )
        self.order3_bonus_final = max(
            0.0,
            float(getattr(args, "adj_order3_bonus", 1.0)),
        )
        configured_order3_bonus_start = float(
            getattr(args, "adj_order3_bonus_start", -1.0)
        )
        self.order3_bonus_start = (
            self.order3_bonus_final
            if configured_order3_bonus_start < 0.0
            else max(0.0, configured_order3_bonus_start)
        )
        self.order3_bonus_anneal_steps = max(
            0,
            int(getattr(args, "adj_order3_bonus_anneal_steps", 0)),
        )
        self.current_order3_bonus = float(self.order3_bonus_start)
        self.sampling_temperature_start = max(
            1e-3,
            float(getattr(args, "adj_sampling_temperature_start", 1.0)),
        )
        self.sampling_temperature_final = max(
            1e-3,
            float(getattr(args, "adj_sampling_temperature_final", 1.0)),
        )
        self.sampling_temperature_anneal_steps = max(
            0,
            int(getattr(args, "adj_sampling_temperature_anneal_steps", 0)),
        )
        self.current_sampling_temperature = float(
            self.sampling_temperature_start
        )
        self.min_order3_ratio_start = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_min_order3_ratio_start", 0.0)),
            ),
        )
        self.min_order3_ratio_final = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_min_order3_ratio_final", 0.0)),
            ),
        )
        self.min_order3_ratio_anneal_steps = max(
            0,
            int(getattr(args, "adj_min_order3_ratio_anneal_steps", 0)),
        )
        self.current_min_order3_ratio = float(
            self.min_order3_ratio_start
        )
        self.order3_quota_score_floor = max(
            0.0,
            float(getattr(args, "adj_order3_quota_score_floor", 0.0)),
        )
        quota_mode = str(
            getattr(args, "adj_order3_quota_mode", "hard")
        ).lower()
        self.order3_quota_mode = (
            "soft" if quota_mode == "soft" else "hard"
        )
        self.order3_soft_quota_coef = max(
            0.0,
            float(getattr(args, "adj_order3_soft_quota_coef", 0.0)),
        )
        self.triplet_balance_coef = max(
            0.0,
            float(getattr(args, "adj_triplet_balance_coef", 0.0)),
        )
        self.use_advantage_triplet_scorer = bool(
            getattr(args, "use_adj_advantage_triplet_scorer", False)
        )
        self.triplet_credit_ema_alpha = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_triplet_credit_ema_alpha", 0.05)),
            ),
        )
        self.triplet_credit_score_coef = max(
            0.0,
            float(getattr(args, "adj_triplet_credit_score_coef", 0.50)),
        )
        self.triplet_credit_score_scale = max(
            1e-6,
            float(getattr(args, "adj_triplet_credit_score_scale", 0.05)),
        )
        self.use_triplet_credit_direct_rank = bool(
            getattr(args, "use_adj_triplet_credit_direct_rank", False)
        )
        self.triplet_credit_rank_coef = max(
            0.0,
            float(getattr(args, "adj_triplet_credit_rank_coef", 0.0)),
        )
        self.triplet_credit_min_multiplier = max(
            1e-3,
            float(getattr(args, "adj_triplet_credit_min_multiplier", 0.25)),
        )
        self.triplet_credit_max_multiplier = max(
            self.triplet_credit_min_multiplier,
            float(getattr(args, "adj_triplet_credit_max_multiplier", 3.0)),
        )
        self.triplet_credit_negative_rank_scale = max(
            0.0,
            float(
                getattr(
                    args,
                    "adj_triplet_credit_negative_rank_scale",
                    1.0,
                )
            ),
        )
        self.triplet_credit_min_positive_fraction = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        args,
                        "adj_triplet_credit_min_positive_fraction",
                        0.0,
                    )
                ),
            ),
        )
        self.triplet_negative_graph_penalty = max(
            0.0,
            float(getattr(args, "adj_triplet_negative_graph_penalty", 0.50)),
        )
        self._pair_index_map = {}
        pair_counter = 0
        for pair_i in range(int(self.num_variable)):
            for pair_j in range(pair_i + 1, int(self.num_variable)):
                self._pair_index_map[(pair_i, pair_j)] = pair_counter
                pair_counter += 1
        self._triplet_index_map = {}
        triplet_counter = 0
        for triplet in combinations(range(int(self.num_variable)), 3):
            self._triplet_index_map[tuple(triplet)] = triplet_counter
            triplet_counter += 1
        self.register_buffer(
            "pair_credit_ema",
            torch.zeros(pair_counter, dtype=torch.float32),
        )
        self.register_buffer(
            "pair_credit_seen",
            torch.zeros(pair_counter, dtype=torch.float32),
        )
        self.register_buffer(
            "triplet_credit_ema",
            torch.zeros(triplet_counter, dtype=torch.float32),
        )
        self.register_buffer(
            "triplet_credit_seen",
            torch.zeros(triplet_counter, dtype=torch.float32),
        )
        self.last_adv_triplet_score_multiplier_mean = 1.0
        self.last_adv_triplet_score_multiplier_min = 1.0
        self.last_adv_triplet_score_multiplier_max = 1.0
        self.last_adv_triplet_score_marginal_mean = 0.0
        self.last_adv_triplet_score_positive_fraction = 0.0
        self.last_adv_triplet_negative_scaled_fraction = 0.0
        self.max_order3_ratio_start = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_max_order3_ratio_start", 1.0)),
            ),
        )
        self.max_order3_ratio_final = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_max_order3_ratio_final", 1.0)),
            ),
        )
        self.max_order3_ratio_anneal_steps = max(
            0,
            int(getattr(args, "adj_max_order3_ratio_anneal_steps", 0)),
        )
        self.current_max_order3_ratio = max(
            self.current_min_order3_ratio,
            float(self.max_order3_ratio_start),
        )
        self.greedy_sample_prob_start = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_greedy_sample_prob_start", 0.0)),
            ),
        )
        self.greedy_sample_prob_final = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_greedy_sample_prob_final", 0.0)),
            ),
        )
        self.greedy_sample_prob_anneal_steps = max(
            0,
            int(getattr(args, "adj_greedy_sample_prob_anneal_steps", 0)),
        )
        self.greedy_sample_prob_cap = max(
            0.0,
            min(
                1.0,
                float(getattr(args, "adj_greedy_sample_prob_cap", 1.0)),
            ),
        )
        self.current_greedy_sample_prob = float(
            self.greedy_sample_prob_start
        )
        self.min_pair_ratio = max(
            0.0,
            min(1.0, float(getattr(args, "adj_min_pair_ratio", 0.0))),
        )
        self.use_order3_credit_gate = bool(
            getattr(args, "use_adj_order3_credit_gate", False)
        )
        self.order3_credit_gate_loss_scale = max(
            1e-8,
            float(getattr(args, "adj_order3_credit_gate_loss_scale", 0.005)),
        )
        self.order3_credit_gate_min_scale = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(args, "adj_order3_credit_gate_min_scale", 0.55)
                ),
            ),
        )
        self.order3_credit_gate_ema_alpha = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(args, "adj_order3_credit_gate_ema_alpha", 0.1)
                ),
            ),
        )
        self.order3_credit_gate_max_delta = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(args, "adj_order3_credit_gate_max_delta", 1.0)
                ),
            ),
        )
        self.use_relative_order3_credit_gate = bool(
            getattr(args, "use_adj_order3_relative_credit_gate", False)
        )
        self.order3_credit_gate_margin = max(
            0.0,
            float(getattr(args, "adj_order3_credit_gate_margin", 0.0)),
        )
        self.order3_credit_loss_ema = 0.0
        self.order3_credit_margin_ema = 0.0
        self.current_order3_credit_gate = 1.0
        self.exploration_mix = max(
            0.0,
            min(1.0, float(getattr(args, "adj_exploration_mix", 0.0))),
        )
        self._triplet_cache = {}
        self._triplet_flat_id_cache = {}
        self._triplet_pair_id_cache = {}

        self.rng = np.random.RandomState(int(getattr(args, "seed", 0)) + 520000)

        gat_heads = args.gat_heads
        gat_slope = args.gat_negative_slope
        triplet_feature_mode = str(
            getattr(args, "adj_triplet_feature_mode", "pair")
        ).lower()
        self.triplet_feature_mode = (
            "synergy" if triplet_feature_mode == "synergy" else "pair"
        )

        self.gat = _GATLayer(
            in_dim=args.hidden_size,
            num_heads=gat_heads,
            negative_slope=gat_slope
        )

        # Three pair scores [s_ij, s_ik, s_jk] produce a bounded residual
        # multiplier around one for the corresponding third-order factor.
        hyperedge_hidden = int(getattr(args, "gat_hyperedge_hidden", getattr(args, "adj_hidden_dim", 32)))
        hyperedge_in_dim = 6 if self.triplet_feature_mode == "synergy" else 3
        self.hyperedge_scorer = nn.Sequential(
            nn.Linear(hyperedge_in_dim, hyperedge_hidden),
            nn.LeakyReLU(gat_slope),
            nn.Linear(hyperedge_hidden, 1)
        )
        # Start triplets on the same scale as their constituent pairs. The
        # former sigmoid gate was always below one and structurally suppressed
        # every triplet before the graph policy had learned anything.
        nn.init.zeros_(self.hyperedge_scorer[-1].weight)
        nn.init.zeros_(self.hyperedge_scorer[-1].bias)

        self.exploration = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.adj_anneal_time,
            decay="linear"
        )

        self.to(device)

    @staticmethod
    def _linear_schedule(start, final, t_env, anneal_steps):
        if t_env is None or anneal_steps <= 0:
            return float(final if anneal_steps <= 0 else start)
        progress = min(max(float(t_env) / float(anneal_steps), 0.0), 1.0)
        return float(start + progress * (final - start))

    def update_order3_credit_gate(
            self,
            order3_factor_rl_loss,
            order2_factor_rl_loss=None):
        """Feed reward-driven triplet credit back into graph hardening.

        ``order3_factor_rl_loss`` should be computed from raw factor-local
        credit, before any order-aware triplet multiplier is applied. Positive
        values mean recent triplet factors received negative local credit.
        run32 showed that using the weighted loss lets the gate react to the
        multiplier itself, not to true triplet quality.  This EMA gate lowers
        the effective order3 quota and greedy argmax mixture only when the
        raw credit signal is actively against the current triplet structure.
        """
        if not self.use_order3_credit_gate:
            self.current_order3_credit_gate = 1.0
            return 1.0
        try:
            loss_value = float(order3_factor_rl_loss)
        except (TypeError, ValueError):
            return float(self.current_order3_credit_gate)
        if not np.isfinite(loss_value):
            return float(self.current_order3_credit_gate)

        margin_value = 0.0
        if self.use_relative_order3_credit_gate:
            try:
                order2_loss_value = float(order2_factor_rl_loss)
            except (TypeError, ValueError):
                order2_loss_value = 0.0
            if not np.isfinite(order2_loss_value):
                order2_loss_value = 0.0
            # Only suppress triplets when they are worse than pair factors by
            # a margin.  Absolute order3 loss stayed positive in run30 even
            # during the best reward/win phases, so using it directly pushed
            # the gate to its floor and kept order3_ratio below target.
            margin_value = (
                loss_value
                - order2_loss_value
                - float(self.order3_credit_gate_margin)
            )
            bad_triplet_credit = max(0.0, margin_value)
        else:
            bad_triplet_credit = max(0.0, loss_value)
        alpha = float(self.order3_credit_gate_ema_alpha)
        self.order3_credit_loss_ema = (
            (1.0 - alpha) * float(self.order3_credit_loss_ema)
            + alpha * bad_triplet_credit
        )
        self.order3_credit_margin_ema = (
            (1.0 - alpha) * float(self.order3_credit_margin_ema)
            + alpha * margin_value
        )
        pressure = min(
            1.0,
            float(self.order3_credit_loss_ema)
            / float(self.order3_credit_gate_loss_scale),
        )
        target_gate = (
            1.0
            - pressure * (1.0 - float(self.order3_credit_gate_min_scale))
        )
        max_delta = float(self.order3_credit_gate_max_delta)
        if max_delta < 1.0:
            delta = target_gate - float(self.current_order3_credit_gate)
            delta = max(-max_delta, min(max_delta, delta))
            target_gate = float(self.current_order3_credit_gate) + delta
        self.current_order3_credit_gate = target_gate
        self.current_order3_credit_gate = max(
            float(self.order3_credit_gate_min_scale),
            min(1.0, float(self.current_order3_credit_gate)),
        )
        return float(self.current_order3_credit_gate)

    def _update_graph_schedules(self, t_env=None):
        """Update rollout graph schedules used by sampling/evaluate_prob.

        Training rollouts are stochastic but evaluation uses argmax.  Annealed
        sampling temperature closes that train/eval topology gap without
        removing early graph exploration.  The third-order prior is also
        annealed so early policies are not forced into triplet-heavy graphs
        before useful action values exist.
        """
        if t_env is not None:
            self.current_order3_bonus = self._linear_schedule(
                self.order3_bonus_start,
                self.order3_bonus_final,
                t_env,
                self.order3_bonus_anneal_steps,
            )
            self.current_sampling_temperature = max(
                1e-3,
                self._linear_schedule(
                    self.sampling_temperature_start,
                    self.sampling_temperature_final,
                    t_env,
                    self.sampling_temperature_anneal_steps,
                ),
            )
            scheduled_min_order3_ratio = self._linear_schedule(
                self.min_order3_ratio_start,
                self.min_order3_ratio_final,
                t_env,
                self.min_order3_ratio_anneal_steps,
            )
            if self.use_order3_credit_gate:
                scheduled_min_order3_ratio *= float(
                    self.current_order3_credit_gate
                )
            self.current_min_order3_ratio = max(
                0.0,
                min(1.0, scheduled_min_order3_ratio),
            )
            self.current_max_order3_ratio = self._linear_schedule(
                self.max_order3_ratio_start,
                self.max_order3_ratio_final,
                t_env,
                self.max_order3_ratio_anneal_steps,
            )
            self.current_max_order3_ratio = max(
                self.current_min_order3_ratio,
                min(1.0, self.current_max_order3_ratio),
            )
            scheduled_greedy_prob = self._linear_schedule(
                self.greedy_sample_prob_start,
                self.greedy_sample_prob_final,
                t_env,
                self.greedy_sample_prob_anneal_steps,
            )
            if self.use_order3_credit_gate:
                scheduled_greedy_prob *= float(
                    self.current_order3_credit_gate
                )
            self.current_greedy_sample_prob = scheduled_greedy_prob
            self.current_greedy_sample_prob = max(
                0.0,
                min(
                    float(self.greedy_sample_prob_cap),
                    self.current_greedy_sample_prob,
                ),
            )
        return (
            float(self.current_order3_bonus),
            float(self.current_sampling_temperature),
            float(self.current_min_order3_ratio),
            float(self.current_max_order3_ratio),
            float(self.current_greedy_sample_prob),
        )

    def _prepare_inputs(self, rnn_obs, dones):
        rnn_obs = to_torch(rnn_obs).to(self.device)

        if rnn_obs.dim() == 2:
            rnn_obs = rnn_obs.unsqueeze(0)

        B, N, _ = rnn_obs.shape
        if N != self.num_variable:
            raise RuntimeError(
                f"fixed-capacity mismatch: graph N={N}, "
                f"max_player_num={self.num_variable}"
            )

        # 兼容没有 torch.nan_to_num 的旧 PyTorch。
        rnn_obs = torch.where(
            torch.isfinite(rnn_obs),
            rnn_obs,
            torch.zeros_like(rnn_obs),
        )

        if dones is None:
            exist_mask = torch.ones(
                B, N, device=self.device, dtype=torch.float32
            )
        else:
            dones_t = to_torch(dones).to(self.device)
            dones_t = dones_t.reshape(B, N, -1)[..., 0]
            exist_mask = (1.0 - dones_t.float()).clamp(0.0, 1.0)

        # 即使 Runner 漏掉一次 hidden 清理，失效节点也不会进入 GAT
        rnn_obs = rnn_obs * exist_mask.unsqueeze(-1)
        return rnn_obs, exist_mask

    def _pair_score(self, A, exist_mask):
        """
        directed GAT attention -> undirected pair score。
        return:
            pair_score: [B, N, N]
            pair_mask:  [B, N, N]
        """
        B, N, _ = A.shape
        node_mask = exist_mask > 0.5

        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        diag_mask = torch.eye(N, dtype=torch.bool, device=A.device).view(1, N, N)
        pair_mask = pair_mask & (~diag_mask)

        pair_score = 0.5 * (A + A.transpose(1, 2))
        pair_score = pair_score * pair_mask.to(pair_score.dtype)
        return pair_score, pair_mask

    def _triplet_indices(self, N, device):
        key = (int(N), str(device))
        if key in self._triplet_cache:
            return self._triplet_cache[key]

        if N < 3:
            triplets = torch.empty((0, 3), dtype=torch.long, device=device)
        else:
            triplets_list = list(combinations(range(N), 3))
            triplets = torch.tensor(triplets_list, dtype=torch.long, device=device) \
                if len(triplets_list) > 0 else torch.empty((0, 3), dtype=torch.long, device=device)

        self._triplet_cache[key] = triplets
        return triplets

    def _triplet_flat_ids(self, triplets):
        key = (int(triplets.shape[0]), str(triplets.device))
        if key in self._triplet_flat_id_cache:
            return self._triplet_flat_id_cache[key]
        ids = []
        for row in triplets.detach().cpu().tolist():
            ids.append(self._triplet_index_map.get(tuple(row), -1))
        tensor_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=triplets.device,
        )
        self._triplet_flat_id_cache[key] = tensor_ids
        return tensor_ids

    def _triplet_pair_flat_ids(self, triplets):
        key = (int(triplets.shape[0]), str(triplets.device))
        if key in self._triplet_pair_id_cache:
            return self._triplet_pair_id_cache[key]
        ids = []
        for i, j, k in triplets.detach().cpu().tolist():
            ids.append(
                [
                    self._pair_index_map.get((min(i, j), max(i, j)), -1),
                    self._pair_index_map.get((min(i, k), max(i, k)), -1),
                    self._pair_index_map.get((min(j, k), max(j, k)), -1),
                ]
            )
        tensor_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=triplets.device,
        )
        self._triplet_pair_id_cache[key] = tensor_ids
        return tensor_ids

    def update_factor_credit_memory(
            self,
            adj_binary,
            raw_local_factor_advantage,
            graph_advantage,
            factor_training_mask):
        """Update candidate-quality EMA from selected graph factors.

        This is intentionally graph-conditioned.  run34 showed that a binary
        graph-positive gate only changes the loss weight; it does not tell the
        triplet scorer which triplets have positive marginal value.  We store a
        small EMA per fixed-capacity pair/triplet candidate and later compare
        each triplet against its internal pair backbone.  Positive credit is
        only assigned when the local residual and the graph-level advantage
        are both positive; locally positive factors inside negative-return
        graphs receive a bounded penalty instead of promotion.
        """
        if not self.use_advantage_triplet_scorer:
            return {}
        with torch.no_grad():
            adj_t = to_torch(adj_binary).detach().to(self.device)
            raw_adv = to_torch(raw_local_factor_advantage).detach().to(
                self.device
            )
            graph_adv = to_torch(graph_advantage).detach().to(self.device)
            mask = to_torch(factor_training_mask).detach().to(self.device)
            if graph_adv.dim() > 1:
                graph_adv = graph_adv.reshape(graph_adv.shape[0], -1)[:, 0]
            else:
                graph_adv = graph_adv.reshape(-1)

            alpha = float(self.triplet_credit_ema_alpha)
            pair_updates = 0
            triplet_updates = 0
            triplet_credit_sum = 0.0
            triplet_marginal_sum = 0.0
            triplet_marginal_positive = 0

            B, _, F = adj_t.shape
            for b in range(B):
                g_adv = float(graph_adv[b].detach().cpu().item())
                for f in range(F):
                    if float(mask[b, f].detach().cpu().item()) <= 0.5:
                        continue
                    nodes_t = torch.where(adj_t[b, :, f] > 0.5)[0]
                    order = int(nodes_t.numel())
                    if order not in (2, 3):
                        continue
                    nodes = sorted(int(x) for x in nodes_t.detach().cpu().tolist())
                    local_adv = float(raw_adv[b, f].detach().cpu().item())
                    if local_adv > 0.0 and g_adv > 0.0:
                        credit = local_adv
                    elif local_adv > 0.0:
                        credit = -float(self.triplet_negative_graph_penalty) * local_adv
                    else:
                        credit = local_adv
                    credit = float(max(-1.0, min(1.0, credit)))

                    if order == 2:
                        pair_id = self._pair_index_map.get(
                            (nodes[0], nodes[1]),
                            -1,
                        )
                        if pair_id >= 0:
                            old = self.pair_credit_ema[pair_id]
                            self.pair_credit_ema[pair_id] = (
                                (1.0 - alpha) * old
                                + alpha * self.pair_credit_ema.new_tensor(
                                    credit
                                )
                            )
                            self.pair_credit_seen[pair_id] = (
                                self.pair_credit_seen[pair_id] + 1.0
                            )
                            pair_updates += 1
                    else:
                        triplet_id = self._triplet_index_map.get(
                            tuple(nodes),
                            -1,
                        )
                        if triplet_id >= 0:
                            old = self.triplet_credit_ema[triplet_id]
                            self.triplet_credit_ema[triplet_id] = (
                                (1.0 - alpha) * old
                                + alpha * self.triplet_credit_ema.new_tensor(
                                    credit
                                )
                            )
                            self.triplet_credit_seen[triplet_id] = (
                                self.triplet_credit_seen[triplet_id] + 1.0
                            )
                            pair_ids = [
                                self._pair_index_map[(nodes[0], nodes[1])],
                                self._pair_index_map[(nodes[0], nodes[2])],
                                self._pair_index_map[(nodes[1], nodes[2])],
                            ]
                            pair_credit = self.pair_credit_ema[
                                torch.tensor(
                                    pair_ids,
                                    dtype=torch.long,
                                    device=self.device,
                                )
                            ].mean()
                            marginal = (
                                self.triplet_credit_ema[triplet_id]
                                - pair_credit
                            )
                            triplet_credit_sum += float(
                                self.triplet_credit_ema[triplet_id]
                                .detach()
                                .cpu()
                                .item()
                            )
                            triplet_marginal_sum += float(
                                marginal.detach().cpu().item()
                            )
                            if float(marginal.detach().cpu().item()) > 0.0:
                                triplet_marginal_positive += 1
                            triplet_updates += 1

            seen_triplets = (
                self.triplet_credit_seen > 0.0
            ).float().mean()
            seen_pairs = (
                self.pair_credit_seen > 0.0
            ).float().mean()
            return {
                "adv_triplet_credit_pair_updates": float(pair_updates),
                "adv_triplet_credit_triplet_updates": float(triplet_updates),
                "adv_triplet_credit_seen_ratio": float(
                    seen_triplets.detach().cpu().item()
                ),
                "adv_pair_credit_seen_ratio": float(
                    seen_pairs.detach().cpu().item()
                ),
                "adv_triplet_credit_mean": (
                    triplet_credit_sum / max(triplet_updates, 1)
                ),
                "adv_triplet_marginal_mean": (
                    triplet_marginal_sum / max(triplet_updates, 1)
                ),
                "adv_triplet_marginal_positive_fraction": (
                    triplet_marginal_positive / max(triplet_updates, 1)
                ),
            }

    def _score_triplets(self, pair_score, triplets):
        """
        pair_score: [B, N, N]
        triplets: [T, 3], 每行是 i<j<k
        return:
            triplet_score: [B, T]
        """
        B = pair_score.shape[0]

        if triplets.numel() == 0:
            return torch.empty(B, 0, dtype=pair_score.dtype, device=pair_score.device)

        i = triplets[:, 0]
        j = triplets[:, 1]
        k = triplets[:, 2]

        sij = pair_score[:, i, j]
        sik = pair_score[:, i, k]
        sjk = pair_score[:, j, k]

        pair_feats = torch.stack([sij, sik, sjk], dim=-1)  # [B, T, 3]
        valid = (pair_feats > 0.0).all(dim=-1)

        mean_score = pair_feats.mean(dim=-1)
        min_score = pair_feats.min(dim=-1)[0]
        max_score = pair_feats.max(dim=-1)[0]
        if self.triplet_feature_mode == "synergy":
            # Avoid torch.std here. Older PyTorch versions can produce
            # non-finite gradients at exactly zero variance, which happens
            # frequently for inactive/invalid triplets during the preflight
            # gradient check.  The range feature carries the same imbalance
            # signal without a sqrt singularity.
            spread_score = max_score - min_score
            balance_feature = (
                min_score / mean_score.clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            spread_feature = (
                spread_score / mean_score.clamp_min(1e-8)
            ).clamp(0.0, 4.0)
            scorer_feats = torch.stack(
                [
                    mean_score,
                    min_score,
                    max_score,
                    spread_score,
                    balance_feature,
                    spread_feature,
                ],
                dim=-1,
            )
        else:
            scorer_feats = pair_feats

        gate_logits = self.hyperedge_scorer(
            scorer_feats.reshape(-1, scorer_feats.shape[-1])
        ).reshape(B, -1)
        multiplier = torch.exp(0.5 * torch.tanh(gate_logits))

        # Stay on the pair-score scale while allowing learned promotion or
        # suppression of cooperative third-order interactions.  Greedy
        # constrained selection compares every triplet against individual
        # strong pairs; without a small order prior, lowering graph entropy can
        # collapse the learned topology toward pair factors even when the task
        # reward depends on three-agent cooperation.
        if self.triplet_balance_coef > 0.0:
            # A triplet with one very weak internal pair is usually a poor
            # Wolfpack coordination candidate: it preserves a high order count
            # but still behaves like a strong pair plus a passenger.  Penalize
            # such unbalanced triplets before quota/greedy selection so high
            # order factors have to be internally coherent.
            balance = (
                min_score
                / mean_score.clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            # Fractional powers have infinite derivative at exactly zero.
            # Invalid triplets and masked inactive nodes often produce zero
            # pair scores; multiplying by ``valid`` after the power is too
            # late because autograd has already seen d(0**0.75).  Use a
            # neutral value for invalid triplets and a tiny floor for valid
            # triplets to keep the preflight gradient finite.
            balance = torch.where(
                valid,
                balance.clamp_min(1e-4),
                torch.ones_like(balance),
            )
            mean_score = mean_score * torch.pow(
                balance,
                float(self.triplet_balance_coef),
            )

        if self.use_advantage_triplet_scorer and triplets.numel() > 0:
            triplet_ids = self._triplet_flat_ids(triplets).clamp_min(0)
            pair_ids = self._triplet_pair_flat_ids(triplets).clamp_min(0)
            triplet_credit = self.triplet_credit_ema[triplet_ids]
            pair_credit = self.pair_credit_ema[pair_ids].mean(dim=-1)
            marginal_credit = triplet_credit - pair_credit
            seen_count = self.triplet_credit_seen[triplet_ids]
            pair_seen_count = self.pair_credit_seen[pair_ids].mean(dim=-1)
            confidence = (
                seen_count / (seen_count + 10.0)
            ).clamp(0.0, 1.0)
            pair_confidence = (
                pair_seen_count / (pair_seen_count + 10.0)
            ).clamp(0.0, 1.0)
            # Old server PyTorch does not provide torch.minimum.
            confidence = torch.where(
                confidence < pair_confidence,
                confidence,
                pair_confidence,
            )
            scaled_credit = torch.tanh(
                marginal_credit / float(self.triplet_credit_score_scale)
            )
            rank_credit = scaled_credit
            if (
                self.use_triplet_credit_direct_rank
                and float(self.triplet_credit_rank_coef) > 0.0
            ):
                negative_scale = float(self.triplet_credit_negative_rank_scale)
                valid_float = valid.to(marginal_credit.dtype)
                valid_count = valid_float.sum().clamp_min(1.0)
                positive_fraction = (
                    (marginal_credit > 0.0).to(marginal_credit.dtype)
                    * valid_float
                ).sum() / valid_count
                if self.triplet_credit_min_positive_fraction > 0.0:
                    positive_gate = (
                        positive_fraction
                        / float(self.triplet_credit_min_positive_fraction)
                    ).clamp(0.0, 1.0)
                    negative_scale = negative_scale * float(
                        positive_gate.detach().cpu().item()
                    )
                rank_credit = torch.where(
                    scaled_credit < 0.0,
                    scaled_credit * negative_scale,
                    scaled_credit,
                )
                credit_multiplier = torch.exp(
                    float(self.triplet_credit_rank_coef)
                    * confidence
                    * rank_credit
                )
                credit_multiplier = torch.clamp(
                    credit_multiplier,
                    min=float(self.triplet_credit_min_multiplier),
                    max=float(self.triplet_credit_max_multiplier),
                )
            else:
                credit_multiplier = torch.exp(
                    float(self.triplet_credit_score_coef)
                    * confidence
                    * scaled_credit
                )
            mean_score = mean_score * credit_multiplier.unsqueeze(0)
            with torch.no_grad():
                self.last_adv_triplet_score_multiplier_mean = float(
                    credit_multiplier.mean().detach().cpu().item()
                )
                self.last_adv_triplet_score_multiplier_min = float(
                    credit_multiplier.min().detach().cpu().item()
                )
                self.last_adv_triplet_score_multiplier_max = float(
                    credit_multiplier.max().detach().cpu().item()
                )
                self.last_adv_triplet_score_marginal_mean = float(
                    marginal_credit.mean().detach().cpu().item()
                )
                self.last_adv_triplet_score_positive_fraction = float(
                    (
                        (marginal_credit > 0.0).to(marginal_credit.dtype)
                        * valid.to(marginal_credit.dtype)
                    )
                    .sum()
                    .div(valid.to(marginal_credit.dtype).sum().clamp_min(1.0))
                    .detach()
                    .cpu()
                    .item()
                )
                self.last_adv_triplet_negative_scaled_fraction = float(
                    (
                        (rank_credit != scaled_credit).to(marginal_credit.dtype)
                        * valid.to(marginal_credit.dtype)
                    )
                    .sum()
                    .div(valid.to(marginal_credit.dtype).sum().clamp_min(1.0))
                    .detach()
                    .cpu()
                    .item()
                )

        triplet_score = (
            mean_score * multiplier * self.current_order3_bonus
        )
        triplet_score = triplet_score * valid.to(triplet_score.dtype)
        return triplet_score

    def _candidate_entropy(self, pair_score, exist_mask):
        """
        基于二阶 + 三阶候选分布计算熵。
        return:
            entropy: [B, N]，会在 trainer 中继续乘 active_masks。
        """
        B, N, _ = pair_score.shape
        device = pair_score.device

        pair_i, pair_j = torch.triu_indices(N, N, offset=1, device=device)
        pair_scores = pair_score[:, pair_i, pair_j]  # [B, P]

        scores = [pair_scores]

        if self.highest_orders >= 3:
            triplets = self._triplet_indices(N, device)
            triplet_scores = self._score_triplets(pair_score, triplets)
            scores.append(triplet_scores)

        all_scores = torch.cat(scores, dim=1)
        all_scores = all_scores.clamp_min(0.0)

        score_sum = all_scores.sum(dim=1, keepdim=True)
        probs = all_scores / score_sum.clamp_min(1e-8)

        entropy_scalar = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        valid_count = (all_scores > 0.0).float().sum(
            dim=1,
            keepdim=True,
        )
        entropy_scalar = entropy_scalar / torch.log(
            valid_count.clamp_min(2.0)
        )

        has_candidate = score_sum > 1e-8
        entropy_scalar = torch.where(has_candidate, entropy_scalar, torch.zeros_like(entropy_scalar))

        # 广播到 agent 维，保持原 trainer 的 entropy 接口 [B, N]
        entropy = entropy_scalar * exist_mask
        return entropy

    def _active_factor_budget(self, num_active, num_valid_candidates):
        """Return the active-roster sparse factor budget."""
        if num_active < 2 or num_valid_candidates <= 0:
            return 0

        min_cover = (int(num_active) + 1) // 2
        sparse_target = int(np.ceil(self.sparsity * float(num_valid_candidates)))
        return min(
            int(self.num_factor),
            int(num_valid_candidates),
            max(min_cover, sparse_target),
        )

    def _candidate_catalog(self, pair_score):
        """Return candidate scores and static node-membership masks."""
        B, N, _ = pair_score.shape
        device = pair_score.device

        pair_i, pair_j = torch.triu_indices(
            N,
            N,
            offset=1,
            device=device,
        )
        pair_scores = pair_score[:, pair_i, pair_j]
        pair_nodes = torch.zeros(
            pair_scores.shape[1],
            N,
            dtype=torch.bool,
            device=device,
        )
        pair_rows = torch.arange(
            pair_scores.shape[1],
            device=device,
        )
        pair_nodes[pair_rows, pair_i] = True
        pair_nodes[pair_rows, pair_j] = True

        scores = [pair_scores]
        node_masks = [pair_nodes]
        if self.highest_orders >= 3:
            triplets = self._triplet_indices(N, device)
            triplet_scores = self._score_triplets(
                pair_score,
                triplets,
            )
            triplet_nodes = torch.zeros(
                triplet_scores.shape[1],
                N,
                dtype=torch.bool,
                device=device,
            )
            if triplets.numel() > 0:
                triplet_rows = torch.arange(
                    triplet_scores.shape[1],
                    device=device,
                ).unsqueeze(1).expand(-1, 3)
                triplet_nodes[triplet_rows, triplets] = True
            scores.append(triplet_scores)
            node_masks.append(triplet_nodes)

        return torch.cat(scores, dim=1), torch.cat(node_masks, dim=0)

    @staticmethod
    def _constrained_probabilities(
            scores,
            candidate_nodes,
            remaining,
            covered,
            active_nodes,
            slots_after=0,
            max_factor_order=3,
            require_connected=False,
            temperature=1.0):
        """
        Conditional distribution for one factor slot.

        While some active agents are uncovered, candidates must add at least
        one new node and their policy weight is multiplied by that coverage
        gain. Once coverage is complete, all remaining candidates compete.
        When connectivity is required, every factor selected after the first
        must also touch the already connected node set while adding at least
        one uncovered node. This builds one connected hypergraph instead of
        several covered but mutually isolated coordination components.
        """
        valid = (scores > 0.0) & remaining
        if not bool(torch.any(valid).item()):
            return torch.zeros_like(scores), valid

        uncovered = active_nodes & (~covered)
        if bool(torch.any(uncovered).item()):
            gains = (
                candidate_nodes
                & uncovered.unsqueeze(0)
            ).long().sum(dim=1)
            uncovered_count = int(uncovered.long().sum().item())
            min_required_gain = max(
                1,
                uncovered_count
                - int(max_factor_order) * int(slots_after),
            )
            eligible = (
                valid
                & (gains >= min_required_gain)
            )
            if (
                require_connected
                and bool(torch.any(covered & active_nodes).item())
            ):
                touches_connected_component = (
                    candidate_nodes
                    & covered.unsqueeze(0)
                ).any(dim=1)
                eligible = eligible & touches_connected_component
            coverage_weight = gains.to(scores.dtype)
        else:
            eligible = valid
            coverage_weight = torch.ones_like(scores)

        weights = (
            scores
            * coverage_weight
            * eligible.to(scores.dtype)
        )
        temperature = max(float(temperature), 1e-3)
        if abs(temperature - 1.0) > 1e-6:
            weights = (
                torch.exp(
                    torch.log(weights.clamp_min(1e-8))
                    / temperature
                )
                * eligible.to(scores.dtype)
            )
        probabilities = weights / weights.sum().clamp_min(1e-8)
        return probabilities, eligible

    def _order3_bounds(
            self,
            factor_budget,
            num_active,
            max_feasible_order3=None):
        """Return scheduled lower/upper triplet counts for one graph.

        The lower bound prevents pair-heavy collapse; the upper bound prevents
        the run26 failure mode where argmax plus quota turned nearly every
        selected factor into a triplet.  Both bounds are expressed over the
        active factor budget, not over the padded factor capacity.
        """
        if (
            self.highest_orders < 3
            or int(num_active) < 3
            or int(factor_budget) <= 0
        ):
            return 0, 0
        min_quota = int(
            np.ceil(
                float(self.current_min_order3_ratio)
                * float(factor_budget)
                - 1e-8
            )
        )
        max_quota = int(
            np.floor(
                float(self.current_max_order3_ratio)
                * float(factor_budget)
                + 1e-8
            )
        )
        min_quota = max(0, min(int(factor_budget), min_quota))
        max_quota = max(min_quota, min(int(factor_budget), max_quota))
        if self.min_pair_ratio > 0.0:
            # Keep at least a small pursuit/pair backbone.  This guard is
            # applied after the order3 band so the two constraints cannot
            # silently request more factors than the active budget allows.
            min_pair_quota = int(
                np.floor(
                    float(self.min_pair_ratio) * float(factor_budget)
                    + 1e-8
                )
            )
            min_pair_quota = max(0, min(int(factor_budget), min_pair_quota))
            max_quota = min(max_quota, int(factor_budget) - min_pair_quota)
            min_quota = min(min_quota, max_quota)
        if max_feasible_order3 is not None:
            feasible = max(0, int(max_feasible_order3))
            min_quota = min(min_quota, feasible)
            max_quota = min(max_quota, feasible)
            max_quota = max(min_quota, max_quota)
        return min_quota, max_quota

    def _apply_order3_band(
            self,
            policy_probs,
            eligible,
            candidate_nodes,
            scores,
            selected_order3,
            factor_budget,
            slot,
            num_active):
        candidate_orders = candidate_nodes.long().sum(dim=1)

        def _quality_triplet_eligible():
            triplet_mask = eligible & (candidate_orders == 3)
            if self.order3_quota_score_floor > 0.0 and bool(
                torch.any(eligible).item()
            ):
                pair_eligible = eligible & (candidate_orders == 2)
                if bool(torch.any(pair_eligible).item()):
                    score_reference = scores[pair_eligible].max()
                else:
                    score_reference = scores[eligible].max()
                score_reference = score_reference.clamp_min(1e-8)
                triplet_mask = triplet_mask & (
                    scores
                    >= score_reference
                    * float(self.order3_quota_score_floor)
                )
            return triplet_mask

        triplet_remaining = int(
            (eligible & (candidate_orders == 3)).long().sum().item()
        )
        min_quota, max_quota = self._order3_bounds(
            factor_budget,
            num_active,
            max_feasible_order3=int(selected_order3) + triplet_remaining,
        )

        if int(selected_order3) >= max_quota:
            pair_eligible = eligible & (candidate_orders == 2)
            if bool(torch.any(pair_eligible).item()):
                restricted_probs = (
                    policy_probs * pair_eligible.to(policy_probs.dtype)
                )
                restricted_sum = restricted_probs.sum()
                if float(restricted_sum.detach().cpu().item()) <= 1e-8:
                    restricted_probs = (
                        pair_eligible.to(policy_probs.dtype)
                        / pair_eligible.float().sum().clamp_min(1.0)
                    )
                else:
                    restricted_probs = (
                        restricted_probs / restricted_sum.clamp_min(1e-8)
                    )
                return restricted_probs, pair_eligible, True

        needed_order3 = max(0, min_quota - int(selected_order3))
        remaining_slots = int(factor_budget) - int(slot)
        if needed_order3 <= 0 or remaining_slots <= 0:
            return policy_probs, eligible, False
        if needed_order3 < remaining_slots:
            if (
                self.order3_quota_mode == "soft"
                and self.order3_soft_quota_coef > 0.0
            ):
                triplet_eligible = _quality_triplet_eligible()
                if bool(torch.any(triplet_eligible).item()):
                    deficit = float(needed_order3) / float(
                        max(remaining_slots, 1)
                    )
                    # Soft quota nudges selection toward quality triplets
                    # without masking out strong pairs. This directly avoids
                    # the run28/run29 conflict where hard quota kept triplets
                    # even while order3 local PPO credit was negative.
                    triplet_bonus = (
                        1.0
                        + float(self.order3_soft_quota_coef)
                        * deficit
                        * float(self.current_order3_credit_gate)
                    )
                    adjusted = policy_probs * (
                        1.0
                        + triplet_eligible.to(policy_probs.dtype)
                        * (triplet_bonus - 1.0)
                    )
                    adjusted = adjusted * eligible.to(adjusted.dtype)
                    adjusted_sum = adjusted.sum()
                    if float(adjusted_sum.detach().cpu().item()) > 1e-8:
                        adjusted = adjusted / adjusted_sum.clamp_min(1e-8)
                        return adjusted, eligible, False
            return policy_probs, eligible, False

        triplet_eligible = _quality_triplet_eligible()
        if not bool(torch.any(triplet_eligible).item()):
            return policy_probs, eligible, False
        if self.order3_quota_mode == "soft":
            adjusted = policy_probs.clone()
            triplet_bonus = (
                1.0
                + float(self.order3_soft_quota_coef)
                * float(self.current_order3_credit_gate)
            )
            adjusted = adjusted * (
                1.0
                + triplet_eligible.to(policy_probs.dtype)
                * (triplet_bonus - 1.0)
            )
            adjusted = adjusted * eligible.to(adjusted.dtype)
            adjusted_sum = adjusted.sum()
            if float(adjusted_sum.detach().cpu().item()) > 1e-8:
                adjusted = adjusted / adjusted_sum.clamp_min(1e-8)
                return adjusted, eligible, False

        restricted_probs = (
            policy_probs * triplet_eligible.to(policy_probs.dtype)
        )
        restricted_sum = restricted_probs.sum()
        if float(restricted_sum.detach().cpu().item()) <= 1e-8:
            restricted_probs = (
                triplet_eligible.to(policy_probs.dtype)
                / triplet_eligible.float().sum().clamp_min(1.0)
            )
        else:
            restricted_probs = restricted_probs / restricted_sum.clamp_min(
                1e-8
            )
        return restricted_probs, triplet_eligible, True

    def _select_candidates(
            self,
            pair_score,
            exist_mask,
            explore=False,
            t_env=None):
        """
        Build a coverage-safe sparse graph by sequential constrained sampling.

        ``prob_adj`` stores the exact conditional behavior probability for
        every selected factor. The old implementation stored a marginal
        candidate score even though factors were chosen by Gumbel Top-K plus a
        separate greedy pair cover, making the PPO importance ratio invalid.
        """
        B, N, _ = pair_score.shape
        device = pair_score.device
        prob_adj = torch.ones(
            B,
            N,
            self.num_factor,
            device=device,
            dtype=torch.float32,
        ) * 1e-8
        cond_adj = torch.zeros(
            B,
            N,
            self.num_factor,
            device=device,
            dtype=torch.int64,
        )

        _, sampling_temperature, _, _, _ = self._update_graph_schedules(t_env)
        candidate_scores, candidate_nodes = self._candidate_catalog(
            pair_score
        )
        for b in range(B):
            scores = candidate_scores[b]
            valid = scores > 0.0
            num_valid = int(valid.long().sum().item())
            num_active = int(
                (exist_mask[b] > 0.5).long().sum().item()
            )
            factor_budget = self._active_factor_budget(
                num_active=num_active,
                num_valid_candidates=num_valid,
            )
            if factor_budget <= 0:
                continue

            remaining = valid.clone()
            covered = torch.zeros(
                N,
                dtype=torch.bool,
                device=device,
            )
            active_nodes = exist_mask[b] > 0.5
            selected_order3 = 0

            for slot in range(factor_budget):
                policy_probs, eligible = self._constrained_probabilities(
                    scores=scores,
                    candidate_nodes=candidate_nodes,
                    remaining=remaining,
                    covered=covered,
                    active_nodes=active_nodes,
                    slots_after=factor_budget - slot - 1,
                    max_factor_order=self.highest_orders,
                    require_connected=self.require_connected,
                    temperature=sampling_temperature if explore else 1.0,
                )
                policy_probs, eligible, _ = self._apply_order3_band(
                    policy_probs=policy_probs,
                    eligible=eligible,
                    candidate_nodes=candidate_nodes,
                    scores=scores,
                    selected_order3=selected_order3,
                    factor_budget=factor_budget,
                    slot=slot,
                    num_active=num_active,
                )
                eligible_count = int(
                    eligible.long().sum().item()
                )
                if eligible_count <= 0:
                    break

                if explore:
                    epsilon = self.exploration_mix
                    uniform_probs = (
                        eligible.to(policy_probs.dtype)
                        / float(eligible_count)
                    )
                    behavior_probs = (
                        (1.0 - epsilon) * policy_probs
                        + epsilon * uniform_probs
                    )
                    greedy_prob = float(self.current_greedy_sample_prob)
                    greedy_idx = int(torch.argmax(policy_probs).item())
                    if self.rng.rand() < greedy_prob:
                        selected_idx = greedy_idx
                    else:
                        sample_probs = (
                            behavior_probs.detach().cpu().numpy()
                        )
                        sample_probs = np.maximum(sample_probs, 0.0)
                        sample_probs = sample_probs / max(
                            float(sample_probs.sum()),
                            1e-8,
                        )
                        selected_idx = int(
                            self.rng.choice(
                                sample_probs.shape[0],
                                p=sample_probs,
                            )
                        )
                    selected_probability = (
                        (1.0 - greedy_prob) * behavior_probs[selected_idx]
                    )
                    if selected_idx == greedy_idx:
                        selected_probability = (
                            selected_probability
                            + behavior_probs.new_tensor(greedy_prob)
                        )
                else:
                    selected_idx = int(
                        torch.argmax(policy_probs).item()
                    )
                    selected_probability = policy_probs[selected_idx]

                nodes = candidate_nodes[selected_idx]
                prob_adj[b, nodes, slot] = (
                    selected_probability.clamp_min(1e-8)
                )
                cond_adj[b, nodes, slot] = 1
                if int(nodes.long().sum().item()) == 3:
                    selected_order3 += 1
                remaining[selected_idx] = False
                covered = covered | nodes

            if (
                factor_budget > 0
                and not bool(torch.all(covered[active_nodes]).item())
            ):
                raise RuntimeError(
                    "SDDFG constrained graph selection failed to cover all "
                    "active agents"
                )

        return prob_adj, cond_adj

    def sample(self, obs, rnn_obs, use_adj_init, dones, explore=False, t_env=None):
        """
        输出保持原接口:
            prob_adj: [B, N, num_factor]
            cond_adj: [B, N, num_factor]
            entropy:  [B, N]

        当 highest_orders >= 3 时，会同时候选二阶 pair 与三阶 hyperedge。
        """
        rnn_obs, exist_mask = self._prepare_inputs(rnn_obs, dones)
        self._update_graph_schedules(t_env)

        A = self.gat(rnn_obs, exist_mask)
        pair_score, _ = self._pair_score(A, exist_mask)

        # Training uses an epsilon mixture over the exact constrained policy;
        # the selected conditional behavior probability is stored for PPO.
        prob_adj, cond_adj = self._select_candidates(
            pair_score=pair_score,
            exist_mask=exist_mask,
            explore=explore,
            t_env=t_env,
        )

        entropy = self._candidate_entropy(pair_score, exist_mask)
        return prob_adj, cond_adj, entropy

    def evaluate_prob(self, obs, rnn_obs, use_adj_init, dones, adj, t_env=None):
        """
        评估 buffer 中旧 adj 在当前 GAT 参数下的概率。
        train_adj_on_batch 中必须使用这个函数，而不是重新 sample 当前图。

        adj: [B, N, F]
        return:
            prob_adj: [B, N, F]，与旧 adj 的 factor 槽位严格对齐
            entropy:  [B, N]
        """
        rnn_obs, exist_mask = self._prepare_inputs(rnn_obs, dones)
        _, sampling_temperature, _, _, _ = self._update_graph_schedules(t_env)

        adj_t = to_torch(adj).to(self.device)
        if adj_t.dim() == 2:
            adj_t = adj_t.unsqueeze(0)
        adj_t = adj_t[:, :, :self.num_factor]

        B, N, F_total = adj_t.shape

        A = self.gat(rnn_obs, exist_mask)
        pair_score, _ = self._pair_score(A, exist_mask)

        candidate_scores, candidate_nodes = self._candidate_catalog(
            pair_score
        )
        prob_adj = torch.ones(
            B,
            N,
            F_total,
            device=self.device,
            dtype=torch.float32,
        )
        eps = 1e-8

        for b in range(B):
            scores = candidate_scores[b]
            remaining = scores > 0.0
            covered = torch.zeros(
                N,
                dtype=torch.bool,
                device=self.device,
            )
            active_nodes = exist_mask[b] > 0.5
            factor_budget = self._active_factor_budget(
                num_active=int(active_nodes.long().sum().item()),
                num_valid_candidates=int(
                    remaining.long().sum().item()
                ),
            )
            selected_order3 = 0
            num_active = int(active_nodes.long().sum().item())

            for f in range(F_total):
                selected_nodes = adj_t[b, :, f] > 0.5
                n_nodes = int(selected_nodes.long().sum().item())
                if n_nodes == 0:
                    continue
                if n_nodes == 1:
                    # Compatibility for a legacy self factor. Dynamic SDDFG
                    # emits only pair/triplet factors.
                    node_idx = torch.where(selected_nodes)[0][0]
                    prob_adj[b, node_idx, f] = (
                        exist_mask[b, node_idx].clamp_min(eps)
                    )
                    continue

                policy_probs, eligible = self._constrained_probabilities(
                    scores=scores,
                    candidate_nodes=candidate_nodes,
                    remaining=remaining,
                    covered=covered,
                    active_nodes=active_nodes,
                    slots_after=max(factor_budget - f - 1, 0),
                    max_factor_order=self.highest_orders,
                    require_connected=self.require_connected,
                    temperature=sampling_temperature,
                )
                policy_probs, eligible, _ = self._apply_order3_band(
                    policy_probs=policy_probs,
                    eligible=eligible,
                    candidate_nodes=candidate_nodes,
                    scores=scores,
                    selected_order3=selected_order3,
                    factor_budget=factor_budget,
                    slot=f,
                    num_active=num_active,
                )
                if self.exploration_mix > 0.0:
                    eligible_count = eligible.float().sum().clamp_min(1.0)
                    uniform_probs = (
                        eligible.to(policy_probs.dtype) / eligible_count
                    )
                    policy_probs = (
                        (1.0 - self.exploration_mix) * policy_probs
                        + self.exploration_mix * uniform_probs
                    )
                greedy_prob = float(self.current_greedy_sample_prob)
                if greedy_prob > 0.0:
                    greedy_idx = int(torch.argmax(policy_probs).item())
                    greedy_mass = torch.zeros_like(policy_probs)
                    greedy_mass[greedy_idx] = greedy_prob
                    policy_probs = (
                        (1.0 - greedy_prob) * policy_probs
                        + greedy_mass
                    )
                membership_match = torch.all(
                    candidate_nodes
                    == selected_nodes.unsqueeze(0),
                    dim=1,
                )
                membership_match = membership_match & remaining
                matched = torch.where(membership_match)[0]
                if matched.numel() == 0:
                    # Keep a finite diagnostic value for a legacy/corrupted
                    # factor. The trainer validity mask excludes it.
                    prob_adj[b, selected_nodes, f] = eps
                    continue

                selected_idx = int(matched[0].item())
                selected_probability = policy_probs[
                    selected_idx
                ].clamp_min(eps)
                prob_adj[b, selected_nodes, f] = selected_probability
                if n_nodes == 3:
                    selected_order3 += 1
                remaining[selected_idx] = False
                covered = covered | candidate_nodes[selected_idx]

        entropy = self._candidate_entropy(pair_score, exist_mask)
        return prob_adj, entropy

    def load_state(self, source_adjnetwork):
        self.load_state_dict(source_adjnetwork.state_dict())
