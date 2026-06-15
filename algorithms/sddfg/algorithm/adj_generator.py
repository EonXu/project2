import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from utils.util import DecayThenFlatSchedule, to_torch


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
        self._triplet_cache = {}

        gat_heads = args.gat_heads
        gat_slope = args.gat_negative_slope

        self.gat = _GATLayer(
            in_dim=args.hidden_size,
            num_heads=gat_heads,
            negative_slope=gat_slope
        )

        # ===== 修改点：三阶超边 scorer =====
        # 输入为三条 pair score: [s_ij, s_ik, s_jk]
        # 输出为 gate，最终 hyperedge_score = mean(pair_scores) * sigmoid(gate)
        hyperedge_hidden = int(getattr(args, "gat_hyperedge_hidden", getattr(args, "adj_hidden_dim", 32)))
        self.hyperedge_scorer = nn.Sequential(
            nn.Linear(3, hyperedge_hidden),
            nn.LeakyReLU(gat_slope),
            nn.Linear(hyperedge_hidden, 1)
        )

        self.exploration = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.adj_anneal_time,
            decay="linear"
        )

        self.to(device)

    def _prepare_inputs(self, rnn_obs, dones):
        rnn_obs = to_torch(rnn_obs).to(self.device)
        if len(rnn_obs.shape) == 2:
            rnn_obs = rnn_obs.unsqueeze(0)

        B, N, _ = rnn_obs.shape

        if dones is not None:
            dones_t = to_torch(dones).to(self.device)
            dones_t = dones_t.reshape(B, N, -1)[..., 0]
            exist_mask = (1.0 - dones_t.float()).clamp(0.0, 1.0)
        else:
            exist_mask = torch.ones(B, N, device=self.device, dtype=torch.float32)

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

        feats = torch.stack([sij, sik, sjk], dim=-1)  # [B, T, 3]
        valid = (feats > 0.0).all(dim=-1)

        gate_logits = self.hyperedge_scorer(feats.reshape(-1, 3)).reshape(B, -1)
        gate = torch.sigmoid(gate_logits)

        # 与 pair score 保持同一数量级，避免三阶候选因 sigmoid 初值过大而压倒二阶边
        triplet_score = feats.mean(dim=-1) * gate
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

        has_candidate = score_sum > 1e-8
        entropy_scalar = torch.where(has_candidate, entropy_scalar, torch.zeros_like(entropy_scalar))

        # 广播到 agent 维，保持原 trainer 的 entropy 接口 [B, N]
        entropy = entropy_scalar * exist_mask
        return entropy

    def _select_candidates(self, pair_score, exist_mask, explore=False, t_env=None):
        """
        从二阶 pair 与三阶 hyperedge 候选中统一 TopK。
        return:
            prob_adj: [B, N, F]
            cond_adj: [B, N, F]
        """
        B, N, _ = pair_score.shape
        F_total = self.num_factor
        device = pair_score.device

        prob_adj = torch.ones(B, N, F_total, device=device, dtype=torch.float32) * 1e-8
        cond_adj = torch.zeros(B, N, F_total, device=device, dtype=torch.int64)

        pair_i, pair_j = torch.triu_indices(N, N, offset=1, device=device)
        pair_scores = pair_score[:, pair_i, pair_j]  # [B, P]
        num_pairs = pair_scores.shape[1]

        triplets = self._triplet_indices(N, device)
        if self.highest_orders >= 3:
            triplet_scores = self._score_triplets(pair_score, triplets)  # [B, T]
        else:
            triplet_scores = torch.empty(B, 0, dtype=pair_score.dtype, device=device)

        for b in range(B):
            candidate_scores = [pair_scores[b]]
            candidate_kinds = [
                torch.zeros(num_pairs, dtype=torch.long, device=device)  # 0 = pair
            ]
            candidate_local_idx = [
                torch.arange(num_pairs, dtype=torch.long, device=device)
            ]

            if self.highest_orders >= 3 and triplet_scores.shape[1] > 0:
                num_triplets = triplet_scores.shape[1]
                candidate_scores.append(triplet_scores[b])
                candidate_kinds.append(torch.ones(num_triplets, dtype=torch.long, device=device))  # 1 = triplet
                candidate_local_idx.append(torch.arange(num_triplets, dtype=torch.long, device=device))

            all_scores = torch.cat(candidate_scores, dim=0)
            all_kinds = torch.cat(candidate_kinds, dim=0)
            all_local_idx = torch.cat(candidate_local_idx, dim=0)

            valid = all_scores > 0.0
            if not torch.any(valid):
                continue

            valid_scores = all_scores[valid]
            valid_kinds = all_kinds[valid]
            valid_local_idx = all_local_idx[valid]

            k_select = min(F_total, valid_scores.numel())
            topk_vals, topk_order = torch.topk(valid_scores, k=k_select, dim=0)

            for f in range(k_select):
                kind = int(valid_kinds[topk_order[f]].item())
                local_idx = int(valid_local_idx[topk_order[f]].item())
                p = topk_vals[f].clamp_min(1e-8)

                if kind == 0:
                    # 二阶 factor: {i, j}
                    i = pair_i[local_idx]
                    j = pair_j[local_idx]
                    nodes = torch.stack([i, j], dim=0)
                else:
                    # 三阶 factor: {i, j, k}
                    nodes = triplets[local_idx]

                prob_adj[b, nodes, f] = p
                cond_adj[b, nodes, f] = 1

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

        A = self.gat(rnn_obs, exist_mask)
        pair_score, pair_mask = self._pair_score(A, exist_mask)

        # 探索噪声只作用于合法 pair；三阶分数会基于加噪后的 pair_score 计算
        if explore:
            eps = float(self.exploration.eval(t_env if t_env is not None else 0))
            if eps > 0:
                noise = torch.rand_like(pair_score) * eps
                pair_score = (pair_score + noise) * pair_mask.to(pair_score.dtype)
                pair_score = pair_score / pair_score.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                pair_score = pair_score * pair_mask.to(pair_score.dtype)

        prob_adj, cond_adj = self._select_candidates(
            pair_score=pair_score,
            exist_mask=exist_mask,
            explore=explore,
            t_env=t_env
        )

        entropy = self._candidate_entropy(pair_score, exist_mask)
        return prob_adj, cond_adj, entropy

    def evaluate_prob(self, obs, rnn_obs, use_adj_init, dones, adj):
        """
        评估 buffer 中旧 adj 在当前 GAT 参数下的概率。
        train_adj_on_batch 中必须使用这个函数，而不是重新 sample 当前图。

        adj: [B, N, F]
        return:
            prob_adj: [B, N, F]，与旧 adj 的 factor 槽位严格对齐
            entropy:  [B, N]
        """
        rnn_obs, exist_mask = self._prepare_inputs(rnn_obs, dones)

        adj_t = to_torch(adj).to(self.device)
        if adj_t.dim() == 2:
            adj_t = adj_t.unsqueeze(0)
        adj_t = adj_t[:, :, :self.num_factor]

        B, N, F_total = adj_t.shape

        A = self.gat(rnn_obs, exist_mask)
        pair_score, _ = self._pair_score(A, exist_mask)

        prob_adj = torch.ones(B, N, F_total, device=self.device, dtype=torch.float32)
        eps = 1e-8

        for b in range(B):
            for f in range(F_total):
                nodes = torch.where(adj_t[b, :, f] > 0.5)[0]
                n_nodes = int(nodes.numel())

                if n_nodes == 2:
                    i, j = nodes[0], nodes[1]
                    p = pair_score[b, i, j].clamp_min(eps)
                    prob_adj[b, nodes, f] = p

                elif n_nodes == 3:
                    i, j, k = nodes[0], nodes[1], nodes[2]
                    feats = torch.stack([
                        pair_score[b, i, j],
                        pair_score[b, i, k],
                        pair_score[b, j, k],
                    ], dim=0).view(1, 3)

                    valid = bool(torch.all(feats > 0.0).item())
                    if valid:
                        gate = torch.sigmoid(self.hyperedge_scorer(feats)).view(())
                        p = (feats.mean() * gate).clamp_min(eps)
                    else:
                        p = torch.tensor(eps, dtype=torch.float32, device=self.device)

                    prob_adj[b, nodes, f] = p

                elif n_nodes == 1:
                    # 兼容异常旧 buffer；正常 GAT 动态 factor 不应是一阶，一阶 self-factor 由 runner/trainer 追加。
                    i = nodes[0]
                    p = exist_mask[b, i].clamp_min(eps)
                    prob_adj[b, i, f] = p

                else:
                    # 空 factor 或异常 factor：保持 1，trainer 中非 adj==1 位置会被忽略。
                    continue

        entropy = self._candidate_entropy(pair_score, exist_mask)
        return prob_adj, entropy

    def load_state(self, source_adjnetwork):
        self.load_state_dict(source_adjnetwork.state_dict())