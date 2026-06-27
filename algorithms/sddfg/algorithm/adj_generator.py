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
        self.exploration_mix = max(
            0.0,
            min(1.0, float(getattr(args, "adj_exploration_mix", 0.0))),
        )
        self._triplet_cache = {}

        self.rng = np.random.RandomState(int(getattr(args, "seed", 0)) + 520000)

        gat_heads = args.gat_heads
        gat_slope = args.gat_negative_slope

        self.gat = _GATLayer(
            in_dim=args.hidden_size,
            num_heads=gat_heads,
            negative_slope=gat_slope
        )

        # Three pair scores [s_ij, s_ik, s_jk] produce a bounded residual
        # multiplier around one for the corresponding third-order factor.
        hyperedge_hidden = int(getattr(args, "gat_hyperedge_hidden", getattr(args, "adj_hidden_dim", 32)))
        self.hyperedge_scorer = nn.Sequential(
            nn.Linear(3, hyperedge_hidden),
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

        gate_logits = self.hyperedge_scorer(
            feats.reshape(-1, 3)
        ).reshape(B, -1)
        multiplier = torch.exp(0.5 * torch.tanh(gate_logits))

        # Stay on the pair-score scale while allowing learned promotion or
        # suppression of cooperative third-order interactions.
        triplet_score = feats.mean(dim=-1) * multiplier
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
            max_factor_order=3):
        """
        Conditional distribution for one factor slot.

        While some active agents are uncovered, candidates must add at least
        one new node and their policy weight is multiplied by that coverage
        gain. Once coverage is complete, all remaining candidates compete.
        This keeps pair and triplet choices learnable while still guaranteeing
        coverage under the pair-cover factor budget.
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
            coverage_weight = gains.to(scores.dtype)
        else:
            eligible = valid
            coverage_weight = torch.ones_like(scores)

        weights = (
            scores
            * coverage_weight
            * eligible.to(scores.dtype)
        )
        probabilities = weights / weights.sum().clamp_min(1e-8)
        return probabilities, eligible

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

            for slot in range(factor_budget):
                policy_probs, eligible = self._constrained_probabilities(
                    scores=scores,
                    candidate_nodes=candidate_nodes,
                    remaining=remaining,
                    covered=covered,
                    active_nodes=active_nodes,
                    slots_after=factor_budget - slot - 1,
                    max_factor_order=self.highest_orders,
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
                    sample_probs = behavior_probs.detach().cpu().numpy()
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
                    selected_probability = behavior_probs[selected_idx]
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

                policy_probs, _ = self._constrained_probabilities(
                    scores=scores,
                    candidate_nodes=candidate_nodes,
                    remaining=remaining,
                    covered=covered,
                    active_nodes=active_nodes,
                    slots_after=max(factor_budget - f - 1, 0),
                    max_factor_order=self.highest_orders,
                )
                if self.exploration_mix > 0.0:
                    eligible = policy_probs > 0.0
                    eligible_count = eligible.float().sum().clamp_min(1.0)
                    uniform_probs = (
                        eligible.to(policy_probs.dtype) / eligible_count
                    )
                    policy_probs = (
                        (1.0 - self.exploration_mix) * policy_probs
                        + self.exploration_mix * uniform_probs
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
                remaining[selected_idx] = False
                covered = covered | candidate_nodes[selected_idx]

        entropy = self._candidate_entropy(pair_score, exist_mask)
        return prob_adj, entropy

    def load_state(self, source_adjnetwork):
        self.load_state_dict(source_adjnetwork.state_dict())
