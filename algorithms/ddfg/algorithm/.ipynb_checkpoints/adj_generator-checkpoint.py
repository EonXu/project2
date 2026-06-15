# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# 这些工具在你的工程中已存在（学习率调度、epsilon 退火等）
from utils.util import DecayThenFlatSchedule


class _GATLayer(nn.Module):
    """
    一个最小可用的多头 GAT 层（单层注意力，不做消息更新，只产出注意力权 A）。
    用于从节点隐向量 H \in R^{B×N×d} 计算注意力矩阵 A \in R^{B×N×N}。

    - 多头：对每个 head 计算 e_ij，再在 head 维上取平均作为最终 A。
    - Mask：对不存在的智能体（exist_mask==0）的行/列置 -inf，防止被选中。
    """
    def __init__(self, in_dim: int, num_heads: int = 4, negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.negative_slope = negative_slope

        # 这里简化处理：head 输出维度 = in_dim // num_heads
        head_dim = max(1, in_dim // num_heads)
        self.proj = nn.Linear(in_dim, num_heads * head_dim, bias=False)
        # 每个 head 各有一组注意力参数 a_h \in R^{2*head_dim}
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 2 * head_dim))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_vec)

        self.head_dim = head_dim

    def forward(self, H: torch.Tensor, exist_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            H: [B, N, d]
            exist_mask: [B, N] in {0,1}，表示该槽位是否有真实 agent（可选）
        Returns:
            A: [B, N, N]，各行 softmax 后的注意力权重（已做 mask，主对角为 0）
        """
        B, N, _ = H.shape
        x = self.proj(H)                      # [B, N, H*num_heads]
        x = x.view(B, N, self.num_heads, self.head_dim)   # [B, N, H, D]

        # x_i 与 x_j 的拼接：[B, N, N, H, 2D]
        # 广播构造 pair (i,j)
        xi = x.unsqueeze(2)                   # [B, 1, N, H, D] -> [B, N, N, H, D]
        xj = x.unsqueeze(1)                   # [B, N, 1, H, D] -> [B, N, N, H, D]
        xij = torch.cat([xi, xj], dim=-1)     # [B, N, N, H, 2D]

        # e_ij^h = LeakyReLU( a_h^T [W h_i || W h_j] )
        # attn_vec: [H, 2D]，与 xij: [B,N,N,H,2D] 做逐 head 点积
        e = F.leaky_relu(torch.einsum('hd, b n m h d -> b n m h', self.attn_vec, xij),
                         negative_slope=self.negative_slope)  # [B, N, N, H]

        # Mask：不存在的 i 或 j，置为 -inf（不会被 softmax 选中）
        if exist_mask is not None:
            emask_i = exist_mask.unsqueeze(2).unsqueeze(-1)  # [B, N, 1, 1]
            emask_j = exist_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            mask = (emask_i * emask_j).bool()                # [B, N, N, 1]
            e = e.masked_fill(~mask, float('-inf'))

        # 去掉自环（对角线）——一阶因子我们单独以单位阵保留
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=H.device).view(1, N, N, 1)
        e = e.masked_fill(diag_mask.logical_not(), float('-inf'))

        # 对 j 维 softmax：得到每个 i 的邻接注意力分布
        alpha = torch.softmax(e, dim=2)       # [B, N, N, H]
        # 多头平均
        A = alpha.mean(dim=-1)                # [B, N, N]

        # 数值稳定：把对角线置 0
        eye = torch.eye(N, device=H.device).unsqueeze(0)
        A = A * (1.0 - eye)

        return A


class Adj_Generator(nn.Module):
    """
    改进版邻接生成器（GAT 驱动）
    ------------------------------------------------------------
    1) [改进] 使用 GAT 从 RNN 节点隐向量中学习注意力 A \in R^{B×N×N}
    2) [改进] 从注意力中选择 Top-K 二阶边，写入因子-节点 incidence（与原 DDFG 因子/阶数接口完全兼容）
    3) [改进] 支持回合内可变 N：通过 exist_mask 屏蔽无效槽位，固定上界 N_max + mask 训练
    """
    def __init__(self, args, obs_dim, state_dim, act_dim, device):
        super().__init__()
        self.args = args
        self.device = device

        # --- 维度与超参 ---
        self.N_max = args.num_agents                    # 固定上界 N_max
        self.F_extra = getattr(args, "num_factor", 8)   # 额外因子个数（用于二/三阶）
        self.highest_orders = getattr(args, "highest_orders", 2)
        self.hidden = args.hidden_size                  # 与 RNN 输出维度一致

        # [改进] GAT 超参
        self.gat_heads = getattr(args, "gat_heads", 4)
        self.topk_per_node = getattr(args, "gat_topk", 2)   # 每个节点选 K 条边
        self.global_max_edges = getattr(args, "gat_max_edges", self.F_extra)  # 全局最多选多少条边/因子

        # [改进] 创建 GAT 层
        self.gat = _GATLayer(in_dim=self.hidden, num_heads=self.gat_heads)

        # 探索退火（与旧实现保持风格一致）：用于控制是否做随机扰动（如 epsilon）
        self.exploration = DecayThenFlatSchedule(
            getattr(args, "epsilon_start", 0.5),
            getattr(args, "epsilon_finish", 0.05),
            getattr(args, "adj_anneal_time", 50000),
            decay="linear"
        )

    # ---------------------- 工具函数 ----------------------
    @staticmethod
    def _row_entropy(P: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """对每个 i 的分布 P[i,*] 计算熵，并对有效行求均值。
        P: [B, N, N], mask: [B, N] (行有效性)
        return: [B] 每个 batch 的平均熵
        """
        eps = 1e-8
        logp = torch.log(P.clamp_min(eps))
        ent = -(P * logp).sum(dim=-1)  # [B, N]
        if mask is not None:
            ent = (ent * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        else:
            ent = ent.mean(dim=-1)
        return ent

    @staticmethod
    def _unique_sorted_edges(edges_ij_with_p, max_edges):
        """将 (i,j,p) 的列表去重（无向边 i<j），按 p 降序截断到 max_edges。"""
        uniq = {}
        for (i, j, p) in edges_ij_with_p:
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key not in uniq or p > uniq[key]:
                uniq[key] = p
        # 排序并截断
        items = sorted([(k[0], k[1], v) for k, v in uniq.items()], key=lambda x: float(x[2]), reverse=True)
        return items[:max_edges]

    def _pick_pairwise_factors(self, A: torch.Tensor, exist_mask: torch.Tensor):
        """
        从注意力 A 中选择二阶边作为因子，返回每个 batch 的边列表 [(i,j,p), ...]
        - 策略：每个节点 TopK + 全局去重 + 截断到 global_max_edges
        """
        B, N, _ = A.shape
        picked = []
        for b in range(B):
            edges = []
            valid = exist_mask[b].bool() if exist_mask is not None else torch.ones(N, dtype=torch.bool, device=A.device)
            valid_idx = torch.where(valid)[0]
            for i in valid_idx.tolist():
                # 对每个 i，从有效 j 中取 topk
                row = A[b, i]                                    # [N]
                row = row.masked_fill(~valid, float('-inf'))     # 只考虑有效 j
                k = min(self.topk_per_node, int(valid.sum().item()) - 1)  # 排除自身后的可选上限
                if k <= 0:
                    continue
                vals, js = torch.topk(row, k=k)
                for v, j in zip(vals.tolist(), js.tolist()):
                    if j == i:
                        continue
                    if v == float('-inf') or v != v:  # nan/inf
                        continue
                    edges.append((i, j, v))
            # 去重 + 全局截断
            edges = self._unique_sorted_edges(edges, max_edges=self.global_max_edges)
            picked.append(edges)
        return picked  # List[List[(i,j,p)]], len==B

    # ---------------------- 核心接口 ----------------------
    def sample(self, obs, rnn_obs, use_adj_init=False, dones=None,
               explore: bool = False, t_env: int = None, exist_mask: torch.Tensor = None):
        """
        生成邻接（因子-节点 incidence）
        Args:
            obs:  [B, N, obs_dim]（未直接用；如需可在此拼接到 H）
            rnn_obs: [B, N, hidden]（来自 RNN 的节点隐向量）
            use_adj_init: 兼容旧接口（此版不使用该开关）
            dones: [B, 1] 或 [B]，可选
            explore: 是否加探索（此版使用 epsilon 对注意力行加噪）
            t_env: 当前环境步数（用于 epsilon 退火）
            exist_mask: [B, N] in {0,1}，表示该槽位是否有 agent（回合内变 N 时必传）
        Returns:
            prob_adj: [B, N, F]（软概率；前 N 是 unary，后 F_extra 是 pairwise）
            cond_adj: [B, N, F]（0/1 的因子-节点矩阵）
            entropy:  [B]      （来自注意力行分布的平均熵）
        """
        # ------ 准备张量与遮罩 ------
        H = rnn_obs                                  # [B, N, hidden]
        B, N, _ = H.shape
        assert N == self.N_max, "rnn_obs 的 N 维应等于固定上界 N_max（用 mask 指示实际存在的 agent）"

        if exist_mask is None:
            exist_mask = torch.ones(B, N, device=H.device, dtype=torch.float32)
        exist_mask_bool = exist_mask.bool()

        # ------ [改进] 用 GAT 得到注意力 A ------
        A = self.gat(H, exist_mask_bool)             # [B, N, N]

        # 探索（可选）：对每行加入少量均匀噪声并重新归一化
        if explore:
            eps = float(self.exploration.eval(t_env if t_env is not None else 0))
            if eps > 0:
                noise = torch.rand_like(A) * eps
                A = A + noise
                A = A / (A.sum(dim=-1, keepdim=True).clamp_min(1e-6))

        # 记录熵（用于正则/日志）
        entropy = self._row_entropy(A, mask=exist_mask)

        # ------ [改进] 从 A 中选择二阶因子（边） ------
        picked_edges = self._pick_pairwise_factors(A, exist_mask_bool)
        F_total = self.N_max + self.F_extra          # 因子总数：N 个一阶 + 额外二/三阶槽位

        # 初始化因子-节点矩阵
        prob_adj = torch.zeros(B, N, F_total, device=H.device, dtype=torch.float32)
        cond_adj = torch.zeros(B, N, F_total, device=H.device, dtype=torch.float32)

        # 一阶因子：对存在的节点，恒等关联；对不存在的节点，保持 0
        for i in range(self.N_max):
            prob_adj[:, i, i] = exist_mask[:, i]
            cond_adj[:, i, i] = exist_mask[:, i]

        # 二阶因子：把选中的边写入 N..N+F_extra-1 的槽位
        for b in range(B):
            for k, (i, j, p) in enumerate(picked_edges[b][:self.F_extra]):
                f = self.N_max + k
                prob_adj[b, i, f] = float(p)
                prob_adj[b, j, f] = float(p)
                cond_adj[b, i, f] = 1.0
                cond_adj[b, j, f] = 1.0

        # 若 highest_orders==1，则只保留一阶（此处已自然满足）；若==2，已填二阶；==3 可在此处扩展三阶逻辑
        # （三阶可通过对 (i,j,k) 的三元评分 top-T，并把三元因子写到 F 槽位，方法与二阶类似）

        return prob_adj, cond_adj, entropy

    # ---------------------- 训练接口 ----------------------
    def parameters(self):
        """供外部优化器收集参数。"""
        return list(self.gat.parameters())

    def load_state(self, source_adjnetwork):
        """从旧的邻接网络权重中装载（兼容旧接口）。"""
        try:
            self.load_state_dict(source_adjnetwork.state_dict(), strict=False)
        except Exception:
            # 如果旧模型是老版 AdjPolicy 结构，这里允许跳过
            pass
