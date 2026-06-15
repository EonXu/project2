import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import init, to_torch

"""
1.mlp_obs:Linear(obs_dim, self.hidden_dim)
2.hyper_w1:
    nn.Linear(hidden_dim * num_variable, hidden_dim * hidden_dim // 2)
    nn.ReLU(),
    nn.Linear(hidden_dim * hidden_dim // 2, hidden_dim // 2)
3.hyper_b1:
    nn.Linear(hidden_dim * num_variable, num_variable * hidden_dim // 2)
4.hidden_layer:
    torch.tanh(torch.matmul(rnn_obs, w1) + b1)
5.hyper_w2:
    nn.Linear(hidden_dim * num_variable, num_factor * hidden_dim // 2)
    nn.ReLU(),
    nn.Linear(num_factor * hidden_dim // 2, num_factor)
6.hyper_b2:
    nn.Linear(hidden_dim * num_variable, num_variable * hidden_dim // 2)
    nn.ReLU(),
    nn.Linear(num_variable * hidden_dim // 2, num_factor)
7.torch.matmul(hidden_layer, w2) + b2
8.softmax(out_pre, dim=1)
           ┌─────────────┐
           │  obs [B, obs_dim]  │  <-- 每个智能体的观测
           └───────┬─────┘
                   │
             ┌─────▼─────┐
             │  mlp_obs  │  Linear(obs_dim -> hidden_dim)
             └─────┬─────┘
                   │
        ┌──────────▼──────────┐
        │ h_obs [B, hidden_dim] │
        └──────────┬──────────┘
                   │
        ┌──────────▼─────────────┐
        │ global_obs [B, hidden_dim * num_variable] │
        └──────────┬─────────────┘
                   │
        ┌──────────▼─────────────┐
        │  Hypernet w1 & w2      │  <-- 输入 global_obs
        │  Hypernet b1 & b2      │
        └─────┬───────────┬─────┘
              │           │
        w1 [B, hidden_dim, hidden_dim/2]
        b1 [B, num_variable, hidden_dim/2]
        w2 [B, hidden_dim/2, num_factor]
        b2 [B, num_variable, num_factor]
              │           │
              └──────┬────┘
                     │
             ┌───────▼────────┐
             │ hidden_layer   │
             │ tanh(rnn_obs @ w1 + b1) │
             │ shape: [B, num_variable, hidden_dim/2] │
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ output_layer   │
             │ hidden_layer @ w2 + b2 │
             │ shape: [B, num_variable, num_factor] │
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ softmax / log_softmax │
             │  -> out & log_probs  │
             └───────────────────┘

Legend:
B = batch_size
obs_dim = 每个智能体的观测维度
hidden_dim = mlp隐藏层维度
num_variable = 智能体数量
num_factor = Q_tot聚合因子数量

"""


class AdjPolicy(nn.Module):
    """
    用于计算多智能体环境中Q值的总和（Q_tot）。
    通过各智能体的观测值和RNN输出，结合超网络生成的权重，得到总Q值。
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """

    def __init__(self, args, obs_dim, hidden_dim, device, use_ReLU):
        super(AdjPolicy, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.num_variable = args.max_player_num  # 智能体数量
        self.num_factor = args.num_factor  # 因子数量（用于聚合Q值）
        self._use_orthogonal = args.use_orthogonal  # 是否用正交初始化
        self.gain = args.gain

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        def act_init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=self.gain)

        # 2 layer hypernets: output dimensions are same as above case
        # nn.ReLU()

        # ---------- 超网络Hypernet ----------
        # 用于生成权重矩阵w1
        self.hyper_input_dim = self.hidden_dim * self.num_variable  # 超网络输入维度

        '''self.hyper_obs = Hypernet(
            input_dim=obs_dim, hidden_dim=self.hidden_dim,
            main_input_dim=obs_dim, main_output_dim=self.hidden_dim
        )'''
        # 将观测值obs映射到隐藏表示
        self.mlp_obs = init_(nn.Linear(obs_dim, self.hidden_dim))

        self.hyper_w1 = Hypernet(
            input_dim=self.hyper_input_dim, hidden_dim=self.hidden_dim * self.hidden_dim // 2,
            main_input_dim=self.hidden_dim, main_output_dim=self.hidden_dim // 2
        )

        self.hyper_w2 = Hypernet(
            input_dim=self.hyper_input_dim, hidden_dim=self.num_factor * self.hidden_dim // 2,
            main_input_dim=self.hidden_dim // 2, main_output_dim=self.num_factor
        )

        # 偏置b1
        self.hyper_b1 = init_(nn.Linear(self.hyper_input_dim, self.num_variable * self.hidden_dim // 2))

        # 偏置b2
        self.hyper_b2 = Hypernet(
            input_dim=self.hyper_input_dim, hidden_dim=self.num_variable * self.hidden_dim // 2,
            main_input_dim=self.num_variable, main_output_dim=self.num_factor
        )

        # 批归一化
        self.bn1 = nn.BatchNorm1d(self.num_variable)

    def forward(self, obs, rnn_obs, use_adj_init, dones):
        """
        Mask-safe AdjPolicy forward.

        obs:     [B, N, obs_dim]
        rnn_obs: [B, N, hidden_dim]
        dones:   [B, N] / [B, N, 1]，True=死亡/空槽
        return:
            out:       [B, N, F]
            log_probs: [B, N, F]
        """
        batch_size = obs.size(0)

        dones = to_torch(dones).to(self.device).bool()
        if dones.dim() == 3:
            dones = dones[..., 0]
        elif dones.dim() != 2:
            dones = dones.reshape(batch_size, self.num_variable)

        active_masks = (~dones).float()  # [B, N]
        active_masks_3d = active_masks.unsqueeze(-1)  # [B, N, 1]

        # ===== 修改点 1：死亡/空槽 obs 与 rnn_obs 先清零，避免污染 hypernet 的 global_obs =====
        obs = to_torch(obs).to(**self.tpdv) * active_masks_3d
        rnn_obs = to_torch(rnn_obs).to(**self.tpdv) * active_masks_3d

        h_obs = self.mlp_obs(obs) * active_masks_3d

        # 死亡 agent 的 h_obs 已清零，因此 global_obs 不再携带 -1 padding 信息
        global_obs = h_obs.reshape(batch_size, -1)

        w1 = self.hyper_w1(global_obs)
        b1 = self.hyper_b1(global_obs).reshape(batch_size, self.num_variable, -1)
        hidden_layer = torch.tanh(torch.matmul(rnn_obs, w1) + b1)
        hidden_layer = hidden_layer * active_masks_3d

        w2 = self.hyper_w2(global_obs)
        b2 = self.hyper_b2(global_obs).reshape(batch_size, self.num_variable, -1)

        if use_adj_init:
            out_raw = torch.matmul(hidden_layer, w2 * 0.01) + b2 - b2.detach()
        else:
            out_raw = torch.matmul(hidden_layer, w2) + b2

        # ===== 修改点 2：logits 层 mask 仍保留，但后面必须二次清零与重归一化 =====
        out_pre = out_raw.masked_fill(dones.unsqueeze(-1), -1e10)

        out = F.softmax(out_pre, dim=1)
        out = out * active_masks_3d

        # ===== 修改点 3：softmax 后按 agent 维重新归一化；全死亡/全无效 factor 保持全 0 =====
        denom = out.sum(dim=1, keepdim=True).clamp_min(1e-8)
        out = out / denom
        out = out * active_masks_3d

        log_probs = torch.log(out.clamp_min(1e-8))

        return out, log_probs


class Hypernet(nn.Module):
    """
   超网络：根据输入生成另一网络的权重矩阵。
   用于动态生成AdjPolicy中的权重。
   """


    def __init__(self, input_dim, hidden_dim, main_input_dim, main_output_dim):
        super(Hypernet, self).__init__()

        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim
        init_method = nn.init.orthogonal_

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 两层MLP：input -> hidden -> output
        self.multihead_nn = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(hidden_dim, output_dim)),
            )


    def forward(self, x):
        # [...,  main_output_dim + main_output_dim + ... + main_output_dim]
        # [bs, main_input_dim, n_heads * main_output_dim]
        # import pdb;pdb.set_trace()
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.main_output_dim])
