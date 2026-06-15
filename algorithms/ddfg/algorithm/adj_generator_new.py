import sys

sys.path.append('..')
import torch
import torch.nn as nn
from utils.util import gumbel_softmax_mdfg, to_torch, update_linear_schedule, DecayThenFlatSchedule
from algorithms.utils.adj_policy import AdjPolicy

# Network Generator
class Adj_Generator(nn.Module):
    def __init__(self, args, obs_dim, state_dim, act_dim, device):
        super(Adj_Generator, self).__init__()

        # 从 args 中读取邻接网络相关的超参数（隐藏尺寸 / 输出尺寸）
        self.adj_hidden_dim = args.adj_hidden_dim
        self.adj_output_dim = args.adj_output_dim
        # 是否在采样时使用 epsilon-greedy
        self.use_epsilon_greedy = args.use_epsilon_greedy
        # 变量数（通常为智能体数量）
        self.num_variable = args.max_player_num
        # 因子数量（图中因子节点的数目）
        self.num_factor = args.num_factor
        # alpha 参数（在某些算法中用于平滑或正则）
        self.alpha = args.adj_alpha
        # 一个内部计数器
        self.num = 1
        self.device = device
        self.args = args
        # 根据是否将上一步动作作为输入到 RNN，设置 rnn 网络的输入维度
        if self.args.prev_act_inp:
            # 若上一步动作作为输入，则输入维等于 obs_dim + act_dim
            self.rnn_network_input_dim = obs_dim + act_dim
        else:
            # 否则仅用观测作为输入
            self.rnn_network_input_dim = obs_dim
        # RNN 输出维度和隐藏尺寸配置（来自 args.hidden_size）
        self.rnn_out_dim = self.args.hidden_size
        self.rnn_hidden_size = self.args.hidden_size

        # 创建邻接策略网络（AdjPolicy），它接收 obs 与 rnn hidden，输出每条可能连接的概率分布
        # 参数：args, rnn_network_input_dim（每个变量的输入维），hidden_size，device，以及是否用 ReLU
        self.adj_policy = AdjPolicy(args, self.rnn_network_input_dim, args.hidden_size, device, args.use_ReLU)

        # 创建 epsilon 退火策略：从 epsilon_start 到 epsilon_finish，在 adj_anneal_time 步线性衰减
        self.exploration = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.adj_anneal_time,
                                                 decay="linear")
        # gen_matrix 为邻接矩阵的概率
        self.device = device
        self.highest_orders = args.highest_orders
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.to(device)

    # get_hidden_states：把观测和上一步动作/隐藏状态送进 RNN，得到 RNN 的输出与新隐状态
    def get_hidden_states(self, obs, prev_actions, rnn_states):
        # 如果 RNN 输入包含前一步动作，则把 obs 与 prev_actions 在最后一维拼接
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions)
            input_batch = torch.cat((obs, prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)

        # 把 input_batch 与 rnn_states 转为合适的 dtype/device，然后传给 rnn_network
        q_batch, new_rnn_states, no_sequence = self.rnn_network(input_batch.to(**self.tpdv),
                                                                to_torch(rnn_states).to(**self.tpdv))

        return q_batch, new_rnn_states, no_sequence

    def sample(self, obs, rnn_obs, use_adj_init, dones, explore=False, t_env=None):
        """
        生成 mask-safe 邻接。
        输出:
          softmax:  [B, N, F]
          cond_adj: [B, N, F]
          entropy:  [B, N]
        """
        batch_size = obs.shape[0]
        input_batch = to_torch(obs).to(**self.tpdv)

        if len(rnn_obs.shape) == 2:
            rnn_obs_batch = to_torch(rnn_obs).to(**self.tpdv).unsqueeze(0)
        else:
            rnn_obs_batch = to_torch(rnn_obs).to(**self.tpdv)

        agent_dones = to_torch(dones).to(self.device)
        agent_dones = agent_dones.reshape(batch_size, self.num_variable, -1)[..., 0].bool()
        active_masks = (~agent_dones).float()  # [B, N]

        softmax, log_probs = self.adj_policy(input_batch, rnn_obs_batch, use_adj_init, agent_dones)

        # ===== 修改点 1：输出级强制 mask，不能完全依赖 AdjPolicy 内部实现 =====
        active_masks_3d = active_masks.unsqueeze(-1)  # [B, N, 1]
        softmax = softmax * active_masks_3d

        # 按 factor 维度对应的变量分布重归一化；全无效 factor 保持全 0
        denom = softmax.sum(dim=1, keepdim=True).clamp_min(1e-8)
        softmax = softmax / denom
        softmax = softmax * active_masks_3d

        softmax_pre = softmax.transpose(1, 2)  # [B, F, N]
        flat_prob = softmax_pre.reshape(-1, self.num_variable)
        flat_sum = flat_prob.sum(dim=-1, keepdim=True)

        # multinomial 不允许全 0 行；全 0 行仅用于占位，最终 cond_adj 会被阈值 mask 清零
        uniform_prob = torch.ones_like(flat_prob) / float(self.num_variable)
        safe_sample_prob = torch.where(flat_sum > 1e-8, flat_prob / flat_sum.clamp_min(1e-8), uniform_prob)

        if explore:
            if self.use_epsilon_greedy:
                eps = float(self.exploration.eval(t_env if t_env is not None else 0))
                rand_numbers = torch.rand(batch_size, self.num_factor, 1, device=self.device)
                take_random = (rand_numbers < eps).long()

                # ===== 修改点 2：随机采样也只能从有效分布中采，不再用全变量均匀概率直接采死亡 slot =====
                random_indices = torch.multinomial(
                    safe_sample_prob,
                    self.highest_orders,
                    replacement=False
                ).reshape(batch_size, -1, self.highest_orders)

                greedy_indices = torch.topk(
                    flat_prob,
                    k=self.highest_orders,
                    dim=1,
                    largest=True
                )[1].reshape(batch_size, -1, self.highest_orders)

                indices = (1 - take_random) * greedy_indices + take_random * random_indices
            else:
                indices = torch.multinomial(
                    safe_sample_prob,
                    self.highest_orders,
                    replacement=True
                ).reshape(batch_size, -1, self.highest_orders)
        else:
            value, indices = torch.topk(
                flat_prob,
                k=self.highest_orders,
                dim=1,
                largest=True
            )
            value = value.reshape(batch_size, -1, self.highest_orders)
            indices = indices.reshape(batch_size, -1, self.highest_orders)

            if self.highest_orders == 3:
                p_order3 = value[..., 0] ** 3
                p_order2 = 3 * value[..., 1] * value[..., 2] * (value[..., 1] + value[..., 2])
                p_order1 = 6 * value[..., 0] * value[..., 1] * value[..., 2]

                chosen_order3 = (p_order3 > p_order2) & (p_order3 > p_order1)
                tmp_order3 = indices[chosen_order3]
                if tmp_order3.numel() > 0:
                    tmp_order3[:, 1] = tmp_order3[:, 0]
                    tmp_order3[:, 2] = tmp_order3[:, 0]
                    indices[chosen_order3] = tmp_order3

                chosen_order2 = (p_order2 >= p_order3) & (p_order2 > p_order1)
                tmp_order2 = indices[chosen_order2]
                if tmp_order2.numel() > 0:
                    tmp_order2[:, 2] = tmp_order2[:, 0]
                    indices[chosen_order2] = tmp_order2

            elif self.highest_orders == 2:
                chosen = (value[..., 0] ** 2) > (2 * value[..., 1])
                tmp_idx = indices[chosen]
                if tmp_idx.numel() > 0:
                    tmp_idx[:, 1] = tmp_idx[:, 0]
                    indices[chosen] = tmp_idx

        # ===== 修改点 3：entropy 使用 clamp 后的安全概率，并按 active mask 清零 =====
        log_softmax_safe = torch.log(softmax.clamp_min(1e-8))
        entropy = -(softmax * log_softmax_safe).sum(dim=-1) * active_masks

        x = torch.ones_like(softmax, dtype=torch.int64)
        y = torch.zeros_like(softmax, dtype=torch.int64)

        cond_adj_1 = torch.where(softmax > 1e-2, x, y)
        cond_adj_2 = torch.zeros_like(softmax, dtype=torch.int64)
        cond_adj_2 = cond_adj_2.transpose(1, 2).scatter(2, indices, 1).transpose(1, 2)

        cond_adj = cond_adj_1 & cond_adj_2

        # ===== 修改点 4：最终邻接再次按 active mask 清零，确保死亡 slot 不可能进入 factor =====
        cond_adj = cond_adj * active_masks_3d.long()
        softmax = softmax * active_masks_3d

        return softmax, cond_adj, entropy

    def parameters(self):
        parameters_sum = []
        # parameters_sum += self.autoencoder.parameters()
        parameters_sum += self.adj_policy.parameters()

        return parameters_sum

    def load_state(self, source_adjnetwork):
        # self.autoencoder.load_state_dict(source_adjnetwork.autoencoder.state_dict())
        self.adj_policy.load_state_dict(source_adjnetwork.adj_policy.state_dict())
