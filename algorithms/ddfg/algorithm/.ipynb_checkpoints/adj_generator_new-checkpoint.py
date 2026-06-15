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
        self.num_variable = args.num_agents
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

    # sample：主采样函数，根据当前观测与 rnn_obs 生成邻接（及其概率、熵）
    def sample(self, obs, rnn_obs, use_adj_init, dones, explore=False, t_env=None):
        # 采样——得到一个临近矩阵
        # batch_size：样本数
        batch_size = obs.shape[0]
        input_batch = to_torch(obs).to(**self.tpdv)  # 把 obs 转为 tensor 并移动到期望 dtype/device
        agent_dones = to_torch(dones).to(self.device)  # 把 dones 转为 tensor 并放到 device（用于 mask 掉已经结束的样本）
        # 若 rnn_obs 是 2D（即没有时间轴），把它扩展为带时间轴的 batch（unsqueeze）
        if len(rnn_obs.shape) == 2:
            rnn_obs_batch = to_torch(rnn_obs).to(**self.tpdv).unsqueeze(0)
        else:  # 已经带有序列维度则直接转换 dtype/device
            rnn_obs_batch = to_torch(rnn_obs).to(**self.tpdv)

        # 用 adj_policy 获取每条潜在边的 softmax 概率以及 log_probs（形状大致为 [B, num_variable, num_factor] 的概率分布）
        softmax, log_probs = self.adj_policy(input_batch, rnn_obs_batch, use_adj_init, agent_dones)
        # 转置 softmax 以便后续按因子/变量维度做操作（从 [B,var,factor] -> [B,factor,var]）
        softmax_pre = softmax.transpose(1, 2)

        # explore 分支：是否进行探索（随机化采样）
        if explore:
            if self.use_epsilon_greedy:  # 若采用 epsilon-greedy
                # 计算当前 eps（标量），并构造随机掩码决定哪些位置采用随机选择
                eps = torch.tensor(self.exploration.eval(t_env))
                rand_numbers = torch.rand(batch_size, self.num_factor, 1)
                # take_random 为 1 表示该位置采用随机选择（按 eps 概率)
                take_random = torch.where(rand_numbers < eps, torch.ones_like(rand_numbers, dtype=torch.int64),
                                          torch.zeros_like(rand_numbers, dtype=torch.int64)).to(self.device)

                # 准备一个与 softmax_pre 合适形状的随机采样概率（均匀）
                x = softmax_pre.reshape(-1, self.num_variable).shape[0]
                random_probability = softmax_pre.new_ones((x, self.num_variable))
                # 从均匀概率中抽样 highest_orders 个索引（不放回）
                random_indices = torch.multinomial(random_probability, self.highest_orders, replacement=False).reshape(
                    batch_size, -1, self.highest_orders)
                # 计算 greedy top-k 索引（基于 softmax_pre 的值）
                greedy_indices = \
                torch.topk(softmax_pre.reshape(-1, self.num_variable), k=self.highest_orders, dim=1, largest=True)[
                    1].reshape(batch_size, -1, self.highest_orders)
                # 对于被选为随机的位置用 random_indices，否则使用 greedy_indices
                indices = (1 - take_random) * greedy_indices + take_random * random_indices
            else:
                # 若不用 eps-greedy，则直接按 softmax_pre 的分布做多次抽样（允许放回）
                indices = torch.multinomial(softmax_pre.reshape(-1, self.num_variable), self.highest_orders,
                                            replacement=True).reshape(batch_size, -1, self.highest_orders)

        else:
            # 非探索（deterministic）分支：直接取 top-k 最大概率的索引和值
            value, indices = torch.topk(softmax_pre.reshape(-1, self.num_variable), k=self.highest_orders, dim=1,
                                        largest=True)
            # 把 value / indices 恢复成 [batch_size, num_factor, highest_orders] 的形状
            value = value.reshape(batch_size, -1, self.highest_orders)
            indices = indices.reshape(batch_size, -1, self.highest_orders)
            # 特殊处理：当 highest_orders == 3 时，根据三个 top 值的组合概率做一些启发式选择
            if self.highest_orders == 3:
                # 计算三阶/二阶/一阶的组合概率评估（用于在多个候选中做选择）
                p_order3 = value[..., 0] ** 3
                p_order2 = 3 * value[..., 1] * value[..., 2] * (value[..., 1] + value[..., 2])
                p_order1 = 6 * value[..., 0] * value[..., 1] * value[..., 2]
                # 若三阶概率占优，则把后两个索引设为第一个索引
                chosen_order3 = (p_order3 > p_order2) & (p_order3 > p_order1)
                tmp_order3 = indices[chosen_order3]
                tmp_order3[:, 1] = tmp_order3[:, 0]
                tmp_order3[:, 2] = tmp_order3[:, 0]
                indices[chosen_order3] = tmp_order3
                # 若二阶占优，则把第三个索引设为第一个索引（退化为二阶）
                chosen_order2 = (p_order2 >= p_order3) & (p_order2 > p_order1)
                tmp_order2 = indices[chosen_order2]
                tmp_order2[:, 2] = tmp_order2[:, 0]
                indices[chosen_order2] = tmp_order2
            # 当 highest_orders == 2 时也有类似的退化处理逻辑
            elif self.highest_orders == 2:
                chosen = (value[..., 0] ** 2) > (2 * value[..., 1])
                tmp_idx = indices[chosen]
                tmp_idx[:, 1] = tmp_idx[:, 0]
                indices[chosen] = tmp_idx

        # 计算熵（每个位置上的 -p log p，用于正则或记录）
        entropy = -softmax * log_probs

        # 构造两个同 shape 的整型张量作为布尔掩码基底（x 表示 1，y 表示 0）
        x = torch.ones_like(softmax, dtype=torch.int64)
        y = torch.zeros_like(softmax, dtype=torch.int64)
        # 根据是否探索，设定 cond_adj_1：当 softmax > 1e-2 时认为该位置可能连边（阈值防止非常小概率连边）
        if explore:
            cond_adj_1 = torch.where(softmax > 1e-2, x, y)
        else:
            cond_adj_1 = torch.where(softmax > 1e-2, x, y)

        # cond_adj_2 用于把 top-k 或采样得到的 indices 转换为 one-hot 的邻接选择
        cond_adj_2 = torch.zeros_like(softmax, dtype=torch.int64)
        # 先把 cond_adj_2 转置，使 factor 维成为最后一维，方便 scatter（在 dim=2 上 scatter）
        cond_adj_2 = cond_adj_2.transpose(1, 2).scatter(2, indices, 1).transpose(1, 2)

        # rand_numbers = torch.rand(batch_size,self.num_variable,self.num_factor).to(self.device)
        # cond_adj_3 = torch.where(rand_numbers<0.1,x,y)

        # 最终的离散邻接 cond_adj 只有在两类条件同时满足时才为 1（概率阈值 & top-k 位置）
        cond_adj = cond_adj_1 & cond_adj_2

        # if random.random() < 0.9:
        #    cond_adj[:] = 0

        '''使用 dones (即 ~exist_mask) 强制将死亡智能体的边置零
        # dones 形状通常为 [batch, num_agents, 1]，True 表示已结束/死亡
        agent_dones = to_torch(dones).to(self.device)
        # [CRITICAL FIX] 强制屏蔽死亡智能体的连边
        # agent_dones: [batch, num_agents, 1], 1 表示死亡
        # cond_adj: [batch, num_agents, num_factors]
        # 如果 agent_dones[b, i, 0] 为 1，则 agent i 不应连接任何因子，即 cond_adj[b, i, :] 应全为 0
        active_mask = (1 - agent_dones.long())  # [batch, num_agents, 1], 1 表示存活
        cond_adj = cond_adj * active_mask'''

        return softmax, cond_adj, entropy.sum(-2).mean(-1)

    def parameters(self):
        parameters_sum = []
        # parameters_sum += self.autoencoder.parameters()
        parameters_sum += self.adj_policy.parameters()

        return parameters_sum

    def load_state(self, source_adjnetwork):
        # self.autoencoder.load_state_dict(source_adjnetwork.autoencoder.state_dict())
        self.adj_policy.load_state_dict(source_adjnetwork.adj_policy.state_dict())
