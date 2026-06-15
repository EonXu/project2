import numpy as np
import torch
#import torch_scatter  # 用于图神经网络中的散射操作
def scatter_add(src, index, dim=0, out=None, dim_size=None):
    """
    使用 torch 原生函数模拟 torch_scatter.scatter_add
    """
    if dim_size is None:
        dim_size = index.max() + 1

    # 构造输出 tensor 的形状
    out_size = list(src.size())
    out_size[dim] = dim_size

    if out is None:
        out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

    return out.scatter_add_(dim, index, src)
from algorithms.ddfg.algorithm.agent_q_function import AgentQFunction
from algorithms.ddfg.algorithm.agent_v_function import AgentVFunction
# from algorithms.ddfg.algorithm.adj_generator import Adj_Generator
from torch.distributions import Categorical, OneHotCategorical
from utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, \
    avail_choose, to_torch, to_numpy
from algorithms.base.mlp_policy import MLPPolicy
from algorithms.ddfg.algorithm.rnn import RNNBase


class R_DDFGPolicy(MLPPolicy):
    """
    QMIX/VDN Policy Class to compute Q-values and actions (MLP). See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param train: (bool) whether the policy will be trained.
    """

    def __init__(self, config, policy_config, train=True):
        self.args = config["args"]
        self.device = config['device']
        self.obs_space = policy_config["obs_space"]  # 观察空间
        self.n_agents = config["num_agents"]  # 智能体数量
        self.num_factor = config["num_agents"] + self.args.num_factor  # 因子数量 = 智能体数 + 额外因子数
        self.obs_dim = get_dim_from_space(self.obs_space)  # 观察维度
        self.act_space = policy_config["act_space"]  # 动作空间
        self.act_dim = get_dim_from_space(self.act_space)  # 动作维度
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.central_obs_dim = policy_config["cent_obs_dim"]  # 中心化观察维度
        self.discrete_action = is_discrete(self.act_space)  # 是否是离散动作
        self.multidiscrete = is_multidiscrete(self.act_space)  # 是否是多离散动作
        self.hidden_size = self.args.hidden_size  # 隐藏层大小
        self.lamda = self.args.lamda  # 消息传递的阻尼系数
        self.num_rank = self.args.num_rank  # 张量分解的秩:1

        # 网络输入维度设置
        # 如果使用前一时刻的动作作为输入
        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.rnn_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.rnn_network_input_dim = self.obs_dim

        self.rnn_out_dim = self.args.hidden_size #64
        self.rnn_hidden_size = self.args.hidden_size #64
        self.q_network_input_dim = self.rnn_out_dim #64
        self.q_hidden_size = [32, 64, 128]  # Q网络隐藏层大小
        self.highest_orders = self.args.highest_orders  # 最高交互阶数2

        # Q网络输出维度设置
        # 设置不同阶数的Q网络输出维度
        self.q_out_size = []
        self.q_out_size.append(self.act_dim)  # 一阶：单个智能体的动作维度

        # 二阶及以上使用张量分解技术来减少参数数量，输出维度 = 秩 × 动作维度
        for num_orders in range(2, self.highest_orders + 1):
            self.q_out_size.append(self.num_rank * self.act_dim)

        self.tpdv = dict(dtype=torch.float32, device=self.device)  # 张量设备配置
        self.use_vfunction = self.args.use_vfunction  # 是否使用价值函数

        # 网络结构创建
        # Local recurrent q network for the agent
        # 创建RNN网络：处理单个智能体的观察序列
        self.rnn_network = RNNBase(self.args, self.rnn_network_input_dim,
                                   self.rnn_hidden_size, self.rnn_out_dim, self.device)

        # 创建评论家RNN网络：处理中心化观察
        self.rnn_critic_network = RNNBase(self.args, self.central_obs_dim,
                                          self.rnn_hidden_size, self.rnn_out_dim, self.device)

        # 创建不同阶数的Q网络
        # 一阶网络：处理单个智能体 64 → 64 → 14
        # 二阶网络：处理两个智能体之间的交互 64 → 64 → 14
        self.q_network = {num_orders: AgentQFunction(self.args,
                                                     self.rnn_network_input_dim,  # 输入维度：观察维度
                                                     self.q_network_input_dim,  # Q网络输入维度：RNN输出维度
                                                     num_orders,  # 阶数：1,2,3
                                                     self.q_out_size[num_orders - 1],  # 输出维度：根据阶数确定
                                                     self.device) for num_orders in range(1, self.highest_orders + 1)}
        # 如果使用价值函数，创建对应的V网络
        # 一阶V网络: 输入64 → 隐藏层64 → 输出1
        # 二阶V网络: 输入64 → 隐藏层64 → 输出1
        if self.use_vfunction:
            self.v_network = {num_orders: AgentVFunction(self.args,
                                                         self.q_network_input_dim,  # 输入维度
                                                         self.central_obs_dim,  # 中心化状态维度
                                                         num_orders,  # 阶数
                                                         1,  # 输出维度：标量价值
                                                         self.device) for num_orders in
                              range(1, self.highest_orders + 1)}

            # 全局价值网络（处理整个系统的价值） 输入64 → 隐藏层64 → 输出1
            self.vtot_network = AgentVFunction(self.args, self.q_network_input_dim,
                                               self.central_obs_dim, 1, 1, self.device)

        # 如果是训练模式，创建探索策略
        if train:
            self.exploration = DecayThenFlatSchedule(self.args.epsilon_start,
                                                     self.args.epsilon_finish,
                                                     self.args.epsilon_anneal_time,
                                                     decay="linear")

    def get_hidden_states(self, obs, prev_actions, rnn_states):
        """
        获取RNN的隐藏状态
        :param obs: 当前观察
        :param prev_actions: 前一时刻动作
        :param rnn_states: RNN状态
        :return: Q值批次, 新RNN状态, 是否无序列标志
        """
        # 如果使用前一时刻动作，将其与观察拼接
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions)
            input_batch = torch.cat((obs, prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)

        # 通过RNN网络前向传播
        q_batch, new_rnn_states, no_sequence = self.rnn_network(
            input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv))

        return q_batch, new_rnn_states, no_sequence

    """
        集中式 critic 给出的全局 V的估计（时间上可能 shift 一步以做 TD）
        :param obs: 中心化观察
        :param rnn_states: RNN状态
        :return: 全局价值
    """
    def get_vtot(self, obs, rnn_states):
        input_batch = to_torch(obs)

        # 通过评论家RNN网络
        q_batch, new_rnn_states, no_sequence = self.rnn_critic_network(
            input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv))

        # 通过全局价值网络
        vtot = self.vtot_network(q_batch.reshape(-1, self.rnn_out_dim), None, no_sequence)

        return vtot

    def get_v_batch(self, obs_batch, state_batch, adj_input=None, batch_size=1, no_sequence=False, dones=None):
        """
        1.根据邻接矩阵处理批次数据，2.输入价值网络得到各阶的价值并 3.归一化 被get_v_values（）调用
        :param obs_batch: 观察批次 [batch_size, agents, obs_dim]
        :param state_batch: 状态批次
        :param adj_input: 邻接矩阵输入 [batch_size, agents, factors]
        :param batch_size: 批次大小
        :param no_sequence: 是否无序列
        :param dones: 终止标志
        :return: 价值批次, 节点顺序索引, 邻接矩阵, 边数量
        处理流程:
        1. 转置邻接矩阵 → [batch_size, factors, agents]
        2. 按连接数分组 → 一阶、二阶、三阶索引
        3. 提取对应观察 → list_obs_batch[0], [1], [2]
        4. 输入价值网络 → 各阶数的因子独立计算价值
        5. 价值归一化 → 避免某些批次的某些阶数会过度影响结果
        6. 价值整合 → 得到最终状态价值
        """
        # 初始化存储列表：为每个阶数创建空列表
        list_obs_batch = [[] for i in range(self.highest_orders)]  # 按阶数分组的观察
        list_state_batch = [[] for i in range(self.highest_orders)]  # 按阶数分组的状态
        num_edges = 0
        q_batch = []
        idx_node_order = []

        # 转置邻接矩阵，从 [batch, agents, factors] 变为 [batch, factors, agents]以便处理
        # 现在adj_input[i,j]表示第i个批次的第j个因子连接了哪些智能体
        adj_input = adj_input.transpose(1, 2)

        # 1.根据邻接矩阵中每个因子的连接数进行分组
        for i in range(1, self.highest_orders + 1):
            # 找到连接数为i的因子（即处理i阶交互）
            # torch.sum(adj_input,dim=2) 计算每个因子连接了多少个智能体
            idx = torch.where(torch.sum(adj_input, dim=2) == i)

            # 构建索引：[batch索引, 因子索引, 连接的智能体索引1, 智能体索引2, ...]
            # 例如三阶交互：[[0, 3, 1, 2, 5]] 表示batch0的第3个因子连接了智能体1,2,5
            tmp = torch.cat([idx[0].unsqueeze(-1),  # batch索引
                             idx[1].unsqueeze(-1),  # 因子索引
                             torch.where(adj_input[idx])[1].reshape(-1, i)],  # 连接的智能体索引
                            dim=-1)
            idx_node_order.append(tmp)

        # 按阶数分组提取对应的观察和状态数据
        for i in range(self.highest_orders):
            tmp = idx_node_order[i]  # 获取该阶数的所有索引
            len_i = len(tmp)
            # 关键：从obs_batch中提取对应智能体的观察
            if len_i != 0:
                # obs_batch[tmp[:,:1], tmp[:,2:]] 的含义：
                # - tmp[:,:1]: batch索引
                # - tmp[:,2:]: 各阶智能体的索引
                list_obs_batch[i] = obs_batch[tmp[:, :1], tmp[:, 2:]].reshape(len_i, -1)

                # 提取对应的全局状态（所有阶数共享相同的全局状态）
                list_state_batch[i] = state_batch[tmp[:, :1]].reshape(len_i, -1)

        # 2.通过对应的价值网络处理每组数据
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                # 将分组后的数据输入到对应阶数的V网络
                q_batch.append(self.v_network[i + 1](list_obs_batch[i], list_state_batch[i], no_sequence))
            else:
                q_batch.append([])  # 如果没有该阶数的数据，保持为空

        num_edges = adj_input.sum()  # 计算总连接数（边数）

        # 计算每个因子连接的智能体数（用于归一化）
        # idx_factor形状: [batch_size, num_factors]，每个元素表示该因子连接的智能体数
        idx_factor = torch.sum(adj_input.transpose(1, 2), dim=1)

        # 3.对价值进行归一化：除以连接数，避免某些批次的某些阶数会过度影响结果
        for i in range(len(idx_node_order)):
            if len(idx_node_order[i]) != 0:
                # 关键归一化步骤：
                # torch.sum(idx_factor==i+1, dim=-1): 计算每个批次中连接数为(i+1)的因子数量
                # [idx_node_order[i][:,0]]: 提取当前分组对应的批次索引
                # 这样确保每个因子的贡献被正确归一化
                q_batch[i] = q_batch[i] / torch.sum(idx_factor == i + 1, dim=-1)[idx_node_order[i][:, 0]].unsqueeze(-1)

        return q_batch, idx_node_order, adj_input.transpose(1, 2), num_edges

    def get_v_values(self, obs_batch, state_batch, adj_input=None, no_sequence=False, dones=None):
        """
        整合各阶数的价值，计算最终的状态价值。
        :param obs_batch: 观察批次
        :param state_batch: 状态批次
        :param adj_input: 邻接矩阵
        :param no_sequence: 是否无序列
        :param dones: 终止标志
        :return: 价值值
        """

        # 确定批次大小
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]  # 多批次情况：[batch, agents, obs_dim]
        else:
            batch_size = 1  # 单批次情况：[agents, obs_dim]

        # 获取处理后的批次数据
        v_batch, idx_node_order, adj, num_edges = self.get_v_batch(
            obs_batch, to_torch(state_batch).to(**self.tpdv), adj_input, batch_size, no_sequence, dones)

        # 根据批次大小选择不同的计算方式
        if batch_size == 1:  # 单批次：使用局部价值计算（更精确）
            values = self.v_local_values(v_batch, idx_node_order)
        else:  # 多批次：将价值分配到对应的因子
            # 创建全零的价值张量：[batch_size, num_factor, 1]
            f_v = torch.zeros((batch_size, self.num_factor, 1)).to(self.device)

            # 将各阶数计算的价值填充到对应的因子位置
            for i in range(len(idx_node_order)):
                if len(idx_node_order[i]) != 0:
                    # idx_node_order[i][:,0]: 批次索引
                    # idx_node_order[i][:,1]: 因子索引
                    # v_batch[i]: 计算出的价值
                    f_v[idx_node_order[i][:, 0], idx_node_order[i][:, 1]] = v_batch[i]

                # 对所有因子的价值求和，得到最终的状态价值
            values = f_v.sum(dim=1)

        return values

    def v_local_values(self, f_q, idx_node_order):
        """
        处理单批次情况下的价值整合。被get_v_values（）调用
        :param f_q: 因子Q值
        :param idx_node_order: 节点顺序索引
        :return: 因子价值
        """
        num_f = self.num_factor

        # Use the utilities for the chosen actions
        # 初始化价值矩阵：[1, num_factor, 1]
        # 1: 单批次，num_factor: 因子数量，1: 标量价值
        values_f = torch.zeros((1, num_f, 1)).to(self.device)

        # 处理一阶因子（单个智能体）
        if len(idx_node_order[0]) != 0:
            # 将一阶价值分配到对应的因子位置
            # idx_node_order[0][:,1] 是因子索引
            values_f[0, idx_node_order[0][:, 1]] = f_q[0]

        # 处理二阶因子（两个智能体交互）
        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:
            values_f[0, idx_node_order[1][:, 1]] = f_q[1]

        # 处理三阶因子（三个智能体交互）
        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            values_f[0, idx_node_order[2][:, 1]] = f_q[2]
        # Return the Q-values for the given actions
        return values_f  # vadj_inputalues_f[:,:self.num_factor-self.n_agents]

    # 从 obs_batch / rnn_states_batch / adj_input 中提取每个阶对应的观测与 rnn 状态并调用对应的q_network 计算该阶的q值张量。
    def get_rnn_batch(self, obs_batch, rnn_states_batch, adj_input=None, batch_size=1, no_sequence=False, dones=None):

        # 初始化存储列表：为每个阶数创建空列表
        list_obs_batch = [[] for i in range(self.highest_orders)]
        list_rnn_batch = [[] for i in range(self.highest_orders)]
        num_edges = 0
        q_batch = []
        # 用于保存每阶的 idx_node 信息（每个因子/边的索引向量）
        idx_node_order = []

        # 转置邻接矩阵按因子维度进行分组处理：从 [batch_size, agents, factors] 变为 [batch_size, factors, agents]
        adj_input = adj_input.transpose(1, 2)

        # 根据邻接矩阵中因子的阶数进行分组，找到 adj_input 中每个 batch*factor 对应的 agent 个数
        for i in range(1, self.highest_orders + 1):
            # 先对最后一维（agents）求和得到[B, F]，再找等于i的位置
            # idx是一对索引（batch_index, factor_index）
            idx = torch.where(torch.sum(adj_input, dim=2) == i)

            # 最终 tmp 是形状 [num_found, 2 + i] 的矩阵，每行： [batch_idx, factor_idx, agent_idx_1, agent_idx_2, ..., agent_idx_i]
            tmp = torch.cat([
                idx[0].unsqueeze(-1),  # batch索引 [b1, b2, ...]
                idx[1].unsqueeze(-1),  # 因子索引 [f1, f2, ...]
                torch.where(adj_input[idx])[1].reshape(-1, i)  # 该因子连接的智能体索引 [a1, a2, ...]
            ], dim=-1)
            idx_node_order.append(tmp)

        # 提取对应阶数的观察和RNN状态数据
        for i in range(self.highest_orders):
            tmp = idx_node_order[i]  # 获取该阶数的所有索引
            len_i = len(tmp)
            if len_i != 0:  # 如果该阶有找到任何边/因子
                # 从obs_batch中提取对应智能体的观察数据
                # tmp[:,:1]: batch索引 [b1, b2, ...]
                # tmp[:,2:]: 智能体索引 [a1, a2, ...] 或 [a1,a2, a3,a4, ...]
                # 结果形状: [样本数, i * obs_dim] - 将i个智能体的观察拼接起来
                list_obs_batch[i] = obs_batch[tmp[:, :1], tmp[:, 2:]].reshape(len_i, -1)

                # 同样提取对应的RNN隐藏状态
                list_rnn_batch[i] = rnn_states_batch[tmp[:, :1], tmp[:, 2:]].reshape(len_i, -1)

        # 再遍历每个阶，调用对应的 q_network 来计算该阶的Q值
        for i in range(self.highest_orders):
            tmp = idx_node_order[i]
            if len(tmp) != 0:
                # 将分组后的数据输入到对应阶数的Q网络
                # 输入: [样本数, i * obs_dim] 的观察 + [样本数, i * hidden_size] 的RNN状态
                # 输出: [样本数, q_out_size[i]] 的Q值
                q_batch.append(self.q_network[i + 1](list_obs_batch[i], list_rnn_batch[i], no_sequence))
            else:
                q_batch.append([])  # 如果没有该阶数的数据，保持为空

        # 如果有二阶且找到了二阶因子，进行张量重组（关键步骤！）
        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:
            dim = list(q_batch[1].shape[:-1])  # 获取除最后一维外的所有维度
            # 将输出重塑为: [总样本数, 秩, 2, 动作维度]
            # 例如: 原始形状[4, 15] -> [4, 3, 2, 5] (秩=3, 动作维度=5)
            tmp = q_batch[1].view(*[np.prod(dim), self.num_rank, 2, self.act_dim])

            # 使用爱因斯坦求和einsum把 rank 分解的两部分合成一个二阶表 q_ij（乘积、合并 index）
            # tmp[:,:,0] 与 tmp[:,:,1] 分别是两个 rank component（每个 shape [prod(dim), num_rank, act_dim]）
            # einsum 'abi,abj->aij' 表示对 rank (b) 做内积，得到 [prod(dim), act_dim, act_dim]
            q_ij = torch.einsum('abi,abj->aij', tmp[:, :, 0], tmp[:, :, 1])
            q_ij = q_ij.permute(0, 2, 1)  # 调整维度顺序

            # 把 q_ij reshape 成 [prod(dim), -1]（将 act_dim x act_dim 展平为一维），作为二阶 q_batch 的最终表示
            q_batch[1] = q_ij.reshape(np.prod(dim), -1)
            # q_batch[1] = (q_ij+q_ij.permute(0, 2, 1).detach()/2).reshape(np.prod(dim),-1)
            # mport pdb;pdb.set_trace()

        # 若有三阶且存在三阶因子
        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            dim = list(q_batch[2].shape[:-1])
            # 三阶: 重塑为 [样本数, 秩, 3, 动作维度]
            tmp = q_batch[2].view(*[np.prod(dim), self.num_rank, 3, self.act_dim])

            # 对三部分 rank component 依次做三重爱因斯坦积：得到 [prod(dim), act_dim, act_dim, act_dim]
            q_batch[2] = torch.einsum('abi,abj,abk->aijk', tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2]).reshape(
                np.prod(dim), -1)

        # 计算总边数（连接数量）
        num_edges = adj_input.sum()

        # 价值归一化 计算每个 batch 中每个因子所连接的 agent 数量（先转回 [B, N, F] 再按 agent 维度求和）
        idx_factor = torch.sum(adj_input.transpose(1, 2), dim=1)
        for i in range(len(idx_node_order)):
            if len(idx_node_order[i]) != 0:
                # torch.sum(idx_factor==i+1, dim=-1) 给出每个 batch 中每个 factor 是否为第 (i+1) 阶的布尔计数向量
                # 然后用 [idx_node_order[i][:,0]] 按找到的 batch 索引取出对应样本的计数作为分母
                # 最后 unsqueeze( -1 ) 以便广播除法，把 q_batch[i] 逐行除以该计数（做平均）
                q_batch[i] = q_batch[i] / torch.sum(idx_factor == i + 1, dim=-1)[idx_node_order[i][:, 0]].unsqueeze(-1)

        return q_batch, idx_node_order, adj_input.transpose(1, 2), num_edges

    """
        在已知动作的前提下，计算当前时刻的 Q（供训练时候的 TD 误差使用）
        1.走一遍 get_rnn_batch 把 rnn_obs 整形后直接过各阶 q_network[num_orders] 得 payoff；
        2. 按 action_indices（形如 [B, N, 1]）把每个因子对应动作处的值取出来；
        3. 聚合得到 [B, 1] 或 [B] 的“合成 Q”。
        :param obs_batch: (np.ndarray) agent observations from which to compute q values
        :param action_batch: (np.ndarray) if not None, then only return the q values corresponding to actions in action_batch

        :return q_values: (torch.Tensor) computed q values
    """
    def get_q_values(self, obs_batch, rnn_q_states_batch, action_batch, adj_input=None, no_sequence=False, dones=None):
        # 判断 obs_batch 是否带有 batch 维：如果是 3D（B, N, obs_dim），则取 batch_size = B
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:  # ；否则视为单样本，batch_size = 1
            batch_size = 1

        # 调用 get_rnn_batch 去按阶提取每条因子对应的 obs 与 rnn hidden，并得到分解后的q_batch
        q_batch, idx_node_order, adj, num_edges = self.get_rnn_batch(obs_batch.to(**self.tpdv),
                                                                     rnn_q_states_batch.to(**self.tpdv), adj_input,
                                                                     batch_size, no_sequence, dones)

        # 计算给定动作的Q值
        values = self.q_values(q_batch, action_batch.type(torch.int64), idx_node_order, batch_size)

        return values

    """ 该函数把分解得到的各阶因子的 f_q 按动作索引取值并合成为全局值（被 greedy 调）
            主要步骤：
            1) 根据 idx_node_order 索引出每个因子在该动作下的值（idx_a_of_Q）
            2) 对每一阶：用 gather 按动作索引抽取 q；然后用 scatter_add 把因子贡献按 batch 累加
            3) 不同阶的贡献相加得到最终 Q
    """
    def q_values(self, f_q, actions, idx_node_order, batch_size):
        # idx_a_of_Q 用于存放每阶的一组索引（便于从 actions 中批量抽取对应动作）
        idx_a_of_Q = []

        # 遍历各阶,把 batch_idx 与 agent_idx_* 拼接起来
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                # idx_node_order[i] 每行是 [batch_idx, factor_idx, agent_idx_1, agent_idx_2, ...]
                # 我们把 batch_idx 与 agent_idx_* 拼接起来，然后转置，以便用于 actions[batch_idx_vec, agent_idx_vec] 的高级索引
                # 结果是一个尺寸为 (1 + i) x E_i 的矩阵（i 为阶数，E_i 为该阶因子数）
                idx_a_of_Q.append(torch.cat([idx_node_order[i][:, :1], idx_node_order[i][:, 2:]], dim=1).t().squeeze())
            else:
                idx_a_of_Q.append([])

        # 初始化 values，用来累积不同阶的 batch 得分；此处使用 3 行（支持最多 3 阶），每行最终为形 [1, batch_size]
        values = torch.zeros((3, batch_size)).to(self.device)
        # ---------- 处理一阶（order 1） ----------
        if len(idx_node_order[0]) != 0:
            # 使用之前构建的索引从 actions 中批量抽取这些因子对应 agent 的动作
            # idx_a_of_Q[0][0] 是 batch indices，idx_a_of_Q[0][1] 是 agent indices
            edge_actions_1 = actions[idx_a_of_Q[0][0], idx_a_of_Q[0][1]]
            # 若抽出来是 1D 向量（N），则扩展为列向量形 [N, 1] 以便作为 gather 的 index
            if len(edge_actions_1.shape) == 1:
                edge_actions_1 = edge_actions_1.unsqueeze(dim=-1)
            # f_q[0] 是一阶 q 表，典型形状 [N, act_dim]，按最后一维用 gather 抽取每条因子对应动作的 q
            # gather 的结果形状为 [N, 1]
            # 把选出的 q 与一个额外的零行拼接（拼上零行是为了后续用 scatter_add 时构造 sentinel，详见下文）
            tmp1 = torch.cat([f_q[0].gather(dim=-1, index=edge_actions_1), torch.zeros((1, 1)).to(self.device)], dim=0)

            # 构造 scatter 索引：把每个因子对应的 batch_idx 列向量拼上一个 sentinel 索引 batch_size（对应上面拼接的零行）
            # 这样 scatter_add 的输出会有 batch_size+1 行，最后我们去掉最后一行 sentinel 行
            if len(idx_a_of_Q[0][0].shape) == 0:
                # 当只有单个元素时，确保 unsqueeze 后能正确拼接
                index1 = torch.cat([idx_a_of_Q[0][0].unsqueeze(0), torch.tensor([batch_size]).to(self.device)])
            else:
                # 否则直接拼接向量
                index1 = torch.cat([idx_a_of_Q[0][0], torch.tensor([batch_size]).to(self.device)])
                # values = f_q[0].gather(dim=-1, index=edge_actions_1).squeeze(dim=-1).mean(dim=-1)

        # ---------- 处理二阶（order 2） ----------
        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:
            # 二阶时需要把两个 agent 的动作合成一个索引以从扁平化的q向量中取值
            # 合成方式为：action1 * act_dim + action2（确保与 get_rnn_batch 中 flatten 的顺序一致）
            edge_actions_2 = actions[idx_a_of_Q[1][0], idx_a_of_Q[1][1]] * self.act_dim + actions[
                idx_a_of_Q[1][0], idx_a_of_Q[1][2]]
            # 扩为列向量
            if len(edge_actions_2.shape) == 1:
                edge_actions_2 = edge_actions_2.unsqueeze(dim=-1)
            # f_q[1] 是二阶 Q 的扁平化表，typical shape = [二阶因子的数量, act_dim * act_dim]
            # gather 会返回 [二阶因子的数量, 1]，然后拼接 sentinel 的零行用于 scatter_add
            tmp2 = torch.cat([f_q[1].gather(dim=-1, index=edge_actions_2), torch.zeros((1, 1)).to(self.device)], dim=0)
            if len(idx_a_of_Q[1][0].shape) == 0:
                index2 = torch.cat([idx_a_of_Q[1][0].unsqueeze(0), torch.tensor([batch_size]).to(self.device)])
            else:
                index2 = torch.cat([idx_a_of_Q[1][0], torch.tensor([batch_size]).to(self.device)])
            # 用 scatter_add 累加到每个 batch 上，并去掉 sentinel，reshape 为 [1, batch_size]
            values[1] = scatter_add(src=tmp2, index=index2, dim=0)[:-1].reshape(-1, batch_size)

        # ---------- 处理三阶（order 3） ----------
        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            edge_actions_3 = (actions[idx_a_of_Q[2][0], idx_a_of_Q[2][1]] * self.act_dim + actions[
                idx_a_of_Q[2][0], idx_a_of_Q[2][2]]) * self.act_dim + actions[idx_a_of_Q[2][0], idx_a_of_Q[2][3]]
            if len(edge_actions_3.shape) == 1:
                edge_actions_3 = edge_actions_3.unsqueeze(dim=-1)
            tmp3 = torch.cat([f_q[2].gather(dim=-1, index=edge_actions_3), torch.zeros((1, 1)).to(self.device)], dim=0)
            if len(idx_a_of_Q[2][0].shape) == 0:
                index3 = torch.cat([idx_a_of_Q[2][0].unsqueeze(0), torch.tensor([batch_size]).to(self.device)])
            else:
                index3 = torch.cat([idx_a_of_Q[2][0], torch.tensor([batch_size]).to(self.device)])
            values[2] = scatter_add(src=tmp3, index=index3, dim=0)[:-1].reshape(-1, batch_size)

        # Return the Q-values for the given actions 最终把不同阶的 contributions 按列相加，得到每个样本的联合 Q 值（形状 [batch_size]）
        return values.sum(0)

    """ （被 greedy 的 anytime 分支用于记录 per-factor 值） 
        该函数返回每个因子对于当前动作组合的Q，即按因子索引顺序返回 f 中对应动作下的具体数值（没有按 batch 汇总）
    """
    def q_local_values(self, f_q, actions, idx_node_order):
        # 读取因子总数
        num_f = f_q.shape[1]
        # 将 f_q reshape 为 形状 [1, num_f, -1] 以便统一索引（把每个因子的 q放在最后一维）
        f = f_q.reshape(1, num_f, -1)
        idx_a_of_Q = []

        # 遍历所有阶数，构造 idx_a_of_Q 条目
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                # idx_node_order[i] 每行: [batch_idx, factor_idx, agent_idx_1, ..., agent_idx_i]
                # 我们取出第一列（batch_idx）和从第3列开始的 agent 索引，拼接后转置，便于用作 actions 的高级索引
                # 结果形状为 (1+i,连接的智能体数)，其中第0行为 batch indices，第1..i 行为对应 agent indices
                idx_a_of_Q.append(torch.cat([idx_node_order[i][:, :1], idx_node_order[i][:, 2:]], dim=1).t().squeeze())
            else:
                idx_a_of_Q.append([])

        # 形状 [1, num_f, 1]，我们将按照 idx_node_order 中的 factor_idx 把对应 q 填入对应位置
        values_f = torch.zeros((1, num_f, 1)).to(self.device)

        # ---------------- 处理一阶因子（order=1） ----------------
        if len(idx_node_order[0]) != 0:
            # 从 actions 中抽取对应位置的动作：
            # idx_a_of_Q[0][0] 是 batch indices 向量（长度 E1），idx_a_of_Q[0][1] 是 agent indices 向量（长度 E1）
            # actions[idx_a_of_Q[0][0], idx_a_of_Q[0][1]] 返回长度为 E1 的向量，每项为该因子对应 agent 在对应 batch 中的动作编号
            edge_actions_1 = actions[idx_a_of_Q[0][0], idx_a_of_Q[0][1]]
            # 若返回的是 1D 向量（E1），扩成列向量 [E1, 1] 以便后续作为 gather 的 index
            if len(edge_actions_1.shape) == 1:
                edge_actions_1 = edge_actions_1.unsqueeze(dim=-1)
            # f[0, idx_node_order[0][:,1]] 取出这些一阶因子对应的q表，形如 [E1, A]
            # gather 根据 edge_actions_1 在最后一维选择对应动作的q，结果形 [E1, 1]
            # 将这些值放到 values_f 对应的因子位置上（按 factor_idx 填入）
            values_f[0, idx_node_order[0][:, 1]] = f[0][idx_node_order[0][:, 1]].gather(dim=-1, index=edge_actions_1)

        # ---------------- 处理二阶因子（order=2） ----------------
        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:
            # 对于二阶因子，我们需要把两个 agent 的动作合并为一个扁平索引，
            # 合成方式与 get_rnn_batch 中 flatten 的顺序必须一致：index = a1 * A + a2
            edge_actions_2 = (actions[idx_a_of_Q[1][0], idx_a_of_Q[1][1]] * self.act_dim + actions[
                idx_a_of_Q[1][0], idx_a_of_Q[1][2]]) * self.act_dim + actions[idx_a_of_Q[1][0], idx_a_of_Q[1][2]]
            # 若得到一维向量 [E2]，扩展为列向量 [E2,1]
            if len(edge_actions_2.shape) == 1:
                edge_actions_2 = edge_actions_2.unsqueeze(dim=-1)
            values_f[0, idx_node_order[1][:, 1]] = f[0][idx_node_order[1][:, 1]].gather(dim=-1, index=edge_actions_2)

        # ---------------- 处理三阶因子（order=3） ----------------
        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            # 三阶合成索引，顺序必须保持与 get_rnn_batch 中 flatten 一致：
            # index = (a1 * A + a2) * A + a3
            edge_actions_3 = (actions[idx_a_of_Q[2][0], idx_a_of_Q[2][1]] * self.act_dim + actions[
                idx_a_of_Q[2][0], idx_a_of_Q[2][2]]) * self.act_dim + actions[idx_a_of_Q[2][0], idx_a_of_Q[2][3]]

            # 扩展为列向量
            if len(edge_actions_3.shape) == 1:
                edge_actions_3 = edge_actions_3.unsqueeze(dim=-1)

            # 从三阶扁平q表中 gather 出每条三阶因子对应的组合动作q并写入对应的因子槽
            values_f[0, idx_node_order[2][:, 1]] = f[0][idx_node_order[2][:, 1]].gather(dim=-1, index=edge_actions_3)
        # Return the Q-values for the given actions
        # import pdb;pdb.set_trace()
        return values_f

    """在固定状态下，给出每个 agent 的 最优联合动作（即找到使 Q_tot最大的动作组合)
         直接枚举所有组合是指数形复杂度，采用了近似的 Max-Sum 贪心消息传递算法
         1.初始化消息：初始化时，给每个连接（edge）分配一条消息向量 msg[f→i]，长度为 act_dim_i
         2.迭代信息传递：因子节点（factor）计算在固定其他 agent 动作时，对每个 agent 的局部最优响应；然后将该响应作为消息发给对应 agent；
                      agent 节点聚合来自多个因子的消息，形成自己的动作偏好（按 Max-Sum 更新公式传播q信息）
         3.平滑更新:用衰减因子 λ加权更新上一次消息（防止振荡），λ 越大，更新越保守。
         4.归一化与数值稳定性：每轮更新后常对消息做归一化，防止数值爆炸或梯度过大
         5.最终贪心选择：所有消息传播完后，每个 agent 汇总它从所有相关因子处收到的消息，近似自身的全局最优联合动作
    """
    def greedy(self, adj, q_batch, idx_node_order, available_actions, num_edges, batch_size):
        # All relevant tensors should be double to reduce accumulating precision loss
        # batch_size = 2
        # available_actions = available_actions.repeat(2,1,1)
        lamda = self.lamda

        # 把邻接矩阵 adj（通常维度 [B, N, F]）解析成便于批量计算的索引。
        adj_f = torch.full([batch_size, self.num_factor, self.highest_orders], self.n_agents, dtype=torch.int64).to(
            self.device)#填充 n_agents 作为哨兵）:二元因子就只填前两个，剩下位置用哨兵占位。
        adj_edge = torch.where(adj)#把三维 0/1 邻接展开成边列表的索引三元组 (b_idx, n_idx, f_idx)。后面常用 adj_edge[0]（batch 索引）、adj_edge[1]（变量索引）、adj_edge[2]（因子索引）
        adj_tmp = adj.reshape(-1, self.num_factor)#把 [B, N, F] 拉成 [B*N, F] 方便得到一维边索引
        var_edge, f_edge = torch.where(adj_tmp)#得到一维的“变量所在样本×变量索引”的编号（行号），以及对应的因子列号。
        f_add_dim = torch.zeros_like(adj)
        f_add_dim[:] = (torch.arange(0, batch_size, 1) * self.num_factor).unsqueeze(-1).unsqueeze(-1)
        f_edge += f_add_dim.reshape(-1, self.num_factor)[adj_tmp == 1]

        # 把每个 factor 所涉及的 agent 列表填入 adj_f：对于每个因子（在某个 batch 中），记录是哪几个 agent 参与（用于按阶分配）
        idx_factor = torch.sum(adj, dim=1)
        for i in range(1, self.highest_orders + 1):
            tmp = torch.where(idx_factor == i)
            if len(tmp) != 0:
                adj_f[tmp[0], tmp[1], :i] = torch.where(adj.transpose(1, 2)[tmp[0], tmp[1]])[1].reshape(-1, i)

        # 为后面高效并行地处理不同“位置”的边（比如在 3 阶因子里，agent 可能是第 0、1 或 2 个参与者），idx_type[i]：收集“变量在第 i 维”的边索引集合；
        # 分类这些边的类型（idx_type），并为每种类型预先记录需要 max 的维度（num_dim）以便后面用 max(dim=...)
        idx_dim = torch.where(adj_f[adj_edge[0], adj_edge[2]] == adj_edge[1].unsqueeze(dim=-1))[1]
        idx_type = []
        idx_type.append(torch.where(idx_dim == 0))
        idx_type.append(torch.where(idx_dim == 1))
        idx_type.append(torch.where(idx_dim == 2))
        num_dim = torch.tensor([[1, 2], [0, 2], [0, 1]]) + 1

        # 把 q_batch（按阶分出的q列表，一维扁平化的）转换成统一的 3D 表 f_q[batch, factor, a1, a2, a3]，
        # 便于后面同时处理各阶（即把一阶/二阶/三阶都放到一个 3-D action 网格里，空位用重复/冗余方式填）。
        in_q_batch = []
        f_q = torch.zeros((batch_size, self.num_factor, self.act_dim, self.act_dim, self.act_dim)).to(self.device)
        for i in range(len(idx_node_order)):
            in_q_batch.append(q_batch[i]) # ← 保存“原始的第 i+1 阶的Q”
            if len(idx_node_order[i]) != 0: ## 下面开始把它扩成三维 joint 表
                q_batch[i] = q_batch[i].unsqueeze(-1).repeat(1, 1, self.act_dim ** (2 - i))
                q_batch[i] = q_batch[i].reshape((q_batch[i].shape[0], self.act_dim, self.act_dim, self.act_dim))
                f_q[idx_node_order[i][:, 0], idx_node_order[i][:, 1]] = q_batch[i]

        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        # 初始化best_* 系列用于 anytime 扩展，记录迄今为止在迭代过程中找到的最佳联合动作及其价值（便于中途中断时仍可返回良好动作）
        best_value = torch.empty(batch_size, dtype=torch.float64).fill_(-float('inf')).to(
            **self.tpdv)  # [1] device=self.device
        best_f_value = torch.empty(batch_size, self.num_factor, 1, dtype=torch.float64).fill_(-float('inf')).to(
            **self.tpdv)
        best_actions = torch.empty(batch_size, self.n_agents, 1, dtype=torch.int64).to(self.device)  # [1,8,1]

        # Without edges (or iterations), CG would be the same as VDN: mean(f_i)
        # utils_Q 是每个因子收到的 a→Q 消息的和（类似因子对动作组合的评分），初始为 0
        utils_Q = best_value.new_zeros(batch_size, self.num_factor, self.act_dim, self.act_dim, self.act_dim).to(
            **self.tpdv)  # [1,5,5]
        # utils_a 会成为每个 agent 对每个动作的打分（聚合了来自所有相邻因子的 Q→a 消息，再加可用性掩码）
        utils_a = best_value.new_zeros(batch_size, self.n_agents, self.act_dim).to(**self.tpdv)  # [1,8,5]
        # 如果有不可用动作（环境约束），把对应动作得分设为 -inf，使它们永远不会被选中。
        avail_a = best_value.new_zeros(batch_size, self.n_agents, self.act_dim).to(**self.tpdv)
        if available_actions is not None:
            avail_a = avail_a.masked_fill(available_actions == 0, -float('inf'))

        # Perform message passing for self.iterations: [0] are messages to *edges_to*, [1] are messages to *edges_from*
        # 主要消息传递循环（核心：Q->a 与 a->Q 的往返）
        # 为每条边（edge 数量 E）创建两个方向的消息缓冲区：
        if num_edges > 0 and self.args.msg_iterations > 0: #msg_iterations = 4
            # messages_a2Q：从 agent 到对应因子的消息（被保存成和 f_q 对齐的形状 [A,A,A]，以便 later utils_Q 聚合）。
            messages_a2Q = best_value.new_zeros(num_edges, self.act_dim, self.act_dim, self.act_dim).to(
                **self.tpdv)  # [1,11,5]
            # messages_Q2a（[num_edges, A]）：因子→变量的消息，按边顺序存储；每条边给“它对面的变量”一条长度为 A（动作数）的向量。
            messages_Q2a = best_value.new_zeros(num_edges, self.act_dim).to(**self.tpdv)  # [1,11,5]
            zeros_Q2a = torch.full_like(messages_Q2a, 0).to(**self.tpdv)
            zeros_a2Q = torch.full_like(messages_a2Q, 0).to(**self.tpdv)
            inf_joint_Q2a = best_value.new_zeros(num_edges, self.act_dim, self.act_dim, self.act_dim).fill_(
                -float('inf')).to(**self.tpdv)

            # 计算 joint_Q2a（因子端先计算其对 agent 的“联合效用”）
            for iteration in range(self.args.msg_iterations):
                # Recompute messages: joint utility for each edge: "sender Q-value"-"message from receiver"+payoffs/
                # update Q->a
                # import pdb;pdb.set_trace()
                # 对每条边（f->a）计算一个“联合评分表” joint_Q2a
                # 它等于当前所有其它因子给出的 utils_Q（对该因子的累积贡献），加上该因子的本身f_q，再减去 来自该接收端 agent 的消息 messages_a2Q
                # （因为在 Max-Sum 的对偶消息里，发送端应“扣除”接收端已经给因子的意见，防止重复计入）。
                joint_Q2a = utils_Q[adj_edge[0], adj_edge[2]] - messages_a2Q + f_q[adj_edge[0], adj_edge[2]]
                joint_Q2a = torch.where(torch.isnan(joint_Q2a), inf_joint_Q2a, joint_Q2a)

                # Maximize the joint Q-value over the action of the sender
                # 计算 messages_Q2a（因子发给 agent 的消息：对发件人动作做 max）
                # 对于每条边，我们要告诉目标 agent：如果你选择某个动作 a_i，因子在其它参与 agent 能做最优配合时能给你多大贡献。
                for i in range(3):
                    if min(idx_type[i][0].shape) != 0:
                        messages_Q2a[idx_type[i]] = \
                            joint_Q2a[idx_type[i]].max(dim=num_dim[i][0])[0].max(dim=num_dim[i][1] - 1)[0]

                # 归一化，把 messages_Q2a 的每个向量减去它的平均值（在最后一维上），把向量“中心化”。
                if self.args.msg_normalized:
                    messages_Q2a -= torch.where(torch.isinf(messages_Q2a), zeros_Q2a, messages_Q2a).mean(dim=-1,
                                                                                                         keepdim=True)

                # Create the current utilities of all agents, based on the messages
                # 把消息汇聚为 agent 的效用（utils_a）
                # 所有因子发给某 agent 的消息加起来，得到该 agent 每个动作的当前总评分；另外加入 avail_a 以把不可用动作强制设为 -inf
                utils_a = avail_a + scatter_add(src=messages_Q2a, index=var_edge, dim=0).reshape(
                    batch_size, self.n_agents, -1) #[B, N, A]

                # update a->Q 更新 a->Q（agent 发给因子的消息）并做平滑
                if iteration % 2 == 0:  # 交替更新 & 平滑
                    # 计算 agent 给因子的消息：它应表达“如果因子中的其它人都不变，我（作为这个 agent）选择每个动作能带来的边缘贡献”
                    # 用 utils_a（agent 的总评分）扣除已经来自这个因子的消息 messages_Q2a（这样得到 agent 对该因子的真实贡献/反馈）。
                    joint_a2Q = utils_a[adj_edge[0], adj_edge[
                        1]] - messages_Q2a  # [batch_size,num_edge,act_dim] = [1,16,5]
                else:  # 把这个新计算的 joint_a2Q 与旧的 joint_a2Q 做指数平滑（用 lamda 混合）
                    joint_a2Q = lamda * joint_a2Q + (1 - lamda) * (utils_a[adj_edge[0], adj_edge[1]] - messages_Q2a)

                # 将 agent->Q 的向量放回 messages_a2Q（按位置扩展到 joint grid）
                for i in range(3):
                    if min(idx_type[i][0].shape) != 0:
                        # joint_a2Q 是 [E, A]（对每条边，为 agent 给的 A 个分数）。但 messages_a2Q 需要的格式是 [E, A, A, A]
                        messages_a2Q[idx_type[i][0]] = joint_a2Q[idx_type[i][0]].unsqueeze(num_dim[i][0]).unsqueeze(
                            num_dim[i][1])

                '''if self.args.msg_normalized:
                    messages_a2Q -= torch.where(torch.isfinite(messages_a2Q), messages_a2Q, zeros_a2Q).mean(dim=-1, keepdim=True)'''
                # 重建 utils_Q（把 messages_a2Q 按因子聚合回 utils_Q）
                # 把所有边（展平后）的 messages_a2Q（形 [E, A, A, A]）按 f_edge 聚合回每个因子的联合效用 utils_Q[batch,factor,...]。
                # 这相当于把不同 agent 对同一因子的反馈相加，得到该因子的当前“联合评分表”。
                utils_Q = scatter_add(src=messages_a2Q, index=f_edge, dim=0).reshape(batch_size,
                                                                                                   self.num_factor,
                                                                                                   self.act_dim,
                                                                                                   self.act_dim,
                                                                                                   self.act_dim)
                # Anytime extension （中途返回最佳动作）
                if self.args.msg_anytime:
                    # Find currently best actions and the (true) value of these actions
                    # 在每次消息迭代结束时都可以计算一下当前根据 utils_a（agent 的总评分）得出的动作（每个 agent 对应 argmax）
                    actions = utils_a.max(dim=-1, keepdim=True)[1]
                    # 并计算这些动作的真实总 Q（用 self.q_values）
                    #   idx_node_order[i]：第 i+1 阶所有因子的 (batch_idx, factor_idx) 索引（告诉你“第几号因子在第几个 batch”）。
                    #   q_values 会用 idx_node_order 找到该阶每个因子的参与变量是谁，再用 actions 把这些变量对应的动作索引逐维 gather，得到每个因子在该组合下的值，最后按样本聚合成全局值。
                    value = self.q_values(in_q_batch, actions, idx_node_order, batch_size)
                    # Update best_actions only for the batches that have a higher value than best_value
                    # 如果这次得到的 Q 比以前记录的最佳值更好，就更新 best_value / best_actions
                    change = value > best_value
                    if batch_size == 1:
                        f_value = self.q_local_values(f_q, actions, idx_node_order)
                        best_f_value[change] = f_value[change]
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]
                    # best_margin_value[change] = margin_value[change]

        # Return the greedy actions and the corresponding message output averaged across agents
        # 终止并返回 greedy 动作（当没有启用 anytime 或没有边或不迭代时）
        if not self.args.msg_anytime or num_edges == 0 or self.args.msg_iterations <= 0:
            _, best_actions = utils_a.max(dim=-1, keepdim=True)  # 就直接把 utils_a 的 argmax 当作贪心动作返回
        return best_actions, best_value, None, best_f_value

    """
    作用：给定观测、RNN隐状态、（可选）可用动作与图结构，返回当前一步要执行的动作（one-hot）以及贪心值等诊断量。
    内部主要调用：
    ├─ get_rnn_batch(...) → 编码序列、整理图与索引；
    ├─ greedy(...) → 在因子图上做消息传递+贪心选择，产出动作索引；
    │    ├─ q_values      # (anytime) 当前贪心动作下的全局值
    │    └─ q_local_values# (anytime) 当前贪心动作下的每因子值
    └─ 最后根据 explore 做 epsilon-greedy/onehot 输出动作。
    """
    def get_actions(self, obs_batch, rnn_q_states_batch, available_actions=None, t_env=None, explore=False,
                    adj_input=None, no_sequence=False, dones=None):
        # 如果 obs_batch 带有 batch 维（形状 [B, N, obs_dim]），取出 batch_size，否则 batch_size 置为 1（单样本）
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1

        # get_rnn_batch 返回：
        #   q_batch: 按阶的q列表（供 greedy 使用）
        #   idx_node_order: 每阶的因子-agent 索引表
        #   adj: 邻接矩阵（可能经处理）
        #   num_edges: 全批次的边数（展平计）
        q_batch, idx_node_order, adj, num_edges = self.get_rnn_batch(to_torch(obs_batch).to(**self.tpdv),
                                                                     rnn_q_states_batch.to(**self.tpdv), adj_input,
                                                                     batch_size, no_sequence, dones)

        # 调用 greedy（消息传递 + 贪心选择），得到动作索引与相应评估值
        # actions: [B, N, 1] 或 [N, 1] 之类（索引形式），best_value: 每个样本的联合 Q 值，best_f_value: 每因子值
        actions, best_value, best_margin_value, best_f_value = self.greedy(adj, q_batch, idx_node_order,
                                                                           available_actions, num_edges, batch_size)

        # 去掉多余的维度（例如最后的长度为1的维），使 actions 变为 [B, N] 或 [N]
        actions = actions.squeeze()

        # mask the available actions by giving -inf q values to unavailable actions
        # 下面开始把动作索引转换为环境能接受的 one-hot 格式，并根据 explore 决定是否随机替换动作

        if self.multidiscrete:  # 如果是多离散动作空间（multidiscrete），对每个子动作维分别处理再拼接
            onehot_actions = []
            # 对每个子动作
            for i in range(len(self.act_dim)):
                # 取出当前分量的 greedy 索引
                greedy_action = actions[i]
                # 如果启用探索则用 eps-greedy：以 eps 概率替换为随机动作
                if explore:
                    # 评估当前 epsilon（随时间衰减）
                    eps = self.exploration.eval(t_env)
                    # 为每个 agent 生成随机数决定是否取随机动作（注意这里是对每个 agent 的随机决策）
                    rand_number = np.random.rand(self.n_agents)
                    # random actions sample uniformly from action space # 随机动作从均匀分布采样
                    random_action = Categorical(logits=torch.ones(self.n_agents, self.act_dim[i])).sample().numpy()
                    # take_random 为 1 表示该 agent 采用随机动作
                    take_random = (rand_number < eps).astype(int)
                    # 合并 greedy_action 和 random_action：被选为随机的轨迹用 random 替代
                    action = (1 - take_random) * to_numpy(greedy_action) + take_random * random_action
                    # 把索引转为 one-hot
                    onehot_action = make_onehot(action, self.act_dim[i])
                else:  # 不探索时直接把 greedy 索引转为 one-hot
                    onehot_action = make_onehot(greedy_action, self.act_dim[i])

                onehot_actions.append(onehot_action)

            # 将所有子分量在最后一维拼接起来，得到完整的多离散动作一热表示
            onehot_actions = np.concatenate(onehot_actions, axis=-1)

        else:
            if explore:
                # 若启用探索，按 eps-greedy 在 greedy 动作和随机动作间切换
                eps = self.exploration.eval(t_env)
                rand_numbers = np.random.rand(self.n_agents)
                # logits 用于在可用动作中采样随机动作，avail_choose 将 unavailable 掩码进去
                logits = avail_choose(torch.ones(self.n_agents, self.act_dim), available_actions)

                random_actions = Categorical(logits=logits).sample().numpy()
                take_random = (rand_numbers < eps).astype(int)
                # take_random表示在每一条轨迹初始多采取随机动作，后面多采取指定动作
                actions = (1 - take_random) * to_numpy(actions) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                onehot_actions = make_onehot(actions, self.act_dim)

        return onehot_actions, best_value, best_margin_value, best_f_value

    # 在环境或 buffer 初始化时需要随机动作：返回随机的一热动作（遵循 available_actions 掩码）
    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]

        # 若为多维离散动作空间，多次采样每个子动作的 one-hot，然后拼接
        if self.multidiscrete:
            # 对每个子动作维使用均匀 logits（ones），并用 OneHotCategorical 直接采样 one-hot
            random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i
                              in
                              range(len(self.act_dim))]
            # 拼接每个分量的 one-hot，得到完整动作的一热表示
            random_actions = np.concatenate(random_actions, axis=-1)
        else:
            if available_actions is not None:
                # 若有可用动作掩码，先用 avail_choose 把 logits 中不可用动作屏蔽（通常设为 -inf）
                logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                # 然后用 OneHotCategorical 采样（会只在可用动作中采样）
                random_actions = OneHotCategorical(logits=logits).sample().numpy()
            else:# 无掩码时对所有动作均匀采样 one-hot
                random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()

        return random_actions

    # 初始化 RNN 的 hidden states（用于 episode 起始或批量重置）
    def init_hidden(self, num_agents, batch_size):
        if num_agents == -1:# 如果 num_agents == -1，表明希望返回形状为 [batch_size, hidden_size] 的 hidden（没有把 agents 扁平）
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(batch_size * num_agents, self.hidden_size)

    # 返回策略（actor / rnn / q_network 各阶）需要优化的参数集合（用于创建 optimizer）
    def parameters(self):
        parameters_sum = []
        # 把 rnn_network（用于生成 rnn hidden）的参数加入
        parameters_sum += self.rnn_network.parameters()
        # parameters_sum += self.q_network.parameters()
        # 把每个阶的 q_network 参数加入（num_orders 从 1 到 highest_orders）
        for num_orders in range(1, self.highest_orders + 1):
            parameters_sum += self.q_network[num_orders].parameters()

            '''if self.use_vfunction:
                parameters_sum += self.v_network[num_orders].parameters()'''
        return parameters_sum # 返回参数列表（可直接传给 torch.optim）

    # 如果启用了 per-factor vfunction，则返回 v_network（各阶）的参数（供 value 优化器使用）
    def critic_fv_parameters(self):
        # 收集参数的列表
        parameters = []
        # 遍历每个阶，如果 use_vfunction 为 True，则把对应 v_network 的参数累加
        for num_orders in range(1, self.highest_orders + 1):
            if self.use_vfunction:
                parameters += self.v_network[num_orders].parameters()
        # 返回收集到的参数
        return parameters

    # 返回用于 centralized critic 的参数集合（rnn_critic_network 与可能的 vtot_network）
    def critic_vtot_parameters(self):
        # 收集参数
        parameters = []
        # 把 rnn_critic_network（centralized critic 的 RNN/网络）参数加入
        parameters += self.rnn_critic_network.parameters()
        # 如果启用 vfunction，把 vtot_network（把 per-factor v 合成为总 V 的网络）也加入
        if self.use_vfunction:
            parameters += self.vtot_network.parameters()
        # 返回参数列表
        return parameters

    # 从另一个 policy 对象加载权重（硬替换）
    def load_state(self, source_policy):
        # 把 source_policy 中 rnn_network 的参数复制到当前对象
        self.rnn_network.load_state_dict(source_policy.rnn_network.state_dict())
        # 把 centralized critic 的 rnn 权重也复制
        self.rnn_critic_network.load_state_dict(source_policy.rnn_critic_network.state_dict())
        # 如果使用 vfunction，把 tot V 的网络权重也复制
        if self.use_vfunction:
            self.vtot_network.load_state_dict(source_policy.vtot_network.state_dict())
        # 对每个阶，把 q_network（以及可能的 v_network）参数从 source_policy 复制过来
        for num_orders in range(1, self.highest_orders + 1):
            self.q_network[num_orders].load_state_dict(source_policy.q_network[num_orders].state_dict())
            if self.use_vfunction:
                self.v_network[num_orders].load_state_dict(source_policy.v_network[num_orders].state_dict())
