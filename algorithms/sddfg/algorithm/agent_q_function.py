import torch
import torch.nn as nn
from utils.util import init, adj_init
from utils.util import to_torch
import torch.nn.functional as F

"""
1.output_layer
    Linear(input_dim*num_orders,act_dim*num_orders)
2.norm
"""
# 定义单个阶数的 Q 网络（一个小型 MLP），用于给出该阶数在不同动作下的估计值
class AgentQFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, obs_dim, input_dim, num_orders, act_dim, device):
        super(AgentQFunction, self).__init__()
        self.device = device
        # tpdv 是一个方便的字典，后面 to(**self.tpdv) 会把张量放到指定 dtype/device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # 保存是否使用 ReLU 激活的布尔标志（True 表示使用 ReLU，否则使用 tanh）
        self.use_ReLU = args.use_ReLU
        # 保存是否使用正交初始化的布尔标志
        self.use_orthogonal = args.use_orthogonal
        # 根据 use_ReLU 选择激活函数：False -> Tanh, True -> ReLU
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        # 根据 use_orthogonal 选择初始化方法：False -> xavier_uniform, True -> orthogonal
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        # 从 args 中读取 gain（用于初始化时的 scale）
        gain = args.gain
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)
        self.hidden_dim = input_dim
        self.num_orders = num_orders
        self.act_dim = act_dim

        # 输出层：一个线性层把输入（尺寸 input_dim * num_orders） 映射到 (act_dim * num_orders)
        # 这里将多个阶的输出合并在一起（order * act_dim），便于上层统一处理
        # 使用上面定义的 init_ 进行权重和偏置的初始化
        self.output_layer = nn.Sequential(init_(nn.Linear(input_dim*num_orders,act_dim*num_orders)))
        self.to(device)

    # 前向函数：输入 x（RNN隐变量）、rnn_obs（RNN 输出或隐状态展开后向量）、no_sequence 标志
    def forward(self, x, rnn_obs, no_sequence):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        bs = x.shape[0]

        # 把 x 转为 torch 张量并移动到期望的 dtype/device，然后 reshape 成 [bs * num_orders, -1]
        # 目的是把batch与阶展平，以便和 rnn_obs 的对应结构对齐（尽管这里 x 没被直接用于 output_layer）
        x = to_torch(x).to(**self.tpdv).reshape(bs*self.num_orders,-1)
         #[bs*order,obs_dim]
        #rnn_obs = rnn_obs.reshape(bs, self.num_orders,-1)
        rnn_obs = rnn_obs.reshape(bs, -1)
        
        #q_value = self.output_layer(rnn_obs).reshape(bs, -1)
        # 用 output_layer（线性层）对 rnn_obs 做一次线性变换，得到 q_value
        q_value = self.output_layer(rnn_obs)

        #这一步用于做一种特殊的尺度调整（避免在后续组合（不同阶相乘 / 合并）时数值爆炸）
        # Stable signed n-th root used by the low-rank factor product. The old
        # q / abs(q + eps) ** exponent has a zero denominator around q=-eps,
        # exactly where the small-gain output layer is initialized.
        root_exponent = 1.0 - 1.0 / float(self.num_orders)
        q_value_norm = q_value / torch.pow(
            torch.abs(q_value) + 1e-8,
            root_exponent,
        )
        #q_value_norm = q_value
        #[bs,1, hidden_dim] * [bs,hidden_dim, order*act_dim] -> [bs,1,  order*act_dim]-> [bs,order*act_dim]
        
        if no_sequence:
            # 当 no_sequence 为 True 时，q_value_norm 形状可能为 [1, ?, ?]，直接取第一条
            q_value_norm = q_value_norm[0, :, :]

        return q_value_norm
