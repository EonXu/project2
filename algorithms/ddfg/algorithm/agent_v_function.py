import torch
import torch.nn as nn
from utils.util import init, adj_init
from utils.util import to_torch
import torch.nn.functional as F

"""
output_layer
    nn.Linear(input_dim*num_orders,act_dim)
"""

# 定义单个因子/智能体的 V-value 网络（MLP），用于估计某些局部/因子价值
class AgentVFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, state_dim, num_orders, act_dim, device):
        super(AgentVFunction, self).__init__()
        self.device = device
        # tpdv 字典用于统一 to(... ) 时指定 dtype 与 device（例如 to(**self.tpdv)）
        self.tpdv = dict(dtype=torch.float32, device=device)
        # 是否使用 ReLU 激活（True 表示使用 ReLU，否则使用 Tanh）
        self.use_ReLU = args.use_ReLU
        # 是否使用正交初始化（True 表示使用 orthogonal 初始化）
        self.use_orthogonal = args.use_orthogonal
        # 根据 use_ReLU 选择激活函数（这里只是为了与可能的扩展保持一致）
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        # 根据 use_orthogonal 选择初始化方法（xavier 或 orthogonal）
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = args.gain
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)
        self.hidden_dim = input_dim
        self.state_dim = state_dim
        self.num_orders = num_orders

        self.output_layer = nn.Sequential(init_(nn.Linear(input_dim*num_orders,act_dim)))
       
        self.to(device) 

    def forward(self, x, state, no_sequence):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        # 获取 batch size（x 的第一个维度）
        bs = x.shape[0]
        #self.num_orders = num_orders
        #x = to_torch(x).to(**self.tpdv).reshape(bs, self.num_orders,-1)
        # 将输入 x 转换为 torch 张量并移动到期望 dtype/device，随后把所有特征扁平为一行
        x = to_torch(x).to(**self.tpdv).reshape(bs,-1)
        
        v_value = self.output_layer(x)
        #v_value = self.output_layer(torch.cat([x.sum(1),self.hidden_layer(state)],dim=1))

        if no_sequence:
                v_value = v_value[0, :, :]


        return v_value

