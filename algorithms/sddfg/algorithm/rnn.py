import torch.nn as nn
import torch
from utils.util import init, adj_init

"""
1.LayerNorm(input_shape)
2.fc1:
    Linear(input_shape, hidden_size)
    Activation(Relu)
    LayerNorm(hidden_size)
2.fc2:
    Linear(hidden_size, hidden_size)
    Activation(Relu)
    LayerNorm(hidden_size)
3.GRU(hidden_size, out_shape)
4.LayerNorm(out_shape) 
"""

class RNNBase(nn.Module):
    # 类的文档：这个类与 rnn_agent 相同，但不计算动作的值或概率，只输出隐藏状态
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """

    # 构造函数：接收参数配置 args、输入维度 input_shape、两个隐藏维度 hidden_size 和 out_shape，以及 device
    def __init__(self, args, input_shape, hidden_size, out_shape, device=torch.device("cuda:0")):
        nn.Module.__init__(self)
        self.args = args
        self.use_ReLU = self.args.use_ReLU #读取是否使用 ReLU 激活的标志
        # 定义一个字典 tpdv 用来统一张量的 dtype 与 device；**注意**这里使用 float16（见下文可能需要改为 float32）
        self.tpdv = dict(dtype=torch.float16, device=device)
        self.use_orthogonal = self.args.use_orthogonal #读取是否使用正交初始化的标志
        self._use_feature_normalization = args.use_feature_normalization #是否对输入特征做归一化（LayerNorm）的配置

        # 根据 use_ReLU 选择激活函数：False -> Tanh(), True -> ReLU()
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        # 根据 use_orthogonal 选择初始化方法：False -> xavier_uniform_, True -> orthogonal_
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        # 根据激活函数自动计算适合的 gain（例如 ReLU 对应 sqrt(2)）
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_ReLU])

        def init_(m):
            # 这里调用外部 util.init，将 init_method 用于权重，lambda(...) 用于 bias 初始化为 0，并传入 gain
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(init_(nn.Linear(input_shape, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.rnn = nn.GRU(hidden_size, out_shape)

        # 对 GRU 内部参数逐个命名并初始化：bias -> 0, weight -> orthogonal/xavier（取决于 use_orthogonal）
        for name, param in self.rnn.named_parameters():
            # 如果参数名包含 'bias'，将其全部初始化为 0
            if 'bias' in name:
                nn.init.constant_(param, 0)
            # 如果参数名包含 'weight'
            elif 'weight' in name:
                # 当启用正交初始化时，使用 orthogonal_ 初始化权重
                if self.use_orthogonal:
                    nn.init.orthogonal_(param)
                else:# 否则使用 xavier_uniform_ 初始化权重
                    nn.init.xavier_uniform_(param)

        # 若开启 _use_feature_normalization,对输入特征进行归一化
        self.norm = nn.LayerNorm(input_shape)
        # 对 GRU 的输出 x 做归一化
        self.rnn_norm = nn.LayerNorm(out_shape)

        # 将整个模块移动到指定 device（同时也会把参数转换为默认 dtype，注意 tpdv 内 dtype 未自动应用到参数）
        self.to(device)


    def forward(self, inputs, rnn_states):
        no_sequence = False
        # 如果 inputs 是 2D（即没有显式时间维，形状如 [B, input_shape]），则在最前面加一维，变为 [1, B, input_shape]
        if len(inputs.shape) == 2:
            inputs = inputs[None]
        # 如果 rnn_states 是 2D（即没有显式层维，形状如 [B, out_shape]），则在最前面加一维，变为 [1, B, out_shape]
        if len(rnn_states.shape) == 2:
            rnn_states = rnn_states[None]
        # 如果启用了特征归一化，把 inputs 通过 self.norm（LayerNorm）进行归一化处理
        if self._use_feature_normalization:
            inputs = self.norm(inputs)

        x = self.fc1(inputs)
        x = self.fc2(x)

        # 在调用 RNN 前整理其参数内存布局以便 cudnn 使用更快路径（在 GPU 上能提升性能）
        self.rnn.flatten_parameters()
        #输出 x（shape [S, B, out_shape]）和最后一层隐状态 hid（shape [num_layers, B, out_shape]）
        x, hid = self.rnn(x, rnn_states)
        x = self.rnn_norm(x)
        # 返回最后一层隐状态（取 hid[0,:,:] 作为单层 GRU 的隐藏输出，形状 [B, out_shape]）
        # 注：hid 是形状 [num_layers, B, out_shape]，通常 num_layers=1，因此 hid[0,:,:] 提取第 0 层的隐藏状态
        return x, hid[0, :, :], no_sequence