import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import init, get_clones


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_feature_normalization):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N  # 网络层数

        # 选择激活函数：Tanh或ReLU
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        # 选择初始化方法：Xavier均匀分布或正交初始化
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        # 计算增益值，用于初始化
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 第一层：输入层到隐藏层
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),  # 线性层
            active_func,  # 激活函数
            nn.LayerNorm(hidden_size)  # 层归一化
        )

        # 隐藏层模板
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size)
        )

        # 克隆多个隐藏层
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)  # 通过第一层
        for i in range(self._layer_N):
            x = self.fc2[i](x)  # 通过每个隐藏层
        return x

class CONVLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, use_orthogonal, use_ReLU):
        super(CONVLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.conv = nn.Sequential(
                init_(nn.Conv1d(in_channels=input_dim, out_channels=hidden_size//4, kernel_size=3, stride=2, padding=0)), active_func, #nn.BatchNorm1d(hidden_size//4),
                init_(nn.Conv1d(in_channels=hidden_size//4, out_channels=hidden_size//2, kernel_size=3, stride=1, padding=1)), active_func, #nn.BatchNorm1d(hidden_size//2),
                init_(nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)), active_func)#, nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        x = self.conv(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, inputs_dim):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._use_conv1d = args.use_conv1d
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(inputs_dim)
            
        if self._use_conv1d:
            self.conv = CONVLayer(self._stacked_frames, self.hidden_size, self._use_orthogonal, self._use_ReLU)
            random_x = torch.FloatTensor(1, self._stacked_frames, inputs_dim)
            random_out = self.conv(random_x)
            assert len(random_out.shape)==3
            inputs_dim = random_out.size(-1) * random_out.size(-2)

        self.mlp = MLPLayer(inputs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU,self._use_feature_normalization)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        #print(x.shape)
        #print(self.hidden_size)
        #sys.exit()
        return x
      
