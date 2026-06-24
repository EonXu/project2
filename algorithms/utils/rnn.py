import torch
import torch.nn as nn
from .mlp import MLPBase


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal, use_cell=False):
        super(RNNLayer, self).__init__()
        self.use_cell = use_cell
        if self.use_cell:
            self.rnn = nn.GRUCell(inputs_dim, outputs_dim)  # GRU单元
        else:
            self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)  # GRU层

        # 初始化RNN参数
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)  # 偏置初始化为0
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)  # 正交初始化
                else:
                    nn.init.xavier_uniform_(param)  # Xavier均匀初始化

        self.norm = nn.LayerNorm(outputs_dim)  # 层归一化

    def forward(self, x, hxs):
        if self.use_cell:
            # 使用GRU单元
            hxs = self.rnn(x, hxs)
            return hxs
        else:
            self.rnn.flatten_parameters()  # 优化RNN性能
            x, hxs = self.rnn(x, hxs)  # 前向传播
            return x, hxs[0, :, :]  # 返回输出和最后一个隐藏状态


class RNNBase(MLPBase):
    def __init__(self, args, inputs_dim, device=torch.device("cuda:0")):
        super(RNNBase, self).__init__(args, inputs_dim)

        self._recurrent_N = args.recurrent_N
        self._use_cell = args.use_cell
        # 创建RNN层
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size,
                            self._recurrent_N, self._use_orthogonal, self._use_cell)
        self.to(device)  # 移动到指定设备

    def forward(self, x, hxs, rnn_masks=None):
        # 特征归一化
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        # 1D卷积处理（如果使用）
        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        # 通过MLP基础网络
        x = self.mlp(x)

        # 通过RNN层
        if rnn_masks is not None:
            # A fixed Wolfpack slot can leave and later be occupied by a
            # joined/recovered agent. Replay must reset hidden state at the
            # same boundary as online rollout.
            rnn_masks = torch.as_tensor(
                rnn_masks, device=x.device, dtype=x.dtype
            )
            if rnn_masks.dim() == 2:
                rnn_masks = rnn_masks.unsqueeze(-1)
            if x.dim() != 3 or rnn_masks.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "rnn_masks must have shape [sequence, batch, 1], "
                    f"got x={tuple(x.shape)}, masks={tuple(rnn_masks.shape)}"
                )

            outputs = []
            if self._use_cell:
                hidden = hxs[-1] if hxs.dim() == 3 else hxs
                for t in range(x.size(0)):
                    hidden = hidden * rnn_masks[t]
                    hidden = self.rnn.rnn(x[t], hidden)
                    outputs.append(hidden.unsqueeze(0))
                x = torch.cat(outputs, dim=0)
                hxs = hidden
            else:
                hidden = hxs if hxs.dim() == 3 else hxs.unsqueeze(0)
                self.rnn.rnn.flatten_parameters()
                for t in range(x.size(0)):
                    hidden = hidden * rnn_masks[t].unsqueeze(0)
                    out_t, hidden = self.rnn.rnn(x[t:t + 1], hidden)
                    outputs.append(out_t)
                x = torch.cat(outputs, dim=0)
                hxs = hidden[-1]
        elif self._use_cell:
            hxs = self.rnn(x, hxs)
        else:
            x, hxs = self.rnn(x, hxs)

        return x, hxs
