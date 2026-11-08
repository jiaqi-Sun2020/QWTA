import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexLinear
from torch_geometric.nn import TransformerConv

class SimpleGraphTransformerLayer(nn.Module):
    """
    输入: X [B, T, N, C]
    输出: X [B, T, N, out_features]
    """
    def __init__(self, in_features, out_features, edge_index, num_nodes, heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.out_features = out_features
        self.heads = heads

        # TransformerConv: 多头注意力 + 可选全连接
        self.trans = TransformerConv(
            in_features,
            out_features // heads,
            heads=heads,
            dropout=0.1,
            beta=True    # 是否学习 skip-attention
        )

    def forward(self, X):
        B, T, N, C = X.shape
        X = X.reshape(B * T, N, C)

        out_list = []

        for t in range(T):
            x_t = X[t*B:(t+1)*B]      # [B, N, C]
            x_flat = x_t.reshape(-1, C)

            out = self.trans(x_flat, self.edge_index.to(X.device))
            out = out.reshape(B, N, self.out_features)

            out_list.append(out)

        return torch.stack(out_list, dim=1)



# ===== 单个量子时序块 =====
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_vertex, edge_index, num_nodes):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0))
        self.layer = SimpleGraphTransformerLayer(out_channels, out_channels, edge_index, num_nodes)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, N]
        x = self.time_conv(x)
        x = x.permute(0,2,3,1)  # -> [B,T,N,C]
        x = self.layer(x)
        x = x.permute(0,3,1,2)  # -> [B,C,T,N]
        x = self.act(self.norm(x))
        return x


# ===== 主网络，逐时间步门控循环 =====
class Transformer_Model(nn.Module):
    def __init__(self, in_features, hidden_dim, out_steps, n_vertex, edge_index, num_nodes, args):
        """
        in_features: 输入特征维 F
        hidden_dim: 隐藏通道数 C
        out_steps: 输出预测时间步
        args.irregular_time_steps: 输入时间步循环次数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_steps = out_steps
        self.irregular_steps = args.irregular_time_steps
        self.num_nodes = num_nodes

        # 输入映射 F -> hidden_dim
        self.input_fc = nn.Linear(in_features, hidden_dim)

        # 量子时序块列表
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, hidden_dim, n_vertex, edge_index, num_nodes)
            for _ in range(self.irregular_steps)
        ])

        # 复数门控层
        self.fc_gate = nn.Linear(hidden_dim, 1)

        # 输出全连接
        self.output_fc = nn.Linear(hidden_dim, out_steps)

    def forward(self, X):
        """
        X: [B,T,N,F]
        输出: [B, out_steps, N]
        """
        # timestamps = X[0, :, 0, 1]  # GPU 张量
        # time_diffs = timestamps[1:] - timestamps[:-1]  # [T-1] 张量
        # time_diffs = torch.cat([time_diffs, torch.tensor([1.0], device=X.device)])  # [T] 张量
        # # print("\n📊 时间间隔 Δt:", time_diffs)

        B, T, N, F = X.shape
        device = X.device

        # 初始化隐藏状态 h = 0
        h = torch.zeros(B, N, self.hidden_dim, device=device)

        # 逐时间步循环
        for t in range(min(self.irregular_steps, T)):

            # 当前输入映射到隐藏维度
            x_t = self.input_fc(X[:, t])  # [B,N,C]

            # 门控融合
            if t == 0:
                x_in = x_t
            else:
                x_in = h * w + (1 - w) * x_t  # 融合上一隐藏状态和当前输入

            # 增加时间维 [B,C,T,N] 供 block 使用
            x_block = x_in.permute(0,2,1).unsqueeze(2)  # [B,C,1,N]
            h_block = self.blocks[t](x_block)          # [B,C,1,N] ,演化时间 evolution_time
            h = h_block.squeeze(2).permute(0,2,1)     # [B,N,C]

            # 复数门控
            w = self.fc_gate(h)               # [B,N,1]
            w = torch.sigmoid(w.abs())        # 取模 -> [B,N,1]

        # 输出预测
        y = self.output_fc(h)  # [B, N, out_steps]
        # y = y.permute(0,2,1)   # -> [B, out_steps, N]
        return y

import torch
from types import SimpleNamespace


if __name__ == "__main__":
    # ===== 模拟参数 =====
    args = SimpleNamespace(
        irregular_time_steps=5,  # 输入循环次数
    )

    # ===== 模拟图结构 =====
    num_nodes = 10
    edge_index = torch.tensor([[i for i in range(num_nodes - 1)],
                               [i + 1 for i in range(num_nodes - 1)]], dtype=torch.long)
    n_vertex = num_nodes

    # ===== 模拟输入数据 =====
    B, T, N, F = 2, 6, num_nodes, 2  # batch=2, T=6, N=10, F=2
    x = torch.randn(B, T, N, F)

    # ===== 初始化模型 =====
    hidden_dim = 8
    out_steps = 3
    model = Transformer_Model(in_features=F, hidden_dim=hidden_dim, out_steps=out_steps,
                 n_vertex=n_vertex, edge_index=edge_index, num_nodes=num_nodes,
                 args=args)

    # ===== 前向推理 =====
    y = model(x)

    print("输入形状:", x.shape)
    print("输出预测形状:", y.shape)  # [B, out_steps, N]
    print("\n部分输出值:\n", y[0, :, :])  # 打印第一个样本的预测结果

    # ===== 验证每个隐藏状态和门控权重 =====
    print("\n✅ 模型前向推理成功！")