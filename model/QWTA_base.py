"""QWTA Base with one globally shared, signed learnable time scale eta.

Based on model/QWTA_ComplexAct.py (QWTA_CA). Only parameter ownership changes:
all propagation blocks use the same nn.Parameter, model.global_theta.
The propagation phase is exp(-i * lambda * delta_t * global_theta), without
absolute value, clipping, sigmoid, or normalization of eta.

The original temporal convolution, ComplexReLU, magnitude readout, recurrent
fusion and terminal delta_t=1 convention are retained. This is not QWTA_GR or
QWTA_CA_G, and does not add phase clipping or a propagation residual path.

Use: from QWTA_base import QWTA_base
Constructor arguments match QWTA_CA. Existing independent-theta checkpoints
are not directly compatible; do not average their theta values implicitly.
Run this file directly for a synthetic forward/backward sharing check.
"""

import torch
import torch.nn as nn
# 移除对 torch.nn.functional as F 的导入，以避免命名冲突
from complexPyTorch.complexLayers import ComplexLinear
from types import SimpleNamespace


# ==========================================================
# 使用 torch.relu 确保函数调用正确
# ==========================================================
class ComplexReLU(nn.Module):
    def forward(self, input):
        # input is a complex tensor
        # 核心修复：使用 torch.relu 替代 F.relu，避免外部命名冲突
        return torch.complex(torch.relu(input.real), torch.relu(input.imag))


# ==========================================================


class QuantumTemporalGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_index, num_nodes, global_theta, eps=1e-8):
        super().__init__()
        if isinstance(num_nodes, torch.Tensor):
            num_nodes = int(num_nodes.item())
        self.num_nodes = num_nodes
        self.edge_index = edge_index

        self.fc = ComplexLinear(in_features, out_features)

        self.complex_act = ComplexReLU()

        # 构造稠密邻接矩阵（float）
        A = torch.zeros(num_nodes, num_nodes)
        A[edge_index[0], edge_index[1]] = 1.0
        # 对称化（如果你的图是无向的）
        A = (A + A.t()) / 2.0

        # 度矩阵
        degree = A.sum(dim=1)
        # 对称归一化邻接： A_hat = D^-1/2 * A * D^-1/2
        D_inv_sqrt = torch.diag(1.0 / (degree.sqrt() + eps))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt

        # 对称归一化拉普拉斯 L = I - A_hat
        I = torch.eye(num_nodes)
        L = I - A_hat

        self.register_buffer('L', L)

        # ===== 预计算特征分解并缓存 =====
        eigvals, eigvecs = torch.linalg.eigh(L)
        self.register_buffer('eigvals', eigvals)  # [N] real
        self.register_buffer('eigvecs', eigvecs)  # [N,N] real orthonormal

        # ===== 引入可学习的量子演化缩放参数 (theta) =====
        self.global_theta = global_theta
        # =========================================================

    def forward(self, X, evolution_t):
        """
        X: [B, T=1, N, C] (real) -> we convert to complex inside
        evolution_t: [B] tensor (float), Batch 内每个样本的时间间隔 Δt
        """
        device = X.device
        eigvals = self.eigvals.to(device)
        eigvecs = self.eigvecs.to(device)

        B, T, N, C = X.shape

        # 1. 准备 Batch 时间参数 t: [B] -> [B, 1]
        t = evolution_t.to(device).float().view(-1, 1)

        # 2. 构造 Batched 复数相位向量: [B, N]
        i = torch.complex(torch.tensor(0., device=device), torch.tensor(1., device=device))
        #
        phases = torch.exp(-i * self.eigvals.unsqueeze(0) * t * self.global_theta)

        # 3. 构造 Batched 演化算子 U: [B, N, N]
        V = eigvecs.to(dtype=torch.complex64, device=device)
        D_phase = torch.diag_embed(phases)
        V_conj_T = V.conj().T.unsqueeze(0)
        V_unsqueeze = V.unsqueeze(0)
        U = V_unsqueeze @ D_phase @ V_conj_T  # [B, N, N] complex

        # 4. 应用图演化
        Xc = X.to(torch.complex64).squeeze(1)  # [B, N, C] complex
        Xc = torch.matmul(U, Xc)  # [B, N, C] complex

        # 5. 复数线性变换
        X_new = self.fc(Xc)  # [B, N, out_features] complex

        X_new = self.complex_act(X_new)  # [B, N, out_features] complex

        # 6. 返回特征的幅度 ( Reshape 回 [B, T=1, N, out_features])
        X_new = X_new.abs().unsqueeze(1)

        return X_new


# ===== 单个量子时序块 (保持不变) =====
class QWTABlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_vertex, edge_index, num_nodes, global_theta):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.qgnn = QuantumTemporalGCNLayer(out_channels, out_channels, edge_index, num_nodes, global_theta)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, evolution_t):
        # x: [B, C, T, N]
        x = self.time_conv(x)
        x = x.permute(0, 2, 3, 1)  # -> [B,T,N,C]
        x = self.qgnn(x, evolution_t)  # [B,T,N,C]
        x = x.permute(0, 3, 1, 2)  # -> [B,C,T,N]
        x = self.act(self.norm(x))
        return x


# ===== QWTA 主网络，逐时间步门控循环 (保持不变) =====
class QWTA_base(nn.Module):
    def __init__(self, in_features, hidden_dim, out_steps, n_vertex, edge_index, num_nodes, args):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_steps = out_steps
        self.irregular_steps = args.irregular_time_steps
        self.num_nodes = num_nodes

        self.global_theta = nn.Parameter(torch.tensor(1.0))
        self.input_fc = nn.Linear(in_features, hidden_dim)

        self.blocks = nn.ModuleList([
            QWTABlock(hidden_dim, hidden_dim, n_vertex, edge_index, num_nodes, self.global_theta)
            for _ in range(self.irregular_steps)
        ])

        self.fc_gate = nn.Linear(hidden_dim, 1)
        self.output_fc = nn.Linear(hidden_dim, out_steps)

    def forward(self, X):
        B, T, N, F = X.shape
        device = X.device

        # 提取时间戳并计算时间间隔
        timestamps = X[:, :, 0, 1]
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
        ones = torch.ones(B, 1, device=device)
        time_diffs = torch.cat([time_diffs, ones], dim=1)  # 最后一个间隔默认设为1，以便循环匹配

        # 初始化隐藏状态 h = 0
        h = torch.zeros(B, N, self.hidden_dim, device=device)

        # 逐时间步循环
        for t in range(min(self.irregular_steps, T)):

            x_t = self.input_fc(X[:, t])  # [B,N,C]

            # 门控融合
            if t == 0:
                x_in = x_t
            else:
                x_in = h * w + (1 - w) * x_t  # 融合上一隐藏状态和当前输入

            # 增加时间维 [B,C,T,N] 供 block 使用
            x_block = x_in.permute(0, 2, 1).unsqueeze(2)  # [B,C,1,N]

            current_dt = time_diffs[:, t]

            h_block = self.blocks[t](x_block, current_dt)  # [B,C,1,N]
            h = h_block.squeeze(2).permute(0, 2, 1)  # [B,N,C]

            # 复数门控
            w = self.fc_gate(h)  # [B,N,1]
            w = torch.sigmoid(w.abs())  # 取模 -> [B,N,1]

        # 输出预测
        y = self.output_fc(h)  # [B, N, out_steps]
        return y




def _smoke_test():
    """Synthetic verification only; no dataset/checkpoint or training run."""
    torch.manual_seed(42)
    num_nodes, steps, batch_size = 5, 4, 2
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]],
        dtype=torch.long,
    )
    model = QWTA_base(2, 8, 1, num_nodes, edge_index, num_nodes,
                      SimpleNamespace(irregular_time_steps=steps))
    assert all(block.qgnn.global_theta is model.global_theta for block in model.blocks)
    theta_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                        if 'theta' in name]
    assert len(theta_parameters) == 1 and theta_parameters[0][0] == 'global_theta'
    inputs = torch.randn(batch_size, steps, num_nodes, 2)
    timestamps = torch.tensor([[0., 1., 4., 8.], [0., 2., 3., 7.]])
    inputs[:, :, :, 1] = timestamps.unsqueeze(-1)
    prediction = model(inputs)
    assert prediction.shape == (batch_size, num_nodes, 1)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert model.global_theta.grad is not None
    assert torch.isfinite(model.global_theta.grad).all()
    assert model.global_theta.grad.abs().item() > 0
    print('PASS: one shared global_theta parameter across all blocks.')
    print('PASS: forward output shape =', tuple(prediction.shape))
    print('PASS: finite, nonzero gradient reaches global_theta.')


if __name__ == '__main__':
    _smoke_test()
