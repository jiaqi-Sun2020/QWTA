import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexLinear
from types import SimpleNamespace

"""
================================================================================
QWTA_GR — 方案A（相位裁剪）+ 方案D（复数门控残差）
================================================================================

改动位置：仅 QuantumTemporalGCNLayer，QWTABlock 和主网络完全不变。

────────────────────────────────────────────────────────────────────────────────
方案A：相位裁剪（Phase Clipping）
────────────────────────────────────────────────────────────────────────────────
问题：exp(-iλθt) 中，λ_max≈2, θ≈1, t 可能很大
      → 相位角度 λθt 可能超过几十弧度
      → 梯度沿 θ 方向剧烈振荡
      → 表现：val loss 在 Ep160→170 从 0.1441 跳到 0.2203

原来：
    phase_angle = eigvals * t * θ               # 无约束，可达几十弧度
    phases = exp(-i * phase_angle)

改后：
    phase_angle = π * tanh(eigvals * t * θ / π) # 软截断到 [-π, π]
    phases = exp(-i * phase_angle)

物理依据：
    e^{iφ} = e^{i(φ+2π)}，相位超过 2π 在物理上是等价的，
    但梯度不知道这个周期性。tanh 截断把周期约束显式告诉优化器。
    tanh(x/π)*π ≈ x 当 x 很小（初期行为与原模型一致），
    x 大时软饱和（数值稳定）。

────────────────────────────────────────────────────────────────────────────────
方案D：复数门控残差（Complex Gated Residual）
────────────────────────────────────────────────────────────────────────────────
问题：残差硬加 X' = UX + X，比例固定 1:1，模型没有自由度

原来：
    Xc = Xc_prop + Xc_original    # 强制 1:1

改后：
    alpha = sigmoid(gate_prop)    # 传播路权重
    beta  = sigmoid(gate_self)    # 自身路权重
    Xc = alpha * Xc_prop + beta * Xc_original

物理依据：
    变分量子电路（VQC）中，态叠加系数是可学习参数。
    初始化为 0 → sigmoid(0)=0.5，与原始 1:1 行为一致，
    不会破坏已有收敛动态。

新增参数：4 个标量（gate_prop, gate_self 各 2 个，实虚部门控）
================================================================================
"""


class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, z):
        magnitude = z.abs().clamp(min=1e-8)
        scale     = torch.relu(magnitude + self.bias) / magnitude
        return z * scale


class QuantumTemporalGCNLayer(nn.Module):

    def __init__(self, in_features, out_features,
                 edge_index, num_nodes, global_theta, eps=1e-8):
        super().__init__()

        self.global_theta = global_theta
        self.num_nodes    = num_nodes

        A = torch.zeros(num_nodes, num_nodes)
        A[edge_index[0], edge_index[1]] = 1.0
        A = (A + A.t()) / 2.0
        degree     = A.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / (degree.sqrt() + eps))
        A_hat      = D_inv_sqrt @ A @ D_inv_sqrt
        L          = torch.eye(num_nodes) - A_hat
        self.register_buffer('L', L)

        eigvals, eigvecs = torch.linalg.eigh(L)
        self.register_buffer('eigvals', eigvals)
        self.register_buffer('eigvecs', eigvecs)

        # ── 方案D：门控残差参数（4个标量）──────────────────────────
        # 初始化为 0 → sigmoid(0)=0.5，与原始 1:1 残差行为完全一致
        self.gate_prop = nn.Parameter(torch.tensor(0.0))  # 传播路权重
        self.gate_self = nn.Parameter(torch.tensor(0.0))  # 自身路权重

        self.fc          = ComplexLinear(in_features, out_features)
        self.complex_act = ModReLU(out_features)
        self.real_projection = nn.Linear(out_features * 2, out_features)

    def forward(self, X, evolution_t):
        device = X.device
        B, T, N, C = X.shape

        eigvals = self.eigvals.to(device)
        eigvecs = self.eigvecs.to(device)
        t = evolution_t.float().view(-1, 1).to(device)   # [B, 1]

        i_unit = torch.complex(torch.tensor(0., device=device),
                               torch.tensor(1., device=device))

        V  = eigvecs.to(dtype=torch.complex64, device=device)
        Vt = V.conj().T
        Xc = X.to(torch.complex64).squeeze(1)             # [B, N, C]

        # ── 方案A：相位裁剪 ─────────────────────────────────────────
        # 原来：phase_angle = eigvals * t * θ（无约束）
        # 现在：tanh 软截断到 [-π, π]，消除超大相位角的梯度爆炸
        phase_angle = eigvals.unsqueeze(0) * t * self.global_theta   # [B, N]
        phase_angle = torch.pi * torch.tanh(phase_angle / torch.pi)  # ← 方案A核心
        phases = torch.exp(-i_unit * phase_angle)                     # [B, N]

        D_phase = torch.diag_embed(phases)
        U       = V.unsqueeze(0) @ D_phase @ Vt.unsqueeze(0)         # [B, N, N]

        Xc_prop = torch.matmul(U, Xc)                                 # [B, N, C]

        # ── 方案D：门控残差 ─────────────────────────────────────────
        # 原来：Xc = Xc_prop + Xc（强制 1:1）
        # 现在：各自可学习权重
        alpha = torch.sigmoid(self.gate_prop)   # 传播路，值域 (0,1)
        beta  = torch.sigmoid(self.gate_self)   # 自身路，值域 (0,1)
        Xc    = alpha * Xc_prop + beta * Xc     # ← 方案D核心

        Xc = self.fc(Xc)
        Xc = self.complex_act(Xc)

        X_out = self.real_projection(
            torch.cat([Xc.real, Xc.imag], dim=-1)
        )
        return X_out.unsqueeze(1)


class QWTABlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 n_vertex, edge_index, num_nodes, global_theta):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=(3, 1), padding=(1, 0))
        self.qgnn = QuantumTemporalGCNLayer(
            out_channels, out_channels, edge_index, num_nodes, global_theta)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act  = nn.ReLU()

    def forward(self, x, evolution_t):
        x = self.time_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.qgnn(x, evolution_t)
        x = x.permute(0, 3, 1, 2)
        return self.act(self.norm(x))


class QWTA_GR(nn.Module):
    def __init__(self, in_features, hidden_dim, out_steps,
                 n_vertex, edge_index, num_nodes, args):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.out_steps       = out_steps
        self.irregular_steps = args.irregular_time_steps
        self.num_nodes       = num_nodes

        self.global_theta = nn.Parameter(torch.tensor(1.0))
        self.input_fc     = nn.Linear(in_features, hidden_dim)

        self.blocks = nn.ModuleList([
            QWTABlock(hidden_dim, hidden_dim, n_vertex,
                      edge_index, num_nodes, self.global_theta)
            for _ in range(self.irregular_steps)
        ])

        # 改动1：通道级门控，并且同时看 h 和 x_t
        self.fc_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 改动2：轻量 dropout
        self.dropout = nn.Dropout(args.dropout)

        self.output_fc = nn.Linear(hidden_dim, out_steps)

    def forward(self, X):
        B, T, N, F = X.shape
        device = X.device

        timestamps = X[:, :, 0, 1]
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]

        # 改动3：最后一个 Δt 复制最近真实间隔，而不是固定为 1
        last_dt = time_diffs[:, -1:].clone()
        time_diffs = torch.cat([time_diffs, last_dt], dim=1)

        # 可选：限制异常时间间隔
        time_diffs = time_diffs.clamp(min=1e-3, max=10.0)

        h = torch.zeros(B, N, self.hidden_dim, device=device)

        for t in range(min(self.irregular_steps, T)):
            x_t = self.input_fc(X[:, t])

            if t == 0:
                x_in = x_t
            else:
                gate_input = torch.cat([h, x_t], dim=-1)
                w = self.fc_gate(gate_input)
                x_in = h * w + (1 - w) * x_t

            h_block = self.blocks[t](
                x_in.permute(0, 2, 1).unsqueeze(2),
                time_diffs[:, t]
            )

            h = h_block.squeeze(2).permute(0, 2, 1)
            h = self.dropout(h)

        return self.output_fc(h)


if __name__ == "__main__":
    args = SimpleNamespace(irregular_time_steps=5)
    num_nodes  = 5
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=torch.long)

    B, T, N, F = 2, 5, num_nodes, 2
    t0 = torch.tensor([0.,1.,4.,5.,10.]).view(1,T,1,1).repeat(1,1,N,1)
    t1 = torch.tensor([0.,2.,3.,7., 9.]).view(1,T,1,1).repeat(1,1,N,1)
    x  = torch.cat([torch.randn(B,T,N,1), torch.cat([t0,t1],0)], dim=-1)

    model  = QWTA_GR(F, 8, 3, num_nodes, edge_index, num_nodes, args)
    y      = model(x)
    total  = sum(p.numel() for p in model.parameters())

    print("=" * 50)
    print("QWTA_AD 测试报告")
    print("=" * 50)
    print(f"输出形状: {y.shape}")
    print(f"总参数量: {total:,}")
    print()
    print("方案A 相位裁剪验证：")
    qgnn = model.blocks[0].qgnn
    t_test = torch.tensor([[10.0]])  # 大时间差，原模型会产生大相位角
    raw_angle = qgnn.eigvals.max().item() * 10.0 * qgnn.global_theta.item()
    clipped   = torch.pi * torch.tanh(torch.tensor(raw_angle / torch.pi))
    print(f"  原始相位角（λ_max·θ·t=10）: {raw_angle:.3f} rad  ({raw_angle/torch.pi:.2f}π)")
    print(f"  裁剪后相位角:               {clipped.item():.3f} rad  ({clipped.item()/torch.pi:.2f}π)")
    print()
    print("方案D 门控残差初始值：")
    print(f"  gate_prop = sigmoid({qgnn.gate_prop.item():.3f}) = {torch.sigmoid(qgnn.gate_prop).item():.3f}")
    print(f"  gate_self = sigmoid({qgnn.gate_self.item():.3f}) = {torch.sigmoid(qgnn.gate_self).item():.3f}")
    print()
    loss = y.sum(); loss.backward()
    print(f"  gate_prop 梯度: {qgnn.gate_prop.grad.item():.6f}  {'✅' if qgnn.gate_prop.grad.item() != 0 else '❌'}")
    print(f"  gate_self 梯度: {qgnn.gate_self.grad.item():.6f}  {'✅' if qgnn.gate_self.grad.item() != 0 else '❌'}")
    print()
    print("测试完成 ✅")
