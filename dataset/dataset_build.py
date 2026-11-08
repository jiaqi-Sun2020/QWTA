# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse


class METRLADataset(torch.utils.data.Dataset):
    """Construct windows on demand using the historical METR-LA data recipe."""

    def __init__(self, data_dir, history, horizon, retained, data_seed):
        if not all(type(value) is int and value > 0 for value in (history, horizon, retained)):
            raise ValueError("History, horizon and retained steps must be positive integers.")
        if retained > history:
            raise ValueError("Retained steps cannot exceed the original history window.")

        adjacency_path = data_dir / "adj_mat.npy"
        values_path = data_dir / "node_values.npy"
        if not adjacency_path.is_file() or not values_path.is_file():
            raise FileNotFoundError(
                f"Place adj_mat.npy and node_values.npy in {data_dir} or use --data-dir."
            )
        adjacency = np.load(adjacency_path, allow_pickle=False)
        values = np.load(values_path, allow_pickle=False)
        if values.ndim != 3 or values.shape[2] < 1:
            raise ValueError("node_values.npy must have shape [time, nodes, features].")
        count, nodes, _ = values.shape
        if adjacency.shape != (nodes, nodes):
            raise ValueError("Adjacency shape does not match the traffic sensor count.")
        windows = count - history - horizon + 1
        if windows <= 0:
            raise ValueError("Traffic data is shorter than the history and prediction window.")

        # Preserve the original per-feature normalization and floating-point order.
        values = values.transpose((1, 2, 0)).astype(np.float32)
        values -= values.mean(axis=(0, 2), keepdims=True)
        scales = values.std(axis=(0, 2), keepdims=True)
        if not np.isfinite(scales).all() or np.any(scales == 0):
            raise ValueError("METR-LA features must have finite, positive standard deviations.")
        values /= scales
        self.velocity = torch.from_numpy(values[:, 0, :].copy())
        self.edge_index, self.edge_attr = dense_to_sparse(torch.from_numpy(adjacency))
        self.history = history
        self.horizon = horizon
        rng = random.Random(data_seed)
        self.indices = torch.tensor(
            [sorted(rng.sample(range(history), retained)) for _ in range(windows)]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        positions = self.indices[index]
        speed = self.velocity[:, index + positions].T
        timestamps = positions[:, None].expand_as(speed)
        x = torch.stack((speed, timestamps), dim=-1)
        end = index + self.history
        y = self.velocity[:, end:end + self.horizon]
        return x, y


def load_metrla(args):
    module_dir = Path(__file__).resolve().parent
    project_data = module_dir.parent / "data"
    default_data = project_data if project_data.is_dir() else module_dir / "data"
    data_dir = Path(getattr(args, "data_dir", default_data))
    dataset = METRLADataset(
        data_dir, args.in_time_range, args.pre_time_range,
        args.irregular_time_steps, getattr(args, "data_seed", 42),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader.edge_index = dataset.edge_index
    dataloader.edge_attr = dataset.edge_attr
    return dataloader

# ═══════════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════════

def select_dataset(args):
    name = getattr(args, "dataset_name", "METRLA")
    if name != "METRLA":
        raise ValueError(f"Unsupported dataset {name!r}. Use METRLA.")
    return load_metrla(args)


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def visualize_batch(batch, title='批次可视化'):
    """
    可视化一个 batch 中第 0 个样本、第 0 个节点的速度值与时间间隔 Δt。

    batch : [B, T, N, 2]
    """
    import matplotlib.pyplot as plt

    timestamps = batch[0, :, 0, 1].numpy()
    velocity   = batch[0, :, 0, 0].numpy()
    time_diffs = [round(float(timestamps[i+1] - timestamps[i]), 2) for i in range(len(timestamps)-1)]

    print(f"\n📌 [{title}] 节点 0 时间间隔 Δt：{time_diffs}")

    plt.figure(figsize=(12, 4))
    plt.plot(velocity, marker='o', label='值（feature 0）')
    for i, dt in enumerate(time_diffs):
        plt.text(i, velocity[i], f"Δ{dt}", fontsize=9, color='red')
    plt.xlabel('采样顺序（不规则）')
    plt.ylabel('特征值')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── METRLA 测试 ──────────────────────────────────────────────
    print("=" * 60)
    print("测试 METRLA")
    print("=" * 60)
    args_metrla = SimpleNamespace(
        dataset_name='METRLA',
        in_time_range=12,
        pre_time_range=1,
        irregular_time_steps=5,
        batch_size=4,
    )
    dl = select_dataset(args_metrla)
    x, y = next(iter(dl))
    print(f"✅ x: {x.shape}   y: {y.shape}")
    print(f"   edge_index: {dl.edge_index.shape}")
    visualize_batch(x, title='METRLA — 节点 0 不规则采样')
