# QWTA

QWTA uses continuous-time quantum walks to forecast traffic speed on the METR-LA sensor graph from irregularly sampled observations.

[English](README.md) · [简体中文](README_zh.md)

---

## 📁 Directory Structure

```text
QWTA/
├── runs/                    # Run outputs
├── dataset/                 # METR-LA data and loaders
│   ├── data/                # Local dataset files
│   ├── dataset_build.py     # Build forecasting windows
│   ├── dataset_split.py     # Split data in chronological order
│   └── dataset_test.py      # Inspect sample shapes
├── requirements.txt         # Python dependencies
├── main.py                  # Training entry point
├── model/                   # QWTA variants and classical baselines
│   ├── QWTA_base.py         # QWTA-Base
│   ├── QWTA_GR.py           # QWTA-GR
│   ├── GAT.py               # Graph attention baseline
│   ├── GCN.py               # Graph convolution baseline
│   ├── GIN.py               # Graph isomorphism baseline
│   └── Transformer.py       # Graph Transformer baseline
├── config.json              # Model and training settings
├── test.py                  # Evaluate a saved checkpoint
├── train.py                 # Train, validate and save checkpoints
└── utils/                   # Configuration and checkpoint helpers
```

## 🛠 Prerequisites

### Python Environment

Use Python 3.10. Run the commands below from the repository root.

```powershell
python -m pip install -r requirements.txt
```

### METR-LA Data

Place `adj_mat.npy` and `node_values.npy` in `data/`. When that directory is absent, QWTA uses `dataset/data/`. Pass `--data-dir` to select another local copy.

### Reference Workstation

Windows 10 Pro for Workstations (22H2), Intel Core i9-14900K, NVIDIA RTX 4090 (24 GB VRAM), and 32 GB DDR5 RAM.

## 🚀 Quick Start

### Train

Run a CPU smoke test with one epoch and one batch of two windows from each split.

```powershell
python -B main.py --smoke --model QWTA_GR --output-dir runs/smoke
```

The run writes `best.pth`, `last.pth`, `config.json`, and `result.json` under `runs/smoke/QWTA_GR/`. Training reloads the checkpoint with the lowest validation MSE before reporting test MSE and MAE.

### Evaluate a Checkpoint

Evaluate the latest QWTA-GR smoke run in PowerShell.

```powershell
$run = Get-ChildItem -LiteralPath .\runs\smoke\QWTA_GR -Directory |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
python -B test.py --checkpoint (Join-Path $run.FullName 'best.pth')
```

### Configure a Full Run

Omit `--smoke` for full training. Select a Table I model with `--model`. Available models are `QWTA_base`, `QWTA_GR`, `GAT`, `GCN`, `Transformer`, and `GIN`. Historical configurations can use `QWTA_AD` as an alias for `QWTA_GR`.

The default [configuration](config.json) uses MASK3, retaining 9 of 12 observations, and a chronological 70/20/10 split for training, validation, and testing. Use `--config` to change the mask, seed, or training schedule. QWTA-GR requires at least two retained observations. Setting `checkpoint_path` loads model weights and starts a new optimizer.

## 🎯 JNU Quantum Physics Lab

**Welcome to JNU Quantum Physics Lab!** For questions about QWTA and related research, use the profiles below.

- 📜 **Project author** [Jiaqi Sun](https://github.com/jiaqi-Sun2020)
- 🎓 **Research supervisor** [Zhihao Bian, School of Science, Jiangnan University](https://science.jiangnan.edu.cn/info/1018/8646.htm)
