# QWTA

QWTA 使用连续时间量子行走在 METR-LA 传感器图上传播信号，根据不规则采样的历史观测预测交通速度。

[English](README.md) · [简体中文](README_zh.md)

---

## 📁 目录结构

```text
QWTA/
├── runs/                    # 运行产物
├── dataset/                 # METR-LA 数据与加载代码
│   ├── data/                # 本地数据文件
│   ├── dataset_build.py     # 构建预测窗口
│   ├── dataset_split.py     # 按时间顺序划分数据
│   └── dataset_test.py      # 检查样本形状
├── requirements.txt         # Python 依赖
├── main.py                  # 训练入口
├── model/                   # QWTA 模型与经典基线
│   ├── QWTA_base.py         # QWTA-Base 模型
│   ├── QWTA_GR.py           # QWTA-GR 模型
│   ├── GAT.py               # 图注意力基线
│   ├── GCN.py               # 图卷积基线
│   ├── GIN.py               # 图同构网络基线
│   └── Transformer.py       # 图 Transformer 基线
├── config.json              # 模型与训练配置
├── test.py                  # 评估已保存的模型权重
├── train.py                 # 训练、验证与保存权重
└── utils/                   # 配置与权重管理工具
```

## 🛠 环境准备

### Python 环境

使用 Python 3.10，并在仓库根目录运行下列命令。

```powershell
python -m pip install -r requirements.txt
```

### METR-LA 数据

将 `adj_mat.npy` 和 `node_values.npy` 放入 `data/`。该目录不存在时，QWTA 会读取 `dataset/data/`。也可用 `--data-dir` 指定其他本地数据目录。

### 参考工作站

Windows 10 Pro for Workstations（22H2）、Intel Core i9-14900K、NVIDIA RTX 4090（24 GB 显存）、32 GB DDR5 内存。

## 🚀 快速开始

### 训练

在 CPU 上进行冒烟测试。训练 1 个 epoch，训练集、验证集和测试集各取 1 批，每批包含 2 个样本窗口。

```powershell
python -B main.py --smoke --model QWTA_GR --output-dir runs/smoke
```

程序将 `best.pth`、`last.pth`、`config.json` 和 `result.json` 写入 `runs/smoke/QWTA_GR/` 下的运行目录。训练结束后，程序重新加载验证 MSE 最低的权重，再报告测试 MSE 和 MAE。

### 配置完整训练

完整训练时去掉 `--smoke`。通过 `--model` 选择表 I 的模型，可选 `QWTA_base`、`QWTA_GR`、`GAT`、`GCN`、`Transformer` 和 `GIN`。历史配置中的 `QWTA_AD` 是 `QWTA_GR` 的别名。

默认[配置](config.json)使用 MASK3，从 12 步历史观测中保留 9 步，并按时间顺序以 70/20/10 划分训练集、验证集和测试集。通过 `--config` 调整掩码、随机种子或训练计划。QWTA-GR 至少需要两个保留观测。设置 `checkpoint_path` 会载入模型权重并初始化新的优化器。

## 🎯 江南大学量子物理实验室

**欢迎了解江南大学量子物理实验室！** 如对 QWTA 或相关研究有疑问，可通过以下方式联系。

- 📜 **项目作者** [Jiaqi Sun](https://github.com/jiaqi-Sun2020) · [6241206031@stu.jiangnan.edu.cn](mailto:6241206031@stu.jiangnan.edu.cn)
- 👥 **同门** [Tianhao Li](https://github.com/quarter-qwalk) · [6251208002@stu.jiangnan.edu.cn](mailto:6251208002@stu.jiangnan.edu.cn)
- 🎓 **研究导师** [边志浩，江南大学理学院](https://science.jiangnan.edu.cn/info/1018/8646.htm) · [zhihaobian@jiangnan.edu.cn](mailto:zhihaobian@jiangnan.edu.cn)
