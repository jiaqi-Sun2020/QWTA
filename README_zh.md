# QWTA 表 I 最小训练

以下命令在仓库根目录运行。准备 Python 3.10 环境。METR-LA 加载器从 `data/` 读取
`adj_mat.npy` 和 `node_values.npy`。若 `data/` 不存在，则读取 `dataset/data/`。
也可以用 `--data-dir` 指定其他本地目录。

```powershell
python -m pip install -r requirements.txt
python -B main.py --smoke --model QWTA_GR
```

可选模型为 `QWTA_base`、`QWTA_GR`、`GAT`、`GCN`、`Transformer` 和
`GIN`。历史配置中的 `QWTA_AD` 对应这里的 GR。

冒烟模式在 CPU 上训练 1 个 epoch，训练、验证、测试各读取 1 个批次
（每批 2 个样本）。程序保存 `best.pth` 和 `last.pth`，严格重载验证最优权重，
然后报告测试 MSE 和 MAE。输出目录以终端打印的路径为准。

将下面命令中的示例目录换成终端打印的运行目录，即可单独验证 checkpoint。
脚本读取 checkpoint 旁边的 `config.json`。

```powershell
python -B test.py --checkpoint .\runs\QWTA_GR\run_EXAMPLE\best.pth
```

完整训练去掉 `--smoke`。默认配置为 MASK3，即从 12 步历史中保留 9 步，
按时间顺序以 70/20/10 划分。复制并调整 `config.json` 后，用 `--config` 加载。
历史 GR 至少需要两个保留观测。`checkpoint_path` 用于初始化权重，优化器重新创建。
