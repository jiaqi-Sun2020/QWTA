# QWTA Table I training

This project trains one METR-LA forecasting model at a time using the historical
model implementations. Supported models are `QWTA_base`, `QWTA_GR`, `GAT`,
`GCN`, `Transformer` and `GIN`. `QWTA_AD` is the historical name for this GR.

## Setup

Run commands from the repository root in a Python 3.10 environment.

```powershell
python -m pip install -r requirements.txt
```

The METR-LA loader reads `adj_mat.npy` and `node_values.npy` from `data/`.
When that directory is absent, it uses `dataset/data/`. Use `--data-dir` to
select another local directory.

## Minimal training

```powershell
python -B main.py --smoke --model QWTA_GR
```

The smoke run uses one epoch and one batch of two windows from each of the
chronological train, validation and test splits. It saves `best.pth` and
`last.pth`, strictly reloads the validation-best weights, and reports test MSE
and MAE. The console prints the run directory under `runs/QWTA_GR/`.

To evaluate the saved checkpoint independently, substitute the printed run
directory in the following command. The evaluator reads the adjacent config.

```powershell
python -B test.py --checkpoint .\runs\QWTA_GR\run_EXAMPLE\best.pth
```

For a full run, remove `--smoke`. The default config selects MASK3 (9 retained
observations in a 12-step window). Edit a copy of `config.json` and pass
`--config` to set another MASK, seed or training schedule. The default split
is 70/20/10 in chronological order. `checkpoint_path` initializes weights
with strict loading and starts a new optimizer.

The original GR implementation requires at least two retained observations.
Smoke metrics come from two windows per split and serve as a pipeline check.
