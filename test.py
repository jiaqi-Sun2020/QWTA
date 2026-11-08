# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
from pathlib import Path

from model.model_get import get_model
from train import get_device, get_loaders, run_epoch
from utils.config import json2args


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to config.json beside the checkpoint.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()

    args = json2args(options.config or options.checkpoint.with_name("config.json"))
    args.checkpoint_path = str(options.checkpoint.resolve())
    args.device = options.device
    if options.data_dir is not None:
        args.data_dir = str(options.data_dir.resolve())
    device = get_device(args)
    _, _, loader, edge_index = get_loaders(args)
    x, _ = loader.dataset[0]
    model = get_model(args, x.shape[-1], edge_index, x.shape[-2], device)
    metrics = run_epoch(model, loader, device, getattr(args, "max_batches", None))
    print(f"Test MSE: {metrics['mse']:.6f} | Test MAE: {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
