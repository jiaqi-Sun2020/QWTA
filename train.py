# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

from itertools import islice
import json
from pathlib import Path
import random
from tempfile import mkdtemp

import numpy as np
import torch

from dataset.dataset_build import select_dataset
from dataset.dataset_split import split_dataset
from model.model_get import get_model
from utils.config import args2json
from utils.save import save_checkpoint


def get_loaders(args):
    data = select_dataset(args)
    return (*split_dataset(args, data), data.edge_index)


def get_device(args):
    device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and getattr(args, "max_batches", None) is not None:
        torch.set_num_threads(2)
    return device


def run_epoch(model, loader, device, max_batches=None, optimizer=None):
    training = optimizer is not None
    model.train(training)
    squared_error = absolute_error = 0.0
    observed = 0
    with torch.set_grad_enabled(training):
        for x, y in islice(loader, max_batches):
            x, y = x.to(device), y.to(device)
            if y.ndim == 4 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            valid = ~torch.isnan(y)
            count = int(valid.sum())
            if count == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if prediction.shape != y.shape:
                raise ValueError(f"Prediction shape {prediction.shape} differs from label shape {y.shape}.")
            if not torch.isfinite(prediction).all() or torch.isinf(y).any():
                raise FloatingPointError("Non-finite prediction or infinite target.")
            error = prediction[valid] - y[valid]
            batch_squared = error.square().sum()
            if not torch.isfinite(batch_squared):
                raise FloatingPointError("Non-finite squared error.")
            if training:
                (batch_squared / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            squared_error += batch_squared.item()
            absolute_error += error.detach().abs().sum().item()
            observed += count
    if observed == 0:
        raise ValueError("No observed targets in the selected batches.")
    return {"mse": squared_error / observed, "mae": absolute_error / observed}


def train(args):
    max_batches = getattr(args, "max_batches", None)
    for name in ("batch_size", "num_epochs", "save_freq", "patience"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError("max_batches must be a positive integer.")

    args.seed = getattr(args, "seed", 42)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args)

    train_loader, val_loader, test_loader, edge_index = get_loaders(args)
    x, _ = train_loader.dataset[0]
    model = get_model(args, x.shape[-1], edge_index, x.shape[-2], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1
    )

    output_root = Path(args.save_path) / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run_", dir=output_root)).resolve()
    args2json(args, run_dir / "config.json")
    print(f"{args.model} on {device} | run: {run_dir}")

    best_val_mse = float("inf")
    stale_checks = 0
    for epoch in range(1, args.num_epochs + 1):
        training = run_epoch(model, train_loader, device, max_batches, optimizer)
        print(f"Epoch {epoch:03d} | Train MSE: {training['mse']:.6f}")
        scheduler.step()
        if epoch % args.save_freq != 0 and epoch != args.num_epochs:
            continue
        validation = run_epoch(model, val_loader, device, max_batches)
        print(f"Epoch {epoch:03d} | Val MSE: {validation['mse']:.6f} | Val MAE: {validation['mae']:.6f}")
        if validation["mse"] < best_val_mse:
            best_val_mse = validation["mse"]
            best_epoch = epoch
            stale_checks = 0
            best_path = save_checkpoint(model, optimizer, epoch, run_dir, "best")
        else:
            stale_checks += 1
            if stale_checks >= args.patience:
                break

    save_checkpoint(model, optimizer, epoch, run_dir, "last")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test = run_epoch(model, test_loader, device, max_batches)
    print(f"Best epoch {best_epoch} | Test MSE: {test['mse']:.6f} | Test MAE: {test['mae']:.6f}")
    result = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test["mse"],
        "test_mae": test["mae"],
        "max_batches_per_split": max_batches,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=4) + "\n", encoding="utf-8")
    return run_dir
