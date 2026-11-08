# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
from pathlib import Path

from model.model_get import model_dict
from train import train
from utils.config import json2args


def main(config_path=None):
    parser = argparse.ArgumentParser(description="Train one Table I forecasting model.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--model", choices=model_dict)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true", help="One CPU epoch and one batch per split.")
    options = parser.parse_args([] if config_path is not None else None)

    args = json2args(config_path if config_path is not None else options.config)
    if options.model is not None:
        args.model = options.model
    if options.data_dir is not None:
        args.data_dir = str(options.data_dir.resolve())
    if options.output_dir is not None:
        args.save_path = str(options.output_dir.resolve())
    if options.smoke:
        args.num_epochs = args.save_freq = args.max_batches = 1
        args.batch_size = 2
        args.device = "cpu"
    if options.device is not None:
        args.device = options.device
    return train(args)


if __name__ == "__main__":
    main()
