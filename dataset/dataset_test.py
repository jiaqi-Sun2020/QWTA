import argparse
from pathlib import Path

if __package__:
    from .dataset_build import load_metrla, visualize_batch
else:
    from dataset_build import load_metrla, visualize_batch


def main():
    parser = argparse.ArgumentParser(description="Inspect locally stored METR-LA samples.")
    parser.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--plot", action="store_true", help="Plot the first sample with Matplotlib.")
    parser.set_defaults(in_time_range=12, pre_time_range=1, irregular_time_steps=9, batch_size=2)
    args = parser.parse_args()
    loader = load_metrla(args)
    x, y = next(iter(loader))
    intervals = loader.dataset.indices.diff(dim=1)
    print(f"Windows: {len(loader.dataset)} | X: {tuple(x.shape)} | y: {tuple(y.shape)}")
    print(f"Retained intervals: {intervals.min().item()}–{intervals.max().item()} time steps")
    if args.plot:
        visualize_batch(x, title="METR-LA")


if __name__ == "__main__":
    main()
