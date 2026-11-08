import math

from torch.utils.data import DataLoader, Subset


def split_dataset(args, dataloader):
    """Split forecast windows chronologically, preserving the graph for each split."""
    dataset = dataloader.dataset
    total = len(dataset)
    ratios = getattr(args, "split_ratio", (0.7, 0.2, 0.1))
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("split_ratio must contain three positive ratios summing to 1.")
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    if not 0 < train_end < val_end < total:
        raise ValueError("Dataset is too short for three nonempty splits.")

    loaders = []
    for start, end in ((0, train_end), (train_end, val_end), (val_end, total)):
        loader = DataLoader(Subset(dataset, range(start, end)), batch_size=args.batch_size)
        loader.edge_index = dataloader.edge_index
        loader.edge_attr = dataloader.edge_attr
        loaders.append(loader)
    return tuple(loaders)
