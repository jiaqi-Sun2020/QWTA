from pathlib import Path

import torch


def save_checkpoint(model, optimizer, epoch, file_path, name):
    directory = Path(file_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pth"
    temporary = path.with_suffix(".tmp")
    try:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
