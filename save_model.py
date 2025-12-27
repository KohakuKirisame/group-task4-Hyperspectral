import os
import torch

def save_checkpoint(
    encoder,
    classifier,
    optimizer,
    epoch,
    global_step,
    path='./resources/model',
    best_metric=None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder_state_dict": encoder.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    torch.save(checkpoint, path)

def load_checkpoint(
    path,
    encoder,
    classifier,
    optimizer=None,
    device="cpu",
    strict=True
):
    checkpoint = torch.load(path, map_location=device)

    encoder.load_state_dict(
        checkpoint["encoder_state_dict"],
        strict=strict
    )
    classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=strict
    )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_metric = checkpoint.get("best_metric", None)

    return epoch, global_step, best_metric
