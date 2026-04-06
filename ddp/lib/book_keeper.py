import json
from typing import Optional

import numpy as np
import torch
from torchvision import models

from . import utils


def plot_training_results(
    train_losses: torch.Tensor,
    train_accs: torch.Tensor,
    val_losses: torch.Tensor,
    val_accs: torch.Tensor,
    val_precisions: torch.Tensor,
    test_accs: Optional[torch.Tensor],
    test_precisions: Optional[torch.Tensor],
    config_dict,
    save_dir,
):
    lr = config_dict["optimizer"]["lr"] if "lr" in config_dict["optimizer"] else None
    momentum = (
        config_dict["optimizer"]["momentum"]
        if "momentum" in config_dict["optimizer"]
        else None
    )
    weight_decay = (
        config_dict["optimizer"]["weight_decay"]
        if "weight_decay" in config_dict["optimizer"]
        else None
    )
    batch_size = (
        config_dict["data"]["batch_size"]
        if "batch_size" in config_dict["data"]
        else None
    )
    plot_title = "lr={lr}".format(lr=lr) if lr is not None else ""
    plot_title += ", momentum={momentum}".format(momentum=momentum) if momentum is not None else ""
    plot_title += ", weight_decay={weight_decay}".format(weight_decay=weight_decay) if weight_decay is not None else ""
    plot_title += ", batch_size={batch_size}".format(batch_size=batch_size) if batch_size is not None else ""
    
    x = np.arange(train_losses.shape[0])
    # train-val loss
    utils.plot_and_save(
        x,
        [train_losses, val_losses],
        "num epochs",
        "loss",
        np.arange(0, config_dict["train"]["num_epochs"] + 10, 10),
        None, # np.arange(0, 1.1, 0.1),
        ["train loss", "val loss"],
        plot_title,
        f"{save_dir}/loss.png",
    )

    # train-val accuracy
    utils.plot_and_save(
        x,
        [train_accs, val_accs, test_accs],
        "num epochs",
        "accuracy",
        np.arange(0, config_dict["train"]["num_epochs"] + 10, 10),
        np.arange(0, 1.1, 0.1),
        ["train accuracy", "val accuracy", "test accuracy"],
        plot_title,
        f"{save_dir}/accuracy.png",
    )

    # val precision
    utils.plot_and_save(
        x,
        [val_precisions, test_precisions],
        "num epochs",
        "precision",
        np.arange(0, config_dict["train"]["num_epochs"] + 10, 10),
        np.arange(0, 1.1, 0.1),
        ["val precision", "test precision"],
        plot_title,
        f"{save_dir}/val_precision.png",
    )


def save_config(save_path, config_dict):
    with open(save_path, "w") as f:
        save_dict = {k: str(v) for k, v in config_dict.items()}
        f.write(json.dumps(save_dict, indent=2))


def save_training_results(
    save_path,
    train_losses: torch.Tensor,
    train_accs: torch.Tensor,
    val_losses: torch.Tensor,
    val_accs: torch.Tensor,
    val_precisions: torch.Tensor,
    test_accs: Optional[torch.Tensor],
    test_precisions: Optional[torch.Tensor],
):
    with open(save_path, "w") as f:
        for i in range(len(train_losses)):
            obj = {
                "epoch": i,
                "train_loss": round(train_losses[i, 0].item(), 6),
                "train_acc": round(train_accs[i, 0].item(), 6),
                "val_loss": round(val_losses[i, 0].item(), 6),
                "val_acc": round(val_accs[i, 0].item(), 6),
                "val_precision": round(val_precisions[i, 0].item(), 6),
            }
            if test_accs is not None:
                obj["test_acc"] = round(test_accs[i, 0].item(), 6)
                obj["test_precision"] = round(test_precisions[i, 0].item(), 6)
            f.write(json.dumps(obj) + "\n")


def save_model(save_path, model: models.ResNet):
    torch.save(model.state_dict(), save_path)
