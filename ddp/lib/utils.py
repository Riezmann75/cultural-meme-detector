import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist


def reduce_tensor(tensor: torch.Tensor, average=True):
    """Sum or average tensor across all processes.

    Args:
        tensor (Tensor): tensor to be reduced.
        average (bool, optional): decide if the reduced sum needs to be averaged or not. Defaults to True.

    Returns:
        rt (Tensor): reduced tensor.
    """
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= torch.distributed.get_world_size()
    return rt


def plot_and_save(
    x,
    ys: list[Optional[torch.Tensor]],
    x_label,
    y_label,
    x_ticks,
    y_ticks,
    labels,
    plt_title,
    save_path,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for y, label in zip(ys, labels):
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).cpu().detach().numpy()
            plt.plot(x, y, label=label)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.legend()
    plt.title(plt_title)
    plt.savefig(save_path)
    plt.close()
