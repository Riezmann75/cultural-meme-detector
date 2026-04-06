import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ddp.lib.book_keeper import (
    plot_training_results,
    save_config,
    save_training_results,
    save_model,
)
from ddp.lib.loader import load_data
from ddp.lib.trainer import Trainer

from torch.profiler import profile

import gc

from ddp.lib.models import SiglipMemeClassifier


def train_one_config(
    rank: int,
    is_distributed: bool,
    config_index: int,
    config_dict: dict,
    data_dir: str,
    save_dir: str,
    profiler: profile = None,
    model_path: str = None,
):
    if is_distributed:
        setup(rank)
    os.makedirs(save_dir, exist_ok=True)

    num_classes = 2

    # Model setup
    model = load_model(rank, is_distributed, num_classes, model_path)

    train_loader, val_loader, test_loader = load_data(
        is_distributed,
        data_dir,
        config_dict["data"]["transforms"],
        config_dict["data"]["batch_size"],
        config_dict["data"]["num_workers"],
    )
    criterion_cfg = config_dict.get("criterion", {})
    criterion = nn.CrossEntropyLoss(**criterion_cfg)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config_dict["optimizer"]["lr"],
        # momentum=config_dict["optimizer"]["momentum"],
        weight_decay=config_dict["optimizer"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config_dict["train"]["num_epochs"]
    )

    # train model
    trainer = Trainer(
        is_distributed,
        rank,
        model,
        criterion,
        optimizer,
        scheduler,
        profiler,
    )
    (
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        val_precisions,
        test_accs,
        test_precisions,
    ) = trainer.train(
        train_loader,
        val_loader,
        test_loader,
        config_index,
        config_dict["train"]["num_epochs"],
    )
    if rank == 0:
        save_index = len(
            [
                name
                for name in os.listdir(save_dir)
                if os.path.isdir(os.path.join(save_dir, name))
            ]
        )
        config_folder = f"{save_dir}/config_{save_index}"
        os.makedirs(config_folder, exist_ok=True)
        config_save_path = os.path.join(config_folder, "config.json")
        training_results_save_path = os.path.join(config_folder, "result.jsonl")
        model_save_path = os.path.join(config_folder, "resnet50.pth")

        plot_training_results(
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            val_precisions,
            test_accs,
            test_precisions,
            config_dict,
            config_folder,
        )

        save_training_results(
            training_results_save_path,
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            val_precisions,
            test_accs,
            test_precisions,
        )
        # save_model(model_save_path, model)
        save_config(config_save_path, config_dict)

    model = cleanup(model)


def load_model(rank, is_distributed, num_classes=3, model_path=None):
    model = SiglipMemeClassifier()
    if is_distributed:
        model = DDP(
            model.to(rank),
            device_ids=[rank],
            output_device=rank,
            bucket_cap_mb=25,
            find_unused_parameters=bool(model_path),
        )
    else:
        model = model.to(rank)

    if model_path is not None:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        model.load_state_dict(
            torch.load(model_path, map_location=map_location, weights_only=True)
        )

        # Freeze all layers except the FC layer
        for param in model.module.parameters():
            param.requires_grad = False
        nn.init.kaiming_normal_(model.module.fc.weight)
        model.module.fc.bias.data.zero_()
        for param in model.module.fc.parameters():
            param.requires_grad = True
    return model


def cleanup(model):
    gc.collect()
    model = None
    torch.cuda.empty_cache()
    return model


def setup(rank):
    # initialize the process group
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
