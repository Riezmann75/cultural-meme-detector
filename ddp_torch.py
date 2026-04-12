import os
import time

import torch
from torch import distributed as dist
from torchvision.transforms import v2

from ddp.run import train_one_config
from transforms.padder import Pad

if __name__ == "__main__":
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transforms_list = [
        v2.ToImage(),
        Pad(target_size=256),
        # v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ]

    data_dir = "dataset/split_dataset"
    today_date = time.strftime("%Y%m%d")
    save_dir = f"results/{today_date}"
    world_size = int(os.getenv("LOCAL_WORLD_SIZE", 1))
    config_dicts = [
        {
            "criterion": {
                "label_smoothing": 0.1,
            },
            "optimizer": {"lr": 0.002, "weight_decay": 0},
            "data": {
                "batch_size": 32,
                "transforms": {
                    "common": transforms_list,
                    "train": [
                        v2.RandomPerspective(
                            distortion_scale=0.1,
                            p=0.3,
                        ),
                        v2.ColorJitter(
                            brightness=0.1,
                            contrast=0.1,
                            saturation=0.1,
                        ),
                    ],
                },
                "num_workers": world_size * 3,
            },
            "train": {
                "num_epochs": 100,
            },
        },
    ]
    is_distributed = world_size > 1
    local_rank = int(os.environ["LOCAL_RANK"])
    for index, config in enumerate(config_dicts):
        train_one_config(
            local_rank,
            is_distributed,
            index,
            config,
            data_dir,
            save_dir,
            model_path=None,
        )
    if local_rank == 0:
        dist.destroy_process_group()
