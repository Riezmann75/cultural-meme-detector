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
        # Pad(target_size=500),
        v2.Resize((224, 224), antialias=True),
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
                "label_smoothing": 0,
            },
            "optimizer": {"lr": 0.01, "weight_decay": 0},
            "data": {
                "batch_size": 32,
                "transforms": {
                    "common": transforms_list,
                    "train": [
                        # v2.RandomPerspective(
                        #     distortion_scale=0.5,
                        #     p=0.3,
                        # ),
                        # v2.RandomGrayscale(p=0.3),
                        # v2.RandomApply(
                        #     [
                        #         v2.GaussianBlur(
                        #             kernel_size=(5, 9),
                        #             sigma=(0.1, 5),
                        #         ),
                        #     ]
                        # ),
                    ],
                },
                "num_workers": world_size * 2,
            },
            "train": {
                "num_epochs": 100,
            },
        },
    ]
    is_distributed = world_size > 1
    if is_distributed:
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
    else:
        for index, config in enumerate(config_dicts):
            train_one_config(
                0,
                is_distributed,
                index,
                config,
                data_dir,
                save_dir,
                model_path="experiments/finetune_vie_detector/apr_02_2025/config_31/resnet50.pth",
            )
