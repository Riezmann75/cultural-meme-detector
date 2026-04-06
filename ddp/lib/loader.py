from typing import Optional
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from transforms.padder import Pad
from ddp.lib.dataset.subset import SubSet

from ddp.lib.dataset.meme import MemeDataset


def resize_with_padding(image, target_size):
    image_width, image_height = image.size
    scale = target_size / max(image_width, image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    image = image.resize((new_width, new_height))

    delta_w = target_size - new_width
    delta_h = target_size - new_height

    padded_w = delta_w // 2
    padded_h = delta_h // 2

    padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded_img.paste(image, (padded_w, padded_h))

    return padded_img


def load_data(
    is_distributed,
    root_dir,
    transforms,
    batch_size=32,
    num_workers=0,
):
    common_transforms = transforms["common"]
    train_transforms = transforms["train"]

    train_data = MemeDataset(
        root_dir=root_dir,
        split="train",
        transform=v2.Compose(common_transforms + train_transforms),
    )
    val_data = MemeDataset(
        root_dir=root_dir, split="val", transform=v2.Compose(common_transforms)
    )
    test_data = MemeDataset(
        root_dir=root_dir, split="test", transform=v2.Compose(common_transforms)
    )

    train_sampler, val_sampler, test_sampler = None, None, None
    train_loader, val_loader, test_loader = None, None, None

    if is_distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        val_sampler = DistributedSampler(val_data, shuffle=False)
        test_sampler = DistributedSampler(test_data, shuffle=False)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=test_sampler,
        shuffle=False,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class DataFetcher:
    def __init__(self, loader: DataLoader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_inputs, self.labels = next(self.loader)
        except StopIteration:
            self.next_inputs, self.labels = None, None
            return

        # load data concurrently using new cuda stream
        with torch.cuda.stream(self.stream):
            self.next_inputs = self.next_inputs.cuda(non_blocking=True)
            self.labels = self.labels.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
        labels = self.labels
        # load the next data to the device's default stream
        if inputs is not None:
            inputs.record_stream(torch.cuda.current_stream())
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream())
        self.preload()
        return inputs, labels
