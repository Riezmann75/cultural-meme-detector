import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from ddp.lib.utils import reduce_tensor

from .loader import DataFetcher

from .early_stopper import EarlyStopper


class Trainer:
    def __init__(
        self,
        is_distributed,
        rank,
        model,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer,
        scheduler,
        profiler: torch.profiler.profile = None,
    ):
        self.is_distributed = is_distributed
        self.model = model
        self.rank = rank
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.profiler = profiler
        self.early_stopper = EarlyStopper(patience=7, delta=0.001)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        config_index: Optional[int],
        num_epochs=10,
    ):
        train_losses = torch.zeros(
            (num_epochs, 1), device=self.rank, requires_grad=False
        )
        train_accs = torch.zeros((num_epochs, 1), device=self.rank, requires_grad=False)
        val_losses = torch.zeros((num_epochs, 1), device=self.rank, requires_grad=False)
        val_accs = torch.zeros((num_epochs, 1), device=self.rank, requires_grad=False)
        val_precisions = torch.zeros(
            (num_epochs, 1), device=self.rank, requires_grad=False
        )
        test_accs, test_precisions = None, None
        if test_loader:
            test_accs = torch.zeros(
                (num_epochs, 1), device=self.rank, requires_grad=False
            )
            test_precisions = torch.zeros(
                (num_epochs, 1), device=self.rank, requires_grad=False
            )
        for epoch in tqdm(
            range(num_epochs), desc=f"Training with config {config_index}"
        ):
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)
                
            # Train model
            self.model.train()
            running_loss, running_correct, *_ = self.consume_data(train_loader)

            # update learning rate
            self.scheduler.step()

            # calculate average loss and accuracy
            num_device = int(os.getenv("LOCAL_WORLD_SIZE", 1))

            train_loss = num_device * running_loss / len(train_loader.dataset)
            train_acc = num_device * running_correct / len(train_loader.dataset)

            # update history
            train_losses[epoch] = train_loss.reshape(-1, 1)
            train_accs[epoch] = train_acc.reshape(-1, 1)

            # validate model
            self.model.eval()
            with torch.no_grad():
                (
                    running_loss,
                    running_correct,
                    true_positive,
                    false_positive,
                ) = self.consume_data(val_loader, requires_grad=False)

            val_precision = torch.div(true_positive, (true_positive + false_positive))
            val_loss = num_device * running_loss / len(val_loader.dataset)
            val_acc = num_device * running_correct / len(val_loader.dataset)

            # update history
            val_losses[epoch] = val_loss.reshape(-1, 1)
            val_accs[epoch] = val_acc.reshape(-1, 1)
            val_precisions[epoch] = val_precision.reshape(-1, 1)

            # test model
            if test_loader:
                self.model.eval()
                with torch.no_grad():
                    (
                        running_loss,
                        running_correct,
                        true_positive,
                        false_positive,
                    ) = self.consume_data(test_loader, requires_grad=False)

                running_correct = reduce_tensor(running_correct, average=False)
                true_positive = reduce_tensor(true_positive)
                false_positive = reduce_tensor(false_positive)

                test_precision = torch.div(
                    true_positive, (true_positive + false_positive)
                )
                test_acc = running_correct / len(test_loader.dataset)
                test_precisions[epoch] = test_precision.reshape(-1, 1)
                test_accs[epoch] = test_acc.reshape(-1, 1)

            # if self.early_stopper(train_loss):
            #     print(f"Early stopping at epoch {epoch}")
            #     break

        torch.cuda.empty_cache()
        return (
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            val_precisions,
            test_accs,
            test_precisions,
        )

    def consume_data(self, data_loader: DataLoader, requires_grad=True):
        running_loss = torch.zeros(1, device=self.rank, requires_grad=False)
        running_correct = torch.zeros(1, device=self.rank, requires_grad=False)
        true_positive = torch.zeros(1, device=self.rank, requires_grad=False)
        false_positive = torch.zeros(1, device=self.rank, requires_grad=False)

        prefetcher = DataFetcher(data_loader)
        inputs, labels = prefetcher.next()

        while inputs is not None:
            self.optimizer.zero_grad()
            # if requires_grad:
            #     mix_up = v2.MixUp(alpha=12e-3, num_classes=3)
            #     inputs, labels = mix_up(inputs, labels)

            outputs = self.model(inputs)
            loss = self.criterion.forward(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if requires_grad:
                loss.backward()
                self.optimizer.step()
                if self.profiler:
                    self.profiler.step()
                # labels = torch.max(labels, 1)[1]

            batch_running_correct = torch.sum(preds == labels)
            batch_true_positive = torch.sum(preds & labels)
            batch_false_positive = torch.sum(preds & ~labels)

            running_loss += loss * inputs.size(0)
            running_correct += batch_running_correct
            true_positive += batch_true_positive
            false_positive += batch_false_positive

            inputs, labels = prefetcher.next()
        return running_loss, running_correct, true_positive, false_positive
