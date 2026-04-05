import os
import time
import json
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from lib.plot import plot_losses
from models import SiglipMemeClassifier
from dataset import MemeDataset


def train(
    model, train_loader, val_loader, optimizer, criterion, scheduler, hyper_dict, device
):
    """
    Refactored train function with Early Stopping and validation monitoring.
    """
    EPOCHS = hyper_dict.get("epochs", 10)
    patience = hyper_dict.get("patience", 3)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0
        for imgs, lbls in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False
        ):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for val_imgs, val_lbls in val_loader:
                val_imgs, val_lbls = val_imgs.to(device), val_lbls.to(device)
                val_logits = model(val_imgs)
                v_loss = criterion(val_logits, val_lbls)
                running_val_loss += v_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # # Early Stopping Logic
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     # Save the best model
        #     torch.save(model.state_dict(), "best_model_checkpoint.pth")
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f"Early stopping triggered at epoch {epoch+1}")
        #         break

    return train_losses, validation_losses


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            _, predicted = torch.max(logits, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
    return correct / total if total > 0 else 0


if __name__ == "__main__":
    # --- GRID SEARCH PARAMETERS ---
    GRID_PARAMS = {
        "lr": [5e-2, 1e-2, 5e-3, 1e-3],  # Increased LR range
        "batch_size": [128],
        "epochs": [100],
        "weight_decay": [1e-2, 1e-3, 8e-4],  # Increased WD
        "label_smoothing": [0.1, 0.2],  # New hyperparam
        "dropout": [0, 0.3, 0.5],  # New hyperparam
    }
    
    # GRID_PARAMS = {
    #     "lr": [1e-2],  # Increased LR range
    #     "batch_size": [128],
    #     "epochs": [100],
    #     "weight_decay": [0],  # Increased WD
    #     "label_smoothing": [0],  # New hyperparam
    #     "dropout": [0],  # New hyperparam
    # }

    DATA_ROOT = "dataset/split_dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TODAY_DATE = time.strftime("%Y%m%d")
    LOG_DIR = f"results/{TODAY_DATE}"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "experiment_log.jsonl")

    # ENHANCED Data Augmentation to prevent overfitting
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    keys, values = zip(*GRID_PARAMS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(
        f"Starting Grid Search with {len(combinations)} configs (Regularization Focused)."
    )

    for i, config in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Config: {config}")

        train_ds = MemeDataset(DATA_ROOT, "train", transform=train_transform)
        val_ds = MemeDataset(DATA_ROOT, "val", transform=val_transform)

        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=4)

        # Update model with dynamic dropout if SiglipMemeClassifier supports it
        # Assuming we passed config['dropout'] to the model init
        model = SiglipMemeClassifier(dropout=config.get("dropout", 0)).to(DEVICE)

        # LABEL SMOOTHING: Prevents the model from becoming overconfident
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get("label_smoothing", 0.0)
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"]
        )

        train_losses, val_losses = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            scheduler,
            config,
            DEVICE,
        )

        train_acc = evaluate(model, train_loader, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {**config, "optimizer": "Adam"},
            "metrics": {
                "final_train_acc": round(train_acc, 4),
                "final_val_acc": round(val_acc, 4),
                "avg_train_losses": [round(l, 6) for l in train_losses],
                "avg_val_losses": [round(l, 6) for l in val_losses],
            },
        }

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Config {i+1} Result: Val Acc {val_acc:.4f}.")

    print(f"\nGrid search complete.")
