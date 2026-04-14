import os
import torch
import json
import time
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image


# --- 1. THE MLLM DATASET ---
class QwenMemeDataset(Dataset):
    def __init__(self, data_root, split, processor):
        self.data_root = data_root
        self.split = split
        self.processor = processor

        self.samples = []
        split_dir = os.path.join(data_root, split)

        # Map your folder names to the binary labels
        label_map = {"negative": 0, "positive": 1}

        if os.path.exists(split_dir):
            for lang in os.listdir(split_dir):
                lang_dir = os.path.join(split_dir, lang)
                if os.path.isdir(lang_dir):
                    for class_name, label in label_map.items():
                        class_dir = os.path.join(lang_dir, class_name)
                        if os.path.exists(class_dir) and os.path.isdir(class_dir):
                            for img_name in os.listdir(class_dir):
                                if img_name.lower().endswith(
                                    (".png", ".jpg", ".jpeg", ".webp")
                                ):
                                    self.samples.append(
                                        (os.path.join(class_dir, img_name), label)
                                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        target_text = "YES" if label == 1 else "NO"

        # 1. Create ONLY the user prompt first
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Memory safety cap
                    {
                        "type": "text",
                        "text": "Task: Classify if this image is a cultural meme from Vietnam or Indonesia. Output only YES or NO.",
                    },
                ],
            }
        ]

        # 2. Get the raw text of the prompt (with the generation trigger)
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Append our target to create the full sequence for training
        full_text = prompt_text + target_text

        image_inputs, video_inputs = process_vision_info(prompt_messages)

        # 4. Tokenize the full sequence
        inputs = self.processor(
            text=[full_text], images=image_inputs, padding=True, return_tensors="pt"
        )

        # Squeeze sequence dims
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"].squeeze(0)

        # --- THE MASKING FIX ---
        labels = inputs["input_ids"].clone()

        prompt_only_inputs = self.processor(
            text=[prompt_text], images=image_inputs, return_tensors="pt"
        )
        prompt_length = prompt_only_inputs.input_ids.shape[1]

        labels[:prompt_length] = -100
        inputs["labels"] = labels

        return inputs


def custom_collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=151643
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100
    )

    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    if "mm_token_type_ids" in batch[0]:
        mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            [item["mm_token_type_ids"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        batch_dict["mm_token_type_ids"] = mm_token_type_ids

    return batch_dict


# --- 2. TRAINING LOOP ---
def train():
    ACCUMULATION_STEPS = 4
    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    BATCH_SIZE = 2  # Physical batch size of 2 per GPU
    EPOCHS = 10
    LR = 2e-4

    # --- EARLY STOPPING CONFIG ---
    USE_EARLY_STOPPING = False
    PATIENCE = 2

    if accelerator.is_main_process:
        today = time.strftime("%Y%m%d")
        base_dir = os.path.join("results", today)
        os.makedirs(base_dir, exist_ok=True)

        config_idx = 0
        while os.path.exists(os.path.join(base_dir, f"config_{config_idx}")):
            config_idx += 1
        out_dir = os.path.join(base_dir, f"config_{config_idx}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving experiment to: {out_dir}")

        log_file = os.path.join(out_dir, "experiment_log.jsonl")
        hyperparams = {
            "model_id": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "use_early_stopping": USE_EARLY_STOPPING,
            "patience": PATIENCE,
            "scheduler": "CosineAnnealingLR",
        }
        with open(log_file, "a") as f:
            f.write(json.dumps({"type": "config", "data": hyperparams}) + "\n")

    if accelerator.is_main_process:
        print("Loading Processor and Model...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": accelerator.process_index}
    )

    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print()
    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    train_ds = QwenMemeDataset("dataset/split_dataset", "train", processor)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    val_ds = QwenMemeDataset("dataset/split_dataset", "val", processor)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # 1. Prepare model, optimizer, and loaders FIRST so the dataloader length is adjusted for multiple GPUs
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 2. Calculate exact total update steps for the Scheduler
    total_update_steps = math.ceil(len(train_loader) / ACCUMULATION_STEPS) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_update_steps, eta_min=1e-6
    )

    # 3. Prepare scheduler with accelerate
    scheduler = accelerator.prepare(scheduler)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        loop = tqdm(train_loader, disable=not accelerator.is_main_process)
        for step, batch in enumerate(loop):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()  # Accelerate safely handles stepping this only on optimizer updates
                optimizer.zero_grad()

            total_train_loss += loss.item()

            # Log current LR in the progress bar
            current_lr = scheduler.get_last_lr()[0]
            loop.set_description(
                f"Epoch [{epoch+1}/{EPOCHS}] LR: {current_lr:.2e} Loss: {loss.item():.4f}"
            )

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        val_loop = tqdm(
            val_loader,
            desc="Validating",
            disable=not accelerator.is_main_process,
            leave=False,
        )

        with torch.no_grad():
            for batch in val_loop:
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        avg_val_loss_tensor = torch.tensor(avg_val_loss, device=accelerator.device)
        avg_val_loss_tensor = accelerator.reduce(avg_val_loss_tensor, reduction="mean")
        global_avg_val_loss = avg_val_loss_tensor.item()

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1} completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {global_avg_val_loss:.4f}"
            )

            with open(log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "epoch_metrics",
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": global_avg_val_loss,
                        }
                    )
                    + "\n"
                )

        # --- EARLY STOPPING & SAVING ---
        if global_avg_val_loss < best_val_loss:
            best_val_loss = global_avg_val_loss
            epochs_no_improve = 0

            if accelerator.is_main_process:
                print(
                    f"➔ Validation loss improved to {best_val_loss:.4f}. Saving best model checkpoint..."
                )
                unwrapped_model = accelerator.unwrap_model(model)
                lora_dir = os.path.join(out_dir, "qwen2_meme_lora")
                unwrapped_model.save_pretrained(lora_dir)
                processor.save_pretrained(lora_dir)
        else:
            epochs_no_improve += 1
            if accelerator.is_main_process:
                if USE_EARLY_STOPPING:
                    print(
                        f"➔ Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}"
                    )
                else:
                    print(f"➔ Validation loss did not improve. (Early stopping is OFF)")

            if USE_EARLY_STOPPING and epochs_no_improve >= PATIENCE:
                if accelerator.is_main_process:
                    print(
                        f"\n[!] Early stopping triggered! Validation loss has not improved for {PATIENCE} epochs."
                    )
                break

    if accelerator.is_main_process:
        print(
            "Training Complete! The best performing model weights are saved in your results folder."
        )


if __name__ == "__main__":
    train()
