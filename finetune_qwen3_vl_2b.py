import argparse
from datetime import datetime
import json
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator


class MemeReasoningDataset(Dataset):
    def __init__(self, jsonl_file, system_prompt, image_root_dir):
        self.jsonl_file = jsonl_file
        self.system_prompt = system_prompt
        self.image_root_dir = image_root_dir
        self.positive_image_dir = os.path.join(image_root_dir, "positive")
        self.negative_image_dir = os.path.join(image_root_dir, "negative")

        pos_files = (
            set(os.listdir(self.positive_image_dir))
            if os.path.exists(self.positive_image_dir)
            else set()
        )
        neg_files = (
            set(os.listdir(self.negative_image_dir))
            if os.path.exists(self.negative_image_dir)
            else set()
        )

        with open(jsonl_file, "r") as f:
            self.data = [json.loads(line) for line in f]
            self.data = [
                annot
                for annot in self.data
                if annot["image_path"].split("/")[-1] in pos_files
                or annot["image_path"].split("/")[-1] in neg_files
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        reasoning = item["reason"]
        image_path = item["image_path"]

        label = 1 if "positive" in image_path else 0

        return reasoning, image_path, label


class MemeCollateFn:
    # --- FIX 2: Multiprocessing Crash ---
    # We pass the processor here, NOT the model. Pickling an entire 3B parameter model
    # to send to 4 dataloader workers causes an immediate OOM/Pickle crash.
    def __init__(self, processor, system_prompt):
        self.processor = processor
        self.system_prompt = system_prompt

    def __call__(self, batch):
        processed_inputs = []
        labels = []
        for reasoning, image_path, label in batch:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": f"{self.system_prompt}\n\n{reasoning}",
                        },
                    ],
                }
            ]

            image_inputs, video_inputs = process_vision_info(messages)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            processed_inputs.append(
                {
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "pixel_values": inputs["images"].squeeze(0),
                    "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
                }
            )
            labels.append(label)

        pixel_values = tuple([item["pixel_values"] for item in processed_inputs])
        image_grid_thw = tuple([item["image_grid_thw"] for item in processed_inputs])

        max_length = max(input["input_ids"].shape[1] for input in processed_inputs)
        batch_size = len(processed_inputs)

        # --- FIX 3: Padding Token ---
        # We dynamically fetch the pad token. If None, fallback to eos_token_id.
        pad_token_id = (
            self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id
        )

        attn_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        input_ids = torch.full(
            (batch_size, max_length), fill_value=pad_token_id, dtype=torch.long
        )

        for i, input in enumerate(processed_inputs):
            length = input["input_ids"].shape[1]
            input_ids[i, -length:] = input["input_ids"]
            attn_mask[i, -length:] = 1

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
        }

        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        return inputs, labels


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(
    processor,  # Changed from model to processor
    jsonl_file,
    image_root_dir,
    system_prompt: str,
    batch_size=1,
    num_workers=1,
    **kwargs,
):
    dataset = MemeReasoningDataset(
        jsonl_file=jsonl_file,
        system_prompt=system_prompt,
        image_root_dir=image_root_dir,
    )  # Removed the unexpected kwargs 'model=model' that would throw a TypeError

    collate_fn = MemeCollateFn(processor=processor, system_prompt=system_prompt)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # Consider setting True for train loader outside this function
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )

    return dataloader, dataset


def get_output_dir(base_results_path="results/finetuned-qwen3-vl-2b"):
    date_str = datetime.now().strftime("%Y%m%d")
    date_path = os.path.join(base_results_path, date_str)
    os.makedirs(date_path, exist_ok=True)

    existing_configs = [d for d in os.listdir(date_path) if d.startswith("config_")]
    config_numbers = []
    for d in existing_configs:
        try:
            num = int(d.split("_")[1])
            config_numbers.append(num)
        except (ValueError, IndexError):
            continue

    next_num = max(config_numbers) + 1 if config_numbers else 1
    config_folder_name = f"config_{next_num}"
    config_path = os.path.join(date_path, config_folder_name)
    os.makedirs(config_path, exist_ok=True)
    return config_path


class QwenVLBinaryClassifier(nn.Module):
    def __init__(self, model_id, lora_config):
        super().__init__()

        base_model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

        self.backbone = get_peft_model(base_model, lora_config)
        hidden_size = self.backbone.config.hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1, dtype=torch.bfloat16),
        )

    def forward(
        self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        last_layer_hidden_states = outputs.hidden_states[-1]
        final_token_embedding = last_layer_hidden_states[:, -1, :]
        logits = self.classifier_head(final_token_embedding)

        return logits


if __name__ == "__main__":
    seed_everything(42)

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-2B with a Binary Classification Head"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Pass this flag to save the model after training",
    )
    args = parser.parse_args()

    output_dir = get_output_dir(base_results_path="results/finetuned-qwen3-vl-2b")

    print("Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    print("Building Model...")
    model = QwenVLBinaryClassifier(MODEL_ID, lora_config)

    BATCH_SIZE = 4
    EPOCHS = 15
    LR = 1e-4

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    accelerator = Accelerator()

    # Pass the processor instead of the model here!
    train_loader, _ = create_dataloader(
        processor=processor,
        jsonl_file="generated_responses.jsonl",
        image_root_dir="dataset/sample-cultural-meme-detector/split_dataset/train/vi",
        system_prompt="Given the image and the reasoning provided, classify if the meme is a cultural meme or common meme.",
        batch_size=BATCH_SIZE,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )

    validation_loader, _ = create_dataloader(
        processor=processor,
        jsonl_file="generated_responses.jsonl",
        image_root_dir="dataset/sample-cultural-meme-detector/split_dataset/val/vi",
        system_prompt="Given the image and the reasoning provided, classify if the meme is a cultural meme or common meme.",
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    test_loader, _ = create_dataloader(
        processor=processor,
        jsonl_file="generated_responses.jsonl",
        image_root_dir="dataset/sample-cultural-meme-detector/split_dataset/test/vi",
        system_prompt="Given the image and the reasoning provided, classify if the meme is a cultural meme or common meme.",
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    (
        model,
        optimizer,
        train_loader,
        validation_loader,
        test_loader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
        test_loader,
        scheduler,
    )

    global_train_losses = []
    global_val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        train_loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
            disable=not accelerator.is_main_process,
        )

        for batch in train_loop:
            optimizer.zero_grad()
            inputs, labels = batch

            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )

            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()

            total_train_loss += loss.item()
            if accelerator.is_main_process:
                train_loop.set_postfix(loss=total_train_loss / (train_loop.n + 1))

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_loss_tensor = torch.tensor(avg_train_loss, device=accelerator.device)
        global_train_loss = accelerator.reduce(
            avg_train_loss_tensor, reduction="mean"
        ).item()

        model.eval()
        val_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        all_gathered_losses = []

        val_loop = tqdm(
            validation_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Val]",
            disable=not accelerator.is_main_process,
        )

        with torch.no_grad():
            for batch in val_loop:
                inputs, labels = batch

                logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                )

                individual_losses = val_loss_fn(logits, labels)
                gathered_losses = accelerator.gather_for_metrics(individual_losses)
                all_gathered_losses.extend(gathered_losses.cpu().tolist())

                if accelerator.is_main_process:
                    current_avg = (
                        sum([l[0] for l in all_gathered_losses])
                        / len(all_gathered_losses)
                        if all_gathered_losses
                        else 0
                    )
                    val_loop.set_postfix(val_loss=current_avg)

        global_val_loss = (
            sum([l[0] for l in all_gathered_losses]) / len(all_gathered_losses)
            if all_gathered_losses
            else 0.0
        )

        scheduler.step()

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1} Summary | Train Loss: {global_train_loss:.4f} | Val Loss: {global_val_loss:.4f}"
            )
            global_train_losses.append(global_train_loss)
            global_val_losses.append(global_val_loss)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            log_dict = {
                "num_epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "Lora_config": {
                    "r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "target_modules": lora_config.target_modules,
                    "lora_dropout": lora_config.lora_dropout,
                    "bias": lora_config.bias,
                    "task_type": lora_config.task_type,
                },
                "train_losses": global_train_losses,
                "val_losses": global_val_losses,
            }
            json.dump(log_dict, f, indent=2)

    if args.save_model:
        if accelerator.is_main_process:
            import os

            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving Fine-Tuned Model to {output_dir}...")

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.backbone.save_pretrained(
                os.path.join(output_dir, "lora_backbone")
            )
            torch.save(
                unwrapped_model.classifier_head.state_dict(),
                os.path.join(output_dir, "custom_head.pth"),
            )
            print("[Success] Model saved.")
    else:
        if accelerator.is_main_process:
            print("Training complete. Skipping model saving as requested.")
