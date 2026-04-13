import os
import torch
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

        # 1. Create ONLY the user prompt first (Note the strict command prompt!)
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "max_pixels": 262144,
                    },  # Memory safety cap
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

        # --- THE MASKING FIX (Breaks Mode Collapse) ---
        labels = inputs["input_ids"].clone()

        # FIX: Tokenize the prompt WITH images to find out the EXACT token length
        # Without passing images here, it doesn't count the hundreds of image patches!
        prompt_only_inputs = self.processor(
            text=[prompt_text], images=image_inputs, return_tensors="pt"
        )
        prompt_length = prompt_only_inputs.input_ids.shape[1]

        # Set everything EXCEPT the answer to -100 so the model ignores it during loss calculation
        labels[:prompt_length] = -100

        inputs["labels"] = labels

        return inputs


def custom_collate_fn(batch):
    # Pad sequences to the longest in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=151643
    )  # Qwen pad token
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100
    )  # -100 is ignored by CrossEntropyLoss

    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    # Add M-RoPE Token type IDs if they exist
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
    # Initialize Accelerate with Gradient Accumulation
    accelerator = Accelerator(gradient_accumulation_steps=4)

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    BATCH_SIZE = 1  # Physical batch size of 1 per GPU
    EPOCHS = 5
    LR = 2e-5

    if accelerator.is_main_process:
        print("Loading Processor and Model...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Load model in bfloat16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": accelerator.process_index}
    )

    # --- LORA SETUP ---
    model.gradient_checkpointing_enable()
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
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # --- DATA & OPTIMIZER ---
    train_ds = QwenMemeDataset("dataset/split_dataset", "train", processor)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Prepare everything with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # --- THE LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, disable=not accelerator.is_main_process)
        for step, batch in enumerate(loop):
            # Accumulate gradients seamlessly using Accelerate context manager
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Save the LoRA adapter
    if accelerator.is_main_process:
        print("Saving LoRA adapter...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("qwen2_meme_lora")
        processor.save_pretrained("qwen2_meme_lora")
        print("Training Complete!")


if __name__ == "__main__":
    train()
