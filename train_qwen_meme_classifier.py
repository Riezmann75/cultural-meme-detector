import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
            # 1. Iterate through language folders ('id', 'vi')
            for lang in os.listdir(split_dir):
                lang_dir = os.path.join(split_dir, lang)
                
                if os.path.isdir(lang_dir):
                    # 2. Iterate through the class folders ('negative', 'positive')
                    for class_name, label in label_map.items():
                        class_dir = os.path.join(lang_dir, class_name)
                        
                        if os.path.exists(class_dir) and os.path.isdir(class_dir):
                            # 3. Collect all images
                            for img_name in os.listdir(class_dir):
                                # Optional: filter to ensure we only grab images
                                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                    self.samples.append((os.path.join(class_dir, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Target text generation: 'positive' folder -> YES, 'negative' folder -> NO
        target_text = "YES" if label == 1 else "NO"

        # Formulate the prompt using Qwen's Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Is this image a cultural meme specific to Vietnam or Indonesia? Answer only YES or NO."},
                ],
            },
            {
                "role": "assistant",
                "content": target_text
            }
        ]

        # The processor handles the dynamic resolution and tokenization
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # --- NEW FIX: Only squeeze text tensors ---
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        # We deliberately DO NOT squeeze inputs["pixel_values"] or inputs["image_grid_thw"]
        
        # NEW: Squeeze the M-RoPE token type IDs
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"].squeeze(0)
        
        # For causal LM, labels are the input_ids
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    
def custom_collate_fn(batch):
    # Pad sequences to the longest in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=151643) # Qwen pad token
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    
    # NEW: Pad the multimodal token type IDs (padding with 0 is standard for text/ignore)
    mm_token_type_ids = torch.nn.utils.rnn.pad_sequence([item['mm_token_type_ids'] for item in batch], batch_first=True, padding_value=0)
    
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids # <-- NEW: Passing it to the model
    }

# --- 2. TRAINING LOOP ---
def train():
    # Initialize Accelerate (Handles DDP across your 2 GPUs automatically)
    accelerator = Accelerator()
    
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    BATCH_SIZE = 4 # Keep this small! MLLMs use a lot of VRAM.
    EPOCHS = 5
    LR = 2e-5 # MLLMs need much smaller learning rates than SigLIP
    
    if accelerator.is_main_process:
        print("Loading Processor and Model...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Load model in bfloat16 to save memory and speed up training on modern GPUs
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index} # Assigns model to the correct GPU
    )
    
    # --- LORA SETUP ---
    # We freeze the 2B parameters and only train a small "adapter" (approx ~20M parameters)
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # --- DATA & OPTIMIZER ---
    train_ds = QwenMemeDataset("dataset/split_dataset", "train", processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Prepare everything with accelerator (this replaces all your manual DDP rank/device code!)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # --- THE LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Only show progress bar on the main GPU
        loop = tqdm(train_loader, disable=not accelerator.is_main_process)
        for batch in loop:
            optimizer.zero_grad()
            
            # Forward pass (Loss is calculated automatically internally for causal LMs)
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass using accelerator
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Save the LoRA adapter
    if accelerator.is_main_process:
        print("Saving LoRA adapter...")
        # Unwrap the model to save it properly
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("qwen2_meme_lora")
        processor.save_pretrained("qwen2_meme_lora")
        print("Training Complete!")

if __name__ == "__main__":
    train()
