import os
import glob
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_latest_config_dir(base_results_dir="results", target_date=None):
    """Finds the most recently created config directory, optionally for a specific date."""
    if target_date:
        latest_date_dir = os.path.join(base_results_dir, target_date)
        if not os.path.exists(latest_date_dir):
            raise ValueError(f"No results directory found for date: {target_date}")
    else:
        date_dirs = sorted(glob.glob(os.path.join(base_results_dir, "202*")))
        if not date_dirs:
            raise ValueError("No results directory found.")
        latest_date_dir = date_dirs[-1]
    
    config_dirs = sorted(glob.glob(os.path.join(latest_date_dir, "config_*")), 
                         key=lambda x: int(x.split('_')[-1]))
    if not config_dirs:
        raise ValueError(f"No config directories found in {latest_date_dir}.")
    return config_dirs[-1]

def gather_data(data_root, split="test"):
    samples = []
    split_dir = os.path.join(data_root, split)
    label_map = {"negative": 0, "positive": 1}
    
    if not os.path.exists(split_dir):
        print(f"Warning: Directory {split_dir} does not exist.")
        return samples

    for lang in os.listdir(split_dir):
        lang_dir = os.path.join(split_dir, lang)
        if os.path.isdir(lang_dir):
            for class_name, label in label_map.items():
                class_dir = os.path.join(lang_dir, class_name)
                if os.path.exists(class_dir) and os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            # We now append the lang so we can sort false predictions by language
                            samples.append((os.path.join(class_dir, img_name), label, lang))
    return samples

def evaluate_split(model, processor, samples, split_name):
    y_true, y_pred = [], []
    false_preds = {}

    print(f"\nEvaluating {split_name.upper()} split on {len(samples)} images...")
    
    for img_path, true_label, lang in tqdm(samples, desc=f"{split_name.capitalize()}"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "max_pixels": 262144},
                    {"type": "text", "text": "Task: Classify if this image is a cultural meme from Vietnam or Indonesia. Output only YES or NO."},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=5)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().lower()

        pred_label = 1 if "yes" in output_text else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)

        # Track false predictions
        if pred_label != true_label:
            if lang not in false_preds:
                false_preds[lang] = {"false_positives": [], "false_negatives": []}
            if pred_label == 1:
                false_preds[lang]["false_positives"].append(img_path)
            else:
                false_preds[lang]["false_negatives"].append(img_path)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Not Meme", "Meme"], zero_division=0)
    
    return acc, cm, report, false_preds

def evaluate(target_date=None):
    BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    DATA_ROOT = "dataset/split_dataset"
    
    # 1. Locate the latest experiment (or specific date)
    config_dir = get_latest_config_dir(target_date=target_date)
    lora_dir = os.path.join(config_dir, "qwen2_meme_lora")
    log_file = os.path.join(config_dir, "experiment_log.jsonl")
    
    print(f"Evaluating Experiment: {config_dir}")
    
    # 2. Parse the JSONL to get training history
    epochs, train_losses, val_losses = [], [], []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get("type") == "epoch_metrics":
                    epochs.append(data["epoch"])
                    train_losses.append(data["train_loss"])
                    val_losses.append(data["val_loss"])
    
    # 3. Load Model
    print("Loading Processor and Model...")
    processor = AutoProcessor.from_pretrained(lora_dir)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    # 4. Evaluate all splits
    splits_to_eval = ["train", "val", "test"]
    all_false_preds = {}
    test_acc = 0.0

    with torch.no_grad():
        for split in splits_to_eval:
            samples = gather_data(DATA_ROOT, split)
            if not samples:
                continue
            
            acc, cm, report, split_false_preds = evaluate_split(model, processor, samples, split)
            all_false_preds[split] = split_false_preds
            
            if split == "test":
                test_acc = acc
            
            print(f"\n=== {split.upper()} RESULTS ===")
            print(f"Accuracy: {acc:.4f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(report)

    # 5. Save False Predictions to JSON
    fp_filename = os.path.join(config_dir, "false_predictions.json")
    with open(fp_filename, "w", encoding="utf-8") as f:
        json.dump(all_false_preds, f, indent=4)
    print(f"\n[Success] False predictions saved to: {fp_filename}")

    # 6. Plot and Save Graph
    if epochs:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, marker='o', label='Training Loss', color='blue')
        plt.plot(epochs, val_losses, marker='o', label='Validation Loss', color='red')
        plt.title(f'Train vs. Validation Loss (Test Acc = {test_acc:.2f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_filename = f"Train vs. Validation loss, final acc.={test_acc:.2f}.png"
        plot_path = os.path.join(config_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Success] Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    # Specify a date like "20260413" or leave as None to automatically grab the closest/latest date
    TARGET_DATE = None 
    evaluate(target_date=TARGET_DATE)
