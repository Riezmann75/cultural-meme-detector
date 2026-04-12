import os
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report, confusion_matrix

def gather_test_data(data_root, split="test"):
    """
    Crawls the test directory using the same language/class structure as training.
    Returns a list of tuples: (image_path, ground_truth_label)
    """
    samples = []
    split_dir = os.path.join(data_root, split)
    label_map = {"negative": 0, "positive": 1}
    
    if not os.path.exists(split_dir):
        raise ValueError(f"Directory {split_dir} does not exist.")

    for lang in os.listdir(split_dir):
        lang_dir = os.path.join(split_dir, lang)
        if os.path.isdir(lang_dir):
            for class_name, label in label_map.items():
                class_dir = os.path.join(lang_dir, class_name)
                if os.path.exists(class_dir) and os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            samples.append((os.path.join(class_dir, img_name), label))
    return samples

def evaluate():
    BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    LORA_DIR = "qwen2_meme_lora" # Where you saved the adapter
    DATA_ROOT = "dataset/split_dataset"
    
    print("1. Loading Processor and Base Model...")
    processor = AutoProcessor.from_pretrained(LORA_DIR)
    
    # Load base model in bfloat16
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map="auto" # Automatically places it on the best GPU
    )
    
    print("2. Injecting LoRA Adapter...")
    # This applies your trained fine-tune weights to the base model
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    print("3. Gathering Test Images...")
    test_samples = gather_test_data(DATA_ROOT, split="test")
    print(f"Found {len(test_samples)} test images.")

    y_true = []
    y_pred = []

    print("4. Starting Evaluation (Batch Size = 1 for safety)...")
    with torch.no_grad():
        for img_path, true_label in tqdm(test_samples):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            # Note: We do NOT append the assistant's answer here. 
            # We want the model to generate it!
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "max_pixels": 262144}, # Keep the safety cap!
                        {"type": "text", "text": "Is this image a cultural meme specific to Vietnam or Indonesia? Answer only YES or NO."},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate output
            # max_new_tokens is very small (5) because we only want a YES/NO answer.
            generated_ids = model.generate(**inputs, max_new_tokens=5)
            
            # Slice the generated_ids to only get the NEW tokens (ignore the prompt tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode the text
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip().lower()

            # Robust parsing (in case the model says "Yes.", "yes", or "Yes it is")
            if "yes" in output_text:
                pred_label = 1
            else:
                # If it says "no", or hallucinates something else, we count it as negative
                pred_label = 0

            y_true.append(true_label)
            y_pred.append(pred_label)

    print("\n=== EVALUATION RESULTS ===")
    print("Label Map: 0 = Not Meme (Negative), 1 = Meme (Positive)\n")
    print(confusion_matrix(y_true, y_pred))
    print("\n")
    print(classification_report(y_true, y_pred, target_names=["Not Meme", "Meme"]))

if __name__ == "__main__":
    evaluate()
