import os
import re
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class MemeFilterVLM:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-32B-Instruct"):
        """
        Initializes the massive Qwen2.5-VL-32B model.
        Automatically distributes the weights across available GPUs.
        """
        print(f"Loading processor from {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"Loading {model_id} across GPUs. This may take a few minutes...")
        # device_map="auto" is crucial here to split the 32B model across your 4 GPUs
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" 
        )
        self.model.eval()
        print("Model loaded successfully!")

    def _extract_tags(self, text):
        """Helper function to safely extract content from <reason> and <answer> tags."""
        result = {
            "status": "PARSE_ERROR",
            "reasoning": "",
            "raw_output": text
        }
        
        reason_match = re.search(r'<reason>(.*?)</reason>', text, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        
        if reason_match:
            result["reasoning"] = reason_match.group(1).strip()
            
        if answer_match:
            answer_text = answer_match.group(1).strip().upper()
            if "UNKNOWN" in answer_text:
                result["status"] = "UNKNOWN"
            elif "KNOWN" in answer_text:
                result["status"] = "KNOWN"
            else:
                result["status"] = answer_text
                
        return result

    def infer_meme(self, image_path):
        """
        Analyzes a single image to determine if it's a global meme or a local/unknown one.
        Returns a dictionary containing the structured reasoning.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        # The Zero-Shot Filtering System Prompt
        system_prompt = (
            "You are an expert archivist of internet culture. "
            "Your only job is to determine if this image is a 'Common Meme'. "
            "A Common Meme is defined as a meme that can be easily understood through universally shared experiences, "
            "basic facts, or global internet culture (e.g., relatable daily struggles, standard reaction faces). "
            "Strict Rules:\n"
            "1. DO NOT GUESS. If the humor relies on hyper-local slang, regional figures, or requires "
            "specific cultural context from Southeast Asia (Vietnam, Indonesia), classify it as UNKNOWN.\n"
            "2. Only classify as KNOWN if the joke relies entirely on a universally shared experience or fact.\n"
            "Format your response EXACTLY as follows:\n"
            "<reason>\n"
            "Explain your thought process step-by-step. Identify any text, visual tropes, or cultural markers.\n"
            "</reason>\n"
            "<answer>\n"
            "KNOWN or UNKNOWN\n"
            "</answer>"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    # Keep max_pixels reasonable to prevent OOM errors during inference
                    {"type": "image", "image": image, "max_pixels": 1003520}, 
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

        # Prepare inputs exactly as Qwen2.5-VL expects
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate the response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512, # Increased to allow room for the reasoning block
                temperature=0.1,    # Low temperature for highly deterministic, robotic parsing
                do_sample=False
            )
            
        # Trim off the prompt tokens from the generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode text
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Extract tags and return
        return self._extract_tags(output_text)

def process_image_folder(analyzer, base_folder, output_jsonl="meme_analysis_results.jsonl"):
    """
    Scans a base folder for 'positive' and 'negative' subdirectories,
    runs the VLM on all images, and saves the results to a JSONL file.
    """
    from tqdm import tqdm
    
    subdirs = ["positive", "negative"]
    image_paths = []
    
    # Gather all images
    for subdir in subdirs:
        dir_path = os.path.join(base_folder, subdir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_paths.append((os.path.join(dir_path, filename), subdir))
        else:
            print(f"Warning: Subdirectory '{dir_path}' not found.")

    if not image_paths:
        print(f"No images found in {base_folder}/[positive|negative]")
        return

    print(f"Found {len(image_paths)} images. Starting inference...")
    
    # Process and save incrementally
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        for img_path, ground_truth in tqdm(image_paths, desc="Analyzing Images"):
            result = analyzer.infer_meme(img_path)
            if result:
                # Add metadata to the result so we know where it came from
                result["image_path"] = img_path
                result["original_folder"] = ground_truth
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
    print(f"\nFinished processing! Results saved to {output_jsonl}")

# --- Quick Test Block ---
if __name__ == "__main__":
    # Initialize the class
    analyzer = MemeFilterVLM()
    
    # Define your base folder containing 'positive' and 'negative' subfolders
    # e.g., "dataset/raw_memes/vietnamese"
    TARGET_FOLDER = "dataset/sample-common-meme-detector" 
    
    if os.path.exists(TARGET_FOLDER):
        process_image_folder(analyzer, TARGET_FOLDER)
    else:
        print(f"Please create the folder '{TARGET_FOLDER}' and add 'positive'/'negative' subfolders inside it.")
