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

        # Crucial for batch generation: padding must be on the left
        self.processor.tokenizer.padding_side = 'left'

        print(f"Loading {model_id} across GPUs. This may take a few minutes...")
        # device_map="auto" is crucial here to split the 32B model across your GPUs
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully!")

    def _extract_tags(self, text):
        """Helper function to safely extract content from <reason> and <answer> tags."""
        result = {"status": "PARSE_ERROR", "reasoning": "", "raw_output": text}

        reason_match = re.search(
            r"<reason>(.*?)</reason>", text, re.DOTALL | re.IGNORECASE
        )
        answer_match = re.search(
            r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE
        )

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

    def infer_meme_batch(self, image_paths):
        """
        Analyzes a batch of images to determine if they are global memes or local/unknown ones.
        Returns a list of dictionaries containing the structured reasoning for each image.
        """
        system_prompt = (
            "You are an expert archivist of internet culture. "
            "Your only job is to determine if this image is a 'Common Meme'. "
            "A Common Meme is defined as a meme that it's humor idea is universally shared, "
            "basic fact, or global internet culture (e.g., relatable daily struggles, standard reaction faces). "
            "Strict Rules:\n"
            "1. DO NOT GUESS. If you can confidently identify the humor idea and it completely relies on hyper-local slang, regional figures, or requires "
            "specific cultural context from Southeast Asia (Vietnam, Indonesia), or there is no joke, classify it as UNKNOWN.\n"
            "2. Be careful with the meme's vocabulary, if it uses local slang or references, it tends to be UNKNOWN.\n"
            "3. If you are not certain how the visual objects in the image contribute to the humor and the humor idea is not clear, classify as UNKNOWN.\n"
            "4. Look at the overall meaning, only classify the meme as KNOWN if the humor idea relies entirely on a universally shared experience or fact.\n"
            "5. There are local words used for emotional expresssion, you can ignore them if the humor is still clear without understanding those words.\n"
            "Put your reasoning trace and your final conclusion EXACTLY using the tags as follows:\n"
            "<reason>Explain your thought process step-by-step. Identify any text, visual tropes, or cultural markers.</reason>\n"
            "<answer> KNOWN or UNKNOWN </answer>"
        )

        valid_paths = []
        messages_batch = []

        # 1. Load images and construct the prompt for each image in the batch
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                valid_paths.append(path)
                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": system_prompt},
                        ],
                    }
                ])
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        if not messages_batch:
            return []

        # 2. Process texts and images for the entire batch
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = self.processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # 3. Generate the response for the batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Increased to allow room for the reasoning block
                temperature=0.1,  # Low temperature for highly deterministic, robotic parsing
                do_sample=False,
            )

        # 4. Trim prompt tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 5. Extract tags and map results back to their original file paths
        final_results = []
        for path, text in zip(valid_paths, output_texts):
            result = self._extract_tags(text.strip())
            result["image_path"] = path
            final_results.append(result)

        return final_results


def process_image_folder(
    analyzer, base_folder, output_jsonl="meme_analysis_results.jsonl", batch_size=4
):
    """
    Scans a base folder for 'positive' and 'negative' subdirectories,
    runs the VLM on all images in batches, and saves the results to a JSONL file.
    """
    from tqdm import tqdm

    subdirs = ["positive", "negative"]
    image_paths = []

    # Gather all images
    for subdir in subdirs:
        dir_path = os.path.join(base_folder, subdir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    image_paths.append((os.path.join(dir_path, filename), subdir))
        else:
            print(f"Warning: Subdirectory '{dir_path}' not found.")

    if not image_paths:
        print(f"No images found in {base_folder}/[positive|negative]")
        return

    print(f"Found {len(image_paths)} images. Starting batched inference (Batch size: {batch_size})...")

    # Helper function to yield batches
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    # Process and save incrementally
    with open(output_jsonl, "a", encoding="utf-8") as f:
        batches = list(chunker(image_paths, batch_size))
        for batch in tqdm(batches, desc="Analyzing Batches"):
            
            # Extract just the paths for the model
            batch_paths = [item[0] for item in batch]
            
            # Run batched inference
            results = analyzer.infer_meme_batch(batch_paths)
            
            # Write results back with their ground truth metadata
            for res in results:
                # Match the image path back to its ground_truth folder ("positive" or "negative")
                ground_truth = next((item[1] for item in batch if item[0] == res["image_path"]), "unknown")
                res["original_folder"] = ground_truth
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\nFinished processing! Results saved to {output_jsonl}")


# --- Quick Test Block ---
if __name__ == "__main__":
    # Initialize the class
    analyzer = MemeFilterVLM()

    # Define your base folder containing 'positive' and 'negative' subfolders
    TARGET_FOLDER = "dataset/sample-common-meme-detector"

    # Set batch_size depending on your GPU memory (2, 4, or 8)
    BATCH_SIZE = 3

    if os.path.exists(TARGET_FOLDER):
        process_image_folder(analyzer, TARGET_FOLDER, batch_size=BATCH_SIZE)
    else:
        print(
            f"Please create the folder '{TARGET_FOLDER}' and add 'positive'/'negative' subfolders inside it."
        )
