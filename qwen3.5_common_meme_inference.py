import os
import re
import json
import torch
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


class MemeFilterVLM:
    def __init__(self, model_id="Qwen/Qwen3.5-VL-27B-Instruct", system_prompt=None):
        """
        Initializes the Qwen3.5 model dynamically using generic Auto classes.
        Automatically distributes the weights across available GPUs.
        """
        print(f"Loading processor from {model_id}...")
        # trust_remote_code=True allows loading new/custom architectures not natively in transformers yet
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Crucial for batch generation: padding must be on the left
        self.processor.tokenizer.padding_side = "left"

        print(f"Loading {model_id} across GPUs. This may take a few minutes...")
        # Replaced the hardcoded Qwen2_5 class with the generic AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.system_prompt = system_prompt
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
            elif (
                "COMMON" in answer_text
            ):  # Adapted to look for COMMON based on new prompt
                result["status"] = "COMMON"
            else:
                result["status"] = answer_text

        return result

    def infer_meme_batch(self, image_paths):
        """
        Analyzes a batch of images to determine if they are common memes or local/unknown ones.
        """
        valid_paths = []
        messages_batch = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                valid_paths.append(path)
                messages_batch.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": self.system_prompt},
                            ],
                        }
                    ]
                )
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        if not messages_batch:
            return []

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages_batch
        ]

        # Uses qwen_vl_utils to parse the images out of the message dictionary
        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = self.processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        final_results = []
        for path, text in zip(valid_paths, output_texts):
            result = self._extract_tags(text.strip())
            result["image_path"] = path
            final_results.append(result)

        return final_results


def get_output_dir(base_results_path="results/common-meme-detection/qwen-3.5"):
    """
    Creates and returns the directory path: base/YYYYMMDD/config_[X]
    """
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


def process_image_folder(analyzer, base_folder, output_folder, batch_size=4):
    """
    Scans a base folder for 'positive' and 'negative' subdirectories,
    runs inference, and saves results to the specified output_folder.
    """
    subdirs = ["positive", "negative"]
    image_paths = []

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

    output_jsonl = os.path.join(output_folder, "meme_analysis_results.jsonl")

    prompt_file = os.path.join(output_folder, "prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f_prompt:
        f_prompt.write(analyzer.system_prompt)

    print(f"Saving results to: {output_folder}")
    print(
        f"Found {len(image_paths)} images. Starting batched inference (Batch size: {batch_size})..."
    )

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    with open(output_jsonl, "a", encoding="utf-8") as f:
        batches = list(chunker(image_paths, batch_size))
        for batch in tqdm(batches, desc="Analyzing Batches"):
            batch_paths = [item[0] for item in batch]
            results = analyzer.infer_meme_batch(batch_paths)

            for res in results:
                ground_truth = next(
                    (item[1] for item in batch if item[0] == res["image_path"]),
                    "unknown",
                )

                ordered_res = {
                    "image_name": os.path.basename(res["image_path"]),
                    "ground_truth": ground_truth,
                    "predicted_result": res["status"],
                    "reasoning": res.get("reasoning", ""),
                    "image_path": res["image_path"],
                    "raw_output": res.get("raw_output", ""),
                }

                f.write(json.dumps(ordered_res, ensure_ascii=False) + "\n")

    print(f"\nFinished processing! Results saved in {output_folder}")


if __name__ == "__main__":
    system_prompt = """
    You are an expert archivist of internet culture. Your only job is to determine if this image is a 'common image,meme or unknown'. A Common Meme is defined as a meme that it's humor idea is universally shared, a common image is a image that is not memebasic fact, or global internet culture (e.g., relatable daily struggles, standard reaction faces). Strict Rules:
    1. DO NOT GUESS. If you can confidently identify the humor idea and it completely relies on hyper-local slang, regional figures, or requires specific cultural context from Southeast Asia (Vietnam, Indonesia), or there is no joke, classify it as UNKNOWN.
    2. If you are not certain how the visual objects in the image contribute to the humor and the humor idea is not clear, classify as UNKNOWN.
    3. If its' a meme, look at the overall meaning, only classify the meme as COMMON if the basic ideas can be understood based on a universally shared experience or fact.
    4. There are emotional slangs, you can ignore them if the humor is still clear without understanding those words.
    Strict analysis steps:
    1. Translate: Translate the text.
    2. Test universality: If you translate the joke into English and show it to someone in a different country, would they 'get' the basic idea?
    Put your reasoning trace and your final conclusion EXACTLY using the tags as follows:
    <reason>Explain your thought process step-by-step. Identify any text, visual tropes, or cultural markers.</reason>
    <answer> COMMON or UNKNOWN </answer>
    """
    output_dir = get_output_dir()

    # NOTE: Set the model_id to the exact Vision-Language variant you wish to use.
    # We default here to a generalized "Qwen/Qwen3.5-VL-27B-Instruct" to show the pattern.
    analyzer = MemeFilterVLM(
        model_id="Qwen/Qwen3.5-27B", system_prompt=system_prompt
    )

    TARGET_FOLDER = "dataset/sample-common-meme-detector"
    BATCH_SIZE = 4

    if os.path.exists(TARGET_FOLDER):
        process_image_folder(analyzer, TARGET_FOLDER, output_dir, batch_size=BATCH_SIZE)
    else:
        print(
            f"Please create the folder '{TARGET_FOLDER}' with 'positive'/'negative' subfolders."
        )
