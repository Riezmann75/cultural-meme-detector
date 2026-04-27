from datetime import datetime
import json
import os
import re

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
from PIL import Image
from qwen_vl_utils import process_vision_info


class MemeQADataset(Dataset):
    def __init__(
        self,
        model,
        image_root,
        system_prompt: str,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.image_root = image_root

    def __len__(self):
        return len(os.listdir(self.image_root))

    def __getitem__(self, idx):
        image_name = os.listdir(self.image_root)[idx]
        image_path = f"{self.image_root}/{image_name}"
        # config = {
        #     "tokenize": True,
        #     "add_generation_prompt": True,
        #     "return_dict": True,
        #     "return_tensors": "pt",
        # }
        input = self.model.process_input(
            image_path=image_path,
            question=self.system_prompt,
        )

        return {
            "input_ids": input["input_ids"],
            "attention_mask": input["attention_mask"],
            "pixel_values": input["pixel_values"],
            "image_grid_thw": input["image_grid_thw"],
            "original_idx": idx,
        }


def collate_fn(batch, tokenizer=None):

    pixel_values = tuple([item["pixel_values"] for item in batch])
    image_grid_thw = tuple([item["image_grid_thw"] for item in batch])

    max_length = max(input["input_ids"].shape[1] for input in batch)
    batch_size = len(batch)

    pad_token_id = tokenizer.eos_token_id if tokenizer is not None else 0

    attn_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    input_ids = torch.full(
        (batch_size, max_length), fill_value=pad_token_id, dtype=torch.long
    )

    original_indices = [item["original_idx"] for item in batch]

    for i, input in enumerate(batch):
        length = input["input_ids"].shape[1]
        input_ids[i, -length:] = input["input_ids"]
        attn_mask[i, -length:] = 1

    # note: not sure about the correct way to batch the pixel values
    # at the moment this seems to work with Qwen3-VL-8B-Instruct
    # but not sure why this works
    # the forward method of Qwen3-VL-8B says it expects pixel values of shape (batch_size, num_channels, height, width)
    # but the pixel values after processing an input is of shape (height x width, num_channels)
    # the torch.cat is done on dimension 0
    # TODO: check the pixel values concatenations
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "original_indices": original_indices,
        "pixel_values": torch.cat(
            pixel_values, dim=0
        ),  # need to cat the pixel values, but not sure if this is the correct way to batch the pixel values
        "image_grid_thw": torch.cat(
            image_grid_thw, dim=0
        ),  # need to cat the image_grid_thw
    }

    return inputs


def create_dataloader(
    model,
    image_root,
    system_prompt: str,
    batch_size=1,
    num_workers=1,
):
    dataset = MemeQADataset(
        model=model,
        image_root=image_root,
        system_prompt=system_prompt,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer=model.tokenizer,
        ),
    )

    return dataloader, dataset


class BaseModel:
    def __init__(self):
        """
        Base model class for vision-language tasks.
        """
        self.generator = None
        self.processor = None

    def process_input(
        self,
        image_path,
        init_prompt,
        question,
        **configs,
    ):
        pass

    def generate(self, **kwargs):
        pass

    def parse_batched_input(self, batch_inputs):
        # override this function if the model requires special parsing of batched inputs before feeding into the model
        pass

    # def _generate_config(self):
    #     pass


class Qwen35(BaseModel):
    def __init__(self, model_str="Qwen/Qwen3.5-9B"):
        super().__init__()

        from transformers import Qwen3_5ForConditionalGeneration

        self.generator = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_str, dtype="auto", device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_str,
            use_fast=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_str,
            use_fast=True,
        )

    def process_input(
        self,
        image_path,
        question,
    ):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_path),
                    },
                    {
                        "type": "text",
                        "text": question,
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

        return inputs

    def parse_batched_input(self, batch_inputs):
        # override this function if the model requires special parsing of batched inputs before feeding into the model
        # remove key: original_indices
        return {k: v for k, v in batch_inputs.items() if k != "original_indices"}


def get_output_dir(base_results_path="results/finetuned-siglip"):
    """
    Creates and returns the directory path: base/YYYYMMDD/config_[X]
    """
    # 1. Create date folder
    date_str = datetime.now().strftime("%Y%m%d")
    date_path = os.path.join(base_results_path, date_str)
    os.makedirs(date_path, exist_ok=True)

    # 2. Determine config_X folder name
    existing_configs = [d for d in os.listdir(date_path) if d.startswith("config_")]

    # Extract numbers and find the next one
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


if __name__ == "__main__":
    model = Qwen35(model_str="Qwen/Qwen3.5-27B")

    processor = model.processor
    generator = model.generator

    max_new_tokens = 512
    # local_lang = "Indonesian"
    batch_size = 8
    temperature = 0.5
    top_p = 0.9
    top_k = 50
    repetition_penalty = 1.0

    device = "auto"

    dataloader, dataset = create_dataloader(
        model=model,
        image_root="dataset/sample-cultural-meme-detector/all",
        system_prompt=(
            "You are an expert archivist of internet culture. Your task is to find the cues explaining why this meme should be categorized as either a cultural meme or a common meme.\n"
            "A cultural meme is a meme that may require specific cultural knowledge, context, or references to understand and appreciate. When translated, the cultural context may be lost or the meme may not be understood in the same way.\n"
            "A common meme is a meme that is widely recognized and understood across different cultures and contexts. It relies on universal humor, visual elements, or themes that can be appreciated by a broad audience regardless of cultural background.\n"
            "Strict rules:\n"
            "1. Do not make decision. Just give your reasoning.\n"
            "2. Don't make conclusion in your reasoning, just provide the evidence and cues that support your classification.\n"
            "3. Focus on the visual elements, text, pun, and any cultural markers that can be identified.\n"
            "4. Ignore the slang for emotional expression.\n"
            "5. Using local language does not necessarily make a meme a cultural meme, it depends on the context and the visual elements of the meme.\n"
            "Enclose your answer between reasoning trace and answer using the tags as follows:\n"
            "<reason>Give your reasoning here. You should extract the text, visual cues that support your classification.</reason>"
            "For example: <reason>The meme seems to use a pun that requires knowledge about the language and cultural context to understand.</reason>"
        ),
        batch_size=8,
        num_workers=4,
    )

    output_dir = get_output_dir(base_results_path="results/qwen3.5-meme-inference")
    output_path = os.path.join(output_dir, "generated_responses.jsonl")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reasoning_tags: tuple = ("<reason>", "</reason>")
    new_key = "reason"

    for batch in tqdm(dataloader, desc="Processing image batches"):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        responses_list = []
        batch_indices = batch["original_indices"]
        # Process batch
        # inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        inputs = model.parse_batched_input(batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generator.eval()
        # Generate responses from VLM
        generated_ids = generator.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # print(generated_ids)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        generated_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )  # decode from token ids to text

        # Extract answers from generated texts
        search_pattern = rf"{reasoning_tags[0]}(.*?){reasoning_tags[1]}"

        for generated_text in generated_texts:
            if reasoning_tags is not None:
                match = re.search(search_pattern, generated_text, re.DOTALL)
                answer = match.group(1).strip() if match else generated_text
                # start_tag, end_tag = answer_tags
                # start_idx = generated_text.find(start_tag) + len(start_tag)
                # end_idx = generated_text.find(end_tag)
                # answer = generated_text[start_idx:end_idx].strip() if start_idx != -1 and end_idx != -1 else generated_text
            else:
                answer = generated_text
            responses_list.append(answer)

        for idx, response, raw_text in zip(
            batch_indices, responses_list, generated_texts
        ):
            data = {}
            data["image_path"] = (
                dataset.image_root + "/" + os.listdir(dataset.image_root)[idx]
            )
            data[new_key] = response
            data["raw_generated_text"] = raw_text

            with open(output_path, "a", encoding="utf-8") as outfile:  # 'a' for append
                outstring = json.dumps(data, ensure_ascii=False) + "\n"
                outfile.write(outstring)
