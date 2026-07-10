import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVQAModel:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens=128):
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading {model_name}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print("Model loaded.")

       
    @torch.inference_mode()
    def generate_response(self, image_path_or_url, prompt_text):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path_or_url
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
  

def load_dataset(json_path, base_folder):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    for item in dataset:
        item["image_path"] = os.path.join(base_folder, item["related_images"])
    return dataset


