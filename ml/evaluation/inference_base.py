import os
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_config(config_path="ml/configs/data_pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_baseline_inference():
    config = load_config()
    out_dir = Path(config['dataset']['output_dir'])
    test_file = out_dir / "test.jsonl"
    
    if not test_file.exists():
        raise FileNotFoundError(f"{test_file} not found. Run Sprint 03 first.")
        
    print("Loading model and processor...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Load in bfloat16 to fit nicely in 16GB T4 VRAM without quantization for baseline
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    with open(test_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    predictions = []
    
    min_pixels = config['dataset'].get('min_pixels', 3136)
    max_pixels = config['dataset'].get('max_pixels', 602112)
    
    print(f"Running inference on {len(records)} test samples...")
    for i, record in enumerate(tqdm(records)):
        messages = record['messages'][0:1] # only use the user message prompt
        
        # Inject min/max pixels to prevent GPU hanging on huge images
        for msg in messages:
            for content in msg['content']:
                if content['type'] == 'image':
                    content['min_pixels'] = min_pixels
                    content['max_pixels'] = max_pixels
        
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        predictions.append({
            "index": i,
            "raw_output": output_text,
            "ground_truth": json.loads(record['messages'][1]['content'][0]['text'])
        })
        
    out_path = Path("ml/evaluation/baseline_predictions.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
        
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    run_baseline_inference()
