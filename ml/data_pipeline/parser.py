import os
import json
import yaml
import glob
from pathlib import Path
from tqdm import tqdm
import hashlib

def load_config(config_path="ml/configs/data_pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_mock_data(img_path):
    # Generates deterministic mock OCR data based on the image filename
    # since MIDV-500 JSONs only provide document 'quad' bounding boxes, 
    # not the actual text transcriptions for Name, DOB, etc.
    seed_str = img_path.stem
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    
    first_names = ["John", "Jane", "Alice", "Bob", "Carlos", "Diana"]
    last_names = ["Doe", "Smith", "Johnson", "Garcia", "Kim"]
    
    name = f"{first_names[h % len(first_names)]} {last_names[(h // 10) % len(last_names)]}"
    dob = f"{(h % 28) + 1:02d}/{(h % 12) + 1:02d}/{1960 + (h % 40)}"
    id_num = f"ID-{h % 100000000:08d}"
    addr = f"{h % 9999} Fake St, City {h % 100}"
    
    return {
        "Name": name,
        "DOB": dob,
        "ID Number": id_num,
        "Address": addr
    }

def parse_midv500(config):
    data_path = Path(config['dataset']['path'])
    out_path = Path(config['dataset']['output_dir'])
    out_path.mkdir(parents=True, exist_ok=True)
    
    max_samples = config['dataset'].get('max_samples', 1000)
    prompt = config['instruction_prompt']
    fields_to_extract = config['fields_to_extract']
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.png']:
        image_files.extend(data_path.rglob(ext))
        
    image_files = sorted(image_files)
    dataset_records = []
    
    print(f"Found {len(image_files)} images. Parsing up to {max_samples}...")
    
    for img_path in tqdm(image_files[:max_samples]):
        # Since MIDV-500 doesn't have OCR text labels, we use mock deterministic data
        # so we can still train the model to output the correct JSON schema.
        extracted_data = generate_mock_data(img_path)
                
        # Format for Qwen2-VL
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path.absolute())},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(extracted_data)}
                    ]
                }
            ]
        }
        dataset_records.append(record)
        
    out_file = out_path / "parsed_dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in dataset_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Successfully parsed {len(dataset_records)} records to {out_file}")

if __name__ == "__main__":
    config = load_config()
    parse_midv500(config)
