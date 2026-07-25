import os
import json
import yaml
import glob
from pathlib import Path
from tqdm import tqdm

def load_config(config_path="ml/configs/data_pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_midv500(config):
    data_path = Path(config['dataset']['path'])
    out_path = Path(config['dataset']['output_dir'])
    out_path.mkdir(parents=True, exist_ok=True)
    
    max_samples = config['dataset'].get('max_samples', 100000)
    prompt = config['instruction_prompt']
    fields_to_extract = config['fields_to_extract']
    
    # MIDV-500 typically has images and ground truth JSON files.
    # The structure varies, but generally we look for .tif/.jpg files and their corresponding .json
    
    # Heuristic search for images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.png']:
        image_files.extend(data_path.rglob(ext))
        
    image_files = sorted(image_files)
    
    dataset_records = []
    
    print(f"Found {len(image_files)} images. Parsing up to {max_samples}...")
    
    for img_path in tqdm(image_files[:max_samples]):
        # Try to find corresponding JSON
        # MIDV-500 ground truth JSON usually has the same name as the image but in a ground_truth folder
        # For this parser, we assume a simple structure or fallback to null if no gt found.
        # Note: You might need to adjust the gt_path logic based on the exact structure of your MIDV-500 dump.
        gt_path = img_path.with_suffix('.json')
        
        # Fallback if structure is e.g. "images/img.tif" and "ground_truth/img.json"
        if not gt_path.exists():
            gt_path = img_path.parent.parent / "ground_truth" / (img_path.stem + ".json")
            
        extracted_data = {field: None for field in fields_to_extract}
        
        if gt_path.exists():
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                    
                # Map MIDV fields to our target fields (simplistic mapping, update based on actual MIDV JSON schema)
                # MIDV-500 usually contains 'field' dict with 'value'
                if isinstance(gt_data, dict):
                    # Try to extract fields heuristically
                    for k, v in gt_data.items():
                        k_lower = k.lower()
                        val_str = str(v.get('value', v)) if isinstance(v, dict) else str(v)
                        
                        if 'name' in k_lower or 'surname' in k_lower:
                            extracted_data['Name'] = val_str
                        elif 'date' in k_lower and ('birth' in k_lower or 'dob' in k_lower):
                            extracted_data['DOB'] = val_str
                        elif 'number' in k_lower or 'id' in k_lower:
                            extracted_data['ID Number'] = val_str
                        elif 'address' in k_lower:
                            extracted_data['Address'] = val_str
            except Exception as e:
                print(f"Error parsing {gt_path}: {e}")
                
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
