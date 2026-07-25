import os
import json
import yaml
import random
from pathlib import Path

def load_config(config_path="ml/configs/data_pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def split_dataset(config):
    out_path = Path(config['dataset']['output_dir'])
    parsed_file = out_path / "parsed_dataset.jsonl"
    
    if not parsed_file.exists():
        raise FileNotFoundError(f"{parsed_file} does not exist. Run parser.py first.")
        
    with open(parsed_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    # Fixed seed for reproducibility
    random.seed(config['dataset']['split']['seed'])
    random.shuffle(records)
    
    total = len(records)
    train_ratio = config['dataset']['split']['train_ratio']
    val_ratio = config['dataset']['split']['val_ratio']
    
    train_idx = int(total * train_ratio)
    val_idx = train_idx + int(total * val_ratio)
    
    train_records = records[:train_idx]
    val_records = records[train_idx:val_idx]
    test_records = records[val_idx:]
    
    splits = {
        "train.jsonl": train_records,
        "val.jsonl": val_records,
        "test.jsonl": test_records
    }
    
    for split_name, split_data in splits.items():
        split_path = out_path / split_name
        with open(split_path, "w", encoding="utf-8") as f:
            for r in split_data:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(split_data)} records to {split_path}")

if __name__ == "__main__":
    config = load_config()
    split_dataset(config)
