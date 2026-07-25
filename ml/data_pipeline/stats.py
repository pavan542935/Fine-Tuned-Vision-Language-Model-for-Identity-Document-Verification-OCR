import json
import yaml
from pathlib import Path
from collections import Counter
import statistics

def load_config(config_path="ml/configs/data_pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_stats(config):
    out_path = Path(config['dataset']['output_dir'])
    
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        split_file = out_path / split
        if not split_file.exists():
            print(f"Split {split} not found. Skipping.")
            continue
            
        print(f"\n--- Statistics for {split} ---")
        
        with open(split_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
            
        print(f"Total samples: {len(records)}")
        
        field_presence = Counter()
        field_lengths = {f: [] for f in config['fields_to_extract']}
        
        for r in records:
            # assistant output is the JSON string
            assistant_content = r['messages'][1]['content'][0]['text']
            try:
                data = json.loads(assistant_content)
                for field in config['fields_to_extract']:
                    val = data.get(field)
                    if val is not None and str(val).strip() != "None" and str(val).strip() != "":
                        field_presence[field] += 1
                        field_lengths[field].append(len(str(val)))
            except Exception as e:
                pass
                
        print("\nField Presence (non-null):")
        for field in config['fields_to_extract']:
            count = field_presence[field]
            print(f"  {field}: {count} / {len(records)} ({count/len(records)*100:.1f}%)")
            
        print("\nField Lengths (characters):")
        for field in config['fields_to_extract']:
            lengths = field_lengths[field]
            if lengths:
                print(f"  {field}: min={min(lengths)}, max={max(lengths)}, avg={statistics.mean(lengths):.1f}")
            else:
                print(f"  {field}: N/A (no data)")

if __name__ == "__main__":
    config = load_config()
    generate_stats(config)
