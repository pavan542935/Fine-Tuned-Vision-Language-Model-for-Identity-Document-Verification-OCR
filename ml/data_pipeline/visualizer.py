import os
import json
import yaml
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_config(config_path="ml/configs/data_pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_previews(config, num_samples=5):
    out_path = Path(config['dataset']['output_dir'])
    train_file = out_path / "train.jsonl"
    preview_dir = Path("ml/data_pipeline/samples_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    if not train_file.exists():
        print(f"File {train_file} not found. Run splitter.py first.")
        return
        
    with open(train_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    if not records:
        print("No records found.")
        return
        
    samples = random.sample(records, min(num_samples, len(records)))
    
    for i, r in enumerate(samples):
        # Extract path
        img_path = None
        user_content = r['messages'][0]['content']
        for item in user_content:
            if item['type'] == 'image':
                img_path = item['image']
                break
                
        assistant_content = r['messages'][1]['content'][0]['text']
        try:
            data = json.loads(assistant_content)
        except:
            data = {"Error": "Failed to parse JSON"}
            
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Simple text overlay for preview
            text_overlay = "\n".join([f"{k}: {v}" for k, v in data.items()])
            
            # Attempt to draw text (using default font as we don't have custom fonts guaranteed)
            draw.text((10, 10), text_overlay, fill=(255, 0, 0))
            
            save_path = preview_dir / f"preview_{i}.jpg"
            img.save(save_path)
            print(f"Saved preview: {save_path}")
            
            # Print to console as well for quick verification
            print(f"\nSample {i}: {img_path}")
            print(f"Labels: {text_overlay}")
        else:
            print(f"Image not found at {img_path}")

if __name__ == "__main__":
    config = load_config()
    generate_previews(config)
