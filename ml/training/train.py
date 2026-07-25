import os
import json
import yaml
import torch
import tempfile
import PIL.Image
import subprocess
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

def load_config(config_path="ml/configs/train_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class Qwen2VLDataset(Dataset):
    def __init__(self, data_path, processor, min_pixels=3136, max_pixels=250880):
        self.processor = processor
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.data)

    def _convert_tif(self, tif_path):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_path = temp_file.name
        temp_file.close()
        
        if not os.path.exists(tif_path):
            img = PIL.Image.new('RGB', (800, 600), color='black')
            img.save(temp_path)
            return temp_path

        try:
            subprocess.run(['convert', str(tif_path), temp_path], check=True, capture_output=True)
            return temp_path
        except Exception:
            try:
                img = PIL.Image.open(str(tif_path))
                img.convert("RGB").save(temp_path)
                return temp_path
            except Exception:
                img = PIL.Image.new('RGB', (800, 600), color='white')
                img.save(temp_path)
                return temp_path

    def __getitem__(self, i):
        record = self.data[i]
        messages = record['messages'] # [user_msg, assistant_msg]
        
        temp_files = []
        for msg in messages:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        img_path = content['image']
                        if str(img_path).lower().endswith(('.tif', '.tiff')):
                            new_path = self._convert_tif(img_path)
                            if new_path:
                                content['image'] = new_path
                                temp_files.append(new_path)
                        content['min_pixels'] = self.min_pixels
                        content['max_pixels'] = self.max_pixels

        # 1. Full sequence
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 2. Prompt only (to find length for masking)
        prompt_text = self.processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
        
        prompt_inputs = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        )
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        )
        
        for tf in temp_files:
            if os.path.exists(tf): os.remove(tf)
            
        labels = inputs['input_ids'].clone()
        # Mask out the prompt so we only compute loss on the JSON response
        labels[0, :prompt_len] = -100
        
        inputs['labels'] = labels
        return {k: v.squeeze(0) for k, v in inputs.items()}

def custom_collator(features, processor):
    batch = {}
    input_ids = [f['input_ids'] for f in features]
    labels = [f['labels'] for f in features]
    
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    batch['input_ids'] = input_ids_padded
    batch['labels'] = labels_padded
    batch['attention_mask'] = (input_ids_padded != pad_id).long()
    
    if 'pixel_values' in features[0]:
        batch['pixel_values'] = torch.cat([f['pixel_values'] for f in features], dim=0)
    if 'image_grid_thw' in features[0]:
        batch['image_grid_thw'] = torch.cat([f['image_grid_thw'] for f in features], dim=0)
        
    return batch

def train():
    config = load_config()
    train_cfg = config['training']
    lora_cfg = config['lora']
    
    with open("ml/configs/data_pipeline.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
        
    out_dir = Path(data_cfg['dataset']['output_dir'])
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    
    print("Loading model and processor...")
    model_id = train_cfg['model_id']
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 4-bit Quantization is required to fit training on a 16GB T4 GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias'],
        task_type=lora_cfg['task_type']
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("Preparing datasets...")
    train_dataset = Qwen2VLDataset(train_file, processor)
    val_dataset = Qwen2VLDataset(val_file, processor)
    
    training_args = TrainingArguments(
        output_dir=train_cfg['output_dir'],
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=train_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        learning_rate=float(train_cfg['learning_rate']),
        weight_decay=train_cfg['weight_decay'],
        warmup_ratio=train_cfg['warmup_ratio'],
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        save_strategy=train_cfg['save_strategy'],
        eval_strategy=train_cfg['eval_strategy'],
        logging_steps=train_cfg['logging_steps'],
        bf16=train_cfg['bf16'],
        remove_unused_columns=False, # CRITICAL: keep pixel_values
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda features: custom_collator(features, processor)
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {train_cfg['output_dir']}")
    trainer.save_model(train_cfg['output_dir'])
    processor.save_pretrained(train_cfg['output_dir'])

if __name__ == "__main__":
    train()
