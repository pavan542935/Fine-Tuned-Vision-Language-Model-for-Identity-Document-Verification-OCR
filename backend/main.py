import os
import json
import yaml
import torch
import tempfile
import PIL.Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import io

app = FastAPI(title="IDVLM API")

# Add CORS middleware to allow the React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None

def load_model():
    print("Loading base model...")
    with open("ml/configs/train_config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)['training']
        
    model_id = train_cfg['model_id']
    lora_dir = train_cfg['output_dir']
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
    )
    
    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, lora_dir)
    peft_model = peft_model.merge_and_unload()
    proc = AutoProcessor.from_pretrained(model_id)
    return peft_model, proc

@app.on_event("startup")
async def startup_event():
    global model, processor
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")

@app.post("/api/extract")
async def extract_data(file: UploadFile = File(...)):
    global model, processor
    if model is None:
        try:
            model, processor = load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
            
    try:
        contents = await file.read()
        image = PIL.Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": 3136, "max_pixels": 250880},
                {"type": "text", "text": "Extract the Name, DOB, ID Number, and Address from this identity document. Output strictly in JSON format."}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    try:
        start = output_text.find('{')
        end = output_text.rfind('}')
        if start != -1 and end != -1:
            json_str = output_text[start:end+1]
            return json.loads(json_str)
        return {"error": "Failed to parse JSON", "raw": output_text}
    except:
        return {"error": "Invalid JSON", "raw": output_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
