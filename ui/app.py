import gradio as gr
import torch
import json
import yaml
from pathlib import Path
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import PIL.Image

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
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

model, processor = None, None

def extract_info(image):
    global model, processor
    if model is None:
        model, processor = load_model()
        
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


custom_css = """
body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    padding: 2rem;
}
h1 {
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 3rem !important;
    margin-bottom: 0.5rem;
}
.gradio-button.primary {
    background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
    border: none;
    border-radius: 12px;
    transition: all 0.3s ease;
    font-weight: 600;
}
.gradio-button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🔍 Identity Document Verification OCR")
    gr.Markdown("<h3 style='text-align: center; color: #94a3b8; font-weight: 400;'>Powered by Fine-Tuned Qwen2-VL</h3>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Identity Document")
            submit_btn = gr.Button("Extract Data", variant="primary")
            
        with gr.Column(scale=1):
            output_json = gr.JSON(label="Extracted Information")
            
    submit_btn.click(fn=extract_info, inputs=input_image, outputs=output_json)

if __name__ == "__main__":
    demo.launch(share=True)
