# Project Conventions: Identity Document Verification & OCR

## Technology Stack
- **Python:** 3.12+
- **Package Manager:** `uv` or `pip`
- **CUDA Environment:** CUDA 12.x
- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT

## Development Guidelines
1. **Type Hints:** Use complete type hints for functions, especially for tensor shapes (e.g., `images: torch.Tensor # shape: [B, C, H, W]`).
2. **Reproducibility:** 
   - All experiments must be logged.
   - Set fixed seeds (`torch.manual_seed(42)`, `numpy.random.seed(42)`, etc.) for reproducibility.
3. **Configuration:** 
   - Never hardcode parameters (learning rate, batch size, image dimensions). 
   - All configurations must reside in YAML files under `/ml/configs/`.
4. **Data & Checkpoints:**
   - NEVER commit datasets or model checkpoints to git.
   - Ensure `/data/` and all output weights (e.g., `.pt`, `.safetensors`) are strictly `.gitignore`d.
5. **GPU Acceleration Context (Google Colab):**
   - We will be training on Google Colab (T4 GPU). 
   - **Important:** `flash-attention-2` is optional. T4 GPUs often struggle or do not support the latest `flash-attn` versions. It is considered a nice-to-have but not required for this project.
