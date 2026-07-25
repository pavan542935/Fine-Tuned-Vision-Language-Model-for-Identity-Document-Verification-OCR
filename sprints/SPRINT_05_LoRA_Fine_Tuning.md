# SPRINT 05 — LoRA Fine-Tuning

## Phase Goal
Fine-tune the Qwen2-VL-2B model using QLoRA (4-bit quantization + LoRA adapters) on our parsed dataset. The goal is to teach the model to output perfect JSON extraction for identity documents.

## Scope
**In Scope:** 
- Training script utilizing Hugging Face `Trainer` and `peft`.
- 4-bit Quantization to fit training safely into a Colab T4's 16GB VRAM.
- Custom dataset and collator to correctly mask the prompt so loss is only calculated on the assistant's JSON output.

**Out of Scope:** 
- Full parameter fine-tuning (impossible on T4).
- Inference (reserved for Sprint 06).

## File/Folder Structure
- `/ml/configs/train_config.yaml` (Created)
- `/ml/training/train.py` (Created)
- `/sprints/SPRINT_05_LoRA_Fine_Tuning.md` (Created)

## Task Checklist
- [x] Write `train_config.yaml` with optimal LoRA hyperparameters (rank 16).
- [x] Write `train.py` including dataset processing, token masking, and 4-bit QLoRA setup.
- [ ] You (the user) run `train.py` on the Colab GPU.

## Dependencies
- Sprint 03 complete (Train/val splits exist).

## Manual Steps Required From Me
1. The training will take place directly on your Google Drive (`/content/drive/MyDrive/idvlm/models/qwen2-vl-lora`), which means the weights will safely persist even if Colab disconnects!
2. Run the training script:
   ```bash
   !python ml/training/train.py
   ```
3. Training on 800 images for 1 epoch on a T4 will take some time (usually around 30 to 60 minutes). Sit back and watch the loss drop!
4. Let me know when it says "Saving model to...".

## Definition of Done
- `adapter_model.safetensors` successfully saved to your Google Drive.
- Training loss decreases successfully over the epoch.

---

## Continue?
**Summary:** The training script is written! It uses bitsandbytes 4-bit quantization to ensure we don't OOM during the backward pass, and it sets up a LoRA adapter with Rank 16. It also uses the bulletproof ImageMagick loader we perfected in Sprint 04.

**Questions before proceeding:**
1. Are you ready to kick off the training run in Colab? Let me know once the script successfully finishes!
