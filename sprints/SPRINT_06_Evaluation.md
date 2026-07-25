# SPRINT 06 — Evaluation & Benchmarking

## Phase Goal
Evaluate the LoRA fine-tuned model on the hold-out test set and compare its performance against the baseline. Our goal is to see parsing failures drop to near 0% and exact matching to significantly increase.

## Scope
**In Scope:** 
- Inference script for the fine-tuned model (`inference_lora.py`).
- Updated `scorer.py` to compare Baseline vs LoRA.

**Out of Scope:** 
- Frontend/UI (reserved for Sprint 07).

## File/Folder Structure
- `/ml/evaluation/inference_lora.py` (Created)
- `/ml/evaluation/scorer.py` (Updated)
- `/sprints/SPRINT_06_Evaluation.md` (Created)

## Task Checklist
- [x] Write `inference_lora.py` that merges the base model with our newly trained PEFT adapter.
- [x] Update `scorer.py` to output a comparative markdown table side-by-side.
- [ ] You (the user) run `inference_lora.py`.
- [ ] You (the user) run `scorer.py` and report the final leap in accuracy.

## Manual Steps Required From Me
1. Pull the latest code and run the LoRA inference script on the test set:
   ```bash
   %cd /content/repo
   !git pull
   !python ml/evaluation/inference_lora.py
   ```
2. Run the updated scoring script to generate the final comparison report:
   ```bash
   !python ml/evaluation/scorer.py
   ```
3. Copy-paste the resulting Markdown table! Let's see if we hit our target.

## Definition of Done
- `lora_predictions.json` generated successfully.
- Final report shows significant improvement over the 0% exact match accuracy of the base model.
