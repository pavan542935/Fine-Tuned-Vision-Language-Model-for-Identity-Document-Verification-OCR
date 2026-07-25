# SPRINT 04 — Baseline Evaluation (pre-fine-tune)

## Phase Goal
Evaluate the base `Qwen2-VL-2B-Instruct` model on our test dataset before applying any fine-tuning. This establishes the baseline performance (which is expected to be poor or heavily hallucinated, as the base model isn't specialized for this JSON extraction task on these specific documents).

## Scope
**In Scope:** 
- Inference script to run the base model over the 100 test samples.
- Scoring script using Exact Match and Fuzzy Match.
- Robust parsing to handle the base model failing to output valid JSON.

**Out of Scope:** 
- Any LoRA fine-tuning (reserved for Sprint 05).

## File/Folder Structure
- `/ml/evaluation/inference_base.py` (Created)
- `/ml/evaluation/scorer.py` (Created)
- `/sprints/SPRINT_04_Baseline_Eval.md` (Created)

## Task Checklist
- [x] Write `inference_base.py` to batch-process the test set using Qwen2-VL.
- [x] Write `scorer.py` to extract JSON from messy outputs and calculate exact/fuzzy metrics.
- [ ] You (the user) run `inference_base.py` on the Colab GPU.
- [ ] You (the user) run `scorer.py` and report the baseline numbers.

## Dependencies
- Sprint 03 complete (`test.jsonl` exists).

## Manual Steps Required From Me
1. Note: Inference with Qwen2-VL runs in `bfloat16` to fit in the T4's 16GB VRAM.
2. Run the baseline inference in Colab (this might take ~5-15 minutes for 100 images depending on the GPU):
   ```bash
   !python ml/evaluation/inference_base.py
   ```
3. Run the scorer script:
   ```bash
   !python ml/evaluation/scorer.py
   ```
4. Read the printed markdown report. **Please copy and paste the markdown report table into our chat** so I can record the baseline numbers.

## Definition of Done
- `baseline_predictions.json` generated.
- `baseline_report.md` generated.
- Baseline numbers pasted back to me in the chat.

---

## Continue?
**Summary:** Scripts for the baseline evaluation have been written. `inference_base.py` will load the base model in `bfloat16` to prevent OOM, and `scorer.py` uses robust JSON parsing (finding the outermost braces) to handle the base model's tendency to add conversational filler.

**Questions before proceeding:**
1. Did the inference run successfully on the GPU without OOM (out of memory)?
2. What are the baseline exact match and fuzzy match numbers for the 4 fields?
