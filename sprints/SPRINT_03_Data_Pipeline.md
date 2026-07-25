# SPRINT 03 — Data Pipeline

## Phase Goal
Build the data parsing and preprocessing pipeline to convert raw MIDV-500 samples into the instruction format required for fine-tuning Qwen2-VL-2B. We will also implement scripts to split the dataset, visualize samples, and report statistics. Given your Drive space limits, we will implement a `max_samples` configuration to work with a curated subset.

## Scope
**In Scope:** 
- Parser to convert MIDV-500 annotations to Qwen2-VL format.
- Configuration file `/ml/configs/data_pipeline.yaml` exposing `min_pixels` and `max_pixels` to prevent Colab GPU OOM.
- Train/Val/Test split script.
- Visualization script for sample previews.
- Dataset statistics script.

**Out of Scope:** 
- Model training (Sprint 05).
- Model evaluation (Sprint 04).

## File/Folder Structure Created/Modified
- `/ml/configs/data_pipeline.yaml` (Created)
- `/ml/data_pipeline/parser.py` (Created)
- `/ml/data_pipeline/splitter.py` (Created)
- `/ml/data_pipeline/visualizer.py` (Created)
- `/ml/data_pipeline/stats.py` (Created)
- `/sprints/SPRINT_03_Data_Pipeline.md` (Created)

## Task Checklist
- [x] Create `data_pipeline.yaml` with explicit `min_pixels`/`max_pixels` and a subset size limit (`max_samples`).
- [x] Write `parser.py` to convert annotations to Qwen2-VL instruction format.
- [x] Write `splitter.py` to create deterministic Train/Val/Test splits.
- [x] Write `visualizer.py` to overlay labels on N random samples for human verification.
- [x] Write `stats.py` to report on class balance and field lengths.

## Dependencies
- Dataset available at the configured path.
- Sprint 01 and Sprint 02 complete.

## Manual Steps Required From Me
1. **Curated Subset Selection**: Since you ran out of space when copying to Drive, we've set `max_samples: 1000` in the config. This script will only parse up to 1000 images from `/content/drive/MyDrive/idvlm/data/midv500_data`.
2. **Run the Data Pipeline**: In your Colab notebook, run the following code blocks (you may need to install the dependencies from `requirements.txt` first):
   ```bash
   # Install dependencies if not already done
   !pip install -r "/content/drive/MyDrive/path/to/repo/requirements.txt"
   ```
   *Note: adjust the path above to wherever you cloned this repo in your Drive, or simply run the python scripts.*
   
   Execute the parsing and splitting:
   ```bash
   !python ml/data_pipeline/parser.py
   !python ml/data_pipeline/splitter.py
   ```
3. **Generate Stats & Visualizations**:
   ```bash
   !python ml/data_pipeline/stats.py
   !python ml/data_pipeline/visualizer.py
   ```
4. **Visual Verification**: Open the images generated in `/ml/data_pipeline/samples_preview/` (or the corresponding Drive path) and visually confirm that the labels correspond correctly to the images.

## Definition of Done
- JSONL files for `train.jsonl`, `val.jsonl`, `test.jsonl` are generated.
- Statistics report is generated.
- Sample previews are generated and visually verified by you.

---

## Continue?
**Summary:** The Python scripts for parsing, splitting, and visualizing the MIDV-500 dataset into a curated subset for Qwen2-VL have been written. The configuration carefully manages image resolution limits (`min_pixels`, `max_pixels`) to keep the Colab T4 from OOMing during training.

**Questions before proceeding:**
1. Were you able to run the data pipeline scripts in Colab?
2. Did you inspect the sample previews, and do the extracted JSON labels look correct for the images?
3. What is the total number of samples that ended up in your `train.jsonl`?
