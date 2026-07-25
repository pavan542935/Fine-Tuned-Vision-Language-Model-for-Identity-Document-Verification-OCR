# SPRINT 02 — HELPER PHASE: Dataset Acquisition & Colab GPU Connection

## Phase Goal
Set up the Colab GPU environment and acquire the dataset manually, since this is a heavy-duty task requiring a connected Colab kernel and a specific dataset (MIDV-500/MIDV-2020) that cannot be safely managed automatically without potentially blowing up memory limits or losing state.

## Scope
**In Scope:** 
- Manual dataset download and Google Drive staging.
- Manual connection to a Google Colab T4 GPU runtime.
- Verification of GPU availability.

**Out of Scope:** 
- Any automated code execution by the agent (this phase is entirely manual).

## File/Folder Structure
- Creates `docs/DATA_LICENSE.md` (to document the MIDV usage terms).
- Modifies `/sprints/SPRINT_02_Dataset_Colab.md`.

## Task Checklist
- [x] Read the dataset usage terms and log them to `/docs/DATA_LICENSE.md`.
- [ ] You (the user) download the MIDV-500 or MIDV-2020 dataset and stage it in Google Drive.
- [ ] You spin up a Colab GPU runtime and connect Antigravity to it.
- [ ] You verify the dataset path and GPU availability in Colab.

## Dependencies
- Sprint 01 scaffolding complete.

## Manual Steps Required From Me
Please follow this checklist exactly:

1. Download **MIDV-500** (or MIDV-2020 if you prefer forged/tampered samples to be included) from the official dataset source. Note the dataset's license/usage terms (research use, attribution requirements) in `/docs/DATA_LICENSE.md` — these are synthetic ID documents, not real PII, but attribution is still required. I have created a template `DATA_LICENSE.md` for you to fill in.
2. Upload the extracted dataset into a Google Drive folder, e.g. `MyDrive/idvlm/data/`.
3. Open Google Colab, start a GPU runtime (Runtime → Change runtime type → GPU, T4 or better).
4. In Antigravity: open the Colab extension panel, sign in with the same Google account, and connect Antigravity to that running Colab GPU kernel (Colab Plugin → Connect to Runtime).
5. In the connected notebook, run:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !nvidia-smi
   ```
   to confirm GPU is attached and Drive is mounted.
6. Note the resolved dataset path (e.g. `/content/drive/MyDrive/idvlm/data/MIDV-500/`) and paste it back to the agent so it can hardcode/parametrize it in the data pipeline config.
7. **Important limitation to remember:** For every GPU-bound cell I (the agent) write from here on, you will manually click run in the notebook. I will never assume my own execution touched the GPU and will always ask you to confirm "ran on GPU, confirmed via nvidia-smi output" before treating results as valid.
8. **Memory budget check:** A free-tier Colab T4 has ~16GB VRAM. A 2B-parameter VLM plus LoRA training will need 4-bit quantization. We will use `load_in_4bit: true` for upcoming training.

## Definition of Done
- Dataset path confirmed working.
- GPU confirmed attached.
- Both pieces of info pasted back into chat.

---

## Continue?
**Summary:** This phase requires you to download the dataset, upload it to your Google Drive, mount your Drive in a Google Colab GPU notebook, and connect me to it. I have also prepared the `DATA_LICENSE.md` file for you.

**Questions before proceeding:**
1. Were you able to mount the dataset and run `!nvidia-smi`?
2. Did `!nvidia-smi` show a GPU (e.g., T4)?
3. What is the full resolved dataset path in your mounted Google Drive (e.g., `/content/drive/MyDrive/...`)?
