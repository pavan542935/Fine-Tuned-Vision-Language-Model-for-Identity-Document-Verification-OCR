# Fine-Tuned Vision-Language Model for Identity Document Verification & OCR

A production-grade machine learning project focused on identity document verification, OCR field extraction, and forgery detection. It utilizes a fine-tuned **Qwen2-VL-2B** via LoRA/PEFT, bundled with a FastAPI backend and a React frontend demo.

## Project Structure
- `/data/` - Git-ignored dataset directory.
- `/ml/data_pipeline/` - Data loading and preprocessing.
- `/ml/training/` - Model LoRA fine-tuning scripts.
- `/ml/evaluation/` - Performance and extraction accuracy metrics.
- `/ml/robustness/` - Data augmentation harness for evaluating model robustness.
- `/ml/tampering/` - Forgery/tampering simulation and classification.
- `/ml/configs/` - Centralized YAML configuration files.
- `/backend/` - FastAPI service for serving the fine-tuned model.
- `/frontend/` - React application for uploading and verifying ID documents.
- `/sprints/` - Sprint tracking markdown files to organize the development process.
- `/docs/` - Project documentation.

## Setup
(Setup instructions will be added in Sprint 08)
