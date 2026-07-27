# Project Interview Prep: Fine-Tuned Vision-Language Model for Identity Document Verification & OCR

## PHASE 1 — Orientation

### 1. Actual Top-Level Folder Structure
Here is the real folder structure found in the project root:
- `backend/` — Contains the FastAPI server code (`main.py`) which handles inference requests.
- `frontend/` — Contains the React/Vite frontend application where users upload ID images.
- `ml/` — The core Machine Learning directory containing scripts for the entire ML lifecycle:
  - `configs/` — YAML configurations for training (e.g., `train_config.yaml` with LoRA hyperparams).
  - `data_pipeline/` — Scripts for parsing, splitting, and analyzing the dataset (`parser.py`, `splitter.py`, etc.).
  - `evaluation/` — Scripts for calculating OCR accuracy and JSON format adherence.
  - `training/` — Contains `train.py` which executes the LoRA fine-tuning process.
  - `inference/`, `robustness/`, `tampering/` — Scripts for testing model reliability against adversarial or damaged inputs.
- `docs/` — Contains documentation like `RUNBOOK.md` detailing how to start the Colab GPU backend and connect the local UI.
- `data/` — Contains the raw and processed images/JSON used for fine-tuning.

### 2. Actual Tech Stack

**Machine Learning & Backend:**
- **Language:** Python
- **Web Framework:** FastAPI (served via Uvicorn)
- **Base Model:** Qwen2-VL-2B-Instruct (A lightweight, state-of-the-art Vision-Language Model)
- **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation)
- **Libraries:** PyTorch (`torch`), HuggingFace Transformers, `peft`, `qwen_vl_utils`
- **Tunneling:** `localtunnel` (exposing Colab's local port to the internet)

**Frontend:**
- **Core:** React 19, Vite
- **Styling:** Tailwind CSS (implied via modern React ecosystem)
- **Icons:** `lucide-react`
- **API Communication:** Native JavaScript `fetch` API

### 3. Executive Summary
**What problem this solves:** Traditional OCR pipelines (like Tesseract) struggle with unstructured documents, varied layouts, and returning data in a structured schema. They extract raw text, forcing developers to write brittle regex patterns to find names, DOBs, and ID numbers.
**Why it was built:** To provide an end-to-end system that not only reads the text on an Identity Document but also *understands* it semantically, outputting strict, perfectly formatted JSON containing specific fields (Name, DOB, ID Number, Address).
**Who the users are:** FinTech companies, KYC (Know Your Customer) compliance teams, and security systems requiring automated, accurate extraction of user identity data from uploaded photos.
**What value it provides:** By fine-tuning a Vision-Language Model (VLM), the system achieves near-perfect accuracy in extracting specific fields regardless of lighting, ID orientation, or layout variations. It eliminates the need for regex and multi-step OCR pipelines.
**What breaks if it didn't exist:** Companies would have to rely on expensive third-party APIs (like Google Cloud Vision) combined with manual human review teams to extract identity data, leading to slower onboarding and higher operational costs.

### 4. Ambiguities & Clarifications Needed
- **Deployment:** The backend requires a GPU (T4/A100) to run inference fast enough for a good UX. Currently, it is orchestrated using Google Colab for free GPU access, with localtunnel bridging the gap to the frontend. For production, this would need to migrate to a paid cloud GPU provider (e.g., RunPod, AWS EC2, Hugging Face Spaces).
- **Data Privacy:** Because this deals with PII (Personally Identifiable Information), sending ID cards to a free Colab instance over `localtunnel` is strictly for portfolio/development purposes. A production version would require a VPC and stringent encryption.

## PHASE 2 — End-to-End Workflow & Architecture

### System Architecture Diagram
```text
[ React Frontend (Vercel/Local) ]
         |
    (POST /api/extract with Multipart Image Form)
         v
[ Localtunnel Public URL ]
         |
[ FastAPI Backend (Google Colab T4 GPU) ] 
         |
    (Image Preprocessing via qwen_vl_utils)
         |
[ Qwen2-VL Base Model + LoRA Adapter ]
         |
    (Structured JSON Output generation)
         v
[ FastAPI returns JSON Response ]
```

### Stage-by-Stage Request Flow

#### Stage 1: User Request (Frontend)
**Plain English:** The user uploads a photo of their ID card and clicks "Extract". The website bundles the image into a file package and sends it over the internet to our AI server.
**Technical:** The **React frontend** creates a `FormData` object containing the uploaded `File`. It sends an **HTTP POST request** using `fetch` to the `/api/extract` endpoint. A custom header `Bypass-Tunnel-Reminder: true` is included to bypass `localtunnel`'s anti-spam screen.

#### Stage 2: API Routing & Validation (backend/main.py)
**Plain English:** The server receives the image. If the AI model hasn't been loaded into the graphics card (GPU) yet, it takes a few seconds to load it. It also checks if the file is a valid image.
**Technical:** The **FastAPI router** accepts the request via `UploadFile`. The model is loaded lazily via the `load_model()` function, which reads `ml/configs/train_config.yaml` to locate the base model (`Qwen/Qwen2-VL-2B-Instruct`) and the LoRA adapter. The image bytes are parsed using `PIL.Image`.

#### Stage 3: Prompt & Vision Processing
**Plain English:** We tell the AI exactly what we want: "Look at this image and extract Name, DOB, ID Number, and Address in strict JSON format." The image is resized so the AI can process it efficiently.
**Technical:** The system constructs a **multimodal prompt** matching the Qwen2-VL chat template format. It uses `process_vision_info` to generate `image_inputs` and resizes the image to fit within defined pixel constraints (`min_pixels: 3136`, `max_pixels: 250880`) to manage GPU VRAM.

#### Stage 4: LLM Inference
**Plain English:** The AI's brain (running on the GPU) analyzes the pixels and the text simultaneously, and predicts the correct JSON text character by character.
**Technical:** The inputs are passed to the `Qwen2VLForConditionalGeneration` model residing on the `cuda` device. `torch.no_grad()` is used to disable gradient calculation (saving memory since we are only inferencing, not training). The model generates token IDs up to `max_new_tokens=128`.

#### Stage 5: Response Parsing & Display
**Plain English:** The server takes the raw text generated by the AI, finds the `{` and `}` brackets, chops out the JSON, and sends it back to the website to be displayed neatly.
**Technical:** The backend decodes the tokens into a string. A `try/except` block is used to perform **string parsing**, searching for `{` and `}` to extract the JSON substring. `json.loads()` validates it before returning it with an **HTTP 200 OK**. If it fails, it returns an error dictionary.

## PHASE 3 — Tech Stack Deep Dive

### Technologies Used

| Technology | Problem it Solves & Why it Fits | Alternatives & Why Rejected | Advantages | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL-2B** | **Plain English:** We need an AI that can see images and read text. Qwen2 is incredibly smart for its small size.<br>**Technical:** A state-of-the-art **Vision-Language Model (VLM)**. At 2 billion parameters, it's small enough to run on a free T4 GPU but highly capable of OCR reasoning. | *Alternatives:* LLaVA, GPT-4V API.<br>*Rejected because:* GPT-4V costs money and violates data privacy. LLaVA is heavier and often less accurate on dense text than Qwen2-VL. | Runs on cheap hardware (T4), natively understands varied resolutions. | Still requires ~6-8GB of VRAM, making CPU deployment impossible. |
| **LoRA (PEFT)** | **Plain English:** Training a massive AI from scratch costs millions. LoRA lets us teach an existing AI a new trick (JSON formatting) by only tweaking a tiny fraction of its brain.<br>**Technical:** **Low-Rank Adaptation**. Freezes the pre-trained model weights and injects trainable rank decomposition matrices into the Transformer architecture. | *Alternatives:* Full Fine-Tuning, Prompt Engineering.<br>*Rejected because:* Full FT requires massive multi-GPU clusters. Prompt Engineering alone is often too brittle for strict JSON extraction on noisy IDs. | Reduces trainable parameters by 99%, prevents catastrophic forgetting. | Adding the adapter slightly slows down initial load times. |
| **FastAPI** | **Plain English:** A fast bridge between our React website and our Python AI script.<br>**Technical:** A modern, high-performance web framework for building APIs with Python. | *Alternatives:* Flask, Django.<br>*Rejected because:* We need async I/O to handle heavy inference loads without blocking the main server thread. | Async native, incredibly fast, auto-generates Swagger UI. | Relies heavily on Pydantic, which has a learning curve. |
| **Vite + React** | **Plain English:** Tools to build a fast, modern website UI.<br>**Technical:** React is a component-based UI library; Vite is a lightning-fast build tool using native ES modules. | *Alternatives:* Next.js.<br>*Rejected because:* This is an SPA (Single Page Application) that doesn't need SSR (Server-Side Rendering) or complex routing. | Instant hot-reloading during development. | React requires client-side JS execution. |
| **Google Colab** | **Plain English:** A free computer in the cloud with a powerful graphics card (GPU) needed to run the AI.<br>**Technical:** A hosted Jupyter notebook service providing free access to NVIDIA T4 GPUs. | *Alternatives:* RunPod, AWS EC2, Local GPU.<br>*Rejected because:* AWS and RunPod cost money per hour. Most laptops lack a dedicated Nvidia GPU with enough VRAM. | 100% Free, pre-installed with CUDA and PyTorch. | Ephemeral (wipes data on restart), max 12-hour sessions, requires tunneling to expose APIs. |

## PHASE 4 — Data & AI/ML Pipeline

### The ML Fine-Tuning Pipeline (Stage-by-Stage)
**Plain English:** Before the app was built, we had to teach the AI. We gathered hundreds of ID images, wrote the correct JSON answers for them, and trained the AI to match the image to the JSON.
**Technical:** The project implements a complete **Supervised Fine-Tuning (SFT)** pipeline.

1. **Data Generation & Parsing (`ml/data_pipeline`):** Synthetic or real ID documents are parsed into image-text pairs. The target text is structured perfectly into JSON format representing Name, DOB, ID, and Address.
2. **Configuration (`ml/configs/train_config.yaml`):** The training hyperparameters are defined:
   - `num_train_epochs`: 1 (often enough for a model to learn JSON syntax).
   - `learning_rate`: 2.0e-5 with a `cosine` scheduler.
   - `bf16`: True (Mixed precision training to save VRAM and speed up training).
   - `LoRA Target Modules`: `q_proj`, `k_proj`, `v_proj`, etc. (Applying LoRA to the attention mechanism).
3. **Training Execution (`ml/training/train.py`):** The base model (`Qwen2-VL-2B-Instruct`) is loaded. The `peft` library injects the LoRA adapter. The model is trained using Backpropagation, updating only the LoRA weights.
4. **Adapter Saving:** The final output is just the LoRA adapter (a small file containing only the learned differences), saved to Google Drive or a local directory.
5. **Inference (Backend):** During production, the Base Model + LoRA adapter are merged dynamically to process user requests.

### API Endpoint
#### `POST /api/extract`
**Plain English:** The main door to our AI. You send an image, it sends back JSON data.
**Technical:** Accepts `multipart/form-data` containing the file.
- **Validation:** Handles missing files or invalid formats via `try/except` blocks around `PIL.Image.open()`.
- **Response:** Returns `{"Name": "...", "DOB": "...", ...}`. If the model hallucinates non-JSON text, the backend parses out the substring between `{` and `}` to salvage the response.

## PHASE 5 — Security, Deployment, Error Handling

### Deployment Strategy
**Plain English:** The frontend lives permanently online. The backend only spins up when we turn on our Colab notebook. We use a tunnel to connect them.
**Technical:**
- **Frontend:** Hosted statically on platforms like Vercel or Netlify. The API URL is injected via environment variables (`import.meta.env.VITE_API_URL`).
- **Backend:** Runs in a Google Colab cell. `localtunnel` exposes port 8000 to the public internet. Because the `loca.lt` URL changes every session, the Vercel environment variable must be updated for each demo, or the user passes a header (`Bypass-Tunnel-Reminder: true`) to bypass the tunnel warning page.

### Error Handling & Edge Cases
**Plain English:** If the user uploads a corrupted file or the AI spits out garbage, the app catches it and shows a clean error instead of crashing.
**Technical:** 
- **Invalid Images:** Handled by `HTTPException(status_code=400, detail="Invalid image file")`.
- **Model Parsing Failures:** Handled by a strict parsing algorithm. If `json.loads(json_str)` fails, it gracefully falls back to returning `{"error": "Invalid JSON", "raw": output_text}`.

### Performance Optimizations
**Technical:**
- **Lazy Loading:** The model is not loaded when the FastAPI server starts; it is loaded inside the `@app.on_event("startup")` hook or upon the first request to prevent timeout crashes during initialization.
- **bfloat16 (BF16):** Loading the model in `torch.bfloat16` halves the memory footprint compared to FP32, allowing it to fit on a 16GB T4 GPU.
- **Vision Resolution Limits:** Restricting `max_pixels: 250880` ensures the Vision Transformer doesn't consume exponential VRAM for ultra-high-resolution uploads.

## PHASE 6 — Design Decisions & Real Challenges

### Top 5 Design Decisions
1. **LoRA vs. Full Fine-Tuning**
   - **Decision:** Use LoRA.
   - **Why:** Full fine-tuning a 2B parameter model requires vast amounts of VRAM (A100s). LoRA allows us to train on a consumer/free GPU by freezing the base weights.
2. **Qwen2-VL vs. Traditional OCR Pipeline**
   - **Decision:** Use an end-to-end VLM.
   - **Why:** Traditional pipelines (Tesseract -> Regex -> Heuristics) break if the ID layout changes slightly. Qwen2-VL understands semantic context, making it layout-agnostic.
3. **Google Colab + Localtunnel vs. Paid Cloud GPUs**
   - **Decision:** Use Colab + `localtunnel` for backend orchestration.
   - **Why:** Keeps the project 100% free for portfolio and development purposes, avoiding $50-$100/month cloud costs.
4. **JSON Substring Parsing vs. Strict Grammar Decoding**
   - **Decision:** Use `output_text[start:end+1]` to extract JSON from raw output.
   - **Why:** Even fine-tuned models sometimes prepend "Here is the extracted data:". Substring parsing is a robust, lightweight way to extract the JSON without needing complex grammar-constrained decoding frameworks like `Outlines`.
5. **Vite vs. Create React App**
   - **Decision:** Scaffold the frontend with Vite.
   - **Why:** Vastly superior development speed and Hot Module Replacement (HMR) compared to the deprecated CRA.

### Top 5 Technical Challenges Handled
1. **Challenge:** **GPU VRAM Exhaustion (OOM).**
   - **Solution:** Loaded the base model in `torch.bfloat16`, limited `max_new_tokens` to 128, and constrained image resolutions (`max_pixels`).
2. **Challenge:** **Connecting a Static Frontend to a Dynamic Colab Backend.**
   - **Solution:** Implemented `localtunnel` to create a public web address. Updated the React fetch call to pass the `Bypass-Tunnel-Reminder` header to skip the anti-bot screen.
3. **Challenge:** **Hallucinations during formatting.**
   - **Solution:** Fine-tuned the model explicitly on JSON structure, overriding its natural tendency to write conversational markdown.
4. **Challenge:** **Handling Non-Standard Uploads.**
   - **Solution:** Converted all uploads directly into an RGB `PIL.Image` in memory (`io.BytesIO`), normalizing JPEGs, PNGs, and preventing Alpha-channel crashes.
5. **Challenge:** **Model Loading Timeouts.**
   - **Solution:** Centralized the model loading logic to ensure the heavy weights are loaded once into global memory, rather than reloading on every API request.

## PHASE 7 — Interview Prep Pack

### Beginner Level (Core Concepts)
1. **Q:** What is OCR?
   **A:** Optical Character Recognition. Traditionally it means turning pixels into text. In our project, we do *Semantic OCR* using a Vision-Language Model.
2. **Q:** What is Qwen2-VL?
   **A:** A Vision-Language Model developed by Alibaba, capable of processing both text and images to answer questions or extract data.
3. **Q:** What is LoRA?
   **A:** Low-Rank Adaptation. A fine-tuning technique that trains only a tiny adapter network while keeping the massive base model frozen, saving memory and compute.
4. **Q:** Why do we need a GPU for this project?
   **A:** Vision-Language Models perform millions of matrix multiplications to process an image. CPUs are too slow for this; GPUs process them in parallel, reducing response times from minutes to seconds.
5. **Q:** What does FastAPI do in this architecture?
   **A:** It acts as the API Gateway, receiving the image from the React frontend, passing it to the ML model, and returning the JSON response.
6. **Q:** How do you pass an image from React to Python?
   **A:** Using a `FormData` object containing the file, sent via a `POST` request with `multipart/form-data` encoding.
7. **Q:** What is `localtunnel` used for?
   **A:** It creates a public HTTPS URL that forwards internet traffic to the local port (8000) running inside our Google Colab instance.
8. **Q:** Why do we ask the model to output JSON?
   **A:** JSON is easily parsed by our frontend (and downstream systems) to display the data clearly or save it to a database, compared to messy plain text.
9. **Q:** What is `PIL` (Pillow)?
   **A:** The Python Imaging Library, used to open and process the raw image bytes into a format the AI model can read.
10. **Q:** What is VRAM?
    **A:** Video RAM (memory on the GPU). Loading a 2B parameter model requires significant VRAM, which is why we must optimize memory usage.

### Intermediate Level (Architecture & Implementation)
11. **Q:** How does LoRA prevent "Catastrophic Forgetting"?
    **A:** Because the original base model weights are frozen, the model retains all its general pre-trained knowledge. The LoRA adapter simply acts as a small "lens" applied on top for the specific task.
12. **Q:** Why do we load the model in `bfloat16`?
    **A:** `bfloat16` uses 16 bits per parameter instead of 32 (FP32). This halves the VRAM required to load the model with almost zero loss in model accuracy.
13. **Q:** How does the backend handle the model generating conversational text alongside the JSON?
    **A:** The backend parses the raw output using `output_text.find('{')` and `rfind('}')` to extract just the dictionary portion, ignoring conversational padding like "Here is your data:".
14. **Q:** What is the purpose of `max_new_tokens=128`?
    **A:** It caps the length of the model's output. Since we only need Name, DOB, ID, and Address, 128 tokens is plenty. This prevents the model from rambling and saves compute time.
15. **Q:** How do we restrict the image size processed by the model?
    **A:** By passing `min_pixels` and `max_pixels` to `process_vision_info`, which intelligently scales down massive iPhone photos before feeding them to the Vision Transformer.
16. **Q:** What does `torch.no_grad()` do?
    **A:** It disables gradient calculation during inference. This saves significant memory and compute power since we aren't updating the model weights.
17. **Q:** Why do we use Vite over Create React App (CRA)?
    **A:** Vite uses native ES modules to serve code during development, making server start and Hot Module Replacement (HMR) virtually instantaneous, unlike CRA which bundles the entire app first.
18. **Q:** How does the `Bypass-Tunnel-Reminder` header work?
    **A:** Localtunnel added an interstitial warning page to prevent phishing. Sending this specific header tells their servers we are a legitimate API client, bypassing the HTML screen and reaching our FastAPI app.
19. **Q:** What happens if the JSON from the model is malformed?
    **A:** `json.loads()` will throw an exception. The backend catches this and returns a custom error JSON (`{"error": "Invalid JSON"}`) so the frontend doesn't crash trying to parse a 500 error page.
20. **Q:** How do you handle CORS in this app?
    **A:** We add `CORSMiddleware` to FastAPI with `allow_origins=["*"]`, enabling the Vercel frontend to fetch data from the Colab backend without browser security blocks.

### Senior Level (Design Tradeoffs & Scaling)
21. **Q:** How would you scale this architecture to handle 10,000 requests per minute?
    **A:** I would containerize the backend using Docker, deploy it to a Kubernetes cluster with autoscaling GPU nodes (e.g., AWS EKS with G4 instances), and place a Load Balancer in front. Colab and localtunnel would be entirely removed.
22. **Q:** What are the security implications of this system regarding PII?
    **A:** Sending ID cards containing PII across a public `localtunnel` is a major security risk for production. In production, we'd need end-to-end TLS, SOC2 compliance, a VPC (Virtual Private Cloud), and strict data-retention policies (deleting images from memory immediately after inference).
23. **Q:** How would you improve the extraction accuracy without retraining the model?
    **A:** I would implement **Grammar-Constrained Decoding** (using a library like `Outlines` or `guidance`). This forces the LLM at the token-generation level to *only* output valid JSON matching a specific Pydantic schema.
24. **Q:** Why fine-tune a model when you could just use Few-Shot Prompting?
    **A:** Few-shot prompting works, but it consumes context window space (increasing latency) and is brittle on complex ID layouts. Fine-tuning ingrains the JSON formatting behavior into the model's weights, making it faster and far more robust.
25. **Q:** How do you evaluate the success of this model in your ML pipeline?
    **A:** By running an `evaluation` script on a holdout test set. We would measure **JSON validation success rate** (did it output valid JSON?) and **Character Error Rate (CER)** for the actual text extracted against ground-truth labels.
26. **Q:** What is the tradeoff of using a 2B model vs an 8B model (like LLaVA-1.5-8B)?
    **A:** The 2B model is cheaper and faster to run, fitting comfortably on a T4. An 8B model might have slightly higher reasoning capabilities on highly obscured documents but requires more expensive hardware (A10G or A100) and suffers from higher inference latency.
27. **Q:** If the API gets stuck on a request, how do you handle it?
    **A:** FastAPI supports asynchronous execution, but PyTorch inference is fundamentally blocking on the GPU thread. To prevent the server from hanging, we'd need to offload inference to a Celery worker queue or use asynchronous inference servers like vLLM.
28. **Q:** What are the tradeoffs of using LoRA over QLoRA?
    **A:** QLoRA adds 4-bit quantization to the base model, reducing memory even further at the cost of a slight performance hit and slower training times due to dequantization overhead. Since a 2B model in BF16 fits on a T4, standard LoRA is sufficient and faster.

### "Why did you choose X?" (Tech Stack Defense)
29. **Why Qwen2-VL over GPT-4V?** 
    **A:** GPT-4V is closed-source, costs money per API call, and sends sensitive PII (ID cards) to OpenAI's servers. Qwen2-VL is open-weights, runs locally for free, and ensures complete data privacy.
30. **Why LoRA instead of Full Fine Tuning?** 
    **A:** Full FT of a 2B model would require multi-GPU clusters and risks catastrophic forgetting. LoRA achieves identical results using 1% of the trainable parameters on a single free GPU.
31. **Why FastAPI over Django?** 
    **A:** Django is a heavy MVC framework built for traditional database-backed web apps. We only needed a lightweight, high-performance API gateway to receive images and return JSON.
32. **Why Google Colab for the backend?** 
    **A:** It provides free, on-demand access to NVIDIA T4 GPUs, perfect for prototyping and portfolio demonstrations without incurring AWS/GCP hosting costs.
33. **Why Tailwind CSS?** 
    **A:** It allows for rapid, utility-first UI development without needing to manage separate, bloated CSS files or worry about class-name collisions.

## PHASE 8 — One-Page Revision Sheet & Glossary

### 10-Minute Revision Sheet

**Project:** ID Verification & OCR via Fine-Tuned VLM
**Workflow:** React UI (Multipart Form) -> FastAPI -> Image Resizing -> Qwen2-VL (LoRA) Inference -> JSON Extraction -> React UI.
**Tech Stack:**
- **Backend/ML:** Python, FastAPI, PyTorch, Qwen2-VL, LoRA (PEFT), HuggingFace.
- **Frontend:** React, Vite, Tailwind.
- **Infrastructure:** Vercel (Frontend), Google Colab + localtunnel (Backend).
**Key Design Choices:**
- **LoRA Fine-Tuning:** Trained an adapter to force strict JSON output without unfreezing the 2B base model.
- **bfloat16 Precision:** Halved VRAM requirements to fit on a free T4 GPU.
- **Robust Parsing:** Implemented substring extraction (`text[start:end]`) to salvage JSON even if the model hallucinates conversational padding.
**Common Gotchas:**
- `localtunnel` requires the `Bypass-Tunnel-Reminder` header.
- Colab sessions die after 12 hours or inactivity; the backend must be manually restarted.
- VRAM OOM (Out Of Memory) errors occur if the image resolution isn't capped via `max_pixels`.

### Glossary of Key Technical Terms
1. **bfloat16:** A 16-bit floating point format optimized for deep learning, saving memory.
2. **Catastrophic Forgetting:** When an AI forgets its general knowledge during fine-tuning. Prevented by LoRA.
3. **CORS:** Cross-Origin Resource Sharing. Configured in FastAPI to allow React to communicate with it.
4. **CUDA:** NVIDIA's parallel computing platform, required to run PyTorch on GPUs.
5. **FastAPI:** The Python framework handling incoming HTTP requests.
6. **Inference:** The phase where a trained model analyzes new data and makes predictions.
7. **JSON Extraction:** Using string matching to pull `{...}` dictionaries from raw LLM text.
8. **localtunnel:** The tool exposing the Colab server's local port 8000 to a public URL.
9. **LoRA (PEFT):** Low-Rank Adaptation. A technique to fine-tune massive models efficiently.
10. **max_new_tokens:** A constraint limiting how long the AI's response can be.
11. **Multipart Form Data:** The HTTP encoding type required for uploading files (images).
12. **OCR (Optical Character Recognition):** Converting images of text into machine-encoded text.
13. **PIL (Pillow):** Python library used to parse image bytes into RGB formats.
14. **SFT (Supervised Fine-Tuning):** Training an AI on exact pairs of input (images) and expected output (JSON).
15. **Vite:** The frontend build tool providing fast, native ES module serving.
16. **VLM (Vision-Language Model):** An AI model capable of understanding both images and text simultaneously.
17. **VRAM:** Video RAM on the GPU. The primary bottleneck for running LLMs/VLMs.
