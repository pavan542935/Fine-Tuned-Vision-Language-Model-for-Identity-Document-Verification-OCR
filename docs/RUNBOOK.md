# 🚀 IDVLM Project Runbook

Since the model requires a GPU, the backend is designed to run on Google Colab, while the UI runs on your local computer. Follow this guide to boot up the project anytime.

## Step 1: Start the AI Backend (Google Colab)

1. Open a new Google Colab notebook.
2. Go to **Runtime > Change runtime type** and ensure **T4 GPU** is selected!
3. Paste the following block of code into a cell and run it:

```python
# 1. Mount your Google Drive (where the model is saved)
from google.colab import drive
drive.mount('/content/drive')

import os

# 2. Clone the repository if it doesn't exist, or pull latest changes
%cd /content
if not os.path.exists('repo'):
    !git clone https://github.com/pavan542935/Fine-Tuned-Vision-Language-Model-for-Identity-Document-Verification-OCR repo

%cd /content/repo
!git fetch --all
!git reset --hard origin/main

# 3. Install dependencies
!pip install -r requirements.txt
!pip install fastapi==0.112.2 starlette==0.37.2 uvicorn python-multipart

# 4. Start the FastAPI backend server in the background
import subprocess
subprocess.Popen(["python", "backend/main.py"])

# 5. Create a public tunnel so your React app can reach the Colab GPU
!npm install -g localtunnel
!lt --port 8000
```

4. The cell will eventually print a URL like: `your url is: https://some-random-words.loca.lt`
5. **Copy this URL.**

---

## Step 2: Connect the Frontend (Local Machine)

1. Open your code editor and go to `frontend/src/App.jsx`.
2. Scroll to **Line 55**.
3. Replace the fetch URL with the new URL you just got from Colab. 
   *(Make sure to keep the `/api/extract` at the very end!)*
   
   Example:
   ```javascript
   const response = await fetch('https://some-random-words.loca.lt/api/extract', {
   ```
4. Save the file.

---

## Step 3: Run the UI

1. Open a terminal on your computer.
2. Navigate to the frontend folder:
   ```bash
   cd "C:\Fine-Tuned Vision-Language Model for Identity Document Verification & OCR\frontend"
   ```
3. Start the UI using `npx`:
   ```bash
   npx vite
   ```
4. Open `http://localhost:5173` in your browser.

**⚠️ Troubleshooting (Failed to fetch):**
If clicking "Extract Data" throws a `Failed to fetch` error, localtunnel is blocking the request to verify you are a human.
1. Open a new tab in your browser and go to your Colab `loca.lt` link directly.
2. It will show a "Friendly Reminder" warning screen with an IP address at the top.
3. Type that exact IP address into the box and click **Continue**.
4. Go back to your React app (`http://localhost:5173`) and try extracting again!
