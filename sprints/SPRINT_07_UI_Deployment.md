# SPRINT 07 — User Interface & Deployment

## Phase Goal
Deploy our newly fine-tuned model into a beautiful, interactive web interface using Gradio, allowing anyone to upload an ID and instantly see the extracted JSON data.

## Scope
**In Scope:** 
- Gradio web application (`ui/app.py`).
- Premium dark-mode glassmorphism aesthetic styling.
- Public Colab sharing link.

**Out of Scope:** 
- Production deployment to Vercel/AWS (out of scope for this notebook phase).

## File/Folder Structure
- `/ui/app.py` (Created)
- `/sprints/SPRINT_07_UI_Deployment.md` (Created)

## Task Checklist
- [x] Write `ui/app.py` utilizing `gradio` and embedding the LoRA weights.
- [x] Apply modern UI styling and CSS.
- [x] You (the user) run `app.py` on Colab to get the public link!

## Manual Steps Required From Me
This is the final test! Let's boot up the web app directly from Colab.

1. Pull the UI code:
   ```bash
   %cd /content/repo
   !git pull
   ```
2. Start the web server:
   ```bash
   !python ui/app.py
   ```
3. Colab will give you a **public Gradio link** (e.g., `https://<random-hash>.gradio.live`). Click it!
4. Upload an image of an ID card and hit **Extract Data**.

## Definition of Done
- Web UI is running.
- You can successfully extract data visually through the interface.
