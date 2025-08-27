# NextPlay Financial Discord Bot

This project implements two AI-powered assistants for Discord:

1. **Chat Bot (`/chat`)** – Fast educational finance chatbot using `gpt-4o-mini`.  
   - No memory, no file upload.  
   - Sub-2s responses.  

2. **Coach Bot (`/coach`)** – File-based assistant powered by OpenAI Assistants API with `file_search`.  
   - Upload PDFs or TXT files and ask evidence-based questions.  
   - Returns **exact quotes** with filename + page number.  
   - Provides educational-only answers with compliance guardrails.  
   - Includes `/reset` to start a fresh session.

---

## Features
- **Retrieval-first answers** – strict Q&A pipeline, summaries only on request.  
- **Evidence-based citations** – quotes matched to page numbers in source PDFs.  
- **Guardrails & disclaimers** – avoids financial advice, adds compliance disclaimer.  
- **Error handling** – clear messages for empty or corrupted files.  
- **Stable sessions** – per-user memory, `/reset` clears state.  
- **Discord-native** – runs fully inside Discord with slash commands.  
- **Hosting ready** – deployable on [Render](https://render.com/) or similar PaaS.

---

## Tech Stack
- **Discord.py** (`app_commands` for slash commands)  
- **OpenAI Python SDK** (Assistants API + file_search tool)  
- **PyMuPDF** (`fitz`) for PDF parsing and page-level text extraction  
- **aiohttp** for async file downloads  
- **Python 3.10+**

---

## Environment Variables
Create a `.env` file (or set variables in Render):

DISCORD_BOT_TOKEN=xxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxx
NPF_ASSISTANT_ID=xxxxxxxxxxxxxxxxxx
CHAT_CHANNEL=chat
COACH_CHANNEL=coach
RATE_LIMIT_PER_MINUTE=5


---

## Running Locally
1. Clone repo and install dependencies:
   ```bash
   pip install -r requirements.txt
2. Set your .env.

3. Run the bot:
    python main.py

4. Invite the bot to your Discord server.

## Deployment
1. Push code to GitHub.

2. Connect repo to Render.

3. Create a Background Worker service.

4. Add environment variables.

5. Start command:
    python main.py

6. Bot runs 24/7 with auto-restart.

## Notes
- Page numbers: Assistants API doesn’t provide them directly; this project adds local page-matching logic.

- Compliance: Every answer includes the disclaimer:


*This information is for educational purposes only and not financial advice. Please consult a licensed financial professional before making any investment decisions.*

## License
Proprietary – for NextPlay Financial.

---