# services/openai_coach.py
import os, time, io, aiohttp
from typing import Optional, Tuple, List
from openai import OpenAI

_client = None
_assistant_id = os.getenv("NPF_ASSISTANT_ID")  # asst_...
# ALLOWED_TYPES: PDFs (+ optional TXT)
ALLOWED_TYPES = {"application/pdf", "text/plain"}  # removed image types and OCR path
MAX_FILE_MB = 15

def _client_once():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

# simple in-memory per-user thread store for Phase 1
_THREAD_BY_USER = {}

def get_or_create_thread(user_id: str) -> str:
    if user_id in _THREAD_BY_USER:
        return _THREAD_BY_USER[user_id]
    client = _client_once()
    thread = client.beta.threads.create()
    _THREAD_BY_USER[user_id] = thread.id
    return thread.id

async def fetch_attachment_bytes(url: str, content_type: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            r.raise_for_status()
            data = await r.read()
            return data

def upload_file_to_openai(file_bytes: bytes, filename: str, mime: str) -> str:
    client = _client_once()
    f = client.files.create(
        file=(filename, io.BytesIO(file_bytes), mime),
        purpose="assistants",
    )
    return f.id

def post_user_message(thread_id: str, content: str, file_ids: Optional[List[str]] = None):
    client = _client_once()
    attachments = []
    if file_ids:
        for fid in file_ids:
            attachments.append({"file_id": fid, "tools": [{"type":"file_search"}]})
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content or "Please analyze the file(s).",
        attachments=attachments if attachments else None,
    )

def run_and_wait(thread_id: str, assistant_id: str, timeout_s: int = 90) -> Tuple[str, List[dict]]:
    client = _client_once()
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    t0 = time.time()
    while True:
        time.sleep(1.0)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status in ("completed", "failed", "requires_action", "cancelled", "expired"):
            break
        if time.time() - t0 > timeout_s:
            break

    if run.status != "completed":
        return ("Sorry, the analysis took too long or failed. Please try again.", [])

    messages = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
    if not messages.data:
        return ("No response produced.", [])
    msg = messages.data[0]

    text_parts = []
    citations = []

    for c in msg.content:
        if c.type == "text":
            text_parts.append(c.text.value.strip())
            anns = c.text.annotations or []
            for ann in anns:
                # normalize pydantic models to dict to avoid attribute errors across SDK versions
                try:
                    ad = ann.model_dump() if hasattr(ann, "model_dump") else dict(ann)
                except Exception:
                    ad = {}
                if ad.get("type") == "file_citation":
                    fc = ad.get("file_citation") or {}
                    citations.append({
                        "file_id": fc.get("file_id"),
                        # quote may be absent in some SDK versions
                        "quote": (fc.get("quote") or "").strip()
                    })

    return ("\n".join(text_parts).strip() or "No text response.", citations)

def format_with_citations(answer: str, citations: List[dict]) -> str:
    if not citations:
        return answer
    seen = set()
    lines = []
    idx = 1
    for c in citations:
        key = (c.get("file_id"), c.get("quote",""))
        if key in seen:
            continue
        seen.add(key)
        snippet = (c.get("quote","") or "").strip()
        if not snippet:
            snippet = f"See source {c.get('file_id','unknown')}"
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        lines.append(f"[{idx}] {snippet}")
        idx += 1
    return f"{answer}\n\n**Citations:**\n" + "\n".join(lines)

async def coach_answer(user_id: str, question: Optional[str], attachment: Optional[dict]) -> str:
    """
    attachment: dict with keys {url, filename, content_type, size}
    """
    if not _assistant_id:
        return "Assistant is not configured yet. Please set NPF_ASSISTANT_ID."

    file_ids = []
    if attachment:
        ct = attachment["content_type"]
        size = attachment["size"]
        if ct not in ALLOWED_TYPES:
            return "Unsupported file type. Please upload PDF, TXT, PNG or JPG."
        if size > MAX_FILE_MB * 1024 * 1024:
            return f"File too large. Please keep under {MAX_FILE_MB} MB."
        data = await fetch_attachment_bytes(attachment["url"], ct)
        fid = upload_file_to_openai(data, attachment["filename"], ct)
        file_ids.append(fid)

    thread_id = get_or_create_thread(user_id)
    post_user_message(thread_id, question or "", file_ids if file_ids else None)
    answer, cites = run_and_wait(thread_id, _assistant_id)
    return format_with_citations(answer, cites)

def reset_user_thread(user_id: str):
    if user_id in _THREAD_BY_USER:
        del _THREAD_BY_USER[user_id]
