# services/openai_coach.py
import os, time, io, aiohttp
from typing import Optional, Tuple, List, Dict
from openai import OpenAI

_client = None
_assistant_id = os.getenv("NPF_ASSISTANT_ID")  # asst_...
# Phase-1 scope: PDFs (+ optional TXT), no OCR/images
ALLOWED_TYPES = {"application/pdf", "text/plain"}
MAX_FILE_MB = 15

# In-memory stores (Phase 1)
_THREAD_BY_USER: Dict[str, str] = {}
_VECTOR_BY_USER: Dict[str, str] = {}


def _client_once():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def get_or_create_thread(user_id: str) -> str:
    """Per-user Assistants thread (Phase 1, in-memory)."""
    if user_id in _THREAD_BY_USER:
        return _THREAD_BY_USER[user_id]
    client = _client_once()
    thread = client.beta.threads.create()
    _THREAD_BY_USER[user_id] = thread.id
    return thread.id


def has_vstore(user_id: str) -> bool:
    return user_id in _VECTOR_BY_USER


def get_or_create_vstore(user_id: str) -> str:
    """Per-user vector store used by file_search for retrieval/quotes."""
    if user_id in _VECTOR_BY_USER:
        return _VECTOR_BY_USER[user_id]
    vs = _client_once().beta.vector_stores.create(name=f"npf-vs-{user_id}")
    _VECTOR_BY_USER[user_id] = vs.id
    return vs.id


def add_file_to_vstore(vector_store_id: str, file_id: str):
    _client_once().beta.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id
    )


async def fetch_attachment_bytes(url: str, content_type: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            r.raise_for_status()
            return await r.read()


def upload_file_to_openai(file_bytes: bytes, filename: str, mime: str) -> str:
    client = _client_once()
    f = client.files.create(
        file=(filename, io.BytesIO(file_bytes), mime),
        purpose="assistants",
    )
    return f.id


def post_user_message(thread_id: str, content: str, file_ids: Optional[List[str]] = None):
    """Post the user's question and (optionally) attach files (with file_search tool)."""
    client = _client_once()
    attachments = []
    if file_ids:
        for fid in file_ids:
            attachments.append({"file_id": fid, "tools": [{"type": "file_search"}]})
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content or "Please analyze the file(s).",
        attachments=attachments if attachments else None,
    )


def run_and_wait(thread_id: str, assistant_id: str, vstore_id: Optional[str], timeout_s: int = 90) -> Tuple[str, List[dict]]:
    """Kick off a run (bound to vector store) and wait for completion."""
    client = _client_once()

    # Require a vector store for retrieval answers (enforces quote-first behavior via Assistant prompt)
    tool_resources = {"file_search": {"vector_stores": [vstore_id]}} if vstore_id else None

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        tool_resources=tool_resources
    )

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

    text_parts: List[str] = []
    citations: List[dict] = []

    # Defensive parsing across SDK versions
    for c in msg.content:
        if getattr(c, "type", None) == "text":
            text_parts.append(c.text.value.strip())
            anns = c.text.annotations or []
            for ann in anns:
                try:
                    ad = ann.model_dump() if hasattr(ann, "model_dump") else dict(ann)  # type: ignore
                except Exception:
                    ad = {}
                if ad.get("type") == "file_citation":
                    fc = ad.get("file_citation") or {}
                    citations.append({
                        "file_id": fc.get("file_id"),
                        # quote may be missing in some SDK versions
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
        key = (c.get("file_id"), c.get("quote", ""))
        if key in seen:
            continue
        seen.add(key)
        snippet = (c.get("quote", "") or "").strip()
        if not snippet:
            snippet = f"See source {c.get('file_id', 'unknown')}"
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        lines.append(f"[{idx}] {snippet}")
        idx += 1
    return f"{answer}\n\n**Citations:**\n" + "\n".join(lines)


async def coach_answer(user_id: str, question: Optional[str], attachment: Optional[dict]) -> str:
    """
    attachment: dict with keys {url, filename, content_type, size}
    Behavior (Phase 1):
      - Requires at least one PDF/TXT uploaded per session (per-user vector store).
      - Retrieval-first answers (quote + page number if available) via Assistant prompt.
      - Summaries only when user explicitly asks to summarize.
    """
    if not _assistant_id:
        return "Assistant is not configured yet. Please set NPF_ASSISTANT_ID."

    vstore_id: Optional[str] = None
    file_ids: List[str] = []

    # Handle upload (optional per call; required at least once per session)
    if attachment:
        ct = attachment["content_type"]
        size = attachment["size"]
        if ct not in ALLOWED_TYPES:
            return "Unsupported file type. Please upload PDF or TXT."
        if size > MAX_FILE_MB * 1024 * 1024:
            return f"File too large. Please keep under {MAX_FILE_MB} MB."

        data = await fetch_attachment_bytes(attachment["url"], ct)
        fid = upload_file_to_openai(data, attachment["filename"], ct)
        file_ids.append(fid)

        # Ensure vector store exists and attach the file(s)
        vstore_id = get_or_create_vstore(user_id)
        for fid in file_ids:
            add_file_to_vstore(vstore_id, fid)

    # If no upload in this call, require an existing vector store (enforces upload-before-QA)
    if not file_ids:
        if not has_vstore(user_id):
            return "Please upload a PDF or TXT to start a new session."
        vstore_id = _VECTOR_BY_USER[user_id]

    # Post the user question and run
    thread_id = get_or_create_thread(user_id)
    post_user_message(thread_id, question or "", file_ids if file_ids else None)
    answer, cites = run_and_wait(thread_id, _assistant_id, vstore_id=vstore_id)
    return format_with_citations(answer, cites)


def reset_user_thread(user_id: str):
    """Clear both thread and vector store so user must upload again (client requirement)."""
    _THREAD_BY_USER.pop(user_id, None)
    _VECTOR_BY_USER.pop(user_id, None)
