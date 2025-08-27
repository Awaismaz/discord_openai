# services/openai_coach.py
import os, io, re, time, difflib, asyncio
from typing import Optional, Tuple, List, Dict

import aiohttp
from openai import OpenAI
import fitz  # PyMuPDF

# ------------------------------------------------------
# Config / Globals
# ------------------------------------------------------
_client = None
_assistant_id = os.getenv("NPF_ASSISTANT_ID")  # asst_...
ALLOWED_TYPES = {"application/pdf", "text/plain"}
MAX_FILE_MB = 15
MIN_FILE_BYTES = 1024            # treat <1KB as effectively empty
MIN_TEXT_CHARS_NORM = 40         # minimum normalized text to consider a PDF "not blank"

# In-memory state (Phase 1)
_THREAD_BY_USER: Dict[str, str] = {}                   # user_id -> thread_id
_HAS_FILE_IN_SESSION: Dict[str, bool] = {}             # user_id -> bool
_FILE_MAP: Dict[str, str] = {}                         # file_id -> original filename
_PAGE_INDEX: Dict[str, Dict[str, List[str]]] = {}      # user_id -> file_id -> [normalized page text]
_FILES_BY_USER: Dict[str, List[str]] = {}              # user_id -> [file_id,...]

# ------------------------------------------------------
# Utilities
# ------------------------------------------------------
def _client_once():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("[COACHDEBUG] OpenAI client initialized")
    return _client

def get_or_create_thread(user_id: str) -> str:
    if user_id in _THREAD_BY_USER:
        return _THREAD_BY_USER[user_id]
    client = _client_once()
    thread = client.beta.threads.create()
    _THREAD_BY_USER[user_id] = thread.id
    _HAS_FILE_IN_SESSION[user_id] = False
    print(f"[COACHDEBUG] New thread for {user_id}: {thread.id}")
    return thread.id

def _infer_mime_from_name(filename: str) -> Optional[str]:
    fn = (filename or "").lower()
    if fn.endswith(".pdf"): return "application/pdf"
    if fn.endswith(".txt"): return "text/plain"
    return None

_ws_rx = re.compile(r"\s+", re.UNICODE)
def _norm(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")
    s = _ws_rx.sub(" ", s).strip().lower()
    return s

def _strip_page_tag(snippet: str) -> str:
    # remove trailing (page X) or p. X
    return re.sub(r"\(?\b(?:page|p\.)\s*\d+\)?$", "", snippet, flags=re.IGNORECASE).strip()

async def fetch_attachment_bytes(url: str) -> bytes:
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as r:
            print(f"[COACHDEBUG] Fetch attachment HTTP {r.status}")
            if r.status >= 400:
                raise RuntimeError(f"FETCH_ERROR_HTTP_{r.status}")
            b = await r.read()
            print(f"[COACHDEBUG] Fetched {len(b)} bytes")
            return b

def upload_file_to_openai(file_bytes: bytes, filename: str, mime: str) -> str:
    client = _client_once()
    print(f"[COACHDEBUG] Uploading to OpenAI: {filename} ({mime}), {len(file_bytes)} bytes")
    try:
        f = client.files.create(
            file=(filename, io.BytesIO(file_bytes), mime),
            purpose="assistants",
        )
    except Exception as e:
        print(f"[COACHDEBUG] OpenAI upload error: {e}")
        raise RuntimeError("UPLOAD_ERROR_CORRUPTED") from e
    _FILE_MAP[f.id] = filename
    print(f"[COACHDEBUG] Uploaded file_id={f.id} -> {filename}")
    return f.id

# ------------------------------------------------------
# PDF indexing (Option A)
# ------------------------------------------------------
def extract_pdf_pages(pdf_bytes: bytes) -> List[str]:
    """Return list of normalized text per page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_norm: List[str] = []
    print(f"[COACHDEBUG] PyMuPDF pages: {doc.page_count}")
    for i, p in enumerate(doc, start=1):
        text = p.get_text("text") or ""
        n = _norm(text)
        pages_norm.append(n)
        if i <= 3:
            print(f"[COACHDEBUG] Page {i} chars={len(text)} (sample normalized)={n[:80]!r}")
    doc.close()
    return pages_norm

def index_pages(user_id: str, file_id: str, pages_norm: List[str]) -> None:
    _PAGE_INDEX.setdefault(user_id, {})[file_id] = pages_norm
    print(f"[COACHDEBUG] Indexed pages for user={user_id}, file_id={file_id}, pages={len(pages_norm)}")

def _probe_snippets(q: str) -> List[str]:
    # generate several probes from the quote (beginning/middle/end)
    q = q.strip()
    L = len(q)
    if L < 30:
        return [q]
    slices = [
        q[:90],
        q[max(0, L//2 - 45): min(L, L//2 + 45)],
        q[-90:]
    ]
    out, seen = [], set()
    for s in slices:
        s = s.strip()
        if len(s) >= 20 and s not in seen:
            seen.add(s); out.append(s)
    return out or [q[:120]]

def locate_page(user_id: str, file_id: str, quoted_snippet: str) -> Optional[int]:
    """Find 1-based page number where quoted_snippet appears (exact -> fuzzy)."""
    pages = _PAGE_INDEX.get(user_id, {}).get(file_id)
    if not pages:
        print("[COACHDEBUG] No page index available for locate_page")
        return None

    q_raw = _strip_page_tag(quoted_snippet or "")
    q = _norm(q_raw)
    if not q or len(q) < 12:
        print(f"[COACHDEBUG] Quote too short to match: {q_raw!r}")
        return None

    probes = [_norm(s) for s in _probe_snippets(q_raw)]
    print(f"[COACHDEBUG] Probes: {[p[:50] for p in probes]}")

    # 1) exact substring on any probe
    for i, page_text in enumerate(pages, start=1):
        for probe in probes:
            if probe and probe in page_text:
                print(f"[COACHDEBUG] Exact match on page {i}")
                return i

    # 2) fuzzy best-match
    best_page, best_ratio = None, 0.0
    for i, page_text in enumerate(pages, start=1):
        for probe in probes:
            ratio = difflib.SequenceMatcher(a=probe, b=page_text).quick_ratio()
            if ratio > best_ratio:
                best_ratio, best_page = ratio, i
    print(f"[COACHDEBUG] Best fuzzy ratio={best_ratio:.3f} on page {best_page}")
    if best_ratio is not None and best_ratio >= 0.82:
        return best_page
    return None

# ------------------------------------------------------
# Assistant messaging
# ------------------------------------------------------
def post_user_message(thread_id: str, content: str, file_ids: Optional[List[str]] = None):
    client = _client_once()
    attachments = []
    if file_ids:
        for fid in file_ids:
            attachments.append({"file_id": fid, "tools": [{"type": "file_search"}]})
    print(f"[COACHDEBUG] Posting message to thread={thread_id} with {len(attachments)} attachments")
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content or "Please analyze the file(s).",
        attachments=attachments if attachments else None,
    )

async def run_and_wait(thread_id: str, assistant_id: str, timeout_s: int = 90) -> Tuple[str, List[dict]]:
    """Async poll to avoid blocking Discord heartbeat."""
    client = _client_once()
    print(f"[COACHDEBUG] Starting run for thread={thread_id}, assistant={assistant_id}")

    run = await asyncio.to_thread(
        client.beta.threads.runs.create,
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    t0 = time.time()
    while True:
        await asyncio.sleep(1.0)
        run = await asyncio.to_thread(
            client.beta.threads.runs.retrieve,
            thread_id=thread_id,
            run_id=run.id
        )
        if run.status in ("completed", "failed", "requires_action", "cancelled", "expired"):
            print(f"[COACHDEBUG] Run status={run.status}")
            break
        if time.time() - t0 > timeout_s:
            print("[COACHDEBUG] Run timeout")
            break

    if run.status != "completed":
        return ("Sorry, the analysis took too long or failed. Please try again.", [])

    messages = await asyncio.to_thread(
        client.beta.threads.messages.list,
        thread_id=thread_id,
        order="desc",
        limit=1
    )
    if not messages.data:
        return ("No response produced.", [])
    msg = messages.data[0]

    text_parts: List[str] = []
    citations: List[dict] = []

    for c in msg.content:
        if getattr(c, "type", None) == "text":
            text_val = c.text.value.strip()
            text_parts.append(text_val)
            print(f"[COACHDEBUG] Model text len={len(text_val)}")
            anns = c.text.annotations or []
            print(f"[COACHDEBUG] Annotations count={len(anns)}")
            for ann in anns:
                try:
                    ad = ann.model_dump() if hasattr(ann, "model_dump") else dict(ann)  # type: ignore
                except Exception:
                    ad = {}
                if ad.get("type") == "file_citation":
                    fc = ad.get("file_citation") or {}
                    fid = fc.get("file_id")
                    quote = (fc.get("quote") or "").strip()
                    print(f"[COACHDEBUG] Citation: file_id={fid}, quote_len={len(quote)}")
                    citations.append({"file_id": fid, "quote": quote})

    answer = ("\n".join(text_parts).strip() or "No text response.")
    return (answer, citations)

# ------------------------------------------------------
# Citation formatting + fallback synthesis
# ------------------------------------------------------
_QUOTE_RX = re.compile(r'“([^”]+)”|"([^"]+)"')

def _extract_quote_from_answer(answer: str) -> Optional[str]:
    m = _QUOTE_RX.search(answer or "")
    if not m:
        return None
    q = m.group(1) or m.group(2)
    q = (q or "").strip()
    return q if len(q) >= 12 else None

def synthesize_citations_from_answer(answer: str, user_id: str) -> List[dict]:
    quote = _extract_quote_from_answer(answer)
    if not quote:
        print("[COACHDEBUG] No explicit quote found in answer for fallback")
        return []
    # Try each known file for this user
    file_ids = _FILES_BY_USER.get(user_id, [])
    for fid in file_ids:
        page = locate_page(user_id, fid, quote)
        if page:
            print(f"[COACHDEBUG] Fallback synthesized citation on page {page} for file {fid}")
            return [{"file_id": fid, "quote": quote}]
    if file_ids:
        print("[COACHDEBUG] Fallback could not locate page; returning minimal citation")
        return [{"file_id": file_ids[-1], "quote": quote}]
    return []

def format_with_citations(answer: str, citations: List[dict], user_id: str) -> str:
    if not citations:
        print("[COACHDEBUG] No citations to format")
        return answer

    seen = set()
    lines = []
    idx = 1

    for c in citations:
        fid = c.get("file_id")
        filename = _FILE_MAP.get(fid, fid)
        snippet = (c.get("quote", "") or "").strip()

        page = locate_page(user_id, fid, snippet) if (fid and snippet) else None

        if not snippet:
            snippet = f"See source {filename}"
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."

        page_part = f", page {page}" if page else ", page n/a"
        key = (fid, snippet, page)
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"[{idx}] {snippet} ({filename}{page_part})")
        idx += 1

    return f"{answer}\n\n**Citations:**\n" + "\n".join(lines)

# ------------------------------------------------------
# Main entry
# ------------------------------------------------------
async def coach_answer(user_id: str, question: Optional[str], attachment: Optional[dict]) -> str:
    """
    Phase 1 behavior:
      - Strict preflight (reject empty/corrupted/image-only PDFs BEFORE upload).
      - Build per-page index after successful upload.
      - Retrieval-first answers (prompt enforces); summaries only if asked.
      - Page numbers via local page matching. Fallback when annotations absent.
    """
    if not _assistant_id:
        return "Assistant is not configured yet. Please set NPF_ASSISTANT_ID."

    thread_id = get_or_create_thread(user_id)
    file_ids: List[str] = []

    fid: Optional[str] = None
    pages_norm_pre: Optional[List[str]] = None  # normalized per-page text from preflight

    if attachment:
        ct = attachment.get("content_type") or _infer_mime_from_name(attachment.get("filename", ""))
        size = attachment.get("size", 0)
        print(f"[COACHDEBUG] Upload received: ct={ct}, size={size}, name={attachment.get('filename')}")

        # Basic guards
        if ct not in ALLOWED_TYPES:
            return "Unsupported file type. Please upload PDF or TXT."
        if size == 0:
            return "This file is empty, no analysis possible."
        if size < MIN_FILE_BYTES:
            print(f"[COACHDEBUG] Very small file ({size} bytes) — treat as empty")
            return "This file is empty, no analysis possible."
        if size > MAX_FILE_MB * 1024 * 1024:
            return f"File too large. Please keep under {MAX_FILE_MB} MB."

        # Download bytes
        try:
            data = await fetch_attachment_bytes(attachment["url"])
        except RuntimeError:
            return "I couldn’t download the file from Discord. Please re-upload and try again."
        except Exception:
            return "Unexpected error fetching the file. Please try again."
        if not data or len(data) < MIN_FILE_BYTES:
            print(f"[COACHDEBUG] Downloaded tiny payload ({len(data) if data else 0} bytes)")
            return "This file is empty, no analysis possible."
        print(f"[COACHDEBUG] Downloaded bytes={len(data)}")

        # ---- Strict preflight BEFORE OpenAI upload ----
        if ct == "application/pdf":
            try:
                doc = fitz.open(stream=data, filetype="pdf")
                page_count = doc.page_count
                if page_count == 0:
                    doc.close()
                    print("[COACHDEBUG] PDF has 0 pages -> corrupted")
                    return "This file is corrupted and cannot be read."

                raw_text_lens, norm_text_lens = [], []
                pages_norm_pre = []
                for p in doc:
                    t = p.get_text("text") or ""
                    raw_text_lens.append(len(t))
                    n = _norm(t)
                    norm_text_lens.append(len(n))
                    pages_norm_pre.append(n)
                doc.close()

                total_text_raw = sum(raw_text_lens)
                total_text_norm = sum(norm_text_lens)
                print(f"[COACHDEBUG] PDF preflight: pages={page_count}, "
                      f"total_text_raw={total_text_raw}, total_text_norm={total_text_norm}, "
                      f"first3_raw={raw_text_lens[:3]}, first3_norm={norm_text_lens[:3]}")

                if total_text_norm < MIN_TEXT_CHARS_NORM:
                    return ("I couldn’t find searchable text in this document. "
                            "If it’s a scanned/image-only PDF, text extraction isn’t enabled in this phase.")
            except Exception as e:
                print(f"[COACHDEBUG] PyMuPDF preflight error: {e}")
                return "This file is corrupted and cannot be read."

        elif ct == "text/plain":
            try:
                txt = (data.decode("utf-8", errors="ignore")).strip()
            except Exception:
                txt = ""
            if len(txt) == 0:
                print("[COACHDEBUG] TXT is empty after decode/strip")
                return "This file is empty, no analysis possible."
            # pages_norm_pre remains None for TXT

        # ---- Upload to OpenAI (safe now) ----
        try:
            fid = upload_file_to_openai(data, attachment["filename"], ct)
        except RuntimeError as e:
            if str(e) == "UPLOAD_ERROR_CORRUPTED":
                return "This file is corrupted and cannot be read."
            return "There was a problem processing this file. Please try another file."
        except Exception:
            return "There was a problem processing this file. Please try another file."

        # Track user->file for fallback matching
        _FILES_BY_USER.setdefault(user_id, []).append(fid)

        # Build page index AFTER upload (using preflight pages if available)
        if ct == "application/pdf":
            if pages_norm_pre is None:
                try:
                    print("[COACHDEBUG] pages_norm_pre missing; extracting after upload")
                    pages_norm_pre = extract_pdf_pages(data)
                except Exception as e:
                    print(f"[COACHDEBUG] Page extraction failed post-upload: {e}")
                    pages_norm_pre = []
            index_pages(user_id, fid, pages_norm_pre)
        else:
            index_pages(user_id, fid, [])  # TXT: no pages

        file_ids.append(fid)
        _HAS_FILE_IN_SESSION[user_id] = True  # only after full validation + upload success

    # Require prior valid upload if none in this call
    if not file_ids and not _HAS_FILE_IN_SESSION.get(user_id, False):
        return "Please upload a PDF or TXT to start a new session."

    # Ask the question
    post_user_message(thread_id, question or "", file_ids if file_ids else None)
    answer, cites = await run_and_wait(thread_id, _assistant_id)

    # If the Assistant didn't return annotations, synthesize from the answer's quoted text
    if not cites:
        fallback_cites = synthesize_citations_from_answer(answer, user_id)
        if fallback_cites:
            cites = fallback_cites

    # Optional nudge if model tried to summarize (still format either way)
    if not any(ch in answer for ch in ['"', '“', '”']) and not cites:
        print("[COACHDEBUG] No quotes detected in answer; model likely summarized (prompt should prevent).")

    return format_with_citations(answer, cites, user_id=user_id)

# ------------------------------------------------------
# Reset
# ------------------------------------------------------
def reset_user_thread(user_id: str):
    _THREAD_BY_USER.pop(user_id, None)
    _HAS_FILE_IN_SESSION.pop(user_id, None)
    for fid in list(_FILE_MAP.keys()):
        _FILE_MAP.pop(fid, None)
    _PAGE_INDEX.pop(user_id, None)
    _FILES_BY_USER.pop(user_id, None)
    print(f"[COACHDEBUG] Reset session for {user_id}")
