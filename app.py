"""Triton Requirements Guide - FastAPI app."""
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from courses import as_context_items, courses_taken, find_codes, implied_by

# ---------- Config: change settings here ----------
EMBED_MODEL = "gemini-embedding-001"
GENERATE_MODEL = "gemini-3.5-flash-lite"  # one constant; flash-lite is faster and the free tier is stricter on full flash
CHUNK_WORDS = 400                     # target words per chunk
CHUNK_OVERLAP = 40                    # words repeated between neighbouring chunks
TOP_K = 8                             # chunks kept for the question itself
MAX_CONTEXT_CHUNKS = 12               # most chunks sent to Gemini
DAILY_LIMIT = 10                      # questions per IP per day on the server key
MAX_QUESTION_CHARS = 400
ANSWER_MAX_WORDS = 300                # raised from 150 so elective lists fit
FEEDBACK_FORM_URL = ""                # paste your Google Form link here to show the feedback link
# --------------------------------------------------

RULES = f"""You answer student questions about UCSD requirements using ONLY the numbered sources below.
- If the sources do not contain the answer, say the catalog pages indexed here do not cover it and suggest checking with an academic advisor.
- Keep the answer under {ANSWER_MAX_WORDS} words, in plain language. Plain text only: no markdown, no asterisks, no bullet symbols.
- Cite sources like [1], [2], matching the source numbers.
- Never invent course codes, unit counts, or policies.
- Course entries in the sources list each course's prerequisites and end with a note: "Ready" means the student has met them, "Missing" names what is still needed, and a note to check the text means it was too complex to check automatically. If the student says they finished a course, its own prerequisites are already counted as finished.
- If asked what to take next, list required (core) courses before electives, and recommend only courses that appear in the major's requirements in the sources and are marked Ready. Mention important courses that are Missing a prerequisite as later steps, naming what is missing. If a missing prerequisite is itself a Ready course, suggest taking it first. Say which prerequisites you assumed.
- Do not say when courses are offered. The catalog does not list that; the department website has sample plans.
- End with one line telling the student to confirm with their advisor."""

load_dotenv()
app = FastAPI()
usage = {}  # (hashed IP, date) -> questions asked on the server key. Memory only, never written to disk.

CHUNKS = json.loads(Path("data/index.json").read_text())
VECTORS = np.load("data/vectors.npy")


def make_client(user_key=None):
    return genai.Client(api_key=user_key or os.environ["GEMINI_API_KEY"])


def retrieve(client, queries):
    """Return the closest chunks for one question or a list of queries (the first is the student's)."""
    queries = [queries] if isinstance(queries, str) else queries
    response = client.models.embed_content(
        model=EMBED_MODEL, contents=queries,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
    picked = []
    for n, embedding in enumerate(response.embeddings):
        q = np.array(embedding.values, dtype="float32")
        q /= np.linalg.norm(q)
        best = np.argsort(VECTORS @ q)[::-1][:TOP_K if n == 0 else 4]  # unit vectors: dot = cosine
        picked += [i for i in best if i not in picked]
    return [CHUNKS[i] for i in picked[:MAX_CONTEXT_CHUNKS]]


def gather_context(client, question):
    """Return (sources for the answer, courses the student says they took)."""
    taken = courses_taken(question)
    queries = [question]
    if taken:  # "what next" questions need each part of the major's requirements, so search them separately
        queries += [f"{part}: {question}" for part in
                    ("lower-division requirements", "upper-division core courses", "electives")]
    chunks = retrieve(client, queries)
    # Candidates: courses named in the best-matching program's requirements (lower-division first).
    program_chunks = [c for c in chunks if "/curric/" in c["url"] and c.get("heading")]
    if program_chunks:
        program = program_chunks[0]["heading"].split(" / ")[0]
        program_chunks = sorted((c for c in program_chunks if c["heading"].startswith(program)),
                                key=lambda c: c["heading"])
    named = find_codes(" ".join(c["text"] for c in program_chunks))
    done, groups = implied_by(taken)
    candidates = [c for c in named if c not in done][:80]
    items = as_context_items(list(done) + candidates, done, groups)
    return chunks + items, taken


def build_prompt(question, sources, taken):
    numbered = "\n\n".join(f"[{i}] {s['title']} - {s['heading']}\n{s['text']}" if s.get("heading")
                            else f"[{i}] {s['title']}\n{s['text']}" for i, s in enumerate(sources, 1))
    done = f"\nThe student says they finished: {', '.join(taken)}." if taken else ""
    return f"{RULES}\n\nSources:\n{numbered}\n{done}\nQuestion: {question}"


ANSWER_CONFIG = types.GenerateContentConfig(
    temperature=0.2, thinking_config=types.ThinkingConfig(thinking_level="LOW"))


def cited_sources(answer, sources):
    """Keep only the sources the answer actually cites, e.g. [2]."""
    groups = re.findall(r"\[([\d,\s]+)\]", answer)  # handles [3] and [3, 10]
    numbers = sorted({int(n) for g in groups for n in re.findall(r"\d+", g)})
    return [{"n": n, "title": sources[n - 1]["title"], "url": sources[n - 1]["url"],
             "excerpt": sources[n - 1]["text"][:500]} for n in numbers if 1 <= n <= len(sources)]


def counter_key(request):
    """Hashed IP + today's date. Raw IPs are never kept."""
    forwarded = request.headers.get("x-forwarded-for")  # set by Render / Hugging Face proxies
    ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
    today = date.today()
    for old in [k for k in usage if k[1] != today]:
        del usage[old]
    return hashlib.sha256(ip.encode()).hexdigest()[:16], today


def explain(error):
    """Turn a Gemini error into a short readable message."""
    if isinstance(error, errors.ClientError) and error.code == 429:
        return "Gemini is busy or its rate limit was hit. Please wait a minute and try again."
    if isinstance(error, errors.ClientError) and error.code in (400, 401, 403):
        return "Gemini rejected the request. If you pasted your own key, check that it is correct."
    return "Gemini is unavailable right now. Please try again."


def check_request(body, request, user_key):
    """Validate the question and the daily limit. Returns (question, counter key)."""
    question = body.question.strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Questions must be 1 to {MAX_QUESTION_CHARS} characters.")
    key = counter_key(request)
    if not user_key and usage.get(key, 0) >= DAILY_LIMIT:
        raise HTTPException(429, "You have used today's free questions. Paste your own free "
                                 "Gemini key in the sidebar to keep going.")
    return question, key


def finish(answer, sources, key, user_key):
    """Count the question (server key only) and build the response body."""
    if not user_key:
        usage[key] = usage.get(key, 0) + 1
    remaining = None if user_key else DAILY_LIMIT - usage[key]
    return {"answer": answer, "citations": cited_sources(answer, sources),
            "model": GENERATE_MODEL, "remaining": remaining}


class Question(BaseModel):
    question: str


@app.get("/api/sources")
def sources():
    pages = {c["url"]: {"title": c["title"], "url": c["url"], "scraped": c["scraped"]} for c in CHUNKS}
    return list(pages.values())


@app.get("/api/config")
def config():
    return {"feedback_url": FEEDBACK_FORM_URL, "last_updated": max(c["scraped"] for c in CHUNKS)}


@app.post("/api/ask")
def ask(body: Question, request: Request, x_gemini_key: str | None = Header(default=None)):
    """Plain JSON answer (used by tests and scripts)."""
    question, key = check_request(body, request, x_gemini_key)
    try:
        client = make_client(x_gemini_key)
        found, taken = gather_context(client, question)
        response = client.models.generate_content(
            model=GENERATE_MODEL, contents=build_prompt(question, found, taken), config=ANSWER_CONFIG)
    except errors.APIError as error:
        raise HTTPException(429 if error.code == 429 else 502, explain(error))
    return finish(response.text, found, key, x_gemini_key)


@app.post("/api/ask-stream")
def ask_stream(body: Question, request: Request, x_gemini_key: str | None = Header(default=None)):
    """Same as /api/ask, but sends the answer as it is written (one JSON object per line)."""
    question, key = check_request(body, request, x_gemini_key)
    try:
        client = make_client(x_gemini_key)
        found, taken = gather_context(client, question)
    except errors.APIError as error:
        raise HTTPException(429 if error.code == 429 else 502, explain(error))

    def events():
        answer = ""
        try:
            for part in client.models.generate_content_stream(
                    model=GENERATE_MODEL, contents=build_prompt(question, found, taken), config=ANSWER_CONFIG):
                answer += part.text or ""
                yield json.dumps({"delta": part.text or ""}) + "\n"
        except errors.APIError as error:
            yield json.dumps({"error": explain(error)}) + "\n"
            return
        yield json.dumps({"done": finish(answer, found, key, x_gemini_key)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.exception_handler(HTTPException)
def readable_errors(request, error):
    return JSONResponse({"message": error.detail}, status_code=error.status_code)


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
