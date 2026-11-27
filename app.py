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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from courses import as_context_items, find_codes, with_prereqs

# ---------- Config: change settings here ----------
EMBED_MODEL = "gemini-embedding-001"
GENERATE_MODEL = "gemini-3.5-flash"   # one constant; 2.5 models are closed to new users
CHUNK_WORDS = 400                     # target words per chunk
CHUNK_OVERLAP = 40                    # words repeated between neighbouring chunks
TOP_K = 8                             # chunks kept for the question itself
EXTRA_SEARCHES = 3                    # extra searches the planner may add (top 4 chunks each)
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
- Course entries in the sources list each course's prerequisites. If the student says they finished a course, treat that course's listed prerequisites as finished too (when it says "A or B", say which you are assuming).
- If asked what to take next, recommend only courses that appear in the major's requirements in the sources and whose listed prerequisites the student has met. Say which prerequisites you assumed, and name any missing prerequisite.
- Do not say when courses are offered. The catalog does not list that; the department website has sample plans.
- End with one line telling the student to confirm with their advisor."""

PLANNER = """A UCSD student asked a question about degree requirements. Reply with JSON only:
{"courses_taken": ["CSE 12"], "searches": ["..."]}
- courses_taken: every course the student says they finished, written with its department. If a bare number follows a department they named ("cse 12 and 21"), reuse that department.
- searches: up to 3 short catalog search queries. If they ask what to take next, include a query for their major's requirements (lower-division, core, electives)."""

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


def plan_question(client, question):
    """Ask Gemini which courses the student finished and what else to search for."""
    try:
        response = client.models.generate_content(
            model=GENERATE_MODEL, contents=f"{PLANNER}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")))  # keep planning fast
        return json.loads(response.text)
    except Exception:
        return {}  # planning is a bonus; fall back to a plain search


def gather_context(client, question):
    """Return (sources for the answer, courses the student says they took)."""
    plan = plan_question(client, question)
    searches = [s for s in plan.get("searches", []) if isinstance(s, str)][:EXTRA_SEARCHES]
    taken = find_codes(" ".join(map(str, plan.get("courses_taken", []))) + " " + question)
    chunks = retrieve(client, [question] + searches)
    # Course codes named in the requirement pages are candidates; look up their prerequisites.
    named = find_codes(" ".join(c["text"] for c in chunks if "/curric/" in c["url"]))
    codes = with_prereqs(taken) + [c for c in named if c not in taken][:20]
    return chunks + as_context_items(list(dict.fromkeys(codes))), taken


def generate_answer(client, question, sources, taken):
    numbered = "\n\n".join(f"[{i}] {s['title']}\n{s['text']}" for i, s in enumerate(sources, 1))
    done = f"\nThe student says they finished: {', '.join(taken)}." if taken else ""
    prompt = f"{RULES}\n\nSources:\n{numbered}\n{done}\nQuestion: {question}"
    response = client.models.generate_content(
        model=GENERATE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2, thinking_config=types.ThinkingConfig(thinking_level="LOW")))
    return response.text


def cited_sources(answer, sources):
    """Keep only the sources the answer actually cites, e.g. [2]."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
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
    question = body.question.strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Questions must be 1 to {MAX_QUESTION_CHARS} characters.")
    key = counter_key(request)
    used = usage.get(key, 0)
    if not x_gemini_key and used >= DAILY_LIMIT:
        raise HTTPException(429, "You have used today's free questions. Paste your own free "
                                 "Gemini key in the sidebar to keep going.")
    try:
        client = make_client(x_gemini_key)
        found, taken = gather_context(client, question)
        answer = generate_answer(client, question, found, taken)
    except errors.ClientError as error:
        if error.code == 429:
            raise HTTPException(429, "Gemini is busy or its rate limit was hit. Please wait a minute and try again.")
        if error.code in (400, 401, 403):
            raise HTTPException(400, "Gemini rejected the request. If you pasted your own key, check that it is correct.")
        raise HTTPException(502, "Gemini returned an error. Please try again.")
    except errors.ServerError:
        raise HTTPException(502, "Gemini is unavailable right now. Please try again.")
    if not x_gemini_key:
        usage[key] = used + 1
    remaining = None if x_gemini_key else DAILY_LIMIT - usage[key]
    return {"answer": answer, "citations": cited_sources(answer, found),
            "model": GENERATE_MODEL, "remaining": remaining}


@app.exception_handler(HTTPException)
def readable_errors(request, error):
    return JSONResponse({"message": error.detail}, status_code=error.status_code)


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
