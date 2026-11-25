"""Triton Requirements Guide - FastAPI app."""
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

# ---------- Config: change settings here ----------
EMBED_MODEL = "gemini-embedding-001"
GENERATE_MODEL = "gemini-3.5-flash"   # one constant; 2.5 models are closed to new users, 3.7+ were overloaded
CHUNK_WORDS = 400                     # target words per chunk
CHUNK_OVERLAP = 40                    # words repeated between neighbouring chunks
TOP_K = 8                             # chunks sent to Gemini (more pages are indexed now)
DAILY_LIMIT = 10                      # questions per IP per day on the server key
MAX_QUESTION_CHARS = 400
ANSWER_MAX_WORDS = 300                # raised from 150 so elective lists fit
# --------------------------------------------------

RULES = f"""You answer student questions about UCSD requirements using ONLY the numbered chunks below.
- If the chunks do not contain the answer, say the catalog pages indexed here do not cover it and suggest checking with an academic advisor.
- Keep the answer under {ANSWER_MAX_WORDS} words, in plain language. Plain text only: no markdown, no asterisks, no bullet symbols.
- Cite sources like [1], [2], matching the chunk numbers.
- Never invent course codes, unit counts, or policies.
- End with one line telling the student to confirm with their advisor."""

load_dotenv()
app = FastAPI()
usage = {}  # (ip, date) -> questions asked with the server key; resets when the server restarts


def load_index():
    chunks = json.loads(Path("data/index.json").read_text())
    return chunks, np.load("data/vectors.npy")


CHUNKS, VECTORS = load_index()


def make_client(user_key=None):
    return genai.Client(api_key=user_key or os.environ["GEMINI_API_KEY"])


def retrieve(client, question):
    """Embed the question and return the TOP_K most similar chunks (with their numbers)."""
    response = client.models.embed_content(
        model=EMBED_MODEL, contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
    q = np.array(response.embeddings[0].values, dtype="float32")
    q /= np.linalg.norm(q)
    scores = VECTORS @ q  # vectors are unit length, so this is cosine similarity
    best = np.argsort(scores)[::-1][:TOP_K]
    return [CHUNKS[i] for i in best]


def generate_answer(client, question, chunks):
    numbered = "\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(chunks, 1))
    prompt = f"{RULES}\n\nChunks:\n{numbered}\n\nQuestion: {question}"
    response = client.models.generate_content(
        model=GENERATE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2))
    return response.text


def cited_sources(answer, chunks):
    """Keep only the chunks the answer actually cites, e.g. [2]."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [{"n": n, "title": chunks[n - 1]["title"], "url": chunks[n - 1]["url"],
             "excerpt": chunks[n - 1]["text"][:500]} for n in numbers if 1 <= n <= len(chunks)]


def client_ip(request):
    forwarded = request.headers.get("x-forwarded-for")  # set by Render / Hugging Face proxies
    return forwarded.split(",")[0].strip() if forwarded else request.client.host


class Question(BaseModel):
    question: str


@app.get("/api/sources")
def sources():
    pages = {c["url"]: {"title": c["title"], "url": c["url"], "scraped": c["scraped"]} for c in CHUNKS}
    return list(pages.values())


@app.post("/api/ask")
def ask(body: Question, request: Request, x_gemini_key: str | None = Header(default=None)):
    question = body.question.strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Questions must be 1 to {MAX_QUESTION_CHARS} characters.")
    key = (client_ip(request), date.today())
    used = usage.get(key, 0)
    if not x_gemini_key and used >= DAILY_LIMIT:
        raise HTTPException(429, "You have used today's free questions. Paste your own free "
                                 "Gemini key in the sidebar to keep going.")
    try:
        client = make_client(x_gemini_key)
        chunks = retrieve(client, question)
        answer = generate_answer(client, question, chunks)
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
    return {"answer": answer, "citations": cited_sources(answer, chunks),
            "model": GENERATE_MODEL, "remaining": remaining}


@app.exception_handler(HTTPException)
def readable_errors(request, error):
    return JSONResponse({"message": error.detail}, status_code=error.status_code)


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
