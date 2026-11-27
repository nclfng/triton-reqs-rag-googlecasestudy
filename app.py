"""Triton Requirements Guide - FastAPI app."""
import hashlib
import json
import os
import re
from collections import OrderedDict
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
GENERATE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]  # tried in order; the next one is used if a model is out of free quota
CHUNK_WORDS = 400                     # target words per chunk
CHUNK_OVERLAP = 40                    # words repeated between neighbouring chunks
TOP_K = 8                             # chunks kept for the question itself
MAX_CONTEXT_CHUNKS = 12               # most chunks sent to Gemini
DAILY_LIMIT = 10                      # questions per IP per day on the server key
MAX_QUESTION_CHARS = 400
CACHE_SIZE = 200                      # answers kept in memory for repeat questions
ANSWER_MAX_WORDS = 300                # raised from 150 so elective lists fit
FEEDBACK_FORM_URL = ""                # paste your Google Form link here to show the feedback link
# --------------------------------------------------

RULES = f"""You answer student questions about UCSD requirements using ONLY the numbered sources in the message.
- If the sources do not contain the answer, say the catalog pages indexed here do not cover it and suggest checking with an academic advisor.
- Keep the answer under {ANSWER_MAX_WORDS} words, in plain language. Plain text only: no markdown, no asterisks, no bullet symbols.
- Cite sources like [1], [2], matching the source numbers. Cite only the source that states the fact, and put the citation after that fact.
- Never invent course codes, unit counts, or policies. Copy numbers exactly as the source states them.
- Totals: when a source gives a total with its parts, state the total and then each part with its units, including small parts such as a 4-unit ethics requirement.
- Prerequisites: the sources say "any one of" when a student needs just one course from a list, and "all of these are required" when every item is needed. Report them the same way and never turn an "any one of" list into a plain list. Do not mention enrollment restrictions (such as which majors may enroll) unless the student asks about them.
- If asked what to take next: first list the major's required courses marked Ready, in a sensible order (programming and math first). Name required courses that are Missing a prerequisite only as later steps, and if the missing prerequisite is itself Ready, say to take that first. Never present a Missing course as one to take now. A student who lists no finished courses (for example a first-year) has finished nothing, so recommend only courses with no unmet prerequisites.
- Mention electives only briefly (at most three) and only courses from the major's own department or closely related ones (for example COGS or DSC for AI). Never list courses from unrelated departments just because they appear in an elective list, and never suggest introductory courses below the level of what the student already finished.
- Do not say when courses are offered. The catalog does not list that; the department website has sample plans.
- Keep it short: a few short paragraphs, then end with one line telling the student to confirm with their advisor."""

FINISHED = re.compile(r"\b(took|taken|finished|completed|passed|already|have done|done with)\b", re.I)
# Only added to the message when the sources contain readiness notes (planning questions).
# Kept out of RULES: with these lines in the system instruction the model refused simple
# questions, because it looked for notes that were not there.
READINESS_NOTES = """Notes on the course entries: each ends with a readiness note. "Ready" means the student has met the prerequisites, "Missing" names what is still needed, and a note to check the text means it was too complex to check automatically. A line starting "Because of prerequisites" lists courses the student has also finished. If the student names finished courses, open the answer with one sentence saying what those imply, for example: "Since you finished CSE 12 and CSE 21, I counted CSE 8B or CSE 11 and CSE 20 or MATH 15A as done." Do not write generic assumption sentences such as "we assume none"."""

PLANNING = re.compile(r"\b(next|take|taking|first[- ]year|freshman|sophomore|junior|senior|recommend\w*|plan|schedule|should i)\b", re.I)

load_dotenv()
app = FastAPI()
answer_cache = OrderedDict()  # hash of question -> finished answer. Memory only, never written to disk.
usage = {}  # (hashed IP, date) -> questions asked on the server key. Memory only, never written to disk.

BASE = Path(__file__).parent  # file paths work no matter where the server starts
CHUNKS = json.loads((BASE / "data/index.json").read_text())
VECTORS = np.load(BASE / "data/vectors.npy")


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


def required_codes(chunks):
    """Course codes from requirement items, skipping "choose from this list" and elective items."""
    found = []
    for chunk in chunks:
        for item in re.split(r"\s(?=\d+\.\s+\*\*)", chunk["text"]):  # numbered items: "1. **_Math:_** ..."
            if not re.search(r"chosen from|choose|select|elective", item, re.I):
                found += find_codes(re.sub(r"\(\s*or [^)]*\)", "", item))  # skip "(or MATH 15A ...)" substitutes
    return list(dict.fromkeys(found))


def gather_context(client, question):
    """Return (sources for the answer, courses the student says they took)."""
    # Only courses the student says they finished count as done ("I took CSE 12"), not every course named
    taken = courses_taken(question) if FINISHED.search(question) else []
    planning = bool(taken) or bool(PLANNING.search(question))  # "what should I take" style questions
    queries = [question]
    if planning:  # search each part of the major's requirements separately
        queries += [f"{part}: {question}" for part in
                    ("lower-division requirements", "upper-division core courses", "electives")]
    chunks = retrieve(client, queries)
    if not planning:
        # A question about named courses ("prerequisites for PHYS 2A") is answered from the course
        # table first, so the answer cites the course page and not an unrelated program page.
        return as_context_items(find_codes(question)) + chunks, taken
    # Candidates: courses named in the best-matching program's requirements (lower-division first).
    program_chunks = [c for c in chunks if "/curric/" in c["url"] and c.get("heading")]
    if program_chunks:
        program = program_chunks[0]["heading"].split(" / ")[0]
        program_chunks = sorted((c for c in program_chunks if c["heading"].startswith(program)),
                                key=lambda c: c["heading"])
    named = required_codes(program_chunks)  # required courses only; electives stay in the text sources
    done, groups = implied_by(taken)
    candidates = [c for c in named if c not in done][:80]
    return chunks + as_context_items(list(done) + candidates, done, groups), taken


def build_message(question, sources, taken):
    """The user message: numbered sources, what the student finished, and the question.
    The rules go in the model's system instruction (see answer_config)."""
    numbered = "\n\n".join(f"[{i}] {s['title']} - {s['heading']}\n{s['text']}" if s.get("heading")
                            else f"[{i}] {s['title']}\n{s['text']}" for i, s in enumerate(sources, 1))
    finished = ""
    if taken:
        done, groups = implied_by(taken)
        also = sorted(done - set(taken))
        finished = f"\nThe student says they finished: {', '.join(taken)}."
        if also or groups:
            options = "; ".join("one of " + " or ".join(sorted(g)) for g in groups if len(g) > 1)
            finished += " Because of prerequisites, they have also finished: " + "; ".join(
                part for part in (", ".join(also), options) if part) + "."
    notes = f"\n{READINESS_NOTES}\n" if re.search(r"\b(Ready|Missing)\b[: ]", numbered) else ""
    return f"Sources:\n{numbered}\n{notes}{finished}\nQuestion: {question}"


COUNTING = re.compile(r"\bhow many\b|\btotal\b|\bunits\b", re.I)


def answer_config(question):
    """Fast answers by default. Questions that add up units get a little more thinking, because
    the fastest setting tends to drop one part of a total (for example the 4-unit ethics item)."""
    level = "MEDIUM" if COUNTING.search(question) else "MINIMAL"
    return types.GenerateContentConfig(system_instruction=RULES, temperature=0.2,
                                       thinking_config=types.ThinkingConfig(thinking_level=level))


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


def explain(error, user_key=None):
    """Turn a Gemini error into a short readable message."""
    if isinstance(error, errors.ClientError) and error.code == 429:
        if "per day" in str(error).lower():  # the daily free quota, not a short rate limit
            if user_key:
                return "Your Gemini key has used up its free quota for today. Try again tomorrow."
            return ("The project owner's free Gemini quota has run out for today. This tool runs on "
                    "Google's free API tier, which has a small daily limit. Please try again tomorrow, "
                    "or paste your own free Gemini key in the sidebar to keep going.")
        return "Gemini is busy right now. Please wait a minute and try again."
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


def stream_text(client, prompt, used, config):
    """Yield the answer text. If a model is out of free quota, switch to the next one."""
    for model in GENERATE_MODELS:
        try:
            for part in client.models.generate_content_stream(model=model, contents=prompt, config=config):
                used["model"] = model
                yield part.text or ""
            return
        except errors.ClientError as error:
            if error.code != 429 or model == GENERATE_MODELS[-1]:
                raise


def cache_key(question):
    """Same question (ignoring case and spacing) gives the same key. Only the hash is kept."""
    return hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()


def cached_reply(question, key, user_key):
    """Return a saved answer for this question, or None. Cached answers cost no Gemini quota."""
    reply = answer_cache.get(cache_key(question))
    if reply:
        remaining = None if user_key else DAILY_LIMIT - usage.get(key, 0)
        return {**reply, "remaining": remaining}


def finish(question, answer, sources, key, user_key, model):
    """Count the question (server key only), save the answer in the cache, build the response body."""
    if not user_key:
        usage[key] = usage.get(key, 0) + 1
    reply = {"answer": answer, "citations": cited_sources(answer, sources), "model": model}
    answer_cache[cache_key(question)] = reply
    if len(answer_cache) > CACHE_SIZE:
        answer_cache.popitem(last=False)  # drop the oldest
    return {**reply, "remaining": None if user_key else DAILY_LIMIT - usage[key]}


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
    if reply := cached_reply(question, key, x_gemini_key):
        return reply
    try:
        client = make_client(x_gemini_key)
        found, taken = gather_context(client, question)
        used = {}
        answer = "".join(stream_text(client, build_message(question, found, taken), used, answer_config(question)))
    except errors.APIError as error:
        raise HTTPException(429 if error.code == 429 else 502, explain(error, x_gemini_key))
    return finish(question, answer, found, key, x_gemini_key, used.get("model", GENERATE_MODELS[0]))


@app.post("/api/ask-stream")
def ask_stream(body: Question, request: Request, x_gemini_key: str | None = Header(default=None)):
    """Same as /api/ask, but sends the answer as it is written (one JSON object per line)."""
    question, key = check_request(body, request, x_gemini_key)
    if reply := cached_reply(question, key, x_gemini_key):
        text = json.dumps({"delta": reply["answer"]}) + "\n" + json.dumps({"done": reply}) + "\n"
        return StreamingResponse(iter([text]), media_type="application/x-ndjson")
    try:
        client = make_client(x_gemini_key)
        found, taken = gather_context(client, question)
    except errors.APIError as error:
        raise HTTPException(429 if error.code == 429 else 502, explain(error, x_gemini_key))

    def events():
        answer, used = "", {}
        try:
            for text in stream_text(client, build_message(question, found, taken), used, answer_config(question)):
                answer += text
                yield json.dumps({"delta": text}) + "\n"
        except errors.APIError as error:
            yield json.dumps({"error": explain(error, x_gemini_key)}) + "\n"
            return
        yield json.dumps({"done": finish(question, answer, found, key, x_gemini_key, used.get("model", GENERATE_MODELS[0]))}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.exception_handler(HTTPException)
def readable_errors(request, error):
    return JSONResponse({"message": error.detail}, status_code=error.status_code)


@app.get("/")
def home():
    return FileResponse(BASE / "static/index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
