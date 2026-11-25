# Triton Requirements Guide

Triton Requirements RAG for Google Finals Case Study.

A small question-answering tool for UCSD general college requirements (Warren College) and the BS in Artificial Intelligence. A student asks a question and gets a short answer with numbered citations linking to the catalog page it came from. If the indexed pages do not contain the answer, it says so instead of guessing.

## How it works

1. `scrape.py` reads `urls.txt` and saves each public catalog page as markdown (via Context.dev) in `data/raw/`. Pages already saved are skipped.
2. `build_index.py` splits the pages by heading into chunks of about 400 words, embeds each chunk with Gemini, and saves `data/index.json` and `data/vectors.npy`.
3. `app.py` embeds the question, finds the 8 closest chunks by cosine similarity (plain numpy, no vector database), and asks a Gemini Flash model to answer only from those chunks, citing them as [1], [2].
4. The page in `static/` is plain HTML, CSS and JS.

Indexed pages (about 50, listed in `urls.txt`): university-wide degree and graduation requirements, all seven undergraduate colleges and their writing/core sequences, the Jacobs School of Engineering and its majors (BENG, CSE including the BS in Artificial Intelligence, ECE, MAE, NANO, SE), Data Science, Cognitive Science, Math, Physics, Chemistry, Biology, and course descriptions for the engineering and prerequisite departments.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add GEMINI_API_KEY and CONTEXT_DEV_API_KEY
```

To rebuild the data (optional, the built index is committed):

```
python scrape.py
python build_index.py
```

The free Gemini tier limits embeddings to about 100 chunks a minute, so `build_index.py` pauses and retries when it hits the limit.

## Run

```
uvicorn app:app --reload
```

Open http://localhost:8000. Settings (models, chunk size, top-k, daily limit) are at the top of `app.py`.

## Usage limits

- With the server key: 10 questions per IP per day, tracked in memory (resets when the server restarts).
- With a key pasted in the sidebar: no cap. It is sent in an `X-Gemini-Key` header for that request only and is never logged or stored.
- Questions are limited to 400 characters.

## Evaluation

Fill `eval/questions.json` with `{question, expected_url_substring}` pairs, then run `python eval/run_eval.py`. It reports how often the expected page appears in the top 8 retrieved chunks.

## Deploy

The built `data/index.json` and `data/vectors.npy` are committed, so the deployed app does not scrape or embed anything. No billing account is needed.

**Hugging Face Spaces:** create a new Space with the Docker SDK, push this repo to it, and add `GEMINI_API_KEY` under Settings, Secrets.

**Render (free tier):** create a Web Service from this repo with the Docker runtime, and add `GEMINI_API_KEY` as an environment variable.

## Limitations

- Only the pages in `urls.txt` are indexed, so non-engineering majors and departments such as Economics or Psychology are not covered, and course descriptions exist only for the listed departments.
- Answers are capped at 150 words, so a long list of electives may be summarized rather than complete.
- Catalog pages change. Re-run the scrape and index to refresh, and always confirm with an academic advisor.
- The daily limit is per server process and resets on restart. IPs behind a shared proxy may share a count.
- Retrieval uses fixed-size chunks, so a requirement split across chunks can be missed.
- Answers come from a language model and can be wrong even with sources.
