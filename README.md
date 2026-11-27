# Triton Requirements Guide

A requirements lookup for UC San Diego students. Ask a question about college general education, university graduation requirements, or an engineering major, and get a short answer with numbered citations that link to the catalog page it came from. If the indexed pages do not contain the answer, it says so.

Example questions:

- How many upper-division electives do I need for the BS in Artificial Intelligence?
- I took CSE 12 and 21. What should I take next as an AI major?
- What are the writing requirements at Warren College?

## How it works

1. **Scrape.** `scrape.py` reads `urls.txt` and saves each public catalog page as markdown (using Context.dev) in `data/raw/`. Pages already saved are skipped.
2. **Index.** `build_index.py` splits the pages by heading into chunks of about 400 words, embeds them with Gemini, and writes `data/index.json` and `data/vectors.npy`.
3. **Course table.** `build_courses.py` reads the course description pages and writes `data/courses.json`, a table of every course with its title, units, and prerequisites.
4. **Answer.** For each question, `app.py`:
   - finds the courses the student says they finished (for example, "CSE 12 and 21");
   - searches the chunks by cosine similarity (numpy, no vector database), including separate searches for the major's lower-division, upper-division, and elective requirements;
   - treats a finished course's prerequisites as finished too, so CSE 12 implies CSE 8B or CSE 11;
   - takes the courses named in the major's requirements and checks each one's prerequisites against what the student has finished, marking it Ready or Missing;
   - sends the sources to Gemini, which writes the answer from them and cites them. The answer streams to the page as it is written.

Prerequisites are checked in code, not by the model. The catalog writes "A or B and C or D" to mean (A or B) and (C or D), and `courses.py` reads it that way.

The page in `static/` is plain HTML, CSS, and JavaScript.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add `GEMINI_API_KEY` and `CONTEXT_DEV_API_KEY` to `.env`. The built index is committed, so you only need the Gemini key to run the app.

To rebuild the data after changing `urls.txt`:

```
python scrape.py
python build_index.py
python build_courses.py
```

The free Gemini tier allows 1,000 embedding requests a day and 30,000 tokens a minute. `build_index.py` waits when it hits either limit and saves its progress in `data/embed_cache.json`, so it can be rerun and will continue where it stopped.

## Run

```
uvicorn app:app --no-access-log
```

Open http://localhost:8000. Settings (models, chunk size, number of sources, daily limit, answer length, feedback form link) are at the top of `app.py`.

## Privacy and limits

- Questions and IP addresses are never written to disk. The daily counter is held in memory as a hashed IP and resets when the server restarts. Run uvicorn with `--no-access-log` so IPs do not appear in server logs.
- Finished answers are cached in memory (up to 200, keyed by a hash of the question) so repeat questions are instant and use no Gemini quota. The cache is never written to disk and clears on restart.
- With the server key: 10 questions per IP per day. Cached answers do not count.
- With a key pasted into the sidebar: no cap. It is sent in an `X-Gemini-Key` header for that request only and is not stored or logged.
- Questions are limited to 400 characters.

## Evaluation

Put `{question, expected_url_substring}` pairs in `eval/questions.json` and run:

```
python eval/run_eval.py
```

It prints whether the expected page was among the sources sent to Gemini for each question, and a pass rate.

## Deploy

The built data files are committed, so the deployed app does not scrape or embed anything and needs no billing account.

- **Hugging Face Spaces:** create a Docker Space from this repo and add `GEMINI_API_KEY` as a secret.
- **Render (free tier):** create a Web Service with the Docker runtime and set `GEMINI_API_KEY` as an environment variable.

## Limitations

- Only the pages in `urls.txt` are indexed. Physics, Chemistry, Cognitive Science, Biology, Admission, and some course description pages are scraped but not yet indexed (see the end of `urls.txt`).
- The catalog does not say when courses are offered, and the sample four-year plans live on department websites, so recommendations are based on requirements and prerequisites only.
- Prerequisites are checked automatically only when they list plain course codes. Conditions like test scores or minimum grades are passed to the model as text and flagged for the student to check.
- The free Gemini tier has a low daily request limit for generation (as low as 20 a day on some models). If the limit is hit, the app shows a message to try again later or paste a personal key.
- Answers come from a language model and can be wrong. Confirm with an academic advisor.
