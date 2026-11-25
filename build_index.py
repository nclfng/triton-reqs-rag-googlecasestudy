"""Chunk data/raw/*.md by heading, embed each chunk with Gemini, save the index."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from app import CHUNK_OVERLAP, CHUNK_WORDS, EMBED_MODEL

BATCH_SIZE = 20  # chunks per request; the free tier also caps tokens per minute (30,000)


def read_page(path):
    """Split a saved file into its header fields and the markdown body."""
    head, _, body = path.read_text().partition("\n\n")
    fields = dict(line.split(": ", 1) for line in head.splitlines())
    return fields, body


def split_by_heading(body):
    """Return (heading, text) sections, starting a new one at every # heading."""
    sections, heading, lines = [], "", []
    for line in body.splitlines():
        if re.match(r"#{1,6} ", line):
            sections.append((heading, "\n".join(lines)))
            heading, lines = line.lstrip("# ").strip(), []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))
    return [(h, t) for h, t in sections if t.strip()]


def split_long_text(text):
    """Cut a long section into ~CHUNK_WORDS pieces that overlap a little."""
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)]


def chunk_page(fields, body):
    chunks = []
    for heading, text in split_by_heading(body):
        for piece in split_long_text(text):
            chunks.append({
                "title": fields["TITLE"], "url": fields["SOURCE_URL"],
                "scraped": fields["SCRAPED"], "heading": heading, "text": piece,
            })
    return chunks


CACHE_FILE = Path("data/embed_cache.json")  # text hash -> vector, so a rerun never repeats work


def embed_batch(client, batch):
    """Embed one batch, waiting out Gemini's free-tier limits (per minute and per day)."""
    while True:
        try:
            return client.models.embed_content(
                model=EMBED_MODEL, contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
        except errors.ClientError as error:
            if error.code != 429:
                raise
            per_day = "per day" in str(error)
            wait = 1800 if per_day else 40
            print(f"{'daily' if per_day else 'per-minute'} limit hit, waiting {wait}s...", flush=True)
            time.sleep(wait)


def embed(client, texts):
    """Embed texts in batches, saving each finished batch to the cache. Returns vectors in order."""
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"{len(cache)} cached, {len(todo)} to embed", flush=True)
    for start in range(0, len(todo), BATCH_SIZE):
        ids = todo[start:start + BATCH_SIZE]
        response = embed_batch(client, [texts[i] for i in ids])
        for i, e in zip(ids, response.embeddings):
            cache[keys[i]] = e.values
        CACHE_FILE.write_text(json.dumps(cache))
        print(f"embedded {start + len(ids)}/{len(todo)}", flush=True)
    return [cache[k] for k in keys]


def main():
    load_dotenv()
    chunks = []
    for path in sorted(Path("data/raw").glob("*.md")):
        chunks += chunk_page(*read_page(path))
    print(f"{len(chunks)} chunks")
    # Put the heading in front so the embedding knows what the text is about.
    texts = [f"{c['title']} - {c['heading']}\n{c['text']}" for c in chunks]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    vectors = np.array(embed(client, texts), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # unit length: dot product = cosine
    np.save("data/vectors.npy", vectors)
    Path("data/index.json").write_text(json.dumps(chunks, indent=1))
    print("saved data/index.json and data/vectors.npy")


if __name__ == "__main__":
    main()
