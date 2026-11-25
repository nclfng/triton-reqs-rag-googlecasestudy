"""Chunk data/raw/*.md by heading, embed each chunk with Gemini, save the index."""
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app import CHUNK_OVERLAP, CHUNK_WORDS, EMBED_MODEL

BATCH_SIZE = 50  # chunks per embedding request


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


def embed(client, texts):
    """Embed texts in batches. Returns a list of vectors."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        response = client.models.embed_content(
            model=EMBED_MODEL, contents=texts[i:i + BATCH_SIZE],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
        vectors += [e.values for e in response.embeddings]
        print(f"embedded {len(vectors)}/{len(texts)}")
    return vectors


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
