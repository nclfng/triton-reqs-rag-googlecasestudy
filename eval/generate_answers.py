"""Answer every question in eval/questions.json with the app's own retrieval and rules.

Run from the project root:  python eval/generate_answers.py
Writes eval/our_answers.json, which compare.ipynb reads. Uses your Gemini key from .env.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from google.genai import errors

from app import answer_config, build_message, cited_sources, gather_context, make_client, stream_text


def answer(client, question):
    sources, taken = gather_context(client, question)
    used = {}
    for attempt in range(4):  # the free tier limits requests per minute; wait and retry
        try:
            text = "".join(stream_text(client, build_message(question, sources, taken), used,
                                      answer_config(question)))
            break
        except errors.ClientError as error:
            if error.code != 429 or attempt == 3:
                raise
            time.sleep(45)
    return {"answer": text.strip(), "model": used.get("model"),
            "sources": [c["url"] for c in cited_sources(text, sources)]}


def main():
    client = make_client()
    results = []
    for item in json.loads(Path("eval/questions.json").read_text()):
        start = time.time()
        result = {"question": item["question"], **answer(client, item["question"])}
        result["seconds"] = round(time.time() - start, 1)
        results.append(result)
        print(f"{result['seconds']:>5}s  {item['question'][:70]}")
    Path("eval/our_answers.json").write_text(json.dumps(results, indent=1))
    print("saved eval/our_answers.json")


if __name__ == "__main__":
    main()
