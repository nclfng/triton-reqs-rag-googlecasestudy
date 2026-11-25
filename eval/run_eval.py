"""Check retrieval: does the expected source page show up in the top 5 chunks?

Run from the project root:  python eval/run_eval.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import make_client, retrieve  # the same retrieval code the app uses


def main():
    questions = json.loads(Path("eval/questions.json").read_text())
    client = make_client()
    hits = 0
    print(f"{'#':<3} {'result':<6} {'rank':<5} question")
    for i, item in enumerate(questions, 1):
        chunks = retrieve(client, item["question"])
        urls = [c["url"] for c in chunks]
        ranks = [r for r, u in enumerate(urls, 1) if item["expected_url_substring"] in u]
        hits += bool(ranks)
        rank = ranks[0] if ranks else "-"
        print(f"{i:<3} {'PASS' if ranks else 'FAIL':<6} {rank:<5} {item['question'][:70]}")
        time.sleep(1)  # stay under the free-tier embedding rate limit
    print(f"\nExpected source in top 5: {hits}/{len(questions)} = {hits / len(questions):.0%}")


if __name__ == "__main__":
    main()
