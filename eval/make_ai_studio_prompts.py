"""Write paste-ready prompts for testing in Google AI Studio (no answer is generated here).

Run from the project root:  python eval/make_ai_studio_prompts.py
Creates eval/ai_studio/:
  system_instructions.txt   paste into the "System instructions" box
  NN_*.txt                  one per eval question; paste into the chat box (sources + question)
These are exactly what app.py sends to the model, so AI Studio answers match the live app.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import RULES, build_message, gather_context, make_client


def main():
    out = Path("eval/ai_studio")
    out.mkdir(exist_ok=True)
    (out / "system_instructions.txt").write_text(RULES + "\n")
    client = make_client()
    for n, item in enumerate(json.loads(Path("eval/questions.json").read_text()), 1):
        sources, taken = gather_context(client, item["question"])
        slug = re.sub(r"[^a-z0-9]+", "-", item["question"].lower()).strip("-")[:40]
        (out / f"{n:02d}_{slug}.txt").write_text(build_message(item["question"], sources, taken) + "\n")
        print(f"wrote {n:02d}_{slug}.txt ({len(sources)} sources)")


if __name__ == "__main__":
    main()
