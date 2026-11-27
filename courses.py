"""Course lookup: exact prerequisites for named courses, read from data/courses.json."""
import json
import re
from pathlib import Path

COURSES = json.loads(Path("data/courses.json").read_text())
CODE = re.compile(r"\b([A-Za-z]{2,4})\s?(\d{1,3}[A-Za-z]{0,2})\b")


def find_codes(text):
    """Catalog course codes mentioned in text, e.g. 'cse12' -> 'CSE 12'."""
    found = []
    for dept, number in CODE.findall(text):
        code = f"{dept.upper()} {number.upper()}"
        if code in COURSES and code not in found:
            found.append(code)
    return found


def with_prereqs(codes, depth=2):
    """The courses plus the courses their prerequisites name, a few levels deep."""
    seen, level = list(codes), list(codes)
    for _ in range(depth):
        level = [p for c in level for p in find_codes(COURSES[c]["prereq"]) if p not in seen]
        level = list(dict.fromkeys(level))
        seen += level
    return seen


def describe(code):
    course = COURSES[code]
    prereq = course["prereq"] or "none listed"
    return f"{code}. {course['title']} ({course['units']} units). Prerequisites: {prereq}"


def as_context_items(codes):
    """Group course entries by catalog page so each page can be cited once."""
    pages = {}
    for code in codes:
        page = pages.setdefault(COURSES[code]["url"], {"title": COURSES[code]["page"], "lines": []})
        page["lines"].append(describe(code))
    return [{"title": p["title"], "url": url, "text": "\n".join(p["lines"])} for url, p in pages.items()]
