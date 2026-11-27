"""Course lookup: prerequisites for named courses, read from data/courses.json."""
import json
import re
from pathlib import Path

COURSES = json.loads((Path(__file__).parent / "data/courses.json").read_text())
CODE = re.compile(r"\b([A-Za-z]{2,4})\s?(\d{1,3}[A-Za-z]{0,2})\b")
# "CSE 12 and 21, 100": a code followed by more bare numbers that reuse its department
CODE_LIST = re.compile(r"\b([A-Za-z]{2,4})\s?(\d{1,3}[A-Za-z]{0,2})((?:\s*(?:,|and|&)\s*\d{1,3}[A-Za-z]{0,2}\b)*)", re.I)
CLEAN_CODE = re.compile(r"[A-Z]{2,4} \d{1,3}[A-Z]{0,2}(?:-[A-Z])?")  # "MATH 10A-B" is a two-course sequence
BARE_NUMBER = re.compile(r"\d{1,3}[A-Z]{0,2}(?:-[A-Z])?")


def find_codes(text):
    """Catalog course codes mentioned in text, e.g. 'cse12' -> 'CSE 12'."""
    found = []
    for dept, number in CODE.findall(text):
        code = f"{dept.upper()} {number.upper()}"
        if code in COURSES and code not in found:
            found.append(code)
    return found


def courses_taken(text):
    """Like find_codes, but 'cse 12 and 21' also gives CSE 21."""
    found = []
    for dept, number, rest in CODE_LIST.findall(text):
        numbers = [number] + re.findall(r"\d{1,3}[A-Za-z]{0,2}", rest)
        for n in numbers:
            code = f"{dept.upper()} {n.upper()}"
            if code in COURSES and code not in found:
                found.append(code)
    return found


def prereq_groups(code):
    """Read 'A or B and C' the way the catalog means it: (A or B) AND C.
    Returns a list of groups, each a list of alternatives. None if the text is too complex."""
    text = COURSES[code]["prereq"].split(";")[0].strip()
    groups, dept = [], ""
    for part in re.split(r"\s+and\s+", text) if text else []:
        alternatives = []
        for a in re.split(r"\s+or\s+|,\s*", part):
            a = a.strip(" ()")
            if BARE_NUMBER.fullmatch(a) and dept:  # "MATH 20A or 20B": 20B reuses MATH
                a = f"{dept} {a}"
            if not CLEAN_CODE.fullmatch(a):
                return None
            dept = a.split()[0]
            alternatives.append(a)
        groups.append(alternatives)
    return groups


def implied_by(taken):
    """Courses the student must have finished, and groups where at least one was finished.
    Example: CSE 12 needs (CSE 8B or CSE 11), so one of those two is done."""
    done, groups, queue = set(taken), [], list(taken)
    while queue:
        for alternatives in prereq_groups(queue.pop()) or []:
            groups.append(set(alternatives))
            if len(alternatives) == 1 and alternatives[0] not in done:
                done.add(alternatives[0])
                queue.append(alternatives[0])
    return done, groups


def readiness(code, done, groups):
    """Say whether the student has met a course's prerequisites."""
    if any(code in g for g in groups):
        return "Probably already finished (it is one of the options for a course the student took)"
    needed = prereq_groups(code)
    if needed is None:
        return "Check the prerequisite text yourself (too complex to check automatically)"
    missing = []
    for alternatives in needed:
        met = any(a in done for a in alternatives) or any(g <= set(alternatives) for g in groups)
        if not met:
            missing.append(" or ".join(alternatives))
    return "Ready: all prerequisites met" if not missing else "Missing: " + "; ".join(missing)


def prereq_sentence(code):
    """Prerequisites in words, so 'any one of' and 'all of' cannot be misread."""
    raw = COURSES[code]["prereq"]
    groups = prereq_groups(code) if ";" not in raw else None
    if not raw:
        return "none listed"
    if groups is None:
        return raw  # too complex to restate safely, keep the catalog's own wording
    parts = [g[0] if len(g) == 1 else "any one of " + ", ".join(g) for g in groups]
    if len(parts) == 1:
        return parts[0]
    return "all of these are required: " + "; ".join(f"({n}) {p}" for n, p in enumerate(parts, 1))


def describe(code, status=""):
    course = COURSES[code]
    line = f"{code}. {course['title']} ({course['units']} units). Prerequisites: {prereq_sentence(code)}"
    return f"{line}. {status}" if status else line


def as_context_items(codes, done=None, groups=None):
    """Group course entries by catalog page so each page can be cited once.
    Courses not in `done` also get a readiness note."""
    pages = {}
    for code in codes:
        status = readiness(code, done, groups) if done is not None and code not in done else ""
        page = pages.setdefault(COURSES[code]["url"], {"title": COURSES[code]["page"], "lines": []})
        page["lines"].append(describe(code, status))
    return [{"title": p["title"], "url": url, "text": "\n".join(p["lines"])} for url, p in pages.items()]
