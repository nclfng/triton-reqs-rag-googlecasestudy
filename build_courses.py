"""Read the saved course-description pages and write data/courses.json.

This lets the app look up one course's prerequisites exactly, with no embedding calls.
Run it after scrape.py (and again after adding pages). Pages in data/raw_deferred are included.
"""
import json
import re
from pathlib import Path

# A course starts with a line like "CSE 12. Basic Data Structures (4)"
ENTRY = re.compile(r"^([A-Z]{2,4}) (\d{1,3}[A-Z]{0,2})\. (.+?) \(([^)]*)\)( \*\*.*)?\s*$")
PREREQ = re.compile(r"\*\*_+Prerequisites:_+\*\*\s*(.*)")


def clean_prereq(text):
    """Keep just the prerequisites. Drops enrollment restrictions ("restricted to undergraduates",
    "restricted to CS25 majors") and credit-exclusion notes ("may not receive credit"): neither
    is a course the student has to take first."""
    sentence = text.split(". ")[0].rstrip(".").strip()
    clauses = [c.strip() for c in sentence.split(";")]
    kept = [c for c in clauses if c and not c.lower().startswith("restricted")
            and "may not receive credit" not in c.lower()]
    return "; ".join(kept)


def parse_page(path):
    """Return {code: course info} for one saved course-description page."""
    head, _, body = path.read_text().partition("\n\n")
    fields = dict(line.split(": ", 1) for line in head.splitlines())
    courses, code = {}, None
    for line in body.splitlines():
        match = ENTRY.match(line.replace("\xa0", " ").strip())
        if match:
            code = f"{match.group(1)} {match.group(2)}"
            courses[code] = {"title": match.group(3), "units": match.group(4), "prereq": "",
                             "url": fields["SOURCE_URL"], "page": fields["TITLE"]}
        elif code and (found := PREREQ.search(line)):
            courses[code]["prereq"] = clean_prereq(found.group(1))
    return courses


def main():
    courses = {}
    for folder in ("data/raw", "data/raw_deferred"):
        for path in sorted(Path(folder).glob("*.md")):
            if "SOURCE_URL: https://catalog.ucsd.edu/courses/" in path.read_text()[:300]:
                courses.update(parse_page(path))
    Path("data/courses.json").write_text(json.dumps(courses, indent=1))
    print(f"saved {len(courses)} courses to data/courses.json")


if __name__ == "__main__":
    main()
