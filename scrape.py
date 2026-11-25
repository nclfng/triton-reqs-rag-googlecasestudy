"""Scrape every URL in urls.txt to markdown with Context.dev and save to data/raw/."""
import hashlib
import os
from datetime import date
from pathlib import Path

from context.dev import ContextDev
from dotenv import load_dotenv

RAW_DIR = Path("data/raw")


def read_urls(path="urls.txt"):
    """Return non-empty lines that are not comments."""
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def filename_for(url):
    """Stable file name from the URL, so we can tell if a page was already scraped."""
    return hashlib.md5(url.encode()).hexdigest()[:12] + ".md"


def scrape_page(client, url):
    """Ask Context.dev for the page as markdown. Returns (title, markdown)."""
    result = client.web.scrape(url=url, formats={"markdown": True})
    if not result.markdown.success:
        raise RuntimeError(f"markdown failed: {result.markdown.error_code}")
    title = (result.metadata.title if result.metadata else None) or url
    if "/courses/" in url:  # same title as the program page, so tell them apart
        title += " - Course Descriptions"
    return title, result.markdown.data


def save_page(path, url, title, markdown):
    """Header lines first, then a blank line, then the page text."""
    header = f"SOURCE_URL: {url}\nTITLE: {title}\nSCRAPED: {date.today()}\n\n"
    path.write_text(header + markdown)


def main():
    load_dotenv()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = ContextDev(api_key=os.environ["CONTEXT_DEV_API_KEY"])
    for url in read_urls():
        path = RAW_DIR / filename_for(url)
        if path.exists():
            print(f"skip (already saved): {url}")
            continue
        try:
            title, markdown = scrape_page(client, url)
            save_page(path, url, title, markdown)
            print(f"saved: {title}")
        except Exception as error:
            print(f"FAILED {url}: {error}")


if __name__ == "__main__":
    main()
