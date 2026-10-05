"""Check that every relative link, image and #anchor in the Markdown files resolves.

    uv run python scripts/check_links.py

External URLs are not fetched. Links inside fenced code blocks are ignored.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SKIP = ("http://", "https://", "mailto:", "tel:")


def markdown_files(root):
    names = [root / "README.md", root / "DATA_ATTRIBUTION.md"]
    found = [p for p in names if p.exists()]
    for folder in ("docs", "results"):
        found += sorted((root / folder).rglob("*.md"))
    return found


def slug(text):
    """GitHub-style anchor: lowercase, drop punctuation, spaces to hyphens."""
    text = re.sub(r"`|\*|_(?=\S)|(?<=\S)_", "", text.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def anchors(path):
    seen, out, fenced = {}, set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        match = None if fenced else HEADING.match(line)
        if match:
            base = slug(match.group(2))
            count = seen.get(base, 0)
            seen[base] = count + 1
            out.add(base if count == 0 else f"{base}-{count}")
    return out


def links(path):
    fenced = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for match in LINK.finditer(re.sub(r"`[^`]*`", "", line)):
            yield number, match.group(1)


def check(root):
    problems, cache = [], {}
    for path in markdown_files(root):
        for number, target in links(path):
            if target.startswith(SKIP) or target.startswith("#") and not target[1:]:
                continue
            file_part, _, fragment = target.partition("#")
            destination = path if not file_part else (path.parent / unquote(file_part)).resolve()
            where = f"{path.relative_to(root)}:{number}"
            if not destination.exists():
                problems.append(f"{where}: missing file {target}")
            elif fragment and destination.suffix == ".md":
                if destination not in cache:
                    cache[destination] = anchors(destination)
                if fragment not in cache[destination]:
                    problems.append(f"{where}: missing anchor {target}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    problems = check(args.root.resolve())
    for problem in problems:
        print(problem)
    print(f"{len(problems)} broken link(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
