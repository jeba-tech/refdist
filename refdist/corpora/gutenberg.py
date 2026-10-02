"""S1, S2: English and American literary prose from Project Gutenberg, dated by author lifespan."""

from __future__ import annotations

import csv
import re
import time
from typing import Iterator

import numpy as np

from refdist import config
from refdist.corpora.base import DocRecord, clean, fetch, raw_dir, take_windows, windows

CFG = config.SOURCES["gutenberg"]
_YEARS = re.compile(r"(\d{3,4})\??-(\d{3,4})\??")
_DRAMA = re.compile(r"^[A-Z][A-Z .'-]{2,}\.\s")
_NOTE = re.compile(r"^\[[^\]]{0,14}\]")
_VERSE_OR_DRAMA = re.compile(r"poetry|poems|drama|\bplays\b|\bverses?\b|hymn|\bsongs\b|ballad|sonnet", re.I)
_SKIP = ("gutenberg", "[illustration", "[footnote", "transcriber", "chapter ", "contents")


def _lifespan(authors: str) -> tuple[str, int, int] | None:
    """Single-author works only; multi-author volumes cannot be dated by one lifespan."""
    parts = [a.strip() for a in authors.split(";") if a.strip()]
    if len(parts) != 1:
        return None
    m = _YEARS.search(parts[0])
    if not m:
        return None
    name = parts[0][: m.start()].strip(" ,")
    return name, int(m.group(1)), int(m.group(2))


def _eligible(stratum: str) -> list[tuple[str, str, int]]:
    rule = CFG[stratum]
    path = fetch(CFG["catalog"], raw_dir("gutenberg") / "pg_catalog.csv")
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Type"] != "Text" or r["Language"] != "en":
                continue
            loccs = [x.strip() for x in r["LoCC"].split(";")]
            if not any(l.startswith(p) for l in loccs for p in CFG["locc_prefixes"]):
                continue
            # LoCC PR/PS covers all literature, verse included. A line-length
            # heuristic misses long-lined verse, so poetry and drama are
            # excluded at book level from the catalogue's own labels.
            if _VERSE_OR_DRAMA.search(" ".join((r["Subjects"], r["Title"], r["Bookshelves"]))):
                continue
            life = _lifespan(r["Authors"])
            if not life:
                continue
            name, born, died = life
            if "death_min" in rule and rule["death_min"] <= died <= rule["death_max"]:
                year = died  # upper bound on when the text was written
            elif "birth_min" in rule and rule["birth_min"] <= born <= rule["birth_max"]:
                year = int(np.clip(born + 35, 1900, 1921))  # declared proxy: active at ~35
            else:
                continue
            out.append((r["Text#"], name, year))
    return out


def _body(raw: str) -> str:
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF", raw)
    return raw[start.end() if start else 0: end.start() if end else len(raw)]


def _prose_paragraphs(body: str) -> list[str]:
    paras = re.split(r"\n\s*\n", body.replace("\r\n", "\n"))
    out = []
    for p in paras[len(paras) // 20:]:  # skip front matter: title page, contents, preface
        lines = [l for l in p.split("\n") if l.strip()]
        if not lines:
            continue
        if len(lines) >= 3 and np.mean([len(l) for l in lines]) < 45:
            continue  # verse
        text = re.sub(r"_([^_]+)_", r"\1", " ".join(l.strip() for l in lines))
        low = text.lower()
        letters = [c for c in text if c.isalpha()]
        if any(s in low for s in _SKIP) or _DRAMA.match(text):
            continue
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.3:
            continue  # headings
        if _NOTE.match(text):
            continue  # editorial notes and endnotes: "[42:1] ...", "[1580.]"
        out.append(clean(text))
    return out


def _book_text(book_id: str) -> str | None:
    """Cached on disk, so a rebuild never re-downloads."""
    dest = raw_dir("gutenberg") / "books" / f"{book_id}.txt"
    if not dest.exists():
        time.sleep(CFG["request_delay_s"])
        try:
            fetch(CFG["text_url"].format(id=book_id), dest)
        except Exception:
            return None
    return dest.read_text(encoding="utf-8", errors="ignore")


def candidates(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    """
    One random book per author, authors in random order. Sampling books
    directly lets prolific authors dominate; after the per-author cap that left
    S1 with only 257 authors and 762 documents.
    """
    by_author: dict[str, list[tuple[str, int]]] = {}
    for book_id, author, year in _eligible(stratum):
        by_author.setdefault(author, []).append((book_id, year))
    authors = sorted(by_author)
    target = config.CORPUS["candidates_per_stratum"]
    produced = 0
    for i in rng.permutation(len(authors))[: CFG["max_books"]]:
        author = authors[i]
        books = by_author[author]
        book_id, year = books[int(rng.integers(0, len(books)))]
        raw = _book_text(book_id)
        if raw is None:
            continue
        wins = take_windows(windows(_prose_paragraphs(_body(raw)), rng), CFG["max_windows_per_source"], rng)
        for w in wins:
            produced += 1
            yield DocRecord(
                stratum=stratum, source="gutenberg", source_doc_id=author, year=year,
                text=w, license="Public domain (US)", extra={"book_id": book_id},
            )
        if produced >= target:
            return
