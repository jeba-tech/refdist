"""
S5, S6: Stack Exchange answers from non-programming sites.

Code, preformatted text and blockquotes are removed: blockquotes are usually
someone else's words (dictionary entries, quoted posts), which would break
single-author attribution. Posts edited after 2021 are excluded outright.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Iterator

import numpy as np

from refdist import config
from refdist.corpora.base import DocRecord, clean, fetch, raw_dir, take_windows, windows

CFG = config.SOURCES["stackexchange"]
_BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br"}
_DROP = {"pre", "code", "blockquote"}


class _Paragraphs(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth_drop = 0
        self.buf: list[str] = []
        self.paras: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _DROP:
            self.depth_drop += 1

    def handle_endtag(self, tag):
        if tag in _DROP:
            self.depth_drop = max(0, self.depth_drop - 1)
        elif tag in _BLOCK and not self.depth_drop:
            self._flush()

    def handle_data(self, data):
        if not self.depth_drop:
            self.buf.append(data)

    def _flush(self):
        t = clean(" ".join("".join(self.buf).split()))
        if t:
            self.paras.append(t)
        self.buf = []


def paragraphs(html: str) -> list[str]:
    p = _Paragraphs()
    p.feed(html)
    p._flush()
    return p.paras


def _posts_xml(site: str):
    import py7zr
    d = raw_dir("stackexchange") / site
    xml = d / "Posts.xml"
    if not xml.exists():
        archive = fetch(CFG["dump"].format(site=site), raw_dir("stackexchange") / f"{site}.7z")
        with py7zr.SevenZipFile(archive) as z:
            z.extract(path=d, targets=["Posts.xml"])
    return xml


def _license(year: int) -> str:
    return "CC BY-SA 2.5" if year < 2011 else "CC BY-SA 3.0" if year < 2018 else "CC BY-SA 4.0"


def candidates(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    st = config.STRATA[stratum]
    quota = CFG["reservoir_per_site"]
    for site in CFG["sites"]:
        reservoir: list[dict] = []
        seen = 0
        for _, el in ET.iterparse(_posts_xml(site), events=("end",)):
            if el.tag != "row":
                continue
            a = el.attrib
            el.clear()
            if a.get("PostTypeId") != "2" or "CommunityOwnedDate" in a:
                continue
            year = int(a["CreationDate"][:4])
            if not st["year_min"] <= year <= st["year_max"]:
                continue
            if a.get("LastEditDate", "")[:10] > CFG["max_last_edit"]:
                continue
            words = len(a.get("Body", "").split())
            if not 80 <= words <= 700:  # cheap prefilter before exact token counts
                continue
            seen += 1
            if len(reservoir) < quota:
                reservoir.append(a)
            else:
                j = int(rng.integers(0, seen))
                if j < quota:
                    reservoir[j] = a
        for a in reservoir:
            year = int(a["CreationDate"][:4])
            author = a.get("OwnerUserId") or f"anon-{a['Id']}"
            for w in take_windows(windows(paragraphs(a["Body"]), rng), 1, rng):
                yield DocRecord(
                    stratum=stratum, source="stackexchange", source_doc_id=f"{site}:{author}",
                    year=year, text=w, license=_license(year),
                    extra={"site": site, "post_id": a["Id"]},
                )
