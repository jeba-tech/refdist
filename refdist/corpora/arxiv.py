"""
S3, S4: arXiv abstracts by submission year.

Equal quotas per archive in both eras, so the S3/S4 contrast differs in era and
not in field mix (cs and cond-mat grew enormously between the two periods).
"""

from __future__ import annotations

import datetime as dt
import math
import re
import time
import urllib.parse
import urllib.request
from typing import Iterator

import numpy as np

from refdist import config
from refdist.corpora.base import UA, DocRecord, clean

CFG = config.SOURCES["arxiv"]
_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
_OVERSAMPLE = 2.0      # most 1990s abstracts fall below 150 tokens and are dropped downstream
_WEEKS_PER_YEAR = 2    # random one-week windows sampled per archive-year


def _query(search: str, start: int, n: int) -> str:
    """One API call. Raises after retries: a silent empty result once hid a bug
    that made every archive-year look empty."""
    url = (f"{CFG['api']}?search_query={urllib.parse.quote(search, safe=':*[]+')}"
           f"&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=ascending")
    for attempt in range(4):
        try:
            time.sleep(CFG["request_delay_s"])
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                body = r.read().decode("utf-8")
            if "<opensearch:totalResults" in body:
                return body
        except Exception:
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"arXiv API failed for {search!r} start={start}")


def _total(search: str) -> int:
    # max_results=0 is rejected by the API; ask for one result to read the count.
    return int(re.search(r"<opensearch:totalResults[^>]*>(\d+)<", _query(search, 0, 1)).group(1))


def _search(archive: str, day0: dt.date, day1: dt.date) -> str:
    return f"cat:{archive}+AND+submittedDate:[{day0:%Y%m%d}0000+TO+{day1:%Y%m%d}2359]"


def candidates(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    """
    Per archive-year, a quota drawn from random one-week windows. Whole-year
    queries fail here: the API returns nothing beyond an offset of roughly
    10,000, so large archive-years (cs 2019 has ~52k papers) would be
    under-sampled at random. A week always stays far below that limit.
    """
    st = config.STRATA[stratum]
    years = list(range(st["year_min"], st["year_max"] + 1))
    per_archive = math.ceil(config.CORPUS["candidates_per_stratum"] * _OVERSAMPLE / len(CFG["archives"]))
    per_week = math.ceil(per_archive / len(years) / _WEEKS_PER_YEAR)
    seen: set[str] = set()
    for archive in CFG["archives"]:
        for y in years:
            for off in sorted(rng.choice(358, size=_WEEKS_PER_YEAR, replace=False)):
                d0 = dt.date(y, 1, 1) + dt.timedelta(days=int(off))
                search = _search(archive, d0, d0 + dt.timedelta(days=6))
                total = _total(search)
                if total == 0:
                    continue
                n = min(per_week, total, 200)
                start = int(rng.integers(0, total - n + 1))
                for e in _ENTRY.findall(_query(search, start, n)):
                    # Everything after /abs/: old-style ids are "hep-th/9906033v1", and the
                    # bare number is shared across archives.
                    aid = re.search(r"<id>[^<]*/abs/([^<]+)</id>", e).group(1)
                    pub = re.search(r"<published>(\d{4})", e)
                    summ = re.search(r"<summary>(.*?)</summary>", e, re.S)
                    if not (pub and summ) or aid in seen:
                        continue
                    text = clean(re.sub(r"\s+", " ", summ.group(1)))
                    if text.count("$") > CFG["max_latex_dollars"]:
                        continue
                    seen.add(aid)
                    yield DocRecord(
                        stratum=stratum, source="arxiv", source_doc_id=aid, year=int(pub.group(1)),
                        text=text, license="arXiv metadata, CC0 1.0",
                        extra={"archive": archive.rstrip(".*")},
                    )
