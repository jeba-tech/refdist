"""
Shared corpus machinery: the document record, the length ruler, segmentation
into 150-400 token windows, near-duplicate removal, and cached downloads.

Every adapter yields candidate DocRecords. build.py applies the same filters to
all strata, so no stratum gets special treatment after its source is parsed.
"""

from __future__ import annotations

import functools
import hashlib
import html
import json
import re
import shutil
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from refdist import config, paths


@dataclass
class DocRecord:
    stratum: str
    source: str
    source_doc_id: str
    year: int
    text: str
    license: str
    l1: str | None = None
    proficiency: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def doc_id(self) -> str:
        return f"{self.stratum}-{self.text_sha256[:12]}"

    def row(self) -> dict:
        d = asdict(self)
        d["meta"] = json.dumps(d.pop("extra"), sort_keys=True)  # e.g. arXiv archive, Gutenberg book id
        d["doc_id"] = self.doc_id
        d["text_sha256"] = self.text_sha256
        return d


# ── length ruler ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _ruler():
    from transformers import AutoTokenizer
    t = config.CORPUS["length_tokenizer"]
    return AutoTokenizer.from_pretrained(t["model_id"], revision=t["revision"])


def n_tokens(text: str) -> int:
    return len(_ruler()(text, add_special_tokens=False)["input_ids"])


# ── text hygiene ─────────────────────────────────────────────────────────────

_WS = re.compile(r"[ \t\r\f\v]+")
_SENT = re.compile(r"([.!?][\"')\]]?)\s+(?=[\"'(\[]?[A-Z0-9])")


_ENTITY = re.compile(r"&(?:[a-z]{2,8}|#\d{2,5}|#x[0-9a-f]{2,4});", re.I)


def unescape(text: str) -> str:
    """HTML/XML entities, however many times escaped. Several sources store text
    escaped; left in place, "&amp;gt;" inflates perplexity for storage reasons,
    not because of anything the writer did."""
    for _ in range(4):
        new = html.unescape(text)
        if new == text:
            break
        text = new
    return text


_C1 = re.compile("[\u0080-\u009f]")


def _cp1252(m: re.Match) -> str:
    """C1 control characters are Windows-1252 punctuation decoded as Latin-1
    (U+0097 was an em dash). Restore the intended character, else drop it."""
    try:
        return bytes([ord(m.group())]).decode("cp1252")
    except UnicodeDecodeError:
        return ""


def clean(text: str) -> str:
    text = _C1.sub(_cp1252, unescape(text))
    text = text.replace("\u00a0", " ").replace("\ufeff", "").replace("\u200b", "")
    lines = [_WS.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


_ABBREV = re.compile(
    r"(?:^|[\s(])(?:Mr|Mrs|Ms|Dr|St|Prof|Jr|Sr|Rev|Gen|Col|Capt|Lt|vs|etc|e\.g|i\.e|cf|al|No|Vol|U\.S)\.$"
)


def sentences(text: str) -> list[str]:
    parts = _SENT.split(text)  # [s0, punct0, s1, punct1, ..., sn]
    out = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    merged: list[str] = []
    for x in (x.strip() for x in out):
        if not x:
            continue
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] += " " + x
        else:
            merged.append(x)
    return merged


# ── segmentation ─────────────────────────────────────────────────────────────

def windows(
    units: Iterable[str],
    rng: np.random.Generator,
    joiner: str = "\n\n",
    lo: int | None = None,
    hi: int | None = None,
) -> Iterator[str]:
    """
    Pack consecutive units (paragraphs, speaker turns, tweets) into windows of
    lo..hi tokens. Each window draws a random target length so the pool spans
    the whole range instead of piling up just above `lo`, which the length
    matching step would otherwise be unable to balance.
    """
    lo = lo or config.CORPUS["min_tokens"]
    hi = hi or config.CORPUS["max_tokens"]
    cur: list[str] = []
    cur_n = 0
    sep = n_tokens(joiner)  # separators are tokens too; uncounted, they overshoot `hi`
    target = int(rng.integers(lo, hi + 1))
    for u in units:
        u = u.strip()
        if not u:
            continue
        c = n_tokens(u) + (sep if cur else 0)
        if c > hi:
            # One oversized unit: flush and split it at sentence boundaries. A
            # unit with no sentence boundary (a long list, a table) cannot fit a
            # window and is dropped; recursing on it would never terminate.
            if lo <= cur_n <= hi:
                yield joiner.join(cur)
            cur, cur_n = [], 0
            pieces = sentences(u)
            if len(pieces) > 1:
                yield from windows(pieces, rng, " ", lo, hi)
            target = int(rng.integers(lo, hi + 1))
            continue
        if cur and cur_n + c > hi:
            if cur_n >= lo:
                yield joiner.join(cur)
            cur, cur_n = [], 0
            c -= sep
            target = int(rng.integers(lo, hi + 1))
        cur.append(u)
        cur_n += c
        if cur_n >= target:
            yield joiner.join(cur)
            cur, cur_n = [], 0
            target = int(rng.integers(lo, hi + 1))
    if lo <= cur_n <= hi:
        yield joiner.join(cur)


def take_windows(texts: Iterable[str], k: int, rng: np.random.Generator) -> list[str]:
    """At most k windows from one source document, chosen at random."""
    pool = list(texts)
    if len(pool) <= k:
        return pool
    idx = sorted(rng.choice(len(pool), size=k, replace=False))
    return [pool[i] for i in idx]


# ── near-duplicate removal ───────────────────────────────────────────────────

_PRIME = (1 << 61) - 1


def minhash_dedup(texts: list[str], threshold: float, seed: int, n_perm: int = 64,
                  bands: int = 16) -> list[int]:
    """
    Indices to keep after near-duplicate removal (MinHash + LSH).

    Python's str hash is salted per process, so shingles are re-hashed with a
    fixed seed below; the result is deterministic across runs.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, n_perm, dtype=np.uint64)
    b = rng.integers(0, _PRIME, n_perm, dtype=np.uint64)
    sigs = np.empty((len(texts), n_perm), dtype=np.uint64)
    for i, t in enumerate(texts):
        w = t.lower().split()
        sh = np.array(
            [int(hashlib.blake2b(" ".join(w[j:j + 5]).encode(), digest_size=4).hexdigest(), 16)
             for j in range(max(1, len(w) - 4))],
            dtype=np.uint64,
        )
        sigs[i] = ((np.outer(sh, a) + b) % _PRIME).min(axis=0)

    rows = n_perm // bands
    buckets: dict[tuple, list[int]] = {}
    for i in range(len(texts)):
        for bnd in range(bands):
            buckets.setdefault((bnd, sigs[i, bnd * rows:(bnd + 1) * rows].tobytes()), []).append(i)

    dropped: set[int] = set()
    for members in buckets.values():
        for x in range(len(members)):
            i = members[x]
            if i in dropped:
                continue
            for j in members[x + 1:]:
                if j not in dropped and (sigs[i] == sigs[j]).mean() >= threshold:
                    dropped.add(j)
    return [i for i in range(len(texts)) if i not in dropped]


# ── downloads ────────────────────────────────────────────────────────────────

UA = {"User-Agent": "refdist-research/0.1 (https://github.com/jeba-tech/refdist)"}


def fetch(url: str, dest: Path, byte_range: tuple[int, int] | None = None,
          retries: int = 4) -> Path:
    """Download once into data/raw; later calls return the cached file."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = dict(UA)
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return dest


def raw_dir(source: str) -> Path:
    d = paths.raw() / source
    d.mkdir(parents=True, exist_ok=True)
    return d


Adapter = Callable[[str, np.random.Generator], Iterator[DocRecord]]
