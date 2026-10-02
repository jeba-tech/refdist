"""
S10 TwitterAAE, S11 legal opinions, S12 AMI meeting transcripts.

All three have units far shorter or far longer than a 150-400 token window, so
each is segmented differently; the rules are declared in the OSF registration.
"""

from __future__ import annotations

import json
import lzma
import re
import struct
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import defaultdict
from typing import Iterator

import numpy as np

from refdist import config
from refdist.corpora.base import UA, DocRecord, clean, fetch, raw_dir, take_windows, unescape, windows


# ── S10 TwitterAAE ───────────────────────────────────────────────────────────

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")


def _aa_member() -> bytes:
    """
    The AA-aligned file is the second member of a 5.9GB zip, ~72MB in. Fetch only
    its byte range instead of the whole archive.
    """
    cfg = config.SOURCES["twitteraae"]
    out = raw_dir("twitteraae") / "twitteraae_all_aa"
    if out.exists():
        return out.read_bytes()
    head = urllib.request.urlopen(urllib.request.Request(
        cfg["zip"], headers={**UA, "Range": "bytes=0-65535"}), timeout=120).read()
    off = 0
    while True:
        _, _, _, comp, _, _, _, csize, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", head[off:off + 30])
        name = head[off + 30: off + 30 + nlen].decode()
        start = off + 30 + nlen + xlen
        if name == cfg["member"]:
            break
        off = start + csize
    blob = fetch(cfg["zip"], raw_dir("twitteraae") / "aa_member.deflate", (start, start + csize - 1))
    data = zlib.decompressobj(-15).decompress(blob.read_bytes()) if comp == 8 else blob.read_bytes()
    out.write_bytes(data)
    return data


def twitteraae(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    by_user: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in _aa_member().decode("utf-8", "ignore").splitlines():
        cols = line.split("\t")
        if len(cols) < 10:
            continue
        try:
            ts, user, text = json.loads(cols[1]), json.loads(cols[2]), json.loads(cols[5])
            aa = float(cols[6])
        except (ValueError, json.JSONDecodeError):
            continue
        if not aa >= 0.8 or text.startswith("RT @"):  # nan fails >=; retweets aren't the user's words
            continue
        # The release stores tweets HTML-escaped ("&gt;", "&amp;"). Left as-is,
        # these strings would inflate this stratum's perplexity and so deflate
        # its false-positive rate: an artefact of storage, not of the writers.
        text = _MENTION.sub("@user", _URL.sub("", unescape(text))).strip()
        if len(text.split()) >= 3:
            by_user[user].append((ts, text))
    users = sorted(by_user)
    for i in rng.permutation(len(users)):
        u = users[i]
        tweets = sorted(set(by_user[u]), key=lambda t: _ts(t[0]))
        year = _ts(tweets[0][0]).year
        for w in take_windows(windows([t for _, t in tweets], rng, joiner="\n"), 1, rng):
            yield DocRecord(stratum=stratum, source="twitteraae", source_doc_id=u, year=year,
                            text=w, license="TwitterAAE research release (Blodgett et al., 2016)")


def _ts(s: str):
    import datetime as dt
    return dt.datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")


# ── S11 legal opinions ───────────────────────────────────────────────────────

_CITE = re.compile(
    r"\b\d+\s+(?:U\.\s?S\.|S\.\s?Ct\.|L\.\s?Ed|F\.\s?(?:2d|3d|4th|Supp)|N\.[EWY]\.\s?\d?d|"
    r"So\.\s?\d?d|P\.\s?\d?d|A\.\s?\d?d|S\.[EW]\.\s?\d?d|Cal\.|N\.Y\.)"
)
_STAR_PAGE = re.compile(r"\*\d+\s?|\{¶\s*\d+\}\s?")  # "*228" pagination, "{¶ 40}" numbering
_BOILERPLATE = re.compile(
    r"^(?:Before |BEFORE |Argued |Submitted |Decided |Filed |Opinion by |PER CURIAM\b)|"
    r"\bfor (?:appellants?|appellees?|plaintiffs?|defendants?|petitioners?|respondents?)\b|"
    r"\bJJ\.|\bP\.J\."
)


def _stream_records(url: str) -> Iterator[dict]:
    dec = lzma.LZMADecompressor()
    buf = b""
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        while chunk := r.read(1 << 20):
            buf += dec.decompress(chunk)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln.strip():
                    yield json.loads(ln)


def legal(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    cfg = config.SOURCES["legal"]
    pool_size, scan = 3000, 30000
    pool: list[dict] = []
    seen = 0
    for rec in _stream_records(cfg["shard"].format(n=0)):
        year = int(rec["created_timestamp"][-4:])
        if year > cfg["max_created_year"]:
            continue
        seen += 1
        if len(pool) < pool_size:
            pool.append(rec)
        elif (j := int(rng.integers(0, seen))) < pool_size:
            pool[j] = rec
        if seen >= scan:
            break
    for rec in pool:
        paras = [_STAR_PAGE.sub("", clean(p)) for p in rec["text"].split("\n")]
        paras = [p for p in paras if len(p.split()) >= 8
                 and len(_CITE.findall(p)) <= cfg["max_citations_per_paragraph"]
                 and "\ufffd" not in p            # characters lost in the source's encoding
                 and not _BOILERPLATE.search(p)]  # bench and counsel listings: not the judge's prose
        for w in take_windows(windows(paras, rng), 3, rng):
            yield DocRecord(stratum=stratum, source="legal", source_doc_id=rec["url"],
                            year=int(rec["created_timestamp"][-4:]), text=w,
                            license="Public domain (US court opinions)")


# ── S12 AMI meetings ─────────────────────────────────────────────────────────


def _turns(z: zipfile.ZipFile, meeting: str, gap: float) -> list[str]:
    """
    Utterances are formed per speaker first (split on pauses longer than `gap`),
    then interleaved by start time. Sorting raw words across speakers instead
    would chop a sentence apart every time someone else says "yeah".
    """
    utts = []
    for name in z.namelist():
        if not re.search(rf"words/{meeting}\.[A-Z]\.words\.xml$", name):
            continue
        spk = name.split(".")[-3]
        words = []
        for el in ET.fromstring(z.read(name)):
            if el.tag == "w" and el.text and el.get("starttime") is not None:
                st = float(el.get("starttime"))
                words.append((st, float(el.get("endtime") or st), el.text, el.get("punc") == "true"))
        words.sort()
        cur, cur_start, last_end = [], 0.0, None
        for st, en, tok, punc in words:
            if cur and last_end is not None and st - last_end > gap:
                utts.append((cur_start, spk, cur))
                cur = []
            if not cur:
                cur_start = st
            if punc and cur:
                cur[-1] += tok
            elif not punc:
                cur.append(tok)
            last_end = en
        if cur:
            utts.append((cur_start, spk, cur))
    utts.sort(key=lambda u: u[0])
    turns: list[list[str]] = []
    last_spk = None
    for _, spk, toks in utts:
        if turns and spk == last_spk:
            turns[-1].extend(toks)
        else:
            turns.append(list(toks))
        last_spk = spk
    return [clean(" ".join(t)) for t in turns if t]


def spoken(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    cfg = config.SOURCES["spoken"]
    z = zipfile.ZipFile(fetch(cfg["zip"], raw_dir("ami") / "ami_public_manual.zip"))
    meetings = sorted({m.group(1) for n in z.namelist()
                       if (m := re.search(r"words/([A-Z]{2}\d{4}[a-d])\.", n))})
    for i in rng.permutation(len(meetings)):
        mt = meetings[i]
        wins = take_windows(windows(_turns(z, mt, cfg["turn_gap_s"]), rng, joiner="\n"),
                            cfg["max_windows_per_source"], rng)
        for w in wins:
            yield DocRecord(stratum=stratum, source="ami", source_doc_id=mt, year=cfg["year"],
                            text=w, license="CC BY 4.0 (AMI Meeting Corpus)")
