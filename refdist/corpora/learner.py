"""
S7 PELIC, S8 ICNALE (L2 English), S9 ASAP-AES (US student essays).

PELIC and ICNALE carry the writer's L1 and proficiency, enabling a
within-stratum dose-response analysis. ICNALE and ASAP need a one-time manual
download (registration / Kaggle competition rules); their adapters read from
data/raw and yield nothing, with a clear message, if the files are absent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from refdist import config, paths
from refdist.corpora.base import DocRecord, clean, fetch, raw_dir, take_windows, windows


# ── S7 PELIC ─────────────────────────────────────────────────────────────────

def pelic(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    cfg = config.SOURCES["pelic"]
    df = pd.read_csv(fetch(cfg["url"], raw_dir("pelic") / "PELIC_compiled.csv"))
    df = df[df["L1"].notna() & (df["L1"] != "English")]
    # Drafts of the same answer are near-duplicates: keep the final version only.
    df = df.sort_values("version").groupby(["anon_id", "question_id"], as_index=False).tail(1)
    df = df.iloc[rng.permutation(len(df))]
    for r in df.itertuples():
        year = int(str(r.semester)[:4])
        for w in take_windows(windows(clean(str(r.text)).split("\n"), rng), 1, rng):
            yield DocRecord(
                stratum=stratum, source="pelic", source_doc_id=str(r.anon_id), year=year,
                text=w, license="PELIC terms (github.com/ELI-Data-Mining-Group/PELIC-dataset)",
                l1=str(r.L1), proficiency=str(r.level_id),
                extra={"question_id": int(r.question_id)},
            )


# ── S8 ICNALE ────────────────────────────────────────────────────────────────

_CEFR = re.compile(r"^(A2|B1|B2|C1|C2|XX)")


def icnale(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    root = paths.raw() / config.SOURCES["icnale"]["manual_dir"]
    # Written Essays module only ("WE_"): "WEP_" (Written Essays Plus, released
    # 2026) may postdate 2021 and is excluded. Merged and tagged copies are skipped
    # so each essay is read exactly once.
    # The WE 2.6 zip holds every essay twice (Unclassified and Classified copies)
    # plus merged/tagged concatenations; read the one unmerged copy only.
    files = sorted(f for f in root.rglob("WE_*.txt")
                   if f.parent.name == "WE_0_Unclassified_Unmerged") if root.exists() else []
    if not files:
        print(f"[S8] ICNALE not found in {root}. Register at "
              "https://language.sakura.ne.jp/icnale/ and extract the Written Essays there.")
        return
    for i in rng.permutation(len(files)):
        f: Path = files[i]
        parts = f.stem.split("_")  # WE_CHN_PTJ0_001_B1_1: module, region, task, student, CEFR
        if len(parts) < 5 or parts[0] != "WE":
            continue
        country = parts[1]
        if country == "ENS":  # native-speaker control group
            continue
        cefr = next((m.group(1) for p in parts[4:] if (m := _CEFR.match(p))), None)
        writer = f"{country}_{parts[3]}"  # one student writes both topics: cap per student, not per essay
        text = clean(f.read_text(encoding="utf-8", errors="ignore"))
        for w in take_windows(windows(text.split("\n"), rng), 1, rng):
            yield DocRecord(
                # Upper bound: Ishikawa (2013) already reports the module's full
                # 5,600 essays by 2,800 writers, so none postdates 2013.
                stratum=stratum, source="icnale", source_doc_id=writer, year=2013, text=w,
                license="ICNALE terms of use", l1=country, proficiency=cefr,
            )


# ── S9 ASAP-AES ──────────────────────────────────────────────────────────────

# ASAP anonymised named entities as @PERSON1, @CAPS2, ... Left in place, these
# tokens inflate perplexity in a way no real writer produces. Essays with many
# are dropped; the rest get a fixed, declared substitution.
_PLACEHOLDER = re.compile(r"@([A-Z]+)\d*")
# Six of eight sets were transcribed from handwriting; unreadable words were
# written as "???", "illegible" or "not legible" (ASAP data description).
_ILLEGIBLE = re.compile(r"\?\?\?|\billegible\b|\bnot legible\b", re.I)
_FILL = {
    "PERSON": "Alex", "CAPS": "Sam", "LOCATION": "the town", "ORGANIZATION": "the company",
    "DATE": "that day", "TIME": "that time", "MONEY": "some money", "PERCENT": "a percent",
    "NUM": "two", "MONTH": "May", "DR": "Dr. Lee", "CITY": "the city", "STATE": "the state",
    "EMAIL": "an email", "STREET": "the street",
}


def asap(stratum: str, rng: np.random.Generator) -> Iterator[DocRecord]:
    cfg = config.SOURCES["asap"]
    f = paths.raw() / cfg["manual_file"]
    if not f.exists():
        print(f"[S9] ASAP not found at {f}. Accept the rules at "
              "https://www.kaggle.com/competitions/asap-aes and place training_set_rel3.tsv there.")
        return
    df = pd.read_csv(f, sep="\t", encoding="latin-1", usecols=["essay_id", "essay_set", "essay"])
    df = df.iloc[rng.permutation(len(df))]
    for r in df.itertuples():
        if len(_PLACEHOLDER.findall(r.essay)) > cfg["max_placeholders"]:
            continue
        if _ILLEGIBLE.search(r.essay):  # transcriber marks, not the student's words
            continue
        text = clean(_PLACEHOLDER.sub(lambda m: _FILL.get(m.group(1), "it"), r.essay))
        for w in take_windows(windows(re.split(r"\n+|\s{3,}", text), rng), 1, rng):
            yield DocRecord(
                stratum=stratum, source="asap", source_doc_id=str(r.essay_id), year=2012,
                text=w, license="Kaggle ASAP-AES competition terms",
                extra={"essay_set": int(r.essay_set)},
            )
