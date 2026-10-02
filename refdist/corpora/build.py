"""
Build the frozen human-only corpus: 1,000 length-matched documents per stratum.

Every stratum passes through the same filters in the same order:
  1. adapter candidates  (cached to data/interim so rebuilds never re-download)
  2. year window and the hard 2021 cutoff
  3. 150-400 tokens on the shared length ruler
  4. exact and near-duplicate removal
  5. per-source cap (author / writer / user / case / meeting)
  6. length-matched sample: equal quotas across 5 length bins

A built stratum is immutable. S6 above all: it sets every detector threshold,
so rebuilding it after scoring would silently move every FPR in the study.

Usage:  python -m refdist.corpora.build S6 S5 S3 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from refdist import config, paths, provenance
from refdist.corpora import ADAPTERS
from refdist.corpora.base import _ENTITY, minhash_dedup, n_tokens


def _seed(stratum: str) -> int:
    return config.CORPUS["seed"] * 1000 + int(stratum[1:])


def stratum_file(stratum: str) -> Path:
    d = paths.processed() / "strata"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stratum}.parquet"


def _candidates(stratum: str, rng: np.random.Generator) -> pd.DataFrame:
    f = paths.interim() / "candidates" / f"{stratum}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    source = config.STRATA[stratum]["source"]
    rows = [r.row() for r in ADAPTERS[source](stratum, rng)]
    df = pd.DataFrame(rows)
    if not df.empty:
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f, index=False)
    return df


def _cap_per_source(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    df = df.assign(_r=rng.random(len(df))).sort_values("_r")
    return df.groupby("source_doc_id", sort=False).head(cap).drop(columns="_r")


def _edges() -> np.ndarray:
    lo, hi, k = config.CORPUS["min_tokens"], config.CORPUS["max_tokens"], config.CORPUS["length_match_bins"]
    return np.linspace(lo, hi + 1, k + 1)


def _bins(n_tokens: pd.Series) -> np.ndarray:
    k = config.CORPUS["length_match_bins"]
    return np.clip(np.digitize(n_tokens, _edges()) - 1, 0, k - 1)


def match_key(stratum: str, df: pd.DataFrame) -> pd.Series:
    """The field a within-source pair must share: arXiv archive, Stack Exchange site."""
    src = config.STRATA[stratum]["source"]
    if src == "arxiv":
        return df["meta"].map(lambda m: json.loads(m).get("archive", "?"))
    if src == "stackexchange":
        return df["source_doc_id"].str.split(":").str[0]
    return pd.Series("all", index=df.index)


def pair_target(stratum: str) -> dict[tuple[int, str], int] | None:
    """
    Within-source pairs (S3/S4, S5/S6) must differ in era only. The member built
    second copies its partner's joint counts over (length bin x field), so the
    pair shares both its length distribution and its field mix. Matching length
    alone is not enough: 1990s astro-ph abstracts survive the 150-token minimum
    far more often than math or cs ones, which skews the field mix by era.
    """
    for a, b in config.WITHIN_SOURCE_PAIRS:
        partner = b if stratum == a else a if stratum == b else None
        if partner and stratum_file(partner).exists():
            p = pd.read_parquet(stratum_file(partner))
            cells = pd.Series(list(zip(_bins(p["n_tokens"]), match_key(partner, p)))).value_counts()
            return {(int(bn), str(key)): int(v) for (bn, key), v in cells.items()}
    return None


def length_matched_sample(df: pd.DataFrame, n: int, rng: np.random.Generator,
                          target: dict[tuple[int, str], int] | None = None,
                          key: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Default: equal quotas across fixed length bins, so every stratum has the same
    length distribution. For the second member of a within-source pair: the
    partner's exact (length bin x field) counts. Any shortfall is filled from the
    same length bin first, then the nearest bins, and recorded.
    """
    k = config.CORPUS["length_match_bins"]
    df = df.assign(_bin=_bins(df["n_tokens"]),
                   _key=key.to_numpy() if key is not None and target else "all",
                   _r=rng.random(len(df))).sort_values("_r")
    quota = target or {(b, "all"): n // k for b in range(k)}
    avail = df.groupby(["_bin", "_key"]).size().to_dict()
    take = {c: min(q, int(avail.get(c, 0))) for c, q in quota.items()}
    for c, v in avail.items():
        take.setdefault(c, 0)
    short = n - sum(take.values())
    while short > 0:
        spare = [c for c in take if avail.get(c, 0) > take[c]]
        if not spare:
            break
        need_bins = {c[0] for c in quota if take.get(c, 0) < quota[c]}
        spare.sort(key=lambda c: (min((abs(c[0] - b) for b in need_bins), default=0), take[c]))
        take[spare[0]] += 1
        short -= 1
    picked = pd.concat([g.head(take[c]) for c, g in df.groupby(["_bin", "_key"]) if take.get(c, 0)])
    per_bin = {int(b): int(sum(v for c, v in take.items() if c[0] == b)) for b in range(k)}
    report = {"bin_edges": _edges().round(1).tolist(), "per_bin": per_bin,
              "shortfall": n - len(picked), "matched_to_partner": target is not None}
    if target:
        report["partner_cell_mismatch"] = int(sum(abs(take.get(c, 0) - q) for c, q in target.items())
                                              + sum(v for c, v in take.items() if c not in target))
    return picked.drop(columns=["_bin", "_key", "_r"]), report


def build_stratum(stratum: str, force: bool = False) -> dict:
    out = stratum_file(stratum)
    if out.exists() and not force:
        return {"stratum": stratum, "status": "frozen (already built)"}
    if out.exists() and stratum == config.CALIBRATION_STRATUM:
        raise RuntimeError(
            f"{stratum} is the calibration stratum and is already built. Rebuilding it would "
            "move every detector threshold. Delete it by hand only if nothing has been scored.")

    st = config.STRATA[stratum]
    df = _candidates(stratum, np.random.default_rng(_seed(stratum)))
    # Separate stream for sampling: a rebuild from cached candidates must pick
    # exactly the documents the original build picked.
    rng = np.random.default_rng(_seed(stratum) + 1)
    log = {"stratum": stratum, "name": st["name"], "candidates": len(df)}
    if df.empty:
        log["status"] = "NO CANDIDATES (source missing?)"
        return log

    df = df[(df["year"] >= st["year_min"]) & (df["year"] <= st["year_max"])
            & (df["year"] <= config.CORPUS["hard_cutoff_year"])]
    log["after_year"] = len(df)

    df = df.assign(n_tokens=[n_tokens(t) for t in df["text"]], n_chars=df["text"].str.len())
    df = df[df["n_tokens"].between(config.CORPUS["min_tokens"], config.CORPUS["max_tokens"])]
    log["after_length"] = len(df)

    df = df.drop_duplicates("text_sha256").reset_index(drop=True)
    keep = minhash_dedup(df["text"].tolist(), config.CORPUS["minhash_threshold"], _seed(stratum))
    df = df.iloc[keep]
    log["after_dedup"] = len(df)

    cap = config.SOURCES[st["source"]].get("max_windows_per_source", config.CORPUS["max_windows_per_source"])
    df = _cap_per_source(df, cap, rng)
    log["after_source_cap"] = len(df)
    log["distinct_sources"] = int(df["source_doc_id"].nunique())

    df, lm = length_matched_sample(df, config.CORPUS["docs_per_stratum"], rng,
                                   pair_target(stratum), match_key(stratum, df))
    log["length_matching"] = lm
    log["docs_with_html_entities"] = int(df["text"].str.contains(_ENTITY).sum())
    log["final"] = len(df)
    log["status"] = "built" if lm["shortfall"] == 0 else f"built with shortfall {lm['shortfall']}"

    tmp = out.with_suffix(".tmp.parquet")
    df.sort_values("doc_id").to_parquet(tmp, index=False)
    tmp.replace(out)  # atomic: a concurrent assemble() never sees a half-written stratum
    log["sha256"] = provenance.sha256_file(out)
    return log


def assemble() -> pd.DataFrame:
    """documents.parquet = concatenation of every built stratum."""
    files = sorted((paths.processed() / "strata").glob("S*.parquet"), key=lambda p: int(p.stem[1:]))
    docs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tmp = paths.documents_file().with_suffix(".tmp.parquet")
    docs.to_parquet(tmp, index=False)
    tmp.replace(paths.documents_file())
    manifests = {p.stem: json.loads(p.read_text()) for p in sorted((paths.processed() / "manifests").glob("S*.json"))}
    (paths.processed() / "manifest.json").write_text(json.dumps(
        {"strata": manifests, "n_documents": len(docs), "_provenance": provenance.stamp()}, indent=2))
    return docs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("strata", nargs="*", default=list(config.STRATA))
    ap.add_argument("--force", action="store_true", help="rebuild non-calibration strata")
    args = ap.parse_args(argv)

    mdir = paths.processed() / "manifests"
    mdir.mkdir(exist_ok=True)
    for s in args.strata:
        log = build_stratum(s, force=args.force)
        print(json.dumps(log, indent=1))
        if log.get("status", "").startswith("built"):
            provenance.write_json(mdir / f"{s}.json", log)
    docs = assemble()
    print(f"\ndocuments.parquet: {len(docs)} docs across {docs['stratum'].nunique()} strata")


if __name__ == "__main__":
    main()
