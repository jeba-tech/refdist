"""
Resumable scoring on a Kaggle T4. Each stage loads one model (or pair) once and
caches per-token statistics for every group of items, flushing as it goes, so
a killed session resumes where it stopped.

Stages
  P   independent predictor (SmolLM2-360M)         -> PPL_indep, all strata
  A   reference models R1-R5                        -> Fast-DetectGPT, PPL_r
  B   Binoculars pairs (R4, R5)                     -> Binoculars
  C   trained classifiers D3-D5                     -> P(AI)
  derive  numpy over caches                          -> scores.parquet (CPU, seconds)

Groups are the 12 strata, "control" (positive control, validity gate only) and
"variants" (H5 transforms of S6 documents).

FORECAST FIREWALL: detector stages (A, B, C) refuse S10-S12 until the
pre-registered forecast file exists. Stage P may score them, because PPL_indep
is the forecast's input, not its outcome.

Usage (Kaggle):  python -m refdist.scoring.runner P
                 python -m refdist.scoring.runner A --refs R1 R2
                 python -m refdist.scoring.runner derive
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import pandas as pd

from refdist import config, paths, provenance
from refdist.scoring import detectors, logprob_cache as lc
from refdist.scoring import models, trained

PREDICTOR_KEY = "PPL_indep"


# ── items ────────────────────────────────────────────────────────────────────

def groups() -> dict[str, pd.DataFrame]:
    """group name -> DataFrame(doc_id, text). Missing inputs are simply absent."""
    out: dict[str, pd.DataFrame] = {}
    if paths.documents_file().exists():
        docs = pd.read_parquet(paths.documents_file(), columns=["doc_id", "stratum", "text"])
        for s, g in docs.groupby("stratum"):
            out[s] = g[["doc_id", "text"]]
    if paths.control_file().exists():
        c = pd.read_parquet(paths.control_file(), columns=["doc_id", "text"])
        out["control"] = c
    if paths.variants_file().exists():
        v = pd.read_parquet(paths.variants_file(), columns=["variant_id", "text"])
        out["variants"] = v.rename(columns={"variant_id": "doc_id"})
    return out


def forecast_committed() -> bool:
    return paths.forecast_file().exists()


def registered() -> bool:
    """The OSF registration must be live before any study model scores a study document."""
    return bool(config.STUDY["osf"].strip()) or os.environ.get("REFDIST_OSF_URL", "").strip() != ""


def allowed(stage: str, group: str) -> bool:
    # Pre-registration firewall: only the positive control (generated text,
    # outside every hypothesis) may be scored before the registration exists.
    if group != "control" and not registered():
        print(f"  [prereg] {group} skipped: set STUDY['osf'] in config.py (or REFDIST_OSF_URL) "
              "to the live OSF registration first")
        return False
    if stage != "P" and group in config.FORECAST_STRATA and not forecast_committed():
        print(f"  [firewall] {group} skipped: commit prereg/forecast_S10_S12.json first")
        return False
    return True


# ── ledger ───────────────────────────────────────────────────────────────────

def ledger() -> set[tuple[str, str, str]]:
    f = paths.ledger_file()
    if not f.exists():
        return set()
    return {(r["stage"], r["model"], r["group"]) for r in map(json.loads, f.read_text().splitlines())}


def record(stage: str, model: str, group: str, n: int) -> None:
    with open(paths.ledger_file(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage, "model": model, "group": group, "n": n,
                            **provenance.stamp()}) + "\n")


def _run_groups(stage: str, model_key: str, fn, only: list[str] | None) -> None:
    done = ledger()
    for group, items in groups().items():
        if only and group not in only:
            continue
        if (stage, model_key, group) in done or not allowed(stage, group):
            continue
        lc.run_cached(fn, model_key, group, items)
        record(stage, model_key, group, len(items))


# ── stages ───────────────────────────────────────────────────────────────────

def stage_single(stage: str, model_key: str, model_id: str, revision: str, only=None) -> None:
    model, tok = models.load_causal_lm(model_id, revision)
    _run_groups(stage, model_key, lambda t, i: lc.token_stats(model, tok, t, i), only)
    models.free(model)


def stage_P(only=None) -> None:
    ip = config.INDEPENDENT_PREDICTOR
    stage_single("P", PREDICTOR_KEY, ip["model_id"], ip["revision"], only)


def stage_A(refs: list[str], only=None) -> None:
    for r in refs:
        ref = config.REFERENCE_MODELS[r]
        stage_single("A", r, ref["model_id"], ref["revision"], only)


def stage_B(refs: list[str], only=None) -> None:
    for r in refs:
        ref = config.REFERENCE_MODELS[r]
        observer, tok = models.load_causal_lm(ref["model_id"], ref["revision"])
        performer, _ = models.load_causal_lm(ref["binoculars_pair"], ref["binoculars_pair_revision"])
        _run_groups("B", f"{r}-pair", lambda t, i: lc.pair_stats(observer, performer, tok, t, i), only)
        models.free(observer, performer)


def stage_C(dets: list[str], only=None) -> None:
    for d in dets:
        det = config.DETECTORS[d]
        model, tok = models.load_classifier(det["model_id"], det["revision"])
        done = ledger()
        for group, items in groups().items():
            if (only and group not in only) or ("C", d, group) in done or not allowed("C", group):
                continue
            out = trained.classify(model, tok, items["text"].tolist(), items["doc_id"].tolist(),
                                   det["ai_label_index"])
            f = lc.cache_path(d, group)
            f.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(f, index=False)
            record("C", d, group, len(items))
        models.free(model)


# ── derive ───────────────────────────────────────────────────────────────────

def derive() -> pd.DataFrame:
    """Every document-level score from the caches. CPU only; seconds, not GPU-hours."""
    run_id = provenance.new_run_id()
    frames = []

    def add(df: pd.DataFrame, instance: str, detector: str, reference: str | None, group: str):
        frames.append(df.rename(columns={"doc_id": "item_id"}).assign(
            instance=instance, detector=detector, reference=reference, group=group))

    for group in groups():
        f = lc.cache_path(PREDICTOR_KEY, group)
        if f.exists():
            lp = detectors.log_perplexity(pd.read_parquet(f))
            add(lp.rename(columns={"log_ppl": "score"}), PREDICTOR_KEY, PREDICTOR_KEY, None, group)
        for r in config.DETECTORS["D1"]["references"]:
            f = lc.cache_path(r, group)
            if f.exists():
                cache = pd.read_parquet(f)
                add(detectors.fast_detectgpt(cache), f"D1@{r}", "D1", r, group)
                lp = detectors.log_perplexity(cache)
                add(lp.rename(columns={"log_ppl": "score"}), f"PPL@{r}", "PPL", r, group)
        for r in config.DETECTORS["D2"]["references"]:
            f = lc.cache_path(f"{r}-pair", group)
            if f.exists():
                add(detectors.binoculars(pd.read_parquet(f)), f"D2@{r}", "D2", r, group)
        for d in config.TRAINED_DETECTORS:
            f = lc.cache_path(d, group)
            if f.exists():
                add(pd.read_parquet(f), d, d, None, group)

    scores = pd.concat(frames, ignore_index=True).assign(
        run_id=run_id, config_hash=config.config_hash())
    scores.to_parquet(paths.scores_file(), index=False)
    return scores


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["P", "A", "B", "C", "derive"])
    ap.add_argument("--refs", nargs="*", default=None)
    ap.add_argument("--dets", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.stage == "P":
        stage_P(a.groups)
    elif a.stage == "A":
        stage_A(a.refs or config.DETECTORS["D1"]["references"], a.groups)
    elif a.stage == "B":
        stage_B(a.refs or config.DETECTORS["D2"]["references"], a.groups)
    elif a.stage == "C":
        stage_C(a.dets or config.TRAINED_DETECTORS, a.groups)
    else:
        s = derive()
        print(s.groupby(["instance", "group"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
