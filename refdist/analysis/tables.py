"""
The analysis tables every hypothesis draws from, built once from scores.parquet.

  cell table   one row per (detector instance, stratum): FPR, mean PPL_indep,
               mean PPL under the instance's own reference, mean length
  doc table    one row per (instance, document): score, flag, PPL_indep

Detector instances are the rows of H1-H4 and H6; PPL instances (PPL@R*) and the
independent predictor are covariates, never outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from refdist import config, paths
from refdist.metrics import flags as fl
from refdist.scoring.runner import PREDICTOR_KEY


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_parquet(paths.scores_file())
    docs = pd.read_parquet(paths.documents_file(), columns=["doc_id", "stratum", "n_tokens"])
    return scores, docs


def detector_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[scores["detector"].str.match(r"^D\d$")]


def doc_table(scores: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    thresholds = fl.load_thresholds()
    det = detector_scores(scores)
    # Instances without a frozen threshold failed the positive control. Keeping
    # them would compare scores to NaN and report a false 0% FPR.
    det = det[det["item_id"].isin(docs["doc_id"]) & det["instance"].isin(thresholds)]
    flagged = fl.flag(det, thresholds)
    indep = scores.loc[scores["instance"] == PREDICTOR_KEY, ["item_id", "score"]].rename(
        columns={"score": "log_ppl_indep"})
    return (flagged.merge(indep, on="item_id", how="left")
                   .merge(docs, left_on="item_id", right_on="doc_id"))


def cell_table(scores: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    d = doc_table(scores, docs)
    cells = d.groupby(["instance", "detector", "reference", "stratum"], dropna=False).agg(
        fpr=("flagged", "mean"), n=("flagged", "size"),
        log_ppl_indep=("log_ppl_indep", "mean"), length=("n_tokens", "mean"),
    ).reset_index()
    own = scores[scores["detector"] == "PPL"].merge(docs, left_on="item_id", right_on="doc_id")
    own = own.groupby(["reference", "stratum"])["score"].mean().rename("log_ppl_ref").reset_index()
    cells = cells.merge(own, on=["reference", "stratum"], how="left")
    cells["family"] = np.where(cells["detector"].isin(config.TRAINED_DETECTORS), "trained", "zero_shot")
    cells["logit_fpr"] = _logit(cells["fpr"], cells["n"])
    return cells


def _logit(p: pd.Series, n: pd.Series) -> pd.Series:
    """Empirical logit with a half-count correction, finite at FPR = 0 or 1."""
    k = p * n
    return np.log((k + 0.5) / (n - k + 0.5))


def source_in_pretraining() -> pd.DataFrame:
    """Pre-declared covariate, coded from published data documentation (prereg/source_overlap.csv).
    "unknown" cells are returned as NaN and dropped from the robustness model only."""
    f = paths.forecast_file().parent / "source_overlap.csv"
    t = pd.read_csv(f, dtype=str)
    t["sip"] = pd.to_numeric(t["source_in_pretraining"], errors="coerce")
    return t[["reference", "stratum", "sip"]]
