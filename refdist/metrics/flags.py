"""
Thresholds, false-positive rates and agreement.

Every detector instance (detector x reference) gets one threshold, set so that
TARGET_FPR of the calibration stratum scores strictly above it. Thresholds are
written once and never refit: refitting after other strata are scored would
let their results leak into the operating point.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from refdist import config, paths, provenance


def instance_key(detector: str, reference: str | None) -> str:
    return f"{detector}@{reference}" if reference else detector


def calibrate(scores: pd.DataFrame, documents: pd.DataFrame) -> dict[str, float]:
    """scores: item_id, instance, score. Threshold per instance on the calibration stratum."""
    cal_ids = set(documents.loc[documents["stratum"] == config.CALIBRATION_STRATUM, "doc_id"])
    cal = scores[scores["item_id"].isin(cal_ids)]
    q = 1.0 - config.TARGET_FPR
    return {inst: float(np.quantile(g["score"].to_numpy(), q, method="higher"))
            for inst, g in cal.groupby("instance")}


def freeze_thresholds(thresholds: dict[str, float]) -> None:
    f = paths.thresholds_file()
    if f.exists():
        old = json.loads(f.read_text())["thresholds"]
        changed = {k for k in thresholds if k in old and not np.isclose(old[k], thresholds[k])}
        if changed:
            raise RuntimeError(f"thresholds already frozen and would change for {sorted(changed)}")
        thresholds = {**thresholds, **old}
    provenance.write_json(f, {"target_fpr": config.TARGET_FPR,
                              "calibration_stratum": config.CALIBRATION_STRATUM,
                              "thresholds": thresholds})


def load_thresholds() -> dict[str, float]:
    return json.loads(paths.thresholds_file().read_text())["thresholds"]


def flag(scores: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    t = scores["instance"].map(thresholds)
    return scores.assign(threshold=t, flagged=(scores["score"] > t).astype(np.int8))


def fpr_table(flags: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """FPR per (instance, stratum) with a Wilson 95% interval."""
    df = flags.merge(documents[["doc_id", "stratum"]], left_on="item_id", right_on="doc_id")
    g = df.groupby(["instance", "stratum"])["flagged"].agg(["sum", "count"]).reset_index()
    p = g["sum"] / g["count"]
    z = 1.96
    denom = 1 + z**2 / g["count"]
    centre = (p + z**2 / (2 * g["count"])) / denom
    half = z * np.sqrt(p * (1 - p) / g["count"] + z**2 / (4 * g["count"] ** 2)) / denom
    return g.assign(fpr=p, fpr_lo=centre - half, fpr_hi=centre + half)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def kappa_matrix(flags: pd.DataFrame) -> pd.DataFrame:
    wide = flags.pivot_table(index="item_id", columns="instance", values="flagged").dropna()
    inst = list(wide.columns)
    rows = []
    for i, x in enumerate(inst):
        for y in inst[i + 1:]:
            rows.append({"a": x, "b": y,
                         "kappa": cohen_kappa(wide[x].to_numpy(), wide[y].to_numpy())})
    return pd.DataFrame(rows)
