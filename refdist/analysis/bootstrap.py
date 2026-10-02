"""
Cluster bootstrap over documents for the headline coefficients.

Documents are resampled with replacement within each stratum, so every
resample keeps the design (1,000 docs per stratum) and every detector instance
sees the same resampled documents. Cell FPRs are recomputed from the resampled
flags and the H1 / H2 models refitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from refdist import config


def _cells(d: pd.DataFrame, own: pd.DataFrame) -> pd.DataFrame:
    c = d.groupby(["instance", "detector", "reference", "stratum"], dropna=False).agg(
        fpr=("flagged", "mean"), log_ppl_indep=("log_ppl_indep", "mean"),
        length=("n_tokens", "mean")).reset_index()
    return c.merge(own, on=["reference", "stratum"], how="left")


def coefficients(doc_table: pd.DataFrame, own_ppl: pd.DataFrame, n: int | None = None,
                 seed: int = 0) -> pd.DataFrame:
    """own_ppl: reference, stratum, log_ppl_ref (per-cell mean). Returns one row per resample."""
    n = n or config.ANALYSIS["bootstrap_resamples"]
    rng = np.random.default_rng(seed)
    docs_by_stratum = {s: g["item_id"].unique() for s, g in doc_table.groupby("stratum")}
    by_doc = {k: g for k, g in doc_table.groupby("item_id")}
    rows = []
    for b in range(n):
        picks = [rng.choice(ids, size=len(ids), replace=True) for ids in docs_by_stratum.values()]
        d = pd.concat([by_doc[i] for i in np.concatenate(picks)], ignore_index=True)
        c = _cells(d, own_ppl)
        row = {"b": b}
        try:
            row["h1_coef"] = smf.ols("fpr ~ log_ppl_indep + length + C(instance)", c).fit().params["log_ppl_indep"]
        except Exception:
            row["h1_coef"] = np.nan
        z = c[(c["detector"] == "D1") & c["log_ppl_ref"].notna()]
        try:
            row["h2_coef"] = smf.ols("fpr ~ log_ppl_ref + length + C(stratum) + C(reference)", z).fit().params["log_ppl_ref"]
        except Exception:
            row["h2_coef"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def interval(boot: pd.DataFrame, col: str) -> tuple[float, float]:
    v = boot[col].dropna()
    return float(v.quantile(0.025)), float(v.quantile(0.975))
