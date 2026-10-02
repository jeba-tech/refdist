"""
Pre-registered out-of-sample forecast.

  commit:  fit FPR ~ log PPL_indep + length + C(instance) on S1-S9, predict each
           detector instance's FPR on S10-S12 from their PPL_indep alone, write
           prereg/forecast_S10_S12.json and print its SHA-256. Post that hash
           publicly (OSF comment) BEFORE any detector touches S10-S12. The scoring
           runner refuses to score those strata until this file exists.

  reveal:  after scoring, compare predictions with actual FPRs. The file's hash
           is re-checked first, so a forecast edited after the fact is caught.

Usage:  python -m refdist.analysis.forecast commit
        python -m refdist.analysis.forecast reveal
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import statsmodels.formula.api as smf

from refdist import config, paths, provenance
from refdist.analysis import tables
from refdist.scoring.runner import PREDICTOR_KEY

FORMULA = "fpr ~ log_ppl_indep + length + C(instance)"


def commit() -> str:
    out = paths.forecast_file()
    if out.exists():
        raise RuntimeError(f"{out} already exists; a forecast is committed once.")
    scores, docs = tables.load()
    cells = tables.cell_table(scores, docs)
    fit_cells = cells[cells["stratum"].isin(config.FIT_STRATA)]
    if set(fit_cells["stratum"]) != set(config.FIT_STRATA):
        raise RuntimeError(f"fit strata incomplete: have {sorted(set(fit_cells['stratum']))}")
    if cells["stratum"].isin(config.FORECAST_STRATA).any():
        raise RuntimeError("detector scores already exist for forecast strata: forecast would not be blind")

    fit = smf.ols(FORMULA, fit_cells).fit()
    ppl = scores[scores["instance"] == PREDICTOR_KEY].merge(docs, left_on="item_id", right_on="doc_id")
    target = ppl[ppl["stratum"].isin(config.FORECAST_STRATA)].groupby("stratum").agg(
        log_ppl_indep=("score", "mean"), length=("n_tokens", "mean")).reset_index()
    grid = target.merge(pd.DataFrame({"instance": sorted(fit_cells["instance"].unique())}), how="cross")
    grid["fpr_pred"] = fit.predict(grid).clip(0, 1)

    payload = {
        "fit_strata": config.FIT_STRATA,
        "forecast_strata": config.FORECAST_STRATA,
        "model_formula": FORMULA,
        "coefficients": fit.params.to_dict(),
        "predictions": {s: g.set_index("instance")["fpr_pred"].round(4).to_dict()
                        for s, g in grid.groupby("stratum")},
        "predictor_log_ppl": target.set_index("stratum")["log_ppl_indep"].round(4).to_dict(),
        "provenance": provenance.stamp(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    digest = provenance.sha256_file(out)
    out.with_suffix(".sha256").write_text(digest + "\n")
    return digest


def reveal() -> pd.DataFrame:
    f = paths.forecast_file()
    recorded = f.with_suffix(".sha256").read_text().strip()
    if provenance.sha256_file(f) != recorded:
        raise RuntimeError("forecast file does not match its committed hash")
    pred = json.loads(f.read_text())["predictions"]
    scores, docs = tables.load()
    cells = tables.cell_table(scores, docs)
    act = cells[cells["stratum"].isin(config.FORECAST_STRATA)][["stratum", "instance", "fpr"]]
    rows = [{"stratum": s, "instance": i, "fpr_pred": p} for s, d in pred.items() for i, p in d.items()]
    cmp = pd.DataFrame(rows).merge(act, on=["stratum", "instance"])
    cmp["abs_error"] = (cmp["fpr_pred"] - cmp["fpr"]).abs()
    return cmp


if __name__ == "__main__":
    if sys.argv[1:] == ["commit"]:
        h = commit()
        print(f"Forecast committed. SHA-256:\n  {h}\nPost this hash on OSF now, before scoring S10-S12.")
    elif sys.argv[1:] == ["reveal"]:
        c = reveal()
        print(c.round(3).to_string(index=False))
        print(f"\nMAE {c['abs_error'].mean():.3f}  (by stratum: {c.groupby('stratum')['abs_error'].mean().round(3).to_dict()})")
    else:
        print(__doc__)
