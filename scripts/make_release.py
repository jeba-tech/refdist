"""
Builds the derived-statistics release promised in the registration (Section 10):
scores, perplexities and false-positive rates, with no text from any source.

  release/
    README.md                       what each file is, and what is left out
    document_scores.csv.gz          one row per human document: length, perplexities,
                                    all ten detector scores, registered flags
    edit_scores.csv.gz              one row per H5 edited document (S6 only)
    control_scores.csv.gz           the 500 machine-written positive-control texts
    population_fpr.csv              FPR per instance x population, Wilson 95% interval
    population_summary.csv          per population: perplexities, length, length bands
    forecast_reveal.csv             the pre-committed S10-S12 forecast against observed FPRs
    results/                        registered results, thresholds, validity, audit model,
                                    exploratory results, scoring ledger
    SHA256SUMS                      checksums of every file above

Usage:  REFDIST_DATA=data/kaggle/phase2/data python scripts/make_release.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from refdist import paths  # noqa: E402
from refdist.analysis import tables  # noqa: E402
from refdist.analysis.forecast import reveal  # noqa: E402

OUT = ROOT / "release"
INSTANCES = ["D1@R1", "D1@R2", "D1@R3", "D1@R4", "D1@R5", "D2@R4", "D2@R5", "D3", "D4", "D5"]
REFS = ["R1", "R2", "R3", "R4", "R5"]
FORBIDDEN = {"text", "sites", "source_doc_id", "meta"}  # text, edit snippets, source identifiers

R = json.loads((paths.results() / "results.json").read_text())
EXP = json.loads((ROOT / "data" / "results" / "exploratory_all_instances.json").read_text())
THR = json.loads((paths.results() / "thresholds.json").read_text())["thresholds"]
INCLUDED = R["included_instances"]

scores, docs = tables.load()
# one column per scorer: detector instances, the independent predictor, each reference's log-PPL
scores["key"] = scores["instance"]
scores.loc[scores["detector"] == "PPL", "key"] = "log_ppl_" + scores["reference"]
scores.loc[scores["detector"] == "PPL_indep", "key"] = "log_ppl_indep"
wide = scores.pivot_table(index=["item_id", "group"], columns="key", values="score").reset_index()
wide.columns.name = None
score_cols = ["log_ppl_indep"] + [f"log_ppl_{r}" for r in REFS] + INSTANCES


def flags(df: pd.DataFrame) -> pd.DataFrame:
    # registered flags: score above the frozen S6 threshold, included instances only
    for i in INCLUDED:
        df[f"flag_{i}"] = (df[i] > THR[i]).astype("int8")
    return df


# ---- documents -------------------------------------------------------------------------
d = wide[~wide["group"].isin(["control", "variants"])].rename(columns={"item_id": "doc_id"})
meta = pd.read_parquet(ROOT / "data" / "processed" / "documents.parquet",  # metadata columns only, never text
                       columns=["doc_id", "stratum", "source", "year", "n_tokens", "license"])
d = d.merge(meta, on="doc_id", how="left", validate="one_to_one")
assert (d["group"] == d["stratum"]).all() and len(d) == 12_000
d = flags(d.rename(columns={"stratum": "population", "n_tokens": "n_tokens_gpt2"}))
d = d[["doc_id", "population", "source", "year", "license", "n_tokens_gpt2"] + score_cols
      + [f"flag_{i}" for i in INCLUDED]]
d = d.assign(_o=d["population"].str[1:].astype(int)).sort_values(["_o", "doc_id"]).drop(columns="_o")

# ---- H5 edited documents ------------------------------------------------------------
var = pd.read_parquet(paths.variants_file())
e = wide[wide["group"] == "variants"].rename(columns={"item_id": "variant_id"}).drop(columns="group")
e = var[["variant_id", "doc_id", "transform", "intensity", "n_sites_eligible", "n_sites_applied", "n_tokens"]] \
    .rename(columns={"n_tokens": "n_tokens_gpt2"}).merge(e, on="variant_id", how="inner", validate="one_to_one")
assert len(e) == len(var)
e = e[[c for c in e.columns if c not in INSTANCES or e[c].notna().any()]]

# ---- positive control -----------------------------------------------------------------
c = wide[wide["group"] == "control"].rename(columns={"item_id": "control_id"}).drop(columns="group")
c = c[[col for col in c.columns if col == "control_id" or c[col].notna().any()]]

# ---- population tables ------------------------------------------------------------------
rows = []
for i in INSTANCES:
    registered = i in INCLUDED
    for s, g in d.groupby("population"):
        if registered:
            k, n = int(g[f"flag_{i}"].sum()), len(g)
            fpr = k / n
        else:  # exploratory threshold (same rule, scratch copy): rate only
            fpr, n = EXP["fpr_table"][i][s], len(g)
            k = round(fpr * n)
        lo, hi = proportion_confint(k, n, method="wilson")
        rows.append({"instance": i, "population": s, "n": n, "n_flagged": k, "fpr": round(fpr, 4),
                     "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                     "status": "registered" if registered else "exploratory (failed positive control)"})
fpr_tab = pd.DataFrame(rows)

man = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())["strata"]
summ = d.groupby("population").agg(n=("doc_id", "size"), mean_tokens=("n_tokens_gpt2", "mean"),
                                   **{f"mean_{col}": (col, "mean") for col in score_cols[:6]}).round(4)
for b in range(5):
    summ[f"band{b}_docs"] = [man[s]["length_matching"]["per_bin"][str(b)] for s in summ.index]
summ = summ.reset_index()
summ = summ.assign(_o=summ["population"].str[1:].astype(int)).sort_values("_o").drop(columns="_o")

fc = reveal()[["stratum", "instance", "fpr_pred", "fpr", "abs_error"]] \
    .rename(columns={"stratum": "population", "fpr": "fpr_observed"}).round(4)

# ---- safety checks: nothing that is or contains source text ---------------------------
for name, df in {"documents": d, "edits": e, "control": c}.items():
    assert not FORBIDDEN & set(df.columns), f"{name}: forbidden column"
    for col in [k for k in df.columns if not pd.api.types.is_numeric_dtype(df[k])]:
        longest = df[col].astype(str).str.len().max()
        assert longest <= 80, f"{name}.{col}: values up to {longest} chars, may be text"

# ---- write -----------------------------------------------------------------------------
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "results").mkdir(parents=True)
d.to_csv(OUT / "document_scores.csv.gz", index=False, float_format="%.6g")
e.to_csv(OUT / "edit_scores.csv.gz", index=False, float_format="%.6g")
c.to_csv(OUT / "control_scores.csv.gz", index=False, float_format="%.6g")
fpr_tab.to_csv(OUT / "population_fpr.csv", index=False)
summ.to_csv(OUT / "population_summary.csv", index=False)
fc.to_csv(OUT / "forecast_reveal.csv", index=False)
for f in ("results.json", "thresholds.json", "validity.json", "precision.json", "audit_model.json", "ledger.jsonl"):
    shutil.copy(paths.results() / f, OUT / "results" / f)
shutil.copy(ROOT / "data" / "results" / "exploratory_all_instances.json", OUT / "results")
shutil.copy(ROOT / "scripts" / "release_README.md", OUT / "README.md")

sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}"
        for p in sorted(OUT.rglob("*")) if p.is_file()]
(OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print(f"wrote {len(sums)} files to {OUT}")
print(f"documents {d.shape}, edits {e.shape}, control {c.shape}, fpr rows {len(fpr_tab)}")
