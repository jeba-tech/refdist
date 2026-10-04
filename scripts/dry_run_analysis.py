"""
Dry run of the full analysis on SYNTHETIC scores with a known ground truth.

Documents get a latent log-perplexity per reference model; every detector's
score is built to fall as perplexity rises. If H1, H2, H5 and the forecast
fail to recover that planted effect, the analysis code is wrong. A second run
with the effect removed must NOT confirm anything.

Nothing here touches real data: it writes to a temporary data root.

Usage:  python scripts/dry_run_analysis.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def synth(effect: float, seed: int = 0) -> None:
    from refdist import config, paths
    from refdist.metrics import flags as fl
    from refdist.scoring.runner import PREDICTOR_KEY

    rng = np.random.default_rng(seed)
    strata = list(config.STRATA)
    base = {s: rng.normal(3.0, 0.4) for s in strata}                 # stratum typicality
    refs = config.DETECTORS["D1"]["references"]
    ref_shift = {(r, s): rng.normal(0, 0.25) for r in refs for s in strata}  # reference-specific familiarity

    docs, rows = [], []
    for s in strata:
        for i in range(300):
            did = f"{s}-{i:04d}"
            lp = base[s] + rng.normal(0, 0.35)
            n_tok = int(rng.integers(150, 401))
            docs.append({"doc_id": did, "stratum": s, "n_tokens": n_tok, "text": "x"})
            rows.append({"item_id": did, "instance": PREDICTOR_KEY, "detector": PREDICTOR_KEY,
                         "reference": None, "group": s, "score": lp + rng.normal(0, 0.05)})
            for r in refs:
                lpr = lp + ref_shift[(r, s)]
                rows.append({"item_id": did, "instance": f"PPL@{r}", "detector": "PPL",
                             "reference": r, "group": s, "score": lpr})
                rows.append({"item_id": did, "instance": f"D1@{r}", "detector": "D1", "reference": r,
                             "group": s, "score": -effect * lpr + rng.normal(0, 1)})
            for r in config.DETECTORS["D2"]["references"]:
                lpr = lp + ref_shift[(r, s)]
                rows.append({"item_id": did, "instance": f"D2@{r}", "detector": "D2", "reference": r,
                             "group": s, "score": -effect * lpr + rng.normal(0, 1)})
            for d in config.TRAINED_DETECTORS:
                rows.append({"item_id": did, "instance": d, "detector": d, "reference": None,
                             "group": s, "score": -effect * 0.7 * lp + rng.normal(0, 1)})

    # H5 variants of S6 documents: each transform shifts perplexity by its own amount.
    s6 = [d["doc_id"] for d in docs if d["stratum"] == config.CALIBRATION_STRATUM][:120]
    lp_of = {r["item_id"]: r["score"] for r in rows if r["instance"] == PREDICTOR_KEY}
    var_rows = []
    for t_i, t in enumerate(config.TRANSFORMS):
        for did in s6:
            for p in (0.25, 0.5, 1.0):
                vid = f"{did}|{t}|{p:.2f}"
                dlp = (t_i - 2.5) * 0.08 * p + rng.normal(0, 0.02)
                var_rows.append({"variant_id": vid, "doc_id": did, "transform": t, "intensity": p, "text": "x"})
                rows.append({"item_id": vid, "instance": PREDICTOR_KEY, "detector": PREDICTOR_KEY,
                             "reference": None, "group": "variants", "score": lp_of[did] + dlp})
                for inst in [f"D1@{r}" for r in refs] + [f"D2@{r}" for r in config.DETECTORS["D2"]["references"]] + list(config.TRAINED_DETECTORS):
                    det = inst.split("@")[0]
                    rows.append({"item_id": vid, "instance": inst, "detector": det,
                                 "reference": inst.split("@")[1] if "@" in inst else None,
                                 "group": "variants", "score": np.nan, "_src": did, "_dlp": dlp})

    scores = pd.DataFrame(rows)
    # Variant scores = source score shifted by the planted effect on the perplexity change.
    src_score = scores[scores["group"] != "variants"].set_index(["instance", "item_id"])["score"]
    m = scores["_src"].notna()
    keys = list(zip(scores.loc[m, "instance"], scores.loc[m, "_src"]))
    scale = np.where(scores.loc[m, "detector"].isin(config.TRAINED_DETECTORS), 0.7, 1.0)
    scores.loc[m, "score"] = src_score.reindex(keys).to_numpy() - effect * scale * scores.loc[m, "_dlp"].to_numpy() \
        + rng.normal(0, 0.05, m.sum())
    scores = scores.drop(columns=["_src", "_dlp"])

    # Positive control: machine text scores high. D5 is planted with its label
    # backwards (scores low) to exercise the registered flip-once rule.
    ctrl = []
    insts = scores.loc[scores["detector"].str.match(r"^D\d$"), "instance"].unique()
    for inst in insts:
        det = inst.split("@")[0]
        sign = 1
        s6 = scores[(scores["instance"] == inst) & (scores["group"] == config.CALIBRATION_STRATUM)]["score"]
        for i in range(200):
            ctrl.append({"item_id": f"C-{i:04d}", "instance": inst, "detector": det,
                         "reference": inst.split("@")[1] if "@" in inst else None, "group": "control",
                         "score": s6.mean() + sign * 3 * s6.std() + rng.normal(0, s6.std())})
    scores = pd.concat([scores, pd.DataFrame(ctrl)], ignore_index=True)
    # A detector with a backwards label is backwards on EVERY text, so D5 is
    # inverted everywhere; the registered flip-once rule must restore it exactly.
    d5 = scores["detector"] == "D5"
    scores.loc[d5, "score"] = 1 - scores.loc[d5, "score"]

    # Real scores carry each scorer's own token count; Phase 1 crashed on the
    # resulting column clash, so the synthetic data must carry it too.
    scores["n_tokens"] = rng.integers(140, 420, len(scores))

    pd.DataFrame(docs).to_parquet(paths.documents_file(), index=False)
    pd.DataFrame(var_rows).to_parquet(paths.variants_file(), index=False)
    scores.to_parquet(paths.scores_file(), index=False)


def run(effect: float) -> dict:
    tmp = tempfile.mkdtemp(prefix="refdist_dry_")
    os.environ["REFDIST_DATA"] = tmp
    for mod in [m for m in sys.modules if m.startswith("refdist")]:
        del sys.modules[mod]
    from refdist.analysis import report
    synth(effect)
    return report.run(n_boot=40)


def summarise(tag: str, r: dict) -> None:
    print()
    print(f"=== {tag} ===")
    g = r["validity"]
    pc = {x["instance"]: (round(x["auroc"], 2), "FLIPPED" if x["flipped"] else "") for x in g["positive_control"]}
    print("positive control:", {k: v for k, v in pc.items() if k in ("D1@R1", "D3", "D5")}, "| excluded:", g["excluded_instances"])
    print("negative control S6 FPR:", sorted({round(x["fpr"], 3) for x in g["negative_control"]}))
    h = r["H1"]; print(f"H1  coef {h['coef_log_ppl']:+.3f} p={h['p']:.2g} dR2={h['delta_r2']:.2f} atten={h['attenuation']:.2f} -> {h['verdict']}")
    for k in ("H2", "H2_matched_R2_R3"):
        h = r[k]; print(f"{k:<16} coef {h['coef_log_ppl_ref']:+.3f} p={h['p']:.2g} shift p={h['profile_shift_p']:.3f} -> {h['verdict']}")
    h = r["H3"]; print(f"H3  coef {h['coef_log_ppl']:+.3f} dR2={h['delta_r2']:.2f} trained-shift p={h.get('trained_profile_shift_p', float('nan')):.3f} -> {h['verdict']}")
    h = r["H4"]; print(f"H4  MAE {h['mae']:.3f} skill {h['skill']:+.2f} -> {h['verdict']}")
    h = r["H5"]["overall"]; print(f"H5  {len(h['confirmed_instances'])}/{h['n_instances']} instances -> {h['verdict']}")
    h = r["H6"]; print(f"H6  kappa gap {h['kappa_same_minus_diff']:+.3f} p={h['p']:.3f} -> {h['verdict']}")
    print("bootstrap 95% CI  H1", tuple(round(x, 3) for x in r["bootstrap"]["h1_coef_95ci"]),
          " H2", tuple(round(x, 3) for x in r["bootstrap"]["h2_coef_95ci"]))
    print("DECISION:", r["decision"]["scorer_distance_hypothesis"])


if __name__ == "__main__":
    summarise("PLANTED EFFECT (should confirm)", run(effect=2.0))
    summarise("NO EFFECT (should not confirm)", run(effect=0.0))
