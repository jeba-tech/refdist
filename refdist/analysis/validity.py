"""
The three validity gates from the registration, run before any hypothesis.

  positive control   each detector instance must separate generated control text
                     from S6 at AUROC >= 0.80, or it is excluded. RADAR's label
                     orientation is undocumented: if its AUROC is below 0.5 its
                     label is flipped ONCE, the flip is recorded, and it must then
                     clear the same bar. No other detector may be flipped.
  negative control   S6 FPR must be 5% +/- 0.5% for every instance (thresholds
                     were set on S6, so anything else means the pipeline is wrong).
  length invariance  within S6, FPR must not differ by more than 3 points across
                     length quartiles.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from refdist import config, paths, provenance
from refdist.metrics import flags as fl

FLIPPABLE = {"D5"}  # undocumented label orientation (RADAR)


def positive_control(scores: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    s6_ids = set(docs.loc[docs["stratum"] == config.CALIBRATION_STRATUM, "doc_id"])
    det = scores[scores["detector"].str.match(r"^D\d$")]
    rows = []
    for inst, g in det.groupby("instance"):
        human = g[g["item_id"].isin(s6_ids)]["score"].to_numpy()
        machine = g[g["group"] == "control"]["score"].to_numpy()
        if len(machine) == 0:
            continue
        y = np.r_[np.zeros(len(human)), np.ones(len(machine))]
        auc = roc_auc_score(y, np.r_[human, machine])
        det_key = inst.split("@")[0]
        flipped = det_key in FLIPPABLE and auc < 0.5
        if flipped:
            auc = 1 - auc
        rows.append({"instance": inst, "auroc": auc, "flipped": flipped,
                     "passed": auc >= config.CONTROL["min_auroc"],
                     "n_human": len(human), "n_control": len(machine)})
    return pd.DataFrame(rows)


def apply_flips(scores: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    """P(AI) -> 1 - P(AI) for the recorded flips; nothing else is touched."""
    flip = set(gate.loc[gate["flipped"], "instance"])
    if not flip:
        return scores
    m = scores["instance"].isin(flip)
    return scores.assign(score=np.where(m, 1 - scores["score"], scores["score"]))


def negative_control(flags: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    t = fl.fpr_table(flags, docs)
    t = t[t["stratum"] == config.CALIBRATION_STRATUM][["instance", "fpr"]]
    return t.assign(passed=(t["fpr"] - config.TARGET_FPR).abs() <= 0.005)


def length_invariance(flags: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    d = flags.merge(docs[["doc_id", "stratum", "n_tokens"]], left_on="item_id", right_on="doc_id")
    d = d[d["stratum"] == config.CALIBRATION_STRATUM]
    d = d.assign(quartile=pd.qcut(d["n_tokens"], 4, labels=False, duplicates="drop"))
    q = d.groupby(["instance", "quartile"])["flagged"].mean().unstack()
    spread = q.max(axis=1) - q.min(axis=1)
    return pd.DataFrame({"instance": spread.index, "fpr_spread": spread.values,
                         "passed": spread.values <= config.ANALYSIS["length_quartile_fpr_margin"]})


def run(scores: pd.DataFrame, docs: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Returns (scores with recorded flips applied, instances that passed the
    positive control, report). Thresholds are calibrated on S6 and frozen here,
    AFTER the orientation decision and BEFORE any other stratum is examined.
    """
    gate = positive_control(scores, docs)
    if gate.empty:
        raise RuntimeError("no control scores: run generate_control and score the 'control' group first")
    scores = apply_flips(scores, gate)
    passed = gate.loc[gate["passed"], "instance"].tolist()
    if not passed:
        raise RuntimeError("positive control failed for every detector: the study stops here (registered gate)")

    det = scores[scores["instance"].isin(passed) & (scores["group"] != "control")
                 & (scores["group"] != "variants")]
    fl.freeze_thresholds(fl.calibrate(det, docs))
    flags = fl.flag(det, fl.load_thresholds())
    report = {
        "positive_control": gate.to_dict("records"),
        "negative_control": negative_control(flags, docs).to_dict("records"),
        "length_invariance": length_invariance(flags, docs).to_dict("records"),
        "included_instances": passed,
        "excluded_instances": gate.loc[~gate["passed"], "instance"].tolist(),
    }
    provenance.write_json(paths.results() / "validity.json", report)
    return scores, passed, report
