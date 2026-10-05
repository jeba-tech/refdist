"""
H1-H6 exactly as registered. Every function returns a dict of numbers plus a
verdict computed from the thresholds in config.ANALYSIS, so the confirmation
rule cannot drift from the registration.

Sign convention: the predictor is log perplexity. Lower perplexity (more
predictable text) is expected to raise the false-positive rate, so the
registered prediction for every perplexity coefficient is NEGATIVE.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from refdist import config
from refdist.metrics.flags import cohen_kappa

A = config.ANALYSIS
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def _verdict(ok: bool, fail: bool) -> str:
    return "CONFIRMED" if ok else "REJECTED" if fail else "PARTIAL"


def _marginal_r2(fit) -> float:
    """Nakagawa marginal R2: variance of fixed-effect predictions over total."""
    vf = float(np.var(fit.predict(), ddof=0)) if hasattr(fit, "predict") else 0.0
    vr = float(np.sum(np.diag(np.atleast_2d(fit.cov_re)))) if hasattr(fit, "cov_re") else 0.0
    return vf / (vf + vr + float(fit.scale))


# ── H1 unification ───────────────────────────────────────────────────────────

def h1(cells: pd.DataFrame, docs: pd.DataFrame, family: str | None = None) -> dict:
    """
    Cell level: FPR ~ log PPL_indep + length + (1 | instance); dR2 from adding PPL.
    Document level: flagged ~ log PPL_indep + length + C(instance) [+ C(stratum)],
    cluster-robust by document. Stratum identity must not absorb the perplexity
    effect: attenuation < 30% means within-stratum perplexity still predicts flags.
    """
    c = cells if family is None else cells[cells["family"] == family]
    d = docs if family is None else docs[docs["detector"].isin(c["detector"].unique())]

    full = smf.mixedlm("fpr ~ log_ppl_indep + length", c, groups=c["instance"]).fit(reml=False)
    base = smf.mixedlm("fpr ~ length", c, groups=c["instance"]).fit(reml=False)
    d_r2 = _marginal_r2(full) - _marginal_r2(base)
    coef, p = full.params["log_ppl_indep"], full.pvalues["log_ppl_indep"]

    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["item_id"])[0]})
    g0 = smf.glm("flagged ~ log_ppl_indep + n_tokens + C(instance)", d,
                 family=sm.families.Binomial()).fit(**kw)
    g1 = smf.glm("flagged ~ log_ppl_indep + n_tokens + C(instance) + C(stratum)", d,
                 family=sm.families.Binomial()).fit(**kw)
    b0, b1 = g0.params["log_ppl_indep"], g1.params["log_ppl_indep"]
    attenuation = 1 - b1 / b0 if b0 != 0 else np.nan

    ok = coef < 0 and p < A["alpha"] and d_r2 >= A["h1_min_delta_r2"] and attenuation < A["h1_max_coef_attenuation"]
    fail = not (coef < 0 and p < A["alpha"]) or d_r2 < A["h1_fail_delta_r2"]
    return {"family": family or "all", "coef_log_ppl": coef, "p": p, "delta_r2": d_r2,
            "doc_coef_without_stratum": b0, "doc_coef_with_stratum": b1,
            "doc_coef_with_stratum_p": g1.pvalues["log_ppl_indep"],
            "attenuation": attenuation, "n_cells": len(c), "verdict": _verdict(ok, fail)}


# ── H2 reference shift ───────────────────────────────────────────────────────

def h2(cells: pd.DataFrame, docs: pd.DataFrame, refs: list[str] | None = None,
       n_perm: int = 2000, seed: int = 0) -> dict:
    """
    Same text, different reference. With stratum and reference fixed effects,
    the perplexity coefficient is identified only by how much a given reference
    finds a given stratum unusually predictable -- the reference-shift effect.
    Plus a permutation test that the stratum profile of false positives differs
    across references (reference labels shuffled within each document).
    """
    c = cells[(cells["detector"] == "D1") & cells["log_ppl_ref"].notna()]
    d = docs[docs["detector"] == "D1"]
    if refs:
        c, d = c[c["reference"].isin(refs)], d[d["reference"].isin(refs)]

    present = sorted(c["reference"].unique())
    if len(present) < 2:
        # A reference swap needs two references. If the positive control removed
        # one, the test cannot run; "rejected" would misreport a missing test.
        return {"refs": refs or "all", "references_available": present,
                "verdict": "NOT ESTIMABLE",
                "reason": "fewer than two references passed the positive control"}

    fe = smf.ols("fpr ~ log_ppl_ref + length + C(stratum) + C(reference)", c).fit(cov_type="HC3")
    coef, p = fe.params["log_ppl_ref"], fe.pvalues["log_ppl_ref"]

    shift_stat, p_perm = profile_shift(d, "reference", n_perm, seed)

    profiles = c.pivot_table(index="stratum", columns="reference", values="fpr")
    tau = {f"{a}~{b}": stats.kendalltau(profiles[a], profiles[b]).statistic
           for a, b in itertools.combinations(profiles.columns, 2)}

    robust = _h2_pretraining_robustness(c)
    ok = coef < 0 and p < A["alpha"] and p_perm < A["alpha"]
    fail = p_perm >= A["alpha"]
    return {"refs": refs or "all", "coef_log_ppl_ref": coef, "p": p,
            "profile_shift_stat": shift_stat, "profile_shift_p": p_perm, "kendall_tau": tau,
            "robustness_source_in_pretraining": robust,
            "n_cells": len(c), "verdict": _verdict(ok, fail)}


def profile_shift(d: pd.DataFrame, column: str, n_perm: int, seed: int) -> tuple[float, float]:
    """
    Does the stratum profile of flags differ across `column` (references, or
    trained detectors)? Statistic: squared stratum x column interaction in flag
    rates. Null: shuffle the column labels within each document, which is
    exchangeable if every column flags a given document with equal probability.
    """
    wide = d.pivot_table(index=["item_id", "stratum"], columns=column, values="flagged").dropna()
    strata = wide.index.get_level_values("stratum").to_numpy()
    y = wide.to_numpy()

    def interaction(mat: np.ndarray) -> float:
        prof = pd.DataFrame(mat).groupby(strata).mean().to_numpy()
        prof = prof - prof.mean(0, keepdims=True) - prof.mean(1, keepdims=True) + prof.mean()
        return float((prof ** 2).sum())

    obs = interaction(y)
    rng = np.random.default_rng(seed)
    idx = np.tile(np.arange(y.shape[1]), (len(y), 1))
    null = np.array([interaction(np.take_along_axis(y, rng.permuted(idx, axis=1), 1)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (1 + n_perm)


def _h2_pretraining_robustness(c: pd.DataFrame) -> dict:
    """H2 with source_in_pretraining added; 'unknown' cells dropped from this model only."""
    from refdist.analysis.tables import source_in_pretraining
    m = c.merge(source_in_pretraining(), on=["reference", "stratum"], how="left").dropna(subset=["sip"])
    if m["sip"].nunique() < 2 or m["reference"].nunique() < 2:
        return {"status": "not estimable (covariate constant after dropping unknown cells)"}
    fit = smf.ols("fpr ~ log_ppl_ref + sip + length + C(stratum) + C(reference)", m).fit(cov_type="HC3")
    return {"coef_log_ppl_ref": fit.params["log_ppl_ref"], "p": fit.pvalues["log_ppl_ref"],
            "coef_sip": fit.params["sip"], "p_sip": fit.pvalues["sip"], "n_cells": len(m)}


# ── H3 trained detectors ─────────────────────────────────────────────────────

def h3(cells: pd.DataFrame, docs: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    """
    H1 on trained classifiers only (no scorer LM, so no construction effect),
    plus the profile-shift test across D3-D5, which differ in training corpus.
    """
    out = h1(cells, docs, family="trained")
    trained = docs[docs["detector"].isin(config.TRAINED_DETECTORS)]
    if trained["instance"].nunique() >= 2:
        stat, p = profile_shift(trained, "instance", n_perm, seed)
        out.update({"trained_profile_shift_stat": stat, "trained_profile_shift_p": p})
    return out


# ── H4 audit validity ────────────────────────────────────────────────────────

def h4(cells: pd.DataFrame) -> dict:
    """
    Leave-one-stratum-out FPR prediction; MAE in FPR units (0.10 = 10 points).

    Low MAE alone proves nothing: if FPR barely varies, any model is "accurate".
    The perplexity model must also beat a baseline that knows each detector's
    average FPR but not perplexity (skill = 1 - MAE_model / MAE_baseline).
    """
    errs, base_errs = [], []
    for s in cells["stratum"].unique():
        train, test = cells[cells["stratum"] != s], cells[cells["stratum"] == s]
        fit = smf.ols("fpr ~ log_ppl_indep + length + C(instance)", train).fit()
        base = smf.ols("fpr ~ length + C(instance)", train).fit()
        errs.append(np.abs(fit.predict(test) - test["fpr"]))
        base_errs.append(np.abs(base.predict(test) - test["fpr"]))
    mae, mae_base = float(pd.concat(errs).mean()), float(pd.concat(base_errs).mean())
    skill = 1 - mae / mae_base if mae_base > 0 else 0.0
    ok = mae <= A["h4_max_mae_confirm"] and skill >= A["h4_min_skill"]
    fail = mae > A["h4_max_mae_partial"] or skill <= 0
    return {"mae": mae, "mae_baseline": mae_base, "skill": skill, "verdict": _verdict(ok, fail)}


# ── H5 text-side intervention ────────────────────────────────────────────────

def h5_deltas(scores: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (instance, variant): change in detector score and in log PPL_indep
    relative to the untouched source document. Score changes are divided by the
    instance's score SD on the calibration stratum so detectors share one scale.
    """
    from refdist.scoring.runner import PREDICTOR_KEY
    det = scores[scores["detector"].str.match(r"^D\d$")]
    src = det[det["group"] == config.CALIBRATION_STRATUM].set_index(["instance", "item_id"])["score"]
    sd = det[det["group"] == config.CALIBRATION_STRATUM].groupby("instance")["score"].std()
    var = det[det["group"] == "variants"].merge(
        variants[["variant_id", "doc_id", "transform", "intensity"]], left_on="item_id", right_on="variant_id")
    var["score_src"] = src.reindex(list(zip(var["instance"], var["doc_id"]))).to_numpy()
    var["dz"] = (var["score"] - var["score_src"]) / var["instance"].map(sd)

    ppl = scores[scores["instance"] == PREDICTOR_KEY].set_index("item_id")["score"]
    var["dlogppl"] = var["item_id"].map(ppl).to_numpy() - var["doc_id"].map(ppl).to_numpy()
    return var.dropna(subset=["dz", "dlogppl"])


def h5(deltas: pd.DataFrame) -> dict:
    """
    Per detector instance: dz ~ dlogppl + (1 + dlogppl | transform). A common
    slope (small random-slope SD relative to the slope) means the detector
    responds to the perplexity change, whichever edit produced it.
    """
    out = {}
    for inst, g in deltas.groupby("instance"):
        fit = smf.mixedlm("dz ~ dlogppl", g, groups=g["transform"], re_formula="~dlogppl").fit(reml=True)
        slope, p = fit.params["dlogppl"], fit.pvalues["dlogppl"]
        sd_slope = float(np.sqrt(max(fit.cov_re.iloc[1, 1], 0)))
        ratio = sd_slope / abs(slope) if slope else np.inf
        per_t = {t: smf.ols("dz ~ dlogppl", gt).fit().params["dlogppl"] for t, gt in g.groupby("transform")}
        ok = slope < 0 and p < A["alpha"] and ratio <= A["h5_max_random_slope_ratio"]
        out[inst] = {"slope": slope, "p": p, "random_slope_sd": sd_slope, "sd_ratio": ratio,
                     "per_transform_slope": per_t, "n": len(g),
                     "verdict": _verdict(ok, not (slope < 0 and p < A["alpha"]))}
    return out


# ── H6 agreement structure ───────────────────────────────────────────────────

def h6(docs: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> dict:
    """Detector instances sharing a reference distribution flag the same documents."""
    wide = docs.pivot_table(index="item_id", columns="instance", values="flagged").dropna()
    inst = list(wide.columns)
    fam = {i: config.reference_family(i.split("@")[0], i.split("@")[1] if "@" in i else None) for i in inst}
    pairs = [(a, b, cohen_kappa(wide[a].to_numpy(), wide[b].to_numpy()))
             for a, b in itertools.combinations(inst, 2)]
    k = np.array([p[2] for p in pairs])

    def gap(f):
        same = np.array([f[a] == f[b] for a, b, _ in pairs])
        return k[same].mean() - k[~same].mean() if same.any() and (~same).any() else np.nan

    obs = gap(fam)
    rng = np.random.default_rng(seed)
    labels = list(fam.values())
    null = np.array([gap(dict(zip(inst, rng.permutation(labels)))) for _ in range(n_perm)])
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    return {"kappa_same_minus_diff": obs, "p": p,
            "pairs": [{"a": a, "b": b, "kappa": kk, "same_family": fam[a] == fam[b]} for a, b, kk in pairs],
            "verdict": _verdict(obs > 0 and p < A["alpha"], not (obs > 0 and p < A["alpha"]))}


def h5_overall(per_instance: dict, included: list[str]) -> dict:
    """
    Registered aggregate: H5 holds if confirmed for at least half of the detector
    instances that passed the positive control. Slope p-values are Holm-adjusted
    across those instances before the per-instance verdicts are recounted.
    """
    from statsmodels.stats.multitest import multipletests
    inst = [i for i in included if i in per_instance]
    if not inst:
        return {"verdict": "NOT RUN", "n_instances": 0}
    p_adj = multipletests([per_instance[i]["p"] for i in inst], method="holm")[1]
    confirmed = []
    for i, pa in zip(inst, p_adj):
        r = per_instance[i]
        r["p_holm"] = float(pa)
        if r["slope"] < 0 and pa < A["alpha"] and r["sd_ratio"] <= A["h5_max_random_slope_ratio"]:
            confirmed.append(i)
    share = len(confirmed) / len(inst)
    return {"confirmed_instances": confirmed, "share": share, "n_instances": len(inst),
            "verdict": "CONFIRMED" if share >= 0.5 else "REJECTED"}
