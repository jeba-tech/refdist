"""
Registered follow-up E: is the reference-swap effect carried by the reference's entropy?

Fast-DetectGPT (analytic, sampling model = scoring model) scores a document as

    (sum logp - sum mu) / sqrt(sum var)  =  sqrt(n) * (H - NLL) / sqrt(V)

with, per document under reference r:
    NLL = -mean logp   the reference's log-perplexity on the text (registered H2 predictor)
    H   = -mean mu     the reference's entropy: the log-loss it expects from its own samples
    V   =  mean var    the per-token variance of that expectation
    n   =  tokens      the reference tokenizer's token count

The registered H2 test found that NLL does not explain how swapping R3 -> R4 changes
which populations are flagged. This module asks which of the four components does.
For every subset T of components it scores each document with T taken from the
target reference and the rest from the base reference, re-sets the threshold on S6
by the registered rule (5% FPR, numpy quantile 'higher', flag if score > threshold),
and measures how far the FPR profile over the other 11 populations has moved
towards the target's profile. Shapley values over the components split that
movement into shares. Binoculars (score = -NLL_perf / X) is decomposed the same way
into its two components.

Usage:
    python -m refdist.analysis.entropy_followup --synthetic   # validation on simulated data
    python -m refdist.analysis.entropy_followup               # the registered analysis;
        refuses to run until prereg/followup_entropy_osf.txt holds the OSF registration link
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from refdist import config, paths, provenance

FD_COMPONENTS = ("nll", "entropy", "var", "n")
BINO_COMPONENTS = ("nll", "xent")
CAL = config.CALIBRATION_STRATUM
POPS = [f"S{i}" for i in range(1, 13)]
REG_FILE = Path(__file__).resolve().parents[2] / "prereg" / "followup_entropy_osf.txt"
SEED = config.CORPUS["seed"]
N_BOOT = config.ANALYSIS["bootstrap_resamples"]


# ---- scores from components ----------------------------------------------------------
def fd_score(c: dict[str, np.ndarray]) -> np.ndarray:
    return np.sqrt(c["n"]) * (c["entropy"] - c["nll"]) / np.sqrt(np.clip(c["var"], 1e-8, None))


def bino_score(c: dict[str, np.ndarray]) -> np.ndarray:
    return -c["nll"] / np.clip(c["xent"], 1e-8, None)


# ---- FPR profile with the registered threshold rule ------------------------------------
def fpr_profile(score: np.ndarray, pop: np.ndarray, others: list[str]) -> np.ndarray:
    t = np.quantile(score[pop == CAL], 1.0 - config.TARGET_FPR, method="higher")
    flagged = score > t
    return np.array([flagged[pop == k].mean() for k in others])


def decompose(base: dict, target: dict, pop: np.ndarray, score_fn, comps: tuple[str, ...]) -> dict:
    """Shapley shares of each component in moving the FPR profile from base towards target."""
    others = [k for k in POPS if k != CAL and (pop == k).any()]
    prof = {}
    for r in range(len(comps) + 1):
        for T in itertools.combinations(comps, r):
            mixed = {c: (target[c] if c in T else base[c]) for c in comps}
            prof[frozenset(T)] = fpr_profile(score_fn(mixed), pop, others)
    goal = prof[frozenset(comps)]
    dist = {T: np.abs(p - goal).sum() for T, p in prof.items()}
    total = dist[frozenset()]                       # base-to-target profile distance
    v = {T: total - d for T, d in dist.items()}     # how much of it T closes
    m = len(comps)
    phi = {}
    for x in comps:
        rest = [c for c in comps if c != x]
        phi[x] = sum(math.factorial(len(S)) * math.factorial(m - len(S) - 1) / math.factorial(m)
                     * (v[frozenset(S) | {x}] - v[frozenset(S)])
                     for r in range(m) for S in itertools.combinations(rest, r))
    share = {x: (phi[x] / total if total > 0 else float("nan")) for x in comps}
    single = {x: (v[frozenset([x])] / total if total > 0 else float("nan")) for x in comps}
    return {"total_distance": float(total), "share": share, "single_swap_share": single}


def bootstrap(base: dict, target: dict, pop: np.ndarray, score_fn, comps, n: int = N_BOOT,
              seed: int = SEED) -> pd.DataFrame:
    """Resample documents within each population; base and target use the same documents."""
    rng = np.random.default_rng(seed)
    idx = {k: np.flatnonzero(pop == k) for k in np.unique(pop)}
    rows = []
    for _ in range(n):
        pick = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in idx.values()])
        r = decompose({c: base[c][pick] for c in comps}, {c: target[c][pick] for c in comps},
                      pop[pick], score_fn, comps)
        rows.append(r["share"])
    return pd.DataFrame(rows)


def ci(x: pd.Series) -> list[float]:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


# ---- registered verdicts -------------------------------------------------------------
def verdict_e1(point: dict, boot: pd.DataFrame) -> dict:
    """Confirmed: entropy share > 0.5 and its 95% CI lead over every other component excludes 0.
    Rejected: entropy share <= 0.25 (no more than an equal split of four). Else partial."""
    s = point["share"]
    leads = {x: ci(boot["entropy"] - boot[x]) for x in FD_COMPONENTS if x != "entropy"}
    if s["entropy"] > 0.5 and all(lo > 0 for lo, _ in leads.values()):
        v = "CONFIRMED"
    elif s["entropy"] <= 0.25:
        v = "REJECTED"
    else:
        v = "PARTIAL"
    return {"verdict": v, "entropy_share": s["entropy"], "entropy_share_ci": ci(boot["entropy"]),
            "entropy_lead_ci": leads}


def verdict_e2(point: dict, boot: pd.DataFrame) -> dict:
    """Confirmed: normaliser share > 0.5 and the 95% CI of (xent - nll) share lies above 0.
    Rejected: that CI lies below 0. Else partial."""
    lo, hi = ci(boot["xent"] - boot["nll"])
    v = "CONFIRMED" if point["share"]["xent"] > 0.5 and lo > 0 else "REJECTED" if hi < 0 else "PARTIAL"
    return {"verdict": v, "xent_share": point["share"]["xent"], "xent_share_ci": ci(boot["xent"]),
            "xent_minus_nll_ci": [lo, hi]}


def verdict_e3(pairs: dict) -> dict:
    """Secondary: entropy has the largest share in at least 6 of the 10 reference pairs."""
    top = {k: max(v["share"], key=v["share"].get) for k, v in pairs.items()}
    k = sum(t == "entropy" for t in top.values())
    return {"verdict": "SUPPORTED" if k >= 6 else "NOT SUPPORTED", "entropy_largest_in": k,
            "of": len(pairs), "largest_component": top}


# ---- loading the cached per-token statistics (study data) -----------------------------
def _require_registration() -> str:
    if not REG_FILE.exists() or "osf.io/" not in REG_FILE.read_text():
        raise RuntimeError(f"Not registered. Put the OSF link of the follow-up registration in {REG_FILE} "
                           "before running this on study data.")
    return REG_FILE.read_text().strip()


def _fd_components(ref: str) -> pd.DataFrame:
    from refdist.scoring.logprob_cache import load
    parts = []
    for k in POPS:
        c = load(ref, k)
        g = c.groupby("doc_id", sort=False)
        parts.append(pd.DataFrame({"nll": -g["logp"].mean(), "entropy": -g["mu"].mean(),
                                   "var": g["var"].mean(), "n": g.size().astype(float),
                                   "pop": k}))
    return pd.concat(parts)


def _bino_components(ref: str) -> pd.DataFrame:
    from refdist.scoring.logprob_cache import load
    parts = []
    for k in POPS:
        c = load(f"{ref}-pair", k)
        g = c.groupby("doc_id", sort=False)
        parts.append(pd.DataFrame({"nll": g["ce"].mean(), "xent": g["xent"].mean(), "pop": k}))
    return pd.concat(parts)


def _aligned(tables: dict[str, pd.DataFrame], comps) -> tuple[dict, np.ndarray]:
    ids = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    assert len(ids) == 12_000, f"expected 12,000 documents in every reference, got {len(ids)}"
    first = next(iter(tables.values())).loc[ids]
    out = {r: {c: t.loc[ids, c].to_numpy(float) for c in comps} for r, t in tables.items()}
    return out, first["pop"].to_numpy()


def _check_against_registered_scores(comps: dict, pop_ids: list[str], instance: str, fn) -> float:
    s = pd.read_parquet(paths.results() / "scores.parquet", columns=["item_id", "instance", "score"])
    s = s[s["instance"] == instance].set_index("item_id").loc[pop_ids, "score"].to_numpy()
    return float(np.max(np.abs(fn(comps) - s)))


def _shift_p(instances: list[str], ids: list[str], pop: np.ndarray) -> float:
    """Precondition: the registered profile-shift permutation test between two instances,
    on their frozen registered flags. A decomposition is only read if the shift is real."""
    from refdist.analysis.hypotheses import profile_shift
    from refdist.metrics.flags import load_thresholds
    thr = load_thresholds()
    s = pd.read_parquet(paths.results() / "scores.parquet", columns=["item_id", "instance", "score"])
    s = s[s["instance"].isin(instances) & s["item_id"].isin(ids)]
    s = s.assign(flagged=(s["score"] > s["instance"].map(thr)).astype(np.int8),
                 stratum=s["item_id"].map(dict(zip(ids, pop))))
    return profile_shift(s, "instance", 2000, 0)[1]


def run() -> dict:
    link = _require_registration()
    fd = {r: _fd_components(r) for r in ("R1", "R2", "R3", "R4", "R5")}
    fdc, pop = _aligned(fd, FD_COMPONENTS)
    ids = sorted(set.intersection(*(set(t.index) for t in fd.values())))
    bino = {r: _bino_components(r) for r in ("R4", "R5")}
    bc, bpop = _aligned(bino, BINO_COMPONENTS)
    assert (bpop == pop).all()

    # the components must reproduce the scores the registered analysis used
    check = {f"D1@{r}": _check_against_registered_scores(fdc[r], ids, f"D1@{r}", fd_score) for r in fdc}
    check |= {f"D2@{r}": _check_against_registered_scores(bc[r], ids, f"D2@{r}", bino_score) for r in bc}
    assert max(check.values()) < 1e-4, f"components do not reproduce registered scores: {check}"

    out = {"registration": link, "score_reproduction_max_abs_diff": check}
    e1 = decompose(fdc["R3"], fdc["R4"], pop, fd_score, FD_COMPONENTS)
    e1_boot = bootstrap(fdc["R3"], fdc["R4"], pop, fd_score, FD_COMPONENTS)
    p1 = _shift_p(["D1@R3", "D1@R4"], ids, pop)
    out["E1"] = {**e1, "shift_p": p1,
                 **(verdict_e1(e1, e1_boot) if p1 < 0.05 else {"verdict": "NOT ESTIMABLE"}),
                 "reverse_R4_to_R3": decompose(fdc["R4"], fdc["R3"], pop, fd_score, FD_COMPONENTS)}
    e2 = decompose(bc["R4"], bc["R5"], pop, bino_score, BINO_COMPONENTS)
    e2_boot = bootstrap(bc["R4"], bc["R5"], pop, bino_score, BINO_COMPONENTS)
    p2 = _shift_p(["D2@R4", "D2@R5"], ids, pop)
    out["E2"] = {**e2, "shift_p": p2,
                 **(verdict_e2(e2, e2_boot) if p2 < 0.05 else {"verdict": "NOT ESTIMABLE"})}
    pairs = {f"{a}->{b}": decompose(fdc[a], fdc[b], pop, fd_score, FD_COMPONENTS)
             for a, b in itertools.combinations(("R1", "R2", "R3", "R4", "R5"), 2)}
    out["E3_secondary"] = {"pairs": pairs, **verdict_e3(pairs)}
    out["descriptive_means"] = {
        r: pd.DataFrame({**fdc[r], "pop": pop}).groupby("pop").mean().round(4).to_dict() for r in fdc}
    provenance.write_json(paths.results() / "followup_entropy.json", out)
    return out


# ---- synthetic validation -----------------------------------------------------------------
def simulate(scenario: str, n_per_pop: int = 1000, seed: int = SEED) -> tuple[dict, dict, np.ndarray]:
    """Base and target references on fake documents. Population-level shifts are planted in
    one component ('entropy', 'var', 'nll'); 'null' plants none."""
    rng = np.random.default_rng(seed)
    pop = np.repeat(POPS, n_per_pop)
    k = np.array([int(p[1:]) for p in pop])
    eff = np.where(pop == CAL, 0.0, rng.normal(0, 1, 12)[k - 1] * 0.06)   # per-population shift
    base = {"nll": rng.normal(3.0, 0.3, len(pop)), "var": rng.lognormal(1.0, 0.1, len(pop)),
            "n": rng.integers(150, 400, len(pop)).astype(float)}
    base["entropy"] = base["nll"] + rng.normal(-0.15, 0.12, len(pop))
    target = {c: v.copy() for c, v in base.items()}
    noise = lambda: rng.normal(0, 0.01, len(pop))
    target["nll"] = base["nll"] + 0.02 + noise()
    target["entropy"] = base["entropy"] + 0.02 + noise()
    if scenario == "entropy":
        target["entropy"] = target["entropy"] + eff
    elif scenario == "nll":
        target["nll"] = target["nll"] - eff
    elif scenario == "var":
        target["var"] = base["var"] * np.exp(-12 * eff)
    return base, target, pop


def validate(n_boot: int = 200) -> dict:
    res = {}
    for sc in ("entropy", "nll", "var", "null"):
        b, t, p = simulate(sc)
        point = decompose(b, t, p, fd_score, FD_COMPONENTS)
        boot = bootstrap(b, t, p, fd_score, FD_COMPONENTS, n=n_boot)
        res[sc] = {"total_distance": round(point["total_distance"], 3),
                   "share": {k: round(v, 2) for k, v in point["share"].items()},
                   "E1_verdict": verdict_e1(point, boot)["verdict"]}
    # Binoculars: plant the shift in the normaliser, then in the numerator
    for sc, comp in (("bino_xent", "xent"), ("bino_nll", "nll")):
        rng = np.random.default_rng(SEED)
        pop = np.repeat(POPS, 1000)
        k = np.array([int(q[1:]) for q in pop])
        eff = np.where(pop == CAL, 0.0, rng.normal(0, 1, 12)[k - 1] * 0.05)
        b = {"nll": rng.normal(3.0, 0.3, len(pop))}
        b["xent"] = b["nll"] * rng.normal(1.05, 0.04, len(pop))
        t = {c: v * (1 + rng.normal(0, 0.002, len(pop))) for c, v in b.items()}
        t[comp] = t[comp] * (1 + eff)
        point = decompose(b, t, pop, bino_score, BINO_COMPONENTS)
        boot = bootstrap(b, t, pop, bino_score, BINO_COMPONENTS, n=n_boot)
        res[sc] = {"total_distance": round(point["total_distance"], 3),
                   "share": {k2: round(v, 2) for k2, v in point["share"].items()},
                   "E2_verdict": verdict_e2(point, boot)["verdict"]}
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["--synthetic"]:
        print(json.dumps(validate(), indent=1))
    elif not sys.argv[1:]:
        r = run()
        print(json.dumps({k: r[k].get("verdict") for k in ("E1", "E2", "E3_secondary")}, indent=1))
    else:
        print(__doc__)
