"""
Estimate a detector's false-positive rate on a population of human texts.

Fitting (from the study, run by the registered report):
    for each detector instance, a logistic model on all human documents
        P(flagged) = logistic(b0 + b1 * log PPL_indep + b2 * length/100)
    whose coefficients and covariance are saved to results/audit_model.json.

Estimating (for a caller's own texts):
    score the texts with the same independent model (SmolLM2-360M), average the
    predicted flag probabilities, and propagate both coefficient uncertainty and
    the caller's sampling variability into a 95% interval.

The output is a POPULATION estimate: "about this share of these human-written
texts would be flagged". It is not, and must never be used as, a verdict on any
individual document.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from refdist import config, paths

LEN_SCALE = 100.0


def _design(log_ppl: np.ndarray, n_tokens: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(log_ppl), log_ppl, n_tokens / LEN_SCALE])


def fit(doc_table: pd.DataFrame, validity: dict | None = None) -> dict:
    """doc_table: instance, flagged, log_ppl_indep, n_tokens (human study documents)."""
    models = {}
    for inst, g in doc_table.dropna(subset=["log_ppl_indep"]).groupby("instance"):
        X = _design(g["log_ppl_indep"].to_numpy(), g["n_tokens"].to_numpy(float))
        res = sm.GLM(g["flagged"].to_numpy(float), X, family=sm.families.Binomial()).fit()
        # Between-population spread: how far each study population's actual FPR
        # sits from the model's prediction, on the logit scale. Without it the
        # interval covers only coefficient and sampling noise and is far too
        # narrow for a population the study never saw.
        pred = pd.Series(res.predict(X), index=g.index)
        dev = []
        for _, gs in g.groupby("stratum"):
            k, n = gs["flagged"].sum(), len(gs)
            obs = np.log((k + 0.5) / (n - k + 0.5))
            p = pred.loc[gs.index].mean()
            dev.append(obs - np.log(p / (1 - p)))
        models[inst] = {"coef": res.params.tolist(), "cov": res.cov_params().tolist(),
                        "population_sd_logit": float(np.std(dev, ddof=1)) if len(dev) > 2 else 0.0,
                        "n_docs": int(len(g)), "n_populations": len(dev)}
    out = {
        "predictor": config.INDEPENDENT_PREDICTOR,
        "length_tokenizer": config.CORPUS["length_tokenizer"],
        "target_fpr_on_calibration": config.TARGET_FPR,
        "instances": models,
        "validity": validity or {},
    }
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _point(model: dict, instance: str, log_ppl: np.ndarray, n_tok: np.ndarray) -> float:
    X = _design(log_ppl, n_tok.astype(float))
    return float(np.mean(1 / (1 + np.exp(-X @ np.array(model["instances"][instance]["coef"])))))


def calibrate(doc_table: pd.DataFrame, level: float = 0.95) -> float:
    """
    Split-conformal half-width on the logit scale. Each study population is held
    out in turn, the model refitted on the rest, and the miss |logit(observed FPR)
    - logit(predicted FPR)| recorded for every detector. The finite-sample
    (n+1)-adjusted quantile of those misses gives ~level coverage for a new
    population, without assuming how populations differ.
    """
    misses = []
    for held, gh in doc_table.groupby("stratum"):
        m = fit(doc_table[doc_table["stratum"] != held])
        for inst, g in gh.dropna(subset=["log_ppl_indep"]).groupby("instance"):
            if inst not in m["instances"]:
                continue
            k, n = g["flagged"].sum(), len(g)
            obs = np.log((k + 0.5) / (n - k + 0.5))
            misses.append(abs(obs - _logit(_point(m, inst, g["log_ppl_indep"].to_numpy(), g["n_tokens"].to_numpy()))))
    misses = np.sort(misses)
    rank = min(len(misses) - 1, int(np.ceil((len(misses) + 1) * level)) - 1)
    return float(misses[rank])


def save(model: dict, path: Path | None = None) -> Path:
    path = path or paths.results() / "audit_model.json"
    path.write_text(json.dumps(model, indent=1))
    return path


def featurize(texts: list[str]) -> pd.DataFrame:
    """log PPL_indep and length for arbitrary texts, exactly as computed in the study."""
    from refdist.corpora.base import n_tokens
    from refdist.scoring import detectors, logprob_cache
    from refdist.scoring.models import load_causal_lm
    ip = config.INDEPENDENT_PREDICTOR
    model, tok = load_causal_lm(ip["model_id"], ip["revision"])
    ids = [f"u{i}" for i in range(len(texts))]
    lp = detectors.log_perplexity(logprob_cache.token_stats(model, tok, texts, ids)).set_index("doc_id")
    return pd.DataFrame({"log_ppl_indep": lp.loc[ids, "log_ppl"].to_numpy(),
                         "n_tokens": [n_tokens(t) for t in texts]})


def estimate(features: pd.DataFrame, model: dict, instance: str, draws: int = 2000,
             seed: int = 0) -> dict:
    m = model["instances"][instance]
    X = _design(features["log_ppl_indep"].to_numpy(), features["n_tokens"].to_numpy(float))
    point = float(np.mean(1 / (1 + np.exp(-X @ np.array(m["coef"])))))
    rng = np.random.default_rng(seed)
    betas = rng.multivariate_normal(m["coef"], m["cov"], size=draws)
    # t, not normal: the spread itself is estimated from only ~11 populations.
    df = max(m.get("n_populations", 30) - 1, 1)
    shifts = m.get("population_sd_logit", 0.0) * rng.standard_t(df, size=draws)
    sims = np.empty(draws)
    for i, (b, u) in enumerate(zip(betas, shifts)):
        rows = rng.integers(0, len(X), len(X))  # caller's sample is itself a sample
        sims[i] = np.mean(1 / (1 + np.exp(-(X[rows] @ b + u))))  # u: this population's own offset
    lo, hi = np.quantile(sims, [0.025, 0.975])
    if "conformal_q_logit" in model:  # calibrated on held-out study populations; preferred
        q = model["conformal_q_logit"]
        c = _logit(point)
        lo, hi = min(lo, 1 / (1 + np.exp(-(c - q)))), max(hi, 1 / (1 + np.exp(-(c + q))))
    span = (features["n_tokens"].min(), features["n_tokens"].max())
    warnings = []
    if span[0] < config.CORPUS["min_tokens"] or span[1] > config.CORPUS["max_tokens"]:
        warnings.append(f"texts outside the study's {config.CORPUS['min_tokens']}-"
                        f"{config.CORPUS['max_tokens']} token range: extrapolating")
    if len(features) < 30:
        warnings.append("fewer than 30 texts: the interval is wide by design")
    return {"instance": instance, "n_texts": len(features), "fpr_estimate": point,
            "fpr_95ci": (float(lo), float(hi)), "warnings": warnings}
