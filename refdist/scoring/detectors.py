"""
Document scores from cached per-token statistics.

Convention for every detector: HIGHER score = MORE machine-like. Binoculars is
negated to follow it. A single orientation keeps thresholding and FPR code
detector-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fast_detectgpt(cache: pd.DataFrame) -> pd.DataFrame:
    """
    Analytic Fast-DetectGPT (Bao et al., 2024) with sampling model = scoring model:
        (sum logp - sum mu) / sqrt(sum var)
    """
    g = cache.groupby("doc_id", sort=False)
    s = g[["logp", "mu", "var"]].sum()
    n = g.size()
    score = (s["logp"] - s["mu"]) / np.sqrt(s["var"].clip(lower=1e-8))
    return pd.DataFrame({"doc_id": s.index, "score": score.values, "n_tokens": n.values})


def log_perplexity(cache: pd.DataFrame) -> pd.DataFrame:
    """Mean negative log-likelihood per token. PPL = exp(this)."""
    g = cache.groupby("doc_id", sort=False)
    lppl = -g["logp"].mean()
    return pd.DataFrame({"doc_id": lppl.index, "log_ppl": lppl.values, "n_tokens": g.size().values})


def binoculars(pair_cache: pd.DataFrame) -> pd.DataFrame:
    g = pair_cache.groupby("doc_id", sort=False)
    m = g[["ce", "xent"]].mean()
    raw = m["ce"] / m["xent"].clip(lower=1e-8)
    return pd.DataFrame({"doc_id": m.index, "score": (-raw).values, "n_tokens": g.size().values})
