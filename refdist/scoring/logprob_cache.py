"""
Pass A: one forward pass per model caches per-token statistics.

For each scored token t (every token after the first, given its prefix):
    logp  = log p(x_t | x_<t)                       observed log-likelihood
    mu    = E_{x~p}[log p(x | x_<t)]                expected log-likelihood (= -entropy)
    var   = Var_{x~p}[log p(x | x_<t)]

Fast-DetectGPT (analytic, sampling model = scoring model), raw perplexity and
PPL_indep are all closed-form functions of these three columns. Re-deriving
every score after a bug fix is therefore numpy over the cache, not a GPU rerun.

Pass B (Binoculars) needs two models' distributions at once and is cached per
pair as (ce_performer, xent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from refdist import paths

MAX_TOKENS = 512
TOKEN_BUDGET = 4096  # padded tokens per batch; keeps B*T*V logits within T4 memory


def _batches(texts: list[str], ids: list[str], tok) -> Iterable[tuple[list[str], dict]]:
    """Length-sorted batches under a padded-token budget; pad waste dominates otherwise."""
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = np.argsort(lens)
    batch: list[int] = []
    for i in order:
        longest = max([min(lens[j], MAX_TOKENS) for j in batch + [i]])
        if batch and longest * (len(batch) + 1) > TOKEN_BUDGET:
            yield _encode(batch, texts, ids, tok)
            batch = []
        batch.append(i)
    if batch:
        yield _encode(batch, texts, ids, tok)


def _encode(idx: list[int], texts, ids, tok):
    enc = tok(
        [texts[i] for i in idx],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        add_special_tokens=False,
    )
    return [ids[i] for i in idx], enc


def _assert_finite(mask: torch.Tensor, **tensors: torch.Tensor) -> None:
    """A non-finite statistic must never reach the cache: it would silently poison a score."""
    m = mask.bool()
    for name, t in tensors.items():
        if not torch.isfinite(t[m]).all():
            raise FloatingPointError(f"non-finite {name} in scoring batch; refusing to cache")


CHUNK = 1024  # token positions per statistics slice: bounds memory at ~4 x CHUNK x vocab floats


def _token_terms(logits: torch.Tensor, targets: torch.Tensor):
    """logp, mu, var for N positions; logits [N, V]. Sliced so a 151k vocab fits a T4."""
    out = [[], [], []]
    for i in range(0, logits.shape[0], CHUNK):
        lp = torch.log_softmax(logits[i:i + CHUNK].float(), dim=-1)
        p = lp.exp()
        out[0].append(lp.gather(-1, targets[i:i + CHUNK, None]).squeeze(-1))
        pl = torch.nan_to_num(p * lp, nan=0.0)          # 0 * log 0 = 0, not NaN
        mu = pl.sum(-1)
        out[1].append(mu)
        out[2].append(torch.nan_to_num(pl * lp, nan=0.0).sum(-1) - mu.square())
        del lp, p, pl
    return (torch.cat(o) for o in out)


def _pair_terms(obs_logits: torch.Tensor, perf_logits: torch.Tensor, targets: torch.Tensor):
    ce, xent = [], []
    for i in range(0, obs_logits.shape[0], CHUNK):
        perf_lp = torch.log_softmax(perf_logits[i:i + CHUNK].float(), dim=-1)
        ce.append(-perf_lp.gather(-1, targets[i:i + CHUNK, None]).squeeze(-1))
        obs_p = torch.softmax(obs_logits[i:i + CHUNK].float(), dim=-1)
        xent.append(-torch.nan_to_num(obs_p * perf_lp, nan=0.0).sum(-1))
        del perf_lp, obs_p
    return torch.cat(ce), torch.cat(xent)


def _split(values: list[np.ndarray], lengths: list[int], batch_ids: list[str], names: list[str]) -> list[pd.DataFrame]:
    rows, start = [], 0
    for doc_id, n in zip(batch_ids, lengths):
        cols = {"doc_id": doc_id, "token_idx": np.arange(n, dtype=np.int32)}
        for name, v in zip(names, values):
            x = v[start:start + n].astype(np.float32)
            cols[name] = np.clip(x, 0, None) if name == "var" else x
        rows.append(pd.DataFrame(cols))
        start += n
    return rows


@torch.inference_mode()
def token_stats(model, tok, texts: list[str], ids: list[str]) -> pd.DataFrame:
    """Per-token (logp, mu, var) for every document, long format."""
    dev = next(model.parameters()).device
    rows = []
    for batch_ids, enc in _batches(texts, ids, tok):
        input_ids = enc["input_ids"].to(dev)
        mask = enc["attention_mask"].to(dev)
        tmask = mask[:, 1:].bool()
        logits = model(input_ids=input_ids, attention_mask=mask).logits[:, :-1][tmask]  # real positions only
        logp, mu, var = _token_terms(logits, input_ids[:, 1:][tmask])
        del logits
        _assert_finite(torch.ones_like(logp, dtype=torch.bool), logp=logp, mu=mu, var=var)
        lengths = tmask.sum(1).tolist()
        rows += _split([x.cpu().numpy() for x in (logp, mu, var)], lengths, batch_ids, ["logp", "mu", "var"])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["doc_id", "token_idx", "logp", "mu", "var"])


@torch.inference_mode()
def pair_stats(observer, performer, tok, texts: list[str], ids: list[str]) -> pd.DataFrame:
    """
    Binoculars per-token terms. Observer and performer share a tokenizer.

    ce    = -log p_performer(x_t | x_<t)
    xent  = -sum_v p_observer(v) log p_performer(v)
    Binoculars = mean(ce) / mean(xent); low = machine-like (Hans et al., 2024).
    """
    dev = next(observer.parameters()).device
    rows = []
    for batch_ids, enc in _batches(texts, ids, tok):
        input_ids = enc["input_ids"].to(dev)
        mask = enc["attention_mask"].to(dev)
        tmask = mask[:, 1:].bool()
        obs = observer(input_ids=input_ids, attention_mask=mask).logits[:, :-1][tmask]
        perf = performer(input_ids=input_ids, attention_mask=mask).logits[:, :-1][tmask]
        ce, xent = _pair_terms(obs, perf, input_ids[:, 1:][tmask])
        del obs, perf
        _assert_finite(torch.ones_like(ce, dtype=torch.bool), ce=ce, xent=xent)
        rows += _split([x.cpu().numpy() for x in (ce, xent)], tmask.sum(1).tolist(), batch_ids, ["ce", "xent"])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["doc_id", "token_idx", "ce", "xent"])


def cache_path(model_key: str, group: str) -> Path:
    return paths.logprob_cache(model_key) / f"{group}.parquet"


def cached_doc_ids(model_key: str, group: str) -> set[str]:
    """Doc ids already cached, across the main file and any partial flushes."""
    d = paths.logprob_cache(model_key)
    done: set[str] = set()
    for f in d.glob(f"{group}*.parquet"):
        done |= set(pd.read_parquet(f, columns=["doc_id"])["doc_id"].unique())
    return done


def run_cached(
    stats_fn,
    model_key: str,
    group: str,
    docs: pd.DataFrame,
    flush_every: int = 200,
) -> None:
    """
    Cache stats for `docs` (columns doc_id, text) under (model_key, group).

    Resumable: already-cached doc ids are skipped, and results flush to part
    files every `flush_every` docs, so a Kaggle session kill loses one chunk.
    """
    done = cached_doc_ids(model_key, group)
    todo = docs[~docs["doc_id"].isin(done)]
    if todo.empty:
        return
    d = paths.logprob_cache(model_key)
    start = len(list(d.glob(f"{group}.part*.parquet")))
    chunks = range(0, len(todo), flush_every)
    for k, i in enumerate(tqdm(chunks, desc=f"{model_key}:{group}")):
        chunk = todo.iloc[i:i + flush_every]
        df = stats_fn(chunk["text"].tolist(), chunk["doc_id"].tolist())
        df.to_parquet(d / f"{group}.part{start + k:04d}.parquet", index=False)
    consolidate(model_key, group)


def consolidate(model_key: str, group: str) -> None:
    d = paths.logprob_cache(model_key)
    parts = sorted(d.glob(f"{group}.part*.parquet"))
    main = d / f"{group}.parquet"
    frames = ([pd.read_parquet(main)] if main.exists() else []) + [pd.read_parquet(p) for p in parts]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["doc_id", "token_idx"])
    tmp = d / f"{group}.tmp.parquet"
    df.to_parquet(tmp, index=False)
    tmp.replace(main)
    for p in parts:
        p.unlink()


def load(model_key: str, group: str) -> pd.DataFrame:
    return pd.read_parquet(cache_path(model_key, group))
