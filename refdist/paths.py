"""
Environment-aware data roots.

On Kaggle, inputs are read-only under /kaggle/input and only /kaggle/working
persists (20GB). Locally everything lives under ./data. Set REFDIST_DATA to
override either.
"""

from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()


def data_root() -> Path:
    if env := os.environ.get("REFDIST_DATA"):
        return Path(env)
    if ON_KAGGLE:
        return Path("/kaggle/working/data")
    return Path(__file__).resolve().parent.parent / "data"


def _d(*parts: str) -> Path:
    p = data_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw() -> Path:
    return _d("raw")


def interim() -> Path:
    return _d("interim")


def processed() -> Path:
    return _d("processed")


def cache() -> Path:
    return _d("cache")


def results() -> Path:
    return _d("results")


def logprob_cache(model_id: str) -> Path:
    return _d("cache", "logprob", model_id.replace("/", "__"))


def documents_file() -> Path:
    return processed() / "documents.parquet"


def variants_file() -> Path:
    return processed() / "variants.parquet"


def control_file() -> Path:
    # Deliberately a separate file: generated text never enters documents.parquet.
    return processed() / "control.parquet"


def scores_file() -> Path:
    return results() / "scores.parquet"


def ledger_file() -> Path:
    return results() / "ledger.jsonl"


def thresholds_file() -> Path:
    return results() / "thresholds.json"


def forecast_file() -> Path:
    return Path(__file__).resolve().parent.parent / "prereg" / "forecast_S10_S12.json"
