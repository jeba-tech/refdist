"""Length matching and within-source pair matching."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refdist.corpora.build import _bins, length_matched_sample  # noqa: E402


def _pool(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)],
                         "n_tokens": rng.integers(150, 401, n),
                         "field": rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1])})


def test_default_gives_equal_length_bins():
    out, rep = length_matched_sample(_pool(), 1000, np.random.default_rng(1))
    assert len(out) == 1000 and rep["shortfall"] == 0
    assert set(rep["per_bin"].values()) == {200}


def test_default_fills_a_thin_bin_from_neighbours_and_reports_it():
    df = _pool()
    df = df[~((df["n_tokens"] > 350) & (df.index % 10 != 0))]  # starve the longest bin
    out, rep = length_matched_sample(df, 1000, np.random.default_rng(1))
    assert len(out) == 1000 and rep["per_bin"][4] < 200


def test_pair_target_reproduces_partner_cells_exactly():
    df = _pool()
    target = {(0, "a"): 100, (0, "b"): 50, (2, "c"): 30, (4, "a"): 20}
    out, rep = length_matched_sample(df, 200, np.random.default_rng(1), target, df["field"])
    got = pd.Series(list(zip(_bins(out["n_tokens"]), out["field"]))).value_counts().to_dict()
    assert {(int(b), k): v for (b, k), v in got.items()} == target
    assert rep["partner_cell_mismatch"] == 0


def test_pair_shortfall_prefers_same_length_bin():
    df = _pool()
    df = df[~((_bins(df["n_tokens"]) == 2) & (df["field"] == "c"))]  # cell (2, c) empty
    target = {(2, "c"): 30, (0, "a"): 70}
    out, rep = length_matched_sample(df, 100, np.random.default_rng(1), target, df["field"])
    assert len(out) == 100
    assert (_bins(out["n_tokens"]) == 2).sum() == 30  # length profile kept, field substituted
