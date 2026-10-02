"""
Build variants.parquet for H5.

Per transform: up to TRANSFORM_N_DOCS documents from the calibration stratum
that contain at least one eligible site, sampled with a fixed seed; if fewer are
eligible, all are used. Transforms differ enormously in how often they apply to
real forum text (T4 hits 94% of documents, T5 about 30%), so one shared document
set would leave the sparse transforms with almost no data.

Intensity 0 is not stored: it is the source document itself, whose scores
already exist, and every H5 delta is taken against it.

Usage:  python -m refdist.transforms.apply
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from refdist import config, paths, provenance
from refdist.corpora.base import n_tokens
from refdist.transforms.base import apply
from refdist.transforms.rules import TRANSFORMS


def build() -> pd.DataFrame:
    docs = pd.read_parquet(paths.documents_file())
    src = docs[docs["stratum"] == config.TRANSFORM_SOURCE_STRATUM].sort_values("doc_id")
    seed = config.CORPUS["seed"]
    rows, selection = [], {}
    for key, t in TRANSFORMS.items():
        eligible = src[[len(t.sites(x)) > 0 for x in src["text"]]]
        rng = np.random.default_rng([seed, int(key[1:])])
        n = min(config.TRANSFORM_N_DOCS, len(eligible))
        chosen = eligible.iloc[np.sort(rng.choice(len(eligible), size=n, replace=False))]
        selection[key] = {"eligible_docs": len(eligible), "selected": n}
        for r in chosen.itertuples():
            for p in config.TRANSFORM_INTENSITIES:
                if p == 0:
                    continue
                text, sites, n_elig = apply(t, r.text, p, r.doc_id, seed)
                rows.append({
                    "variant_id": f"{r.doc_id}|{key}|{p:.2f}",
                    "doc_id": r.doc_id, "transform": key, "intensity": p, "text": text,
                    "n_sites_eligible": n_elig, "n_sites_applied": len(sites),
                    "sites": json.dumps([[s.start, s.end, s.original, s.replacement] for s in sites]),
                    "n_tokens": n_tokens(text), "text_sha256": provenance.sha256_text(text),
                })
    v = pd.DataFrame(rows)
    v.to_parquet(paths.variants_file(), index=False)
    provenance.write_json(paths.processed() / "variants_manifest.json",
                          {"selection": selection, "n_variants": len(v),
                           "intensities": config.TRANSFORM_INTENSITIES})
    return v


if __name__ == "__main__":
    v = build()
    print(v.groupby("transform").agg(docs=("doc_id", "nunique"), variants=("variant_id", "size"),
                                     mean_sites=("n_sites_eligible", "mean")).round(2))
