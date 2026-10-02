"""
refdist-audit: estimate how often a detector would wrongly flag YOUR human-written texts.

    refdist-audit --texts ./essays/ --detector D1@R1
    refdist-audit --texts essays.csv --column text --detector all

Reports a population false-positive rate with a 95% interval. It never labels an
individual text: the study behind this tool shows detectors respond to how
predictable text is, not to who wrote it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from refdist import paths
from refdist.audit import estimate as E


def _read(path: Path, column: str) -> list[str]:
    if path.is_dir():
        return [f.read_text(encoding="utf-8", errors="ignore") for f in sorted(path.glob("*.txt"))]
    if path.suffix == ".csv":
        return pd.read_csv(path)[column].dropna().astype(str).tolist()
    return [t for t in path.read_text(encoding="utf-8", errors="ignore").split("\n\n") if t.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", type=Path, required=True, help="folder of .txt, a .csv, or a text file")
    ap.add_argument("--column", default="text", help="CSV column holding the text")
    ap.add_argument("--detector", default="all", help="instance, e.g. D1@R1, D3; or 'all'")
    ap.add_argument("--model", type=Path, default=None, help="audit_model.json (default: results/)")
    a = ap.parse_args(argv)

    model = json.loads((a.model or paths.results() / "audit_model.json").read_text())
    texts = [t.strip() for t in _read(a.texts, a.column) if t.strip()]
    if not texts:
        sys.exit("no texts found")
    print(f"scoring {len(texts)} texts with {model['predictor']['model_id']} ...", file=sys.stderr)
    feats = E.featurize(texts)
    insts = list(model["instances"]) if a.detector == "all" else [a.detector]

    print(f"\nEstimated false-positive rate on these {len(texts)} human-written texts")
    print(f"(each detector set to a {model['target_fpr_on_calibration']:.0%} FPR on its calibration forum text)\n")
    for inst in insts:
        r = E.estimate(feats, model, inst)
        lo, hi = r["fpr_95ci"]
        print(f"  {inst:10s} {r['fpr_estimate']:6.1%}   95% interval {lo:5.1%} - {hi:5.1%}")
    for w in E.estimate(feats, model, insts[0])["warnings"]:
        print(f"\n  note: {w}")
    v = model.get("validity", {})
    if v:
        print(f"\n  model validity (leave-one-population-out, registered H4): MAE {v.get('mae', float('nan')):.3f}, "
              f"skill {v.get('skill', float('nan')):+.2f} over a perplexity-blind baseline")
    print("\nThis is a population estimate. It says nothing about whether any single text was "
          "written by a person or a machine.")


if __name__ == "__main__":
    main()
