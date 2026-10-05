"""
EXPLORATORY analysis -- not part of the registered decision.

The registered positive-control gate (AUROC >= 0.80) excluded four detector
instances: D1@R1 (0.788), D1@R2 (0.792), D1@R5 (0.779) and D3 (0.502). This
script reruns H1-H6 with all ten instances, thresholds calibrated on S6 exactly
as registered, so the paper can show what the gate changed. It works in a
scratch copy, so the frozen registered thresholds are never touched. Every
output is labelled exploratory and must be reported as such.

Usage:  python scripts/exploratory_all_instances.py <phase2 data dir> <scratch dir>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

src, scratch = Path(sys.argv[1]), Path(sys.argv[2])
for sub in ("processed", "results"):
    (scratch / sub).mkdir(parents=True, exist_ok=True)
for f in ("documents.parquet", "variants.parquet", "control.parquet"):
    shutil.copy(src / "processed" / f, scratch / "processed" / f)
shutil.copy(src / "results" / "scores.parquet", scratch / "results" / "scores.parquet")
os.environ["REFDIST_DATA"] = str(scratch)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from refdist import config, paths  # noqa: E402
from refdist.analysis import hypotheses as H, tables, validity  # noqa: E402
from refdist.metrics import flags as fl  # noqa: E402

scores, docs = tables.load()
gate = validity.positive_control(scores, docs)
det = scores[scores["detector"].str.match(r"^D\d$") & ~scores["group"].isin(["control", "variants"])]
thr = fl.calibrate(det, docs)  # all ten instances, same rule as registered
(paths.results() / "thresholds.json").write_text(json.dumps({"thresholds": thr, "_note": "EXPLORATORY"}))

cells = tables.cell_table(scores, docs)
dt = tables.doc_table(scores, docs)
per = H.h5(H.h5_deltas(scores, pd.read_parquet(paths.variants_file())))
res = {
    "_label": "EXPLORATORY: all 10 detector instances, positive-control gate not applied",
    "positive_control_auroc": gate.set_index("instance")["auroc"].round(3).to_dict(),
    "H1": H.h1(cells, dt),
    "H2": H.h2(cells, dt),
    "H2_matched_R2_R3": H.h2(cells, dt, refs=list(config.MATCHED_DATA_PAIR)),
    "H3": H.h3(cells, dt),
    "H4": H.h4(cells),
    "H5": {"overall": H.h5_overall(per, list(thr)), "per_instance": per},
    "H6": H.h6(dt),
}
fpr = cells.pivot_table(index="stratum", columns="instance", values="fpr")
res["fpr_table"] = fpr.round(4).to_dict()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o.item() if hasattr(o, "item") else o


out = Path(__file__).resolve().parent.parent / "data" / "results" / "exploratory_all_instances.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(clean(res), indent=1))
print(f"EXPLORATORY results -> {out}")
for k in ("H1", "H2", "H2_matched_R2_R3", "H3", "H4", "H6"):
    print(f"  {k:18s} {res[k]['verdict']}")
print(f"  {'H5':18s} {res['H5']['overall']['verdict']} ({len(res['H5']['overall']['confirmed_instances'])}/{res['H5']['overall']['n_instances']})")
