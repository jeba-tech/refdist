"""
The whole registered analysis, in registered order, in one command.

  1. validity gates (positive control -> recorded flips -> thresholds frozen on
     S6 -> negative control -> length invariance)
  2. H1-H6 on instances that passed the positive control
  3. bootstrap intervals for the H1 and H2 coefficients
  4. the decision rule of Section 8

Usage:  python -m refdist.analysis.report [--boot 1000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from refdist import config, paths, provenance
from refdist.analysis import bootstrap, hypotheses as H, tables, validity


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run(n_boot: int) -> dict:
    scores, docs = tables.load()
    scores, included, gate = validity.run(scores, docs)
    scores.to_parquet(paths.scores_file(), index=False)  # recorded flips applied

    scores = scores[scores["instance"].isin(included) | ~scores["detector"].str.match(r"^D\d$")]
    cells = tables.cell_table(scores, docs)
    dt = tables.doc_table(scores, docs)

    res = {"validity": gate, "included_instances": included}
    res["H1"] = H.h1(cells, dt)
    res["H2"] = H.h2(cells, dt)
    res["H2_matched_R2_R3"] = H.h2(cells, dt, refs=list(config.MATCHED_DATA_PAIR))
    res["H3"] = H.h3(cells, dt)
    res["H4"] = H.h4(cells)
    if paths.variants_file().exists() and (scores["group"] == "variants").any():
        per = H.h5(H.h5_deltas(scores, pd.read_parquet(paths.variants_file())))
        res["H5"] = {"overall": H.h5_overall(per, included), "per_instance": per}
    res["H6"] = H.h6(dt)

    from refdist.audit import estimate as audit
    audit_model = audit.fit(dt, validity={k: res["H4"][k] for k in ("mae", "skill", "verdict")})
    audit_model["conformal_q_logit"] = audit.calibrate(dt)
    audit.save(audit_model)

    own = cells[["reference", "stratum", "log_ppl_ref"]].dropna().drop_duplicates()
    boot = bootstrap.coefficients(dt, own, n=n_boot)
    res["bootstrap"] = {"n": n_boot, "h1_coef_95ci": bootstrap.interval(boot, "h1_coef"),
                        "h2_coef_95ci": bootstrap.interval(boot, "h2_coef")}

    core = [res["H1"]["verdict"], res["H2"]["verdict"], res.get("H5", {}).get("overall", {}).get("verdict")]
    n_conf = sum(v == "CONFIRMED" for v in core)
    res["decision"] = {
        "core_verdicts": {"H1": core[0], "H2": core[1], "H5": core[2]},
        "scorer_distance_hypothesis": ("CONFIRMED" if n_conf >= 2 else "SUGGESTIVE" if n_conf == 1
                                       else "REJECTED"),
    }
    provenance.write_json(paths.results() / "results.json", _jsonable(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=config.ANALYSIS["bootstrap_resamples"])
    a = ap.parse_args(argv)
    r = run(a.boot)
    print(json.dumps(_jsonable({k: v.get("verdict") if isinstance(v, dict) and "verdict" in v else None
                                for k, v in r.items() if k.startswith("H")}), indent=1))
    print("H5:", r.get("H5", {}).get("overall", {}).get("verdict"))
    print("DECISION:", r["decision"])
    print(f"-> {paths.results() / 'results.json'}")


if __name__ == "__main__":
    main()
