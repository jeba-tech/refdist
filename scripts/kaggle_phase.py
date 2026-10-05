"""
Runs one phase of the study inside a Kaggle kernel. Called by the kernel stub
that kaggle/launch.py pushes; never run locally.

  phase 0  GPU harness check (Milestone 0) + positive-control generation.
           Scores no study document, so it may run before OSF registration.
  phase 1  Score S1-S9, control and H5 variants with every model; validity
           gates; freeze thresholds; commit the forecast. Requires the live
           OSF registration (enforced by the runner).
  phase 2  Score the held-out S10-S12; forecast reveal; full registered report.

Each phase's /kaggle/working output is attached as input to the next phase, so
outputs accumulate and every stage stays resumable through the ledger.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
DATA = WORK / "data"
CODE = Path(__file__).resolve().parent.parent


def sh(*args: str) -> None:
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=CODE)


def gather_inputs() -> None:
    """Corpus parquets from the bundle; accumulated data and forecast from earlier phases."""
    proc = DATA / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    # Earlier phases' outputs first (Kaggle mounts them at notebooks/<user>/<slug>)...
    prev_runs = glob.glob("/kaggle/input/notebooks/*/*/") + glob.glob("/kaggle/input/refdist-phase*/")
    # A run that ended in an error can't be attached as a kernel source, so its
    # outputs may arrive as an uploaded dataset instead (e.g. refdist-phase1-out).
    for t in glob.glob("/kaggle/input/datasets/**/results/thresholds.json", recursive=True):
        prev_runs.append(str(Path(t).parent.parent.parent) + "/")
    for run in sorted(set(prev_runs)):
        if os.path.isdir(os.path.join(run, "data")):
            shutil.copytree(os.path.join(run, "data"), DATA, dirs_exist_ok=True)
        for f in glob.glob(os.path.join(run, "code", "prereg", "forecast_S10_S12.*")):
            shutil.copy(f, CODE / "prereg" / Path(f).name)
    # ...then the bundle, which is the source of truth for the corpus: a stale
    # documents.parquet from an earlier phase must never win.
    # Only the bundle's top level: other inputs hold caches with the same file
    # names (every detector caches a "variants.parquet"), and a name search once
    # picked one of those instead of the corpus.
    bundle = (glob.glob("/kaggle/input/datasets/*/refdist-bundle") + glob.glob("/kaggle/input/refdist-bundle"))[0]
    for name in ("documents.parquet", "variants.parquet"):
        if os.path.isfile(os.path.join(bundle, name)):
            shutil.copy(os.path.join(bundle, name), proc / name)
    if os.path.isfile(os.path.join(bundle, "precision.json")) and not (DATA / "results" / "precision.json").exists():
        (DATA / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(bundle, "precision.json"), DATA / "results" / "precision.json")
    import pyarrow.parquet as pq
    expected = {"documents.parquet": {"doc_id", "stratum", "text"}, "variants.parquet": {"variant_id", "doc_id", "text"},
                "control.parquet": {"doc_id", "text"}}
    for name, cols in expected.items():
        if (proc / name).exists():
            missing = cols - set(pq.read_schema(proc / name).names)
            if missing:
                raise SystemExit(f"{name} is the wrong file (missing {sorted(missing)})")
    print("previous runs:", prev_runs)
    print("inputs:", sorted(p.name for p in proc.iterdir()),
          "| results:", sorted(p.name for p in (DATA / "results").iterdir()) if (DATA / "results").exists() else [])


def phase0() -> None:
    sh("scripts/milestone0_local.py")
    sh("-m", "refdist.scoring.generate_control")


def phase0b() -> None:
    """Top up the positive control after normalisation (raw generations are reused)."""
    sh("-m", "refdist.scoring.generate_control")


def phase1() -> None:
    from refdist.scoring.runner import registered
    if not registered():
        raise SystemExit("phase 1 needs the live OSF registration URL (REFDIST_OSF_URL)")
    for stage in (["P"], ["A", "--refs", "R1", "R2", "R3"], ["A", "--refs", "R4", "R5"], ["B"], ["C"]):
        sh("-m", "refdist.scoring.runner", *stage)
    sh("-m", "refdist.scoring.runner", "derive")
    sh("-c", "from refdist.analysis import tables, validity; s, d = tables.load(); "
             "s, inc, g = validity.run(s, d); "
             "print('included', inc); print('excluded', g['excluded_instances'])")
    sh("-m", "refdist.analysis.forecast", "commit")
    digest = (CODE / "prereg" / "forecast_S10_S12.sha256").read_text().strip()
    (DATA / "results" / "PHASE1_DONE.json").write_text(json.dumps({"forecast_sha256": digest}))
    print(f"\nFORECAST SHA-256: {digest}\nPost this on OSF before phase 2.")


def phase2() -> None:
    if not (CODE / "prereg" / "forecast_S10_S12.json").exists():
        raise SystemExit("phase 2 needs phase 1's committed forecast as input")
    for stage in ("A", "B", "C"):
        sh("-m", "refdist.scoring.runner", stage, "--groups", "S10", "S11", "S12")
    sh("-m", "refdist.scoring.runner", "derive")
    sh("-m", "refdist.analysis.forecast", "reveal")
    sh("-m", "refdist.analysis.report")


if __name__ == "__main__":
    os.environ.setdefault("REFDIST_DATA", str(DATA))
    sys.path.insert(0, str(CODE))
    gather_inputs()
    {"0": phase0, "0b": phase0b, "1": phase1, "2": phase2}[sys.argv[1]]()
