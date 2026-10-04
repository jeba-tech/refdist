"""
Drive the Kaggle side of the study from this machine with the official CLI.

  python kaggle/launch.py bundle          upload/refresh the PRIVATE dataset refdist-bundle
                                          (code + documents.parquet + variants.parquet)
  python kaggle/launch.py push 0|1|2      push and start phase N on a T4
  python kaggle/launch.py status 0|1|2    run status
  python kaggle/launch.py pull 0|1|2      download phase N outputs into data/kaggle/phaseN

The dataset and kernels are private: several sources (PELIC, TwitterAAE,
ICNALE, ASAP) may not be redistributed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle" / "_build"
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def kaggle(*args: str, check: bool = True) -> str:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # Windows cp1252 crashes on log text
    out = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", env=env)
    if check and out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    return (out.stdout + out.stderr).strip()


def user() -> str:
    if u := os.environ.get("KAGGLE_USERNAME"):
        return u
    for f in (Path.home() / ".kaggle" / "kaggle.json",):
        if f.exists():
            return json.loads(f.read_text())["username"]
    raise SystemExit("Kaggle username unknown: set KAGGLE_USERNAME or create ~/.kaggle/kaggle.json")


def bundle() -> None:
    d = BUILD / "bundle"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("refdist", "scripts", "prereg"):
            for f in (ROOT / sub).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, Path("code") / f.relative_to(ROOT))
        z.write(ROOT / "pyproject.toml", "code/pyproject.toml")
    for name in ("documents.parquet", "variants.parquet"):
        shutil.copy(ROOT / "data" / "processed" / name, d / name)
    prec = ROOT / "data" / "kaggle" / "phase0" / "data" / "results" / "precision.json"
    if prec.exists():
        shutil.copy(prec, d / "precision.json")  # Milestone 0's measured fp16/fp32 choice per model
    slug = f"{user()}/refdist-bundle"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "refdist-bundle", "id": slug, "licenses": [{"name": "other"}]}, indent=1))
    exists = "refdist-bundle" in kaggle("datasets", "list", "--mine", "-s", "refdist-bundle", check=False)
    if exists:
        print(kaggle("datasets", "version", "-p", str(d), "-m", "refresh", "--dir-mode", "skip"))
    else:
        print(kaggle("datasets", "create", "-p", str(d), "--dir-mode", "skip"))


STUB = '''import glob, os, shutil, subprocess, sys, zipfile
PHASE = "{phase}"
OSF = "{osf}"
cfg = glob.glob("/kaggle/input/datasets/**/code/refdist/config.py", recursive=True)  # the bundle, never an old run's copy
if not cfg:
    for z in glob.glob("/kaggle/input/**/code.zip", recursive=True):
        zipfile.ZipFile(z).extractall("/kaggle/working/_src")
    cfg = glob.glob("/kaggle/working/_src/**/code/refdist/config.py", recursive=True)
src = os.path.dirname(os.path.dirname(sorted(cfg)[0]))
shutil.copytree(src, "/kaggle/working/code", dirs_exist_ok=True)
shutil.rmtree("/kaggle/working/_src", ignore_errors=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.56", "accelerate",
                "statsmodels==0.15.0", "wordfreq==3.1.1", "py7zr"], check=True)
env = dict(os.environ, REFDIST_DATA="/kaggle/working/data", REFDIST_OSF_URL=OSF,
           PYTHONPATH="/kaggle/working/code")
subprocess.run([sys.executable, "/kaggle/working/code/scripts/kaggle_phase.py", PHASE],
               check=True, env=env, cwd="/kaggle/working/code")
'''


PREV = {"0": None, "0b": "0", "1": "0b", "2": "1"}  # each phase builds on the previous one's output


def push(phase: str, osf: str = "") -> None:
    if phase not in ("0", "0b") and not osf:
        from refdist import config
        osf = config.STUDY["osf"]
        if not osf:
            raise SystemExit("phase 1/2 need the OSF registration URL: set STUDY['osf'] in config.py")
    d = BUILD / f"phase{phase}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(STUB.format(phase=phase, osf=osf))
    u = user()
    meta = {
        "id": f"{u}/refdist-phase{phase}", "title": f"refdist-phase{phase}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{u}/refdist-bundle"],
        "kernel_sources": [f"{u}/refdist-phase{PREV[phase]}"] if PREV.get(phase) else [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(kaggle("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4"))


def status(phase: str) -> None:
    print(kaggle("kernels", "status", f"{user()}/refdist-phase{phase}"))


def pull(phase: str) -> None:
    out = ROOT / "data" / "kaggle" / f"phase{phase}"
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", f"{user()}/refdist-phase{phase}", "-p", str(out), "-o"))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"bundle": bundle, "push": push, "status": status, "pull": pull}[cmd](*rest)
