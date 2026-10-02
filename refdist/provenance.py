"""Stamps every artifact with the configuration and code state that produced it."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import uuid
from pathlib import Path

from refdist import __version__, config


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def new_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def stamp() -> dict:
    return {
        "refdist_version": __version__,
        "config_hash": config.config_hash(),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**payload, "_provenance": stamp()}, indent=2), encoding="utf-8")
