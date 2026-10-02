"""
Transform protocol for H5.

A transform finds every eligible edit site in a document and knows the
replacement for each. Applying it at intensity p edits the first ceil(p * n)
sites of a fixed per-document permutation, so intensities are nested: the 25%
edit set is a subset of the 50% set, which is a subset of the 100% set. That
makes the dose-response within a document a clean monotone series.

Every applied edit is recorded, so any variant can be audited or reversed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Site:
    start: int
    end: int
    original: str
    replacement: str


class Transform:
    key: str = ""
    name: str = ""
    direction: str = ""

    def sites(self, text: str) -> list[Site]:
        raise NotImplementedError


def _non_overlapping(sites: list[Site]) -> list[Site]:
    out, last_end = [], -1
    for s in sorted(sites, key=lambda s: (s.start, -s.end)):
        if s.start >= last_end:
            out.append(s)
            last_end = s.end
    return out


def site_order(doc_id: str, transform_key: str, n: int, seed: int) -> np.ndarray:
    """Fixed permutation per (doc, transform), independent of intensity."""
    h = int(hashlib.sha256(f"{seed}:{doc_id}:{transform_key}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h).permutation(n)


def apply(transform: Transform, text: str, intensity: float, doc_id: str,
          seed: int) -> tuple[str, list[Site], int]:
    """Returns (new_text, applied_sites, n_eligible)."""
    sites = _non_overlapping(transform.sites(text))
    k = math.ceil(intensity * len(sites) - 1e-9)  # any positive intensity edits >= 1 site
    if k == 0:
        return text, [], len(sites)
    chosen = sorted((sites[i] for i in site_order(doc_id, transform.key, len(sites), seed)[:k]),
                    key=lambda s: s.start)
    parts, pos = [], 0
    for s in chosen:
        parts.append(text[pos:s.start])
        parts.append(s.replacement)
        pos = s.end
    parts.append(text[pos:])
    return "".join(parts), chosen, len(sites)


def match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl
