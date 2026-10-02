"""
Milestone 0 (CPU locally, or GPU on Kaggle): prove the scoring math before spending GPU hours.

Three checks, each of which catches a bug that would otherwise corrupt every
score in the study without raising an error:

  1. Analytic Fast-DetectGPT terms (mu, var) match a Monte-Carlo estimate.
  2. Batched scoring equals one-document-at-a-time scoring (padding/mask bugs).
  3. Machine text scores higher than human text (orientation / sign bugs).

Human text is real public-domain prose from Project Gutenberg. Machine text is
sampled from the same small model, so this is a white-box sanity check of the
harness, not a detection benchmark.

Usage:  python scripts/milestone0_local.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refdist.scoring import detectors, logprob_cache  # noqa: E402
from refdist.scoring.models import load_causal_lm  # noqa: E402

MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GUTENBERG = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"  # Pride and Prejudice
N = 20
torch.manual_seed(0)


def human_paragraphs(n: int) -> list[str]:
    raw = urllib.request.urlopen(GUTENBERG, timeout=60).read().decode("utf-8", "ignore")
    body = raw.split("*** START OF")[1].split("*** END OF")[0]
    paras = [re.sub(r"\s+", " ", p).strip() for p in body.split("\r\n\r\n")]
    paras = [p for p in paras if 120 <= len(p.split()) <= 220 and "CHAPTER" not in p]
    return paras[10:10 + n]


@torch.inference_mode()
def machine_texts(model, tok, prompts: list[str]) -> list[str]:
    out = []
    for p in prompts:
        ids = tok(" ".join(p.split()[:8]), return_tensors="pt").input_ids.to(model.device)
        gen = model.generate(ids, max_new_tokens=200, do_sample=True, top_k=50,
                             pad_token_id=tok.eos_token_id)
        out.append(tok.decode(gen[0], skip_special_tokens=True))
    return out


@torch.inference_mode()
def check_analytic_vs_montecarlo(model, tok, text: str, samples: int = 20000) -> None:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :128].to(model.device)
    lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1).cpu()
    mu = (lp.exp() * lp).sum(-1)
    var = (lp.exp() * lp.square()).sum(-1) - mu.square()
    draws = torch.multinomial(lp.exp(), samples, replacement=True)  # [T, S]
    sampled = lp.gather(1, draws).sum(0)                            # sum over positions
    mc_mu, mc_var = sampled.mean().item(), sampled.var().item()
    an_mu, an_var = mu.sum().item(), var.sum().item()
    print(f"  sum mu : analytic {an_mu:9.3f}  monte-carlo {mc_mu:9.3f}")
    print(f"  sum var: analytic {an_var:9.3f}  monte-carlo {mc_var:9.3f}")
    assert abs(an_mu - mc_mu) < 0.02 * abs(an_mu) + 0.5, "mu mismatch"
    assert abs(an_var - mc_var) < 0.05 * an_var + 0.5, "var mismatch"


PRECISION_MAX_DIFF_SD = 0.05   # fp16 kept only if |fp16 - fp32| < 5% of the score SD ...
PRECISION_MIN_SPEARMAN = 0.999  # ... and the document ranking is essentially unchanged


def check_precision(human: list[str], machine: list[str]) -> dict:
    """
    [4] GPU only. For every model the study runs, score the same 40 non-study
    texts in fp16 and fp32. fp16 is kept only where it reproduces fp32; the
    verdict is written to results/precision.json and read by the runner.
    """
    from scipy.stats import spearmanr
    from refdist import config, paths
    from refdist.provenance import write_json
    from refdist.scoring.models import free
    texts, ids = human + machine, [f"t{i}" for i in range(len(human) + len(machine))]
    models = {config.INDEPENDENT_PREDICTOR["model_id"]: config.INDEPENDENT_PREDICTOR["revision"]}
    for r in config.REFERENCE_MODELS.values():
        models[r["model_id"]] = r["revision"]
        if r.get("binoculars_pair"):
            models[r["binoculars_pair"]] = r["binoculars_pair_revision"]
    verdict = {}
    for mid, rev in models.items():
        res = {}
        for dt in (torch.float32, torch.float16):
            m, t = load_causal_lm(mid, rev, dtype=dt)
            try:
                res[dt] = detectors.fast_detectgpt(logprob_cache.token_stats(m, t, texts, ids)).set_index("doc_id")["score"]
            except FloatingPointError:
                res[dt] = None
            free(m)
            torch.cuda.empty_cache()
        a, b = res[torch.float32], res[torch.float16]
        if b is None:
            verdict[mid] = {"dtype": "float32", "reason": "fp16 produced non-finite values"}
        else:
            b = b.reindex(a.index)
            ratio = float((a - b).abs().max() / a.std())
            rho = float(spearmanr(a, b).statistic)
            keep16 = ratio < PRECISION_MAX_DIFF_SD and rho > PRECISION_MIN_SPEARMAN
            verdict[mid] = {"dtype": "float16" if keep16 else "float32",
                            "max_diff_over_sd": ratio, "spearman": rho}
        print(f"  {mid:45s} -> {verdict[mid]}")
    write_json(paths.results() / "precision.json", {"models": verdict,
               "rule": {"max_diff_over_sd": PRECISION_MAX_DIFF_SD, "min_spearman": PRECISION_MIN_SPEARMAN}})
    return verdict


def main() -> None:
    dev = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"loading {MODEL} in fp32 on {dev} ...")
    # fp32: these checks test the scoring logic, so rounding must not be able to fail them.
    model, tok = load_causal_lm(MODEL, REVISION, dtype=torch.float32)

    human = human_paragraphs(N)
    print(f"human paragraphs: {len(human)}")
    machine = machine_texts(model, tok, human)

    print()
    print("[1] analytic terms vs Monte-Carlo")
    check_analytic_vs_montecarlo(model, tok, human[0])
    print("  PASS")

    print()
    print("[2] batched == single-document")
    ids = [f"h{i}" for i in range(len(human))]
    batched = detectors.fast_detectgpt(logprob_cache.token_stats(model, tok, human, ids))
    singles = [detectors.fast_detectgpt(logprob_cache.token_stats(model, tok, [t], [i]))
               for t, i in zip(human, ids)]
    single = dict(zip(ids, [s["score"].iloc[0] for s in singles]))
    diff = max(abs(r.score - single[r.doc_id]) for r in batched.itertuples())
    print(f"  max |batched - single| = {diff:.2e}")
    assert diff < 1e-3, "padding or masking changes scores"
    print("  PASS")

    print()
    print("[3] machine text scores higher than human text")
    mids = [f"m{i}" for i in range(len(machine))]
    mscore = detectors.fast_detectgpt(logprob_cache.token_stats(model, tok, machine, mids))
    y = [0] * len(batched) + [1] * len(mscore)
    s = list(batched["score"]) + list(mscore["score"])
    auroc = roc_auc_score(y, s)
    print(f"  human mean {batched['score'].mean():6.2f} | machine mean {mscore['score'].mean():6.2f}"
          f" | AUROC {auroc:.3f}")
    assert auroc >= 0.80, "harness fails positive control"
    print("  PASS")

    if torch.cuda.is_available():
        print()
        print("[4] fp16 vs fp32, every study model, non-study texts")
        check_precision(human, machine)

    print()
    print(f"Milestone 0 ({dev}): all checks passed.")


if __name__ == "__main__":
    main()
