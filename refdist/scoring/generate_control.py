"""
Positive control: the ONLY place in the study where text is generated.

500 answers written by Qwen2.5-1.5B-Instruct, each matched to one S6 document's
site and length. Its single purpose is to prove the scoring harness works: a
detector that cannot separate these from S6 at AUROC >= 0.80 is excluded, and
the exclusion is reported. The output goes to control.parquet, which is never
concatenated into documents.parquet and is never described as a benchmark.

Run on a GPU (Kaggle):  python -m refdist.scoring.generate_control
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from refdist import config, paths, provenance
from refdist.corpora.base import clean, n_tokens, take_windows, windows
from refdist.scoring.models import load_causal_lm

SITE_TOPIC = {
    "english": "English language and usage",
    "travel": "travel",
    "cooking": "cooking",
}


def _prompt(site: str, first_sentence: str, n_words: int) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            f"Write an answer to a question on the {SITE_TOPIC.get(site, site)} Stack Exchange site. "
            f"The question is about the topic of this sentence: \"{first_sentence}\". "
            f"Write about {n_words} words as a knowledgeable community member. "
            f"Output only the answer text."
        ),
    }]


def _raw_file():
    return paths.processed() / "control_raw.parquet"


@torch.inference_mode()
def generate() -> pd.DataFrame:
    """
    Generate until n_docs normalised, in-window control documents exist. The
    first n_docs seeds are a fixed sample of S6; if normalisation drops some,
    further seeds come from a second fixed permutation of the remaining S6
    documents. Raw generations are kept in control_raw.parquet for audit.
    """
    cc = config.CONTROL
    docs = pd.read_parquet(paths.documents_file())
    s6 = docs[docs["stratum"] == cc["match_stratum"]].sort_values("doc_id")
    rng = np.random.default_rng(config.CORPUS["seed"])
    first = np.sort(rng.choice(len(s6), size=cc["n_docs"], replace=False))
    rest = np.setdiff1d(np.arange(len(s6)), first)
    order = np.concatenate([first, np.random.default_rng(config.CORPUS["seed"] + 1).permutation(rest)])
    seeds = s6.iloc[order]

    raw_f = _raw_file()
    if not raw_f.exists() and paths.control_file().exists() and "seed_doc_id" in pd.read_parquet(paths.control_file()):
        old = pd.read_parquet(paths.control_file())
        if (old["n_tokens"] > config.CORPUS["max_tokens"]).any():  # pre-normalisation output
            paths.control_file().replace(raw_f)
    rows = pd.read_parquet(raw_f).to_dict("records") if raw_f.exists() else []
    done = {r["seed_doc_id"] for r in rows}

    model = tok = None
    pos = 0
    while len(final := _final(rows)) < cc["n_docs"] and pos < len(seeds):
        need = cc["n_docs"] - len(final)
        todo = seeds.iloc[pos:][~seeds.iloc[pos:]["doc_id"].isin(done)].head(max(8, int(need * 1.3)))
        pos = seeds.index.get_loc(todo.index[-1]) + 1 if len(todo) else len(seeds)
        if model is None:
            model, tok = load_causal_lm(cc["model_id"], cc["revision"])
            tok.padding_side = "left"
        for i in tqdm(range(0, len(todo), 8), desc="control"):
            batch = todo.iloc[i:i + 8]
            prompts = [tok.apply_chat_template(
                _prompt(r.source_doc_id.split(":")[0], r.text.split(". ")[0][:200], int(len(r.text.split()) * 1.1)),
                tokenize=False, add_generation_prompt=True) for r in batch.itertuples()]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=560, do_sample=True, temperature=0.7,
                                 top_p=0.9, pad_token_id=tok.pad_token_id)
            for r, seq in zip(batch.itertuples(), out[:, enc["input_ids"].shape[1]:]):
                text = tok.decode(seq, skip_special_tokens=True).strip()
                rows.append({"doc_id": f"C-{r.doc_id}", "seed_doc_id": r.doc_id, "text": text,
                             "n_tokens": n_tokens(text), "generator": cc["model_id"],
                             "text_sha256": provenance.sha256_text(text)})
                done.add(r.doc_id)
            pd.DataFrame(rows).to_parquet(raw_f, index=False)  # resumable
    final = _final(rows)
    final.to_parquet(paths.control_file(), index=False)
    return final


def _final(rows: list[dict]) -> pd.DataFrame:
    """Normalised windows in seed order, first n_docs within the token window."""
    if not rows:
        return pd.DataFrame(columns=["doc_id", "text", "n_tokens"])
    f = finalize(pd.DataFrame(rows))
    f = f[f["n_tokens"].between(config.CORPUS["min_tokens"], config.CORPUS["max_tokens"])]
    return f.head(config.CONTROL["n_docs"])


_MD = [(re.compile(r"\*\*(.+?)\*\*"), r"\1"), (re.compile(r"__(.+?)__"), r"\1"),
       (re.compile(r"(?m)^#{1,6}\s*"), ""), (re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+"), ""),
       (re.compile(r"`([^`]*)`"), r"\1")]


def normalize(text: str) -> str:
    """
    The same surface treatment the human corpus received: Stack Exchange HTML
    was flattened to plain paragraphs, so markdown emphasis, headers and list
    markers are removed here. Otherwise a detector could separate control from
    S6 on asterisks alone, and the gate would prove nothing.
    """
    for pat, rep in _MD:
        text = pat.sub(rep, text)
    return clean(text)


def finalize(raw: pd.DataFrame) -> pd.DataFrame:
    """One 150-400 token window per generation, cut exactly as human documents are."""
    rows = []
    for r in raw.itertuples():
        rng = np.random.default_rng(int(r.text_sha256[:8], 16))
        for w in take_windows(windows(normalize(r.text).split("\n"), rng), 1, rng):
            rows.append({"doc_id": r.doc_id, "seed_doc_id": r.seed_doc_id, "text": w,
                         "n_tokens": n_tokens(w), "generator": r.generator,
                         "text_sha256": provenance.sha256_text(w)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    c = generate()
    print(f"{len(c)} control documents, median {int(c['n_tokens'].median())} tokens")
