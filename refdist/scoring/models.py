"""Model loading with pinned revisions. The only place weights are fetched."""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    # T4 has no bf16; fp16 halves memory. CPU runs stay in fp32.
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _measured_dtype(model_id: str) -> torch.dtype | None:
    """Precision chosen for this model by Milestone 0's fp16-vs-fp32 check, if it ran."""
    import json
    from refdist import paths
    f = paths.results() / "precision.json"
    if not f.exists() or not torch.cuda.is_available():
        return None
    entry = json.loads(f.read_text()).get("models", {}).get(model_id)
    return getattr(torch, entry["dtype"]) if entry else None


PROBE = ("The committee met on Tuesday to review the proposal. After a long discussion, "
         "they agreed that the budget needed revision before the next meeting in March.")


@torch.inference_mode()
def _finite(model, tok) -> bool:
    ids = tok(PROBE, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    lp = torch.log_softmax(model(ids).logits.float(), -1)
    return bool(torch.isfinite(lp).all())


def load_causal_lm(model_id: str, revision: str, dtype: torch.dtype | None = None):
    """
    fp16 on GPU unless the model overflows there. Several bf16-trained models
    (Qwen2.5, OLMo-2) can produce inf/NaN in fp16, and a T4 has no native bf16,
    so a probe batch decides; on failure the model is reloaded in fp32.
    """
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dt = dtype or _measured_dtype(model_id) or _dtype()
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dt).to(device())
    model.eval()
    if dtype is None and dt == torch.float16 and not _finite(model, tok):
        print(f"  [precision] {model_id}: non-finite in fp16, reloading in fp32")
        free(model)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                     dtype=torch.float32).to(device())
        model.eval()
    model.refdist_dtype = str(next(model.parameters()).dtype)
    return model, tok


def load_classifier(model_id: str, revision: str):
    # Always fp32: these RoBERTa classifiers are small (125-355M), and fp32 means
    # their scores need no precision check of their own.
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, dtype=torch.float32
    ).to(device())
    model.eval()
    return model, tok


def free(*objs) -> None:
    """Callers must also drop their own references (e.g. `model = None`)."""
    import gc
    for o in objs:
        if hasattr(o, "to"):
            o.to("cpu")  # moves weights off the GPU even if a reference survives
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
