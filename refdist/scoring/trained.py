"""Pass C: trained classifiers (D3-D5). One logit per document, no scorer LM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


@torch.inference_mode()
def classify(model, tok, texts: list[str], ids: list[str], ai_label_index: int,
             batch_size: int = 16) -> pd.DataFrame:
    """P(AI) at the detector's documented AI label index. Higher = more machine-like."""
    dev = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    out_ids, out_scores = [], []
    for i in tqdm(range(0, len(order), batch_size), leave=False):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(dev)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, ai_label_index]
        out_ids += [ids[j] for j in idx]
        out_scores += probs.cpu().tolist()
    return pd.DataFrame({"doc_id": out_ids, "score": out_scores})
