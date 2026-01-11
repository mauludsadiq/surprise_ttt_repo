from __future__ import annotations

from typing import Iterable, List, Tuple

import torch.nn as nn


def select_params(model: nn.Module, subset: str) -> List[nn.Parameter]:
    subset = subset.lower()
    ps: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if subset == "all":
            ps.append(p)
        elif subset == "mlp":
            # in torch TransformerEncoderLayer, FFN weights are typically in linear1/linear2
            if any(k in name for k in ["linear1", "linear2", "dim_feedforward"]):
                ps.append(p)
        elif subset == "ln":
            if "norm" in name or "ln" in name:
                ps.append(p)
        elif subset == "head":
            if name.endswith("head.weight") or "head" in name:
                ps.append(p)
        else:
            raise ValueError(f"Unknown subset: {subset}")

    if not ps:
        # fallback: if name matching fails due to upstream changes, just use all
        if subset != "all":
            return select_params(model, "all")
    return ps
