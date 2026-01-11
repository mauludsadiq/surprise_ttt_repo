from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .models import toy_lm as toy_lm_mod


def _resolve_model_class() -> type:
    for name in ("ToyLM", "ToyLMModel", "ToyTransformerLM", "ToyLanguageModel"):
        if hasattr(toy_lm_mod, name):
            v = getattr(toy_lm_mod, name)
            if isinstance(v, type) and issubclass(v, nn.Module):
                return v
    for v in toy_lm_mod.__dict__.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise ImportError("No nn.Module subclass found in surprise_ttt.models.toy_lm")


def _resolve_cfg_class() -> Optional[type]:
    for name in ("ToyLMCfg", "ToyLMConfig", "ToyTransformerCfg", "ToyLMcfg"):
        if hasattr(toy_lm_mod, name):
            v = getattr(toy_lm_mod, name)
            if isinstance(v, type):
                return v
    return None


def make_toy_model(
    *,
    vocab_size: int = 256,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    max_len: int = 256,
    dropout: float = 0.0,
    device: str = "cpu",
) -> nn.Module:
    Model = _resolve_model_class()
    Cfg = _resolve_cfg_class()

    # If toy_lm exposes an explicit factory, use it as the single source of truth.
    if hasattr(toy_lm_mod, "make_toy_model"):
        return toy_lm_mod.make_toy_model(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout,
            device=device,
        )

    kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
    )

    sig = inspect.signature(Model.__init__)
    if "cfg" in sig.parameters and Cfg is not None:
        cfg = Cfg(**kwargs)
        m = Model(cfg)
    else:
        try:
            m = Model(**kwargs)
        except TypeError:
            if Cfg is None:
                raise
            cfg = Cfg(**kwargs)
            m = Model(cfg)

    return m.to(torch.device(device))
