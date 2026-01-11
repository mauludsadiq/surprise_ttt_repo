from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ToyLMConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 2048


class ToyTransformerLM(nn.Module):
    """Tiny Transformer encoder LM.

    Inputs: token ids [B, T]
    Outputs: logits [B, T, V]
    """

    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg.num_layers, enable_nested_tensor=False)
        except TypeError:
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)

        # causal mask: True blocks attention
        mask = torch.triu(torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1)
        x = self.enc(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def loss_next_token(self, ids: torch.Tensor) -> torch.Tensor:
        # predict token t from prefix up to t-1
        logits = self.forward(ids[:, :-1])
        targets = ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def make_toy_model(
    vocab_size: int,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    max_len: int = 2048,
    dropout: float = 0.1,
    device: str = "cpu",
) -> "ToyTransformerLM":
    cfg = ToyLMConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
    )
    m = ToyTransformerLM(cfg)
    return m.to(device)

ToyLM = ToyTransformerLM
