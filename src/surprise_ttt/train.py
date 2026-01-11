from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .data import TokenBatch, iter_token_batches, load_text, make_stream_ids
from .models.toy_lm import ToyLMConfig, ToyTransformerLM
from .tokenizer import SimpleTokenizer
from .utils import set_seed


@dataclass(frozen=True)
class PretrainCfg:
    seed: int = 7
    batch_tokens: int = 128
    steps: int = 500
    lr: float = 3e-4
    weight_decay: float = 0.0
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 2048


def pretrain_toy_lm(
    corpus_path: str,
    out_path: str,
    cfg: PretrainCfg,
    device: str = "cpu",
) -> None:
    set_seed(cfg.seed)
    text = load_text(corpus_path)
    tok = SimpleTokenizer.build(text)
    stream = make_stream_ids(tok, text)

    model = ToyTransformerLM(
        ToyLMConfig(
            vocab_size=tok.vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
        )
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    it = iter_token_batches(stream, batch_tokens=cfg.batch_tokens, batch_size=1)
    pbar = tqdm(range(cfg.steps), desc="pretrain", ncols=100)
    model.train()
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter_token_batches(stream, batch_tokens=cfg.batch_tokens, batch_size=1)
            batch = next(it)

        ids = batch.ids.to(device)
        loss = model.loss_next_token(ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss.detach().item()))

    payload = {
        "tokenizer": {"itos": tok.itos},
        "model_cfg": model.cfg.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, out_path)
