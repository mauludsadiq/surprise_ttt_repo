from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .config import TTTcfg
from .data import iter_token_batches, load_text, make_stream_ids
from .models.toy_lm import ToyLMConfig, ToyTransformerLM
from .tokenizer import SimpleTokenizer
from .updater import AnchoredSurpriseTTT
from .utils import set_seed


@dataclass(frozen=True)
class TTTResult:
    losses: List[float]
    gammas: List[float]
    lambdas: List[float]
    s: List[float]
    s_ema: List[float]


def load_checkpoint(path: str, device: str = "cpu") -> tuple[ToyTransformerLM, SimpleTokenizer]:
    ckpt = torch.load(path, map_location=device)
    tok = SimpleTokenizer(stoi={t: i for i, t in enumerate(ckpt["tokenizer"]["itos"])}, itos=ckpt["tokenizer"]["itos"])
    model_cfg = ToyLMConfig(**ckpt["model_cfg"])
    model = ToyTransformerLM(model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, tok


def run_ttt(
    corpus_path: str,
    ckpt_path: str,
    out_path: str,
    cfg: TTTcfg,
    device: str = "cpu",
) -> TTTResult:
    set_seed(cfg.seed)

    model, tok = load_checkpoint(ckpt_path, device=device)
    text = load_text(corpus_path)
    stream = make_stream_ids(tok, text)

    updater = AnchoredSurpriseTTT(
        model=model,
        subset=cfg.update_subset,
        ema_alpha=cfg.ema.alpha,
        gate_kind=cfg.gate.kind,
        eta=cfg.gate.eta,
        tau=cfg.gate.tau,
        eps=cfg.gate.eps,
        temp=cfg.gate.temp,
        lambda_=cfg.retention.lambda_,
        lambda_schedule=cfg.retention.schedule,
        inv_c=cfg.retention.inv_c,
        decay_beta=cfg.retention.decay_beta,
    )

    losses: List[float] = []
    gammas: List[float] = []
    lambdas: List[float] = []
    s: List[float] = []
    s_ema: List[float] = []

    model.train()
    it = iter_token_batches(stream, batch_tokens=cfg.batch_tokens, batch_size=1)
    pbar = tqdm(range(cfg.max_steps), desc="ttt", ncols=110)
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter_token_batches(stream, batch_tokens=cfg.batch_tokens, batch_size=1)
            batch = next(it)
        ids = batch.ids.to(device)
        loss = model.loss_next_token(ids)
        stats = updater.step_update(loss)
        losses.append(stats.loss)
        gammas.append(stats.gamma)
        lambdas.append(stats.lambda_t)
        s.append(stats.s)
        s_ema.append(stats.s_ema)
        pbar.set_postfix(loss=stats.loss, gamma=stats.gamma, lam=stats.lambda_t, s=stats.s_ema)

    payload = {
        "cfg": cfg.model_dump(),
        "tokenizer": {"itos": tok.itos},
        "model_cfg": model.cfg.__dict__,
        "state_dict": model.state_dict(),
        "metrics": {
            "losses": losses,
            "gammas": gammas,
            "lambdas": lambdas,
            "s": s,
            "s_ema": s_ema,
        },
    }
    torch.save(payload, out_path)
    return TTTResult(losses=losses, gammas=gammas, lambdas=lambdas, s=s, s_ema=s_ema)
