from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
try:
    from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
    _HAS_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
    _HAS_TORCH_AMP = False

from surprise_ttt.ema import EMA
from surprise_ttt.param_subset import UpdateSubset, mark_trainable_subset
from surprise_ttt.sdp import configure_sdp
from surprise_ttt.runtime_defaults import apply_runtime_defaults


def iter_microbatches(ids: torch.Tensor, batch_tokens: int) -> Iterable[torch.Tensor]:
    b, t = ids.shape
    bt = max(1, int(batch_tokens))
    start = 0
    while start < t - 1:
        end = min(t - 1, start + bt)
        yield ids[:, start : end + 1]
        start = end


def budget_gate(eta: float, tau: float, s: float, eps: float = 1e-8) -> float:
    if s <= 0.0:
        return float(eta)
    g = tau / (s + eps)
    if g > 1.0:
        g = 1.0
    if g < 0.0:
        g = 0.0
    return float(eta * g)


def _autocast_dtype(amp_dtype: str) -> torch.dtype:
    if (amp_dtype or "bf16").lower() == "fp16":
        return torch.float16
    return torch.bfloat16

def _make_scaler(cfg):
    enabled = bool(getattr(cfg, 'amp', False)) and torch.cuda.is_available() and _autocast_dtype(getattr(cfg, 'amp_dtype', 'bf16')) == torch.float16
    if _HAS_TORCH_AMP:
        return AmpGradScaler('cuda', enabled=enabled)
    return AmpGradScaler(enabled=enabled)

def _amp_ctx(use_amp: bool, dtype: torch.dtype):
    if _HAS_TORCH_AMP:
        return amp_autocast('cuda', enabled=use_amp, dtype=dtype)
    return amp_autocast(enabled=use_amp, dtype=dtype)


def _snapshot_params(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _apply_update(
    params: List[torch.nn.Parameter],
    params0: List[torch.Tensor],
    lr: float,
    lam: float,
) -> None:
    with torch.no_grad():
        for p, p0 in zip(params, params0):
            if p.grad is None:
                continue
            p.add_(p.grad, alpha=-lr)
            if lam != 0.0:
                p.add_(p0 - p, alpha=lr * lam)


@dataclass
class AdvCfg:
    eta: float = 1e-3
    tau: float = 1.0
    lam: float = 0.0
    alpha: float = 0.9
    eps: float = 1e-8

    batch_tokens: int = 1
    amp: bool = False
    amp_dtype: str = "bf16"
    update_scope: str = "all"
    update_blocks: str = "all"
    sdp: str = "auto"


def run_ttt_adv(
    model: torch.nn.Module,
    ids: torch.Tensor,
    cfg: AdvCfg,
    device: torch.device,
) -> Dict[str, float]:
    apply_runtime_defaults(cfg)
    configure_sdp(getattr(cfg, "sdp", "auto"))

    model.to(device)
    model.train()

    trainable = mark_trainable_subset(
        model,
        UpdateSubset(scope=getattr(cfg, "update_scope", "all"), blocks=getattr(cfg, "update_blocks", "all")),
    )
    params0 = _snapshot_params(trainable)

    ema = EMA(alpha=float(cfg.alpha))
    scaler = _make_scaler(cfg)

    total_loss = 0.0
    steps = 0
    total_s = 0.0
    total_gamma = 0.0

    for chunk in iter_microbatches(ids.to(device), int(cfg.batch_tokens)):
        x = chunk[:, :-1]
        y = chunk[:, 1:]

        for p in trainable:
            if p.grad is not None:
                p.grad = None

        use_amp = bool(cfg.amp) and torch.cuda.is_available()
        with _amp_ctx(use_amp, _autocast_dtype(cfg.amp_dtype)):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(torch.optim.SGD(trainable, lr=1.0))
        else:
            loss.backward()

        s = 0.0
        with torch.no_grad():
            for p in trainable:
                if p.grad is None:
                    continue
                s += float(p.grad.detach().float().pow(2).sum().item())
        s = float(s ** 0.5)

        s_bar = float(ema.update(s))
        gamma = budget_gate(float(cfg.eta), float(cfg.tau), s_bar, float(cfg.eps))

        if scaler.is_enabled():
            for p in trainable:
                if p.grad is not None:
                    p.grad.detach_()
            _apply_update(trainable, params0, gamma, float(cfg.lam))
            scaler.update()
        else:
            _apply_update(trainable, params0, gamma, float(cfg.lam))

        total_loss += float(loss.detach().float().item())
        total_s += s_bar
        total_gamma += gamma
        steps += 1

    return {
        "steps": float(steps),
        "loss_mean": total_loss / max(1, steps),
        "surprise_ema_mean": total_s / max(1, steps),
        "gamma_mean": total_gamma / max(1, steps),
        "trainable_params": float(sum(p.numel() for p in trainable)),
    }
