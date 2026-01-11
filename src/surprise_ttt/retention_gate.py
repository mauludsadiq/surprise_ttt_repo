from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import math

Number = Union[float, "torch.Tensor"]


def _is_torch(x) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _clip(x: Number, lo: float, hi: float) -> Number:
    if _is_torch(x):
        import torch
        return torch.clamp(x, min=lo, max=hi)
    return max(lo, min(hi, float(x)))


def retention_lambda_miras(
    *,
    lam0: float,
    s_t: Number,
    sbar_t: Number,
    s_ref: float = 1.0,
    spike_k: float = 4.0,
    eps: float = 1e-8,
    lam_max: float = 0.5,
) -> Number:
    if _is_torch(s_t) or _is_torch(sbar_t):
        import torch
        s_t = s_t if _is_torch(s_t) else torch.tensor(float(s_t))
        sbar_t = sbar_t if _is_torch(sbar_t) else torch.tensor(float(sbar_t))

        sustained = sbar_t / (sbar_t + s_ref)
        r = s_t / (sbar_t + eps)
        spike = torch.relu(r - 1.0)
        spike_suppress = 1.0 / (1.0 + spike_k * spike)
        lam = lam0 * sustained * spike_suppress
        return torch.clamp(lam, 0.0, lam_max)

    sustained = float(sbar_t) / (float(sbar_t) + s_ref)
    r = float(s_t) / (float(sbar_t) + eps)
    spike = max(0.0, r - 1.0)
    spike_suppress = 1.0 / (1.0 + spike_k * spike)
    lam = lam0 * sustained * spike_suppress
    return _clip(lam, 0.0, lam_max)
