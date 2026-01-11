from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .ema import EMA
from .gating import BudgetGate, SigmoidGate
from .param_select import select_params


@dataclass
class UpdateStats:
    loss: float
    s: float
    s_ema: float
    gamma: float
    lambda_t: float
    grad_norm: float
    delta_norm: float


class AnchoredSurpriseTTT:
    """Applies surprise-gated TTT update with anchored retention.

    Update:
      θ <- θ - γ(∇ℓ(θ) + λ(θ-θ0))

    Implementation detail: θ0 is stored as detached clones (anchor copy).
    """

    def __init__(
        self,
        model: nn.Module,
        subset: str,
        ema_alpha: float,
        gate_kind: str,
        eta: float,
        tau: float,
        eps: float,
        temp: float,
        lambda_: float,
        lambda_schedule: str = "constant",
        inv_c: float = 1.0,
        decay_beta: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.subset = subset
        self.params: List[nn.Parameter] = select_params(model, subset)
        self.anchor: List[torch.Tensor] = [p.detach().clone() for p in self.params]
        self.ema = EMA(alpha=ema_alpha)
        self.lambda0 = float(lambda_)
        self.lambda_schedule = lambda_schedule
        self.inv_c = float(inv_c)
        self.decay_beta = float(decay_beta)
        self.step = 0

        if gate_kind == "budget":
            self.gate = BudgetGate(eta=eta, tau=tau, eps=eps)
        elif gate_kind == "sigmoid":
            self.gate = SigmoidGate(eta=eta, tau=tau, temp=temp)
        else:
            raise ValueError(f"Unknown gate kind: {gate_kind}")

        self.device = device

    def _lambda_t(self, s_ema: float) -> float:
        if self.lambda_schedule == "constant":
            return self.lambda0
        if self.lambda_schedule == "inv_surprise":
            # larger surprise => smaller retention (allow more drift)
            return self.lambda0 * (self.inv_c / (s_ema + 1e-12))
        if self.lambda_schedule == "exp_decay":
            return self.lambda0 * torch.exp(torch.tensor(-self.decay_beta * self.step)).item()
        raise ValueError(f"Unknown lambda_schedule: {self.lambda_schedule}")

    @torch.no_grad()
    def _apply_anchor_grad(self, lambda_t: float) -> None:
        # Adds retention gradient: λ(θ-θ0)
        for p, a in zip(self.params, self.anchor):
            if p.grad is None:
                continue
            p.grad.add_(lambda_t * (p.data - a))

    @staticmethod
    def _grad_norm(params: List[nn.Parameter]) -> float:
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            total += float(p.grad.detach().pow(2).sum().item())
        return float(total) ** 0.5

    def step_update(self, loss: torch.Tensor) -> UpdateStats:
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        s = self._grad_norm(self.params)
        s_ema = self.ema.update(s)
        gamma = float(self.gate(s_ema))
        lambda_t = self._lambda_t(s_ema)

        # Add retention gradient and then take step with effective step size gamma
        self._apply_anchor_grad(lambda_t=lambda_t)

        before = [p.data.clone() for p in self.params]
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-gamma)
        after = self.params

        delta2 = 0.0
        for b, p in zip(before, after):
            delta2 += float((p.data - b).pow(2).sum().item())
        delta_norm = delta2 ** 0.5

        self.step += 1
        return UpdateStats(
            loss=float(loss.detach().item()),
            s=s,
            s_ema=s_ema,
            gamma=gamma,
            lambda_t=lambda_t,
            grad_norm=s,
            delta_norm=delta_norm,
        )
