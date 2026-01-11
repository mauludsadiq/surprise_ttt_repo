from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class BudgetGate:
    """gamma = eta * clip(tau / (s + eps), 0, 1)

    Interpretable as a step budget / KL budget proxy.
    """

    eta: float
    tau: float
    eps: float = 1e-8

    def __call__(self, s: float) -> float:
        r = self.tau / (s + self.eps)
        return self.eta * clip(r, 0.0, 1.0)


@dataclass(frozen=True)
class SigmoidGate:
    """Smooth gate: gamma = eta * sigmoid((tau - s)/temp)."""

    eta: float
    tau: float
    temp: float = 1.0

    def __call__(self, s: float) -> float:
        z = (self.tau - s) / max(self.temp, 1e-12)
        return self.eta * (1.0 / (1.0 + math.exp(-z)))
