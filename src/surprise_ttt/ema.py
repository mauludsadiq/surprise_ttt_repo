from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EMA:
    """Simple exponential moving average tracker.

    update: x -> state := alpha * state + (1-alpha) * x

    alpha in [0,1). Larger alpha => longer memory.
    """

    alpha: float
    value: float = 0.0
    initialized: bool = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
            return self.value
        self.value = self.alpha * self.value + (1.0 - self.alpha) * x
        return self.value

    @property
    def time_constant(self) -> float:
        # Approx tokens for 1-e^{-1} response
        if self.alpha >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - self.alpha)
