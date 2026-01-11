from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Quadratic:
    """Quadratic objective: l(theta)=0.5*(theta-theta*)^T H (theta-theta*)"""
    H: np.ndarray
    theta_star: np.ndarray
    theta0: np.ndarray


def step_quadratic(theta: np.ndarray, q: Quadratic, gamma: float, lam: float) -> np.ndarray:
    grad = q.H @ (theta - q.theta_star)
    # retention grad: lam*(theta-theta0)
    grad = grad + lam * (theta - q.theta0)
    return theta - gamma * grad


def stable_gamma_upper(H: np.ndarray, lam: float) -> float:
    eigmax = float(np.linalg.eigvalsh(H).max())
    return 2.0 / (eigmax + lam)


def non_osc_gamma_upper(H: np.ndarray, lam: float) -> float:
    eigmax = float(np.linalg.eigvalsh(H).max())
    return 1.0 / (eigmax + lam)
