import numpy as np
from surprise_ttt.stability import Quadratic, step_quadratic, non_osc_gamma_upper


def test_non_osc_contracts_in_quadratic() -> None:
    # H = diag([2, 10]) has eigmax=10
    H = np.diag([2.0, 10.0])
    theta_star = np.array([0.0, 0.0])
    theta0 = np.array([0.0, 0.0])
    q = Quadratic(H=H, theta_star=theta_star, theta0=theta0)

    lam = 1.0
    gamma = 0.9 * non_osc_gamma_upper(H, lam)

    theta = np.array([10.0, -10.0])
    n0 = np.linalg.norm(theta - theta_star)
    for _ in range(50):
        theta = step_quadratic(theta, q, gamma=gamma, lam=lam)

    n1 = np.linalg.norm(theta - theta_star)
    assert n1 < 1e-3 * n0
