import torch

from surprise_ttt.retention_gate import retention_lambda_miras


def test_miras_regime_noise_vs_flashbulb():
    lam0 = 0.2

    lam_flash = retention_lambda_miras(
        lam0=lam0,
        s_t=10.0,
        sbar_t=1.0,
        s_ref=1.0,
        spike_k=4.0,
        lam_max=0.5,
    )

    lam_noise = retention_lambda_miras(
        lam0=lam0,
        s_t=10.0,
        sbar_t=10.0,
        s_ref=1.0,
        spike_k=4.0,
        lam_max=0.5,
    )

    assert float(lam_noise) > float(lam_flash)


def test_retention_pull_prevents_divergence_trust_radius():
    torch.manual_seed(0)

    d = 100
    theta0 = torch.zeros(d)
    theta = torch.zeros(d)

    grad = torch.ones(d) * 10.0
    gamma = 0.05
    step = -gamma * grad
    step_norm = float(step.norm())

    lam = retention_lambda_miras(
        lam0=0.2,
        s_t=10.0,
        sbar_t=10.0,
        s_ref=1.0,
        spike_k=4.0,
        lam_max=0.5,
    )
    lam = float(lam)

    assert 0.0 < lam < 1.0

    for _ in range(200):
        theta = theta + step + lam * (theta0 - theta)

    final_dist = float((theta - theta0).norm())

    radius = step_norm / lam
    assert final_dist <= radius * 1.05
