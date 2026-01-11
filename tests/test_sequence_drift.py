import torch

from surprise_ttt.retention_gate import retention_lambda_miras


def test_sequence_level_drift_grow_then_shrink():
    print("TEST  Sequence-level drift: ||θ-θ0|| grows in clean phase, shrinks in sustained-noise phase (MIRAS).")

    torch.manual_seed(0)

    d = 256
    theta0 = torch.zeros(d)
    theta = theta0.clone()

    alpha = 0.95
    sbar = None

    eta = 0.05
    tau = 1.0
    eps = 1e-8

    lam0 = 0.25
    s_ref = 1.0
    spike_k = 4.0
    lam_max = 0.5

    def step(theta, grad, s_t, sbar):
        if sbar is None:
            sbar = float(s_t)
        else:
            sbar = alpha * float(sbar) + (1.0 - alpha) * float(s_t)

        gamma = eta * min(1.0, tau / (float(s_t) + eps))

        lam = retention_lambda_miras(
            lam0=lam0,
            s_t=float(s_t),
            sbar_t=float(sbar),
            s_ref=s_ref,
            spike_k=spike_k,
            eps=eps,
            lam_max=lam_max,
        )
        lam = float(lam)

        theta = theta - gamma * grad + lam * (theta0 - theta)
        return theta, sbar, gamma, lam

    dist0 = float((theta - theta0).norm())

    clean_steps = 60
    s_clean = 0.2
    grad_clean = torch.ones(d)

    dists_clean = []
    gammas_clean = []
    lams_clean = []

    for _ in range(clean_steps):
        theta, sbar, gamma, lam = step(theta, grad_clean, s_clean, sbar)
        dists_clean.append(float((theta - theta0).norm()))
        gammas_clean.append(float(gamma))
        lams_clean.append(float(lam))

    dist_clean_end = dists_clean[-1]

    assert dist_clean_end > dist0 + 0.5, f"Expected drift to grow in clean phase. start={dist0:.4f} end={dist_clean_end:.4f}"
    assert min(gammas_clean) > 0.0
    assert max(lams_clean) < 0.25, f"Clean phase retention should stay relatively small. max_lam={max(lams_clean):.4f}"

    noise_steps = 120
    s_noise = 10.0

    dists_noise = []
    gammas_noise = []
    lams_noise = []

    for _ in range(noise_steps):
        grad_noise = torch.randn(d)
        theta, sbar, gamma, lam = step(theta, grad_noise, s_noise, sbar)
        dists_noise.append(float((theta - theta0).norm()))
        gammas_noise.append(float(gamma))
        lams_noise.append(float(lam))

    dist_noise_end = dists_noise[-1]

    assert max(lams_noise) > max(lams_clean), "Expected MIRAS retention to be stronger under sustained high surprise."
    assert sum(gammas_noise) / len(gammas_noise) < sum(gammas_clean) / len(gammas_clean), "Expected budget gate to shrink step size in high surprise."
    assert dist_noise_end < dist_clean_end * 0.85, f"Expected drift to shrink in sustained noise. clean_end={dist_clean_end:.4f} noise_end={dist_noise_end:.4f}"

    tail = dists_noise[-40:]
    assert tail[-1] < tail[0], f"Expected decreasing trend late in noise phase. tail_start={tail[0]:.4f} tail_end={tail[-1]:.4f}"

    print("PASS  Sequence-level drift behaves correctly under MIRAS: grow then shrink.")
