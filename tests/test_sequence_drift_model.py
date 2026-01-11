import torch
import torch.nn.functional as F

from surprise_ttt.ema import EMA
from surprise_ttt.param_subset import UpdateSubset, mark_trainable_subset
from surprise_ttt.retention_gate import retention_lambda_miras
from surprise_ttt.testing import make_toy_model


def _subset_snapshot(params):
    return [p.detach().clone() for p in params]


def _subset_l2(params, snap):
    s = 0.0
    for p, p0 in zip(params, snap):
        s += float((p.detach() - p0).pow(2).sum())
    return s ** 0.5


def _l2_of_tensors(xs):
    s = 0.0
    for x in xs:
        s += float(x.detach().pow(2).sum())
    return s ** 0.5


def _flat_tensors(tensors):
    if not tensors:
        return torch.empty(0)
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)


def test_sequence_level_drift_on_model_subset_grow_then_shrink():
    print("TEST  Model-bound drift: subset ||θ-θ0|| grows on clean text, shrinks on sustained noise (MIRAS + budget + EMA).")

    torch.manual_seed(0)
    device = torch.device("cpu")

    model = make_toy_model(
        vocab_size=256,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_len=256,
        dropout=0.0,
        device="cpu",
    )
    model.train()

    trainable = mark_trainable_subset(model, UpdateSubset(scope="mlp", blocks="last"))
    theta0 = _subset_snapshot(trainable)

    ema = EMA(alpha=0.95)

    eta = 5e-2
    tau = 1.0
    eps = 1e-8

    lam0 = 0.25
    s_ref = 1.0
    spike_k = 4.0
    lam_max = 0.5

    def one_step(ids):
        model.zero_grad(set_to_none=True)
        ids = ids.to(device)

        x = ids[:, :-1]
        y = ids[:, 1:]

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        grads = torch.autograd.grad(loss, trainable, retain_graph=False, create_graph=False)

        g2 = 0.0
        for g in grads:
            g2 = g2 + float(g.detach().pow(2).sum())
        s_t = (g2 ** 0.5)

        sbar = ema.update(s_t)

        gamma = eta * min(1.0, tau / (s_t + eps))

        lam_t = retention_lambda_miras(
            lam0=lam0,
            s_t=s_t,
            sbar_t=sbar,
            s_ref=s_ref,
            spike_k=spike_k,
            eps=eps,
            lam_max=lam_max,
        )
        lam_t = float(lam_t)

        grad_deltas = []
        ret_deltas = []

        # u BEFORE the parameter update (use this for drift-direction diagnostics)
        u_before = _flat_tensors([p.detach() - p0 for p, p0 in zip(trainable, theta0)])

        with torch.no_grad():
            for p, g, p0 in zip(trainable, grads, theta0):
                p_before = p.detach().clone()

                grad_delta = (-gamma) * g.detach()
                p_after_grad = p_before + grad_delta

                ret_delta = lam_t * (p0 - p_after_grad)

                p.copy_(p_after_grad + ret_delta)

                grad_deltas.append(grad_delta)
                ret_deltas.append(ret_delta)

        grad_norm = _l2_of_tensors(grad_deltas)
        ret_norm = _l2_of_tensors(ret_deltas)
        u_vec = u_before
        d_grad = _flat_tensors(grad_deltas)
        d_ret = _flat_tensors(ret_deltas)
        d_tot = d_grad + d_ret
        dot_grad = float((u_vec * d_grad).sum()) if u_vec.numel() else 0.0
        dot_ret = float((u_vec * d_ret).sum()) if u_vec.numel() else 0.0
        dot_total = float((u_vec * d_tot).sum()) if u_vec.numel() else 0.0


        return float(loss.detach()), float(s_t), float(sbar), float(gamma), float(lam_t), float(grad_norm), float(ret_norm), float(dot_total), float(dot_grad), float(dot_ret)

    dist_start = _subset_l2(trainable, theta0)

    clean_steps = 60
    T = 64
    clean_ids = torch.full((1, T), 7, dtype=torch.long)

    dists_clean = []
    grad_norms_clean = []
    ret_norms_clean = []

    for _ in range(clean_steps):
        _, _, _, _, _, gnn, rnn, _, _, _ = one_step(clean_ids)
        dists_clean.append(_subset_l2(trainable, theta0))
        grad_norms_clean.append(gnn)
        ret_norms_clean.append(rnn)

    dist_clean_end = dists_clean[-1]
    assert dist_clean_end > dist_start + 0.25, f"Expected growth on clean phase. start={dist_start:.4f} end={dist_clean_end:.4f}"

    clean_g = sum(grad_norms_clean) / len(grad_norms_clean)
    clean_r = sum(ret_norms_clean) / len(ret_norms_clean)
    assert clean_g > clean_r, f"Expected learning-dominant updates on clean phase: avg||Δgrad||={clean_g:.6f} avg||Δret||={clean_r:.6f}"

    noise_steps = 120
    dists_noise = []
    grad_norms_noise = []
    ret_norms_noise = []
    dots_total_noise = []
    dots_grad_noise = []
    dots_ret_noise = []

    for _ in range(noise_steps):
        noise_ids = torch.randint(low=0, high=256, size=(1, T), dtype=torch.long)
        _, _, _, _, _, gnn, rnn, dot_total, dot_grad, dot_ret = one_step(noise_ids)
        dists_noise.append(_subset_l2(trainable, theta0))
        grad_norms_noise.append(gnn)
        ret_norms_noise.append(rnn)
        dots_total_noise.append(dot_total)
        dots_grad_noise.append(dot_grad)
        dots_ret_noise.append(dot_ret)

    dist_noise_end = dists_noise[-1]
    assert dist_noise_end < dist_clean_end * 0.90, f"Expected shrink under sustained noise. clean_end={dist_clean_end:.4f} noise_end={dist_noise_end:.4f}"
    tail_n = 40
    tail = dists_noise[-tail_n:]
    tail_mean = sum(tail) / len(tail)
    tail_max = max(tail)
    tail_min = min(tail)
    assert tail_max <= tail_min + 0.02, f"Expected bounded tail under sustained noise. tail_min={tail_min:.4f} tail_max={tail_max:.4f}"
    assert tail_mean <= 0.08, f"Expected small tail mean under sustained noise. tail_mean={tail_mean:.4f}"
    noise_g_tail = sum(grad_norms_noise[-tail_n:]) / tail_n
    noise_r_tail = sum(ret_norms_noise[-tail_n:]) / tail_n

    dots_tail = dots_total_noise[-tail_n:]
    neg_frac = sum(1 for z in dots_tail if z < 0.0) / tail_n
    mean_dot = sum(dots_tail) / tail_n
    assert neg_frac >= 0.60, f"Expected most noise-tail steps to project back toward θ0: neg_frac={neg_frac:.3f}"
    assert mean_dot < 0.0, f"Expected negative mean projection in noise tail (net pull to θ0): mean_dot={mean_dot:.6e}"

    dot_ret_tail = sum(dots_ret_noise[-tail_n:]) / tail_n
    dot_grad_tail = sum(dots_grad_noise[-tail_n:]) / tail_n
    assert dot_ret_tail < 0.0, f"Expected retention component to pull toward θ0 in tail: dot_ret_tail={dot_ret_tail:.6e}"
    assert abs(dot_ret_tail) >= abs(dot_grad_tail), (
        f"Expected retention radial pull to dominate avg grad radial component in tail: |dot_ret|={abs(dot_ret_tail):.6e} |dot_grad|={abs(dot_grad_tail):.6e}"
    )

    print("PASS  Model-bound drift contract holds: grow on clean, shrink on sustained noise; dominance flips (grad→retention).")
