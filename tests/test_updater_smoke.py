import torch

from surprise_ttt.models.toy_lm import ToyLMConfig, ToyTransformerLM
from surprise_ttt.updater import AnchoredSurpriseTTT


def test_updater_runs_and_changes_params() -> None:
    # Force single-threaded to avoid CI thread issues.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    torch.manual_seed(0)
    cfg = ToyLMConfig(
        vocab_size=50,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_len=128,
        dropout=0.0,
    )
    model = ToyTransformerLM(cfg)
    updater = AnchoredSurpriseTTT(
        model=model,
        subset="all",
        ema_alpha=0.9,
        gate_kind="budget",
        eta=1e-3,
        tau=1.0,
        eps=1e-8,
        temp=1.0,
        lambda_=1e-2,
    )

    ids = torch.randint(0, 50, (1, 33), dtype=torch.long)
    loss = model.loss_next_token(ids)
    before = [p.detach().clone() for p in model.parameters()]
    updater.step_update(loss)
    after = [p.detach().clone() for p in model.parameters()]

    changed = any((a - b).abs().sum().item() > 0 for a, b in zip(after, before))
    assert changed
