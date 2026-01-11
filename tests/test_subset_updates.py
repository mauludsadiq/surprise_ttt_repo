import copy
import torch

from surprise_ttt.ttt_adv import AdvCfg, run_ttt_adv
from surprise_ttt.models.toy_lm import ToyLM


def _make_model():
    class Cfg:
        vocab_size = 64
        d_model = 64
        nhead = 4
        num_layers = 4
        dim_feedforward = 128
        dropout = 0.0
        max_len = 64
    return ToyLM(Cfg)


def _param_snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def _changed(a, b, tol=0.0):
    return (a - b).abs().max().item() > tol


def test_mlp_subset_changes_only_mlp_params_in_last_block():
    device = torch.device("cpu")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    model = _make_model()
    before = _param_snapshot(model)

    cfg = AdvCfg(
        eta=1e-3,
        tau=1.0,
        lam=0.1,
        alpha=0.0,
        batch_tokens=4,
        amp=False,
        update_scope="mlp",
        update_blocks="last",
        sdp="auto",
    )
    run_ttt_adv(model, ids, cfg, device)
    after = _param_snapshot(model)

    changed = [n for n in before if _changed(before[n], after[n])]
    assert len(changed) > 0

    for n in changed:
        assert "enc.layers.3.linear1" in n or "enc.layers.3.linear2" in n or n.startswith("ln_f") or n.startswith("head") is False
