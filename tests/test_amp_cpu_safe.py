import torch

from surprise_ttt.ttt_adv import AdvCfg, run_ttt_adv
from surprise_ttt.models.toy_lm import ToyLM


def _make_model():
    class Cfg:
        vocab_size = 64
        d_model = 64
        nhead = 4
        num_layers = 2
        dim_feedforward = 128
        dropout = 0.0
        max_len = 64
    return ToyLM(Cfg)


def test_amp_flags_do_not_crash_on_cpu():
    device = torch.device("cpu")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    model = _make_model()
    cfg = AdvCfg(batch_tokens=4, amp=True, amp_dtype="bf16", update_scope="all", update_blocks="all")
    stats = run_ttt_adv(model, ids, cfg, device)
    assert stats["steps"] >= 1
