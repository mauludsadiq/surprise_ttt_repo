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
        max_len = 256
    return ToyLM(Cfg)


def test_batch_tokens_controls_number_of_update_steps():
    device = torch.device("cpu")

    ids = torch.tensor([[i % 64 for i in range(1, 65)]], dtype=torch.long)

    m1 = _make_model()
    s1 = run_ttt_adv(m1, ids, AdvCfg(batch_tokens=8, update_scope="all", update_blocks="all"), device)

    m2 = _make_model()
    s2 = run_ttt_adv(m2, ids, AdvCfg(batch_tokens=32, update_scope="all", update_blocks="all"), device)

    assert s1["steps"] > s2["steps"]
