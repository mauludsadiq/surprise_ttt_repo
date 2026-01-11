from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from surprise_ttt.models import toy_lm as toy_lm_mod


def _resolve_toy_lm_class():
    import torch.nn as nn
    for name in ("ToyLM","ToyLMModel","ToyTransformerLM","ToyLanguageModel"):
        if hasattr(toy_lm_mod, name):
            return getattr(toy_lm_mod, name)
    for v in toy_lm_mod.__dict__.values():
        try:
            if isinstance(v, type) and issubclass(v, nn.Module):
                return v
        except Exception:
            pass
    raise ImportError("No nn.Module subclass found in surprise_ttt.models.toy_lm")

ToyLM = _resolve_toy_lm_class()
from surprise_ttt.ttt_adv import AdvCfg, run_ttt_adv


class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for i, c in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str):
        vocab = sorted(set(text))
        if "\n" not in vocab:
            vocab.append("\n")
        return cls(vocab)

    def encode(self, s: str):
        return [self.stoi.get(c, 0) for c in s]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surprise-ttt-adv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--corpus", type=str, default="data/sample_corpus.txt")
    p.add_argument("--prompt", type=str, default="Hello world\n")

    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument("--batch-tokens", type=int, default=32)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--update-scope", type=str, default="mlp", choices=["all", "mlp"])
    p.add_argument("--update-blocks", type=str, default="last_quarter")
    p.add_argument("--sdp", type=str, default="auto", choices=["auto", "flash", "mem_efficient", "math"])

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    return p


def main() -> None:
    args = build_parser().parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text + args.prompt)
    vocab_size = len(tok.vocab)

    class Cfg:
        vocab_size = vocab_size
        d_model = args.d_model
        nhead = args.nhead
        num_layers = args.num_layers
        dim_feedforward = args.ff
        dropout = args.dropout
        max_len = args.max_len

    model = ToyLM(Cfg)

    device = torch.device(args.device)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)

    cfg = AdvCfg(
        eta=args.eta,
        tau=args.tau,
        lam=args.lam,
        alpha=args.alpha,
        eps=args.eps,
        batch_tokens=args.batch_tokens,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        update_scope=args.update_scope,
        update_blocks=args.update_blocks,
        sdp=args.sdp,
    )

    stats = run_ttt_adv(model, ids, cfg, device)

    print("ADV_CFG", asdict(cfg))
    print("STATS", stats)


if __name__ == "__main__":
    main()
