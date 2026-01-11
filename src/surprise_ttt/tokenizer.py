from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import re


@dataclass
class SimpleTokenizer:
    """A tiny deterministic tokenizer.

    - splits on whitespace and punctuation
    - builds vocab from corpus

    This is intentionally simple to keep the repo self-contained.
    """

    stoi: Dict[str, int]
    itos: List[str]

    @staticmethod
    def build(text: str, min_freq: int = 1) -> "SimpleTokenizer":
        tokens = tokenize(text)
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and tok not in vocab:
                vocab.append(tok)
        stoi = {t: i for i, t in enumerate(vocab)}
        return SimpleTokenizer(stoi=stoi, itos=vocab)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        toks = tokenize(text)
        ids = []
        if add_bos:
            ids.append(self.stoi["<bos>"])
        for t in toks:
            ids.append(self.stoi.get(t, self.stoi["<unk>"]))
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                out.append("<unk>")
            else:
                out.append(self.itos[i])
        return " ".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,;:!?()\[\]{}\-]")

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)
