from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch

from .tokenizer import SimpleTokenizer


@dataclass(frozen=True)
class TokenBatch:
    ids: torch.Tensor  # [B,T]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_stream_ids(tok: SimpleTokenizer, text: str) -> List[int]:
    return tok.encode(text, add_bos=True, add_eos=True)


def iter_token_batches(
    stream: List[int], batch_tokens: int, batch_size: int = 1
) -> Iterator[TokenBatch]:
    # For simplicity, batch_size default 1 (one long stream sliced into chunks).
    # You can extend this to multiple documents later.
    assert batch_size == 1, "Toy iterator currently supports batch_size=1"
    i = 0
    n = len(stream)
    while i + batch_tokens + 1 <= n:
        chunk = stream[i : i + batch_tokens + 1]  # +1 for next-token targets
        ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
        yield TokenBatch(ids=ids)
        i += batch_tokens
