from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DeviceCfg:
    device: str = "cpu"
    dtype: str = "float32"

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def torch_dtype(self) -> torch.dtype:
        m = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.dtype not in m:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return m[self.dtype]


def env_default_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    return int(v)
