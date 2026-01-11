from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class UpdateSubset:
    scope: str = "all"
    blocks: str = "all"


def parse_blocks(spec: str, n_layers: int) -> List[int]:
    s = (spec or "all").strip().lower()
    if s in ("all", "*"):
        return list(range(n_layers))
    if s in ("last", "last1"):
        return [max(0, n_layers - 1)]
    if s in ("last_quarter", "last-quarter", "lq"):
        k = max(1, n_layers // 4)
        return list(range(n_layers - k, n_layers))
    m = re.match(r"^range:(\d+)-(\d+)$", s)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        a = max(0, min(a, n_layers - 1))
        b = max(0, min(b, n_layers - 1))
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))
    m = re.match(r"^list:([\d,]+)$", s)
    if m:
        xs = [int(x) for x in m.group(1).split(",") if x.strip() != ""]
        ys = sorted({x for x in xs if 0 <= x < n_layers})
        return ys
    raise ValueError(f"Unrecognized blocks spec: {spec!r}")


def infer_num_layers(model: nn.Module) -> int:
    if hasattr(model, "enc") and hasattr(model.enc, "layers"):
        try:
            return len(model.enc.layers)
        except Exception:
            pass
    return 0


def mark_trainable_subset(model: nn.Module, subset: UpdateSubset) -> List[torch.nn.Parameter]:
    n_layers = infer_num_layers(model)
    layer_ids: List[int] = []
    if n_layers > 0:
        layer_ids = parse_blocks(subset.blocks, n_layers)

    for p in model.parameters():
        p.requires_grad_(False)

    trainable: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        ok = False

        if subset.scope == "all":
            if n_layers == 0:
                ok = True
            else:
                for i in layer_ids:
                    if name.startswith(f"enc.layers.{i}."):
                        ok = True
                        break
                if not ok and name.startswith("ln_f"):
                    ok = True
                if not ok and name.startswith("head"):
                    ok = True

        elif subset.scope in ("mlp", "ffn"):
            if n_layers == 0:
                ok = "linear" in name
            else:
                for i in layer_ids:
                    prefix = f"enc.layers.{i}."
                    if name.startswith(prefix + "linear1.") or name.startswith(prefix + "linear2."):
                        ok = True
                        break


        elif subset.scope in ("ln", "layernorm"):
            if n_layers == 0:
                ok = ("ln" in name.lower()) or ("norm" in name.lower())
            else:
                for i in layer_ids:
                    prefix = f"enc.layers.{i}."
                    if name.startswith(prefix) and ("norm" in name.lower() or ".norm" in name.lower()):
                        ok = True
                        break
                if not ok and name.startswith("ln_f"):
                    ok = True

        elif subset.scope in ("head",):
            ok = name.startswith("head.")

        else:
            raise ValueError(f"Unknown update scope: {subset.scope!r}")

        if ok:
            p.requires_grad_(True)
            trainable.append(p)

    if not trainable:
        raise RuntimeError("No trainable parameters selected; check scope/blocks settings")

    return trainable
