from __future__ import annotations

from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "batch_tokens": 1,
    "amp": False,
    "amp_dtype": "bf16",
    "update_scope": "all",
    "update_blocks": "all",
    "sdp": "auto",
}


def apply_runtime_defaults(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        return cfg

    for k, v in DEFAULTS.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
