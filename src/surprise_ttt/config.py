from __future__ import annotations

from pydantic import BaseModel, Field


class GateCfg(BaseModel):
    kind: str = Field(default="budget", description="budget|sigmoid")
    eta: float = 5e-4
    tau: float = 1.0
    eps: float = 1e-8
    temp: float = 1.0


class RetentionCfg(BaseModel):
    lambda_: float = Field(default=1e-3, alias="lambda")
    schedule: str = Field(default="constant", description="constant|inv_surprise|exp_decay")
    inv_c: float = 1.0
    decay_beta: float = 0.0


class EMAcfg(BaseModel):
    alpha: float = 0.95


class TTTcfg(BaseModel):
    seed: int = 7
    batch_tokens: int = 64
    max_steps: int = 200
    lr: float = 5e-4
    gate: GateCfg = GateCfg()
    retention: RetentionCfg = RetentionCfg()
    ema: EMAcfg = EMAcfg()
    update_subset: str = Field(
        default="mlp",
        description="which parameters to update: all|mlp|ln|head",
    )
