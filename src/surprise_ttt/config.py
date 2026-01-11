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
    schedule: str = Field(default="constant", description="constant|inv_surprise|exp_decay|miras")
    inv_c: float = 1.0
    decay_beta: float = 0.0
    s_ref: float = 1.0
    spike_k: float = 4.0
    lam_max: float = 0.5


class EMAcfg(BaseModel):
    alpha: float = 0.95


class TTTcfg(BaseModel):
    seed: int = 7
    batch_tokens: int = 64
    max_steps: int = 200
    lr: float = 5e-4

    gate: GateCfg = Field(default_factory=GateCfg)
    retention: RetentionCfg = Field(default_factory=RetentionCfg)
    ema: EMAcfg = Field(default_factory=EMAcfg)

    update_subset: str = Field(
        default="mlp",
        description="which parameters to update: all|mlp|ln|head",
    )

    update_blocks: str = Field(
        default="all",
        description="which blocks to update: all|last|last_quarter|range:a-b|list:i,j",
    )

    amp: bool = Field(default=False, description="enable mixed precision (CUDA only)")
    amp_dtype: str = Field(default="bf16", description="bf16|fp16")
    sdp: str = Field(default="auto", description="auto|flash|mem_efficient|math")
