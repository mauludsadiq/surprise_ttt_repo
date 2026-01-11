from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .config import TTTcfg
from .train import PretrainCfg, pretrain_toy_lm
from .ttt import run_ttt
from .sdp import configure_sdp

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def pretrain(
    corpus: str = typer.Option(..., help="Path to text corpus"),
    out: str = typer.Option("runs/pretrained.pt", help="Output checkpoint path"),
    steps: int = typer.Option(500, help="Optimization steps"),
    batch_tokens: int = typer.Option(128, help="Tokens per step"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    device: str = typer.Option("cpu", help="cpu|cuda"),
) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    configure_sdp(sdp)
    cfg = PretrainCfg(steps=steps, batch_tokens=batch_tokens, lr=lr)
    pretrain_toy_lm(corpus_path=corpus, out_path=out, cfg=cfg, device=device)
    rprint(f"[green]Wrote[/green] {out}")


@app.command()
def ttt(
    corpus: str = typer.Option(..., help="Path to text corpus"),
    ckpt: str = typer.Option(..., help="Pretrained checkpoint path"),
    out: str = typer.Option("runs/ttt.pt", help="Output checkpoint path"),
    cfg_json: Optional[str] = typer.Option(None, help="Optional JSON config override"),
    max_steps: int = typer.Option(200, help="TTT steps"),
    batch_tokens: int = typer.Option(64, help="Tokens per TTT step"),
    gate_eta: float = typer.Option(5e-4, help="Base lr eta inside gate"),
    gate_tau: float = typer.Option(1.0, help="Surprise budget tau"),
    ema_alpha: float = typer.Option(0.95, help="EMA alpha"),
    lam: float = typer.Option(1e-3, help="Retention lambda"),
    lam_schedule: str = typer.Option("constant", help="constant|inv_surprise|exp_decay|miras"),
    subset: str = typer.Option("mlp", help="all|mlp|ln|head"),
    update_blocks: str = typer.Option("all", help="all|last|last_quarter|range:a-b|list:i,j"),
    amp: bool = typer.Option(False, help="Enable mixed precision on CUDA"),
    amp_dtype: str = typer.Option("bf16", help="bf16|fp16"),
    sdp: str = typer.Option("auto", help="auto|flash|mem_efficient|math"),
    device: str = typer.Option("cpu", help="cpu|cuda"),
) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cfg = TTTcfg(max_steps=max_steps, batch_tokens=batch_tokens)
    cfg.gate.eta = gate_eta
    cfg.gate.tau = gate_tau
    cfg.ema.alpha = ema_alpha
    cfg.retention.lambda_ = lam
    cfg.retention.schedule = lam_schedule
    cfg.update_subset = subset
    cfg.update_blocks = update_blocks
    cfg.amp = amp
    cfg.amp_dtype = amp_dtype
    cfg.sdp = sdp

    if cfg_json is not None:
        overrides = json.loads(Path(cfg_json).read_text(encoding="utf-8"))
        cfg = TTTcfg.model_validate({**cfg.model_dump(), **overrides})

    run_ttt(corpus_path=corpus, ckpt_path=ckpt, out_path=out, cfg=cfg, device=device)
    rprint(f"[green]Wrote[/green] {out}")
