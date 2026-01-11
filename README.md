# Surprise-Gated Test-Time Training (TTT) + Anchored Forgetting (MIRAS-style)

This repository implements **surprise-gated test-time training** with an **anchored forgetting term**
(aka retention / pullback to pretrained weights), plus experiments and unit tests.

Core update (per step `t`, token or micro-batch):

\[
\theta_{t+1}
= \theta_t - \gamma_t\Big(\nabla \ell_t(\theta_t) + \lambda_t (\theta_t - \theta_0)\Big)
\]

- `θ0`: anchor parameters (frozen pretrained copy)
- `ℓt`: inner self-supervised loss (default: next-token cross-entropy)
- `λt`: retention (pullback) coefficient
- `γt`: surprise-gated effective step size

Surprise is measured by gradient norm (or squared norm), smoothed by EMA:

\[
s_t = \|\nabla \ell_t(\theta_t)\|_2,\qquad
\tilde s_{t+1} = \alpha \tilde s_t + (1-\alpha) s_t
\]

A default gate implements a step/KL budget:

\[
\gamma_t = \eta \cdot \mathrm{clip}\left(\frac{\tau}{\tilde s_t + \varepsilon}, 0, 1\right)
\]

## What’s included (not minimal)

- `src/surprise_ttt/`
  - gating functions (budget, sigmoid, piecewise)
  - EMA trackers (for `s` and `s^2`)
  - Anchored updater that applies the coupled update to a **selected parameter subset**
  - Toy language model (Transformer encoder LM) + simple tokenizer
  - Pretrain + TTT runners
  - Stability micro-simulator on quadratic objectives
- `tests/`
  - gate + EMA unit tests
  - quadratic stability test for contraction
  - anchored update sanity tests
- `docs/`
  - math notes for the coupled system + stability bounds
- `cli/` entry points via `typer` (`sttt`)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# 1) pretrain a toy LM on a tiny corpus
sttt pretrain --corpus data/sample_corpus.txt --out runs/pretrained.pt

# 2) run surprise-gated TTT with anchored forgetting
sttt ttt --corpus data/sample_corpus.txt --ckpt runs/pretrained.pt --out runs/ttt.pt

# 3) run tests
pytest -q
```

## Notes

- This is a *research/engineering scaffold*: the LM is toy by design.
- For large models, you would:
  - restrict updates to a subset (e.g., selected MLPs/blocks)
  - use micro-batches (`--batch-tokens`) and mixed precision
  - use fused attention kernels
