from __future__ import annotations

import torch


def configure_sdp(backend: str) -> None:
    backend = (backend or "auto").lower()

    if not torch.cuda.is_available():
        return

    if not hasattr(torch.backends, "cuda"):
        return

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(backend in ("auto", "flash"))
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(backend in ("auto", "mem_efficient"))
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(backend in ("auto", "math"))

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        enable_flash = backend in ("auto", "flash")
        enable_mem = backend in ("auto", "mem_efficient")
        enable_math = backend in ("auto", "math")
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash,
                enable_mem_efficient=enable_mem,
                enable_math=enable_math,
            )
        except TypeError:
            pass
