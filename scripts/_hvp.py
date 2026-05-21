"""Shared HVP helpers + a one-load-many-layer model loader.

The original Phase 0A `directional_curvature.py:run_for_layer` reloaded the
1.5B model in fp32 for every layer it analyzed (~35 s of HF weight load each
time). Experiment A already worked around that; this module promotes that
pattern to a reusable utility so subsequent sweeps (depth/matrix scans with
20-40 layers) only pay the load once.

Public API:
    load_model_for_hvp(model_name, device=None, dtype=torch.float32) -> nn.Module
    freeze_all_but(model, layer_name) -> nn.Parameter
    loss_fn(model, input_ids, attention_mask) -> torch.Tensor (scalar NLL)
    directional_curvature(model, target_param, direction, ids, am)
        -> (vHv, ||v||^2, vHv/||v||^2)
    matched_norm_random(shape, target_norm, generator, device, dtype) -> Tensor

All helpers are dtype-/device-aware. The model is loaded with eager attention
because the SDPA backward path does not implement double-backward (Pearlmutter
HVP) under HF Qwen2 in transformers 4.46.x — already documented in
`docs/decisions.md` (2026-04-26 entry).
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ---------- model loader ----------

def load_model_for_hvp(
    model_name: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> torch.nn.Module:
    """Load a HF causal-LM in a configuration that supports Pearlmutter HVPs.

    - fp32 by default so HVP estimates are not bf16-noisy.
    - `attn_implementation="eager"` because the SDPA / flash backward path
      raises `derivative for ..._attention_backward is not implemented` under
      double-backward.
    - `use_cache=False` + gradient checkpointing disabled so the autograd
      graph is intact for the second backward pass.
    - All parameters left with their default `requires_grad`; callers should
      use `freeze_all_but` to select the target layer before each HVP.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if verbose:
        print(f"[hvp.load] {model_name} -> {device} ({dtype})  attn=eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if verbose:
        print(f"[hvp.load] done in {time.time() - t0:.1f}s")
    return model


# ---------- parameter selection ----------

def freeze_all_but(model: torch.nn.Module, layer_name: str) -> torch.nn.Parameter:
    """Toggle requires_grad so only `layer_name` will be differentiated.

    Cheap: O(num_params) iteration over a flat named_parameters list, no I/O.
    Use between HVP calls to switch which layer is being probed without
    reloading the model.
    """
    target: Optional[torch.nn.Parameter] = None
    for n, p in model.named_parameters():
        if n == layer_name:
            p.requires_grad_(True)
            target = p
        else:
            p.requires_grad_(False)
    if target is None:
        avail = [k for k, _ in model.named_parameters() if "layers." in k][:8]
        raise KeyError(f"layer {layer_name!r} not found. Sample: {avail}")
    return target


# ---------- loss ----------

def loss_fn(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-token NLL on non-pad positions (causal LM, label-shifted)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # (B, T, V)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().to(torch.bool)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)
    losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    masked = losses * flat_mask.to(losses.dtype)
    return masked.sum() / flat_mask.sum().clamp_min(1)


# ---------- directional HVP (Pearlmutter) ----------

def directional_curvature(
    model: torch.nn.Module,
    target_param: torch.nn.Parameter,
    direction: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute (vHv, ||v||^2, vHv/||v||^2) along `direction` restricted to
    `target_param`.

    Implemented as double-backward (Pearlmutter trick): g = ∇L, then
    Hv = ∇(g·v); finally v^T H v = sum(Hv * v).
    """
    if direction.shape != target_param.shape:
        raise ValueError(
            f"direction shape {direction.shape} != param shape {target_param.shape}"
        )
    v = direction.to(target_param.device, dtype=target_param.dtype)
    model.zero_grad(set_to_none=True)
    L = loss_fn(model, input_ids, attention_mask)
    g = torch.autograd.grad(L, target_param, create_graph=True)[0]
    inner = (g * v).sum()
    Hv = torch.autograd.grad(inner, target_param, retain_graph=False)[0]
    vHv = float((Hv * v).sum().item())
    norm2 = float((v * v).sum().item())
    return vHv, norm2, vHv / max(norm2, 1e-30)


# ---------- matched-norm random vector ----------

def matched_norm_random(
    shape: torch.Size,
    target_norm: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A Gaussian of shape `shape` rescaled to have Frobenius norm `target_norm`."""
    v = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    cur = v.norm()
    if cur > 0:
        v = v * (target_norm / cur)
    return v
