"""Load both public checkpoints, compute layer deltas, and save them.

Output: results/<pair>/deltas/<safe_layer>.pt with keys:
  - 'delta'       : torch.float32 tensor, shape = layer's weight shape
  - 'earlier'     : torch.float32 tensor (kept for SVD step + curvature probe)
  - 'shape'       : list[int]
  - 'layer_name'  : str
  - 'frob_delta'  : float, Frobenius norm of the delta
  - 'frob_earlier': float, Frobenius norm of the earlier weight
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from _common import load_config, results_root, safe_layer_filename


@torch.no_grad()
def get_param(model: torch.nn.Module, name: str) -> torch.Tensor:
    sd = dict(model.named_parameters())
    if name not in sd:
        avail = [k for k in sd.keys() if "layers.13" in k][:10]
        raise KeyError(f"layer {name!r} not found. Sample available: {avail}")
    return sd[name].detach().to(torch.float32).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out = results_root(cfg["pair_name"])
    deltas_dir = out / "deltas"

    print(f"[load] earlier = {cfg['earlier_model']}")
    earlier = AutoModelForCausalLM.from_pretrained(
        cfg["earlier_model"], torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    earlier.eval()

    print(f"[load] later   = {cfg['later_model']}")
    later = AutoModelForCausalLM.from_pretrained(
        cfg["later_model"], torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    later.eval()

    manifest: dict[str, dict] = {}
    for layer in cfg["selected_layers"]:
        We = get_param(earlier, layer)
        Wl = get_param(later, layer)
        if We.shape != Wl.shape:
            raise RuntimeError(f"shape mismatch for {layer}: {We.shape} vs {Wl.shape}")
        delta = Wl - We
        frob_d = float(delta.norm().item())
        frob_e = float(We.norm().item())
        rec = {
            "delta": delta,
            "earlier": We,
            "shape": list(We.shape),
            "layer_name": layer,
            "frob_delta": frob_d,
            "frob_earlier": frob_e,
        }
        fname = deltas_dir / f"{safe_layer_filename(layer)}.pt"
        torch.save(rec, fname)
        manifest[layer] = {
            "shape": list(We.shape),
            "frob_delta": frob_d,
            "frob_earlier": frob_e,
            "rel_frob_delta": frob_d / max(frob_e, 1e-12),
            "file": str(fname.relative_to(out)),
        }
        print(
            f"[delta] {layer}  shape={tuple(We.shape)}  "
            f"||dW||_F={frob_d:.4e}  ||W||_F={frob_e:.4e}  "
            f"rel={frob_d/max(frob_e,1e-12):.4e}"
        )

    with open(out / "deltas_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {len(manifest)} deltas under {deltas_dir}")


if __name__ == "__main__":
    main()
