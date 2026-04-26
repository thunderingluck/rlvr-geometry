"""Top-k SVD of the EARLIER checkpoint's selected layers.

Reads results/<pair>/deltas/<safe_layer>.pt (which already includes the earlier
weight) and writes results/<pair>/svd/<safe_layer>.pt with:
  - 'U_k' : (m, k) left singular vectors
  - 'S_k' : (k,)   top-k singular values
  - 'Vt_k': (k, n) right singular vectors (rows)
  - 'k'   : int
  - 'tail_energy_frac': float, sum_{i>k} sigma_i^2 / ||W||_F^2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import load_config, results_root, safe_layer_filename


@torch.no_grad()
def topk_svd(W: torch.Tensor, k: int) -> dict:
    # Full SVD on a 1536x1536 (or 1536x8960) matrix is sub-second on CPU.
    # Use full_matrices=False to get the economy form.
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k_eff = min(k, S.numel())
    total_energy = float((S ** 2).sum().item())
    tail_energy = float((S[k_eff:] ** 2).sum().item()) if k_eff < S.numel() else 0.0
    return {
        "U_k": U[:, :k_eff].contiguous(),
        "S_k": S[:k_eff].contiguous(),
        "Vt_k": Vt[:k_eff, :].contiguous(),
        "k": k_eff,
        "tail_energy_frac": tail_energy / max(total_energy, 1e-30),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out = results_root(cfg["pair_name"])
    svd_dir = out / "svd"
    deltas_dir = out / "deltas"

    k = int(cfg["svd_top_k"])
    summary: dict[str, dict] = {}
    for layer in cfg["selected_layers"]:
        rec_path = deltas_dir / f"{safe_layer_filename(layer)}.pt"
        rec = torch.load(rec_path, map_location="cpu", weights_only=True)
        W = rec["earlier"]
        result = topk_svd(W.to(torch.float32), k)
        out_path = svd_dir / f"{safe_layer_filename(layer)}.pt"
        torch.save(result, out_path)
        s_top = float(result["S_k"][0].item())
        s_kth = float(result["S_k"][-1].item())
        summary[layer] = {
            "k": result["k"],
            "sigma_1": s_top,
            "sigma_k": s_kth,
            "tail_energy_frac": result["tail_energy_frac"],
            "file": str(out_path.relative_to(out)),
        }
        print(
            f"[svd] {layer}  k={result['k']}  "
            f"sigma_1={s_top:.4e}  sigma_k={s_kth:.4e}  "
            f"tail_energy={result['tail_energy_frac']:.4e}"
        )

    with open(out / "svd_manifest.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
