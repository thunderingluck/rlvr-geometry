"""Experiment A: k-sweep + coordinate-mask comparison.

For each layer L and each k in cfg.experiment_a.k_sweep, build two operator pairs:

  (A) Subspace projection  (Wedin sin-Theta / CLAUDE.md "principal directions"):
        P_sub^(k)(dW) = U_k U_k^T dW V_k V_k^T
        P_sub_comp    = dW - P_sub^(k)(dW)

  (B) Coordinate mask  (the source paper's "principal mask" operator,
        Liu et al. 2025c / Fig. 5; rule for principal weights in Sec 4.2):
        W_k = U_k diag(S_k) V_k^T              (rank-k reconstruction of W_0)
        M_princ = TopAlpha(|W_k(i,j)|)         (boolean mask, alpha fraction)
        P_mask^(k,a)(dW) = M_princ * dW        (entrywise selection)
        P_mask_comp      = (1 - M_princ) * dW

For each layer we run F2 (directional curvature, v^T H v / ||v||^2 via Pearlmutter
HVP) on the existing fixed minibatch, the same loss (token NLL), the same target
parameter, and the same eager-attention model as Phase 0A. The direction set per
layer is:

    realized                                   (= dW, k-independent)
    random_seed_{s}     for s in seeds         (matched-Frobenius, k-independent)
    sub_princ_k{k}      for k in k_sweep
    sub_comp_k{k}       for k in k_sweep
    mask_princ_k{k}_a{a}    for k in k_sweep, alpha = cfg.coord_mask_alpha
    mask_comp_k{k}_a{a}     for k in k_sweep
    mask_princ_k{k}_a{a}    for k in k_sweep, every a in cfg.extra_alphas
    mask_comp_k{k}_a{a}     for k in k_sweep, every a in cfg.extra_alphas

Outputs: results/<pair>/experiment_a/<safe_layer>.json with per-direction stats
and Frobenius norms, plus results/<pair>/experiment_a/full_svd_<safe_layer>.pt
caching the full SVD of W_0 so subsequent runs are cheap.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import torch

from _common import device, load_config, results_root, safe_layer_filename
from _hvp import (
    directional_curvature,
    freeze_all_but,
    load_model_for_hvp,
    matched_norm_random,
)


# ---------- operator implementations ----------

def subspace_projection(
    dW: torch.Tensor, U_k: torch.Tensor, Vt_k: torch.Tensor
) -> torch.Tensor:
    """P_U dW P_V where P_U = U_k U_k^T, P_V = V_k V_k^T."""
    return U_k @ (U_k.T @ dW @ Vt_k.T) @ Vt_k


def coord_mask_from_rank_k(
    U_k: torch.Tensor, S_k: torch.Tensor, Vt_k: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Top-alpha boolean mask of |W_k| where W_k = U_k diag(S_k) V_k^T.

    Matches the paper's M_princ = Top_alpha(|W_0^(k)(i,j)|) definition. Returns
    a {0,1} float tensor with the same shape as W_k.
    """
    W_k = (U_k * S_k.unsqueeze(0)) @ Vt_k
    flat = W_k.abs().flatten()
    n = flat.numel()
    keep = max(1, int(round(alpha * n)))
    # Top-`keep` indices by magnitude.
    threshold = torch.topk(flat, keep, largest=True, sorted=False).values.min()
    mask = (W_k.abs() >= threshold).to(W_k.dtype)
    # Edge case: ties at threshold could give slightly more than `keep` entries.
    # That is acceptable; we record the realized count for diagnostics.
    return mask


# ---------- per-layer runner ----------

def compute_full_svd(W: torch.Tensor, cache_path: Path) -> dict:
    """Compute or load the cached full economy SVD of W (CPU, float32)."""
    if cache_path.exists():
        rec = torch.load(cache_path, map_location="cpu", weights_only=True)
        return rec
    print(f"[svd-full] computing on shape {tuple(W.shape)}...")
    t0 = time.time()
    U, S, Vt = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    print(f"[svd-full] done in {time.time() - t0:.1f}s, rank = {S.numel()}")
    rec = {"U": U.contiguous(), "S": S.contiguous(), "Vt": Vt.contiguous()}
    torch.save(rec, cache_path)
    return rec


def build_directions(
    dW: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    Vt: torch.Tensor,
    k_sweep: Iterable[int],
    alpha_list: list[float],
    n_random: int,
    random_seed_base: int = 1000,
) -> list[tuple[str, torch.Tensor, dict]]:
    """Returns list of (name, tensor, meta) tuples.

    meta carries `kind`, `k`, `alpha`, `frob_v`, `mask_density` (for coord_mask).
    """
    out: list[tuple[str, torch.Tensor, dict]] = []
    frob_dW = float(dW.norm().item())

    # 1) realized (k-independent)
    out.append(("realized", dW.clone(), {"kind": "realized", "frob_v": frob_dW}))

    # 2) random matched-Frobenius (k-independent)
    for s in range(n_random):
        gen = torch.Generator(device="cpu").manual_seed(random_seed_base + s)
        rdir = matched_norm_random(
            dW.shape, frob_dW, gen, device=torch.device("cpu"), dtype=torch.float32
        )
        out.append(
            (f"random_seed_{s}", rdir, {"kind": "random", "seed": random_seed_base + s,
                                         "frob_v": float(rdir.norm().item())})
        )

    rank = S.numel()
    # 3) subspace projection: principal + complement at each k
    for k in k_sweep:
        k_eff = min(int(k), rank)
        U_k = U[:, :k_eff]
        S_k = S[:k_eff]
        Vt_k = Vt[:k_eff, :]
        v_p = subspace_projection(dW, U_k, Vt_k)
        v_c = dW - v_p
        out.append(
            (f"sub_princ_k{k_eff}", v_p,
             {"kind": "subspace_principal", "k": k_eff, "frob_v": float(v_p.norm().item())})
        )
        out.append(
            (f"sub_comp_k{k_eff}", v_c,
             {"kind": "subspace_complement", "k": k_eff, "frob_v": float(v_c.norm().item())})
        )

    # 4) coordinate mask: principal + complement at each (k, alpha)
    for k in k_sweep:
        k_eff = min(int(k), rank)
        U_k = U[:, :k_eff]
        S_k = S[:k_eff]
        Vt_k = Vt[:k_eff, :]
        for alpha in alpha_list:
            mask = coord_mask_from_rank_k(U_k, S_k, Vt_k, alpha)
            v_p = mask * dW
            v_c = (1.0 - mask) * dW
            density = float(mask.mean().item())
            a_tag = f"{alpha:.2f}".replace("0.", "0p")
            out.append(
                (f"mask_princ_k{k_eff}_a{a_tag}", v_p,
                 {"kind": "mask_principal", "k": k_eff, "alpha": alpha,
                  "mask_density": density,
                  "frob_v": float(v_p.norm().item())})
            )
            out.append(
                (f"mask_comp_k{k_eff}_a{a_tag}", v_c,
                 {"kind": "mask_complement", "k": k_eff, "alpha": alpha,
                  "mask_density": density,
                  "frob_v": float(v_c.norm().item())})
            )

    return out


def run_for_layer(
    cfg: dict,
    layer_name: str,
    model: torch.nn.Module,
    minibatches: list[dict],
    dev: torch.device,
    out_dir: Path,
) -> dict:
    """Run all directions for one layer using a single shared model."""
    pair_out = results_root(cfg["pair_name"])
    delta_rec = torch.load(
        pair_out / "deltas" / f"{safe_layer_filename(layer_name)}.pt",
        map_location="cpu", weights_only=True,
    )
    W = delta_rec["earlier"].to(torch.float32)
    dW = delta_rec["delta"].to(torch.float32)

    svd_cache = out_dir / f"full_svd_{safe_layer_filename(layer_name)}.pt"
    full = compute_full_svd(W, svd_cache)
    U, S, Vt = full["U"], full["S"], full["Vt"]

    exp_cfg = cfg["experiment_a"]
    directions = build_directions(
        dW=dW,
        U=U, S=S, Vt=Vt,
        k_sweep=exp_cfg["k_sweep"],
        alpha_list=[exp_cfg["coord_mask_alpha"]] + list(exp_cfg.get("extra_alphas", [])),
        n_random=int(exp_cfg["num_random_seeds"]),
    )
    print(f"[layer] {layer_name}: {len(directions)} directions, shape={tuple(W.shape)}")

    target = freeze_all_but(model, layer_name)

    per_dir: dict[str, dict] = {}
    t_layer = time.time()
    for di, (name, v, meta) in enumerate(directions):
        if meta["frob_v"] < 1e-30:
            # Degenerate (e.g., complement of full-rank projection when k >= min(m,n)).
            per_dir[name] = {
                "meta": meta,
                "skipped_reason": "frob_v ~ 0 (degenerate direction)",
            }
            print(f"  [{di+1}/{len(directions)}] {name}  SKIP (frob_v={meta['frob_v']:.2e})")
            continue

        per_mb_curv: list[float] = []
        per_mb_vHv: list[float] = []
        t0 = time.time()
        for mi, mb in enumerate(minibatches):
            ids = mb["input_ids"].to(dev)
            am = mb["attention_mask"].to(dev)
            vHv, n2, c = directional_curvature(model, target, v, ids, am)
            per_mb_curv.append(c)
            per_mb_vHv.append(vHv)
        t = torch.tensor(per_mb_curv)
        per_dir[name] = {
            "meta": meta,
            "norm_squared": float((v * v).sum().item()),
            "frob_norm": float(v.norm().item()),
            "per_minibatch_curvature": per_mb_curv,
            "per_minibatch_vHv": per_mb_vHv,
            "mean_curvature": float(t.mean().item()),
            "std_curvature": float(t.std(unbiased=False).item() if len(per_mb_curv) > 1 else 0.0),
            "min_curvature": float(t.min().item()),
            "max_curvature": float(t.max().item()),
        }
        dt = time.time() - t0
        print(
            f"  [{di+1}/{len(directions)}] {name}  "
            f"mean_curv={per_dir[name]['mean_curvature']:+.3e}  "
            f"std={per_dir[name]['std_curvature']:.2e}  "
            f"frob={meta['frob_v']:.3e}  ({dt:.1f}s)"
        )
    print(f"[layer] {layer_name} done in {time.time() - t_layer:.1f}s")

    # Aggregate random.
    rand_means = [v["mean_curvature"] for k, v in per_dir.items()
                  if k.startswith("random_seed_") and "skipped_reason" not in v]
    if rand_means:
        rt = torch.tensor(rand_means)
        per_dir["random_aggregate"] = {
            "n_seeds": len(rand_means),
            "mean_of_seed_means": float(rt.mean().item()),
            "std_of_seed_means": float(rt.std(unbiased=False).item() if len(rand_means) > 1 else 0.0),
            "seed_means": rand_means,
        }

    return {
        "layer_name": layer_name,
        "earlier_model": cfg["earlier_model"],
        "later_model": cfg["later_model"],
        "shape": list(W.shape),
        "k_sweep": list(exp_cfg["k_sweep"]),
        "coord_mask_alpha": exp_cfg["coord_mask_alpha"],
        "extra_alphas": list(exp_cfg.get("extra_alphas", [])),
        "frob_dW": float(dW.norm().item()),
        "frob_W": float(W.norm().item()),
        "directions": per_dir,
        "objective": exp_cfg["loss"],
        "num_minibatches": len(minibatches),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", default=None,
                    help="single layer name (default: every selected_layer)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    pair_out = results_root(cfg["pair_name"])
    out_subdir = cfg["experiment_a"].get("out_subdir", "experiment_a")
    out_dir = pair_out / out_subdir
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    if args.layer:
        layers = [args.layer]
    else:
        layers = list(cfg["selected_layers"])

    # Load model ONCE (reuse across layers). HVP requires eager attn (Phase 0A note).
    dev = device()
    model = load_model_for_hvp(cfg["earlier_model"], device=dev, dtype=torch.float32)

    mb_payload = torch.load(pair_out / cfg.get("minibatch_file", "minibatch.pt"),
                             map_location="cpu", weights_only=False)
    minibatches = mb_payload["minibatches"]
    print(f"[init] {len(minibatches)} minibatches  bs={mb_payload['batch_size']}  "
          f"seq_len={mb_payload['seq_len']}")

    for layer in layers:
        print(f"\n=== experiment_a :: {layer} ===")
        summary = run_for_layer(cfg, layer, model, minibatches, dev, out_dir)
        path = out_dir / f"{safe_layer_filename(layer)}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[write] {path}")


if __name__ == "__main__":
    main()
