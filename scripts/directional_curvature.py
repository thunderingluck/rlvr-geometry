"""Directional curvature v^T H v / ||v||^2 along chosen layer-restricted dirs.

For each minibatch and each direction in {realized, principal, nonprincipal,
random_seed_*}:

    L(theta)         = mean token NLL on the minibatch (CrossEntropyLoss with
                       attention-mask + label-shift; pad tokens ignored)
    g(theta)         = autograd.grad(L, W_layer, create_graph=True)
    Hv               = autograd.grad((g * v).sum(), W_layer)[0]
    curvature        = (Hv * v).sum() / (v * v).sum()

We restrict v to a single layer (zero everywhere else) and run autograd only
through that layer's weight by toggling requires_grad. This makes one Hv
roughly the cost of a single backward pass.

The shared HVP helpers (model loader, freeze_all_but, loss_fn,
directional_curvature, matched_norm_random) live in `_hvp.py`; this script
imports them and just orchestrates direction construction per layer.

Public re-exports (back-compat) — older code imports these from here:
    directional_curvature, freeze_all_but, matched_norm_random, loss_fn

Outputs results/<pair>/curvature/<safe_layer>.json with per-direction stats
and per-minibatch raw values for stability assessment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import device, load_config, results_root, safe_layer_filename
from _hvp import (  # noqa: F401  (re-export for back-compat)
    directional_curvature,
    freeze_all_but,
    load_model_for_hvp,
    loss_fn,
    matched_norm_random,
)


def project_principal(dW: torch.Tensor, U_k: torch.Tensor, Vt_k: torch.Tensor) -> torch.Tensor:
    # principal projection P_U dW P_V where P_U = U_k U_k^T, P_V = V_k V_k^T.
    # dW: (m, n);  U_k: (m, k);  Vt_k: (k, n)
    return U_k @ (U_k.T @ dW @ Vt_k.T) @ Vt_k


def run_for_layer(
    cfg: dict,
    layer_name: str,
    model: torch.nn.Module,
    minibatches: list[dict],
    dev: torch.device,
) -> dict:
    """Run all directions for one layer using a shared, already-loaded model.

    `model` is expected to be the result of `_hvp.load_model_for_hvp(...)`. The
    caller owns the lifetime (don't `del model` here).
    """
    out = results_root(cfg["pair_name"])

    target = freeze_all_but(model, layer_name)
    print(f"[curv] target = {layer_name}  shape={tuple(target.shape)}  device={target.device}")

    # Load delta + svd.
    delta_rec = torch.load(out / "deltas" / f"{safe_layer_filename(layer_name)}.pt",
                            map_location="cpu", weights_only=True)
    svd_rec = torch.load(out / "svd" / f"{safe_layer_filename(layer_name)}.pt",
                          map_location="cpu", weights_only=True)

    dW = delta_rec["delta"].to(torch.float32)
    U_k = svd_rec["U_k"].to(torch.float32)
    Vt_k = svd_rec["Vt_k"].to(torch.float32)

    dW_principal = project_principal(dW, U_k, Vt_k)
    dW_nonprincipal = dW - dW_principal

    frob_dW = float(dW.norm().item())
    frob_p = float(dW_principal.norm().item())
    frob_np = float(dW_nonprincipal.norm().item())
    print(f"[curv] ||dW||_F={frob_dW:.4e}  ||P(dW)||_F={frob_p:.4e}  "
          f"||(I-P)dW||_F={frob_np:.4e}  rank-k k={U_k.shape[1]}")

    # Build random matched-norm directions, one per seed.
    n_random = int(cfg["curvature"]["num_random_seeds"])
    random_dirs: list[tuple[str, torch.Tensor]] = []
    for s in range(n_random):
        gen = torch.Generator(device="cpu").manual_seed(1000 + s)
        rdir = matched_norm_random(dW.shape, frob_dW, gen,
                                   device=torch.device("cpu"),
                                   dtype=torch.float32)
        random_dirs.append((f"random_seed_{s}", rdir))

    direction_set = [
        ("realized", dW),
        ("principal", dW_principal),
        ("nonprincipal", dW_nonprincipal),
    ] + random_dirs

    print(f"[curv] {len(minibatches)} minibatches  bs={minibatches[0]['input_ids'].shape[0]}  "
          f"seq_len={minibatches[0]['input_ids'].shape[1]}")

    per_dir: dict[str, dict] = {}
    for dname, d in direction_set:
        norms = float((d * d).sum().item())
        vals: list[float] = []
        vHv_list: list[float] = []
        for mi, mb in enumerate(minibatches):
            ids = mb["input_ids"].to(dev)
            am = mb["attention_mask"].to(dev)
            vHv, n2, c = directional_curvature(model, target, d, ids, am)
            vals.append(c)
            vHv_list.append(vHv)
            print(f"  [{dname}] mb={mi}  vHv={vHv:+.4e}  ||v||^2={n2:.4e}  curv={c:+.4e}")
        t = torch.tensor(vals)
        per_dir[dname] = {
            "norm_squared": norms,
            "frob_norm": float(norms ** 0.5),
            "per_minibatch_curvature": vals,
            "per_minibatch_vHv": vHv_list,
            "mean_curvature": float(t.mean().item()),
            "std_curvature": float(t.std(unbiased=False).item() if len(vals) > 1 else 0.0),
            "min_curvature": float(t.min().item()),
            "max_curvature": float(t.max().item()),
        }

    # Aggregate random across seeds.
    rand_means = [v["mean_curvature"] for k, v in per_dir.items() if k.startswith("random_seed_")]
    if rand_means:
        rt = torch.tensor(rand_means)
        per_dir["random_aggregate"] = {
            "n_seeds": len(rand_means),
            "mean_of_seed_means": float(rt.mean().item()),
            "std_of_seed_means": float(rt.std(unbiased=False).item() if len(rand_means) > 1 else 0.0),
            "seed_means": rand_means,
        }

    summary = {
        "layer_name": layer_name,
        "earlier_model": cfg["earlier_model"],
        "later_model": cfg["later_model"],
        "svd_top_k": int(cfg["svd_top_k"]),
        "frob": {"dW": frob_dW, "principal": frob_p, "nonprincipal": frob_np},
        "directions": per_dir,
        "objective": cfg["curvature"]["loss"],
        "num_minibatches": len(minibatches),
        "minibatch_used_continuations": None,  # filled by caller if available
        "objective_proxy_note": cfg.get("notes", ""),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", default=None,
                    help="single layer name to run (default: cfg.primary_layer)")
    ap.add_argument("--all-layers", action="store_true",
                    help="run on every layer in cfg.selected_layers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = results_root(cfg["pair_name"])

    if args.all_layers:
        layers = list(cfg["selected_layers"])
    else:
        layers = [args.layer or cfg["primary_layer"]]

    dev = device()
    # Load the earlier checkpoint ONCE and reuse it for every requested layer.
    # Previously this function reloaded the 1.5B fp32 weights inside the per-layer
    # loop (~35 s per layer); a 35-cell sweep would burn ~20 min on I/O alone.
    model = load_model_for_hvp(cfg["earlier_model"], device=dev,
                                dtype=torch.float32)

    # Pre-load minibatches once.
    mb_payload = torch.load(out / "minibatch.pt", map_location="cpu", weights_only=False)
    minibatches = mb_payload["minibatches"]

    for layer in layers:
        print(f"\n=== curvature for {layer} ===")
        summary = run_for_layer(cfg, layer, model, minibatches, dev)
        summary["minibatch_used_continuations"] = bool(mb_payload.get("used_continuations", False))
        path = out / "curvature" / f"{safe_layer_filename(layer)}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[curv] wrote {path}")


if __name__ == "__main__":
    main()
