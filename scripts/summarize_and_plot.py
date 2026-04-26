"""Aggregate per-layer curvature JSONs into summary.json + comparison plot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, results_root, safe_layer_filename


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = results_root(cfg["pair_name"])

    per_layer: dict[str, dict] = {}
    for layer in cfg["selected_layers"]:
        cur = out / "curvature" / f"{safe_layer_filename(layer)}.json"
        if not cur.exists():
            print(f"[skip] no curvature file for {layer}")
            continue
        with open(cur) as f:
            per_layer[layer] = json.load(f)

    if not per_layer:
        raise RuntimeError("no curvature files found")

    summary = {
        "pair_name": cfg["pair_name"],
        "earlier_model": cfg["earlier_model"],
        "later_model": cfg["later_model"],
        "svd_top_k": int(cfg["svd_top_k"]),
        "objective": cfg["curvature"]["loss"],
        "objective_proxy_note": cfg.get("notes", ""),
        "per_layer": {},
    }

    for layer, rec in per_layer.items():
        dirs = rec["directions"]
        # Headline ranking by mean curvature.
        keys = ["realized", "principal", "nonprincipal"]
        if "random_aggregate" in dirs:
            random_mean = dirs["random_aggregate"]["mean_of_seed_means"]
        else:
            random_mean = None
        ranking = sorted(
            ((k, dirs[k]["mean_curvature"]) for k in keys),
            key=lambda kv: kv[1],
            reverse=True,
        )
        principal_vs_nonprincipal = (
            dirs["principal"]["mean_curvature"] - dirs["nonprincipal"]["mean_curvature"]
        )
        # Stability: max relative std across primary directions.
        rel_stds = []
        for k in keys:
            m = dirs[k]["mean_curvature"]
            s = dirs[k]["std_curvature"]
            if abs(m) > 1e-30:
                rel_stds.append(abs(s) / abs(m))
        max_rel_std = max(rel_stds) if rel_stds else float("nan")

        summary["per_layer"][layer] = {
            "frob": rec["frob"],
            "mean_curvature": {k: dirs[k]["mean_curvature"] for k in keys},
            "std_curvature": {k: dirs[k]["std_curvature"] for k in keys},
            "random_mean_curvature": random_mean,
            "ranking_high_to_low": ranking,
            "principal_minus_nonprincipal": principal_vs_nonprincipal,
            "principal_sharper_than_nonprincipal": principal_vs_nonprincipal > 0,
            "max_relative_std_over_primary": max_rel_std,
        }

    # ---------- plot ----------
    layers = list(per_layer.keys())
    dir_keys = ["realized", "principal", "nonprincipal", "random"]

    def get_dir_mean_std(rec, k):
        if k == "random":
            ra = rec["directions"].get("random_aggregate")
            if ra is None:
                return float("nan"), 0.0
            return ra["mean_of_seed_means"], ra["std_of_seed_means"]
        d = rec["directions"][k]
        return d["mean_curvature"], d["std_curvature"]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(layers) * len(dir_keys) * 0.8), 4.8))
    width = 0.18
    x = np.arange(len(layers))
    colors = {"realized": "#1f77b4", "principal": "#d62728",
              "nonprincipal": "#2ca02c", "random": "#7f7f7f"}
    for i, k in enumerate(dir_keys):
        means = []
        stds = []
        for layer in layers:
            m, s = get_dir_mean_std(per_layer[layer], k)
            means.append(m)
            stds.append(s)
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=k,
               color=colors[k], capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([l.split("model.layers.")[-1] for l in layers],
                        rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Directional curvature  vᵀ H v / ||v||²")
    ax.set_title(
        f"{cfg['pair_name']}: directional curvature by direction class\n"
        f"earlier={cfg['earlier_model'].split('/')[-1]}  "
        f"later={cfg['later_model'].split('/')[-1]}  k={cfg['svd_top_k']}"
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    plot_path = out / "plots" / "directional_curvature_bars.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {plot_path}")

    sp = out / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {sp}")

    # Print compact textual summary.
    print("\n=== ranking by direction (mean curvature, high to low) ===")
    for layer, rec in summary["per_layer"].items():
        items = ", ".join(f"{k}={v:+.3e}" for k, v in rec["ranking_high_to_low"])
        rand = rec["random_mean_curvature"]
        rand_str = f"  random={rand:+.3e}" if rand is not None else ""
        print(f"  {layer}:  {items}{rand_str}  [P-NP={rec['principal_minus_nonprincipal']:+.3e}]")


if __name__ == "__main__":
    main()
