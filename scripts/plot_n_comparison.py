"""Produce a side-by-side n=3 vs n=16 sign-flip grid.

Reads results/<pair>/experiment_a_n3/summary.json and
      results/<pair>/experiment_a/summary.json,
and writes results/<pair>/experiment_a/plots/sign_flip_grid_n3_vs_n16.png.

The figure has two rows (n=3 above, n=16 below) and one column per layer,
so the reader can see cells move from gray (undetermined) to colored
(significant) as minibatch count rises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from _common import load_config, results_root
from summarize_experiment_a import _short_layer_name


def cell_value_and_label(r: dict | None) -> tuple[float, str, str, str, bool]:
    if r is None:
        return float("nan"), "", "", "", False
    pmc = r["princ_minus_comp"]
    undet = r["is_undetermined_2sigma"]
    val = 0.0 if undet else (1.0 if pmc > 0 else -1.0)
    snr = r["pmc_snr"]
    if not (snr == snr) or snr in (float("inf"), float("-inf")):
        big_z = "z=∞"
    else:
        big_z = f"z={snr:+.1f}"
    pmc_str = f"P−C={pmc:+.1e}"
    count_str = f"{r['per_mb_princ_sharper_count']}/{r['per_mb_total']}"
    return val, big_z, pmc_str, count_str, undet


def render_panel(
    ax, summary: dict, layer: str, op_alpha_pairs: list[tuple[str, float | None, str]],
    ks: list[int], cmap, draw_xticks: bool, draw_yticks: bool, draw_xlabel: bool,
) -> None:
    cells = np.full((len(op_alpha_pairs), len(ks)), np.nan)
    label_grid: list[list[tuple[float, str, str, str, bool]]] = [
        [(float("nan"), "", "", "", False)] * len(ks) for _ in op_alpha_pairs
    ]
    for ri, (op, a, _label) in enumerate(op_alpha_pairs):
        for ci, k in enumerate(ks):
            rec = next((r for r in summary["rows"]
                        if r["layer"] == layer and r["operator"] == op
                        and r["alpha"] == a and r["k"] == k), None)
            v, big_z, pmc_str, count_str, undet = cell_value_and_label(rec)
            cells[ri, ci] = v
            label_grid[ri][ci] = (v, big_z, pmc_str, count_str, undet)

    masked = np.ma.array(cells, mask=np.isnan(cells))
    ax.imshow(masked, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto",
               interpolation="nearest")
    for ri in range(len(op_alpha_pairs)):
        for ci in range(len(ks)):
            if cells[ri, ci] == 0.0:
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1, hatch="////",
                    fill=False, edgecolor="#888888", lw=0,
                ))
    ax.set_xticks(range(len(ks)))
    if draw_xticks:
        ax.set_xticklabels([str(k) for k in ks], fontsize=10)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(len(op_alpha_pairs)))
    if draw_yticks:
        ax.set_yticklabels([lab for _o, _a, lab in op_alpha_pairs], fontsize=10)
    else:
        ax.set_yticklabels([])
    if draw_xlabel:
        ax.set_xlabel("k (SVD rank)", fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    for ri in range(len(op_alpha_pairs)):
        for ci in range(len(ks)):
            v, big_z, pmc_str, count_str, undet = label_grid[ri][ci]
            if not big_z:
                continue
            base_col = "white" if v != 0.0 else "black"
            ax.text(ci, ri - 0.20, big_z, ha="center", va="center",
                     fontsize=10, fontweight="bold", color=base_col)
            ax.text(ci, ri + 0.13, pmc_str, ha="center", va="center",
                     fontsize=6.5, color=base_col)
            ax.text(ci, ri + 0.30, count_str, ha="center", va="center",
                     fontsize=6.5, color=base_col)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n3-subdir", default="experiment_a_n3")
    ap.add_argument("--n16-subdir", default="experiment_a")
    ap.add_argument("--out-name", default="sign_flip_grid_n3_vs_n16.png")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pair_out = results_root(cfg["pair_name"])

    with open(pair_out / args.n3_subdir / "summary.json") as f:
        n3 = json.load(f)
    with open(pair_out / args.n16_subdir / "summary.json") as f:
        n16 = json.load(f)

    layers = sorted(set(r["layer"] for r in n16["rows"]))
    ks = sorted(set(r["k"] for r in n16["rows"]))
    op_alpha_pairs: list[tuple[str, float | None, str]] = [("subspace", None, "subspace")]
    alphas = sorted(set(r["alpha"] for r in n16["rows"] if r["operator"] == "coord_mask"))
    for a in alphas:
        op_alpha_pairs.append(("coord_mask", a, f"coord α={a:.2f}"))

    cmap = ListedColormap(["#3b6fb6", "#e8e8e8", "#c83737"])

    n_rows = 2
    n_cols = len(layers)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 2.8 + 0.42 * len(op_alpha_pairs) * n_rows),
        squeeze=False,
    )

    for ci, layer in enumerate(layers):
        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]
        render_panel(ax_top, n3, layer, op_alpha_pairs, ks, cmap,
                      draw_xticks=False, draw_yticks=(ci == 0), draw_xlabel=False)
        render_panel(ax_bot, n16, layer, op_alpha_pairs, ks, cmap,
                      draw_xticks=True, draw_yticks=(ci == 0), draw_xlabel=True)
        ax_top.set_title(_short_layer_name(layer), fontsize=12, pad=6)
        if ci == 0:
            ax_top.set_ylabel("n = 3", fontsize=11, labelpad=10)
            ax_bot.set_ylabel("n = 16", fontsize=11, labelpad=10)

    fig.suptitle(
        "Sign-flip grid: principal − complement directional curvature\n"
        f"top: n=3 (Phase 0A draw)  ·  bottom: n=16 (extended draw)  ·  "
        f"noise floor: |P−C| < 2σ_paired",
        fontsize=12, y=0.995,
    )

    legend_handles = [
        Patch(facecolor="#c83737", edgecolor="black", label="P > C  (paper-matching)"),
        Patch(facecolor="#3b6fb6", edgecolor="black", label="P < C  (inversion)"),
        Patch(facecolor="#e8e8e8", edgecolor="#888888", hatch="////",
               label="undetermined (within 2σ_paired)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = pair_out / args.n16_subdir / "plots" / args.out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
