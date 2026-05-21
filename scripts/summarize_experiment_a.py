"""Aggregate experiment_a per-layer JSONs into a summary + plots.

Produces:
  - results/<pair>/experiment_a/summary.json: structured table of P-vs-C ordering
    for each (layer, operator, k, alpha), plus a sign-flip table.
  - results/<pair>/experiment_a/plots/curvature_vs_k.png: mean curvature with
    error bars vs k, three subplots (one per layer), one row per (operator, alpha).
  - results/<pair>/experiment_a/plots/sign_flip_grid.png: heatmap of
    sign(P - C) for each (operator, alpha) x k cell, faceted by layer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, results_root, safe_layer_filename


def collect(cfg: dict) -> tuple[Path, dict[str, dict]]:
    pair_out = results_root(cfg["pair_name"])
    out_dir = pair_out / cfg["experiment_a"].get("out_subdir", "experiment_a")
    per_layer: dict[str, dict] = {}
    for layer in cfg["selected_layers"]:
        path = out_dir / f"{safe_layer_filename(layer)}.json"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with open(path) as f:
            per_layer[layer] = json.load(f)
    if not per_layer:
        raise RuntimeError("no experiment_a per-layer JSONs found")
    return out_dir, per_layer


def parse_name(name: str) -> dict | None:
    """Recover (kind, k, alpha) from direction name."""
    if name.startswith("sub_princ_k"):
        return {"role": "princ", "operator": "subspace", "k": int(name[len("sub_princ_k"):]),
                "alpha": None}
    if name.startswith("sub_comp_k"):
        return {"role": "comp", "operator": "subspace", "k": int(name[len("sub_comp_k"):]),
                "alpha": None}
    if name.startswith("mask_princ_") or name.startswith("mask_comp_"):
        role = "princ" if name.startswith("mask_princ_") else "comp"
        rest = name.replace("mask_princ_", "").replace("mask_comp_", "")
        # rest = "k{K}_a{A}"
        kpart, apart = rest.split("_a")
        k = int(kpart[1:])
        alpha = float(apart.replace("0p", "0."))
        return {"role": role, "operator": "coord_mask", "k": k, "alpha": alpha}
    return None


def build_summary(per_layer: dict[str, dict]) -> dict:
    """For each (layer, operator, k, alpha) collect princ + comp mean/std and
    a P-vs-C ordering tag."""
    rows: list[dict] = []
    for layer, rec in per_layer.items():
        dirs = rec["directions"]
        # Index princ/comp by (operator, k, alpha)
        pairs: dict[tuple[str, int, float | None], dict] = {}
        for name, dd in dirs.items():
            meta = parse_name(name)
            if meta is None:
                continue
            if "skipped_reason" in dd:
                continue
            key = (meta["operator"], meta["k"], meta["alpha"])
            pairs.setdefault(key, {})
            pairs[key][meta["role"]] = {
                "name": name,
                "mean_curvature": dd["mean_curvature"],
                "std_curvature": dd["std_curvature"],
                "min_curvature": dd["min_curvature"],
                "max_curvature": dd["max_curvature"],
                "frob_norm": dd["frob_norm"],
                "per_minibatch_curvature": dd["per_minibatch_curvature"],
            }
        for (op, k, alpha), pc in pairs.items():
            if "princ" not in pc or "comp" not in pc:
                continue
            p, c = pc["princ"], pc["comp"]
            row = {
                "layer": layer,
                "operator": op,
                "k": k,
                "alpha": alpha,
                "frob_princ": p["frob_norm"],
                "frob_comp": c["frob_norm"],
                "princ_mean": p["mean_curvature"],
                "comp_mean": c["mean_curvature"],
                "princ_std": p["std_curvature"],
                "comp_std": c["std_curvature"],
                "princ_minus_comp": p["mean_curvature"] - c["mean_curvature"],
                "princ_sharper": p["mean_curvature"] > c["mean_curvature"],
                # Per-minibatch sign agreement (rough robustness check):
                "per_mb_sign_pmc": [
                    pv - cv
                    for pv, cv in zip(p["per_minibatch_curvature"], c["per_minibatch_curvature"])
                ],
            }
            row["per_mb_princ_sharper_count"] = sum(1 for d in row["per_mb_sign_pmc"] if d > 0)
            row["per_mb_total"] = len(row["per_mb_sign_pmc"])
            # Reference: random + realized for the same layer
            if "random_aggregate" in dirs:
                row["random_mean"] = dirs["random_aggregate"]["mean_of_seed_means"]
            if "realized" in dirs:
                row["realized_mean"] = dirs["realized"]["mean_curvature"]
            rows.append(row)
    return {"rows": rows}


def plot_curvature_vs_k(per_layer: dict[str, dict], summary: dict, out_path: Path,
                          cfg: dict) -> None:
    layers = list(per_layer.keys())
    # operator-alpha row label
    op_alpha_pairs: list[tuple[str, float | None, str]] = [("subspace", None, "subspace")]
    alphas = sorted(set(r["alpha"] for r in summary["rows"] if r["operator"] == "coord_mask"))
    for a in alphas:
        op_alpha_pairs.append(("coord_mask", a, f"coord_mask α={a:.2f}"))

    rows_n = len(op_alpha_pairs)
    cols_n = len(layers)
    fig, axes = plt.subplots(rows_n, cols_n,
                              figsize=(4.5 * cols_n, 2.7 * rows_n),
                              sharex=True)
    if rows_n == 1:
        axes = axes.reshape(1, -1)
    if cols_n == 1:
        axes = axes.reshape(-1, 1)

    for ci, layer in enumerate(layers):
        rec = per_layer[layer]
        dirs = rec["directions"]
        realized = dirs.get("realized", {}).get("mean_curvature", None)
        realized_std = dirs.get("realized", {}).get("std_curvature", 0.0)
        random_mean = (dirs.get("random_aggregate", {}) or {}).get("mean_of_seed_means", None)
        for ri, (op, a, label) in enumerate(op_alpha_pairs):
            ax = axes[ri, ci]
            sel = [r for r in summary["rows"] if r["layer"] == layer
                   and r["operator"] == op and r["alpha"] == a]
            sel.sort(key=lambda r: r["k"])
            ks = [r["k"] for r in sel]
            pm = [r["princ_mean"] for r in sel]
            ps = [r["princ_std"] for r in sel]
            cm = [r["comp_mean"] for r in sel]
            cs = [r["comp_std"] for r in sel]
            ax.errorbar(ks, pm, yerr=ps, marker="o", label="principal", color="#d62728",
                        capsize=3, lw=1.5)
            ax.errorbar(ks, cm, yerr=cs, marker="s", label="complement", color="#2ca02c",
                        capsize=3, lw=1.5)
            if realized is not None:
                ax.axhline(realized, color="#1f77b4", lw=1.0, ls="--",
                           label="realized" if (ri == 0 and ci == 0) else None)
            if random_mean is not None:
                ax.axhline(random_mean, color="#7f7f7f", lw=1.0, ls=":",
                           label="random" if (ri == 0 and ci == 0) else None)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.grid(axis="both", linestyle=":", alpha=0.5)
            if ci == 0:
                ax.set_ylabel(f"{label}\n vᵀHv/||v||²", fontsize=9)
            if ri == 0:
                short = layer.split("model.layers.")[-1]
                ax.set_title(short, fontsize=10)
            if ri == rows_n - 1:
                ax.set_xlabel("k (SVD rank)")
    # Legend on top-left
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Experiment A: directional curvature vs k\n"
        f"earlier={cfg['earlier_model'].split('/')[-1]}  "
        f"later={cfg['later_model'].split('/')[-1]}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_sign_flip_grid(summary: dict, out_path: Path, cfg: dict) -> None:
    """Heatmap showing sign(princ - comp) for each (operator, alpha) x k cell,
    one subplot per layer. +1 (red) = principal sharper than complement
    (paper's claim); -1 (blue) = inversion (principal lower curvature)."""
    layers = sorted(set(r["layer"] for r in summary["rows"]))
    ks = sorted(set(r["k"] for r in summary["rows"]))
    op_alpha_pairs: list[tuple[str, float | None, str]] = [("subspace", None, "subspace")]
    alphas = sorted(set(r["alpha"] for r in summary["rows"] if r["operator"] == "coord_mask"))
    for a in alphas:
        op_alpha_pairs.append(("coord_mask", a, f"coord α={a:.2f}"))

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 2.5 + 0.4 * len(op_alpha_pairs)),
                              sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        cells = np.zeros((len(op_alpha_pairs), len(ks)))
        text = [[""] * len(ks) for _ in op_alpha_pairs]
        for ri, (op, a, _label) in enumerate(op_alpha_pairs):
            for ci, k in enumerate(ks):
                rec = next((r for r in summary["rows"]
                            if r["layer"] == layer and r["operator"] == op
                            and r["alpha"] == a and r["k"] == k), None)
                if rec is None:
                    cells[ri, ci] = np.nan
                    text[ri][ci] = ""
                else:
                    pmc = rec["princ_minus_comp"]
                    cells[ri, ci] = 1.0 if pmc > 0 else -1.0
                    text[ri][ci] = (
                        f"{pmc:+.1e}\n"
                        f"{rec['per_mb_princ_sharper_count']}/{rec['per_mb_total']}"
                    )
        # mask NaN
        masked = np.ma.array(cells, mask=np.isnan(cells))
        im = ax.imshow(masked, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_yticks(range(len(op_alpha_pairs)))
        ax.set_yticklabels([lab for _o, _a, lab in op_alpha_pairs], fontsize=9)
        ax.set_xlabel("k")
        ax.set_title(layer.split("model.layers.")[-1], fontsize=10)
        for ri in range(len(op_alpha_pairs)):
            for ci in range(len(ks)):
                if text[ri][ci]:
                    ax.text(ci, ri, text[ri][ci], ha="center", va="center",
                             fontsize=7, color="black")
    fig.suptitle(
        "Sign of (principal − complement) mean curvature  "
        "[red = principal sharper (paper); blue = inverted]\n"
        "Per-cell label: P−C value  +  per-minibatch count where P > C",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def print_text_table(summary: dict) -> None:
    print("\n=== experiment_a summary (per (layer, operator, k, alpha)) ===")
    header = ("layer", "op", "k", "alpha", "P_mean", "C_mean", "P-C",
              "P>C", "per-mb")
    fmt = "{:36s}  {:10s}  {:>4s}  {:>5s}  {:>10s}  {:>10s}  {:>11s}  {:>4s}  {}"
    print(fmt.format(*header))
    rows = sorted(summary["rows"], key=lambda r: (r["layer"], r["operator"],
                                                    r["alpha"] if r["alpha"] is not None else -1.0,
                                                    r["k"]))
    for r in rows:
        a_str = f"{r['alpha']:.2f}" if r["alpha"] is not None else "—"
        print(fmt.format(
            r["layer"].split("model.layers.")[-1],
            r["operator"],
            str(r["k"]),
            a_str,
            f"{r['princ_mean']:+.2e}",
            f"{r['comp_mean']:+.2e}",
            f"{r['princ_minus_comp']:+.2e}",
            "Y" if r["princ_sharper"] else "N",
            f"{r['per_mb_princ_sharper_count']}/{r['per_mb_total']}",
        ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir, per_layer = collect(cfg)
    summary = build_summary(per_layer)

    # Persist summary
    sp = out_dir / "summary.json"
    with open(sp, "w") as f:
        json.dump(
            {
                "pair_name": cfg["pair_name"],
                "earlier_model": cfg["earlier_model"],
                "later_model": cfg["later_model"],
                "k_sweep": cfg["experiment_a"]["k_sweep"],
                "coord_mask_alpha": cfg["experiment_a"]["coord_mask_alpha"],
                "extra_alphas": cfg["experiment_a"].get("extra_alphas", []),
                "rows": summary["rows"],
            },
            f, indent=2,
        )
    print(f"[summary] wrote {sp}")

    # Plots
    plot_curvature_vs_k(per_layer, summary,
                         out_dir / "plots" / "curvature_vs_k.png", cfg)
    plot_sign_flip_grid(summary,
                         out_dir / "plots" / "sign_flip_grid.png", cfg)

    print_text_table(summary)


if __name__ == "__main__":
    main()
