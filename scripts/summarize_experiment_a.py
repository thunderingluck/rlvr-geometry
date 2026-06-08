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
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def _pmc_std(diffs: list[float]) -> float:
    """Population std of the per-minibatch (princ - comp) gap.

    We use the *paired* difference (same minibatch -> same loss surface, so P
    and C share noise that should partly cancel). Using std of the difference
    directly is the right noise estimate for the contrast we're testing.
    Population (unbiased=False) matches existing convention in the per-direction
    std_curvature fields.
    """
    if not diffs:
        return 0.0
    t = torch.tensor(diffs, dtype=torch.float64)
    if t.numel() < 2:
        return 0.0
    return float(t.std(unbiased=False).item())


def _binomial_two_sided_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for H0: P(P>C) = 0.5.

    p = sum of PMFs whose value is <= the observed PMF. No scipy dep.
    Returns 1.0 for n == 0.
    """
    if n == 0:
        return 1.0
    from math import comb
    pmfs = [comb(n, i) * (0.5 ** n) for i in range(n + 1)]
    observed = pmfs[k]
    eps = 1e-12  # ties on the observed PMF should all be included
    return min(1.0, sum(pmf for pmf in pmfs if pmf <= observed + eps))


def _median_and_mad(diffs: list[float]) -> tuple[float, float]:
    """Median and median absolute deviation (raw, no scaling)."""
    if not diffs:
        return 0.0, 0.0
    arr = sorted(diffs)
    n = len(arr)
    med = arr[n // 2] if n % 2 else 0.5 * (arr[n // 2 - 1] + arr[n // 2])
    abs_dev = sorted(abs(x - med) for x in diffs)
    mad = abs_dev[n // 2] if n % 2 else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
    return med, mad


def build_summary(per_layer: dict[str, dict], noise_z: float = 2.0) -> dict:
    """For each (layer, operator, k, alpha) collect princ + comp mean/std and
    a P-vs-C ordering tag.

    Noise floor: a cell is `undetermined` when |princ - comp mean| <
    noise_z * std(per_minibatch_pmc). With only ~3 minibatches the std itself
    is noisy, so we ALSO log a more conservative `undetermined_strict` using
    the t-distribution at the actual df (3 mbs -> df=2 -> t_{0.975} ≈ 4.30).
    """
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
            per_mb_pmc = [
                pv - cv
                for pv, cv in zip(p["per_minibatch_curvature"], c["per_minibatch_curvature"])
            ]
            pmc_mean = p["mean_curvature"] - c["mean_curvature"]
            pmc_std = _pmc_std(per_mb_pmc)
            n_mb = len(per_mb_pmc)
            # Heuristic 2σ noise floor (user-requested).
            noise_floor = noise_z * pmc_std
            is_undetermined = abs(pmc_mean) < noise_floor
            # SNR for plot labels: signal-to-paired-noise ratio.
            snr = (pmc_mean / pmc_std) if pmc_std > 0 else float("inf")
            # Strict variant: t_{0.975, df=n-1} CI bound. Hard-coded small-n table
            # to avoid scipy dependency; n=2 -> t=12.71, n=3 -> t=4.30,
            # n=4 -> t=3.18, n>=5 -> 2.78/2.57/2.45/...
            t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                       6: 2.571, 7: 2.447, 8: 2.365}
            t_crit = t_table.get(n_mb, 1.96)
            se = pmc_std / (n_mb ** 0.5) if n_mb else float("inf")
            is_undetermined_strict = abs(pmc_mean) < t_crit * se
            # Sign-test on the per-minibatch (P-C) signs. Heavy-tailed σ can
            # tank the z-test even when 13+ of 16 minibatches agree on the
            # sign of the gap; the sign-test catches that.
            k_pos = sum(1 for d in per_mb_pmc if d > 0)
            sign_test_p = _binomial_two_sided_p(k_pos, n_mb) if n_mb else 1.0
            if sign_test_p >= 0.05 or n_mb < 4:
                sign_test_verdict = "undet"
            else:
                sign_test_verdict = "paper" if k_pos > n_mb / 2 else "inverted"
            # Robust location/scale: median (P-C) and MAD across minibatches.
            pmc_median, pmc_mad = _median_and_mad(per_mb_pmc)
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
                "princ_minus_comp": pmc_mean,
                "princ_sharper": pmc_mean > 0,
                "per_mb_sign_pmc": per_mb_pmc,
                "pmc_std": pmc_std,
                "pmc_snr": snr,
                "pmc_median": pmc_median,
                "pmc_mad": pmc_mad,
                "noise_floor_2sigma": noise_floor,
                "is_undetermined_2sigma": is_undetermined,
                "is_undetermined_strict_t": is_undetermined_strict,
                "sign_test_p": sign_test_p,
                "sign_test_verdict": sign_test_verdict,
                "n_mb": n_mb,
            }
            row["per_mb_princ_sharper_count"] = k_pos
            row["per_mb_total"] = n_mb
            # Reference: random + realized for the same layer
            if "random_aggregate" in dirs:
                row["random_mean"] = dirs["random_aggregate"]["mean_of_seed_means"]
            if "realized" in dirs:
                row["realized_mean"] = dirs["realized"]["mean_curvature"]
            rows.append(row)
    return {"rows": rows, "noise_z": noise_z}


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


def _short_layer_name(layer: str) -> str:
    """e.g. 'model.layers.13.self_attn.q_proj.weight' -> 'L13 q_proj'."""
    m = re.match(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.([^.]+)\.weight", layer)
    if m:
        return f"L{m.group(1)} {m.group(2)}"
    return layer.split("model.layers.")[-1].replace(".weight", "")


def plot_sign_flip_grid(summary: dict, out_path: Path, cfg: dict) -> None:
    """Heatmap showing sign(princ - comp) for each (operator, alpha) x k cell,
    one subplot per layer.

    Paper-figure design choices:
      - Three discrete colors (blue / light-gray / red) plus a hatch pattern on
        undetermined cells so the figure is legible in B&W and color-blind safe.
      - Per-cell label is dominated by the z-score (the statistic the noise
        floor compares against); P-C magnitude and per-mb count appear as
        smaller annotations.
      - Embedded legend below the grid.
      - dpi=200 for camera-ready output.

    Cell color encoding:
        +1 (red)   = principal sharper than complement (paper's claim)
        -1 (blue)  = inversion (principal lower curvature than complement)
         0 (gray)  = UNDETERMINED: |P-C| < noise_z * std(per-mb P-C)
    """
    layers = sorted(set(r["layer"] for r in summary["rows"]))
    ks = sorted(set(r["k"] for r in summary["rows"]))
    op_alpha_pairs: list[tuple[str, float | None, str]] = [("subspace", None, "subspace")]
    alphas = sorted(set(r["alpha"] for r in summary["rows"] if r["operator"] == "coord_mask"))
    for a in alphas:
        op_alpha_pairs.append(("coord_mask", a, f"coord α={a:.2f}"))

    noise_z = summary.get("noise_z", 2.0)
    # Sample one row to learn n_mb (assumed constant across cells).
    n_mb = None
    for r in summary["rows"]:
        if "n_mb" in r:
            n_mb = r["n_mb"]; break

    # Three discrete colors. RdBu_r centered at 0 gives red/white/blue; we want
    # a slightly less-bright gray for undetermined so the boundary is visible.
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#3b6fb6", "#e8e8e8", "#c83737"])  # blue, gray, red

    fig, axes = plt.subplots(
        1, len(layers),
        figsize=(5.0 * len(layers), 3.2 + 0.42 * len(op_alpha_pairs)),
        sharey=True,
    )
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        cells = np.full((len(op_alpha_pairs), len(ks)), np.nan)
        # Per-cell label parts: (big_z, small_value, small_count, is_undet)
        labels: list[list[tuple[str, str, str, bool]]] = [
            [("", "", "", False)] * len(ks) for _ in op_alpha_pairs
        ]
        for ri, (op, a, _label) in enumerate(op_alpha_pairs):
            for ci, k in enumerate(ks):
                rec = next((r for r in summary["rows"]
                            if r["layer"] == layer and r["operator"] == op
                            and r["alpha"] == a and r["k"] == k), None)
                if rec is None:
                    continue
                pmc = rec["princ_minus_comp"]
                undet = rec["is_undetermined_2sigma"]
                if undet:
                    cells[ri, ci] = 0.0
                else:
                    cells[ri, ci] = 1.0 if pmc > 0 else -1.0
                snr = rec["pmc_snr"]
                if not (snr == snr) or snr in (float("inf"), float("-inf")):
                    big_z = "z=∞"
                else:
                    big_z = f"z={snr:+.1f}"
                pmc_str = f"P−C={pmc:+.1e}"
                count_str = f"{rec['per_mb_princ_sharper_count']}/{rec['per_mb_total']}"
                labels[ri][ci] = (big_z, pmc_str, count_str, undet)

        masked = np.ma.array(cells, mask=np.isnan(cells))
        ax.imshow(masked, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto",
                   interpolation="nearest")

        # Hatching for undetermined cells (B&W safety).
        for ri in range(len(op_alpha_pairs)):
            for ci in range(len(ks)):
                if cells[ri, ci] == 0.0:
                    ax.add_patch(plt.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        hatch="////", fill=False, edgecolor="#888888", lw=0,
                    ))

        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks], fontsize=10)
        ax.set_yticks(range(len(op_alpha_pairs)))
        ax.set_yticklabels([lab for _o, _a, lab in op_alpha_pairs], fontsize=10)
        ax.set_xlabel("k (SVD rank)", fontsize=10)
        ax.set_title(_short_layer_name(layer), fontsize=12, pad=6)
        ax.tick_params(axis="both", which="both", length=0)

        for ri in range(len(op_alpha_pairs)):
            for ci in range(len(ks)):
                big_z, pmc_str, count_str, undet = labels[ri][ci]
                if not big_z:
                    continue
                # Choose label color for contrast: white on red/blue, black on gray.
                base_col = "white" if cells[ri, ci] != 0.0 else "black"
                ax.text(ci, ri - 0.20, big_z, ha="center", va="center",
                         fontsize=11, fontweight="bold", color=base_col)
                ax.text(ci, ri + 0.13, pmc_str, ha="center", va="center",
                         fontsize=7, color=base_col)
                ax.text(ci, ri + 0.30, count_str, ha="center", va="center",
                         fontsize=7, color=base_col)
                # Sign-test marker: small badge in the corner of the cell when
                # the binomial sign test is significant (p<0.05). Black border,
                # color = sign-test verdict color. Lets the reader see at a
                # glance which "undetermined under z" cells are highly
                # directional under a sign test.
                rec_sig = next((r for r in summary["rows"]
                                 if r["layer"] == layer and r["operator"] == op
                                 and r["alpha"] == a and r["k"] == ks[ci]
                                 ), None)
                if rec_sig and rec_sig.get("sign_test_verdict", "undet") != "undet":
                    sv = rec_sig["sign_test_verdict"]
                    badge_color = "#c83737" if sv == "paper" else "#3b6fb6"
                    ax.add_patch(plt.Circle(
                        (ci + 0.38, ri - 0.38), 0.07,
                        facecolor=badge_color, edgecolor="black", lw=0.8,
                        zorder=10,
                    ))

    # Suptitle + embedded legend.
    n_str = f"n_mb = {n_mb}" if n_mb is not None else ""
    fig.suptitle(
        f"Directional curvature: principal vs complement (sign-flip grid, {noise_z}σ noise floor, {n_str})",
        fontsize=12, y=0.995,
    )

    # Legend axes at the bottom.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor="#c83737", edgecolor="black", label="P > C  (paper-matching, z-test)"),
        Patch(facecolor="#3b6fb6", edgecolor="black", label="P < C  (inversion, z-test)"),
        Patch(facecolor="#e8e8e8", edgecolor="#888888", hatch="////",
               label=f"undetermined under z (|P−C| < {noise_z}σ_paired)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b6fb6",
                markeredgecolor="black", markersize=8,
                label="sign-test significant at p<0.05 (badge in cell corner)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def print_text_table(summary: dict) -> None:
    noise_z = summary.get("noise_z", 2.0)
    print(f"\n=== experiment_a summary  (z-test floor: |P-C| < {noise_z}σ_paired; "
          f"sign-test: binomial two-sided p<0.05) ===")
    header = ("layer", "op", "k", "α", "P_mean", "C_mean", "P-C",
              "σ_pmc", "z", "z?", "k/n", "p_sign", "sign?")
    fmt = ("{:36s}  {:10s}  {:>4s}  {:>5s}  {:>10s}  {:>10s}  {:>11s}  "
           "{:>9s}  {:>6s}  {:>4s}  {:>5s}  {:>7s}  {:>8s}")
    print(fmt.format(*header))
    rows = sorted(summary["rows"], key=lambda r: (r["layer"], r["operator"],
                                                    r["alpha"] if r["alpha"] is not None else -1.0,
                                                    r["k"]))
    counts = {"red_z": 0, "blue_z": 0, "undet_z": 0,
              "red_sign": 0, "blue_sign": 0, "undet_sign": 0,
              "z_undet_but_sign_resolved": 0, "total": 0}
    for r in rows:
        a_str = f"{r['alpha']:.2f}" if r["alpha"] is not None else "—"
        if r["is_undetermined_2sigma"]:
            z_verdict = "?"
            counts["undet_z"] += 1
        elif r["princ_sharper"]:
            z_verdict = "Y"
            counts["red_z"] += 1
        else:
            z_verdict = "N"
            counts["blue_z"] += 1
        sv = r.get("sign_test_verdict", "undet")
        if sv == "paper":
            sign_str = "Y"; counts["red_sign"] += 1
        elif sv == "inverted":
            sign_str = "N"; counts["blue_sign"] += 1
        else:
            sign_str = "?"; counts["undet_sign"] += 1
        if r["is_undetermined_2sigma"] and sv != "undet":
            counts["z_undet_but_sign_resolved"] += 1
        counts["total"] += 1
        snr = r["pmc_snr"]
        z_str = "inf" if snr == float("inf") else f"{snr:+.2f}"
        print(fmt.format(
            r["layer"].split("model.layers.")[-1],
            r["operator"],
            str(r["k"]),
            a_str,
            f"{r['princ_mean']:+.2e}",
            f"{r['comp_mean']:+.2e}",
            f"{r['princ_minus_comp']:+.2e}",
            f"{r['pmc_std']:.2e}",
            z_str,
            z_verdict,
            f"{r['per_mb_princ_sharper_count']}/{r['per_mb_total']}",
            f"{r.get('sign_test_p', 1.0):.1e}" if r.get('sign_test_p') is not None else "—",
            sign_str,
        ))
    print(
        f"\n[counts] total={counts['total']}  |  "
        f"z-test: paper={counts['red_z']} inv={counts['blue_z']} undet={counts['undet_z']}  |  "
        f"sign-test: paper={counts['red_sign']} inv={counts['blue_sign']} undet={counts['undet_sign']}  |  "
        f"z-undet-but-sign-resolved={counts['z_undet_but_sign_resolved']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir, per_layer = collect(cfg)
    noise_z = float(cfg["experiment_a"].get("noise_z", 2.0))
    summary = build_summary(per_layer, noise_z=noise_z)

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
                "noise_z": noise_z,
                "noise_floor_rule": (
                    f"|princ_minus_comp| < {noise_z} * std_paired(per_mb_sign_pmc) "
                    f"=> is_undetermined_2sigma"
                ),
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
