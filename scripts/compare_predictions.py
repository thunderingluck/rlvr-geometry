"""Score the observed n=16 Experiment A result against the pre-registered
n=3 -> n=16 predictions in docs/preregistration_n16.md.

The predictions in that file use the naive 1/sqrt(n) scaling
  z_pred(n=16) = z_obs(n=3) * sqrt(16/3) ≈ 2.31 * z_obs(n=3).

This script:
  1. Loads the n=3 summary (results/<pair>/experiment_a_n3/summary.json) and
     reconstructs the predicted z + predicted verdict for every cell.
  2. Loads the n=16 summary (results/<pair>/experiment_a/summary.json) and
     records the observed z + observed verdict.
  3. Classifies each cell into one of:
        match          - predicted verdict == observed verdict
        miss_undet     - predicted significant, observed undetermined
        miss_resolved  - predicted undetermined, observed significant
        flip           - predicted blue, observed red (or vice versa)
  4. Aggregates counts into the four pre-registered "Stories" (A/B/C/D) so
     the decision rule from docs/preregistration_n16.md is mechanical, not
     narrated.

Outputs:
  - results/<pair>/experiment_a/prediction_check.json: per-cell scoring
  - prints a Markdown-style summary table + Story verdict
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _common import load_config, results_root


# Significance threshold (same 2σ that the noise-floor logic uses).
NOISE_Z = 2.0
SCALE = math.sqrt(16.0 / 3.0)  # ≈ 2.309


def verdict_from_z(z: float, threshold: float = NOISE_Z) -> str:
    if abs(z) < threshold:
        return "undet"
    return "paper" if z > 0 else "inverted"


def key(row: dict) -> tuple:
    return (row["layer"], row["operator"], row["k"],
            row["alpha"] if row["alpha"] is not None else -1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="experiment_a config (used to locate the pair's results dir)")
    ap.add_argument("--n3-subdir", default="experiment_a_n3",
                    help="subdir under the pair holding the n=3 summary.json")
    ap.add_argument("--n16-subdir", default="experiment_a",
                    help="subdir under the pair holding the n=16 summary.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    pair_out = results_root(cfg["pair_name"])
    n3_path = pair_out / args.n3_subdir / "summary.json"
    n16_path = pair_out / args.n16_subdir / "summary.json"

    with open(n3_path) as f:
        n3 = json.load(f)
    with open(n16_path) as f:
        n16 = json.load(f)

    n3_by_key = {key(r): r for r in n3["rows"]}
    n16_by_key = {key(r): r for r in n16["rows"]}

    rows: list[dict] = []
    missing_pred = []
    missing_obs = []
    for k_, r3 in n3_by_key.items():
        if k_ not in n16_by_key:
            missing_obs.append(k_)
            continue
        r16 = n16_by_key[k_]
        z3 = r3["pmc_snr"]
        z_pred = z3 * SCALE
        z_obs = r16["pmc_snr"]
        v_pred = verdict_from_z(z_pred)
        v_obs = verdict_from_z(z_obs)

        if v_pred == v_obs:
            outcome = "match"
        elif v_pred == "undet":
            outcome = "miss_resolved"
        elif v_obs == "undet":
            outcome = "miss_undet"
        else:
            # both significant, opposite signs
            outcome = "flip"

        rows.append({
            "layer": r16["layer"],
            "operator": r16["operator"],
            "k": r16["k"],
            "alpha": r16["alpha"],
            "z_n3": z3,
            "z_pred": z_pred,
            "z_obs": z_obs,
            "verdict_pred": v_pred,
            "verdict_obs": v_obs,
            "outcome": outcome,
            "ratio_obs_over_pred": (z_obs / z_pred) if abs(z_pred) > 1e-9 else None,
            "princ_minus_comp_n3": r3["princ_minus_comp"],
            "princ_minus_comp_n16": r16["princ_minus_comp"],
            "pmc_std_n3": r3["pmc_std"],
            "pmc_std_n16": r16["pmc_std"],
        })

    for k_ in n16_by_key:
        if k_ not in n3_by_key:
            missing_pred.append(k_)

    # Counts
    counts = {"match": 0, "miss_undet": 0, "miss_resolved": 0, "flip": 0}
    by_verdict_obs = {"paper": 0, "inverted": 0, "undet": 0}
    sig_predicted = 0
    sig_predicted_cleared = 0
    for r in rows:
        counts[r["outcome"]] += 1
        by_verdict_obs[r["verdict_obs"]] += 1
        if r["verdict_pred"] != "undet":
            sig_predicted += 1
            if r["verdict_obs"] != "undet":
                sig_predicted_cleared += 1

    # Story classification
    flip_count = counts["flip"]
    cleared_frac = sig_predicted_cleared / max(sig_predicted, 1)
    nonzero_obs_z = [abs(r["z_obs"]) for r in rows
                     if r["verdict_pred"] != "undet" and abs(r["z_pred"]) > 0]
    nonzero_pred_z = [abs(r["z_pred"]) for r in rows
                      if r["verdict_pred"] != "undet" and abs(r["z_pred"]) > 0]
    if nonzero_pred_z:
        median_ratio = sorted(
            abs(r["z_obs"]) / abs(r["z_pred"])
            for r in rows if r["verdict_pred"] != "undet" and abs(r["z_pred"]) > 0
        )[len(nonzero_pred_z) // 2]
    else:
        median_ratio = float("nan")

    if flip_count > 0:
        story = "D"
        story_label = "predictions flipped sign — n=3 means were biased, not just noisy"
    elif cleared_frac >= 0.80:
        story = "A"
        story_label = ("data had signal, n=3 just didn't have power "
                       f"({sig_predicted_cleared}/{sig_predicted} = "
                       f"{cleared_frac*100:.0f}% of predicted-significant cells cleared)")
    elif cleared_frac >= 0.30:
        story = "B"
        story_label = ("σ at n=3 underestimated true σ; signs survive but "
                       f"magnitudes shrink (median |z_obs|/|z_pred| = {median_ratio:.2f})")
    else:
        story = "C"
        story_label = ("predicted-significant cells refuse to clear; per-cell "
                       "claims not defensible at this n")

    # Write JSON
    out_dir = pair_out / args.n16_subdir
    out_path = out_dir / "prediction_check.json"
    payload = {
        "noise_z": NOISE_Z,
        "scale_n3_to_n16": SCALE,
        "counts": counts,
        "by_observed_verdict": by_verdict_obs,
        "sig_predicted": sig_predicted,
        "sig_predicted_cleared": sig_predicted_cleared,
        "cleared_fraction": cleared_frac,
        "median_obs_over_pred_ratio_on_predicted_sig": median_ratio,
        "story": story,
        "story_label": story_label,
        "missing_in_n16": missing_obs,
        "missing_in_n3": missing_pred,
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] {out_path}")

    # Print table
    print(f"\n=== prediction check: n=3 -> n=16 (scale = sqrt(16/3) = {SCALE:.3f}) ===\n")
    print(f"{'layer':36s}  {'op':10s}  {'k':>4s}  {'α':>5s}  "
           f"{'z(n=3)':>8s}  {'z_pred':>8s}  {'z(n=16)':>8s}  "
           f"{'pred':>10s}  {'obs':>10s}  {'outcome':>14s}")
    for r in sorted(rows, key=lambda r: (r["layer"], r["operator"],
                                           r["alpha"] if r["alpha"] is not None else -1,
                                           r["k"])):
        a_str = f"{r['alpha']:.2f}" if r["alpha"] is not None else "—"
        print(f"{r['layer'].split('model.layers.')[-1]:36s}  "
               f"{r['operator']:10s}  {str(r['k']):>4s}  {a_str:>5s}  "
               f"{r['z_n3']:>+8.2f}  {r['z_pred']:>+8.2f}  {r['z_obs']:>+8.2f}  "
               f"{r['verdict_pred']:>10s}  {r['verdict_obs']:>10s}  {r['outcome']:>14s}")

    print(
        f"\n=== aggregate counts (out of {len(rows)} cells) ===\n"
        f"  match          : {counts['match']}\n"
        f"  miss_undet     : {counts['miss_undet']}  (predicted sig, observed undet)\n"
        f"  miss_resolved  : {counts['miss_resolved']}  (predicted undet, observed sig)\n"
        f"  flip           : {counts['flip']}\n"
        f"\n=== observed verdict distribution ===\n"
        f"  paper-matching: {by_verdict_obs['paper']}\n"
        f"  inverted      : {by_verdict_obs['inverted']}\n"
        f"  undetermined  : {by_verdict_obs['undet']}\n"
        f"\n=== STORY {story}: {story_label} ===\n"
    )


if __name__ == "__main__":
    main()
