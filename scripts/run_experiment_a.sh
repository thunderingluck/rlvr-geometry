#!/usr/bin/env bash
# Experiment A: k-sweep + coordinate-mask comparison.
# Reuses Phase 0A's cached deltas + minibatch; only adds new direction
# definitions and re-runs F2 (directional curvature) under each.
#
# Usage:
#   bash scripts/run_experiment_a.sh                  # default config
#   bash scripts/run_experiment_a.sh path/to/cfg.json
#   bash scripts/run_experiment_a.sh cfg.json --layer model.layers.13.self_attn.q_proj.weight

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/env.sh"

CFG="${1:-$ROOT/configs/analysis/experiment_a.json}"
shift || true
EXTRA_ARGS=("$@")

cd "$ROOT"

# Precondition: Phase 0A must have run so that deltas/<layer>.pt and
# minibatch.pt exist. We do NOT recompute those here.
echo "===== [1/2] experiment_a_ksweep (HVPs over k + operator sweep) ====="
python "$HERE/experiment_a_ksweep.py" --config "$CFG" "${EXTRA_ARGS[@]}"

echo "===== [2/2] summarize_experiment_a (summary.json + plots) ====="
python "$HERE/summarize_experiment_a.py" --config "$CFG"

echo "Done. Results in $RLVR_RESULTS/$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["pair_name"])' "$CFG")/experiment_a"
