#!/usr/bin/env bash
# End-to-end Phase 0A pipeline: deltas -> SVD -> minibatch -> curvature -> plot.
#
# Usage:
#   bash scripts/analyze_public_pair.sh                  # default config
#   bash scripts/analyze_public_pair.sh path/to/cfg.json
#   bash scripts/analyze_public_pair.sh cfg.json --all-layers

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/env.sh"

CFG="${1:-$ROOT/configs/analysis/public_pair_deepscaler.json}"
shift || true
EXTRA_ARGS=("$@")

cd "$ROOT"

echo "===== [1/5] extract_checkpoints ====="
python "$HERE/extract_checkpoints.py" --config "$CFG"

echo "===== [2/5] compute_svd ====="
python "$HERE/compute_svd.py" --config "$CFG"

echo "===== [3/5] build_minibatch ====="
python "$HERE/build_minibatch.py" --config "$CFG"

echo "===== [4/5] directional_curvature ====="
python "$HERE/directional_curvature.py" --config "$CFG" "${EXTRA_ARGS[@]}"

echo "===== [5/5] summarize_and_plot ====="
python "$HERE/summarize_and_plot.py" --config "$CFG"

echo "Done. Results in $RLVR_RESULTS"
