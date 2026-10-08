# rlvr-geometry

Direct curvature verification of RLVR's Three-Gate Theory.

## What this project does

Tests whether RLVR genuinely follows lower-curvature directions during training, and whether SVD-derived principal directions are a good proxy for those curvature directions — the inferential gap in the [Zhu et al. 2025](https://arxiv.org/abs/2511.08567) paper.

**Phase 0A** (done): public-endpoint pilot on the `DeepSeek-R1-Distill-Qwen-1.5B → DeepScaleR-1.5B-Preview` pair. Validates the measurement harness before running any self-trained RL.

**Experiment A** (done): k-sweep + coordinate-mask comparison at n=3 minibatches, then a pre-registered scale-up to n=16. Headline: at n=16, every cell of the (layer × operator × k × α) grid lands inside the 2σ noise floor — the n=3 signal does not survive better variance estimates. See [Experiment A findings](#experiment-a-findings) below.


## Model pair

| Role | Model | Notes |
|---|---|---|
| Earlier (base of RL run) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2.5-1.5B SFT-distilled from DeepSeek-R1 reasoning traces |
| Later (RL endpoint) | `agentica-org/DeepScaleR-1.5B-Preview` | GRPO-trained from the earlier checkpoint on a 40K math corpus |

ΔW = later − earlier is the weight change from the GRPO stage only. Direct ancestry is documented on both model cards.

## Setup

### Prerequisites
- Python 3.11
- CUDA 12.1-compatible GPU (tested on NVIDIA L40S, 48 GB)
- ~15 GB free disk for model weights + venv

### Create a virtual environment

```bash
# Create and activate (adjust path as needed)
python3.11 -m venv /path/to/envs/rlvr
source /path/to/envs/rlvr/bin/activate

# Install torch with CUDA 12.1 wheels
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note on CUDA version**: replace `cu121` with `cu118` or `cu124` to match your driver. Check with `nvidia-smi`.

### Point HF cache off a quota-limited filesystem (if needed)

```bash
export HF_HOME=/path/to/scratch/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster downloads
```

Or source `scripts/env.sh` which sets these and the project paths for the ORCD cluster:

```bash
source scripts/env.sh
```

## Running Phase 0A

```bash
source scripts/env.sh   # sets PATH, HF_HOME, RLVR_RESULTS, etc.

# Full pipeline: download → deltas → SVD → minibatch → curvature → plot
bash scripts/analyze_public_pair.sh

# Or step by step:
python scripts/extract_checkpoints.py --config configs/analysis/public_pair_deepscaler.json
python scripts/compute_svd.py         --config configs/analysis/public_pair_deepscaler.json
python scripts/build_minibatch.py     --config configs/analysis/public_pair_deepscaler.json
python scripts/directional_curvature.py --config configs/analysis/public_pair_deepscaler.json
python scripts/directional_curvature.py --config configs/analysis/public_pair_deepscaler.json --all-layers
python scripts/summarize_and_plot.py  --config configs/analysis/public_pair_deepscaler.json
```

Outputs land in `results/public_pair_deepscaler/`:

```
results/public_pair_deepscaler/
  deltas/          # per-layer ΔW and earlier W as .pt tensors
  svd/             # top-k SVD (U_k, S_k, Vt_k) for each layer
  curvature/       # per-layer directional curvature JSON
  plots/           # comparison bar chart
  minibatch.pt     # frozen tokenized minibatch
  summary.json     # aggregated findings
```

## Running Experiment A (k-sweep + coord_mask)

Experiment A reuses Phase 0A's cached `deltas/` and `minibatch.pt`; it only re-runs the directional-curvature step under additional operators (subspace projection at k ∈ {16, 64, 256, 512} and coordinate-mask at α ∈ {0.05, 0.30, 0.50}).

```bash
# n=16 run (uses minibatch_n16.pt, the n=3 batch extended to 16 prompts)
bash scripts/run_experiment_a.sh configs/analysis/experiment_a.json

# Score the observed n=16 grid against the pre-registered n=3 → n=16 predictions
python scripts/compare_predictions.py --config configs/analysis/experiment_a.json

# Side-by-side n=3 vs n=16 sign-flip grid
python scripts/plot_n_comparison.py --config configs/analysis/experiment_a.json
```

Outputs land in `results/public_pair_deepscaler/experiment_a/`:

```
results/public_pair_deepscaler/experiment_a/
  model_layers_13_*.json   # raw per-cell curvature
  summary.json             # aggregated grid + z-test + sign-test verdicts
  prediction_check.json    # n=3 → n=16 scoring vs docs/preregistration_n16.md
  plots/                   # sign-flip grid, n=3-vs-n=16 comparison
```

## Key implementation notes

**Curvature objective**: token-level cross-entropy NLL on a fixed minibatch of math prompts + greedy continuations from the *earlier* checkpoint. This is a tractable proxy for the GRPO objective — not the true GRPO Hessian, which would require rollout data. Flagged explicitly in `summary.json`.

**HVP method**: Pearlmutter double-backward via `torch.autograd.grad(..., create_graph=True)`. Requires `attn_implementation="eager"` — the fused SDPA paths do not implement double-backward.

**Principal subspace**: rank-k singular subspace projection of ΔW (`U_k Uᵀ_k ΔW V_k Vᵀ_k`). This is the geometric subspace definition from CLAUDE.md, distinct from the paper's coordinate-mask definition (top-α magnitude entries of the rank-k reconstruction). See `docs/decisions.md`.

**Selected layers** (layer 13 of 28): `self_attn.q_proj`, `self_attn.o_proj`, `mlp.down_proj`.

## Phase 0A findings

See `results/public_pair_deepscaler/summary.json` and `docs/notes.md` for full details.

Brief: the realized RL delta is ~94–96% off-principal by Frobenius energy across all three layers. The per-layer Frobenius-energy and SVD-overlap measurements are robust and reused downstream.

The original n=3 directional-curvature result — "principal-subspace projection of ΔW has *lower* curvature than the non-principal complement for q_proj and o_proj" — is **superseded by Experiment A at n=16** (next section). It does not survive the larger sample.

## Experiment A findings

See `results/public_pair_deepscaler/experiment_a/{summary.json, prediction_check.json}` and `docs/preregistration_n16.md`.

**Grid.** Three layers (L13 `q_proj`, `o_proj`, `mlp.down_proj`) × two operators (rank-k SVD subspace projection, and coordinate-mask at α ∈ {0.05, 0.30, 0.50}) × k ∈ {16, 64, 256, 512}. 48 cells total.

**n=3 → n=16, pre-registered.** The n=3 grid was used to write `docs/preregistration_n16.md`, which predicted 31 cells would clear the 2σ noise floor at n=16 under naive 1/√n scaling.

**n=16 result.** 0 of 31 predicted-significant cells cleared. All 48/48 cells are undetermined under the 2σ z-test, and all 48/48 are also undetermined under a per-minibatch binomial sign-test (added in this run). No cells flipped sign. Median |z_obs| / |z_pred| ≈ 0.16 — meaning the paired standard deviation at n=16 was roughly 6× larger than the n=3 estimate suggested, because at n=3 the σ estimate itself had df=2 and was severely downward-biased.

`prediction_check.json` classifies this as **Story C** ("predicted-significant cells refuse to clear; per-cell claims not defensible at this n"). Operationally: at this layer subset / minibatch budget the principal-vs-complement directional-curvature gap is below our noise floor everywhere — we cannot defend either the paper's "principal is sharper" reading or the n=3 inversion finding from these data.

**What we *can* still say** from the n=16 measurements:
- Frobenius-energy off-principal fraction (~94–96%) is unchanged, since it is a closed-form quantity on ΔW and does not depend on HVP variance.
- The variance structure itself is informative: the per-minibatch (P−C) signal scatters heavy-tailed, and the n=3 σ was an unreliable basis for power calculations. This is logged so later experiments use n ≥ 16 from the start and report sign-test alongside z-test.

## Project structure

```
rlvr-geometry/
  README.md
  requirements.txt
  configs/
    analysis/
      public_pair_deepscaler.json
      experiment_a.json
  scripts/
    env.sh                       # cluster env vars (ORCD-specific)
    _common.py                   # shared helpers + fixed prompts
    _hvp.py                      # Pearlmutter HVP utilities
    extract_checkpoints.py
    compute_svd.py
    build_minibatch.py
    directional_curvature.py
    summarize_and_plot.py
    analyze_public_pair.sh       # Phase 0A end-to-end orchestrator
    experiment_a_ksweep.py       # k-sweep + coord_mask HVP runner
    summarize_experiment_a.py    # builds summary.json + sign-flip grid (z + sign test)
    compare_predictions.py       # scores n=16 vs n=3 pre-registration
    plot_n_comparison.py         # side-by-side n=3 vs n=16 sign-flip grid
    run_experiment_a.sh          # Experiment A orchestrator
  results/
    public_pair_deepscaler/      # Phase 0A + Experiment A outputs
      experiment_a/              # n=16 run
      experiment_a_n3/           # n=3 run (kept for pre-registration scoring)
  docs/
    notes.md                     # Phase 0A run notes and interpretation
    decisions.md                 # implementation choices log
    preregistration_n16.md       # pre-registered predictions for n=16 scale-up
```
