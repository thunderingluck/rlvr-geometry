# rlvr-geometry

Direct curvature verification of RLVR's Three-Gate Theory.

## What this project does

Tests whether RLVR genuinely follows lower-curvature directions during training, and whether SVD-derived principal directions are a good proxy for those curvature directions — the inferential gap in the [Zhu et al. 2025](https://arxiv.org/abs/2511.08567) paper.

**Phase 0A** (done): public-endpoint pilot on the `DeepSeek-R1-Distill-Qwen-1.5B → DeepScaleR-1.5B-Preview` pair. Validates the measurement harness before running any self-trained RL.

See `CLAUDE.md` for the full research plan, hypotheses, and scope.

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

## Key implementation notes

**Curvature objective**: token-level cross-entropy NLL on a fixed minibatch of math prompts + greedy continuations from the *earlier* checkpoint. This is a tractable proxy for the GRPO objective — not the true GRPO Hessian, which would require rollout data. Flagged explicitly in `summary.json`.

**HVP method**: Pearlmutter double-backward via `torch.autograd.grad(..., create_graph=True)`. Requires `attn_implementation="eager"` — the fused SDPA paths do not implement double-backward.

**Principal subspace**: rank-k singular subspace projection of ΔW (`U_k Uᵀ_k ΔW V_k Vᵀ_k`). This is the geometric subspace definition from CLAUDE.md, distinct from the paper's coordinate-mask definition (top-α magnitude entries of the rank-k reconstruction). See `docs/decisions.md`.

**Selected layers** (layer 13 of 28): `self_attn.q_proj`, `self_attn.o_proj`, `mlp.down_proj`.

## Phase 0A findings

See `results/public_pair_deepscaler/summary.json` and `docs/notes.md` for full details.

Brief: the realized RL delta is ~94–96% off-principal by Frobenius energy across all three layers. Under NLL curvature, the principal-subspace projection of ΔW has *lower* curvature than the non-principal complement for the two attention layers (q_proj, o_proj), contrary to the strongest reading of Gate II. The MLP down_proj estimate is noisy and inconclusive. Rankings are stable across minibatches for the attention layers.

## Project structure

```
rlvr-geometry/
  CLAUDE.md                      # full research plan
  README.md
  requirements.txt
  configs/
    analysis/
      public_pair_deepscaler.json
  scripts/
    env.sh                       # cluster env vars (ORCD-specific)
    _common.py                   # shared helpers + fixed prompts
    extract_checkpoints.py
    compute_svd.py
    build_minibatch.py
    directional_curvature.py
    summarize_and_plot.py
    analyze_public_pair.sh       # end-to-end orchestrator
  results/
    public_pair_deepscaler/      # Phase 0A outputs
  docs/
    notes.md                     # Phase 0A run notes and interpretation
    decisions.md                 # implementation choices log
```
