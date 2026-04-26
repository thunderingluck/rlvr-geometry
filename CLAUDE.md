# CLAUDE.md

## Project: Direct curvature verification of RLVR's Three-Gate Theory

### North star
Test whether RLVR genuinely follows **lower-curvature directions** during training, and whether the paper's **SVD-derived principal directions / principal-weight masks** are actually a good proxy for those curvature directions.

This project exists to close one precise inferential gap:
- The current RLVR paper shows that RLVR preserves spectral structure, rotates principal subspaces less than SFT, and avoids **SVD-defined principal weights**.
- But that does **not yet directly prove** that RLVR avoids **high-curvature directions** of the RL objective.
- We therefore want a direct curvature study, starting with a **public-checkpoint endpoint pilot** and only then moving to a **self-trained trajectory study**.

## Primary research question
Do RL-finetuned checkpoints occupy lower-curvature directions than matched SFT-style or principal-subspace directions, and do SVD-defined principal directions align with true curvature directions strongly enough to justify Gate II's interpretation?

## Core hypotheses
1. **RL low-curvature occupancy hypothesis**
   RLVR updates place more energy in mid-/low-curvature bands than matched SFT updates.

2. **Proxy-validity hypothesis**
   SVD-defined principal directions overlap with true high-curvature directions only partially; they are a noisy proxy, not an identity.

3. **Gate II refinement hypothesis**
   Gate II is directionally correct — RLVR is more curvature-avoiding than SFT — but the strongest version of "principal weights == high-curvature directions" may be overstated.

4. **Magnitude/precision confound hypothesis**
   Some apparent off-principal behavior is real geometry; some apparent sparsity is amplified by weight magnitude and bf16 visibility thresholds.

## Explicit definitions
Use terms carefully.

- **Principal directions (weight-SVD sense):** top singular subspaces of a weight matrix or positions selected from rank-k reconstructions.
- **Curvature directions:** top eigendirections of the Hessian or Fisher / Gauss-Newton proxy of the actual training objective.
- **RL objective:** the GRPO training surrogate actually used at the checkpoint under analysis, not a generic language-model loss.
- **Directional curvature:** quadratic form along a vector `v`, e.g. `v^T H v / ||v||^2`.

Never conflate these without measuring their alignment.

## Scope
### In scope
- public endpoint checkpoint pilot first
- 1.5B-scale models first
- per-layer SVD diagnostics
- directional sharpness / critical sharpness along chosen directions
- Fisher / Hessian proxies
- RL endpoint vs base comparison
- later: self-trained GRPO trajectory study with dense checkpointing
- later: RL vs matched SFT comparison

### Out of scope for the first phase
- 7B+ scale confirmation
- full NTK atlas
- full-model exact Hessian eigenspectra
- PEFT method design beyond small proof-of-concept ablations
- large benchmark sweeps
- claiming trajectory-level conclusions before training our own run

## Recommended public checkpoint pair (start here)
### Primary pair
- **Base / earlier checkpoint:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Public RL endpoint:** `agentica-org/DeepScaleR-1.5B-Preview`

Why this pair:
- direct ancestry is explicit in the public model card
- both are public Hugging Face checkpoints
- 1.5B scale keeps curvature probing manageable
- this is the cleanest public endpoint pair for a static pilot

### Optional upstream chain for a second endpoint comparison
- `Qwen/Qwen2.5-Math-1.5B` → `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

This gives a second public delta and helps check whether any curvature signature is specific to the DeepScaleR stage or already present in the earlier reasoning-distillation stage.

## Recommended experiment order
### Phase 0A — public endpoint pilot (mandatory)
Before any RL training, validate the analysis pipeline on **public checkpoints**.

Default target:
- base = `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- later = `agentica-org/DeepScaleR-1.5B-Preview`

Deliverables:
- load both checkpoints
- compute parameter deltas for selected layers
- compute per-layer SVD on the earlier checkpoint
- compute directional curvature estimates on a fixed minibatch for:
  - realized delta direction
  - projection onto top-k SVD subspace
  - complementary subspace
  - matched random direction
- verify estimator stability across repeated probes / minibatches / seeds

Exit criterion:
- directional curvature estimates are numerically stable enough to rank directions consistently
- at least one layer shows whether principal vs non-principal projections differ meaningfully in curvature

### Phase 0B — public endpoint extension (optional but recommended)
Run the same analysis on:
- `Qwen/Qwen2.5-Math-1.5B` → `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

Purpose:
- test whether the same geometric signature appears on a second public pair
- reduce the risk that results are idiosyncratic to one model family transition

### Phase 1 — short-horizon self-trained RL pilot
Only after Phase 0 works, run a small GRPO pilot at 1.5B scale.

Suggested settings:
- 100–200 update steps
- save checkpoints every 10–20 steps early, then every 25–50 steps
- fixed held-out probe minibatch for curvature analysis
- small math / code-style verifiable dataset
- framework: OpenRLHF

At each analyzed checkpoint:
1. compute per-layer SVD for selected matrices
2. compute realized update vector between checkpoints
3. estimate directional curvature for:
   - realized RL update
   - principal-subspace projection of that update
   - non-principal projection
   - matched random directions
4. estimate diagonal Fisher / Hessian proxy only as a secondary diagnostic

Exit criterion:
- clear signal on whether RL update directions are lower-curvature than principal-subspace projections
- dense checkpoints are saved and reload cleanly for analysis

### Phase 2 — RL vs SFT comparison
Using the same base model and prompts, compare:
- RL gradient / realized update direction
- matched SFT gradient direction
- random matched-norm direction

Primary question:
- does RL occupy lower-curvature directions than SFT at the same checkpoint?

### Phase 3 — robustness and causal checks
Only after Phase 1–2 succeed:
- KL sweep / clipping conservatism sweep
- bf16 vs fp32 serialization visibility check
- limited rebasing / orthogonal rotation intervention where feasible
- PEFT ablations: LoRA vs PiSSA vs mask baselines

## Metrics that matter most
### Primary
- directional curvature of realized RL endpoint deltas or RL updates
- directional curvature of SFT gradients / updates
- alignment between SVD-defined principal subspaces and estimated high-curvature eigenspaces
- fraction of update energy in top / mid / low curvature bands

### Secondary
- weight-spectrum drift
- principal-angle rotation of weight-SVD subspaces
- overlap with SVD-derived principal masks
- overlap with low-magnitude masks
- forward KL to base / reference policy
- pass@1 or task reward on held-out eval set

### Tertiary
- bf16-visible update sparsity
- Jaccard overlap / consensus maps

## Methodological rules
1. **Do not rely on diagonal Hessian alone.**
   Diagonal Fisher / Hessian estimates are screening tools, not decisive evidence about directional geometry.

2. **Prefer directional measurements.**
   Use Hessian-vector products / critical sharpness / band occupancy wherever possible.

3. **Keep the objective consistent.**
   Curvature must be measured for the actual checkpoint and actual RL surrogate being optimized.

4. **Disentangle geometry from precision.**
   Always separate actual parameter deltas from bf16-visible mask effects.

5. **Disentangle geometry from magnitude.**
   Low-magnitude weights may be easy places for visible updates; do not mistake that for low curvature.

6. **Use matched comparisons.**
   RL vs SFT comparisons must share prompts, base model, and comparable evaluation protocol.

7. **Do not overclaim from endpoint-only public pairs.**
   Public checkpoint analysis can justify an endpoint geometry claim, not a full training-dynamics claim.

8. **Log dense trajectories once self-training starts.**
   Endpoint-only analysis is not enough for the final paper.

## Default layer subset for pilot analysis
Unless there is a strong reason otherwise, start with these per-layer targets:
- attention `q_proj`
- attention `o_proj`
- MLP `down_proj`

These are the most interpretable early targets for testing stripe/locality and SVD-vs-curvature mismatch.

## Directory and artifact structure
```text
project/
  configs/
    rl/
    sft/
    analysis/
  scripts/
    analyze_public_pair.sh
    run_grpo.sh
    run_sft.sh
    extract_checkpoints.py
    compute_svd.py
    directional_curvature.py
    fisher_diag.py
    compare_rl_sft.py
  notebooks/
  results/
    public_pair_deepscaler/
      deltas/
      svd/
      curvature/
      plots/
      summary.json
    pilot_run_001/
      checkpoints/
      deltas/
      svd/
      curvature/
      plots/
      summary.json
  docs/
    notes.md
    decisions.md
```

## What counts as success
A successful pilot produces at least one of the following:
1. evidence that realized RL endpoint deltas or updates are lower-curvature than matched SFT directions;
2. evidence that SVD principal directions are only a partial proxy for top curvature directions;
3. evidence that the apparent off-principal effect is mostly explained by magnitude / precision rather than curvature.

All three are publishable outcomes if the measurements are credible.

## What not to claim too early
Do **not** claim any of the following from early pilot results:
- universal proof of Gate II
- universal invalidation of principal-weight methods
- universal RL law across tasks / model families
- curvature conclusions from mask overlap alone
- trajectory conclusions from one public endpoint pair

## Immediate next step
The first step is **not** a full multi-GPU training campaign.

The first step is to **de-risk the measurement stack on a public pair**.

### Actionable first task
Build a minimal analysis harness that takes:
- earlier checkpoint = `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- later checkpoint = `agentica-org/DeepScaleR-1.5B-Preview`
- a fixed minibatch
- a selected layer
and outputs:
1. the layer delta `ΔW`
2. top-k SVD subspaces of the earlier checkpoint's weight matrix
3. projection of `ΔW` into principal vs non-principal subspaces
4. directional curvature / critical sharpness along:
   - `ΔW`
   - `Proj_principal(ΔW)`
   - `Proj_nonprincipal(ΔW)`
   - random matched-norm direction
5. a small JSON summary + one comparison plot

### Concrete acceptance criteria for this first task
- runs on one layer of the DeepSeek-R1-Distill → DeepScaleR pair without crashing
- produces stable curvature rankings across repeated probes
- makes clear whether the principal projection is sharper than the non-principal projection

Only after this works should we launch the checkpointed GRPO run.

## If a first training experiment must be launched immediately
Run a **100-step GRPO pilot** on the cheapest compatible 1.5B model with checkpoints every 10–20 steps and analyze only 3 layers. Do not start with a 1000-step campaign.

## Decision memo
The updated plan is:
- **Yes:** direct curvature verification of Gate II is still the right next move.
- **Yes:** start with a public endpoint pair before training anything.
- **Yes:** use `DeepSeek-R1-Distill-Qwen-1.5B` → `DeepScaleR-1.5B-Preview` as the default public pair.
- **Yes:** compare RL directions to SFT directions once the measurement harness works.
- **Refinement:** treat diagonal Hessian / Hutchinson as secondary; make directional curvature the primary object.
- **Refinement:** keep endpoint and trajectory claims separate.
- **Refinement:** make the first training milestone a short, checkpoint-dense GRPO pilot only after the public-pair harness is validated.
