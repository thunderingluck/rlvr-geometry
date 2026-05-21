# Phase 0A — Public-Pair Curvature Pilot: Run Notes

## Pair
- earlier = `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- later   = `agentica-org/DeepScaleR-1.5B-Preview`

## Pipeline (run via `bash scripts/analyze_public_pair.sh`)
1. `extract_checkpoints.py` — loads both HF checkpoints in fp32, saves per-layer ΔW and earlier W to `results/public_pair_deepscaler/deltas/`.
2. `compute_svd.py` — top-k SVD (k=64) of earlier W for the selected layers.
3. `build_minibatch.py` — tokenizes 12 fixed math prompts, greedy-generates continuations from the **earlier** checkpoint, packs into 3 minibatches of size 2 × seq_len 256. Saved as `minibatch.pt` so every curvature probe sees identical input.
4. `directional_curvature.py` — for each layer, builds 4 direction classes:
   - `realized`     = ΔW
   - `principal`    = U_k Uᵀ_k ΔW V_k Vᵀ_k   (rank-k subspace projection)
   - `nonprincipal` = ΔW − principal
   - `random_seed_{0,1,2}` = matched-Frobenius-norm Gaussian
   Computes `vᵀHv / ‖v‖²` via Pearlmutter HVP (autograd) on each of 3 minibatches, with the model in fp32 and **eager attention** (the SDPA backward path raises `derivative for ..._attention_backward is not implemented` under double-backward).
5. `summarize_and_plot.py` — aggregates `summary.json` and produces `plots/directional_curvature_bars.png`.

## Key implementation choices
- **Curvature objective** is token-level cross-entropy NLL on the fixed minibatch (pad-masked, label-shifted). This is a **proxy** for the GRPO objective at the checkpoint, *not* the true GRPO Hessian. Recorded explicitly in `summary.json["objective_proxy_note"]`. Rule 3 in CLAUDE.md flags this as a known gap of an endpoint-only public-pair study.
- **Layer-restricted direction**: `requires_grad` is enabled only on the target layer's `.weight`, all other params frozen. So `vᵀHv` measures the diagonal block `H_{WW}` of the Hessian projected onto v — exactly the layer-restricted directional curvature.
- **Principal subspace** = top-k left/right singular subspaces (Wedin/sin-Θ definition).  This is *different* from the paper's "principal mask = top-α magnitude entries of the rank-k reconstruction"; both are valid but distinct projection operators. CLAUDE.md "Explicit definitions" calls for the subspace form, which is what we implement.
- **Random direction** is matched in Frobenius norm to ΔW; the curvature reported is normalized (`vᵀHv / ‖v‖²`), so the matching only affects numerical scale, not the comparison.

## Directional curvature  vᵀHv / ‖v‖²  (mean over 3 minibatches, ranking from `summary.json`)

| Layer | realized | principal | nonprincipal | random | P − NP |
|---|---|---|---|---|---|
| `model.layers.13.self_attn.q_proj` | +5.97e-4 | +9.66e-5 | +5.91e-4 | +1.60e-5 | **−4.95e-4** |
| `model.layers.13.self_attn.o_proj` | +5.25e-4 | +1.57e-5 | +5.59e-4 | +4.67e-5 | **−5.43e-4** |
| `model.layers.13.mlp.down_proj`    | +1.95e-4 | +3.75e-4 | +1.98e-4 | −2.5e-7  | **+1.77e-4** |

`||dW||_F` rel-Frobenius (||dW||_F / ||W||_F) ≈ 1.0 × 10⁻³ on all three layers — consistent with the paper's "small RL endpoint drift". The principal subspace captures only ~9 % (q_proj), 5.5 % (o_proj), 4.5 % (down_proj) of the delta's energy — the realized RL update is dominantly off-principal, which already echoes Gate II at the energy-distribution level.

## Acceptance-criterion check (CLAUDE.md "Concrete acceptance criteria")
1. **Runs on one layer without crashing** — runs on all three layers; one fix needed (`attn_implementation="eager"` for HVP).
2. **Stable curvature rankings across repeated probes** — for **q_proj** and **o_proj** the ranking `nonprincipal ≈ realized > principal > random` holds on every minibatch. For **mlp.down_proj** the principal projection's curvature varies in sign across minibatches (one negative), so its relative ranking is NOT stable. Per-direction relative std ≈ 90 % for q_proj realized/nonprincipal — high in absolute terms but the *ranking* is preserved.
3. **Principal vs. non-principal differ meaningfully** — yes, by ~6× on q_proj and ~36× on o_proj.

## What this does NOT yet show
- This is *NLL* curvature, not GRPO curvature. The claim "RL avoids high-curvature directions of the RL objective" cannot be tested without the actual reward model / advantages from the GRPO pipeline.
- Endpoint-only: we see geometry of the *final* delta, not the trajectory. CLAUDE.md rule 7 forbids over-claiming a training-dynamics conclusion from this alone.
- The principal subspace projection has very small ‖v‖²; the resulting `vᵀHv` is a small numerator and a small denominator. mlp.down_proj shows this signal is at the edge of estimator noise on this minibatch size — flagged for follow-up.

## Provisional reading (informal, for next-step planning)
- For the two attention layers, the principal-subspace projection of ΔW lies in *lower* curvature directions than its complement, **opposite** to the strongest reading of Gate II ("principal ⇒ high curvature"). This is consistent with Hypotheses 2 and 3 in CLAUDE.md (proxy validity / Gate II refinement).
- The MLP down-projection shows the opposite ordering but with high estimator noise — needs more minibatches before drawing any conclusion.

## Next obvious de-risking steps (suggested, not yet done)
- ~~Re-run with the **paper's principal-mask definition** (top-α |W^(k)_ij| as a coordinate mask) alongside the subspace projection, so we can directly compare the two proxies.~~ Done — see Experiment A below.
- Increase the number of minibatches (e.g. 8–16) to tighten error bars on the down_proj layer.
- Apply Hypothesis 4 controls: factor magnitude out (`Mlow ∩ M_princ`-style) before drawing geometric conclusions.
- Repeat on the `Qwen2.5-Math-1.5B → DeepSeek-R1-Distill-Qwen-1.5B` pair (Phase 0B).

---

# Experiment A — k-sweep + coordinate-mask comparison

## Question
Does the curvature inversion observed at k=64 with SVD-subspace projection survive
(a) varying k ∈ {16, 64, 256, 512}, and (b) switching from subspace projection to
the source paper's coordinate-mask principal operator
(`M_princ = Top_α(|W₀^(k)(i,j)|)`)?

## Setup
- Same earlier checkpoint, minibatch, NLL proxy loss, and target layers as Phase 0A.
- Cached ΔW (Phase 0A) is reused; only the direction set changes.
- Operators: (1) **subspace** projection `P_U dW P_V` at k ∈ {16,64,256,512};
  (2) **coord_mask** projection `M ⊙ dW` where `M = Top_α(|W₀^(k)|)` for the
  same k and α ∈ {0.05, 0.30, 0.50}.
- Per layer: 36 direction tensors × 3 minibatches = 108 HVPs.
- Driver: `scripts/run_experiment_a.sh`.

## Headline (sign of principal − complement mean curvature; red = principal sharper, the paper claim)

`results/public_pair_deepscaler/experiment_a/plots/sign_flip_grid.png`

| Layer | Operator/α \ k | 16 | 64 | 256 | 512 |
|---|---|---|---|---|---|
| **q_proj** | subspace          | inverted | inverted | inverted | inverted |
| q_proj     | coord α=0.05      | inverted | inverted | inverted | inverted |
| q_proj     | coord α=0.30      | inverted | inverted | inverted | inverted |
| q_proj     | coord α=0.50      | inverted | inverted | inverted | inverted |
| **o_proj** | subspace          | inverted | inverted | inverted | inverted |
| o_proj     | coord α=0.05      | inverted | inverted | inverted | inverted |
| o_proj     | coord α=0.30      | **paper** | inverted | inverted | inverted |
| o_proj     | coord α=0.50      | **paper** | **paper** | inverted | inverted |
| **mlp.down_proj** | subspace   | **paper** | **paper** | inverted | inverted |
| mlp.down_proj  | coord α=0.05  | inverted | inverted | inverted | inverted |
| mlp.down_proj  | coord α=0.30  | **paper** | inverted | inverted | inverted |
| mlp.down_proj  | coord α=0.50  | **paper** | inverted | ≈0 | ≈0 |

## Interpretation (which of the three user-named branches actually fired)

The three pre-registered branches were:
1. **Inversion flips under coord-mask** → "paper right but only for their operator"
2. **Inversion holds under both, all k** → "robust structural finding"
3. **k-sensitive** → "fragile, saves us from publishing brittle result"

All three fired — **layer-dependent**:

- **q_proj — Branch 2 (robust).** 16/16 (operator × k × α) cells show the inversion (principal < complement curvature). Per-minibatch agreement is also high (0/3 or 1/3 minibatches show paper ordering, i.e. 2/3 or 3/3 agree with inversion). On q_proj the curvature inversion is genuinely robust to operator choice and rank choice.

- **o_proj — Branch 1 (operator flip).** Subspace projection inverts at every k.
  Coord-mask at α=0.50 with small k (16, 64) recovers the paper's ordering
  (principal sharper than complement), then flips back to inverted at k=256,512.
  Coord-mask at α=0.30 only flips at k=16. So on o_proj, the paper's reading
  is operator-and-rank-specific: it holds in a thin slice near `(k≈16, α≈0.5)`
  and fails everywhere else, including the subspace operator.

- **mlp.down_proj — Branch 3 (k-sensitive).** This is the layer that supposedly
  *confirmed* the paper at k=64 in Phase 0A. The k-sweep undermines that: subspace
  projection gives the paper's ordering at k=16 (+6.7e-4 P-C) and k=64 (+1.8e-4
  P-C), but **flips to inverted** at k=256 (-2.1e-4) and k=512 (-8.5e-5).
  Coord-mask α=0.05 inverts at every k. α=0.30 supports paper only at k=16.
  α=0.50 is essentially noise at k=256,512 (|P-C| ≈ 1e-6 << per-minibatch std).
  The Phase 0A k=64 confirmation of Gate II on this layer does not generalize.

## Methodological implications

- **The Phase 0A headline finding ("RL update lies in lower-curvature directions
  than principal subspaces, for q/o_proj") survives the k-sweep + operator
  cross-check on attention layers q_proj and o_proj — for q_proj completely,
  for o_proj over all combinations except a narrow `(k≤64, α≥0.30)` slice of
  the coord-mask operator.**
- **The Phase 0A k=64 result on mlp.down_proj that "matched the paper" is not
  k-robust; that conclusion should be retracted from the running narrative
  until estimator noise is reduced (more minibatches / larger batch / multiple
  random projections).**
- Numerical sanity: at k=64 with subspace operator, every direction's mean
  curvature reproduces the Phase 0A value to 4+ sig figs (e.g. q_proj sub_princ
  +9.665e-5 vs Phase 0A +9.665e-5). The two pipelines are computing the same
  HVP on the same minibatch.
- Per-minibatch variability is high: relative std ≈ 80–100 % on principal
  directions. With 3 minibatches alone, several `princ_minus_comp` values are
  within ±1σ of zero and should be treated as undetermined, not as flips.
  Concretely the mlp.down_proj coord α=0.50 cells at k=256,512 (|P-C| ≈ 1e-6)
  are noise, not signal.

## Open questions Experiment A surfaced
- The k=16 + α=0.50 corner where the paper's ordering holds on o_proj and
  mlp.down_proj corresponds to a coord-mask whose density is 50 % but whose
  *mask entries* are dominated by `|W_k|`'s largest values when k is small
  (i.e. close to the top singular component alone). This is intuitively the
  "most concentrated principal weights" condition. The fact that *only* this
  corner agrees with the paper is consistent with Hypothesis 4 (magnitude /
  precision confound): the mask there picks up high-magnitude entries of
  `W_0`, which are also where the model is willing to take larger absolute
  updates. Worth disentangling.
- The flips on mlp.down_proj across k for the *subspace* operator are
  particularly informative — they say the principal-subspace projection of
  ΔW does not have a stable curvature signature on this MLP layer, even
  before any operator choice. The paper's "principal weights have higher
  curvature" picture cannot be cleanly tested on this layer without first
  resolving this k-instability.

## Correction: noise floor + statistical undeterminability (2026-05-21)

Adding the heuristic noise-floor flag `|P − C| < 2·σ_paired(per_mb_pmc)` to the
sign-flip plot (where σ_paired is the std across minibatches of the per-mb
*difference* P_mb − C_mb) collapses most of the colored cells in the table
above to "undetermined". With n=3 minibatches:

- 48 cells total
- 0 cells are red (P>C) and survive the 2σ test
- 6 cells are blue (P<C) and survive: o_proj subspace at k=256 (z=−2.7) and
  k=512 (z=−5.3); mlp.down_proj coord α=0.05 at k=64 (z=−4.5) and k=512
  (z=−4.8); mlp.down_proj coord α=0.30 at k=256 (z=−3.5) and k=512 (z=−5.7).
- 42 cells are within 2σ of zero and therefore undetermined at this minibatch
  count.

The strict small-n t-rule (`|P − C| < t_{0.975,n−1} · σ_paired/√n`) agrees:
42 undetermined under both rules.

Implications for the layer-by-layer story we wrote above:

- **q_proj** ostensibly had a "robust inversion" because every cell's mean
  was negative and per-mb signs agreed. But every q_proj cell has z ∈
  (−1.5, −0.8), so the apparent uniformity is *not* statistically resolvable
  with 3 minibatches. The qualitative direction is consistent with inversion
  but the magnitude can't be ranked against zero yet.
- **o_proj subspace** at k=256 and k=512 is the cleanest single signal in
  the experiment: large z scores and per-mb sign 0/3 — robustly inverted.
  o_proj subspace at k=16 and k=64 remains undetermined.
- **mlp.down_proj**'s previously highlighted "Phase 0A confirms paper at
  k=64" result is in the noise zone (z=+0.2). The flip pattern across k for
  the subspace operator is also entirely in the noise zone (every z ∈
  (−0.5, +1.1)). The only signal that survives on this layer is the
  coordinate-mask operator at low α (0.05 and 0.30) at the larger ks (256,
  512), all of which point to inversion.

**Net headline after the noise-floor correction:** the only statistically
defensible Experiment A result with the current minibatch count is the
*inversion* on o_proj at large k (subspace) and on mlp.down_proj at large k
(coord-mask, low α). No cell shows the paper's ordering above noise. The
prior qualitative summary that q_proj's inversion is "robust" needs to be
softened to "directionally consistent but currently statistically
indistinguishable from zero at n=3 minibatches". Increasing to 8–16
minibatches is now load-bearing before any follow-on experiment, not optional.

Plot: `results/public_pair_deepscaler/experiment_a/plots/sign_flip_grid.png`
(red/blue cells = decisive; light-gray cells with `[u]` label = undetermined).
Per-row noise-floor numbers are persisted to `summary.json` under
`rows[].pmc_std`, `rows[].pmc_snr`, `rows[].noise_floor_2sigma`,
`rows[].is_undetermined_2sigma`, `rows[].is_undetermined_strict_t`.

## Engineering: shared HVP helpers + one-load model loader (2026-05-21)

The Phase 0A driver and Experiment A driver previously each had their own
model-load code path (Phase 0A reloaded the 1.5B fp32 weights once per layer
inside `run_for_layer`; Experiment A loaded once at the top). Both now share
`scripts/_hvp.py` which exposes `load_model_for_hvp`, `freeze_all_but`,
`loss_fn`, `directional_curvature`, `matched_norm_random`. The shared loader
sets fp32 + eager attention + `use_cache=False` + gradient-checkpointing-off,
all of which are needed for the Pearlmutter HVP path. Verified by re-running
both pipelines and bit-exactly reproducing every per-minibatch curvature
value from the prior summaries. This matters because the next planned sweep
(Experiment C, depth × matrix-type scan) is on the order of 35 layers; the
previous per-layer reload pattern would have wasted ~20 minutes of I/O for
no analytical benefit.
