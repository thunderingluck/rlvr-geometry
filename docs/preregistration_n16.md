# Pre-registration: Experiment A at n=16 minibatches

**Written 2026-05-21, before any n=16 HVPs have run.**

This document fixes the predictions for the n=16 run *before* we observe the
result, so we can distinguish "the n=3 directional signal was real and just
needed more samples" from "I'm narrating whatever the n=16 grid happens to
show."

## Setup
- Identical to the published n=3 Experiment A run except `num_minibatches = 16`,
  with the first 3 of those 16 minibatches *exactly* equal to the n=3 batch
  (same `FIXED_PROMPTS[:6]`, same continuations) — so the comparison isolates
  variance reduction from any change in the prompt distribution.
- Same earlier checkpoint, same target layers
  (q_proj/o_proj/mlp.down_proj at layer 13), same NLL proxy loss, same operator
  set (subspace projection + coord_mask at α ∈ {0.05, 0.30, 0.50}), same
  k_sweep ∈ {16, 64, 256, 512}.
- Same noise-floor rule: a cell is "undetermined" iff
  `|princ - comp mean| < 2 · σ_paired(per-mb P − C)`.

## Scaling assumption
Standard-error-of-the-mean scales as 1/√n. If σ_paired is unchanged, then
`z(n=16) ≈ z(n=3) · √(16/3) ≈ 2.31 · z(n=3)`. Predictions below apply that
naive scaling to each cell's observed n=3 z-score.

**Caveat I expect to surface:** σ_paired at n=3 is itself an unstable estimate
(df = 2). If the true σ is *larger* than the n=3 estimate, observed z(n=16) will
come in *below* prediction across the board. Conversely if σ at n=3 was an
overestimate, observed z(n=16) overshoots prediction. Direction-of-flip
predictions should be robust to this; magnitude predictions are not.

## The headline question
Does q_proj — which at n=3 showed every cell directionally inverted but
*no* cell statistically resolved — convert that directional consistency into
2σ-significant inversion at n=16?

Naive scaling says yes, with **high-k subspace cells flipping from undetermined
to significant first** because those already had the largest |z| at n=3.

## Predicted cell verdicts at n=16

Convention: `paper` = P>C above 2σ (matches the source paper's claim);
`inverted` = P<C above 2σ; `undet` = within 2σ of zero. The `predicted z`
column is `z(n=3) × √(16/3) = 2.31 × z(n=3)`. Threshold is |z_pred| > 2.

### q_proj  (n=3 verdict: every cell undetermined, all directionally inverted)

| operator/α | k | z(n=3) | z_pred(n=16) | verdict_pred(n=16) |
|---|---|---|---|---|
| subspace | 16  | −1.45 | −3.35 | **inverted** |
| subspace | 64  | −1.19 | −2.75 | **inverted** |
| subspace | 256 | −1.19 | −2.75 | **inverted** |
| subspace | 512 | −1.51 | −3.49 | **inverted** |
| coord α=0.05 | 16  | −1.24 | −2.87 | **inverted** |
| coord α=0.05 | 64  | −1.09 | −2.52 | **inverted** |
| coord α=0.05 | 256 | −1.20 | −2.78 | **inverted** |
| coord α=0.05 | 512 | −1.22 | −2.82 | **inverted** |
| coord α=0.30 | 16  | −0.89 | −2.06 | inverted (borderline) |
| coord α=0.30 | 64  | −1.01 | −2.33 | **inverted** |
| coord α=0.30 | 256 | −1.13 | −2.61 | **inverted** |
| coord α=0.30 | 512 | −1.27 | −2.94 | **inverted** |
| coord α=0.50 | 16  | −0.83 | −1.92 | undet (borderline) |
| coord α=0.50 | 64  | −1.31 | −3.03 | **inverted** |
| coord α=0.50 | 256 | −1.22 | −2.82 | **inverted** |
| coord α=0.50 | 512 | −1.26 | −2.91 | **inverted** |

Summary q_proj prediction: 14/16 cells inverted-significant, 2/16 borderline
(coord α=0.30 k=16 expected to clear, coord α=0.50 k=16 expected to stay
undetermined). Zero "paper" cells predicted.

### o_proj  (n=3 verdict: 2/16 inverted-significant, rest undetermined)

| operator/α | k | z(n=3) | z_pred(n=16) | verdict_pred(n=16) |
|---|---|---|---|---|
| subspace | 16  | −1.06 | −2.45 | **inverted** |
| subspace | 64  | −1.23 | −2.84 | **inverted** |
| subspace | 256 | −2.69 | −6.22 | **inverted (very strong)** |
| subspace | 512 | −5.27 | −12.2 | **inverted (very strong)** |
| coord α=0.05 | 16  | −1.36 | −3.14 | **inverted** |
| coord α=0.05 | 64  | −0.82 | −1.89 | undet (borderline) |
| coord α=0.05 | 256 | −0.95 | −2.20 | **inverted** |
| coord α=0.05 | 512 | −0.86 | −1.99 | undet (borderline) |
| coord α=0.30 | 16  | +0.22 | +0.51 | undet |
| coord α=0.30 | 64  | −0.76 | −1.76 | undet |
| coord α=0.30 | 256 | −0.27 | −0.62 | undet |
| coord α=0.30 | 512 | −0.86 | −1.99 | undet (borderline) |
| coord α=0.50 | 16  | +0.52 | +1.20 | undet |
| coord α=0.50 | 64  | +0.73 | +1.69 | undet |
| coord α=0.50 | 256 | −0.19 | −0.44 | undet |
| coord α=0.50 | 512 | −1.07 | −2.47 | **inverted** |

Summary o_proj prediction: 7/16 inverted-significant, 9/16 undetermined.

### mlp.down_proj  (n=3 verdict: 4/16 inverted-significant, all in coord_mask)

| operator/α | k | z(n=3) | z_pred(n=16) | verdict_pred(n=16) |
|---|---|---|---|---|
| subspace | 16  | +1.10 | +2.54 | **paper** |
| subspace | 64  | +0.21 | +0.49 | undet |
| subspace | 256 | −0.47 | −1.09 | undet |
| subspace | 512 | −0.36 | −0.83 | undet |
| coord α=0.05 | 16  | −1.35 | −3.12 | **inverted** |
| coord α=0.05 | 64  | −4.49 | −10.4 | **inverted (very strong)** |
| coord α=0.05 | 256 | −1.76 | −4.07 | **inverted** |
| coord α=0.05 | 512 | −4.80 | −11.1 | **inverted (very strong)** |
| coord α=0.30 | 16  | +0.59 | +1.36 | undet |
| coord α=0.30 | 64  | −0.90 | −2.08 | inverted (borderline) |
| coord α=0.30 | 256 | −3.48 | −8.04 | **inverted (very strong)** |
| coord α=0.30 | 512 | −5.68 | −13.1 | **inverted (very strong)** |
| coord α=0.50 | 16  | +1.58 | +3.65 | **paper** |
| coord α=0.50 | 64  | −0.48 | −1.11 | undet |
| coord α=0.50 | 256 | −0.03 | −0.07 | undet |
| coord α=0.50 | 512 | +0.01 | +0.02 | undet |

Summary mlp.down_proj prediction:
- 2 cells are predicted to come in as **paper-matching** (P>C):
  subspace k=16 (z_pred=+2.5) and coord_mask α=0.50 k=16 (z_pred=+3.7).
- 6 cells predicted as inverted-significant (the coord_mask α=0.05/0.30 high-k cells).
- The rest undetermined.
- This makes mlp.down_proj the most operator-and-k-dependent layer: small-k
  with full-magnitude masks (subspace k=16, coord_mask α=0.50 k=16) shows
  paper ordering; strict principal masks (α=0.05 / α=0.30 at large k) show
  inversion. The Phase 0A "k=64 confirms paper" cell stays undetermined,
  vindicating the noise-floor correction.

## Aggregate counts predicted at n=16 (across all 48 cells)

| verdict (predicted) | count |
|---|---|
| inverted (significant) | 27 + 6 borderline-leaning-inverted = up to 33 |
| paper-matching (significant) | 2 |
| undetermined (within 2σ) | 13 ± 4 depending on borderlines |

Compare to the **observed n=3 outcome**: 6 inverted, 0 paper, 42 undetermined.

## Pre-registered diagnostic outcomes

After the n=16 run lands I will write up the comparison in
`docs/notes.md` under an "n=16 outcome" heading. The diagnostic categories I
will use:

1. **Story A — "data had signal, n=3 just didn't have power":** ≥ 80 % of the
   predicted-significant cells actually clear 2σ at n=16, and the q_proj layer
   moves from "all undetermined" to "majority inverted". This rehabilitates the
   Phase 0A reading on attention layers and is the cleanest outcome.

2. **Story B — "predictions overshot because n=3 σ underestimates true σ":**
   The signs go the right way (no flips in direction) but z values come in
   systematically below prediction (say, < 70 % of predicted |z|). Significant
   cells still emerge, just fewer than predicted. Reading: σ_paired at n=3 was
   an underestimate; the true measurement is noisier than n=3 implied. This
   would weaken any strong quantitative claim but keep the qualitative
   inversion finding on the attention layers.

3. **Story C — "predicted-significant cells refuse to clear":** Even
   high-|z(n=3)| cells stay undetermined at n=16. This would mean the n=3 z
   values were already inflated by lucky alignment of 3 noisy samples; the
   underlying P−C gap is even smaller than the n=3 mean. Reading: no
   per-cell inversion claim is defensible at this minibatch count; the
   methodological paper (rather than the structural-finding paper) is what we
   have.

4. **Story D — predictions flip in sign:** Cells with predicted negative z come
   in with positive z above 2σ, or vice versa. Reading: the n=3 mean was
   biased, not just noisy. This would mean the n=3 result was qualitatively
   wrong, not just statistically weak. Would require a deeper post-mortem
   (minibatch sampling, prompt distribution, etc.).

The decision rule:
- Story A → keep the q_proj inversion as a load-bearing claim; iterate on
  Experiment C (depth × matrix sweep) next.
- Story B → soften quantitative claims, keep qualitative inversion claim,
  add a "σ stability across n" subsection.
- Story C → reframe as a methodological paper centered on the noise-floor
  finding itself; do not claim per-layer geometry.
- Story D → halt and audit before any further sweeps.
