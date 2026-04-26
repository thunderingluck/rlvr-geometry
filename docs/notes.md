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
- Re-run with the **paper's principal-mask definition** (top-α |W^(k)_ij| as a coordinate mask) alongside the subspace projection, so we can directly compare the two proxies.
- Increase the number of minibatches (e.g. 8–16) to tighten error bars on the down_proj layer.
- Apply Hypothesis 4 controls: factor magnitude out (`Mlow ∩ M_princ`-style) before drawing geometric conclusions.
- Repeat on the `Qwen2.5-Math-1.5B → DeepSeek-R1-Distill-Qwen-1.5B` pair (Phase 0B).
