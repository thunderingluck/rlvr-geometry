# Decisions log

## 2026-04-26  Phase 0A scaffold

- **Filesystem layout**: `/orcd/home/002/evag/code/rlvr-geometry` is at the user's home quota hard limit (201 GB / 200 GB), so `mkdir` failed there. Resolution: real project tree lives on `/orcd/scratch/orcd/014/evag/rlvr-geometry/`, with `configs/`, `scripts/`, `notebooks/`, `docs/`, `results/` symlinked back into the home path. The venv (`/orcd/scratch/orcd/014/evag/envs/rlvr`) and HF cache (`/orcd/scratch/orcd/014/evag/hf_cache`) also live on scratch. `scripts/env.sh` sets the relevant env vars.
- **Env**: `python -m venv` (not conda — conda's solver is broken on this node). Python 3.11.14, torch 2.5.1+cu121, transformers 4.46.3, accelerate 1.2.1.
- **Curvature objective**: token-level cross-entropy NLL on a fixed minibatch as a proxy for the GRPO objective. Reason: we don't have the GRPO reward / rollout pipeline for a public endpoint pair. Flagged in `summary.json` and `docs/notes.md` as a known proxy gap (CLAUDE.md rule 3).
- **Principal subspace definition**: rank-k singular subspace projection (Wedin's sin-Θ definition), `U_k Uᵀ_k ΔW V_k Vᵀ_k`. CLAUDE.md "Explicit definitions" picks this one. Note that the paper's "principal mask" is a coordinate mask of top-α |W^(k)_{ij}|; same name, different operator. Worth implementing the paper's mask form too as a follow-up.
- **HVP via autograd**: Pearlmutter trick. Required `attn_implementation="eager"` because the SDPA backward path (mem-efficient and flash) does not implement double-backward. Eager attention is slower but is the only path that works for HVP under HF's Qwen2 implementation as of transformers 4.46.3.
- **Layer subset**: `model.layers.13.{self_attn.q_proj, self_attn.o_proj, mlp.down_proj}.weight` per CLAUDE.md "Default layer subset for pilot analysis"; layer index 13 borrowed from the paper's exemplar in Fig. 2.
- **k = 64** for the SVD subspace. Captures ~30 % (q_proj/o_proj) and ~14 % (down_proj) of the spectral energy. Arbitrary; revisit if the principal projection is too tiny to give meaningful curvature signal.
- **Number of probes**: 3 minibatches × 3 random seeds. Enough to demonstrate ranking stability for q/o_proj; insufficient to nail down sign of curvature on principal projection of mlp.down_proj. Increase before running Phase 0B.
