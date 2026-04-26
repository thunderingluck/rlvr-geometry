#!/usr/bin/env bash
# Source this file before running any harness scripts:
#   source scripts/env.sh
#
# Reason: home filesystem is at quota, so the venv, HF cache, and tmp all live
# on /orcd/scratch/orcd/014/evag. Project root stays on /orcd/home with subdirs
# symlinked back to scratch.

SCRATCH=/orcd/scratch/orcd/014/evag
export RLVR_VENV="$SCRATCH/envs/rlvr"
export PATH="$RLVR_VENV/bin:$PATH"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"

# Project root (real path, not the symlinked subdirs).
export RLVR_ROOT=/orcd/home/002/evag/code/rlvr-geometry
export RLVR_RESULTS="$RLVR_ROOT/results"

mkdir -p "$HF_HUB_CACHE" "$TMPDIR" "$RLVR_RESULTS"
