"""Shared helpers for the public-pair analysis harness."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def project_root() -> Path:
    return Path(os.environ.get("RLVR_ROOT", "/orcd/home/002/evag/code/rlvr-geometry"))


def results_root(pair_name: str) -> Path:
    base = Path(os.environ.get("RLVR_RESULTS", str(project_root() / "results")))
    out = base / pair_name
    (out / "deltas").mkdir(parents=True, exist_ok=True)
    (out / "svd").mkdir(parents=True, exist_ok=True)
    (out / "curvature").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hardcoded fixed math prompts. Kept in-repo so we don't depend on a dataset
# download and so the minibatch is exactly reproducible. These are paraphrased
# textbook-style problems chosen to lie in the math-reasoning distribution
# both checkpoints were trained on.
FIXED_PROMPTS: list[str] = [
    "Problem: Find all real solutions to the equation x^2 - 5x + 6 = 0.\nSolution:",
    "Problem: Compute the sum 1 + 2 + 3 + ... + 100.\nSolution:",
    "Problem: A right triangle has legs of length 3 and 4. What is the length of the hypotenuse?\nSolution:",
    "Problem: How many positive integers less than 100 are divisible by both 4 and 6?\nSolution:",
    "Problem: Evaluate the integral of x^2 from 0 to 3.\nSolution:",
    "Problem: If f(x) = 2x + 1, what is f(f(3))?\nSolution:",
    "Problem: Solve for x: log_2(x) + log_2(x-2) = 3.\nSolution:",
    "Problem: What is the remainder when 7^100 is divided by 5?\nSolution:",
    "Problem: A circle has area 25*pi. What is its circumference?\nSolution:",
    "Problem: How many distinct ways can the letters of MISSISSIPPI be arranged?\nSolution:",
    "Problem: Find the derivative of g(x) = x*sin(x).\nSolution:",
    "Problem: Two dice are rolled. What is the probability that the sum is exactly 7?\nSolution:",
]
