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
#
# Stability contract:
#   - FIXED_PROMPTS[:6]   are the original Phase-0A prompts (bs=2, nb=3).
#                         Do not reorder or modify these — every downstream
#                         result file ties back to this slice via the saved
#                         minibatch.
#   - FIXED_PROMPTS[6:12] are the next six original prompts (unused by Phase 0A
#                         but kept for backwards reproducibility).
#   - FIXED_PROMPTS[12:]  are extensions added 2026-05-21 for the n=16
#                         Experiment-A re-run (needs bs=2, nb=16 -> 32 prompts).
#                         New prompts go here, not inserted upstream.
FIXED_PROMPTS: list[str] = [
    # ----- original Phase 0A prompts (do not reorder) -----
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
    # ----- additions for the n=16 (32-prompt) Experiment A run -----
    "Problem: Find the smallest positive integer n such that n^2 > 1000.\nSolution:",
    "Problem: Evaluate the limit as x -> 0 of sin(3x)/x.\nSolution:",
    "Problem: A bag contains 5 red and 7 blue marbles. Two are drawn without replacement. What is the probability that both are red?\nSolution:",
    "Problem: Solve the system x + y = 7, 2x - y = 5.\nSolution:",
    "Problem: How many ways are there to choose 3 books from a shelf of 10 distinct books?\nSolution:",
    "Problem: Find the area of a regular hexagon with side length 2.\nSolution:",
    "Problem: What is the value of the infinite geometric sum 1 + 1/3 + 1/9 + 1/27 + ...?\nSolution:",
    "Problem: Compute the greatest common divisor of 252 and 198.\nSolution:",
    "Problem: For what value of k does the equation x^2 + kx + 9 = 0 have a double root?\nSolution:",
    "Problem: Find the equation of the tangent line to y = x^3 at the point (1, 1).\nSolution:",
    "Problem: Let z = 3 + 4i. What is |z|, the modulus of z?\nSolution:",
    "Problem: How many degrees does the hour hand of a clock turn through in 25 minutes?\nSolution:",
    "Problem: Evaluate (2 + 3i)(1 - i).\nSolution:",
    "Problem: A sequence is defined by a_1 = 2 and a_{n+1} = 3 a_n + 1. Find a_4.\nSolution:",
    "Problem: Find all integer solutions to 3x + 5y = 23 with x, y >= 0.\nSolution:",
    "Problem: What is the coefficient of x^3 in the expansion of (1 + x)^7?\nSolution:",
    "Problem: A line passes through (1, 2) and (4, 11). What is its slope?\nSolution:",
    "Problem: Find the sum of the first 20 odd positive integers.\nSolution:",
    "Problem: For which positive integers n is n^2 + 1 divisible by 5?\nSolution:",
    "Problem: Evaluate the definite integral of cos(x) from 0 to pi/2.\nSolution:",
]
assert len(FIXED_PROMPTS) >= 32, (
    f"FIXED_PROMPTS shrank to {len(FIXED_PROMPTS)}; the n=16 Experiment A "
    f"needs at least 32 prompts (bs=2 * nb=16)."
)
