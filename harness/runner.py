"""Benchmark harness for comparing gradient‑descent variants.

The script loads a simple quadratic test function, runs each implementation,
records runtime and final loss, and writes a markdown summary to
``results/comparison.md``.
"""

import time
import numpy as np
from pathlib import Path
from implementations.naive import NaiveMomentumGD
from implementations.optimized import OptimizedMomentumGD

# Quadratic test problem: f(x) = 0.5 * x^T A x - b^T x
A = np.array([[3.0, 0.5], [0.5, 1.0]])
b = np.array([1.0, 2.0])


def func(x: np.ndarray) -> float:
    return 0.5 * x @ A @ x - b @ x


def grad(x: np.ndarray) -> np.ndarray:
    return A @ x - b


def run_variant(optimizer, name: str) -> dict:
    start = time.perf_counter()
    x_opt, n_iter = optimizer.minimize(func, grad, np.zeros(2))
    elapsed = time.perf_counter() - start
    final_loss = func(x_opt)
    return {
        "name": name,
        "iterations": n_iter,
        "time_sec": elapsed,
        "final_loss": final_loss,
        "solution": x_opt.tolist(),
    }


def main() -> None:
    results = []
    results.append(run_variant(NaiveMomentumGD(lr=0.1, momentum=0.8, max_iter=500), "Naive"))
    results.append(run_variant(OptimizedMomentumGD(lr=0.1, momentum=0.8, max_iter=500), "Optimized"))

    out_path = Path(__file__).parent.parent / "results" / "comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Gradient Descent with Momentum Benchmark\n\n")
        f.write("| Implementation | Iterations | Time (s) | Final Loss | Solution (x) |\n")
        f.write("|----------------|------------|----------|------------|--------------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['iterations']} | {r['time_sec']:.6f} | {r['final_loss']:.6e} | {r['solution']} |\n")

if __name__ == "__main__":
    main()
