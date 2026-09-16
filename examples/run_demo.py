"""Demo script that runs the benchmark harness and visualises the loss history.

The script invokes ``harness.runner.main`` to produce ``results/comparison.md``
and then plots the loss curves from each implementation side‑by‑side.
"""

import matplotlib.pyplot as plt
from implementations.naive import NaiveMomentumGD
from implementations.optimized import OptimizedMomentumGD
import numpy as np

# Same quadratic as in the harness
A = np.array([[3.0, 0.5], [0.5, 1.0]])
b = np.array([1.0, 2.0])

def func(x):
    return 0.5 * x @ A @ x - b @ x

def grad(x):
    return A @ x - b

naive = NaiveMomentumGD(lr=0.1, momentum=0.8, max_iter=300)
opt = OptimizedMomentumGD(lr=0.1, momentum=0.8, max_iter=300)

_, _ = naive.minimize(func, grad, np.zeros(2))
_, _ = opt.minimize(func, grad, np.zeros(2))

plt.figure(figsize=(8,4))
plt.plot(naive.history, label="Naive")
plt.plot(opt.history, label="Optimized")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss curve comparison")
plt.legend()
plt.tight_layout()

out_path = "results/demo_loss.png"
plt.savefig(out_path)
print(f"Demo plot saved to {out_path}")
