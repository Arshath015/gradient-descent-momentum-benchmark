"""Optimized implementation of gradient descent with momentum.

The optimizer uses vectorised NumPy operations and pre‑allocates history storage.
"""

import numpy as np
from typing import Callable, Tuple

class OptimizedMomentumGD:
    """Performance‑oriented momentum GD.

    Parameters are identical to :class:`NaiveMomentumGD` but the internal loop
    avoids Python‑level list appends and performs a single NumPy allocation for
    the history array.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, max_iter: int = 1000, tol: float = 1e-6):
        self.lr = lr
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.history: np.ndarray = np.empty(max_iter, dtype=float)

    def minimize(self, func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> Tuple[np.ndarray, int]:
        x = x0.astype(float).copy()
        v = np.zeros_like(x)
        for i in range(self.max_iter):
            g = grad(x)
            self.history[i] = func(x)
            if np.linalg.norm(g) <= self.tol:
                self.history = self.history[:i+1]
                break
            v = self.momentum * v - self.lr * g
            x = x + v
        else:
            self.history = self.history[:self.max_iter]
        return x, i + 1
