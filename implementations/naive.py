"""Naive implementation of gradient descent with momentum.

The algorithm updates parameters according to:
    v = mu * v - lr * grad
    theta = theta + v
where mu is the momentum coefficient and lr the learning rate.
"""

import numpy as np
from typing import Callable, Iterable, Tuple

class NaiveMomentumGD:
    """Simple, clear implementation of momentum‑based gradient descent.

    Parameters
    ----------
    lr: float
        Learning rate.
    momentum: float
        Momentum coefficient (mu), typically in [0, 1).
    max_iter: int
        Maximum number of iterations.
    tol: float
        Tolerance for the norm of the gradient for early stopping.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, max_iter: int = 1000, tol: float = 1e-6):
        self.lr = lr
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.history: list[float] = []

    def minimize(self, func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> Tuple[np.ndarray, int]:
        """Run the optimizer.

        Returns
        -------
        x_opt: np.ndarray
            Final parameter vector.
        n_iter: int
            Number of iterations performed.
        """
        x = x0.astype(float).copy()
        v = np.zeros_like(x)
        for i in range(self.max_iter):
            g = grad(x)
            self.history.append(func(x))
            if np.linalg.norm(g) <= self.tol:
                break
            v = self.momentum * v - self.lr * g
            x = x + v
        return x, i + 1
