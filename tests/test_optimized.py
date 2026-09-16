import numpy as np
from implementations.optimized import OptimizedMomentumGD

def rosenbrock(x):
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    a = 1.0
    b = 100.0
    df_dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    df_dy = 2 * b * (x[1] - x[0]**2)
    return np.array([df_dx, df_dy])

def test_converges_close_to_minimum():
    opt = OptimizedMomentumGD(lr=0.001, momentum=0.9, max_iter=5000, tol=1e-6)
    x_opt, it = opt.minimize(rosenbrock, grad_rosenbrock, np.array([-1.2, 1.0]))
    assert np.allclose(x_opt, np.array([1.0, 1.0]), atol=1e-2)
    assert it < 5000

def test_history_length():
    opt = OptimizedMomentumGD(lr=0.01, momentum=0.5, max_iter=50)
    _, _ = opt.minimize(rosenbrock, grad_rosenbrock, np.array([0.0, 0.0]))
    assert len(opt.history) <= 50
