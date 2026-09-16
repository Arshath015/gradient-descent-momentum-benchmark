import numpy as np
from implementations.naive import NaiveMomentumGD

def quadratic(x):
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    return 0.5 * x @ A @ x - b @ x

def grad_quadratic(x):
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    return A @ x - b

def test_converges_to_zero():
    opt = NaiveMomentumGD(lr=0.2, momentum=0.5, max_iter=200, tol=1e-8)
    x_opt, it = opt.minimize(quadratic, grad_quadratic, np.array([5.0, -3.0]))
    assert np.allclose(x_opt, np.zeros(2), atol=1e-4)
    assert it < 200

def test_edge_zero_gradient():
    opt = NaiveMomentumGD(lr=0.1, momentum=0.9, max_iter=10)
    x0 = np.zeros(2)
    x_opt, it = opt.minimize(quadratic, grad_quadratic, x0)
    assert np.array_equal(x_opt, x0)
    assert it == 1
