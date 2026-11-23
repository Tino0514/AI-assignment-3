"""Sanity tests for the gamma-EigenGame implementation (run: python -m pytest tests)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gamma_eigengame import (gamma_eigengame, gamma_pseudo_gradient, identity_operator,
                             init_players, matrix_operator, rayleigh_quotients)
from metrics import exact_sgep, longest_streak, sqrtm_psd, subspace_error, whiten_truth


def random_spd(d, rng, cond=10.0):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * np.linspace(1.0, cond, d)) @ Q.T / cond


def test_reduces_to_eigengame_when_B_is_identity():
    """Proposition 3: for B = I the update is (I - v v^T)[A v - sum_j <v,A v_j> v_j]."""
    rng = np.random.default_rng(0)
    d, k = 20, 5
    A = random_spd(d, rng)
    V = init_players(d, k, rng)
    G = gamma_pseudo_gradient(V, A @ V, V)
    for i in range(k):
        v = V[:, i]
        target = A @ v - sum((v @ A @ V[:, j]) * V[:, j] for j in range(i))
        target = target - (v @ target) * v
        assert np.allclose(G[:, i], target, atol=1e-10)


def test_deterministic_sgep_matches_scipy():
    rng = np.random.default_rng(1)
    d, k = 30, 5
    X = rng.standard_normal((d, d))
    A = X @ np.diag(np.linspace(0.1, 3.0, d)) @ X.T / d
    B = random_spd(d, rng, cond=5.0)
    lam, Vt = exact_sgep(A, B, k)
    Bh = sqrtm_psd(B)
    W = whiten_truth(Vt, Bh)
    V, _ = gamma_eigengame(lambda t, r: (matrix_operator(A), matrix_operator(B)),
                           d, k, n_iter=20000, lr=0.2, seed=0)
    assert subspace_error(V, W, Bh) < 1e-6
    assert longest_streak(V, W, Bh) == k
    assert np.allclose(rayleigh_quotients(V, matrix_operator(A), matrix_operator(B)), lam, rtol=1e-4)


def test_stochastic_pca_converges():
    rng = np.random.default_rng(2)
    n, d, k = 5000, 40, 4
    scales = np.concatenate([[4.0, 3.0, 2.5, 2.0], np.linspace(1.0, 0.5, d - 4)])
    X = rng.standard_normal((n, d)) * scales[None, :]
    C = X.T @ X / n
    lam, Vt = exact_sgep(C, None, k)

    def ops(t, r):
        idx = r.integers(0, n, 128)
        Xb = X[idx]
        return (lambda V: Xb.T @ (Xb @ V) / len(idx)), identity_operator

    V, _ = gamma_eigengame(ops, d, k, n_iter=4000, lr=lambda t: 0.02 / (1 + t / 1000),
                           stochastic=True, seed=0)
    assert subspace_error(V, Vt) < 1e-2


if __name__ == "__main__":
    test_reduces_to_eigengame_when_B_is_identity()
    test_deterministic_sgep_matches_scipy()
    test_stochastic_pca_converges()
    print("all tests passed")
