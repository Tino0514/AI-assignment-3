"""
baselines.py
============

Classical / earlier streaming eigen-solvers used as baselines for PCA:

* alpha-EigenGame  (Gemp et al., ICLR 2021)   - the predecessor of gamma-EigenGame
* GHA              (Sanger, 1989)            - Generalized Hebbian Algorithm
* Oja + QR         (Oja, 1982; stochastic orthogonal/power iteration)

Every solver takes an operator get_A(t, rng) returning A_t (V -> A_t V), so
that the same minibatch stream can be fed to every method.
"""

from __future__ import annotations

import time

import numpy as np

from gamma_eigengame import GameHistory, init_players, normalize_columns


def _run(step, get_A, d, k, n_iter, lr, seed, callback, log_every, orthonormal_init=False):
    rng = np.random.default_rng(seed)
    V = init_players(d, k, rng)
    if orthonormal_init:
        V, _ = np.linalg.qr(V)
    lr_fn = lr if callable(lr) else (lambda t: lr)
    hist = GameHistory()
    t0 = time.perf_counter()
    elapsed = 0.0
    for t in range(n_iter + 1):
        if callback is not None and (t % log_every == 0 or t == n_iter):
            elapsed += time.perf_counter() - t0
            hist.iters.append(t)
            hist.seconds.append(elapsed)
            hist.metrics.append(callback(t, V))
            t0 = time.perf_counter()
        if t == n_iter:
            break
        A_t = get_A(t, rng)
        V = step(V, A_t(V), lr_fn(t))
        if not np.all(np.isfinite(V)):          # diverged -> stop early
            V = np.nan_to_num(V)
            break
    return V, hist


# ----------------------------------------------------------------- alpha-EG
def alpha_eigengame_step(V, AV, lr):
    """alpha-EigenGame (parallel version):
        grad_i  = 2 [ A v_i - sum_{j<i} (<v_i,Av_j>/<v_j,Av_j>) A v_j ]
        grad_i <- grad_i - <grad_i, v_i> v_i       (Riemannian projection)
        v_i    <- normalise(v_i + lr * grad_i)
    """
    G = V.T @ AV                                     # G_ij = <v_i, A v_j>
    coef = np.tril(G / np.diag(G)[None, :], -1)      # j < i
    grad = 2.0 * (AV - AV @ coef.T)
    grad = grad - V * np.einsum("dk,dk->k", grad, V)[None, :]
    return normalize_columns(V + lr * grad)


def alpha_eigengame(get_A, d, k, n_iter, lr, seed=0, callback=None, log_every=10):
    return _run(alpha_eigengame_step, get_A, d, k, n_iter, lr, seed, callback, log_every)


# --------------------------------------------------------------------- GHA
def gha_step(V, AV, lr):
    """Sanger's rule in matrix form:  dW = A W - W triu(W^T A W)."""
    return V + lr * (AV - V @ np.triu(V.T @ AV))


def gha(get_A, d, k, n_iter, lr, seed=0, callback=None, log_every=10):
    return _run(gha_step, get_A, d, k, n_iter, lr, seed, callback, log_every)


# -------------------------------------------------------------- Oja + QR
def oja_qr_step(V, AV, lr):
    """Stochastic orthogonal iteration:  W <- QR(W + lr * A W)."""
    Q, R = np.linalg.qr(V + lr * AV)
    return Q * np.sign(np.diag(R))[None, :]


def oja_qr(get_A, d, k, n_iter, lr, seed=0, callback=None, log_every=10):
    return _run(oja_qr_step, get_A, d, k, n_iter, lr, seed, callback, log_every,
                orthonormal_init=True)
