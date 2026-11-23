"""
gamma_eigengame.py
==================

NumPy implementation of **gamma-EigenGame** from

    I. Gemp, C. Chen, B. McWilliams,
    "The Symmetric Generalized Eigenvalue Problem as a Nash Equilibrium",
    ICLR 2023 (arXiv:2206.04993).

The method finds the top-k solutions of the symmetric generalized
eigenvalue problem (SGEP)

        A v = lambda B v,      A symmetric,  B symmetric positive definite,

by letting k "players" play a game.  Player i owns a unit vector v_i and
maximises the utility

  u_i = <v_i,Av_i>/<v_i,Bv_i>
        - sum_{j<i} <v_j,Av_j> <v_i,Bv_j>^2 / (<v_j,Bv_j>^2 <v_i,Bv_i>)

whose Nash equilibrium is the set of top-k generalized eigenvectors.

Special cases used in this project
  * PCA : A = covariance matrix C,          B = I
  * LDA : A = between-class scatter S_b,    B = within-class scatter S_w (+ eps I)

Only matrix-vector products with A and B are needed, so A and B are passed
as *operators* (callables V -> A @ V) and are never formed explicitly when
the data are large.  All k players are updated simultaneously (the
"parallel" version of the game), which in NumPy simply means that we work
on the d x k matrix V whose columns are the players.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

Operator = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def normalize_columns(V: np.ndarray) -> np.ndarray:
    """Project every column of V back onto the unit sphere (retraction)."""
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def init_players(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """v_i ~ N(0, I_d) followed by normalisation (as in Algorithms 1 and 2)."""
    return normalize_columns(rng.standard_normal((d, k)))


def matrix_operator(M: np.ndarray) -> Operator:
    """Wrap an explicit (d x d) matrix as an operator V -> M @ V."""
    return lambda V: M @ V


def identity_operator(V: np.ndarray) -> np.ndarray:
    return V


# ---------------------------------------------------------------------------
# the gamma-EigenGame pseudo-gradient  (Eq. 7 of the paper, vectorised)
# ---------------------------------------------------------------------------
def gamma_pseudo_gradient(
    V: np.ndarray,
    AV: np.ndarray,
    BV: np.ndarray,
    BV_parents: Optional[np.ndarray] = None,
    rho: float = 0.0,
) -> np.ndarray:
    """Return the gamma-EigenGame update direction of all k players.

    For player i (v = v_i) and its parents j < i:

        reward_i  = <v,Bv> A v - <v,Av> B v
        penalty_i = sum_{j<i} <v, A y_j> ( <v,Bv> [B y]_j - <v,[B y]_j> B v )
        grad_i    = reward_i - penalty_i

    with y_j = v_j / sqrt(max(<v_j,[Bv]_j>, rho)).

    Parameters
    ----------
    V          : (d, k) current strategies, unit-norm columns.
    AV, BV     : (d, k) A @ V and B @ V (exact or minibatch estimates).
    BV_parents : (d, k) estimate of B @ V used for the *parents*.  In the
                 stochastic Algorithm 2 this is the running-average auxiliary
                 variable [B v]; in the deterministic Algorithm 1 it equals BV.
    rho        : clip value that lower-bounds <v_j, B v_j> (Algorithm 2).
    """
    if BV_parents is None:
        BV_parents = BV

    vAv = np.einsum("dk,dk->k", V, AV)                     # <v_i, A v_i>
    vBv = np.einsum("dk,dk->k", V, BV)                     # <v_i, B v_i>
    parent_sq = np.maximum(np.einsum("dk,dk->k", V, BV_parents), rho)
    scale = 1.0 / np.sqrt(parent_sq)                       # 1/||v_j||_B

    BY = BV_parents * scale[None, :]                       # [B y_j]
    C = (V.T @ AV) * scale[None, :]                        # C_ij = <v_i, A y_j>
    L = np.tril(C, -1)                                     # keep parents j < i
    D = V.T @ BY                                           # D_ij = <v_i, [B y]_j>

    rewards = vBv[None, :] * AV - vAv[None, :] * BV
    penalties = (BY @ L.T) * vBv[None, :] - BV * np.sum(L * D, axis=1)[None, :]
    return rewards - penalties


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
@dataclass
class GameHistory:
    iters: list = field(default_factory=list)
    seconds: list = field(default_factory=list)
    metrics: list = field(default_factory=list)


def gamma_eigengame(
    get_ops: Callable[[int, np.random.Generator], Tuple[Operator, Operator]],
    d: int,
    k: int,
    n_iter: int,
    lr: float | Callable[[int], float],
    stochastic: bool = False,
    aux_lr: float | Callable[[int], float] = 1.0,
    rho: float = 0.0,
    seed: int = 0,
    V0: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, np.ndarray], dict]] = None,
    log_every: int = 10,
) -> Tuple[np.ndarray, GameHistory]:
    """Run gamma-EigenGame.

    get_ops(t, rng) must return a pair of operators (A_t, B_t).  For the
    deterministic Algorithm 1 they are the exact A and B; for the stochastic
    Algorithm 2 they are unbiased minibatch estimates.

    stochastic=False  ->  Algorithm 1 (parents use the exact B v_j).
    stochastic=True   ->  Algorithm 2 (parents use the auxiliary running
                          estimate [B v]_j, updated with step aux_lr, and the
                          normaliser is clipped from below by rho).
    """
    rng = np.random.default_rng(seed)
    V = init_players(d, k, rng) if V0 is None else normalize_columns(V0.copy())
    BV_aux = V.copy()                      # [B v]_i <- v_i^0  (Algorithm 2)
    lr_fn = lr if callable(lr) else (lambda t: lr)
    aux_fn = aux_lr if callable(aux_lr) else (lambda t: aux_lr)

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

        A_t, B_t = get_ops(t, rng)
        AV = A_t(V)
        BV = B_t(V)
        if stochastic:
            grad = gamma_pseudo_gradient(V, AV, BV, BV_aux, rho)
            BV_aux = BV_aux + aux_fn(t) * (BV - BV_aux)
        else:
            grad = gamma_pseudo_gradient(V, AV, BV, None, rho)
        V = normalize_columns(V + lr_fn(t) * grad)
    return V, hist


def rayleigh_quotients(V: np.ndarray, A_op: Operator, B_op: Operator) -> np.ndarray:
    """Generalized eigenvalue estimates lambda_i = <v_i,Av_i>/<v_i,Bv_i>."""
    return np.einsum("dk,dk->k", V, A_op(V)) / np.einsum("dk,dk->k", V, B_op(V))
