"""
metrics.py
==========

Ground truth and evaluation metrics.

* exact_sgep        : top-k generalized eigenpairs with scipy.linalg.eigh(A, B)
* subspace_error    : 1 - tr(U* P)/k   (Appendix A.1 of the gamma-EigenGame paper)
* longest_streak    : number of leading eigenvectors recovered in the right order
                      within an angle of pi/8 (metric of the EigenGame papers)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def exact_sgep(A: np.ndarray, B: np.ndarray | None, k: int):
    """Top-k eigenvalues (descending) and B-orthonormal eigenvectors of A v = l B v."""
    d = A.shape[0]
    w, V = sla.eigh(A, B, subset_by_index=[d - k, d - 1])
    return w[::-1], V[:, ::-1]


def sqrtm_psd(B: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(B)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


def whiten_truth(V_true: np.ndarray, B_half: np.ndarray | None) -> np.ndarray:
    """W = B^{1/2} V*  -> orthonormal eigenvectors of B^{-1/2} A B^{-1/2}."""
    W = V_true if B_half is None else B_half @ V_true
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def subspace_error(V_hat: np.ndarray, W_true: np.ndarray, B_half: np.ndarray | None = None) -> float:
    """Normalised subspace distance in the whitened (B^{1/2}) coordinates."""
    k = W_true.shape[1]
    W_hat = V_hat if B_half is None else B_half @ V_hat
    if not np.all(np.isfinite(W_hat)):
        return 1.0
    with np.errstate(all="ignore"):
        W_hat = W_hat / np.linalg.norm(W_hat, axis=0, keepdims=True)
        Q, _ = np.linalg.qr(W_hat)
        err = 1.0 - np.sum((W_true.T @ Q) ** 2) / k
    return float(np.clip(err, 0.0, 1.0)) if np.isfinite(err) else 1.0   # diverged run -> 1


def per_vector_angles(V_hat, W_true, B_half=None):
    W_hat = V_hat if B_half is None else B_half @ V_hat
    W_hat = W_hat / np.linalg.norm(W_hat, axis=0, keepdims=True)
    cos = np.abs(np.einsum("dk,dk->k", W_hat, W_true))
    return np.arccos(np.clip(cos, 0.0, 1.0))


def longest_streak(V_hat, W_true, B_half=None, thresh=np.pi / 8) -> int:
    ang = per_vector_angles(V_hat, W_true, B_half)
    if not np.all(np.isfinite(ang)):
        return 0
    bad = np.where(ang >= thresh)[0]
    return int(bad[0]) if bad.size else int(len(ang))
