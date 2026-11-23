"""
Experiment 3 - Face recognition on ORL / Olivetti faces (40 people x 10 images,
64 x 64 = 4,096 pixels) with Eigenfaces (PCA) and Fisherfaces (PCA + LDA),
where every eigen-decomposition is replaced by gamma-EigenGame.

Protocol: 10 random splits, 5 training and 5 test images per person,
1-nearest-neighbour classifier in the projected space.

  Eigenfaces  : PCA,  A = covariance (matrix-free, A V = X'(X V)/n),  B = I
  Fisherfaces : PCA to m = 60 dims, then LDA with A = S_b, B = S_w (k = 39)
"""
import os
import time

from common import RESULTS, np, save_json
from sklearn.neighbors import KNeighborsClassifier

from data import load_faces, scatter_matrices, split_faces
from gamma_eigengame import gamma_eigengame, identity_operator, matrix_operator
from metrics import exact_sgep, longest_streak, sqrtm_psd, subspace_error, whiten_truth

N_SPLITS = 10
K_PCA = 100            # eigenfaces learned by gamma-EigenGame
M_FISHER = 60          # PCA dimension before LDA (Fisherfaces)
K_LDA = 39             # c - 1 discriminant directions
M_LIST = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
F_LIST = [5, 10, 15, 20, 25, 30, 35, 39]
PCA_ITERS, PCA_LR = 3000, 2.0
LDA_ITERS, LDA_LR = 120000, 5.0

X, y, shape = load_faces()


def power_norm(op, d, iters=30, seed=0):
    """Spectral norm of a PSD operator by power iteration (matrix-free)."""
    v = np.random.default_rng(seed).standard_normal((d, 1))
    for _ in range(iters):
        v = op(v); v /= np.linalg.norm(v)
    return float(v[:, 0] @ op(v)[:, 0])


def knn_acc(Z, ytr, Zt, yte):
    return KNeighborsClassifier(1).fit(Z, ytr).score(Zt, yte)


rec = dict(raw=[], eig_exact={m: [] for m in M_LIST}, eig_eg={m: [] for m in M_LIST},
           fish_exact={f: [] for f in F_LIST}, fish_eg={f: [] for f in F_LIST},
           pca_err60=[], pca_err100=[], pca_streak=[], lda_err=[], lda_streak=[],
           t_svd=[], t_eg_pca=[], t_eigh_lda=[], t_eg_lda=[])
keep = {}
for s in range(N_SPLITS):
    rng = np.random.default_rng(s)
    tr, te = split_faces(y, 5, rng)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    mu = Xtr.mean(0)
    Xc, Xtc = Xtr - mu, Xte - mu
    n, d = Xc.shape
    rec["raw"].append(knn_acc(Xtr, ytr, Xte, yte))

    # ---------------- exact Eigenfaces (SVD) and Fisherfaces (eigh)
    t0 = time.perf_counter()
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P_exact = Vt[:K_PCA].T
    rec["t_svd"].append(time.perf_counter() - t0)
    for m in M_LIST:
        rec["eig_exact"][m].append(knn_acc(Xc @ P_exact[:, :m], ytr, Xtc @ P_exact[:, :m], yte))
    Z = Xc @ P_exact[:, :M_FISHER]
    Sb, Sw, *_ = scatter_matrices(Z, ytr)
    t0 = time.perf_counter()
    _, W_exact = exact_sgep(Sb, Sw, K_LDA)
    rec["t_eigh_lda"].append(time.perf_counter() - t0)
    for f in F_LIST:
        Wf = W_exact[:, :f]
        rec["fish_exact"][f].append(knn_acc(Z @ Wf, ytr, Xtc @ P_exact[:, :M_FISHER] @ Wf, yte))

    # ---------------- gamma-EigenGame Eigenfaces  (B = I, matrix-free A)
    t0 = time.perf_counter()
    lam1 = power_norm(lambda V: Xc.T @ (Xc @ V) / n, d)
    Xs = Xc / np.sqrt(lam1)
    A_op = lambda V: Xs.T @ (Xs @ V) / n
    P_eg, _ = gamma_eigengame(lambda t, r: (A_op, identity_operator), d, K_PCA, PCA_ITERS, PCA_LR, seed=s)
    rec["t_eg_pca"].append(time.perf_counter() - t0)
    rec["pca_err60"].append(subspace_error(P_eg[:, :M_FISHER], P_exact[:, :M_FISHER]))
    rec["pca_err100"].append(subspace_error(P_eg, P_exact))
    rec["pca_streak"].append(longest_streak(P_eg, P_exact))
    for m in M_LIST:
        rec["eig_eg"][m].append(knn_acc(Xc @ P_eg[:, :m], ytr, Xtc @ P_eg[:, :m], yte))

    # ---------------- gamma-EigenGame Fisherfaces (LDA on the gamma-EG PCA space)
    P60 = P_eg[:, :M_FISHER]
    Zg = Xc @ P60
    Sb_g, Sw_g, *_ = scatter_matrices(Zg, ytr)
    t0 = time.perf_counter()
    A1 = matrix_operator(Sb_g / power_norm(matrix_operator(Sb_g), M_FISHER))
    B1 = matrix_operator(Sw_g / power_norm(matrix_operator(Sw_g), M_FISHER))
    W_eg, _ = gamma_eigengame(lambda t, r: (A1, B1), M_FISHER, K_LDA, LDA_ITERS, LDA_LR, seed=s)
    rec["t_eg_lda"].append(time.perf_counter() - t0)
    _, W_ref = exact_sgep(Sb_g, Sw_g, K_LDA)          # reference in the same PCA space
    Bh = sqrtm_psd(Sw_g)
    rec["lda_err"].append(subspace_error(W_eg, whiten_truth(W_ref, Bh), Bh))
    rec["lda_streak"].append(longest_streak(W_eg, whiten_truth(W_ref, Bh), Bh))
    for f in F_LIST:
        Wf = W_eg[:, :f]
        rec["fish_eg"][f].append(knn_acc(Zg @ Wf, ytr, Xtc @ P60 @ Wf, yte))

    if s == 0:
        keep = dict(mu=mu, P_exact=P_exact, P_eg=P_eg, F_exact=P_exact[:, :M_FISHER] @ W_exact, F_eg=P60 @ W_eg)
    print(f"split {s}: raw {rec['raw'][-1]:.3f} | eigenfaces(60) exact {rec['eig_exact'][60][-1]:.3f} "
          f"EG {rec['eig_eg'][60][-1]:.3f} | fisherfaces(39) exact {rec['fish_exact'][39][-1]:.3f} "
          f"EG {rec['fish_eg'][39][-1]:.3f} | PCA err60 {rec['pca_err60'][-1]:.4f} streak {rec['pca_streak'][-1]} "
          f"| LDA err {rec['lda_err'][-1]:.4f} streak {rec['lda_streak'][-1]} "
          f"| t_eg_pca {rec['t_eg_pca'][-1]:.1f}s t_eg_lda {rec['t_eg_lda'][-1]:.1f}s", flush=True)

summary = {}
for key in ["raw", "pca_err60", "pca_err100", "pca_streak", "lda_err", "lda_streak", "t_svd", "t_eg_pca", "t_eigh_lda", "t_eg_lda"]:
    summary[key] = (float(np.mean(rec[key])), float(np.std(rec[key])))
for key in ["eig_exact", "eig_eg", "fish_exact", "fish_eg"]:
    summary[key] = {str(m): (float(np.mean(v)), float(np.std(v))) for m, v in rec[key].items()}
save_json(dict(records={k: (v if isinstance(v, list) else {str(a): b for a, b in v.items()}) for k, v in rec.items()},
               summary=summary), "exp3_faces.json")
for k, v in summary.items():
    print(k, v)

# ---------------------------------------------------------------- figures
np.savez_compressed(os.path.join(RESULTS, "exp3_components_split0.npz"), shape=np.array(shape),
                    **{k: keep[k][:, :8] for k in ["P_exact", "P_eg", "F_exact", "F_eg"]})
from plot_exp3 import plot  # noqa: E402

plot()
print("done")
