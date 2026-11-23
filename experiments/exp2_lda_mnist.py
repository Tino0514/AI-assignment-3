"""
Experiment 2 - Linear Discriminant Analysis on MNIST with gamma-EigenGame.

LDA = SGEP with A = S_b (between-class scatter) and B = S_w + eps*I
(within-class scatter, ridge-regularised because constant border pixels make
S_w singular).  We learn the k = c - 1 = 9 discriminant directions

  * with the deterministic Algorithm 1 (exact A, B), and
  * with the stochastic Algorithm 2 where B is estimated from minibatches of
    b in {16, 64, 256} training images (A uses the 10 class means only),
  * ablation: Algorithm 2 *without* the auxiliary running estimate of B v.

Ground truth: scipy.linalg.eigh(S_b, S_w + eps I).  The learned projections
are evaluated by 1-NN and nearest-class-mean (NCM) accuracy on the 10,000 test
images.
"""
import time

from common import np, plt, save_json, savefig
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from data import load_mnist, scatter_matrices
from gamma_eigengame import gamma_eigengame, matrix_operator
from metrics import exact_sgep, longest_streak, sqrtm_psd, subspace_error, whiten_truth

EPS = 0.1
K = 9
BATCHES = [16, 64, 256]
SEEDS = [0, 1, 2]
T = 10000
LOG = 100
LR0 = 1.0
AUX_LR = 0.1

Xtr, ytr, Xte, yte = load_mnist()
n, d = Xtr.shape
Sb, Sw, means, priors, mu = scatter_matrices(Xtr, ytr)
B = Sw + EPS * np.eye(d)

t0 = time.perf_counter()
lam, Vtrue = exact_sgep(Sb, B, K)
t_exact = time.perf_counter() - t0
Bh = sqrtm_psd(B)
Wtrue = whiten_truth(Vtrue, Bh)
print("generalized eigenvalues", lam.round(3), "eigh time %.2fs" % t_exact)

# scale A and B to unit spectral norm (does not change the eigenvectors)
sA, sB = np.linalg.eigvalsh(Sb)[-1], np.linalg.eigvalsh(B)[-1]
Mc = means - mu
A_op = lambda V: (Mc.T @ (priors[:, None] * (Mc @ V))) / sA      # S_b V from class means
Xw = Xtr - means[ytr]                                            # within-class centred data
RHO = 0.5 * EPS / sB                                             # lower bound on sigma_min(B)

Sb_full, B_full = matrix_operator(Sb), matrix_operator(B)


def callback(t, V):
    rq = np.einsum("dk,dk->k", V, Sb @ V) / np.einsum("dk,dk->k", V, B @ V)
    return dict(err=subspace_error(V, Wtrue, Bh), streak=longest_streak(V, Wtrue, Bh), rq=rq)


def stochastic_ops(b):
    def ops(t, rng):
        Xb = Xw[rng.integers(0, n, b)]
        return A_op, (lambda V: (Xb.T @ (Xb @ V) / b + EPS * V) / sB)
    return ops


def pack(h):
    return dict(err=[m["err"] for m in h.metrics], streak=[m["streak"] for m in h.metrics],
                rq=[m["rq"] for m in h.metrics], seconds=h.seconds[-1], iters=h.iters)


res = dict(lam=lam, t_exact=t_exact, eps=EPS, runs={})
final_V = {}

# (a) deterministic Algorithm 1 with the exact (scaled) matrices
A1, B1 = matrix_operator(Sb / sA), matrix_operator(B / sB)
V, h = gamma_eigengame(lambda t, r: (A1, B1), d, K, 3000, 1.0, seed=0, callback=callback, log_every=20)
res["runs"]["deterministic"] = pack(h)
final_V["deterministic"] = V
print("Algorithm 1: err %.2e streak %d time %.1fs" % (h.metrics[-1]["err"], h.metrics[-1]["streak"], h.seconds[-1]))

# (b) stochastic Algorithm 2 and (c) ablation without auxiliary variable
for b in BATCHES:
    for aux in [True, False]:
        runs = []
        for s in SEEDS:
            V, h = gamma_eigengame(stochastic_ops(b), d, K, T, lambda t: LR0 / (1 + t / 2000.0),
                                   stochastic=aux, aux_lr=AUX_LR, rho=RHO, seed=s,
                                   callback=callback, log_every=LOG)
            runs.append(pack(h))
            if s == 0:
                final_V[f"b{b}_aux{aux}"] = V
        res["runs"][f"b{b}_aux{aux}"] = runs
        fe = np.array([r["err"][-1] for r in runs])
        fs = np.array([r["streak"][-1] for r in runs])
        print(f"Algorithm 2 b={b:4d} aux={aux!s:5s}: err {fe.mean():.2e}+-{fe.std():.1e} "
              f"streak {fs.mean():.1f} time {np.mean([r['seconds'] for r in runs]):.1f}s", flush=True)

# ---------------------------------------------------------- classification
def evaluate(P):
    Z, Zt = (Xtr - mu) @ P, (Xte - mu) @ P
    nn = KNeighborsClassifier(1).fit(Z, ytr).score(Zt, yte)
    cm = np.stack([Z[ytr == c].mean(0) for c in range(10)])
    ncm = float((np.argmin(((Zt[:, None, :] - cm[None]) ** 2).sum(-1), 1) == yte).mean())
    return nn, ncm


acc = {}
acc["PCA (9 comps, exact)"] = evaluate(PCA(K).fit(Xtr).components_.T)
acc["LDA exact (scipy eigh)"] = evaluate(Vtrue)
acc["LDA gamma-EG Alg.1 (deterministic)"] = evaluate(final_V["deterministic"])
for b in BATCHES:
    acc[f"LDA gamma-EG Alg.2 (b={b})"] = evaluate(final_V[f"b{b}_auxTrue"])
sk = LinearDiscriminantAnalysis(solver="svd").fit(Xtr, ytr)
acc["sklearn LDA classifier (reference)"] = (float(sk.score(Xte, yte)), float("nan"))
for k_, v in acc.items():
    print(f"{k_:40s} 1-NN {v[0]:.4f}  NCM {v[1]:.4f}")
res["accuracy"] = acc
save_json(res, "exp2_lda_mnist.json")

# ---------------------------------------------------------------- figures
from plot_exp2 import plot  # noqa: E402

plot(res)

# 2-D embeddings of the test set
rng = np.random.default_rng(0)
sub = rng.choice(len(Xte), 3000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, P) in zip(axes, [("exact LDA (scipy eigh)", Vtrue), ("gamma-EigenGame, Algorithm 2 (b = 256)", final_V["b256_auxTrue"])]):
    P = P[:, :2] * np.sign(np.sum(P[:, :2] * Vtrue[:, :2], 0))[None, :]
    P = P / np.sqrt(np.einsum("dk,dk->k", P, B @ P))[None, :]   # same scaling for both: v'Bv = 1
    Z = (Xte - mu) @ P
    ax.scatter(Z[sub, 0], Z[sub, 1], s=3, color="#b9b8b2", lw=0)
    for c in range(10):
        Zc = Z[yte == c]
        m_, S_ = Zc.mean(0), np.cov(Zc.T)
        ev, evec = np.linalg.eigh(S_)
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        ax.add_patch(Ellipse(m_, 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]), angle=ang, fill=False, lw=0.9, color="#2a78d6"))
        ax.text(m_[0], m_[1], str(c), ha="center", va="center", fontsize=10, weight="bold", color="#0b0b0b")
    ax.set_title(name); ax.set_xlabel("discriminant 1"); ax.set_ylabel("discriminant 2")
savefig(fig, "fig6_lda_embedding.png")

# discriminant vectors as images
fig, axes = plt.subplots(2, 9, figsize=(10, 2.6))
Vg = final_V["b256_auxTrue"]
for i in range(9):
    u = Vtrue[:, i] / np.linalg.norm(Vtrue[:, i])
    v = Vg[:, i] * np.sign(Vg[:, i] @ u)
    for row, img in enumerate([u, v]):
        ax = axes[row, i]
        ax.imshow(img.reshape(28, 28), cmap="RdBu_r", vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, i].set_title(f"LD {i + 1}", fontsize=8)
axes[0, 0].set_ylabel("exact", fontsize=8)
axes[1, 0].set_ylabel("gamma-EG", fontsize=8)
savefig(fig, "fig7_lda_directions.png")
print("done")
