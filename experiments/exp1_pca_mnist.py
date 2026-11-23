"""
Experiment 1 - Streaming PCA on MNIST (gamma-EigenGame with B = I).

Top-k = 16 principal components of the 60,000 MNIST training images (d = 784)
are learned from minibatches of size b in {8, 32, 256}.  gamma-EigenGame is
compared with alpha-EigenGame (ICLR 2021), GHA (Sanger 1989) and Oja + QR.
Ground truth comes from scipy.linalg.eigh on the full covariance matrix.
"""
import time

from common import np, plt, save_json, savefig

from baselines import alpha_eigengame, gha, oja_qr
from data import load_mnist
from gamma_eigengame import gamma_eigengame, identity_operator
from metrics import exact_sgep, longest_streak, subspace_error

K = 16
BATCHES = [8, 32, 256]
LR_GRID = [0.03, 0.1, 0.3]
SEEDS = [0, 1, 2]
T = 10000
LOG = 100
schedule = lambda lr0: (lambda t: lr0 / (1.0 + t / 2000.0))   # square-summable decay

Xtr, ytr, _, _ = load_mnist()
X = Xtr - Xtr.mean(0)
n, d = X.shape
C = X.T @ X / n
t0 = time.perf_counter()
lam, Vtrue = exact_sgep(C, None, K)
t_exact = time.perf_counter() - t0
X = X / np.sqrt(lam[0])                 # rescale so that ||C||_2 = 1
print("top eigenvalues", lam.round(3), "exact eigh time %.2fs" % t_exact)


def run(method, b, lr0, seed):
    def get_A(t, rng):
        idx = rng.integers(0, n, b)
        Xb = X[idx]
        return lambda V: Xb.T @ (Xb @ V) / b

    cb = lambda t, V: (subspace_error(V, Vtrue), longest_streak(V, Vtrue))
    lr = schedule(lr0)
    if method == "gamma-EigenGame":
        V, h = gamma_eigengame(lambda t, r: (get_A(t, r), identity_operator), d, K, T, lr,
                               stochastic=True, seed=seed, callback=cb, log_every=LOG)
    elif method == "alpha-EigenGame":
        V, h = alpha_eigengame(get_A, d, K, T, lr, seed=seed, callback=cb, log_every=LOG)
    elif method == "GHA":
        V, h = gha(get_A, d, K, T, lr, seed=seed, callback=cb, log_every=LOG)
    else:
        V, h = oja_qr(get_A, d, K, T, lr, seed=seed, callback=cb, log_every=LOG)
    err = [m[0] for m in h.metrics]
    streak = [m[1] for m in h.metrics]
    if len(err) < len(range(0, T + 1, LOG)):           # diverged -> pad with failure
        pad = len(range(0, T + 1, LOG)) - len(err)
        err += [1.0] * pad
        streak += [0] * pad
    return V, dict(err=err, streak=streak, seconds=h.seconds[-1])


METHODS = ["gamma-EigenGame", "alpha-EigenGame", "GHA", "Oja+QR"]
results = {"iters": list(range(0, T + 1, LOG)), "lam": lam, "t_exact": t_exact, "runs": {}}
best_V = None
for b in BATCHES:
    for m in METHODS:
        # 1) tune the initial step size on seed 0
        tune = {}
        for lr0 in LR_GRID:
            _, r = run(m, b, lr0, 0)
            tune[lr0] = r
        best = min(LR_GRID, key=lambda l: tune[l]["err"][-1])
        runs = [tune[best]]
        # 2) two more seeds with the selected step size
        for s in SEEDS[1:]:
            V, r = run(m, b, best, s)
            runs.append(r)
            if m == "gamma-EigenGame" and b == 256 and s == SEEDS[-1]:
                best_V = V
        key = f"{m}|{b}"
        results["runs"][key] = dict(lr0=best, tune={str(k): v["err"][-1] for k, v in tune.items()},
                                    err=[r["err"] for r in runs], streak=[r["streak"] for r in runs],
                                    seconds=[r["seconds"] for r in runs])
        fe = np.array([r["err"][-1] for r in runs])
        fs = np.array([r["streak"][-1] for r in runs])
        print(f"b={b:4d} {m:16s} lr0={best:<5} final err {fe.mean():.4f}+-{fe.std():.4f} "
              f"streak {fs.mean():.1f} time {np.mean([r['seconds'] for r in runs]):.1f}s", flush=True)

save_json(results, "exp1_pca_mnist.json")

# ---------------------------------------------------------------- figures
from plot_exp1 import plot  # noqa: E402

plot(results, K)

# eigen-digits: exact vs gamma-EigenGame
fig, axes = plt.subplots(2, 8, figsize=(9, 2.6))
for i in range(8):
    u = Vtrue[:, i]
    v = best_V[:, i] * np.sign(best_V[:, i] @ u)
    for row, img in enumerate([u, v]):
        ax = axes[row, i]
        ax.imshow(img.reshape(28, 28), cmap="RdBu_r", vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, i].set_title(f"PC {i + 1}", fontsize=8)
axes[0, 0].set_ylabel("exact (eigh)", fontsize=8)
axes[1, 0].set_ylabel("gamma-EG", fontsize=8)
savefig(fig, "fig3_pca_eigendigits.png")
print("done")
