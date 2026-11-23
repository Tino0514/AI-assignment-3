"""Figures 4-5 of Experiment 2, drawn from results/exp2_lda_mnist.json
(called by exp2_lda_mnist.py, or run on its own to re-draw the figures)."""
import json
import os

from common import RESULTS, np, plt, savefig

BATCHES = [16, 64, 256]


def plot(res):
    K = len(res["lam"])
    lam = res["lam"]
    BCOL = {16: "#eb6834", 64: "#1baf7a", 256: "#2a78d6"}
    BSTY = {16: ":", 64: "--", 256: "-"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    r = res["runs"]["deterministic"]
    ax1.plot(r["iters"], r["err"], color="#52514e", lw=1.6, label="Algorithm 1 (full batch)")
    for b in BATCHES:
        E = np.array([q["err"] for q in res["runs"][f"b{b}_auxTrue"]])
        it = res["runs"][f"b{b}_auxTrue"][0]["iters"]
        ax1.plot(it, E.mean(0), BSTY[b], color=BCOL[b], label=f"Algorithm 2, b = {b}")
        ax1.fill_between(it, E.min(0), E.max(0), color=BCOL[b], alpha=0.15, lw=0)
    ax1.set_yscale("log"); ax1.set_xlabel("iteration"); ax1.set_ylabel("subspace error (log scale)")
    ax1.set_title("LDA on MNIST: convergence to scipy eigh(S_b, S_w + eps I)")
    ax1.legend(loc="upper right")
    xs = np.arange(len(BATCHES)); w = 0.36
    for j, (aux, col, lab) in enumerate([(True, "#2a78d6", "with auxiliary [Bv]"), (False, "#eb6834", "without auxiliary [Bv]")]):
        F = np.array([[q["err"][-1] for q in res["runs"][f"b{b}_aux{aux}"]] for b in BATCHES])
        ax2.bar(xs + (j - 0.5) * (w + 0.02), F.mean(1), w, yerr=F.std(1), color=col, label=lab,
                capsize=2, error_kw=dict(lw=0.8, ecolor="#52514e"))
    ax2.set_yscale("log"); ax2.set_xticks(xs); ax2.set_xticklabels([f"b = {b}" for b in BATCHES])
    ax2.set_ylabel("final subspace error (log scale)"); ax2.set_title("Ablation: auxiliary variable")
    ax2.legend(loc="upper right"); ax2.grid(axis="x", visible=False)
    savefig(fig, "fig4_lda_convergence.png")

    # eigenvalue trajectories (unbiasedness across batch sizes)
    ramp = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for ax, b in zip(axes, BATCHES):
        q = res["runs"][f"b{b}_auxTrue"][0]
        RQ = np.array(q["rq"])[1:]                     # drop t = 0 for the log axis
        its = np.array(q["iters"])[1:]
        for i in range(K):
            ax.axhline(lam[i], color="#8a8984", lw=0.7, ls="--")
            ax.plot(its, RQ[:, i], color=ramp[i], lw=1.2)
        ax.set_title(f"b = {b}"); ax.set_xlabel("iteration (log scale)"); ax.set_xscale("log")
        ax.set_xlim(its[0], its[-1])
    axes[0].set_ylabel("generalized Rayleigh quotient")
    savefig(fig, "fig5_lda_eigenvalues.png")


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "exp2_lda_mnist.json")) as f:
        plot(json.load(f))
    print("figures re-drawn")
