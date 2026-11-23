"""Figures 1-2 of Experiment 1, drawn from results/exp1_pca_mnist.json
(called by exp1_pca_mnist.py, or run on its own to re-draw the figures)."""
import json
import os

from common import COLORS, RESULTS, STYLES, np, plt, savefig

METHODS = ["gamma-EigenGame", "alpha-EigenGame", "GHA", "Oja+QR"]
BATCHES = [8, 32, 256]


def plot(results, k=16):
    it = np.array(results["iters"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, b in zip(axes, BATCHES):
        for m in METHODS:
            E = np.array(results["runs"][f"{m}|{b}"]["err"])
            ax.plot(it, E.mean(0), STYLES[m], color=COLORS[m], label=m)
            ax.fill_between(it, E.min(0), E.max(0), color=COLORS[m], alpha=0.15, lw=0)   # min-max over seeds
        ax.set_yscale("log")
        ax.set_title(f"minibatch size b = {b}")
        ax.set_xlabel("iteration")
    axes[0].set_ylabel("subspace error (log scale)")
    axes[2].legend(loc="upper right")
    savefig(fig, "fig1_pca_subspace_error.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, b in zip(axes, BATCHES):
        for m in METHODS:
            S = np.array(results["runs"][f"{m}|{b}"]["streak"])
            ax.plot(it, S.mean(0), STYLES[m], color=COLORS[m], label=m)
        ax.set_title(f"minibatch size b = {b}")
        ax.set_xlabel("iteration")
        ax.set_ylim(0, k + 0.5)
    axes[0].set_ylabel(f"longest correct streak (of {k})")
    axes[2].legend(loc="lower right")
    savefig(fig, "fig2_pca_streak.png")


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "exp1_pca_mnist.json")) as f:
        plot(json.load(f))
    print("figures re-drawn")
