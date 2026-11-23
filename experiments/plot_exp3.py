"""Figures 8-9 of Experiment 3, drawn from results/exp3_faces.json and
results/exp3_components_split0.npz (called by exp3_faces.py, or run on its own)."""
import json
import os

from common import COLORS, RESULTS, np, plt, savefig

M_LIST = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
F_LIST = [5, 10, 15, 20, 25, 30, 35, 39]


def plot():
    with open(os.path.join(RESULTS, "exp3_faces.json")) as f:
        rec = json.load(f)["records"]
    comp = np.load(os.path.join(RESULTS, "exp3_components_split0.npz"))
    shape = tuple(comp["shape"])

    # ---- Figure 8: recognition rate vs number of components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    raw = np.mean(rec["raw"]) * 100
    for ax, keys, xs, title, xl in [
            (ax1, ("eig_exact", "eig_eg"), M_LIST, "Eigenfaces (PCA) + 1-NN", "number of eigenfaces"),
            (ax2, ("fish_exact", "fish_eg"), F_LIST, "Fisherfaces (PCA-60 + LDA) + 1-NN", "number of Fisherfaces")]:
        for key, lab, col, ls, mk in [(keys[0], "exact (SVD / scipy eigh)", COLORS["exact"], "--", "s"),
                                      (keys[1], "gamma-EigenGame", COLORS["gamma-EigenGame"], "-", "o")]:
            M = np.array([rec[key][str(m)] for m in xs]) * 100
            ax.errorbar(xs, M.mean(1), yerr=M.std(1), color=col, ls=ls, marker=mk, ms=4, capsize=2, lw=1.4, label=lab)
        ax.axhline(raw, color="#8a8984", lw=0.8, ls=":")
        ax.text(xs[0], raw + 0.5, "raw pixels, 1-NN", ha="left", va="bottom", fontsize=7, color="#52514e")
        ax.set_title(title); ax.set_xlabel(xl)
    ax1.set_ylabel("recognition rate (%)\nmean ± std over 10 splits")
    ax1.legend(loc="lower right")
    savefig(fig, "fig8_faces_accuracy.png")

    # ---- Figure 9: first eight eigenfaces / Fisherfaces (split 0)
    rows = [("P_exact", "Eigenfaces\nSVD"), ("P_eg", "Eigenfaces\ngamma-EG"),
            ("F_exact", "Fisherfaces\neigh"), ("F_eg", "Fisherfaces\ngamma-EG")]
    fig, axes = plt.subplots(4, 8, figsize=(9, 5.4), gridspec_kw=dict(hspace=0.12, wspace=0.05))
    for r, (key, name) in enumerate(rows):
        ref_key = "P_exact" if r < 2 else "F_exact"
        for i in range(8):
            v = comp[key][:, i] * np.sign(comp[key][:, i] @ comp[ref_key][:, i])
            ax = axes[r, i]
            ax.imshow(v.reshape(shape), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"{i + 1}", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
    savefig(fig, "fig9_faces_components.png")


if __name__ == "__main__":
    plot()
    print("figures re-drawn")
