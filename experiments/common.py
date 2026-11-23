"""Shared settings for the experiment scripts (paths, plotting style, JSON I/O)."""
import json
import os
import sys

# single-threaded BLAS: much faster for the small d x k matrix products used here
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical palette, fixed order (one colour per method, never cycled)
COLORS = {
    "gamma-EigenGame": "#2a78d6",
    "alpha-EigenGame": "#eb6834",
    "GHA": "#1baf7a",
    "Oja+QR": "#eda100",
    "exact": "#52514e",
}
STYLES = {"gamma-EigenGame": "-", "alpha-EigenGame": "--", "GHA": "-.", "Oja+QR": ":", "exact": "--"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.facecolor": "white",
})


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def savefig(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
