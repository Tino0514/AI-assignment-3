"""
data.py - dataset loaders (scikit-learn caches the files in ~/scikit_learn_data).

* MNIST           : 70,000 handwritten digits, 28x28 = 784 pixels, 10 classes
                    (standard split: first 60,000 train / last 10,000 test)
* ORL / Olivetti  : 400 face images of 40 people (10 each), 64x64 = 4,096 pixels
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_openml


def load_mnist():
    m = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = m.data.astype(np.float64) / 255.0
    y = m.target.astype(int)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def load_faces():
    f = fetch_olivetti_faces()
    return f.data.astype(np.float64), f.target.astype(int), f.images.shape[1:]


def split_faces(y, n_train_per_class, rng):
    """Random split with n_train_per_class images of every person for training."""
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        tr.extend(idx[:n_train_per_class])
        te.extend(idx[n_train_per_class:])
    return np.array(tr), np.array(te)


def scatter_matrices(X, y):
    """Between-class S_b, within-class S_w and class means (all divided by n)."""
    n, d = X.shape
    mu = X.mean(0)
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(0) for c in classes])
    priors = np.array([(y == c).mean() for c in classes])
    Mc = means - mu
    Sb = (Mc * priors[:, None]).T @ Mc
    Xw = X - means[np.searchsorted(classes, y)]
    Sw = Xw.T @ Xw / n
    return Sb, Sw, means, priors, mu
