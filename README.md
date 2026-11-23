# γ-EigenGame for PCA and LDA, applied to digit and face recognition

CISC3024 Pattern Recognition, AI Assignment #3. All code in this repository was written by an AI agent (Claude Code, Anthropic); the student wrote none of it.

This repository contains a NumPy implementation of **γ-EigenGame** from

> I. Gemp, C. Chen, B. McWilliams. *The Symmetric Generalized Eigenvalue Problem as a Nash Equilibrium.* ICLR 2023. [arXiv:2206.04993](https://arxiv.org/abs/2206.04993)

γ-EigenGame turns the generalized eigenvalue problem **A v = λ B v** into a game between *k* players. The Nash equilibrium of that game is the set of top-*k* generalized eigenvectors. Both PCA and LDA are special cases:

| Task | A | B |
|---|---|---|
| PCA (Eigenfaces) | covariance matrix C | I |
| LDA (Fisherfaces) | between-class scatter S_b | within-class scatter S_w (+ εI) |

The algorithm needs only matrix-vector products. It works with minibatches (Algorithm 2 of the paper) and never inverts or factorizes a d × d matrix.

## Repository layout

```
src/gamma_eigengame.py   γ-EigenGame: vectorised pseudo-gradient (Eq. 7), Algorithms 1 & 2
src/baselines.py         α-EigenGame (ICLR 2021), GHA (Sanger 1989), Oja + QR
src/metrics.py           scipy ground truth, subspace error, longest-correct-streak
src/data.py              MNIST / ORL-Olivetti loaders, scatter matrices
experiments/exp1_pca_mnist.py   streaming PCA on MNIST, 4 solvers, batch sizes 8/32/256
experiments/exp2_lda_mnist.py   LDA on MNIST, Algorithm 1 vs 2, auxiliary-variable ablation
experiments/exp3_faces.py       Eigenfaces + Fisherfaces on ORL faces, 10 random splits
tests/test_gamma_eigengame.py   sanity tests (Prop. 3 reduction, agreement with scipy)
results/                        logs, JSON results and figures produced by the scripts
```

## Running the code

```bash
pip install -r requirements.txt
python tests/test_gamma_eigengame.py
python experiments/exp1_pca_mnist.py   # ~10 min on one CPU core
python experiments/exp2_lda_mnist.py   # ~3 min
python experiments/exp3_faces.py       # ~10 min
```

On first use, scikit-learn downloads MNIST (OpenML) and the Olivetti faces (figshare) to `~/scikit_learn_data`.

## Main results

| Experiment | γ-EigenGame | Exact solver |
|---|---|---|
| PCA, MNIST, k = 16, minibatch 8: subspace error | 0.0063 (α-EigenGame: 0.49) | 0 |
| LDA, MNIST, 9 directions: 1-NN test accuracy | 90.5–90.9 % | 90.7 % |
| ORL faces, Eigenfaces (60): recognition rate | 87.95 % | 87.95 % |
| ORL faces, Fisherfaces (39): recognition rate | 94.6 % | 95.6 % |

The raw numbers are in `results/*.json` and `results/*_log.txt`, and the plots are in `results/figures/`.
