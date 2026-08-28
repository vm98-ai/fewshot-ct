"""
LaplacianShot (Ziko et al., ICML 2020): transductive few-shot inference via
bound optimization of a Laplacian-regularized energy

Adapted from the official implementation:
  https://github.com/imtiazziko/LaplacianShot
  https://github.com/SegoleneMartin/PADDLE/blob/main/src/methods/laplacianshot.py
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import torch
from .base import FewShotMethod, batched_prototypes


def _affinity(x: np.ndarray, knn: int) -> sparse.csc_matrix:
    n = x.shape[0]
    knn = min(knn, n)
    nbrs = NearestNeighbors(n_neighbors=knn).fit(x)
    _, ind = nbrs.kneighbors(x)
    row = np.repeat(range(n), knn - 1)
    col = ind[:, 1:].flatten()
    data = np.ones(n * (knn - 1))
    return sparse.csc_matrix((data, (row, col)), shape=(n, n))


def _normalize(y: np.ndarray) -> np.ndarray:
    y = y - y.max(axis=1, keepdims=True)
    y = np.exp(y)
    return y / y.sum(axis=1, keepdims=True)


class LaplacianShot(FewShotMethod):
    name = "LaplacianShot (transductive)"

    def __init__(self, knn: int = 5, bound_lambda: float = 0.7, iters: int = 20):
        self.knn = knn
        self.bound_lambda = bound_lambda
        self.iters = iters

    def _bound_update(self, unary: np.ndarray, kernel: sparse.csc_matrix):
        Y = _normalize(-unary)
        old_e = float("inf")
        for _ in range(self.iters):
            pairwise = kernel.dot(Y)
            Y = _normalize(-unary - self.bound_lambda * pairwise)
            e = (Y * np.log(np.maximum(Y, 1e-20))
                 + unary * Y - self.bound_lambda * pairwise * Y).sum()
            if abs(e - old_e) <= 1e-6 * abs(old_e):
                break
            old_e = e
        return Y

    def run(self, support, query, y_s, y_q, n_ways):
        centroids = batched_prototypes(support, y_s, n_ways)  # [T, W, D]
        n_task = support.size(0)
        accs, all_probs = [], []

        for t in range(n_task):
            c = centroids[t].detach().cpu().numpy()            # [W, D]
            q = query[t].detach().cpu().numpy()                # [Q, D]
            # unary_i(k) = ||q_i - c_k||^2
            unary = ((q[:, None, :] - c[None, :, :]) ** 2).sum(-1)  # [Q, W]
            W = _affinity(q, self.knn)
            Y = self._bound_update(unary, W)                   # [Q, W] soft labels
            preds = Y.argmax(1)
            acc = float((preds == y_q[t].cpu().numpy()).mean())
            accs.append(acc)
            all_probs.append(torch.from_numpy(Y).float())

        probs = torch.stack(all_probs, dim=0)
        return {"acc": torch.tensor(accs), "probs": probs, "logits": None}
