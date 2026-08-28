"""
N-way K-shot episodic task sampler.

Works on precomputed [N_samples, feature_dim] embeddings + integer labels,
and can draw support and query from *different* label-indexed pools -- this
is how we implement the domain-shift experiment (support drawn from axial
CT, query drawn from coronal/sagittal CT of the same organ classes).
"""
from __future__ import annotations
import numpy as np
import torch


class EpisodicSampler:
    def __init__(self, features: np.ndarray, labels: np.ndarray, seed: int = 0):
        self.features = features
        self.labels = labels
        self.classes = np.unique(labels)
        self.by_class = {c: np.where(labels == c)[0] for c in self.classes}
        self.rng = np.random.default_rng(seed)

    def sample_task(self, n_ways: int, k_shot: int, q_shots: int,
                     query_pool: "EpisodicSampler | None" = None):
        query_pool = query_pool or self
        common_classes = np.intersect1d(self.classes, query_pool.classes)
        chosen = self.rng.choice(common_classes, size=n_ways, replace=False)

        sx, sy, qx, qy = [], [], [], []
        for local_label, c in enumerate(chosen):
            s_idx = self.rng.choice(self.by_class[c], size=k_shot, replace=False)
            q_pool_idx = query_pool.by_class[c]
            # avoid overlap when query_pool is self
            if query_pool is self:
                q_pool_idx = np.setdiff1d(q_pool_idx, s_idx)
            q_idx = self.rng.choice(q_pool_idx, size=min(q_shots, len(q_pool_idx)), replace=False)

            sx.append(self.features[s_idx]); sy.append(np.full(k_shot, local_label))
            qx.append(query_pool.features[q_idx]); qy.append(np.full(len(q_idx), local_label))

        return (np.concatenate(sx), np.concatenate(sy),
                np.concatenate(qx), np.concatenate(qy))

    def sample_batch(self, n_tasks: int, n_ways: int, k_shot: int, q_shots: int,
                      query_pool: "EpisodicSampler | None" = None, device="cpu"):
        """Stack n_tasks episodes into the [n_task, n, dim] tensors methods/ expects."""
        S_x, S_y, Q_x, Q_y = [], [], [], []
        for _ in range(n_tasks):
            sx, sy, qx, qy = self.sample_task(n_ways, k_shot, q_shots, query_pool)
            S_x.append(sx); S_y.append(sy); Q_x.append(qx); Q_y.append(qy)
        # q_shots may differ slightly per class if a class pool is small; trim to min
        min_q = min(q.shape[0] for q in Q_x)
        Q_x = [q[:min_q] for q in Q_x]
        Q_y = [q[:min_q] for q in Q_y]

        to_t = lambda arr, dtype: torch.tensor(np.stack(arr), dtype=dtype, device=device)
        return {
            "x_s": to_t(S_x, torch.float32),
            "y_s": to_t(S_y, torch.long),
            "x_q": to_t(Q_x, torch.float32),
            "y_q": to_t(Q_y, torch.long),
        }
