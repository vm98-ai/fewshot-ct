"""
PADDLE: PrimAl-Dual minimum-Description-LEngth inference
(S. Martin, M. Boudiaf, E. Chouzenoux, J.-C. Pesquet, I. Ben Ayed,
"Towards Practical Few-Shot Query Sets: Transductive MDL Inference", NeurIPS 2022)

This is a direct, dependency-light port of the official implementation so the
update equations match the published method exactly (only the surrounding
experiment/config plumbing was stripped out):
  https://github.com/SegoleneMartin/PADDLE/blob/main/src/methods/paddle.py
"""
from __future__ import annotations
import torch
from .base import FewShotMethod, get_one_hot, batched_prototypes, accuracy


class PADDLE(FewShotMethod):
    name = "PADDLE (transductive MDL, ref [2])"

    def __init__(self, alpha: float = 5.0, iters: int = 50):
        self.alpha = alpha
        self.iters = iters

    def _logits(self, samples, w):
        n_task = samples.size(0)
        return (samples.matmul(w.transpose(1, 2))
                - 0.5 * (w ** 2).sum(2).view(n_task, 1, -1)
                - 0.5 * (samples ** 2).sum(2).view(n_task, -1, 1))

    @staticmethod
    def _A(p):
        """Averaging operator A: mean over the query dimension."""
        return p.sum(1) / p.size(1)

    @staticmethod
    def _A_adj(v, n_query):
        """Adjoint of A: broadcast a per-class dual value back to every query."""
        return v.unsqueeze(1).repeat(1, n_query, 1) / n_query

    def run(self, support, query, y_s, y_q, n_ways):
        n_query = query.size(1)
        y_s_one_hot = get_one_hot(y_s, n_ways)
        n_task = support.size(0)

        w = batched_prototypes(support, y_s, n_ways)          # [T, W, D]
        v = torch.zeros(n_task, n_ways, device=support.device)  # dual variable

        for _ in range(self.iters):
            # u-update: soft query labels
            logits_q = self._logits(query, w).detach()
            u = (logits_q + self.alpha * self._A_adj(v, n_query)).softmax(2)

            # v-update: dual ascent (closed form)
            v = torch.log(self._A(u) + 1e-6) + 1

            # w-update: weighted centroid recompute using labeled support + soft query labels
            num = torch.einsum('bkq,bqd->bkd', u.transpose(1, 2), query) \
                + torch.einsum('bkq,bqd->bkd', y_s_one_hot.transpose(1, 2), support)
            den = u.sum(1) + y_s_one_hot.sum(1)
            w = num / den.unsqueeze(2)

        logits_q = self._logits(query, w).detach()
        probs = logits_q.softmax(-1)
        return {"acc": accuracy(logits_q, y_q), "probs": probs, "logits": logits_q}
