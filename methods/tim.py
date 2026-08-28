"""
TIM-GD: Transductive Information Maximization (Boudiaf et al., NeurIPS 2020),
gradient-descent variant, as used in the PADDLE benchmark.

Adapted from:
  https://github.com/mboudiaf/TIM  (original TIM authors)
  https://github.com/SegoleneMartin/PADDLE/blob/main/src/methods/tim.py
    (batched multi-task version used by the OPIS group's few-shot benchmark)
"""
from __future__ import annotations
import torch
from .base import (FewShotMethod, get_one_hot, batched_prototypes,
                    squared_euclidean_logits, entropy, cond_entropy, accuracy)


class TIM_GD(FewShotMethod):
    name = "TIM-GD (transductive)"

    def __init__(self, temp: float = 15.0, iters: int = 1000, lr: float = 1e-3,
                 loss_weights=(1.0, 0.1, 0.1)):
        self.temp = temp
        self.iters = iters
        self.lr = lr
        self.loss_weights = loss_weights  # (CE on support, marginal-entropy, cond-entropy)

    def run(self, support, query, y_s, y_q, n_ways):
        w = batched_prototypes(support, y_s, n_ways).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s, n_ways)

        for _ in range(self.iters):
            logits_s = squared_euclidean_logits(support, w, self.temp)
            logits_q = squared_euclidean_logits(query, w, self.temp)

            ce = -(y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            marg_ent = -(q_probs.mean(1) * torch.log(q_probs.mean(1) + 1e-12)).sum(1).sum(0)
            cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)

            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * marg_ent
                                                 - self.loss_weights[2] * cond_ent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits_q = squared_euclidean_logits(query, w, self.temp)
            probs = logits_q.softmax(-1)
        return {"acc": accuracy(logits_q, y_q), "probs": probs, "logits": logits_q}