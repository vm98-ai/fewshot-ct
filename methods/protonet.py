"""
Inductive nearest-centroid baseline (ProtoNet, Snell et al. NeurIPS 2017).

Each query is classified independently against class prototypes computed
from the support set -- no information is shared across query samples.
This is the reference point every transductive method below is compared to.
"""
from __future__ import annotations
import torch
from .base import FewShotMethod, batched_prototypes, squared_euclidean_logits, accuracy


class ProtoNet(FewShotMethod):
    name = "ProtoNet (inductive)"

    def __init__(self, temp: float = 1.0):
        self.temp = temp

    def run(self, support, query, y_s, y_q, n_ways):
        centroids = batched_prototypes(support, y_s, n_ways)
        logits = squared_euclidean_logits(query, centroids, self.temp)
        probs = logits.softmax(-1)
        return {"acc": accuracy(logits, y_q), "probs": probs, "logits": logits}
