"""
Adapted from:
  - S. Martin et al., "Towards Practical Few-Shot Query Sets: Transductive MDL
    Inference" (NeurIPS 2022) -- https://github.com/SegoleneMartin/PADDLE
  - M. Boudiaf et al., "TIM: Transductive Information Maximization" (NeurIPS 2020)
    -- https://github.com/mboudiaf/TIM
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def get_one_hot(y: torch.Tensor, n_ways: int | None = None) -> torch.Tensor:
    """[n_task, n] long labels -> [n_task, n, n_ways] one-hot."""
    if n_ways is None:
        n_ways = int(y.max().item()) + 1
    eye = torch.eye(n_ways, device=y.device, dtype=torch.float32)
    return eye[y]


def batched_prototypes(support: torch.Tensor, y_s: torch.Tensor, n_ways: int) -> torch.Tensor:
    """Class-mean prototypes per task. Returns [n_task, n_ways, dim]."""
    one_hot = get_one_hot(y_s, n_ways)                       # [T, S, W]
    counts = one_hot.sum(1).clamp_min(1e-6).unsqueeze(-1)    # [T, W, 1]
    weights = one_hot.transpose(1, 2).matmul(support)        # [T, W, D]
    return weights / counts


def squared_euclidean_logits(samples: torch.Tensor, centroids: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """
    Negative squared-Euclidean-distance logits between samples and centroids,
    expanded as ||s||^2 - 2 s.c + ||c||^2 (the trick used in PADDLE/TIM so it's
    a single batched matmul instead of an O(n^2) explicit distance loop).
    Returns [n_task, n_samples, n_ways].
    """
    n_task = samples.size(0)
    logits = temp * (
        samples.matmul(centroids.transpose(1, 2))
        - 0.5 * (centroids ** 2).sum(2).view(n_task, 1, -1)
        - 0.5 * (samples ** 2).sum(2).view(n_task, -1, 1)
    )
    return logits


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Marginal entropy H(Y) over the query batch. probs: [T, Q, W] -> [T, 1]."""
    marginal = probs.mean(1)
    return -(marginal * torch.log(marginal + 1e-12)).sum(1, keepdim=True)


def cond_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Conditional entropy H(Y|X), averaged over the query batch. -> [T, 1]."""
    return -(probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)


def mutual_information(probs: torch.Tensor) -> torch.Tensor:
    return entropy(probs) - cond_entropy(probs)


def accuracy(logits: torch.Tensor, y_q: torch.Tensor) -> torch.Tensor:
    """Per-task accuracy. logits: [T, Q, W], y_q: [T, Q] -> [T]."""
    preds = logits.argmax(-1)
    return (preds == y_q).float().mean(-1)


class FewShotMethod:
    """Common interface every method below implements."""

    name = "base"

    def run(self, support: torch.Tensor, query: torch.Tensor,
            y_s: torch.Tensor, y_q: torch.Tensor, n_ways: int) -> dict:
        """
        Returns a dict with at least:
          'acc'    : [n_task] float tensor of per-task query accuracy
          'probs'  : [n_task, n_query, n_ways] final soft predictions
        """
        raise NotImplementedError
