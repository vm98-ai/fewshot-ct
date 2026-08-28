"""
Robustness sweep: for a fixed n-way/k-shot setting, corrupts the *query*
images with increasing severity (noise / contrast / brightness) and tracks
how far each method's accuracy falls -- the "robustness assessment" called
out explicitly in the FM2AI posting. Transductive methods that lean hard on
query-query similarity (LaplacianShot, PADDLE) are expected to degrade
faster under corruption than the inductive ProtoNet baseline once the
corruption is severe enough to break the query manifold structure; showing
*where* that crossover happens is the point of this script.

Usage:
    python scripts/run_robustness.py --corruption gaussian_noise
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.medmnist_data import load_plane
from data.episodic_sampler import EpisodicSampler
from features.extract_features import extract_features
from methods import METHOD_REGISTRY
from robustness.corruptions import CORRUPTIONS, SEVERITY_LEVELS


def sample_task_no_leak(support_sampler, query_sampler, n_ways, k_shot, q_shots):
    rng = support_sampler.rng
    common_classes = np.intersect1d(support_sampler.classes, query_sampler.classes)
    chosen = rng.choice(common_classes, size=n_ways, replace=False)

    sx, sy, qx, qy = [], [], [], []
    for local_label, c in enumerate(chosen):
        s_idx = rng.choice(support_sampler.by_class[c], size=k_shot, replace=False)
        q_pool_idx = np.setdiff1d(query_sampler.by_class[c], s_idx)  # explicit, index-based exclusion
        q_idx = rng.choice(q_pool_idx, size=min(q_shots, len(q_pool_idx)), replace=False)

        sx.append(support_sampler.features[s_idx]); sy.append(np.full(k_shot, local_label))
        qx.append(query_sampler.features[q_idx]); qy.append(np.full(len(q_idx), local_label))

    return (np.concatenate(sx), np.concatenate(sy),
            np.concatenate(qx), np.concatenate(qy))


def sample_batch_no_leak(support_sampler, query_sampler, n_tasks, n_ways, k_shot, q_shots, device):
    """Batched version of sample_task_no_leak; mirrors EpisodicSampler.sample_batch."""
    S_x, S_y, Q_x, Q_y = [], [], [], []
    for _ in range(n_tasks):
        sx, sy, qx, qy = sample_task_no_leak(support_sampler, query_sampler, n_ways, k_shot, q_shots)
        S_x.append(sx); S_y.append(sy); Q_x.append(qx); Q_y.append(qy)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "biomedclip"])
    p.add_argument("--corruption", default="gaussian_noise", choices=list(CORRUPTIONS))
    p.add_argument("--n_tasks", type=int, default=200)
    p.add_argument("--n_ways", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_shots", type=int, default=15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    imgs, labels, _ = load_plane("axial", split="test")

    # clean support features (support is never corrupted -- it's the known
    # labeled reference the on-site adaptation starts from)
    clean_feats = extract_features(imgs, backbone=args.backbone, device=args.device)
    support_sampler = EpisodicSampler(clean_feats, labels)

    print(f"\nCorruption: {args.corruption} | {args.n_ways}-way {args.k_shot}-shot")
    print(f"{'severity':>10} " + " ".join(f"{k:>22}" for k in METHOD_REGISTRY))

    for severity in SEVERITY_LEVELS[args.corruption]:
        corrupt_fn = CORRUPTIONS[args.corruption]
        corrupted_imgs = corrupt_fn(imgs, severity)
        corrupt_feats = extract_features(corrupted_imgs, backbone=args.backbone, device=args.device)
        query_sampler = EpisodicSampler(corrupt_feats, labels)

        batch = sample_batch_no_leak(
            support_sampler, query_sampler,
            args.n_tasks, args.n_ways, args.k_shot, args.q_shots, args.device,
        )

        row = []
        for key, cls in METHOD_REGISTRY.items():
            method = cls()
            out = method.run(batch["x_s"], batch["x_q"], batch["y_s"], batch["y_q"], args.n_ways)
            acc = out["acc"].float()
            row.append(f"{acc.mean().item()*100:6.2f} +/- {acc.std().item()*100:4.2f}%")
        print(f"{severity:>10} " + " ".join(row))


if __name__ == "__main__":
    main()