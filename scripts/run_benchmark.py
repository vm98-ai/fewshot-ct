"""
Main entry point: compares ProtoNet / TIM / PADDLE / LaplacianShot on
few-shot organ classification from real abdominal CT slices (MedMNIST
OrganA/C/SMNIST), with and without a train-axial / test-coronal domain
shift, across several shot counts.

Usage:
    python scripts/run_benchmark.py --backbone resnet18 --n_tasks 200
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


def build_samplers(backbone: str, device: str, size: int = 28):
    samplers = {}
    for plane in ["axial", "coronal", "sagittal"]:
        imgs, labels, _ = load_plane(plane, split="test", size=size)
        feats = extract_features(imgs, backbone=backbone, device=device)
        samplers[plane] = EpisodicSampler(feats, labels)
    return samplers


def run_setting(samplers, query_plane, n_tasks, n_ways, k_shot, q_shots, device):
    support_sampler = samplers["axial"]        # support always drawn "in-domain"
    query_sampler = samplers[query_plane]       # query may be a different plane
    same_domain = (query_plane == "axial")

    batch = support_sampler.sample_batch(
        n_tasks, n_ways, k_shot, q_shots,
        query_pool=None if same_domain else query_sampler, device=device,
    )

    results = {}
    for key, cls in METHOD_REGISTRY.items():
        method = cls()
        out = method.run(batch["x_s"], batch["x_q"], batch["y_s"], batch["y_q"], n_ways)
        acc = out["acc"].float()
        results[key] = (acc.mean().item(), acc.std().item())
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "biomedclip"])
    p.add_argument("--n_tasks", type=int, default=200)
    p.add_argument("--n_ways", type=int, default=5)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    p.add_argument("--q_shots", type=int, default=15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Extracting {args.backbone} features for axial/coronal/sagittal splits...")
    samplers = build_samplers(args.backbone, args.device)

    print(f"\n{'plane':>10} {'shot':>5} " + " ".join(f"{k:>22}" for k in METHOD_REGISTRY))
    for plane in ["axial", "coronal", "sagittal"]:
        for k in args.shots:
            res = run_setting(samplers, plane, args.n_tasks, args.n_ways, k, args.q_shots, args.device)
            row = " ".join(f"{res[m][0]*100:6.2f} +/- {res[m][1]*100:4.2f}%" for m in METHOD_REGISTRY)
            print(f"{plane:>10} {k:>5} {row}")


if __name__ == "__main__":
    main()
