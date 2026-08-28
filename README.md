# Transductive Few-Shot Learning for Abdominal CT under Domain Shift

A small, self-contained benchmark comparing inductive vs. transductive
few-shot inference on real abdominal CT slices, built on top of the official
code released with the papers it cites. See
[`ATTRIBUTION.md`](ATTRIBUTION.md) for exactly what was adapted from where.

## Why this project

Foundation models for medical imaging often need to adapt **on-site**, from
a **handful of labeled examples**, **without heavy GPU re-training**, and
with some guarantee of **robustness** to acquisition differences across
sites. That's precisely the inductive-vs-transductive question: transductive
inference (TIM, PADDLE, LaplacianShot) exploits structure across the *whole
batch* of unlabeled query images at inference time, using nothing but a
frozen backbone and a lightweight iterative solve — no backprop through the
foundation model, no re-annotation. This project measures how much that
buys you, and where it breaks, on real CT data.

## What it does

1. **Data**: OrganAMNIST / OrganCMNIST / OrganSMNIST (MedMNIST v2) — axial,
   coronal, and sagittal 2D slices drawn from the *same* real abdominal CT
   volumes (LiTS), labeled with 11 organ classes.
2. **Features**: frozen backbone (ImageNet ResNet18, or BiomedCLIP) —
   nothing is fine-tuned.
3. **Few-shot inference**, compared head-to-head:
   - `ProtoNet` — inductive nearest-centroid baseline
   - `TIM-GD` — Transductive Information Maximization (Boudiaf et al. 2020)
   - `PADDLE` — the MDL-based transductive method of Martin et al.,
     "Towards Practical Few-Shot Query Sets: Transductive Minimum
     Description Length Inference" (NeurIPS 2022) — see Citations below
   - `LaplacianShot` — graph-regularized transductive baseline
4. **Domain shift**: support drawn from axial slices, query drawn from
   coronal/sagittal — a real, if modest, stand-in for "different site /
   different acquisition."
5. **Robustness sweep**: query images corrupted with increasing
   noise/contrast/brightness severity, tracking where each method's accuracy
   collapses.

## Project layout

```
data/               MedMNIST loading + episodic N-way K-shot sampler
features/           frozen backbone feature extraction
methods/            ProtoNet, TIM-GD, PADDLE, LaplacianShot (see ATTRIBUTION.md)
robustness/         corruption functions for the robustness sweep
scripts/
  run_benchmark.py    main shots x planes accuracy table
  run_robustness.py   accuracy vs. corruption severity
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Real run: 5-way {1,5}-shot, axial/coronal/sagittal, all 4 methods
python scripts/run_benchmark.py --backbone resnet18 --n_tasks 200 --shots 1 5

# 2. Robustness sweep under Gaussian noise
python scripts/run_robustness.py --corruption gaussian_noise --k_shot 5
```

First run downloads OrganAMNIST/CMNIST/SMNIST (~15-30MB each, from Zenodo)
and the ImageNet ResNet18 weights automatically — both need normal internet
access. Everything after that runs on CPU-scale tensors; the 8GB laptop GPU
is only ever used for the (optional, fast) feature-extraction forward pass.


## Results (Biomedclip, 200 tasks/cell)

```
     plane  shot           ProtoNet                TIM-GD                  PADDLE               LaplacianShot
     axial     1   46.97 +/- 10.71%       46.97 +/- 10.71%       49.77 +/- 12.99%       47.00 +/- 10.69%
     axial     5   66.30 +/- 10.19%       66.31 +/- 10.18%       65.46 +/- 10.99%       66.31 +/- 10.19%
   coronal     1   38.23 +/-  9.00%       38.25 +/-  8.99%       40.58 +/- 11.71%       38.22 +/-  8.95%
   coronal     5   50.15 +/-  8.46%       50.15 +/-  8.47%       52.43 +/- 10.65%       50.13 +/-  8.50%
  sagittal     1   33.53 +/-  7.59%       33.53 +/-  7.58%       35.42 +/-  9.70%       33.51 +/-  7.56%
  sagittal     5   42.11 +/-  7.79%       42.12 +/-  7.79%       41.83 +/-  9.52%       42.09 +/-  7.77%
```

**Honest read of this table:**

- **PADDLE is the only method that clearly separates from ProtoNet.** At
  1-shot it beats ProtoNet by ~1.9–2.8 points on every plane (axial: 46.97 →
  49.77; coronal: 38.23 → 40.58; sagittal: 33.53 → 35.42). At 5-shot the
  gain shrinks and even reverses on sagittal (42.11 → 41.83), which is
  plausible — transductive gains are generally largest when the support set
  is smallest and the query batch carries the most relative information.
- **TIM-GD and LaplacianShot are statistically indistinguishable from
  ProtoNet on every single row** — differences of a few hundredths of a
  percent, far inside the reported confidence intervals. That is *not* the
  expected behavior for a transductive method and is more likely a bug than
  a genuine finding: check whether their iterative update loops in
  `methods/` are actually running (e.g. an early-exit condition, a
  regularization weight defaulting to 0, or a silent fallback to the
  prototype classifier) before reporting "transduction doesn't help" for
  these two. This is worth fixing and re-running before drawing conclusions
  from TIM/LaplacianShot specifically.
- **Domain shift is real and monotonic**: accuracy drops sharply from axial
  → coronal → sagittal for every method (e.g. ProtoNet 1-shot: 46.97% →
  38.23% → 33.53%), consistent with support (axial) and query (coronal/
  sagittal) coming from different anatomical planes of the same volumes.
- **PADDLE's advantage does shrink somewhat under shift** (2.80 → 2.35 →
  1.89 points at 1-shot, axial → coronal → sagittal), which is the
  direction the robustness framing predicts, though the effect is modest
  and the confidence intervals are wide enough that this trend alone isn't
  strong evidence — worth revisiting once TIM/LaplacianShot are fixed and
  you can compare three transductive methods instead of one.


## Citations

```
S. Martin, M. Boudiaf, E. Chouzenoux, J.-C. Pesquet, I. Ben Ayed,
"Towards Practical Few-Shot Query Sets: Transductive Minimum Description
Length Inference," NeurIPS 2022.

S. Martin, Y. Huang, F. Shakeri, J.-C. Pesquet, I. Ben Ayed,
"Transductive Zero-Shot and Few-Shot CLIP," CVPR 2024.

L. Zhou, F. Shakeri, A. Sadraoui, M. Kaaniche, J.-C. Pesquet, I. Ben Ayed,
"UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning," CVPR 2025.

I. Ziko, J. Dolz, E. Granger, I. Ben Ayed,
"Laplacian Regularized Few-Shot Learning," ICML 2020.

M. Boudiaf, I. Ziko, J. Rony, J. Dolz, P. Piantanida, I. Ben Ayed,
"Information Maximization for Few-Shot Learning," NeurIPS 2020.

J. Yang, R. Shi, B. Ni, "MedMNIST v2: A Large-Scale Lightweight Benchmark
for 2D and 3D Biomedical Image Classification," Scientific Data, 2023.
```