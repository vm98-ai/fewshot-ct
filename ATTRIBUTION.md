# Attribution

The transductive few-shot methods in `methods/` are adapted from the official open-source implementations released alongside the papers they're based on:

| File | Adapted from | Paper |
|---|---|---|
| `methods/paddle_mdl.py` | https://github.com/SegoleneMartin/PADDLE (`src/methods/paddle.py`) | S. Martin, M. Boudiaf, E. Chouzenoux, J.-C. Pesquet, I. Ben Ayed, "Towards Practical Few-Shot Query Sets: Transductive MDL Inference," NeurIPS 2022. |
| `methods/tim.py` | https://github.com/SegoleneMartin/PADDLE (`src/methods/tim.py`), originally https://github.com/mboudiaf/TIM | M. Boudiaf et al., "Information Maximization for Few-Shot Learning," NeurIPS 2020. |
| `methods/laplacian_shot.py` | https://github.com/SegoleneMartin/PADDLE (`src/methods/laplacianshot.py`), originally https://github.com/imtiazziko/LaplacianShot | I. Ziko et al., "Laplacian Regularized Few-Shot Learning," ICML 2020. |

Also inspected while designing this project (not directly vendored, but the batched-task API convention — `[n_task, n_shot, feature_dim]` tensors, support/query separation, `get_one_hot` helper, etc. — follows these repos' convention so this project's code reads naturally alongside them):
- https://github.com/SegoleneMartin/transductive-CLIP — S. Martin, Y. Huang, F. Shakeri, J.-C. Pesquet, I. Ben Ayed, "Transductive Zero-Shot and Few-Shot CLIP," CVPR 2024.
- https://github.com/ZhouLong0/UNEM-Transductive — L. Zhou, F. Shakeri, A. Sadraoui, M. Kaaniche, J.-C. Pesquet, I. Ben Ayed, "UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning," CVPR 2025.

## What "adapted" means concretely

The core mathematical updates (PADDLE's primal-dual `u`/`v`/`w` block coordinate descent, TIM's mutual-information objective, LaplacianShot's bound-optimization energy) are reproduced faithfully because they *are* the contribution of those papers. What was stripped out or rewritten:

- the `sacred`-based experiment/config framework and CLI plumbing
- dataset-specific loaders for miniImageNet/tieredImageNet/CUB (replaced with the MedMNIST abdominal-CT loader in `data/`)
- backbone training code (this project only ever uses **frozen** pretrained features)
- logging/plotting utilities

## License note

No `LICENSE` file was present in any of the three source repositories at the time this project was built (August 2026). Code was adapted here for a personal, educational project with full citation of the original authors and papers above. If you plan to publish, redistribute, or build on this beyond a personal learning exercise, check each repo's page directly for any license added since, and consider reaching out to the authors as good practice.