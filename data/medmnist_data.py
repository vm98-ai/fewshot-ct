"""
Loads the OrganAMNIST / OrganCMNIST / OrganSMNIST splits from MedMNIST v2.

These three datasets are 2D slices (axial / coronal / sagittal respectively)
extracted from the *same* underlying real abdominal CT volumes (from the
LiTS -- Liver Tumor Segmentation -- benchmark), labeled with 11 organ
classes. That makes them a genuine, if small-scale, stand-in for the
"on-site adaptation across acquisition/plane" scenario in the FM2AI project:
same organs, real CT data, different imaging plane per dataset.
"""
from __future__ import annotations
import numpy as np

PLANES = {"axial": "organamnist", "coronal": "organcmnist", "sagittal": "organsmnist"}


def load_plane(plane: str, split: str = "test", size: int = 28):
    """
    Returns (images, labels): images as [N, H, W] uint8 in [0,255],
    labels as [N] int in 0..10 (11 organ classes).
    """
    import medmnist
    from medmnist import INFO

    assert plane in PLANES, f"plane must be one of {list(PLANES)}"
    flag = PLANES[plane]
    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, download=True, size=size)
    images = ds.imgs                      # [N, H, W]
    labels = ds.labels.squeeze(-1)         # [N]
    return images, labels, info["label"]


def load_all_planes(split: str = "test", size: int = 28):
    """Convenience: load axial/coronal/sagittal at once."""
    out = {}
    for plane in PLANES:
        out[plane] = load_plane(plane, split=split, size=size)
    return out


if __name__ == "__main__":
    imgs, labels, class_names = load_plane("axial", split="test")
    print(f"axial test set: {imgs.shape}, classes: {class_names}")
