from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _to_rgb_tensor_batch(images: np.ndarray) -> torch.Tensor:
    """[N, H, W] uint8 grayscale -> [N, 3, H, W] float in [0,1]."""
    x = torch.from_numpy(images).float() / 255.0
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    return x


def build_backbone(name: str = "resnet18", device: str = "cpu"):
    if name == "resnet18":
        import torchvision.models as models
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            net = models.resnet18(weights=weights)
        except Exception as e:  # offline sandbox / no internet
            print(f"[warn] could not fetch pretrained ResNet18 weights ({e}); "
                  f"falling back to random init. Re-run with internet access "
                  f"for meaningful features.")
            net = models.resnet18(weights=None)
        net.fc = nn.Identity()  # 512-d embeddings
        net.eval().to(device)
        preprocess = _to_rgb_tensor_batch
        dim = 512
        return net, preprocess, dim

    elif name == "biomedclip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model.eval().to(device)

        def preprocess(images):
            x = _to_rgb_tensor_batch(images)
            return torch.nn.functional.interpolate(x, size=224, mode="bilinear")

        dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else 512
        return model.visual, preprocess, dim

    else:
        raise ValueError(f"unknown backbone {name}")


@torch.no_grad()
def extract_features(images: np.ndarray, backbone: str = "resnet18",
                      device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    net, preprocess, dim = build_backbone(backbone, device)
    feats = []
    for i in range(0, len(images), batch_size):
        batch = preprocess(images[i:i + batch_size]).to(device)
        f = net(batch)
        if isinstance(f, tuple):
            f = f[0]
        feats.append(f.view(f.size(0), -1).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)
