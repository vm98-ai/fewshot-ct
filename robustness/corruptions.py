from __future__ import annotations
import numpy as np


def gaussian_noise(images: np.ndarray, severity: float) -> np.ndarray:
    noise = np.random.normal(0, severity * 255, images.shape)
    return np.clip(images.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def contrast_shift(images: np.ndarray, factor: float) -> np.ndarray:
    mean = images.mean(axis=(1, 2), keepdims=True)
    return np.clip((images.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def brightness_shift(images: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(images.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)


CORRUPTIONS = {
    "none": lambda x, s: x,
    "gaussian_noise": gaussian_noise,
    "contrast": contrast_shift,
    "brightness": brightness_shift,
}

# severity sweeps used in the robustness report
SEVERITY_LEVELS = {
    "none": [0.0],
    "gaussian_noise": [0.02, 0.05, 0.1, 0.2],
    "contrast": [0.8, 0.6, 1.4, 1.8],
    "brightness": [-0.1, -0.2, 0.1, 0.2],
}
