"""Image and mask transforms for MedSAM input pipeline."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as RGB uint8 numpy array (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask_binary(path: str) -> np.ndarray:
    """Load a mask as binary uint8 numpy array (H, W). Values > 0 become 1."""
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    return (arr > 0).astype(np.uint8)


def resize_image_1024(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize image to 1024x1024 for MedSAM, return (resized, original_size)."""
    from skimage.transform import resize as sk_resize
    original_size = (image.shape[0], image.shape[1])
    resized = sk_resize(image, (1024, 1024), order=3, preserve_range=True,
                        anti_aliasing=True).astype(np.uint8)
    return resized, original_size


def resize_mask_256(mask: np.ndarray) -> np.ndarray:
    """Resize mask to 256x256 (MedSAM output resolution) using nearest interpolation."""
    from skimage.transform import resize as sk_resize
    return sk_resize(mask, (256, 256), order=0, preserve_range=True,
                     anti_aliasing=False).astype(np.uint8)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 image to CHW float32 tensor normalized to [0,1]."""
    t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return t


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert HW binary mask to (1, H, W) float32 tensor."""
    return torch.from_numpy(mask).unsqueeze(0).float()
