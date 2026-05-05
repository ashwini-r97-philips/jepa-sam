"""SSL pretraining dataset — images only, with augmentation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.io import load_json

logger = logging.getLogger(__name__)


class SSLDataset(Dataset):
    """Dataset for self-supervised pretraining (MAE / JEPA).

    Returns only images (no masks, no boxes). Applies augmentations
    to maximize diversity from a small dataset.

    Args:
        split_file: Path to JSON split file (list of {image, mask} dicts).
        image_size: Target image size (square).
        augment: Whether to apply data augmentation.
        limit: Limit number of samples (smoke test).
    """

    def __init__(
        self,
        split_file: str,
        image_size: int = 1024,
        augment: bool = True,
        limit: Optional[int] = None,
    ):
        self.samples: List[Dict[str, str]] = load_json(split_file)
        if limit is not None:
            self.samples = self.samples[:limit]
        self.image_size = image_size
        self.augment = augment
        self._build_transforms()
        logger.info("SSLDataset: %d images, augment=%s", len(self.samples), augment)

    def _build_transforms(self) -> None:
        """Build torchvision transforms for SSL pretraining."""
        import torchvision.transforms as T

        if self.augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.5, 1.0),
                                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),  # [0, 1] range
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size),
                         interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.samples[idx]
        image_path = entry["image"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return {"image": image_tensor, "image_id": Path(image_path).stem}
