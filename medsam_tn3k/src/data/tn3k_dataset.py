"""TN3K dataset class for MedSAM training and inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.box_utils import bbox_to_1024, mask_to_bbox
from src.data.transforms import (
    image_to_tensor,
    load_image_rgb,
    load_mask_binary,
    mask_to_tensor,
    resize_image_1024,
    resize_mask_256,
)
from src.utils.io import load_json


class TN3KDataset(Dataset):
    """PyTorch Dataset for TN3K with MedSAM pre-processing.

    Modes:
        train_supervised: returns image, mask, box, metadata.
        inference: returns image, mask (for eval), box, metadata.
        unlabeled: returns image, box=None, metadata (mask not used).

    Args:
        split_file: Path to JSON split file (list of {image, mask} dicts).
        boxes_file: Path to boxes.json mapping image_id -> [xmin, ymin, xmax, ymax].
        mode: One of 'train_supervised', 'inference', 'unlabeled'.
        limit: If set, only load first N samples (smoke test).
    """

    MODES = ("train_supervised", "inference", "unlabeled")

    def __init__(
        self,
        split_file: str,
        boxes_file: str,
        mode: str = "inference",
        limit: Optional[int] = None,
    ):
        assert mode in self.MODES, f"mode must be one of {self.MODES}, got {mode}"
        self.mode = mode
        self.samples: List[Dict[str, str]] = load_json(split_file)
        if limit is not None:
            self.samples = self.samples[:limit]

        self.boxes: Dict[str, List[int]] = load_json(boxes_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.samples[idx]
        image_path = entry["image"]
        mask_path = entry["mask"]
        image_id = Path(image_path).stem

        # Load image
        image_rgb = load_image_rgb(image_path)
        original_h, original_w = image_rgb.shape[:2]
        image_1024, _ = resize_image_1024(image_rgb)
        image_tensor = image_to_tensor(image_1024)

        # Oracle box
        bbox_orig = self.boxes.get(image_id)
        if bbox_orig is None and self.mode != "unlabeled":
            raise KeyError(f"No box found for image_id={image_id}")
        bbox_1024 = (
            bbox_to_1024(bbox_orig, original_h, original_w)
            if bbox_orig is not None
            else [0, 0, 0, 0]
        )
        box_tensor = torch.tensor(bbox_1024, dtype=torch.float32)

        result: Dict[str, Any] = {
            "image": image_tensor,
            "box": box_tensor,
            "image_id": image_id,
            "original_size": (original_h, original_w),
        }

        # Mask
        if self.mode in ("train_supervised", "inference"):
            mask = load_mask_binary(mask_path)
            mask_256 = resize_mask_256(mask)
            result["mask"] = mask_to_tensor(mask_256)
            # Original-resolution mask only for inference (varies in size, can't batch)
            if self.mode == "inference":
                result["mask_original"] = torch.from_numpy(mask).unsqueeze(0).float()

        return result
