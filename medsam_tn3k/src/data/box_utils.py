"""Bounding box utilities for oracle box computation."""
from __future__ import annotations

import numpy as np


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Compute tight bounding box [xmin, ymin, xmax, ymax] from a binary mask.

    Args:
        mask: 2-D binary array (H, W) with nonzero pixels as foreground.

    Returns:
        [xmin, ymin, xmax, ymax] in pixel coordinates.

    Raises:
        ValueError: If mask is entirely zero (no foreground).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("Empty mask: no foreground pixels found.")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # bbox format: xmin, ymin, xmax, ymax  (x=col, y=row)
    return [int(cmin), int(rmin), int(cmax), int(rmax)]


def bbox_to_1024(bbox: list[int], original_h: int, original_w: int,
                 target_size: int = 1024) -> list[int]:
    """Scale a bounding box from original image size to MedSAM 1024x1024 input space."""
    xmin, ymin, xmax, ymax = bbox
    scale_x = target_size / original_w
    scale_y = target_size / original_h
    return [
        int(round(xmin * scale_x)),
        int(round(ymin * scale_y)),
        int(round(xmax * scale_x)),
        int(round(ymax * scale_y)),
    ]
