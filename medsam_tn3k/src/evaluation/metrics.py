"""Segmentation evaluation metrics."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient for binary masks.

    Args:
        pred: Binary prediction (H, W) or flattened.
        target: Binary ground truth (H, W) or flattened.

    Returns:
        Dice score in [0, 1].
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    intersection = (pred & target).sum()
    return float(2.0 * intersection / (pred.sum() + target.sum()))


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU (Jaccard index) for binary masks.

    Args:
        pred: Binary prediction (H, W) or flattened.
        target: Binary ground truth (H, W) or flattened.

    Returns:
        IoU score in [0, 1].
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return float(intersection / union) if union > 0 else 0.0


def hd95_score(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Compute 95th percentile Hausdorff Distance.

    Returns None if scipy/medpy is not available or if either mask is empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        logger.debug("scipy not available, skipping HD95")
        return None

    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return None

    # Surface distances
    pred_border = pred ^ _erode(pred)
    target_border = target ^ _erode(target)

    if pred_border.sum() == 0 or target_border.sum() == 0:
        return None

    dt_target = distance_transform_edt(~target_border)
    dt_pred = distance_transform_edt(~pred_border)

    dist_pred_to_target = dt_target[pred_border]
    dist_target_to_pred = dt_pred[target_border]

    all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion by 1 pixel (3x3 kernel)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    include_hd95: bool = False,
) -> Dict[str, Optional[float]]:
    """Compute all metrics for a single prediction-target pair."""
    metrics: Dict[str, Optional[float]] = {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
    }
    if include_hd95:
        metrics["hd95"] = hd95_score(pred, target)
    return metrics
