"""Evaluation orchestrator: runs metrics over a set of predictions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_metrics
from src.utils.io import save_json

logger = logging.getLogger(__name__)


def evaluate_predictions(
    pred_dir: str,
    split_file: str,
    output_dir: str,
    split_name: str = "test",
    include_hd95: bool = False,
) -> Dict[str, Any]:
    """Evaluate predicted masks against ground truth.

    Args:
        pred_dir: Directory containing predicted mask PNGs (named by image_id).
        split_file: Path to the split JSON file with ground-truth mask paths.
        output_dir: Where to save metrics JSON and per-case CSV.
        split_name: Name for output files (e.g., 'val', 'test').
        include_hd95: Whether to compute HD95 metric.

    Returns:
        Aggregate metrics dict.
    """
    from src.utils.io import load_json
    import pandas as pd

    samples = load_json(split_file)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_case: List[Dict[str, Any]] = []

    for entry in samples:
        image_id = Path(entry["image"]).stem
        pred_path = pred_dir / f"{image_id}.png"
        gt_path = entry["mask"]

        if not pred_path.exists():
            logger.warning("Missing prediction for %s, skipping", image_id)
            continue

        pred = np.array(Image.open(pred_path).convert("L")) > 0
        gt = np.array(Image.open(gt_path).convert("L")) > 0

        m = compute_metrics(pred.astype(np.uint8), gt.astype(np.uint8),
                            include_hd95=include_hd95)
        m["image_id"] = image_id
        per_case.append(m)

    # Aggregate
    agg: Dict[str, Any] = {}
    for key in ["dice", "iou", "hd95"]:
        values = [c[key] for c in per_case if c.get(key) is not None]
        if values:
            agg[f"{key}_mean"] = round(float(np.mean(values)), 5)
            agg[f"{key}_std"] = round(float(np.std(values)), 5)
            agg[f"{key}_median"] = round(float(np.median(values)), 5)
    agg["n_cases"] = len(per_case)

    # Save
    save_json(agg, output_dir / f"metrics_{split_name}.json")
    df = pd.DataFrame(per_case)
    df.to_csv(output_dir / f"metrics_{split_name}_per_case.csv", index=False)
    logger.info("Saved metrics for %d cases: %s", len(per_case), agg)

    return agg
