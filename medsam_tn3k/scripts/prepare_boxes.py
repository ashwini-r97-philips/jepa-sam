#!/usr/bin/env python3
"""Compute oracle bounding boxes from ground-truth masks.

Usage:
    python scripts/prepare_boxes.py --split_dir data/splits/seed0

Reads all split JSON files and computes tight bounding boxes from masks.
Saves: data/splits/seed0/boxes.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.box_utils import mask_to_bbox
from src.data.transforms import load_mask_binary
from src.utils.io import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute oracle bounding boxes from masks.")
    parser.add_argument("--split_dir", type=str, default="data/splits/seed0",
                        help="Directory containing split JSON files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images (smoke test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    split_dir = Path(args.split_dir)
    boxes_file = split_dir / "boxes.json"

    if boxes_file.exists():
        existing = load_json(boxes_file)
        logger.info("Boxes file already exists with %d entries. Skipping.", len(existing))
        return

    # Collect all unique mask paths from all splits
    all_masks = {}  # image_id -> mask_path
    for split_file in sorted(split_dir.glob("*.json")):
        if split_file.name == "boxes.json":
            continue
        samples = load_json(split_file)
        for entry in samples:
            image_id = Path(entry["image"]).stem
            all_masks[image_id] = entry["mask"]

    logger.info("Computing boxes for %d unique images", len(all_masks))

    if args.limit is not None:
        items = list(all_masks.items())[:args.limit]
        all_masks = dict(items)
        logger.info("Limited to %d images (smoke test)", len(all_masks))

    boxes = {}
    errors = []
    for i, (image_id, mask_path) in enumerate(sorted(all_masks.items())):
        try:
            mask = load_mask_binary(mask_path)
            bbox = mask_to_bbox(mask)
            boxes[image_id] = bbox
        except ValueError as e:
            logger.error("Empty mask for %s: %s", image_id, e)
            errors.append(image_id)

        if (i + 1) % 500 == 0:
            logger.info("Processed %d/%d", i + 1, len(all_masks))

    if errors:
        logger.warning("%d empty masks found: %s", len(errors), errors[:10])

    save_json(boxes, boxes_file)
    logger.info("Saved %d boxes to %s", len(boxes), boxes_file)


if __name__ == "__main__":
    main()
