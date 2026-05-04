#!/usr/bin/env python3
"""Prepare reproducible train/val/test splits and labeled/unlabeled subsets for TN3K.

Usage:
    python scripts/prepare_splits.py --data_dir data/raw/TN3K --out_dir data/splits/seed0 --seed 0

Output files:
    train.json, val.json, test.json
    train_labeled_10pct.json, train_unlabeled_90pct.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.split_utils import labeled_unlabeled_split, stratified_split
from src.utils.io import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TN3K dataset splits.")
    parser.add_argument("--data_dir", type=str, default="data/raw/TN3K",
                        help="Root directory of downloaded TN3K dataset.")
    parser.add_argument("--out_dir", type=str, default="data/splits/seed0",
                        help="Output directory for split JSON files.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--labeled_fraction", type=float, default=0.1,
                        help="Fraction of training set to label.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    return parser.parse_args()


def discover_samples(data_dir: Path) -> List[Dict[str, str]]:
    """Discover image-mask pairs from the TN3K directory.

    TN3K typical structure:
      - image/ or images/ folder with PNGs/JPGs
      - mask/ or masks/ or label/ or labels/ folder with corresponding masks

    Also checks for official train/test split directories.
    """
    # Try common directory layouts
    image_dirs = ["image", "images", "Image", "Images"]
    mask_dirs = ["mask", "masks", "label", "labels", "Mask", "Masks", "Label", "Labels"]

    image_dir = None
    mask_dir = None

    for d in image_dirs:
        candidate = data_dir / d
        if candidate.is_dir():
            image_dir = candidate
            break

    for d in mask_dirs:
        candidate = data_dir / d
        if candidate.is_dir():
            mask_dir = candidate
            break

    # If not found at top level, search inside subdirectories (train/test)
    if image_dir is None or mask_dir is None:
        # TN3K may have train-image/, test-image/, train-mask/, test-mask/ structure
        samples = _discover_split_structure(data_dir)
        if samples:
            return samples

        # Search one level deeper
        for subdir in sorted(data_dir.iterdir()):
            if subdir.is_dir():
                for d in image_dirs:
                    candidate = subdir / d
                    if candidate.is_dir() and image_dir is None:
                        image_dir = candidate
                for d in mask_dirs:
                    candidate = subdir / d
                    if candidate.is_dir() and mask_dir is None:
                        mask_dir = candidate

    if image_dir is None or mask_dir is None:
        logger.error("Could not find image/mask directories in %s", data_dir)
        logger.info("Directory contents: %s", list(data_dir.iterdir()))
        sys.exit(1)

    logger.info("Found image_dir=%s, mask_dir=%s", image_dir, mask_dir)
    return _pair_images_masks(image_dir, mask_dir)


def _discover_split_structure(data_dir: Path) -> List[Dict[str, str]]:
    """Handle TN3K structure with train-image/test-image/train-mask/test-mask dirs."""
    samples = []
    for prefix in ["train", "test", "val"]:
        for img_suffix in ["-image", "-images", "_image", "_images", "/image", "/images"]:
            for msk_suffix in ["-mask", "-masks", "_mask", "_masks", "-label", "-labels",
                               "/mask", "/masks", "/label", "/labels"]:
                img_dir = data_dir / f"{prefix}{img_suffix}"
                msk_dir = data_dir / f"{prefix}{msk_suffix}"
                if img_dir.is_dir() and msk_dir.is_dir():
                    pairs = _pair_images_masks(img_dir, msk_dir)
                    samples.extend(pairs)
                    logger.info("Found %d pairs in %s / %s", len(pairs), img_dir.name, msk_dir.name)

    # Also check for subdirs like train/images, train/masks
    for prefix in ["train", "test", "val"]:
        prefix_dir = data_dir / prefix
        if prefix_dir.is_dir():
            for img_name in ["image", "images", "Image", "Images"]:
                for msk_name in ["mask", "masks", "label", "labels", "Mask", "Masks"]:
                    img_dir = prefix_dir / img_name
                    msk_dir = prefix_dir / msk_name
                    if img_dir.is_dir() and msk_dir.is_dir():
                        pairs = _pair_images_masks(img_dir, msk_dir)
                        samples.extend(pairs)
                        logger.info("Found %d pairs in %s/%s / %s/%s",
                                    len(pairs), prefix, img_name, prefix, msk_name)

    return samples


def _pair_images_masks(image_dir: Path, mask_dir: Path) -> List[Dict[str, str]]:
    """Pair images with masks by matching stem names."""
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in image_exts])
    masks = {f.stem: f for f in mask_dir.iterdir() if f.suffix.lower() in image_exts}

    samples = []
    for img in images:
        mask_file = masks.get(img.stem)
        if mask_file is not None:
            samples.append({
                "image": str(img.resolve()),
                "mask": str(mask_file.resolve()),
            })
        else:
            logger.warning("No mask found for image %s", img.name)

    return samples


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Check if splits already exist
    if (out_dir / "train.json").exists():
        logger.info("Splits already exist at %s. Skipping.", out_dir)
        # Print counts
        for name in ["train", "val", "test", "train_labeled_10pct", "train_unlabeled_90pct"]:
            p = out_dir / f"{name}.json"
            if p.exists():
                data = load_json(p)
                logger.info("  %s: %d samples", name, len(data))
        return

    # Discover all samples
    all_samples = discover_samples(data_dir)
    if not all_samples:
        logger.error("No image-mask pairs found in %s", data_dir)
        sys.exit(1)
    logger.info("Discovered %d image-mask pairs total", len(all_samples))

    # Verify every image has a corresponding mask
    for s in all_samples:
        assert Path(s["mask"]).exists(), f"Mask file missing: {s['mask']}"
        assert Path(s["image"]).exists(), f"Image file missing: {s['image']}"

    # Check for duplicates
    image_stems = [Path(s["image"]).stem for s in all_samples]
    assert len(image_stems) == len(set(image_stems)), "Duplicate image stems detected!"

    # Create train/val/test split
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    splits = stratified_split(all_samples, ratios, seed=args.seed)

    logger.info("Split counts: train=%d, val=%d, test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Verify no leakage
    train_stems = {Path(s["image"]).stem for s in splits["train"]}
    val_stems = {Path(s["image"]).stem for s in splits["val"]}
    test_stems = {Path(s["image"]).stem for s in splits["test"]}
    assert not (train_stems & val_stems), "Train-val leakage!"
    assert not (train_stems & test_stems), "Train-test leakage!"
    assert not (val_stems & test_stems), "Val-test leakage!"

    # Save splits
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(splits["train"], out_dir / "train.json")
    save_json(splits["val"], out_dir / "val.json")
    save_json(splits["test"], out_dir / "test.json")

    # Create labeled/unlabeled split within train
    labeled, unlabeled = labeled_unlabeled_split(
        splits["train"], labeled_fraction=args.labeled_fraction, seed=args.seed
    )
    logger.info("Labeled/unlabeled split: labeled=%d, unlabeled=%d",
                len(labeled), len(unlabeled))

    save_json(labeled, out_dir / "train_labeled_10pct.json")
    save_json(unlabeled, out_dir / "train_unlabeled_90pct.json")

    logger.info("All splits saved to %s", out_dir)


if __name__ == "__main__":
    main()
