#!/usr/bin/env python3
"""Download TN3K dataset from Hugging Face.

Usage:
    python scripts/download_tn3k.py --out_dir data/raw/TN3K --repo_id haifan-gong/TN3K
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TN3K dataset from Hugging Face.")
    parser.add_argument("--out_dir", type=str, default="data/raw/TN3K",
                        help="Output directory for downloaded data.")
    parser.add_argument("--repo_id", type=str, default="haifan-gong/TN3K",
                        help="Hugging Face repo ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    out_dir = Path(args.out_dir)

    # Idempotent: skip if already downloaded
    if out_dir.exists() and any(out_dir.rglob("*.png")) or any(out_dir.rglob("*.jpg")):
        image_count = len(list(out_dir.rglob("*.png"))) + len(list(out_dir.rglob("*.jpg")))
        if image_count > 0:
            logger.info("Dataset already exists at %s (%d images). Skipping download.",
                        out_dir, image_count)
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading TN3K from Hugging Face repo: %s", args.repo_id)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("Download complete: %s", out_dir)
    except Exception as e:
        logger.error("Failed to download dataset: %s", e)
        logger.info(
            "Alternative: manually download from https://huggingface.co/datasets/%s "
            "and extract into %s", args.repo_id, out_dir
        )
        sys.exit(1)

    # Verify
    image_count = len(list(out_dir.rglob("*.png"))) + len(list(out_dir.rglob("*.jpg")))
    logger.info("Verified: %d image files in %s", image_count, out_dir)


if __name__ == "__main__":
    main()
