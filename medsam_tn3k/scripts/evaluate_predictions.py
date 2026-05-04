#!/usr/bin/env python3
"""Standalone evaluation of predicted masks against ground truth.

Usage:
    python scripts/evaluate_predictions.py \
        --pred_dir outputs/stage0_initial_infer/test \
        --split_file data/splits/seed0/test.json \
        --output_dir outputs/stage0_initial_infer \
        --split_name test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import evaluate_predictions
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory with predicted mask PNGs.")
    parser.add_argument("--split_file", type=str, required=True,
                        help="Path to split JSON with ground-truth mask paths.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save metrics JSON and CSV.")
    parser.add_argument("--split_name", type=str, default="test",
                        help="Name for output files (e.g., val, test).")
    parser.add_argument("--include_hd95", action="store_true",
                        help="Include HD95 metric (requires scipy).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")

    metrics = evaluate_predictions(
        pred_dir=args.pred_dir,
        split_file=args.split_file,
        output_dir=args.output_dir,
        split_name=args.split_name,
        include_hd95=args.include_hd95,
    )
    logger.info("Aggregate metrics: %s", metrics)


if __name__ == "__main__":
    main()
