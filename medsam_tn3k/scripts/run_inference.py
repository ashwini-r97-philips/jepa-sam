#!/usr/bin/env python3
"""Run MedSAM inference on val/test splits using oracle boxes.

Usage:
    python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml
    python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml --limit 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tn3k_dataset import TN3KDataset
from src.evaluation.evaluator import evaluate_predictions
from src.models.medsam_loader import load_medsam_model, medsam_inference
from src.utils.io import load_yaml, save_json, ensure_dir
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedSAM inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (smoke test).")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Which splits to run inference on.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path from config.")
    return parser.parse_args()


def run_inference_on_split(
    model: torch.nn.Module,
    split_file: str,
    boxes_file: str,
    output_dir: str,
    device: str,
    batch_size: int = 1,
    limit: int | None = None,
) -> None:
    """Run inference on a single split and save predicted masks."""
    dataset = TN3KDataset(
        split_file=split_file,
        boxes_file=boxes_file,
        mode="inference",
        limit=limit,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pred_dir = Path(output_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for batch in tqdm(loader, desc=f"Inference → {pred_dir.name}"):
        image = batch["image"]
        box = batch["box"]
        image_ids = batch["image_id"]
        original_sizes = batch["original_size"]  # tuple of tensors

        pred_masks = medsam_inference(model, image, box)  # (B, 1, 256, 256)

        # Save each prediction as PNG at original resolution
        for i in range(pred_masks.shape[0]):
            mask_256 = pred_masks[i, 0].cpu().numpy().astype(np.uint8) * 255
            orig_h = original_sizes[0][i].item()
            orig_w = original_sizes[1][i].item()
            mask_pil = Image.fromarray(mask_256).resize((orig_w, orig_h), Image.NEAREST)
            mask_pil.save(pred_dir / f"{image_ids[i]}.png")

    logger.info("Saved %d predictions to %s", len(dataset), pred_dir)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    setup_logging(config.get("log_level", "INFO"))
    seed_everything(config.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    checkpoint = args.checkpoint or config["checkpoint_path"]
    model = load_medsam_model(checkpoint_path=checkpoint, device=device,
                              freeze_image_encoder=True)

    output_base = Path(config["output_dir"])
    split_dir = config["split_dir"]
    boxes_file = config["boxes_file"]
    batch_size = config.get("batch_size", 1)

    for split_name in args.splits:
        split_file = str(Path(split_dir) / f"{split_name}.json")
        pred_dir = str(output_base / split_name)
        logger.info("Running inference on %s split", split_name)
        run_inference_on_split(
            model=model,
            split_file=split_file,
            boxes_file=boxes_file,
            output_dir=pred_dir,
            device=device,
            batch_size=batch_size,
            limit=args.limit,
        )

        # Evaluate
        include_hd95 = config.get("include_hd95", False)
        metrics = evaluate_predictions(
            pred_dir=pred_dir,
            split_file=split_file,
            output_dir=str(output_base),
            split_name=split_name,
            include_hd95=include_hd95,
        )
        logger.info("Metrics [%s]: %s", split_name, metrics)

    logger.info("All inference complete. Results in %s", output_base)


if __name__ == "__main__":
    main()
