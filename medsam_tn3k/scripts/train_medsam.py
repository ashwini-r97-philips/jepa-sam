#!/usr/bin/env python3
"""Fine-tune MedSAM on TN3K with supervised segmentation loss.

Usage:
    # Stage 1: 100% labels
    python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml

    # Stage 2: 10% labels
    python scripts/train_medsam.py --config configs/medsam_stage2_10pct_train.yaml

    # Smoke test
    python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml --limit 5 --epochs 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tn3k_dataset import TN3KDataset
from src.models.medsam_loader import load_medsam_model
from src.training.trainer import MedSAMTrainer
from src.utils.io import load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM on TN3K.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per split (smoke test).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    setup_logging(config.get("log_level", "INFO"))
    seed_everything(config.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Override config values from CLI
    if args.epochs is not None:
        config["epochs"] = args.epochs

    # Load model
    model = load_medsam_model(
        checkpoint_path=config["checkpoint_path"],
        device=device,
        freeze_image_encoder=config.get("freeze_image_encoder", True),
    )

    # Datasets
    train_dataset = TN3KDataset(
        split_file=config["train_split"],
        boxes_file=config["boxes_file"],
        mode="train_supervised",
        limit=args.limit,
    )
    val_dataset = TN3KDataset(
        split_file=config["val_split"],
        boxes_file=config["boxes_file"],
        mode="train_supervised",
        limit=args.limit,
    )

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Train
    trainer = MedSAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
    )
    trainer.train()

    logger.info("Training complete. Best checkpoint: %s/checkpoints/best.pt",
                config["output_dir"])
    logger.info("Run inference with:")
    logger.info("  python scripts/run_inference.py --config %s --checkpoint %s/checkpoints/best.pt",
                args.config.replace("train", "infer") if "train" in args.config else args.config,
                config["output_dir"])


if __name__ == "__main__":
    main()
