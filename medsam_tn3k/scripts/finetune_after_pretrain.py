#!/usr/bin/env python3
"""Fine-tune SAM with a pretrained encoder (from MAE or JEPA) using differential learning rates.

Shared script for Stage 3 (MAE fine-tune) and Stage 4 (JEPA fine-tune).
Loads a pretrained encoder checkpoint into SAM, then fine-tunes with:
  - encoder_lr: low LR to preserve SSL-learned features
  - decoder_lr: high LR to adapt decoder to shifted embeddings

Usage:
    python scripts/finetune_after_pretrain.py --config configs/stage3_mae_finetune.yaml
    python scripts/finetune_after_pretrain.py --config configs/stage4_jepa_finetune.yaml
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
from src.models.encoder_utils import load_encoder_checkpoint, load_pretrained_encoder_into_sam
from src.models.medsam_loader import load_medsam_model
from src.training.finetune_trainer import DiffLRFineTuner
from src.utils.io import load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAM after SSL pretraining.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (smoke test).")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    setup_logging(config.get("log_level", "INFO"))
    seed_everything(config.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if args.epochs:
        config["epochs"] = args.epochs

    # Load base SAM model
    logger.info("Loading MedSAM model...")
    sam_model = load_medsam_model(
        checkpoint_path=config["medsam_checkpoint"],
        device=device,
        freeze_image_encoder=False,
    )

    # Load pretrained encoder weights
    pretrained_path = config["pretrained_encoder_checkpoint"]
    logger.info("Loading pretrained encoder from: %s", pretrained_path)
    encoder_sd = load_encoder_checkpoint(pretrained_path, device=device)
    load_pretrained_encoder_into_sam(sam_model, encoder_sd)

    # Datasets
    train_dataset = TN3KDataset(
        split_file=config["train_split"],
        mode="train_supervised",
        image_size=config.get("image_size", 1024),
        mask_size=config.get("mask_size", 256),
        augment=True,
        limit=args.limit,
    )
    val_dataset = TN3KDataset(
        split_file=config["val_split"],
        mode="inference",
        image_size=config.get("image_size", 1024),
        mask_size=config.get("mask_size", 256),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    logger.info("Train: %d samples, Val: %d samples", len(train_dataset), len(val_dataset))

    # Run fine-tuning with differential LR trainer
    finetuner = DiffLRFineTuner(
        model=sam_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
    )
    finetuner.train()

    logger.info("Fine-tuning complete. Best model at: %s/checkpoints/best.pt", config["output_dir"])


if __name__ == "__main__":
    main()
