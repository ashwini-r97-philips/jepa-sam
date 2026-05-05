#!/usr/bin/env python3
"""MAE pretraining on unlabeled TN3K images using SAM's ViT-B encoder.

Usage:
    python scripts/pretrain_mae.py --config configs/mae_pretrain.yaml
    python scripts/pretrain_mae.py --config configs/mae_pretrain.yaml --limit 10 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ssl_dataset import SSLDataset
from src.models.encoder_utils import extract_encoder_state_dict
from src.models.mae_model import MAEForSAM
from src.models.medsam_loader import load_medsam_model
from src.utils.io import load_yaml, ensure_dir
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE pretraining on TN3K unlabeled data.")
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

    epochs = args.epochs or config.get("epochs", 300)
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")

    # Load SAM model to get the encoder
    logger.info("Loading SAM encoder from MedSAM checkpoint...")
    sam_model = load_medsam_model(
        checkpoint_path=config["medsam_checkpoint"],
        device=device,
        freeze_image_encoder=False,
    )
    sam_encoder = sam_model.image_encoder

    # Build MAE model
    mae_model = MAEForSAM(
        sam_encoder=sam_encoder,
        decoder_embed_dim=config.get("decoder_embed_dim", 512),
        decoder_depth=config.get("decoder_depth", 4),
        decoder_num_heads=config.get("decoder_num_heads", 8),
        mask_ratio=config.get("mask_ratio", 0.75),
        norm_pix_loss=config.get("norm_pix_loss", True),
    ).to(device)

    n_params = sum(p.numel() for p in mae_model.parameters() if p.requires_grad)
    logger.info("MAE model: %d trainable parameters", n_params)

    # Dataset
    dataset = SSLDataset(
        split_file=config["unlabeled_split"],
        image_size=config.get("image_size", 1024),
        augment=True,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    logger.info("Dataset: %d images, %d batches/epoch", len(dataset), len(loader))

    # Optimizer
    optimizer = AdamW(
        mae_model.parameters(),
        lr=config.get("lr", 1.5e-4),
        weight_decay=config.get("weight_decay", 0.05),
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    log_file = output_dir / "logs" / "pretrain_log.jsonl"
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        mae_model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            images = batch["image"].to(device)
            loss, _, _ = mae_model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "lr": current_lr,
            "elapsed_s": round(elapsed, 1),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %d/%d  loss=%.5f  lr=%.2e  (%.1fs)",
                        epoch, epochs, avg_loss, current_lr, elapsed)

        # Save best encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            encoder_sd = extract_encoder_state_dict(sam_model)
            # SAM encoder is sam_encoder which is mae_model.sam_encoder — same object
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": {
                    k: v.cpu() for k, v in mae_model.sam_encoder.state_dict().items()
                },
                "loss": best_loss,
            }, str(output_dir / "checkpoints" / "best_encoder.pt"))

        # Save last encoder every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": {
                    k: v.cpu() for k, v in mae_model.sam_encoder.state_dict().items()
                },
                "loss": avg_loss,
            }, str(output_dir / "checkpoints" / f"encoder_epoch{epoch}.pt"))

    # Final save
    torch.save({
        "epoch": epochs,
        "encoder_state_dict": {
            k: v.cpu() for k, v in mae_model.sam_encoder.state_dict().items()
        },
        "loss": avg_loss,
    }, str(output_dir / "checkpoints" / "last_encoder.pt"))

    logger.info("MAE pretraining complete. Best loss=%.5f", best_loss)
    logger.info("Encoder saved to: %s/checkpoints/best_encoder.pt", output_dir)


if __name__ == "__main__":
    main()
