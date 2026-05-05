#!/usr/bin/env python3
"""I-JEPA pretraining on unlabeled TN3K images using SAM's ViT-B encoder.

Usage:
    python scripts/pretrain_jepa.py --config configs/jepa_pretrain.yaml
    python scripts/pretrain_jepa.py --config configs/jepa_pretrain.yaml --limit 10 --epochs 5
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
from src.data.jepa_masking import JEPAMaskCollator, collate_jepa_masks
from src.models.jepa_model import JEPAForSAM
from src.models.medsam_loader import load_medsam_model
from src.utils.io import load_yaml, ensure_dir
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="I-JEPA pretraining on TN3K unlabeled data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (smoke test).")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    return parser.parse_args()


def pad_indices(index_list: list, max_len: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length."""
    padded = []
    for idx in index_list:
        if len(idx) < max_len:
            pad = torch.full((max_len - len(idx),), pad_value, dtype=idx.dtype, device=idx.device)
            padded.append(torch.cat([idx, pad]))
        else:
            padded.append(idx[:max_len])
    return torch.stack(padded)


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

    # Build JEPA model
    jepa_model = JEPAForSAM(
        sam_encoder=sam_encoder,
        predictor_embed_dim=config.get("predictor_embed_dim", 384),
        predictor_depth=config.get("predictor_depth", 6),
        predictor_num_heads=config.get("predictor_num_heads", 12),
        ema_momentum_start=config.get("ema_momentum_start", 0.996),
        ema_momentum_end=config.get("ema_momentum_end", 1.0),
    ).to(device)

    n_ctx_params = sum(p.numel() for p in jepa_model.context_encoder.parameters() if p.requires_grad)
    n_pred_params = sum(p.numel() for p in jepa_model.predictor.parameters() if p.requires_grad)
    logger.info("JEPA: context_encoder=%d, predictor=%d trainable params", n_ctx_params, n_pred_params)

    # Mask collator
    mask_collator = JEPAMaskCollator(
        grid_size=config.get("grid_size", 64),
        num_pred_masks=config.get("num_pred_masks", 4),
        pred_mask_scale=tuple(config.get("pred_mask_scale", [0.15, 0.2])),
        enc_mask_scale=tuple(config.get("enc_mask_scale", [0.85, 1.0])),
    )

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

    # Optimizer (context encoder + predictor; target encoder is EMA-only)
    optimizer = AdamW(
        [
            {"params": jepa_model.context_encoder.parameters(), "lr": config.get("encoder_lr", 1e-4)},
            {"params": jepa_model.predictor.parameters(), "lr": config.get("predictor_lr", 1e-4)},
        ],
        weight_decay=config.get("weight_decay", 0.05),
        betas=(0.9, 0.95),
    )
    total_steps = epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    log_file = output_dir / "logs" / "pretrain_log.jsonl"
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        jepa_model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            images = batch["image"].to(device)
            B = images.shape[0]

            # Generate masks for this batch
            context_masks, target_masks = collate_jepa_masks(B, mask_collator, device)

            # Pad context indices to same length
            max_ctx = max(len(c) for c in context_masks)
            context_indices = pad_indices(context_masks, max_ctx)

            # Flatten and pad target indices
            # Combine all target block indices per sample
            target_flat = []
            for tgt_list in target_masks:
                combined = torch.cat(tgt_list) if tgt_list else torch.zeros(1, dtype=torch.long, device=device)
                target_flat.append(combined)
            max_tgt = max(len(t) for t in target_flat)
            target_indices = pad_indices(target_flat, max_tgt)

            # Forward
            loss = jepa_model(images, context_indices, target_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            momentum = jepa_model.get_ema_momentum(global_step, total_steps)
            jepa_model.update_target_encoder(momentum)

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "lr": current_lr,
            "momentum": round(momentum, 6),
            "elapsed_s": round(elapsed, 1),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %d/%d  loss=%.5f  lr=%.2e  mom=%.4f  (%.1fs)",
                        epoch, epochs, avg_loss, current_lr, momentum, elapsed)

        # Save best target encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": {
                    k: v.cpu() for k, v in jepa_model.target_encoder.state_dict().items()
                },
                "loss": best_loss,
            }, str(output_dir / "checkpoints" / "best_encoder.pt"))

        # Periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": {
                    k: v.cpu() for k, v in jepa_model.target_encoder.state_dict().items()
                },
                "context_encoder_state_dict": {
                    k: v.cpu() for k, v in jepa_model.context_encoder.state_dict().items()
                },
                "predictor_state_dict": {
                    k: v.cpu() for k, v in jepa_model.predictor.state_dict().items()
                },
                "loss": avg_loss,
            }, str(output_dir / "checkpoints" / f"checkpoint_epoch{epoch}.pt"))

    # Final save
    torch.save({
        "epoch": epochs,
        "encoder_state_dict": {
            k: v.cpu() for k, v in jepa_model.target_encoder.state_dict().items()
        },
        "loss": avg_loss,
    }, str(output_dir / "checkpoints" / "last_encoder.pt"))

    logger.info("JEPA pretraining complete. Best loss=%.5f", best_loss)
    logger.info("Target encoder saved to: %s/checkpoints/best_encoder.pt", output_dir)


if __name__ == "__main__":
    main()
