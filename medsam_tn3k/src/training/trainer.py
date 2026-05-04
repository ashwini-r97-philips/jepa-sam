"""MedSAM fine-tuning trainer."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.training.losses import DiceBCELoss

logger = logging.getLogger(__name__)


class MedSAMTrainer:
    """Supervised fine-tuning loop for MedSAM prompt decoder.

    The image encoder is frozen; only the prompt encoder and mask decoder
    are updated (or just the mask decoder, depending on config).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        self.device = next(model.parameters()).device
        self.epochs = config.get("epochs", 50)
        self.lr = config.get("lr", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)

        # Only optimize unfrozen parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable))
        self.optimizer = AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = DiceBCELoss()

        self.log_file = self.output_dir / "logs" / "train_log.jsonl"
        self.best_val_loss = float("inf")

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "elapsed_s": round(elapsed, 1),
            }
            logger.info("Epoch %d/%d  train_loss=%.5f  val_loss=%.5f  (%.1fs)",
                        epoch, self.epochs, train_loss, val_loss, elapsed)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Save last checkpoint
            self._save_checkpoint(self.output_dir / "checkpoints" / "last.pt", epoch)

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(self.output_dir / "checkpoints" / "best.pt", epoch)
                logger.info("New best model at epoch %d (val_loss=%.5f)", epoch, val_loss)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        # Keep image encoder in eval mode (frozen)
        self.model.image_encoder.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            box = batch["box"].to(self.device)

            # Forward
            with torch.no_grad():
                image_embedding = self.model.image_encoder(image)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=box, masks=None,
            )
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            loss = self.criterion(low_res_logits, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            box = batch["box"].to(self.device)

            image_embedding = self.model.image_encoder(image)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=box, masks=None,
            )
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            loss = self.criterion(low_res_logits, mask)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, str(path))
