"""Differential LR fine-tuning trainer for pretrained encoder + MedSAM decoder.

Used after MAE or JEPA pretraining to fine-tune the full model on labeled data
with different learning rates for encoder vs decoder.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.training.losses import DiceBCELoss

logger = logging.getLogger(__name__)


class DiffLRFineTuner:
    """Fine-tuning trainer with differential learning rates.

    Encoder uses a lower LR to preserve pretrained representations.
    Decoder/prompt encoder uses a higher LR to adapt to the (possibly shifted)
    embedding space.

    Args:
        model: Full SAM model with pretrained encoder loaded.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Configuration dict with encoder_lr, decoder_lr, epochs, patience, etc.
        output_dir: Where to save checkpoints and logs.
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
        self.epochs = config.get("epochs", 30)
        self.encoder_lr = config.get("encoder_lr", 1e-5)
        # Support separate prompt_encoder / mask_decoder LRs; fall back to decoder_lr
        default_decoder_lr = config.get("decoder_lr", 1e-4)
        self.prompt_encoder_lr = config.get("prompt_encoder_lr", default_decoder_lr)
        self.mask_decoder_lr = config.get("mask_decoder_lr", default_decoder_lr)
        self.grad_accum_steps = config.get("grad_accum_steps", 1)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.patience = config.get("patience", 10)

        # Build parameter groups with differential LR
        encoder_params = list(model.image_encoder.parameters())
        prompt_encoder_params = list(model.prompt_encoder.parameters())
        mask_decoder_params = list(model.mask_decoder.parameters())

        # Unfreeze encoder for fine-tuning
        for p in encoder_params:
            p.requires_grad = True

        n_encoder = sum(p.numel() for p in encoder_params if p.requires_grad)
        n_prompt = sum(p.numel() for p in prompt_encoder_params if p.requires_grad)
        n_mask_dec = sum(p.numel() for p in mask_decoder_params if p.requires_grad)
        logger.info("Fine-tune params: encoder=%d (lr=%.1e), prompt_encoder=%d (lr=%.1e), mask_decoder=%d (lr=%.1e)",
                    n_encoder, self.encoder_lr, n_prompt, self.prompt_encoder_lr, n_mask_dec, self.mask_decoder_lr)

        self.optimizer = AdamW([
            {"params": encoder_params, "lr": self.encoder_lr},
            {"params": prompt_encoder_params, "lr": self.prompt_encoder_lr},
            {"params": mask_decoder_params, "lr": self.mask_decoder_lr},
        ], weight_decay=self.weight_decay)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        self.criterion = DiceBCELoss()

        self.log_file = self.output_dir / "logs" / "train_log.jsonl"
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train(self) -> None:
        epochs_no_improve = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            elapsed = time.time() - t0

            enc_lr = self.optimizer.param_groups[0]["lr"]
            prompt_lr = self.optimizer.param_groups[1]["lr"]
            mask_dec_lr = self.optimizer.param_groups[2]["lr"]
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "encoder_lr": enc_lr,
                "prompt_encoder_lr": prompt_lr,
                "mask_decoder_lr": mask_dec_lr,
                "elapsed_s": round(elapsed, 1),
            }
            logger.info("Epoch %d/%d  train=%.5f  val=%.5f  enc_lr=%.2e  prompt_lr=%.2e  dec_lr=%.2e  (%.1fs)",
                        epoch, self.epochs, train_loss, val_loss, enc_lr, prompt_lr, mask_dec_lr, elapsed)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            self.scheduler.step()

            # Save last
            self._save_checkpoint(self.output_dir / "checkpoints" / "last.pt", epoch)

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_no_improve = 0
                self._save_checkpoint(self.output_dir / "checkpoints" / "best.pt", epoch)
                logger.info("New best at epoch %d (val_loss=%.5f)", epoch, val_loss)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, self.patience)
                break

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            box = batch["box"].to(self.device)

            # Full forward through encoder (gradients flow)
            image_embedding = self.model.image_encoder(image)

            # Per-sample decoder loop (SAM decoder limitation)
            logits_list = []
            for i in range(image.shape[0]):
                sparse_emb, dense_emb = self.model.prompt_encoder(
                    points=None, boxes=box[i:i+1], masks=None,
                )
                logits_i, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding[i:i+1],
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                logits_list.append(logits_i)
            low_res_logits = torch.cat(logits_list, dim=0)

            loss = self.criterion(low_res_logits, mask) / self.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
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

            logits_list = []
            for i in range(image.shape[0]):
                sparse_emb, dense_emb = self.model.prompt_encoder(
                    points=None, boxes=box[i:i+1], masks=None,
                )
                logits_i, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding[i:i+1],
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                logits_list.append(logits_i)
            low_res_logits = torch.cat(logits_list, dim=0)

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
