"""Masked Autoencoder (MAE) adapted for SAM's ViT-B encoder.

Architecture:
- Encoder: SAM's ViT-B image encoder (1024x1024 input, 64x64 patch grid = 4096 patches)
- Decoder: Lightweight transformer for pixel reconstruction
- Mask ratio: 75% random uniform
- Loss: MSE on masked patches with per-patch normalization

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder for MAE pixel reconstruction."""

    def __init__(
        self,
        num_patches: int = 4096,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        patch_size: int = 16,
        in_chans: int = 3,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size

        # Project encoder output to decoder dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sincos positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )

        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Prediction head: project to pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        # Initialize sincos pos embed
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_embed_dim, int(self.num_patches ** 0.5)
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N_visible, encoder_embed_dim) encoded visible patches.
            ids_restore: (B, N_total) indices to unshuffle patches.

        Returns:
            pred: (B, N_total, patch_size^2 * 3) pixel predictions for all patches.
        """
        # Project to decoder dim
        x = self.decoder_embed(x)

        # Append mask tokens for masked positions
        B, N_vis, D = x.shape
        N_total = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N_total - N_vis, -1)
        x_ = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle to original positions
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore_expanded)

        # Add positional embedding
        x_ = x_ + self.decoder_pos_embed

        # Transformer blocks
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)

        # Predict pixels
        pred = self.decoder_pred(x_)
        return pred


class TransformerBlock(nn.Module):
    """Simple transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEForSAM(nn.Module):
    """Masked Autoencoder using SAM's ViT-B as the encoder.

    This wraps SAM's image_encoder for MAE pretraining. The encoder
    processes only visible (unmasked) patches, and a lightweight decoder
    reconstructs the masked patch pixels.

    Args:
        sam_encoder: SAM's image_encoder module (ViT-B).
        decoder_embed_dim: Decoder hidden dimension.
        decoder_depth: Number of decoder transformer blocks.
        decoder_num_heads: Decoder attention heads.
        mask_ratio: Fraction of patches to mask.
        norm_pix_loss: Per-patch normalize reconstruction target.
    """

    def __init__(
        self,
        sam_encoder: nn.Module,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.sam_encoder = sam_encoder
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # SAM ViT-B specifics
        self.patch_size = 16
        self.img_size = 1024
        self.grid_size = self.img_size // self.patch_size  # 64
        self.num_patches = self.grid_size * self.grid_size  # 4096
        self.encoder_embed_dim = 768  # SAM ViT-B embed dim

        # Decoder
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            encoder_embed_dim=256,  # SAM encoder neck outputs 256-dim
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            patch_size=self.patch_size,
            in_chans=3,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patch sequences.

        Args:
            imgs: (B, 3, H, W)

        Returns:
            patches: (B, N, patch_size^2 * 3)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.img_size
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def random_masking(
        self, B: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random mask for the patch grid.

        Returns:
            ids_keep: (B, N_visible) indices of visible patches.
            ids_restore: (B, N) indices to unshuffle.
            mask: (B, N) binary mask (1=masked, 0=visible).
        """
        N = self.num_patches
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones(B, N, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward_encoder(
        self, imgs: torch.Tensor, ids_keep: torch.Tensor
    ) -> torch.Tensor:
        """Run SAM encoder on full image and extract visible patch embeddings.

        Note: SAM's encoder doesn't support partial-patch input natively.
        We run the full encoder and then select visible patch positions from
        the spatial output.

        Args:
            imgs: (B, 3, 1024, 1024)
            ids_keep: (B, N_visible) indices of visible patches.

        Returns:
            visible_embeddings: (B, N_visible, 256) encoder features at visible positions.
        """
        # SAM encoder outputs (B, 256, 64, 64) after the neck
        features = self.sam_encoder(imgs)  # (B, 256, 64, 64)

        # Reshape to patch sequence: (B, 4096, 256)
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Select visible patches
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, C)
        visible = torch.gather(features_flat, dim=1, index=ids_keep_expanded)

        return visible

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss on masked patches only.

        Args:
            imgs: (B, 3, 1024, 1024) original images.
            pred: (B, N, patch_size^2*3) predictions.
            mask: (B, N) binary mask (1=masked).
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # average over masked patches
        return loss

    def forward(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full MAE forward pass.

        Returns:
            loss: Reconstruction loss on masked patches.
            pred: (B, N, patch_size^2*3) predictions.
            mask: (B, N) binary mask.
        """
        B = imgs.shape[0]
        device = imgs.device

        ids_keep, ids_restore, mask = self.random_masking(B, device)
        visible = self.forward_encoder(imgs, ids_keep)
        pred = self.decoder(visible, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> "np.ndarray":
    """Generate 2D sincos positional embedding."""
    import numpy as np

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1).T  # (N, 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 0])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: "np.ndarray") -> "np.ndarray":
    """Generate 1D sincos positional embedding from positions."""
    import numpy as np

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
