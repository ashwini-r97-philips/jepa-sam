"""I-JEPA model adapted for SAM's ViT-B encoder.

Architecture:
- Context encoder: SAM's ViT-B image_encoder, processes context patches
- Target encoder: EMA copy of context encoder, processes full image
- Predictor: Lightweight transformer that predicts target representations from context
- Loss: Smooth L1 between predicted and target representations at masked positions

Reference: Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture" (CVPR 2023)
"""
from __future__ import annotations

import copy
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class JEPAPredictor(nn.Module):
    """Lightweight transformer predictor for I-JEPA.

    Takes context encoder output at visible positions and predicts
    target representations at masked positions.

    Args:
        encoder_embed_dim: Dimension of encoder output (256 for SAM neck output).
        predictor_embed_dim: Internal predictor dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_patches: Total patches in grid (4096 for 64x64).
    """

    def __init__(
        self,
        encoder_embed_dim: int = 256,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        num_patches: int = 4096,
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.num_patches = num_patches

        # Project encoder dim to predictor dim
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Learnable mask token for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Positional embedding (fixed sincos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder dim
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        # Sincos pos embed
        from src.models.mae_model import get_2d_sincos_pos_embed
        grid_size = int(self.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.predictor_embed_dim, grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

    def forward(
        self,
        context_features: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target representations from context.

        Args:
            context_features: (B, N_ctx, encoder_embed_dim) context encoder output.
            context_indices: (B, N_ctx) indices of context patches.
            target_indices: (B, N_tgt) indices of target patches.

        Returns:
            predictions: (B, N_tgt, encoder_embed_dim) predicted target features.
        """
        B = context_features.shape[0]
        N_ctx = context_indices.shape[1]
        N_tgt = target_indices.shape[1]

        # Project context to predictor dim
        x_ctx = self.input_proj(context_features)  # (B, N_ctx, pred_dim)

        # Add positional embedding to context tokens
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        x_ctx = x_ctx + ctx_pos

        # Create mask tokens for target positions
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)
        tgt_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        x_tgt = mask_tokens + tgt_pos

        # Concatenate context + target tokens
        x = torch.cat([x_ctx, x_tgt], dim=1)  # (B, N_ctx + N_tgt, pred_dim)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Extract only target predictions
        x_pred = x[:, N_ctx:, :]  # (B, N_tgt, pred_dim)

        # Project back to encoder dim
        predictions = self.output_proj(x_pred)  # (B, N_tgt, encoder_embed_dim)
        return predictions


class PredictorBlock(nn.Module):
    """Transformer block for the predictor."""

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


class JEPAForSAM(nn.Module):
    """I-JEPA model using SAM's ViT-B encoder.

    Args:
        sam_encoder: SAM's image_encoder module.
        predictor_embed_dim: Predictor hidden dimension.
        predictor_depth: Predictor transformer depth.
        predictor_num_heads: Predictor attention heads.
        ema_momentum_start: Initial EMA momentum.
        ema_momentum_end: Final EMA momentum.
    """

    def __init__(
        self,
        sam_encoder: nn.Module,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 12,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
    ):
        super().__init__()
        self.context_encoder = sam_encoder

        # Target encoder is an EMA copy
        self.target_encoder = copy.deepcopy(sam_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # SAM encoder output dim after neck
        encoder_embed_dim = 256  # SAM ViT-B neck output
        grid_size = 64  # 1024 / 16
        num_patches = grid_size * grid_size

        self.predictor = JEPAPredictor(
            encoder_embed_dim=encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            num_patches=num_patches,
        )

        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.num_patches = num_patches

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA update of target encoder from context encoder."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

    def get_ema_momentum(self, step: int, total_steps: int) -> float:
        """Linear ramp of EMA momentum from start to end."""
        return self.ema_momentum_start + (
            self.ema_momentum_end - self.ema_momentum_start
        ) * step / max(total_steps, 1)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing JEPA loss.

        Args:
            images: (B, 3, 1024, 1024) input images.
            context_indices: (B, N_ctx) indices of context patches.
            target_indices: (B, N_tgt) indices of target patches.

        Returns:
            loss: Smooth L1 loss between predictions and targets.
        """
        B = images.shape[0]

        # Target encoder: full image → all patch features
        with torch.no_grad():
            target_features = self.target_encoder(images)  # (B, 256, 64, 64)
            target_features = target_features.permute(0, 2, 3, 1).reshape(
                B, self.num_patches, -1
            )  # (B, 4096, 256)
            # Layer normalize targets
            target_features = F.layer_norm(
                target_features, (target_features.size(-1),)
            )
            # Extract target positions
            tgt_expanded = target_indices.unsqueeze(-1).expand(
                -1, -1, target_features.size(-1)
            )
            h = torch.gather(target_features, dim=1, index=tgt_expanded)  # (B, N_tgt, 256)

        # Context encoder: full image → select context positions
        context_features = self.context_encoder(images)  # (B, 256, 64, 64)
        context_features = context_features.permute(0, 2, 3, 1).reshape(
            B, self.num_patches, -1
        )  # (B, 4096, 256)
        # Extract context positions
        ctx_expanded = context_indices.unsqueeze(-1).expand(
            -1, -1, context_features.size(-1)
        )
        z_ctx = torch.gather(context_features, dim=1, index=ctx_expanded)  # (B, N_ctx, 256)

        # Predictor: predict target features from context
        z_pred = self.predictor(z_ctx, context_indices, target_indices)  # (B, N_tgt, 256)

        # Loss: smooth L1 between predicted and target
        loss = F.smooth_l1_loss(z_pred, h)
        return loss
