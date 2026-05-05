"""Block masking for I-JEPA on a 2D patch grid.

Generates non-overlapping context and target masks for the JEPA objective.
Context: large contiguous block (e.g., 85-100% of patches).
Target: multiple small blocks (e.g., 4 blocks at 15-20% scale each).

Reference: Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture" (CVPR 2023)
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch


class JEPAMaskCollator:
    """Generates block masks for I-JEPA pretraining.

    For a 64x64 patch grid (from 1024x1024 image with patch_size=16):
    - Context mask: indices of patches the encoder sees
    - Target masks: indices of patches the predictor must predict

    Args:
        grid_size: Patch grid dimension (64 for 1024px image with p=16).
        num_pred_masks: Number of target blocks to predict.
        pred_mask_scale: (min, max) scale for target block area as fraction of total.
        enc_mask_scale: (min, max) scale for context block area.
        aspect_ratio: (min, max) aspect ratio for blocks.
        min_keep: Minimum number of context patches to keep.
    """

    def __init__(
        self,
        grid_size: int = 64,
        num_pred_masks: int = 4,
        pred_mask_scale: Tuple[float, float] = (0.15, 0.2),
        enc_mask_scale: Tuple[float, float] = (0.85, 1.0),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        min_keep: int = 10,
    ):
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_pred_masks = num_pred_masks
        self.pred_mask_scale = pred_mask_scale
        self.enc_mask_scale = enc_mask_scale
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep

    def _sample_block(
        self, scale: Tuple[float, float], aspect_ratio: Tuple[float, float]
    ) -> Tuple[int, int, int, int]:
        """Sample a random block within the grid.

        Returns:
            (top, left, height, width) of the block.
        """
        # Target area
        area = self.num_patches
        target_area = random.uniform(scale[0], scale[1]) * area

        # Aspect ratio
        log_ratio = (math.log(aspect_ratio[0]), math.log(aspect_ratio[1]))
        ar = math.exp(random.uniform(*log_ratio))

        h = int(round(math.sqrt(target_area * ar)))
        w = int(round(math.sqrt(target_area / ar)))

        h = min(h, self.grid_size)
        w = min(w, self.grid_size)
        h = max(h, 1)
        w = max(w, 1)

        top = random.randint(0, self.grid_size - h)
        left = random.randint(0, self.grid_size - w)

        return top, left, h, w

    def _block_to_indices(self, top: int, left: int, h: int, w: int) -> List[int]:
        """Convert block coordinates to flat patch indices."""
        indices = []
        for r in range(top, top + h):
            for c in range(left, left + w):
                indices.append(r * self.grid_size + c)
        return indices

    def __call__(self) -> Tuple[List[int], List[List[int]]]:
        """Generate context and target masks for one sample.

        Returns:
            context_indices: List of patch indices the encoder receives.
            target_indices: List of lists (one per target block).
        """
        # Generate target blocks
        target_indices_all: List[List[int]] = []
        target_set = set()

        for _ in range(self.num_pred_masks):
            for _attempt in range(10):
                top, left, h, w = self._sample_block(
                    self.pred_mask_scale, self.aspect_ratio
                )
                indices = self._block_to_indices(top, left, h, w)
                # Check no overlap with existing targets
                if not target_set.intersection(indices):
                    target_indices_all.append(indices)
                    target_set.update(indices)
                    break

        # Context = everything NOT in target
        all_indices = set(range(self.num_patches))
        context_set = all_indices - target_set

        # Optionally subsample context for enc_mask_scale < 1.0
        # (by default enc_mask_scale is high so we keep most context)
        max_context = int(self.enc_mask_scale[1] * self.num_patches)
        context_indices = sorted(context_set)
        if len(context_indices) > max_context:
            context_indices = sorted(random.sample(context_indices, max_context))

        # Ensure minimum context
        if len(context_indices) < self.min_keep:
            context_indices = sorted(random.sample(list(all_indices), self.min_keep))

        return context_indices, target_indices_all


def collate_jepa_masks(
    batch_size: int,
    collator: JEPAMaskCollator,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
    """Generate masks for a batch.

    Returns:
        context_masks: List of (N_ctx_i,) tensors with context indices per sample.
        target_masks: List of lists of (N_tgt_j,) tensors per sample per target block.
    """
    context_masks = []
    target_masks = []
    for _ in range(batch_size):
        ctx, tgts = collator()
        context_masks.append(torch.tensor(ctx, dtype=torch.long, device=device))
        target_masks.append([
            torch.tensor(t, dtype=torch.long, device=device) for t in tgts
        ])
    return context_masks, target_masks
