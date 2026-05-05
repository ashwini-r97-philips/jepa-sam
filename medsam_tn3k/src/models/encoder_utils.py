"""Utilities for extracting and loading encoder weights between SAM and pretrained models."""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_encoder_state_dict(sam_model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract the image encoder state dict from a full SAM model.

    Returns keys like 'patch_embed.proj.weight', 'blocks.0.norm1.weight', etc.
    (strips the 'image_encoder.' prefix).
    """
    state_dict = OrderedDict()
    prefix = "image_encoder."
    for key, val in sam_model.state_dict().items():
        if key.startswith(prefix):
            state_dict[key[len(prefix):]] = val
    logger.info("Extracted encoder state dict: %d keys", len(state_dict))
    return state_dict


def load_pretrained_encoder_into_sam(
    sam_model: nn.Module,
    encoder_state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> nn.Module:
    """Load pretrained encoder weights into a full SAM model's image_encoder.

    Args:
        sam_model: Full SAM model.
        encoder_state_dict: State dict with keys relative to image_encoder
            (e.g., 'blocks.0.norm1.weight', NOT 'image_encoder.blocks.0.norm1.weight').
        strict: Whether to require exact key match.

    Returns:
        The SAM model with updated encoder.
    """
    missing, unexpected = sam_model.image_encoder.load_state_dict(
        encoder_state_dict, strict=strict
    )
    if missing:
        logger.warning("Missing keys when loading encoder: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys when loading encoder: %s", unexpected[:10])
    logger.info("Loaded pretrained encoder into SAM (%d parameters)",
                sum(p.numel() for p in sam_model.image_encoder.parameters()))
    return sam_model


def load_encoder_checkpoint(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load an encoder checkpoint file.

    Handles both formats:
      - Direct state dict: {'blocks.0.norm1.weight': ...}
      - Wrapped: {'encoder_state_dict': {...}, 'epoch': ..., ...}
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        logger.info("Loading encoder from wrapped checkpoint (epoch=%s)",
                    ckpt.get("epoch", "?"))
        return ckpt["encoder_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("blocks.") for k in ckpt.keys()):
        return ckpt
    else:
        raise ValueError(
            f"Unrecognized encoder checkpoint format. Keys: {list(ckpt.keys())[:5]}"
        )
