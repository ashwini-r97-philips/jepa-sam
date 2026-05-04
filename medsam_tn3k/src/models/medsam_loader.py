"""MedSAM model loader and adapter.

MedSAM is based on the Segment Anything Model (SAM) with a ViT-B image encoder,
fine-tuned on large-scale medical image datasets.

Checkpoint: Download from https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fax37 
Place at: checkpoints/medsam_vit_b.pth

Requires: segment-anything package
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_medsam_model(
    checkpoint_path: str = "checkpoints/medsam_vit_b.pth",
    device: str = "cpu",
    freeze_image_encoder: bool = True,
) -> nn.Module:
    """Load MedSAM model from checkpoint.

    Args:
        checkpoint_path: Path to medsam_vit_b.pth.
        device: Target device.
        freeze_image_encoder: If True, freeze the image encoder (default for fine-tuning).

    Returns:
        SAM model with loaded weights.
    """
    from segment_anything import sam_model_registry

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MedSAM checkpoint not found at {checkpoint_path}. "
            "Download from https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fax37 "
            "and place at checkpoints/medsam_vit_b.pth"
        )

    logger.info("Loading MedSAM model from %s", checkpoint_path)

    # Detect checkpoint format: training checkpoint vs raw state_dict
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    is_training_ckpt = isinstance(ckpt, dict) and "model_state_dict" in ckpt

    if is_training_ckpt:
        # Training checkpoint (best.pt / last.pt) — build model first, then load weights
        logger.info("Detected training checkpoint (epoch=%s)", ckpt.get("epoch", "?"))
        model = sam_model_registry["vit_b"]()
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Raw MedSAM checkpoint (medsam_vit_b.pth)
        model = sam_model_registry["vit_b"]()
        if isinstance(ckpt, dict) and all(k.startswith(("image_encoder", "prompt_encoder", "mask_decoder")) for k in list(ckpt.keys())[:3]):
            model.load_state_dict(ckpt)
        else:
            # Let SAM handle it via its built-in loader
            model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))

    model = model.to(device)

    if freeze_image_encoder:
        logger.info("Freezing image encoder")
        for param in model.image_encoder.parameters():
            param.requires_grad = False

    return model


@torch.no_grad()
def medsam_inference(
    model: nn.Module,
    image: torch.Tensor,
    box_1024: torch.Tensor,
    target_size: tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """Run MedSAM inference on a batch of images.

    Args:
        model: SAM model.
        image: (B, 3, 1024, 1024) normalized image tensor.
        box_1024: (B, 4) bounding box in 1024x1024 coordinate space.
        target_size: Output mask resolution (default 256x256).

    Returns:
        Binary mask tensor (B, 1, H, W) at target_size resolution.
    """
    device = next(model.parameters()).device
    image = image.to(device)
    box_1024 = box_1024.to(device)

    # Image embedding (batched — standard ViT)
    image_embedding = model.image_encoder(image)

    # SAM decoder expects single-image input; loop per sample
    preds = []
    for i in range(image.shape[0]):
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=box_1024[i:i+1],
            masks=None,
        )
        low_res_logits, _ = model.mask_decoder(
            image_embeddings=image_embedding[i:i+1],
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        preds.append(low_res_logits)

    low_res_logits = torch.cat(preds, dim=0)

    # Sigmoid and threshold
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = (low_res_pred > 0.5).float()
    return low_res_pred
