#!/usr/bin/env python3
"""SSL Checkpoint Sweep: evaluate multiple SSL pretraining epochs on downstream segmentation.

For each SSL checkpoint epoch:
  1. Zero-shot evaluation (SSL encoder injected into MedSAM, no finetuning)
  2. Downstream finetuning (10% labels, fixed hyperparameters)
  3. Downstream evaluation on test split

Epoch 0 is always the MedSAM baseline (no SSL encoder loaded).

Usage:
    python scripts/run_ssl_checkpoint_sweep.py \
        --method mae \
        --ssl-config configs/mae_pretrain.yaml \
        --finetune-config configs/sweep_finetune_10pct.yaml \
        --checkpoint-epochs 0 1 2 5 10 20 50 \
        --output-dir results/ssl_sweeps/mae_run_001

    python scripts/run_ssl_checkpoint_sweep.py \
        --method jepa \
        --ssl-config configs/jepa_pretrain.yaml \
        --finetune-config configs/sweep_finetune_10pct.yaml \
        --checkpoint-epochs 0 1 2 5 10 20 50 \
        --output-dir results/ssl_sweeps/jepa_run_001
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tn3k_dataset import TN3KDataset
from src.evaluation.evaluator import evaluate_predictions
from src.models.encoder_utils import load_encoder_checkpoint, load_pretrained_encoder_into_sam
from src.models.medsam_loader import load_medsam_model, medsam_inference
from src.training.finetune_trainer import DiffLRFineTuner
from src.utils.io import load_yaml, save_json, ensure_dir
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything
from src.utils.sweep_results import (
    save_sweep_row,
    compile_sweep_summary,
    save_config_snapshot,
    load_sweep_rows,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SSL checkpoint sweep: evaluate pretraining duration vs downstream performance."
    )
    parser.add_argument("--method", type=str, required=True, choices=["mae", "jepa"],
                        help="SSL method (mae or jepa).")
    parser.add_argument("--ssl-config", type=str, required=True,
                        help="Path to SSL pretraining YAML config.")
    parser.add_argument("--finetune-config", type=str, required=True,
                        help="Path to downstream finetune YAML config.")
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+", required=True,
                        help="SSL pretraining epochs to evaluate (0 = MedSAM baseline).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for sweep results.")
    parser.add_argument("--skip-ssl-train", action="store_true",
                        help="Only use existing checkpoints; do not run SSL training.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples for smoke testing.")
    parser.add_argument("--eval-split", type=str, default="test",
                        help="Split to evaluate on (default: test).")
    return parser.parse_args()


def _get_ssl_checkpoint_path(ssl_config: Dict, epoch: int) -> Path:
    """Resolve path to an SSL encoder checkpoint for a given epoch."""
    ssl_output_dir = Path(ssl_config["output_dir"])
    return ssl_output_dir / "checkpoints" / f"encoder_epoch{epoch}.pt"


def _get_ssl_log(ssl_config: Dict) -> Optional[Dict[int, Dict]]:
    """Load SSL training log and index by epoch."""
    log_path = Path(ssl_config["output_dir"]) / "logs" / "pretrain_log.jsonl"
    if not log_path.exists():
        return None
    epoch_log = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                epoch_log[entry["epoch"]] = entry
    return epoch_log


def _run_zero_shot_eval(
    model: torch.nn.Module,
    finetune_config: Dict,
    output_dir: Path,
    eval_split: str,
    device: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run zero-shot inference + evaluation (no finetuning)."""
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    split_dir = finetune_config["split_dir"]
    boxes_file = finetune_config["boxes_file"]
    split_file = str(Path(split_dir) / f"{eval_split}.json")

    pred_dir = output_dir / "zeroshot" / eval_split
    ensure_dir(pred_dir)

    dataset = TN3KDataset(
        split_file=split_file,
        boxes_file=boxes_file,
        mode="inference",
        limit=limit,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Zero-shot inference", leave=False):
            image = batch["image"]
            box = batch["box"]
            image_ids = batch["image_id"]
            original_sizes = batch["original_size"]

            pred_masks = medsam_inference(model, image, box)

            for i in range(pred_masks.shape[0]):
                mask_256 = pred_masks[i, 0].cpu().numpy().astype(np.uint8) * 255
                orig_h = original_sizes[0][i].item()
                orig_w = original_sizes[1][i].item()
                mask_pil = Image.fromarray(mask_256).resize((orig_w, orig_h), Image.NEAREST)
                mask_pil.save(pred_dir / f"{image_ids[i]}.png")

    metrics = evaluate_predictions(
        pred_dir=str(pred_dir),
        split_file=split_file,
        output_dir=str(output_dir / "zeroshot"),
        split_name=eval_split,
        include_hd95=False,
    )
    return metrics


def _run_downstream_finetune(
    model: torch.nn.Module,
    finetune_config: Dict,
    output_dir: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run downstream finetuning and return trainer info."""
    train_dataset = TN3KDataset(
        split_file=finetune_config["train_split"],
        boxes_file=finetune_config["boxes_file"],
        mode="train_supervised",
        limit=limit,
    )
    val_dataset = TN3KDataset(
        split_file=finetune_config["val_split"],
        boxes_file=finetune_config["boxes_file"],
        mode="inference",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=finetune_config.get("batch_size", 2),
        shuffle=True,
        num_workers=finetune_config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=finetune_config.get("batch_size", 2),
        shuffle=False,
        num_workers=finetune_config.get("num_workers", 4),
        pin_memory=True,
    )

    ft_output_dir = str(output_dir / "finetune")
    ft_config = dict(finetune_config)
    ft_config["output_dir"] = ft_output_dir

    finetuner = DiffLRFineTuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=ft_config,
        output_dir=ft_output_dir,
    )
    finetuner.train()

    return {
        "finetune_best_epoch": finetuner.best_epoch,
        "finetune_best_val_loss": finetuner.best_val_loss,
        "finetune_train_samples": len(train_dataset),
        "finetune_val_samples": len(val_dataset),
        "checkpoint_path": str(Path(ft_output_dir) / "checkpoints" / "best.pt"),
    }


def _run_downstream_eval(
    checkpoint_path: str,
    finetune_config: Dict,
    output_dir: Path,
    eval_split: str,
    device: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Load finetuned checkpoint and evaluate on test."""
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    model = load_medsam_model(
        checkpoint_path=checkpoint_path,
        device=device,
        freeze_image_encoder=True,
    )

    split_dir = finetune_config["split_dir"]
    boxes_file = finetune_config["boxes_file"]
    split_file = str(Path(split_dir) / f"{eval_split}.json")

    pred_dir = output_dir / "downstream" / eval_split
    ensure_dir(pred_dir)

    dataset = TN3KDataset(
        split_file=split_file,
        boxes_file=boxes_file,
        mode="inference",
        limit=limit,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Downstream inference", leave=False):
            image = batch["image"]
            box = batch["box"]
            image_ids = batch["image_id"]
            original_sizes = batch["original_size"]

            pred_masks = medsam_inference(model, image, box)

            for i in range(pred_masks.shape[0]):
                mask_256 = pred_masks[i, 0].cpu().numpy().astype(np.uint8) * 255
                orig_h = original_sizes[0][i].item()
                orig_w = original_sizes[1][i].item()
                mask_pil = Image.fromarray(mask_256).resize((orig_w, orig_h), Image.NEAREST)
                mask_pil.save(pred_dir / f"{image_ids[i]}.png")

    metrics = evaluate_predictions(
        pred_dir=str(pred_dir),
        split_file=split_file,
        output_dir=str(output_dir / "downstream"),
        split_name=eval_split,
        include_hd95=False,
    )
    return metrics


def _epoch_already_completed(output_dir: Path, epoch: int, jsonl_path: Path) -> bool:
    """Check if this epoch has already been completed (for resume support)."""
    if not jsonl_path.exists():
        return False
    rows = load_sweep_rows(jsonl_path)
    return any(r.get("ssl_epoch") == epoch and r.get("error") is None for r in rows)


def run_single_checkpoint(
    method: str,
    ssl_epoch: int,
    ssl_config: Dict,
    finetune_config: Dict,
    output_dir: Path,
    eval_split: str,
    device: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline for a single SSL checkpoint epoch."""
    epoch_dir = output_dir / f"epoch_{ssl_epoch}"
    ensure_dir(epoch_dir)

    row: Dict[str, Any] = {
        "method": method,
        "ssl_epoch": ssl_epoch,
        "ssl_train_loss": None,
        "ssl_val_loss": None,
        "ssl_lr": None,
        "ssl_mask_ratio": ssl_config.get("mask_ratio"),
        "ssl_checkpoint_path": None,
    }

    # Get SSL training log info for this epoch
    ssl_log = _get_ssl_log(ssl_config)
    if ssl_log and ssl_epoch in ssl_log:
        entry = ssl_log[ssl_epoch]
        row["ssl_train_loss"] = entry.get("loss")
        row["ssl_lr"] = entry.get("lr")

    # --- Step 1: Load model with or without SSL encoder ---
    logger.info("Loading MedSAM base model...")
    sam_model = load_medsam_model(
        checkpoint_path=finetune_config["medsam_checkpoint"],
        device=device,
        freeze_image_encoder=False,
    )

    if ssl_epoch == 0:
        # Baseline: no SSL encoder, use MedSAM as-is
        row["ssl_checkpoint_path"] = "baseline (MedSAM)"
        logger.info("Epoch 0: using MedSAM baseline (no SSL encoder)")
    else:
        # Load SSL encoder checkpoint
        ckpt_path = _get_ssl_checkpoint_path(ssl_config, ssl_epoch)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SSL checkpoint not found: {ckpt_path}. "
                f"Run pretraining first or use --skip-ssl-train to skip missing checkpoints."
            )
        row["ssl_checkpoint_path"] = str(ckpt_path)
        logger.info("Loading SSL encoder from: %s", ckpt_path)
        encoder_sd = load_encoder_checkpoint(str(ckpt_path), device=device)
        load_pretrained_encoder_into_sam(sam_model, encoder_sd)

    # --- Step 2: Zero-shot evaluation ---
    logger.info("Running zero-shot evaluation (epoch %d)...", ssl_epoch)
    zs_metrics = _run_zero_shot_eval(
        model=sam_model,
        finetune_config=finetune_config,
        output_dir=epoch_dir,
        eval_split=eval_split,
        device=device,
        limit=limit,
    )
    row["zero_shot_dice_mean"] = zs_metrics.get("dice_mean")
    row["zero_shot_dice_std"] = zs_metrics.get("dice_std")
    row["zero_shot_iou_mean"] = zs_metrics.get("iou_mean")
    row["zero_shot_iou_std"] = zs_metrics.get("iou_std")
    row["zero_shot_n_cases"] = zs_metrics.get("n_cases")
    logger.info("Zero-shot [epoch %d]: Dice=%.5f, IoU=%.5f",
                ssl_epoch, zs_metrics.get("dice_mean", 0), zs_metrics.get("iou_mean", 0))

    # --- Step 3: Downstream finetuning ---
    logger.info("Running downstream finetuning (epoch %d)...", ssl_epoch)
    # Need to reload model fresh for finetuning (zero-shot eval may have changed state)
    sam_model_ft = load_medsam_model(
        checkpoint_path=finetune_config["medsam_checkpoint"],
        device=device,
        freeze_image_encoder=False,
    )
    if ssl_epoch > 0:
        ckpt_path = _get_ssl_checkpoint_path(ssl_config, ssl_epoch)
        encoder_sd = load_encoder_checkpoint(str(ckpt_path), device=device)
        load_pretrained_encoder_into_sam(sam_model_ft, encoder_sd)

    ft_info = _run_downstream_finetune(
        model=sam_model_ft,
        finetune_config=finetune_config,
        output_dir=epoch_dir,
        limit=limit,
    )
    row["finetune_train_samples"] = ft_info["finetune_train_samples"]
    row["finetune_val_samples"] = ft_info["finetune_val_samples"]
    row["finetune_best_epoch"] = ft_info["finetune_best_epoch"]
    row["finetune_best_val_loss"] = ft_info["finetune_best_val_loss"]

    # --- Step 4: Downstream evaluation ---
    logger.info("Running downstream evaluation (epoch %d)...", ssl_epoch)
    ds_metrics = _run_downstream_eval(
        checkpoint_path=ft_info["checkpoint_path"],
        finetune_config=finetune_config,
        output_dir=epoch_dir,
        eval_split=eval_split,
        device=device,
        limit=limit,
    )
    row["dice_mean"] = ds_metrics.get("dice_mean")
    row["dice_std"] = ds_metrics.get("dice_std")
    row["dice_median"] = ds_metrics.get("dice_median")
    row["iou_mean"] = ds_metrics.get("iou_mean")
    row["iou_std"] = ds_metrics.get("iou_std")
    row["iou_median"] = ds_metrics.get("iou_median")
    row["n_cases"] = ds_metrics.get("n_cases")
    logger.info("Downstream [epoch %d]: Dice=%.5f, IoU=%.5f",
                ssl_epoch, ds_metrics.get("dice_mean", 0), ds_metrics.get("iou_mean", 0))

    return row


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    seed_everything(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    ssl_config = load_yaml(args.ssl_config)
    finetune_config = load_yaml(args.finetune_config)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    jsonl_path = output_dir / "sweep_results.jsonl"

    # Save config snapshot
    save_config_snapshot(
        configs={
            "ssl_config": ssl_config,
            "finetune_config": finetune_config,
            "cli_args": vars(args),
        },
        output_dir=output_dir,
        extra={
            "method": args.method,
            "checkpoint_epochs": args.checkpoint_epochs,
            "device": device,
        },
    )

    checkpoint_epochs = sorted(args.checkpoint_epochs)
    logger.info("Sweep: method=%s, epochs=%s, output=%s", args.method, checkpoint_epochs, output_dir)

    for ssl_epoch in checkpoint_epochs:
        # Resume support: skip already-completed epochs
        if _epoch_already_completed(output_dir, ssl_epoch, jsonl_path):
            logger.info("Skipping epoch %d (already completed)", ssl_epoch)
            continue

        logger.info("=" * 60)
        logger.info("Processing SSL epoch %d / %s", ssl_epoch, checkpoint_epochs)
        logger.info("=" * 60)

        try:
            row = run_single_checkpoint(
                method=args.method,
                ssl_epoch=ssl_epoch,
                ssl_config=ssl_config,
                finetune_config=finetune_config,
                output_dir=output_dir,
                eval_split=args.eval_split,
                device=device,
                limit=args.limit,
            )
            row["error"] = None
        except FileNotFoundError as e:
            if args.skip_ssl_train:
                logger.warning("Skipping epoch %d: %s", ssl_epoch, e)
                row = {
                    "method": args.method,
                    "ssl_epoch": ssl_epoch,
                    "error": str(e),
                }
            else:
                raise
        except Exception as e:
            logger.error("Failed at epoch %d: %s", ssl_epoch, e)
            logger.error(traceback.format_exc())
            row = {
                "method": args.method,
                "ssl_epoch": ssl_epoch,
                "error": str(e),
            }

        # Append result
        save_sweep_row(row, jsonl_path)
        logger.info("Saved result for epoch %d", ssl_epoch)

    # Compile summary
    logger.info("Compiling sweep summary...")
    compile_sweep_summary(jsonl_path, output_dir)

    # Generate plots
    try:
        from scripts.plot_sweep_results import generate_plots
        generate_plots(str(jsonl_path), str(output_dir / "plots"))
        logger.info("Plots saved to %s/plots/", output_dir)
    except Exception as e:
        logger.warning("Plot generation failed (non-fatal): %s", e)

    logger.info("Sweep complete! Results at: %s", output_dir)


if __name__ == "__main__":
    main()
