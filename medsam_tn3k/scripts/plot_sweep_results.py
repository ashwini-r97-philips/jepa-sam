#!/usr/bin/env python3
"""Generate diagnostic plots from SSL checkpoint sweep results.

Usage:
    python scripts/plot_sweep_results.py --results results/ssl_sweeps/mae_run_001/sweep_results.jsonl
    python scripts/plot_sweep_results.py --results results/ssl_sweeps/mae_run_001/sweep_results.jsonl --output-dir results/ssl_sweeps/mae_run_001/plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_plots(jsonl_path: str, output_dir: str) -> None:
    """Generate all diagnostic plots from sweep JSONL results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.sweep_results import load_sweep_rows

    rows = load_sweep_rows(jsonl_path)
    if not rows:
        print("No results to plot.")
        return

    # Filter out errored rows
    rows = [r for r in rows if r.get("error") is None]
    if not rows:
        print("All rows have errors, nothing to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [r["ssl_epoch"] for r in rows]
    method = rows[0].get("method", "SSL")

    # --- Plot 1: SSL epoch vs zero-shot Dice ---
    zs_dice = [r.get("zero_shot_dice_mean") for r in rows]
    if any(v is not None for v in zs_dice):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = [(e, d) for e, d in zip(epochs, zs_dice) if d is not None]
        if valid:
            es, ds = zip(*valid)
            ax.plot(es, ds, "o-", color="tab:red", linewidth=2, markersize=8)
            ax.set_xlabel("SSL Pretraining Epoch", fontsize=12)
            ax.set_ylabel("Zero-Shot Dice (mean)", fontsize=12)
            ax.set_title(f"{method.upper()}: SSL Epoch vs Zero-Shot Dice", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(output_dir / "ssl_epoch_vs_zeroshot_dice.png", dpi=150)
        plt.close(fig)

    # --- Plot 2: SSL epoch vs downstream Dice ---
    ds_dice = [r.get("dice_mean") for r in rows]
    if any(v is not None for v in ds_dice):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = [(e, d) for e, d in zip(epochs, ds_dice) if d is not None]
        if valid:
            es, ds_vals = zip(*valid)
            ax.plot(es, ds_vals, "s-", color="tab:blue", linewidth=2, markersize=8)
            ax.set_xlabel("SSL Pretraining Epoch", fontsize=12)
            ax.set_ylabel("Downstream Dice (mean)", fontsize=12)
            ax.set_title(f"{method.upper()}: SSL Epoch vs Downstream Dice (10% finetune)", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(output_dir / "ssl_epoch_vs_downstream_dice.png", dpi=150)
        plt.close(fig)

    # --- Plot 3: SSL epoch vs downstream IoU ---
    ds_iou = [r.get("iou_mean") for r in rows]
    if any(v is not None for v in ds_iou):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = [(e, d) for e, d in zip(epochs, ds_iou) if d is not None]
        if valid:
            es, ious = zip(*valid)
            ax.plot(es, ious, "^-", color="tab:green", linewidth=2, markersize=8)
            ax.set_xlabel("SSL Pretraining Epoch", fontsize=12)
            ax.set_ylabel("Downstream IoU (mean)", fontsize=12)
            ax.set_title(f"{method.upper()}: SSL Epoch vs Downstream IoU (10% finetune)", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(output_dir / "ssl_epoch_vs_downstream_iou.png", dpi=150)
        plt.close(fig)

    # --- Plot 4: SSL loss vs downstream Dice ---
    ssl_loss = [r.get("ssl_train_loss") for r in rows]
    if any(v is not None for v in ssl_loss) and any(v is not None for v in ds_dice):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = [(l, d) for l, d in zip(ssl_loss, ds_dice) if l is not None and d is not None]
        if valid:
            losses, dices = zip(*valid)
            ax.plot(losses, dices, "D-", color="tab:purple", linewidth=2, markersize=8)
            ax.set_xlabel("SSL Training Loss", fontsize=12)
            ax.set_ylabel("Downstream Dice (mean)", fontsize=12)
            ax.set_title(f"{method.upper()}: SSL Loss vs Downstream Dice", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # lower loss → right side
            fig.tight_layout()
            fig.savefig(output_dir / "ssl_loss_vs_downstream_dice.png", dpi=150)
        plt.close(fig)

    # --- Combined plot: zero-shot + downstream Dice on same axes ---
    if any(v is not None for v in zs_dice) and any(v is not None for v in ds_dice):
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_zs = [(e, d) for e, d in zip(epochs, zs_dice) if d is not None]
        valid_ds = [(e, d) for e, d in zip(epochs, ds_dice) if d is not None]
        if valid_zs:
            es, ds_vals = zip(*valid_zs)
            ax.plot(es, ds_vals, "o--", color="tab:red", linewidth=2, markersize=8,
                    label="Zero-shot Dice")
        if valid_ds:
            es, ds_vals = zip(*valid_ds)
            ax.plot(es, ds_vals, "s-", color="tab:blue", linewidth=2, markersize=8,
                    label="Downstream Dice (10% finetune)")
        ax.set_xlabel("SSL Pretraining Epoch", fontsize=12)
        ax.set_ylabel("Dice (mean)", fontsize=12)
        ax.set_title(f"{method.upper()}: Zero-Shot vs Downstream Dice", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(output_dir / "combined_zeroshot_vs_downstream.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SSL checkpoint sweep results.")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to sweep_results.jsonl file.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: sibling 'plots/' dir).")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(args.results).parent / "plots")

    generate_plots(args.results, output_dir)


if __name__ == "__main__":
    main()
