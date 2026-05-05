#!/bin/bash
#SBATCH --job-name=mae_finetune
#SBATCH --partition=<partition>           # TODO: set your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/stage3_mae_finetune_%j.out
#SBATCH --error=slurm_logs/stage3_mae_finetune_%j.err

# ============================================================
# Stage 3, Part 2: Fine-tune with MAE-pretrained encoder
# Uses differential LR (encoder_lr=1e-5, decoder_lr=1e-4)
# ============================================================

set -euo pipefail

# TODO: Activate your environment
# conda activate medsam2_lora

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm_logs

echo "=== Stage 3: MAE Fine-tuning (Differential LR) ==="
python scripts/finetune_after_pretrain.py --config configs/stage3_mae_finetune.yaml

echo "=== Stage 3: Inference ==="
python scripts/run_inference.py \
    --config configs/stage3_mae_finetune.yaml \
    --checkpoint outputs/stage3_mae_finetune/checkpoints/best_model.pt \
    --splits val test

echo "=== Stage 3 MAE fine-tuning complete ==="
echo "Best model: outputs/stage3_mae_finetune/checkpoints/best_model.pt"
