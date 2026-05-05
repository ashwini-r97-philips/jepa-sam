#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --partition=<partition>           # TODO: set your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/stage3_mae_pretrain_%j.out
#SBATCH --error=slurm_logs/stage3_mae_pretrain_%j.err

# ============================================================
# Stage 3, Part 1: MAE pretraining on 90% unlabeled data
# ============================================================

set -euo pipefail

# TODO: Activate your environment
# conda activate medsam2_lora

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm_logs

echo "=== Stage 3: MAE Pretraining ==="
python scripts/pretrain_mae.py --config configs/mae_pretrain.yaml

echo "=== MAE pretraining complete ==="
echo "Encoder checkpoint: outputs/stage3_mae_pretrain/checkpoints/best_encoder.pt"
