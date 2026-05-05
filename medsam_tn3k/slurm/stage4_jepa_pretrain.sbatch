#!/bin/bash
#SBATCH --job-name=jepa_pretrain
#SBATCH --partition=<partition>           # TODO: set your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/stage4_jepa_pretrain_%j.out
#SBATCH --error=slurm_logs/stage4_jepa_pretrain_%j.err

# ============================================================
# Stage 4, Part 1: I-JEPA pretraining on 90% unlabeled data
# ============================================================

set -euo pipefail

# TODO: Activate your environment
# conda activate medsam2_lora

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm_logs

echo "=== Stage 4: I-JEPA Pretraining ==="
python scripts/pretrain_jepa.py --config configs/jepa_pretrain.yaml

echo "=== JEPA pretraining complete ==="
echo "Target encoder checkpoint: outputs/stage4_jepa_pretrain/checkpoints/best_encoder.pt"
