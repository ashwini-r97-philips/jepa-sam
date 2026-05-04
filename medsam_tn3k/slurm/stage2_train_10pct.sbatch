#!/bin/bash
#SBATCH --job-name=medsam_stage2_train
#SBATCH --partition=<partition>           # TODO: set your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/stage2_train_%j.out
#SBATCH --error=slurm_logs/stage2_train_%j.err

# ============================================================
# Stage 2: Fine-tune MedSAM with 10% labeled training masks
# The 90% unlabeled set is NOT used here (reserved for JEPA/SSL)
# ============================================================

set -euo pipefail

# TODO: Adjust module loads for your cluster
# module load python/3.10
# module load cuda/11.8

# TODO: Activate your environment
# source /path/to/venv/bin/activate
# conda activate medsam_tn3k

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm_logs

echo "=== Stage 2: Training with 10% labels ==="
python scripts/train_medsam.py --config configs/medsam_stage2_10pct_train.yaml

echo "=== Stage 2: Inference with best checkpoint ==="
python scripts/run_inference.py \
    --config configs/medsam_stage2_10pct_train.yaml \
    --checkpoint outputs/stage2_10pct/checkpoints/best.pt \
    --splits val test

echo "=== Stage 2 complete ==="
echo "Checkpoints: outputs/stage2_10pct/checkpoints/"
echo "Metrics: outputs/stage2_10pct/"
