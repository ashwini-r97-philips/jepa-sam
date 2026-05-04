#!/bin/bash
#SBATCH --job-name=medsam_stage1_train
#SBATCH --partition=<partition>           # TODO: set your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/stage1_train_%j.out
#SBATCH --error=slurm_logs/stage1_train_%j.err

# ============================================================
# Stage 1: Fine-tune MedSAM with 100% training masks
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

echo "=== Stage 1: Training with 100% labels ==="
python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml

echo "=== Stage 1: Inference with best checkpoint ==="
python scripts/run_inference.py \
    --config configs/medsam_stage0_infer.yaml \
    --checkpoint outputs/stage1_100pct/checkpoints/best.pt \
    --splits val test

# Move inference outputs to stage1 directory
mkdir -p outputs/stage1_100pct/val outputs/stage1_100pct/test
# Re-run inference pointing output to stage1 dir
python scripts/run_inference.py \
    --config configs/medsam_stage1_100pct_train.yaml \
    --checkpoint outputs/stage1_100pct/checkpoints/best.pt \
    --splits val test

echo "=== Stage 1 complete ==="
echo "Checkpoints: outputs/stage1_100pct/checkpoints/"
echo "Metrics: outputs/stage1_100pct/"
