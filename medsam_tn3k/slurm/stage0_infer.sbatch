#!/bin/bash
#SBATCH --job-name=medsam_stage0_infer
#SBATCH --partition=<partition>           # TODO: set your partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/stage0_infer_%j.out
#SBATCH --error=slurm_logs/stage0_infer_%j.err

# ============================================================
# Stage 0: Initial MedSAM inference (baseline, no fine-tuning)
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

echo "=== Stage 0: Data preparation ==="

# Download dataset (idempotent)
python scripts/download_tn3k.py --out_dir data/raw/TN3K --repo_id haifan-gong/TN3K

# Prepare splits (idempotent)
python scripts/prepare_splits.py --data_dir data/raw/TN3K --out_dir data/splits/seed0 --seed 0

# Compute oracle boxes (idempotent)
python scripts/prepare_boxes.py --split_dir data/splits/seed0

echo "=== Stage 0: Running initial MedSAM inference ==="
python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml

echo "=== Stage 0 complete ==="
echo "Results: outputs/stage0_initial_infer/"
