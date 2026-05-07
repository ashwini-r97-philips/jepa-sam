#!/bin/bash
# Full experiment run script for Jarvislabs A100 40GB
# Runs all stages sequentially: pretrain → finetune → inference for MAE+JEPA, then baselines.
#
# Usage (background, survives browser/terminal close):
#   nohup bash online_gpu_full_run.sh > run_master.log 2>&1 &
#
# Monitor:
#   tail -f run_master.log
#
# Check if still running:
#   ps aux | grep online_gpu_full_run

set -euo pipefail

# Reduce CUDA memory fragmentation (important for large encoders on 40GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LOG_DIR="outputs/run_logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_stage() {
    local name="$1"
    local cmd="$2"
    local logfile="$LOG_DIR/${name}.log"
    log "=== START: $name ==="
    if eval "$cmd" 2>&1 | tee "$logfile"; then
        log "=== DONE:  $name ==="
    else
        log "=== FAILED: $name (see $logfile) ==="
        exit 1
    fi
}

# ============================================================
# STAGE 3: MAE Pretrain → Finetune → Inference
# ============================================================

run_stage "stage3_mae_pretrain" \
    "python scripts/pretrain_mae.py --config configs/mae_pretrain.yaml"

run_stage "stage3_mae_finetune" \
    "python scripts/finetune_after_pretrain.py --config configs/stage3_mae_finetune.yaml"

run_stage "stage3_mae_inference" \
    "python scripts/run_inference.py \
        --config configs/stage3_mae_finetune.yaml \
        --checkpoint outputs/stage3_mae_finetune/checkpoints/best.pt \
        --splits val test"

# ============================================================
# STAGE 4: JEPA Pretrain → Finetune → Inference
# ============================================================

run_stage "stage4_jepa_pretrain" \
    "python scripts/pretrain_jepa.py --config configs/jepa_pretrain.yaml"

run_stage "stage4_jepa_finetune" \
    "python scripts/finetune_after_pretrain.py --config configs/stage4_jepa_finetune.yaml"

run_stage "stage4_jepa_inference" \
    "python scripts/run_inference.py \
        --config configs/stage4_jepa_finetune.yaml \
        --checkpoint outputs/stage4_jepa_finetune/checkpoints/best.pt \
        --splits val test"

# ============================================================
# STAGE 0: Baseline MedSAM inference (no training)
# ============================================================

run_stage "stage0_inference" \
    "python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml"

# ============================================================
# STAGE 1: Fine-tune with 100% labels → Inference
# ============================================================

run_stage "stage1_train" \
    "python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml"

run_stage "stage1_inference" \
    "python scripts/run_inference.py \
        --config configs/medsam_stage1_100pct_train.yaml \
        --checkpoint outputs/stage1_100pct/checkpoints/best.pt \
        --splits val test"

# ============================================================
# STAGE 2: Fine-tune with 10% labels → Inference
# ============================================================

run_stage "stage2_train" \
    "python scripts/train_medsam.py --config configs/medsam_stage2_10pct_train.yaml"

run_stage "stage2_inference" \
    "python scripts/run_inference.py \
        --config configs/medsam_stage2_10pct_train.yaml \
        --checkpoint outputs/stage2_10pct/checkpoints/best.pt \
        --splits val test"

# ============================================================
log "=== ALL STAGES COMPLETE ==="
log "Results in: outputs/"
