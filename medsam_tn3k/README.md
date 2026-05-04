# MedSAM + TN3K Semi-Supervised Experiment Scaffold

First-stage experiment: supervised baselines for MedSAM fine-tuning on TN3K thyroid nodule segmentation, using oracle bounding boxes.

## Experiment Overview

| Stage | Model | Training labels | Unlabeled data used? | Prompt | Output |
|---|---|---|---|---|---|
| 0 | Initial MedSAM | none | no | oracle box | baseline inference |
| 1 | Fine-tuned MedSAM | 100% masks | no | oracle box | upper-bound supervised |
| 2 | Fine-tuned MedSAM | 10% masks | no | oracle box | low-label baseline |

## Directory Structure

```
medsam_tn3k/
├── configs/                          # YAML experiment configs
│   ├── default.yaml
│   ├── medsam_stage0_infer.yaml
│   ├── medsam_stage1_100pct_train.yaml
│   └── medsam_stage2_10pct_train.yaml
├── scripts/                          # Runnable scripts
│   ├── download_tn3k.py
│   ├── prepare_splits.py
│   ├── prepare_boxes.py
│   ├── run_inference.py
│   ├── train_medsam.py
│   └── evaluate_predictions.py
├── slurm/                            # SLURM batch scripts
│   ├── stage0_infer.sbatch
│   ├── stage1_train_100pct.sbatch
│   └── stage2_train_10pct.sbatch
├── src/                              # Library code
│   ├── data/
│   │   ├── tn3k_dataset.py           # PyTorch Dataset
│   │   ├── transforms.py             # Image/mask transforms
│   │   ├── split_utils.py            # Split logic
│   │   └── box_utils.py              # Oracle box computation
│   ├── models/
│   │   └── medsam_loader.py          # MedSAM loading + inference
│   ├── training/
│   │   ├── trainer.py                # Fine-tuning loop
│   │   └── losses.py                 # Dice + BCE loss
│   ├── evaluation/
│   │   ├── metrics.py                # Dice, IoU, HD95
│   │   └── evaluator.py              # Evaluation orchestrator
│   └── utils/
│       ├── io.py                     # JSON/YAML I/O
│       ├── seed.py                   # Reproducible seeding
│       └── logging.py                # Logging setup
├── outputs/                          # Experiment outputs (git-ignored)
├── data/                             # Data directory (git-ignored)
│   ├── raw/TN3K/                     # Downloaded dataset
│   └── splits/seed0/                 # Split files
├── checkpoints/                      # Model checkpoints (git-ignored)
└── requirements.txt
```

## Environment Setup

```bash
cd medsam_tn3k/

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### MedSAM Checkpoint

Download the MedSAM ViT-B checkpoint and place it at:

```
medsam_tn3k/checkpoints/medsam_vit_b.pth
```

Download link: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fax37

The `segment-anything` package is required:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Quick Start (Local)

All commands assume you are in the `medsam_tn3k/` directory.

### 1. Download TN3K Dataset

```bash
python scripts/download_tn3k.py --out_dir data/raw/TN3K --repo_id haifan-gong/TN3K
```

### 2. Create Splits

```bash
python scripts/prepare_splits.py --data_dir data/raw/TN3K --out_dir data/splits/seed0 --seed 0
```

This creates:
- `data/splits/seed0/train.json` — full training set (70%)
- `data/splits/seed0/val.json` — validation set (10%)
- `data/splits/seed0/test.json` — test set (20%)
- `data/splits/seed0/train_labeled_10pct.json` — 10% labeled subset
- `data/splits/seed0/train_unlabeled_90pct.json` — 90% unlabeled (reserved for JEPA/SSL)

### 3. Compute Oracle Bounding Boxes

```bash
python scripts/prepare_boxes.py --split_dir data/splits/seed0
```

### 4. Stage 0: Baseline Inference (Initial MedSAM)

```bash
python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml
```

Smoke test with 5 samples:
```bash
python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml --limit 5
```

### 5. Stage 1: Fine-tune with 100% Labels

```bash
# Train
python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml

# Inference with best checkpoint
python scripts/run_inference.py \
    --config configs/medsam_stage1_100pct_train.yaml \
    --checkpoint outputs/stage1_100pct/checkpoints/best.pt
```

### 6. Stage 2: Fine-tune with 10% Labels

```bash
# Train
python scripts/train_medsam.py --config configs/medsam_stage2_10pct_train.yaml

# Inference with best checkpoint
python scripts/run_inference.py \
    --config configs/medsam_stage2_10pct_train.yaml \
    --checkpoint outputs/stage2_10pct/checkpoints/best.pt
```

## SLURM Execution

Edit the `.sbatch` files to set your partition, modules, and environment activation, then:

```bash
# Stage 0: data prep + baseline inference
sbatch slurm/stage0_infer.sbatch

# Stage 1: 100% supervised training + inference
sbatch slurm/stage1_train_100pct.sbatch

# Stage 2: 10% supervised training + inference
sbatch slurm/stage2_train_10pct.sbatch
```

## Expected Outputs

```
outputs/
├── stage0_initial_infer/
│   ├── val/                           # Predicted masks
│   ├── test/
│   ├── metrics_val.json
│   ├── metrics_test.json
│   ├── metrics_val_per_case.csv
│   └── metrics_test_per_case.csv
├── stage1_100pct/
│   ├── checkpoints/best.pt
│   ├── checkpoints/last.pt
│   ├── logs/train_log.jsonl
│   ├── val/
│   ├── test/
│   ├── metrics_val.json
│   └── metrics_test.json
└── stage2_10pct/
    ├── checkpoints/best.pt
    ├── checkpoints/last.pt
    ├── logs/train_log.jsonl
    ├── val/
    ├── test/
    ├── metrics_val.json
    └── metrics_test.json
```

## Results Template

| Stage | Split | Dice (mean±std) | IoU (mean±std) | HD95 (mean±std) |
|---|---|---|---|---|
| 0 (initial) | val | — | — | — |
| 0 (initial) | test | — | — | — |
| 1 (100% labels) | val | — | — | — |
| 1 (100% labels) | test | — | — | — |
| 2 (10% labels) | val | — | — | — |
| 2 (10% labels) | test | — | — | — |

## Smoke Test

Run a quick test with limited samples to verify the pipeline:

```bash
# Test data loading and box computation
python scripts/prepare_boxes.py --split_dir data/splits/seed0 --limit 5

# Test inference (requires checkpoint + GPU)
python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml --limit 5

# Test training loop (requires checkpoint + GPU)
python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml --limit 5 --epochs 2
```

## Notes

- **Oracle boxes only**: All bounding boxes are computed from ground-truth masks. No jittered/noisy boxes.
- **No SSL yet**: The 90% unlabeled split (`train_unlabeled_90pct.json`) is prepared but not used. It is reserved for future JEPA/SSL work.
- **Frozen image encoder**: During fine-tuning, only the prompt encoder and mask decoder are updated.
- **HD95 is optional**: Requires `scipy` and `medpy`. Enable via `include_hd95: true` in config.
- **All scripts are idempotent**: Re-running will skip completed steps.
