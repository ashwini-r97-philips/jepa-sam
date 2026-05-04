cd medsam_tn3k/

# Data prep (run once)
python scripts/download_tn3k.py
python scripts/prepare_splits.py
python scripts/prepare_boxes.py

# Stage 0: baseline inference
python scripts/run_inference.py --config configs/medsam_stage0_infer.yaml

# Stage 1: train 100% → infer
python scripts/train_medsam.py --config configs/medsam_stage1_100pct_train.yaml
python scripts/run_inference.py --config configs/medsam_stage1_100pct_train.yaml --checkpoint outputs/stage1_100pct/checkpoints/best.pt

# Stage 2: train 10% → infer
python scripts/train_medsam.py --config configs/medsam_stage2_10pct_train.yaml
python scripts/run_inference.py --config configs/medsam_stage2_10pct_train.yaml --checkpoint outputs/stage2_10pct/checkpoints/best.pt