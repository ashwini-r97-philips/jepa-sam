# MAE sweep (assumes pretraining already ran with checkpoint_epochs saving)
python scripts/run_ssl_checkpoint_sweep.py \
  --method mae \
  --ssl-config configs/mae_pretrain.yaml \
  --finetune-config configs/sweep_finetune_10pct.yaml \
  --checkpoint-epochs 0 1 2 5 10 20 50 \
  --output-dir results/ssl_sweeps/mae_run_001 \
  --skip-ssl-train

# JEPA sweep
python scripts/run_ssl_checkpoint_sweep.py \
  --method jepa \
  --ssl-config configs/jepa_pretrain.yaml \
  --finetune-config configs/sweep_finetune_10pct.yaml \
  --checkpoint-epochs 0 1 2 5 10 20 50 \
  --output-dir results/ssl_sweeps/jepa_run_001 \
  --skip-ssl-train

# Smoke test (5 samples, 2 epochs)
python scripts/run_ssl_checkpoint_sweep.py \
  --method mae \
  --ssl-config configs/mae_pretrain.yaml \
  --finetune-config configs/sweep_finetune_10pct.yaml \
  --checkpoint-epochs 0 1 \
  --output-dir results/ssl_sweeps/test_run \
  --skip-ssl-train --limit 5