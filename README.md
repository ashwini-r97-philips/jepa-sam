# jepa-sam

Exploring whether Joint Embedding Predictive Architecture (JEPA) can be infused into Segment Anything Model (SAM) based training pipelines to improve segmentation, particularly in low-label and semi-supervised regimes.

The core idea: use JEPA's self-supervised representation learning to boost SAM/MedSAM fine-tuning when labeled data is scarce — learning from unlabeled images via predictive embeddings, then transferring that knowledge into the segmentation decoder.

## Repository Structure

Each subdirectory contains a self-contained experiment with its own README detailing the specific use case, setup, and results.

| Folder | Use Case |
|---|---|
| `medsam_tn3k/` | MedSAM + TN3K thyroid nodule segmentation — supervised baselines and semi-supervised scaffolding |