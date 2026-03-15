# RFM: GPT-2 Residual Feature Mapping with SAE

This project collects GPT-2 layer activations, trains a Sparse Autoencoder (SAE), and produces token-level feature mapping reports for interpretation.

## Quick Workflow

1. Activation extraction: `runner.py`
2. SAE training: `train_runner.py`
3. Feature mapping (token + strength): `sae/mapping.py`
4. Visualization reports: `report_plots.py`

## Setup

```bash
uv sync
```

## Run

```bash
# 1) Build activation chunks
python runner.py --config config.json

# 2) Train SAE
python train_runner.py --config config.json

# 3) Generate feature mapping reports
python -m sae.mapping --config config.json

# 4) Generate mapping visualizations
python report_plots.py --mode mapping --output-dir reports/feature_mapping/viz
```

## Main Outputs

- `reports/feature_mapping/feature_mapping_events.csv`
- `reports/feature_mapping/feature_mapping_feature_summary.csv`
- `reports/feature_mapping/feature_mapping_feature_summary_token_pairs.csv`
- `reports/feature_mapping/feature_mapping_summary.txt`

## Config Notes

Feature mapping settings are controlled under `feature-mapping` in `config.json`:

- `model_path`
- `device`
- `tokenizer_name`
- `top_k`
- `strength_threshold`
- `count`
