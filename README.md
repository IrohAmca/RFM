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

Single-command pipeline:

```bash
python run_pipeline.py --config config.json
```

Useful flags:

- `--skip-viz`: run extraction + training + mapping only
- `--continue-on-error`: continue remaining steps even if one fails

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

## Scalable Project Design (Per Model)

Use one config per model and keep outputs isolated under `runs/<model>/`.

- Activation chunks: `runs/<model>/activations/`
- SAE checkpoints: `runs/<model>/checkpoints/`
- Mapping reports: `runs/<model>/reports/feature_mapping/`

Recommended config layout:

- `configs/models/gpt2-small.emotion.json`
- `configs/models/turkish-gpt2.emotion.json`

Run each model independently with its own config:

```bash
python runner.py --config configs/models/gpt2-small.emotion.json
python train_runner.py --config configs/models/gpt2-small.emotion.json
python -m sae.mapping --config configs/models/gpt2-small.emotion.json
```

Implementation note:

- `extractor_factory.py` centralizes extractor selection.
- `project_layout.py` centralizes model-scoped path conventions.

## Per-Model Project Layout

Use one config file per model, and keep outputs isolated by model under `runs/`.

Suggested layout:

- `configs/models/gpt2-small.emotion.json`
- `configs/models/turkish-gpt2.emotion.json`
- `runs/gpt2-small/activations`
- `runs/gpt2-small/checkpoints`
- `runs/gpt2-small/reports/feature_mapping`
- `runs/turkish-gpt2/activations`
- `runs/turkish-gpt2/checkpoints`
- `runs/turkish-gpt2/reports/feature_mapping`

Run each model with its own config:

```bash
python runner.py --config configs/models/gpt2-small.emotion.json
python train_runner.py --config configs/models/gpt2-small.emotion.json
python -m sae.mapping --config configs/models/gpt2-small.emotion.json
python report_plots.py --mode mapping --output-dir runs/gpt2-small/reports/feature_mapping/viz

python runner.py --config configs/models/turkish-gpt2.emotion.json
python train_runner.py --config configs/models/turkish-gpt2.emotion.json
python -m sae.mapping --config configs/models/turkish-gpt2.emotion.json
python report_plots.py --mode mapping --output-dir runs/turkish-gpt2/reports/feature_mapping/viz
```
