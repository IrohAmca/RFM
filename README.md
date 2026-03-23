# RFM: Residual Feature Mapping with SAE

Research pipeline for analyzing LLM internal representations using Sparse Autoencoders (SAEs). Extracts layer activations, trains SAEs, produces token-level feature maps, and supports emotion feature discovery + steering experiments.

## Setup

```bash
uv sync
```

For contributor tooling:

```bash
uv sync --extra dev
```

## Project Structure

```
rfm/                  # Main Python package
├── config.py         # ConfigManager — JSON/TOML merge, dot-notation, validation
├── layout.py         # Layer-scoped path conventions
├── extractors/       # TransformerLens + HuggingFace CausalLM backends
├── data/             # Dataset loading and activation chunk management
├── sae/              # SAE model, training (dead feature analysis), mapping
├── steering/         # Feature steering hooks, activation patching, emotion probe
└── viz/              # Training metrics and mapping visualizations

cli/                  # CLI entry points
├── extract.py        # Activation extraction
├── train.py          # SAE training (sparsity sweep support)
├── pipeline.py       # End-to-end pipeline
└── steer.py          # Feature steering, patching, emotion discovery

configs/models/       # Per-model JSON config files
runs/<model>/         # Model outputs (activations, checkpoints, reports)
tests/                # 45 unit tests
```

## Usage

### Full Pipeline

```bash
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json --skip-viz
```

### Step by Step

```bash
# 1) Extract activations (multi-layer supported)
python -m cli.extract --config configs/models/gpt2-small.emotion.json

# 2) Train SAE
python -m cli.train --config configs/models/gpt2-small.emotion.json

# 3) Run feature mapping
python -m rfm.sae.mapping --config configs/models/gpt2-small.emotion.json

# 4) Generate visualizations
python -m rfm.viz.plots --mode mapping --config configs/models/gpt2-small.emotion.json
python -m rfm.viz.plots --mode all --config configs/models/gpt2-small.emotion.json
```

### Feature Steering

```bash
# Discover which features correlate with which emotions
python -m cli.steer discover --config configs/models/gpt2-small.emotion.json

# Amplify or suppress a feature during generation
python -m cli.steer steer \
  --config configs/models/gpt2-small.emotion.json \
  --layer blocks.11.hook_resid_post \
  --feature-id 4231 --alpha 5.0 \
  --prompt "I feel very"

# Causal validation via activation patching
python -m cli.steer patch \
  --config configs/models/gpt2-small.emotion.json \
  --layer blocks.11.hook_resid_post \
  --clean "I feel very happy today" \
  --patch "I feel very sad today"
```

## Configuration

Each model gets its own config file; all outputs are isolated under `runs/<model>/`.

**Multi-layer extraction** — set `extraction.target` to a list:

```json
"extraction": {
  "target": [
    "blocks.0.hook_resid_post",
    "blocks.6.hook_resid_post",
    "blocks.11.hook_resid_post"
  ]
}
```

Outputs are automatically organized per layer:

```
runs/gpt2-small/activations/blocks_6_hook_resid_post/
runs/gpt2-small/checkpoints/blocks_6_hook_resid_post/sae.pt
runs/gpt2-small/reports/feature_mapping/blocks_6_hook_resid_post/
```

**Supported backends:**
- `transformer_lens` (default) — GPT-2 and other TransformerLens-compatible models
- `hf` — Any HuggingFace CausalLM (`"extractor_backend": "hf"` in config)

## Tests

```bash
uv run pytest -q
```
