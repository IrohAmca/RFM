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

For new analysis and dashboard features:

```bash
uv sync --extra autointerp --extra dashboard --extra dev
```

## Project Structure

```
rfm/                  # Main Python package
├── config.py         # ConfigManager — JSON/TOML merge, dot-notation, validation
├── layout.py         # Layer-scoped path conventions
├── extractors/       # TransformerLens + HuggingFace CausalLM backends
├── data/             # Dataset loading and activation chunk management
├── sae/              # SAE models (Vanilla, TopK, Gated), training (dead feature analysis), mapping
├── analysis/         # LLM Autointerp and Feature Clustering
├── dashboard/        # Streamlit interactive analysis dashboard
├── steering/         # Feature steering hooks, activation patching, emotion probe
└── viz/              # Training metrics and mapping visualizations

cli/                  # CLI entry points
├── extract.py        # Activation extraction
├── train.py          # SAE training (sparsity sweep support)
├── pipeline.py       # End-to-end pipeline
├── steer.py          # Feature steering, patching, emotion discovery
└── analyze.py        # Autointerp and Clustering CLI

configs/models/       # Per-model JSON config files
runs/<model>/         # Model outputs (activations, checkpoints, reports)
tests/                # Unit tests
```

## Usage

### Full Pipeline

```bash
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json --skip-viz
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json --skip-extract --skip-train
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json --from-step mapping
uv run python -m cli.pipeline --config configs/models/qwen3-0.6B.safety.json --from-hook 27
```

### Step by Step

```bash
# 1) Extract activations (multi-layer supported)
python -m cli.extract --config configs/models/gpt2-small.emotion.json

# 2) Train SAE
python -m cli.train --config configs/models/gpt2-small.emotion.json

# 3) Run feature mapping
python -m rfm.sae.mapping --config configs/models/gpt2-small.emotion.json

# 4) Run LLM Auto-interpretation
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json --top-n 50

# 5) Generate interactive dashboard
uv run streamlit run rfm/dashboard/app.py

# 6) Generate static visual reports
uv run python -m rfm.viz.plots --mode all --config configs/models/gpt2-small.emotion.json
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
uv run python -m cli.steer patch \
  --config configs/models/gpt2-small.emotion.json \
  --layer blocks.11.hook_resid_post \
  --clean "I feel very happy today" \
  --patch "I feel very sad today"
```

### Advanced Analysis

```bash
# Feature Clustering (Cosine similarity between feature decoders)
uv run python -m cli.analyze cluster --config configs/models/gpt2-small.emotion.json

# Run analysis only for one layer
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json --layer blocks.11.hook_resid_post

# Use Groq OpenAI-compatible endpoint with rate-limit friendly pacing
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json --groq --model llama-3.1-8b-instant --request-delay 11

# Restart autointerp from scratch (ignore existing partial JSON)
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json --no-resume

```

API key and base URL requirements for autointerp:

- Default endpoint is OpenAI. Provide key via `OPENAI_API_KEY` or `--api-key`.
- For Groq, use `--groq` and provide key via `GROQ_API_KEY` (or `--api-key`).
- For any OpenAI-compatible provider, use `--base-url <provider_url>` and `--api-key <provider_key>`.

Examples:

```bash
# OpenAI (default endpoint)
set OPENAI_API_KEY=your_openai_key
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json

# Groq shortcut (uses https://api.groq.com/openai/v1)
set GROQ_API_KEY=your_groq_key
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json --groq --model llama-3.1-8b-instant

# Generic OpenAI-compatible endpoint
uv run python -m cli.analyze autointerp --config configs/models/gpt2-small.emotion.json --base-url https://your-provider.example/v1 --api-key your_provider_key
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

To resume a multi-layer pipeline from a later hook without re-running earlier hooks, use `--from-hook`.
It accepts either the exact target string or just the layer index, for example `--from-hook blocks.27.hook_resid_post` or `--from-hook 27`.

**SAE Architectures:**
By default, the pipeline trains a Vanilla Sparse Autoencoder with L1 penalty. You can enable state-of-the-art architectures in the configuration:
- `vanilla` (Standard L1 SAE)
- `topk` (Anthropic's Top-K sparsity, better reconstruction)
- `gated` (DeepMind's Gated SAE, decoupled magnitude and gating)

```json
"sae": {
  "architecture": "topk",
  "topk_k": 32,
  "hidden_dim": 3072,
  "sparsity_weight": 0.0
}
```

**Supported backends:**
- `transformer_lens` (default) — GPT-2 and other TransformerLens-compatible models
- `hf` — Any HuggingFace CausalLM (`"extractor_backend": "hf"` in config)

## Tests

```bash
uv run pytest -q
```
