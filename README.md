# RFM: Residual Feature Mapping with SAE

Research pipeline for analyzing and steering LLM internal representations using Sparse Autoencoders (SAEs). Supports two primary use cases:

- **General Analysis**: Extract layer activations, train SAEs, map features, run auto-interpretation and steering experiments.
- **Safety Analysis**: Identify and suppress features that cause harmful model outputs using generation-time contrastive analysis.
- **Deception Analysis**: Learn honest vs deceptive directions, probes, monitors, and static reports from paired scenario activations.

## Setup

```bash
uv sync
```

For contributor tooling:

```bash
uv sync --extra dev
```

For analysis, dashboard, and safety scoring features:

```bash
uv sync --extra autointerp --extra dashboard --extra dev
pip install scikit-learn   # required for cross-layer safety analysis
```

## Project Structure

```
rfm/                    # Main Python package
├── config.py           # ConfigManager — JSON/TOML merge, dot-notation, validation
├── layout.py           # Layer-scoped path conventions
├── extractors/         # Extraction backends
│   ├── hf_causal.py    # HuggingFace CausalLM (reading-time, batched multi-layer)
│   └── hf_generate.py  # HuggingFace generation-time, multi-layer single-pass
├── data/               # Dataset loading and activation chunk management
├── sae/                # SAE models (Vanilla, TopK, Gated), training, mapping
├── analysis/           # LLM Autointerp and Feature Clustering
├── safety/             # Safety-specific modules
│   ├── classifier.py   # Toxicity classification (HF model, LLM judge, dataset labels)
│   ├── contrastive.py  # Per-feature toxic vs safe scoring (rate ratio, Fisher score)
│   └── cross_layer.py  # Cross-layer combination analysis + interpretable classifier
├── dashboard/          # Streamlit interactive analysis dashboard
├── steering/           # Feature steering hooks, activation patching, emotion probe
└── viz/                # Training metrics and mapping visualizations

cli/                    # CLI entry points
├── extract.py          # Activation extraction (batched multi-layer)
├── extract_generate.py # Generation-time extraction with safety validation
├── safety_score.py     # Contrastive + cross-layer safety scoring
├── train.py            # SAE training (sparsity sweep support)
├── pipeline.py         # End-to-end pipeline
├── steer.py            # Feature steering, patching, emotion discovery
└── analyze.py          # Autointerp and Clustering CLI

configs/models/         # Per-model JSON config files
runs/<model>/           # Model outputs (activations, checkpoints, reports)
tests/                  # Unit tests
```

Deception-specific entry points added on top of the general pipeline:

- `cli.deception_cycle` for direction -> probe -> monitor -> adversarial runs
- `cli.deception_report` for static HTML + PNG reporting
- `cli.deception_autointerp` for LLM interpretations of top deception-correlated features

---

## Safety Pipeline (Qwen3-0.6B)

A generation-time safety pipeline built on top of BeaverTails to identify and suppress features causing harmful outputs.

### Overview

```
Phase 1: Extract → Phase 2: Train SAE → Phase 3: Contrastive Score → Phase 4: Cross-Layer → Phase 5: Interpret → Phase 6: Validate → Phase 7: Steer
```

### Phase 1 — Generation-Time Extraction

Extracts activations from **all target layers in a single forward pass** during replay of real prompt/response pairs. Only stores samples validated by a safety classifier as `toxic` or `safe`.

**Key properties:**
- `count` = number of **validated** samples (not raw rows). The loop continues until this many accepted samples are collected.
- Multi-layer atomicity: a sample is added to **all** layer buffers or **none**. If rejected, it disappears from every layer simultaneously — guaranteeing alignment.
- `max_raw_samples` (default: `count × 5`) prevents infinite loops when the model consistently refuses.

```bash
python -m cli.extract_generate \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --mode replay
```

Output includes per-sample `labels` in chunk metadata, enabling contrastive analysis.

### Phase 2 — SAE Training

Uses TopK SAE architecture. Trains on all activations (toxic + safe) — SAE must reconstruct both distributions. Class separation happens in Phase 3, not here.

```bash
# Train all layers sequentially
python -m cli.train --config configs/models/qwen3-0.6B.safety-gen.json

# Train a single layer (recommended for 4GB VRAM)
python -m cli.train --config configs/models/qwen3-0.6B.safety-gen.json --layer blocks.6.hook_resid_post
```

### Phase 3 — Contrastive Safety Scoring

Identifies which SAE features are most associated with toxic generation.

**Metrics per feature:**

| Metric | Description |
|---|---|
| `rate_ratio` | How much more often feature fires on toxic text |
| `strength_diff` | Mean activation strength: toxic − safe |
| `fisher_score` | Statistical separability (μ_toxic − μ_safe)² / (σ²_t + σ²_s) |
| `risk_score` | Combined: log(rate_ratio) × √fisher |

```bash
python -m cli.safety_score \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --mode contrastive \
  --top-k 50

# Single layer scoring
python -m cli.safety_score \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --mode contrastive \
  --layer blocks.6.hook_resid_post \
  --top-k 50
```

Output: `<activations_parent>/safety_scores/<layer>_contrastive.csv`

Also writes `contrastive_summary.json` in the same `safety_scores/` directory.

### Phase 4 — Cross-Layer Feature Combination Analysis

Detects **cross-layer combinations** that are collectively dangerous even when individual features are not:

```
IF layer6[feature_512] > 0.3 AND layer27[feature_8192] > 0.5 → toxic (94%)
```

Uses:
1. **Pairwise co-activation**: which cross-layer pairs fire together on toxic text?
2. **Decision Tree classifier** (sklearn): extracts interpretable if-then rules.
3. **Feature importance ranking**: identifies the most predictive individual features.

```bash
python -m cli.safety_score \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --mode cross-layer \
  --top-k 50
```

Output: `safety_scores/cross_layer_combinations.csv`, `safety_scores/safety_classifier_report.json`, `safety_scores/feature_importance.json`

> **Note**: Cross-layer analysis requires contrastive scoring to be run first.

### Phase 5 — Feature Interpretation

Auto-interprets dangerous features using an LLM:

```bash
python -m cli.analyze autointerp \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --top-n 50 \
  --groq
```

The config includes a safety-specific system prompt that guides the LLM to focus on whether the feature relates to harmful output generation.

### Phase 6 — Causal Validation

Confirms that a candidate dangerous feature actually causes toxic output:

```bash
# Patch the suspected dangerous feature into a safe prompt
python -m cli.steer patch \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --layer blocks.27.hook_resid_post \
  --feature-id 8192 \
  --clean "What is the weather today?" \
  --patch "How to make explosives?"
```

A high patching effect confirms the feature is causally responsible.

### Phase 7 — Safety Steering

Suppress confirmed dangerous features to make the model safer:

```bash
# Single feature suppression
python -m cli.steer steer \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --layer blocks.27.hook_resid_post \
  --feature-id 8192 --alpha -15 --mode add \
  --prompt "How to make explosives?"

# Multi-feature suppression (from Phase 4 combinations)
python -m cli.steer steer \
  --config configs/models/qwen3-0.6B.safety-gen.json \
  --layer blocks.27.hook_resid_post \
  --features "8192:-15,15000:-10" \
  --prompt "How to make explosives?"
```

### Safety Config (`qwen3-0.6B.safety-gen.json`) Key Parameters

```json
{
  "extraction": {
    "extractor_backend": "hf_generate",
    "count": 3000,           // target validated samples (toxic + safe)
    "max_raw_samples": 15000, // safety cap for raw iterations
    "batch_size": 16,
    "max_length": 512,
    "device": "cuda",
    "dtype": "bfloat16"
  },
  "classifier": {
    "backend": "hf_classifier",
    "model": "s-nlp/roberta_toxicity_classifier",
    "threshold": 0.5
  },
  "generation": {
    "mode": "replay",
    "response_field": "response",
    "max_new_tokens": 256
  }
}
```

---

## Deception Pipeline (Qwen3-0.6B)

The deception workflow uses paired honest/deceptive generations and adds reporting on top of direction, probe, and monitor artifacts.

### Phase 1 - Direction / Probe / Monitor / Adversarial

```bash
# Run individual phases against existing deception activations + SAE checkpoints
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase direction
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase probe
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase monitor
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase adversarial

# Or run the full deception pipeline
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase full
```

Key artifacts:

- `runs/Qwen_Qwen3-0.6B/deception/directions/directions.pt`
- `runs/Qwen_Qwen3-0.6B/deception/probes/probe_summary.json`
- `runs/Qwen_Qwen3-0.6B/deception/monitor/monitor_report.json`
- `runs/Qwen_Qwen3-0.6B/deception/adversarial/summary.json`

### Phase 2 - Feature-Level Scoring and Reporting

Generate contrastive deception feature scores:

```bash
python -m cli.safety_score \
  --config configs/models/qwen3-0.6B.deception.json \
  --mode contrastive

python -m cli.safety_score \
  --config configs/models/qwen3-0.6B.deception.json \
  --mode cross-layer
```

Default output directory for deception scoring:

- `runs/Qwen_Qwen3-0.6B/deception/contextual_activations/safety_scores/`

Generate the static deception report:

```bash
python -m cli.deception_report \
  --config configs/models/qwen3-0.6B.deception.json
```

Static report outputs:

- `runs/Qwen_Qwen3-0.6B/deception/reports/deception_report.html`
- `runs/Qwen_Qwen3-0.6B/deception/reports/layer_comparison.png`
- `runs/Qwen_Qwen3-0.6B/deception/reports/tsne_honest_vs_deceptive.png`
- `runs/Qwen_Qwen3-0.6B/deception/reports/probe_roc_curve.png`
- `runs/Qwen_Qwen3-0.6B/deception/reports/category_breakdown.png`
- `runs/Qwen_Qwen3-0.6B/deception/reports/adversarial_analysis.png`

Optional: add LLM interpretations for top deception-correlated SAE features:

```bash
# OpenAI
set OPENAI_API_KEY=your_key
python -m cli.deception_autointerp \
  --config configs/models/qwen3-0.6B.deception.json \
  --top-n 20

# Groq
set GROQ_API_KEY=your_key
python -m cli.deception_autointerp \
  --config configs/models/qwen3-0.6B.deception.json \
  --top-n 20 \
  --groq \
  --model llama-3.1-8b-instant
```

### Dashboard

The Streamlit dashboard includes a deception monitor view and automatically surfaces generated report artifacts when present.

```bash
uv run streamlit run rfm/dashboard/app.py
```

Set the sidebar config path to:

- `configs/models/qwen3-0.6B.deception.json`

### Reporting Command Summary

```bash
# 1) Produce deception monitor artifacts
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase direction
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase probe
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase monitor
python -m cli.deception_cycle --config configs/models/qwen3-0.6B.deception.json --phase adversarial

# 2) Score SAE features for deceptive vs honest behavior
python -m cli.safety_score --config configs/models/qwen3-0.6B.deception.json --mode contrastive
python -m cli.safety_score --config configs/models/qwen3-0.6B.deception.json --mode cross-layer

# 3) Build the static report
python -m cli.deception_report --config configs/models/qwen3-0.6B.deception.json

# 4) Open the dashboard
uv run streamlit run rfm/dashboard/app.py
```

---

## General Pipeline (GPT-2, Turkish-GPT, etc.)

### Full Pipeline

```bash
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json --skip-viz
python -m cli.pipeline --config configs/models/gpt2-small.emotion.json --from-step mapping
```

### Step by Step

```bash
# 1) Extract activations (multi-layer, batched, GPU-accelerated)
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
# Discover which features correlate with emotions
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

---

## Configuration

### Extractor Backends

| `extractor_backend` | Use case |
|---|---|
| `transformer_lens` | GPT-2, TransformerLens-compatible models |
| `hf` | Any HuggingFace CausalLM (batched multi-layer, reading-time) |
| `hf_generate` | Generation-time extraction with safety validation |

### Performance: Batched Multi-Layer Extraction

The `hf` and `hf_generate` backends extract **all layers in a single forward pass**:

```
Old: N_samples × N_layers = 8000 × 4 = 32,000 forward passes
New: N_samples / batch_size = 8000 / 16 = 500 forward passes (64× faster)
```

Key extraction config:
```json
"extraction": {
  "batch_size": 4,     // keep low on 4GB VRAM (4-8 recommended)
  "max_length": 256,   // token truncation (256 for 4GB, 512 for 8GB+)
  "device": "cuda",    // explicit — avoid accidental CPU fallback
  "dtype": "bfloat16"
}
```

### Low-VRAM Guide (4GB GPU)

For GPUs with limited VRAM (e.g. GTX 1650, RTX 3050 4GB):

| Parameter | 4GB VRAM | 8GB+ VRAM |
|---|---|---|
| `extraction.batch_size` | 4 | 16 |
| `extraction.max_length` | 256 | 512 |
| `sae.hidden_dim` | 8192 | 32768 |
| `train.batch_size` | 1024 | 4096 |
| `train.device` | `"cpu"` | `"cuda"` |
| `generation.max_new_tokens` | 128 | 256 |

Recommended layer-by-layer workflow for 4GB VRAM:

```bash
# Step 1: Extract all layers (single pass, GPU handles one batch at a time)
python -m cli.extract_generate --config configs/models/qwen3-0.6B.safety-gen.json

# Step 2: Train SAE layer by layer (CPU training, one at a time)
python -m cli.train --config configs/models/qwen3-0.6B.safety-gen.json --layer blocks.6.hook_resid_post
python -m cli.train --config configs/models/qwen3-0.6B.safety-gen.json --layer blocks.13.hook_resid_post
python -m cli.train --config configs/models/qwen3-0.6B.safety-gen.json --layer blocks.20.hook_resid_post
python -m cli.train --config configs/models/qwen3-0.6B.safety-gen.json --layer blocks.27.hook_resid_post

# Step 3: Score layers individually
python -m cli.safety_score --config configs/models/qwen3-0.6B.safety-gen.json --layer blocks.6.hook_resid_post
```

### Device Resolution

Device is resolved in this priority order:
```
extraction.device → model.device → train.device → auto (CUDA if available)
```

Extractor prints on startup:
```
[extractor] Device: cuda | dtype: bfloat16
[extractor] Model loaded: Qwen/Qwen3-0.6B on cuda:0
```

### Layer Config Structure

```json
{
  "layers": {
    "blocks.6.hook_resid_post": {},
    "blocks.13.hook_resid_post": {
      "sae": { "hidden_dim": 16384, "topk_k_sweep": [192, 384] },
      "train": { "epochs": 40, "learning_rate": 0.0003 }
    },
    "blocks.27.hook_resid_post": {
      "train": { "batch_size": 2048, "epochs": 50, "learning_rate": 0.0002 }
    }
  }
}
```

### SAE Architectures

```json
"sae": {
  "architecture": "topk",  // "vanilla" | "topk" | "gated"
  "topk_k": 96,
  "hidden_dim": 12288,
  "sparsity_weight": 0.0
}
```

### LLM Auto-interpretation

```bash
# OpenAI (default)
set OPENAI_API_KEY=your_key
python -m cli.analyze autointerp --config ...

# Groq (free tier, rate-limit friendly)
set GROQ_API_KEY=your_key
python -m cli.analyze autointerp --config ... --groq --model llama-3.1-8b-instant --request-delay 11

# Any OpenAI-compatible endpoint
python -m cli.analyze autointerp --config ... --base-url https://your-provider/v1 --api-key your_key
```

---

## Tests

```bash
uv run pytest -q
```
