"""CLI: Generation-time activation extraction for safety analysis.

Extracts hidden states during model *generation* (not just reading), which is
critical for finding features that *cause* harmful output. Supports two modes:

  replay   – Use (prompt, response) pairs from a labelled dataset (e.g. BeaverTails).
             Cheap, supports pre-labelled toxicity.
  generate – Let the model generate its own response to red-team prompts.
             Better for finding the model's natural harmful tendencies.

Usage:
    python -m cli.extract_generate --config configs/models/qwen3-0.6B.safety.json

The config should include a ``generation`` section (see below) and can use datasets
that provide prompt+response+label triples such as PKU-Alignment/BeaverTails.
"""

import argparse
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

from rfm.config import ConfigManager
from rfm.extractors.hf_generate import HFGenerationExtractor
from rfm.data import BaseDataLoader
from rfm.layout import (
    resolve_activations_dir,
    resolve_requested_targets,
    sanitize_layer_name,
    sanitize_model_name,
)
from rfm.safety.classifier import SafetyClassifier, create_classifier_from_config


def resolve_dtype(name):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(str(name).lower(), torch.bfloat16)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract generation-time activations for safety analysis."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument(
        "--mode", type=str, default="replay",
        choices=["replay", "generate"],
        help="'replay' uses dataset responses; 'generate' lets the model produce its own.",
    )
    return parser.parse_args()


def flush_chunk(
    buffer_acts, buffer_tks, buffer_lens, buffer_labels,
    target, model_name, chunk_index, output_dir, output_prefix,
):
    """Save a chunk of activations with per-sequence toxicity labels."""
    if not buffer_acts:
        return chunk_index

    combined_acts = torch.cat(buffer_acts, dim=0)
    combined_tks = torch.cat(buffer_tks, dim=0)

    safe_model_name = sanitize_model_name(model_name)
    filename = f"{output_prefix}_{safe_model_name}_{target.replace('.', '_')}_{chunk_index}.pt"
    save_path = Path(output_dir) / filename

    torch.save(
        {
            "activations": combined_acts,
            "tokens": combined_tks,
            "metadata": {
                "model_name": model_name,
                "target_layer": target,
                "chunk_id": chunk_index,
                "token_lengths": list(buffer_lens),
                "labels": list(buffer_labels),  # per-sequence label
                "extraction_mode": "generation",
            },
        },
        save_path,
    )
    n_toxic = sum(1 for l in buffer_labels if l == "toxic")
    n_safe = sum(1 for l in buffer_labels if l == "safe")
    print(
        f"[extract_gen] Saved chunk {chunk_index} → {save_path} "
        f"({combined_acts.shape[0]} tokens, {n_toxic} toxic / {n_safe} safe sequences)"
    )

    buffer_acts.clear()
    buffer_tks.clear()
    buffer_lens.clear()
    buffer_labels.clear()
    return chunk_index + 1


def _classify_sample(row, config, classifier: SafetyClassifier, response_text: str | None = None):
    """Determine if a sample is toxic or safe.

    Priority order:
      1. If the dataset has a label field → use it (trusted human labels).
      2. Otherwise, run the SafetyClassifier on the response text.
    """
    safety_cfg = config.get("safety", {})
    label_field = safety_cfg.get("label_field", "is_safe")

    # Try dataset label first
    label_value = row.get(label_field)
    if label_value is not None:
        result = classifier.classify_from_row(row, label_field=label_field)
        return result["label"], result["score"]

    # No dataset label → classify the response text with the model
    if response_text and classifier.backend != "dataset":
        result = classifier.classify(response_text)
        return result["label"], result["score"]

    return "unknown", 0.0


def extract_generation_target(target, extractor, dataloader, config, mode="replay", classifier=None):
    """Extract activations from one target layer during generation."""
    gen_cfg = config.get("generation", {})
    chunk_size = int(config.get("extraction.chunk_size", 500_000))
    count = int(config.get("extraction.count", 2000))
    output_prefix = config.get("extraction.output_prefix", "gen_activations")
    activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))

    # Fields
    prompt_field = config.get("dataloader.text_field", "prompt")
    response_field = gen_cfg.get("response_field", config.get("dataloader.response_field", "response"))

    # Generation params
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_p = float(gen_cfg.get("top_p", 0.95))

    # Safety classifier (create default if not passed)
    if classifier is None:
        classifier = create_classifier_from_config(config)

    output_dir = resolve_activations_dir(config, target=target)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    buffer_acts = []
    buffer_tks = []
    buffer_lens = []
    buffer_labels = []
    buffer_scores = []
    chunk_index = 0
    skipped = 0
    label_counts = {"toxic": 0, "safe": 0, "unknown": 0}

    for i, row in enumerate(
        tqdm(islice(dataloader, count), total=count, desc=f"Extracting gen [{target}]")
    ):
        prompt_text = row.get(prompt_field, "")
        if not prompt_text:
            skipped += 1
            continue

        try:
            if mode == "replay":
                response_text = row.get(response_field, "")
                if not response_text:
                    skipped += 1
                    continue
                result = extractor.extract_replay(prompt_text, response_text, target)
            else:
                result = extractor.extract_generate(
                    prompt_text, target,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # For generate mode, decode the response for classification
                response_text = extractor.tokenizer.decode(
                    result["tokens"].tolist(), skip_special_tokens=True
                )
        except Exception as e:
            print(f"[extract_gen] Warning: skipping sample {i}: {e}")
            skipped += 1
            continue

        # Classify: dataset label first, then classifier on response text
        label, score = _classify_sample(row, config, classifier, response_text)
        label_counts[label] = label_counts.get(label, 0) + 1

        acts = result["activations"].to(activation_dtype)
        tks = result["tokens"]

        if acts.shape[0] == 0:
            skipped += 1
            continue

        buffer_acts.append(acts)
        buffer_tks.append(tks)
        buffer_lens.append(int(tks.shape[0]))
        buffer_labels.append(label)
        buffer_scores.append(score)

        current_size = sum(t.shape[0] for t in buffer_acts)
        if current_size >= chunk_size:
            chunk_index = flush_chunk(
                buffer_acts, buffer_tks, buffer_lens, buffer_labels,
                target, extractor.model_name, chunk_index,
                output_dir, output_prefix,
            )

    # Flush remaining
    flush_chunk(
        buffer_acts, buffer_tks, buffer_lens, buffer_labels,
        target, extractor.model_name, chunk_index,
        output_dir, output_prefix,
    )

    print(
        f"[extract_gen] Complete for target: {target} → {output_dir}\n"
        f"  Samples: toxic={label_counts.get('toxic', 0)} safe={label_counts.get('safe', 0)} "
        f"unknown={label_counts.get('unknown', 0)} skipped={skipped}"
    )


def main():
    args = parse_args()
    base_config = ConfigManager.from_file(args.config)

    extractor = HFGenerationExtractor(base_config)

    # Create safety classifier once (shared across targets)
    classifier = create_classifier_from_config(base_config)
    print(f"[extract_gen] Safety classifier: backend={classifier.backend}, model={classifier.model_name}")

    dataloader = BaseDataLoader(base_config)
    dataloader.load()

    targets = resolve_requested_targets(base_config)
    for target in targets:
        dataloader.load()
        extract_generation_target(
            target, extractor, dataloader,
            base_config.for_target(target),
            mode=args.mode,
            classifier=classifier,
        )


if __name__ == "__main__":
    main()
