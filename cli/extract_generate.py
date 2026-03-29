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
from pathlib import Path

import torch
from tqdm import tqdm

from rfm.config import ConfigManager
from rfm.extractors.hf_generate import HFGenerationExtractor
from rfm.data import BaseDataLoader
from rfm.layout import (
    resolve_activations_dir,
    resolve_requested_targets,
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
    n_toxic = sum(1 for la in buffer_labels if la == "toxic")
    n_safe = sum(1 for la in buffer_labels if la == "safe")
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


def extract_all_targets(targets, extractor, dataloader, config, mode="replay", classifier=None):
    """Extract activations for ALL target layers in a SINGLE forward pass per sample.

    Instead of running the model N times (once per layer), this runs it ONCE
    and slices out all requested layers.  This is critical for:
      1. Speed:  N× faster (only 1 forward pass per sample).
      2. Cross-layer analysis:  All layers' activations are perfectly aligned
         to the same sample → enables feature combination detection.
    """
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

    # Safety classifier
    if classifier is None:
        classifier = create_classifier_from_config(config)

    # Per-layer output directories and buffers
    output_dirs = {}
    buffers = {}  # target → {acts, tks, lens, labels, scores, chunk_index}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        out_dir = resolve_activations_dir(target_config, target=target)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output_dirs[target] = out_dir
        buffers[target] = {
            "acts": [], "tks": [], "lens": [], "labels": [], "scores": [],
            "chunk_index": 0,
        }

    skipped = 0
    rejected = 0
    validated = 0  # count of accepted (toxic + safe) samples
    label_counts = {"toxic": 0, "safe": 0, "unknown": 0}
    valid_labels = {"toxic", "safe"}
    model_name = extractor.model_name

    # count = TARGET number of validated samples (not raw)
    # max_raw = safety limit to prevent infinite iteration
    max_raw = int(config.get("extraction.max_raw_samples", count * 5))

    print(f"[extract_gen] Multi-layer extraction: {len(targets)} layers in 1 forward pass")
    print(f"[extract_gen] Target: {count} validated samples (max {max_raw} raw)")
    print(f"[extract_gen] Targets: {targets}")

    pbar = tqdm(total=count, desc="Validated samples")
    raw_processed = 0

    for row in dataloader:
        if validated >= count or raw_processed >= max_raw:
            break
        raw_processed += 1

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
                # Single forward pass → all layers
                multi_result = extractor.extract_replay_multi(prompt_text, response_text, targets)
            else:
                # Single forward pass → all layers
                multi_result = extractor.extract_generate_multi(
                    prompt_text, targets,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # Decode response for classification (use any target's tokens — they're all the same)
                first_target = targets[0]
                response_text = extractor.tokenizer.decode(
                    multi_result[first_target]["tokens"].tolist(), skip_special_tokens=True
                )
        except Exception as e:
            print(f"[extract_gen] Warning: skipping sample {raw_processed}: {e}")
            skipped += 1
            continue

        # Classify (same label for all layers — it's the same sample)
        label, score = _classify_sample(row, config, classifier, response_text)
        label_counts[label] = label_counts.get(label, 0) + 1

        # ── ATOMIC GUARD: reject unknown from ALL layers ──────────────────
        if label not in valid_labels:
            rejected += 1
            continue  # skips the entire per-target loop below

        # ACCEPTED — increment validated count
        validated += 1
        pbar.update(1)

        # Distribute to per-layer buffers (all layers or none)
        for target in targets:
            result = multi_result[target]
            acts = result["activations"].to(activation_dtype)
            tks = result["tokens"]

            if acts.shape[0] == 0:
                continue

            buf = buffers[target]
            buf["acts"].append(acts)
            buf["tks"].append(tks)
            buf["lens"].append(int(tks.shape[0]))
            buf["labels"].append(label)
            buf["scores"].append(score)

            # Check if this layer's buffer needs flushing
            current_size = sum(t.shape[0] for t in buf["acts"])
            if current_size >= chunk_size:
                buf["chunk_index"] = flush_chunk(
                    buf["acts"], buf["tks"], buf["lens"], buf["labels"],
                    target, model_name, buf["chunk_index"],
                    output_dirs[target], output_prefix,
                )

    pbar.close()

    # Flush remaining for each layer
    for target in targets:
        buf = buffers[target]
        flush_chunk(
            buf["acts"], buf["tks"], buf["lens"], buf["labels"],
            target, model_name, buf["chunk_index"],
            output_dirs[target], output_prefix,
        )

    yield_rate = validated / max(raw_processed, 1) * 100
    print(
        f"\n[extract_gen] Complete for all {len(targets)} layers\n"
        f"  Validated: {validated}/{count} target "
        f"(toxic={label_counts.get('toxic', 0)} safe={label_counts.get('safe', 0)})\n"
        f"  Rejected (unknown): {rejected} | Skipped (error/empty): {skipped}\n"
        f"  Raw processed: {raw_processed} → Yield rate: {yield_rate:.1f}%\n"
        f"  Output dirs:"
    )
    for t, d in output_dirs.items():
        print(f"    {t} → {d}")


def main():
    args = parse_args()
    base_config = ConfigManager.from_file(args.config)

    extractor = HFGenerationExtractor(base_config)

    # Create safety classifier once
    classifier = create_classifier_from_config(base_config)
    print(f"[extract_gen] Safety classifier: backend={classifier.backend}, model={classifier.model_name}")

    dataloader = BaseDataLoader(base_config)
    dataloader.load()

    targets = resolve_requested_targets(base_config)
    print(f"[extract_gen] Requested targets: {targets}")

    # All layers extracted in a single forward pass per sample
    extract_all_targets(
        targets, extractor, dataloader, base_config,
        mode=args.mode,
        classifier=classifier,
    )


if __name__ == "__main__":
    main()

