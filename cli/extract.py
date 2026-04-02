"""CLI: Activation extraction from language models.

Supports batched multi-layer extraction: a SINGLE forward pass per batch
extracts ALL requested layers simultaneously.

Speed:
    Old (one sample × one layer per call):  N_samples × N_layers forward passes
    New (batch × all layers in one call):   ceil(N_samples / batch_size) passes

Example: 8000 samples, batch=16, 4 layers → 500 vs 32000 forward passes = 64×.
"""

import argparse
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

from rfm.config import ConfigManager
from rfm.extractors import ExtractorFactory
from rfm.data import BaseDataLoader
from rfm.layout import (
    resolve_activations_dir,
    resolve_requested_targets,
    sanitize_model_name,
)


def resolve_dtype(name):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(str(name).lower(), torch.bfloat16)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations from an LLM.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--layer", type=str, default=None,
                        help="Extract activations for a specific layer only (e.g. blocks.6.hook_resid_post). "
                             "If not set, extracts all configured layers.")
    return parser.parse_args()


def flush_chunk(buffer_acts, buffer_tks, buffer_lens, target, model_name, chunk_index, output_dir, output_prefix, labels=None):
    if not buffer_acts:
        return chunk_index

    combined_acts = torch.cat(buffer_acts, dim=0)
    combined_tks = torch.cat(buffer_tks, dim=0)

    filename = f"{output_prefix}_{sanitize_model_name(model_name)}_{target.replace('.', '_')}_{chunk_index}.pt"
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
                "labels": list(labels) if labels is not None else None,
            },
        },
        save_path,
    )
    n_total = combined_acts.shape[0]
    label_info = ""
    if labels:
        n_toxic = sum(1 for la in labels if la == "toxic")
        n_safe = sum(1 for la in labels if la == "safe")
        label_info = f", {n_toxic} toxic / {n_safe} safe"
    print(f"[extract] Saved chunk {chunk_index} → {save_path} ({n_total} tokens{label_info})")

    buffer_acts.clear()
    buffer_tks.clear()
    buffer_lens.clear()
    return chunk_index + 1


def extract_all_targets_batched(targets, extractor, dataloader, config):
    """Batched multi-layer extraction with atomic per-sample validation.

    Correctness guarantee:
        A sample is added to buffer_layer_A and buffer_layer_B
        **at exactly the same time** or **not at all**.

        If the safety classifier rejects sample i (label='unknown' or
        classification fails), sample i is dropped from EVERY layer's
        buffer — so all layers always have the same number of samples.
    """
    from rfm.safety.classifier import create_classifier_from_config

    chunk_size = int(config.get("extraction.chunk_size", 1_000_000))
    count = int(config.get("extraction.count", 100))
    output_prefix = config.get("extraction.output_prefix", "activations")
    activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))
    text_field = config.get("dataloader.text_field", "content")
    batch_size = int(config.get("extraction.batch_size", 16))
    max_length = int(config.get("extraction.max_length", 512))

    # Safety classifier — validates each sample before saving
    classifier = create_classifier_from_config(config)
    valid_labels = {"toxic", "safe"}
    print(f"[extract] Classifier: backend={classifier.backend}, model={classifier.model_name}")

    # Per-layer output dirs and buffers
    output_dirs = {}
    buffers = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        out_dir = resolve_activations_dir(target_config, target=target)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output_dirs[target] = out_dir
        buffers[target] = {"acts": [], "tks": [], "lens": [], "labels": [], "chunk_index": 0}

    model_name = extractor.model_name
    all_rows = list(islice(dataloader, count))
    total_batches = (len(all_rows) + batch_size - 1) // batch_size
    label_counts = {"toxic": 0, "safe": 0, "unknown": 0, "skipped": 0}

    print("\n[extract] Batched multi-layer extraction (with validation)")
    print(f"  Samples: {len(all_rows)}  Batch: {batch_size}  Layers: {len(targets)}")
    print(f"  Forward passes: {total_batches} (était {len(all_rows) * len(targets)} sans batching)\n")

    for batch_start in tqdm(range(0, len(all_rows), batch_size), total=total_batches, desc="Extracting [batched]"):
        batch_rows = all_rows[batch_start: batch_start + batch_size]
        texts = [row.get(text_field, "") for row in batch_rows]

        # Keep track of original indices (so we can discard atomically)
        valid_pairs = [(i, t) for i, t in enumerate(texts) if t]
        if not valid_pairs:
            continue
        valid_indices, valid_texts = zip(*valid_pairs)

        try:
            batch_results = extractor.extract_batch_multi(list(valid_texts), targets, max_length=max_length)
        except Exception as e:
            print(f"[extract] Warning: batch {batch_start // batch_size} failed ({e}), skipping.")
            label_counts["skipped"] += len(valid_texts)
            continue

        # ── Classify all texts in the batch ──────────────────────────────
        # Run classifier on all texts at once (HF backend benefits from batch)
        try:
            classify_results = classifier.classify_batch(list(valid_texts))
        except Exception:
            classify_results = []
            for t in valid_texts:
                try:
                    classify_results.append(classifier.classify(t))
                except Exception:
                    classify_results.append({"label": "unknown", "score": 0.0})

        # ── Atomic add: only valid samples, ALL layers together ───────────
        for local_idx, (cls_result, sample_text) in enumerate(zip(classify_results, valid_texts)):
            label = cls_result["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

            if label not in valid_labels:
                # REJECT: skip this sample from EVERY layer — alignment preserved
                label_counts["skipped"] = label_counts.get("skipped", 0) + 1
                continue

            # ACCEPT: add to ALL layer buffers atomically
            for target in targets:
                layer_acts_list = batch_results[target]
                acts_2d = layer_acts_list[local_idx].to(activation_dtype)
                sample_tks = torch.zeros(acts_2d.shape[0], dtype=torch.long)
                buf = buffers[target]
                buf["acts"].append(acts_2d)
                buf["tks"].append(sample_tks)
                buf["lens"].append(int(acts_2d.shape[0]))
                buf["labels"].append(label)

        # Flush per-layer if needed
        for target in targets:
            buf = buffers[target]
            if sum(t.shape[0] for t in buf["acts"]) >= chunk_size:
                buf["chunk_index"] = flush_chunk(
                    buf["acts"], buf["tks"], buf["lens"],
                    target, model_name, buf["chunk_index"],
                    output_dirs[target], output_prefix,
                    labels=buf["labels"],
                )
                buf["labels"].clear()

    # Final flush for each layer
    for target in targets:
        buf = buffers[target]
        flush_chunk(
            buf["acts"], buf["tks"], buf["lens"],
            target, model_name, buf["chunk_index"],
            output_dirs[target], output_prefix,
            labels=buf["labels"],
        )
        print(f"[extract] Complete: {target} → {output_dirs[target]}")

    print(
        f"\n[extract] Validation summary: "
        f"toxic={label_counts.get('toxic', 0)} "
        f"safe={label_counts.get('safe', 0)} "
        f"unknown/rejected={label_counts.get('unknown', 0)} "
        f"skipped={label_counts.get('skipped', 0)}"
    )


def _extract_single_target_legacy(target, extractor, dataloader, config):
    """Legacy single-sample extraction (fallback for non-HF extractors)."""
    chunk_size = int(config.get("extraction.chunk_size", 1_000_000))
    count = int(config.get("extraction.count", 100))
    output_prefix = config.get("extraction.output_prefix", "activations")
    activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))
    text_field = config.get("dataloader.text_field", "content")
    output_dir = resolve_activations_dir(config, target=target)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    buffer_acts, buffer_tks, buffer_lens = [], [], []
    chunk_index = 0

    for text in tqdm(islice(dataloader, count), total=count, desc=f"Extracting [{target}]"):
        sample_text = text.get(text_field, "")
        if not sample_text:
            continue
        activations = extractor.extract(sample_text, target)
        tokens = extractor.to_tokens(sample_text).squeeze(0)
        acts_2d = activations.reshape(-1, activations.shape[-1]).to(activation_dtype).cpu()
        buffer_acts.append(acts_2d)
        buffer_tks.append(tokens.cpu())
        buffer_lens.append(int(tokens.shape[0]))
        if sum(t.shape[0] for t in buffer_acts) >= chunk_size:
            chunk_index = flush_chunk(buffer_acts, buffer_tks, buffer_lens,
                                      target, extractor.model_name, chunk_index,
                                      output_dir, output_prefix)

    flush_chunk(buffer_acts, buffer_tks, buffer_lens,
                target, extractor.model_name, chunk_index, output_dir, output_prefix)
    print(f"[extract] Complete: {target} → {output_dir}")


def main():
    args = parse_args()
    base_config = ConfigManager.from_file(args.config)

    extractor = ExtractorFactory.create(base_config)
    dataloader = BaseDataLoader(base_config)
    dataloader.load()

    if args.layer:
        targets = [args.layer]
    else:
        targets = resolve_requested_targets(base_config)

    if hasattr(extractor, "extract_batch_multi"):
        extract_all_targets_batched(targets, extractor, dataloader, base_config)
    else:
        # Fallback for TransformerLens / non-HF extractors
        for target in targets:
            dataloader.load()
            _extract_single_target_legacy(
                target, extractor, dataloader, base_config.for_target(target)
            )


if __name__ == "__main__":
    main()
