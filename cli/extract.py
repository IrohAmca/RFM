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
    return parser.parse_args()


def flush_chunk(buffer_acts, buffer_tks, buffer_lens, target, model_name, chunk_index, output_dir, output_prefix):
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
            },
        },
        save_path,
    )
    print(f"[extract] Saved chunk {chunk_index} → {save_path} ({combined_acts.shape[0]} tokens)")

    buffer_acts.clear()
    buffer_tks.clear()
    buffer_lens.clear()
    return chunk_index + 1


def extract_all_targets_batched(targets, extractor, dataloader, config):
    """Batched multi-layer extraction: ALL layers in ONE forward pass per batch."""
    chunk_size = int(config.get("extraction.chunk_size", 1_000_000))
    count = int(config.get("extraction.count", 100))
    output_prefix = config.get("extraction.output_prefix", "activations")
    activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))
    text_field = config.get("dataloader.text_field", "content")
    batch_size = int(config.get("extraction.batch_size", 16))
    max_length = int(config.get("extraction.max_length", 512))

    # Per-layer output dirs and buffers
    output_dirs = {}
    buffers = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        out_dir = resolve_activations_dir(target_config, target=target)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output_dirs[target] = out_dir
        buffers[target] = {"acts": [], "tks": [], "lens": [], "chunk_index": 0}

    model_name = extractor.model_name

    # Pre-collect all rows (needed to display progress correctly)
    all_rows = list(islice(dataloader, count))
    total_batches = (len(all_rows) + batch_size - 1) // batch_size

    print(f"\n[extract] Batched multi-layer extraction")
    print(f"  Samples: {len(all_rows)}  Batch size: {batch_size}  Layers: {len(targets)}")
    print(f"  Forward passes: {total_batches} (was {len(all_rows) * len(targets)} without batching)\n")

    for batch_start in tqdm(range(0, len(all_rows), batch_size), total=total_batches, desc="Extracting [batched]"):
        batch_rows = all_rows[batch_start: batch_start + batch_size]
        texts = [row.get(text_field, "") for row in batch_rows]
        texts = [t for t in texts if t]
        if not texts:
            continue

        try:
            # ONE forward pass → all layers, all texts
            batch_results = extractor.extract_batch_multi(texts, targets, max_length=max_length)
        except Exception as e:
            print(f"[extract] Warning: batch {batch_start // batch_size} failed ({e}), skipping.")
            continue

        for target in targets:
            layer_acts_list = batch_results[target]  # list of [seq_len, d_model]
            buf = buffers[target]

            for sample_acts in layer_acts_list:
                acts_2d = sample_acts.to(activation_dtype)
                # Token IDs: use zeros as placeholder (activations are what SAE needs)
                sample_tks = torch.zeros(acts_2d.shape[0], dtype=torch.long)
                buf["acts"].append(acts_2d)
                buf["tks"].append(sample_tks)
                buf["lens"].append(int(acts_2d.shape[0]))

            # Flush if buffer is large enough
            if sum(t.shape[0] for t in buf["acts"]) >= chunk_size:
                buf["chunk_index"] = flush_chunk(
                    buf["acts"], buf["tks"], buf["lens"],
                    target, model_name, buf["chunk_index"],
                    output_dirs[target], output_prefix,
                )

    # Final flush for each layer
    for target in targets:
        buf = buffers[target]
        flush_chunk(
            buf["acts"], buf["tks"], buf["lens"],
            target, model_name, buf["chunk_index"],
            output_dirs[target], output_prefix,
        )
        print(f"[extract] Complete: {target} → {output_dirs[target]}")


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
