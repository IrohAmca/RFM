"""CLI: Activation extraction from language models."""

import argparse
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

from rfm.config import ConfigManager
from rfm.extractors import ExtractorFactory
from rfm.data import BaseDataLoader
from rfm.layout import default_activations_dir, sanitize_layer_name, sanitize_model_name


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


def _resolve_targets(config):
    raw = config.get("extraction.target")
    if isinstance(raw, list):
        return raw
    return [raw]


def extract_single_target(target, extractor, dataloader, config):
    chunk_size = int(config.get("extraction.chunk_size", 1_000_000))
    count = int(config.get("extraction.count", 100))
    output_prefix = config.get("extraction.output_prefix", "activations")
    activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))
    text_field = config.get("dataloader.text_field", "content")

    output_dir = config.get("extraction.output_dir", ".")
    if output_dir in (None, "", "."):
        output_dir = default_activations_dir(config)

    targets = _resolve_targets(config)
    if len(targets) > 1:
        output_dir = str(Path(output_dir) / sanitize_layer_name(target))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    buffer_acts = []
    buffer_tks = []
    buffer_lens = []
    chunk_index = 0

    for i, text in enumerate(
        tqdm(islice(dataloader, count), total=count, desc=f"Extracting [{target}]")
    ):
        sample_text = text.get(text_field, "")
        if not sample_text:
            continue

        activations = extractor.extract(sample_text, target)
        tokens = extractor.to_tokens(sample_text).squeeze(0)

        acts_2d = activations.reshape(-1, activations.shape[-1]).to(activation_dtype).cpu()
        if acts_2d.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"Activation/token length mismatch: acts={acts_2d.shape[0]} tokens={tokens.shape[0]}"
            )

        buffer_acts.append(acts_2d)
        buffer_tks.append(tokens.cpu())
        buffer_lens.append(int(tokens.shape[0]))

        current_size = sum([t.shape[0] for t in buffer_acts])
        if current_size >= chunk_size:
            chunk_index = flush_chunk(
                buffer_acts, buffer_tks, buffer_lens,
                target, extractor.model_name, chunk_index,
                output_dir, output_prefix,
            )

    flush_chunk(
        buffer_acts, buffer_tks, buffer_lens,
        target, extractor.model_name, chunk_index,
        output_dir, output_prefix,
    )
    print(f"[extract] Complete for target: {target} → {output_dir}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)

    extractor = ExtractorFactory.create(config)
    dataloader = BaseDataLoader(config)
    dataloader.load()

    targets = _resolve_targets(config)
    for target in targets:
        dataloader.load()
        extract_single_target(target, extractor, dataloader, config)


if __name__ == "__main__":
    main()
