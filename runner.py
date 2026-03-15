import argparse
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

from base import BaseDataLoader
from config_manager import ConfigManager
from extractor_factory import ExtractorFactory
from project_layout import default_activations_dir


def chunk_saver(
    buffer_tensor,
    token_buffer_tensor,
    token_lengths,
    target_layer,
    model_name,
    index,
    output_dir,
    output_prefix,
):
    safe_target = target_layer.replace(".", "_")
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")
    filename = f"{output_prefix}_{safe_model}_{safe_target}_{index}.pt"
    file_path = Path(output_dir) / filename

    chunk_data = {
        "metadata": {
            "model_name": model_name,
            "target_layer": target_layer,
            "d_in": int(buffer_tensor.shape[-1]),
            "chunk_id": index,
            "num_rows": int(buffer_tensor.shape[0]),
            "num_tokens": int(token_buffer_tensor.shape[0]),
            "token_lengths": list(token_lengths),
        },
        "activations": buffer_tensor,
        "tokens": token_buffer_tensor,
    }

    torch.save(chunk_data, file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect GPT-2 activations for SAE")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON/TOML config. If omitted, built-in defaults are used.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name):
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(str(dtype_name).lower(), torch.bfloat16)


def flush_chunk(
    buffer_acts,
    buffer_tks,
    buffer_lens,
    target,
    model_name,
    chunk_index,
    output_dir,
    output_prefix,
):
    if not buffer_acts:
        return chunk_index

    buffer_tensor = torch.cat(buffer_acts, dim=0)
    token_buffer_tensor = torch.cat(buffer_tks, dim=0)

    chunk_saver(
        buffer_tensor,
        token_buffer_tensor,
        buffer_lens,
        target,
        model_name,
        chunk_index,
        output_dir,
        output_prefix,
    )

    buffer_acts.clear()
    buffer_tks.clear()
    buffer_lens.clear()
    return chunk_index + 1


args = parse_args()
config = ConfigManager.from_file(args.config)

target = config.get("extraction.target")
chunk_size = int(config.get("extraction.chunk_size", 1_000_000))
count = int(config.get("extraction.count", 100))
output_dir = config.get("extraction.output_dir", ".")
if output_dir in (None, "", "."):
    output_dir = default_activations_dir(config)
output_prefix = config.get("extraction.output_prefix", "activations")
activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))
text_field = config.get("dataloader.text_field", "content")

Path(output_dir).mkdir(parents=True, exist_ok=True)

extractor = ExtractorFactory.create(config)
dataloader = BaseDataLoader(config)
dataloader.load()

buffer_acts = []
buffer_tks = []
buffer_lens = []

chunk_index = 0

for i, text in enumerate(
    tqdm(islice(dataloader, count), total=count, desc="Extracting")
):
    sample_text = text.get(text_field, "")
    if not sample_text:
        continue

    activations = extractor.extract(sample_text, target)
    tokens = extractor.model.to_tokens(sample_text).squeeze(0)

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
            buffer_acts,
            buffer_tks,
            buffer_lens,
            target,
            extractor.model_name,
            chunk_index,
            output_dir,
            output_prefix,
        )

chunk_index = flush_chunk(
    buffer_acts,
    buffer_tks,
    buffer_lens,
    target,
    extractor.model_name,
    chunk_index,
    output_dir,
    output_prefix,
)
