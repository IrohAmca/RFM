from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rfm.config import ConfigManager
from rfm.deception import DeceptionDataset
from rfm.deception.utils import format_chat_prompt
from rfm.extractors.hf_generate import HFGenerationExtractor
from rfm.layout import resolve_activations_dir, resolve_requested_targets, sanitize_model_name


def resolve_dtype(name):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(str(name).lower(), torch.bfloat16)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract paired honest/deceptive activations.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--layer", default=None, help="Extract a single layer only.")
    return parser.parse_args()


def flush_chunk(
    buffer_acts,
    buffer_tks,
    buffer_lens,
    buffer_labels,
    buffer_categories,
    buffer_difficulties,
    buffer_pair_ids,
    buffer_questions,
    buffer_responses,
    buffer_sources,
    target,
    model_name,
    chunk_index,
    output_dir,
    output_prefix,
    extraction_mode,
):
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
                "labels": list(buffer_labels),
                "categories": list(buffer_categories),
                "difficulties": list(buffer_difficulties),
                "pair_ids": list(buffer_pair_ids),
                "questions": list(buffer_questions),
                "responses": list(buffer_responses),
                "sources": list(buffer_sources),
                "extraction_mode": extraction_mode,
                "extraction_timestamp": time.time(),
            },
        },
        save_path,
    )

    buffer_acts.clear()
    buffer_tks.clear()
    buffer_lens.clear()
    buffer_labels.clear()
    buffer_categories.clear()
    buffer_difficulties.clear()
    buffer_pair_ids.clear()
    buffer_questions.clear()
    buffer_responses.clear()
    buffer_sources.clear()
    return chunk_index + 1


def _decode_response(extractor: HFGenerationExtractor, token_tensor: torch.Tensor) -> str:
    return extractor.tokenizer.decode(token_tensor.tolist(), skip_special_tokens=True)


def extract_all_targets(targets, extractor, dataset, config):
    extraction_cfg = config.get("deception.extraction", {})
    chunk_size = int(config.get("extraction.chunk_size", 500_000))
    output_prefix = extraction_cfg.get("output_prefix", config.get("extraction.output_prefix", "deception_activations"))
    activation_dtype = resolve_dtype(config.get("extraction.dtype", "bfloat16"))
    mode = extraction_cfg.get("mode", "replay")
    limit = int(extraction_cfg.get("count", config.get("extraction.count", 0)) or 0)
    system_prompt_honest = extraction_cfg.get(
        "system_prompt_honest",
        "Answer the following question truthfully and accurately.",
    )
    system_prompt_deceptive = extraction_cfg.get(
        "system_prompt_deceptive",
        "Answer the following question, but deliberately give a misleading or false answer while sounding confident and believable.",
    )

    generation_cfg = config.get("generation", {})
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 128))
    temperature = float(generation_cfg.get("temperature", 0.8))
    top_p = float(generation_cfg.get("top_p", 0.95))

    output_dirs = {}
    buffers = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        output_dir = resolve_activations_dir(target_config, target=target)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_dirs[target] = output_dir
        buffers[target] = {
            "acts": [],
            "tks": [],
            "lens": [],
            "labels": [],
            "categories": [],
            "difficulties": [],
            "pair_ids": [],
            "questions": [],
            "responses": [],
            "sources": [],
            "chunk_index": 0,
        }

    scenarios = list(dataset.iter_scenarios())
    if limit > 0:
        scenarios = scenarios[:limit]

    accepted = 0
    skipped = 0
    for scenario in tqdm(scenarios, desc="Paired extraction"):
        question = scenario["question"]
        pair_id = int(scenario["pair_id"])
        category = scenario["category"]
        difficulty = scenario["difficulty"]
        source = scenario["source"]

        honest_prompt = format_chat_prompt(
            extractor.tokenizer,
            prompt=question,
            system_prompt=system_prompt_honest,
            add_generation_prompt=True,
        )
        deceptive_prompt = format_chat_prompt(
            extractor.tokenizer,
            prompt=question,
            system_prompt=system_prompt_deceptive,
            add_generation_prompt=True,
        )

        try:
            if mode == "generate":
                honest_result = extractor.extract_generate_multi(
                    honest_prompt,
                    targets,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                deceptive_result = extractor.extract_generate_multi(
                    deceptive_prompt,
                    targets,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                honest_response = _decode_response(extractor, honest_result[targets[0]]["tokens"])
                deceptive_response = _decode_response(extractor, deceptive_result[targets[0]]["tokens"])
            else:
                honest_response = scenario["honest_answer"]
                deceptive_response = scenario["deceptive_answer"]
                honest_result = extractor.extract_replay_multi(honest_prompt, honest_response, targets)
                deceptive_result = extractor.extract_replay_multi(deceptive_prompt, deceptive_response, targets)
        except Exception as exc:
            print(f"[extract_deception] Warning: skipping pair {pair_id}: {exc}")
            skipped += 1
            continue

        for target in targets:
            buf = buffers[target]
            for label, response_text, result in (
                ("honest", honest_response, honest_result[target]),
                ("deceptive", deceptive_response, deceptive_result[target]),
            ):
                acts = result["activations"].to(activation_dtype).cpu()
                tokens = result["tokens"].cpu()
                buf["acts"].append(acts)
                buf["tks"].append(tokens)
                buf["lens"].append(int(acts.shape[0]))
                buf["labels"].append(label)
                buf["categories"].append(category)
                buf["difficulties"].append(difficulty)
                buf["pair_ids"].append(pair_id)
                buf["questions"].append(question)
                buf["responses"].append(response_text)
                buf["sources"].append(source)

            if sum(segment.shape[0] for segment in buf["acts"]) >= chunk_size:
                buf["chunk_index"] = flush_chunk(
                    buf["acts"],
                    buf["tks"],
                    buf["lens"],
                    buf["labels"],
                    buf["categories"],
                    buf["difficulties"],
                    buf["pair_ids"],
                    buf["questions"],
                    buf["responses"],
                    buf["sources"],
                    target,
                    extractor.model_name,
                    buf["chunk_index"],
                    output_dirs[target],
                    output_prefix,
                    f"deception_{mode}",
                )

        accepted += 1

    for target in targets:
        buf = buffers[target]
        flush_chunk(
            buf["acts"],
            buf["tks"],
            buf["lens"],
            buf["labels"],
            buf["categories"],
            buf["difficulties"],
            buf["pair_ids"],
            buf["questions"],
            buf["responses"],
            buf["sources"],
            target,
            extractor.model_name,
            buf["chunk_index"],
            output_dirs[target],
            output_prefix,
            f"deception_{mode}",
        )
        print(f"[extract_deception] Complete: {target} -> {output_dirs[target]}")

    print(f"[extract_deception] Accepted pairs={accepted} skipped={skipped}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)
    extractor = HFGenerationExtractor(config)
    dataset = DeceptionDataset(config=config, mode="paired")
    dataset.load()

    targets = [args.layer] if args.layer else resolve_requested_targets(config)
    extract_all_targets(targets, extractor, dataset, config)


if __name__ == "__main__":
    main()
