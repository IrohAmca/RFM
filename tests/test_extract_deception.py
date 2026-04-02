import json

import torch

from cli.extract_deception import extract_all_targets
from rfm.config import ConfigManager
from rfm.deception.deception_dataset import DeceptionDataset


class _Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"


class _FakeExtractor:
    def __init__(self):
        self.model_name = "test/model"
        self.tokenizer = _Tokenizer()

    def extract_replay_multi(self, prompt, response, targets):
        value = 0.0 if response.startswith("H") else 1.0
        payload = {}
        for target in targets:
            payload[target] = {
                "activations": torch.tensor([[value, value + 0.5]], dtype=torch.float32),
                "tokens": torch.tensor([1], dtype=torch.long),
            }
        return payload


def _write_dataset(path):
    rows = [
        {
            "question": "Q1",
            "honest_answer": "H1",
            "deceptive_answer": "D1",
            "category": "context_contradiction",
            "difficulty": "easy",
            "metadata": {"failure_mode": "contradiction"},
        },
        {
            "question": "Q2",
            "honest_answer": "H2",
            "deceptive_answer": "D2",
            "category": "omission",
            "difficulty": "medium",
            "metadata": {"failure_mode": "omission"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _make_config(output_dir, count):
    return ConfigManager(
        {
            "model_name": "test/model",
            "layers": {
                "blocks.0.hook_resid_post": {},
            },
            "extraction": {
                "output_dir": str(output_dir),
                "chunk_size": 10_000,
                "dtype": "float32",
            },
            "deception": {
                "extraction": {
                    "mode": "replay",
                    "count": count,
                    "output_prefix": "deception_contextual_activations",
                },
            },
        }
    )


def test_extract_deception_resumes_from_existing_chunks(tmp_path):
    dataset_path = tmp_path / "scenarios.jsonl"
    _write_dataset(dataset_path)

    dataset = DeceptionDataset(input_path=dataset_path, mode="paired")
    dataset.load()
    extractor = _FakeExtractor()
    targets = ["blocks.0.hook_resid_post"]

    extract_all_targets(targets, extractor, dataset, _make_config(tmp_path / "acts", count=1))
    extract_all_targets(targets, extractor, dataset, _make_config(tmp_path / "acts", count=2))

    meta_files = sorted((tmp_path / "acts").glob("*.meta.json"))
    assert len(meta_files) == 2

    meta_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in meta_files]
    assert meta_payloads[0]["pair_ids"] == [0, 0]
    assert meta_payloads[1]["pair_ids"] == [1, 1]
