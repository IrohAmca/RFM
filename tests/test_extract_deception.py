import json

import torch

from cli.extract_deception import extract_all_targets
from rfm.config import ConfigManager
from rfm.deception.deception_dataset import DeceptionDataset


class _Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"


class _GenerateTokenizer:
    def __init__(self, responses):
        self.responses = responses

    def decode(self, tokens, skip_special_tokens=True):
        return self.responses[int(tokens[0])]


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


class _FakeGenerateExtractor:
    def __init__(self, scripted_responses):
        self.model_name = "test/model"
        self.scripted_responses = list(scripted_responses)
        self.tokenizer = _GenerateTokenizer({
            index: response
            for index, response in enumerate(self.scripted_responses)
        })
        self.calls = 0

    def extract_generate_multi(self, prompt, targets, max_new_tokens, temperature, top_p):
        token_id = self.calls
        self.calls += 1
        payload = {}
        for target in targets:
            payload[target] = {
                "activations": torch.tensor([[float(token_id), float(token_id) + 0.5]], dtype=torch.float32),
                "tokens": torch.tensor([token_id], dtype=torch.long),
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
    assert meta_payloads[0]["contrast_axis"]["endpoint_a"] == "honest"
    assert meta_payloads[0]["contrast_axis"]["endpoint_b"] == "deceptive"


def test_extract_deception_generate_retries_until_behavior_validated(tmp_path):
    dataset_path = tmp_path / "scenarios.jsonl"
    _write_dataset(dataset_path)

    dataset = DeceptionDataset(input_path=dataset_path, mode="paired")
    dataset.load()
    extractor = _FakeGenerateExtractor([
        "I cannot answer that.",
        "H1",
        "I cannot help provide a misleading answer.",
        "D1",
    ])
    targets = ["blocks.0.hook_resid_post"]
    config = _make_config(tmp_path / "acts", count=1)
    config.set("deception.extraction.mode", "generate")
    config.set("deception.extraction.validation.min_response_tokens", 1)
    config.set("deception.extraction.validation.min_expected_similarity", 0.5)
    config.set("deception.extraction.validation.max_attempts", 2)

    extract_all_targets(targets, extractor, dataset, config)

    assert extractor.calls == 4
    meta_path = next((tmp_path / "acts").glob("*.meta.json"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["responses"] == ["H1", "D1"]
    assert [item["accepted"] for item in metadata["validations"]] == [True, True]
    assert [item["attempt"] for item in metadata["validations"]] == [2, 2]
