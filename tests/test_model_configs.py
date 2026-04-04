import json
from pathlib import Path


CONFIG_DIR = Path("configs/models")


def _load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_all_model_configs_include_patterns_block():
    for path in CONFIG_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "patterns" in payload, f"{path.name} is missing the canonical patterns block"


def test_example_and_safety_configs_include_axis_metadata():
    example = _load_config("config.example.json")
    safety = _load_config("qwen3-0.6B.safety.json")
    safety_gen = _load_config("qwen3-0.6B.safety-gen.json")

    assert example["contrast_axis"]["id"] == "contrast_axis"
    assert safety["contrast_axis"]["endpoint_a"] == "safe"
    assert safety["contrast_axis"]["endpoint_b"] == "toxic"
    assert safety_gen["contrast_axis"]["endpoint_a"] == "safe"
    assert safety_gen["contrast_axis"]["endpoint_b"] == "toxic"
