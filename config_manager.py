from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "model_name": "gpt2-small",
    "model": {
        "device": None,
        "dtype": None,
    },
    "dataloader": {
        "dataset_name": "bigcode/the-stack-smol",
        "dataset_config_name": None,
        "split": "train",
        "data_dir": None,
        "streaming": True,
        "text_field": "content",
        "filter_field": None,
        "filter_values": None,
    },
    "extraction": {
        "extractor_backend": "transformer_lens",
        "target": "blocks.0.hook_resid_post",
        "count": 100,
        "chunk_size": 1_000_000,
        "dtype": "bfloat16",
        "output_dir": ".",
        "output_prefix": "activations",
    },
    "datasets": {
        "path": [],
    },
    "sae": {
        "hidden_dim": 3072,
        "sparsity_weight": 1e-3,
        "sparsity_sweep": [],
    },
    "train": {
        "batch_size": 1024,
        "epochs": 5,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "device": "cuda",
        "validation_split": 0.1,
        "split_seed": 42,
        "feature_activity_threshold": 1e-3,
        "output_model_path": "checkpoints/sae.pt",
    },
}


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ConfigManager:
    def __init__(self, config=None):
        config = config or {}
        self._config = _deep_merge(DEFAULT_CONFIG, config)

    @classmethod
    def from_file(cls, file_path=None):
        if file_path is None:
            return cls()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(data)

        if suffix == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "TOML config requires Python 3.11+ for tomllib or installing tomli."
                ) from exc
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return cls(data)

        raise ValueError("Unsupported config format. Use .json or .toml")

    def as_dict(self):
        return copy.deepcopy(self._config)

    def get(self, key, default=None):
        if "." not in key:
            return self._config.get(key, default)

        current = self._config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set(self, key, value):
        """Set a config value using dot-notation key."""
        parts = key.split(".")
        current = self._config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def section(self, name):
        value = self._config.get(name, {})
        if not isinstance(value, dict):
            return {}
        return copy.deepcopy(value)

    def validate(self):
        """Validate config structure. Returns list of error messages (empty = valid)."""
        errors = []

        target = self.get("extraction.target")
        if target is None:
            errors.append("extraction.target is required")
        elif isinstance(target, list):
            if not target:
                errors.append("extraction.target list must not be empty")
            for t in target:
                if not isinstance(t, str) or not t.strip():
                    errors.append(f"extraction.target contains invalid entry: {t!r}")
        elif not isinstance(target, str) or not target.strip():
            errors.append(f"extraction.target must be a string or list, got: {type(target).__name__}")

        model_name = self.get("model_name")
        if not model_name or not isinstance(model_name, str):
            errors.append("model_name is required and must be a non-empty string")

        return errors

