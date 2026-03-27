import pytest
from rfm.config import ConfigManager


class TestConfigManagerGet:
    def test_top_level_key(self):
        cfg = ConfigManager({"model_name": "test-model"})
        assert cfg.get("model_name") == "test-model"

    def test_dot_notation_access(self):
        cfg = ConfigManager({"extraction": {"target": "blocks.0.hook_resid_post"}})
        assert cfg.get("extraction.target") == "blocks.0.hook_resid_post"

    def test_missing_key_returns_default(self):
        cfg = ConfigManager({})
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_missing_key_returns_none(self):
        cfg = ConfigManager({})
        assert cfg.get("nonexistent.key") is None


class TestConfigManagerDeepMerge:
    def test_override_nested_value(self):
        cfg = ConfigManager({"sae": {"hidden_dim": 9999}})
        assert cfg.get("sae.hidden_dim") == 9999
        assert cfg.get("sae.sparsity_weight") == 1e-3

    def test_add_new_section(self):
        cfg = ConfigManager({"custom": {"key": "value"}})
        assert cfg.get("custom.key") == "value"

    def test_default_values_preserved(self):
        cfg = ConfigManager({})
        assert cfg.get("model_name") == "gpt2-small"
        assert cfg.get("extraction.target") == "blocks.0.hook_resid_post"

    def test_layers_without_explicit_target_clear_default_target(self):
        cfg = ConfigManager(
            {
                "layers": {
                    "blocks.6.hook_resid_post": {},
                }
            }
        )
        assert cfg.get("extraction.target") is None


class TestConfigManagerSet:
    def test_set_simple_key(self):
        cfg = ConfigManager({})
        cfg.set("model_name", "new-model")
        assert cfg.get("model_name") == "new-model"

    def test_set_nested_key(self):
        cfg = ConfigManager({})
        cfg.set("extraction.target", "blocks.11.hook_resid_post")
        assert cfg.get("extraction.target") == "blocks.11.hook_resid_post"

    def test_set_creates_intermediate_dicts(self):
        cfg = ConfigManager({})
        cfg.set("new.deeply.nested.key", 42)
        assert cfg.get("new.deeply.nested.key") == 42


class TestConfigManagerSection:
    def test_section_returns_copy(self):
        cfg = ConfigManager({"sae": {"hidden_dim": 3072}})
        section = cfg.section("sae")
        section["hidden_dim"] = 9999
        assert cfg.get("sae.hidden_dim") == 3072

    def test_missing_section_returns_empty_dict(self):
        cfg = ConfigManager({})
        assert cfg.section("nonexistent") == {}


class TestConfigManagerForTarget:
    def test_for_target_applies_layer_specific_sections(self):
        cfg = ConfigManager(
            {
                "layers": {
                    "blocks.13.hook_resid_post": {
                        "sae": {"hidden_dim": 24576, "topk_k": 192},
                        "train": {"epochs": 40, "learning_rate": 3e-4},
                    }
                }
            }
        )

        target_cfg = cfg.for_target("blocks.13.hook_resid_post")

        assert target_cfg.get("extraction.target") == "blocks.13.hook_resid_post"
        assert target_cfg.get("sae.hidden_dim") == 24576
        assert target_cfg.get("sae.topk_k") == 192
        assert target_cfg.get("train.epochs") == 40
        assert target_cfg.get("train.learning_rate") == 3e-4


class TestConfigManagerValidation:
    def test_valid_config(self):
        cfg = ConfigManager({"model_name": "gpt2-small"})
        errors = cfg.validate()
        assert len(errors) == 0

    def test_missing_target(self):
        cfg = ConfigManager({})
        cfg.set("extraction.target", None)
        errors = cfg.validate()
        assert any("extraction.target" in e for e in errors)

    def test_layers_only_config_is_valid(self):
        cfg = ConfigManager(
            {
                "layers": {
                    "blocks.0.hook_resid_post": {},
                    "blocks.6.hook_resid_post": {"train": {"epochs": 10}},
                }
            }
        )
        assert cfg.validate() == []

    def test_empty_target_list(self):
        cfg = ConfigManager({})
        cfg.set("extraction.target", [])
        errors = cfg.validate()
        assert any("empty" in e for e in errors)

    def test_target_list_with_valid_entries(self):
        cfg = ConfigManager({})
        cfg.set("extraction.target", ["blocks.0.hook_resid_post", "blocks.6.hook_resid_post"])
        errors = cfg.validate()
        assert len(errors) == 0

    def test_target_list_with_invalid_entry(self):
        cfg = ConfigManager({})
        cfg.set("extraction.target", ["blocks.0.hook_resid_post", ""])
        errors = cfg.validate()
        assert any("invalid" in e for e in errors)


class TestConfigManagerFromFile:
    def test_none_returns_defaults(self):
        cfg = ConfigManager.from_file(None)
        assert cfg.get("model_name") == "gpt2-small"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ConfigManager.from_file("nonexistent_file.json")

    def test_unsupported_format_raises(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"key: value")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported"):
                ConfigManager.from_file(f.name)

    def test_toml_file_loads(self):
        import tempfile

        toml_content = """
model_name = "test-model"

[extraction]
target = "blocks.11.hook_resid_post"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content.encode("utf-8"))
            f.flush()

            cfg = ConfigManager.from_file(f.name)

        assert cfg.get("model_name") == "test-model"
        assert cfg.get("extraction.target") == "blocks.11.hook_resid_post"
