import pytest
from rfm.layout import (
    sanitize_model_name,
    sanitize_layer_name,
    model_slug,
    default_activations_dir,
    default_checkpoint_path,
    default_feature_mapping_dir,
    _is_multi_layer,
)
from rfm.config import ConfigManager


class TestSanitize:
    def test_sanitize_model_name_slash(self):
        assert sanitize_model_name("ytu-ce-cosmos/turkish-gpt2") == "ytu-ce-cosmos_turkish-gpt2"

    def test_sanitize_model_name_backslash(self):
        assert sanitize_model_name("path\\model") == "path_model"

    def test_sanitize_model_name_spaces(self):
        assert sanitize_model_name("my model name") == "my_model_name"

    def test_sanitize_layer_name(self):
        assert sanitize_layer_name("blocks.6.hook_resid_post") == "blocks_6_hook_resid_post"

    def test_sanitize_layer_name_simple(self):
        assert sanitize_layer_name("layer.0") == "layer_0"


class TestModelSlug:
    def test_with_config_manager(self):
        cfg = ConfigManager({"model_name": "gpt2-small"})
        assert model_slug(cfg) == "gpt2-small"

    def test_with_dict(self):
        assert model_slug({"model_name": "test/model"}) == "test_model"

    def test_default(self):
        cfg = ConfigManager({})
        assert model_slug(cfg) == "gpt2-small"


class TestMultiLayerDetection:
    def test_single_target_is_not_multi(self):
        cfg = ConfigManager({"extraction": {"target": "blocks.0.hook_resid_post"}})
        assert not _is_multi_layer(cfg)

    def test_list_target_is_multi(self):
        cfg = ConfigManager({"extraction": {"target": ["blocks.0.hook_resid_post", "blocks.6.hook_resid_post"]}})
        assert _is_multi_layer(cfg)

    def test_single_element_list_is_not_multi(self):
        cfg = ConfigManager({"extraction": {"target": ["blocks.0.hook_resid_post"]}})
        assert not _is_multi_layer(cfg)


class TestDefaultPaths:
    def test_single_layer_activations(self):
        cfg = ConfigManager({"model_name": "gpt2-small"})
        path = default_activations_dir(cfg)
        assert "gpt2-small" in path
        assert "activations" in path

    def test_multi_layer_activations_with_target(self):
        cfg = ConfigManager({
            "model_name": "gpt2-small",
            "extraction": {"target": ["blocks.0.hook_resid_post", "blocks.6.hook_resid_post"]},
        })
        path = default_activations_dir(cfg, target="blocks.6.hook_resid_post")
        assert "blocks_6_hook_resid_post" in path

    def test_single_layer_checkpoint(self):
        cfg = ConfigManager({"model_name": "gpt2-small"})
        path = default_checkpoint_path(cfg)
        assert path.endswith("sae.pt")

    def test_multi_layer_checkpoint_with_target(self):
        cfg = ConfigManager({
            "model_name": "gpt2-small",
            "extraction": {"target": ["blocks.0.hook_resid_post", "blocks.11.hook_resid_post"]},
        })
        path = default_checkpoint_path(cfg, target="blocks.11.hook_resid_post")
        assert "blocks_11_hook_resid_post" in path
        assert path.endswith("sae.pt")

    def test_feature_mapping_dir(self):
        cfg = ConfigManager({"model_name": "gpt2-small"})
        path = default_feature_mapping_dir(cfg)
        assert "feature_mapping" in path
