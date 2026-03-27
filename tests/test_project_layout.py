import pytest
from rfm.layout import (
    sanitize_model_name,
    sanitize_layer_name,
    model_slug,
    default_activations_dir,
    default_checkpoint_path,
    default_feature_mapping_dir,
    resolve_activations_dir,
    _is_multi_layer,
    resolve_requested_targets,
    select_targets_from,
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

    def test_pipeline_from_hook_keeps_multi_layer_layout(self):
        cfg = ConfigManager({
            "model_name": "gpt2-small",
            "extraction": {
                "target": [
                    "blocks.0.hook_resid_post",
                    "blocks.6.hook_resid_post",
                ]
            },
            "pipeline": {"from_hook": "6"},
        })
        assert _is_multi_layer(cfg)
        assert "blocks_6_hook_resid_post" in default_activations_dir(cfg, target="blocks.6.hook_resid_post")

    def test_target_specific_config_from_layers_stays_multi_layer_for_paths(self):
        cfg = ConfigManager({
            "model_name": "Qwen/Qwen3-0.6B",
            "layers": {
                "blocks.0.hook_resid_post": {},
                "blocks.13.hook_resid_post": {"sae": {"hidden_dim": 24576}},
                "blocks.27.hook_resid_post": {"sae": {"hidden_dim": 32768}},
            },
        })
        target_cfg = cfg.for_target("blocks.13.hook_resid_post")

        assert _is_multi_layer(target_cfg)
        assert default_activations_dir(
            target_cfg,
            target="blocks.13.hook_resid_post",
        ).endswith("runs\\Qwen_Qwen3-0.6B\\activations\\blocks_13_hook_resid_post")


class TestRequestedTargets:
    def test_exact_hook_match(self):
        cfg = ConfigManager({
            "extraction": {
                "target": [
                    "blocks.0.hook_resid_post",
                    "blocks.6.hook_resid_post",
                    "blocks.11.hook_resid_post",
                ]
            },
            "pipeline": {"from_hook": "blocks.6.hook_resid_post"},
        })
        assert resolve_requested_targets(cfg) == [
            "blocks.6.hook_resid_post",
            "blocks.11.hook_resid_post",
        ]

    def test_numeric_hook_match(self):
        cfg = ConfigManager({
            "extraction": {
                "target": [
                    "blocks.0.hook_resid_post",
                    "blocks.6.hook_resid_post",
                    "blocks.11.hook_resid_post",
                ]
            },
            "pipeline": {"from_hook": "11"},
        })
        assert resolve_requested_targets(cfg) == ["blocks.11.hook_resid_post"]

    def test_missing_hook_raises(self):
        with pytest.raises(ValueError, match="not found"):
            select_targets_from(
                ["blocks.0.hook_resid_post", "blocks.6.hook_resid_post"],
                "27",
            )

    def test_layers_section_is_used_when_target_missing(self):
        cfg = ConfigManager({
            "layers": {
                "blocks.0.hook_resid_post": {},
                "blocks.6.hook_resid_post": {},
                "blocks.11.hook_resid_post": {},
            },
            "pipeline": {"from_hook": "6"},
        })
        assert resolve_requested_targets(cfg) == [
            "blocks.6.hook_resid_post",
            "blocks.11.hook_resid_post",
        ]


class TestDefaultPaths:
    def test_single_layer_activations(self):
        cfg = ConfigManager({"model_name": "gpt2-small"})
        path = default_activations_dir(cfg)
        assert "gpt2-small" in path
        assert "activations" in path

    def test_multi_layer_activations_with_target(self):
        cfg = ConfigManager({
            "model_name": "gpt2-small",
            "layers": {
                "blocks.0.hook_resid_post": {},
                "blocks.6.hook_resid_post": {},
            },
        })
        path = default_activations_dir(cfg, target="blocks.6.hook_resid_post")
        assert "blocks_6_hook_resid_post" in path

    def test_resolve_activations_dir_uses_custom_output_dir(self):
        cfg = ConfigManager({
            "model_name": "gpt2-small",
            "layers": {
                "blocks.0.hook_resid_post": {},
                "blocks.6.hook_resid_post": {},
            },
            "extraction": {"output_dir": "custom_acts"},
        })
        path = resolve_activations_dir(cfg, target="blocks.6.hook_resid_post")
        assert path.endswith("custom_acts\\blocks_6_hook_resid_post")

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
