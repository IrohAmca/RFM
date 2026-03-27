from rfm.config import ConfigManager


def test_for_target_handles_missing_layer_without_mutating_defaults():
    cfg = ConfigManager(
        {
            "sae": {"hidden_dim": 12288},
            "layers": {
                "blocks.13.hook_resid_post": {
                    "sae": {"hidden_dim": 24576},
                }
            },
        }
    )

    target_cfg = cfg.for_target("blocks.6.hook_resid_post")

    assert target_cfg.get("sae.hidden_dim") == 12288
    assert cfg.get("sae.hidden_dim") == 12288
