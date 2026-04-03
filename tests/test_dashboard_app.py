from rfm.config import ConfigManager
from rfm.dashboard.app import _pipeline_status


def test_pipeline_status_uses_configured_scenario_cache(tmp_path):
    dec_dir = tmp_path / "runs" / "demo_model" / "deception"
    dec_dir.mkdir(parents=True)

    scenario_path = tmp_path / "custom_cache" / "scenarios.jsonl"
    scenario_path.parent.mkdir(parents=True)
    scenario_path.write_text("", encoding="utf-8")

    config = ConfigManager(
        {
            "deception": {
                "scenario_generator": {
                    "cache_path": str(scenario_path),
                }
            }
        }
    )

    status = _pipeline_status(str(dec_dir), config)
    assert status["scenarios"] is True
