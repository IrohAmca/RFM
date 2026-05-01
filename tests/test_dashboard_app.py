import json

from rfm.config import ConfigManager
from rfm.dashboard.app import _pipeline_status, load_extraction_metadata


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


def test_load_extraction_metadata_uses_hashable_config_path(tmp_path):
    source_id = "5-gemmascope-transcoder-16k"
    other_source_id = "12-gemmascope-transcoder-16k"
    output_dir = tmp_path / "feature_store"
    source_dir = output_dir / source_id
    source_dir.mkdir(parents=True)
    (source_dir / "chunk.meta.json").write_text(
        json.dumps(
            {
                "chunk_id": 0,
                "source_id": source_id,
                "labels": ["truthful", "sycophantic"],
                "token_lengths": [4, 5],
                "categories": ["false_math", "false_math"],
                "difficulties": ["medium", "medium"],
                "questions": ["Q", "Q"],
                "responses": ["A", "B"],
                "sources": ["synthetic", "synthetic"],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "google/gemma-3-4b-it",
                "layers": {source_id: {}, other_source_id: {}},
                "extraction": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    df = load_extraction_metadata(str(config_path))

    assert len(df) == 2
    assert set(df["label"]) == {"truthful", "sycophantic"}
    assert df["layer"].iloc[0] == source_id
