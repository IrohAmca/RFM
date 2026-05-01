import pytest

from cli.sycophancy_cycle import run_extract, run_generate, run_patterns, run_probe, run_validate
from rfm.config import ConfigManager


def _config(tmp_path):
    return ConfigManager(
        {
            "model_name": "google/gemma-3-4b-it",
            "contrast_axis": {
                "id": "sycophancy",
                "endpoint_a": "truthful",
                "endpoint_b": "sycophantic",
                "display_name_a": "Truthful",
                "display_name_b": "Sycophantic",
            },
            "sycophancy": {
                "scenario_count": 8,
                "scenario_path": str(tmp_path / "runs" / "gemma" / "sycophancy" / "scenarios.jsonl"),
                "feature_store": {"top_k": 4},
                "gemma_scope": {
                    "source_priority": ["resid_post_all"],
                    "layers": [9, 17],
                    "width": "16k",
                    "d_sae": 12,
                    "l0_fallback": ["medium", "small", "big", "large"],
                },
            },
            "train": {"device": "cpu"},
            "patterns": {
                "aggregation_candidates": ["mean", "max"],
                "cv_folds": 2,
                "feature_pool": {"endpoint_a": 3, "endpoint_b": 3, "interaction": 1},
                "max_interaction_features_per_layer": 1,
            },
        }
    )


def test_sycophancy_cycle_dry_run_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)

    scenario_path = run_generate(cfg, limit=8)
    assert scenario_path.exists()

    probe = run_probe(cfg, limit=2, dry_run=True)
    assert probe["status"] == "ok"

    extract = run_extract(cfg, limit=8, dry_run=True)
    assert extract["written"]

    patterns = run_patterns(cfg)
    assert patterns["selected_aggregation"] in {"mean", "max"}

    validation = run_validate(cfg)
    assert validation["status"] == "ok"


def test_sycophancy_extract_respects_disabled_auto_generate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    cfg.set("sycophancy.auto_generate_scenarios", False)

    with pytest.raises(FileNotFoundError, match="run generate first"):
        run_extract(cfg, dry_run=True)
