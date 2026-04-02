import torch

from cli.deception_cycle import run_direction, run_monitor, run_probe
from rfm.config import ConfigManager


def _write_chunk(path):
    torch.save(
        {
            "activations": torch.tensor(
                [
                    [0.0, 0.1],
                    [0.1, 0.0],
                    [1.0, 1.1],
                    [1.2, 0.9],
                    [0.0, 0.0],
                    [0.2, 0.1],
                    [1.1, 1.0],
                    [0.9, 1.2],
                ],
                dtype=torch.float32,
            ),
            "tokens": torch.zeros(8, dtype=torch.long),
            "metadata": {
                "labels": ["honest", "deceptive", "honest", "deceptive"],
                "token_lengths": [2, 2, 2, 2],
                "pair_ids": [0, 0, 1, 1],
                "categories": ["factual_lying"] * 4,
                "difficulties": ["medium"] * 4,
                "questions": ["Q1", "Q1", "Q2", "Q2"],
            },
        },
        path,
    )


def test_deception_cycle_direction_probe_monitor_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    _write_chunk(acts_dir / "chunk.pt")

    cfg = ConfigManager(
        {
            "model_name": "test/model",
            "layers": {
                "blocks.0.hook_resid_post": {},
            },
            "extraction": {
                "output_dir": str(acts_dir),
            },
            "train": {
                "device": "cpu",
            },
            "deception": {
                "direction": {
                    "method": "mean_diff",
                    "validation_split": 0.5,
                },
                "probe": {
                    "cross_validation_folds": 2,
                },
                "monitor": {
                    "alert_threshold": 0.5,
                },
            },
        }
    )

    directions = run_direction(cfg, "blocks.0.hook_resid_post")
    assert "blocks.0.hook_resid_post" in directions

    probe_summary = run_probe(cfg, "blocks.0.hook_resid_post")
    assert probe_summary["blocks.0.hook_resid_post"]["validation_accuracy"] >= 0.5

    monitor_report = run_monitor(cfg, "blocks.0.hook_resid_post")
    assert monitor_report["detection_rate"] >= 0.5
