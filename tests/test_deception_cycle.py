import torch

from cli.deception_cycle import _split_pairs, run_direction, run_extract, run_monitor, run_phase, run_probe
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


def test_run_extract_auto_generates_scenarios(monkeypatch):
    calls = {"generate": 0, "extract": 0}

    monkeypatch.setattr(
        "cli.deception_cycle.run_generate",
        lambda config: calls.__setitem__("generate", calls["generate"] + 1) or [],
    )

    class _FakeExtractor:
        def __init__(self, config):
            self.model_name = "test/model"
            self.tokenizer = object()

    class _FakeDataset:
        def __init__(self, config=None, mode="paired"):
            self.config = config
            self.mode = mode

        def load(self):
            return None

    monkeypatch.setattr("cli.deception_cycle.HFGenerationExtractor", _FakeExtractor)
    monkeypatch.setattr("cli.deception_cycle.DeceptionDataset", _FakeDataset)
    monkeypatch.setattr(
        "cli.deception_cycle.extract_all_targets",
        lambda targets, extractor, dataset, config: calls.__setitem__("extract", calls["extract"] + 1),
    )

    cfg = ConfigManager(
        {
            "model_name": "test/model",
            "layers": {
                "blocks.0.hook_resid_post": {},
            },
            "deception": {
                "extraction": {
                    "auto_generate_scenarios": True,
                },
            },
        }
    )

    run_extract(cfg, "blocks.0.hook_resid_post", ensure_scenarios=True)

    assert calls["generate"] == 1
    assert calls["extract"] == 1


def test_run_phase_train_delegates(monkeypatch):
    captured = {}

    def _fake_run_train(config, layer_override=None):
        captured["layer"] = layer_override
        return {"started": True}

    monkeypatch.setattr("cli.deception_cycle.run_train", _fake_run_train)

    cfg = ConfigManager({"model_name": "test/model"})
    result = run_phase(cfg, "train", "blocks.0.hook_resid_post")

    assert result == {"started": True}
    assert captured["layer"] == "blocks.0.hook_resid_post"


def test_run_phase_full_includes_train(monkeypatch):
    calls = []

    def _recorder(name):
        def _inner(*args, **kwargs):
            calls.append((name, kwargs))
            return {}

        return _inner

    monkeypatch.setattr("cli.deception_cycle.run_generate", _recorder("generate"))
    monkeypatch.setattr("cli.deception_cycle.run_extract", _recorder("extract"))
    monkeypatch.setattr("cli.deception_cycle.run_train", _recorder("train"))
    monkeypatch.setattr("cli.deception_cycle.run_direction", _recorder("direction"))
    monkeypatch.setattr("cli.deception_cycle.run_probe", _recorder("probe"))
    monkeypatch.setattr("cli.deception_cycle.run_monitor", _recorder("monitor"))
    monkeypatch.setattr("cli.deception_cycle.run_adversarial", _recorder("adversarial"))

    cfg = ConfigManager({"model_name": "test/model"})
    run_phase(cfg, "full", "blocks.0.hook_resid_post")

    assert [name for name, _ in calls] == [
        "generate",
        "extract",
        "train",
        "direction",
        "probe",
        "monitor",
        "adversarial",
    ]
    assert calls[1][1] == {"ensure_scenarios": False}


def test_split_pairs_is_deterministic_and_preserves_pair_alignment():
    honest = torch.arange(12, dtype=torch.float32).view(6, 2)
    deceptive = honest + 100

    split_a = _split_pairs(honest, deceptive, validation_split=0.33, seed=7)
    split_b = _split_pairs(honest, deceptive, validation_split=0.33, seed=7)

    for tensor_a, tensor_b in zip(split_a, split_b):
        assert torch.equal(tensor_a, tensor_b)

    train_h, train_d, val_h, val_d = split_a
    assert torch.equal(train_d - train_h, torch.full_like(train_h, 100))
    assert torch.equal(val_d - val_h, torch.full_like(val_h, 100))
