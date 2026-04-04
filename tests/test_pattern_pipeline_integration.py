import json

from torch import nn
import torch

from cli.deception_autointerp import _load_top_features_from_bundle
from cli.deception_cycle import run_direction, run_monitor, run_patterns, run_probe
from cli.safety_score import cmd_cross_layer
from rfm.config import ConfigManager
from rfm.dashboard.app import (
    _pipeline_status,
    load_deception_autointerp,
    load_pattern_bundle_artifact,
    load_pattern_report_artifact,
)
from rfm.deception.reporting import generate_deception_report
from rfm.deception.utils import deception_run_dir
from rfm.layout import sanitize_layer_name
from rfm.patterns import ContrastAxisSpec, axis_monitor_from_bundle, load_pattern_bundle, pattern_artifact_paths


class IdentitySAE(nn.Module):
    def forward(self, x):
        return x, x

    def to(self, device):
        return self

    def eval(self):
        return self


def _write_chunk(path, label_a, label_b, *, offset=0.0):
    activations = []
    labels = []
    token_lengths = []
    pair_ids = []
    categories = []
    difficulties = []
    questions = []

    for pair_id in range(8):
        category = "sycophancy" if pair_id % 2 == 0 else "omission"
        difficulty = "easy" if pair_id < 4 else "hard"
        base_a = torch.tensor([offset + 0.05 * pair_id, offset + 0.10], dtype=torch.float32)
        base_b = base_a + torch.tensor([1.0, 1.1], dtype=torch.float32)

        for label, base in [(label_a, base_a), (label_b, base_b)]:
            activations.extend([base, base + torch.tensor([0.03, -0.02], dtype=torch.float32)])
            labels.append(label)
            token_lengths.append(2)
            pair_ids.append(pair_id)
            categories.append(category)
            difficulties.append(difficulty)
            questions.append(f"Q{pair_id}")

    torch.save(
        {
            "activations": torch.stack(activations),
            "tokens": torch.zeros(len(activations), dtype=torch.long),
            "metadata": {
                "labels": labels,
                "token_lengths": token_lengths,
                "pair_ids": pair_ids,
                "categories": categories,
                "difficulties": difficulties,
                "questions": questions,
            },
        },
        path,
    )


def _fake_causal_validation(_, motifs):
    effect_rows = []
    for index, motif in enumerate(motifs):
        supported = index == 0
        effect_rows.append(
            {
                "name": motif["name"],
                "kind": motif["kind"],
                "sign": motif.get("sign", "endpoint_b"),
                "supports_causal_effect": supported,
                "validation_backend": "generation_monitor",
                "aligned_ablation_shift": 0.08 if supported else 0.0,
                "aligned_amplification_shift": 0.03 if supported else 0.0,
            }
        )
    return effect_rows, {
        "status": "ok",
        "reason": "",
        "evaluated": len(effect_rows),
        "supported": sum(1 for row in effect_rows if row["supports_causal_effect"]),
        "prompt_source": "chunk_metadata",
        "monitor_backend": "generation_monitor",
    }


def test_cmd_cross_layer_persists_causal_validation_to_bundle_and_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    acts_dir = tmp_path / "acts"
    targets = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    for index, target in enumerate(targets):
        layer_dir = acts_dir / sanitize_layer_name(target)
        layer_dir.mkdir(parents=True)
        _write_chunk(layer_dir / "chunk.pt", "accept", "reject", offset=0.05 * index)

    monkeypatch.setattr("cli.safety_score.resolve_best_checkpoint", lambda config, target=None: "fake.pt")
    monkeypatch.setattr("cli.safety_score.load_sae_checkpoint", lambda *args, **kwargs: (IdentitySAE(), None))
    monkeypatch.setattr("cli.safety_score.MotifCausalValidator.evaluate", _fake_causal_validation)

    cfg = ConfigManager(
        {
            "model_name": "test/model",
            "contrast_axis": {
                "id": "review_axis",
                "endpoint_a": "accept",
                "endpoint_b": "reject",
                "display_name_a": "Accept",
                "display_name_b": "Reject",
                "pair_key_fields": ["pair_id", "label", "question"],
            },
            "layers": {target: {} for target in targets},
            "extraction": {
                "output_dir": str(acts_dir),
            },
            "train": {
                "device": "cpu",
            },
            "patterns": {
                "cv_folds": 2,
            },
        }
    )

    cmd_cross_layer(cfg, targets, top_k=5, output_base=None)

    axis = ContrastAxisSpec.from_config(cfg)
    bundle = load_pattern_bundle(cfg, axis)
    report = json.loads(pattern_artifact_paths(cfg, axis)["report"].read_text(encoding="utf-8"))

    assert bundle["analysis"]["causal_validation"]["status"] == "ok"
    assert report["analysis"]["causal_validation"]["supported"] == 1
    assert len(bundle["analysis"]["stable_motifs"]) == 1
    assert len(report["analysis"]["stable_motifs"]) == 1
    assert any(row["status"] == "hypothesis" for row in bundle["analysis"]["motif_candidates"])
    assert bundle["analysis"]["intervention_effects"][0]["validation_backend"] == "generation_monitor"


def test_run_patterns_feeds_dashboard_report_and_autointerp_consumers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    acts_dir = tmp_path / "acts"
    targets = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    for index, target in enumerate(targets):
        layer_dir = acts_dir / sanitize_layer_name(target)
        layer_dir.mkdir(parents=True)
        _write_chunk(layer_dir / "chunk.pt", "honest", "deceptive", offset=0.05 * index)

    scenario_path = tmp_path / "runs" / "test_model" / "deception" / "contextual_scenarios.jsonl"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text("", encoding="utf-8")

    monkeypatch.setattr("cli.deception_cycle.resolve_best_checkpoint", lambda config, target=None: "fake.pt")
    monkeypatch.setattr("cli.deception_cycle.load_sae_checkpoint", lambda *args, **kwargs: (IdentitySAE(), None))

    cfg = ConfigManager(
        {
            "model_name": "test/model",
            "contrast_axis": {
                "id": "deception",
                "endpoint_a": "honest",
                "endpoint_b": "deceptive",
                "display_name_a": "Honest",
                "display_name_b": "Deceptive",
                "pair_key_fields": ["pair_id", "label", "question"],
            },
            "layers": {target: {} for target in targets},
            "extraction": {
                "output_dir": str(acts_dir),
            },
            "train": {
                "device": "cpu",
                "split_seed": 42,
            },
            "deception": {
                "scenario_generator": {
                    "cache_path": str(scenario_path),
                },
                "direction": {
                    "method": "mean_diff",
                    "validation_split": 0.25,
                },
                "probe": {
                    "cross_validation_folds": 2,
                },
                "monitor": {
                    "alert_threshold": {target: 0.5 for target in targets},
                },
            },
            "patterns": {
                "cv_folds": 2,
            },
        }
    )

    run_direction(cfg)
    run_probe(cfg)

    axis = ContrastAxisSpec.from_config(cfg)
    monitor = axis_monitor_from_bundle(axis_spec=axis, bundle={}, config=cfg)
    assert monitor is not None
    assert set(monitor.monitored_layers) == set(targets)

    monkeypatch.setattr("cli.deception_cycle.MotifCausalValidator.evaluate", _fake_causal_validation)
    run_patterns(cfg)
    run_monitor(cfg)

    pattern_paths = pattern_artifact_paths(cfg, axis)
    bundle = load_pattern_bundle_artifact(str(pattern_paths["bundle"]))
    report = load_pattern_report_artifact(str(pattern_paths["report"]))

    assert bundle["analysis"]["causal_validation"]["status"] == "ok"
    assert report["analysis"]["causal_validation"]["supported"] == 1

    target = targets[0]
    safe = sanitize_layer_name(target)
    autointerp_path = deception_run_dir(cfg, "autointerp", f"{safe}_interpretations.json")
    autointerp_path.parent.mkdir(parents=True, exist_ok=True)
    autointerp_path.write_text(json.dumps({"1": "Deceptive coordination feature"}), encoding="utf-8")

    top_features = _load_top_features_from_bundle(cfg, target, top_n=2)
    rows = list(bundle["layers"][target]["feature_scores"])
    rows.sort(
        key=lambda row: (
            float(row.get("delta", 0.0)),
            abs(float(row.get("effect_size", 0.0))),
        ),
        reverse=True,
    )
    assert top_features == [int(row["feature_id"]) for row in rows[:2]]
    assert load_deception_autointerp(str(deception_run_dir(cfg)), target)[1] == "Deceptive coordination feature"

    status = _pipeline_status(str(deception_run_dir(cfg)), cfg)
    assert status["scenarios"] is True
    assert status["patterns"] is True
    assert status["probes"] is True
    assert status["monitor"] is True

    report_result = generate_deception_report(
        config=cfg,
        config_path="configs/test.deception.json",
        output_dir=None,
        top_features=5,
        max_projection_points=16,
    )
    html_text = report_result["html_path"].read_text(encoding="utf-8")
    assert "Pattern report" in html_text
    assert report_result["html_path"].exists()
