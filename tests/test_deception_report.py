import csv
import json

import torch

from cli.deception_cycle import run_direction, run_monitor, run_probe
from rfm.config import ConfigManager
from rfm.deception.reporting import generate_deception_report
from rfm.deception.utils import deception_run_dir
from rfm.layout import sanitize_layer_name


def _write_chunk(path, offset=0.0):
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
        honest_base = torch.tensor([offset + 0.05 * pair_id, offset + 0.10], dtype=torch.float32)
        deceptive_base = honest_base + torch.tensor([1.0, 1.1], dtype=torch.float32)

        for label, base in [("honest", honest_base), ("deceptive", deceptive_base)]:
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


def _write_contrastive_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_id",
                "deceptive_rate",
                "honest_rate",
                "rate_ratio",
                "fisher_score",
                "risk_score",
                "direction",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "feature_id": 10,
                    "deceptive_rate": 0.40,
                    "honest_rate": 0.05,
                    "rate_ratio": 8.0,
                    "fisher_score": 0.30,
                    "risk_score": 0.90,
                    "direction": "deceptive",
                },
                {
                    "feature_id": 21,
                    "deceptive_rate": 0.22,
                    "honest_rate": 0.08,
                    "rate_ratio": 2.75,
                    "fisher_score": 0.15,
                    "risk_score": 0.45,
                    "direction": "deceptive",
                },
            ]
        )


def test_generate_deception_report_writes_html_and_pngs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    targets = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    acts_dir = tmp_path / "acts"
    for index, target in enumerate(targets):
        layer_dir = acts_dir / sanitize_layer_name(target)
        layer_dir.mkdir(parents=True)
        _write_chunk(layer_dir / "chunk.pt", offset=0.05 * index)

    cfg = ConfigManager(
        {
            "model_name": "test/model",
            "layers": {target: {} for target in targets},
            "extraction": {
                "output_dir": str(acts_dir),
            },
            "train": {
                "device": "cpu",
                "split_seed": 42,
            },
            "deception": {
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
        }
    )

    run_direction(cfg)
    run_probe(cfg)
    run_monitor(cfg)

    dec_dir = deception_run_dir(cfg)
    safety_dir = dec_dir / "contextual_activations" / "safety_scores"
    summary_dir = tmp_path / "runs" / "test_model" / "safety_scores"
    summary_dir.mkdir(parents=True)

    summary_payload = {}
    for index, target in enumerate(targets):
        safe = sanitize_layer_name(target)
        _write_contrastive_csv(safety_dir / f"{safe}_contrastive.csv")
        (dec_dir / "probes").mkdir(parents=True, exist_ok=True)
        (dec_dir / "probes" / f"{safe}_sae_features.json").write_text(
            json.dumps(
                [
                    {
                        "feature_id": 10 + index,
                        "cosine_similarity": 0.75 - 0.05 * index,
                        "alignment": "aligned",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_payload[target] = [{"feature_id": 10 + index, "risk_score": 0.8, "rate_ratio": 7.0}]

    (summary_dir / "contrastive_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    adversarial_dir = dec_dir / "adversarial"
    adversarial_dir.mkdir(parents=True, exist_ok=True)
    (adversarial_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_missed": 3,
                "by_category": {"omission": 2, "sycophancy": 1},
                "by_difficulty": {"easy": 1, "hard": 2},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = generate_deception_report(
        config=cfg,
        config_path="configs/test.deception.json",
        output_dir=None,
        top_features=5,
        max_projection_points=16,
    )

    report_dir = result["report_dir"]
    html_path = result["html_path"]

    assert report_dir.exists()
    assert html_path.exists()
    assert "Deception Monitor Report" in html_path.read_text(encoding="utf-8")
    assert "Top Risky Features" in html_path.read_text(encoding="utf-8")

    for chart_name in [
        "layer_comparison.png",
        "tsne_honest_vs_deceptive.png",
        "probe_roc_curve.png",
        "category_breakdown.png",
        "adversarial_analysis.png",
    ]:
        chart_path = report_dir / chart_name
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0
