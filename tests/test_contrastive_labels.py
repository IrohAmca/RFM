import csv
import json

import torch
from torch import nn

from cli.safety_score import cmd_contrastive
from rfm.config import ConfigManager
from rfm.safety.contrastive import ContrastiveScorer


class IdentitySAE(nn.Module):
    def forward(self, x):
        return x, x

    def to(self, device):
        return self

    def eval(self):
        return self


def test_contrastive_scorer_supports_deception_labels(tmp_path):
    chunk_dir = tmp_path / "acts"
    chunk_dir.mkdir()
    torch.save(
        {
            "activations": torch.tensor(
                [
                    [0.0, 0.1],
                    [0.2, 0.1],
                    [1.0, 0.8],
                    [0.9, 1.1],
                ],
                dtype=torch.float32,
            ),
            "tokens": torch.zeros(4, dtype=torch.long),
            "metadata": {
                "labels": ["honest", "deceptive"],
                "token_lengths": [2, 2],
            },
        },
        chunk_dir / "chunk.pt",
    )

    scorer = ContrastiveScorer(IdentitySAE(), device="cpu")
    scores = scorer.score_from_chunks(
        chunk_dir,
        positive_label="deceptive",
        negative_label="honest",
    )

    assert scores
    assert "deceptive_rate" in scores[0]
    top = scorer.top_dangerous(scores, top_k=1, direction="deceptive")
    assert top


def test_cmd_contrastive_writes_summary_next_to_resolved_safety_scores(tmp_path, monkeypatch):
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    (acts_dir / "chunk.pt").write_bytes(b"stub")

    class FakeScorer:
        def __init__(self, sae_model, device="cpu"):
            self.sae_model = sae_model
            self.device = device

        def score_from_chunks(self, chunk_dir, positive_label="deceptive", negative_label="honest"):
            return [
                {
                    "feature_id": 7,
                    f"{positive_label}_rate": 0.4,
                    f"{negative_label}_rate": 0.1,
                    "rate_ratio": 4.0,
                    "fisher_score": 0.25,
                    "risk_score": 0.6,
                    "direction": positive_label,
                }
            ]

        def save_scores(self, scores, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(scores[0].keys()))
                writer.writeheader()
                writer.writerows(scores)

        def top_dangerous(self, scores, top_k=50, direction="deceptive"):
            return scores[:top_k]

    monkeypatch.setattr("cli.safety_score.resolve_best_checkpoint", lambda config, target=None: "fake.pt")
    monkeypatch.setattr("cli.safety_score.load_sae_checkpoint", lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr("cli.safety_score.ContrastiveScorer", FakeScorer)

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
            "deception": {},
        }
    )

    cmd_contrastive(cfg, ["blocks.0.hook_resid_post"], top_k=5, output_base=None)

    scores_dir = tmp_path / "safety_scores"
    summary_path = scores_dir / "contrastive_summary.json"
    csv_path = scores_dir / "blocks_0_hook_resid_post_contrastive.csv"

    assert csv_path.exists()
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "blocks.0.hook_resid_post" in payload
