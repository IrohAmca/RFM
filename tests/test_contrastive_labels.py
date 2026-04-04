import torch
from torch import nn

from cli.safety_score import cmd_contrastive
from rfm.config import ConfigManager
from rfm.patterns import ContrastAxisSpec, load_pattern_bundle, pattern_artifact_paths
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


def test_cmd_contrastive_writes_axis_neutral_pattern_bundle(tmp_path, monkeypatch):
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    torch.save(
        {
            "activations": torch.tensor(
                [
                    [0.1, 0.2],
                    [0.0, 0.1],
                    [0.8, 0.9],
                    [1.0, 1.1],
                    [0.2, 0.0],
                    [0.1, 0.2],
                    [1.1, 0.9],
                    [0.9, 1.2],
                ],
                dtype=torch.float32,
            ),
            "tokens": torch.zeros(8, dtype=torch.long),
            "metadata": {
                "labels": ["accept", "reject", "accept", "reject"],
                "token_lengths": [2, 2, 2, 2],
                "pair_ids": [0, 0, 1, 1],
                "categories": ["eval"] * 4,
                "difficulties": ["medium"] * 4,
                "questions": ["Q1", "Q1", "Q2", "Q2"],
            },
        },
        acts_dir / "chunk.pt",
    )

    monkeypatch.setattr("cli.safety_score.resolve_best_checkpoint", lambda config, target=None: "fake.pt")
    monkeypatch.setattr("cli.safety_score.load_sae_checkpoint", lambda *args, **kwargs: (IdentitySAE(), None))

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
            "layers": {
                "blocks.0.hook_resid_post": {},
            },
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

    cmd_contrastive(cfg, ["blocks.0.hook_resid_post"], top_k=5, output_base=None)

    axis = ContrastAxisSpec.from_config(cfg)
    paths = pattern_artifact_paths(cfg, axis)
    bundle = load_pattern_bundle(cfg, axis)

    assert paths["bundle"].exists()
    assert paths["report"].exists()
    assert bundle["axis"]["endpoint_a"] == "accept"
    assert bundle["axis"]["endpoint_b"] == "reject"
    assert "blocks.0.hook_resid_post" in bundle["layers"]
    assert bundle["layers"]["blocks.0.hook_resid_post"]["feature_scores"]
