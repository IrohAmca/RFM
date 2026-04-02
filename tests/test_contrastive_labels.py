import torch
from torch import nn

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
