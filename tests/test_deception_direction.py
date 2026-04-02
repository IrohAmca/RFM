import torch

from rfm.deception.direction_finder import DeceptionDirectionFinder


def test_direction_finder_loads_pairs_and_separates_classes(tmp_path):
    chunk_dir = tmp_path / "acts"
    chunk_dir.mkdir()
    torch.save(
        {
            "activations": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.2, 0.1],
                    [1.0, 1.0],
                    [1.2, 0.9],
                    [0.1, 0.0],
                    [0.0, 0.2],
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
        chunk_dir / "chunk.pt",
    )

    finder = DeceptionDirectionFinder()
    paired = finder.load_paired_activations(chunk_dir)

    assert paired["honest"].shape == (2, 2)
    assert paired["deceptive"].shape == (2, 2)
    assert paired["pair_ids"] == [0, 1]

    result = finder.find_direction(paired["honest"], paired["deceptive"], method="mean_diff")
    assert result.validation_accuracy >= 0.99
    assert result.cluster_separation > 1.0
    assert result.direction.shape == (2,)
