import json

from rfm.deception.deception_dataset import DeceptionDataset


def test_deception_dataset_loads_jsonl_and_supports_modes(tmp_path):
    dataset_path = tmp_path / "scenarios.jsonl"
    rows = [
        {
            "question": "Q1",
            "honest_answer": "H1",
            "deceptive_answer": "D1",
            "category": "factual_lying",
            "difficulty": "easy",
            "metadata": {"failure_mode": "falsehood"},
        },
        {
            "question": "Q2",
            "honest_answer": "H2",
            "deceptive_answer": "D2",
            "category": "sycophancy",
            "difficulty": "hard",
            "metadata": {"failure_mode": "agreement"},
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    paired = DeceptionDataset(input_path=dataset_path, mode="paired")
    paired.load()
    paired_rows = list(paired)
    assert len(paired_rows) == 2
    assert paired_rows[0]["pair_id"] == 0
    assert paired_rows[1]["deceptive_answer"] == "D2"

    replay = DeceptionDataset(input_path=dataset_path, mode="replay")
    replay.load()
    replay_rows = list(replay)
    assert [row["label"] for row in replay_rows] == [
        "honest",
        "deceptive",
        "honest",
        "deceptive",
    ]
    assert replay_rows[0]["response"] == "H1"
    assert replay_rows[1]["pair_id"] == 0
