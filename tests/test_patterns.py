import pytest
import torch

from rfm.patterns import ContrastAxisSpec, SequenceRecord, aggregate_sequence_activations, validate_layer_alignment


def test_validate_layer_alignment_rejects_misaligned_sequence_order():
    axis = ContrastAxisSpec(
        axis_id="review_axis",
        endpoint_a="accept",
        endpoint_b="reject",
        display_name_a="Accept",
        display_name_b="Reject",
        pair_key_fields=("pair_id", "label", "question"),
    )
    reference = [
        SequenceRecord(label="accept", pair_id=0, question="Q1", category="c", difficulty="d", token_length=2),
        SequenceRecord(label="reject", pair_id=0, question="Q1", category="c", difficulty="d", token_length=2),
    ]
    misaligned = [
        SequenceRecord(label="accept", pair_id=0, question="Q2", category="c", difficulty="d", token_length=2),
        SequenceRecord(label="reject", pair_id=0, question="Q1", category="c", difficulty="d", token_length=2),
    ]

    with pytest.raises(ValueError, match="Layer alignment failed"):
        validate_layer_alignment(
            {
                "layer_a": reference,
                "layer_b": misaligned,
            },
            axis,
        )


def test_aggregate_sequence_activations_supports_topk_and_lastk():
    activations = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
            [10.0, 0.0],
            [6.0, 8.0],
        ],
        dtype=torch.float32,
    )
    token_lengths = [3, 2]

    topk = aggregate_sequence_activations(activations, token_lengths, method="topk_mean_2")
    lastk = aggregate_sequence_activations(activations, token_lengths, method="lastk_mean_2")

    assert torch.allclose(topk[0], torch.tensor([2.5, 3.0]))
    assert torch.allclose(topk[1], torch.tensor([8.0, 4.0]))
    assert torch.allclose(lastk[0], torch.tensor([2.5, 2.5]))
    assert torch.allclose(lastk[1], torch.tensor([8.0, 4.0]))
