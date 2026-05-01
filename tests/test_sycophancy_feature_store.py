import torch

from rfm.patterns import ContrastAxisSpec
from rfm.patterns.data import dense_token_features_from_sparse_payload
from rfm.patterns.discovery import PatternDiscoveryAnalyzer
from rfm.sycophancy.feature_store import dense_to_sparse_topk, write_feature_store_chunk


def _axis():
    return ContrastAxisSpec(
        axis_id="sycophancy",
        endpoint_a="truthful",
        endpoint_b="sycophantic",
        display_name_a="Truthful",
        display_name_b="Sycophantic",
    )


def test_sparse_missing_features_are_zero():
    payload = {
        "d_sae": 5,
        "feature_indices": torch.tensor([[1, -1], [3, 4]]),
        "feature_values": torch.tensor([[2.0, 7.0], [1.5, 0.5]]),
    }

    dense = dense_token_features_from_sparse_payload(payload)

    assert dense.shape == (2, 5)
    assert dense[0, 1] == 2.0
    assert dense[0, 0] == 0.0
    assert dense[1, 3] == 1.5
    assert dense[1, 4] == 0.5


def _write_synthetic_store(
    path,
    source_id,
    *,
    feature_id=5,
    d_sae=12,
    offset=0.0,
    skip_pair_ids=None,
    reverse_records=False,
):
    skip_pair_ids = set(skip_pair_ids or [])
    sparse_rows = []
    metadata = {
        "labels": [],
        "token_lengths": [],
        "pair_ids": [],
        "categories": [],
        "difficulties": [],
        "questions": [],
        "responses": [],
        "sources": [],
    }
    records = []
    for pair_id in range(12):
        if pair_id in skip_pair_ids:
            continue
        family = "false_math" if pair_id < 6 else "false_health"
        for label in ["truthful", "sycophantic"]:
            records.append((pair_id, family, label))
    if reverse_records:
        records.reverse()

    for pair_id, family, label in records:
        dense = torch.zeros((2, d_sae), dtype=torch.float32)
        if label == "sycophantic":
            dense[:, feature_id] = 3.0 + offset + 0.01 * pair_id
        else:
            dense[:, 1] = 0.2 + offset
        sparse_rows.append(dense_to_sparse_topk(dense, top_k=4, source_id=source_id, token_strings=["a", "b"]))
        metadata["labels"].append(label)
        metadata["token_lengths"].append(2)
        metadata["pair_ids"].append(pair_id)
        metadata["categories"].append(family)
        metadata["difficulties"].append("medium")
        metadata["questions"].append(f"Q{pair_id}")
        metadata["responses"].append(label)
        metadata["sources"].append("synthetic")
    write_feature_store_chunk(
        output_dir=path,
        source_id=source_id,
        chunk_index=0,
        sparse_rows=sparse_rows,
        metadata={**metadata, "target_layer": source_id, "chunk_id": 0, "contrast_axis": _axis().to_dict()},
    )


def test_preencoded_pattern_analysis_finds_planted_sycophancy_feature(tmp_path):
    source = "9-gemmascope-2-res-16k"
    store = tmp_path / source
    _write_synthetic_store(store, source)

    analyzer = PatternDiscoveryAnalyzer({}, axis_spec=_axis(), device="cpu")
    result = analyzer.analyze(
        {source: store},
        preencoded=True,
        include_controls=True,
        stable_interactions_only=True,
        cv_folds=3,
        top_endpoint_a=3,
        top_endpoint_b=3,
        top_interaction=0,
    )

    top_b = [row for row in result["layer_feature_scores"][source] if row["delta"] > 0]
    assert int(top_b[0]["feature_id"]) == 5
    assert result["controls"]["length_only"]["status"] in {"ok", "skipped"}
    assert "label_shuffle" in result["controls"]


def test_stable_interaction_pool_is_capped(tmp_path):
    source_a = "9-gemmascope-2-res-16k"
    source_b = "17-gemmascope-2-res-16k"
    _write_synthetic_store(tmp_path / source_a, source_a, feature_id=5)
    _write_synthetic_store(tmp_path / source_b, source_b, feature_id=6, offset=0.1)

    analyzer = PatternDiscoveryAnalyzer({}, axis_spec=_axis(), device="cpu")
    result = analyzer.analyze(
        {source_a: tmp_path / source_a, source_b: tmp_path / source_b},
        preencoded=True,
        include_controls=True,
        stable_interactions_only=True,
        max_interaction_features_per_layer=1,
        cv_folds=3,
        top_endpoint_a=4,
        top_endpoint_b=4,
        top_interaction=4,
    )

    assert len(result["feature_pools"][source_a]["interaction_combined"]) <= 1
    assert len(result["feature_pools"][source_b]["interaction_combined"]) <= 1


def test_preencoded_layers_filter_and_reorder_common_records(tmp_path):
    source_a = "9-gemmascope-2-res-16k"
    source_b = "17-gemmascope-2-res-16k"
    _write_synthetic_store(tmp_path / source_a, source_a, feature_id=5)
    _write_synthetic_store(
        tmp_path / source_b,
        source_b,
        feature_id=6,
        skip_pair_ids={0, 1},
        reverse_records=True,
    )

    analyzer = PatternDiscoveryAnalyzer({}, axis_spec=_axis(), device="cpu")
    result = analyzer.analyze(
        {source_a: tmp_path / source_a, source_b: tmp_path / source_b},
        preencoded=True,
        cv_folds=2,
        top_endpoint_a=2,
        top_endpoint_b=2,
        top_interaction=0,
    )

    record_filter = result["alignment_report"]["record_filter"]
    assert record_filter["common_record_count"] == 20
    assert record_filter["dropped_record_counts"][source_a] == 4
    assert record_filter["dropped_record_counts"][source_b] == 0
    assert source_b in record_filter["reordered_layers"]
