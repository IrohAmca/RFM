from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rfm.patterns.spec import ContrastAxisSpec


@dataclass(frozen=True)
class SequenceRecord:
    label: str
    pair_id: int
    question: str
    category: str
    difficulty: str
    token_length: int
    response: str = ""
    source: str = ""
    chunk_id: int = -1
    sequence_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "pair_id": self.pair_id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "token_length": self.token_length,
            "response": self.response,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "sequence_index": self.sequence_index,
        }

    def alignment_key(self, fields: tuple[str, ...]) -> tuple[Any, ...]:
        return tuple(getattr(self, field, None) for field in fields)


@dataclass
class PairedActivationSet:
    endpoint_a: torch.Tensor
    endpoint_b: torch.Tensor
    pair_ids: list[int]
    categories: list[str]
    difficulties: list[str]
    questions: list[str]


@dataclass
class PairedSequenceIndex:
    axis: ContrastAxisSpec
    records: list[SequenceRecord]
    ordered_keys: list[tuple[Any, ...]]
    pair_keys: list[tuple[Any, ...]]

    def to_report(self, layer_name: str) -> dict[str, Any]:
        return {
            "layer": layer_name,
            "sequence_count": len(self.records),
            "pair_count": len({record.pair_id for record in self.records}),
            "pair_key_fields": list(self.axis.pair_group_fields),
            "alignment_key_fields": list(self.axis.pair_key_fields),
        }


def aggregate_sequence_activations(
    activations: torch.Tensor,
    token_lengths: list[int],
    method: str = "mean",
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(f"Expected [N_tokens, d_model] activations, got {tuple(activations.shape)}")

    rows = []
    offset = 0
    method = str(method)
    for length in token_lengths:
        segment = activations[offset: offset + int(length)]
        offset += int(length)
        if segment.numel() == 0:
            continue
        if method == "mean":
            rows.append(segment.mean(dim=0))
        elif method == "max":
            rows.append(segment.max(dim=0).values)
        elif method == "last":
            rows.append(segment[-1])
        elif method.startswith("topk_mean_"):
            k = max(int(method.rsplit("_", 1)[-1]), 1)
            topk = min(k, segment.shape[0])
            values = torch.topk(segment, k=topk, dim=0).values
            rows.append(values.mean(dim=0))
        elif method.startswith("lastk_mean_"):
            k = max(int(method.rsplit("_", 1)[-1]), 1)
            rows.append(segment[-min(k, segment.shape[0]):].mean(dim=0))
        else:
            raise ValueError(f"Unsupported aggregation method: {method!r}")

    if not rows:
        return torch.empty(0, activations.shape[-1], dtype=activations.dtype)
    return torch.stack(rows, dim=0)


def _records_from_metadata(metadata: dict[str, Any], *, chunk_id: int, sequence_offset: int = 0) -> list[SequenceRecord]:
    labels = list(metadata.get("labels", []))
    token_lengths = [int(length) for length in metadata.get("token_lengths", [])]
    pair_ids = list(metadata.get("pair_ids", []))
    categories = list(metadata.get("categories", []))
    difficulties = list(metadata.get("difficulties", []))
    questions = list(metadata.get("questions", []))
    responses = list(metadata.get("responses", []))
    sources = list(metadata.get("sources", []))

    if len(pair_ids) != len(labels):
        pair_ids = list(range(sequence_offset, sequence_offset + len(labels)))
    if len(categories) != len(labels):
        categories = ["unknown"] * len(labels)
    if len(difficulties) != len(labels):
        difficulties = ["unknown"] * len(labels)
    if len(questions) != len(labels):
        questions = [""] * len(labels)
    if len(responses) != len(labels):
        responses = [""] * len(labels)
    if len(sources) != len(labels):
        sources = [""] * len(labels)

    return [
        SequenceRecord(
            label=str(label),
            pair_id=int(pair_id),
            question=str(question),
            category=str(category),
            difficulty=str(difficulty),
            token_length=int(length),
            response=str(response),
            source=str(source),
            chunk_id=int(chunk_id),
            sequence_index=sequence_offset + index,
        )
        for index, (label, length, pair_id, category, difficulty, question, response, source) in enumerate(
            zip(labels, token_lengths, pair_ids, categories, difficulties, questions, responses, sources)
        )
    ]


def load_sequence_records(
    chunk_dir: str | Path,
    *,
    aggregation: str = "mean",
    pattern: str = "*.pt",
) -> tuple[torch.Tensor, list[SequenceRecord]]:
    chunk_dir = Path(chunk_dir)
    files = sorted(chunk_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No activation chunks found in {chunk_dir}")

    vectors = []
    records: list[SequenceRecord] = []
    offset = 0
    for chunk_id, file_path in enumerate(files):
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        chunk_records = _records_from_metadata(metadata, chunk_id=metadata.get("chunk_id", chunk_id), sequence_offset=offset)
        token_lengths = [record.token_length for record in chunk_records]
        seq_vectors = aggregate_sequence_activations(payload["activations"].float(), token_lengths, method=aggregation)
        if seq_vectors.shape[0] != len(chunk_records):
            raise ValueError(
                f"Aggregation mismatch for {file_path}: got {seq_vectors.shape[0]} vectors for {len(chunk_records)} records."
            )
        vectors.append(seq_vectors)
        records.extend(chunk_records)
        offset += len(chunk_records)

    return torch.cat(vectors, dim=0), records


def build_paired_activation_set(
    chunk_dir: str | Path,
    axis: ContrastAxisSpec,
    *,
    aggregation: str = "mean",
    pattern: str = "*.pt",
) -> PairedActivationSet:
    vectors, records = load_sequence_records(chunk_dir, aggregation=aggregation, pattern=pattern)

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for vector, record in zip(vectors, records):
        pair_key = record.alignment_key(axis.pair_group_fields)
        bucket = grouped.setdefault(
            pair_key,
            {
                "pair_id": record.pair_id,
                "category": record.category,
                "difficulty": record.difficulty,
                "question": record.question,
            },
        )
        bucket[record.label] = vector.detach().cpu()

    endpoint_a = []
    endpoint_b = []
    pair_ids = []
    categories = []
    difficulties = []
    questions = []
    for pair_key in sorted(grouped):
        bucket = grouped[pair_key]
        if axis.endpoint_a not in bucket or axis.endpoint_b not in bucket:
            continue
        endpoint_a.append(bucket[axis.endpoint_a])
        endpoint_b.append(bucket[axis.endpoint_b])
        pair_ids.append(int(bucket["pair_id"]))
        categories.append(str(bucket["category"]))
        difficulties.append(str(bucket["difficulty"]))
        questions.append(str(bucket["question"]))

    if not endpoint_a:
        raise ValueError(f"No complete paired activations found for axis {axis.axis_id} in {chunk_dir}")

    return PairedActivationSet(
        endpoint_a=torch.stack(endpoint_a, dim=0),
        endpoint_b=torch.stack(endpoint_b, dim=0),
        pair_ids=pair_ids,
        categories=categories,
        difficulties=difficulties,
        questions=questions,
    )


def _compare_record_fields(reference: SequenceRecord, current: SequenceRecord, fields: tuple[str, ...]) -> dict[str, Any] | None:
    mismatches = {}
    for field in fields:
        if getattr(reference, field, None) != getattr(current, field, None):
            mismatches[field] = {
                "reference": getattr(reference, field, None),
                "current": getattr(current, field, None),
            }
    if mismatches:
        return mismatches
    return None


def validate_layer_alignment(
    records_by_layer: dict[str, list[SequenceRecord]],
    axis: ContrastAxisSpec,
) -> dict[str, Any]:
    layers = list(records_by_layer)
    report = {
        "ok": True,
        "reference_layer": layers[0] if layers else None,
        "layers_checked": layers,
        "alignment_key_fields": list(axis.pair_key_fields),
        "issues": [],
    }
    if not layers:
        return report

    reference_layer = layers[0]
    reference_records = records_by_layer[reference_layer]
    reference_keys = [record.alignment_key(axis.pair_key_fields) for record in reference_records]
    checked_fields = ("pair_id", "label", "question", "category", "difficulty", "token_length")
    for layer_name in layers[1:]:
        current_records = records_by_layer[layer_name]
        if len(current_records) != len(reference_records):
            report["ok"] = False
            report["issues"].append(
                {
                    "layer": layer_name,
                    "type": "sequence_count_mismatch",
                    "reference_count": len(reference_records),
                    "current_count": len(current_records),
                }
            )
            continue

        for index, (reference_record, current_record) in enumerate(zip(reference_records, current_records)):
            if reference_keys[index] != current_record.alignment_key(axis.pair_key_fields):
                report["ok"] = False
                report["issues"].append(
                    {
                        "layer": layer_name,
                        "type": "alignment_key_mismatch",
                        "index": index,
                        "reference_key": list(reference_keys[index]),
                        "current_key": list(current_record.alignment_key(axis.pair_key_fields)),
                    }
                )
                break

            mismatch = _compare_record_fields(reference_record, current_record, checked_fields)
            if mismatch:
                report["ok"] = False
                report["issues"].append(
                    {
                        "layer": layer_name,
                        "type": "metadata_mismatch",
                        "index": index,
                        "mismatch": mismatch,
                    }
                )
                break

    if not report["ok"]:
        first_issue = report["issues"][0]
        raise ValueError(f"Layer alignment failed for axis {axis.axis_id}: {first_issue}")
    return report
