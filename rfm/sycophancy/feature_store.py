from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rfm.layout import model_slug
from rfm.patterns import ContrastAxisSpec
from rfm.patterns.paths import axis_run_dir


@dataclass(frozen=True)
class SparseTokenFeatures:
    source_id: str
    d_sae: int
    feature_indices: torch.Tensor
    feature_values: torch.Tensor
    token_strings: list[str]
    backend: str

    def __post_init__(self):
        if self.feature_indices.shape != self.feature_values.shape:
            raise ValueError(
                "feature_indices and feature_values must have matching shapes. "
                f"Got {tuple(self.feature_indices.shape)} vs {tuple(self.feature_values.shape)}."
            )
        if self.feature_indices.ndim != 2:
            raise ValueError(f"Sparse token features must be [n_tokens, top_k], got {tuple(self.feature_indices.shape)}")
        if int(self.d_sae) <= 0:
            raise ValueError("d_sae must be positive.")
        if len(self.token_strings) != int(self.feature_indices.shape[0]):
            raise ValueError(
                "token_strings must align to feature rows. "
                f"Got {len(self.token_strings)} strings for {self.feature_indices.shape[0]} rows."
            )


def sycophancy_run_dir(config, *parts: str) -> Path:
    axis = ContrastAxisSpec.from_config(config)
    if axis.axis_id != "sycophancy":
        axis = ContrastAxisSpec(
            axis_id="sycophancy",
            endpoint_a="truthful",
            endpoint_b="sycophantic",
            display_name_a="Truthful",
            display_name_b="Sycophantic",
        )
    return axis_run_dir(config, axis, *parts)


def source_slug(source_id: str) -> str:
    return str(source_id).replace("/", "_").replace("\\", "_").replace(".", "_")


def feature_store_dir(config, source_id: str | None = None) -> Path:
    base = sycophancy_run_dir(config, "feature_store")
    if source_id:
        return base / source_slug(source_id)
    return base


def neuronpedia_cache_dir(config) -> Path:
    return sycophancy_run_dir(config, "cache", "neuronpedia")


def default_scenario_path(config) -> Path:
    return sycophancy_run_dir(config, "scenarios.jsonl")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    rows = []
    if not input_path.exists():
        return rows
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def response_topk_to_sparse(
    *,
    topk_by_token: list[dict[str, Any]],
    token_offset: int,
    d_sae: int,
    top_k: int,
    source_id: str,
    backend: str,
) -> SparseTokenFeatures:
    token_offset = max(int(token_offset), 0)
    top_k = int(top_k)
    d_sae = int(d_sae)
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if d_sae <= 0:
        raise ValueError("d_sae must be positive.")
    rows = list(topk_by_token)[token_offset:]
    if not rows:
        raise ValueError("No response-token feature rows after applying token_offset.")

    indices = torch.full((len(rows), top_k), -1, dtype=torch.long)
    values = torch.zeros((len(rows), top_k), dtype=torch.float32)
    token_strings: list[str] = []
    for row_index, row in enumerate(rows):
        token_strings.append(str(row.get("token", "")))
        features = list(row.get("top_features", []) or [])[: int(top_k)]
        for feature_slot, item in enumerate(features):
            feature_index = int(item["feature_index"])
            if feature_index < 0 or feature_index >= d_sae:
                continue
            indices[row_index, feature_slot] = feature_index
            values[row_index, feature_slot] = float(item["activation_value"])

    return SparseTokenFeatures(
        source_id=source_id,
        d_sae=int(d_sae),
        feature_indices=indices,
        feature_values=values,
        token_strings=token_strings,
        backend=backend,
    )


def dense_to_sparse_topk(
    dense: torch.Tensor,
    *,
    top_k: int,
    source_id: str,
    token_strings: list[str] | None = None,
    backend: str = "local_saelens",
) -> SparseTokenFeatures:
    dense = dense.detach().cpu().float()
    if dense.ndim != 2:
        raise ValueError(f"Dense feature tensor must be [n_tokens, d_sae], got {tuple(dense.shape)}")
    if dense.shape[1] <= 0:
        raise ValueError("Dense feature tensor must have at least one feature column.")
    k = min(max(int(top_k), 1), dense.shape[1])
    values, indices = torch.topk(dense, k=k, dim=1)
    if k < int(top_k):
        pad = int(top_k) - k
        indices = torch.cat([indices, torch.full((dense.shape[0], pad), -1, dtype=torch.long)], dim=1)
        values = torch.cat([values, torch.zeros((dense.shape[0], pad), dtype=torch.float32)], dim=1)
    return SparseTokenFeatures(
        source_id=source_id,
        d_sae=int(dense.shape[1]),
        feature_indices=indices,
        feature_values=values,
        token_strings=list(token_strings or [""] * dense.shape[0]),
        backend=backend,
    )


def write_feature_store_chunk(
    *,
    output_dir: str | Path,
    source_id: str,
    chunk_index: int,
    sparse_rows: list[SparseTokenFeatures],
    metadata: dict[str, Any],
) -> Path:
    if not sparse_rows:
        raise ValueError("Cannot write an empty feature-store chunk.")

    d_sae = int(sparse_rows[0].d_sae)
    top_k = int(sparse_rows[0].feature_indices.shape[1])
    if any(int(row.d_sae) != d_sae for row in sparse_rows):
        raise ValueError("All sparse rows in a chunk must share d_sae.")
    if any(int(row.feature_indices.shape[1]) != top_k for row in sparse_rows):
        raise ValueError("All sparse rows in a chunk must share top_k.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_path = output / f"{source_slug(source_id)}_{int(chunk_index)}.pt"
    meta_path = save_path.with_suffix(".meta.json")
    payload_metadata = dict(metadata)
    payload_metadata.update(
        {
            "source_id": source_id,
            "d_sae": d_sae,
            "top_k": top_k,
            "schema": "preencoded_sparse_feature_store_v1",
            "written_at": time.time(),
        }
    )
    payload = {
        "schema_version": 1,
        "source_id": source_id,
        "d_sae": d_sae,
        "feature_indices": torch.cat([row.feature_indices for row in sparse_rows], dim=0),
        "feature_values": torch.cat([row.feature_values for row in sparse_rows], dim=0),
        "token_strings": [token for row in sparse_rows for token in row.token_strings],
        "metadata": payload_metadata,
    }
    torch.save(payload, save_path)
    meta_path.write_text(json.dumps(payload_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return save_path


def model_cache_key(config) -> str:
    return model_slug(config).lower()
