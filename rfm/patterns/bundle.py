from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from rfm.patterns.paths import pattern_artifact_paths
from rfm.patterns.spec import ContrastAxisSpec


def _sanitize_json(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _sanitize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _default_bundle(axis_spec: ContrastAxisSpec) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "axis": axis_spec.to_dict(),
        "layers": {},
        "monitor": {},
        "analysis": {},
        "adversarial": {},
        "artifacts": {},
    }


def load_pattern_bundle(config, axis_spec: ContrastAxisSpec | None = None) -> dict[str, Any]:
    axis = axis_spec or ContrastAxisSpec.from_config(config)
    bundle_path = pattern_artifact_paths(config, axis)["bundle"]
    if not bundle_path.exists():
        return _default_bundle(axis)
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return _default_bundle(axis)
    bundle = copy.deepcopy(payload)
    bundle.setdefault("schema_version", 2)
    bundle.setdefault("axis", axis.to_dict())
    bundle.setdefault("layers", {})
    bundle.setdefault("monitor", {})
    bundle.setdefault("analysis", {})
    bundle.setdefault("adversarial", {})
    bundle.setdefault("artifacts", {})
    return bundle


def build_pattern_report(bundle: dict[str, Any]) -> dict[str, Any]:
    axis = dict(bundle.get("axis", {}))
    layers = {}
    for layer_name, payload in bundle.get("layers", {}).items():
        scores = list(payload.get("feature_scores", []) or [])
        top_positive = [row for row in scores if float(row.get("delta", 0.0)) > 0][:10]
        top_negative = [row for row in scores if float(row.get("delta", 0.0)) < 0][:10]
        direction_payload = dict(payload.get("direction", {}) or {})
        direction_payload.pop("direction", None)
        layers[layer_name] = {
            "direction": _sanitize_json(direction_payload),
            "probe": _sanitize_json(payload.get("probe", {})),
            "top_endpoint_b_features": _sanitize_json(top_positive),
            "top_endpoint_a_features": _sanitize_json(top_negative),
            "aggregation": _sanitize_json(payload.get("aggregation", {})),
        }

    analysis = dict(bundle.get("analysis", {}))
    report = {
        "schema_version": int(bundle.get("schema_version", 1)),
        "axis": axis,
        "layers": layers,
        "monitor": _sanitize_json(bundle.get("monitor", {})),
        "analysis": {
            "alignment_report": _sanitize_json(analysis.get("alignment_report", {})),
            "aggregation_benchmark": _sanitize_json(analysis.get("aggregation_benchmark", [])),
            "selected_aggregation": analysis.get("selected_aggregation"),
            "model_metrics": _sanitize_json(analysis.get("model_metrics", {})),
            "motif_candidates": _sanitize_json(list(analysis.get("motif_candidates", []) or [])[:200]),
            "stable_motifs": _sanitize_json(list(analysis.get("stable_motifs", []) or [])[:100]),
            "stable_interactions": _sanitize_json(list(analysis.get("stable_interactions", []) or [])[:100]),
            "coactivation_stats": _sanitize_json(list(analysis.get("coactivation_stats", []) or [])[:100]),
            "feature_importance": _sanitize_json(list(analysis.get("feature_importance", []) or [])[:50]),
            "decision_tree": _sanitize_json(analysis.get("decision_tree", {})),
            "intervention_effects": _sanitize_json(list(analysis.get("intervention_effects", []) or [])[:100]),
            "manual_interventions": _sanitize_json(list(analysis.get("manual_interventions", []) or [])[:50]),
            "causal_validation": _sanitize_json(analysis.get("causal_validation", {})),
        },
        "adversarial": _sanitize_json(bundle.get("adversarial", {})),
        "artifacts": _sanitize_json(bundle.get("artifacts", {})),
    }
    return report


def save_pattern_bundle(config, bundle: dict[str, Any], axis_spec: ContrastAxisSpec | None = None) -> dict[str, Path]:
    axis = axis_spec or ContrastAxisSpec.from_config(config)
    paths = pattern_artifact_paths(config, axis)
    paths["pattern_dir"].mkdir(parents=True, exist_ok=True)
    bundle["axis"] = axis.to_dict()
    torch.save(bundle, paths["bundle"])
    report = build_pattern_report(bundle)
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def update_pattern_bundle(
    config,
    *,
    axis_spec: ContrastAxisSpec | None = None,
    layer_updates: dict[str, dict[str, Any]] | None = None,
    monitor: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    adversarial: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = load_pattern_bundle(config, axis_spec)

    if layer_updates:
        for layer_name, payload in layer_updates.items():
            bucket = bundle["layers"].setdefault(layer_name, {})
            bucket.update(payload)
    if monitor:
        bundle["monitor"].update(monitor)
    if analysis:
        bundle["analysis"].update(analysis)
    if adversarial:
        bundle["adversarial"].update(adversarial)
    if artifacts:
        bundle["artifacts"].update(artifacts)

    save_pattern_bundle(config, bundle, axis_spec)
    return bundle


def write_pattern_report(config, axis_spec: ContrastAxisSpec | None = None) -> Path:
    bundle = load_pattern_bundle(config, axis_spec)
    paths = save_pattern_bundle(config, bundle, axis_spec)
    return paths["report"]


def append_pattern_analysis_rows(
    config,
    *,
    key: str,
    rows: list[dict[str, Any]],
    axis_spec: ContrastAxisSpec | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    bundle = load_pattern_bundle(config, axis_spec)
    analysis = bundle.setdefault("analysis", {})
    existing = list(analysis.get(key, []) or [])
    existing.extend(rows)
    if max_rows is not None:
        existing = existing[-max(int(max_rows), 1):]
    analysis[key] = existing
    save_pattern_bundle(config, bundle, axis_spec)
    return bundle
