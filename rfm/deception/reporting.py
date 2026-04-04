from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rfm.deception.deception_monitor import DeceptionMonitor
from rfm.deception.deception_probe import DeceptionProbe
from rfm.deception.direction_finder import DeceptionDirectionFinder, DirectionResult
from rfm.deception.utils import deception_run_dir
from rfm.layout import resolve_activations_dir, resolve_requested_targets, sanitize_layer_name
from rfm.patterns import AxisProbeState, ContrastAxisSpec, load_pattern_bundle, pattern_artifact_paths
from rfm.viz.plots import (
    build_adversarial_analysis_chart,
    build_category_breakdown_heatmap,
    build_deception_layer_comparison,
    build_deception_tsne_scatter,
    build_probe_roc_curve,
)


def deception_report_dir(config) -> Path:
    return deception_run_dir(config, "reports")


def deception_report_paths(report_dir: str | Path) -> dict[str, Path]:
    base = Path(report_dir)
    return {
        "report_dir": base,
        "html": base / "deception_report.html",
        "layer_comparison": base / "layer_comparison.png",
        "tsne": base / "tsne_honest_vs_deceptive.png",
        "roc": base / "probe_roc_curve.png",
        "category": base / "category_breakdown.png",
        "adversarial": base / "adversarial_analysis.png",
    }


def _pretty_layer_name(layer_name: str) -> str:
    match = re.search(r"(?:blocks|layer)\.(\d+)", str(layer_name))
    if match:
        return f"L{match.group(1)}"
    return str(layer_name)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _split_pairs(
    honest: torch.Tensor,
    deceptive: torch.Tensor,
    validation_split: float,
    seed: int | None = None,
):
    n = honest.shape[0]
    if n != deceptive.shape[0]:
        raise ValueError("Honest and deceptive activations must contain the same number of samples.")
    if n < 2 or validation_split <= 0:
        return honest, deceptive, honest, deceptive

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    order = torch.randperm(n, generator=generator)
    honest = honest[order]
    deceptive = deceptive[order]

    val_count = min(max(int(round(n * validation_split)), 1), n - 1)
    train_count = n - val_count
    return honest[:train_count], deceptive[:train_count], honest[train_count:], deceptive[train_count:]


def _load_directions(path: Path) -> dict[str, DirectionResult]:
    if not path.exists():
        return {}
    finder = DeceptionDirectionFinder()
    return finder.load(path)


def _load_probe(path: Path) -> DeceptionProbe | None:
    if not path.exists():
        return None
    probe = DeceptionProbe()
    probe.load(path)
    return probe


def _directions_from_bundle(bundle_layers: dict[str, dict]) -> dict[str, DirectionResult]:
    directions = {}
    for layer_name, payload in bundle_layers.items():
        direction_payload = payload.get("direction")
        if isinstance(direction_payload, dict) and "direction" in direction_payload:
            directions[layer_name] = DirectionResult.from_dict(direction_payload)
    return directions


def _probe_from_bundle(payload: dict) -> DeceptionProbe | None:
    probe_payload = payload.get("probe_state")
    if not isinstance(probe_payload, dict) or "weight" not in probe_payload:
        return None
    probe = DeceptionProbe()
    probe.state = AxisProbeState.from_dict(probe_payload)
    return probe

def _sample_projection_inputs(
    honest: torch.Tensor,
    deceptive: torch.Tensor,
    max_points_per_label: int,
    seed: int = 42,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def _take_rows(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] <= max_points_per_label:
            return values
        order = torch.randperm(values.shape[0], generator=generator)[:max_points_per_label]
        return values[order]

    honest_sel = _take_rows(honest.detach().cpu().float())
    deceptive_sel = _take_rows(deceptive.detach().cpu().float())
    return honest_sel, deceptive_sel


def _project_2d(
    honest: torch.Tensor,
    deceptive: torch.Tensor,
    max_points_per_label: int,
    *,
    label_a: str = "honest",
    label_b: str = "deceptive",
) -> dict:
    honest_sel, deceptive_sel = _sample_projection_inputs(honest, deceptive, max_points_per_label=max_points_per_label)
    sample = torch.cat([honest_sel, deceptive_sel], dim=0)
    labels = [label_a] * honest_sel.shape[0] + [label_b] * deceptive_sel.shape[0]

    if sample.shape[0] < 2:
        points = [{"x": float(index), "y": 0.0, "label": label} for index, label in enumerate(labels)]
        return {"method": "degenerate", "points": points}

    try:
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, sample.shape[0] - 1))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="random",
        ).fit_transform(sample.numpy())
        method = "tsne"
    except Exception:
        centered = sample - sample.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[: min(2, vh.shape[0])].T
        coords = (centered @ basis).numpy()
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
        method = "pca"

    points = [{"x": float(x), "y": float(y), "label": label} for (x, y), label in zip(coords, labels)]
    return {"method": method, "points": points}


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[list[float], list[float], float]:
    if y_true.size == 0 or y_score.size == 0:
        return [], [], 0.0

    positives = max(int((y_true == 1).sum()), 0)
    negatives = max(int((y_true == 0).sum()), 0)
    if positives == 0 or negatives == 0:
        return [], [], 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    change_idx = np.where(np.diff(y_score))[0]
    idx = np.r_[change_idx, y_true.size - 1]

    tpr = np.r_[0.0, tps[idx] / positives, 1.0]
    fpr = np.r_[0.0, fps[idx] / negatives, 1.0]
    integrate = getattr(np, "trapezoid", np.trapz)
    auc = float(integrate(tpr, fpr))
    return fpr.tolist(), tpr.tolist(), auc


def _validation_scores_for_probe(
    probe: DeceptionProbe | None,
    paired: dict,
    validation_split: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if probe is None:
        return np.array([]), np.array([])

    _, _, val_h, val_d = _split_pairs(
        paired["honest"],
        paired["deceptive"],
        validation_split=validation_split,
        seed=split_seed,
    )
    if val_h.shape[0] == 0:
        val_h = paired["honest"]
        val_d = paired["deceptive"]

    scores = []
    labels = []
    for vector in val_h:
        _, probability = probe.predict(vector)
        scores.append(probability)
        labels.append(0)
    for vector in val_d:
        _, probability = probe.predict(vector)
        scores.append(probability)
        labels.append(1)
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def _pair_records(paired: dict) -> dict[int, dict]:
    records = {}
    pair_ids = paired.get("pair_ids", [])
    categories = paired.get("categories", [])
    difficulties = paired.get("difficulties", [])
    questions = paired.get("questions", [])

    for index, pair_id in enumerate(pair_ids):
        records[int(pair_id)] = {
            "honest": paired["honest"][index],
            "deceptive": paired["deceptive"][index],
            "category": categories[index] if index < len(categories) else "unknown",
            "difficulty": difficulties[index] if index < len(difficulties) else "unknown",
            "question": questions[index] if index < len(questions) else "",
        }
    return records


def _build_category_heatmap(
    targets: list[str],
    paired_by_layer: dict[str, dict],
    monitor: DeceptionMonitor,
) -> tuple[list[str], list[str], np.ndarray]:
    if not targets or not paired_by_layer:
        return [], [], np.zeros((0, 0), dtype=np.float32)

    record_maps = {layer: _pair_records(paired) for layer, paired in paired_by_layer.items()}
    common_pair_ids = None
    for layer in targets:
        layer_ids = set(record_maps.get(layer, {}).keys())
        common_pair_ids = layer_ids if common_pair_ids is None else common_pair_ids & layer_ids
    if not common_pair_ids:
        return [], [], np.zeros((0, 0), dtype=np.float32)

    first_layer = targets[0]
    categories = sorted({record_maps[first_layer][pair_id]["category"] for pair_id in common_pair_ids})
    category_hits = {
        category: {layer: 0 for layer in targets}
        for category in categories
    }
    category_hits.update({category: {**category_hits[category], "ensemble": 0, "_count": 0} for category in categories})

    for pair_id in sorted(common_pair_ids):
        category = record_maps[first_layer][pair_id]["category"]
        category_hits[category]["_count"] += 1

        layer_inputs = {}
        for layer in targets:
            deceptive = record_maps[layer][pair_id]["deceptive"].unsqueeze(0)
            layer_inputs[layer] = deceptive
            score = monitor._score_layer(layer, deceptive)
            if score >= float(monitor.thresholds.get(layer, 0.5)):
                category_hits[category][layer] += 1

        ensemble = monitor.score_generation(layer_inputs)
        if ensemble.alert:
            category_hits[category]["ensemble"] += 1

    column_labels = [_pretty_layer_name(layer) for layer in targets] + ["ensemble"]
    matrix = np.zeros((len(categories), len(column_labels)), dtype=np.float32)
    for row_index, category in enumerate(categories):
        count = max(int(category_hits[category]["_count"]), 1)
        for col_index, layer in enumerate(targets):
            matrix[row_index, col_index] = category_hits[category][layer] / count
        matrix[row_index, -1] = category_hits[category]["ensemble"] / count
    return categories, column_labels, matrix


def _top_risk_rows(rows: list[dict], top_k: int) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: abs(_to_float(row.get("effect_size"), _to_float(row.get("delta"), 0.0))),
        reverse=True,
    )
    top_rows = []
    for row in ordered[: max(int(top_k), 1)]:
        top_rows.append(
            {
                "feature_id": _to_int(row.get("feature_id")),
                "delta": _to_float(row.get("delta")),
                "effect_size": _to_float(row.get("effect_size")),
                "activation_rate_a": _to_float(row.get("activation_rate_a")),
                "activation_rate_b": _to_float(row.get("activation_rate_b")),
            }
        )
    return top_rows


def _alignment_rows(alignment_path: Path, feature_rows: list[dict], top_k: int) -> list[dict]:
    if not alignment_path.exists():
        return []

    feature_rows_by_feature = {_to_int(row.get("feature_id")): row for row in feature_rows}
    raw = _read_json(alignment_path, [])
    rows = []
    for item in raw[: max(int(top_k), 1)]:
        feature_id = _to_int(item.get("feature_id"))
        feature_row = feature_rows_by_feature.get(feature_id, {})
        rows.append(
            {
                "feature_id": feature_id,
                "cosine_similarity": _to_float(item.get("cosine_similarity")),
                "alignment": item.get("alignment", ""),
                "delta": _to_float(feature_row.get("delta")),
                "effect_size": _to_float(feature_row.get("effect_size")),
            }
        )
    return rows


def _format_scalar(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        if abs(value) >= 10000:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return html.escape(str(value))


def _table_html(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "<p class='muted'>No data available.</p>"

    columns = columns or list(rows[0].keys())
    header_html = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_format_scalar(row.get(column))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _image_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _build_html_report(
    *,
    config_path: str,
    config,
    report_dir: Path,
    layer_rows: list[dict],
    monitor_report: dict,
    chart_paths: dict[str, Path],
    risky_features: dict[str, list[dict]],
    alignment_tables: dict[str, list[dict]],
    pattern_report_path: Path | None,
):
    axis = ContrastAxisSpec.from_config(config)
    ddr = _to_float(monitor_report.get("detection_rate"))
    fpr = _to_float(monitor_report.get("false_positive_rate"))
    precision = _to_float(monitor_report.get("precision"))
    pairs_evaluated = _to_int(monitor_report.get("pairs_evaluated"))

    best_direction = max(layer_rows, key=lambda row: row.get("direction_accuracy", 0.0), default=None)
    best_probe = max(layer_rows, key=lambda row: row.get("probe_validation_accuracy", 0.0), default=None)

    gallery = []
    for title, key in [
        ("Layer Comparison", "layer_comparison"),
        (f"{axis.display_name_a} vs {axis.display_name_b} Projection", "tsne"),
        ("Probe ROC", "roc"),
        ("Category Breakdown", "category"),
        ("Adversarial Analysis", "adversarial"),
    ]:
        path = chart_paths.get(key)
        if path and path.exists():
            gallery.append(
                "<section class='card image-card'>"
                f"<h3>{html.escape(title)}</h3>"
                f"<img src='{_image_data_uri(path)}' alt='{html.escape(title)}' />"
                "</section>"
            )

    risky_sections = []
    for layer_name, rows in risky_features.items():
        risky_sections.append(
            f"<section class='card'><h3>{html.escape(_pretty_layer_name(layer_name))}: Top Signed Features</h3>"
            + _table_html(rows, ["feature_id", "delta", "effect_size", "activation_rate_a", "activation_rate_b"])
            + "</section>"
        )

    alignment_sections = []
    for layer_name, rows in alignment_tables.items():
        alignment_sections.append(
            f"<section class='card'><h3>{html.escape(_pretty_layer_name(layer_name))}: SAE Direction Alignment</h3>"
            + _table_html(rows, ["feature_id", "cosine_similarity", "alignment", "delta", "effect_size"])
            + "</section>"
        )

    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    pattern_note = (
        f"<p class='muted'>Pattern report: {html.escape(str(pattern_report_path))}</p>"
        if pattern_report_path
        else "<p class='muted'>No canonical pattern report was found.</p>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Deception Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f7fb;
      color: #1f2937;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
    }}
    p {{
      line-height: 1.55;
    }}
    .muted {{
      color: #6b7280;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 20px 0 28px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      margin: 20px 0 28px;
    }}
    .card {{
      background: white;
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    .metric-value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .metric-label {{
      font-size: 13px;
      color: #6b7280;
    }}
    .image-card img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }}
    th {{
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="card">
      <h1>Deception Monitor Report</h1>
      <p class="muted">Generated: {html.escape(generated_at)}</p>
      <p>Config: <code>{html.escape(config_path)}</code></p>
      <p>Model: <code>{html.escape(str(config.get('model_name', 'unknown')))}</code></p>
      <p>Report directory: <code>{html.escape(str(report_dir))}</code></p>
      {pattern_note}
    </section>

    <div class="grid">
      <section class="card">
        <div class="metric-value">{ddr:.1%}</div>
        <div class="metric-label">Detection Rate</div>
      </section>
      <section class="card">
        <div class="metric-value">{fpr:.1%}</div>
        <div class="metric-label">False Positive Rate</div>
      </section>
      <section class="card">
        <div class="metric-value">{precision:.1%}</div>
        <div class="metric-label">Precision</div>
      </section>
      <section class="card">
        <div class="metric-value">{pairs_evaluated}</div>
        <div class="metric-label">Pairs Evaluated</div>
      </section>
      <section class="card">
        <div class="metric-value">{html.escape(best_direction['layer_label']) if best_direction else '-'}</div>
        <div class="metric-label">Best Direction Layer</div>
      </section>
      <section class="card">
        <div class="metric-value">{html.escape(best_probe['layer_label']) if best_probe else '-'}</div>
        <div class="metric-label">Best Probe Layer</div>
      </section>
    </div>

    <section class="card">
      <h2>Per-Layer Metrics</h2>
      {_table_html(layer_rows, [
          "layer_label",
          "direction_accuracy",
          "probe_validation_accuracy",
          "cluster_separation",
          "explained_variance",
          "threshold",
          "top_risk_feature",
          "top_risk_score",
      ])}
    </section>

    <div class="gallery">
      {''.join(gallery)}
    </div>

    <section class="card">
      <h2>Signed Feature Tables</h2>
      {''.join(risky_sections) if risky_sections else "<p class='muted'>No feature score tables were found.</p>"}
    </section>

    <section class="card">
      <h2>SAE Feature to Direction Alignment</h2>
      {''.join(alignment_sections) if alignment_sections else "<p class='muted'>No SAE alignment reports were found.</p>"}
    </section>
  </div>
</body>
</html>"""


def generate_deception_report(
    config,
    config_path: str,
    output_dir: str | None = None,
    top_features: int = 10,
    max_projection_points: int = 250,
):
    axis = ContrastAxisSpec.from_config(config)
    targets = resolve_requested_targets(config)
    report_dir = Path(output_dir) if output_dir else deception_report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)

    dec_dir = deception_run_dir(config)
    pattern_paths = pattern_artifact_paths(config, axis)
    pattern_bundle = load_pattern_bundle(config, axis)
    bundle_layers = pattern_bundle.get("layers", {})
    directions = _load_directions(dec_dir / "directions" / "directions.pt") or _directions_from_bundle(bundle_layers)
    probe_summary = _read_json(dec_dir / "probes" / "probe_summary.json", {})
    if not probe_summary:
        probe_summary = {
            layer_name: dict(payload.get("probe", {}) or {})
            for layer_name, payload in bundle_layers.items()
            if payload.get("probe")
        }
    monitor_report = dict(pattern_bundle.get("monitor", {}) or {}) or _read_json(dec_dir / "monitor" / "monitor_report.json", {})
    adversarial_summary = dict(pattern_bundle.get("adversarial", {}) or {}) or _read_json(dec_dir / "adversarial" / "summary.json", {})
    thresholds = monitor_report.get("thresholds", {})

    validation_split = float(config.get("deception.direction.validation_split", 0.2))
    split_seed = int(config.get("deception.direction.split_seed", config.get("train.split_seed", 42)))
    finder = DeceptionDirectionFinder(
        aggregation=config.get("deception.direction.aggregation", "mean"),
    )

    probes: dict[str, DeceptionProbe] = {}
    paired_by_layer: dict[str, dict] = {}
    projection_payloads: dict[str, dict] = {}
    roc_payloads: dict[str, dict] = {}
    risky_features: dict[str, list[dict]] = {}
    alignment_tables: dict[str, list[dict]] = {}
    layer_rows: list[dict] = []

    for target in targets:
        safe = sanitize_layer_name(target)
        target_config = config.for_target(target) if hasattr(config, "for_target") else config

        feature_rows = list(bundle_layers.get(target, {}).get("feature_scores", []) or [])
        risky_features[target] = _top_risk_rows(feature_rows, top_k=top_features)

        alignment_path = dec_dir / "probes" / f"{safe}_sae_features.json"
        alignment_tables[target] = _alignment_rows(alignment_path, feature_rows, top_k=top_features)

        direction = directions.get(target)
        probe_info = probe_summary.get(target, {})
        threshold = _to_float(thresholds.get(target, direction.threshold if direction else 0.0))
        top_risk = risky_features[target][0] if risky_features[target] else {}
        layer_rows.append(
            {
                "layer": target,
                "layer_label": _pretty_layer_name(target),
                "direction_accuracy": float(direction.validation_accuracy) if direction else 0.0,
                "probe_validation_accuracy": _to_float(probe_info.get("validation_accuracy")),
                "cluster_separation": float(direction.cluster_separation) if direction else 0.0,
                "explained_variance": float(direction.explained_variance) if direction else 0.0,
                "threshold": threshold,
                "top_risk_feature": top_risk.get("feature_id"),
                "top_risk_score": top_risk.get("effect_size"),
            }
        )

        probe = _load_probe(dec_dir / "probes" / f"{safe}.pt") or _probe_from_bundle(bundle_layers.get(target, {}))
        if probe is not None:
            probes[target] = probe

        chunk_dir = Path(resolve_activations_dir(target_config, target=target))
        if chunk_dir.exists():
            try:
                paired = finder.load_paired_activations(chunk_dir)
            except (FileNotFoundError, ValueError):
                paired = None
            if paired is not None:
                paired_by_layer[target] = paired
                projection_payloads[target] = {
                    "layer_label": _pretty_layer_name(target),
                    **_project_2d(
                        paired["honest"],
                        paired["deceptive"],
                        max_points_per_label=max_projection_points,
                        label_a=axis.display_name_a,
                        label_b=axis.display_name_b,
                    ),
                }
                y_true, y_score = _validation_scores_for_probe(
                    probe=probe,
                    paired=paired,
                    validation_split=validation_split,
                    split_seed=split_seed,
                )
                fpr, tpr, auc = _roc_curve(y_true, y_score)
                roc_payloads[target] = {
                    "layer_label": _pretty_layer_name(target),
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": auc,
                }

    layer_rows.sort(key=lambda row: row["layer_label"])

    available_targets = [target for target in targets if target in paired_by_layer]
    monitor = DeceptionMonitor(
        directions=directions,
        probes=probes,
        thresholds={layer: _to_float(value, 0.5) for layer, value in thresholds.items()},
        ensemble_method=config.get("deception.monitor.ensemble_method", "weighted_average"),
    )
    categories, heatmap_columns, heatmap_matrix = _build_category_heatmap(available_targets, paired_by_layer, monitor)

    chart_paths = deception_report_paths(report_dir)
    build_deception_layer_comparison(layer_rows, chart_paths["layer_comparison"])
    build_deception_tsne_scatter(projection_payloads, chart_paths["tsne"])
    build_probe_roc_curve(roc_payloads, chart_paths["roc"])
    build_category_breakdown_heatmap(categories, heatmap_columns, heatmap_matrix, chart_paths["category"])
    build_adversarial_analysis_chart(adversarial_summary, chart_paths["adversarial"])

    pattern_report_path = pattern_paths["report"] if pattern_paths["report"].exists() else None
    html_text = _build_html_report(
        config_path=config_path,
        config=config,
        report_dir=report_dir,
        layer_rows=layer_rows,
        monitor_report=monitor_report,
        chart_paths=chart_paths,
        risky_features=risky_features,
        alignment_tables=alignment_tables,
        pattern_report_path=pattern_report_path,
    )
    chart_paths["html"].write_text(html_text, encoding="utf-8")

    return {
        "report_dir": report_dir,
        "html_path": chart_paths["html"],
        "chart_paths": chart_paths,
    }
