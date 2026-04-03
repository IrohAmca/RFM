import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rfm.layout import resolve_requested_targets

DEFAULT_METRICS = [
    "train_loss",
    "train_recon",
    "train_sparse",
    "train_active_rate",
    "val_loss",
    "val_recon",
    "val_sparse",
    "val_active_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAE training or feature-mapping charts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file. When set, report paths are derived automatically.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["training", "mapping", "all"],
        help="Which report family to generate.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing .pt checkpoint files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="Checkpoint file glob pattern.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for generated png/csv files.",
    )
    parser.add_argument(
        "--mapping-events-csv",
        type=str,
        default="reports/feature_mapping/feature_mapping_events.csv",
        help="Path to feature mapping events CSV.",
    )
    parser.add_argument(
        "--mapping-summary-csv",
        type=str,
        default="reports/feature_mapping/feature_mapping_feature_summary.csv",
        help="Path to feature mapping summary CSV.",
    )
    parser.add_argument(
        "--mapping-token-pairs-csv",
        type=str,
        default="reports/feature_mapping/feature_mapping_feature_summary_token_pairs.csv",
        help="Path to feature mapping token-pairs CSV.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=30,
        help="Top-N features to include in mapping charts.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", [])
    cfg = payload.get("config", {})
    sae_cfg = cfg.get("sae", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}

    return {
        "path": str(path),
        "name": path.stem,
        "history": history,
        "sparsity_weight": sae_cfg.get("sparsity_weight"),
        "hidden_dim": sae_cfg.get("hidden_dim"),
        "epochs": train_cfg.get("epochs"),
    }


def _history_epoch_rows(history):
    return [
        step for step in history
        if isinstance(step, dict) and _is_number(step.get("epoch"))
    ]


def final_metrics(run):
    if not run["history"]:
        return None

    epoch_rows = _history_epoch_rows(run["history"])
    if not epoch_rows:
        return None

    last = epoch_rows[-1]
    return {
        "name": run["name"],
        "checkpoint": run["path"],
        "lambda": run["sparsity_weight"],
        "hidden_dim": run["hidden_dim"],
        "epochs": run["epochs"],
        "epoch": last.get("epoch"),
        "train_loss": last.get("train_loss"),
        "train_recon": last.get("train_recon"),
        "train_sparse": last.get("train_sparse"),
        "train_active_rate": last.get("train_active_rate"),
        "val_loss": last.get("val_loss"),
        "val_recon": last.get("val_recon"),
        "val_sparse": last.get("val_sparse"),
        "val_active_rate": last.get("val_active_rate"),
    }


def write_summary_csv(rows, output_path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_number(value):
    return isinstance(value, (int, float))


def build_pareto_chart(rows, output_path: Path):
    valid = [
        row
        for row in rows
        if _is_number(row.get("val_active_rate")) and _is_number(row.get("val_recon"))
    ]
    if not valid:
        return

    x_vals = [r["val_active_rate"] for r in valid]
    y_vals = [r["val_recon"] for r in valid]
    labels = [r["name"] for r in valid]
    colors = [r.get("lambda") if _is_number(r.get("lambda")) else 0.0 for r in valid]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    scatter = ax.scatter(x_vals, y_vals, c=colors, cmap="viridis", s=70, alpha=0.9)

    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x_vals[i], y_vals[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("lambda")
    ax.set_title("SAE Pareto: Reconstruction vs Sparsity")
    ax.set_xlabel("val_active_rate (lower = sparser)")
    ax.set_ylabel("val_recon (lower = better reconstruction)")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_lambda_tradeoff_chart(rows, output_path: Path):
    valid = [row for row in rows if _is_number(row.get("lambda"))]
    valid.sort(key=lambda r: float(r["lambda"]))
    if not valid:
        return

    x_values = [r["lambda"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax2 = ax1.twinx()

    if any(_is_number(r.get("val_recon")) for r in valid):
        ax1.plot(
            x_values,
            [r.get("val_recon") for r in valid],
            marker="o",
            linewidth=1.8,
            label="val_recon",
            color="#1f77b4",
        )
    if any(_is_number(r.get("val_active_rate")) for r in valid):
        ax2.plot(
            x_values,
            [r.get("val_active_rate") for r in valid],
            marker="s",
            linewidth=1.8,
            label="val_active_rate",
            color="#d62728",
        )

    ax1.set_xscale("log")
    ax1.set_title("Lambda Tradeoff Curve")
    ax1.set_xlabel("sparsity_weight (lambda)")
    ax1.set_ylabel("val_recon", color="#1f77b4")
    ax2.set_ylabel("val_active_rate", color="#d62728")
    ax1.grid(True, alpha=0.25)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    if lines:
        ax1.legend(lines, labels, loc="best")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_epoch_metric_charts(runs, output_dir: Path):
    for metric in DEFAULT_METRICS:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        has_trace = False

        for run in runs:
            history = _history_epoch_rows(run.get("history", []))
            x_vals = []
            y_vals = []
            for step in history:
                x = step.get("epoch")
                y = step.get(metric)
                if _is_number(x) and _is_number(y):
                    x_vals.append(x)
                    y_vals.append(y)

            if x_vals:
                has_trace = True
                ax.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    linewidth=1.6,
                    label=run["name"],
                )

        if not has_trace:
            plt.close(fig)
            continue

        ax.set_title(f"Epoch Curve: {metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(output_dir / f"epoch_{metric}.png", dpi=160)
        plt.close(fig)


def _read_csv_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_mapping_feature_importance_chart(summary_rows, output_path: Path, top_n=30):
    if not summary_rows:
        return

    rows = [
        {
            "feature_id": _to_int(r.get("feature_id")),
            "count_active": _to_int(r.get("count_active")),
            "p95_strength": _to_float(r.get("p95_strength")),
        }
        for r in summary_rows
    ]
    rows.sort(key=lambda r: (r["count_active"], r["p95_strength"]), reverse=True)
    rows = rows[: max(1, int(top_n))]
    if not rows:
        return

    labels = [str(r["feature_id"]) for r in rows]
    counts = [r["count_active"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(labels, counts, color="#2a9d8f", alpha=0.9)
    ax.set_title("Top Features by Activity Count")
    ax.set_xlabel("feature_id")
    ax.set_ylabel("count_active")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=70)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_mapping_strength_histogram(events_rows, output_path: Path):
    strengths = [_to_float(r.get("strength")) for r in events_rows]
    strengths = [x for x in strengths if x > 0.0]
    if not strengths:
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.hist(strengths, bins=60, color="#264653", alpha=0.9)
    ax.set_title("Feature Strength Distribution")
    ax.set_xlabel("strength")
    ax.set_ylabel("frequency")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_mapping_token_heatmap(token_pair_rows, output_path: Path, top_features=20, top_tokens=20):
    if not token_pair_rows:
        return

    parsed = [
        {
            "feature_id": _to_int(r.get("feature_id")),
            "token_str": str(r.get("token_str", "")),
            "count": _to_int(r.get("count")),
        }
        for r in token_pair_rows
    ]

    feature_total = {}
    token_total = {}
    for row in parsed:
        feature_total[row["feature_id"]] = feature_total.get(row["feature_id"], 0) + row["count"]
        token_total[row["token_str"]] = token_total.get(row["token_str"], 0) + row["count"]

    feature_ids = [
        fid
        for fid, _ in sorted(feature_total.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_features))]
    ]
    tokens = [
        tok
        for tok, _ in sorted(token_total.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_tokens))]
    ]

    if not feature_ids or not tokens:
        return

    feature_to_i = {fid: i for i, fid in enumerate(feature_ids)}
    token_to_j = {tok: j for j, tok in enumerate(tokens)}

    mat = np.zeros((len(feature_ids), len(tokens)), dtype=np.float32)
    for row in parsed:
        i = feature_to_i.get(row["feature_id"])
        j = token_to_j.get(row["token_str"])
        if i is not None and j is not None:
            mat[i, j] += row["count"]

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_title("Feature-Token Count Heatmap")
    ax.set_xlabel("token_str")
    ax.set_ylabel("feature_id")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_ids)))
    ax.set_yticklabels([str(fid) for fid in feature_ids], fontsize=8)
    fig.colorbar(image, ax=ax, label="count")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_mapping_sequence_timeline(events_rows, output_path: Path):
    if not events_rows:
        return

    seq_counts = {}
    for row in events_rows:
        seq = _to_int(row.get("sequence_idx"), default=-1)
        if seq >= 0:
            seq_counts[seq] = seq_counts.get(seq, 0) + 1

    if not seq_counts:
        return

    selected_seq = max(seq_counts.items(), key=lambda kv: kv[1])[0]
    selected = [r for r in events_rows if _to_int(r.get("sequence_idx"), -1) == selected_seq]
    if not selected:
        return

    x_vals = [_to_int(r.get("token_idx_in_sequence")) for r in selected]
    y_vals = [_to_int(r.get("feature_id")) for r in selected]
    sizes = [max(10.0, _to_float(r.get("strength")) * 12.0) for r in selected]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.scatter(x_vals, y_vals, s=sizes, alpha=0.5, c="#e76f51")
    ax.set_title(f"Feature Timeline for sequence_idx={selected_seq}")
    ax.set_xlabel("token_idx_in_sequence")
    ax.set_ylabel("feature_id")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_placeholder_chart(output_path: Path, title: str, message: str):
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_deception_layer_comparison(layer_rows, output_path: Path):
    if not layer_rows:
        _save_placeholder_chart(
            output_path,
            "Deception Layer Comparison",
            "No direction/probe metrics were available.",
        )
        return

    labels = [row.get("layer_label", row.get("layer", f"layer_{idx}")) for idx, row in enumerate(layer_rows)]
    direction_acc = [float(row.get("direction_accuracy", 0.0) or 0.0) for row in layer_rows]
    probe_val = [float(row.get("probe_validation_accuracy", 0.0) or 0.0) for row in layer_rows]
    separation = [float(row.get("cluster_separation", 0.0) or 0.0) for row in layer_rows]
    explained = [float(row.get("explained_variance", 0.0) or 0.0) for row in layer_rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].bar(x - width / 2, direction_acc, width=width, label="direction_accuracy", color="#4e79a7")
    axes[0].bar(x + width / 2, probe_val, width=width, label="probe_validation_accuracy", color="#f28e2b")
    axes[0].axhline(0.7, color="#888888", linestyle="--", linewidth=1, label="direction target")
    axes[0].axhline(0.75, color="#c44e52", linestyle=":", linewidth=1, label="probe target")
    axes[0].set_title("Direction and Probe Accuracy by Layer")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x - width / 2, separation, width=width, label="cluster_separation", color="#59a14f")
    axes[1].bar(x + width / 2, explained, width=width, label="explained_variance", color="#9c755f")
    axes[1].axhline(0.3, color="#888888", linestyle="--", linewidth=1, label="separation target")
    axes[1].set_title("Layer Separation and Explained Variance")
    axes[1].set_ylabel("score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_deception_tsne_scatter(tsne_payloads, output_path: Path):
    if not tsne_payloads:
        _save_placeholder_chart(
            output_path,
            "Honest vs Deceptive Projection",
            "No activation projection data was available.",
        )
        return

    items = list(tsne_payloads.items())
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(5, rows * 4.5)), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for index, (layer_name, payload) in enumerate(items):
        ax = axes[index // cols, index % cols]
        points = payload.get("points", [])
        if not points:
            ax.axis("off")
            ax.set_title(payload.get("layer_label", layer_name))
            ax.text(0.5, 0.5, "No points available", ha="center", va="center")
            continue

        honest = [point for point in points if point.get("label") == "honest"]
        deceptive = [point for point in points if point.get("label") == "deceptive"]
        if honest:
            ax.scatter(
                [point["x"] for point in honest],
                [point["y"] for point in honest],
                s=18,
                alpha=0.65,
                label="honest",
                color="#4e79a7",
            )
        if deceptive:
            ax.scatter(
                [point["x"] for point in deceptive],
                [point["y"] for point in deceptive],
                s=18,
                alpha=0.65,
                label="deceptive",
                color="#e15759",
            )
        method = payload.get("method", "projection")
        ax.set_title(f"{payload.get('layer_label', layer_name)} ({method})")
        ax.set_xlabel("component_1")
        ax.set_ylabel("component_2")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)

    for index in range(len(items), rows * cols):
        axes[index // cols, index % cols].axis("off")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_probe_roc_curve(roc_curves, output_path: Path):
    if not roc_curves:
        _save_placeholder_chart(
            output_path,
            "Probe ROC Curve",
            "No probe validation scores were available.",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888", label="random")

    has_curve = False
    for layer_name, payload in roc_curves.items():
        fpr = payload.get("fpr", [])
        tpr = payload.get("tpr", [])
        if not fpr or not tpr:
            continue
        has_curve = True
        auc = float(payload.get("auc", 0.0))
        label = f"{payload.get('layer_label', layer_name)} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, linewidth=2, label=label)

    if not has_curve:
        plt.close(fig)
        _save_placeholder_chart(
            output_path,
            "Probe ROC Curve",
            "ROC inputs were present but empty.",
        )
        return

    ax.set_title("Probe ROC Curves")
    ax.set_xlabel("false_positive_rate")
    ax.set_ylabel("true_positive_rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_category_breakdown_heatmap(categories, column_labels, values, output_path: Path):
    if not categories or values is None or np.size(values) == 0:
        _save_placeholder_chart(
            output_path,
            "Category Breakdown",
            "No category-level detection data was available.",
        )
        return

    matrix = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, max(4, len(categories) * 0.8)), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_title("Category-Level Detection Rate")
    ax.set_xlabel("layer")
    ax.set_ylabel("category")
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            text_color = "white" if value >= 0.55 else "black"
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color=text_color)

    fig.colorbar(image, ax=ax, label="detection_rate")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_adversarial_analysis_chart(summary, output_path: Path):
    if not summary:
        _save_placeholder_chart(
            output_path,
            "Adversarial Analysis",
            "No adversarial summary was available.",
        )
        return

    by_category = summary.get("by_category", {}) or {}
    by_difficulty = summary.get("by_difficulty", {}) or {}
    total_missed = int(summary.get("total_missed", 0) or 0)

    if total_missed <= 0 and not by_category and not by_difficulty:
        _save_placeholder_chart(
            output_path,
            "Adversarial Analysis",
            "No missed deceptive samples were recorded.",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    if by_category:
        axes[0].bar(list(by_category.keys()), list(by_category.values()), color="#e15759", alpha=0.9)
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_title("Missed Samples by Category")
        axes[0].set_ylabel("missed_count")
        axes[0].grid(axis="y", alpha=0.25)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No category failures.", ha="center", va="center")

    if by_difficulty:
        axes[1].bar(list(by_difficulty.keys()), list(by_difficulty.values()), color="#4e79a7", alpha=0.9)
        axes[1].set_title("Missed Samples by Difficulty")
        axes[1].set_ylabel("missed_count")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No difficulty failures.", ha="center", va="center")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _resolve_targets(config):
    return resolve_requested_targets(config)


def _mode_output_dir(base_config, mode, target, override_dir):
    from rfm.layout import _is_multi_layer, default_feature_mapping_dir, model_slug, sanitize_layer_name

    if override_dir and override_dir != "reports":
        output_dir = Path(override_dir) / mode
        if _is_multi_layer(base_config):
            output_dir = output_dir / sanitize_layer_name(target)
        return output_dir

    if mode == "mapping":
        return Path(default_feature_mapping_dir(base_config, target=target)) / "viz"

    output_dir = Path("runs") / model_slug(base_config) / "reports" / "training"
    if _is_multi_layer(base_config):
        output_dir = output_dir / sanitize_layer_name(target)
    return output_dir


def _configured_training_args(base_config, target, pattern):
    target_config = base_config.for_target(target)
    from rfm.layout import resolve_checkpoint_path

    checkpoint_path = Path(resolve_checkpoint_path(target_config, target=target))
    return {
        "checkpoints_dir": str(checkpoint_path.parent),
        "pattern": pattern,
    }


def _configured_mapping_args(base_config, target, top_features):
    target_config = base_config.for_target(target)
    from rfm.layout import default_feature_mapping_dir

    mapping_cfg = target_config.get("feature-mapping", {})
    mapping_dir = Path(default_feature_mapping_dir(target_config, target=target))
    events_path = Path(mapping_cfg.get("event_output_path", str(mapping_dir / "feature_mapping_events.csv")))
    summary_path = Path(mapping_cfg.get("summary_csv_output_path", str(mapping_dir / "feature_mapping_feature_summary.csv")))
    token_pairs_path = summary_path.with_name(summary_path.stem + "_token_pairs.csv")
    return {
        "mapping_events_csv": str(events_path),
        "mapping_summary_csv": str(summary_path),
        "mapping_token_pairs_csv": str(token_pairs_path),
        "top_features": top_features,
    }


def run_training_reports(args, output_dir: Path):
    checkpoints_dir = Path(args.checkpoints_dir)

    paths = sorted(checkpoints_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found in '{checkpoints_dir}' with pattern '{args.pattern}'."
        )

    runs = [load_checkpoint(path) for path in paths]
    rows = [row for row in (final_metrics(run) for run in runs) if row is not None]

    if not rows:
        raise RuntimeError(
            "Checkpoint files were found but none contain training history."
        )

    rows.sort(key=lambda r: str(r.get("name", "")))

    write_summary_csv(rows, output_dir / "summary.csv")
    build_pareto_chart(rows, output_dir / "pareto_recon_vs_sparsity.png")
    build_lambda_tradeoff_chart(rows, output_dir / "lambda_tradeoff.png")
    build_epoch_metric_charts(runs, output_dir)

    print(f"Generated training reports in: {output_dir.resolve()}")
    print(f"Runs included: {len(rows)}")


def run_mapping_reports(args, output_dir: Path):
    events_path = Path(args.mapping_events_csv)
    summary_path = Path(args.mapping_summary_csv)
    token_pairs_path = Path(args.mapping_token_pairs_csv)

    events_rows = _read_csv_rows(events_path)
    summary_rows = _read_csv_rows(summary_path)
    token_pair_rows = _read_csv_rows(token_pairs_path)

    build_mapping_feature_importance_chart(
        summary_rows=summary_rows,
        output_path=output_dir / "mapping_feature_importance.png",
        top_n=args.top_features,
    )
    build_mapping_strength_histogram(
        events_rows=events_rows,
        output_path=output_dir / "mapping_strength_histogram.png",
    )
    build_mapping_token_heatmap(
        token_pair_rows=token_pair_rows,
        output_path=output_dir / "mapping_feature_token_heatmap.png",
        top_features=min(args.top_features, 20),
        top_tokens=20,
    )
    build_mapping_sequence_timeline(
        events_rows=events_rows,
        output_path=output_dir / "mapping_sequence_timeline.png",
    )

    print(f"Generated mapping reports in: {output_dir.resolve()}")
    print(f"Events rows: {len(events_rows)}")
    print(f"Feature summary rows: {len(summary_rows)}")


def run_reports_from_config(args):
    from rfm.config import ConfigManager

    base_config = ConfigManager.from_file(args.config)
    targets = _resolve_targets(base_config)

    for target in targets:
        if len(targets) > 1:
            print(f"[viz] Processing target layer: {target}")

        if args.mode in {"training", "all"}:
            training_output_dir = _mode_output_dir(base_config, "training", target, args.output_dir)
            training_output_dir.mkdir(parents=True, exist_ok=True)
            run_training_reports(
                SimpleNamespace(**_configured_training_args(base_config, target, args.pattern)),
                training_output_dir,
            )

        if args.mode in {"mapping", "all"}:
            mapping_output_dir = _mode_output_dir(base_config, "mapping", target, args.output_dir)
            mapping_output_dir.mkdir(parents=True, exist_ok=True)
            run_mapping_reports(
                SimpleNamespace(**_configured_mapping_args(base_config, target, args.top_features)),
                mapping_output_dir,
            )


def main():
    args = parse_args()

    if args.config:
        run_reports_from_config(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "training":
        run_training_reports(args, output_dir)
        return

    if args.mode == "all":
        training_output_dir = output_dir / "training"
        mapping_output_dir = output_dir / "mapping"
        training_output_dir.mkdir(parents=True, exist_ok=True)
        mapping_output_dir.mkdir(parents=True, exist_ok=True)
        run_training_reports(args, training_output_dir)
        run_mapping_reports(args, mapping_output_dir)
        return

    run_mapping_reports(args, output_dir)


if __name__ == "__main__":
    main()
