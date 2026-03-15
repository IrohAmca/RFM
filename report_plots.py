import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


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
        description="Generate SAE training charts from checkpoint histories."
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
    return parser.parse_args()


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu")
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


def final_metrics(run):
    if not run["history"]:
        return None

    last = run["history"][-1]
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
            history = run.get("history", [])
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


def main():
    args = parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Generated reports in: {output_dir.resolve()}")
    print(f"Runs included: {len(rows)}")


if __name__ == "__main__":
    main()
