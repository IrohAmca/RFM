"""CLI: contrastive safety scoring and cross-layer analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.layout import default_safety_scores_dir, resolve_activations_dir, resolve_best_checkpoint, resolve_requested_targets
from rfm.sae.model import load_sae_checkpoint
from rfm.safety.contrastive import ContrastiveScorer

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("cli.safety_score")


def parse_args():
    parser = argparse.ArgumentParser(description="Safety scoring for SAE features.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--mode",
        default="contrastive",
        choices=["contrastive", "cross-layer"],
        help="Scoring mode.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features to report.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--layer", type=str, default=None, help="Score a specific layer only.")
    return parser.parse_args()


def _contrastive_labels(config):
    positive_label = config.get("contrastive.positive_label")
    negative_label = config.get("contrastive.negative_label")
    if positive_label and negative_label:
        return positive_label, negative_label
    if config.get("deception", None):
        return positive_label or "deceptive", negative_label or "honest"
    return positive_label or "toxic", negative_label or "safe"


def _resolve_safety_output_dir(config, target: str | None, output_base: str | None) -> Path:
    if output_base:
        return Path(output_base)
    return Path(default_safety_scores_dir(config, target=target))


def cmd_contrastive(config, targets, top_k, output_base):
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")
    positive_label, negative_label = _contrastive_labels(config)
    all_layer_results = {}
    summary_output_dir = None

    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        print(f"\n{'=' * 60}")
        print(f"[safety] Contrastive scoring: {target}")
        print(f"{'=' * 60}")

        try:
            sae_path = resolve_best_checkpoint(target_config, target=target)
        except Exception as exc:
            logger.warning("No SAE checkpoint for %s: %s. Skipping.", target, exc)
            continue

        sae_model, _ = load_sae_checkpoint(sae_path, device=device)
        scorer = ContrastiveScorer(sae_model, device=device)

        chunk_dir = resolve_activations_dir(target_config, target=target)
        if not Path(chunk_dir).exists():
            logger.warning("No activation chunks at %s. Run extraction first.", chunk_dir)
            continue

        try:
            scores = scorer.score_from_chunks(
                chunk_dir,
                positive_label=positive_label,
                negative_label=negative_label,
            )
        except ValueError as exc:
            logger.error("Scoring failed for %s: %s", target, exc)
            continue

        output_dir = _resolve_safety_output_dir(config, target, output_base)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{target.replace('.', '_')}_contrastive.csv"
        scorer.save_scores(scores, csv_path)
        summary_output_dir = output_dir

        top_dangerous = scorer.top_dangerous(scores, top_k=top_k, direction=positive_label)
        all_layer_results[target] = top_dangerous

        pos_rate_key = f"{positive_label}_rate"
        neg_rate_key = f"{negative_label}_rate"
        print(f"\n  Top {min(top_k, len(top_dangerous))} {positive_label}-associated features:")
        print(f"  {'ID':>6} {pos_rate_key:>16} {neg_rate_key:>16} {'ratio':>8} {'fisher':>10} {'risk':>8}")
        print(f"  {'-' * 74}")
        for row in top_dangerous[:20]:
            print(
                f"  {row['feature_id']:>6d} "
                f"{row[pos_rate_key]:>16.4f} "
                f"{row[neg_rate_key]:>16.4f} "
                f"{row['rate_ratio']:>8.2f} "
                f"{row['fisher_score']:>10.4f} "
                f"{row['risk_score']:>8.2f}"
            )

    if all_layer_results and summary_output_dir is not None:
        summary_path = summary_output_dir / "contrastive_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            target: [
                {
                    "feature_id": feature["feature_id"],
                    "risk_score": feature["risk_score"],
                    "rate_ratio": feature["rate_ratio"],
                }
                for feature in features
            ]
            for target, features in all_layer_results.items()
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[safety] Summary saved to {summary_path}")


def cmd_cross_layer(config, targets, top_k, output_base):
    from rfm.safety.cross_layer import CrossLayerAnalyzer

    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")
    positive_label, negative_label = _contrastive_labels(config)
    print(f"\n{'=' * 60}")
    print("[safety] Cross-layer feature combination analysis")
    print(f"  Layers: {targets}")
    print(f"{'=' * 60}")

    sae_models = {}
    chunk_dirs = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        try:
            sae_path = resolve_best_checkpoint(target_config, target=target)
            sae_model, _ = load_sae_checkpoint(sae_path, device=device)
            sae_models[target] = sae_model
        except Exception as exc:
            logger.warning("No SAE for %s: %s. Skipping this layer.", target, exc)
            continue

        chunk_dir = resolve_activations_dir(target_config, target=target)
        if Path(chunk_dir).exists():
            chunk_dirs[target] = chunk_dir
        else:
            logger.warning("No chunks for %s. Skipping.", target)

    available_targets = [target for target in targets if target in sae_models and target in chunk_dirs]
    if len(available_targets) < 2:
        logger.error("Need at least 2 layers with SAE + chunks for cross-layer analysis.")
        return

    analyzer = CrossLayerAnalyzer(
        sae_models={target: sae_models[target] for target in available_targets},
        device=device,
    )

    output_dir = _resolve_safety_output_dir(config, available_targets[0], output_base)
    contrastive_features = {}
    for target in available_targets:
        csv_path = output_dir / f"{target.replace('.', '_')}_contrastive.csv"
        if not csv_path.exists():
            continue
        import csv

        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            features = [int(row["feature_id"]) for row in reader if float(row["risk_score"]) > 0]
            contrastive_features[target] = features[:top_k]
            logger.info("  %s: %s risky features from contrastive CSV", target, len(contrastive_features[target]))

    results = analyzer.analyze(
        chunk_dirs={target: chunk_dirs[target] for target in available_targets},
        risky_features=contrastive_features if contrastive_features else None,
        top_k_per_layer=top_k,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if results.get("combinations"):
        combo_path = output_dir / "cross_layer_combinations.csv"
        import csv

        with open(combo_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results["combinations"][0].keys()))
            writer.writeheader()
            writer.writerows(results["combinations"])
        print(f"\n[safety] Cross-layer combinations saved to {combo_path}")

    if results.get("classifier_report"):
        report_path = output_dir / "safety_classifier_report.json"
        report_path.write_text(json.dumps(results["classifier_report"], indent=2), encoding="utf-8")
        print(f"[safety] Classifier report saved to {report_path}")

    if results.get("feature_importance"):
        importance_path = output_dir / "feature_importance.json"
        importance_path.write_text(json.dumps(results["feature_importance"], indent=2), encoding="utf-8")
        print(f"[safety] Feature importance saved to {importance_path}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)
    targets = [args.layer] if args.layer else resolve_requested_targets(config)

    if args.mode == "contrastive":
        cmd_contrastive(config, targets, args.top_k, args.output_dir)
    else:
        cmd_cross_layer(config, targets, args.top_k, args.output_dir)


if __name__ == "__main__":
    main()
