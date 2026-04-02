"""CLI: Contrastive safety scoring and cross-layer analysis.

Modes:
    contrastive   — Per-layer feature risk scoring (toxic vs safe activation patterns)
    cross-layer   — Cross-layer feature combination analysis

Usage:
    python -m cli.safety_score --config configs/models/qwen3-0.6B.safety-gen.json --mode contrastive
    python -m cli.safety_score --config configs/models/qwen3-0.6B.safety-gen.json --mode cross-layer
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.layout import resolve_activations_dir, resolve_best_checkpoint, resolve_requested_targets
from rfm.sae.model import load_sae_checkpoint
from rfm.safety.contrastive import ContrastiveScorer

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("cli.safety_score")


def parse_args():
    parser = argparse.ArgumentParser(description="Safety scoring for SAE features.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--mode", default="contrastive",
        choices=["contrastive", "cross-layer"],
        help="Scoring mode.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features to report.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--layer", type=str, default=None,
                        help="Score a specific layer only (e.g. blocks.6.hook_resid_post). "
                             "If not set, scores all configured layers.")
    return parser.parse_args()


def cmd_contrastive(config, targets, top_k, output_base):
    """Per-layer contrastive safety scoring."""
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")

    all_layer_results = {}

    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        print(f"\n{'='*60}")
        print(f"[safety] Contrastive scoring: {target}")
        print(f"{'='*60}")

        # 1. Load SAE checkpoint
        try:
            sae_path = resolve_best_checkpoint(target_config, target=target)
        except Exception as e:
            logger.warning(f"No SAE checkpoint for {target}: {e}. Skipping.")
            continue

        sae_model, _ = load_sae_checkpoint(sae_path, device=device)
        scorer = ContrastiveScorer(sae_model, device=device)

        # 2. Find activation chunks
        chunk_dir = resolve_activations_dir(target_config, target=target)
        if not Path(chunk_dir).exists():
            logger.warning(f"No activation chunks at {chunk_dir}. Run extraction first.")
            continue

        # 3. Score
        try:
            scores = scorer.score_from_chunks(chunk_dir)
        except ValueError as e:
            logger.error(f"Scoring failed for {target}: {e}")
            continue

        # 4. Save full results
        output_dir = Path(output_base) if output_base else Path(chunk_dir).parent / "safety_scores"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{target.replace('.', '_')}_contrastive.csv"
        scorer.save_scores(scores, csv_path)

        # 5. Print top dangerous features
        top_dangerous = scorer.top_dangerous(scores, top_k=top_k, direction="toxic")
        all_layer_results[target] = top_dangerous

        print(f"\n  Top {min(top_k, len(top_dangerous))} toxic-associated features:")
        print(f"  {'ID':>6} {'toxic_rate':>10} {'safe_rate':>10} {'ratio':>8} {'fisher':>10} {'risk':>8}")
        print(f"  {'─'*60}")
        for s in top_dangerous[:20]:
            print(
                f"  {s['feature_id']:>6d} "
                f"{s['toxic_rate']:>10.4f} "
                f"{s['safe_rate']:>10.4f} "
                f"{s['rate_ratio']:>8.2f} "
                f"{s['fisher_score']:>10.4f} "
                f"{s['risk_score']:>8.2f}"
            )

    # Save combined summary
    if all_layer_results:
        summary_path = (Path(output_base) if output_base else Path("safety_scores")) / "contrastive_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {}
        for target, feats in all_layer_results.items():
            summary[target] = [
                {"feature_id": f["feature_id"], "risk_score": f["risk_score"], "rate_ratio": f["rate_ratio"]}
                for f in feats
            ]

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[safety] Summary saved to {summary_path}")


def cmd_cross_layer(config, targets, top_k, output_base):
    """Cross-layer feature combination analysis using interpretable classifier."""
    from rfm.safety.cross_layer import CrossLayerAnalyzer

    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("[safety] Cross-layer feature combination analysis")
    print(f"  Layers: {targets}")
    print(f"{'='*60}")

    # Load SAE models and chunk directories
    sae_models = {}
    chunk_dirs = {}

    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config

        try:
            sae_path = resolve_best_checkpoint(target_config, target=target)
            sae_model, _ = load_sae_checkpoint(sae_path, device=device)
            sae_models[target] = sae_model
        except Exception as e:
            logger.warning(f"No SAE for {target}: {e}. Skipping this layer.")
            continue

        chunk_dir = resolve_activations_dir(target_config, target=target)
        if Path(chunk_dir).exists():
            chunk_dirs[target] = chunk_dir
        else:
            logger.warning(f"No chunks for {target}. Skipping.")

    available_targets = [t for t in targets if t in sae_models and t in chunk_dirs]
    if len(available_targets) < 2:
        logger.error("Need at least 2 layers with SAE + chunks for cross-layer analysis.")
        return

    analyzer = CrossLayerAnalyzer(
        sae_models={t: sae_models[t] for t in available_targets},
        device=device,
    )

    # Load contrastive scores to select top features per layer
    output_dir = Path(output_base) if output_base else Path("safety_scores")
    contrastive_features = {}
    for target in available_targets:
        csv_path = output_dir / f"{target.replace('.', '_')}_contrastive.csv"
        if csv_path.exists():
            import csv
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                feats = [int(row["feature_id"]) for row in reader if float(row["risk_score"]) > 0]
                contrastive_features[target] = feats[:top_k]
                logger.info(f"  {target}: {len(contrastive_features[target])} risky features from contrastive CSV")

    # Build joint feature matrix
    results = analyzer.analyze(
        chunk_dirs={t: chunk_dirs[t] for t in available_targets},
        risky_features=contrastive_features if contrastive_features else None,
        top_k_per_layer=top_k,
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    if results.get("combinations"):
        combo_path = output_dir / "cross_layer_combinations.csv"
        import csv
        with open(combo_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results["combinations"][0].keys()))
            writer.writeheader()
            writer.writerows(results["combinations"])
        print(f"\n[safety] Cross-layer combinations saved to {combo_path}")

    if results.get("classifier_report"):
        report_path = output_dir / "safety_classifier_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results["classifier_report"], f, indent=2)
        print(f"[safety] Classifier report saved to {report_path}")

    if results.get("feature_importance"):
        importance_path = output_dir / "feature_importance.json"
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(results["feature_importance"], f, indent=2)
        print(f"[safety] Feature importance saved to {importance_path}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)

    if args.layer:
        targets = [args.layer]
    else:
        targets = resolve_requested_targets(config)

    if args.mode == "contrastive":
        cmd_contrastive(config, targets, args.top_k, args.output_dir)
    elif args.mode == "cross-layer":
        cmd_cross_layer(config, targets, args.top_k, args.output_dir)


if __name__ == "__main__":
    main()
