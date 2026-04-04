"""CLI: axis-neutral feature scoring and motif discovery."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.layout import resolve_activations_dir, resolve_best_checkpoint, resolve_requested_targets
from rfm.patterns import (
    ContrastAxisSpec,
    MotifCausalValidator,
    PatternDiscoveryAnalyzer,
    analysis_payload_from_result,
    apply_causal_validation,
    layer_payload_from_result,
    load_pattern_bundle,
    pattern_artifact_paths,
    update_pattern_bundle,
)
from rfm.sae.model import load_sae_checkpoint

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("cli.safety_score")


def parse_args():
    parser = argparse.ArgumentParser(description="Axis-neutral feature scoring and motif discovery.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--mode",
        default="contrastive",
        choices=["contrastive", "feature-score", "cross-layer", "motif-discovery"],
        help="Scoring mode. `contrastive` and `feature-score` are aliases; `cross-layer` and `motif-discovery` are aliases.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Number of top features or motifs to print.")
    parser.add_argument("--output-dir", default=None, help="Deprecated. Canonical outputs now go to the pattern artifact directory.")
    parser.add_argument("--layer", type=str, default=None, help="Score a specific layer only.")
    return parser.parse_args()


def _canonical_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in {"contrastive", "feature-score"}:
        return "feature-score"
    return "motif-discovery"


def _device(config) -> str:
    return config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")


def _axis_spec(config) -> ContrastAxisSpec:
    return ContrastAxisSpec.from_config(config)


def _pattern_kwargs(config) -> dict:
    return {
        "aggregation_candidates": list(config.get("patterns.aggregation_candidates", ["mean", "topk_mean_4", "lastk_mean_8", "max"])),
        "cv_folds": int(config.get("patterns.cv_folds", config.get("deception.probe.cross_validation_folds", 5))),
        "top_endpoint_a": int(config.get("patterns.feature_pool.endpoint_a", 12)),
        "top_endpoint_b": int(config.get("patterns.feature_pool.endpoint_b", 12)),
        "top_interaction": int(config.get("patterns.feature_pool.interaction", 8)),
        "stability_min_fraction": float(config.get("patterns.stability_min_fraction", 0.6)),
        "max_tree_depth": int(config.get("patterns.max_tree_depth", 5)),
        "min_interaction_gain": float(config.get("patterns.min_interaction_gain", 0.005)),
        "intervention_min_shift": float(config.get("patterns.intervention_min_shift", 0.01)),
    }


def _resolve_available_inputs(config, targets: list[str], *, device: str) -> tuple[dict[str, torch.nn.Module], dict[str, str]]:
    sae_models: dict[str, torch.nn.Module] = {}
    chunk_dirs: dict[str, str] = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        try:
            sae_path = resolve_best_checkpoint(target_config, target=target)
            sae_model, _ = load_sae_checkpoint(sae_path, device=device)
        except Exception as exc:
            logger.warning("Skipping %s: SAE checkpoint unavailable (%s)", target, exc)
            continue

        chunk_dir = resolve_activations_dir(target_config, target=target)
        if not Path(chunk_dir).exists():
            logger.warning("Skipping %s: activation chunks not found at %s", target, chunk_dir)
            continue

        sae_models[target] = sae_model
        chunk_dirs[target] = chunk_dir
    return sae_models, chunk_dirs


def _warn_output_dir(output_base: str | None) -> None:
    if output_base:
        logger.warning("Ignoring --output-dir=%s. Canonical outputs are written to the pattern artifact directory.", output_base)


def _print_top_features(target: str, rows: list[dict], axis: ContrastAxisSpec, top_k: int) -> None:
    top_b = [row for row in rows if float(row.get("delta", 0.0)) > 0][:top_k]
    top_a = [row for row in rows if float(row.get("delta", 0.0)) < 0][:top_k]
    print(f"\n[patterns] Layer: {target}")
    print(f"  Top {min(len(top_b), top_k)} {axis.endpoint_b}-associated features")
    print(f"  {'ID':>6} {'delta':>10} {'effect':>10} {'A_rate':>10} {'B_rate':>10} {'q':>10}")
    print(f"  {'-' * 62}")
    for row in top_b[:top_k]:
        print(
            f"  {int(row['feature_id']):>6d} "
            f"{float(row['delta']):>10.4f} "
            f"{float(row['effect_size']):>10.4f} "
            f"{float(row['activation_rate_a']):>10.4f} "
            f"{float(row['activation_rate_b']):>10.4f} "
            f"{float(row['q_value']):>10.4g}"
        )
    print(f"  Top {min(len(top_a), top_k)} {axis.endpoint_a}-associated features")
    print(f"  {'ID':>6} {'delta':>10} {'effect':>10} {'A_rate':>10} {'B_rate':>10} {'q':>10}")
    print(f"  {'-' * 62}")
    for row in top_a[:top_k]:
        print(
            f"  {int(row['feature_id']):>6d} "
            f"{float(row['delta']):>10.4f} "
            f"{float(row['effect_size']):>10.4f} "
            f"{float(row['activation_rate_a']):>10.4f} "
            f"{float(row['activation_rate_b']):>10.4f} "
            f"{float(row['q_value']):>10.4g}"
        )


def _maybe_validate_causally(config, axis: ContrastAxisSpec, sae_models, chunk_dirs, result: dict) -> dict:
    stable_candidates = [item for item in result.get("motif_candidates", []) if item.get("status") == "stable"]
    if not stable_candidates:
        result = dict(result)
        result["causal_validation"] = {
            "status": "skipped",
            "reason": "no_stable_candidates",
            "evaluated": 0,
            "supported": 0,
        }
        return result

    bundle = load_pattern_bundle(config, axis)
    validator = MotifCausalValidator(
        config=config,
        axis_spec=axis,
        bundle=bundle,
        sae_models=sae_models,
        chunk_dirs=chunk_dirs,
    )
    try:
        effect_rows, summary = validator.evaluate(stable_candidates)
    except Exception as exc:
        logger.warning("Skipping causal validation: %s", exc)
        result = dict(result)
        result["causal_validation"] = {
            "status": "skipped",
            "reason": str(exc),
            "evaluated": 0,
            "supported": 0,
        }
        return result

    if summary.get("status") != "ok" or not effect_rows:
        result = dict(result)
        result["causal_validation"] = summary
        return result
    return apply_causal_validation(result, effect_rows, summary=summary)


def cmd_contrastive(config, targets, top_k, output_base):
    _warn_output_dir(output_base)
    device = _device(config)
    axis = _axis_spec(config)
    sae_models, chunk_dirs = _resolve_available_inputs(config, list(targets), device=device)
    available = [target for target in targets if target in sae_models and target in chunk_dirs]
    if not available:
        logger.error("No layers available for feature scoring.")
        return {}

    layer_updates = {}
    for target in available:
        analyzer = PatternDiscoveryAnalyzer({target: sae_models[target]}, axis_spec=axis, device=device)
        result = analyzer.analyze({target: chunk_dirs[target]}, **_pattern_kwargs(config))
        layer_updates[target] = layer_payload_from_result(result, target)
        _print_top_features(target, layer_updates[target]["feature_scores"], axis, top_k)
        print(f"  Selected aggregation: {result['selected_aggregation']}")

    update_pattern_bundle(config, axis_spec=axis, layer_updates=layer_updates)
    paths = pattern_artifact_paths(config, axis)
    print(f"\n[patterns] Bundle saved to {paths['bundle']}")
    print(f"[patterns] Report saved to {paths['report']}")
    return {
        "layer_updates": layer_updates,
        "paths": paths,
    }


def cmd_cross_layer(config, targets, top_k, output_base):
    _warn_output_dir(output_base)
    device = _device(config)
    axis = _axis_spec(config)
    sae_models, chunk_dirs = _resolve_available_inputs(config, list(targets), device=device)
    available = [target for target in targets if target in sae_models and target in chunk_dirs]
    if len(available) < 2:
        logger.error("Cross-layer motif discovery requires at least two layers with SAE checkpoints and activation chunks.")
        return {}

    analyzer = PatternDiscoveryAnalyzer(
        {target: sae_models[target] for target in available},
        axis_spec=axis,
        device=device,
    )
    result = analyzer.analyze({target: chunk_dirs[target] for target in available}, **_pattern_kwargs(config))
    result = _maybe_validate_causally(config, axis, {target: sae_models[target] for target in available}, {target: chunk_dirs[target] for target in available}, result)
    layer_updates = {target: layer_payload_from_result(result, target) for target in available}
    update_pattern_bundle(
        config,
        axis_spec=axis,
        layer_updates=layer_updates,
        analysis=analysis_payload_from_result(result),
    )

    print(f"\n[patterns] Cross-layer motif discovery")
    print(f"  Axis: {axis.axis_id} ({axis.endpoint_a} vs {axis.endpoint_b})")
    print(f"  Layers: {', '.join(available)}")
    print(f"  Selected aggregation: {result['selected_aggregation']}")
    print(f"  Stable motifs: {len(result['stable_motifs'])}")
    print(f"  Stable interactions: {len(result['stable_interactions'])}")

    if result["stable_interactions"]:
        print(f"  Top {min(len(result['stable_interactions']), top_k)} stable interactions")
        print(f"  {'stability':>10} {'sign':>10}  motif")
        print(f"  {'-' * 72}")
        for item in result["stable_interactions"][:top_k]:
            print(
                f"  {float(item['stability']):>10.3f} "
                f"{str(item['sign']):>10}  "
                f"{item['name']}"
            )

    paths = pattern_artifact_paths(config, axis)
    print(f"\n[patterns] Bundle saved to {paths['bundle']}")
    print(f"[patterns] Report saved to {paths['report']}")
    return {
        "result": result,
        "layer_updates": layer_updates,
        "paths": paths,
    }


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)
    targets = [args.layer] if args.layer else resolve_requested_targets(config)
    mode = _canonical_mode(args.mode)

    if mode == "feature-score":
        cmd_contrastive(config, targets, args.top_k, args.output_dir)
    else:
        cmd_cross_layer(config, targets, args.top_k, args.output_dir)


if __name__ == "__main__":
    main()
