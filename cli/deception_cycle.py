from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cli.extract_deception import extract_all_targets
from cli.train import run_training
from rfm.config import ConfigManager
from rfm.deception import (
    AdversarialSearch,
    DeceptionDataset,
    DeceptionDirectionFinder,
    DeceptionMonitor,
    DeceptionProbe,
    DirectionResult,
    ScenarioGenerator,
)
from rfm.deception.utils import deception_run_dir
from rfm.extractors.hf_generate import HFGenerationExtractor
from rfm.layout import (
    resolve_activations_dir,
    resolve_best_checkpoint,
    resolve_requested_targets,
    sanitize_layer_name,
)
from rfm.patterns import (
    ContrastAxisSpec,
    MotifCausalValidator,
    PatternDiscoveryAnalyzer,
    analysis_payload_from_result,
    apply_causal_validation,
    layer_payload_from_result,
    load_pattern_bundle,
    update_pattern_bundle,
)
from rfm.sae.model import load_sae_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run the deception-detection cycle.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--phase",
        default="full",
        choices=["full", "generate", "extract", "train", "direction", "probe", "patterns", "monitor", "adversarial"],
        help="Which phase to run.",
    )
    parser.add_argument("--cycle", type=int, default=1, help="Number of iterations to run.")
    parser.add_argument("--layer", default=None, help="Restrict to a single layer.")
    return parser.parse_args()


def _split_pairs(
    honest: torch.Tensor,
    deceptive: torch.Tensor,
    validation_split: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = honest.shape[0]
    if n != deceptive.shape[0]:
        raise ValueError("Honest/deceptive tensors must contain the same number of pairs.")
    if n < 2 or validation_split <= 0:
        return honest, deceptive, honest, deceptive

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    indices = torch.randperm(n, generator=generator)
    honest = honest[indices]
    deceptive = deceptive[indices]

    val_count = min(max(int(round(n * validation_split)), 1), n - 1)
    train_count = n - val_count
    return (
        honest[:train_count],
        deceptive[:train_count],
        honest[train_count:],
        deceptive[train_count:],
    )


def _targets(config, layer_override: str | None = None) -> list[str]:
    return [layer_override] if layer_override else resolve_requested_targets(config)


def _direction_path(config) -> Path:
    return deception_run_dir(config, "directions", "directions.pt")


def _probe_path(config, target: str) -> Path:
    return deception_run_dir(config, "probes", f"{sanitize_layer_name(target)}.pt")


def _monitor_path(config) -> Path:
    return deception_run_dir(config, "monitor", "monitor_bundle.pt")


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
        print(f"[deception_cycle] Skipping causal validation: {exc}")
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


def run_generate(config) -> list[dict]:
    generator = ScenarioGenerator.from_config(config)
    section = config.get("deception.scenario_generator", {})
    scenarios = generator.generate(
        categories=section.get("categories"),
        samples_per_category=int(section.get("samples_per_category", 100)),
        cache_path=section.get("cache_path"),
        resume=True,
    )
    print(f"[deception_cycle] Generated/loaded {len(scenarios)} scenarios")
    return [scenario.to_dict() for scenario in scenarios]


def run_extract(
    config,
    layer_override: str | None = None,
    *,
    ensure_scenarios: bool = True,
) -> None:
    if ensure_scenarios and bool(config.get("deception.extraction.auto_generate_scenarios", True)):
        run_generate(config)
    extractor = HFGenerationExtractor(config)
    dataset = DeceptionDataset(config=config, mode="paired")
    dataset.load()
    extract_all_targets(_targets(config, layer_override), extractor, dataset, config)


def run_train(config, layer_override: str | None = None) -> None:
    run_training(config, layer=layer_override)


def run_direction(config, layer_override: str | None = None) -> dict[str, DirectionResult]:
    axis = _axis_spec(config)
    finder = DeceptionDirectionFinder(
        aggregation=config.get("deception.direction.aggregation", "mean"),
    )
    validation_split = float(config.get("deception.direction.validation_split", 0.2))
    split_seed = int(config.get("deception.direction.split_seed", config.get("train.split_seed", 42)))
    method = config.get("deception.direction.method", "mean_diff")
    min_cluster = float(config.get("deception.direction.min_cluster_separation", 0.0))

    results = {}
    layer_updates = {}
    for target in _targets(config, layer_override):
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        chunk_dir = resolve_activations_dir(target_config, target=target)
        paired = finder.load_paired_activations(chunk_dir)
        train_h, train_d, val_h, val_d = _split_pairs(
            paired["honest"],
            paired["deceptive"],
            validation_split,
            split_seed,
        )
        result = finder.find_direction(train_h, train_d, method=method)
        if val_h.shape[0] > 0:
            validation = finder.validate_direction(result.direction, val_h, val_d)
            result = DirectionResult(
                method=result.method,
                direction=result.direction,
                explained_variance=result.explained_variance,
                cluster_separation=validation["cluster_separation"],
                validation_accuracy=validation["accuracy"],
                threshold=validation["threshold"],
            )
        finder.directions[target] = result
        results[target] = result
        layer_updates[target] = {
            "direction": result.to_dict()
        }
        status = "OK" if result.cluster_separation >= min_cluster else "LOW_SEPARATION"
        print(
            f"[deception_cycle] Direction {target}: "
            f"acc={result.validation_accuracy:.3f} "
            f"sep={result.cluster_separation:.3f} "
            f"var={result.explained_variance:.3f} "
            f"{status}"
        )

    finder.save(_direction_path(config))
    if layer_updates:
        update_pattern_bundle(config, axis_spec=axis, layer_updates=layer_updates)
    return results


def _load_directions(config) -> dict[str, DirectionResult]:
    finder = DeceptionDirectionFinder()
    return finder.load(_direction_path(config))


def run_probe(config, layer_override: str | None = None) -> dict[str, dict]:
    axis = _axis_spec(config)
    summary = {}
    layer_updates = {}
    validation_split = float(config.get("deception.direction.validation_split", 0.2))
    split_seed = int(config.get("deception.direction.split_seed", config.get("train.split_seed", 42)))
    cv_folds = int(config.get("deception.probe.cross_validation_folds", 5))
    directions = _load_directions(config) if _direction_path(config).exists() else {}
    finder = DeceptionDirectionFinder(
        aggregation=config.get("deception.direction.aggregation", "mean"),
    )

    for target in _targets(config, layer_override):
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        chunk_dir = resolve_activations_dir(target_config, target=target)
        paired = finder.load_paired_activations(chunk_dir)
        train_h, train_d, val_h, val_d = _split_pairs(
            paired["honest"],
            paired["deceptive"],
            validation_split,
            split_seed,
        )

        probe = DeceptionProbe()
        state = probe.train(train_h, train_d, cv_folds=cv_folds)

        honest_acc = []
        deceptive_acc = []
        for vector in val_h:
            honest_acc.append(1 if probe.predict(vector)[0] == "honest" else 0)
        for vector in val_d:
            deceptive_acc.append(1 if probe.predict(vector)[0] == "deceptive" else 0)
        val_accuracy = (
            (sum(honest_acc) + sum(deceptive_acc)) / max(len(honest_acc) + len(deceptive_acc), 1)
            if val_h.shape[0] > 0 else state.training_accuracy
        )

        probe_path = _probe_path(config, target)
        probe.save(probe_path)

        sae_features = []
        if target in directions:
            try:
                sae_path = resolve_best_checkpoint(target_config, target=target)
                sae_model, _ = load_sae_checkpoint(
                    sae_path,
                    device=config.get("train.device", "cpu"),
                )
                sae_features = probe.explain_with_sae(
                    sae_model,
                    direction=directions[target].direction,
                    top_k=20,
                )
                sae_report_path = deception_run_dir(config, "probes", f"{sanitize_layer_name(target)}_sae_features.json")
                sae_report_path.write_text(json.dumps(sae_features, indent=2), encoding="utf-8")
            except Exception:
                sae_features = []

        summary[target] = {
            "backend": state.backend,
            "training_accuracy": state.training_accuracy,
            "cv_accuracy": state.cv_accuracy,
            "validation_accuracy": float(val_accuracy),
            "probe_path": str(probe_path),
            "sae_feature_count": len(sae_features),
        }
        layer_updates[target] = {
            "probe": summary[target],
            "probe_state": state.to_dict(),
        }
        print(
            f"[deception_cycle] Probe {target}: "
            f"train_acc={state.training_accuracy:.3f} "
            f"val_acc={float(val_accuracy):.3f}"
        )

    summary_path = deception_run_dir(config, "probes", "probe_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_pattern_bundle(
        config,
        axis_spec=axis,
        layer_updates=layer_updates,
        artifacts={"probe_summary_path": str(summary_path)},
    )
    return summary


def run_patterns(config, layer_override: str | None = None) -> dict:
    axis = _axis_spec(config)
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")
    targets = _targets(config, layer_override)
    sae_models = {}
    chunk_dirs = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        try:
            sae_path = resolve_best_checkpoint(target_config, target=target)
            sae_model, _ = load_sae_checkpoint(sae_path, device=device)
            sae_models[target] = sae_model
        except Exception as exc:
            print(f"[deception_cycle] Skipping {target}: SAE checkpoint unavailable ({exc})")
            continue

        chunk_dir = resolve_activations_dir(target_config, target=target)
        if Path(chunk_dir).exists():
            chunk_dirs[target] = chunk_dir
        else:
            print(f"[deception_cycle] Skipping {target}: activation chunks not found at {chunk_dir}")

    available = [target for target in targets if target in sae_models and target in chunk_dirs]
    if not available:
        raise ValueError("Pattern discovery requires at least one layer with SAE checkpoint and activation chunks.")

    analyzer = PatternDiscoveryAnalyzer(
        {target: sae_models[target] for target in available},
        axis_spec=axis,
        device=device,
    )
    result = analyzer.analyze(
        {target: chunk_dirs[target] for target in available},
        **_pattern_kwargs(config),
    )
    result = _maybe_validate_causally(
        config,
        axis,
        {target: sae_models[target] for target in available},
        {target: chunk_dirs[target] for target in available},
        result,
    )

    layer_updates = {}
    for target in available:
        layer_updates[target] = layer_payload_from_result(result, target)

    update_pattern_bundle(
        config,
        axis_spec=axis,
        layer_updates=layer_updates,
        analysis=analysis_payload_from_result(result),
    )
    print(
        f"[deception_cycle] Patterns: layers={len(available)} "
        f"agg={result['selected_aggregation']} "
        f"stable_motifs={len(result['stable_motifs'])}"
    )
    return result


def _load_probes(config, targets: list[str]) -> dict[str, DeceptionProbe]:
    probes = {}
    for target in targets:
        path = _probe_path(config, target)
        if not path.exists():
            continue
        probe = DeceptionProbe()
        probe.load(path)
        probes[target] = probe
    return probes


def run_monitor(config, layer_override: str | None = None) -> dict:
    axis = _axis_spec(config)
    targets = _targets(config, layer_override)
    directions = _load_directions(config)
    probes = _load_probes(config, targets)
    threshold_cfg = config.get("deception.monitor.alert_threshold", 0.7)
    if isinstance(threshold_cfg, dict):
        thresholds = {layer: float(threshold_cfg.get(layer, 0.7)) for layer in targets}
    else:
        thresholds = {layer: float(threshold_cfg) for layer in targets}

    monitor = DeceptionMonitor(
        directions={layer: directions[layer] for layer in targets if layer in directions},
        probes=probes,
        thresholds=thresholds,
        ensemble_method=config.get("deception.monitor.ensemble_method", "weighted_average"),
    )

    finder = DeceptionDirectionFinder(
        aggregation=config.get("deception.direction.aggregation", "mean"),
    )
    layer_pairs = {}
    for target in targets:
        target_config = config.for_target(target) if hasattr(config, "for_target") else config
        chunk_dir = resolve_activations_dir(target_config, target=target)
        layer_pairs[target] = finder.load_paired_activations(chunk_dir)

    reference_target = targets[0]
    n_pairs = layer_pairs[reference_target]["honest"].shape[0]
    tp = fp = tn = fn = 0
    for index in range(n_pairs):
        honest_score = monitor.score_generation(
            {layer: layer_pairs[layer]["honest"][index].unsqueeze(0) for layer in targets}
        )
        deceptive_score = monitor.score_generation(
            {layer: layer_pairs[layer]["deceptive"][index].unsqueeze(0) for layer in targets}
        )
        if deceptive_score.alert:
            tp += 1
        else:
            fn += 1
        if honest_score.alert:
            fp += 1
        else:
            tn += 1

    report = {
        "pairs_evaluated": n_pairs,
        "detection_rate": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "thresholds": thresholds,
        "layers": targets,
    }

    monitor_payload = {
        "directions": {layer: directions[layer].to_dict() for layer in targets if layer in directions},
        "thresholds": thresholds,
        "report": report,
    }
    _monitor_path(config).parent.mkdir(parents=True, exist_ok=True)
    torch.save(monitor_payload, _monitor_path(config))

    report_path = deception_run_dir(config, "monitor", "monitor_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_pattern_bundle(
        config,
        axis_spec=axis,
        monitor=report,
        artifacts={
            "direction_path": str(_direction_path(config)),
            "monitor_bundle_path": str(_monitor_path(config)),
            "monitor_report_path": str(report_path),
        },
    )
    print(
        f"[deception_cycle] Monitor: DDR={report['detection_rate']:.3f} "
        f"FPR={report['false_positive_rate']:.3f}"
    )
    return report


def run_adversarial(config, layer_override: str | None = None) -> dict:
    axis = _axis_spec(config)
    targets = _targets(config, layer_override)
    directions = _load_directions(config)
    probes = _load_probes(config, targets)
    threshold_cfg = config.get("deception.monitor.alert_threshold", 0.7)
    if isinstance(threshold_cfg, dict):
        thresholds = {layer: float(threshold_cfg.get(layer, 0.7)) for layer in targets}
    else:
        thresholds = {layer: float(threshold_cfg) for layer in targets}

    monitor = DeceptionMonitor(
        directions={layer: directions[layer] for layer in targets if layer in directions},
        probes=probes,
        thresholds=thresholds,
        ensemble_method=config.get("deception.monitor.ensemble_method", "weighted_average"),
    )

    extractor = HFGenerationExtractor(config)
    generator = ScenarioGenerator.from_config(config)
    searcher = AdversarialSearch(generator)
    missed = searcher.search(
        monitor=monitor,
        target_model=extractor.model,
        tokenizer=extractor.tokenizer,
        n_attempts=int(config.get("deception.adversarial.attempts_per_cycle", 25)),
        categories=config.get("deception.scenario_generator.categories"),
        system_prompt_deceptive=config.get("deception.extraction.system_prompt_deceptive"),
        append_to=config.get("deception.scenario_generator.cache_path"),
    )
    summary = searcher.categorize_failures(missed)

    output_path = deception_run_dir(config, "adversarial", "missed_samples.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(missed, indent=2), encoding="utf-8")
    summary_path = deception_run_dir(config, "adversarial", "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_pattern_bundle(
        config,
        axis_spec=axis,
        adversarial=summary,
        artifacts={
            "adversarial_summary_path": str(summary_path),
            "missed_samples_path": str(output_path),
        },
    )
    print(f"[deception_cycle] Adversarial search: {summary['total_missed']} missed samples")
    return summary


def run_phase(config, phase: str, layer_override: str | None = None):
    if phase == "generate":
        return run_generate(config)
    if phase == "extract":
        return run_extract(config, layer_override, ensure_scenarios=True)
    if phase == "train":
        return run_train(config, layer_override)
    if phase == "direction":
        return run_direction(config, layer_override)
    if phase == "probe":
        return run_probe(config, layer_override)
    if phase == "patterns":
        return run_patterns(config, layer_override)
    if phase == "monitor":
        return run_monitor(config, layer_override)
    if phase == "adversarial":
        return run_adversarial(config, layer_override)

    run_generate(config)
    run_extract(config, layer_override, ensure_scenarios=False)
    run_train(config, layer_override)
    run_direction(config, layer_override)
    run_probe(config, layer_override)
    run_patterns(config, layer_override)
    run_monitor(config, layer_override)
    return run_adversarial(config, layer_override)


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)

    for cycle_index in range(1, args.cycle + 1):
        print(f"[deception_cycle] Cycle {cycle_index}/{args.cycle} phase={args.phase}")
        run_phase(config, args.phase, args.layer)


if __name__ == "__main__":
    main()
