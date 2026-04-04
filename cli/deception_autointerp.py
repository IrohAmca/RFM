"""CLI: LLM-based auto-interpretation of top axis-associated SAE features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rfm.config import ConfigManager
from rfm.deception.deception_autointerp import DeceptionFeatureAutoInterp
from rfm.deception.utils import deception_run_dir
from rfm.layout import (
    resolve_activations_dir,
    resolve_best_checkpoint,
    resolve_requested_targets,
    sanitize_layer_name,
)
from rfm.patterns import ContrastAxisSpec, load_model_and_tokenizer, load_pattern_bundle, pattern_artifact_paths
from rfm.sae.model import load_sae_checkpoint

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpret top endpoint-B features from the canonical pattern bundle via LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to config file.")
    parser.add_argument(
        "--layer",
        default=None,
        help="Single layer to interpret (e.g. blocks.20.hook_resid_post). Defaults to all configured layers.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of top endpoint-B-associated features to interpret per layer.")
    parser.add_argument("--top-k-contexts", type=int, default=8, help="Number of top activating examples to retrieve per feature.")
    parser.add_argument("--request-delay", type=float, default=2.5, help="Seconds between LLM requests.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use for interpretation.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key override.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for an OpenAI-compatible API.")
    parser.add_argument("--groq", action="store_true", help="Use Groq's OpenAI-compatible endpoint and GROQ_API_KEY.")
    parser.add_argument("--local", action="store_true", help="Use the locally cached HF model instead of an external API.")
    parser.add_argument("--local-max-new-tokens", type=int, default=96, help="Generation length for local interpretation mode.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results and re-interpret from scratch.")
    return parser.parse_args()


def _resolve_api_key(args) -> str | None:
    if args.api_key:
        return args.api_key
    if args.groq:
        return os.environ.get("GROQ_API_KEY", "") or None
    return os.environ.get("OPENAI_API_KEY", "") or None


def _resolve_scenarios_path(config) -> str | None:
    primary = config.get("deception.scenario_generator.cache_path")
    sibling_fallback = None
    if primary:
        sibling_fallback = str(Path(primary).with_name("scenarios.jsonl"))
    fallback = deception_run_dir(config, "scenarios.jsonl")
    candidates = [primary, sibling_fallback, str(fallback)]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path.stat().st_size > 0:
            return str(path)
    return primary


def _load_top_features_from_bundle(config, target: str, top_n: int) -> list[int]:
    axis = ContrastAxisSpec.from_config(config)
    paths = pattern_artifact_paths(config, axis)
    bundle = load_pattern_bundle(config, axis)
    rows = list(bundle.get("layers", {}).get(target, {}).get("feature_scores", []) or [])
    if not rows:
        raise FileNotFoundError(
            f"No feature scores found for {target} in pattern bundle: {paths['bundle']}\n"
            "Run: python -m cli.deception_cycle --phase patterns  or  python -m cli.pattern_score --mode feature-score"
        )
    rows.sort(
        key=lambda row: (
            float(row.get("delta", 0.0)),
            abs(float(row.get("effect_size", 0.0))),
        ),
        reverse=True,
    )
    return [int(row["feature_id"]) for row in rows[:top_n]]


def _output_path(config, target: str) -> Path:
    safe = sanitize_layer_name(target)
    return deception_run_dir(config, "autointerp", f"{safe}_interpretations.json")


def run_autointerp(
    config,
    target: str,
    args,
    *,
    api_key: str | None,
    base_url: str | None,
    use_local: bool,
    runtime_model=None,
    runtime_tokenizer=None,
) -> None:
    target_config = config.for_target(target) if hasattr(config, "for_target") else config
    axis = ContrastAxisSpec.from_config(config)
    pattern_paths = pattern_artifact_paths(config, axis)

    print(f"\n[autointerp] Layer: {target}")
    print(f"  Loading top-{args.top_n} {axis.endpoint_b}-associated features from {pattern_paths['bundle']} ...")
    feature_ids = _load_top_features_from_bundle(config, target, args.top_n)
    print(f"  Features selected: {feature_ids[:10]}{'...' if len(feature_ids) > 10 else ''}")

    sae_path = resolve_best_checkpoint(target_config, target=target)
    device = config.get("train.device", "cpu")
    print(f"  Loading SAE from {sae_path} ...")
    sae_model, _ = load_sae_checkpoint(sae_path, device=device)

    chunk_dir = resolve_activations_dir(target_config, target=target)
    scenarios_path = _resolve_scenarios_path(config)
    interp = DeceptionFeatureAutoInterp(
        sae_model=sae_model,
        chunk_dir=chunk_dir,
        scenarios_path=scenarios_path,
        device=device,
    )

    print(f"  Encoding activations + finding top-{args.top_k_contexts} contexts per feature ...")
    contexts = interp.find_top_contexts(feature_ids, top_k=args.top_k_contexts)

    out_path = _output_path(config, target)
    existing: dict[int, str] = {}
    if not args.no_resume and out_path.exists():
        try:
            raw = json.loads(out_path.read_text(encoding="utf-8"))
            existing = {int(k): v for k, v in raw.items()}
            skip = sum(1 for fid in feature_ids if fid in existing)
            if skip:
                print(f"  Resuming: {skip}/{len(feature_ids)} already interpreted.")
        except Exception as exc:
            print(f"  Warning: could not load existing results: {exc}")

    if use_local:
        print(f"  Calling local model ({config.get('model_name')}) for interpretation ...")
        results = interp.interpret_features_locally(
            feature_contexts=contexts,
            model=runtime_model,
            tokenizer=runtime_tokenizer,
            max_new_tokens=args.local_max_new_tokens,
            request_delay=args.request_delay,
            existing_results=existing,
        )
    else:
        print(f"  Calling LLM ({args.model}) for interpretation ...")
        results = interp.interpret_features(
            feature_contexts=contexts,
            api_key=api_key,
            model=args.model,
            base_url=base_url,
            request_delay=args.request_delay,
            existing_results=existing,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved {len(results)} interpretations -> {out_path}")

    print(f"\n  Top interpretations for {target}:")
    for fid in feature_ids[:10]:
        print(f"    F{fid:>5}: {results.get(fid, '[missing]')}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)

    api_key = _resolve_api_key(args)
    use_local = bool(args.local or not api_key)
    if use_local and not args.local:
        print("[autointerp] No API key found. Falling back to the locally cached model.")
    base_url = None
    runtime_model = None
    runtime_tokenizer = None
    if use_local:
        runtime_model, runtime_tokenizer, _ = load_model_and_tokenizer(config)
    else:
        base_url = args.base_url
        if args.groq:
            base_url = GROQ_BASE_URL
        if base_url is None:
            base_url = "https://api.openai.com/v1"

    targets = [args.layer] if args.layer else resolve_requested_targets(config)
    for target in targets:
        run_autointerp(
            config,
            target,
            args,
            api_key=api_key,
            base_url=base_url,
            use_local=use_local,
            runtime_model=runtime_model,
            runtime_tokenizer=runtime_tokenizer,
        )


if __name__ == "__main__":
    main()
