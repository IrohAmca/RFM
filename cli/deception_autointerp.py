"""CLI: LLM-based auto-interpretation of top deception SAE features.

Usage
-----
  # Interpret top-20 features from blocks.20 using Groq (set GROQ_API_KEY in env):
  python -m cli.deception_autointerp \\
      --config configs/models/qwen3-0.6B.deception.json \\
      --layer blocks.20.hook_resid_post \\
      --top-n 20 --groq

  # Using OpenAI (set OPENAI_API_KEY in env):
  python -m cli.deception_autointerp \\
      --config configs/models/qwen3-0.6B.deception.json \\
      --top-n 15 --model gpt-4o-mini

  Required env vars (one of):
    GROQ_API_KEY    — used when --groq flag is set
    OPENAI_API_KEY  — used as fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from rfm.config import ConfigManager
from rfm.deception.deception_autointerp import DeceptionFeatureAutoInterp
from rfm.deception.utils import deception_run_dir
from rfm.layout import (
    default_safety_scores_dir,
    resolve_activations_dir,
    resolve_best_checkpoint,
    resolve_requested_targets,
    sanitize_layer_name,
)
from rfm.sae.model import load_sae_checkpoint

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpret top deception SAE features via LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to config file.")
    parser.add_argument(
        "--layer",
        default=None,
        help="Single layer to interpret (e.g. blocks.20.hook_resid_post). "
             "Defaults to all configured layers.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top deception features to interpret per layer.",
    )
    parser.add_argument(
        "--top-k-contexts",
        type=int,
        default=8,
        help="Number of top activating examples to retrieve per feature.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.5,
        help="Seconds between LLM requests (rate-limit guard).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for interpretation.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override. Otherwise read from env vars.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--groq",
        action="store_true",
        help="Use Groq's OpenAI-compatible endpoint and GROQ_API_KEY.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing partial results and re-interpret from scratch.",
    )
    return parser.parse_args()


def _resolve_api_key(args) -> str:
    """Read API key from env vars — matches the pattern used in cli/analyze.py."""
    if args.api_key:
        return args.api_key
    if args.groq:
        key = os.environ.get("GROQ_API_KEY", "")
        if key:
            return key
        raise ValueError(
            "GROQ_API_KEY not set in environment. "
            "Run: set GROQ_API_KEY=<your-key>  (Windows) or export GROQ_API_KEY=<key>  (Linux/Mac)"
        )
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    raise ValueError(
        "No API key found in environment.\n"
        "  For Groq: set GROQ_API_KEY=<key> and add --groq flag\n"
        "  For OpenAI: set OPENAI_API_KEY=<key>"
    )


def _load_top_features_from_csv(csv_path: Path, top_n: int) -> list[int]:
    """Read contrastive CSV and return top-N feature IDs sorted by abs(risk_score)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Contrastive scores CSV not found: {csv_path}\n"
            "Run: python -m cli.safety_score --config ... --mode contrastive"
        )
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {"feature_id": int(row["feature_id"]), "risk_score": float(row["risk_score"])}
                )
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: abs(r["risk_score"]), reverse=True)
    return [r["feature_id"] for r in rows[:top_n]]


def _output_path(config, target: str) -> Path:
    safe = sanitize_layer_name(target)
    return deception_run_dir(config, "autointerp", f"{safe}_interpretations.json")


def _safety_scores_dir(config, target: str) -> Path:
    """Locate the contrastive CSV directory for a given layer."""
    return Path(default_safety_scores_dir(config, target=target))


def run_autointerp(config, target: str, args, api_key: str, base_url: str) -> None:
    target_config = config.for_target(target) if hasattr(config, "for_target") else config

    # ── 1. Load top-N feature IDs from contrastive CSV ──────────────────
    safety_dir = _safety_scores_dir(config, target)
    csv_path = safety_dir / f"{sanitize_layer_name(target)}_contrastive.csv"
    print(f"\n[autointerp] Layer: {target}")
    print(f"  Loading top-{args.top_n} features from {csv_path} ...")
    feature_ids = _load_top_features_from_csv(csv_path, args.top_n)
    print(f"  Features selected: {feature_ids[:10]}{'...' if len(feature_ids) > 10 else ''}")

    # ── 2. Load SAE checkpoint ───────────────────────────────────────────
    sae_path = resolve_best_checkpoint(target_config, target=target)
    device = config.get("train.device", "cpu")
    print(f"  Loading SAE from {sae_path} ...")
    sae_model, _ = load_sae_checkpoint(sae_path, device=device)

    # ── 3. Locate activation chunk directory and scenarios ───────────────
    chunk_dir = resolve_activations_dir(target_config, target=target)
    scenarios_path = config.get("deception.scenario_generator.cache_path")

    interp = DeceptionFeatureAutoInterp(
        sae_model=sae_model,
        chunk_dir=chunk_dir,
        scenarios_path=scenarios_path,
        device=device,
    )

    # ── 4. Find top activating contexts ─────────────────────────────────
    print(f"  Encoding activations + finding top-{args.top_k_contexts} contexts per feature ...")
    contexts = interp.find_top_contexts(feature_ids, top_k=args.top_k_contexts)

    # ── 5. Load existing interpretations (resume) ────────────────────────
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

    # ── 6. Interpret via LLM ─────────────────────────────────────────────
    print(f"  Calling LLM ({args.model}) for interpretation ...")
    results = interp.interpret_features(
        feature_contexts=contexts,
        api_key=api_key,
        model=args.model,
        base_url=base_url,
        request_delay=args.request_delay,
        existing_results=existing,
    )

    # ── 7. Save ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ Saved {len(results)} interpretations → {out_path}")

    # ── 8. Print top results ─────────────────────────────────────────────
    print(f"\n  Top interpretations for {target}:")
    for fid in feature_ids[:10]:
        interp_text = results.get(fid, "[missing]")
        print(f"    F{fid:>5}: {interp_text}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)

    api_key = _resolve_api_key(args)
    base_url = args.base_url
    if args.groq:
        base_url = GROQ_BASE_URL
    if base_url is None:
        base_url = "https://api.openai.com/v1"

    targets = [args.layer] if args.layer else resolve_requested_targets(config)
    for target in targets:
        run_autointerp(config, target, args, api_key, base_url)


if __name__ == "__main__":
    main()
