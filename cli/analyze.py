"""
CLI for analyzing SAE features via auto-interpretation and clustering.
"""

import argparse
import json
from pathlib import Path

from rfm.config import ConfigManager
from rfm.analysis.autointerp import FeatureAutoInterp, GROQ_BASE_URL
from rfm.analysis.clustering import FeatureClustering


def _resolve_targets(config: ConfigManager, layer: str | None = None) -> list[str]:
    if layer:
        return [layer]
    raw = config.get("extraction.target")
    if isinstance(raw, list):
        return raw
    if raw:
        return [raw]
    return ["blocks.0.hook_resid_post"]


def _resolve_events_csv(mapping_dir: Path, slug: str) -> Path | None:
    candidates = [
        mapping_dir / "feature_mapping_events.csv",
        mapping_dir / f"{slug}_feature_events.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_summary_csv(mapping_dir: Path, slug: str) -> Path | None:
    candidates = [
        mapping_dir / "feature_mapping_feature_summary.csv",
        mapping_dir / f"{slug}_feature_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _summary_feature_ids(summary_csv: Path, top_n: int) -> list[int]:
    import pandas as pd

    summary_df = pd.read_csv(summary_csv)
    count_col = "num_events" if "num_events" in summary_df.columns else "count_active" if "count_active" in summary_df.columns else None
    if count_col is None or "feature_id" not in summary_df.columns:
        raise ValueError("summary CSV missing feature_id or count column")
    return summary_df.sort_values(by=count_col, ascending=False).head(top_n)["feature_id"].tolist()


def compute_autointerp(
    config: ConfigManager,
    top_n: int = 50,
    model: str = "gpt-4o-mini",
    api_key: str = None,
    layer: str = None,
    base_url: str = None,
    request_delay: float = 0.0,
    resume: bool = True,
):
    """Run auto-interpretation for features."""
    from rfm.layout import model_slug, default_feature_mapping_dir

    slug = model_slug(config)
    targets = _resolve_targets(config, layer=layer)
    missing_targets = []

    for target in targets:
        mapping_dir = Path(default_feature_mapping_dir(config, target=target))
        events_csv = _resolve_events_csv(mapping_dir, slug)
        output_path = mapping_dir / f"{slug}_autointerp_results.json"

        if events_csv is None:
            missing_targets.append((target, mapping_dir))
            continue

        print(f"[autointerp] Target layer: {target}")
        print(f"Loading events from {events_csv}...")

        # Use custom system prompt from config if defined (e.g. safety-focused prompt)
        system_prompt = config.get("autointerp.system_prompt", None)

        interp = FeatureAutoInterp(
            events_csv,
            backend="openai",
            api_key=api_key,
            base_url=base_url,
            request_delay=request_delay,
            system_prompt=system_prompt,
        )

        top_k_contexts = config.get("autointerp.contexts_per_feature", 15)

        try:
            summary_csv = _resolve_summary_csv(mapping_dir, slug)
            if summary_csv is not None:
                print(f"Selecting top {top_n} most active features from {summary_csv}...")
                top_features = _summary_feature_ids(summary_csv, top_n)
            else:
                top_features = interp.df["feature_id"].unique()[:top_n].tolist()
        except Exception as e:
            print(f"Error selecting top features: {e}. Falling back to first found features.")
            top_features = interp.df["feature_id"].unique()[:top_n].tolist()

        # Load previously saved results for resume support
        existing_results = {}
        if resume and output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Keys in the JSON are string feature IDs; convert to int for consistency
                existing_results = {int(k): v for k, v in raw.items()}
                already_done = len([fid for fid in top_features if fid in existing_results])
                if already_done:
                    print(f"  Resuming: {already_done}/{len(top_features)} features already interpreted, skipping them.")
            except Exception as e:
                print(f"  Warning: could not load existing results from {output_path}: {e}")

        print(f"Running LLM interpretation on {len(top_features)} features...")
        results = interp.interpret_features(
            top_features,
            top_k=top_k_contexts,
            model=model,
            existing_results=existing_results,
        )

        interp.save_interpretations(results, output_path)
        print(f"Auto-interpretation complete for {target}. Saved to {output_path}")

    if missing_targets:
        if len(missing_targets) == len(targets):
            target, mapping_dir = missing_targets[0]
            print(
                f"Error: Could not find events CSV under {mapping_dir} "
                f"for target {target}. Have you run feature mapping?"
            )
            return
        for target, mapping_dir in missing_targets:
            print(f"Warning: Skipping {target}; no events CSV found under {mapping_dir}")


def compute_clusters(config: ConfigManager, clusters: int = 20, layer: str = None):
    """Run hierarchical clustering on SAE features."""
    from rfm.layout import default_feature_mapping_dir, model_slug, resolve_best_checkpoint

    slug = model_slug(config)
    for target in _resolve_targets(config, layer=layer):
        checkpoint_path = Path(resolve_best_checkpoint(config, target=target))
        mapping_dir = Path(default_feature_mapping_dir(config, target=target))

        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}. Have you trained the SAE?")
            continue

        print(f"[cluster] Target layer: {target}")
        print(f"Loading SAE decoder from {checkpoint_path}...")
        fc = FeatureClustering(checkpoint_path)

        print(f"Clustering into {clusters} groups via cosine distance...")
        cluster_dict, leaves_order = fc.cluster_features(n_clusters=clusters, metric="cosine", method="average")

        output_path = mapping_dir / f"{slug}_feature_clusters.json"

        import json
        with open(output_path, "w", encoding="utf-8") as f:
             json.dump({
                 "metadata": {"n_clusters": clusters, "metric": "cosine", "target_layer": target, "leaves_order": leaves_order},
                 "clusters": cluster_dict
             }, f, indent=2)

        print(f"Saved cluster assignments to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="RFM Feature Analysis Tool")
    parser.add_argument("command", choices=["autointerp", "cluster"], help="Command to run")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")

    # Autointerp args
    parser.add_argument("--top-n", type=int, default=50, help="Number of top features to interpret (autointerp only)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use (autointerp only)")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API key (autointerp only)")
    parser.add_argument("--layer", type=str, default=None, help="Specific target layer to analyze. Defaults to all configured targets.")
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for an OpenAI-compatible API (e.g. https://api.groq.com/openai/v1). "
             "Overrides the default OpenAI endpoint."
    )
    parser.add_argument(
        "--groq", action="store_true",
        help="Shortcut: use Groq's OpenAI-compatible endpoint. "
             "Set GROQ_API_KEY in the environment (or use --api-key)."
    )
    parser.add_argument(
        "--request-delay", type=float, default=0.0,
        help="Seconds to wait between successive LLM requests. "
             "Increase this (e.g. 11) when hitting TPM rate limits on Groq free tier."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing partial results and re-interpret all features from scratch."
    )

    # Cluster args
    parser.add_argument("--clusters", type=int, default=20, help="Number of clusters to form (cluster only)")

    args = parser.parse_args()

    config = ConfigManager.from_file(args.config)

    if args.command == "autointerp":
        base_url = args.base_url
        if args.groq:
            base_url = GROQ_BASE_URL
        compute_autointerp(
            config,
            top_n=args.top_n,
            model=args.model,
            api_key=args.api_key,
            layer=args.layer,
            base_url=base_url,
            request_delay=args.request_delay,
            resume=not args.no_resume,
        )
    elif args.command == "cluster":
        compute_clusters(config, clusters=args.clusters, layer=args.layer)

if __name__ == "__main__":
    main()
