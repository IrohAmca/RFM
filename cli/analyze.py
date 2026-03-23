"""
CLI for analyzing SAE features via auto-interpretation and clustering.
"""

import argparse
from pathlib import Path

from rfm.config import ConfigManager
from rfm.analysis.autointerp import FeatureAutoInterp
from rfm.analysis.clustering import FeatureClustering


def compute_autointerp(config: ConfigManager, top_n: int = 50, model: str = "gpt-4o-mini", api_key: str = None):
    """Run auto-interpretation for features."""
    from rfm.layout import model_slug, default_feature_mapping_dir
    
    slug = model_slug(config)
    mapping_dir = Path(default_feature_mapping_dir(config))
    events_csv = mapping_dir / f"{slug}_feature_events.csv"
    
    # We output to the same mapping dir
    output_path = mapping_dir / f"{slug}_autointerp_results.json"
    
    if not events_csv.exists():
        print(f"Error: Could not find events CSV at {events_csv}. Have you run feature mapping?")
        return
        
    print(f"Loading events from {events_csv}...")
    interp = FeatureAutoInterp(events_csv, backend="openai", api_key=api_key)
    
    # Use config top_k if available or default
    top_k_contexts = config.get("autointerp.contexts_per_feature", 15)
    
    # Normally we would read the summary CSV to sort by event count and take the top_n most active features
    try:
        import pandas as pd
        summary_csv = mapping_dir / f"{slug}_feature_summary.csv"
        if summary_csv.exists():
            print(f"Selecting top {top_n} most active features...")
            summ_df = pd.read_csv(summary_csv)
            # sort by num_events
            top_features = summ_df.sort_values(by="num_events", ascending=False).head(top_n)["feature_id"].tolist()
        else:
             # Just pick first N if summary not found
             top_features = interp.df["feature_id"].unique()[:top_n].tolist()
    except Exception as e:
        print(f"Error selecting top features: {e}. Falling back to first found features.")
        top_features = interp.df["feature_id"].unique()[:top_n].tolist()
        
    print(f"Running LLM interpretation on {len(top_features)} features...")
    results = interp.interpret_features(top_features, top_k=top_k_contexts, model=model)
    
    interp.save_interpretations(results, output_path)
    print("Auto-interpretation complete.")


def compute_clusters(config: ConfigManager, clusters: int = 20):
    """Run hierarchical clustering on SAE features."""
    from rfm.layout import default_checkpoint_path, model_slug, default_feature_mapping_dir
    
    checkpoint_path = Path(default_checkpoint_path(config))
    mapping_dir = Path(default_feature_mapping_dir(config))
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}. Have you trained the SAE?")
        return
        
    print(f"Loading SAE decoder from {checkpoint_path}...")
    fc = FeatureClustering(checkpoint_path)
    
    print(f"Clustering into {clusters} groups via cosine distance...")
    cluster_dict, leaves_order = fc.cluster_features(n_clusters=clusters, metric="cosine", method="average")
    
    # Save the cluster dictionary
    slug = model_slug(config)
    output_path = mapping_dir / f"{slug}_feature_clusters.json"
    
    import json
    with open(output_path, "w", encoding="utf-8") as f:
         json.dump({
             "metadata": {"n_clusters": clusters, "metric": "cosine"},
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
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (autointerp only)")
    
    # Cluster args
    parser.add_argument("--clusters", type=int, default=20, help="Number of clusters to form (cluster only)")
    
    args = parser.add_argument_args() if hasattr(parser, "add_argument_args") else parser.parse_args()
    
    config = ConfigManager.from_file(args.config)
    
    if args.command == "autointerp":
        compute_autointerp(config, top_n=args.top_n, model=args.model, api_key=args.api_key)
    elif args.command == "cluster":
        compute_clusters(config, clusters=args.clusters)

if __name__ == "__main__":
    main()
