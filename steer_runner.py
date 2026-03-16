"""CLI tool for SAE feature steering, patching, and emotion discovery.

Usage:
    python steer_runner.py discover --config CONFIG --layer LAYER
    python steer_runner.py steer --config CONFIG --layer LAYER --feature-id FID --alpha A --prompt TEXT
    python steer_runner.py patch --config CONFIG --layer LAYER --feature-id FID --clean TEXT --patch TEXT
"""

import argparse
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from config_manager import ConfigManager
from project_layout import default_checkpoint_path, default_feature_mapping_dir
from sae.model import SparseAutoEncoder
from steering.hook import SteeringHook
from steering.patching import activation_patch, batch_feature_patching
from steering.emotion_probe import EmotionProbe


def parse_args():
    parser = argparse.ArgumentParser(description="SAE feature steering and discovery.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run.")

    # --- discover ---
    discover = subparsers.add_parser("discover", help="Find emotion-correlated features.")
    discover.add_argument("--config", required=True, help="Path to config file.")
    discover.add_argument("--layer", default=None, help="Target layer for discovery.")
    discover.add_argument("--events-csv", default=None, help="Path to events CSV (override).")
    discover.add_argument("--dataset-name", default="dair-ai/emotion", help="HF dataset for labels.")
    discover.add_argument("--label-field", default="label", help="Label field in dataset.")
    discover.add_argument("--top-k", type=int, default=15, help="Top features per emotion.")
    discover.add_argument("--output", default=None, help="Output CSV path.")

    # --- steer ---
    steer = subparsers.add_parser("steer", help="Steer model output via SAE features.")
    steer.add_argument("--config", required=True, help="Path to config file.")
    steer.add_argument("--layer", required=True, help="Target layer for steering.")
    steer.add_argument("--feature-id", type=int, required=True, help="Feature index.")
    steer.add_argument("--alpha", type=float, default=5.0, help="Steering strength.")
    steer.add_argument("--mode", default="add", choices=["add", "ablate", "clamp"], help="Steering mode.")
    steer.add_argument("--prompt", required=True, help="Input prompt.")
    steer.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate.")

    # --- patch ---
    patch = subparsers.add_parser("patch", help="Activation patching for causal validation.")
    patch.add_argument("--config", required=True, help="Path to config file.")
    patch.add_argument("--layer", required=True, help="Target layer for patching.")
    patch.add_argument("--feature-id", type=int, default=None, help="Feature index (or auto-select top-k).")
    patch.add_argument("--clean", required=True, help="Clean prompt.")
    patch.add_argument("--patch", required=True, help="Patch prompt.")
    patch.add_argument("--top-k", type=int, default=10, help="Top-k features if no feature-id.")
    patch.add_argument("--metric", default="logit_diff", choices=["logit_diff", "kl_divergence"])

    return parser.parse_args()


def _load_model_and_sae(config, layer=None):
    """Load the TransformerLens model and trained SAE."""
    model_name = config.get("model_name", "gpt2-small")
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")

    model = HookedTransformer.from_pretrained(model_name)
    model.to(device)

    # Load SAE
    mapping_cfg = config.section("feature-mapping") if hasattr(config, "section") else {}
    sae_path = mapping_cfg.get("model_path") or default_checkpoint_path(config)

    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae_config = checkpoint.get("config", {}).get("sae", {})
    input_dim = checkpoint["state_dict"]["b_pre"].shape[0]
    hidden_dim = int(sae_config.get("hidden_dim", 3072))

    sae_model = SparseAutoEncoder(input_dim, hidden_dim)
    sae_model.load_state_dict(checkpoint["state_dict"])
    sae_model.to(device)
    sae_model.eval()

    return model, sae_model


def cmd_discover(args):
    """Find SAE features correlated with emotion labels."""
    config = ConfigManager.from_file(args.config)

    events_csv = args.events_csv
    if not events_csv:
        mapping_dir = default_feature_mapping_dir(config)
        events_csv = str(Path(mapping_dir) / "feature_mapping_events.csv")

    probe = EmotionProbe(events_csv)

    # Load labels from HF dataset
    from datasets import load_dataset
    ds = load_dataset(args.dataset_name, split="train")

    label_names = None
    if hasattr(ds.features.get(args.label_field, None), "names"):
        label_names = ds.features[args.label_field].names

    probe.load_labels_from_dataset(ds, label_field=args.label_field, label_names=label_names)

    summary = probe.summary(top_k_per_emotion=args.top_k)

    print("\n" + "=" * 70)
    print("EMOTION FEATURE DISCOVERY RESULTS")
    print("=" * 70)

    for emotion, features in summary.items():
        if not features:
            continue
        print(f"\n--- {emotion.upper()} ---")
        for f in features[:5]:
            print(
                f"  feature={f['feature_id']:>5d}  "
                f"count={f['count']:>4d}  "
                f"mean_str={f['mean_strength']:.4f}  "
                f"specificity={f['specificity']:.2f}"
            )

    if args.output:
        probe.write_summary_csv(args.output, top_k_per_emotion=args.top_k)
    else:
        default_output = str(Path(events_csv).parent / "emotion_feature_ranking.csv")
        probe.write_summary_csv(default_output, top_k_per_emotion=args.top_k)


def cmd_steer(args):
    """Steer model output by amplifying/suppressing a feature."""
    config = ConfigManager.from_file(args.config)
    model, sae_model = _load_model_and_sae(config)

    print(f"\n--- Clean Output (no steering) ---")
    clean_tokens = model.generate(
        args.prompt, max_new_tokens=args.max_tokens, temperature=0.7
    )
    clean_text = model.to_string(clean_tokens[0])
    print(clean_text)

    print(f"\n--- Steered Output (feature={args.feature_id}, alpha={args.alpha}, mode={args.mode}) ---")
    hook = SteeringHook.apply(
        model=model,
        target_layer=args.layer,
        sae_model=sae_model,
        feature_id=args.feature_id,
        alpha=args.alpha,
        mode=args.mode,
    )
    steered_tokens = model.generate(
        args.prompt, max_new_tokens=args.max_tokens, temperature=0.7
    )
    steered_text = model.to_string(steered_tokens[0])
    print(steered_text)

    model.reset_hooks()


def cmd_patch(args):
    """Run activation patching for causal validation."""
    config = ConfigManager.from_file(args.config)
    model, sae_model = _load_model_and_sae(config)

    if args.feature_id is not None:
        result = activation_patch(
            model=model,
            sae_model=sae_model,
            clean_text=args.clean,
            patch_text=args.patch,
            target_layer=args.layer,
            feature_id=args.feature_id,
            metric=args.metric,
        )
        print(f"\n--- Activation Patching Result ---")
        print(f"  Feature:       {result['feature_id']}")
        print(f"  Layer:         {result['target_layer']}")
        print(f"  Metric:        {result['metric']}")
        print(f"  Effect:        {result['effect']:.6f}")
        print(f"  Clean f_mean:  {result['clean_feature_mean']:.4f}")
        print(f"  Patch f_mean:  {result['patch_feature_mean']:.4f}")
    else:
        results = batch_feature_patching(
            model=model,
            sae_model=sae_model,
            clean_text=args.clean,
            patch_text=args.patch,
            target_layer=args.layer,
            top_k=args.top_k,
            metric=args.metric,
        )
        print(f"\n--- Batch Patching Results (top {len(results)} features) ---")
        for r in results:
            print(
                f"  feature={r['feature_id']:>5d}  "
                f"effect={r['effect']:.6f}  "
                f"clean_f={r['clean_feature_mean']:.4f}  "
                f"patch_f={r['patch_feature_mean']:.4f}"
            )


def main():
    args = parse_args()
    if args.command is None:
        print("Please specify a command: discover, steer, or patch")
        print("Run with --help for details.")
        sys.exit(1)

    commands = {
        "discover": cmd_discover,
        "steer": cmd_steer,
        "patch": cmd_patch,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
