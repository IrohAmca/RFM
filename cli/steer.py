"""CLI: SAE feature steering, patching, and emotion discovery."""

import argparse
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from rfm.config import ConfigManager
from rfm.layout import default_checkpoint_path, default_feature_mapping_dir
from rfm.sae.model import SparseAutoEncoder
from rfm.steering.hook import SteeringHook
from rfm.steering.patching import activation_patch, batch_feature_patching
from rfm.steering.emotion_probe import EmotionProbe


def parse_args():
    parser = argparse.ArgumentParser(description="SAE feature steering and discovery.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run.")

    discover = subparsers.add_parser("discover", help="Find emotion-correlated features.")
    discover.add_argument("--config", required=True)
    discover.add_argument("--layer", default=None)
    discover.add_argument("--events-csv", default=None)
    discover.add_argument("--dataset-name", default="dair-ai/emotion")
    discover.add_argument("--label-field", default="label")
    discover.add_argument("--top-k", type=int, default=15)
    discover.add_argument("--output", default=None)

    steer = subparsers.add_parser("steer", help="Steer model output via SAE features.")
    steer.add_argument("--config", required=True)
    steer.add_argument("--layer", required=True)
    steer.add_argument("--feature-id", type=int, required=True)
    steer.add_argument("--alpha", type=float, default=5.0)
    steer.add_argument("--mode", default="add", choices=["add", "ablate", "clamp"])
    steer.add_argument("--prompt", required=True)
    steer.add_argument("--max-tokens", type=int, default=50)

    patch = subparsers.add_parser("patch", help="Activation patching for causal validation.")
    patch.add_argument("--config", required=True)
    patch.add_argument("--layer", required=True)
    patch.add_argument("--feature-id", type=int, default=None)
    patch.add_argument("--clean", required=True)
    patch.add_argument("--patch", required=True)
    patch.add_argument("--top-k", type=int, default=10)
    patch.add_argument("--metric", default="logit_diff", choices=["logit_diff", "kl_divergence"])

    return parser.parse_args()


def _load_model_and_sae(config):
    model_name = config.get("model_name", "gpt2-small")
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")

    model = HookedTransformer.from_pretrained(model_name)
    model.to(device)

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
    config = ConfigManager.from_file(args.config)
    events_csv = args.events_csv
    if not events_csv:
        mapping_dir = default_feature_mapping_dir(config)
        events_csv = str(Path(mapping_dir) / "feature_mapping_events.csv")

    probe = EmotionProbe(events_csv)

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
            print(f"  feature={f['feature_id']:>5d}  count={f['count']:>4d}  mean_str={f['mean_strength']:.4f}  specificity={f['specificity']:.2f}")

    output = args.output or str(Path(events_csv).parent / "emotion_feature_ranking.csv")
    probe.write_summary_csv(output, top_k_per_emotion=args.top_k)


def cmd_steer(args):
    config = ConfigManager.from_file(args.config)
    model, sae_model = _load_model_and_sae(config)

    print(f"\n--- Clean Output ---")
    clean_tokens = model.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=0.7)
    print(model.to_string(clean_tokens[0]))

    print(f"\n--- Steered (feature={args.feature_id}, alpha={args.alpha}, mode={args.mode}) ---")
    SteeringHook.apply(model=model, target_layer=args.layer, sae_model=sae_model,
                       feature_id=args.feature_id, alpha=args.alpha, mode=args.mode)
    steered_tokens = model.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=0.7)
    print(model.to_string(steered_tokens[0]))
    model.reset_hooks()


def cmd_patch(args):
    config = ConfigManager.from_file(args.config)
    model, sae_model = _load_model_and_sae(config)

    if args.feature_id is not None:
        result = activation_patch(model=model, sae_model=sae_model, clean_text=args.clean,
                                  patch_text=args.patch, target_layer=args.layer,
                                  feature_id=args.feature_id, metric=args.metric)
        print(f"\n--- Patching Result ---")
        print(f"  Feature: {result['feature_id']}  Effect: {result['effect']:.6f}")
    else:
        results = batch_feature_patching(model=model, sae_model=sae_model, clean_text=args.clean,
                                         patch_text=args.patch, target_layer=args.layer,
                                         top_k=args.top_k, metric=args.metric)
        print(f"\n--- Batch Patching ({len(results)} features) ---")
        for r in results:
            print(f"  feature={r['feature_id']:>5d}  effect={r['effect']:.6f}")


def main():
    args = parse_args()
    if args.command is None:
        print("Commands: discover, steer, patch. Use --help.")
        sys.exit(1)
    {"discover": cmd_discover, "steer": cmd_steer, "patch": cmd_patch}[args.command](args)


if __name__ == "__main__":
    main()
