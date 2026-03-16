"""CLI: SAE feature steering, patching, and emotion discovery."""

import argparse
import sys
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.layout import default_checkpoint_path, default_feature_mapping_dir
from rfm.sae.model import SparseAutoEncoder
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
    steer.add_argument("--feature-id", type=int, default=None, help="Single feature to steer.")
    steer.add_argument("--alpha", type=float, default=5.0)
    steer.add_argument("--mode", default="add", choices=["add", "ablate", "clamp"])
    steer.add_argument("--features", default=None,
                       help="Multi-feature steering: comma-separated 'id:alpha' pairs, e.g. '3129:-15,3583:15'."
                            " If set, --feature-id and --alpha are ignored.")
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


def _resolve_targets(config):
    raw = config.get("extraction.target")
    if isinstance(raw, list):
        return raw
    if raw:
        return [raw]
    return ["blocks.0.hook_resid_post"]


def _load_model_and_sae(config, target):
    model_name = config.get("model_name", "gpt2-small")
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Always load raw HuggingFace for steering to avoid TransformerLens type issues
    print(f"[steer] Loading {model_name} via HuggingFace AutoModelForCausalLM...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mapping_cfg = config.section("feature-mapping") if hasattr(config, "section") else {}
    sae_path = mapping_cfg.get("model_path")
    if not sae_path:
        default_path = Path(default_checkpoint_path(config, target=target))
        if default_path.exists():
            sae_path = str(default_path)
        else:
            sweep_val = config.get("sae.sparsity_weight", 0.005)
            sweep_path = default_path.parent / f"sae_lambda_{sweep_val}.pt"
            if sweep_path.exists():
                sae_path = str(sweep_path)
            else:
                available = list(default_path.parent.glob("sae_lambda_*.pt"))
                if available:
                    sae_path = str(available[0])
                else:
                    sae_path = str(default_path)

    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae_config = checkpoint.get("config", {}).get("sae", {})
    input_dim = checkpoint["state_dict"]["b_pre"].shape[0]
    hidden_dim = int(sae_config.get("hidden_dim", 3072))

    sae_model = SparseAutoEncoder(input_dim, hidden_dim)
    sae_model.load_state_dict(checkpoint["state_dict"])
    sae_model.to(device)
    sae_model.eval()

    return hf_model, tokenizer, sae_model


def cmd_discover(args):
    config = ConfigManager.from_file(args.config)
    
    target = args.layer
    if not target:
        targets = _resolve_targets(config)
        target = targets[0] if targets else "blocks.0.hook_resid_post"
        
    events_csv = args.events_csv
    if not events_csv:
        mapping_dir = default_feature_mapping_dir(config, target=target)
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
    hf_model, tokenizer, sae_model = _load_model_and_sae(config, args.layer)
    from rfm.steering.hook import HFSteeringHook

    device = sae_model.W_dec.device
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print("\n--- Clean Output ---")
    with torch.no_grad():
        clean_out = hf_model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=0.7, do_sample=True)
    print(tokenizer.decode(clean_out[0], skip_special_tokens=True))

    # Build feature list: either from --features or single --feature-id/--alpha
    if args.features:
        feature_configs = []
        for item in args.features.split(","):
            parts = item.strip().split(":")
            fid = int(parts[0])
            falpha = float(parts[1]) if len(parts) > 1 else args.alpha
            feature_configs.append({"feature_id": fid, "alpha": falpha, "mode": args.mode})
    else:
        if args.feature_id is None:
            print("Error: provide --feature-id or --features")
            sys.exit(1)
        feature_configs = [{"feature_id": args.feature_id, "alpha": args.alpha, "mode": args.mode}]

    label = args.features if args.features else f"feature={args.feature_id}, alpha={args.alpha}, mode={args.mode}"
    print(f"\n--- Steered ({label}) ---")

    handles = []
    for fc in feature_configs:
        h = HFSteeringHook.apply(
            hf_model=hf_model, target_layer=args.layer, sae_model=sae_model,
            feature_id=fc["feature_id"], alpha=fc["alpha"], mode=fc["mode"]
        )
        handles.append(h)

    with torch.no_grad():
        steered_out = hf_model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=0.7, do_sample=True)
    print(tokenizer.decode(steered_out[0], skip_special_tokens=True))

    for h in handles:
        h.remove()


def cmd_patch(args):
    config = ConfigManager.from_file(args.config)
    model, sae_model = _load_model_and_sae(config, args.layer)

    if args.feature_id is not None:
        result = activation_patch(model=model, sae_model=sae_model, clean_text=args.clean,
                                  patch_text=args.patch, target_layer=args.layer,
                                  feature_id=args.feature_id, metric=args.metric)
        print("\n--- Patching Result ---")
        print(f"  Feature: {result['feature_id']}  Effect: {result['effect']:.6f}")
    else:
        results = batch_feature_patching(model=model, sae_model=sae_model, clean_text=args.clean,
                                         patch_text=args.patch, target_layer=args.layer,
                                         top_k=args.top_k, metric=args.metric)
        print(f"\n--- Batch Patching ({len(results)} features) ---")
        for r in results:
            print(f"  feature={r['feature_id']:>5d}  effect={r['effect']:.6f}")


def main():
    if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    args = parse_args()
    if args.command is None:
        print("Commands: discover, steer, patch. Use --help.")
        sys.exit(1)
    {"discover": cmd_discover, "steer": cmd_steer, "patch": cmd_patch}[args.command](args)


if __name__ == "__main__":
    main()
