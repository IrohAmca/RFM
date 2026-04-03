"""CLI: SAE feature steering, deception direction steering, and patching.

Commands
--------
  steer      — amplify/suppress/ablate SAE features during generation
  deception  — suppress deception direction from trained monitor bundle
  patch      — activation patching for causal feature validation
  discover   — find emotion/label-correlated SAE features

Examples
--------
  # Standard SAE feature steering
  python -m cli.steer steer --config ... --layer blocks.20.hook_resid_post \\
      --feature-id 4192 --alpha -10 --mode add --prompt "Tell me about X"

  # Multi-feature steering (comma-separated id:alpha pairs)
  python -m cli.steer steer --config ... --layer blocks.20.hook_resid_post \\
      --features "4192:-10,2022:-8" --prompt "Tell me about X"

  # Suppress deception direction (uses monitor_bundle.pt directions)
  python -m cli.steer deception --config configs/models/qwen3-0.6B.deception.json \\
      --prompt "What is the capital of France?" --alpha -5.0

  # Activation patching
  python -m cli.steer patch --config ... --layer blocks.20.hook_resid_post \\
      --clean "The sky is blue" --patch "The sky is red"
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.deception.utils import deception_run_dir
from rfm.layout import default_feature_mapping_dir, resolve_best_checkpoint, resolve_requested_targets
from rfm.sae.model import load_sae_checkpoint
from rfm.steering.hook import HFSteeringHook, resolve_hf_target_module
from rfm.steering.patching import activation_patch, batch_feature_patching


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE feature steering, deception suppression, and causal patching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run.")

    # ── steer ──────────────────────────────────────────────────────────
    steer = subparsers.add_parser("steer", help="Steer model output via SAE feature directions.")
    steer.add_argument("--config", required=True)
    steer.add_argument("--layer", required=True, help="Layer to hook (e.g. blocks.20.hook_resid_post).")
    steer.add_argument("--feature-id", type=int, default=None, help="Single feature to steer.")
    steer.add_argument("--alpha", type=float, default=5.0, help="Steering strength (negative = suppress).")
    steer.add_argument("--mode", default="add", choices=["add", "ablate", "clamp"],
                       help="Steering mode: add=inject direction, ablate=remove, clamp=fix value.")
    steer.add_argument("--features", default=None,
                       help="Multi-feature: comma-separated 'id:alpha' pairs, e.g. '4192:-10,2022:-8'.")
    steer.add_argument("--prompt", required=True)
    steer.add_argument("--system-prompt", default=None, help="System prompt for chat models.")
    steer.add_argument("--max-tokens", type=int, default=100)
    steer.add_argument("--temperature", type=float, default=0.7)
    steer.add_argument("--no-chat-template", action="store_true",
                       help="Bypass chat template (raw prompt only).")

    # ── deception ──────────────────────────────────────────────────────
    deception = subparsers.add_parser(
        "deception",
        help="Suppress/amplify deception directions from the trained monitor bundle.",
    )
    deception.add_argument("--config", required=True)
    deception.add_argument("--prompt", required=True)
    deception.add_argument("--system-prompt", default=None, help="System prompt for chat models.")
    deception.add_argument(
        "--alpha", type=float, default=-5.0,
        help="Steering alpha per layer. Negative = suppress deception (default: -5.0).",
    )
    deception.add_argument(
        "--layers", default=None,
        help="Comma-separated layers to steer. Defaults to all layers in monitor bundle.",
    )
    deception.add_argument("--max-tokens", type=int, default=150)
    deception.add_argument("--temperature", type=float, default=0.7)
    deception.add_argument("--show-scores", action="store_true",
                           help="Run deception monitor on both outputs and show comparison scores.")

    # ── patch ──────────────────────────────────────────────────────────
    patch = subparsers.add_parser("patch", help="Activation patching for causal validation.")
    patch.add_argument("--config", required=True)
    patch.add_argument("--layer", required=True)
    patch.add_argument("--feature-id", type=int, default=None)
    patch.add_argument("--clean", required=True)
    patch.add_argument("--patch", required=True)
    patch.add_argument("--top-k", type=int, default=10)
    patch.add_argument("--metric", default="logit_diff", choices=["logit_diff", "kl_divergence"])

    # ── discover ───────────────────────────────────────────────────────
    discover = subparsers.add_parser("discover", help="Find label-correlated SAE features.")
    discover.add_argument("--config", required=True)
    discover.add_argument("--layer", default=None)
    discover.add_argument("--events-csv", default=None)
    discover.add_argument("--dataset-name", default="dair-ai/emotion")
    discover.add_argument("--label-field", default="label")
    discover.add_argument("--top-k", type=int, default=15)
    discover.add_argument("--output", default=None)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.get("model_name", "gpt2")
    device = config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = config.get("extraction.dtype", "float32")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str, torch.float32)

    print(f"[steer] Loading {model_name} → {device} ({dtype_str})")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, device


def _load_sae_for_target(config, target: str, device: str):
    target_cfg = config.for_target(target) if hasattr(config, "for_target") else config
    sae_path = resolve_best_checkpoint(target_cfg, target=target)
    sae, _ = load_sae_checkpoint(sae_path, device=device)
    sae.eval()
    return sae


def _format_prompt(tokenizer, prompt: str, system_prompt: str | None, use_chat: bool) -> str:
    """Format prompt using chat template if available, with plain-text fallback."""
    if not use_chat:
        return prompt
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def _generate(model, tokenizer, prompt_text: str, device: str, max_tokens: int, temperature: float) -> str:
    ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def _print_comparison(clean_out: str, steered_out: str, label: str = "") -> None:
    width = 72
    print(f"\n{'─' * width}")
    print("  CLEAN OUTPUT")
    print(f"{'─' * width}")
    print(clean_out.strip())
    print(f"\n{'─' * width}")
    print(f"  STEERED OUTPUT  {label}")
    print(f"{'─' * width}")
    print(steered_out.strip())
    print(f"{'─' * width}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_steer(args):
    config = ConfigManager.from_file(args.config)
    model, tokenizer, device = _load_model_and_tokenizer(config)
    sae = _load_sae_for_target(config, args.layer, device)

    use_chat = not getattr(args, "no_chat_template", False)
    prompt_text = _format_prompt(tokenizer, args.prompt, getattr(args, "system_prompt", None), use_chat)

    # ── Build feature config list ─────────────────────────────────────
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

    label = (
        args.features if args.features
        else f"F{args.feature_id} α={args.alpha} mode={args.mode}"
    )

    # ── Generate clean ────────────────────────────────────────────────
    clean_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)

    # ── Apply steering hooks + generate ──────────────────────────────
    handles = []
    for fc in feature_configs:
        h = HFSteeringHook.apply(
            hf_model=model, target_layer=args.layer, sae_model=sae,
            feature_id=fc["feature_id"], alpha=fc["alpha"], mode=fc["mode"],
        )
        handles.append(h)

    steered_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)
    for h in handles:
        h.remove()

    _print_comparison(clean_out, steered_out, label=label)


def cmd_deception(args):
    """Steer using the deception direction vectors from the monitor bundle."""
    config = ConfigManager.from_file(args.config)
    model, tokenizer, device = _load_model_and_tokenizer(config)

    # ── Load direction vectors from monitor bundle ────────────────────
    dec_dir = deception_run_dir(config)
    directions_path = dec_dir / "directions" / "directions.pt"
    if not directions_path.exists():
        print(
            f"[steer] Direction file not found: {directions_path}\n"
            "Run: python -m cli.deception_cycle --phase direction"
        )
        sys.exit(1)

    all_directions = torch.load(directions_path, map_location="cpu", weights_only=False)

    target_layers = (
        [l.strip() for l in args.layers.split(",")]
        if args.layers
        else list(all_directions.keys())
    )
    print(f"[steer] Deception steering on {len(target_layers)} layer(s), α={args.alpha}")
    for layer in target_layers:
        if layer not in all_directions:
            print(f"  Warning: {layer} not in directions file; skipping.")

    use_chat = True
    prompt_text = _format_prompt(tokenizer, args.prompt, args.system_prompt, use_chat)

    # ── Generate clean ────────────────────────────────────────────────
    clean_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)

    # ── Attach direction-based steering hooks ─────────────────────────
    handles = []
    hooked_layers = []
    for layer in target_layers:
        if layer not in all_directions:
            continue
        direction_payload = all_directions[layer]
        direction = (
            direction_payload["direction"]
            if isinstance(direction_payload, dict)
            else direction_payload.direction
        )
        direction = direction.detach().to(device, dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-12)

        try:
            target_module = resolve_hf_target_module(model, layer)
        except ValueError as exc:
            print(f"  Warning: Could not resolve {layer}: {exc}")
            continue

        def _make_hook(d, a):
            def hook_fn(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                act = output[0] if is_tuple else output
                steered = act + a * d.to(act.device, act.dtype)
                return (steered,) + output[1:] if is_tuple else steered
            return hook_fn

        handle = target_module.register_forward_hook(_make_hook(direction, args.alpha))
        handles.append(handle)
        hooked_layers.append(layer)

    print(f"  Hooked layers: {hooked_layers}")
    steered_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)

    for h in handles:
        h.remove()

    _print_comparison(
        clean_out, steered_out,
        label=f"deception direction α={args.alpha} ({len(hooked_layers)} layers)",
    )

    # ── Optional: show deception monitor scores ───────────────────────
    if getattr(args, "show_scores", False):
        try:
            from rfm.deception import DeceptionDirectionFinder, DeceptionMonitor, DirectionResult
            directions_dict = {}
            thresholds = {}
            for layer, payload in all_directions.items():
                if layer not in hooked_layers:
                    continue
                if isinstance(payload, dict):
                    dr = DirectionResult.from_dict(payload)
                else:
                    dr = payload
                directions_dict[layer] = dr
                thresholds[layer] = float(dr.threshold)

            monitor = DeceptionMonitor(
                directions=directions_dict,
                thresholds=thresholds,
            )
            for label, text in [("clean", clean_out), ("steered", steered_out)]:
                score = monitor.score_replay(
                    model=model, tokenizer=tokenizer,
                    prompt=args.prompt, response=text,
                    system_prompt=args.system_prompt,
                )
                print(
                    f"  [{label}] deception_prob={score.deception_probability:.3f}  "
                    f"alert={score.alert}  per_layer={score.per_layer_scores}"
                )
        except Exception as exc:
            print(f"  Warning: Could not compute monitor scores: {exc}")


def cmd_patch(args):
    config = ConfigManager.from_file(args.config).for_target(args.layer)
    model, tokenizer, device = _load_model_and_tokenizer(config)
    sae = _load_sae_for_target(config, args.layer, device)

    if args.feature_id is not None:
        result = activation_patch(
            model=model, sae_model=sae, clean_text=args.clean,
            patch_text=args.patch, target_layer=args.layer,
            feature_id=args.feature_id, metric=args.metric, tokenizer=tokenizer,
        )
        print(f"\n[patch] Feature {result['feature_id']}  Effect: {result['effect']:.6f}")
    else:
        results = batch_feature_patching(
            model=model, sae_model=sae, clean_text=args.clean,
            patch_text=args.patch, target_layer=args.layer,
            top_k=args.top_k, metric=args.metric, tokenizer=tokenizer,
        )
        print(f"\n[patch] Top {len(results)} features by causal effect:")
        print(f"  {'Feature':>8}  {'Effect':>12}")
        print(f"  {'─' * 25}")
        for r in results:
            print(f"  {r['feature_id']:>8d}  {r['effect']:>12.6f}")


def cmd_discover(args):
    from rfm.steering.emotion_probe import EmotionProbe

    base_config = ConfigManager.from_file(args.config)
    target = args.layer
    if not target:
        targets = resolve_requested_targets(base_config)
        target = targets[0] if targets else "blocks.0.hook_resid_post"
    config = base_config.for_target(target)

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
    print("EMOTION / LABEL FEATURE DISCOVERY")
    print("=" * 70)
    for emotion, features in summary.items():
        if not features:
            continue
        print(f"\n--- {emotion.upper()} ---")
        for f in features[:5]:
            print(
                f"  feature={f['feature_id']:>5d}  count={f['count']:>4d}  "
                f"mean_str={f['mean_strength']:.4f}  specificity={f['specificity']:.2f}"
            )

    output = args.output or str(Path(events_csv).parent / "emotion_feature_ranking.csv")
    probe.write_summary_csv(output, top_k_per_emotion=args.top_k)
    print(f"\n[discover] Written to {output}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.command is None:
        print("Commands: steer, deception, patch, discover. Use --help for details.")
        sys.exit(1)

    dispatch = {
        "steer": cmd_steer,
        "deception": cmd_deception,
        "patch": cmd_patch,
        "discover": cmd_discover,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
