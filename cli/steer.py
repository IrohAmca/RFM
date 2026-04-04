"""CLI: feature steering, motif intervention, and causal patching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.deception.utils import deception_run_dir
from rfm.layout import default_feature_mapping_dir, resolve_best_checkpoint, resolve_requested_targets
from rfm.patterns import (
    ContrastAxisSpec,
    append_pattern_analysis_rows,
    axis_monitor_from_bundle,
    load_model_and_tokenizer as load_runtime_model_and_tokenizer,
    load_pattern_bundle,
    motif_feature_configs,
    motif_members,
    register_feature_interventions,
)
from rfm.sae.model import load_sae_checkpoint
from rfm.steering.hook import HFSteeringHook, resolve_hf_target_module
from rfm.steering.patching import activation_patch, batch_feature_patching


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE feature steering, motif intervention, deception suppression, and causal patching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run.")

    steer = subparsers.add_parser("steer", help="Steer model output via SAE feature directions.")
    steer.add_argument("--config", required=True)
    steer.add_argument("--layer", required=True, help="Layer to hook (for example blocks.20.hook_resid_post).")
    steer.add_argument("--feature-id", type=int, default=None, help="Single feature to steer.")
    steer.add_argument("--alpha", type=float, default=5.0, help="Steering strength. Negative values suppress.")
    steer.add_argument("--mode", default="add", choices=["add", "ablate", "clamp"])
    steer.add_argument("--features", default=None, help="Comma-separated id:alpha pairs, for example 4192:-10,2022:-8.")
    steer.add_argument("--prompt", required=True)
    steer.add_argument("--system-prompt", default=None, help="System prompt for chat models.")
    steer.add_argument("--max-tokens", type=int, default=100)
    steer.add_argument("--temperature", type=float, default=0.7)
    steer.add_argument("--no-chat-template", action="store_true", help="Bypass chat template.")

    deception = subparsers.add_parser("deception", help="Suppress or amplify deception directions from the monitor bundle.")
    deception.add_argument("--config", required=True)
    deception.add_argument("--prompt", required=True)
    deception.add_argument("--system-prompt", default=None, help="System prompt for chat models.")
    deception.add_argument("--alpha", type=float, default=-5.0, help="Per-layer steering alpha. Negative suppresses deception.")
    deception.add_argument("--layers", default=None, help="Comma-separated layers to steer. Defaults to all monitored layers.")
    deception.add_argument("--max-tokens", type=int, default=150)
    deception.add_argument("--temperature", type=float, default=0.7)
    deception.add_argument("--show-scores", action="store_true", help="Replay deception monitor scores on both outputs.")

    patch = subparsers.add_parser("patch", help="Activation patching for causal validation.")
    patch.add_argument("--config", required=True)
    patch.add_argument("--layer", required=True)
    patch.add_argument("--feature-id", type=int, default=None)
    patch.add_argument("--clean", required=True)
    patch.add_argument("--patch", required=True)
    patch.add_argument("--top-k", type=int, default=10)
    patch.add_argument("--metric", default="logit_diff", choices=["logit_diff", "kl_divergence"])

    motif = subparsers.add_parser("motif", help="Ablate or amplify a promoted motif from the canonical pattern bundle.")
    motif.add_argument("--config", required=True)
    motif.add_argument("--prompt", required=True)
    motif.add_argument("--system-prompt", default=None, help="System prompt for chat models.")
    motif.add_argument("--motif-name", default=None, help="Exact motif name from analysis.* in the pattern bundle.")
    motif.add_argument("--motif-index", type=int, default=0, help="Motif index to use when --motif-name is omitted.")
    motif.add_argument(
        "--source",
        default="stable_motifs",
        choices=["stable_motifs", "stable_interactions", "motif_candidates"],
        help="Which analysis bucket in the canonical bundle to select from.",
    )
    motif.add_argument("--action", default="ablate", choices=["ablate", "amplify"])
    motif.add_argument("--alpha", type=float, default=5.0, help="Amplification strength for add-mode motif interventions.")
    motif.add_argument("--max-tokens", type=int, default=150)
    motif.add_argument("--temperature", type=float, default=0.0)
    motif.add_argument("--show-scores", action="store_true", help="Replay monitor scores on both generations.")
    motif.add_argument("--write-bundle", action="store_true", help="Append the run to analysis.manual_interventions.")
    motif.add_argument("--no-chat-template", action="store_true", help="Bypass chat template.")

    discover = subparsers.add_parser("discover", help="Find label-correlated SAE features.")
    discover.add_argument("--config", required=True)
    discover.add_argument("--layer", default=None)
    discover.add_argument("--events-csv", default=None)
    discover.add_argument("--dataset-name", default="dair-ai/emotion")
    discover.add_argument("--label-field", default="label")
    discover.add_argument("--top-k", type=int, default=15)
    discover.add_argument("--output", default=None)

    return parser.parse_args()


def _load_model_and_tokenizer(config):
    model, tokenizer, device = load_runtime_model_and_tokenizer(config)
    model_name = config.get("model_name", "gpt2")
    dtype_str = config.get("extraction.dtype", "float32")
    print(f"[steer] Loading {model_name} -> {device} ({dtype_str})")
    return model, tokenizer, device


def _load_sae_for_target(config, target: str, device: str):
    target_cfg = config.for_target(target) if hasattr(config, "for_target") else config
    sae_path = resolve_best_checkpoint(target_cfg, target=target)
    sae, _ = load_sae_checkpoint(sae_path, device=device)
    sae.eval()
    return sae


def _format_prompt(tokenizer, prompt: str, system_prompt: str | None, use_chat: bool) -> str:
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
    print(f"\n{'-' * width}")
    print("  CLEAN OUTPUT")
    print(f"{'-' * width}")
    print(clean_out.strip())
    print(f"\n{'-' * width}")
    print(f"  STEERED OUTPUT  {label}")
    print(f"{'-' * width}")
    print(steered_out.strip())
    print(f"{'-' * width}\n")


def _bundle_rows(bundle: dict, source: str) -> list[dict]:
    analysis = dict(bundle.get("analysis", {}) or {})
    rows = list(analysis.get(source, []) or [])
    if source == "stable_interactions" and not rows:
        rows = [row for row in analysis.get("stable_motifs", []) if row.get("kind") == "interaction"]
    return rows


def _select_motif(bundle: dict, source: str, motif_name: str | None, motif_index: int) -> dict:
    rows = _bundle_rows(bundle, source)
    if not rows:
        raise ValueError(f"No motifs found under analysis.{source} in the canonical pattern bundle.")
    if motif_name:
        for row in rows:
            if str(row.get("name")) == motif_name:
                return row
        raise ValueError(f"Motif {motif_name!r} not found in analysis.{source}.")
    if motif_index < 0 or motif_index >= len(rows):
        raise IndexError(f"motif-index {motif_index} is out of range for analysis.{source} ({len(rows)} motifs).")
    return rows[motif_index]


def cmd_steer(args):
    config = ConfigManager.from_file(args.config)
    model, tokenizer, device = _load_model_and_tokenizer(config)
    sae = _load_sae_for_target(config, args.layer, device)

    use_chat = not getattr(args, "no_chat_template", False)
    prompt_text = _format_prompt(tokenizer, args.prompt, getattr(args, "system_prompt", None), use_chat)

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

    label = args.features if args.features else f"F{args.feature_id} alpha={args.alpha} mode={args.mode}"
    clean_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)

    handles = []
    for feature_cfg in feature_configs:
        handles.append(
            HFSteeringHook.apply(
                hf_model=model,
                target_layer=args.layer,
                sae_model=sae,
                feature_id=feature_cfg["feature_id"],
                alpha=feature_cfg["alpha"],
                mode=feature_cfg["mode"],
            )
        )

    try:
        steered_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)
    finally:
        for handle in handles:
            handle.remove()

    _print_comparison(clean_out, steered_out, label=label)


def cmd_deception(args):
    from rfm.deception import DeceptionMonitor, DirectionResult

    config = ConfigManager.from_file(args.config)
    model, tokenizer, device = _load_model_and_tokenizer(config)

    dec_dir = deception_run_dir(config)
    directions_path = dec_dir / "directions" / "directions.pt"
    if not directions_path.exists():
        print(
            f"[steer] Direction file not found: {directions_path}\n"
            "Run: python -m cli.deception_cycle --phase direction"
        )
        sys.exit(1)

    all_directions = torch.load(directions_path, map_location="cpu", weights_only=False)
    target_layers = [layer.strip() for layer in args.layers.split(",")] if args.layers else list(all_directions.keys())
    print(f"[steer] Deception steering on {len(target_layers)} layer(s), alpha={args.alpha}")

    prompt_text = _format_prompt(tokenizer, args.prompt, args.system_prompt, True)
    clean_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)

    handles = []
    hooked_layers = []
    for layer in target_layers:
        if layer not in all_directions:
            print(f"  Warning: {layer} not in directions file; skipping.")
            continue
        direction_payload = all_directions[layer]
        direction = direction_payload["direction"] if isinstance(direction_payload, dict) else direction_payload.direction
        direction = direction.detach().to(device, dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-12)

        try:
            target_module = resolve_hf_target_module(model, layer)
        except ValueError as exc:
            print(f"  Warning: Could not resolve {layer}: {exc}")
            continue

        def _make_hook(direction_tensor, alpha_value):
            def hook_fn(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                activations = output[0] if is_tuple else output
                steered = activations + alpha_value * direction_tensor.to(activations.device, activations.dtype)
                return (steered,) + output[1:] if is_tuple else steered

            return hook_fn

        handles.append(target_module.register_forward_hook(_make_hook(direction, args.alpha)))
        hooked_layers.append(layer)

    print(f"  Hooked layers: {hooked_layers}")
    try:
        steered_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)
    finally:
        for handle in handles:
            handle.remove()

    _print_comparison(clean_out, steered_out, label=f"deception direction alpha={args.alpha} ({len(hooked_layers)} layers)")

    if getattr(args, "show_scores", False):
        try:
            directions_dict = {}
            thresholds = {}
            for layer, payload in all_directions.items():
                if layer not in hooked_layers:
                    continue
                direction_result = DirectionResult.from_dict(payload) if isinstance(payload, dict) else payload
                directions_dict[layer] = direction_result
                thresholds[layer] = float(direction_result.threshold)

            monitor = DeceptionMonitor(directions=directions_dict, thresholds=thresholds)
            for label, text in [("clean", clean_out), ("steered", steered_out)]:
                score = monitor.score_replay(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    response=text,
                    system_prompt=args.system_prompt,
                )
                print(
                    f"  [{label}] deception_prob={score.deception_probability:.3f} "
                    f"alert={score.alert} per_layer={score.per_layer_scores}"
                )
        except Exception as exc:
            print(f"  Warning: Could not compute monitor scores: {exc}")


def cmd_patch(args):
    config = ConfigManager.from_file(args.config).for_target(args.layer)
    model, tokenizer, device = _load_model_and_tokenizer(config)
    sae = _load_sae_for_target(config, args.layer, device)

    if args.feature_id is not None:
        result = activation_patch(
            model=model,
            sae_model=sae,
            clean_text=args.clean,
            patch_text=args.patch,
            target_layer=args.layer,
            feature_id=args.feature_id,
            metric=args.metric,
            tokenizer=tokenizer,
        )
        print(f"\n[patch] Feature {result['feature_id']}  Effect: {result['effect']:.6f}")
        return

    results = batch_feature_patching(
        model=model,
        sae_model=sae,
        clean_text=args.clean,
        patch_text=args.patch,
        target_layer=args.layer,
        top_k=args.top_k,
        metric=args.metric,
        tokenizer=tokenizer,
    )
    print(f"\n[patch] Top {len(results)} features by causal effect:")
    print(f"  {'Feature':>8}  {'Effect':>12}")
    print(f"  {'-' * 25}")
    for row in results:
        print(f"  {row['feature_id']:>8d}  {row['effect']:>12.6f}")


def cmd_motif(args):
    config = ConfigManager.from_file(args.config)
    axis = ContrastAxisSpec.from_config(config)
    bundle = load_pattern_bundle(config, axis)
    motif = _select_motif(bundle, args.source, args.motif_name, args.motif_index)

    model, tokenizer, device = _load_model_and_tokenizer(config)
    layer_feature_configs = motif_feature_configs(motif, action=args.action, alpha=args.alpha)
    sae_models = {layer_name: _load_sae_for_target(config, layer_name, device) for layer_name in layer_feature_configs}

    use_chat = not getattr(args, "no_chat_template", False)
    prompt_text = _format_prompt(tokenizer, args.prompt, args.system_prompt, use_chat)
    clean_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)

    handles = register_feature_interventions(
        model=model,
        sae_models=sae_models,
        layer_feature_configs=layer_feature_configs,
    )
    try:
        steered_out = _generate(model, tokenizer, prompt_text, device, args.max_tokens, args.temperature)
    finally:
        for handle in handles:
            handle.remove()

    _print_comparison(clean_out, steered_out, label=f"{motif.get('name', 'motif')} [{args.action}]")

    clean_score = None
    steered_score = None
    if getattr(args, "show_scores", False):
        ensemble_method = config.get("deception.monitor.ensemble_method", "weighted_average")
        monitor = axis_monitor_from_bundle(axis_spec=axis, bundle=bundle, ensemble_method=ensemble_method)
        if monitor is None:
            print("  Warning: No monitor/probe state found in the pattern bundle; skipping score replay.")
        else:
            clean_score = monitor.score_replay(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                response=clean_out,
                system_prompt=args.system_prompt,
            )
            steered_score = monitor.score_replay(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                response=steered_out,
                system_prompt=args.system_prompt,
            )
            print(
                f"  [clean] contrast_prob={clean_score.contrast_probability:.3f} "
                f"alert={clean_score.alert} per_layer={clean_score.per_layer_scores}"
            )
            print(
                f"  [intervened] contrast_prob={steered_score.contrast_probability:.3f} "
                f"alert={steered_score.alert} per_layer={steered_score.per_layer_scores}"
            )

    if getattr(args, "write_bundle", False):
        record = {
            "name": motif.get("name"),
            "kind": motif.get("kind"),
            "source": args.source,
            "action": args.action,
            "alpha": float(args.alpha),
            "members": motif_members(motif),
            "prompt": args.prompt,
            "system_prompt": args.system_prompt,
            "clean_response": clean_out[:400],
            "intervention_response": steered_out[:400],
        }
        if clean_score is not None and steered_score is not None:
            record.update(
                {
                    "clean_probability": float(clean_score.contrast_probability),
                    "intervention_probability": float(steered_score.contrast_probability),
                    "probability_delta": float(steered_score.contrast_probability - clean_score.contrast_probability),
                }
            )
        append_pattern_analysis_rows(
            config,
            axis_spec=axis,
            key="manual_interventions",
            rows=[record],
            max_rows=50,
        )
        print("  [bundle] Appended manual intervention record to analysis.manual_interventions")


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

    dataset = load_dataset(args.dataset_name, split="train")
    label_names = None
    if hasattr(dataset.features.get(args.label_field, None), "names"):
        label_names = dataset.features[args.label_field].names
    probe.load_labels_from_dataset(dataset, label_field=args.label_field, label_names=label_names)

    summary = probe.summary(top_k_per_emotion=args.top_k)
    print("\n" + "=" * 70)
    print("EMOTION / LABEL FEATURE DISCOVERY")
    print("=" * 70)
    for emotion, features in summary.items():
        if not features:
            continue
        print(f"\n--- {emotion.upper()} ---")
        for feature in features[:5]:
            print(
                f"  feature={feature['feature_id']:>5d}  count={feature['count']:>4d}  "
                f"mean_str={feature['mean_strength']:.4f}  specificity={feature['specificity']:.2f}"
            )

    output = args.output or str(Path(events_csv).parent / "emotion_feature_ranking.csv")
    probe.write_summary_csv(output, top_k_per_emotion=args.top_k)
    print(f"\n[discover] Written to {output}")


def main():
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.command is None:
        print("Commands: steer, deception, patch, motif, discover. Use --help for details.")
        sys.exit(1)

    dispatch = {
        "steer": cmd_steer,
        "deception": cmd_deception,
        "patch": cmd_patch,
        "motif": cmd_motif,
        "discover": cmd_discover,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
