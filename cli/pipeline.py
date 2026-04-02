"""CLI: End-to-end pipeline execution."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from rfm.config import ConfigManager
from rfm.layout import resolve_requested_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full RFM pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization step.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction step (use existing activations).")
    parser.add_argument("--skip-train", action="store_true", help="Skip SAE training step (use existing checkpoints).")
    parser.add_argument("--skip-mapping", action="store_true", help="Skip feature mapping step.")
    parser.add_argument(
        "--from-step",
        choices=["extract", "train", "mapping", "viz"],
        default=None,
        help="Start pipeline from this step onward (skips all earlier steps).",
    )
    parser.add_argument(
        "--from-hook",
        type=str,
        default=None,
        help="Start from this hook target onward within extraction.target. Accepts an exact target or layer index like '27'.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a step fails.")
    return parser.parse_args()


def run_step(label, cmd, continue_on_error=False):
    print(f"\n{'='*60}")
    print(f"[pipeline] {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[pipeline] WARNING: {label} exited with code {result.returncode}")
        if not continue_on_error:
            sys.exit(result.returncode)
    return result.returncode


def _build_effective_config(config_path, from_hook=None):
    if not from_hook:
        return config_path, None, None

    config = ConfigManager.from_file(config_path)
    config.set("pipeline.from_hook", from_hook)
    selected_targets = resolve_requested_targets(config)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(config.as_dict(), handle, ensure_ascii=False, indent=2)
        temp_path = handle.name

    return temp_path, Path(temp_path), selected_targets


def main():
    args = parse_args()
    python = sys.executable
    temp_config_path = None

    try:
        try:
            effective_config, temp_config_path, selected_targets = _build_effective_config(
                args.config,
                from_hook=args.from_hook,
            )
        except ValueError as exc:
            print(f"[pipeline] ERROR: {exc}")
            sys.exit(2)

        if selected_targets:
            print(f"[pipeline] Starting from hook: {selected_targets[0]}")
            print(f"[pipeline] Target subset: {', '.join(selected_targets)}")

        # --from-step sets skip flags for everything before the chosen step
        STEP_ORDER = ["extract", "train", "mapping", "viz"]
        if args.from_step:
            start_idx = STEP_ORDER.index(args.from_step)
            if start_idx > STEP_ORDER.index("extract"):
                args.skip_extract = True
            if start_idx > STEP_ORDER.index("train"):
                args.skip_train = True
            if start_idx > STEP_ORDER.index("mapping"):
                args.skip_mapping = True

        config = ConfigManager.from_file(args.config)
        extractor_backend = config.get("extraction.extractor_backend", "hf_causal")
        extract_module = "cli.extract_generate" if extractor_backend in ("hf_generate", "generate") else "cli.extract"

        all_steps = [
            ("extract",  "Extraction",      not args.skip_extract,  [python, "-m", extract_module,     "--config", effective_config]),
            ("train",    "SAE Training",    not args.skip_train,    [python, "-m", "cli.train",         "--config", effective_config]),
            ("mapping",  "Feature Mapping", not args.skip_mapping,  [python, "-m", "rfm.sae.mapping",   "--config", effective_config]),
            ("viz",      "Visualization",   not args.skip_viz,      [python, "-m", "rfm.viz.plots", "--mode", "all", "--config", effective_config]),
        ]

        for step_key, label, enabled, cmd in all_steps:
            if enabled:
                run_step(label, cmd, continue_on_error=args.continue_on_error)
            else:
                print(f"\n[pipeline] Skipping: {label}")

        print("\n[pipeline] Done.")
    finally:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
