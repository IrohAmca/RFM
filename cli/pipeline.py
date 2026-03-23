"""CLI: End-to-end pipeline execution."""

import argparse
import subprocess
import sys


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


def main():
    args = parse_args()
    python = sys.executable

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

    all_steps = [
        ("extract",  "Extraction",      not args.skip_extract,  [python, "-m", "cli.extract",      "--config", args.config]),
        ("train",    "SAE Training",    not args.skip_train,    [python, "-m", "cli.train",         "--config", args.config]),
        ("mapping",  "Feature Mapping", not args.skip_mapping,  [python, "-m", "rfm.sae.mapping",   "--config", args.config]),
        ("viz",      "Visualization",   not args.skip_viz,      [python, "-m", "rfm.viz.plots", "--mode", "all", "--config", args.config]),
    ]

    for step_key, label, enabled, cmd in all_steps:
        if enabled:
            run_step(label, cmd, continue_on_error=args.continue_on_error)
        else:
            print(f"\n[pipeline] Skipping: {label}")

    print(f"\n[pipeline] Done.")


if __name__ == "__main__":
    main()
