"""CLI: End-to-end pipeline execution."""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full RFM pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization step.")
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

    steps = [
        ("Extraction", [python, "-m", "cli.extract", "--config", args.config]),
        ("SAE Training", [python, "-m", "cli.train", "--config", args.config]),
        ("Feature Mapping", [python, "-m", "rfm.sae.mapping", "--config", args.config]),
    ]

    if not args.skip_viz:
        steps.append(
            ("Visualization", [python, "-m", "rfm.viz.plots", "--mode", "all", "--config", args.config]),
        )

    for label, cmd in steps:
        run_step(label, cmd, continue_on_error=args.continue_on_error)

    print(f"\n[pipeline] Done.")


if __name__ == "__main__":
    main()
