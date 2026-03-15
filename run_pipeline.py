import argparse
import subprocess
import sys
from pathlib import Path

from config_manager import ConfigManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run extraction, training, mapping, and visualization in sequence."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config JSON/TOML.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization step (report_plots.py).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps even if a step fails.",
    )
    return parser.parse_args()


def run_step(command, cwd):
    print(f"\n[PIPELINE] Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(cwd), check=False)
    print(f"[PIPELINE] Exit code: {result.returncode}")
    return result.returncode


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config = ConfigManager.from_file(args.config)

    mapping_cfg = config.section("feature-mapping") if hasattr(config, "section") else {}

    events_csv = mapping_cfg.get(
        "event_output_path",
        "reports/feature_mapping/feature_mapping_events.csv",
    )
    summary_csv = mapping_cfg.get(
        "summary_csv_output_path",
        "reports/feature_mapping/feature_mapping_feature_summary.csv",
    )
    token_pairs_csv = str(Path(summary_csv).with_name(Path(summary_csv).stem + "_token_pairs.csv"))
    viz_output_dir = str(Path(summary_csv).with_name("viz"))

    steps = [
        [sys.executable, "runner.py", "--config", args.config],
        [sys.executable, "train_runner.py", "--config", args.config],
        [sys.executable, "-m", "sae.mapping", "--config", args.config],
    ]

    if not args.skip_viz:
        steps.append(
            [
                sys.executable,
                "report_plots.py",
                "--mode",
                "mapping",
                "--mapping-events-csv",
                events_csv,
                "--mapping-summary-csv",
                summary_csv,
                "--mapping-token-pairs-csv",
                token_pairs_csv,
                "--output-dir",
                viz_output_dir,
            ]
        )

    failed = []
    for step in steps:
        code = run_step(step, cwd=repo_root)
        if code != 0:
            failed.append((step, code))
            if not args.continue_on_error:
                break

    print("\n[PIPELINE] Finished.")
    if failed:
        print("[PIPELINE] Failed steps:")
        for step, code in failed:
            print(f"  - ({code}) {' '.join(step)}")
        raise SystemExit(1)

    print("[PIPELINE] All steps completed successfully.")


if __name__ == "__main__":
    main()
