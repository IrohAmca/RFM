from __future__ import annotations

import argparse

from rfm.config import ConfigManager
from rfm.deception.reporting import generate_deception_report


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a deception monitoring report from pipeline artifacts.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--top-features", type=int, default=10, help="Top risky features to show per layer.")
    parser.add_argument(
        "--max-projection-points",
        type=int,
        default=250,
        help="Maximum honest/deceptive points per class for the 2D projection plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)
    result = generate_deception_report(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        top_features=args.top_features,
        max_projection_points=args.max_projection_points,
    )
    print(f"[deception_report] Report directory: {result['report_dir'].resolve()}")
    for key, path in result["chart_paths"].items():
        if key == "report_dir":
            continue
        print(f"[deception_report] Wrote {path}")


if __name__ == "__main__":
    main()
