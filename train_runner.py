import argparse
import copy
from pathlib import Path

import torch

from config_manager import ConfigManager
from sae.train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAE from saved activation chunks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON/TOML config. If omitted, built-in defaults are used.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional override for train.output_model_path",
    )
    return parser.parse_args()


def _resolve_save_path(config, override_path=None):
    if override_path:
        return override_path
    return config.get("train.output_model_path", "checkpoints/sae.pt")


def _save_checkpoint(model, config, save_path):
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.as_dict() if hasattr(config, "as_dict") else config,
            "history": getattr(model, "training_history", []),
        },
        output_path,
    )
    print(f"Saved SAE checkpoint to: {output_path}")


def _run_single(config, save_path):
    model = train(config)
    _save_checkpoint(model, config, save_path)


def _run_sweep(config, save_path):
    sweep_values = config.get("sae.sparsity_sweep", [])
    if not isinstance(sweep_values, list) or not sweep_values:
        _run_single(config, save_path)
        return

    base = config.as_dict()
    save_path_obj = Path(save_path)
    for lambda_value in sweep_values:
        run_cfg = copy.deepcopy(base)
        run_cfg.setdefault("sae", {})["sparsity_weight"] = float(lambda_value)
        run_config = ConfigManager(run_cfg)

        print(f"Running sweep with sparsity_weight={float(lambda_value):.6g}")
        model = train(run_config)

        suffix = str(lambda_value).replace(".", "p")
        sweep_path = save_path_obj.with_name(
            f"{save_path_obj.stem}_l1_{suffix}{save_path_obj.suffix}"
        )
        _save_checkpoint(model, run_config, sweep_path)


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)

    save_path = _resolve_save_path(config, args.save_path)
    _run_sweep(config, save_path)


if __name__ == "__main__":
    main()
