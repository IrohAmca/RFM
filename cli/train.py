"""CLI: SAE training."""

import argparse
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.layout import default_activations_dir, default_checkpoint_path
from rfm.sae.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on extracted activations.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    return parser.parse_args()


def _run_sweep(config):
    sae_section = config.section("sae") if hasattr(config, "section") else config.get("sae", {})
    sweep_lambdas = sae_section.get("sparsity_sweep")

    activation_dir = config.get("datasets.path")
    if activation_dir is None:
        act_dir = default_activations_dir(config)
        pt_files = sorted(Path(act_dir).glob("*.pt"))
        if pt_files:
            config.set("datasets.path", [str(p) for p in pt_files])
        else:
            raise FileNotFoundError(f"No activation files found in {act_dir}")

    save_path = config.get("train.save_path") or default_checkpoint_path(config)

    if sweep_lambdas:
        for lam in sweep_lambdas:
            print(f"\n{'='*60}")
            print(f"[sweep] sparsity_weight = {lam}")
            print(f"{'='*60}")
            config.set("sae.sparsity_weight", lam)

            sweep_save = str(Path(save_path).parent / f"sae_lambda_{lam}.pt")
            model = train(config)
            Path(sweep_save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": config.as_dict(),
                "history": model.training_history,
            }, sweep_save)
            print(f"[sweep] Saved → {sweep_save}")
    else:
        model = train(config)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": config.as_dict(),
            "history": model.training_history,
        }, save_path)
        print(f"[train] Saved → {save_path}")


def main():
    args = parse_args()
    config = ConfigManager.from_file(args.config)
    _run_sweep(config)


if __name__ == "__main__":
    main()
