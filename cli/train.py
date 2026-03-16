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
    if not activation_dir:
        raise FileNotFoundError(f"Datasets path not set. Must be injected via main().")

    save_path = config.get("train.save_path")

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


def _resolve_targets(config):
    raw = config.get("extraction.target")
    if isinstance(raw, list):
        return raw
    return [raw]


def main():
    args = parse_args()
    base_config = ConfigManager.from_file(args.config)
    targets = _resolve_targets(base_config)
    
    for target in targets:
        print(f"\n{'#'*60}")
        print(f"[train] Starting SAE training for target: {target}")
        print(f"{'#'*60}")
        
        config = ConfigManager.from_file(args.config)
        config.set("extraction.target", target)
        
        # Resolve target-specific paths using base_config so multi-layer status is preserved
        act_dir = default_activations_dir(base_config, target=target)
        pt_files = sorted(Path(act_dir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No activation files found in {act_dir}")
        config.set("datasets.path", [str(p) for p in pt_files])
        
        save_path = config.get("train.save_path")
        if not save_path:
            save_path = default_checkpoint_path(base_config, target=target)
            config.set("train.save_path", save_path)
        
        _run_sweep(config)


if __name__ == "__main__":
    main()
