"""CLI: SAE training."""

import argparse
from pathlib import Path

import torch

from rfm.config import ConfigManager
from rfm.layout import (
    resolve_activations_dir,
    resolve_checkpoint_path,
    resolve_requested_targets,
)
from rfm.sae.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on extracted activations.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    return parser.parse_args()


def _save_model(model, config, save_path):
    """Save trained model checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": config.as_dict(),
        "history": model.training_history,
    }, save_path)


def _run_sweep(config):
    sae_section = config.section("sae") if hasattr(config, "section") else config.get("sae", {})
    architecture = sae_section.get("architecture", "vanilla")

    activation_dir = config.get("datasets.path")
    if not activation_dir:
        raise FileNotFoundError("Datasets path not set. Must be injected via main().")

    save_path = config.get("train.save_path")

    # ── TopK: sweep over K values, sparsity_weight is irrelevant ──
    if architecture == "topk":
        k_sweep = sae_section.get("topk_k_sweep")
        if k_sweep:
            for k_val in k_sweep:
                print(f"\n{'='*60}")
                print(f"[sweep] topk_k = {k_val}")
                print(f"{'='*60}")
                config.set("sae.topk_k", int(k_val))
                model = train(config)
                sweep_save = str(Path(save_path).parent / f"sae_topk_k_{k_val}.pt")
                _save_model(model, config, sweep_save)
                print(f"[sweep] Saved → {sweep_save}")
        else:
            print(f"[train] Architecture: TopK (k={sae_section.get('topk_k', 32)})")
            model = train(config)
            _save_model(model, config, save_path)
            print(f"[train] Saved → {save_path}")
        return

    # ── Gated: single run, sparsity_weight is used but sweep is unusual ──
    if architecture == "gated":
        sweep_lambdas = sae_section.get("sparsity_sweep")
        if sweep_lambdas:
            for lam in sweep_lambdas:
                print(f"\n{'='*60}")
                print(f"[sweep] sparsity_weight = {lam}")
                print(f"{'='*60}")
                config.set("sae.sparsity_weight", lam)
                model = train(config)
                sweep_save = str(Path(save_path).parent / f"sae_lambda_{lam}.pt")
                _save_model(model, config, sweep_save)
                print(f"[sweep] Saved → {sweep_save}")
        else:
            print("[train] Architecture: Gated")
            model = train(config)
            _save_model(model, config, save_path)
            print(f"[train] Saved → {save_path}")
        return

    # ── Vanilla: sweep over sparsity_weight (lambda) ──
    sweep_lambdas = sae_section.get("sparsity_sweep")
    if sweep_lambdas:
        for lam in sweep_lambdas:
            print(f"\n{'='*60}")
            print(f"[sweep] sparsity_weight = {lam}")
            print(f"{'='*60}")
            config.set("sae.sparsity_weight", lam)
            model = train(config)
            sweep_save = str(Path(save_path).parent / f"sae_lambda_{lam}.pt")
            _save_model(model, config, sweep_save)
            print(f"[sweep] Saved → {sweep_save}")
    else:
        model = train(config)
        _save_model(model, config, save_path)
        print(f"[train] Saved → {save_path}")


def _resolve_targets(config):
    return resolve_requested_targets(config)


def main():
    args = parse_args()
    base_config = ConfigManager.from_file(args.config)
    targets = _resolve_targets(base_config)
    
    for target in targets:
        print(f"\n{'#'*60}")
        print(f"[train] Starting SAE training for target: {target}")
        print(f"{'#'*60}")
        
        config = base_config.for_target(target)
        
        # Resolve target-specific paths using base_config so multi-layer status is preserved
        act_dir = resolve_activations_dir(config, target=target)
        pt_files = sorted(Path(act_dir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No activation files found in {act_dir}")
        config.set("datasets.path", [str(p) for p in pt_files])
        
        save_path = config.get("train.save_path")
        if not save_path:
            save_path = resolve_checkpoint_path(config, target=target)
        config.set("train.save_path", save_path)
        
        _run_sweep(config)


if __name__ == "__main__":
    main()
