from sae.model import SparseAutoEncoder
from base.dataloader import BaseDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def _make_train_val_loaders(activations, batch_size, val_split, seed):
    if not (0.0 <= val_split < 1.0):
        raise ValueError("train.validation_split must be in [0.0, 1.0).")

    num_rows = int(activations.shape[0])
    if num_rows < 2 or val_split == 0.0:
        train_loader = DataLoader(
            TensorDataset(activations),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        return train_loader, None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_rows, generator=generator)

    val_size = max(1, int(num_rows * val_split))
    train_size = max(1, num_rows - val_size)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    if val_idx.numel() == 0:
        train_loader = DataLoader(
            TensorDataset(activations[train_idx]),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        return train_loader, None

    train_loader = DataLoader(
        TensorDataset(activations[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(activations[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate(model, loader, device, active_threshold):
    if loader is None:
        return None

    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    active_rate_sum = 0.0
    steps = 0

    for (x,) in loader:
        x = x.to(device)
        x_hat, f = model(x)
        total_loss, recon_loss, sparsity_loss = model.compute_loss(x, x_hat, f)

        active_rate = (f > active_threshold).float().mean()

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        sparsity_loss_sum += sparsity_loss.item()
        active_rate_sum += active_rate.item()
        steps += 1

    if steps == 0:
        return None

    return {
        "loss": total_loss_sum / steps,
        "recon": recon_loss_sum / steps,
        "sparse": sparsity_loss_sum / steps,
        "active_rate": active_rate_sum / steps,
    }


def train(config):
    dataset = BaseDataset(config)
    dataset.load()

    sae_config = config.get("sae", {})
    train_config = config.get("train", {})

    input_dim = dataset.get_feature_dim()
    hidden_dim = int(sae_config.get("hidden_dim", input_dim * 4))
    sparsity_weight = float(sae_config.get("sparsity_weight", 1e-3))

    batch_size = int(train_config.get("batch_size", 1024))
    epochs = int(train_config.get("epochs", 1))
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    weight_decay = float(train_config.get("weight_decay", 0.0))
    val_split = float(train_config.get("validation_split", 0.1))
    split_seed = int(train_config.get("split_seed", 42))
    active_threshold = float(train_config.get("feature_activity_threshold", 1e-3))

    device = torch.device(
        train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    model = SparseAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_weight=sparsity_weight,
    ).to(device)

    # Initialize pre-bias with the activation mean for centering.
    model.set_pre_bias(dataset.get_mean_activation().to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader, val_loader = _make_train_val_loaders(
        dataset.activations,
        batch_size=batch_size,
        val_split=val_split,
        seed=split_seed,
    )

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        sparsity_loss_sum = 0.0
        active_rate_sum = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for step, (x,) in enumerate(progress, start=1):
            x = x.to(device)

            x_hat, f = model(x)
            total_loss, recon_loss, sparsity_loss = model.compute_loss(x, x_hat, f)
            active_rate = (f > active_threshold).float().mean()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            model.normalize_decoder_columns()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            sparsity_loss_sum += sparsity_loss.item()
            active_rate_sum += active_rate.item()

            progress.set_postfix(
                loss=f"{(total_loss_sum / step):.6f}",
                recon=f"{(recon_loss_sum / step):.6f}",
                sparse=f"{(sparsity_loss_sum / step):.6f}",
                active=f"{(active_rate_sum / step):.4f}",
            )

        num_steps = max(len(train_loader), 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": total_loss_sum / num_steps,
            "train_recon": recon_loss_sum / num_steps,
            "train_sparse": sparsity_loss_sum / num_steps,
            "train_active_rate": active_rate_sum / num_steps,
        }

        val_metrics = _evaluate(model, val_loader, device, active_threshold)
        if val_metrics is not None:
            epoch_metrics.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_recon": val_metrics["recon"],
                    "val_sparse": val_metrics["sparse"],
                    "val_active_rate": val_metrics["active_rate"],
                }
            )

        history.append(epoch_metrics)

        summary = (
            f"epoch={epoch + 1} "
            f"train_loss={epoch_metrics['train_loss']:.6f} "
            f"train_recon={epoch_metrics['train_recon']:.6f} "
            f"train_sparse={epoch_metrics['train_sparse']:.6f} "
            f"train_active={epoch_metrics['train_active_rate']:.4f}"
        )
        if val_metrics is not None:
            summary += (
                f" val_loss={epoch_metrics['val_loss']:.6f} "
                f"val_recon={epoch_metrics['val_recon']:.6f} "
                f"val_sparse={epoch_metrics['val_sparse']:.6f} "
                f"val_active={epoch_metrics['val_active_rate']:.4f}"
            )
        print(summary)

    # Dead feature analysis
    dead_info = _dead_feature_analysis(model, train_loader, device, active_threshold)
    if dead_info:
        print(
            f"\n[Dead Feature Analysis] "
            f"dead={dead_info['dead_count']}/{dead_info['total_features']} "
            f"({dead_info['dead_ratio']:.1%}) "
            f"alive={dead_info['alive_count']}"
        )
    history.append({"dead_feature_analysis": dead_info})

    model.training_history = history
    return model


@torch.no_grad()
def _dead_feature_analysis(model, loader, device, threshold):
    """Identify features that never activate above threshold across the dataset."""
    model.eval()
    hidden_dim = model.hidden_dim
    ever_active = torch.zeros(hidden_dim, dtype=torch.bool, device=device)

    for (x,) in loader:
        x = x.to(device)
        _, f = model(x)
        batch_active = (f > threshold).any(dim=0)
        ever_active |= batch_active

    alive_count = int(ever_active.sum().item())
    dead_count = hidden_dim - alive_count

    return {
        "total_features": hidden_dim,
        "alive_count": alive_count,
        "dead_count": dead_count,
        "dead_ratio": dead_count / max(hidden_dim, 1),
        "dead_feature_ids": torch.nonzero(~ever_active, as_tuple=False).flatten().cpu().tolist(),
    }

