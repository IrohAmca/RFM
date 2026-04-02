"""Balance Matryoshka Sparse Autoencoder.

Addresses the feature hedging vs feature absorption tradeoff identified in
Chanin et al. 2025 (arXiv:2505.11756v2).

Feature hedging: SAE merges correlated features into single latents when the
SAE is too narrow to represent them separately. Worse in narrow SAEs.

Feature absorption: SAE sparsity penalty causes child features to suppress
parent feature firing. Worse in wide SAEs.

Balance Matryoshka SAE uses nested reconstruction losses with per-level
balance coefficients (β_m) to counter both phenomena:
- Inner levels (narrow) get smaller β → reduces hedging pressure
- Outer levels (wide) get full β → maintains absorption control
- Optimal β balances the opposing gradient forces
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfm.sae.model import TopKSAE

logger = logging.getLogger("rfm.sae.matryoshka")


class BalanceMatryoshkaSAE(TopKSAE):
    """Matryoshka SAE with per-level balance coefficients.

    The SAE hidden layer is divided into nested prefixes of increasing size.
    Each prefix is forced to reconstruct the input on its own via a separate
    loss term.  This encourages the SAE to place general, high-frequency
    features in early latents and fine-grained features in later latents.

    The ``balance_multiplier`` controls how strongly inner levels are
    weighted relative to the outermost level (which always has β=1).
    Following the paper's findings, a multiplier of ~0.75 produces the
    best trade-off between hedging and absorption.

    Args:
        input_dim:            Dimensionality of the input activations.
        hidden_dim:           Total dictionary size (number of latents).
        k:                    TopK sparsity parameter.
        matryoshka_levels:    Sizes of the nested inner dictionaries.
                              Must be strictly increasing and all < hidden_dim.
                              The full hidden_dim is appended automatically.
        balance_multiplier:   Compound multiplier between adjacent β values.
                              β_m = multiplier^(n_levels - 1 - m) for level m.
                              0.75 is recommended per Chanin et al.
        aux_alpha:            Dead-feature auxiliary loss coefficient.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 32,
        matryoshka_levels: Sequence[int] | None = None,
        balance_multiplier: float = 0.75,
        aux_alpha: float = 1 / 32,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=k,
            aux_alpha=aux_alpha,
            **kwargs,
        )

        # Build sorted level list and append the full width
        if matryoshka_levels is None:
            # Default: geometric progression up to hidden_dim
            matryoshka_levels = self._default_levels(hidden_dim)

        levels = sorted(int(lv) for lv in matryoshka_levels)

        # Validate
        for lv in levels:
            if lv <= 0 or lv >= hidden_dim:
                raise ValueError(
                    f"matryoshka_levels must be in (0, {hidden_dim}). Got {lv}."
                )

        # Append full width as the outermost level
        if levels[-1] != hidden_dim:
            levels.append(hidden_dim)

        self.matryoshka_levels = levels
        self.balance_multiplier = float(balance_multiplier)

        # Compute β coefficients: outermost level = 1.0,
        # each inner level is multiplied by balance_multiplier
        n = len(self.matryoshka_levels)
        self.betas = []
        for i in range(n):
            # i=0 is innermost (smallest), i=n-1 is outermost (full)
            beta = self.balance_multiplier ** (n - 1 - i)
            self.betas.append(beta)

        logger.info(
            f"BalanceMatryoshkaSAE: levels={self.matryoshka_levels}, "
            f"betas={[round(b, 4) for b in self.betas]}, "
            f"multiplier={self.balance_multiplier}"
        )

    @staticmethod
    def _default_levels(hidden_dim: int) -> list[int]:
        """Generate default geometric levels: [hidden_dim/64, /16, /4]."""
        levels = []
        divisor = 64
        while divisor >= 4:
            lv = max(32, hidden_dim // divisor)
            if lv < hidden_dim:
                levels.append(lv)
            divisor //= 4
        return sorted(set(levels))

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor):
        """Nested reconstruction loss with balance coefficients.

        For each matryoshka level m with prefix size S_m:
            loss_m = β_m * MSE(x, f[:, :S_m] @ W_dec[:S_m, :] + b_pre)

        Total = Σ loss_m + aux_alpha * aux_loss

        Returns:
            total_loss, recon_loss (outer level only), aux_loss
        """
        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        outer_recon_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for level_size, beta in zip(self.matryoshka_levels, self.betas):
            # Reconstruct using only the first `level_size` latents
            f_prefix = f[:, :level_size]
            x_hat_prefix = f_prefix @ self.W_dec[:level_size, :] + self.b_pre
            level_recon = F.mse_loss(x_hat_prefix, x)
            total_loss = total_loss + beta * level_recon

            # Track outermost reconstruction for logging
            if level_size == self.matryoshka_levels[-1]:
                outer_recon_loss = level_recon

        # Auxiliary dead-feature loss from TopKSAE
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self._last_h is not None and self.aux_alpha > 0:
            h = self._last_h
            topk_idx = self._last_topk_idx

            dead_mask = torch.ones_like(h, dtype=torch.bool)
            dead_mask.scatter_(-1, topk_idx, False)
            dead_h = h * dead_mask.float()
            dead_f = F.relu(dead_h)
            dead_recon = dead_f @ self.W_dec + self.b_pre
            aux_loss = F.mse_loss(dead_recon, x)

        total_loss = total_loss + self.aux_alpha * aux_loss

        return total_loss, outer_recon_loss, aux_loss

    def get_level_metrics(self, x: torch.Tensor, f: torch.Tensor) -> dict:
        """Compute per-level reconstruction quality for monitoring.

        Useful during training to track whether inner levels are learning
        general features and outer levels are refining.

        Returns:
            dict mapping level_size → {'recon_loss': float, 'beta': float}
        """
        metrics = {}
        with torch.no_grad():
            for level_size, beta in zip(self.matryoshka_levels, self.betas):
                f_prefix = f[:, :level_size]
                x_hat_prefix = f_prefix @ self.W_dec[:level_size, :] + self.b_pre
                recon = F.mse_loss(x_hat_prefix, x).item()
                metrics[level_size] = {
                    "recon_loss": round(recon, 6),
                    "beta": round(beta, 4),
                    "weighted_loss": round(beta * recon, 6),
                }
        return metrics

    def get_inner_features(self, f: torch.Tensor, level_idx: int = 0) -> torch.Tensor:
        """Extract features from a specific matryoshka level.

        Args:
            f: Full feature activations [batch, hidden_dim]
            level_idx: Which nesting level (0 = innermost/smallest)

        Returns:
            [batch, level_size] feature subset
        """
        if level_idx < 0 or level_idx >= len(self.matryoshka_levels):
            raise IndexError(
                f"level_idx must be in [0, {len(self.matryoshka_levels) - 1}]"
            )
        level_size = self.matryoshka_levels[level_idx]
        return f[:, :level_size]


class MatryoshkaTrainingCallback:
    """Optional training callback that logs per-level metrics.

    Usage in training loop:
        callback = MatryoshkaTrainingCallback(model)
        # After each batch:
        callback.on_batch_end(x, f, epoch, step)
    """

    def __init__(self, model: BalanceMatryoshkaSAE, log_every_n_steps: int = 100):
        self.model = model
        self.log_every_n_steps = log_every_n_steps
        self._step_counter = 0

    def on_batch_end(self, x: torch.Tensor, f: torch.Tensor, epoch: int, step: int):
        self._step_counter += 1
        if self._step_counter % self.log_every_n_steps != 0:
            return

        metrics = self.model.get_level_metrics(x, f)
        parts = [f"[Matryoshka e{epoch} s{step}]"]
        for level_size, m in sorted(metrics.items()):
            parts.append(f"L{level_size}={m['recon_loss']:.5f}(β={m['beta']:.2f})")
        logger.info(" ".join(parts))
