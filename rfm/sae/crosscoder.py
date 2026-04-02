"""Sparse Crosscoder — Cross-layer SAE for unified multi-layer feature tracking.

Unlike independent per-layer SAEs, the crosscoder learns a single shared 
latent space from concatenated activations across all target layers. This 
enables features that *span* layers — tracking how a concept evolves from 
topic recognition → intent classification → output generation.

For deception detection, this is critical:
- Layer 6: "weapon knowledge" (topic feature)
- Layer 13: "instruction-giving intent" (intent feature)
- Layer 27: "harmful output pattern" (output feature)

A single crosscoder latent can capture this entire trajectory, while 
per-layer SAEs require post-hoc combination analysis.

Architecture:
    - Shared encoder: concat(layer_acts) → TopK sparse code
    - Per-layer decoders: sparse code → per-layer reconstructions
    - Total loss = Σ_layer MSE(reconstruction_layer, original_layer)

Research basis: Cross-layer transcoders (CLTs), Sparse Crosscoders (2025-2026)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfm.sae.crosscoder")


class SparseCrosscoder(nn.Module):
    """Cross-layer SAE with shared encoder and per-layer decoders.

    Args:
        layer_dims:     Tuple/list of activation dimensions per layer.
                        E.g., (1024, 1024, 1024, 1024) for 4 layers of Qwen3-0.6B.
        hidden_dim:     Shared dictionary size (latent space dimension).
        k:              TopK sparsity — number of active latents per sample.
        aux_alpha:      Dead-feature resurrection loss coefficient.
        layer_names:    Optional human-readable layer names for logging.
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        hidden_dim: int = 16384,
        k: int = 64,
        aux_alpha: float = 1 / 32,
        layer_names: Sequence[str] | None = None,
    ):
        super().__init__()

        self.layer_dims = tuple(int(d) for d in layer_dims)
        self.n_layers = len(self.layer_dims)
        self.total_input_dim = sum(self.layer_dims)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)
        self.aux_alpha = float(aux_alpha)
        self.layer_names = layer_names or [f"layer_{i}" for i in range(self.n_layers)]

        # Pre-bias (per concatenated input)
        self.b_pre = nn.Parameter(torch.zeros(self.total_input_dim))

        # Shared encoder: concatenated input → latent space
        self.W_enc = nn.Parameter(torch.empty(self.total_input_dim, self.hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_dim))

        # Per-layer decoders: latent space → per-layer reconstruction
        self.decoders = nn.ModuleList()
        for dim in self.layer_dims:
            self.decoders.append(nn.Linear(self.hidden_dim, dim, bias=True))

        # Track for auxiliary loss
        self._last_h = None
        self._last_topk_idx = None

        self._reset_parameters()

        logger.info(
            f"SparseCrosscoder: {self.n_layers} layers, "
            f"input_dims={self.layer_dims}, hidden={self.hidden_dim}, k={self.k}"
        )

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=0.0, nonlinearity="relu")
        for dec in self.decoders:
            nn.init.kaiming_uniform_(dec.weight, a=0.0, nonlinearity="linear")
            nn.init.zeros_(dec.bias)

    @torch.no_grad()
    def set_pre_bias(self, mean_activations: list[torch.Tensor]):
        """Set pre-bias from per-layer mean activations.

        Args:
            mean_activations: list of [dim_i] tensors, one per layer
        """
        if len(mean_activations) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} mean activations, got {len(mean_activations)}"
            )
        combined = torch.cat([m.flatten() for m in mean_activations], dim=0)
        if combined.shape[0] != self.total_input_dim:
            raise ValueError(
                f"Combined mean dim={combined.shape[0]} != total_input_dim={self.total_input_dim}"
            )
        self.b_pre.copy_(combined)

    def forward(
        self, layer_activations: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass through the crosscoder.

        Args:
            layer_activations: list of [batch, dim_i] tensors, one per layer.
                               All must have the same batch_size.

        Returns:
            reconstructions: list of [batch, dim_i] tensors (per-layer reconstructions)
            f:               [batch, hidden_dim] sparse feature activations
        """
        if len(layer_activations) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} layer activations, got {len(layer_activations)}"
            )

        # Concatenate all layers
        x = torch.cat(layer_activations, dim=-1)  # [batch, total_input_dim]

        # Encode
        x_centered = x - self.b_pre
        h = x_centered @ self.W_enc + self.b_enc  # [batch, hidden_dim]

        # TopK sparsity
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        f = torch.zeros_like(h)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))

        # Per-layer reconstruction
        reconstructions = [dec(f) for dec in self.decoders]

        # Store for auxiliary loss
        self._last_h = h
        self._last_topk_idx = topk_idx

        return reconstructions, f

    def compute_loss(
        self,
        layer_activations: list[torch.Tensor],
        reconstructions: list[torch.Tensor],
        f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss = Σ per-layer MSE + auxiliary dead-feature loss.

        Returns:
            total_loss, mean_recon_loss (averaged across layers), aux_loss
        """
        # Per-layer reconstruction losses
        recon_losses = []
        for orig, recon in zip(layer_activations, reconstructions):
            recon_losses.append(F.mse_loss(recon, orig.detach()))

        mean_recon = sum(recon_losses) / len(recon_losses)
        total_loss = sum(recon_losses)

        # Auxiliary dead-feature loss
        aux_loss = torch.tensor(0.0, device=f.device, dtype=f.dtype)
        if self.training and self._last_h is not None and self.aux_alpha > 0:
            h = self._last_h
            topk_idx = self._last_topk_idx

            dead_mask = torch.ones_like(h, dtype=torch.bool)
            dead_mask.scatter_(-1, topk_idx, False)
            dead_h = h * dead_mask.float()
            dead_f = F.relu(dead_h)

            # Reconstruct from dead features → each layer
            dead_recon_losses = []
            for dec, orig in zip(self.decoders, layer_activations):
                dead_recon = dec(dead_f)
                dead_recon_losses.append(F.mse_loss(dead_recon, orig.detach()))

            aux_loss = sum(dead_recon_losses) / len(dead_recon_losses)

        total_loss = total_loss + self.aux_alpha * aux_loss

        return total_loss, mean_recon, aux_loss

    def encode(self, layer_activations: list[torch.Tensor]) -> torch.Tensor:
        """Encode only — returns sparse feature vector.

        Args:
            layer_activations: list of [batch, dim_i] tensors

        Returns:
            [batch, hidden_dim] sparse features (TopK activated)
        """
        x = torch.cat(layer_activations, dim=-1)
        x_centered = x - self.b_pre
        h = x_centered @ self.W_enc + self.b_enc
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        f = torch.zeros_like(h)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))
        return f

    @torch.no_grad()
    def get_feature_layer_attribution(self, feature_id: int) -> dict[str, float]:
        """Analyze which layers a given feature primarily reconstructs.

        Returns dict mapping layer_name → reconstruction weight magnitude.
        Useful for understanding whether a crosscoder feature is primarily
        an early-layer, mid-layer, or late-layer concept.
        """
        if feature_id < 0 or feature_id >= self.hidden_dim:
            raise IndexError(f"feature_id must be in [0, {self.hidden_dim})")

        attributions = {}
        for name, dec in zip(self.layer_names, self.decoders):
            # Weight column for this feature
            weight = dec.weight[:, feature_id]  # [layer_dim]
            attributions[name] = float(weight.norm().item())

        # Normalize to sum=1
        total = sum(attributions.values()) + 1e-12
        attributions = {k: round(v / total, 4) for k, v in attributions.items()}
        return attributions

    @torch.no_grad()
    def get_cross_layer_features(
        self, threshold: float = 0.2
    ) -> list[dict]:
        """Find features that significantly span multiple layers.

        A feature is "cross-layer" if its decoder weights are distributed
        across multiple layers (no single layer dominates by > threshold).

        Returns:
            List of dicts with feature_id and per-layer attributions.
        """
        results = []
        for fid in range(self.hidden_dim):
            attr = self.get_feature_layer_attribution(fid)
            max_attr = max(attr.values())
            if max_attr < (1.0 - threshold):  # No single layer dominates
                results.append({
                    "feature_id": fid,
                    "attributions": attr,
                    "max_layer_weight": max_attr,
                    "n_significant_layers": sum(
                        1 for v in attr.values() if v > threshold / self.n_layers
                    ),
                })

        results.sort(key=lambda x: x["max_layer_weight"])
        logger.info(f"Found {len(results)} cross-layer features (threshold={threshold})")
        return results

    @property
    def input_dim(self):
        return self.total_input_dim
