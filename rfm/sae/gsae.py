"""Graph-Regularized Sparse Autoencoder (GSAE).

Extends TopKSAE with a Laplacian smoothness penalty based on a neuron
co-activation graph. This encourages frequently co-activating neurons 
to share similar feature representations, which helps capture distributed 
safety concepts like "deception" that span multiple neurons.

The co-activation graph is built from a batch of activations (typically during
a warmup phase) and kept on CPU to avoid GPU memory overhead during training.

Key properties:
- Feature representations are more *coherent* for distributed concepts
- Safety concepts are grouped into natural clusters
- Better defense against jailbreak attacks (GCG, AutoDAN)

Reference: GSAE paper (2025-2026), safety steering via spectral vectors
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfm.sae.model import TopKSAE

logger = logging.getLogger("rfm.sae.gsae")


class GraphRegularizedSAE(TopKSAE):
    """GSAE: TopK SAE with Laplacian graph regularization.

    During training, adds a penalty term:
        λ * trace(F^T L F) / batch_size

    where L is the graph Laplacian of the neuron co-activation graph and 
    F is the sparse feature activation matrix.

    This penalty encourages features that frequently co-activate in the
    input data to have similar learned representations — critical for 
    capturing distributed safety concepts.

    The co-activation graph is built on CPU from a warmup batch, then 
    the Laplacian submatrix is transferred to GPU only during loss 
    computation (using sparse representation to minimize transfer cost).

    Args:
        input_dim:     Dimensionality of input activations.
        hidden_dim:    Dictionary size.
        k:             TopK sparsity.
        graph_lambda:  Weight of the Laplacian regularization term.
        graph_knn:     Number of nearest neighbors for KNN graph sparsification.
        aux_alpha:     Dead-feature auxiliary loss coefficient.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 32,
        graph_lambda: float = 0.01,
        graph_knn: int = 10,
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
        self.graph_lambda = float(graph_lambda)
        self.graph_knn = int(graph_knn)

        # Laplacian is built lazily on CPU, transferred to device during loss
        self._laplacian_cpu: torch.Tensor | None = None
        self._graph_built = False

        logger.info(
            f"GraphRegularizedSAE: graph_lambda={self.graph_lambda}, "
            f"graph_knn={self.graph_knn}"
        )

    @torch.no_grad()
    def build_coactivation_graph(
        self,
        activation_batch: torch.Tensor,
        batch_size: int = 4096,
    ) -> None:
        """Build KNN co-activation graph from a batch of raw activations.

        This should be called once during training warmup, using a 
        representative batch of raw activations (before SAE encoding).

        The method:
        1. Encodes activations through current SAE weights
        2. Computes pairwise co-activation frequency (binary)
        3. Sparsifies with KNN
        4. Computes Laplacian L = D - A and stores on CPU

        Args:
            activation_batch: [N, input_dim] raw model activations
            batch_size:       Processing batch size for encoding
        """
        logger.info(f"Building co-activation graph from {activation_batch.shape[0]} samples...")
        device = next(self.parameters()).device

        # Encode in batches
        all_features = []
        for i in range(0, activation_batch.shape[0], batch_size):
            batch = activation_batch[i: i + batch_size].to(device)
            _, f = self.forward(batch)
            all_features.append(f.cpu())

        features = torch.cat(all_features, dim=0)  # [N, hidden_dim]
        n_samples = features.shape[0]

        # Binary co-activation matrix
        binary = (features > 0).float()  # [N, hidden_dim]
        
        # Co-activation: how often do pairs of features both fire?
        # [hidden_dim, hidden_dim] = binary^T @ binary / N
        coact = (binary.T @ binary) / n_samples

        # Zero out diagonal (self-co-activation is always freq=1)
        coact.fill_diagonal_(0.0)

        # KNN sparsification: keep only top-k neighbors per feature
        hidden_dim = features.shape[1]
        effective_k = min(self.graph_knn, hidden_dim - 1)
        
        topk_vals, topk_idx = torch.topk(coact, effective_k, dim=1)
        adj = torch.zeros_like(coact)
        adj.scatter_(1, topk_idx, topk_vals)
        adj = (adj + adj.T) / 2  # Symmetrize

        # Graph Laplacian: L = D - A
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj

        # Normalize: L_norm = D^{-1/2} L D^{-1/2} for numerical stability
        d_inv_sqrt = torch.zeros_like(degree)
        nonzero = degree > 1e-12
        d_inv_sqrt[nonzero] = 1.0 / degree[nonzero].sqrt()
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian = D_inv_sqrt @ laplacian @ D_inv_sqrt

        self._laplacian_cpu = laplacian.cpu()
        self._graph_built = True

        # Stats
        n_edges = (adj > 0).sum().item() // 2
        logger.info(
            f"Graph built: {hidden_dim} nodes, {n_edges} edges, "
            f"avg degree={degree.mean().item():.1f}"
        )

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor):
        """TopK loss + Laplacian graph regularization.

        Graph loss = λ * trace(F^T L F) / batch_size
        This penalizes features that co-activate in data but have 
        dissimilar learned directions.

        Returns:
            total_loss, recon_loss, aux_loss
        """
        total_loss, recon_loss, aux_loss = super().compute_loss(x, x_hat, f)

        # Add graph regularization if Laplacian is available
        if self._graph_built and self._laplacian_cpu is not None and self.graph_lambda > 0:
            L = self._laplacian_cpu.to(f.device, dtype=f.dtype)
            
            # trace(F^T L F) — the graph smoothness penalty
            # Efficient computation: sum of element-wise product of (F @ L) and F
            graph_loss = (f @ L * f).sum() / f.shape[0]
            total_loss = total_loss + self.graph_lambda * graph_loss

        return total_loss, recon_loss, aux_loss

    @property
    def graph_built(self) -> bool:
        """Whether the co-activation graph has been constructed."""
        return self._graph_built

    @torch.no_grad()
    def get_feature_clusters(
        self, n_clusters: int = 20
    ) -> dict[int, list[int]]:
        """Cluster features based on co-activation graph structure.

        Uses spectral clustering on the adjacency matrix to group 
        features that belong to the same distributed concept.

        Returns:
            dict mapping cluster_id → list of feature_ids
        """
        if not self._graph_built:
            raise RuntimeError("Cannot cluster: co-activation graph not built yet.")

        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            logger.warning("scikit-learn required for spectral clustering.")
            return {}

        # Reconstruct adjacency from Laplacian (approximately)
        L = self._laplacian_cpu.numpy()
        import numpy as np
        D = np.diag(np.diag(L))
        A = D - L
        A = np.maximum(A, 0)  # Ensure non-negative

        sc = SpectralClustering(
            n_clusters=min(n_clusters, self.hidden_dim),
            affinity="precomputed",
            random_state=42,
        )
        cluster_labels = sc.fit_predict(A + np.eye(A.shape[0]) * 1e-10)

        clusters: dict[int, list[int]] = {}
        for fid, cid in enumerate(cluster_labels):
            cid = int(cid)
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(fid)

        logger.info(
            f"Spectral clustering: {n_clusters} clusters, "
            f"sizes={[len(v) for v in sorted(clusters.values(), key=len, reverse=True)[:5]]}"
        )
        return clusters
