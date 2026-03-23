"""
Feature clustering based on decoder weight cosine similarity.
Useful for identifying groups of features that represent related concepts.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    pass

logger = logging.getLogger("rfm.analysis.clustering")


class FeatureClustering:
    def __init__(self, sae_model_path: str | Path, device: str = "cpu"):
        """
        Args:
            sae_model_path: Path to the trained SAE checkpoint.
        """
        self.model_path = Path(sae_model_path)
        self.device = torch.device(device)
        self.W_dec = None
        
        # Keep dependencies optional until instantiated
        try:
            import scipy.cluster.hierarchy
        except ImportError:
            raise ImportError(
                "scipy is required for hierarchical clustering. Install it with `pip install scipy`."
            )

        self._load_decoder_weights()

    def _load_decoder_weights(self):
        """Load just the decoder weights from the checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        try:
            state = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # The structure might be different depending on how it was saved
            if "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state

            if "W_dec" not in state_dict:
                raise KeyError(f"W_dec not found in {self.model_path}")
                
            self.W_dec = state_dict["W_dec"].detach()
            logger.info(f"Loaded W_dec with shape {self.W_dec.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load decoder weights: {e}")
            raise

    def compute_similarity_matrix(self, normalize: bool = True) -> torch.Tensor:
        """Compute cosine similarity between all pairs of features."""
        if self.W_dec is None:
            raise ValueError("Decoder weights not loaded.")

        weights = self.W_dec
        if normalize:
            # W_dec should already be normalized, but we ensure it here
            weights = F.normalize(weights, p=2, dim=1)

        # Cosine similarity is just the dot product of normalized vectors
        sim_matrix = torch.mm(weights, weights.t())
        return sim_matrix

    def cluster_features(
        self, 
        n_clusters: int = None, 
        distance_threshold: float = None,
        method: str = "average",
        metric: str = "cosine"
    ) -> Tuple[Dict[int, List[int]], List[int]]:
        """
        Cluster features using hierarchical agglomerative clustering.
        
        Args:
            n_clusters: Number of clusters to form.
            distance_threshold: The linkage distance threshold above which clusters will not be merged.
                Provide either n_clusters or distance_threshold, not both.
            method: Linkage method ('average', 'complete', 'ward', etc.)
            metric: Distance metric to use ('cosine', 'euclidean', etc.)
            
        Returns:
            Tuple of:
                - Dictionary mapping cluster_id -> list of feature_ids
                - Order of features for visualization (leaves order)
        """
        if self.W_dec is None:
            raise ValueError("Decoder weights not loaded.")

        if n_clusters is None and distance_threshold is None:
             # Default to a reasonable number if neither is provided
             n_clusters = max(10, self.W_dec.shape[0] // 50)
             
        if n_clusters is not None and distance_threshold is not None:
             raise ValueError("Provide either n_clusters or distance_threshold, not both.")

        weights_np = self.W_dec.cpu().numpy()
        
        logger.info(f"Computing distance matrix and linkage ({method} linkage, {metric} metric)...")
        # Ward linkage only works with euclidean distance
        if method == "ward" and metric != "euclidean":
             logger.warning("Ward linkage requires euclidean metric. Forcing metric to euclidean.")
             metric = "euclidean"
             
        dist_matrix = pdist(weights_np, metric=metric)
        Z = linkage(dist_matrix, method=method)
        
        leaves_order = leaves_list(Z).tolist()

        if n_clusters is not None:
             logger.info(f"Forming {n_clusters} clusters...")
             labels = fcluster(Z, n_clusters, criterion="maxclust")
        else:
             logger.info(f"Forming clusters with distance threshold {distance_threshold}...")
             labels = fcluster(Z, distance_threshold, criterion="distance")

        # Group features by cluster
        clusters = {}
        for feature_id, cluster_id in enumerate(labels):
            cid = int(cluster_id)
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(int(feature_id))
            
        logger.info(f"Formed {len(clusters)} clusters.")
        return clusters, leaves_order

    def get_most_similar_features(self, feature_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get the top-K most similar features to a given feature (by cosine similarity)."""
        sim_matrix = self.compute_similarity_matrix()
        
        # Get similarities for the target feature
        similarities = sim_matrix[feature_id]
        
        # Zero out the self-similarity so it doesn't return itself as top match
        similarities[feature_id] = -float("inf")
        
        # Get top K
        top_k_sims, top_k_indices = torch.topk(similarities, top_k)
        
        results = []
        for i in range(top_k):
             fid = int(top_k_indices[i].item())
             sim = float(top_k_sims[i].item())
             results.append((fid, sim))
             
        return results

