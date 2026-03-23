"""
Cross-lingual feature alignment and comparison.
Compares two trained SAE models to find features that represent the same concept across different models/languages.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from rfm.analysis.clustering import FeatureClustering

logger = logging.getLogger("rfm.analysis.crosslingual")


class CrossLingualAligner:
    def __init__(self, sae_path_A: str | Path, sae_path_B: str | Path, device: str = "cpu"):
        """
        Args:
            sae_path_A: Path to the first SAE checkpoint (e.g. English model).
            sae_path_B: Path to the second SAE checkpoint (e.g. Turkish model).
        """
        self.sae_path_A = Path(sae_path_A)
        self.sae_path_B = Path(sae_path_B)
        self.device = torch.device(device)
        
        # Load weights
        self.fc_A = FeatureClustering(self.sae_path_A, device=device)
        self.fc_B = FeatureClustering(self.sae_path_B, device=device)
        
        self.W_dec_A = self.fc_A.W_dec
        self.W_dec_B = self.fc_B.W_dec

    def compute_cka(self) -> float:
        """
        Compute Linear Centered Kernel Alignment (Linear CKA) between the two decoder weight spaces.
        This provides a single score of how structurally similar the entire feature spaces are.
        Not well defined if the input embedding dimensions are different, but we can compute it on the similarity matrices.
        """
        # Linear CKA is typically computed on activations, but here we compute it on the similarities
        sim_A = self.fc_A.compute_similarity_matrix()
        sim_B = self.fc_B.compute_similarity_matrix()
        
        # Center the similarity matrices
        n = sim_A.shape[0]
        H = torch.eye(n, device=self.device) - torch.ones((n, n), device=self.device) / n
        
        centered_A = H @ sim_A @ H
        centered_B = H @ sim_B @ H
        
        # Compute Hilbert-Schmidt Independence Criterion
        hsic = torch.sum(centered_A * centered_B)
        var_A = torch.sum(centered_A * centered_A)
        var_B = torch.sum(centered_B * centered_B)
        
        cka = hsic / torch.sqrt(var_A * var_B)
        return float(cka.item())

    def align_features(self, top_k: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """
        Finds aligned features between Model A and Model B using Bipartite matching or Greedy Cosine Similarity.
        This only makes sense if the models share an embedding space (e.g. multilingual model) or 
        if we project them into a shared space. 
        
        Assuming they share an embedding space (same hidden_dim and same tokenizer or aligned token embeddings):
        """
        if self.W_dec_A.shape[1] != self.W_dec_B.shape[1]:
            logger.warning(
                f"Embedding dimensions differ ({self.W_dec_A.shape[1]} vs {self.W_dec_B.shape[1]}). "
                "Cosine similarity requires shared embedding space. Alignment may be invalid."
            )
            
        weights_A = F.normalize(self.W_dec_A, p=2, dim=1)
        weights_B = F.normalize(self.W_dec_B, p=2, dim=1)
        
        # Cross-similarity matrix shape: (num_features_A, num_features_B)
        cross_sim = torch.mm(weights_A, weights_B.t())
        
        alignments = {}
        for i in range(cross_sim.shape[0]):
             # For feature i in Model A, find top K in Model B
             sims = cross_sim[i]
             top_vals, top_idx = torch.topk(sims, top_k)
             
             matches = []
             for v, idx in zip(top_vals, top_idx):
                  matches.append((int(idx.item()), float(v.item())))
                  
             alignments[i] = matches
             
        return alignments

    def generate_report(self, alignments: Dict[int, List[Tuple[int, float]]], threshold: float = 0.5) -> List[Dict]:
        """Generate a summarized list of strongly aligned feature pairs."""
        strong_pairs = []
        for feat_a, matches in alignments.items():
            best_match_b, score = matches[0]
            if score >= threshold:
                strong_pairs.append({
                    "feature_A": feat_a,
                    "feature_B": best_match_b,
                    "similarity": score
                })
                
        # Sort by similarity descending
        strong_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return strong_pairs

