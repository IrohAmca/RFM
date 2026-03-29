"""Statistical validation and category-aware analysis of contrastive scores."""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger("rfm.safety.score_validator")

class ScoreValidator:
    """Validate contrastive scores for statistical robustness."""

    @staticmethod
    def bootstrap_confidence_interval(
        toxic_col: np.ndarray, 
        safe_col: np.ndarray, 
        n_bootstrap: int = 1000, 
        alpha: float = 0.05
    ) -> tuple[float, float]:
        """Compute bootstrap 95% CI for the risk score of a single feature.
        
        Args:
           toxic_col: Array of feature activations for toxic samples.
           safe_col: Array of feature activations for safe samples.
           n_bootstrap: Number of bootstrap iterations.
           alpha: Significance level for CI bounds (e.g. 0.05 -> 95% CI).
           
        Returns:
            (lower_bound, upper_bound) of the risk score.
        """
        n_t = len(toxic_col)
        n_s = len(safe_col)
        
        if n_t < 2 or n_s < 2:
            return (0.0, 0.0)
            
        boot_scores = []
        for _ in range(n_bootstrap):
            idx_t = np.random.randint(0, n_t, size=n_t)
            idx_s = np.random.randint(0, n_s, size=n_s)
            
            b_t = toxic_col[idx_t]
            b_s = safe_col[idx_s]
            
            t_rate = (b_t > 0).mean()
            s_rate = (b_s > 0).mean()
            
            t_mean, t_var = b_t.mean(), b_t.var()
            s_mean, s_var = b_s.mean(), b_s.var()
            
            pooled_std = np.sqrt(((n_t - 1) * t_var + (n_s - 1) * s_var) / max(n_t + n_s - 2, 1))
            cohens_d = (t_mean - s_mean) / max(pooled_std, 1e-8)
            
            rate_ratio = t_rate / max(s_rate, 1e-8)
            log_ratio = np.log(max(rate_ratio, 1e-8))
            
            boot_scores.append(log_ratio * abs(cohens_d))
            
        boot_scores.sort()
        lower_idx = int(n_bootstrap * (alpha / 2))
        upper_idx = int(n_bootstrap * (1 - alpha / 2))
        
        return float(boot_scores[lower_idx]), float(boot_scores[upper_idx])

    @staticmethod
    def category_breakdown(
        features: np.ndarray,
        labels: List[str],
        categories: List[str],
        min_activation_rate: float = 0.001
    ) -> pd.DataFrame:
        """Analyze feature activation rates across detailed harm categories.
        
        Args:
            features: [N_tokens, hidden_dim] activation matrix.
            labels: "toxic" or "safe" per token (or sequence expanded).
            categories: detailed harm category per token.
            
        Returns:
            DataFrame with rows=features, cols=categories, values=activation rates.
        """
        unique_cats = sorted(list(set(categories)))
        n_features = features.shape[1]
        
        results = []
        cat_masks = {cat: (np.array(categories) == cat) & (np.array(labels) == "toxic") for cat in unique_cats}
        
        for fid in range(n_features):
            col = features[:, fid]
            t_rate = (col > 0).mean()
            if t_rate < min_activation_rate:
                continue
                
            row = {"feature_id": fid}
            for cat, mask in cat_masks.items():
                if mask.sum() == 0:
                    row[cat] = 0.0
                else:
                    cat_rate = (col[mask] > 0).mean()
                    row[cat] = cat_rate
                    
            results.append(row)
            
        return pd.DataFrame(results)
