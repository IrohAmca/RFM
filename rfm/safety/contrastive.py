"""Contrastive safety scoring: identify features that differentiate toxic vs safe generation.

Given saved activation chunks (with per-sample labels), loads a trained SAE,
encodes all activations into sparse feature space, and computes per-feature
metrics that quantify how strongly each feature is associated with toxic output.

Metrics:
    - activation_rate_ratio:  How much more often the feature fires on toxic text
    - mean_strength_diff:     How much stronger the feature fires on toxic text
    - fisher_score:           Class separability (standardized mean difference)
    - risk_score:             Combined score for ranking
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger("rfm.safety.contrastive")


class ContrastiveScorer:
    """Score SAE features by their association with toxic vs safe outputs."""

    def __init__(self, sae_model, device: str = "cuda"):
        self.sae = sae_model
        self.device = device
        self.sae.to(device)
        self.sae.eval()

    # ── Load and split activations ──────────────────────────────────────

    @staticmethod
    def load_chunks(chunk_dir: str | Path, pattern: str = "*.pt"):
        """Load all activation chunks from a directory.

        Returns:
            activations: [N_total_tokens, d_model]
            labels:      list of str per *sequence* (not per token)
            token_lengths: list of int per sequence
        """
        chunk_dir = Path(chunk_dir)
        files = sorted(chunk_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {chunk_dir}")

        all_acts = []
        all_labels = []
        all_lens = []

        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            all_acts.append(data["activations"])

            meta = data.get("metadata", {})
            labels = meta.get("labels", [])
            lens = meta.get("token_lengths", [])

            all_labels.extend(labels)
            all_lens.extend(lens)

        activations = torch.cat(all_acts, dim=0)
        return activations, all_labels, all_lens

    @staticmethod
    def split_by_label(activations, labels, token_lengths):
        """Split activations into per-label groups.

        Since labels are per-sequence but activations are per-token,
        we use token_lengths to slice the correct ranges.

        Returns:
            dict mapping label → [N_tokens_for_label, d_model]
        """
        groups = {}
        offset = 0

        for label, length in zip(labels, token_lengths):
            end = offset + length
            if label not in groups:
                groups[label] = []
            groups[label].append(activations[offset:end])
            offset = end

        return {k: torch.cat(v, dim=0) for k, v in groups.items() if v}

    # ── Encode through SAE ──────────────────────────────────────────────

    def encode(self, activations: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Pass activations through SAE encoder → sparse feature vectors.

        Returns: [N, hidden_dim] feature activations (on CPU)
        """
        all_features = []
        with torch.no_grad():
            for start in range(0, activations.shape[0], batch_size):
                batch = activations[start: start + batch_size].to(self.device)
                _, features = self.sae(batch)
                all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    # ── Compute contrastive metrics ─────────────────────────────────────

    def compute_scores(
        self,
        toxic_features: torch.Tensor,
        safe_features: torch.Tensor,
        min_activation_rate: float = 0.001,
        mode: str = "continuous",
        significance_test: bool = True,
    ) -> list[dict]:
        """Compute per-feature contrastive safety scores.

        Args:
            toxic_features: [N_toxic, hidden_dim] sparse feature activations
            safe_features:  [N_safe, hidden_dim]
            min_activation_rate: ignore features below this rate in both classes
            mode: "continuous" or "binary". If binary, activations > 0 become 1.
            significance_test: execute Mann-Whitney U test for p-values.

        Returns:
            List of dicts (one per feature), sorted by risk_score descending.
        """
        if mode == "binary":
            toxic_features = (toxic_features > 0).float()
            safe_features = (safe_features > 0).float()

        hidden_dim = toxic_features.shape[1]
        n_toxic = toxic_features.shape[0]
        n_safe = safe_features.shape[0]

        logger.info(f"Computing contrastive scores ({mode}): {n_toxic} toxic, {n_safe} safe tokens, {hidden_dim} features")

        results = []
        
        mannwhitneyu = None
        if significance_test:
            try:
                from scipy.stats import mannwhitneyu
            except ImportError:
                logger.warning("scipy not installed. Skipping significance_test.")
                significance_test = False

        for fid in range(hidden_dim):
            t_col = toxic_features[:, fid]
            s_col = safe_features[:, fid]

            # Activation rates
            t_rate = (t_col > 0).float().mean().item()
            s_rate = (s_col > 0).float().mean().item()

            # Skip very-low-activity features
            if t_rate < min_activation_rate and s_rate < min_activation_rate:
                continue

            # Mean strengths (only over active tokens)
            t_active = t_col[t_col > 0]
            s_active = s_col[s_col > 0]
            t_strength = t_active.mean().item() if len(t_active) > 0 else 0.0
            s_strength = s_active.mean().item() if len(s_active) > 0 else 0.0

            # ── Metrics ────────────────────────────────────────────────
            # 1. Rate ratio
            rate_ratio = t_rate / max(s_rate, 1e-8)

            # 2. Strength difference
            strength_diff = t_strength - s_strength

            # 3. Fisher score: (μ1 - μ2)² / (σ1² + σ2²)
            t_mean = t_col.mean().item()
            s_mean = s_col.mean().item()
            t_var = t_col.var().item() if n_toxic > 1 else 1e-8
            s_var = s_col.var().item() if n_safe > 1 else 1e-8
            fisher = (t_mean - s_mean) ** 2 / max(t_var + s_var, 1e-12)

            # 4. Cohen's d (standardized effect size)
            pooled_std = np.sqrt(((n_toxic - 1) * t_var + (n_safe - 1) * s_var) / max(n_toxic + n_safe - 2, 1))
            cohens_d = (t_mean - s_mean) / max(pooled_std, 1e-8)

            # 5. Risk score (combined): emphasizes both differential and separability
            log_ratio = np.log(max(rate_ratio, 1e-8))
            risk_score = log_ratio * abs(cohens_d)

            # Direction: positive = toxic-associated, negative = safe-associated
            direction = "toxic" if risk_score > 0 else "safe"
            
            # 6. Significance test (Mann-Whitney U)
            p_value = 1.0
            if significance_test and (t_rate > 0 or s_rate > 0):
                if t_var > 0 or s_var > 0:
                    try:
                        _, p_value = mannwhitneyu(t_col.numpy(), s_col.numpy(), alternative='two-sided')
                    except Exception:
                        p_value = 1.0

            results.append({
                "feature_id": fid,
                "toxic_rate": round(t_rate, 6),
                "safe_rate": round(s_rate, 6),
                "rate_ratio": round(rate_ratio, 4),
                "toxic_strength": round(t_strength, 4),
                "safe_strength": round(s_strength, 4),
                "strength_diff": round(strength_diff, 4),
                "fisher_score": round(fisher, 6),
                "cohens_d": round(cohens_d, 4),
                "risk_score": round(risk_score, 4),
                "p_value": float(p_value),
                "direction": direction,
            })

        # Sort by absolute risk_score descending
        results.sort(key=lambda x: abs(x["risk_score"]), reverse=True)
        return results

    # ── High-level API ──────────────────────────────────────────────────

    def score_from_chunks(
        self,
        chunk_dir: str | Path,
        pattern: str = "*.pt",
        min_activation_rate: float = 0.001,
        mode: str = "continuous",
    ) -> list[dict]:
        """End-to-end: load chunks → encode → score.

        Args:
            chunk_dir: Directory containing activation .pt chunks (with labels).
            pattern:   Glob pattern for chunk files.
            mode:      "continuous" or "binary"

        Returns:
            Sorted list of feature score dicts.
        """
        acts, labels, lens = self.load_chunks(chunk_dir, pattern)

        if not labels:
            raise ValueError(
                f"No labels found in chunks at {chunk_dir}. "
                "Was the extraction run with safety classification enabled?"
            )

        groups = self.split_by_label(acts, labels, lens)

        if "toxic" not in groups:
            raise ValueError("No toxic samples found. Check your extraction/classification.")
        if "safe" not in groups:
            raise ValueError("No safe samples found. Check your extraction/classification.")

        logger.info(
            f"Loaded: {groups['toxic'].shape[0]} toxic tokens, "
            f"{groups['safe'].shape[0]} safe tokens"
        )

        toxic_features = self.encode(groups["toxic"])
        safe_features = self.encode(groups["safe"])

        return self.compute_scores(toxic_features, safe_features, min_activation_rate, mode=mode)

    @staticmethod
    def save_scores(scores: list[dict], output_path: str | Path):
        """Save scores to CSV."""
        import csv
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not scores:
            logger.warning("No scores to save.")
            return

        fieldnames = list(scores[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scores)

        logger.info(f"Saved {len(scores)} feature scores to {output_path}")

    @staticmethod
    def top_dangerous(scores: list[dict], top_k: int = 50, direction: str = "toxic") -> list[dict]:
        """Extract top-K features associated with a given direction."""
        filtered = [s for s in scores if s["direction"] == direction]
        return filtered[:top_k]
