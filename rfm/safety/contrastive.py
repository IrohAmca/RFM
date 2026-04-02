"""Contrastive safety scoring for any two labeled activation classes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("rfm.safety.contrastive")


class ContrastiveScorer:
    """Score SAE features by their association with one label vs another."""

    def __init__(self, sae_model, device: str = "cuda"):
        self.sae = sae_model
        self.device = device
        self.sae.to(device)
        self.sae.eval()

    @staticmethod
    def load_chunks(
        chunk_dir: str | Path,
        pattern: str = "*.pt",
        include_metadata: bool = False,
    ):
        """Load activation chunks from disk.

        Returns:
            activations: [N_total_tokens, d_model]
            labels:      list[str] per sequence
            token_lengths: list[int] per sequence
            metadata:    optional flattened sequence metadata
        """
        chunk_dir = Path(chunk_dir)
        files = sorted(chunk_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {chunk_dir}")

        all_acts = []
        all_labels = []
        all_lens = []
        sequence_metadata = {
            "pair_ids": [],
            "categories": [],
            "difficulties": [],
            "questions": [],
            "responses": [],
            "sources": [],
        }

        for file_path in files:
            data = torch.load(file_path, map_location="cpu", weights_only=False)
            all_acts.append(data["activations"])
            metadata = data.get("metadata", {})
            labels = list(metadata.get("labels", []))
            lens = [int(length) for length in metadata.get("token_lengths", [])]
            all_labels.extend(labels)
            all_lens.extend(lens)

            if include_metadata:
                for key in sequence_metadata:
                    values = metadata.get(key, [])
                    if isinstance(values, list):
                        sequence_metadata[key].extend(values)

        activations = torch.cat(all_acts, dim=0)
        if include_metadata:
            return activations, all_labels, all_lens, sequence_metadata
        return activations, all_labels, all_lens

    @staticmethod
    def split_by_label(activations, labels, token_lengths):
        """Split token activations into label-specific tensors."""
        groups = {}
        offset = 0
        for label, length in zip(labels, token_lengths):
            end = offset + int(length)
            groups.setdefault(label, []).append(activations[offset:end])
            offset = end
        return {key: torch.cat(value, dim=0) for key, value in groups.items() if value}

    def encode(self, activations: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Pass activations through the SAE encoder."""
        all_features = []
        with torch.no_grad():
            for start in range(0, activations.shape[0], batch_size):
                batch = activations[start: start + batch_size].to(self.device)
                _, features = self.sae(batch)
                all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    def compute_scores(
        self,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        min_activation_rate: float = 0.001,
        mode: str = "continuous",
        significance_test: bool = True,
        positive_label: str = "toxic",
        negative_label: str = "safe",
    ) -> list[dict]:
        """Compute per-feature contrastive metrics for two classes."""
        if mode == "binary":
            positive_features = (positive_features > 0).float()
            negative_features = (negative_features > 0).float()

        hidden_dim = positive_features.shape[1]
        n_positive = positive_features.shape[0]
        n_negative = negative_features.shape[0]

        logger.info(
            "Computing contrastive scores (%s): %s=%s, %s=%s, features=%s",
            mode,
            positive_label,
            n_positive,
            negative_label,
            n_negative,
            hidden_dim,
        )

        mannwhitneyu = None
        if significance_test:
            try:
                from scipy.stats import mannwhitneyu
            except ImportError:
                logger.warning("scipy not installed. Skipping significance_test.")
                significance_test = False

        results = []
        for feature_id in range(hidden_dim):
            pos_col = positive_features[:, feature_id]
            neg_col = negative_features[:, feature_id]

            pos_rate = (pos_col > 0).float().mean().item()
            neg_rate = (neg_col > 0).float().mean().item()
            if pos_rate < min_activation_rate and neg_rate < min_activation_rate:
                continue

            pos_active = pos_col[pos_col > 0]
            neg_active = neg_col[neg_col > 0]
            pos_strength = pos_active.mean().item() if len(pos_active) > 0 else 0.0
            neg_strength = neg_active.mean().item() if len(neg_active) > 0 else 0.0

            rate_ratio = pos_rate / max(neg_rate, 1e-8)
            strength_diff = pos_strength - neg_strength

            pos_mean = pos_col.mean().item()
            neg_mean = neg_col.mean().item()
            pos_var = pos_col.var().item() if n_positive > 1 else 1e-8
            neg_var = neg_col.var().item() if n_negative > 1 else 1e-8
            fisher = (pos_mean - neg_mean) ** 2 / max(pos_var + neg_var, 1e-12)

            pooled_std = np.sqrt(
                ((n_positive - 1) * pos_var + (n_negative - 1) * neg_var)
                / max(n_positive + n_negative - 2, 1)
            )
            cohens_d = (pos_mean - neg_mean) / max(pooled_std, 1e-8)
            log_ratio = np.log(max(rate_ratio, 1e-8))
            risk_score = log_ratio * abs(cohens_d)

            direction = positive_label if risk_score > 0 else negative_label

            p_value = 1.0
            if significance_test and (pos_rate > 0 or neg_rate > 0):
                if pos_var > 0 or neg_var > 0:
                    try:
                        _, p_value = mannwhitneyu(
                            pos_col.numpy(),
                            neg_col.numpy(),
                            alternative="two-sided",
                        )
                    except Exception:
                        p_value = 1.0

            results.append(
                {
                    "feature_id": feature_id,
                    f"{positive_label}_rate": round(pos_rate, 6),
                    f"{negative_label}_rate": round(neg_rate, 6),
                    "rate_ratio": round(rate_ratio, 4),
                    f"{positive_label}_strength": round(pos_strength, 4),
                    f"{negative_label}_strength": round(neg_strength, 4),
                    "strength_diff": round(strength_diff, 4),
                    "fisher_score": round(fisher, 6),
                    "cohens_d": round(cohens_d, 4),
                    "risk_score": round(risk_score, 4),
                    "p_value": float(p_value),
                    "direction": direction,
                    "positive_label": positive_label,
                    "negative_label": negative_label,
                }
            )

        results.sort(key=lambda row: abs(row["risk_score"]), reverse=True)
        return results

    def score_from_chunks(
        self,
        chunk_dir: str | Path,
        pattern: str = "*.pt",
        min_activation_rate: float = 0.001,
        mode: str = "continuous",
        positive_label: str = "toxic",
        negative_label: str = "safe",
    ) -> list[dict]:
        """Load chunks, encode them, and compute contrastive scores."""
        activations, labels, token_lengths = self.load_chunks(chunk_dir, pattern)
        if not labels:
            raise ValueError(
                f"No labels found in chunks at {chunk_dir}. "
                "Was the extraction run with classification enabled?"
            )

        groups = self.split_by_label(activations, labels, token_lengths)
        if positive_label not in groups:
            raise ValueError(f"No {positive_label} samples found. Check your extraction/classification.")
        if negative_label not in groups:
            raise ValueError(f"No {negative_label} samples found. Check your extraction/classification.")

        logger.info(
            "Loaded: %s=%s tokens, %s=%s tokens",
            positive_label,
            groups[positive_label].shape[0],
            negative_label,
            groups[negative_label].shape[0],
        )

        positive_features = self.encode(groups[positive_label])
        negative_features = self.encode(groups[negative_label])
        return self.compute_scores(
            positive_features,
            negative_features,
            min_activation_rate=min_activation_rate,
            mode=mode,
            positive_label=positive_label,
            negative_label=negative_label,
        )

    @staticmethod
    def save_scores(scores: list[dict], output_path: str | Path):
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not scores:
            logger.warning("No scores to save.")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0].keys()))
            writer.writeheader()
            writer.writerows(scores)

        logger.info("Saved %s feature scores to %s", len(scores), output_path)

    @staticmethod
    def top_dangerous(scores: list[dict], top_k: int = 50, direction: str = "toxic") -> list[dict]:
        return [row for row in scores if row["direction"] == direction][:top_k]
