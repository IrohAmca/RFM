"""Cross-layer feature combination analysis for safety.

Combines sparse feature vectors from multiple layers to detect
dangerous feature *combinations* that span layers — e.g.,
"weapon topic in layer 6 + instruction intent in layer 27 → dangerous".

Uses scikit-learn DecisionTreeClassifier for interpretable rules.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger("rfm.safety.cross_layer")


class CrossLayerAnalyzer:
    """Deprecated compatibility wrapper for the old safety-specific cross-layer analyzer."""

    def __init__(self, sae_models: dict[str, torch.nn.Module], device: str = "cuda"):
        """
        Args:
            sae_models: dict mapping target_name → trained SAE model
            device:     torch device
        """
        warnings.warn(
            "rfm.safety.cross_layer.CrossLayerAnalyzer is deprecated. "
            "Use cli.pattern_score and the canonical pattern bundle/report flow instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.sae_models = sae_models
        self.device = device
        for m in sae_models.values():
            m.to(device)
            m.eval()

    def _encode_layer(self, sae, activations, batch_size=4096):
        """Encode activations through one SAE."""
        all_f = []
        with torch.no_grad():
            for i in range(0, activations.shape[0], batch_size):
                batch = activations[i: i + batch_size].to(self.device)
                _, f = sae(batch)
                all_f.append(f.cpu())
        return torch.cat(all_f, dim=0)

    def _load_and_split(self, chunk_dir: str | Path):
        """Load chunks and return per-sequence label list + token_lengths."""
        from rfm.safety.contrastive import ContrastiveScorer
        acts, labels, lens = ContrastiveScorer.load_chunks(chunk_dir)
        return acts, labels, lens

    def _aggregate_per_sequence(self, features, token_lengths, method="max"):
        """Aggregate per-token features into per-sequence features.

        Args:
            features:      [N_tokens, hidden_dim]
            token_lengths: list of ints
            method:        'max' (max-pool) or 'mean' (mean-pool)

        Returns:
            [N_sequences, hidden_dim]
        """
        seq_features = []
        offset = 0
        for length in token_lengths:
            seg = features[offset: offset + length]
            if method == "max":
                seq_features.append(seg.max(dim=0).values)
            else:
                seq_features.append(seg.mean(dim=0))
            offset += length
        return torch.stack(seq_features, dim=0)

    def analyze(
        self,
        chunk_dirs: dict[str, str | Path],
        risky_features: dict[str, list[int]] | None = None,
        top_k_per_layer: int = 50,
        max_tree_depth: int = 5,
        positive_label: str = "toxic",
        negative_label: str = "safe",
    ) -> dict:
        """Run full cross-layer analysis.

        Args:
            chunk_dirs:       dict mapping target → chunk directory
            risky_features:   optional pre-computed risky feature IDs per layer
                              (from Phase 3 contrastive scoring)
            top_k_per_layer:  if risky_features not provided, use top-K active features
            max_tree_depth:   depth limit for interpretable decision tree

        Returns:
            dict with keys:
                - combinations:      list of cross-layer feature pairs with co-activation stats
                - classifier_report: decision tree accuracy + rules
                - feature_importance: per-feature importance ranking
        """
        targets = list(chunk_dirs.keys())
        logger.info(f"Cross-layer analysis over {len(targets)} layers")

        # ── Step 1: Load, encode, aggregate per-sequence ────────────────
        layer_seq_features = {}  # target → [N_seq, K_features] (K = subset)
        layer_feature_ids = {}   # target → list of feature IDs used
        labels = None
        n_sequences = None

        for target in targets:
            acts, chunk_labels, lens = self._load_and_split(chunk_dirs[target])

            if labels is None:
                labels = chunk_labels
                n_sequences = len(labels)
            else:
                # Verify alignment
                if len(chunk_labels) != n_sequences:
                    raise ValueError(
                        f"Layer {target} has {len(chunk_labels)} sequences "
                        f"but expected {n_sequences}. Extraction alignment broken!"
                    )

            sae = self.sae_models[target]
            all_features = self._encode_layer(sae, acts)  # [N_tokens, hidden_dim]
            seq_features = self._aggregate_per_sequence(all_features, lens, method="max")

            # Select subset of features (risky or top-active)
            if risky_features and target in risky_features:
                fids = risky_features[target][:top_k_per_layer]
            else:
                # Fallback: top-k by total activation magnitude
                total_act = seq_features.sum(dim=0)
                fids = total_act.topk(min(top_k_per_layer, seq_features.shape[1])).indices.tolist()

            layer_seq_features[target] = seq_features[:, fids]
            layer_feature_ids[target] = fids
            logger.info(f"  {target}: {seq_features.shape[0]} sequences, {len(fids)} features selected")

        # ── Step 2: Build combined feature matrix ───────────────────────
        # Concatenate features from all layers: [N_seq, sum(K_per_layer)]
        combined = torch.cat([layer_seq_features[t] for t in targets], dim=1).numpy()
        combined_feature_names = []
        for target in targets:
            short = target.split(".")[-2] if "." in target else target  # e.g. "6"
            for fid in layer_feature_ids[target]:
                combined_feature_names.append(f"L{short}_F{fid}")

        # Binary labels
        label_map = {positive_label: 1, negative_label: 0}
        y = np.array([label_map.get(la, -1) for la in labels])
        valid_mask = y >= 0
        X = combined[valid_mask]
        y = y[valid_mask]

        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        logger.info(
            "Combined matrix: %s, %s=%s, %s=%s",
            X.shape,
            positive_label,
            n_positive,
            negative_label,
            n_negative,
        )

        # ── Step 3: Pairwise co-activation analysis ─────────────────────
        combinations = self._pairwise_coactivation(X, y, combined_feature_names, targets, layer_feature_ids)

        # ── Step 4: Train interpretable classifier ──────────────────────
        classifier_report, feature_importance = self._train_classifier(
            X, y, combined_feature_names, max_tree_depth
        )

        return {
            "combinations": combinations,
            "classifier_report": classifier_report,
            "feature_importance": feature_importance,
        }

    def _pairwise_coactivation(self, X, y, feature_names, targets, layer_feature_ids):
        """Find cross-layer feature pairs that co-activate more in toxic samples."""
        target_list = list(targets)
        results = []

        # Get column ranges per layer
        layer_ranges = {}
        offset = 0
        for target in target_list:
            n_feats = len(layer_feature_ids[target])
            layer_ranges[target] = (offset, offset + n_feats)
            offset += n_feats

        toxic_mask = y == 1
        safe_mask = y == 0

        # Only check cross-layer pairs (not within same layer)
        for i, t1 in enumerate(target_list):
            for t2 in target_list[i + 1:]:
                r1_start, r1_end = layer_ranges[t1]
                r2_start, r2_end = layer_ranges[t2]

                for c1 in range(r1_start, r1_end):
                    for c2 in range(r2_start, r2_end):
                        # Co-activation: both features > 0
                        both_active = (X[:, c1] > 0) & (X[:, c2] > 0)
                        toxic_co = both_active[toxic_mask].mean()
                        safe_co = both_active[safe_mask].mean()

                        if toxic_co < 0.01:  # skip very rare
                            continue

                        ratio = toxic_co / max(safe_co, 1e-8)
                        if ratio > 2.0:  # only report significant pairs
                            results.append({
                                "feature_1": feature_names[c1],
                                "feature_2": feature_names[c2],
                                "toxic_coactivation": round(float(toxic_co), 4),
                                "safe_coactivation": round(float(safe_co), 4),
                                "ratio": round(float(ratio), 2),
                            })

        results.sort(key=lambda x: x["ratio"], reverse=True)
        logger.info(f"Found {len(results)} significant cross-layer combinations")
        return results[:200]  # cap to top 200

    def _train_classifier(self, X, y, feature_names, max_depth):
        """Train a decision tree and extract interpretable rules."""
        try:
            from sklearn.tree import DecisionTreeClassifier, export_text
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.warning("scikit-learn not installed. Skipping classifier training.")
            return {"error": "scikit-learn required"}, []

        clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=42)
        clf.fit(X, y)

        # Cross-validation score
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")

        # Extract rules as text
        rules = export_text(clf, feature_names=feature_names, max_depth=max_depth)

        # Feature importance
        importance = clf.feature_importances_
        important_features = sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1], reverse=True,
        )

        report = {
            "accuracy": round(float(clf.score(X, y)), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "n_samples": int(len(y)),
            "n_toxic": int((y == 1).sum()),
            "n_safe": int((y == 0).sum()),
            "n_features": int(X.shape[1]),
            "tree_depth": max_depth,
            "rules": rules,
        }

        importance_list = [
            {"feature": name, "importance": round(imp, 6)}
            for name, imp in important_features if imp > 0
        ]

        logger.info(
            f"Decision tree: accuracy={report['accuracy']}, "
            f"CV F1={report['cv_f1_mean']:.3f}±{report['cv_f1_std']:.3f}"
        )
        return report, importance_list
