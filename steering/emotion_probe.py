"""Supervised emotion feature discovery via SAE feature-label correlation.

Given SAE feature activations and emotion labels, identifies which features
are most correlated with specific emotions.
"""

import csv
from collections import defaultdict
from pathlib import Path



class EmotionProbe:
    """Discover SAE features correlated with emotion labels.

    Workflow:
    1. Load feature mapping events CSV (from sae.mapping)
    2. Load emotion labels for corresponding sequences
    3. Compute per-feature, per-emotion statistics
    4. Rank features by mutual information or activation difference
    """

    def __init__(self, events_csv_path, label_map=None):
        """
        Args:
            events_csv_path: Path to feature_mapping_events.csv from sae.mapping.
            label_map: Dict mapping sequence_idx -> emotion_label string.
                       If None, must be provided via set_labels().
        """
        self.events_csv_path = Path(events_csv_path)
        self.label_map = label_map or {}
        self._events = None
        self._feature_emotion_stats = None

    def set_labels(self, label_map):
        """Set emotion labels for sequences.

        Args:
            label_map: Dict mapping sequence_idx (int) -> emotion_label (str).
        """
        self.label_map = label_map
        self._feature_emotion_stats = None  # invalidate cache

    def load_labels_from_dataset(self, dataset, label_field="label", label_names=None):
        """Load emotion labels from a HuggingFace dataset.

        Args:
            dataset: HuggingFace dataset object (non-streaming, indexed).
            label_field: Field name containing the emotion label or index.
            label_names: Optional list mapping label indices to names.

        Returns:
            self (for chaining).
        """
        label_map = {}
        for idx, row in enumerate(dataset):
            label_val = row.get(label_field)
            if label_names and isinstance(label_val, int):
                label_val = label_names[label_val] if label_val < len(label_names) else str(label_val)
            label_map[idx] = str(label_val)
        self.label_map = label_map
        self._feature_emotion_stats = None
        return self

    def _load_events(self):
        """Load events CSV into memory."""
        if self._events is not None:
            return self._events

        with self.events_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._events = list(reader)
        return self._events

    def compute_feature_emotion_stats(self):
        """Compute activation statistics per (feature_id, emotion_label) pair.

        Returns:
            Dict[int, Dict[str, dict]] — feature_id -> emotion -> stats
        """
        if self._feature_emotion_stats is not None:
            return self._feature_emotion_stats

        events = self._load_events()
        stats = defaultdict(lambda: defaultdict(lambda: {
            "count": 0,
            "strength_sum": 0.0,
            "strength_max": 0.0,
            "strengths": [],
        }))

        unlabeled_count = 0
        for event in events:
            seq_idx = int(event.get("sequence_idx", -1))
            emotion = self.label_map.get(seq_idx)
            if emotion is None:
                unlabeled_count += 1
                continue

            feature_id = int(event["feature_id"])
            strength = float(event["strength"])

            entry = stats[feature_id][emotion]
            entry["count"] += 1
            entry["strength_sum"] += strength
            entry["strength_max"] = max(entry["strength_max"], strength)
            entry["strengths"].append(strength)

        if unlabeled_count > 0:
            print(f"[EmotionProbe] Warning: {unlabeled_count} events had no emotion label.")

        self._feature_emotion_stats = dict(stats)
        return self._feature_emotion_stats

    def rank_features_by_emotion(self, emotion, top_k=20):
        """Rank features by their correlation strength with a specific emotion.

        Args:
            emotion: Emotion label string (e.g. 'sadness', 'joy').
            top_k: Number of top features to return.

        Returns:
            List of dicts with feature_id, count, mean_strength, specificity.
        """
        stats = self.compute_feature_emotion_stats()
        all_emotions = set()
        for feat_stats in stats.values():
            all_emotions.update(feat_stats.keys())

        results = []
        for feature_id, emotion_stats in stats.items():
            if emotion not in emotion_stats:
                continue

            target_stats = emotion_stats[emotion]
            target_count = target_stats["count"]
            target_mean = target_stats["strength_sum"] / max(target_count, 1)

            # Compute specificity: how much more active is this feature for
            # the target emotion vs other emotions?
            other_counts = []
            for other_emotion, other_stats in emotion_stats.items():
                if other_emotion != emotion:
                    other_counts.append(other_stats["count"])

            other_mean_count = sum(other_counts) / max(len(other_counts), 1)
            specificity = target_count / max(other_mean_count, 1)

            results.append({
                "feature_id": feature_id,
                "emotion": emotion,
                "count": target_count,
                "mean_strength": target_mean,
                "max_strength": target_stats["strength_max"],
                "specificity": specificity,
            })

        results.sort(key=lambda r: (r["specificity"], r["mean_strength"]), reverse=True)
        return results[:top_k]

    def get_all_emotions(self):
        """Return set of all unique emotion labels."""
        return set(self.label_map.values())

    def summary(self, top_k_per_emotion=10):
        """Generate a summary of top features per emotion.

        Returns:
            Dict[str, list] — emotion -> list of top feature dicts.
        """
        result = {}
        for emotion in sorted(self.get_all_emotions()):
            result[emotion] = self.rank_features_by_emotion(emotion, top_k=top_k_per_emotion)
        return result

    def write_summary_csv(self, output_path, top_k_per_emotion=20):
        """Write emotion-feature ranking to CSV.

        Args:
            output_path: Output CSV path.
            top_k_per_emotion: Number of top features per emotion.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = []
        for emotion in sorted(self.get_all_emotions()):
            ranked = self.rank_features_by_emotion(emotion, top_k=top_k_per_emotion)
            all_results.extend(ranked)

        if not all_results:
            print("[EmotionProbe] No results to write.")
            return

        fieldnames = ["emotion", "feature_id", "count", "mean_strength", "max_strength", "specificity"]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"[EmotionProbe] Summary written to {output_path}")
