from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from rfm.deception.utils import aggregate_sequence_activations

logger = logging.getLogger("rfm.deception.direction_finder")


@dataclass
class DirectionResult:
    method: str
    direction: torch.Tensor
    explained_variance: float
    cluster_separation: float
    validation_accuracy: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "direction": self.direction.detach().cpu(),
            "explained_variance": float(self.explained_variance),
            "cluster_separation": float(self.cluster_separation),
            "validation_accuracy": float(self.validation_accuracy),
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DirectionResult":
        return cls(
            method=str(payload["method"]),
            direction=payload["direction"].detach().cpu(),
            explained_variance=float(payload.get("explained_variance", 0.0)),
            cluster_separation=float(payload.get("cluster_separation", 0.0)),
            validation_accuracy=float(payload.get("validation_accuracy", 0.0)),
            threshold=float(payload.get("threshold", 0.0)),
        )


class DeceptionDirectionFinder:
    """Find per-layer directions that separate honest and deceptive activations."""

    def __init__(self, aggregation: str = "mean"):
        self.aggregation = aggregation
        self.directions: dict[str, DirectionResult] = {}

    def load_paired_activations(
        self,
        chunk_dir: str | Path,
        pattern: str = "*.pt",
    ) -> dict[str, object]:
        chunk_dir = Path(chunk_dir)
        files = sorted(chunk_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No deception activation chunks found in {chunk_dir}")

        pairs: dict[tuple[int, str], dict] = {}
        next_pair_id = 0

        for file_path in files:
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            labels = list(metadata.get("labels", []))
            token_lengths = [int(length) for length in metadata.get("token_lengths", [])]
            pair_ids = list(metadata.get("pair_ids", []))
            categories = list(metadata.get("categories", []))
            difficulties = list(metadata.get("difficulties", []))
            questions = list(metadata.get("questions", []))

            if not labels or not token_lengths:
                continue

            vectors = aggregate_sequence_activations(
                payload["activations"].float(),
                token_lengths,
                method=self.aggregation,
            )

            if len(pair_ids) != len(labels):
                pair_ids = list(range(next_pair_id, next_pair_id + len(labels)))
            if len(categories) != len(labels):
                categories = ["unknown"] * len(labels)
            if len(difficulties) != len(labels):
                difficulties = ["unknown"] * len(labels)
            if len(questions) != len(labels):
                questions = [""] * len(labels)

            next_pair_id = max(next_pair_id, max(pair_ids, default=-1) + 1)

            for vector, label, pair_id, category, difficulty, question in zip(
                vectors,
                labels,
                pair_ids,
                categories,
                difficulties,
                questions,
            ):
                key = (int(pair_id), str(question))
                record = pairs.setdefault(
                    key,
                    {
                        "pair_id": int(pair_id),
                        "question": question,
                        "category": category,
                        "difficulty": difficulty,
                    },
                )
                record[str(label)] = vector.detach().cpu()

        honest = []
        deceptive = []
        pair_ids_out = []
        categories_out = []
        difficulties_out = []
        questions_out = []

        for record in pairs.values():
            if "honest" not in record or "deceptive" not in record:
                continue
            honest.append(record["honest"])
            deceptive.append(record["deceptive"])
            pair_ids_out.append(record["pair_id"])
            categories_out.append(record["category"])
            difficulties_out.append(record["difficulty"])
            questions_out.append(record["question"])

        if not honest:
            raise ValueError(f"No complete honest/deceptive pairs found in {chunk_dir}")

        return {
            "honest": torch.stack(honest, dim=0),
            "deceptive": torch.stack(deceptive, dim=0),
            "pair_ids": pair_ids_out,
            "categories": categories_out,
            "difficulties": difficulties_out,
            "questions": questions_out,
        }

    @staticmethod
    def _normalize(direction: torch.Tensor) -> torch.Tensor:
        return direction / direction.norm().clamp(min=1e-12)

    @staticmethod
    def _orient(direction: torch.Tensor, honest_acts: torch.Tensor, deceptive_acts: torch.Tensor) -> torch.Tensor:
        honest_score = honest_acts @ direction
        deceptive_score = deceptive_acts @ direction
        if deceptive_score.mean() < honest_score.mean():
            return -direction
        return direction

    @staticmethod
    def _threshold(honest_proj: torch.Tensor, deceptive_proj: torch.Tensor) -> float:
        return float((honest_proj.mean() + deceptive_proj.mean()).item() / 2.0)

    def _probe_direction(self, honest_acts: torch.Tensor, deceptive_acts: torch.Tensor) -> torch.Tensor:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning("scikit-learn not installed; falling back to mean-diff probe direction.")
            return deceptive_acts.mean(dim=0) - honest_acts.mean(dim=0)

        x = torch.cat([honest_acts, deceptive_acts], dim=0).numpy()
        y = torch.cat(
            [
                torch.zeros(honest_acts.shape[0], dtype=torch.int64),
                torch.ones(deceptive_acts.shape[0], dtype=torch.int64),
            ],
            dim=0,
        ).numpy()
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(x, y)
        return torch.tensor(clf.coef_[0], dtype=honest_acts.dtype)

    def find_direction(
        self,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
        method: str = "mean_diff",
    ) -> DirectionResult:
        if honest_acts.shape != deceptive_acts.shape:
            raise ValueError(
                "Honest and deceptive activations must have the same shape. "
                f"Got {tuple(honest_acts.shape)} vs {tuple(deceptive_acts.shape)}."
            )

        method = str(method).lower()
        if method == "mean_diff":
            direction = deceptive_acts.mean(dim=0) - honest_acts.mean(dim=0)
        elif method in {"pca", "ccs"}:
            diffs = deceptive_acts - honest_acts
            centered = diffs - diffs.mean(dim=0, keepdim=True)
            _, _, v = torch.linalg.svd(centered, full_matrices=False)
            direction = v[0]
        elif method == "probe":
            direction = self._probe_direction(honest_acts, deceptive_acts)
        else:
            raise ValueError(f"Unsupported direction-finding method: {method}")

        direction = self._normalize(direction.detach().cpu())
        direction = self._orient(direction, honest_acts, deceptive_acts)

        diffs = deceptive_acts - honest_acts
        projected_diffs = diffs @ direction
        total_variance = diffs.var(dim=0, unbiased=False).sum().item()
        explained = 0.0 if total_variance <= 0 else float(projected_diffs.var(unbiased=False).item() / total_variance)

        validation = self.validate_direction(direction, honest_acts, deceptive_acts)
        result = DirectionResult(
            method=method,
            direction=direction,
            explained_variance=explained,
            cluster_separation=validation["cluster_separation"],
            validation_accuracy=validation["accuracy"],
            threshold=validation["threshold"],
        )
        return result

    def validate_direction(
        self,
        direction: torch.Tensor,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
    ) -> dict[str, object]:
        direction = self._normalize(direction.detach().cpu())
        honest_proj = honest_acts @ direction
        deceptive_proj = deceptive_acts @ direction
        threshold = self._threshold(honest_proj, deceptive_proj)

        honest_pred = (honest_proj >= threshold).int()
        deceptive_pred = (deceptive_proj >= threshold).int()
        correct = int((honest_pred == 0).sum().item() + (deceptive_pred == 1).sum().item())
        total = honest_proj.numel() + deceptive_proj.numel()
        accuracy = correct / max(total, 1)

        pooled_var = honest_proj.var(unbiased=False).item() + deceptive_proj.var(unbiased=False).item()
        mean_gap = deceptive_proj.mean().item() - honest_proj.mean().item()
        cluster_separation = float((mean_gap ** 2) / max(pooled_var, 1e-12))

        result = {
            "accuracy": float(accuracy),
            "threshold": float(threshold),
            "honest_projection_mean": float(honest_proj.mean().item()),
            "deceptive_projection_mean": float(deceptive_proj.mean().item()),
            "cluster_separation": cluster_separation,
        }

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return result

        sample = torch.cat([honest_acts, deceptive_acts], dim=0)
        labels = ["honest"] * honest_acts.shape[0] + ["deceptive"] * deceptive_acts.shape[0]
        if sample.shape[0] >= 4:
            coords = TSNE(
                n_components=2,
                perplexity=min(30, max(sample.shape[0] - 1, 2)),
                random_state=42,
                init="random",
            ).fit_transform(sample.numpy())
            result["tsne"] = [
                {"x": float(x), "y": float(y), "label": label}
                for (x, y), label in zip(coords, labels)
            ]
        return result

    def fit_layer(
        self,
        layer_name: str,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
        method: str = "mean_diff",
    ) -> DirectionResult:
        result = self.find_direction(honest_acts, deceptive_acts, method=method)
        self.directions[layer_name] = result
        return result

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {layer: result.to_dict() for layer, result in self.directions.items()},
            output_path,
        )
        return output_path

    def load(self, path: str | Path) -> dict[str, DirectionResult]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.directions = {
            layer: DirectionResult.from_dict(result)
            for layer, result in payload.items()
        }
        return self.directions
