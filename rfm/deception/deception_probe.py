from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfm.deception.utils import sigmoid


@dataclass
class ProbeState:
    weight: torch.Tensor
    bias: float
    backend: str
    training_accuracy: float
    cv_accuracy: float | None = None

    def to_dict(self) -> dict:
        return {
            "weight": self.weight.detach().cpu(),
            "bias": float(self.bias),
            "backend": self.backend,
            "training_accuracy": float(self.training_accuracy),
            "cv_accuracy": None if self.cv_accuracy is None else float(self.cv_accuracy),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProbeState":
        return cls(
            weight=payload["weight"].detach().cpu(),
            bias=float(payload.get("bias", 0.0)),
            backend=str(payload.get("backend", "manual")),
            training_accuracy=float(payload.get("training_accuracy", 0.0)),
            cv_accuracy=None if payload.get("cv_accuracy") is None else float(payload["cv_accuracy"]),
        )


class DeceptionProbe:
    """Linear honest/deceptive probe with sklearn and mean-diff fallback."""

    def __init__(self):
        self.state: ProbeState | None = None

    @staticmethod
    def _ensure_vector(activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim == 1:
            return activations.detach().cpu().float()
        if activations.ndim == 2:
            return activations.mean(dim=0).detach().cpu().float()
        raise ValueError(f"Expected 1D or 2D activations, got {tuple(activations.shape)}")

    def train(
        self,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
        cv_folds: int = 5,
    ) -> ProbeState:
        x = torch.cat([honest_acts, deceptive_acts], dim=0).float()
        y = torch.cat(
            [
                torch.zeros(honest_acts.shape[0], dtype=torch.int64),
                torch.ones(deceptive_acts.shape[0], dtype=torch.int64),
            ],
            dim=0,
        )

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
        except ImportError:
            weight = deceptive_acts.mean(dim=0) - honest_acts.mean(dim=0)
            center = 0.5 * (deceptive_acts.mean(dim=0) + honest_acts.mean(dim=0))
            bias = -float(torch.dot(weight, center).item())
            scores = torch.sigmoid(x @ weight + bias)
            preds = (scores >= 0.5).int()
            accuracy = float((preds == y).float().mean().item())
            self.state = ProbeState(
                weight=weight.detach().cpu(),
                bias=bias,
                backend="mean_diff_fallback",
                training_accuracy=accuracy,
                cv_accuracy=None,
            )
            return self.state

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(x.numpy(), y.numpy())
        weight = torch.tensor(clf.coef_[0], dtype=torch.float32)
        bias = float(clf.intercept_[0])
        preds = torch.tensor(clf.predict(x.numpy()), dtype=torch.int64)
        accuracy = float((preds == y).float().mean().item())

        cv_accuracy = None
        min_class = min(int((y == 0).sum().item()), int((y == 1).sum().item()))
        if min_class >= 2:
            folds = max(2, min(int(cv_folds), min_class))
            cv_accuracy = float(
                cross_val_score(clf, x.numpy(), y.numpy(), cv=folds, scoring="accuracy").mean()
            )

        self.state = ProbeState(
            weight=weight,
            bias=bias,
            backend="sklearn_logistic_regression",
            training_accuracy=accuracy,
            cv_accuracy=cv_accuracy,
        )
        return self.state

    def predict(self, activations: torch.Tensor) -> tuple[str, float]:
        if self.state is None:
            raise ValueError("Probe is not trained.")
        vector = self._ensure_vector(activations)
        score = float(torch.dot(self.state.weight, vector).item() + self.state.bias)
        probability = sigmoid(score)
        label = "deceptive" if probability >= 0.5 else "honest"
        return label, probability

    def explain_with_sae(
        self,
        sae_model,
        direction: torch.Tensor | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        if direction is None:
            if self.state is None:
                raise ValueError("Provide a direction or train the probe first.")
            direction = self.state.weight
        direction = direction.detach().cpu().float()
        if not hasattr(sae_model, "W_dec"):
            raise ValueError("SAE model does not expose W_dec for decoder-direction analysis.")

        decoder = sae_model.W_dec.detach().cpu().float()
        direction = direction / direction.norm().clamp(min=1e-12)
        decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cosine = decoder @ direction
        top_idx = torch.topk(cosine.abs(), k=min(int(top_k), cosine.shape[0])).indices.tolist()
        return [
            {
                "feature_id": int(idx),
                "cosine_similarity": float(cosine[idx].item()),
                "alignment": "aligned" if cosine[idx].item() >= 0 else "opposed",
            }
            for idx in top_idx
        ]

    @staticmethod
    def multi_layer_ensemble(
        layer_probes: dict[str, "DeceptionProbe"],
        activations_by_layer: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
    ) -> dict:
        per_layer = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for layer_name, probe in layer_probes.items():
            if layer_name not in activations_by_layer:
                continue
            _, probability = probe.predict(activations_by_layer[layer_name])
            weight = float((weights or {}).get(layer_name, 1.0))
            per_layer[layer_name] = probability
            weighted_sum += probability * weight
            total_weight += weight

        final_probability = 0.0 if total_weight == 0 else weighted_sum / total_weight
        return {
            "deception_probability": final_probability,
            "per_layer_probabilities": per_layer,
            "label": "deceptive" if final_probability >= 0.5 else "honest",
        }

    def save(self, path: str | Path) -> Path:
        if self.state is None:
            raise ValueError("Probe is not trained.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state.to_dict(), output_path)
        return output_path

    def load(self, path: str | Path) -> ProbeState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.state = ProbeState.from_dict(payload)
        return self.state
