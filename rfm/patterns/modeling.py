from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rfm.patterns.data import build_paired_activation_set
from rfm.patterns.spec import ContrastAxisSpec
from rfm.steering.hook import resolve_hf_target_module


def format_chat_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    lines = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    lines.append(f"User: {prompt}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n".join(lines)


def sigmoid(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.sigmoid(tensor).item())


@dataclass
class DirectionResult:
    method: str
    direction: torch.Tensor
    explained_variance: float
    cluster_separation: float
    validation_accuracy: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "direction": self.direction.detach().cpu(),
            "explained_variance": float(self.explained_variance),
            "cluster_separation": float(self.cluster_separation),
            "validation_accuracy": float(self.validation_accuracy),
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DirectionResult":
        return cls(
            method=str(payload["method"]),
            direction=payload["direction"].detach().cpu(),
            explained_variance=float(payload.get("explained_variance", 0.0)),
            cluster_separation=float(payload.get("cluster_separation", 0.0)),
            validation_accuracy=float(payload.get("validation_accuracy", 0.0)),
            threshold=float(payload.get("threshold", 0.0)),
        )


class AxisDirectionFinder:
    def __init__(self, aggregation: str = "mean", axis_spec: ContrastAxisSpec | None = None):
        self.aggregation = aggregation
        self.axis = axis_spec or ContrastAxisSpec(
            axis_id="contrast_axis",
            endpoint_a="endpoint_a",
            endpoint_b="endpoint_b",
            display_name_a="Endpoint A",
            display_name_b="Endpoint B",
        )
        self.directions: dict[str, DirectionResult] = {}

    def load_paired_activations(self, chunk_dir: str | Path, pattern: str = "*.pt") -> dict[str, Any]:
        paired = build_paired_activation_set(
            chunk_dir,
            self.axis,
            aggregation=self.aggregation,
            pattern=pattern,
        )
        return {
            "endpoint_a": paired.endpoint_a,
            "endpoint_b": paired.endpoint_b,
            "pair_ids": paired.pair_ids,
            "categories": paired.categories,
            "difficulties": paired.difficulties,
            "questions": paired.questions,
        }

    @staticmethod
    def _normalize(direction: torch.Tensor) -> torch.Tensor:
        return direction / direction.norm().clamp(min=1e-12)

    @staticmethod
    def _threshold(endpoint_a_proj: torch.Tensor, endpoint_b_proj: torch.Tensor) -> float:
        return float((endpoint_a_proj.mean() + endpoint_b_proj.mean()).item() / 2.0)

    def _orient(self, direction: torch.Tensor, endpoint_a_acts: torch.Tensor, endpoint_b_acts: torch.Tensor) -> torch.Tensor:
        score_a = endpoint_a_acts @ direction
        score_b = endpoint_b_acts @ direction
        if score_b.mean() < score_a.mean():
            return -direction
        return direction

    def _probe_direction(self, endpoint_a_acts: torch.Tensor, endpoint_b_acts: torch.Tensor) -> torch.Tensor:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return endpoint_b_acts.mean(dim=0) - endpoint_a_acts.mean(dim=0)

        x = torch.cat([endpoint_a_acts, endpoint_b_acts], dim=0).numpy()
        y = torch.cat(
            [
                torch.zeros(endpoint_a_acts.shape[0], dtype=torch.int64),
                torch.ones(endpoint_b_acts.shape[0], dtype=torch.int64),
            ],
            dim=0,
        ).numpy()
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(x, y)
        return torch.tensor(clf.coef_[0], dtype=endpoint_a_acts.dtype)

    def find_direction(
        self,
        endpoint_a_acts: torch.Tensor,
        endpoint_b_acts: torch.Tensor,
        method: str = "mean_diff",
    ) -> DirectionResult:
        if endpoint_a_acts.shape != endpoint_b_acts.shape:
            raise ValueError(
                "Paired endpoint tensors must have the same shape. "
                f"Got {tuple(endpoint_a_acts.shape)} vs {tuple(endpoint_b_acts.shape)}."
            )

        method = str(method).lower()
        if method == "mean_diff":
            direction = endpoint_b_acts.mean(dim=0) - endpoint_a_acts.mean(dim=0)
        elif method in {"pca", "ccs"}:
            diffs = endpoint_b_acts - endpoint_a_acts
            centered = diffs - diffs.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        elif method == "probe":
            direction = self._probe_direction(endpoint_a_acts, endpoint_b_acts)
        else:
            raise ValueError(f"Unsupported direction-finding method: {method}")

        direction = self._normalize(direction.detach().cpu())
        direction = self._orient(direction, endpoint_a_acts, endpoint_b_acts)
        diffs = endpoint_b_acts - endpoint_a_acts
        projected_diffs = diffs @ direction
        total_variance = diffs.var(dim=0, unbiased=False).sum().item()
        explained = 0.0 if total_variance <= 0 else float(projected_diffs.var(unbiased=False).item() / total_variance)

        validation = self.validate_direction(direction, endpoint_a_acts, endpoint_b_acts)
        return DirectionResult(
            method=method,
            direction=direction,
            explained_variance=explained,
            cluster_separation=validation["cluster_separation"],
            validation_accuracy=validation["accuracy"],
            threshold=validation["threshold"],
        )

    def validate_direction(
        self,
        direction: torch.Tensor,
        endpoint_a_acts: torch.Tensor,
        endpoint_b_acts: torch.Tensor,
    ) -> dict[str, Any]:
        direction = self._normalize(direction.detach().cpu())
        endpoint_a_proj = endpoint_a_acts @ direction
        endpoint_b_proj = endpoint_b_acts @ direction
        threshold = self._threshold(endpoint_a_proj, endpoint_b_proj)

        endpoint_a_pred = (endpoint_a_proj >= threshold).int()
        endpoint_b_pred = (endpoint_b_proj >= threshold).int()
        correct = int((endpoint_a_pred == 0).sum().item() + (endpoint_b_pred == 1).sum().item())
        total = endpoint_a_proj.numel() + endpoint_b_proj.numel()
        pooled_var = endpoint_a_proj.var(unbiased=False).item() + endpoint_b_proj.var(unbiased=False).item()
        mean_gap = endpoint_b_proj.mean().item() - endpoint_a_proj.mean().item()
        cluster_separation = float((mean_gap ** 2) / max(pooled_var, 1e-12))

        return {
            "accuracy": float(correct / max(total, 1)),
            "threshold": float(threshold),
            "endpoint_a_projection_mean": float(endpoint_a_proj.mean().item()),
            "endpoint_b_projection_mean": float(endpoint_b_proj.mean().item()),
            "cluster_separation": cluster_separation,
        }

    def fit_layer(
        self,
        layer_name: str,
        endpoint_a_acts: torch.Tensor,
        endpoint_b_acts: torch.Tensor,
        method: str = "mean_diff",
    ) -> DirectionResult:
        result = self.find_direction(endpoint_a_acts, endpoint_b_acts, method=method)
        self.directions[layer_name] = result
        return result

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({layer: result.to_dict() for layer, result in self.directions.items()}, output_path)
        return output_path

    def load(self, path: str | Path) -> dict[str, DirectionResult]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.directions = {layer: DirectionResult.from_dict(result) for layer, result in payload.items()}
        return self.directions


@dataclass
class AxisProbeState:
    weight: torch.Tensor
    bias: float
    backend: str
    training_accuracy: float
    cv_accuracy: float | None = None
    endpoint_a_label: str = "endpoint_a"
    endpoint_b_label: str = "endpoint_b"

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.detach().cpu(),
            "bias": float(self.bias),
            "backend": self.backend,
            "training_accuracy": float(self.training_accuracy),
            "cv_accuracy": None if self.cv_accuracy is None else float(self.cv_accuracy),
            "endpoint_a_label": self.endpoint_a_label,
            "endpoint_b_label": self.endpoint_b_label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AxisProbeState":
        return cls(
            weight=payload["weight"].detach().cpu(),
            bias=float(payload.get("bias", 0.0)),
            backend=str(payload.get("backend", "manual")),
            training_accuracy=float(payload.get("training_accuracy", 0.0)),
            cv_accuracy=None if payload.get("cv_accuracy") is None else float(payload["cv_accuracy"]),
            endpoint_a_label=str(payload.get("endpoint_a_label", "endpoint_a")),
            endpoint_b_label=str(payload.get("endpoint_b_label", "endpoint_b")),
        )


class AxisProbe:
    def __init__(self, axis_spec: ContrastAxisSpec | None = None):
        self.axis = axis_spec or ContrastAxisSpec(
            axis_id="contrast_axis",
            endpoint_a="endpoint_a",
            endpoint_b="endpoint_b",
            display_name_a="Endpoint A",
            display_name_b="Endpoint B",
        )
        self.state: AxisProbeState | None = None

    @staticmethod
    def _ensure_vector(activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim == 1:
            return activations.detach().cpu().float()
        if activations.ndim == 2:
            return activations.mean(dim=0).detach().cpu().float()
        raise ValueError(f"Expected 1D or 2D activations, got {tuple(activations.shape)}")

    def train(
        self,
        endpoint_a_acts: torch.Tensor,
        endpoint_b_acts: torch.Tensor,
        cv_folds: int = 5,
    ) -> AxisProbeState:
        x = torch.cat([endpoint_a_acts, endpoint_b_acts], dim=0).float()
        y = torch.cat(
            [
                torch.zeros(endpoint_a_acts.shape[0], dtype=torch.int64),
                torch.ones(endpoint_b_acts.shape[0], dtype=torch.int64),
            ],
            dim=0,
        )

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
        except ImportError:
            weight = endpoint_b_acts.mean(dim=0) - endpoint_a_acts.mean(dim=0)
            center = 0.5 * (endpoint_b_acts.mean(dim=0) + endpoint_a_acts.mean(dim=0))
            bias = -float(torch.dot(weight, center).item())
            scores = torch.sigmoid(x @ weight + bias)
            preds = (scores >= 0.5).int()
            accuracy = float((preds == y).float().mean().item())
            self.state = AxisProbeState(
                weight=weight.detach().cpu(),
                bias=bias,
                backend="mean_diff_fallback",
                training_accuracy=accuracy,
                cv_accuracy=None,
                endpoint_a_label=self.axis.endpoint_a,
                endpoint_b_label=self.axis.endpoint_b,
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
            cv_accuracy = float(cross_val_score(clf, x.numpy(), y.numpy(), cv=folds, scoring="accuracy").mean())

        self.state = AxisProbeState(
            weight=weight,
            bias=bias,
            backend="sklearn_logistic_regression",
            training_accuracy=accuracy,
            cv_accuracy=cv_accuracy,
            endpoint_a_label=self.axis.endpoint_a,
            endpoint_b_label=self.axis.endpoint_b,
        )
        return self.state

    def predict(self, activations: torch.Tensor) -> tuple[str, float]:
        if self.state is None:
            raise ValueError("Probe is not trained.")
        vector = self._ensure_vector(activations)
        score = float(torch.dot(self.state.weight, vector).item() + self.state.bias)
        probability = sigmoid(score)
        label = self.axis.endpoint_b if probability >= 0.5 else self.axis.endpoint_a
        return label, probability

    def explain_with_sae(self, sae_model, direction: torch.Tensor | None = None, top_k: int = 20) -> list[dict[str, Any]]:
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

    def save(self, path: str | Path) -> Path:
        if self.state is None:
            raise ValueError("Probe is not trained.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state.to_dict(), output_path)
        return output_path

    def load(self, path: str | Path) -> AxisProbeState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.state = AxisProbeState.from_dict(payload)
        return self.state


@dataclass
class AxisMonitorScore:
    contrast_probability: float
    per_layer_scores: dict[str, float]
    alert: bool
    explanation: str


class AxisMonitor:
    def __init__(
        self,
        *,
        axis_spec: ContrastAxisSpec,
        directions: dict[str, torch.Tensor | DirectionResult],
        probes: dict[str, AxisProbe] | None = None,
        thresholds: dict[str, float] | None = None,
        ensemble_method: str = "weighted_average",
    ):
        self.axis = axis_spec
        self.probes = probes or {}
        self.thresholds = thresholds or {}
        self.ensemble_method = ensemble_method
        self.directions: dict[str, torch.Tensor] = {}
        self.direction_thresholds: dict[str, float] = {}
        for layer_name, value in directions.items():
            if isinstance(value, DirectionResult):
                self.directions[layer_name] = value.direction.detach().cpu().float()
                self.direction_thresholds[layer_name] = float(value.threshold)
            elif hasattr(value, "direction") and hasattr(value, "threshold"):
                self.directions[layer_name] = value.direction.detach().cpu().float()
                self.direction_thresholds[layer_name] = float(value.threshold)
            else:
                self.directions[layer_name] = torch.as_tensor(value).detach().cpu().float()
                self.direction_thresholds[layer_name] = 0.0
        self.monitored_layers = list(dict.fromkeys([*self.directions.keys(), *self.probes.keys()]))
        self._captured: dict[str, list[torch.Tensor]] = {}

    def reset(self) -> None:
        self._captured = {layer_name: [] for layer_name in self.monitored_layers}

    def consume_activations(self) -> dict[str, torch.Tensor]:
        aggregated = {}
        for layer_name, chunks in self._captured.items():
            if chunks:
                aggregated[layer_name] = torch.cat(chunks, dim=0)
        return aggregated

    @staticmethod
    def _unwrap_activation(output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output from hooked module, got {type(output).__name__}")
        if output.ndim == 3:
            return output.detach().cpu().reshape(-1, output.shape[-1])
        if output.ndim == 2:
            return output.detach().cpu()
        raise ValueError(f"Unsupported hooked activation shape: {tuple(output.shape)}")

    def create_hooks(self, model) -> list:
        self.reset()
        handles = []
        for layer_name in self.monitored_layers:
            target_module = resolve_hf_target_module(model, layer_name)

            def hook_fn(module, inputs, output, layer_name=layer_name):
                self._captured[layer_name].append(self._unwrap_activation(output))
                return output

            handles.append(target_module.register_forward_hook(hook_fn))
        return handles

    def _score_layer(self, layer_name: str, activations: torch.Tensor) -> float:
        vector = activations.mean(dim=0).detach().cpu().float()
        scores = []

        direction = self.directions.get(layer_name)
        if direction is not None:
            raw_projection = float(torch.dot(vector, direction).item())
            calibrated = raw_projection - self.direction_thresholds.get(layer_name, 0.0)
            scores.append(sigmoid(calibrated))

        probe = self.probes.get(layer_name)
        if probe is not None:
            _, probability = probe.predict(vector)
            scores.append(probability)

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _ensemble_threshold(self, observed_layers: list[str]) -> float:
        layer_thresholds = [float(self.thresholds[layer]) for layer in observed_layers if layer in self.thresholds]
        if not layer_thresholds:
            return 0.5
        if self.ensemble_method == "max":
            return max(layer_thresholds)
        return sum(layer_thresholds) / len(layer_thresholds)

    def score_generation(self, activations: dict[str, torch.Tensor]) -> AxisMonitorScore:
        per_layer_scores = {}
        alert_layers = []
        for layer_name, layer_acts in activations.items():
            score = self._score_layer(layer_name, layer_acts)
            per_layer_scores[layer_name] = score
            if score >= float(self.thresholds.get(layer_name, 0.5)):
                alert_layers.append(layer_name)

        if not per_layer_scores:
            return AxisMonitorScore(
                contrast_probability=0.0,
                per_layer_scores={},
                alert=False,
                explanation="No monitored activations captured.",
            )

        if self.ensemble_method == "max":
            contrast_probability = max(per_layer_scores.values())
        else:
            contrast_probability = sum(per_layer_scores.values()) / len(per_layer_scores)

        ensemble_threshold = self._ensemble_threshold(list(per_layer_scores))
        alert = bool(alert_layers or contrast_probability >= ensemble_threshold)
        explanation = (
            "Alerting layers: " + ", ".join(alert_layers)
            if alert_layers
            else f"No layer crossed its alert threshold. Ensemble threshold={ensemble_threshold:.3f}."
        )
        return AxisMonitorScore(
            contrast_probability=float(contrast_probability),
            per_layer_scores=per_layer_scores,
            alert=alert,
            explanation=explanation,
        )

    def score_replay(self, model, tokenizer, prompt: str, response: str, system_prompt: str | None = None) -> AxisMonitorScore:
        device = next(model.parameters()).device
        prompt_text = format_chat_prompt(tokenizer, prompt=prompt, system_prompt=system_prompt, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        full_ids = tokenizer(prompt_text + response, return_tensors="pt")["input_ids"].to(device)
        response_length = max(int(full_ids.shape[1] - prompt_ids.shape[1]), 0)

        handles = self.create_hooks(model)
        try:
            with torch.no_grad():
                model(input_ids=full_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        captured = self.consume_activations()
        if response_length > 0:
            captured = {
                layer_name: layer_acts[-response_length:]
                for layer_name, layer_acts in captured.items()
                if layer_acts.shape[0] >= response_length
            }
        return self.score_generation(captured)

    def generate_with_monitoring(
        self,
        model,
        tokenizer,
        prompt: str,
        system_prompt: str | None = None,
        **gen_kwargs,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        prompt_text = format_chat_prompt(tokenizer, prompt=prompt, system_prompt=system_prompt, add_generation_prompt=True)
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **gen_kwargs,
            )

        response_ids = output_ids[:, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        score = self.score_replay(model=model, tokenizer=tokenizer, prompt=prompt, response=response_text, system_prompt=system_prompt)
        return {
            "response": response_text,
            "contrast_probability": score.contrast_probability,
            "per_layer_scores": score.per_layer_scores,
            "alert": score.alert,
            "explanation": score.explanation,
        }
