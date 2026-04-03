from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rfm.deception.deception_probe import DeceptionProbe
from rfm.deception.direction_finder import DirectionResult
from rfm.deception.utils import format_chat_prompt, sigmoid
from rfm.steering.hook import resolve_hf_target_module


@dataclass
class DeceptionScore:
    deception_probability: float
    per_layer_scores: dict[str, float]
    alert: bool
    explanation: str


class DeceptionMonitor:
    """Hook-based deception monitor for HuggingFace causal language models."""

    def __init__(
        self,
        directions: dict[str, torch.Tensor | DirectionResult],
        probes: dict[str, DeceptionProbe] | None = None,
        thresholds: dict[str, float] | None = None,
        ensemble_method: str = "weighted_average",
    ):
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
        layer_thresholds = [
            float(self.thresholds[layer_name])
            for layer_name in observed_layers
            if layer_name in self.thresholds
        ]
        if not layer_thresholds:
            return 0.5
        if self.ensemble_method == "max":
            return max(layer_thresholds)
        return sum(layer_thresholds) / len(layer_thresholds)

    def score_generation(self, activations: dict[str, torch.Tensor]) -> DeceptionScore:
        per_layer_scores = {}
        alert_layers = []
        for layer_name, layer_acts in activations.items():
            score = self._score_layer(layer_name, layer_acts)
            per_layer_scores[layer_name] = score
            if score >= float(self.thresholds.get(layer_name, 0.5)):
                alert_layers.append(layer_name)

        if not per_layer_scores:
            return DeceptionScore(
                deception_probability=0.0,
                per_layer_scores={},
                alert=False,
                explanation="No monitored activations captured.",
            )

        if self.ensemble_method == "max":
            deception_probability = max(per_layer_scores.values())
        else:
            deception_probability = sum(per_layer_scores.values()) / len(per_layer_scores)

        ensemble_threshold = self._ensemble_threshold(list(per_layer_scores))
        alert = bool(alert_layers or deception_probability >= ensemble_threshold)
        explanation = (
            "Alerting layers: " + ", ".join(alert_layers)
            if alert_layers
            else f"No layer crossed its alert threshold. Ensemble threshold={ensemble_threshold:.3f}."
        )
        return DeceptionScore(
            deception_probability=float(deception_probability),
            per_layer_scores=per_layer_scores,
            alert=alert,
            explanation=explanation,
        )

    def score_replay(
        self,
        model,
        tokenizer,
        prompt: str,
        response: str,
        system_prompt: str | None = None,
    ) -> DeceptionScore:
        device = next(model.parameters()).device
        prompt_text = format_chat_prompt(
            tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
            add_generation_prompt=True,
        )
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
        prompt_text = format_chat_prompt(
            tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
            add_generation_prompt=True,
        )
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
        score = self.score_replay(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            response=response_text,
            system_prompt=system_prompt,
        )
        return {
            "response": response_text,
            "deception_probability": score.deception_probability,
            "per_layer_scores": score.per_layer_scores,
            "alert": score.alert,
            "explanation": score.explanation,
        }
