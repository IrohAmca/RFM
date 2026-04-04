from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rfm.patterns import AxisMonitor, AxisMonitorScore, ContrastAxisSpec, DirectionResult


@dataclass
class DeceptionScore:
    deception_probability: float
    per_layer_scores: dict[str, float]
    alert: bool
    explanation: str


class DeceptionMonitor(AxisMonitor):
    """Deception benchmark adapter over the shared axis-neutral monitor."""

    def __init__(
        self,
        directions: dict[str, torch.Tensor | DirectionResult],
        probes=None,
        thresholds: dict[str, float] | None = None,
        ensemble_method: str = "weighted_average",
    ):
        super().__init__(
            axis_spec=ContrastAxisSpec(
                axis_id="deception",
                endpoint_a="honest",
                endpoint_b="deceptive",
                display_name_a="Honest",
                display_name_b="Deceptive",
            ),
            directions=directions,
            probes=probes,
            thresholds=thresholds,
            ensemble_method=ensemble_method,
        )

    def _wrap_score(self, score: AxisMonitorScore) -> DeceptionScore:
        return DeceptionScore(
            deception_probability=score.contrast_probability,
            per_layer_scores=score.per_layer_scores,
            alert=score.alert,
            explanation=score.explanation,
        )

    def score_generation(self, activations: dict[str, torch.Tensor]) -> DeceptionScore:
        return self._wrap_score(super().score_generation(activations))

    def score_replay(
        self,
        model,
        tokenizer,
        prompt: str,
        response: str,
        system_prompt: str | None = None,
    ) -> DeceptionScore:
        return self._wrap_score(super().score_replay(model, tokenizer, prompt, response, system_prompt=system_prompt))

    def generate_with_monitoring(
        self,
        model,
        tokenizer,
        prompt: str,
        system_prompt: str | None = None,
        **gen_kwargs,
    ) -> dict[str, Any]:
        result = super().generate_with_monitoring(model, tokenizer, prompt, system_prompt=system_prompt, **gen_kwargs)
        return {
            "response": result["response"],
            "deception_probability": result["contrast_probability"],
            "per_layer_scores": result["per_layer_scores"],
            "alert": result["alert"],
            "explanation": result["explanation"],
        }
