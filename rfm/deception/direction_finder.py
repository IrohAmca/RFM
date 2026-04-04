from __future__ import annotations

from pathlib import Path

import torch

from rfm.patterns import AxisDirectionFinder, ContrastAxisSpec, DirectionResult


class DeceptionDirectionFinder(AxisDirectionFinder):
    """Deception benchmark adapter over the shared axis-neutral direction finder."""

    def __init__(self, aggregation: str = "mean"):
        super().__init__(
            aggregation=aggregation,
            axis_spec=ContrastAxisSpec(
                axis_id="deception",
                endpoint_a="honest",
                endpoint_b="deceptive",
                display_name_a="Honest",
                display_name_b="Deceptive",
            ),
        )

    def load_paired_activations(self, chunk_dir: str | Path, pattern: str = "*.pt") -> dict[str, object]:
        paired = super().load_paired_activations(chunk_dir, pattern=pattern)
        return {
            "honest": paired["endpoint_a"],
            "deceptive": paired["endpoint_b"],
            "pair_ids": paired["pair_ids"],
            "categories": paired["categories"],
            "difficulties": paired["difficulties"],
            "questions": paired["questions"],
        }

    def find_direction(
        self,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
        method: str = "mean_diff",
    ) -> DirectionResult:
        return super().find_direction(honest_acts, deceptive_acts, method=method)

    def validate_direction(
        self,
        direction: torch.Tensor,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
    ) -> dict[str, object]:
        result = super().validate_direction(direction, honest_acts, deceptive_acts)
        result["honest_projection_mean"] = result.pop("endpoint_a_projection_mean")
        result["deceptive_projection_mean"] = result.pop("endpoint_b_projection_mean")
        return result

    def fit_layer(
        self,
        layer_name: str,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
        method: str = "mean_diff",
    ) -> DirectionResult:
        return super().fit_layer(layer_name, honest_acts, deceptive_acts, method=method)
