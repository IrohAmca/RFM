from __future__ import annotations

import torch

from rfm.patterns import AxisProbe, AxisProbeState, ContrastAxisSpec


ProbeState = AxisProbeState


class DeceptionProbe(AxisProbe):
    """Deception benchmark adapter over the shared axis-neutral probe."""

    def __init__(self):
        super().__init__(
            axis_spec=ContrastAxisSpec(
                axis_id="deception",
                endpoint_a="honest",
                endpoint_b="deceptive",
                display_name_a="Honest",
                display_name_b="Deceptive",
            )
        )

    def train(
        self,
        honest_acts: torch.Tensor,
        deceptive_acts: torch.Tensor,
        cv_folds: int = 5,
    ) -> ProbeState:
        return super().train(honest_acts, deceptive_acts, cv_folds=cv_folds)
