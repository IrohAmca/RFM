from __future__ import annotations

from pathlib import Path

import torch

from rfm.patterns.data import aggregate_sequence_activations as aggregate_sequence_activations_generic
from rfm.patterns.paths import axis_run_dir
from rfm.patterns.spec import ContrastAxisSpec


def format_chat_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Format a prompt for chat-tuned tokenizers, with a plain-text fallback."""
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


def aggregate_sequence_activations(
    activations: torch.Tensor,
    token_lengths: list[int],
    method: str = "mean",
) -> torch.Tensor:
    """Aggregate token-level activations into one vector per sequence."""
    return aggregate_sequence_activations_generic(activations, token_lengths, method=method)


def deception_run_dir(config, *parts: str) -> Path:
    axis = ContrastAxisSpec.from_config(config)
    if axis.axis_id != "deception":
        axis = ContrastAxisSpec(
            axis_id="deception",
            endpoint_a="honest",
            endpoint_b="deceptive",
            display_name_a="Honest",
            display_name_b="Deceptive",
        )
    return axis_run_dir(config, axis, *parts)


def sigmoid(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.sigmoid(tensor).item())
