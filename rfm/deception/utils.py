from __future__ import annotations

from pathlib import Path

import torch

from rfm.layout import model_slug


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
    if activations.ndim != 2:
        raise ValueError(f"Expected [N_tokens, d_model] activations, got {tuple(activations.shape)}")

    if method not in {"mean", "max", "last"}:
        raise ValueError(f"Unsupported aggregation method: {method!r}")

    rows = []
    offset = 0
    for length in token_lengths:
        segment = activations[offset: offset + int(length)]
        offset += int(length)
        if segment.numel() == 0:
            continue
        if method == "mean":
            rows.append(segment.mean(dim=0))
        elif method == "max":
            rows.append(segment.max(dim=0).values)
        else:
            rows.append(segment[-1])

    if not rows:
        return torch.empty(0, activations.shape[-1], dtype=activations.dtype)
    return torch.stack(rows, dim=0)


def deception_run_dir(config, *parts: str) -> Path:
    base = Path("runs") / model_slug(config) / "deception"
    for part in parts:
        if part:
            base = base / part
    return base


def sigmoid(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.sigmoid(tensor).item())
