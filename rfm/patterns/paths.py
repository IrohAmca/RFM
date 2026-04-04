from __future__ import annotations

from pathlib import Path

from rfm.layout import model_slug
from rfm.patterns.spec import ContrastAxisSpec


def axis_run_dir(config, axis_spec: ContrastAxisSpec | None = None, *parts: str) -> Path:
    axis = axis_spec or ContrastAxisSpec.from_config(config)
    base = Path("runs") / model_slug(config) / axis.axis_id
    for part in parts:
        if part:
            base = base / part
    return base


def pattern_artifact_paths(config, axis_spec: ContrastAxisSpec | None = None) -> dict[str, Path]:
    base = axis_run_dir(config, axis_spec, "patterns")
    return {
        "pattern_dir": base,
        "bundle": base / "pattern_bundle.pt",
        "report": base / "pattern_report.json",
    }
