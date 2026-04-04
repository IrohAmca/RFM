from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _safe_axis_id(value: str) -> str:
    return str(value or "contrast_axis").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


@dataclass(frozen=True)
class ContrastAxisSpec:
    axis_id: str
    endpoint_a: str
    endpoint_b: str
    display_name_a: str
    display_name_b: str
    orientation: str = "b_minus_a"
    pair_key_fields: tuple[str, ...] = ("pair_id", "label", "question")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "axis_id", _safe_axis_id(self.axis_id))
        object.__setattr__(self, "endpoint_a", str(self.endpoint_a).strip())
        object.__setattr__(self, "endpoint_b", str(self.endpoint_b).strip())
        object.__setattr__(self, "display_name_a", str(self.display_name_a).strip() or self.endpoint_a)
        object.__setattr__(self, "display_name_b", str(self.display_name_b).strip() or self.endpoint_b)
        object.__setattr__(
            self,
            "pair_key_fields",
            tuple(str(field).strip() for field in self.pair_key_fields if str(field).strip()) or ("pair_id", "label", "question"),
        )

    @property
    def endpoint_labels(self) -> tuple[str, str]:
        return self.endpoint_a, self.endpoint_b

    @property
    def pair_group_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self.pair_key_fields if field != "label")

    @property
    def label_lookup(self) -> dict[str, int]:
        return {
            self.endpoint_a: 0,
            self.endpoint_b: 1,
        }

    def other_label(self, label: str) -> str:
        if str(label) == self.endpoint_a:
            return self.endpoint_b
        return self.endpoint_a

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "display_name_a": self.display_name_a,
            "display_name_b": self.display_name_b,
            "orientation": self.orientation,
            "pair_key_fields": list(self.pair_key_fields),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContrastAxisSpec":
        return cls(
            axis_id=str(payload.get("axis_id", "contrast_axis")),
            endpoint_a=str(payload.get("endpoint_a", "endpoint_a")),
            endpoint_b=str(payload.get("endpoint_b", "endpoint_b")),
            display_name_a=str(payload.get("display_name_a", payload.get("endpoint_a", "Endpoint A"))),
            display_name_b=str(payload.get("display_name_b", payload.get("endpoint_b", "Endpoint B"))),
            orientation=str(payload.get("orientation", "b_minus_a")),
            pair_key_fields=tuple(payload.get("pair_key_fields", ("pair_id", "label", "question"))),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def from_config(cls, config) -> "ContrastAxisSpec":
        section = config.get("contrast_axis", {}) if hasattr(config, "get") else {}
        if isinstance(section, dict) and section.get("endpoint_a") and section.get("endpoint_b"):
            return cls(
                axis_id=str(section.get("id", "contrast_axis")),
                endpoint_a=str(section["endpoint_a"]),
                endpoint_b=str(section["endpoint_b"]),
                display_name_a=str(section.get("display_name_a", section["endpoint_a"])),
                display_name_b=str(section.get("display_name_b", section["endpoint_b"])),
                orientation=str(section.get("orientation", "b_minus_a")),
                pair_key_fields=tuple(section.get("pair_key_fields", ("pair_id", "label", "question"))),
                metadata=dict(section.get("metadata", {}) or {}),
            )

        if hasattr(config, "get") and config.get("deception", None):
            return cls(
                axis_id="deception",
                endpoint_a="honest",
                endpoint_b="deceptive",
                display_name_a="Honest",
                display_name_b="Deceptive",
                orientation="b_minus_a",
                pair_key_fields=("pair_id", "label", "question"),
            )

        positive_label = config.get("contrastive.positive_label", "positive") if hasattr(config, "get") else "positive"
        negative_label = config.get("contrastive.negative_label", "negative") if hasattr(config, "get") else "negative"
        return cls(
            axis_id="contrast_axis",
            endpoint_a=str(negative_label),
            endpoint_b=str(positive_label),
            display_name_a=str(negative_label).replace("_", " ").title(),
            display_name_b=str(positive_label).replace("_", " ").title(),
            orientation="b_minus_a",
            pair_key_fields=("pair_id", "label", "question"),
        )
