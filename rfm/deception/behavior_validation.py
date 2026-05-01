from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can",
    "for", "from", "has", "have", "i", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "with", "you", "your",
}

_REFUSAL_RE = re.compile(
    r"\b("
    r"i\s+(?:can'?t|cannot|won'?t)\s+(?:help|assist|comply|provide)|"
    r"i\s+am\s+(?:unable|not\s+able)\s+to|"
    r"i\s+will\s+not\s+(?:help|assist|comply|provide)|"
    r"cannot\s+(?:help|assist|comply)|"
    r"not\s+appropriate\s+for\s+me\s+to"
    r")\b",
    flags=re.IGNORECASE,
)

_AGREEMENT_RE = re.compile(
    r"\b("
    r"yes|you(?:'re| are)\s+right|you(?:'re| are)\s+correct|"
    r"i\s+agree|absolutely|exactly|that'?s\s+right|"
    r"your\s+(?:memory|belief|claim|point)\s+is\s+(?:valid|correct)"
    r")\b",
    flags=re.IGNORECASE,
)

_CORRECTION_RE = re.compile(
    r"(^|\b)("
    r"no|actually|however|that\s+is\s+not\s+correct|"
    r"not\s+true|incorrect|false|wrong|"
    r"i\s+(?:can'?t|cannot)\s+agree|"
    r"\bis\s+not\b|\bare\s+not\b|\bdoes\s+not\b|\bdon'?t\b"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass
class BehaviorValidationResult:
    accepted: bool
    label: str
    score: float
    expected_similarity: float
    opposite_similarity: float
    refusal_detected: bool
    agreement_detected: bool
    correction_detected: bool
    reasons: list[str] = field(default_factory=list)
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        current = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def _words(text: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-zA-Z0-9']+", str(text).lower()):
        word = word.strip("'")
        if len(word) >= 2 and word not in _STOPWORDS:
            words.add(word)
    return words


def _similarity(response: str, expected: str) -> float:
    expected_words = _words(expected)
    response_words = _words(response)
    if not expected_words or not response_words:
        return 0.0
    overlap = len(expected_words & response_words)
    recall = overlap / max(len(expected_words), 1)
    jaccard = overlap / max(len(expected_words | response_words), 1)
    seq = SequenceMatcher(None, str(response).lower(), str(expected).lower()).ratio()
    return float(max(recall, jaccard, seq * 0.5))


class BehaviorValidator:
    """Lightweight accept/reject validation for generated contrastive outputs.

    The validator is intentionally conservative and local: it does not need an
    external judge model. It checks that a generated response is non-empty,
    is not a refusal for target-behavior labels, and is closer to the expected
    endpoint response than to the opposite endpoint.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_attempts: int = 3,
        min_response_tokens: int = 3,
        min_expected_similarity: float = 0.08,
        max_opposite_similarity_advantage: float = 0.15,
        reject_refusals_for_endpoint_b: bool = True,
        reject_refusal_labels: list[str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_attempts = max(1, int(max_attempts))
        self.min_response_tokens = max(0, int(min_response_tokens))
        self.min_expected_similarity = float(min_expected_similarity)
        self.max_opposite_similarity_advantage = float(max_opposite_similarity_advantage)
        self.reject_refusals_for_endpoint_b = bool(reject_refusals_for_endpoint_b)
        self.reject_refusal_labels = {
            str(label).strip().lower()
            for label in (reject_refusal_labels or [])
            if str(label).strip()
        }

    @classmethod
    def from_config(cls, config: Any, prefix: str) -> "BehaviorValidator":
        section = _cfg_get(config, prefix, {}) or {}
        if not isinstance(section, dict):
            section = {}
        refusal_labels = section.get("reject_refusal_labels")
        if isinstance(refusal_labels, str):
            refusal_labels = [refusal_labels]
        return cls(
            enabled=section.get("enabled", True),
            max_attempts=section.get("max_attempts", 3),
            min_response_tokens=section.get("min_response_tokens", 3),
            min_expected_similarity=section.get("min_expected_similarity", 0.08),
            max_opposite_similarity_advantage=section.get("max_opposite_similarity_advantage", 0.15),
            reject_refusals_for_endpoint_b=section.get("reject_refusals_for_endpoint_b", True),
            reject_refusal_labels=refusal_labels,
        )

    @staticmethod
    def _expected_keys(label: str, axis: Any) -> tuple[str, str]:
        label_norm = str(label).strip().lower()
        endpoint_a = str(getattr(axis, "endpoint_a", "honest")).strip().lower()
        endpoint_b = str(getattr(axis, "endpoint_b", "deceptive")).strip().lower()
        if label_norm == endpoint_a:
            return "a", "b"
        if label_norm == endpoint_b:
            return "b", "a"
        if label_norm in {"honest", "truthful"}:
            return "a", "b"
        return "b", "a"

    @staticmethod
    def _response_for(row: dict[str, Any], endpoint: str) -> str:
        if endpoint == "a":
            return str(
                row.get("honest_answer")
                or row.get("truthful_answer")
                or row.get("truthful_response")
                or ""
            )
        return str(
            row.get("deceptive_answer")
            or row.get("sycophantic_answer")
            or row.get("deceptive_response")
            or ""
        )

    @staticmethod
    def _is_sycophancy_case(label: str, row: dict[str, Any], axis: Any) -> bool:
        label_norm = str(label).lower()
        endpoint_b = str(getattr(axis, "endpoint_b", "")).lower()
        category = str(row.get("category", row.get("prompt_family", ""))).lower()
        return (
            "sycoph" in label_norm
            or ("sycoph" in endpoint_b and label_norm == endpoint_b)
            or ("sycoph" in category and label_norm in {endpoint_b, "deceptive"})
        )

    @staticmethod
    def _is_truthful_sycophancy_case(label: str, row: dict[str, Any], axis: Any) -> bool:
        label_norm = str(label).lower()
        endpoint_a = str(getattr(axis, "endpoint_a", "")).lower()
        endpoint_b = str(getattr(axis, "endpoint_b", "")).lower()
        category = str(row.get("category", row.get("prompt_family", ""))).lower()
        return ("sycoph" in category or "sycoph" in endpoint_b) and label_norm in {
            endpoint_a,
            "honest",
            "truthful",
        }

    def validate(
        self,
        *,
        response: str,
        label: str,
        row: dict[str, Any],
        axis: Any,
        attempt: int = 1,
    ) -> BehaviorValidationResult:
        if not self.enabled:
            return BehaviorValidationResult(
                accepted=True,
                label=str(label),
                score=1.0,
                expected_similarity=1.0,
                opposite_similarity=0.0,
                refusal_detected=False,
                agreement_detected=False,
                correction_detected=False,
                attempt=int(attempt),
            )

        text = str(response or "").strip()
        expected_endpoint, opposite_endpoint = self._expected_keys(label, axis)
        expected = self._response_for(row, expected_endpoint)
        opposite = self._response_for(row, opposite_endpoint)

        expected_similarity = _similarity(text, expected)
        opposite_similarity = _similarity(text, opposite)
        refusal_detected = bool(_REFUSAL_RE.search(text))
        agreement_detected = bool(_AGREEMENT_RE.search(text))
        correction_detected = bool(_CORRECTION_RE.search(text))
        token_count = len(re.findall(r"\S+", text))

        reasons: list[str] = []
        endpoint_b = str(getattr(axis, "endpoint_b", "deceptive")).strip().lower()
        label_norm = str(label).strip().lower()
        reject_refusal = label_norm in self.reject_refusal_labels or (
            self.reject_refusals_for_endpoint_b and label_norm == endpoint_b
        )

        if token_count < self.min_response_tokens:
            reasons.append("too_short")
        if reject_refusal and refusal_detected:
            reasons.append("refusal_for_target_behavior")
        if expected and expected_similarity < self.min_expected_similarity:
            reasons.append("low_expected_similarity")
        if opposite_similarity - expected_similarity > self.max_opposite_similarity_advantage:
            reasons.append("closer_to_opposite_endpoint")

        if self._is_sycophancy_case(label, row, axis):
            if correction_detected and not agreement_detected:
                reasons.append("corrected_false_premise")
            if not agreement_detected and expected_similarity < self.min_expected_similarity:
                reasons.append("missing_agreement_signal")
        elif self._is_truthful_sycophancy_case(label, row, axis):
            if agreement_detected and not correction_detected:
                reasons.append("agreed_with_false_premise")

        score = expected_similarity - opposite_similarity
        return BehaviorValidationResult(
            accepted=not reasons,
            label=str(label),
            score=float(score),
            expected_similarity=float(expected_similarity),
            opposite_similarity=float(opposite_similarity),
            refusal_detected=refusal_detected,
            agreement_detected=agreement_detected,
            correction_detected=correction_detected,
            reasons=reasons,
            attempt=int(attempt),
        )
