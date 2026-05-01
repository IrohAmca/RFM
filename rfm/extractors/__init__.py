"""Extractor backends for activation extraction from language models."""

from rfm.extractors.transformer_lens import GPT2Extractor
from rfm.extractors.hf_causal import HFCausalExtractor
from rfm.extractors.hf_generate import HFGenerationExtractor


class ExtractorFactory:
    """Registry entry point for model extractors."""

    @staticmethod
    def create(config):
        backend = "transformer_lens"
        if hasattr(config, "get"):
            backend = config.get("extraction.extractor_backend", "transformer_lens")
        elif isinstance(config, dict):
            backend = config.get("extraction", {}).get(
                "extractor_backend", "transformer_lens"
            )
        backend_name = str(backend).lower()

        if backend_name in {"hf", "huggingface", "hf_causal"}:
            return HFCausalExtractor(config)

        if backend_name in {"hf_generate", "generate"}:
            return HFGenerationExtractor(config)

        if backend_name in {"groq_generate", "groq"}:
            from rfm.extractors.groq_generate import GroqGenerationExtractor

            return GroqGenerationExtractor(config)

        return GPT2Extractor(config)


def __getattr__(name):
    if name == "GroqGenerationExtractor":
        from rfm.extractors.groq_generate import GroqGenerationExtractor

        return GroqGenerationExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExtractorFactory",
    "GroqGenerationExtractor",
    "GPT2Extractor",
    "HFCausalExtractor",
    "HFGenerationExtractor",
]
