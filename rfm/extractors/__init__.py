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
            backend = config.get("extraction", {}).get("extractor_backend", "transformer_lens")

        if str(backend).lower() in {"hf", "huggingface", "hf_causal"}:
            return HFCausalExtractor(config)

        if str(backend).lower() in {"hf_generate", "generate"}:
            return HFGenerationExtractor(config)

        return GPT2Extractor(config)


__all__ = ["ExtractorFactory", "GPT2Extractor", "HFCausalExtractor", "HFGenerationExtractor"]
