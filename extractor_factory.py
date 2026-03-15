from models.gpt2.extractor import GPT2Extractor
from models.hf_causal.extractor import HFCausalExtractor


class ExtractorFactory:
    """Registry entry point for model extractors.

    For now, all HookedTransformer-compatible models use GPT2Extractor.
    Add new extractor classes here when a model family requires custom behavior.
    """

    @staticmethod
    def create(config):
        backend = "transformer_lens"
        if hasattr(config, "get"):
            backend = config.get("extraction.extractor_backend", "transformer_lens")
        elif isinstance(config, dict):
            backend = config.get("extraction", {}).get("extractor_backend", "transformer_lens")

        if str(backend).lower() in {"hf", "huggingface", "hf_causal"}:
            return HFCausalExtractor(config)

        return GPT2Extractor(config)
