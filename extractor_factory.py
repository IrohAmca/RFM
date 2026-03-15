from models.gpt2.extractor import GPT2Extractor


class ExtractorFactory:
    """Registry entry point for model extractors.

    For now, all HookedTransformer-compatible models use GPT2Extractor.
    Add new extractor classes here when a model family requires custom behavior.
    """

    @staticmethod
    def create(config):
        return GPT2Extractor(config)
