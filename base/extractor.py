from transformer_lens import HookedTransformer


class BaseExtractor:
    def __init__(self, config, model=None):
        self.config = config
        self.model_name = self._cfg_get("model_name", "gpt2-small")
        self.model = model

        if self.model is None:
            if self.model_name is None:
                raise ValueError(
                    "Model name must be provided in config if no model is passed."
                )
            self.load_model_from_lens()

    def _cfg_get(self, key, default=None):
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def load_model_from_lens(self):
        model_kwargs = {}
        model_device = self._cfg_get("model.device")
        model_dtype = self._cfg_get("model.dtype")

        if model_device is not None:
            model_kwargs["device"] = model_device
        if model_dtype is not None:
            model_kwargs["dtype"] = model_dtype

        self.model = HookedTransformer.from_pretrained(self.model_name, **model_kwargs)

    def ablate_hook(self, act, hook):
        raise NotImplementedError("Subclasses must implement the ablate_hook method.")

    def capture_hook(self, act, hook):
        raise NotImplementedError("Subclasses must implement the capture_hook method.")

    def extract(self, text, target):
        raise NotImplementedError("Subclasses must implement the extract method.")
