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
        raise NotImplementedError

    def capture_hook(self, act, hook):
        raise NotImplementedError

    def extract(self, text, target):
        raise NotImplementedError


class GPT2Extractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)

    def capture_hook(self, act, hook):
        self.saved[hook.name] = act.detach().cpu()
        return act

    def next_token_id(self, logits):
        return logits[0, -1].argmax().item()

    def next_token_str(self, logits):
        token_id = self.next_token_id(logits)
        return self.model.to_string(token_id)

    def clean_next_token(self, text):
        import torch
        with torch.no_grad():
            clean_output = self.model(text)
        return self.next_token_id(clean_output)

    def convert_to_string(self, token_id):
        return self.model.to_string(token_id)

    def ablate_hook(self, act, hook):
        import torch
        return torch.zeros_like(act)

    def ablate(self, text, target):
        import torch
        self.model.reset_hooks()
        self.model.add_hook(target, self.ablate_hook)
        with torch.no_grad():
            ablated_output = self.model(text)
        self.model.reset_hooks()
        return ablated_output

    def extract(self, text, target):
        import torch
        self.saved = {}
        self.model.reset_hooks()
        self.model.add_hook(target, self.capture_hook)
        with torch.no_grad():
            _ = self.model(text)
        self.model.reset_hooks()
        return self.saved[target]

    def to_tokens(self, text):
        return self.model.to_tokens(text)
