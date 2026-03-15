import torch

from base.extractor import BaseExtractor


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
        with torch.no_grad():
            clean_output = self.model(text)
        return self.next_token_id(clean_output)

    def convert_to_string(self, token_id):
        return self.model.to_string(token_id)

    def ablate_hook(self, act, hook):
        return torch.zeros_like(act)

    def ablate(self, text, target):
        self.model.reset_hooks()
        self.model.add_hook(target, self.ablate_hook)

        with torch.no_grad():
            ablated_output = self.model(text)

        self.model.reset_hooks()

        return ablated_output

    def extract(self, text, target):
        self.saved = {}
        self.model.reset_hooks()
        self.model.add_hook(target, self.capture_hook)

        with torch.no_grad():
            _ = self.model(text)

        self.model.reset_hooks()

        return self.saved[target]

    def to_tokens(self, text):
        return self.model.to_tokens(text)
