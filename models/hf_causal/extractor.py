import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFCausalExtractor:
    """Extractor for Hugging Face causal language models.

    Supports extraction of post-residual block outputs via targets like:
    - blocks.<idx>.hook_resid_post

    For HF hidden states, block i post output corresponds to hidden_states[i + 1].
    """

    def __init__(self, config):
        self.config = config
        self.model_name = self._cfg_get("model_name", "gpt2")
        self.device = self._cfg_get(
            "model.device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = self._resolve_dtype(self._cfg_get("model.dtype"))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def _cfg_get(self, key, default=None):
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def _resolve_dtype(self, dtype_name):
        if dtype_name is None:
            return None
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(str(dtype_name).lower(), None)

    def _target_to_hidden_state_index(self, target):
        match = re.match(r"^blocks\.(\d+)\.hook_resid_post$", str(target))
        if match:
            block_idx = int(match.group(1))
            return block_idx + 1

        raise ValueError(
            "Unsupported target for HFCausalExtractor. "
            "Use format 'blocks.<idx>.hook_resid_post'."
        )

    def to_tokens(self, text):
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"]

    def extract(self, text, target):
        hidden_state_index = self._target_to_hidden_state_index(target)
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        if hidden_state_index >= len(hidden_states):
            raise ValueError(
                f"Requested block index resolves to hidden_states[{hidden_state_index}], "
                f"but model returned {len(hidden_states)} hidden-state tensors."
            )

        return hidden_states[hidden_state_index].detach().cpu()
