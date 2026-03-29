import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFCausalExtractor:
    """Extractor for Hugging Face causal language models.

    Supports extraction of post-residual block outputs via targets like:
    - blocks.<idx>.hook_resid_post
    - layer.<idx>
    """

    def __init__(self, config):
        self.config = config
        self.model_name = self._cfg_get("model_name", "gpt2")

        # Device: check extraction.device → model.device → train.device → auto
        _auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = (
            self._cfg_get("extraction.device")
            or self._cfg_get("model.device")
            or self._cfg_get("train.device")
            or _auto_device
        )
        # dtype: check extraction.dtype → model.dtype
        _dtype_str = self._cfg_get("extraction.dtype") or self._cfg_get("model.dtype")
        self.dtype = self._resolve_dtype(_dtype_str)

        print(f"[extractor] Device: {self.device} | dtype: {_dtype_str or 'default'}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Ensure pad token exists (required for batched inference)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # left-pad for causal LMs

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[extractor] Model loaded: {self.model_name} on {next(self.model.parameters()).device}")

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
            return int(match.group(1)) + 1

        match = re.match(r"^layer\.(\d+)$", str(target))
        if match:
            return int(match.group(1)) + 1

        raise ValueError(
            f"Unsupported target for HFCausalExtractor: {target!r}. "
            "Use format 'blocks.<idx>.hook_resid_post' or 'layer.<idx>'."
        )

    def to_tokens(self, text):
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"]

    def extract(self, text, target):
        """Single-text extraction (backward compat). Prefer extract_batch_multi for speed."""
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

    def extract_batch_multi(
        self,
        texts: list[str],
        targets: list[str],
        max_length: int = 512,
    ) -> dict[str, list[torch.Tensor]]:
        """Batched multi-layer extraction: ONE forward pass for N texts × M layers.

        Instead of N×M individual forward passes, this runs the model exactly once
        per batch, returning all requested layers for all texts simultaneously.

        Args:
            texts:      List of input strings (one batch).
            targets:    Layer target names, e.g. ['blocks.6.hook_resid_post', ...].
            max_length: Truncation length (tokens). Keep ≤512 to fit 4GB VRAM.

        Returns:
            dict mapping target_name → list of [seq_len, d_model] tensors,
            one per input text (padding stripped).
        """
        # Tokenize all texts together with padding
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states  # tuple of [batch, seq, d_model]
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")

        results: dict[str, list[torch.Tensor]] = {t: [] for t in targets}

        # For each target layer, strip padding and collect per-sample tensors
        for target in targets:
            hs_idx = self._target_to_hidden_state_index(target)
            if hs_idx >= len(hidden_states):
                raise ValueError(
                    f"Target {target!r} → hidden_states[{hs_idx}] but only "
                    f"{len(hidden_states)} tensors returned."
                )
            layer_hs = hidden_states[hs_idx].detach().cpu()  # [batch, seq, d_model]

            # Strip left-padding using attention mask
            mask_cpu = attention_mask.cpu()
            for b_idx in range(len(texts)):
                n_real = int(mask_cpu[b_idx].sum().item())  # number of real tokens
                # Left-padded → real tokens are the LAST n_real positions
                results[target].append(layer_hs[b_idx, -n_real:, :])

        return results
