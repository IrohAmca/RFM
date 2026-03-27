"""Generation-time activation extractor.

Captures hidden states while the model **generates** tokens, not just reads them.
This is critical for safety analysis: we need features that *cause* harmful output,
not features that activate when *reading* harmful text.

Supports two modes:
  1. ``generate`` – let the model produce its own response to a prompt
  2. ``replay``  – feed an existing (prompt, response) pair and treat the response
                   tokens as the "generation" phase (cheaper, allows using labelled
                   datasets like BeaverTails directly)
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFGenerationExtractor:
    """Extracts per-token hidden states during generation."""

    def __init__(self, config):
        self.config = config
        self.model_name = self._cfg_get("model_name", "gpt2")
        self.device = self._cfg_get(
            "model.device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = self._resolve_dtype(
            self._cfg_get("extraction.dtype") or self._cfg_get("model.dtype")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    # ── helpers ──────────────────────────────────────────────────────────

    def _cfg_get(self, key, default=None):
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        if isinstance(self.config, dict):
            parts = key.split(".")
            obj = self.config
            for p in parts:
                if isinstance(obj, dict):
                    obj = obj.get(p, default)
                else:
                    return default
            return obj
        return default

    @staticmethod
    def _resolve_dtype(dtype_name):
        if dtype_name is None:
            return None
        mapping = {
            "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        return mapping.get(str(dtype_name).lower(), None)

    def _target_to_hidden_state_index(self, target: str) -> int:
        match = re.match(r"^blocks\.(\d+)\.hook_resid_post$", str(target))
        if match:
            return int(match.group(1)) + 1
        match = re.match(r"^layer\.(\d+)$", str(target))
        if match:
            return int(match.group(1)) + 1
        raise ValueError(
            f"Unsupported target: {target!r}. "
            "Use 'blocks.<idx>.hook_resid_post' or 'layer.<idx>'."
        )

    def to_tokens(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"]

    # ── core: replay mode ────────────────────────────────────────────────

    def extract_replay(
        self,
        prompt: str,
        response: str,
        target: str,
    ) -> Dict[str, torch.Tensor]:
        """Extract activations by replaying an existing (prompt, response) pair.

        The full sequence ``prompt + response`` is fed through the model in a
        single forward pass.  Only the hidden states corresponding to the
        *response* tokens are returned – these approximate generation-time
        activations because the model attends to the same causal context it
        would have seen during actual generation.

        Returns:
            dict with keys:
                - ``activations``: [n_response_tokens, d_model]
                - ``tokens``:      [n_response_tokens]   (token IDs of response)
                - ``prompt_tokens``: [n_prompt_tokens]
                - ``full_tokens``:   [n_total_tokens]
        """
        hs_idx = self._target_to_hidden_state_index(target)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        n_prompt = prompt_ids.shape[1]

        full_text = prompt + response
        full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"]
        full_ids = full_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None or hs_idx >= len(hidden_states):
            raise RuntimeError(
                f"Model returned {len(hidden_states) if hidden_states else 0} "
                f"hidden-state tensors, but index {hs_idx} was requested."
            )

        # Shape: [1, seq_len, d_model] → [seq_len, d_model]
        hs = hidden_states[hs_idx].squeeze(0).detach().cpu()

        # Slice out only the response portion
        response_activations = hs[n_prompt:]
        response_token_ids = full_ids.squeeze(0)[n_prompt:].detach().cpu()

        return {
            "activations": response_activations,
            "tokens": response_token_ids,
            "prompt_tokens": prompt_ids.squeeze(0).detach().cpu(),
            "full_tokens": full_ids.squeeze(0).detach().cpu(),
        }

    # ── core: generate mode ──────────────────────────────────────────────

    def extract_generate(
        self,
        prompt: str,
        target: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """Generate from a prompt and capture hidden states of generated tokens.

        Two-step process:
          1. Generate the full output (prompt + response).
          2. Re-run the full sequence through the model with
             ``output_hidden_states=True`` to capture all layers.

        Returns same dict structure as ``extract_replay``.
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        n_prompt = prompt_ids.shape[1]
        prompt_ids_device = prompt_ids.to(self.device)

        # Step 1: generate
        with torch.no_grad():
            gen_output = self.model.generate(
                input_ids=prompt_ids_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_ids = gen_output  # [1, total_len]

        # Step 2: re-run for hidden states
        hs_idx = self._target_to_hidden_state_index(target)

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None or hs_idx >= len(hidden_states):
            raise RuntimeError(
                f"Model returned {len(hidden_states) if hidden_states else 0} "
                f"hidden-state tensors, but index {hs_idx} was requested."
            )

        hs = hidden_states[hs_idx].squeeze(0).detach().cpu()

        response_activations = hs[n_prompt:]
        response_token_ids = full_ids.squeeze(0)[n_prompt:].detach().cpu()

        return {
            "activations": response_activations,
            "tokens": response_token_ids,
            "prompt_tokens": prompt_ids.squeeze(0).detach().cpu(),
            "full_tokens": full_ids.squeeze(0).detach().cpu(),
        }

    # ── legacy compat: acts as a drop-in for HFCausalExtractor ───────────

    def extract(self, text: str, target: str) -> torch.Tensor:
        """Forward-pass extraction (reading mode) for backward compat."""
        hs_idx = self._target_to_hidden_state_index(target)
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                **encoded, output_hidden_states=True, use_cache=False
            )

        return outputs.hidden_states[hs_idx].detach().cpu()
