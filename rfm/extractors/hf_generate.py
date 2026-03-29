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
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFGenerationExtractor:
    """Extracts per-token hidden states during generation."""

    def __init__(self, config):
        self.config = config
        self.model_name = self._cfg_get("model_name", "gpt2")

        # Device: extraction.device → model.device → train.device → auto
        _auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = (
            self._cfg_get("extraction.device")
            or self._cfg_get("model.device")
            or self._cfg_get("train.device")
            or _auto_device
        )
        # dtype: extraction.dtype → model.dtype
        _dtype_str = self._cfg_get("extraction.dtype") or self._cfg_get("model.dtype")
        self.dtype = self._resolve_dtype(_dtype_str)

        print(f"[gen-extractor] Device: {self.device} | dtype: {_dtype_str or 'default'}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[gen-extractor] Model loaded: {self.model_name} on {next(self.model.parameters()).device}")

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

    # ── internal: single forward pass, all hidden states ──────────────

    def _forward_all_hidden(self, input_ids: torch.Tensor):
        """Run model once, return all hidden-state tensors."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        return outputs.hidden_states

    def _slice_response(
        self,
        hidden_states,
        targets: Sequence[str],
        n_prompt: int,
        full_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract response-portion activations for multiple targets."""
        response_token_ids = full_ids.squeeze(0)[n_prompt:].detach().cpu()
        prompt_token_ids = prompt_ids.squeeze(0).detach().cpu()
        full_token_ids = full_ids.squeeze(0).detach().cpu()

        results = {}
        for target in targets:
            hs_idx = self._target_to_hidden_state_index(target)
            if hs_idx >= len(hidden_states):
                raise RuntimeError(
                    f"Model returned {len(hidden_states)} hidden-state tensors, "
                    f"but target {target!r} resolves to index {hs_idx}."
                )
            hs = hidden_states[hs_idx].squeeze(0).detach().cpu()
            results[target] = {
                "activations": hs[n_prompt:],
                "tokens": response_token_ids,
                "prompt_tokens": prompt_token_ids,
                "full_tokens": full_token_ids,
            }
        return results

    # ── multi-layer replay (SINGLE forward pass) ─────────────────────

    def extract_replay_multi(
        self,
        prompt: str,
        response: str,
        targets: Sequence[str],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract activations for ALL target layers in ONE forward pass.

        Returns:
            dict mapping ``target_name`` → ``{activations, tokens, ...}``
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        n_prompt = prompt_ids.shape[1]

        full_text = prompt + response
        full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"]

        hidden_states = self._forward_all_hidden(full_ids)
        return self._slice_response(hidden_states, targets, n_prompt, full_ids, prompt_ids)

    # ── single-target replay (backward compat) ───────────────────────

    def extract_replay(
        self,
        prompt: str,
        response: str,
        target: str,
    ) -> Dict[str, torch.Tensor]:
        """Extract activations for a single layer (delegates to multi)."""
        multi = self.extract_replay_multi(prompt, response, [target])
        return multi[target]

    # ── multi-layer generate (SINGLE forward pass for extraction) ─────

    def extract_generate_multi(
        self,
        prompt: str,
        targets: Sequence[str],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate from a prompt and capture hidden states for ALL layers.

        Two-step process:
          1. Generate the full output (prompt + response).
          2. Re-run the full sequence *once* with ``output_hidden_states=True``.

        Returns:
            dict mapping ``target_name`` → ``{activations, tokens, ...}``
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

        # Step 2: single forward pass for all hidden states
        hidden_states = self._forward_all_hidden(full_ids)
        return self._slice_response(hidden_states, targets, n_prompt, full_ids, prompt_ids)

    # ── single-target generate (backward compat) ─────────────────────

    def extract_generate(
        self,
        prompt: str,
        target: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """Generate and extract for a single layer (delegates to multi)."""
        multi = self.extract_generate_multi(prompt, [target], max_new_tokens, temperature, top_p)
        return multi[target]

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
