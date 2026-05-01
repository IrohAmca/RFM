"""Groq API-based generation extractor.

Drop-in replacement for text generation paths that call the Groq API instead
of loading a local model. It exposes tokenizer/model compatibility shims for
call sites that share prompt-formatting code with local Hugging Face extractors.
"""

from __future__ import annotations

import os
import time
from typing import Any


class _GroqTokenizerShim:
    """Minimal tokenizer shim for shared prompt-formatting code."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1

    def __call__(self, text: str, **kwargs):
        return {"input_ids": [[0] * max(1, len(str(text).split()))]}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "user":
                parts.append(f"[User]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
        if add_generation_prompt:
            parts.append("[Assistant]:")
        return "\n".join(parts)

    def decode(self, ids, **kwargs):
        return ""

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


class GroqGenerationExtractor:
    """Text generation via Groq API; no local model loading."""

    def __init__(self, config):
        self.config = config
        self.model_name = self._get("model_name", "google/gemma-3-4b-it")
        self.groq_model = self._get("generation.groq_model", "llama-3.3-70b-versatile")
        self.device = self._get("extraction.device", "cpu")

        api_key = self._get("sycophancy.neuronpedia.groq_api_key") or os.getenv(
            "GROQ_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY env var or "
                "'sycophancy.neuronpedia.groq_api_key' in config."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GroqGenerationExtractor. "
                "Install it with: pip install openai"
            ) from exc

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.tokenizer = _GroqTokenizerShim(self.model_name)
        self.model = None
        self._max_retries = int(self._get("generation.max_retries", 3))
        self._retry_delay = float(self._get("generation.retry_delay", 2.0))

        print(f"[groq-extractor] Using Groq API with model: {self.groq_model}")

    def _get(self, key: str, default: Any = None) -> Any:
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        if isinstance(self.config, dict):
            current: Any = self.config
            for part in key.split("."):
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return current
        return default

    def extract_generate_multi(self, *args, **kwargs):
        raise NotImplementedError(
            "Groq extractor does not support hidden state extraction."
        )

    @staticmethod
    def _messages_from_prompt(prompt_text: str) -> list[dict[str, str]]:
        system_prefix = "[System]: "
        user_prefix = "[User]: "
        assistant_prefix = "[Assistant]:"
        system_parts: list[str] = []
        user_parts: list[str] = []
        fallback_parts: list[str] = []

        for raw_line in str(prompt_text).splitlines():
            line = raw_line.strip()
            if not line or line == assistant_prefix:
                continue
            if line.startswith(system_prefix):
                system_parts.append(line[len(system_prefix) :].strip())
            elif line.startswith(user_prefix):
                user_parts.append(line[len(user_prefix) :].strip())
            else:
                fallback_parts.append(raw_line)

        messages: list[dict[str, str]] = []
        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})
        content = "\n".join(user_parts or fallback_parts).strip()
        messages.append({"role": "user", "content": content or str(prompt_text)})
        return messages

    def generate(self, prompt_text: str) -> str:
        """Call Groq API and return the generated assistant text."""
        max_tokens = int(self._get("generation.max_new_tokens", 96))
        temperature = float(self._get("generation.temperature", 0.7))
        top_p = float(self._get("generation.top_p", 0.95))
        messages = self._messages_from_prompt(prompt_text)

        delay = self._retry_delay
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    print(
                        f"[groq-extractor] Attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2.0
        raise RuntimeError(
            f"Groq API call failed after {self._max_retries + 1} attempts: {last_exc}"
        )
