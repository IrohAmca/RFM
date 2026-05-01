from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from rfm.steering.hook import resolve_hf_target_module
from rfm.sycophancy.feature_store import SparseTokenFeatures, dense_to_sparse_topk


SOURCE_FAMILY_TO_RELEASE = {
    "transcoder_all": "gemma-scope-2b-pt-transcoders",
    "mlp_out_all": "gemma-scope-2b-pt-mlp",
    "resid_post_all": "gemma-scope-2b-pt-res",
}

SOURCE_FAMILY_TO_SOURCE_KIND = {
    "transcoder_all": "transcoder",
    "mlp_out_all": "mlp",
    "resid_post_all": "res",
}


@dataclass(frozen=True)
class GemmaScopeSourceSpec:
    source_family: str
    layer: int
    width: str = "16k"
    l0: str = "medium"
    d_sae: int = 16384

    @property
    def source_id(self) -> str:
        kind = SOURCE_FAMILY_TO_SOURCE_KIND.get(self.source_family, self.source_family)
        return f"{self.layer}-gemmascope-{kind}-{self.width}"

    @property
    def release(self) -> str:
        return SOURCE_FAMILY_TO_RELEASE[self.source_family]

    @property
    def sae_id(self) -> str:
        return f"layer_{self.layer}_width_{self.width}_l0_{self.l0}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_family": self.source_family,
            "layer": self.layer,
            "width": self.width,
            "l0": self.l0,
            "d_sae": self.d_sae,
            "source_id": self.source_id,
            "release": self.release,
            "sae_id": self.sae_id,
        }


def _config_get(config, key: str, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    if isinstance(config, dict):
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    return default


def resolve_source_specs(
    config,
    *,
    source_family: str | None = None,
    all_families: bool = False,
) -> list[GemmaScopeSourceSpec]:
    section = _config_get(config, "sycophancy.gemma_scope", {}) or {}
    priority = list(section.get("source_priority", ["transcoder_all", "mlp_out_all", "resid_post_all"]))
    if source_family:
        families = [source_family]
    elif all_families:
        families = priority
    else:
        families = priority[:1]
    layers = [int(layer) for layer in section.get("layers", [9, 17, 22, 29])]
    width = str(section.get("width", "16k")).replace("width_", "")
    d_sae = int(section.get("d_sae", 16384))
    l0_preferences = [str(item).replace("l0_", "") for item in section.get("l0_fallback", ["medium", "small", "big", "large"])]
    primary_l0 = l0_preferences[0] if l0_preferences else "medium"
    specs = []
    for family in families:
        if family not in SOURCE_FAMILY_TO_RELEASE:
            continue
        for layer in layers:
            specs.append(GemmaScopeSourceSpec(source_family=family, layer=layer, width=width, l0=primary_l0, d_sae=d_sae))
    return specs


def fallback_l0_values(config) -> list[str]:
    section = _config_get(config, "sycophancy.gemma_scope", {}) or {}
    values = [
        str(item).replace("l0_", "")
        for item in section.get("l0_fallback", ["medium", "small", "big", "large"])
    ]
    return values or ["medium"]


def source_family_from_source_id(source_id: str) -> str:
    text = str(source_id)
    if "transcoder" in text:
        return "transcoder_all"
    if "-mlp-" in text:
        return "mlp_out_all"
    return "resid_post_all"


def layer_from_source_id(source_id: str) -> int:
    match = re.match(r"^(\d+)-", str(source_id))
    if not match:
        raise ValueError(f"Cannot infer Gemma Scope layer from source id: {source_id!r}")
    return int(match.group(1))


def spec_from_source_id(config, source_id: str, *, l0: str | None = None) -> GemmaScopeSourceSpec:
    section = _config_get(config, "sycophancy.gemma_scope", {}) or {}
    width = str(section.get("width", "16k")).replace("width_", "")
    d_sae = int(section.get("d_sae", 16384))
    return GemmaScopeSourceSpec(
        source_family=source_family_from_source_id(source_id),
        layer=layer_from_source_id(source_id),
        width=width,
        l0=l0 or fallback_l0_values(config)[0],
        d_sae=d_sae,
    )


class LocalGemmaScopeBackend:
    """Best-effort local Gemma Scope 2 encoder using SAELens-loaded SAEs.

    This backend is imported lazily. It is intentionally thin so tests can
    monkeypatch ``sae_loader``/``model`` without downloading Gemma.
    """

    def __init__(
        self,
        *,
        config,
        model=None,
        tokenizer=None,
        device: str | None = None,
        sae_loader=None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or _config_get(config, "extraction.device", _config_get(config, "train.device", "cpu"))
        self.sae_loader = sae_loader
        self._sae_cache: dict[str, Any] = {}

    def _load_sae(self, spec: GemmaScopeSourceSpec):
        last_error: Exception | None = None
        for l0 in fallback_l0_values(self.config):
            candidate = GemmaScopeSourceSpec(
                source_family=spec.source_family,
                layer=spec.layer,
                width=spec.width,
                l0=l0,
                d_sae=spec.d_sae,
            )
            key = f"{candidate.release}:{candidate.sae_id}"
            if key in self._sae_cache:
                return self._sae_cache[key], candidate
            try:
                if self.sae_loader is not None:
                    sae = self.sae_loader(candidate.release, candidate.sae_id)
                    cfg = {}
                else:
                    try:
                        from sae_lens import SAE
                    except ImportError as exc:
                        raise ImportError(
                            "Local Gemma Scope fallback requires the optional 'sae-lens' package. "
                            "Install the gemma-scope extra or use Neuronpedia/API cache."
                        ) from exc
                    sae, cfg, _ = SAE.from_pretrained(release=candidate.release, sae_id=candidate.sae_id)
                self._sae_cache[key] = (sae, cfg)
                return (sae, cfg), candidate
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"No Gemma Scope SAE could be loaded for {spec.source_id}."
        ) from last_error

    @staticmethod
    def _encode_sae(sae, activations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(sae, "encode"):
                return sae.encode(activations).detach().cpu()
            output = sae(activations)
            if isinstance(output, tuple) and len(output) >= 2:
                return output[1].detach().cpu()
            raise ValueError("Loaded SAE does not expose encode() or tuple forward output.")

    def _ensure_runtime(self):
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for local Gemma Scope fallback.") from exc
        model_name = _config_get(self.config, "model_name", "google/gemma-3-4b-it")
        dtype_name = str(_config_get(self.config, "extraction.dtype", "bfloat16")).lower()
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name, torch.float32)
        if str(self.device).startswith("cpu") and dtype in {torch.bfloat16, torch.float16}:
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def extract_response_features(
        self,
        *,
        spec: GemmaScopeSourceSpec,
        prompt_text: str,
        response_text: str,
        top_k: int,
    ) -> SparseTokenFeatures:
        self._ensure_runtime()
        (sae, cfg), loaded_spec = self._load_sae(spec)
        hook_name = ""
        if isinstance(cfg, dict):
            hook_name = str(
                cfg.get("hook_name") or cfg.get("metadata", {}).get("hook_name") or ""
            )
        elif cfg is not None:
            hook_name = str(getattr(cfg, "hook_name", "") or "")
        hook_name = hook_name or f"blocks.{loaded_spec.layer}.hook_resid_post"

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_ids = self.tokenizer(prompt_text + response_text, return_tensors="pt")["input_ids"].to(self.device)
        prompt_len = int(prompt_ids.shape[1])
        captured: list[torch.Tensor] = []
        target_module = resolve_hf_target_module(self.model, hook_name)

        def _hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.append(value.detach())
            return output

        handle = target_module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.model(input_ids=full_ids, use_cache=False)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(f"No activations captured for {hook_name}.")
        activations = captured[-1]
        if activations.ndim == 3:
            activations = activations.squeeze(0)
        response_acts = activations[prompt_len:].detach().to(self.device)
        features = self._encode_sae(sae, response_acts)
        token_strings = self.tokenizer.convert_ids_to_tokens(full_ids[0, prompt_len:].detach().cpu().tolist())
        return dense_to_sparse_topk(
            features,
            top_k=top_k,
            source_id=loaded_spec.source_id,
            token_strings=token_strings,
            backend="local_saelens",
        )
