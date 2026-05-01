from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from rfm.sycophancy.feature_store import SparseTokenFeatures, response_topk_to_sparse


class NeuronpediaAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, kind: str = "api_error"):
        super().__init__(message)
        self.status = status
        self.kind = kind


@dataclass(frozen=True)
class NeuronpediaSource:
    model: str
    source_id: str
    source_set: str
    d_sae: int


Transport = Callable[[str, dict[str, Any], dict[str, str]], tuple[int, str | bytes]]


class NeuronpediaClient:
    """Small cache-aware JSON client for Neuronpedia inference-style APIs."""

    def __init__(
        self,
        *,
        base_url: str = "https://www.neuronpedia.org",
        cache_dir: str | Path,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        transport: Transport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key if api_key is not None else os.getenv("NEURONPEDIA_API_KEY")
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.transport = transport
        self.sleep_fn = sleep_fn

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _cache_path(self, path: str, payload: dict[str, Any]) -> Path:
        raw = json.dumps({"path": path, "payload": payload}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _request(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
        if self.transport is not None:
            status, content = self.transport(url, payload, headers)
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return int(status), str(content)

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST", headers=headers)
        try:
            with request.urlopen(req, timeout=60) as response:
                return int(response.status), response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), body
        except error.URLError as exc:
            raise NeuronpediaAPIError(str(exc), kind="connection_error") from exc

    def post_json(self, path: str, payload: dict[str, Any], *, use_cache: bool = True) -> dict[str, Any]:
        path = "/" + str(path).lstrip("/")
        cache_path = self._cache_path(path, payload)
        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            return cached

        url = f"{self.base_url}{path}"
        headers = self._headers()
        delay = self.retry_base_delay
        last_status: int | None = None
        last_content = ""
        for attempt in range(self.max_retries + 1):
            status, content = self._request(url, payload, headers)
            last_status = status
            last_content = content
            if status == 429 and attempt < self.max_retries:
                self.sleep_fn(delay)
                delay *= 2.0
                continue
            if status < 200 or status >= 300:
                raise NeuronpediaAPIError(content[:500], status=status, kind="http_error")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise NeuronpediaAPIError(
                    f"Neuronpedia returned non-JSON content for {path}: {content[:120]!r}",
                    status=status,
                    kind="malformed_response",
                ) from exc
            if not isinstance(parsed, dict):
                raise NeuronpediaAPIError(
                    f"Neuronpedia response for {path} must be a JSON object.",
                    status=status,
                    kind="malformed_response",
                )
            result = {
                "cache_hit": False,
                "status": status,
                "path": path,
                "request": payload,
                "response": parsed,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result

        raise NeuronpediaAPIError(last_content[:500], status=last_status, kind="rate_limited")

    def tokenize(self, *, model: str, source: str, text: str, path: str = "/api/search-topk-by-token") -> dict[str, Any]:
        """Tokenize text via the search-topk-by-token endpoint which returns tokens.

        The real response format is:
        {"source": "...", "results": [{"position": 0, "token": "Hello", "topFeatures": [...]}, ...]}
        We extract token strings from the results array.
        """
        result = self.post_json(path, {
            "modelId": model,
            "source": source,
            "text": text,
            "numResults": 1,
            "ignoreBos": True,
        })
        response = result["response"]
        results = response.get("results")
        if not isinstance(results, list):
            raise NeuronpediaAPIError("Tokenize response is missing results list.", kind="malformed_response")
        token_strings = [str(item.get("token", "")) for item in results]
        response["token_strings"] = token_strings
        return response

    def topk_by_token_batch(
        self,
        *,
        prompts: list[str],
        source: NeuronpediaSource,
        top_k: int,
        path: str = "/api/search-topk-by-token",
    ) -> dict[str, Any]:
        """Get top-k activating features per token via the official API.

        The real response format is:
        {"source": "...", "results": [
            {"position": 0, "token": "Hello", "topFeatures": [
                {"featureIndex": 3992, "activationValue": 27.6, ...}, ...]},
            ...
        ]}

        We normalise into the downstream format expected by response_topk_to_sparse:
        {"results": [{"tokens": [...], "results": [
            {"token": "Hello", "top_features": [
                {"feature_index": 3992, "activation_value": 27.6}]}, ...]}]}
        """
        text_input = prompts[0] if len(prompts) == 1 else list(prompts)
        payload = {
            "modelId": source.model,
            "source": source.source_id,
            "text": text_input,
            "numResults": int(top_k),
            "ignoreBos": True,
        }
        result = self.post_json(path, payload)
        response = result["response"]
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise NeuronpediaAPIError("Activation response is missing results list.", kind="malformed_response")
        tokens: list[str] = []
        per_token_rows: list[dict[str, Any]] = []
        for item in raw_results:
            tok = str(item.get("token", ""))
            tokens.append(tok)
            top_features = []
            for feat in (item.get("topFeatures") or []):
                top_features.append({
                    "feature_index": int(feat.get("featureIndex", 0)),
                    "activation_value": float(feat.get("activationValue", 0.0)),
                })
            per_token_rows.append({"token": tok, "top_features": top_features})
        return {
            "results": [{
                "tokens": tokens,
                "results": per_token_rows,
            }],
        }

    def extract_response_features(
        self,
        *,
        source: NeuronpediaSource,
        prompt_text: str,
        response_text: str,
        top_k: int,
        tokenize_path: str = "/api/search-topk-by-token",
        activation_path: str = "/api/search-topk-by-token",
    ) -> SparseTokenFeatures:
        full_text = prompt_text + response_text
        prompt_tokens = self.tokenize(
            model=source.model,
            source=source.source_id,
            text=prompt_text,
            path=tokenize_path,
        )
        activation = self.topk_by_token_batch(
            prompts=[full_text],
            source=source,
            top_k=top_k,
            path=activation_path,
        )
        item = activation["results"][0]
        token_rows = list(item.get("results", []) or [])
        tokens = list(item.get("tokens", []) or [])
        prompt_count = len(prompt_tokens.get("token_strings", []) or [])
        if tokens and prompt_count > len(tokens):
            raise NeuronpediaAPIError(
                f"Token alignment failed: prompt has {prompt_count} tokens but full text has {len(tokens)}.",
                kind="token_alignment",
            )
        if len(token_rows) != len(tokens):
            raise NeuronpediaAPIError(
                f"Activation rows and tokens are misaligned: {len(token_rows)} rows vs {len(tokens)} tokens.",
                kind="token_alignment",
            )
        return response_topk_to_sparse(
            topk_by_token=token_rows,
            token_offset=prompt_count,
            d_sae=source.d_sae,
            top_k=top_k,
            source_id=source.source_id,
            backend="neuronpedia",
        )

