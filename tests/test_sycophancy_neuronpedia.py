import pytest

from rfm.sycophancy.neuronpedia import NeuronpediaAPIError, NeuronpediaClient, NeuronpediaSource


def test_neuronpedia_client_uses_cache(tmp_path):
    calls = {"count": 0}

    def transport(url, payload, headers):
        calls["count"] += 1
        return 200, '{"ok": true, "token_strings": ["x"]}'

    client = NeuronpediaClient(cache_dir=tmp_path, transport=transport)
    first = client.post_json("/api/search-topk-by-token", {"model": "m", "text": "x"})
    second = client.post_json("/api/search-topk-by-token", {"model": "m", "text": "x"})

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert calls["count"] == 1


def test_neuronpedia_client_retries_429(tmp_path):
    sleeps = []
    responses = [(429, "too many"), (200, '{"ok": true}')]

    def transport(url, payload, headers):
        return responses.pop(0)

    client = NeuronpediaClient(
        cache_dir=tmp_path,
        transport=transport,
        retry_base_delay=1.5,
        sleep_fn=sleeps.append,
    )
    result = client.post_json("/api/test", {"x": 1}, use_cache=False)

    assert result["response"]["ok"] is True
    assert sleeps == [1.5]


def test_neuronpedia_client_rejects_html_response(tmp_path):
    client = NeuronpediaClient(cache_dir=tmp_path, transport=lambda *args: (200, "<html>not json</html>"))

    with pytest.raises(NeuronpediaAPIError) as exc:
        client.post_json("/api/test", {"x": 1}, use_cache=False)

    assert exc.value.kind == "malformed_response"


def test_neuronpedia_extract_rejects_malformed_activation_schema(tmp_path):
    """Tokenize call returns valid results, but activation call returns unexpected format."""
    def transport(url, payload, headers):
        # Tokenize call (numResults=1)
        if payload.get("numResults") == 1:
            return 200, '{"source": "test", "results": [{"position": 0, "token": "A", "topFeatures": []}]}'
        # Activation call returns bad data
        return 200, '{"unexpected": []}'

    client = NeuronpediaClient(cache_dir=tmp_path, transport=transport)
    source = NeuronpediaSource(model="gemma-2-2b", source_id="9-gemmascope-res-16k", source_set="gemmascope-res-16k", d_sae=16)

    with pytest.raises(NeuronpediaAPIError, match="results"):
        client.extract_response_features(source=source, prompt_text="A", response_text=" B", top_k=4)


def test_neuronpedia_extract_slices_response_tokens(tmp_path):
    """Full end-to-end: tokenize + activation with real Neuronpedia response format."""
    def transport(url, payload, headers):
        # Tokenize call (numResults=1)
        if payload.get("numResults") == 1:
            return 200, '{"source": "test", "results": [{"position": 0, "token": "A", "topFeatures": []}]}'
        # Activation call with full Neuronpedia format
        return 200, (
            '{"source": "test", "results": ['
            '{"position": 0, "token": "A", "topFeatures": [{"featureIndex": 1, "activationValue": 9.0}]},'
            '{"position": 1, "token": " B", "topFeatures": [{"featureIndex": 2, "activationValue": 3.0}]}'
            ']}'
        )

    client = NeuronpediaClient(cache_dir=tmp_path, transport=transport)
    source = NeuronpediaSource(model="gemma-2-2b", source_id="9-gemmascope-res-16k", source_set="gemmascope-res-16k", d_sae=16)

    sparse = client.extract_response_features(source=source, prompt_text="A", response_text=" B", top_k=4)

    assert sparse.feature_indices.shape == (1, 4)
    assert int(sparse.feature_indices[0, 0]) == 2
    assert float(sparse.feature_values[0, 0]) == pytest.approx(3.0)
