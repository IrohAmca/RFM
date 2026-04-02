import json

import pytest

from rfm.deception.scenario_generator import ScenarioGenerator


class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class _FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)

    def create(self, **kwargs):
        if not self.scripted:
            raise RuntimeError("No scripted responses left.")
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeClient:
    def __init__(self, scripted):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(scripted)})()


def _scenario_json(question: str, category: str = "factual_lying") -> str:
    return json.dumps(
        [
            {
                "question": question,
                "honest_answer": f"honest::{question}",
                "deceptive_answer": f"deceptive::{question}",
                "category": category,
                "difficulty": "medium",
                "metadata": {"failure_mode": "test"},
            }
        ]
    )


def test_scenario_generator_appends_batches_and_resumes_after_failure(tmp_path, monkeypatch):
    cache_path = tmp_path / "scenarios.jsonl"
    scripted = [
        _scenario_json("Q1"),
        RuntimeError("synthetic failure"),
    ]
    generator = ScenarioGenerator(
        request_delay=0.0,
        max_retries=0,
        max_samples_per_request=1,
    )
    client = _FakeClient(scripted)
    monkeypatch.setattr(generator, "_get_client", lambda: client)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        generator.generate(
            categories=["factual_lying"],
            samples_per_category=2,
            cache_path=cache_path,
            resume=False,
        )

    partial = ScenarioGenerator.load_cache(cache_path)
    assert len(partial) == 1
    assert partial[0].question == "Q1"

    manifest = ScenarioGenerator.load_manifest(ScenarioGenerator.manifest_path(cache_path))
    assert manifest["status"] == "failed"
    assert manifest["completed_counts"]["factual_lying"] == 1
    assert manifest["last_error"] == "synthetic failure"

    resume_generator = ScenarioGenerator(
        request_delay=0.0,
        max_retries=0,
        max_samples_per_request=1,
    )
    resume_client = _FakeClient([_scenario_json("Q2")])
    monkeypatch.setattr(resume_generator, "_get_client", lambda: resume_client)

    all_rows = resume_generator.generate(
        categories=["factual_lying"],
        samples_per_category=2,
        cache_path=cache_path,
        resume=True,
    )

    assert [row.question for row in all_rows] == ["Q1", "Q2"]
    final_manifest = ScenarioGenerator.load_manifest(ScenarioGenerator.manifest_path(cache_path))
    assert final_manifest["status"] == "completed"
    assert final_manifest["completed_counts"]["factual_lying"] == 2


def test_scenario_generator_retries_rate_limit_with_server_delay(monkeypatch):
    sleeps = []
    scripted = [
        RuntimeError("429 rate_limit exceeded, try again in 2s"),
        _scenario_json("Q1"),
    ]
    generator = ScenarioGenerator(
        request_delay=0.0,
        max_retries=2,
        retry_base_delay=1.0,
        max_samples_per_request=1,
    )
    client = _FakeClient(scripted)
    monkeypatch.setattr(generator, "_get_client", lambda: client)
    monkeypatch.setattr("rfm.deception.scenario_generator.time.sleep", lambda seconds: sleeps.append(seconds))

    rows = generator.generate_category("factual_lying", 1)

    assert len(rows) == 1
    assert rows[0].question == "Q1"
    assert sleeps
    assert sleeps[0] == pytest.approx(3.0)


def test_scenario_generator_salvages_truncated_json_response(monkeypatch):
    truncated = (
        '[{"question":"Q1","honest_answer":"honest::Q1","deceptive_answer":"deceptive::Q1",'
        '"category":"factual_lying","difficulty":"medium","metadata":{"failure_mode":"test"}},'
        '{"question":"Q2","honest_answer":"honest::Q2"'
    )
    generator = ScenarioGenerator(
        request_delay=0.0,
        max_retries=0,
        max_samples_per_request=2,
    )
    client = _FakeClient([truncated])
    monkeypatch.setattr(generator, "_get_client", lambda: client)

    rows = generator.generate_category("factual_lying", 1)

    assert len(rows) == 1
    assert rows[0].question == "Q1"


def test_scenario_generator_retries_after_parse_failure(monkeypatch):
    sleeps = []
    scripted = [
        "not json at all",
        _scenario_json("Q1"),
    ]
    generator = ScenarioGenerator(
        request_delay=0.0,
        max_retries=1,
        retry_base_delay=1.0,
        max_samples_per_request=1,
    )
    client = _FakeClient(scripted)
    monkeypatch.setattr(generator, "_get_client", lambda: client)
    monkeypatch.setattr("rfm.deception.scenario_generator.time.sleep", lambda seconds: sleeps.append(seconds))

    rows = generator.generate_category("factual_lying", 1)

    assert len(rows) == 1
    assert rows[0].question == "Q1"
    assert sleeps == [1.0]
