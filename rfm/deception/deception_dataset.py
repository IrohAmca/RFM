from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from rfm.deception.scenario_generator import ScenarioRecord


class DeceptionDataset:
    """Load scenario-generator outputs and expose paired/replay iterators."""

    def __init__(
        self,
        config=None,
        input_path: str | Path | None = None,
        mode: str = "paired",
    ):
        self.config = config
        self.input_path = Path(input_path) if input_path else self._resolve_input_path()
        self.mode = mode
        self.scenarios: list[ScenarioRecord] = []

    def _resolve_input_path(self) -> Path:
        if self.config is None:
            raise ValueError("input_path is required when config is not provided.")

        if hasattr(self.config, "get"):
            raw_path = self.config.get("deception.scenario_generator.cache_path")
        elif isinstance(self.config, dict):
            raw_path = (
                self.config.get("deception", {})
                .get("scenario_generator", {})
                .get("cache_path")
            )
        else:
            raw_path = None

        if not raw_path:
            raise ValueError("deception.scenario_generator.cache_path is not configured.")
        return Path(raw_path)

    def load(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Deception scenario file not found: {self.input_path}")

        scenarios: list[ScenarioRecord] = []
        suffix = self.input_path.suffix.lower()
        if suffix == ".jsonl":
            with open(self.input_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    scenarios.append(ScenarioRecord.from_dict(json.loads(line)))
        elif suffix == ".json":
            payload = json.loads(self.input_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("scenarios", [])
            if not isinstance(payload, list):
                raise ValueError("JSON deception dataset must be a list or {'scenarios': [...]} object.")
            scenarios = [ScenarioRecord.from_dict(item) for item in payload]
        else:
            raise ValueError(f"Unsupported deception dataset format: {self.input_path.suffix}")

        self.scenarios = [
            scenario
            for scenario in scenarios
            if scenario.question and scenario.honest_answer and scenario.deceptive_answer
        ]

    def get_data(self) -> list[ScenarioRecord]:
        if not self.scenarios:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.scenarios

    def iter_scenarios(self) -> Iterator[dict]:
        if not self.scenarios:
            raise ValueError("Dataset not loaded. Call load() first.")

        for index, scenario in enumerate(self.scenarios):
            yield {
                "pair_id": index,
                "question": scenario.question,
                "honest_answer": scenario.honest_answer,
                "deceptive_answer": scenario.deceptive_answer,
                "category": scenario.category,
                "difficulty": scenario.difficulty,
                "source": scenario.source,
                "metadata": dict(scenario.metadata),
            }

    def iter_replay_rows(self) -> Iterator[dict]:
        for scenario in self.iter_scenarios():
            pair_id = scenario["pair_id"]
            common = {
                "pair_id": pair_id,
                "question": scenario["question"],
                "prompt": scenario["question"],
                "category": scenario["category"],
                "difficulty": scenario["difficulty"],
                "source": scenario["source"],
                "metadata": dict(scenario["metadata"]),
            }
            yield {
                **common,
                "label": "honest",
                "response": scenario["honest_answer"],
            }
            yield {
                **common,
                "label": "deceptive",
                "response": scenario["deceptive_answer"],
            }

    def __iter__(self) -> Iterator[dict]:
        if self.mode == "replay":
            return self.iter_replay_rows()
        return self.iter_scenarios()
