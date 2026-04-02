from __future__ import annotations

from collections import Counter
from pathlib import Path

from rfm.deception.deception_monitor import DeceptionMonitor
from rfm.deception.scenario_generator import ScenarioGenerator, ScenarioRecord


class AdversarialSearch:
    """Search for deceptive scenarios that the current monitor misses."""

    def __init__(self, scenario_generator: ScenarioGenerator):
        self.scenario_generator = scenario_generator

    def search(
        self,
        monitor: DeceptionMonitor,
        target_model,
        tokenizer,
        n_attempts: int = 100,
        categories: list[str] | None = None,
        system_prompt_deceptive: str | None = None,
        append_to: str | Path | None = None,
    ) -> list[dict]:
        category_count = len(categories) if categories else 1
        scenarios = self.scenario_generator.generate(
            categories=categories,
            samples_per_category=max(1, int((n_attempts + category_count - 1) // category_count)),
            cache_path=None,
            resume=False,
        )
        scenarios = scenarios[: int(n_attempts)]

        missed = []
        for scenario in scenarios:
            score = monitor.score_replay(
                model=target_model,
                tokenizer=tokenizer,
                prompt=scenario.question,
                response=scenario.deceptive_answer,
                system_prompt=system_prompt_deceptive,
            )
            if not score.alert:
                missed.append(
                    {
                        "question": scenario.question,
                        "honest_answer": scenario.honest_answer,
                        "deceptive_answer": scenario.deceptive_answer,
                        "category": scenario.category,
                        "difficulty": scenario.difficulty,
                        "source": "adversarial_search",
                        "monitor_score": score.deception_probability,
                        "per_layer_scores": score.per_layer_scores,
                        "metadata": {
                            **scenario.metadata,
                            "monitor_explanation": score.explanation,
                        },
                    }
                )

        if append_to and missed:
            ScenarioGenerator.append_jsonl(
                append_to,
                [
                    ScenarioRecord.from_dict(sample)
                    for sample in missed
                ],
            )

        return missed

    @staticmethod
    def categorize_failures(missed_samples: list[dict]) -> dict:
        category_counts = Counter(sample.get("category", "unknown") for sample in missed_samples)
        difficulty_counts = Counter(sample.get("difficulty", "unknown") for sample in missed_samples)
        return {
            "total_missed": len(missed_samples),
            "by_category": dict(category_counts),
            "by_difficulty": dict(difficulty_counts),
        }
