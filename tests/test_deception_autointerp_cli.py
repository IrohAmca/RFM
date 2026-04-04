import json
from argparse import Namespace

import torch
from torch import nn

from cli.deception_autointerp import _resolve_scenarios_path, run_autointerp
from rfm.config import ConfigManager
from rfm.patterns import ContrastAxisSpec, update_pattern_bundle


class IdentitySAE(nn.Module):
    def forward(self, x):
        return x, x

    def to(self, device):
        return self

    def eval(self):
        return self


def test_resolve_scenarios_path_falls_back_to_nonempty_run_file(tmp_path):
    empty_cache = tmp_path / "runs" / "demo_model" / "deception" / "contextual_scenarios.jsonl"
    empty_cache.parent.mkdir(parents=True, exist_ok=True)
    empty_cache.write_text("", encoding="utf-8")
    fallback = tmp_path / "runs" / "demo_model" / "deception" / "scenarios.jsonl"
    fallback.write_text('{"question":"Q","honest_answer":"A","deceptive_answer":"B"}\n', encoding="utf-8")

    cfg = ConfigManager(
        {
            "model_name": "demo/model",
            "deception": {
                "scenario_generator": {
                    "cache_path": str(empty_cache),
                }
            },
        }
    )

    assert _resolve_scenarios_path(cfg) == str(fallback)


def test_run_autointerp_uses_local_backend_and_writes_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    torch.save(
        {
            "activations": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            "tokens": torch.zeros(2, dtype=torch.long),
            "metadata": {
                "labels": ["honest"],
                "token_lengths": [2],
                "pair_ids": [0],
                "questions": ["Q1"],
            },
        },
        acts_dir / "chunk.pt",
    )

    fallback = tmp_path / "runs" / "test_model" / "deception" / "scenarios.jsonl"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(
        '{"question":"Q1","honest_answer":"Honest","deceptive_answer":"Deceptive","category":"test","difficulty":"easy"}\n',
        encoding="utf-8",
    )

    cfg = ConfigManager(
        {
            "model_name": "test/model",
            "contrast_axis": {
                "id": "deception",
                "endpoint_a": "honest",
                "endpoint_b": "deceptive",
                "display_name_a": "Honest",
                "display_name_b": "Deceptive",
                "pair_key_fields": ["pair_id", "label", "question"],
            },
            "layers": {"blocks.0.hook_resid_post": {}},
            "extraction": {"output_dir": str(acts_dir)},
            "deception": {
                "scenario_generator": {
                    "cache_path": str(tmp_path / "runs" / "test_model" / "deception" / "contextual_scenarios.jsonl"),
                }
            },
            "train": {"device": "cpu"},
        }
    )
    axis = ContrastAxisSpec.from_config(cfg)
    update_pattern_bundle(
        cfg,
        axis_spec=axis,
        layer_updates={
            "blocks.0.hook_resid_post": {
                "feature_scores": [
                    {"feature_id": 7, "delta": 0.5, "effect_size": 0.5},
                    {"feature_id": 3, "delta": 0.4, "effect_size": 0.4},
                ]
            }
        },
    )

    monkeypatch.setattr("cli.deception_autointerp.resolve_best_checkpoint", lambda config, target=None: "fake.pt")
    monkeypatch.setattr("cli.deception_autointerp.load_sae_checkpoint", lambda *args, **kwargs: (IdentitySAE(), None))
    monkeypatch.setattr("rfm.deception.deception_autointerp.DeceptionFeatureAutoInterp.find_top_contexts", lambda self, feature_ids, top_k=8: {fid: [{"question": "Q1", "honest_answer": "Honest", "deceptive_answer": "Deceptive", "category": "test", "difficulty": "easy", "activation": 1.0}] for fid in feature_ids})
    monkeypatch.setattr("rfm.deception.deception_autointerp.DeceptionFeatureAutoInterp.interpret_features_locally", lambda self, **kwargs: {7: "Local interpretation", 3: "Second interpretation"})

    args = Namespace(
        top_n=2,
        top_k_contexts=1,
        request_delay=0.0,
        model="unused",
        local_max_new_tokens=48,
        no_resume=False,
    )

    run_autointerp(
        cfg,
        "blocks.0.hook_resid_post",
        args,
        api_key=None,
        base_url=None,
        use_local=True,
        runtime_model=object(),
        runtime_tokenizer=object(),
    )

    output_path = tmp_path / "runs" / "test_model" / "deception" / "autointerp" / "blocks_0_hook_resid_post_interpretations.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {"7": "Local interpretation", "3": "Second interpretation"}
