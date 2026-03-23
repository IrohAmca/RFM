import torch.nn as nn

from rfm.steering.hook import resolve_hf_target_module
from rfm.viz.plots import final_metrics


class DummyGPT2Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


class DummyGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = DummyGPT2Transformer()


class DummyLlamaBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


class DummyLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyLlamaBackbone()


def test_resolve_hf_target_module_for_gpt2_style_models():
    model = DummyGPT2Model()
    resolved = resolve_hf_target_module(model, "blocks.1.hook_resid_post")
    assert resolved is model.transformer.h[1]


def test_resolve_hf_target_module_for_llama_style_models():
    model = DummyLlamaModel()
    resolved = resolve_hf_target_module(model, "blocks.2.hook_resid_post")
    assert resolved is model.model.layers[2]


def test_final_metrics_ignores_dead_feature_summary_rows():
    run = {
        "name": "sae",
        "path": "checkpoint.pt",
        "history": [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.8},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4},
            {"dead_feature_analysis": {"dead_count": 12}},
        ],
        "sparsity_weight": 0.005,
        "hidden_dim": 3072,
        "epochs": 2,
    }

    result = final_metrics(run)
    assert result["epoch"] == 2
    assert result["train_loss"] == 0.5
    assert result["val_loss"] == 0.4


def test_final_metrics_returns_none_without_epoch_rows():
    run = {
        "name": "sae",
        "path": "checkpoint.pt",
        "history": [{"dead_feature_analysis": {"dead_count": 12}}],
        "sparsity_weight": 0.005,
        "hidden_dim": 3072,
        "epochs": 0,
    }

    assert final_metrics(run) is None
