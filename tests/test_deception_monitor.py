import torch
from torch import nn

from rfm.deception.deception_monitor import DeceptionMonitor
from rfm.deception.deception_probe import DeceptionProbe


class DummyLayer(nn.Module):
    def forward(self, x):
        return x


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer()])
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, use_cache=False):
        return self.model.layers[0](input_ids + self.anchor)


def test_monitor_scores_hooked_layer_outputs():
    probe = DeceptionProbe()
    probe.train(
        honest_acts=torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32),
        deceptive_acts=torch.tensor([[1.0, 0.0], [1.1, 0.1]], dtype=torch.float32),
        cv_folds=2,
    )

    layer_name = "blocks.0.hook_resid_post"
    monitor = DeceptionMonitor(
        directions={layer_name: torch.tensor([1.0, 0.0])},
        probes={layer_name: probe},
        thresholds={layer_name: 0.6},
    )
    model = DummyModel()
    handles = monitor.create_hooks(model)
    try:
        model(input_ids=torch.tensor([[[1.0, 0.0], [1.0, 0.1]]], dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()

    score = monitor.score_generation(monitor.consume_activations())
    assert score.alert is True
    assert score.deception_probability >= 0.6
    assert layer_name in score.per_layer_scores


def test_monitor_respects_configured_threshold_for_global_alert():
    monitor = DeceptionMonitor(
        directions={"blocks.0.hook_resid_post": torch.tensor([1.0])},
        thresholds={"blocks.0.hook_resid_post": 0.7},
    )

    score = monitor.score_generation(
        {"blocks.0.hook_resid_post": torch.tensor([[0.4]], dtype=torch.float32)}
    )

    assert score.deception_probability < 0.7
    assert score.alert is False


def test_monitor_hooks_probe_only_layers():
    probe = DeceptionProbe()
    probe.train(
        honest_acts=torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32),
        deceptive_acts=torch.tensor([[1.0, 0.0], [1.1, 0.1]], dtype=torch.float32),
        cv_folds=2,
    )

    layer_name = "blocks.0.hook_resid_post"
    monitor = DeceptionMonitor(
        directions={},
        probes={layer_name: probe},
        thresholds={layer_name: 0.5},
    )
    model = DummyModel()
    handles = monitor.create_hooks(model)
    try:
        model(input_ids=torch.tensor([[[1.0, 0.0], [1.0, 0.1]]], dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()

    captured = monitor.consume_activations()
    score = monitor.score_generation(captured)
    assert layer_name in captured
    assert score.alert is True
    assert layer_name in score.per_layer_scores
