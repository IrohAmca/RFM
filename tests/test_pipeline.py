import json
import tempfile
from pathlib import Path

from cli.pipeline import _build_effective_config


def test_build_effective_config_injects_from_hook_without_rewriting_targets():
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
        dir=".",
    ) as handle:
        json.dump(
            {
                "model_name": "Qwen/Qwen3-0.6B",
                "extraction": {
                    "target": [
                        "blocks.0.hook_resid_post",
                        "blocks.6.hook_resid_post",
                        "blocks.27.hook_resid_post",
                    ]
                },
            },
            handle,
            ensure_ascii=False,
        )
        config_path = Path(handle.name)

    try:
        effective_config, temp_path, selected_targets = _build_effective_config(
            str(config_path),
            from_hook="27",
        )

        assert temp_path is not None
        assert effective_config == str(temp_path)
        assert selected_targets == ["blocks.27.hook_resid_post"]

        payload = json.loads(temp_path.read_text(encoding="utf-8"))
        assert payload["pipeline"]["from_hook"] == "27"
        assert payload["extraction"]["target"] == [
            "blocks.0.hook_resid_post",
            "blocks.6.hook_resid_post",
            "blocks.27.hook_resid_post",
        ]
    finally:
        config_path.unlink(missing_ok=True)
        if "temp_path" in locals() and temp_path is not None:
            temp_path.unlink(missing_ok=True)
