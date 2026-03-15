from pathlib import Path


def sanitize_model_name(model_name: str) -> str:
    return str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def model_slug(config) -> str:
    model_name = None
    if hasattr(config, "get"):
        model_name = config.get("model_name", "gpt2-small")
    elif isinstance(config, dict):
        model_name = config.get("model_name", "gpt2-small")
    else:
        model_name = "gpt2-small"
    return sanitize_model_name(model_name)


def default_activations_dir(config) -> str:
    return str(Path("runs") / model_slug(config) / "activations")


def default_checkpoint_path(config) -> str:
    return str(Path("runs") / model_slug(config) / "checkpoints" / "sae.pt")


def default_feature_mapping_dir(config) -> str:
    return str(Path("runs") / model_slug(config) / "reports" / "feature_mapping")
