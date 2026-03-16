from pathlib import Path


def sanitize_model_name(model_name: str) -> str:
    return str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def sanitize_layer_name(target_layer: str) -> str:
    """Convert a hook target name to a filesystem-safe slug.

    Example: 'blocks.6.hook_resid_post' -> 'blocks_6_hook_resid_post'
    """
    return str(target_layer).replace(".", "_")


def model_slug(config) -> str:
    model_name = None
    if hasattr(config, "get"):
        model_name = config.get("model_name", "gpt2-small")
    elif isinstance(config, dict):
        model_name = config.get("model_name", "gpt2-small")
    else:
        model_name = "gpt2-small"
    return sanitize_model_name(model_name)


def _resolve_targets(config):
    """Return a list of target layer names from config."""
    raw = None
    if hasattr(config, "get"):
        raw = config.get("extraction.target")
    elif isinstance(config, dict):
        raw = config.get("extraction", {}).get("target")
    if isinstance(raw, list):
        return raw
    if raw:
        return [raw]
    return ["blocks.0.hook_resid_post"]


def _is_multi_layer(config) -> bool:
    return len(_resolve_targets(config)) > 1


def default_activations_dir(config, target=None) -> str:
    base = Path("runs") / model_slug(config) / "activations"
    if target and _is_multi_layer(config):
        return str(base / sanitize_layer_name(target))
    return str(base)


def default_checkpoint_path(config, target=None) -> str:
    base = Path("runs") / model_slug(config) / "checkpoints"
    if target and _is_multi_layer(config):
        return str(base / sanitize_layer_name(target) / "sae.pt")
    return str(base / "sae.pt")


def default_feature_mapping_dir(config, target=None) -> str:
    base = Path("runs") / model_slug(config) / "reports" / "feature_mapping"
    if target and _is_multi_layer(config):
        return str(base / sanitize_layer_name(target))
    return str(base)
