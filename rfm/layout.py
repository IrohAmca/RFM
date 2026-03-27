import re
from pathlib import Path


DEFAULT_TARGET = "blocks.0.hook_resid_post"


def _config_get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)

    if not isinstance(config, dict):
        return default

    if "." not in key:
        return config.get(key, default)

    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


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


def resolve_all_targets(config):
    """Return the full configured list of target layers."""
    raw = _config_get(config, "extraction.target")
    if isinstance(raw, list):
        return raw
    if raw:
        return [raw]
    return [DEFAULT_TARGET]


def _resolve_targets(config):
    return resolve_all_targets(config)


def _hook_index(value):
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    match = re.fullmatch(r"(?:blocks|layer)\.(\d+)(?:\.hook_resid_post)?", text)
    if match:
        return int(match.group(1))
    return None


def select_targets_from(targets, from_hook):
    if from_hook in (None, ""):
        return list(targets)

    requested = str(from_hook).strip()
    if not requested:
        return list(targets)

    targets = list(targets)
    if requested in targets:
        start_idx = targets.index(requested)
        return targets[start_idx:]

    requested_idx = _hook_index(requested)
    if requested_idx is not None:
        for idx, target in enumerate(targets):
            if _hook_index(target) == requested_idx:
                return targets[idx:]

    available = ", ".join(targets)
    raise ValueError(
        f"Hook '{from_hook}' not found in extraction.target. Available targets: {available}"
    )


def resolve_requested_targets(config):
    """Return the configured targets after applying any pipeline hook offset."""
    return select_targets_from(
        resolve_all_targets(config),
        _config_get(config, "pipeline.from_hook"),
    )


def _is_multi_layer(config) -> bool:
    return len(resolve_all_targets(config)) > 1


def is_multi_layer_config(config) -> bool:
    return _is_multi_layer(config)


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


def resolve_best_checkpoint(config, target=None) -> str:
    """Find the best available SAE checkpoint for a given target layer.

    Priority:
    1. `sae.pt` (exact path from config/default)
    2. `sae_topk_k_N.pt` — highest K value (most expressive TopK model)
    3. `sae_lambda_X.pt` — first available vanilla sweep checkpoint
    4. Fallback to default path (may not exist)
    """
    default_path = Path(default_checkpoint_path(config, target=target))

    if default_path.exists():
        return str(default_path)

    ckpt_dir = default_path.parent

    # Priority: TopK sweep checkpoints — pick largest K
    topk_files = list(ckpt_dir.glob("sae_topk_k_*.pt"))
    if topk_files:
        def _k_val(p):
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return 0
        best = max(topk_files, key=_k_val)
        print(f"[layout] sae.pt not found. Using best TopK checkpoint: {best.name}")
        return str(best)

    # Fallback: vanilla lambda sweep checkpoints — pick first
    lambda_files = sorted(ckpt_dir.glob("sae_lambda_*.pt"))
    if lambda_files:
        print(f"[layout] sae.pt not found. Using sweep checkpoint: {lambda_files[0].name}")
        return str(lambda_files[0])

    # Last resort: return default path even if missing (callers will raise)
    return str(default_path)
