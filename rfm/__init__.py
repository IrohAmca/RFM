"""RFM: Residual Feature Mapping with Sparse Autoencoders."""

from rfm.config import ConfigManager
from rfm.layout import model_slug, sanitize_layer_name

__all__ = ["ConfigManager", "model_slug", "sanitize_layer_name"]
