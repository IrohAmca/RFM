"""Steering hooks for SAE feature-level intervention on transformer models.

Provides hooks that can amplify, suppress, ablate, or clamp specific SAE
features in the residual stream during a forward pass.
"""

import torch


class SteeringHook:
    """Inject or suppress SAE feature directions in the residual stream.

    The steering direction is extracted from the SAE decoder matrix:
        direction = W_dec[feature_id]  (unit-normalized by SAE training)

    Modes:
        - amplify/suppress:  act' = act + alpha * direction
        - ablate:            project out the feature direction entirely
        - clamp:             set the feature's activation to a fixed value
    """

    def __init__(self, sae_model, feature_id, alpha=1.0, mode="add"):
        """
        Args:
            sae_model: Trained SparseAutoEncoder instance.
            feature_id: Index of the latent feature to steer.
            alpha: Steering strength. Positive amplifies, negative suppresses.
            mode: One of 'add', 'ablate', 'clamp'.
                  - 'add': act' = act + alpha * direction
                  - 'ablate': remove projection onto direction
                  - 'clamp': set feature activation to alpha value
        """
        if feature_id < 0 or feature_id >= sae_model.hidden_dim:
            raise ValueError(
                f"feature_id={feature_id} out of range [0, {sae_model.hidden_dim})"
            )
        if mode not in ("add", "ablate", "clamp"):
            raise ValueError(f"Unsupported mode: {mode!r}. Use 'add', 'ablate', or 'clamp'.")

        self.feature_id = feature_id
        self.alpha = alpha
        self.mode = mode

        # Extract the decoder direction vector (already unit-normalized by SAE)
        with torch.no_grad():
            self.direction = sae_model.W_dec[feature_id].detach().clone()

        # Keep the full SAE for clamp mode
        self._sae = sae_model

    def hook_fn(self, act, hook):
        """TransformerLens-compatible hook function.

        Args:
            act: Activation tensor of shape [batch, seq_len, d_model].
            hook: TransformerLens HookPoint (unused but required by API).

        Returns:
            Modified activation tensor.
        """
        direction = self.direction.to(act.device, act.dtype)

        if self.mode == "add":
            return act + self.alpha * direction

        if self.mode == "ablate":
            # Project out: act' = act - (act · d̂) * d̂
            d_hat = direction / direction.norm().clamp(min=1e-12)
            proj = (act * d_hat).sum(dim=-1, keepdim=True) * d_hat
            return act - proj

        if self.mode == "clamp":
            return self._clamp_feature(act, direction)

        return act

    def _clamp_feature(self, act, direction):
        """Replace the feature's contribution with a clamped value."""
        sae = self._sae
        device = act.device
        dtype = act.dtype

        # Move SAE parameters to match activation device/dtype
        b_pre = sae.b_pre.to(device, dtype)
        W_enc = sae.W_enc.to(device, dtype)
        b_enc = sae.b_enc.to(device, dtype)

        # Compute current feature activation
        x_centered = act - b_pre
        h = x_centered @ W_enc + b_enc
        f = torch.relu(h)

        # Current contribution of this feature
        current_val = f[..., self.feature_id : self.feature_id + 1]
        target_val = torch.full_like(current_val, self.alpha)

        # Delta contribution: (target - current) * decoder_direction
        delta = (target_val - current_val) * direction.unsqueeze(0)
        return act + delta

    @classmethod
    def apply(cls, model, target_layer, sae_model, feature_id, alpha=1.0, mode="add"):
        """Convenience: create hook and attach to a TransformerLens model.

        Args:
            model: HookedTransformer instance.
            target_layer: Hook point name, e.g. 'blocks.6.hook_resid_post'.
            sae_model: Trained SAE.
            feature_id: Feature index to steer.
            alpha: Steering strength.
            mode: 'add', 'ablate', or 'clamp'.

        Returns:
            SteeringHook instance (call model.reset_hooks() to remove).
        """
        hook = cls(sae_model, feature_id, alpha=alpha, mode=mode)
        model.add_hook(target_layer, hook.hook_fn)
        return hook


class MultiSteeringHook:
    """Apply multiple feature steering directions simultaneously."""

    def __init__(self, hooks):
        """
        Args:
            hooks: List of SteeringHook instances.
        """
        self.hooks = hooks

    def hook_fn(self, act, hook):
        for h in self.hooks:
            act = h.hook_fn(act, hook)
        return act

    @classmethod
    def from_features(cls, sae_model, feature_configs):
        """Create from a list of (feature_id, alpha, mode) tuples.

        Args:
            sae_model: Trained SAE.
            feature_configs: List of dicts with keys 'feature_id', 'alpha', 'mode'.

        Returns:
            MultiSteeringHook instance.
        """
        hooks = []
        for cfg in feature_configs:
            hooks.append(
                SteeringHook(
                    sae_model=sae_model,
                    feature_id=cfg["feature_id"],
                    alpha=cfg.get("alpha", 1.0),
                    mode=cfg.get("mode", "add"),
                )
            )
        return cls(hooks)
