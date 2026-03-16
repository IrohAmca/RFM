"""Activation patching for causal validation of SAE features.

Answers: "Does this SAE feature causally affect the model's output?"

Approach:
1. Run clean prompt → get clean activations at target layer
2. Run patch prompt → get patch activations at target layer
3. Decompose both through SAE to get feature activations
4. Replace target feature's activation in clean run with patch run's value
5. Measure output change (logit diff, KL divergence, etc.)
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def activation_patch(
    model,
    sae_model,
    clean_text,
    patch_text,
    target_layer,
    feature_id,
    metric="logit_diff",
):
    """Measure causal effect of a single SAE feature via activation patching.

    Args:
        model: HookedTransformer instance.
        sae_model: Trained SparseAutoEncoder.
        clean_text: Baseline text prompt.
        patch_text: Text with modified meaning for patching.
        target_layer: Hook point name (e.g. 'blocks.6.hook_resid_post').
        feature_id: SAE feature index to patch.
        metric: 'logit_diff' or 'kl_divergence'.

    Returns:
        Dict with patching results:
            - clean_output: logits from clean run
            - patched_output: logits from patched run
            - effect: scalar measuring the change
            - metric: which metric was used
    """
    device = next(model.parameters()).device
    sae_device = next(sae_model.parameters()).device

    # Step 1: Clean run — capture activations
    clean_cache = {}

    def capture_clean(act, hook):
        clean_cache["act"] = act.detach().clone()
        return act

    model.reset_hooks()
    model.add_hook(target_layer, capture_clean)
    clean_logits = model(clean_text)
    model.reset_hooks()

    # Step 2: Patch run — capture activations
    patch_cache = {}

    def capture_patch(act, hook):
        patch_cache["act"] = act.detach().clone()
        return act

    model.add_hook(target_layer, capture_patch)
    _ = model(patch_text)
    model.reset_hooks()

    clean_act = clean_cache["act"]
    patch_act = patch_cache["act"]

    # Step 3: Decompose through SAE
    sae_model.eval()

    def _get_feature_val(act_tensor):
        # act shape: [batch, seq_len, d_model]
        flat = act_tensor.reshape(-1, act_tensor.shape[-1]).to(sae_device)
        _, f = sae_model(flat)
        return f[:, feature_id]

    clean_feature_vals = _get_feature_val(clean_act)
    patch_feature_vals = _get_feature_val(patch_act)

    # Step 4: Create patched activations
    # Replace the feature's contribution in the clean run
    b_pre = sae_model.b_pre.to(device, clean_act.dtype)
    W_enc = sae_model.W_enc.to(device, clean_act.dtype)
    b_enc = sae_model.b_enc.to(device, clean_act.dtype)
    W_dec = sae_model.W_dec.to(device, clean_act.dtype)

    # Compute feature activations for clean run
    clean_flat = clean_act.reshape(-1, clean_act.shape[-1])
    x_centered = clean_flat - b_pre
    h = x_centered @ W_enc + b_enc
    f_clean = torch.relu(h)

    # Get delta for the target feature
    min_len = min(f_clean.shape[0], patch_feature_vals.shape[0])
    delta_vals = patch_feature_vals[:min_len].to(device) - f_clean[:min_len, feature_id]

    # direction = W_dec[feature_id]
    direction = W_dec[feature_id]

    # Construct patched activations
    patched_act = clean_act.clone()
    patched_flat = patched_act.reshape(-1, patched_act.shape[-1])
    patched_flat[:min_len] += delta_vals.unsqueeze(-1) * direction.unsqueeze(0)
    patched_act = patched_flat.reshape(clean_act.shape)

    # Step 5: Run model with patched activations
    def inject_patched(act, hook):
        return patched_act.to(act.device, act.dtype)

    model.add_hook(target_layer, inject_patched)
    patched_logits = model(clean_text)
    model.reset_hooks()

    # Step 6: Compute effect metric
    effect = _compute_effect(clean_logits, patched_logits, metric)

    return {
        "clean_logits": clean_logits.detach().cpu(),
        "patched_logits": patched_logits.detach().cpu(),
        "effect": float(effect),
        "metric": metric,
        "feature_id": feature_id,
        "target_layer": target_layer,
        "clean_feature_mean": float(clean_feature_vals.mean().item()),
        "patch_feature_mean": float(patch_feature_vals[:min_len].mean().item()),
    }


def _compute_effect(clean_logits, patched_logits, metric):
    """Compute the effect of patching on the output distribution."""
    clean_last = clean_logits[0, -1]
    patched_last = patched_logits[0, -1]

    if metric == "logit_diff":
        return (patched_last - clean_last).abs().mean().item()

    if metric == "kl_divergence":
        clean_probs = F.log_softmax(clean_last, dim=-1)
        patched_probs = F.softmax(patched_last, dim=-1)
        kl = F.kl_div(clean_probs, patched_probs, reduction="sum")
        return kl.item()

    raise ValueError(f"Unknown metric: {metric!r}. Use 'logit_diff' or 'kl_divergence'.")


@torch.no_grad()
def batch_feature_patching(
    model,
    sae_model,
    clean_text,
    patch_text,
    target_layer,
    feature_ids=None,
    top_k=20,
    metric="logit_diff",
):
    """Patch multiple features and rank by causal effect.

    Args:
        model: HookedTransformer instance.
        sae_model: Trained SAE.
        clean_text: Baseline prompt.
        patch_text: Altered prompt.
        target_layer: Hook point name.
        feature_ids: List of feature ids to test. If None, uses top_k most active.
        top_k: If feature_ids is None, test the top_k most active features.
        metric: Effect metric.

    Returns:
        List of dicts sorted by effect (descending).
    """

    if feature_ids is None:
        # Find top_k most active features on the clean prompt
        model.reset_hooks()
        cache = {}

        def capture(act, hook):
            cache["act"] = act.detach().clone()
            return act

        model.add_hook(target_layer, capture)
        _ = model(clean_text)
        model.reset_hooks()

        sae_model.eval()
        act_flat = cache["act"].reshape(-1, cache["act"].shape[-1])
        act_flat = act_flat.to(next(sae_model.parameters()).device)
        _, f = sae_model(act_flat)
        mean_activation = f.mean(dim=0)
        _, top_indices = torch.topk(mean_activation, k=min(top_k, f.shape[-1]))
        feature_ids = top_indices.cpu().tolist()

    results = []
    for fid in feature_ids:
        result = activation_patch(
            model=model,
            sae_model=sae_model,
            clean_text=clean_text,
            patch_text=patch_text,
            target_layer=target_layer,
            feature_id=fid,
            metric=metric,
        )
        results.append(result)

    results.sort(key=lambda r: r["effect"], reverse=True)
    return results
