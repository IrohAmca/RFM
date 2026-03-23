"""Activation patching for causal validation of SAE features."""

import torch
import torch.nn.functional as F

from rfm.steering.hook import resolve_hf_target_module


def _is_transformer_lens_model(model):
    return hasattr(model, "add_hook") and hasattr(model, "reset_hooks")


def _hf_inputs(tokenizer, text, device):
    encoded = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def _capture_activation_and_logits(model, text, target_layer, tokenizer=None):
    if _is_transformer_lens_model(model):
        cache = {}

        def capture_hook(act, hook):
            cache["act"] = act.detach().clone()
            return act

        model.reset_hooks()
        model.add_hook(target_layer, capture_hook)
        logits = model(text)
        model.reset_hooks()
        return logits, cache["act"]

    if tokenizer is None:
        raise ValueError("tokenizer is required for HuggingFace activation patching.")

    cache = {}
    target_module = resolve_hf_target_module(model, target_layer)

    def capture_hook(module, inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        cache["act"] = act.detach().clone()
        return output

    handle = target_module.register_forward_hook(capture_hook)
    inputs = _hf_inputs(tokenizer, text, next(model.parameters()).device)
    outputs = model(**inputs, use_cache=False)
    handle.remove()
    return outputs.logits.detach(), cache["act"]


@torch.no_grad()
def _run_with_injected_activation(model, text, target_layer, patched_act, tokenizer=None):
    if _is_transformer_lens_model(model):

        def inject_hook(act, hook):
            return patched_act.to(act.device, act.dtype)

        model.add_hook(target_layer, inject_hook)
        logits = model(text)
        model.reset_hooks()
        return logits

    if tokenizer is None:
        raise ValueError("tokenizer is required for HuggingFace activation patching.")

    target_module = resolve_hf_target_module(model, target_layer)

    def inject_hook(module, inputs, output):
        steered = patched_act
        if isinstance(output, tuple):
            act = output[0]
            steered = steered.to(act.device, act.dtype)
            return (steered,) + output[1:]
        return steered.to(output.device, output.dtype)

    handle = target_module.register_forward_hook(inject_hook)
    inputs = _hf_inputs(tokenizer, text, next(model.parameters()).device)
    outputs = model(**inputs, use_cache=False)
    handle.remove()
    return outputs.logits.detach()


def _compute_effect(clean_logits, patched_logits, metric):
    clean_last = clean_logits[0, -1]
    patched_last = patched_logits[0, -1]

    if metric == "logit_diff":
        return float((patched_last - clean_last).abs().mean().item())

    if metric == "kl_divergence":
        clean_probs = F.log_softmax(clean_last, dim=-1)
        patched_probs = F.softmax(patched_last, dim=-1)
        return float(F.kl_div(clean_probs, patched_probs, reduction="sum").item())

    raise ValueError(f"Unknown metric: {metric!r}. Use 'logit_diff' or 'kl_divergence'.")


def _feature_values(act_tensor, sae_model, feature_id):
    flat = act_tensor.reshape(-1, act_tensor.shape[-1]).to(next(sae_model.parameters()).device)
    _, f = sae_model(flat)
    return f[:, feature_id]


def _patched_activation(clean_act, patch_feature_vals, sae_model, feature_id, device):
    b_pre = sae_model.b_pre.to(device, clean_act.dtype)
    W_enc = sae_model.W_enc.to(device, clean_act.dtype)
    b_enc = sae_model.b_enc.to(device, clean_act.dtype)
    W_dec = sae_model.W_dec.to(device, clean_act.dtype)

    clean_flat = clean_act.reshape(-1, clean_act.shape[-1]).to(device)
    x_centered = clean_flat - b_pre
    h = x_centered @ W_enc + b_enc
    f_clean = torch.relu(h)

    min_len = min(f_clean.shape[0], patch_feature_vals.shape[0])
    delta_vals = patch_feature_vals[:min_len].to(device) - f_clean[:min_len, feature_id]
    direction = W_dec[feature_id]

    patched_flat = clean_flat.clone()
    patched_flat[:min_len] += delta_vals.unsqueeze(-1) * direction.unsqueeze(0)
    patched_act = patched_flat.reshape(clean_act.shape)
    return patched_act, f_clean, min_len


@torch.no_grad()
def activation_patch(
    model,
    sae_model,
    clean_text,
    patch_text,
    target_layer,
    feature_id,
    metric="logit_diff",
    tokenizer=None,
):
    """Measure the causal effect of a single SAE feature."""
    device = next(model.parameters()).device
    sae_model.eval()

    clean_logits, clean_act = _capture_activation_and_logits(
        model=model,
        text=clean_text,
        target_layer=target_layer,
        tokenizer=tokenizer,
    )
    _, patch_act = _capture_activation_and_logits(
        model=model,
        text=patch_text,
        target_layer=target_layer,
        tokenizer=tokenizer,
    )

    clean_feature_vals = _feature_values(clean_act, sae_model, feature_id)
    patch_feature_vals = _feature_values(patch_act, sae_model, feature_id)
    patched_act, _, min_len = _patched_activation(
        clean_act=clean_act,
        patch_feature_vals=patch_feature_vals,
        sae_model=sae_model,
        feature_id=feature_id,
        device=device,
    )

    patched_logits = _run_with_injected_activation(
        model=model,
        text=clean_text,
        target_layer=target_layer,
        patched_act=patched_act,
        tokenizer=tokenizer,
    )
    effect = _compute_effect(clean_logits, patched_logits, metric)

    return {
        "clean_logits": clean_logits.detach().cpu(),
        "patched_logits": patched_logits.detach().cpu(),
        "effect": effect,
        "metric": metric,
        "feature_id": feature_id,
        "target_layer": target_layer,
        "clean_feature_mean": float(clean_feature_vals.mean().item()),
        "patch_feature_mean": float(patch_feature_vals[:min_len].mean().item()),
    }


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
    tokenizer=None,
):
    """Patch multiple features and rank them by effect size."""
    if feature_ids is None:
        _, clean_act = _capture_activation_and_logits(
            model=model,
            text=clean_text,
            target_layer=target_layer,
            tokenizer=tokenizer,
        )
        act_flat = clean_act.reshape(-1, clean_act.shape[-1]).to(next(sae_model.parameters()).device)
        _, f = sae_model(act_flat)
        mean_activation = f.mean(dim=0)
        _, top_indices = torch.topk(mean_activation, k=min(top_k, f.shape[-1]))
        feature_ids = top_indices.cpu().tolist()

    results = []
    for fid in feature_ids:
        results.append(
            activation_patch(
                model=model,
                sae_model=sae_model,
                clean_text=clean_text,
                patch_text=patch_text,
                target_layer=target_layer,
                feature_id=fid,
                metric=metric,
                tokenizer=tokenizer,
            )
        )

    results.sort(key=lambda row: row["effect"], reverse=True)
    return results
