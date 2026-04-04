import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from rfm.config import ConfigManager
from rfm.deception.reporting import deception_report_paths
from rfm.deception.utils import deception_run_dir
from rfm.layout import (
    default_feature_mapping_dir,
    model_slug,
    resolve_activations_dir,
    resolve_best_checkpoint,
    resolve_requested_targets,
    sanitize_layer_name,
)
from rfm.patterns import (
    ContrastAxisSpec,
    DirectionResult,
    append_pattern_analysis_rows,
    axis_monitor_from_bundle,
    load_model_and_tokenizer as load_runtime_model_and_tokenizer,
    load_pattern_bundle,
    motif_feature_configs,
    motif_members,
    pattern_artifact_paths,
    register_feature_interventions,
)
from rfm.sae.model import load_sae_checkpoint
from rfm.steering.hook import HFSteeringHook, resolve_hf_target_module

st.set_page_config(page_title="RFM Dashboard", layout="wide")


@st.cache_data
def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        return None
    return ConfigManager.from_file(str(path))


@st.cache_resource
def load_runtime_model_resource(config_path):
    config = ConfigManager.from_file(config_path)
    return load_runtime_model_and_tokenizer(config)


@st.cache_resource
def load_sae_resource(config_path, target):
    config = ConfigManager.from_file(config_path)
    target_config = config.for_target(target) if hasattr(config, "for_target") else config
    sae_path = resolve_best_checkpoint(target_config, target=target)
    sae_model, _ = load_sae_checkpoint(sae_path, device=config.get("train.device", "cpu"))
    sae_model.eval()
    return sae_model


@st.cache_data
def load_json_artifact(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_pattern_bundle_artifact(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


@st.cache_data
def load_pattern_report_artifact(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_torch_artifact(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


@st.cache_data
def load_feature_summary(mapping_dir, slug):
    candidates = [
        Path(mapping_dir) / "feature_mapping_feature_summary.csv",
        Path(mapping_dir) / f"{slug}_feature_summary.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    return pd.read_csv(path) if path is not None else None


@st.cache_data
def load_feature_events(mapping_dir, slug):
    candidates = [
        Path(mapping_dir) / "feature_mapping_events.csv",
        Path(mapping_dir) / f"{slug}_feature_events.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    return pd.read_csv(path) if path is not None else None


@st.cache_data
def load_autointerp(mapping_dir, slug):
    path = Path(mapping_dir) / f"{slug}_autointerp_results.json"
    if not path.exists():
        return {}
    return {int(key): value for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


@st.cache_data
def load_deception_autointerp(deception_dir: str, layer_name: str):
    safe = sanitize_layer_name(layer_name)
    path = Path(deception_dir) / "autointerp" / f"{safe}_interpretations.json"
    if not path.exists():
        return {}
    return {int(key): value for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


@st.cache_data
def load_training_history(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return state.get("history", [])


@st.cache_data
def load_extraction_metadata(config):
    targets = resolve_requested_targets(config)
    if not targets:
        return None
    target = targets[0]
    act_dir = Path(resolve_activations_dir(config, target=target))
    meta_files = list(act_dir.glob("*.meta.json"))
    if not meta_files:
        return None

    rows = []
    for meta_path in meta_files:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = payload.get("labels", [])
        categories = payload.get("categories", [])
        difficulties = payload.get("difficulties", [])
        lengths = payload.get("token_lengths", [])
        questions = payload.get("questions", [])
        responses = payload.get("responses", [])
        for index, label in enumerate(labels):
            rows.append(
                {
                    "chunk_id": payload.get("chunk_id", 0),
                    "label": label,
                    "category": categories[index] if index < len(categories) else "unknown",
                    "difficulty": difficulties[index] if index < len(difficulties) else "unknown",
                    "token_length": lengths[index] if index < len(lengths) else 0,
                    "question": questions[index] if index < len(questions) else "",
                    "response": responses[index] if index < len(responses) else "",
                }
            )
    return pd.DataFrame(rows) if rows else None


@st.cache_data
def load_deception_adversarial(deception_dir: str):
    summary_path = Path(deception_dir) / "adversarial" / "summary.json"
    missed_path = Path(deception_dir) / "adversarial" / "missed_samples.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    missed = json.loads(missed_path.read_text(encoding="utf-8")) if missed_path.exists() else []
    return summary, missed


@st.cache_data
def load_deception_report_assets(deception_dir: str):
    paths = deception_report_paths(Path(deception_dir) / "reports")
    return {key: str(value) for key, value in paths.items()}


def _pattern_state(config):
    axis = ContrastAxisSpec.from_config(config)
    paths = pattern_artifact_paths(config, axis)
    bundle = load_pattern_bundle_artifact(str(paths["bundle"])) or {}
    report = load_pattern_report_artifact(str(paths["report"])) or {}
    return axis, paths, bundle, report


def _pattern_layer_scores(bundle: dict, layer_name: str) -> pd.DataFrame | None:
    rows = list(bundle.get("layers", {}).get(layer_name, {}).get("feature_scores", []) or [])
    if not rows:
        return None
    return pd.DataFrame(rows)


def _format_target_label(target: str) -> str:
    parts = str(target).split(".")
    if len(parts) >= 2 and parts[-2].isdigit():
        return f"L{parts[-2]} resid_post"
    return str(target)


def _select_target(config, key: str, label: str = "Layer"):
    targets = resolve_requested_targets(config)
    if not targets:
        return None
    if len(targets) == 1:
        st.caption(f"{label}: `{_format_target_label(targets[0])}`")
        return targets[0]
    return st.selectbox(label, targets, format_func=_format_target_label, key=key)


def _pipeline_status(dec_dir: str, config=None) -> dict:
    base = Path(dec_dir)
    scenario_path = base / "contextual_scenarios.jsonl"
    if config is not None and hasattr(config, "get"):
        configured_path = config.get("deception.scenario_generator.cache_path")
        if configured_path:
            scenario_path = Path(configured_path)
    return {
        "scenarios": scenario_path.exists(),
        "activations": any((base / "contextual_activations").glob("**/*.pt")),
        "checkpoints": any((base / "checkpoints").glob("**/sae.pt")),
        "directions": (base / "directions" / "directions.pt").exists(),
        "probes": (base / "probes" / "probe_summary.json").exists(),
        "patterns": (base.parent / ContrastAxisSpec.from_config(config).axis_id / "patterns" / "pattern_bundle.pt").exists()
        if config is not None
        else False,
        "monitor": (base / "monitor" / "monitor_report.json").exists(),
        "adversarial": (base / "adversarial" / "summary.json").exists(),
        "autointerp": any((base / "autointerp").glob("*.json")) if (base / "autointerp").exists() else False,
    }


def _format_generation_prompt(tokenizer, prompt: str, system_prompt: str | None = None) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            try:
                return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
    if system_prompt:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    return prompt


def _generate_text(model, tokenizer, prompt_text: str, device: str, max_tokens: int, temperature: float) -> str:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_tokens),
        "do_sample": float(temperature) > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if float(temperature) > 0:
        generation_kwargs["temperature"] = float(temperature)

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    response_ids = output_ids[:, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    return response


def _bundle_rows(bundle: dict, source: str) -> list[dict]:
    analysis = dict(bundle.get("analysis", {}) or {})
    rows = list(analysis.get(source, []) or [])
    if source == "stable_interactions" and not rows:
        rows = [row for row in analysis.get("stable_motifs", []) if row.get("kind") == "interaction"]
    return rows


def _select_motif_payload(bundle: dict, source: str, motif_name: str) -> dict:
    rows = _bundle_rows(bundle, source)
    for row in rows:
        if str(row.get("name")) == motif_name:
            return row
    raise ValueError(f"Motif {motif_name!r} not found in analysis.{source}.")


def _render_steering_result(result: dict, *, heading: str) -> None:
    if not result:
        return
    st.subheader(heading)
    cols = st.columns(2)
    with cols[0]:
        st.caption("Clean Output")
        st.write(result.get("clean_output", ""))
    with cols[1]:
        st.caption("Intervened Output")
        st.write(result.get("steered_output", ""))
    if result.get("clean_score") is not None and result.get("steered_score") is not None:
        score_cols = st.columns(3)
        score_cols[0].metric("Clean Score", f"{float(result['clean_score']):.3f}")
        score_cols[1].metric("Intervened Score", f"{float(result['steered_score']):.3f}")
        score_cols[2].metric("Delta", f"{float(result['steered_score'] - result['clean_score']):+.3f}")
    if result.get("details"):
        with st.expander("Details", expanded=False):
            st.json(result["details"])


def _run_feature_steering_ui(config_path, layer: str, feature_id: int, alpha: float, mode: str, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> dict:
    config = ConfigManager.from_file(config_path)
    model, tokenizer, device = load_runtime_model_resource(config_path)
    sae_model = load_sae_resource(config_path, layer)
    prompt_text = _format_generation_prompt(tokenizer, prompt, system_prompt or None)
    clean_output = _generate_text(model, tokenizer, prompt_text, device, max_tokens, temperature)
    handle = HFSteeringHook.apply(
        hf_model=model,
        target_layer=layer,
        sae_model=sae_model,
        feature_id=int(feature_id),
        alpha=float(alpha),
        mode=str(mode),
    )
    try:
        steered_output = _generate_text(model, tokenizer, prompt_text, device, max_tokens, temperature)
    finally:
        handle.remove()
    return {
        "clean_output": clean_output,
        "steered_output": steered_output,
        "details": {
            "layer": layer,
            "feature_id": int(feature_id),
            "alpha": float(alpha),
            "mode": str(mode),
        },
    }


def _run_motif_steering_ui(
    config_path,
    motif_name: str,
    action: str,
    alpha: float,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    write_bundle: bool,
) -> dict:
    config = ConfigManager.from_file(config_path)
    axis = ContrastAxisSpec.from_config(config)
    bundle = load_pattern_bundle(config, axis)
    motif = _select_motif_payload(bundle, "stable_motifs", motif_name)
    model, tokenizer, device = load_runtime_model_resource(config_path)
    prompt_text = _format_generation_prompt(tokenizer, prompt, system_prompt or None)
    layer_feature_configs = motif_feature_configs(motif, action=action, alpha=alpha)
    sae_models = {layer_name: load_sae_resource(config_path, layer_name) for layer_name in layer_feature_configs}

    clean_output = _generate_text(model, tokenizer, prompt_text, device, max_tokens, temperature)
    handles = register_feature_interventions(
        model=model,
        sae_models=sae_models,
        layer_feature_configs=layer_feature_configs,
    )
    try:
        steered_output = _generate_text(model, tokenizer, prompt_text, device, max_tokens, temperature)
    finally:
        for handle in handles:
            handle.remove()

    monitor = axis_monitor_from_bundle(axis_spec=axis, bundle=bundle, ensemble_method=config.get("deception.monitor.ensemble_method", "weighted_average"), config=config)
    clean_score = None
    steered_score = None
    if monitor is not None:
        clean_score = monitor.score_replay(model=model, tokenizer=tokenizer, prompt=prompt, response=clean_output, system_prompt=system_prompt or None).contrast_probability
        steered_score = monitor.score_replay(model=model, tokenizer=tokenizer, prompt=prompt, response=steered_output, system_prompt=system_prompt or None).contrast_probability

    if write_bundle:
        record = {
            "name": motif.get("name"),
            "kind": motif.get("kind"),
            "source": "stable_motifs",
            "action": action,
            "alpha": float(alpha),
            "members": motif_members(motif),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "clean_response": clean_output[:400],
            "intervention_response": steered_output[:400],
        }
        if clean_score is not None and steered_score is not None:
            record.update(
                {
                    "clean_probability": float(clean_score),
                    "intervention_probability": float(steered_score),
                    "probability_delta": float(steered_score - clean_score),
                }
            )
        append_pattern_analysis_rows(config, axis_spec=axis, key="manual_interventions", rows=[record], max_rows=50)

    return {
        "clean_output": clean_output,
        "steered_output": steered_output,
        "clean_score": clean_score,
        "steered_score": steered_score,
        "details": motif,
    }


def _run_direction_steering_ui(config_path, target_layers: list[str], alpha: float, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> dict:
    config = ConfigManager.from_file(config_path)
    dec_dir = deception_run_dir(config)
    directions_path = dec_dir / "directions" / "directions.pt"
    payload = load_torch_artifact(str(directions_path))
    if not payload:
        raise FileNotFoundError(f"Direction file not found: {directions_path}")

    model, tokenizer, device = load_runtime_model_resource(config_path)
    prompt_text = _format_generation_prompt(tokenizer, prompt, system_prompt or None)
    clean_output = _generate_text(model, tokenizer, prompt_text, device, max_tokens, temperature)

    handles = []
    hooked_layers = []
    for layer in target_layers:
        if layer not in payload:
            continue
        direction_result = DirectionResult.from_dict(payload[layer]) if isinstance(payload[layer], dict) else payload[layer]
        direction = direction_result.direction.detach().to(device, dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-12)
        target_module = resolve_hf_target_module(model, layer)

        def _make_hook(direction_tensor, alpha_value):
            def hook_fn(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                activations = output[0] if is_tuple else output
                steered = activations + alpha_value * direction_tensor.to(activations.device, activations.dtype)
                return (steered,) + output[1:] if is_tuple else steered

            return hook_fn

        handles.append(target_module.register_forward_hook(_make_hook(direction, alpha)))
        hooked_layers.append(layer)

    try:
        steered_output = _generate_text(model, tokenizer, prompt_text, device, max_tokens, temperature)
    finally:
        for handle in handles:
            handle.remove()

    axis = ContrastAxisSpec.from_config(config)
    bundle = load_pattern_bundle(config, axis)
    monitor = axis_monitor_from_bundle(axis_spec=axis, bundle=bundle, ensemble_method=config.get("deception.monitor.ensemble_method", "weighted_average"), config=config)
    clean_score = None
    steered_score = None
    if monitor is not None:
        clean_score = monitor.score_replay(model=model, tokenizer=tokenizer, prompt=prompt, response=clean_output, system_prompt=system_prompt or None).contrast_probability
        steered_score = monitor.score_replay(model=model, tokenizer=tokenizer, prompt=prompt, response=steered_output, system_prompt=system_prompt or None).contrast_probability

    return {
        "clean_output": clean_output,
        "steered_output": steered_output,
        "clean_score": clean_score,
        "steered_score": steered_score,
        "details": {
            "hooked_layers": hooked_layers,
            "alpha": float(alpha),
        },
    }


def render_feature_explorer(config, slug):
    st.header("Feature Explorer")
    selected_target = _select_target(config, "feature_layer_select")
    if not selected_target:
        st.warning("No target layers configured.")
        return

    mapping_dir = default_feature_mapping_dir(config, target=selected_target)
    summary_df = load_feature_summary(mapping_dir, slug)
    events_df = load_feature_events(mapping_dir, slug)
    autointerp_data = load_autointerp(mapping_dir, slug)
    if summary_df is not None and events_df is not None:
        st.caption(f"Mapping directory: `{mapping_dir}`")
        st.write(f"Mapped features: **{len(summary_df)}**")
        search_kw = st.text_input("Search interpretation or tokens", "", key="mapping_search")
        sort_by = st.selectbox("Sort by", ["num_events", "mean_activation", "max_activation"], index=0, key="mapping_sort")
        display_df = summary_df.copy()
        display_df["interpretation"] = display_df["feature_id"].map(lambda value: autointerp_data.get(int(value), ""))
        if search_kw:
            mask = (
                display_df["interpretation"].fillna("").str.contains(search_kw, case=False, na=False)
                | display_df.get("top_tokens", pd.Series([""] * len(display_df))).fillna("").str.contains(search_kw, case=False, na=False)
            )
            display_df = display_df[mask]
        display_df = display_df.sort_values(by=sort_by, ascending=False)
        feature_ids = [int(value) for value in display_df["feature_id"].dropna().tolist()]
        if not feature_ids:
            st.info("No features match the current filter.")
            return
        selected_feature = st.selectbox("Feature", feature_ids, format_func=lambda value: f"F{value}", key="feature_select")
        stats = display_df[display_df["feature_id"] == selected_feature].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Events", int(stats.get("num_events", 0)))
        m2.metric("Mean Activation", f"{float(stats.get('mean_activation', 0.0)):.3f}")
        m3.metric("Max Activation", f"{float(stats.get('max_activation', 0.0)):.3f}")
        if stats.get("interpretation"):
            st.info(stats["interpretation"])
        feature_events = events_df[events_df["feature_id"] == selected_feature].sort_values(by="activation", ascending=False).head(20)
        st.dataframe(feature_events, use_container_width=True)
        return

    axis, paths, bundle, _ = _pattern_state(config)
    scores_df = _pattern_layer_scores(bundle, selected_target)
    autointerp_data = load_deception_autointerp(str(deception_run_dir(config)), selected_target)
    if scores_df is None or scores_df.empty:
        st.warning(
            "No feature mapping artifacts or canonical signed feature scores were found. "
            "Run feature mapping or `python -m cli.pattern_score --mode feature-score` first."
        )
        return
    scores_df = scores_df.copy()
    scores_df["association"] = scores_df["delta"].apply(lambda value: axis.endpoint_b if value >= 0 else axis.endpoint_a)
    scores_df["interpretation"] = scores_df["feature_id"].map(lambda value: autointerp_data.get(int(value), ""))
    st.caption(f"Pattern bundle: `{paths['bundle']}`")
    st.dataframe(scores_df.sort_values(by="effect_size", key=abs, ascending=False).head(100), use_container_width=True)


def render_training_metrics(config):
    st.header("Training Metrics")
    selected_target = _select_target(config, "metrics_layer_select")
    if not selected_target:
        st.warning("No target layers configured.")
        return

    target_config = config.for_target(selected_target) if hasattr(config, "for_target") else config
    checkpoint_path = resolve_best_checkpoint(target_config, target=selected_target)
    st.caption(f"Checkpoint: `{checkpoint_path}`")
    history = load_training_history(checkpoint_path)
    if not history:
        st.warning("Training history not found in checkpoint.")
        return

    epoch_metrics = [row for row in history if "epoch" in row]
    if not epoch_metrics:
        st.info("No epoch metrics found.")
        return
    df = pd.DataFrame(epoch_metrics)
    col1, col2 = st.columns(2)
    with col1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Loss"))
        if "val_loss" in df.columns:
            fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val Loss"))
        st.plotly_chart(fig_loss, use_container_width=True)
    with col2:
        fig_active = go.Figure()
        fig_active.add_trace(go.Scatter(x=df["epoch"], y=df["train_active_rate"], name="Train Active Rate"))
        if "val_active_rate" in df.columns:
            fig_active.add_trace(go.Scatter(x=df["epoch"], y=df["val_active_rate"], name="Val Active Rate"))
        st.plotly_chart(fig_active, use_container_width=True)


def render_extraction_overview(config):
    st.header("Extraction Overview")
    meta_df = load_extraction_metadata(config)
    if meta_df is None or meta_df.empty:
        st.warning("No extraction metadata found.")
        return
    st.success(f"Loaded {len(meta_df)} sequence rows.")
    col1, col2 = st.columns(2)
    with col1:
        fig_cat = px.histogram(meta_df, x="category", color="label", barmode="group", title="Category Distribution")
        st.plotly_chart(fig_cat, use_container_width=True)
    with col2:
        fig_len = px.box(meta_df, x="label", y="token_length", color="label", points="all", title="Token Lengths")
        st.plotly_chart(fig_len, use_container_width=True)


def render_signed_feature_analysis(config):
    st.header("Signed Feature Analysis")
    selected_target = _select_target(config, "signed_feature_layer_select")
    if not selected_target:
        st.warning("No target layers configured.")
        return

    axis, paths, bundle, _ = _pattern_state(config)
    scores_df = _pattern_layer_scores(bundle, selected_target)
    if scores_df is None or scores_df.empty:
        st.warning("No signed feature scores found. Run `python -m cli.pattern_score --mode feature-score` first.")
        return

    st.caption(f"Pattern bundle: `{paths['bundle']}`")
    scores_df = scores_df.copy()
    scores_df["association"] = scores_df["delta"].apply(lambda value: axis.endpoint_b if value >= 0 else axis.endpoint_a)

    k1, k2, k3 = st.columns(3)
    k1.metric("Features", len(scores_df))
    k2.metric("Top |Effect Size|", f"{scores_df['effect_size'].abs().max():.4f}")
    k3.metric("Mean |Delta|", f"{scores_df['delta'].abs().mean():.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(
            scores_df,
            x="activation_rate_a",
            y="activation_rate_b",
            color="effect_size",
            size=scores_df["delta"].abs().clip(lower=0.01),
            hover_data=["feature_id", "delta", "effect_size", "q_value"],
            labels={
                "activation_rate_a": f"{axis.display_name_a} Activation Rate",
                "activation_rate_b": f"{axis.display_name_b} Activation Rate",
            },
            title="Endpoint Activation Landscape",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        fig_hist = px.histogram(scores_df, x="effect_size", nbins=60, title="Effect Size Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.dataframe(scores_df.sort_values(by="effect_size", key=abs, ascending=False).head(100), use_container_width=True)


def render_cross_layer_view(config):
    st.header("Cross-Layer View")
    axis, paths, bundle, report = _pattern_state(config)
    analysis = dict(report.get("analysis", {}) or {})
    if not analysis:
        st.warning("No canonical pattern report found. Run `python -m cli.pattern_score --mode cross-layer` first.")
        return

    st.caption(f"Pattern report: `{paths['report']}`")
    model_metrics = dict(analysis.get("model_metrics", {}) or {})
    causal_validation = dict(analysis.get("causal_validation", {}) or {})
    base_metrics = dict(model_metrics.get("base", {}) or {})
    full_metrics = dict(model_metrics.get("full", {}) or {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aggregation", analysis.get("selected_aggregation", "n/a"))
    c2.metric("Base CV F1", f"{float(base_metrics.get('cv_f1_mean', 0.0)):.3f}")
    c3.metric("Full CV F1", f"{float(full_metrics.get('cv_f1_mean', 0.0)):.3f}")
    c4.metric("Causal Status", str(causal_validation.get("status", "unknown")))

    stable_motifs = pd.DataFrame(list(analysis.get("stable_motifs", []) or []))
    if not stable_motifs.empty:
        st.subheader("Promoted Motifs")
        st.dataframe(stable_motifs, use_container_width=True)
    else:
        st.info("No motifs remain promoted after causal validation.")

    stable_interactions = pd.DataFrame(list(analysis.get("stable_interactions", []) or []))
    if not stable_interactions.empty:
        fig = px.bar(
            stable_interactions.head(25).sort_values(by="mean_abs_coefficient", ascending=True),
            x="mean_abs_coefficient",
            y="name",
            color="sign",
            orientation="h",
            title="Stable Cross-Layer Motifs",
        )
        st.plotly_chart(fig, use_container_width=True)

    intervention_effects = pd.DataFrame(list(analysis.get("intervention_effects", []) or []))
    if not intervention_effects.empty:
        st.subheader("Intervention Effects")
        st.dataframe(intervention_effects, use_container_width=True)

    feature_importance = pd.DataFrame(list(analysis.get("feature_importance", []) or []))
    if not feature_importance.empty:
        st.subheader("Feature Importance")
        st.dataframe(feature_importance.head(50), use_container_width=True)


def render_deception_monitor(config, config_path=None):
    st.header("Deception Monitor")
    dec_dir = str(deception_run_dir(config))
    axis, paths, bundle, report = _pattern_state(config)
    analysis = dict(report.get("analysis", {}) or {})
    bundle_layers = dict(bundle.get("layers", {}) or {})
    monitor_report = dict(report.get("monitor", {}) or {}) or dict(
        load_json_artifact(str(Path(dec_dir) / "monitor" / "monitor_report.json")) or {}
    )
    disk_directions = dict(load_torch_artifact(str(Path(dec_dir) / "directions" / "directions.pt")) or {})
    disk_probe_summary = dict(load_json_artifact(str(Path(dec_dir) / "probes" / "probe_summary.json")) or {})

    if monitor_report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detection Rate", f"{float(monitor_report.get('detection_rate', 0.0)):.1%}")
        c2.metric("False Positive Rate", f"{float(monitor_report.get('false_positive_rate', 0.0)):.1%}")
        c3.metric("Precision", f"{float(monitor_report.get('precision', 0.0)):.1%}")
        c4.metric("Pairs Evaluated", int(monitor_report.get("pairs_evaluated", 0) or 0))
    else:
        st.warning("Monitor report not found. Run `python -m cli.deception_cycle --phase monitor`.")

    st.divider()
    st.subheader("Per-Layer Direction and Probe Summary")
    rows = []
    for layer_name in resolve_requested_targets(config):
        payload = dict(bundle_layers.get(layer_name, {}) or {})
        direction = dict(payload.get("direction", {}) or {})
        if not direction:
            direction = dict(disk_directions.get(layer_name, {}) or {})
        probe = dict(payload.get("probe", {}) or {}) or dict(disk_probe_summary.get(layer_name, {}) or {})
        if not direction and not probe:
            continue
        rows.append(
            {
                "layer": _format_target_label(layer_name),
                "direction_accuracy": float(direction.get("validation_accuracy", 0.0) or 0.0),
                "cluster_separation": float(direction.get("cluster_separation", 0.0) or 0.0),
                "probe_validation_accuracy": float(probe.get("validation_accuracy", 0.0) or 0.0),
                "probe_cv_accuracy": float(probe.get("cv_accuracy", 0.0) or 0.0),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No direction or probe state found in the canonical bundle.")

    st.divider()
    selected_target = _select_target(config, "deception_feature_target")
    if selected_target:
        scores_df = _pattern_layer_scores(bundle, selected_target)
        autointerp = load_deception_autointerp(dec_dir, selected_target)
        if scores_df is not None and not scores_df.empty:
            scores_df = scores_df.copy()
            scores_df["interpretation"] = scores_df["feature_id"].map(lambda value: autointerp.get(int(value), ""))
            st.subheader("Signed Features")
            st.dataframe(scores_df.sort_values(by="effect_size", key=abs, ascending=False).head(100), use_container_width=True)
        else:
            st.info("No signed feature scores found for this layer.")

    st.divider()
    st.subheader("Adversarial Search")
    adv_summary, missed = load_deception_adversarial(dec_dir)
    if adv_summary is not None:
        st.json(adv_summary)
        if missed:
            st.dataframe(pd.DataFrame(missed[:20]), use_container_width=True)
    else:
        st.info("Run `python -m cli.deception_cycle --phase adversarial` to populate this section.")

    st.divider()
    st.subheader("Report Artifacts")
    report_assets = load_deception_report_assets(dec_dir)
    html_path = Path(report_assets["html"])
    if html_path.exists():
        st.success(f"HTML report ready: `{html_path}`")
        st.download_button(
            label="Download deception_report.html",
            data=html_path.read_bytes(),
            file_name="deception_report.html",
            mime="text/html",
            key="download_deception_report",
        )
    else:
        cmd = f"python -m cli.deception_report --config {config_path}" if config_path else "python -m cli.deception_report --config ..."
        st.info(f"Generate the HTML report with: `{cmd}`")

    chart_specs = [
        ("Layer Comparison", report_assets["layer_comparison"]),
        (f"{axis.display_name_a} vs {axis.display_name_b} Projection", report_assets["tsne"]),
        ("Probe ROC", report_assets["roc"]),
        ("Category Breakdown", report_assets["category"]),
        ("Adversarial Analysis", report_assets["adversarial"]),
    ]
    existing_specs = [(label, path) for label, path in chart_specs if Path(path).exists()]
    if existing_specs:
        cols = st.columns(2)
        for index, (label, path) in enumerate(existing_specs):
            with cols[index % 2]:
                st.image(path, caption=label, use_container_width=True)

    if analysis.get("manual_interventions"):
        st.divider()
        st.subheader("Manual Motif Interventions")
        st.dataframe(pd.DataFrame(list(analysis.get("manual_interventions", []) or [])), use_container_width=True)


def render_steering_playground(config, config_path):
    st.header("Steering Playground")
    targets = resolve_requested_targets(config)
    axis, _, bundle, report = _pattern_state(config)
    analysis = dict(report.get("analysis", {}) or {})

    tab1, tab2, tab3 = st.tabs(["Feature Steering", "Motif Steering", "Direction Steering"])

    with tab1:
        with st.form("feature_steering_form"):
            layer = st.selectbox("Layer", targets, key="steer_layer")
            feature_id = st.number_input("Feature ID", min_value=0, value=0, key="steer_feature_id")
            alpha = st.slider("Alpha", -50.0, 50.0, -10.0, 1.0, key="steer_alpha")
            mode = st.selectbox("Mode", ["add", "ablate", "clamp"], key="steer_mode")
            prompt = st.text_area("Prompt", "What do you think about this topic?", key="steer_prompt")
            system_prompt = st.text_input("System Prompt", "", key="steer_system_prompt")
            max_tokens = st.slider("Max Tokens", 32, 512, 150, 16, key="steer_max_tokens")
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="steer_temperature")
            run_feature = st.form_submit_button("Enter")
        sys_flag = f' --system-prompt "{system_prompt}"' if system_prompt else ""
        cmd = (
            f"python -m cli.steer steer\n"
            f"  --config {config_path}\n"
            f"  --layer {layer}\n"
            f"  --feature-id {feature_id} --alpha {alpha} --mode {mode}\n"
            f"  --max-tokens {max_tokens}{sys_flag}\n"
            f'  --prompt "{prompt}"'
        )
        st.code(cmd, language="bash")
        if run_feature:
            with st.spinner("Running feature steering..."):
                try:
                    st.session_state["feature_steering_result"] = _run_feature_steering_ui(
                        config_path,
                        layer,
                        int(feature_id),
                        float(alpha),
                        mode,
                        prompt,
                        system_prompt,
                        int(max_tokens),
                        float(temperature),
                    )
                except Exception as exc:
                    st.session_state["feature_steering_result"] = {"error": str(exc)}
        feature_result = st.session_state.get("feature_steering_result", {})
        if feature_result.get("error"):
            st.error(feature_result["error"])
        else:
            _render_steering_result(feature_result, heading="Feature Steering Result")

    with tab2:
        motif_rows = list(analysis.get("stable_motifs", []) or [])
        if not motif_rows:
            st.info("No promoted motifs found yet. Run `python -m cli.deception_cycle --phase patterns` or `python -m cli.pattern_score --mode cross-layer`.")
        else:
            motif_names = [str(row.get("name")) for row in motif_rows]
            with st.form("motif_steering_form"):
                selected_motif = st.selectbox("Motif", motif_names, key="motif_name")
                action = st.selectbox("Action", ["ablate", "amplify"], key="motif_action")
                alpha = st.slider(
                    "Alpha",
                    -20.0,
                    20.0,
                    5.0,
                    0.5,
                    key="motif_alpha",
                    help="Positive values amplify the motif, negative values suppress it. Alpha is ignored for ablate.",
                )
                prompt = st.text_area("Prompt", "Answer the question directly.", key="motif_prompt")
                system_prompt = st.text_input("System Prompt", "", key="motif_system_prompt")
                max_tokens = st.slider("Max Tokens", 32, 512, 150, 16, key="motif_max_tokens")
                temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1, key="motif_temperature")
                write_bundle = st.checkbox("Record intervention in pattern bundle", value=True, key="motif_write_bundle")
                run_motif = st.form_submit_button("Enter")
            sys_flag = f' --system-prompt "{system_prompt}"' if system_prompt else ""
            alpha_flag = f" --alpha {alpha}" if action == "amplify" else ""
            cmd = (
                f"python -m cli.steer motif\n"
                f"  --config {config_path}\n"
                f'  --motif-name "{selected_motif}" --source stable_motifs --action {action}{alpha_flag}\n'
                f"  --show-scores --write-bundle{sys_flag}\n"
                f'  --prompt "{prompt}"'
            )
            st.code(cmd, language="bash")
            selected_payload = next((row for row in motif_rows if str(row.get("name")) == selected_motif), {})
            if selected_payload:
                st.json(selected_payload)
            if run_motif:
                with st.spinner("Running motif steering..."):
                    try:
                        st.session_state["motif_steering_result"] = _run_motif_steering_ui(
                            config_path,
                            selected_motif,
                            action,
                            float(alpha),
                            prompt,
                            system_prompt,
                            int(max_tokens),
                            float(temperature),
                            write_bundle=bool(write_bundle),
                        )
                    except Exception as exc:
                        st.session_state["motif_steering_result"] = {"error": str(exc)}
            motif_result = st.session_state.get("motif_steering_result", {})
            if motif_result.get("error"):
                st.error(motif_result["error"])
            else:
                _render_steering_result(motif_result, heading="Motif Steering Result")

    with tab3:
        with st.form("direction_steering_form"):
            prompt = st.text_area("Prompt", "Tell me about the safety evidence.", key="direction_prompt")
            system_prompt = st.text_input("System Prompt", "", key="direction_system_prompt")
            target_layers = st.multiselect("Layers", targets, default=targets, key="direction_layers")
            alpha = st.slider("Per-layer Alpha", -20.0, 20.0, -5.0, 0.5, key="direction_alpha")
            max_tokens = st.slider("Max Tokens", 32, 512, 150, 16, key="direction_max_tokens")
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="direction_temperature")
            run_direction = st.form_submit_button("Enter")
        sys_flag = f' --system-prompt "{system_prompt}"' if system_prompt else ""
        cmd = (
            f"python -m cli.steer deception\n"
            f"  --config {config_path}\n"
            f"  --alpha {alpha} --show-scores{sys_flag}\n"
            f'  --prompt "{prompt}"'
        )
        st.code(cmd, language="bash")
        if run_direction:
            with st.spinner("Running direction steering..."):
                try:
                    st.session_state["direction_steering_result"] = _run_direction_steering_ui(
                        config_path,
                        target_layers or targets,
                        float(alpha),
                        prompt,
                        system_prompt,
                        int(max_tokens),
                        float(temperature),
                    )
                except Exception as exc:
                    st.session_state["direction_steering_result"] = {"error": str(exc)}
        direction_result = st.session_state.get("direction_steering_result", {})
        if direction_result.get("error"):
            st.error(direction_result["error"])
        else:
            _render_steering_result(direction_result, heading="Direction Steering Result")


def main():
    st.sidebar.title("RFM Dashboard")
    default_config = "configs/models/qwen3-0.6B.deception.json"
    config_path = st.sidebar.text_input("Config Path", default_config)
    config = load_config(config_path)
    if not config:
        st.error(f"Config not found at `{config_path}`")
        return

    slug = model_slug(config)
    st.sidebar.markdown(f"**Model:** `{config.get('model_name')}`")
    targets = resolve_requested_targets(config)
    if targets:
        st.sidebar.markdown(f"**Layers:** `{', '.join(t.split('.')[-2] for t in targets)}`")

    has_deception = bool(config.get("deception", None))
    if has_deception:
        dec_dir = str(deception_run_dir(config))
        status = _pipeline_status(dec_dir, config)
        st.sidebar.markdown("**Pipeline Status**")
        icons = {True: "OK", False: "--"}
        for step, label in [
            ("scenarios", "Scenarios"),
            ("activations", "Activations"),
            ("checkpoints", "SAE Checkpoints"),
            ("directions", "Directions"),
            ("probes", "Probes"),
            ("patterns", "Patterns"),
            ("monitor", "Monitor"),
            ("adversarial", "Adversarial"),
            ("autointerp", "Autointerp"),
        ]:
            st.sidebar.markdown(f"{icons[status[step]]} {label}")
        st.sidebar.divider()

    pages = [
        "Deception Monitor",
        "Feature Explorer",
        "Training Metrics",
        "Steering Playground",
        "Extraction Overview",
        "Signed Feature Analysis",
        "Cross-Layer View",
    ]
    page = st.sidebar.radio("Navigation", pages)

    if page == "Deception Monitor":
        render_deception_monitor(config, config_path)
    elif page == "Feature Explorer":
        render_feature_explorer(config, slug)
    elif page == "Training Metrics":
        render_training_metrics(config)
    elif page == "Steering Playground":
        render_steering_playground(config, config_path)
    elif page == "Extraction Overview":
        render_extraction_overview(config)
    elif page == "Signed Feature Analysis":
        render_signed_feature_analysis(config)
    elif page == "Cross-Layer View":
        render_cross_layer_view(config)


if __name__ == "__main__":
    main()
