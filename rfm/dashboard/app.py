import json
import os
import sys
import subprocess
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
    default_safety_scores_dir,
    default_feature_mapping_dir,
    model_slug,
    resolve_activations_dir,
    resolve_best_checkpoint,
    resolve_requested_targets,
    sanitize_layer_name,
)

st.set_page_config(page_title="RFM Dashboard", layout="wide", page_icon="🧠")


# ==========================================
# Data Loading & Utils
# ==========================================

@st.cache_data
def load_config(config_path):
    if not os.path.exists(config_path):
        return None
    return ConfigManager.from_file(config_path)


@st.cache_data
def load_feature_summary(mapping_dir, slug):
    candidates = [
        Path(mapping_dir) / "feature_mapping_feature_summary.csv",
        Path(mapping_dir) / f"{slug}_feature_summary.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is not None:
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_events(mapping_dir, slug):
    candidates = [
        Path(mapping_dir) / "feature_mapping_events.csv",
        Path(mapping_dir) / f"{slug}_feature_events.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is not None:
        return pd.read_csv(path)
    return None


@st.cache_data
def load_autointerp(mapping_dir, slug):
    path = Path(mapping_dir) / f"{slug}_autointerp_results.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


@st.cache_data
def load_training_history(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return state.get("history", [])
    except Exception as e:
        st.error(f"Error loading checkpoint history: {e}")
        return None


@st.cache_data
def load_contrastive_scores(safety_scores_dir, layer_name, mapping_dir=None, slug=None):
    safe = sanitize_layer_name(layer_name)
    candidates = [Path(safety_scores_dir) / f"{safe}_contrastive.csv"]
    if mapping_dir and slug:
        candidates.append(Path(mapping_dir) / f"{slug}_contrastive_scores.csv")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is not None:
        return pd.read_csv(path)
    return None


@st.cache_data
def load_cross_layer_combinations(safety_scores_dir):
    path = Path(safety_scores_dir) / "cross_layer_combinations.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json_artifact(path_str):
    path = Path(path_str)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ==========================================
# Deception Monitor Data Loaders
# ==========================================

@st.cache_data
def load_deception_monitor_report(deception_dir: str):
    path = Path(deception_dir) / "monitor" / "monitor_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_deception_probe_summary(deception_dir: str):
    path = Path(deception_dir) / "probes" / "probe_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_deception_directions(deception_dir: str):
    path = Path(deception_dir) / "directions" / "directions.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for layer, d in payload.items():
        result[layer] = {
            "validation_accuracy": float(d.get("validation_accuracy", 0)),
            "cluster_separation": float(d.get("cluster_separation", 0)),
            "explained_variance": float(d.get("explained_variance", 0)),
            "threshold": float(d.get("threshold", 0)),
            "method": str(d.get("method", "unknown")),
        }
    return result


@st.cache_data
def load_deception_contrastive_csv(safety_scores_dir: str, layer_name: str):
    safe = sanitize_layer_name(layer_name)
    path = Path(safety_scores_dir) / f"{safe}_contrastive.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


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
    return {
        key: str(value)
        for key, value in paths.items()
    }


@st.cache_data
def load_deception_autointerp(deception_dir: str, layer_name: str):
    safe = sanitize_layer_name(layer_name)
    path = Path(deception_dir) / "autointerp" / f"{safe}_interpretations.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    return {}


@st.cache_data
def load_extraction_metadata(_config):
    targets = resolve_requested_targets(_config)
    if not targets:
        return None

    target = targets[0]
    act_dir = resolve_activations_dir(_config, target=target)

    meta_files = list(Path(act_dir).glob("*.meta.json"))
    if not meta_files:
        return None

    all_data = []
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

            n_seq = len(chunk_data.get("labels", []))
            labels = chunk_data.get("labels", [])
            categories = chunk_data.get("categories", [])
            lengths = chunk_data.get("token_lengths", [])
            p_lengths = chunk_data.get("prompt_lengths", [])
            scores = chunk_data.get("scores", [])

            for i in range(n_seq):
                all_data.append({
                    "chunk_id": chunk_data.get("chunk_id", 0),
                    "label": labels[i] if i < len(labels) else "unknown",
                    "category": categories[i] if i < len(categories) else "unknown",
                    "token_length": lengths[i] if i < len(lengths) else 0,
                    "prompt_length": p_lengths[i] if i < len(p_lengths) else 0,
                    "score": scores[i] if i < len(scores) else 0.0,
                })

    if not all_data:
        return None
    return pd.DataFrame(all_data)


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


def _normalize_feature_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if "count_active" in df.columns and "num_events" not in df.columns:
        df["num_events"] = df["count_active"]
    if "mean_strength" in df.columns and "mean_activation" not in df.columns:
        df["mean_activation"] = df["mean_strength"]
    if "max_strength" in df.columns and "max_activation" not in df.columns:
        df["max_activation"] = df["max_strength"]
    if "feature_id" in df.columns:
        df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce").astype("Int64")
    if "top_tokens" not in df.columns:
        df["top_tokens"] = ""
    return df


def _normalize_feature_events(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    if "strength" in df.columns and "activation" not in df.columns:
        df["activation"] = df["strength"]
    if "prompt_preview" in df.columns and "context" not in df.columns:
        df["context"] = df["prompt_preview"]
    if "feature_id" in df.columns:
        df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce").astype("Int64")
    if "token_str" not in df.columns:
        df["token_str"] = ""
    if "context" not in df.columns:
        df["context"] = ""
    return df


def _contrastive_column_info(scores_df: pd.DataFrame) -> dict:
    positive_label = None
    negative_label = None
    if "positive_label" in scores_df.columns and not scores_df.empty:
        positive_label = str(scores_df["positive_label"].iloc[0])
    if "negative_label" in scores_df.columns and not scores_df.empty:
        negative_label = str(scores_df["negative_label"].iloc[0])

    rate_cols = [col for col in scores_df.columns if col.endswith("_rate")]
    strength_cols = [col for col in scores_df.columns if col.endswith("_strength")]
    pos_rate_col = f"{positive_label}_rate" if positive_label and f"{positive_label}_rate" in scores_df.columns else None
    neg_rate_col = f"{negative_label}_rate" if negative_label and f"{negative_label}_rate" in scores_df.columns else None
    pos_strength_col = (
        f"{positive_label}_strength" if positive_label and f"{positive_label}_strength" in scores_df.columns else None
    )
    neg_strength_col = (
        f"{negative_label}_strength" if negative_label and f"{negative_label}_strength" in scores_df.columns else None
    )

    if pos_rate_col is None and rate_cols:
        pos_rate_col = next((col for col in rate_cols if "deceptive" in col or "toxic" in col), rate_cols[0])
    if neg_rate_col is None:
        neg_rate_col = next((col for col in rate_cols if col != pos_rate_col), None)
    if pos_strength_col is None and strength_cols:
        pos_strength_col = next((col for col in strength_cols if "deceptive" in col or "toxic" in col), strength_cols[0])
    if neg_strength_col is None:
        neg_strength_col = next((col for col in strength_cols if col != pos_strength_col), None)

    if positive_label is None and pos_rate_col:
        positive_label = pos_rate_col.removesuffix("_rate")
    if negative_label is None and neg_rate_col:
        negative_label = neg_rate_col.removesuffix("_rate")

    return {
        "positive_label": positive_label or "positive",
        "negative_label": negative_label or "negative",
        "pos_rate_col": pos_rate_col,
        "neg_rate_col": neg_rate_col,
        "pos_strength_col": pos_strength_col,
        "neg_strength_col": neg_strength_col,
    }


# ==========================================
# General Dashboard Views
# ==========================================

def render_feature_explorer(config, slug, mapping_dir):
    st.header("🔍 Feature Explorer")

    summary_df = load_feature_summary(mapping_dir, slug)
    events_df = load_feature_events(mapping_dir, slug)
    autointerp_data = load_autointerp(mapping_dir, slug)

    if summary_df is None or events_df is None:
        st.warning("Mapping data not found. Please run feature mapping first.")
        return

    st.write(f"Total mapped features: **{len(summary_df)}**")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Features")
        search_kw = st.text_input("Search context/interpretation...", "")

        display_df = summary_df.copy()
        display_df["Interp"] = display_df["feature_id"].map(lambda x: autointerp_data.get(x, "No interpretation"))

        if search_kw:
             display_df = display_df[display_df["Interp"].str.contains(search_kw, case=False)]

        sort_by = st.selectbox("Sort by", ["num_events", "mean_activation", "max_activation"], index=0)
        display_df = display_df.sort_values(by=sort_by, ascending=False)

        selected_feature = st.selectbox(
            "Select Feature:",
            display_df["feature_id"].tolist(),
            format_func=lambda x: f"F{x} ({display_df[display_df['feature_id'] == x][sort_by].values[0]:.2f})"
        )

    with col2:
        if selected_feature is not None:
            st.subheader(f"Feature {selected_feature} Details")

            interp = autointerp_data.get(selected_feature)
            if interp:
                st.info(f"**Auto-Interpretation:** {interp}")

            stats = summary_df[summary_df["feature_id"] == selected_feature].iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Events Found", stats["num_events"])
            m2.metric("Mean Act", f"{stats['mean_activation']:.3f}")
            m3.metric("Max Act", f"{stats['max_activation']:.3f}")

            st.markdown("#### Top Activating Contexts")
            feature_events = events_df[events_df["feature_id"] == selected_feature].sort_values(by="activation", ascending=False).head(20)

            for _, row in feature_events.iterrows():
                ctx = row["context"]
                tok = row["token_str"]
                act = row["activation"]

                ctx_html = str(ctx).replace("<", "&lt;").replace(">", "&gt;")
                tok_html = str(tok).replace("<", "&lt;").replace(">", "&gt;")
                highlighted = ctx_html.replace(tok_html, f"<span style='background-color: rgba(255, 100, 100, {min(act/10.0, 1.0)}); font-weight: bold;'>{tok_html}</span>")

                st.markdown(f"**Act:** `{act:.3f}` | {highlighted}", unsafe_allow_html=True)


def render_training_metrics(config, checkpoint_path):
    st.header("📈 Training Metrics")

    history = load_training_history(checkpoint_path)
    if not history:
        st.warning("Training history not found in checkpoint.")
        return

    epoch_metrics = [h for h in history if "epoch" in h]
    dead_analysis = [h for h in history if "dead_feature_analysis" in h]

    if not epoch_metrics:
        st.info("No epoch metrics found in history.")
        return

    df = pd.DataFrame(epoch_metrics)

    col1, col2 = st.columns(2)

    with col1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Total Loss"))
        if "val_loss" in df.columns:
             fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val Total Loss"))
        fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

        fig_components = go.Figure()
        fig_components.add_trace(go.Scatter(x=df["epoch"], y=df["train_recon"], name="Recon Loss"))
        fig_components.add_trace(go.Scatter(x=df["epoch"], y=df["train_sparse"], name="Sparse Loss"))
        fig_components.update_layout(title="Loss Components", xaxis_title="Epoch")
        st.plotly_chart(fig_components, use_container_width=True)

    with col2:
         fig_act = go.Figure()
         fig_act.add_trace(go.Scatter(x=df["epoch"], y=df["train_active_rate"], name="Train Active Rate"))
         if "val_active_rate" in df.columns:
              fig_act.add_trace(go.Scatter(x=df["epoch"], y=df["val_active_rate"], name="Val Active Rate"))
         fig_act.update_layout(title="Feature Active Rate (L0 proxy)", xaxis_title="Epoch", yaxis_title="Active Rate")
         st.plotly_chart(fig_act, use_container_width=True)

         if dead_analysis:
              st.subheader("Dead Feature Analysis")
              info = dead_analysis[-1]["dead_feature_analysis"]

              m1, m2 = st.columns(2)
              m1.metric("Dead Features", f"{info['dead_count']} / {info['total_features']}")
              m2.metric("Dead Ratio", f"{info['dead_ratio']:.2%}")

              fig_pie = px.pie(
                   values=[info["alive_count"], info["dead_count"]],
                   names=["Alive", "Dead"],
                   title="Feature Liveliness"
              )
              st.plotly_chart(fig_pie, use_container_width=True)


def render_extraction_overview(config):
    st.header("📊 Extraction Overview")

    meta_df = load_extraction_metadata(config)
    if meta_df is None or meta_df.empty:
        st.warning("No metadata files found in the activations directory. Run extract_generate first.")
        return

    st.success(f"Successfully loaded {len(meta_df)} sequences across {meta_df['chunk_id'].nunique()} chunks.")

    col1, col2 = st.columns(2)
    with col1:
        fig_cat = px.histogram(
            meta_df, x="category", color="label",
            barmode="group", title="Category Distribution by Toxicity"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        fig_len = px.box(
            meta_df, x="label", y="token_length", color="label",
            points="all", title="Response Token Lengths"
        )
        st.plotly_chart(fig_len, use_container_width=True)

    st.subheader("Classifier Confidence")
    fig_conf = px.histogram(
        meta_df, x="score", color="label",
        nbins=50, title="Safety Classifier Score Distribution"
    )
    st.plotly_chart(fig_conf, use_container_width=True)


def render_contrastive_analysis(config, slug, mapping_dir):
    st.header("🔬 Contrastive Analysis")
    scores_df = load_contrastive_scores(mapping_dir, slug)
    if scores_df is None:
        st.warning(f"No contrastive scores found for `{slug}`. Run safety scoring pipeline first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(
            scores_df, x="fisher_score", y="rate_ratio",
            color="direction", hover_data=["feature_id", "risk_score"],
            log_x=True, log_y=True, title="Feature Risk Landscape"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        fig_dist = px.violin(
            scores_df, x="direction", y="risk_score",
            box=True, points="all", title="Risk Score Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Top Risk Features")
    st.dataframe(scores_df.sort_values(by="risk_score", key=abs, ascending=False).head(100), use_container_width=True)


def render_cross_layer_view():
    st.header("🧬 Cross-Layer View")
    st.info("Cross-layer feature combinations and Decision Tree rules (NetworkX/Plotly) will be displayed here.")
    st.markdown("Run `cli.safety_score --mode cross-layer` to generate the JSON reports for this view.")


# Feature Explorer / Metrics / Safety View overrides

def render_feature_explorer(config, slug, mapping_dir=None):
    st.header("ğŸ” Feature Explorer")

    selected_target = _select_target(config, "feature_layer_select")
    if not selected_target:
        st.warning("No target layers configured.")
        return

    mapping_dir = default_feature_mapping_dir(config, target=selected_target)
    summary_df = load_feature_summary(mapping_dir, slug)
    events_df = load_feature_events(mapping_dir, slug)
    autointerp_data = load_autointerp(mapping_dir, slug)

    if summary_df is not None and events_df is not None:
        summary_df = _normalize_feature_summary(summary_df)
        events_df = _normalize_feature_events(events_df)

        st.caption(f"Mapping directory: `{mapping_dir}`")
        st.write(f"Total mapped features: **{len(summary_df)}**")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Features")
            search_kw = st.text_input("Search token/interpretation...", "", key="mapping_search")

            display_df = summary_df.copy()
            display_df["Interp"] = display_df["feature_id"].map(
                lambda x: autointerp_data.get(int(x), "No interpretation") if pd.notna(x) else "No interpretation"
            )

            if search_kw:
                search_mask = (
                    display_df["Interp"].fillna("").str.contains(search_kw, case=False, na=False)
                    | display_df["top_tokens"].fillna("").str.contains(search_kw, case=False, na=False)
                )
                display_df = display_df[search_mask]

            sort_options = [col for col in ["num_events", "mean_activation", "max_activation"] if col in display_df.columns]
            sort_by = st.selectbox("Sort by", sort_options, index=0, key="mapping_sort")
            display_df = display_df.sort_values(by=sort_by, ascending=False)

            feature_options = [int(fid) for fid in display_df["feature_id"].dropna().tolist()]
            if not feature_options:
                st.warning("No features match the current filter.")
                return

            selected_feature = st.selectbox(
                "Select Feature:",
                feature_options,
                format_func=lambda x: f"F{x} ({display_df.loc[display_df['feature_id'] == x, sort_by].iloc[0]:.2f})",
                key="mapping_feature_select",
            )

        with col2:
            st.subheader(f"Feature {selected_feature} Details")

            interp = autointerp_data.get(selected_feature)
            if interp:
                st.info(f"**Auto-Interpretation:** {interp}")

            stats = summary_df[summary_df["feature_id"] == selected_feature].iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Events Found", int(stats["num_events"]))
            m2.metric("Mean Act", f"{stats['mean_activation']:.3f}")
            m3.metric("Max Act", f"{stats['max_activation']:.3f}")

            if stats.get("top_tokens"):
                st.caption(f"Top tokens: {stats['top_tokens']}")

            st.markdown("#### Top Activating Contexts")
            feature_events = events_df[events_df["feature_id"] == selected_feature].sort_values(
                by="activation", ascending=False
            ).head(20)

            if feature_events.empty:
                st.info("No activation events found for this feature.")
            else:
                for _, row in feature_events.iterrows():
                    st.markdown(
                        f"**Act:** `{row['activation']:.3f}` | **Token:** `{row.get('token_str', '')}`\n\n"
                        f"{row.get('context', '')}"
                    )
                    st.divider()
        return

    if config.get("deception", None):
        dec_dir = str(deception_run_dir(config))
        safety_scores_dir = default_safety_scores_dir(config, target=selected_target)
        scores_df = load_contrastive_scores(safety_scores_dir, selected_target)
        autointerp_data = load_deception_autointerp(dec_dir, selected_target)

        if scores_df is None or scores_df.empty:
            st.warning(
                "No feature mapping CSVs or deception contrastive scores found for this layer. "
                "Run feature mapping or `python -m cli.safety_score --mode contrastive` first."
            )
            return

        info = _contrastive_column_info(scores_df)
        st.caption(f"Contrastive directory: `{safety_scores_dir}`")
        st.write(f"Total scored features: **{len(scores_df)}**")

        scores_df = scores_df.copy()
        scores_df["interpretation"] = scores_df["feature_id"].map(
            lambda x: autointerp_data.get(int(x), "") if autointerp_data else ""
        )

        search_kw = st.text_input("Search interpretation...", "", key="deception_feature_search")
        if search_kw:
            scores_df = scores_df[scores_df["interpretation"].str.contains(search_kw, case=False, na=False)]

        sort_options = [col for col in ["risk_score", "rate_ratio", "fisher_score", "cohens_d"] if col in scores_df.columns]
        sort_by = st.selectbox("Sort by", sort_options, index=0, key="deception_feature_sort")
        scores_df = scores_df.sort_values(by=sort_by, ascending=False)

        feature_options = scores_df["feature_id"].astype(int).tolist()
        if not feature_options:
            st.warning("No features match the current filter.")
            return

        selected_feature = st.selectbox(
            "Select Feature",
            feature_options,
            format_func=lambda x: f"F{x}",
            key="deception_feature_select",
        )

        feature_row = scores_df[scores_df["feature_id"] == selected_feature].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Score", f"{feature_row['risk_score']:.4f}")
        if info["pos_rate_col"]:
            m2.metric(f"{info['positive_label'].title()} Rate", f"{feature_row[info['pos_rate_col']]:.4f}")
        if info["neg_rate_col"]:
            m3.metric(f"{info['negative_label'].title()} Rate", f"{feature_row[info['neg_rate_col']]:.4f}")
        if "rate_ratio" in feature_row:
            m4.metric("Rate Ratio", f"{feature_row['rate_ratio']:.2f}")

        detail_cols = [
            col for col in [
                "feature_id",
                "risk_score",
                "rate_ratio",
                "fisher_score",
                "cohens_d",
                info["pos_rate_col"],
                info["neg_rate_col"],
                info["pos_strength_col"],
                info["neg_strength_col"],
                "direction",
                "interpretation",
            ]
            if col and col in scores_df.columns
        ]
        st.dataframe(scores_df[detail_cols].head(100), use_container_width=True)

        if autointerp_data and selected_feature in autointerp_data:
            st.info(f"**Auto-Interpretation:** {autointerp_data[selected_feature]}")
        elif not autointerp_data:
            st.caption(
                "LLM interpretation not found for this layer. "
                "Run `python -m cli.deception_autointerp --config ...` to enrich the explorer."
            )
        return

    st.warning("Mapping data not found. Please run feature mapping first.")


def render_training_metrics(config, checkpoint_path=None):
    st.header("ğŸ“ˆ Training Metrics")

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

    epoch_metrics = [h for h in history if "epoch" in h]
    dead_analysis = [h for h in history if "dead_feature_analysis" in h]

    if not epoch_metrics:
        st.info("No epoch metrics found in history.")
        return

    df = pd.DataFrame(epoch_metrics)
    col1, col2 = st.columns(2)

    with col1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Total Loss"))
        if "val_loss" in df.columns:
            fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val Total Loss"))
        fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

        fig_components = go.Figure()
        fig_components.add_trace(go.Scatter(x=df["epoch"], y=df["train_recon"], name="Recon Loss"))
        fig_components.add_trace(go.Scatter(x=df["epoch"], y=df["train_sparse"], name="Sparse Loss"))
        fig_components.update_layout(title="Loss Components", xaxis_title="Epoch")
        st.plotly_chart(fig_components, use_container_width=True)

    with col2:
        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(x=df["epoch"], y=df["train_active_rate"], name="Train Active Rate"))
        if "val_active_rate" in df.columns:
            fig_act.add_trace(go.Scatter(x=df["epoch"], y=df["val_active_rate"], name="Val Active Rate"))
        fig_act.update_layout(title="Feature Active Rate (L0 proxy)", xaxis_title="Epoch", yaxis_title="Active Rate")
        st.plotly_chart(fig_act, use_container_width=True)

        if dead_analysis:
            st.subheader("Dead Feature Analysis")
            info = dead_analysis[-1]["dead_feature_analysis"]

            m1, m2 = st.columns(2)
            m1.metric("Dead Features", f"{info['dead_count']} / {info['total_features']}")
            m2.metric("Dead Ratio", f"{info['dead_ratio']:.2%}")

            fig_pie = px.pie(
                values=[info["alive_count"], info["dead_count"]],
                names=["Alive", "Dead"],
                title="Feature Liveliness"
            )
            st.plotly_chart(fig_pie, use_container_width=True)


def render_contrastive_analysis(config, slug, mapping_dir=None):
    st.header("ğŸ”¬ Contrastive Analysis")

    selected_target = _select_target(config, "contrastive_layer_select")
    if not selected_target:
        st.warning("No target layers configured.")
        return

    mapping_dir = default_feature_mapping_dir(config, target=selected_target)
    safety_scores_dir = default_safety_scores_dir(config, target=selected_target)
    scores_df = load_contrastive_scores(safety_scores_dir, selected_target, mapping_dir=mapping_dir, slug=slug)
    if scores_df is None:
        st.warning(
            f"No contrastive scores found for `{selected_target}`. "
            "Run `python -m cli.safety_score --config ... --mode contrastive` first."
        )
        return

    info = _contrastive_column_info(scores_df)
    st.caption(f"Safety scores directory: `{safety_scores_dir}`")

    k1, k2, k3 = st.columns(3)
    k1.metric("Features", len(scores_df))
    k2.metric("Top Risk Score", f"{scores_df['risk_score'].max():.4f}")
    k3.metric("Mean Risk Score", f"{scores_df['risk_score'].mean():.4f}")

    col1, col2 = st.columns(2)
    with col1:
        if info["pos_rate_col"] and info["neg_rate_col"]:
            fig_scatter = px.scatter(
                scores_df,
                x=info["neg_rate_col"],
                y=info["pos_rate_col"],
                color="risk_score",
                hover_data=["feature_id", "rate_ratio", "fisher_score"],
                title="Feature Risk Landscape",
                labels={
                    info["neg_rate_col"]: f"{info['negative_label'].title()} Rate",
                    info["pos_rate_col"]: f"{info['positive_label'].title()} Rate",
                },
                color_continuous_scale="RdYlGn_r",
            )
            max_ref = float(scores_df[[info["neg_rate_col"], info["pos_rate_col"]]].max().max())
            fig_scatter.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=max_ref,
                y1=max_ref,
                line=dict(color="gray", dash="dash", width=1),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            fig_scatter = px.scatter(
                scores_df,
                x="fisher_score",
                y="rate_ratio",
                color="direction" if "direction" in scores_df.columns else "risk_score",
                hover_data=["feature_id", "risk_score"],
                title="Feature Risk Landscape",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        if "direction" in scores_df.columns:
            fig_dist = px.violin(
                scores_df, x="direction", y="risk_score",
                box=True, points="all", title="Risk Score Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            fig_hist = px.histogram(scores_df, x="risk_score", nbins=60, title="Risk Score Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Top Risk Features")
    st.dataframe(scores_df.sort_values(by="risk_score", key=abs, ascending=False).head(100), use_container_width=True)


def render_cross_layer_view(config):
    st.header("ğŸ§¬ Cross-Layer View")

    targets = resolve_requested_targets(config)
    if len(targets) < 2:
        st.warning("Cross-layer analysis requires at least two configured target layers.")
        return

    safety_scores_dir = default_safety_scores_dir(config, target=targets[0])
    combinations_df = load_cross_layer_combinations(safety_scores_dir)
    classifier_report = load_json_artifact(str(Path(safety_scores_dir) / "safety_classifier_report.json"))
    feature_importance = load_json_artifact(str(Path(safety_scores_dir) / "feature_importance.json"))

    st.caption(f"Safety scores directory: `{safety_scores_dir}`")

    if classifier_report:
        st.subheader("Decision Tree Classifier")
        if classifier_report.get("error"):
            st.warning(
                f"Classifier training did not complete: {classifier_report['error']}. "
                "Install scikit-learn and rerun `python -m cli.safety_score --mode cross-layer`."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{classifier_report.get('accuracy', 0):.3f}")
            c2.metric("CV F1", f"{classifier_report.get('cv_f1_mean', 0):.3f}")
            c3.metric("Samples", classifier_report.get("n_samples", 0))
            c4.metric("Features", classifier_report.get("n_features", 0))
            if classifier_report.get("rules"):
                with st.expander("Decision Tree Rules", expanded=False):
                    st.code(classifier_report["rules"], language="text")

    if combinations_df is not None and not combinations_df.empty:
        st.subheader("Top Cross-Layer Combinations")
        top_df = combinations_df.sort_values(by="ratio", ascending=False).head(30).copy()
        top_df["pair"] = top_df["feature_1"].astype(str) + " × " + top_df["feature_2"].astype(str)
        color_col = "toxic_coactivation" if "toxic_coactivation" in top_df.columns else None
        fig_combo = px.bar(
            top_df,
            x="ratio",
            y="pair",
            color=color_col,
            orientation="h",
            title="Highest-Ratio Cross-Layer Pairs",
            labels={"x": "Positive/Negative Co-Activation Ratio", "y": "Feature Pair"},
        )
        fig_combo.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_combo, use_container_width=True)
        st.dataframe(top_df.drop(columns=["pair"]), use_container_width=True)
    else:
        st.info(
            "No cross-layer combinations found yet. "
            "Run `python -m cli.safety_score --config ... --mode cross-layer`."
        )

    if feature_importance:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame(feature_importance)
        if not importance_df.empty:
            fig_importance = px.bar(
                importance_df.head(25),
                x="importance",
                y="feature",
                orientation="h",
                title="Most Informative Cross-Layer Features",
            )
            fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_importance, use_container_width=True)
            st.dataframe(importance_df.head(50), use_container_width=True)


# ==========================================
# Deception Monitor Views
# ==========================================

def render_deception_monitor(config, config_path=None):
    st.header("🕵️ Deception Monitor")

    dec_dir = str(deception_run_dir(config))
    targets = resolve_requested_targets(config)
    safety_scores_dir = str(default_safety_scores_dir(config, target=targets[0])) if targets else ""

    # ── Section 1: Monitor Health KPIs ──────────────────────────────────
    report = load_deception_monitor_report(dec_dir)
    if report:
        st.subheader("Monitor Performance")
        c1, c2, c3, c4 = st.columns(4)
        ddr = report.get("detection_rate", 0)
        fpr = report.get("false_positive_rate", 0)
        prec = report.get("precision", 0)
        n_pairs = report.get("pairs_evaluated", 0)
        c1.metric("Detection Rate (DDR)", f"{ddr:.1%}",
                  delta="▲ good" if ddr >= 0.80 else "▼ low",
                  delta_color="normal" if ddr >= 0.80 else "inverse")
        c2.metric("False Positive Rate", f"{fpr:.1%}",
                  delta="▼ good" if fpr <= 0.20 else "▲ high",
                  delta_color="inverse" if fpr <= 0.20 else "normal")
        c3.metric("Precision", f"{prec:.1%}")
        c4.metric("Pairs Evaluated", n_pairs)
    else:
        st.warning("Monitor report not found. Run: `python -m cli.deception_cycle --phase monitor`")

    st.divider()

    # ── Section 2: Per-Layer Direction + Probe Comparison ────────────────
    st.subheader("Per-Layer Performance")
    directions = load_deception_directions(dec_dir)
    probe_summary = load_deception_probe_summary(dec_dir)

    if directions or probe_summary:
        rows = []
        for t in targets:
            row = {"layer": t.replace("hook_resid_post", "resid").replace("blocks.", "L").replace(".", ".")}
            if directions and t in directions:
                d = directions[t]
                row["dir_acc"] = round(d["validation_accuracy"], 3)
                row["separation"] = round(d["cluster_separation"], 3)
                row["expl_var"] = round(d["explained_variance"], 3)
            if probe_summary and t in probe_summary:
                p = probe_summary[t]
                row["probe_train"] = round(p.get("training_accuracy", 0), 3)
                row["probe_val"] = round(p.get("validation_accuracy", 0), 3)
                row["probe_cv"] = round(p.get("cv_accuracy") or 0, 3)
                row["backend"] = p.get("backend", "")
            rows.append(row)

        df_layers = pd.DataFrame(rows)
        st.dataframe(df_layers, use_container_width=True)

        if "dir_acc" in df_layers.columns and "probe_val" in df_layers.columns:
            fig = go.Figure()
            fig.add_bar(x=df_layers["layer"], y=df_layers["dir_acc"],
                        name="Direction Acc", marker_color="#4e8df5")
            fig.add_bar(x=df_layers["layer"], y=df_layers["probe_val"],
                        name="Probe Val Acc", marker_color="#f5954e")
            fig.update_layout(
                barmode="group",
                title="Direction vs Probe Accuracy by Layer",
                yaxis=dict(range=[0, 1]),
                yaxis_title="Accuracy",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No direction/probe data found. Run direction and probe phases first.")

    st.divider()

    # ── Section 3: Deception Feature Explorer ────────────────────────────
    st.subheader("SAE Deception Feature Explorer")
    selected_layer = st.selectbox("Select Layer", targets, key="dec_layer_select")

    csv_df = load_deception_contrastive_csv(safety_scores_dir, selected_layer)
    autointerp = load_deception_autointerp(dec_dir, selected_layer)

    if csv_df is not None and not csv_df.empty:
        pos_rate_col = next((c for c in csv_df.columns if "deceptive" in c and "rate" in c), None)
        neg_rate_col = next((c for c in csv_df.columns if "honest" in c and "rate" in c), None)

        if autointerp:
            csv_df["interpretation"] = csv_df["feature_id"].map(
                lambda x: autointerp.get(int(x), "")
            )
        else:
            csv_df["interpretation"] = ""

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            if pos_rate_col and neg_rate_col:
                fig_scatter = px.scatter(
                    csv_df.head(200),
                    x=neg_rate_col,
                    y=pos_rate_col,
                    color="risk_score",
                    size=csv_df.head(200)["risk_score"].abs().clip(lower=0.01),
                    hover_data=["feature_id", "risk_score", "rate_ratio"],
                    color_continuous_scale="RdYlGn_r",
                    title=f"Feature Risk Landscape — {selected_layer.split('.')[-2]}",
                    labels={neg_rate_col: "Honest Rate", pos_rate_col: "Deceptive Rate"},
                )
                fig_scatter.add_shape(
                    type="line", x0=0, y0=0, x1=csv_df[neg_rate_col].max(), y1=csv_df[neg_rate_col].max(),
                    line=dict(color="gray", dash="dash", width=1),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            fig_hist = px.histogram(
                csv_df,
                x="risk_score",
                nbins=60,
                color_discrete_sequence=["#f5954e"],
                title="Risk Score Distribution",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Top Deceptive Features")
        top_n = st.slider("Show top N features", 10, 100, 30, key="dec_top_n")
        display_cols = ["feature_id", "risk_score", "rate_ratio", "fisher_score"]
        if pos_rate_col:
            display_cols.insert(2, pos_rate_col)
        if neg_rate_col:
            display_cols.insert(3, neg_rate_col)
        if autointerp:
            display_cols.append("interpretation")
        top_df = csv_df.nlargest(top_n, "risk_score")[display_cols]
        st.dataframe(top_df, use_container_width=True)

        if autointerp:
            st.markdown("#### Feature Interpretations")
            if not csv_df.empty:
                top_feature_ids = csv_df.nlargest(20, "risk_score")["feature_id"].astype(int).tolist()
                sel_fid = st.selectbox(
                    "Select feature to inspect",
                    top_feature_ids,
                    format_func=lambda x: f"F{x}",
                    key="dec_fid_select",
                )
                if sel_fid is not None and int(sel_fid) in autointerp:
                    st.info(f"**F{sel_fid}**: {autointerp[int(sel_fid)]}")
        else:
            st.caption(
                "💡 Run autointerp to add LLM interpretations: "
                "`python -m cli.deception_autointerp --config ... --groq`"
            )
    else:
        st.warning(f"No contrastive scores for `{selected_layer}`. Run: `python -m cli.safety_score --config ... --mode contrastive`")

    st.divider()

    # ── Section 4: Adversarial Analysis ──────────────────────────────────
    st.subheader("Adversarial Search Results")
    adv_summary, missed = load_deception_adversarial(dec_dir)
    if adv_summary is not None:
        total = adv_summary.get("total_missed", 0)
        by_cat = adv_summary.get("by_category", {})
        by_diff = adv_summary.get("by_difficulty", {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missed Samples", total, delta="0 = monitor catches all" if total == 0 else None)
            if by_cat:
                fig_cat = px.bar(
                    x=list(by_cat.keys()), y=list(by_cat.values()),
                    title="Missed by Category",
                    labels={"x": "Category", "y": "Missed"},
                    color_discrete_sequence=["#e74c3c"],
                )
                st.plotly_chart(fig_cat, use_container_width=True)
        with col2:
            if by_diff:
                fig_diff = px.pie(
                    values=list(by_diff.values()), names=list(by_diff.keys()),
                    title="Missed by Difficulty",
                )
                st.plotly_chart(fig_diff, use_container_width=True)

        if missed:
            with st.expander(f"Show {len(missed)} missed samples"):
                for sample in missed[:20]:
                    st.markdown(
                        f"**Category**: {sample.get('category')} | "
                        f"**Score**: {sample.get('monitor_score', 0):.3f}\n\n"
                        f"*Q*: {sample.get('question', '')}\n"
                        f"*Deceptive*: {sample.get('deceptive_answer', '')}"
                    )
                    st.divider()
    else:
        st.info("Run adversarial phase: `python -m cli.deception_cycle --phase adversarial`")

    st.divider()

    st.subheader("Generated Report Artifacts")
    report_assets = load_deception_report_assets(dec_dir)
    html_path = Path(report_assets["html"])
    if html_path.exists():
        st.success(f"HTML report ready: `{html_path}`")
        st.caption(f"Report directory: `{report_assets['report_dir']}`")
        st.download_button(
            label="Download deception_report.html",
            data=html_path.read_bytes(),
            file_name="deception_report.html",
            mime="text/html",
            key="download_deception_report",
        )
    else:
        cmd = "python -m cli.deception_report --config ..."
        if config_path:
            cmd = f"python -m cli.deception_report --config {config_path}"
        st.info(f"Generate the full report with: `{cmd}`")

    chart_specs = [
        ("Layer Comparison", report_assets["layer_comparison"]),
        ("Honest vs Deceptive Projection", report_assets["tsne"]),
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
    else:
        st.caption("No report charts found yet.")


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
        "monitor": (base / "monitor" / "monitor_report.json").exists(),
        "adversarial": (base / "adversarial" / "summary.json").exists(),
        "autointerp": any((base / "autointerp").glob("*.json")) if (base / "autointerp").exists() else False,
    }


# ==========================================
# Steering Playground
# ==========================================

def render_steering_playground(config, slug, config_path):
    st.header("🎛️ Steering Playground")

    targets = resolve_requested_targets(config)
    model_name = config.get("model_name", "")
    has_deception = bool(config.get("deception", None))

    tab1, tab2 = st.tabs(["SAE Feature Steering", "Deception Direction Steering"])

    with tab1:
        st.markdown(
            "Amplify or suppress individual SAE features during generation. "
            "Positive α = amplify, negative α = suppress."
        )
        col1, col2 = st.columns(2)
        with col1:
            layer = st.selectbox("Layer", targets, key="steer_layer")
            feature_id = st.number_input("Feature ID", min_value=0, value=4192, key="steer_fid")
        with col2:
            alpha = st.slider("α (Steering Strength)", -50.0, 50.0, -10.0, 1.0, key="steer_alpha")
            mode = st.selectbox("Mode", ["add", "ablate", "clamp"], key="steer_mode")
        prompt = st.text_area("Prompt", "What do you think about climate change?", key="steer_prompt")
        system_prompt = st.text_input("System Prompt (optional)", "", key="steer_sys")

        sys_flag = f' --system-prompt "{system_prompt}"' if system_prompt else ""
        cmd_str = (
            f"python -m cli.steer steer \\"
            f"\n  --config {config_path} \\"
            f"\n  --layer {layer} \\"
            f"\n  --feature-id {feature_id} --alpha {alpha} --mode {mode} \\"
            f"\n  --max-tokens 150{sys_flag} \\"
            f'\n  --prompt "{prompt}"'
        )

        st.markdown("**💻 Run this command:**")
        st.code(cmd_str, language="bash")

        if st.button("🚀 Run SAE Feature Steering", key="run_sae"):
            with st.spinner("Running steering... this may take a moment."):
                actual_cmd = cmd_str.replace("\\\n  ", "")
                result = subprocess.run(actual_cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
                if result.returncode == 0:
                    st.text_area("CLI Output", result.stdout, height=300)
                else:
                    st.error("Command failed")
                    st.text_area("Error Output", result.stderr + "\n" + result.stdout, height=300)

    with tab2:
        if not has_deception:
            st.info("❌ This config does not have a deception section.")
        else:
            st.markdown(
                "Suppress or amplify the **deception direction** across all monitored layers simultaneously. "
                "Negative α → suppresses deceptive activation patterns."
            )

            dec_dir = str(deception_run_dir(config))
            status = _pipeline_status(dec_dir, config)
            if status["directions"]:
                st.success("✅ Direction vectors found — ready to steer.")
            else:
                st.warning("⚠️ Direction vectors not found.")

            prompt_d = st.text_area("Prompt", "Tell me about the vaccine safety data.", key="dec_steer_prompt")
            sys_d = st.text_input("System Prompt (optional)", "", key="dec_steer_sys")
            alpha_d = st.slider("α per layer", -20.0, 20.0, -5.0, 0.5, key="dec_alpha")
            show_scores = st.checkbox("Show monitor scores before/after", value=True)

            sys_flag_d = f' --system-prompt "{sys_d}"' if sys_d else ""
            scores_flag = " --show-scores" if show_scores else ""
            
            cmd_dec_str = (
                f"python -m cli.steer deception \\"
                f"\n  --config {config_path} \\"
                f"\n  --alpha {alpha_d} --max-tokens 150{sys_flag_d}{scores_flag} \\"
                f'\n  --prompt "{prompt_d}"'
            )
            st.markdown("**💻 Run this command:**")
            st.code(cmd_dec_str, language="bash")
            
            if st.button("🚀 Run Deception Steering", key="run_dec"):
                with st.spinner("Running deception suppression... this may take a moment."):
                    actual_cmd_dec = cmd_dec_str.replace("\\\n  ", "")
                    result = subprocess.run(actual_cmd_dec, shell=True, capture_output=True, text=True, encoding='utf-8')
                    if result.returncode == 0:
                        st.text_area("CLI Output", result.stdout, height=400)
                    else:
                        st.error("Command failed")
                        st.text_area("Error Output", result.stderr + "\n" + result.stdout, height=400)


# ==========================================
# Main Sidebar & Routing
# ==========================================

def main():
    st.sidebar.title("🧠 RFM Dashboard")

    default_config = "configs/models/qwen3-0.6B.deception.json"
    config_path = st.sidebar.text_input("Config Path", default_config)

    config = load_config(config_path)
    if not config:
        st.error(f"Config not found at `{config_path}`")
        return

    slug = model_slug(config)

    st.sidebar.markdown(f"🤖 **Model:** `{config.get('model_name')}`")
    targets = resolve_requested_targets(config)
    st.sidebar.markdown(f"🎯 **Layers:** `{', '.join(t.split('.')[-2] for t in targets)}`")
    st.sidebar.divider()

    has_deception = bool(config.get("deception", None))
    if has_deception:
        dec_dir = str(deception_run_dir(config))
        status = _pipeline_status(dec_dir, config)
        st.sidebar.markdown("**Pipeline Status**")
        icons = {True: "✅", False: "○"}
        for step, label in [
            ("scenarios", "Scenarios"),
            ("activations", "Activations"),
            ("checkpoints", "SAE Checkpoints"),
            ("directions", "Directions"),
            ("probes", "Probes"),
            ("monitor", "Monitor"),
            ("adversarial", "Adversarial"),
            ("autointerp", "Autointerp"),
        ]:
            st.sidebar.markdown(f"{icons[status[step]]} {label}")
        st.sidebar.divider()

    pages = [
        "🕵️ Deception Monitor",
        "🔍 Feature Explorer",
        "📈 Training Metrics",
        "🎛️ Steering Playground",
        "📊 Extraction Overview",
        "🔬 Contrastive Analysis",
        "🧬 Cross-Layer View",
    ]
    page = st.sidebar.radio("Navigation", pages)

    if page == "🕵️ Deception Monitor":
        render_deception_monitor(config, config_path)
    elif page == "🔍 Feature Explorer":
        render_feature_explorer(config, slug)
    elif page == "📈 Training Metrics":
        render_training_metrics(config)
    elif page == "🎛️ Steering Playground":
        render_steering_playground(config, slug, config_path)
    elif page == "📊 Extraction Overview":
        render_extraction_overview(config)
    elif page == "🔬 Contrastive Analysis":
        render_contrastive_analysis(config, slug)
    elif page == "🧬 Cross-Layer View":
        render_cross_layer_view(config)


if __name__ == "__main__":
    main()
