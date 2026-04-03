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
from rfm.deception.utils import deception_run_dir
from rfm.layout import (
    default_checkpoint_path,
    default_feature_mapping_dir,
    model_slug,
    resolve_activations_dir,
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
    path = Path(mapping_dir) / f"{slug}_feature_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_events(mapping_dir, slug):
    path = Path(mapping_dir) / f"{slug}_feature_events.csv"
    if path.exists():
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
def load_contrastive_scores(mapping_dir, slug):
    path = Path(mapping_dir) / f"{slug}_contrastive_scores.csv"
    if path.exists():
        return pd.read_csv(path)
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


# ==========================================
# Deception Monitor Views
# ==========================================

def render_deception_monitor(config):
    st.header("🕵️ Deception Monitor")

    dec_dir = str(deception_run_dir(config))
    safety_scores_dir = str(Path(dec_dir) / "contextual_activations" / "safety_scores")
    targets = resolve_requested_targets(config)

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
    mapping_dir = default_feature_mapping_dir(config)
    checkpoint_path = default_checkpoint_path(config)

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
        render_deception_monitor(config)
    elif page == "🔍 Feature Explorer":
        render_feature_explorer(config, slug, mapping_dir)
    elif page == "📈 Training Metrics":
        render_training_metrics(config, checkpoint_path)
    elif page == "🎛️ Steering Playground":
        render_steering_playground(config, slug, config_path)
    elif page == "📊 Extraction Overview":
        render_extraction_overview(config)
    elif page == "🔬 Contrastive Analysis":
        render_contrastive_analysis(config, slug, mapping_dir)
    elif page == "🧬 Cross-Layer View":
        render_cross_layer_view()


if __name__ == "__main__":
    main()
