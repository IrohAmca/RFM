import json
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from rfm.config import ConfigManager
from rfm.layout import model_slug, default_feature_mapping_dir, default_checkpoint_path, resolve_requested_targets, resolve_activations_dir

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
            # key strings to int
            return {int(self_k): v for self_k, v in json.load(f).items()}
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

import json
@st.cache_data
def load_extraction_metadata(_config):
    targets = resolve_requested_targets(_config)
    if not targets:
        return None
    
    # We only need to look at the first target's directory because sequence metadata is identical across layers
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
# Pages
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
        # Add search box
        search_kw = st.text_input("Search context/interpreation...", "")
        
        display_df = summary_df.copy()
        display_df["Interp"] = display_df["feature_id"].map(lambda x: autointerp_data.get(x, "No interpretation"))
        
        if search_kw:
             display_df = display_df[display_df["Interp"].str.contains(search_kw, case=False)]
             
        # Sort options
        sort_by = st.selectbox("Sort by", ["num_events", "mean_activation", "max_activation"], index=0)
        display_df = display_df.sort_values(by=sort_by, ascending=False)
        
        # Selection
        selected_feature = st.selectbox(
            "Select Feature:", 
            display_df["feature_id"].tolist(),
            format_func=lambda x: f"F{x} ({display_df[display_df['feature_id'] == x][sort_by].values[0]:.2f})"
        )
        
    with col2:
        if selected_feature is not None:
            st.subheader(f"Feature {selected_feature} Details")
            
            # Autointerp
            interp = autointerp_data.get(selected_feature)
            if interp:
                st.info(f"**Auto-Interpretation:** {interp}")
                
            # Quick Stats
            stats = summary_df[summary_df["feature_id"] == selected_feature].iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Events Found", stats["num_events"])
            m2.metric("Mean Act", f"{stats['mean_activation']:.3f}")
            m3.metric("Max Act", f"{stats['max_activation']:.3f}")
            
            # Top Contexts
            st.markdown("#### Top Activating Contexts")
            feature_events = events_df[events_df["feature_id"] == selected_feature].sort_values(by="activation", ascending=False).head(20)
            
            for _, row in feature_events.iterrows():
                ctx = row["context"]
                tok = row["token_str"]
                act = row["activation"]
                
                # Highlight the token in context
                # Safe HTML escaping
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
        
    # Extract epoch metrics vs dead feature analysis
    epoch_metrics = [h for h in history if "epoch" in h]
    dead_analysis = [h for h in history if "dead_feature_analysis" in h]
    
    if not epoch_metrics:
        st.info("No epoch metrics found in history.")
        return
        
    df = pd.DataFrame(epoch_metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loss Chart
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Total Loss"))
        if "val_loss" in df.columns:
             fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val Total Loss"))
        fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Reconstruction vs Sparsity
        fig_components = go.Figure()
        fig_components.add_trace(go.Scatter(x=df["epoch"], y=df["train_recon"], name="Recon Loss"))
        fig_components.add_trace(go.Scatter(x=df["epoch"], y=df["train_sparse"], name="Sparse Loss"))
        fig_components.update_layout(title="Loss Components", xaxis_title="Epoch")
        st.plotly_chart(fig_components, use_container_width=True)
        
    with col2:
         # Active Rate Chart
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


def render_steering_playground(config, slug):
    st.header("🎛️ Steering Playground")
    st.info("Interactive steering requires a live model instance. Enter a prompt and steer specific features.")
    
    with st.expander("How it works", expanded=True):
         st.markdown("""
         1. **Enter a prompt** that you want the model to complete.
         2. **Select a feature** you discovered in the Feature Explorer.
         3. **Set the Alpha (Multiplier)**. Positive values inject the trait, negative values suppress it.
         4. Click **Run Generation** to compare the clean output vs the steered output.
         """)
         
    prompt = st.text_area("Input Prompt", "The quick brown fox")
    
    col1, col2 = st.columns(2)
    with col1:
         feature_id = st.number_input("Feature ID", min_value=0, value=0)
    with col2:
         alpha = st.slider("Steering Alpha", min_value=-100.0, max_value=100.0, value=20.0, step=1.0)
         
    if st.button("Run Generation (Requires GPU & Environment Setup)", type="primary"):
         st.warning("Live generation within Streamlit is a placeholder in this version. Use `cli/steer.py` for actual generation.")
         # Implementation would involve loading model into memory here or via a persistent background process.
         

def render_extraction_overview(config):
    st.header("📊 Extraction Overview")
    
    meta_df = load_extraction_metadata(config)
    if meta_df is None or meta_df.empty:
        st.warning("No metadata files (`*.meta.json`) found in the activations directory. Run `cli.extract_generate` first to generate data.")
        return
        
    st.success(f"Successfully loaded {len(meta_df)} sequences across {meta_df['chunk_id'].nunique()} chunks.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Category balance
        fig_cat = px.histogram(
            meta_df, x="category", color="label", 
            barmode="group", title="Category Distribution by Toxicity"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col2:
        # Token length distribution
        fig_len = px.box(
            meta_df, x="label", y="token_length", color="label", 
            points="all", title="Response Token Lengths"
        )
        st.plotly_chart(fig_len, use_container_width=True)
        
    # Classifier confidence
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
        # Fisher vs Rate Ratio Scatter
        fig_scatter = px.scatter(
            scores_df, x="fisher_score", y="rate_ratio",
            color="direction", hover_data=["feature_id", "risk_score"],
            log_x=True, log_y=True, title="Feature Risk Landscape"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        # Risk Score Distribution
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
# Main App Structure
# ==========================================
def main():
    st.sidebar.title("🧠 RFM Dashboard")
    
    config_path = st.sidebar.text_input("Config Path", "configs/models/gpt2-small.emotion.json")
    
    config = load_config(config_path)
    if not config:
        st.error(f"Config not found at {config_path}")
        return
        
    slug = model_slug(config)
    mapping_dir = default_feature_mapping_dir(config)
    checkpoint_path = default_checkpoint_path(config)
    
    st.sidebar.markdown(f"**Model:** `{config.get('model_name')}`")
    targets = resolve_requested_targets(config)
    st.sidebar.markdown(f"**Targets:** `{', '.join(targets)}`")
    
    st.sidebar.divider()
    
    pages = [
        "Feature Explorer", "Training Metrics", "Steering Playground", 
        "Extraction Overview", "Contrastive Analysis", "Cross-Layer View"
    ]
    page = st.sidebar.radio("Navigation", pages)
    
    if page == "Feature Explorer":
        render_feature_explorer(config, slug, mapping_dir)
    elif page == "Training Metrics":
        render_training_metrics(config, checkpoint_path)
    elif page == "Steering Playground":
        render_steering_playground(config, slug)
    elif page == "Extraction Overview":
        render_extraction_overview(config)
    elif page == "Contrastive Analysis":
        render_contrastive_analysis(config, slug, mapping_dir)
    elif page == "Cross-Layer View":
        render_cross_layer_view()

if __name__ == "__main__":
    main()
