# -*- coding: utf-8 -*-
"""Critical analysis: Direction-SAE projection, contamination test, and feature graph.

Three independent analyses:
1. Direction → SAE Projection: Find features truly aligned with the deception direction
2. Contamination Test: Check if surface signals dominate
3. Feature Co-activation Graph: Discover feature communities as "thought patterns"

All run on CPU to respect 4GB VRAM constraint.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

# Force UTF-8 stdout for Turkish Windows (cp1254)
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

RUNS_DIR = Path("runs/Qwen_Qwen3-0.6B/deception")
DIRECTIONS_PATH = RUNS_DIR / "directions" / "directions.pt"
ACTIVATIONS_DIR = RUNS_DIR / "contextual_activations"
CHECKPOINTS_DIR = RUNS_DIR / "checkpoints"
OUTPUT_DIR = RUNS_DIR / "critical_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [
    "blocks.6.hook_resid_post",
    "blocks.13.hook_resid_post",
    "blocks.20.hook_resid_post",
    "blocks.27.hook_resid_post",
]


def sanitize(name: str) -> str:
    return name.replace(".", "_")


def load_sae(layer: str):
    path = CHECKPOINTS_DIR / sanitize(layer) / "sae.pt"
    if not path.exists():
        print(f"  [SKIP] No checkpoint for {layer}")
        return None
    from rfm.sae.model import load_sae_checkpoint
    model, ckpt = load_sae_checkpoint(path, device="cpu")
    model.eval()
    return model


def load_chunk(layer: str):
    layer_dir = ACTIVATIONS_DIR / sanitize(layer)
    files = sorted(layer_dir.glob("*.pt"))
    if not files:
        print(f"  [SKIP] No activation chunks for {layer}")
        return None, None
    activation_chunks = []
    merged_metadata = defaultdict(list)
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        activation_chunks.append(data["activations"])
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            if isinstance(value, list):
                merged_metadata[key].extend(value)
            elif isinstance(value, tuple):
                merged_metadata[key].extend(list(value))
            elif torch.is_tensor(value):
                merged_metadata[key].extend(value.detach().cpu().tolist())
            else:
                merged_metadata[key].append(value)
    return torch.cat(activation_chunks, dim=0), dict(merged_metadata)


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: Direction → SAE Projection
# ═══════════════════════════════════════════════════════════════

def analysis_1_direction_projection():
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Direction → SAE Decoder Projection")
    print("=" * 70)
    print("Goal: Find SAE features truly ALIGNED with the deception direction,")
    print("      rather than features that just separate classes trivially.\n")

    if not DIRECTIONS_PATH.exists():
        print("[ERROR] directions.pt not found")
        return {}

    directions = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    results = {}

    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")

        dir_data = directions.get(layer)
        if dir_data is None:
            print(f"  [SKIP] No direction for {layer}")
            continue

        direction = dir_data["direction"].detach().cpu().float()
        direction = direction / direction.norm().clamp(min=1e-12)

        sae = load_sae(layer)
        if sae is None:
            continue

        # W_dec shape: [hidden_dim, input_dim] — each row is a decoder direction
        W_dec = sae.W_dec.detach().cpu().float()
        decoder_norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-12)
        W_dec_normed = W_dec / decoder_norms

        # Cosine similarity between each SAE feature direction and the deception direction
        alignment = W_dec_normed @ direction  # [hidden_dim]

        # Top aligned (deceptive direction)
        top_positive = torch.topk(alignment, k=20)
        # Top opposed (honest direction)
        top_negative = torch.topk(-alignment, k=20)

        print(f"  Direction explained_variance: {dir_data.get('explained_variance', 'N/A')}")
        print(f"  Direction validation_accuracy: {dir_data.get('validation_accuracy', 'N/A')}")
        print(f"  Direction cluster_separation: {dir_data.get('cluster_separation', 'N/A')}")
        print(f"\n  Top 20 features ALIGNED with deception direction:")
        deceptive_features = []
        for rank, (val, idx) in enumerate(zip(top_positive.values.tolist(), top_positive.indices.tolist())):
            print(f"    #{rank+1:2d} Feature {idx:5d}  cosine={val:+.4f}")
            deceptive_features.append({"feature_id": idx, "cosine": round(val, 6), "alignment": "deceptive"})

        print(f"\n  Top 20 features OPPOSED (aligned with honesty):")
        honest_features = []
        for rank, (val, idx) in enumerate(zip(top_negative.values.tolist(), top_negative.indices.tolist())):
            cos = -val  # negate back
            print(f"    #{rank+1:2d} Feature {idx:5d}  cosine={cos:+.4f}")
            honest_features.append({"feature_id": idx, "cosine": round(cos, 6), "alignment": "honest"})

        # Statistics
        abs_alignment = alignment.abs()
        print(f"\n  Alignment statistics:")
        print(f"    Mean |cosine|:  {abs_alignment.mean().item():.6f}")
        print(f"    Max  |cosine|:  {abs_alignment.max().item():.6f}")
        print(f"    Std  |cosine|:  {abs_alignment.std().item():.6f}")
        print(f"    Features with |cosine| > 0.1:  {(abs_alignment > 0.1).sum().item()}")
        print(f"    Features with |cosine| > 0.05: {(abs_alignment > 0.05).sum().item()}")

        results[layer] = {
            "direction_stats": {
                "explained_variance": float(dir_data.get("explained_variance", 0)),
                "validation_accuracy": float(dir_data.get("validation_accuracy", 0)),
                "cluster_separation": float(dir_data.get("cluster_separation", 0)),
            },
            "alignment_stats": {
                "mean_abs": round(abs_alignment.mean().item(), 6),
                "max_abs": round(abs_alignment.max().item(), 6),
                "std": round(abs_alignment.std().item(), 6),
                "count_above_0.1": int((abs_alignment > 0.1).sum().item()),
                "count_above_0.05": int((abs_alignment > 0.05).sum().item()),
            },
            "top_deceptive_features": deceptive_features,
            "top_honest_features": honest_features,
        }

    # Save
    out_path = OUTPUT_DIR / "direction_sae_projection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: Contamination Test
# ═══════════════════════════════════════════════════════════════

def analysis_2_contamination_test():
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Contamination / Trivial Signal Test")
    print("=" * 70)
    print("Goal: Check if non-semantic features (length, position) can classify.\n")

    results = {}

    for layer in LAYERS[:1]:  # Only need one layer for this test
        print(f"\n--- Layer: {layer} ---")
        activations, metadata = load_chunk(layer)
        if activations is None:
            continue

        labels = list(metadata.get("labels", []))
        token_lengths = [int(x) for x in metadata.get("token_lengths", [])]
        responses = list(metadata.get("responses", []))
        categories = list(metadata.get("categories", []))

        if not labels or not token_lengths:
            print("  [SKIP] No labels or token_lengths in metadata")
            continue

        # --- Test A: Response length separation ---
        honest_lens = []
        deceptive_lens = []
        for label, tlen in zip(labels, token_lengths):
            if label == "honest":
                honest_lens.append(tlen)
            elif label == "deceptive":
                deceptive_lens.append(tlen)

        honest_arr = np.array(honest_lens, dtype=np.float64)
        deceptive_arr = np.array(deceptive_lens, dtype=np.float64)

        print(f"  Sample counts: honest={len(honest_lens)}, deceptive={len(deceptive_lens)}")
        print(f"\n  Test A: Token length distribution")
        print(f"    Honest   — mean={honest_arr.mean():.1f}, std={honest_arr.std():.1f}, "
              f"min={honest_arr.min():.0f}, max={honest_arr.max():.0f}")
        print(f"    Deceptive — mean={deceptive_arr.mean():.1f}, std={deceptive_arr.std():.1f}, "
              f"min={deceptive_arr.min():.0f}, max={deceptive_arr.max():.0f}")

        # Cohen's d for length
        pooled_std = np.sqrt((honest_arr.var() + deceptive_arr.var()) / 2)
        cohens_d_length = (deceptive_arr.mean() - honest_arr.mean()) / max(pooled_std, 1e-8)
        print(f"    Cohen's d (length): {cohens_d_length:.4f}")

        # Try to classify using length alone
        threshold = (honest_arr.mean() + deceptive_arr.mean()) / 2
        honest_correct = (honest_arr < threshold).sum()
        deceptive_correct = (deceptive_arr >= threshold).sum()
        length_accuracy = (honest_correct + deceptive_correct) / (len(honest_lens) + len(deceptive_lens))
        print(f"    Length-only accuracy: {length_accuracy:.4f}")

        # --- Test B: Response text length (characters) ---
        honest_char_lens = []
        deceptive_char_lens = []
        for label, resp in zip(labels, responses):
            if label == "honest":
                honest_char_lens.append(len(str(resp)))
            elif label == "deceptive":
                deceptive_char_lens.append(len(str(resp)))

        if honest_char_lens and deceptive_char_lens:
            honest_chars = np.array(honest_char_lens, dtype=np.float64)
            deceptive_chars = np.array(deceptive_char_lens, dtype=np.float64)
            print(f"\n  Test B: Response character length")
            print(f"    Honest   — mean={honest_chars.mean():.1f}")
            print(f"    Deceptive — mean={deceptive_chars.mean():.1f}")
            char_diff = abs(deceptive_chars.mean() - honest_chars.mean())
            print(f"    Mean difference: {char_diff:.1f} chars")

        # --- Test C: Category distribution ---
        honest_cats = Counter()
        deceptive_cats = Counter()
        for label, cat in zip(labels, categories):
            if label == "honest":
                honest_cats[cat] += 1
            elif label == "deceptive":
                deceptive_cats[cat] += 1

        print(f"\n  Test C: Category balance")
        all_cats = sorted(set(honest_cats.keys()) | set(deceptive_cats.keys()))
        for cat in all_cats:
            print(f"    {cat:30s}  honest={honest_cats.get(cat, 0):4d}  deceptive={deceptive_cats.get(cat, 0):4d}")

        # --- Test D: Activation magnitude difference ---
        print(f"\n  Test D: Raw activation magnitude per class")
        honest_acts_list = []
        deceptive_acts_list = []
        offset = 0
        for label, tlen in zip(labels, token_lengths):
            segment = activations[offset:offset+tlen]
            mean_act = segment.mean().item()
            if label == "honest":
                honest_acts_list.append(mean_act)
            elif label == "deceptive":
                deceptive_acts_list.append(mean_act)
            offset += tlen

        h_act = np.array(honest_acts_list)
        d_act = np.array(deceptive_acts_list)
        print(f"    Honest activation mean:    {h_act.mean():.6f} ± {h_act.std():.6f}")
        print(f"    Deceptive activation mean: {d_act.mean():.6f} ± {d_act.std():.6f}")
        act_cohens_d = (d_act.mean() - h_act.mean()) / max(np.sqrt((h_act.var() + d_act.var()) / 2), 1e-8)
        print(f"    Cohen's d (activation): {act_cohens_d:.4f}")

        # --- Test E: Norm of activation vectors ---
        print(f"\n  Test E: Activation vector norm per class")
        honest_norms = []
        deceptive_norms = []
        offset = 0
        for label, tlen in zip(labels, token_lengths):
            segment = activations[offset:offset+tlen].float()
            mean_norm = segment.norm(dim=1).mean().item()
            if label == "honest":
                honest_norms.append(mean_norm)
            elif label == "deceptive":
                deceptive_norms.append(mean_norm)
            offset += tlen

        h_norms = np.array(honest_norms)
        d_norms = np.array(deceptive_norms)
        print(f"    Honest norm mean:    {h_norms.mean():.4f} ± {h_norms.std():.4f}")
        print(f"    Deceptive norm mean: {d_norms.mean():.4f} ± {d_norms.std():.4f}")
        norm_d = (d_norms.mean() - h_norms.mean()) / max(np.sqrt((h_norms.var() + d_norms.var()) / 2), 1e-8)
        print(f"    Cohen's d (norm): {norm_d:.4f}")

        results = {
            "sample_counts": {"honest": len(honest_lens), "deceptive": len(deceptive_lens)},
            "token_length": {
                "honest_mean": round(float(honest_arr.mean()), 2),
                "deceptive_mean": round(float(deceptive_arr.mean()), 2),
                "cohens_d": round(cohens_d_length, 4),
                "length_only_accuracy": round(length_accuracy, 4),
            },
            "activation_magnitude": {
                "honest_mean": round(float(h_act.mean()), 6),
                "deceptive_mean": round(float(d_act.mean()), 6),
                "cohens_d": round(act_cohens_d, 4),
            },
            "activation_norm": {
                "honest_mean": round(float(h_norms.mean()), 4),
                "deceptive_mean": round(float(d_norms.mean()), 4),
                "cohens_d": round(norm_d, 4),
            },
            "categories": {cat: {"honest": honest_cats.get(cat, 0), "deceptive": deceptive_cats.get(cat, 0)} for cat in all_cats},
        }

    out_path = OUTPUT_DIR / "contamination_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: Direction-Projected Feature Scoring
# (Replace trivial contrastive with direction-aligned scoring)
# ═══════════════════════════════════════════════════════════════

def analysis_3_direction_aware_scoring():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Direction-Aware Feature Scoring")
    print("=" * 70)
    print("Goal: Score SAE features by their actual contribution to the")
    print("      deception direction, not by trivial class separation.\n")

    if not DIRECTIONS_PATH.exists():
        print("[ERROR] directions.pt not found")
        return {}

    directions = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    all_results = {}

    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")

        dir_data = directions.get(layer)
        if dir_data is None:
            continue

        direction = dir_data["direction"].detach().cpu().float()
        direction = direction / direction.norm().clamp(min=1e-12)

        sae = load_sae(layer)
        if sae is None:
            continue

        activations, metadata = load_chunk(layer)
        if activations is None:
            continue

        labels = list(metadata.get("labels", []))
        token_lengths = [int(x) for x in metadata.get("token_lengths", [])]
        pair_ids = [int(x) for x in metadata.get("pair_ids", [])]

        # Encode through SAE
        print(f"  Encoding {activations.shape[0]} tokens through SAE...")
        all_features = []
        sae.eval()
        with torch.no_grad():
            for start in range(0, activations.shape[0], 2048):
                batch = activations[start:start+2048].float()
                _, feats = sae(batch)
                all_features.append(feats.cpu())
        all_features = torch.cat(all_features, dim=0)

        # Aggregate per sequence (mean)
        seq_features = []
        offset = 0
        for tlen in token_lengths:
            seg = all_features[offset:offset+tlen]
            seq_features.append(seg.mean(dim=0))
            offset += tlen
        seq_features = torch.stack(seq_features, dim=0)  # [N_seq, hidden_dim]

        # Split by label
        y = np.array([1 if l == "deceptive" else 0 for l in labels])

        # For each feature: compute its contribution via direction projection
        W_dec = sae.W_dec.detach().cpu().float()
        decoder_norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-12)
        W_dec_normed = W_dec / decoder_norms
        alignment = (W_dec_normed @ direction).numpy()  # [hidden_dim]

        # Feature activation difference
        deceptive_mean = seq_features[y == 1].mean(dim=0).numpy()
        honest_mean = seq_features[y == 0].mean(dim=0).numpy()
        delta = deceptive_mean - honest_mean

        # Direction-weighted score: alignment × delta
        # This captures features that are BOTH aligned with the direction AND differentially active
        direction_score = alignment * delta

        # Also compute vanilla effect size for comparison
        deceptive_std = seq_features[y == 1].std(dim=0).numpy()
        honest_std = seq_features[y == 0].std(dim=0).numpy()
        pooled_std = np.sqrt((deceptive_std ** 2 + honest_std ** 2) / 2 + 1e-12)
        vanilla_effect = delta / pooled_std

        # Rank features by direction_score
        dir_ranking = np.argsort(-np.abs(direction_score))
        vanilla_ranking = np.argsort(-np.abs(vanilla_effect))

        print(f"\n  Top 15 by DIRECTION-WEIGHTED score (alignment × delta):")
        dir_top = []
        for rank, idx in enumerate(dir_ranking[:15]):
            idx = int(idx)
            print(f"    #{rank+1:2d} F{idx:5d}  dir_score={direction_score[idx]:+.6f}  "
                  f"alignment={alignment[idx]:+.4f}  delta={delta[idx]:+.6f}  "
                  f"effect_size={vanilla_effect[idx]:+.4f}")
            dir_top.append({
                "feature_id": idx,
                "direction_score": round(float(direction_score[idx]), 6),
                "alignment": round(float(alignment[idx]), 6),
                "delta": round(float(delta[idx]), 6),
                "vanilla_effect_size": round(float(vanilla_effect[idx]), 6),
            })

        print(f"\n  Top 15 by VANILLA effect size (for comparison):")
        van_top = []
        for rank, idx in enumerate(vanilla_ranking[:15]):
            idx = int(idx)
            print(f"    #{rank+1:2d} F{idx:5d}  effect_size={vanilla_effect[idx]:+.4f}  "
                  f"alignment={alignment[idx]:+.4f}  dir_score={direction_score[idx]:+.6f}")
            van_top.append({
                "feature_id": idx,
                "vanilla_effect_size": round(float(vanilla_effect[idx]), 6),
                "alignment": round(float(alignment[idx]), 6),
                "direction_score": round(float(direction_score[idx]), 6),
            })

        # Overlap analysis
        dir_set = set(dir_ranking[:50])
        van_set = set(vanilla_ranking[:50])
        overlap = dir_set & van_set
        only_direction = dir_set - van_set
        only_vanilla = van_set - dir_set

        print(f"\n  === Top-50 Overlap Analysis ===")
        print(f"    Both methods:       {len(overlap)}")
        print(f"    Direction-only:     {len(only_direction)}")
        print(f"    Vanilla-only:       {len(only_vanilla)}")
        print(f"    Overlap ratio:      {len(overlap)/50:.1%}")

        # Dead feature check
        active_mask = (seq_features > 0).any(dim=0)
        alive = int(active_mask.sum().item())
        dead = int(seq_features.shape[1]) - alive
        print(f"\n  === Dead Feature Analysis ===")
        print(f"    Total features: {seq_features.shape[1]}")
        print(f"    Alive: {alive}  Dead: {dead}  ({dead/seq_features.shape[1]:.1%} dead)")

        # Cross-validate direction-projected classifier
        print(f"\n  === Direction-Projected Classifier (5-fold CV) ===")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, roc_auc_score

        # Method A: Vanilla top-50 features
        van_idx = vanilla_ranking[:50]
        X_van = seq_features[:, van_idx].numpy()

        # Method B: Direction-weighted top-50 features
        dir_idx = dir_ranking[:50]
        X_dir = seq_features[:, dir_idx].numpy()

        # Method C: Direction-ONLY (no SAE) — just project onto direction
        seq_raw_list = []
        offset = 0
        for tlen in token_lengths:
            seg = activations[offset:offset+tlen].float()
            seq_raw_list.append(seg.mean(dim=0))
            offset += tlen
        seq_raw = torch.stack(seq_raw_list, dim=0)
        X_proj = (seq_raw @ direction).numpy().reshape(-1, 1)

        # Method D: Direction-only features  > 0.05 alignment
        aligned_mask = np.abs(alignment) > 0.05
        aligned_idx = np.where(aligned_mask)[0]
        X_aligned = seq_features[:, aligned_idx].numpy() if len(aligned_idx) > 0 else X_van

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        methods = {
            "vanilla_top50": X_van,
            "direction_top50": X_dir,
            "raw_direction_1d": X_proj,
            f"aligned_features_{len(aligned_idx)}": X_aligned,
        }

        cv_results = {}
        for name, X in methods.items():
            f1s, aucs = [], []
            for train_idx, test_idx in skf.split(X, y):
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
                clf.fit(X[train_idx], y[train_idx])
                y_pred = clf.predict(X[test_idx])
                y_score = clf.predict_proba(X[test_idx])[:, 1]
                f1s.append(f1_score(y[test_idx], y_pred))
                aucs.append(roc_auc_score(y[test_idx], y_score))

            mean_f1 = np.mean(f1s)
            mean_auc = np.mean(aucs)
            print(f"    {name:35s}  F1={mean_f1:.4f} ± {np.std(f1s):.4f}  "
                  f"AUC={mean_auc:.4f} ± {np.std(aucs):.4f}")
            cv_results[name] = {
                "f1_mean": round(mean_f1, 6),
                "f1_std": round(float(np.std(f1s)), 6),
                "auc_mean": round(mean_auc, 6),
                "auc_std": round(float(np.std(aucs)), 6),
            }

        all_results[layer] = {
            "direction_top15": dir_top,
            "vanilla_top15": van_top,
            "overlap_analysis": {
                "top50_both": len(overlap),
                "top50_direction_only": len(only_direction),
                "top50_vanilla_only": len(only_vanilla),
                "overlap_ratio": round(len(overlap) / 50, 4),
            },
            "dead_features": {
                "total": int(seq_features.shape[1]),
                "alive": alive,
                "dead": dead,
                "dead_ratio": round(dead / seq_features.shape[1], 4),
            },
            "cv_comparison": cv_results,
        }

    out_path = OUTPUT_DIR / "direction_aware_scoring.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")
    return all_results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: Feature Co-activation Graph + Community Detection
# ═══════════════════════════════════════════════════════════════

def analysis_4_feature_communities():
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Feature Co-activation Graph & Community Detection")
    print("=" * 70)
    print("Goal: Discover 'thought modules' — groups of features that")
    print("      co-activate as a coherent pattern.\n")

    if not DIRECTIONS_PATH.exists():
        print("[ERROR] directions.pt not found")
        return {}

    directions = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    all_results = {}

    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")

        sae = load_sae(layer)
        if sae is None:
            continue

        activations, metadata = load_chunk(layer)
        if activations is None:
            continue

        labels = list(metadata.get("labels", []))
        token_lengths = [int(x) for x in metadata.get("token_lengths", [])]

        # Encode
        print(f"  Encoding activations...")
        all_features = []
        with torch.no_grad():
            for start in range(0, activations.shape[0], 2048):
                batch = activations[start:start + 2048].float()
                _, feats = sae(batch)
                all_features.append(feats.cpu())
        all_features = torch.cat(all_features, dim=0)

        # Aggregate per sequence
        seq_features = []
        offset = 0
        for tlen in token_lengths:
            seg = all_features[offset:offset + tlen]
            seq_features.append(seg.mean(dim=0))
            offset += tlen
        seq_features = torch.stack(seq_features, dim=0)  # [N_seq, hidden_dim]

        # Pre-filter to active features only (activation rate > 1%)
        activation_rate = (seq_features > 0).float().mean(dim=0)
        active_idx = torch.where(activation_rate > 0.01)[0].tolist()
        print(f"  Active features (>1% activation rate): {len(active_idx)}/{seq_features.shape[1]}")

        if len(active_idx) < 10:
            print("  [SKIP] Too few active features for community detection")
            continue

        # Limit to top-200 most active to keep graph manageable
        top_active = sorted(active_idx, key=lambda i: activation_rate[i].item(), reverse=True)[:200]
        sub_features = seq_features[:, top_active]  # [N_seq, 200]

        # Build co-activation matrix (correlation of activation patterns)
        # Binarize: feature active or not
        binary = (sub_features > 0).float().numpy()  # [N_seq, 200]
        n_features = binary.shape[1]

        # Compute pairwise Jaccard similarity
        print(f"  Computing co-activation matrix ({n_features}x{n_features})...")
        coact_matrix = np.zeros((n_features, n_features), dtype=np.float32)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                both = (binary[:, i] * binary[:, j]).sum()
                either = np.maximum(binary[:, i], binary[:, j]).sum()
                jaccard = both / max(either, 1)
                coact_matrix[i, j] = jaccard
                coact_matrix[j, i] = jaccard

        # Community detection using spectral clustering
        try:
            from sklearn.cluster import SpectralClustering

            n_clusters = min(15, n_features // 5)
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
                n_init=10,
            )
            cluster_labels = clustering.fit_predict(coact_matrix + np.eye(n_features) * 0.01)

            communities = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_labels):
                communities[int(cluster_id)].append(top_active[idx])

            print(f"\n  Found {len(communities)} feature communities:")

            # Analyze each community
            dir_data = directions.get(layer)
            direction = None
            alignment = None
            if dir_data is not None:
                direction = dir_data["direction"].detach().cpu().float()
                direction = direction / direction.norm().clamp(min=1e-12)
                W_dec = sae.W_dec.detach().cpu().float()
                W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-12)
                alignment = (W_dec_normed @ direction).numpy()

            y = np.array([1 if l == "deceptive" else 0 for l in labels])
            community_report = []
            for cluster_id in sorted(communities.keys()):
                members = communities[cluster_id]
                member_features = seq_features[:, members]
                member_active = (member_features > 0).float()

                # Class-specific activation
                honest_rate = member_active[y == 0].mean().item()
                deceptive_rate = member_active[y == 1].mean().item()
                rate_diff = deceptive_rate - honest_rate

                # Direction alignment of community
                if alignment is not None:
                    community_alignment = np.mean([alignment[m] for m in members])
                else:
                    community_alignment = 0.0

                label = "-> DECEPTIVE" if rate_diff > 0.02 else ("-> HONEST" if rate_diff < -0.02 else "-> NEUTRAL")
                print(f"    Community {cluster_id:2d}: {len(members):3d} features  "
                      f"h_rate={honest_rate:.3f} d_rate={deceptive_rate:.3f} "
                      f"diff={rate_diff:+.3f} align={community_alignment:+.4f} {label}")

                community_report.append({
                    "community_id": int(cluster_id),
                    "size": len(members),
                    "feature_ids": members[:20],  # Save top 20 members
                    "honest_activation_rate": round(honest_rate, 4),
                    "deceptive_activation_rate": round(deceptive_rate, 4),
                    "rate_difference": round(rate_diff, 4),
                    "direction_alignment": round(float(community_alignment), 4),
                    "label": label.replace("-> ", "").strip(),
                })

            all_results[layer] = {
                "active_features": len(active_idx),
                "analyzed_features": n_features,
                "n_communities": len(communities),
                "communities": community_report,
            }

        except ImportError:
            print("  [SKIP] scikit-learn required for clustering")

    out_path = OUTPUT_DIR / "feature_communities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")
    return all_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  RFM Critical Analysis -- Direction Projection + Diagnostics  ")
    print("=" * 64)

    r1 = analysis_1_direction_projection()
    r2 = analysis_2_contamination_test()
    r3 = analysis_3_direction_aware_scoring()
    r4 = analysis_4_feature_communities()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if r2:
        length_acc = r2.get("token_length", {}).get("length_only_accuracy", "N/A")
        print(f"\n  Contamination: Length-only accuracy = {length_acc}")
        if isinstance(length_acc, float) and length_acc > 0.6:
            print("  [!] WARNING: Token length alone can partially classify -- surface signal exists!")
        elif isinstance(length_acc, float):
            print("  [OK] Length alone cannot classify well -- good signal quality")

    if r3:
        for layer, data in r3.items():
            cv = data.get("cv_comparison", {})
            print(f"\n  Layer {layer}:")
            for method, scores in cv.items():
                print(f"    {method:35s}  F1={scores['f1_mean']:.4f} AUC={scores['auc_mean']:.4f}")

            overlap = data.get("overlap_analysis", {})
            print(f"    Direction vs Vanilla overlap (top-50): {overlap.get('overlap_ratio', 'N/A')}")

    if r4:
        for layer, data in r4.items():
            communities = data.get("communities", [])
            deceptive_communities = [c for c in communities if c["label"] == "DECEPTIVE"]
            honest_communities = [c for c in communities if c["label"] == "HONEST"]
            print(f"\n  Layer {layer}: {len(communities)} communities")
            print(f"    Deceptive-leaning: {len(deceptive_communities)}")
            print(f"    Honest-leaning:    {len(honest_communities)}")

    print(f"\n  All results saved to: {OUTPUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
