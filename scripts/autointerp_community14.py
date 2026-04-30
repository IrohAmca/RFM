# -*- coding: utf-8 -*-
"""Community 14 (Layer 20) & top direction-aligned features -- autointerp via Groq.

Loads the 9 features from Layer-20 Community-14 (the strongest deception
thought-module candidate), runs a Groq LLM over sample activating token
contexts, and produces an interpretability report.
"""

import io
import json
import os
import sys
import time
from pathlib import Path
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


RUNS_DIR = Path("runs/Qwen_Qwen3-0.6B/deception")
COMMUNITIES_PATH = RUNS_DIR / "critical_analysis" / "feature_communities.json"
DIRECTION_SCORING_PATH = RUNS_DIR / "critical_analysis" / "direction_aware_scoring.json"
ACTIVATIONS_DIR = RUNS_DIR / "contextual_activations"
CHECKPOINTS_DIR = RUNS_DIR / "checkpoints"
OUTPUT_DIR = RUNS_DIR / "critical_analysis" / "autointerp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOCUS_LAYER = "blocks.20.hook_resid_post"
FOCUS_COMMUNITY_ID = 14   # strongest deceptive community
TOP_DIRECTION_FEATURES = 10  # top direction-aligned features
MAX_CONTEXTS_PER_FEATURE = 12
MAX_TOKENS_CONTEXT = 60   # chars shown per context
GROQ_MODEL = "llama3-70b-8192"


def sanitize(name: str) -> str:
    return name.replace(".", "_")


def load_sae(layer: str):
    path = CHECKPOINTS_DIR / sanitize(layer) / "sae.pt"
    from rfm.sae.model import load_sae_checkpoint
    model, _ = load_sae_checkpoint(path, device="cpu")
    model.eval()
    return model


def load_chunk(layer: str):
    layer_dir = ACTIVATIONS_DIR / sanitize(layer)
    files = sorted(layer_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No activation chunks found in {layer_dir}")
    activation_chunks = []
    merged_metadata: dict[str, list] = {}
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        activation_chunks.append(data["activations"])
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            target = merged_metadata.setdefault(key, [])
            if isinstance(value, list):
                target.extend(value)
            elif isinstance(value, tuple):
                target.extend(list(value))
            elif torch.is_tensor(value):
                target.extend(value.detach().cpu().tolist())
            else:
                target.append(value)
    return torch.cat(activation_chunks, dim=0), merged_metadata


def encode_sae(sae, activations: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    out = []
    sae.eval()
    with torch.no_grad():
        for start in range(0, activations.shape[0], batch_size):
            _, feats = sae(activations[start:start + batch_size].float())
            out.append(feats.cpu())
    return torch.cat(out, dim=0)


def find_top_activating_tokens(
    feature_id: int,
    all_features: torch.Tensor,
    responses: list[str],
    token_lengths: list[int],
    labels: list[str],
    top_k: int = MAX_CONTEXTS_PER_FEATURE,
    target_label: str | None = None,
) -> list[dict]:
    """Find sequences where the feature activates most strongly."""
    seq_maxvals = []
    offset = 0
    for seq_idx, (tlen, label, resp) in enumerate(zip(token_lengths, labels, responses)):
        seg = all_features[offset:offset + tlen, feature_id]
        if target_label and label != target_label:
            offset += tlen
            continue
        maxval = float(seg.max().item())
        seq_maxvals.append((maxval, seq_idx, label, resp))
        offset += tlen

    seq_maxvals.sort(reverse=True)
    results = []
    for maxval, seq_idx, label, resp in seq_maxvals[:top_k]:
        if maxval <= 0:
            break
        preview = str(resp)[:MAX_TOKENS_CONTEXT].replace("\n", " ")
        results.append({
            "max_activation": round(maxval, 4),
            "label": label,
            "response_preview": preview,
        })
    return results


def build_feature_description_prompt(
    feature_id: int,
    layer: str,
    contexts: list[dict],
    direction_alignment: float,
    effect_size: float,
    community_id: int | None = None,
) -> str:
    ctx_lines = []
    for i, ctx in enumerate(contexts, 1):
        ctx_lines.append(f"  [{i}] ({ctx['label']}) activation={ctx['max_activation']:.3f}: \"{ctx['response_preview']}\"")
    contexts_text = "\n".join(ctx_lines) if ctx_lines else "  (no activating contexts found)"

    community_note = f" (part of Community {community_id})" if community_id is not None else ""
    return f"""You are analyzing a sparse autoencoder (SAE) feature from a language model's internal representations.

Feature: F{feature_id} in layer {layer}{community_note}
- Direction alignment (cosine with deception direction): {direction_alignment:+.4f}
- Effect size (deceptive vs honest): {effect_size:+.4f}

The feature activates most strongly in these response contexts:
{contexts_text}

Based on these activation patterns, provide:
1. A concise semantic label (5-10 words) for what this feature represents.
2. A 2-3 sentence explanation of what linguistic or conceptual pattern triggers this feature.
3. How it might relate to deceptive vs. honest generation (given the direction alignment).

Be specific and grounded in the examples. Avoid vague descriptions like "general language patterns."
Reply in JSON: {{"label": "...", "explanation": "...", "deception_role": "..."}}"""


def call_groq(prompt: str, api_key: str, max_retries: int = 3) -> str:
    """Call Groq API with retry."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI interpretability researcher. Reply only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            wait = 10 * (attempt + 1)
            print(f"  [WARN] Groq call failed (attempt {attempt+1}): {exc}. Waiting {wait}s...")
            time.sleep(wait)
    return ""


def parse_llm_response(text: str) -> dict:
    import re
    stripped = text.strip()
    match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"label": "parse_error", "explanation": stripped[:200], "deception_role": "unknown"}


def main():
    print("=" * 65)
    print("  Community 14 + Direction-Top Features -- Autointerp")
    print("=" * 65)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[ERROR] GROQ_API_KEY environment variable not set.")
        print("  Set it with: set GROQ_API_KEY=your_key_here")
        sys.exit(1)

    # Load community data
    communities_data = json.loads(COMMUNITIES_PATH.read_text(encoding="utf-8"))
    layer_communities = communities_data.get(FOCUS_LAYER, {})
    target_community = None
    for c in layer_communities.get("communities", []):
        if c["community_id"] == FOCUS_COMMUNITY_ID:
            target_community = c
            break

    if not target_community:
        print(f"[ERROR] Community {FOCUS_COMMUNITY_ID} not found in {FOCUS_LAYER}")
        sys.exit(1)

    community_feature_ids = target_community["feature_ids"]
    print(f"\nCommunity {FOCUS_COMMUNITY_ID} ({FOCUS_LAYER}): {len(community_feature_ids)} features")
    print(f"  h_rate={target_community['honest_activation_rate']:.3f}  "
          f"d_rate={target_community['deceptive_activation_rate']:.3f}  "
          f"diff={target_community['rate_difference']:+.3f}  "
          f"align={target_community['direction_alignment']:+.4f}")

    # Load direction scoring data to get individual feature alignment scores
    dir_scoring = json.loads(DIRECTION_SCORING_PATH.read_text(encoding="utf-8"))
    layer_dir_data = dir_scoring.get(FOCUS_LAYER, {})

    # Build per-feature alignment + effect lookup from direction_top15 + vanilla_top15
    feature_alignment_map: dict[int, float] = {}
    feature_effect_map: dict[int, float] = {}
    for item in layer_dir_data.get("direction_top15", []):
        feature_alignment_map[item["feature_id"]] = item["alignment"]
        feature_effect_map[item["feature_id"]] = item["vanilla_effect_size"]
    for item in layer_dir_data.get("vanilla_top15", []):
        fid = item["feature_id"]
        if fid not in feature_alignment_map:
            feature_alignment_map[fid] = item["alignment"]
            feature_effect_map[fid] = item["vanilla_effect_size"]

    # Select all community features + top direction-aligned features not already in community
    dir_top15_ids = [item["feature_id"] for item in layer_dir_data.get("direction_top15", [])]
    extra_ids = [fid for fid in dir_top15_ids if fid not in community_feature_ids][:TOP_DIRECTION_FEATURES]
    all_target_features = list(dict.fromkeys(community_feature_ids + extra_ids))

    print(f"\nTotal features to interpret: {len(all_target_features)}")
    print(f"  Community 14 members: {len(community_feature_ids)}")
    print(f"  Additional direction-top features: {len(extra_ids)}")

    # Load activations + encode
    print(f"\nLoading activations for {FOCUS_LAYER}...")
    activations, metadata = load_chunk(FOCUS_LAYER)
    labels = list(metadata.get("labels", []))
    token_lengths = [int(x) for x in metadata.get("token_lengths", [])]
    responses = list(metadata.get("responses", []))

    print("Encoding through SAE...")
    sae = load_sae(FOCUS_LAYER)
    all_features = encode_sae(sae, activations)
    print(f"Done. Shape: {all_features.shape}")

    # Interpret each feature
    results = []
    for i, feature_id in enumerate(all_target_features):
        is_community = feature_id in community_feature_ids
        group_tag = f"community_{FOCUS_COMMUNITY_ID}" if is_community else "direction_top"

        print(f"\n[{i+1}/{len(all_target_features)}] F{feature_id} ({group_tag})")

        # Find top activating contexts
        deceptive_contexts = find_top_activating_tokens(
            feature_id, all_features, responses, token_lengths, labels,
            target_label="deceptive",
        )
        honest_contexts = find_top_activating_tokens(
            feature_id, all_features, responses, token_lengths, labels,
            target_label="honest",
        )
        all_contexts = sorted(
            deceptive_contexts[:6] + honest_contexts[:6],
            key=lambda x: x["max_activation"],
            reverse=True
        )[:MAX_CONTEXTS_PER_FEATURE]

        alignment = feature_alignment_map.get(feature_id, target_community["direction_alignment"])
        effect_size = feature_effect_map.get(feature_id, 0.0)

        print(f"  alignment={alignment:+.4f}  effect_size={effect_size:+.4f}  contexts={len(all_contexts)}")

        # Call Groq
        prompt = build_feature_description_prompt(
            feature_id=feature_id,
            layer=FOCUS_LAYER,
            contexts=all_contexts,
            direction_alignment=alignment,
            effect_size=effect_size,
            community_id=FOCUS_COMMUNITY_ID if is_community else None,
        )
        raw_response = call_groq(prompt, api_key)
        parsed = parse_llm_response(raw_response)

        print(f"  Label: {parsed.get('label', '?')}")
        print(f"  Role:  {parsed.get('deception_role', '?')[:80]}")

        results.append({
            "feature_id": feature_id,
            "layer": FOCUS_LAYER,
            "group": group_tag,
            "direction_alignment": alignment,
            "effect_size": effect_size,
            "deceptive_contexts": deceptive_contexts[:6],
            "honest_contexts": honest_contexts[:6],
            "interpretation": parsed,
        })

        time.sleep(1.5)  # rate limit safety

    # Save
    out_path = OUTPUT_DIR / f"community_{FOCUS_COMMUNITY_ID}_layer20_autointerp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")

    # Summary
    print("\n" + "=" * 65)
    print("INTERPRETATION SUMMARY")
    print("=" * 65)
    community_results = [r for r in results if r["group"] == f"community_{FOCUS_COMMUNITY_ID}"]
    direction_results = [r for r in results if r["group"] == "direction_top"]

    print(f"\nCommunity {FOCUS_COMMUNITY_ID} features:")
    for r in community_results:
        interp = r["interpretation"]
        print(f"  F{r['feature_id']:5d}  align={r['direction_alignment']:+.4f}  "
              f"\"{interp.get('label', '?')}\"")

    print("\nTop direction-aligned features:")
    for r in direction_results:
        interp = r["interpretation"]
        print(f"  F{r['feature_id']:5d}  align={r['direction_alignment']:+.4f}  "
              f"\"{interp.get('label', '?')}\"")

    print(f"\nDone. Report saved to {out_path}")


if __name__ == "__main__":
    main()
