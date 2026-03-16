import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config_manager import ConfigManager
from project_layout import default_checkpoint_path, default_feature_mapping_dir
from sae.model import SparseAutoEncoder


class FeatureMapping:
    def __init__(self, config):
        self.config = config
        device_name = self._cfg_section("feature-mapping").get("device")
        if not device_name:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model_name = self.config.get("model_name", "gpt2-small")
        self.tokenizer_name = self._resolve_tokenizer_name()
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self._token_cache = {}
        self._sequence_cache = {}

    def _resolve_tokenizer_name(self):
        mapping_cfg = self._cfg_section("feature-mapping")
        explicit_name = mapping_cfg.get("tokenizer_name")
        if explicit_name:
            return explicit_name

        alias_map = {
            "gpt2-small": "gpt2",
        }
        return alias_map.get(self.model_name, self.model_name)

    def _cfg_value(self, key, default=None):
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def _dataset_paths(self):
        paths = self._cfg_section("datasets").get("path", [])
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not paths:
            raise ValueError("config.datasets.path must be a non-empty string or list.")
        return paths

    def _decode_token(self, token_id):
        token_id = int(token_id)
        cached = self._token_cache.get(token_id)
        if cached is not None:
            return cached

        token_str = self._tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        self._token_cache[token_id] = token_str
        return token_str

    def _decode_token_for_csv(self, token_id):
        # Escaped representation is easier to read in CSV than raw GPT-2 bytes/spaces.
        return repr(self._decode_token(token_id))

    def _decode_sequence_preview(self, token_ids, max_chars):
        cache_key = tuple(int(t) for t in token_ids)
        cached = self._sequence_cache.get(cache_key)
        if cached is not None:
            return cached

        text = self._tokenizer.decode(list(cache_key), clean_up_tokenization_spaces=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        self._sequence_cache[cache_key] = text
        return text

    def _iter_activation_token_rows(self):
        global_token_idx = 0
        global_sequence_idx = 0
        mapping_cfg = self._cfg_section("feature-mapping")
        max_prompt_preview_chars = int(mapping_cfg.get("prompt_preview_chars", 180))
        default_layer = self._cfg_value("extraction.target", "")

        for data_path in self._dataset_paths():
            payload = torch.load(data_path, map_location="cpu", weights_only=False)
            activations = payload.get("activations")
            tokens = payload.get("tokens")
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            token_lengths = metadata.get("token_lengths", []) if isinstance(metadata, dict) else []
            target_layer = metadata.get("target_layer", default_layer) if isinstance(metadata, dict) else default_layer
            chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) else None

            if activations is None or tokens is None:
                raise ValueError(f"Chunk {data_path} must contain both 'activations' and 'tokens'.")

            if int(activations.shape[0]) != int(tokens.shape[0]):
                raise ValueError(
                    f"Row/token mismatch in {data_path}: activations={activations.shape[0]} tokens={tokens.shape[0]}"
                )

            total_rows = int(activations.shape[0])

            # Prefer sequence-aware iteration when token_lengths metadata is present.
            if token_lengths and sum(int(x) for x in token_lengths) == total_rows:
                start = 0
                for seq_local_idx, seq_len in enumerate(token_lengths):
                    seq_len = int(seq_len)
                    end = start + seq_len
                    seq_token_ids = [int(t.item()) for t in tokens[start:end]]
                    prompt_preview = self._decode_sequence_preview(
                        token_ids=seq_token_ids,
                        max_chars=max_prompt_preview_chars,
                    )

                    for token_idx_in_sequence in range(seq_len):
                        row_idx = start + token_idx_in_sequence
                        yield {
                            "sample_idx": global_token_idx,
                            "sequence_idx": global_sequence_idx,
                            "token_idx_in_sequence": token_idx_in_sequence,
                            "token_idx_in_chunk": row_idx,
                            "activation": activations[row_idx],
                            "token_id": int(tokens[row_idx].item()),
                            "prompt_preview": prompt_preview,
                            "target_layer": target_layer,
                            "chunk_path": data_path,
                            "chunk_id": chunk_id,
                            "sequence_local_idx": seq_local_idx,
                        }
                        global_token_idx += 1

                    start = end
                    global_sequence_idx += 1
            else:
                for row_idx in range(total_rows):
                    token_id = int(tokens[row_idx].item())
                    yield {
                        "sample_idx": global_token_idx,
                        "sequence_idx": global_sequence_idx,
                        "token_idx_in_sequence": row_idx,
                        "token_idx_in_chunk": row_idx,
                        "activation": activations[row_idx],
                        "token_id": token_id,
                        "prompt_preview": self._decode_sequence_preview([token_id], max_prompt_preview_chars),
                        "target_layer": target_layer,
                        "chunk_path": data_path,
                        "chunk_id": chunk_id,
                        "sequence_local_idx": 0,
                    }
                    global_token_idx += 1
                global_sequence_idx += 1

    def get_model_config(self, model_path):
        base_config = torch.load(model_path, map_location="cpu", weights_only=False).get("config", {})

        if not base_config:
            raise ValueError(f"Model checkpoint at {model_path} does not contain config information.")
        return base_config.get("sae", {})

    def load_sae_model(self, input_dim):
        model_path = self._cfg_section("feature-mapping").get("model_path")
        checkpoint_config = self.get_model_config(model_path) if model_path else {}

        hidden_dim = int(checkpoint_config.get("hidden_dim", self._cfg_section("sae").get("hidden_dim", 128)))
        sparsity_weight = checkpoint_config.get("sparsity_weight", 1e-3)

        model = SparseAutoEncoder(input_dim, hidden_dim, sparsity_weight)

        if model_path:
            model.load_model(model_path, device=self.device)

        return model.to(self.device)

    def infer_input_dim(self):
        first_path = self._dataset_paths()[0]
        first_payload = torch.load(first_path, map_location="cpu", weights_only=False)
        activations = first_payload.get("activations")
        if activations is None:
            raise ValueError(f"Chunk {first_path} does not contain 'activations'.")
        return int(activations.shape[-1])
    
    def _cfg_section(self, name):
        if hasattr(self.config, "section"):
            return self.config.section(name)
        if isinstance(self.config, dict):
            return self.config.get(name, {})
        return {}
    
    def select_active_features(self, f_1d, top_k, threshold):
        active_mask = f_1d >= threshold
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()

        if active_indices.numel() == 0:
            return [], []

        active_values = f_1d[active_indices]
        k = min(int(top_k), int(active_values.numel()))
        top_values, top_local_indices = torch.topk(active_values, k=k)
        top_indices = active_indices[top_local_indices]
        return top_indices.tolist(), top_values.tolist()

    def write_events_csv(self, events, file_path):
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_idx",
            "sequence_idx",
            "sequence_local_idx",
            "token_idx_in_chunk",
            "token_idx_in_sequence",
            "token_id",
            "token_str",
            "prompt_preview",
            "target_layer",
            "chunk_id",
            "chunk_path",
            "feature_id",
            "strength",
            "rank",
            "device",
            "checkpoint",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

    def write_summary_files(self, events, summary_csv_path, summary_txt_path):
        output_csv = Path(summary_csv_path)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        feature_values = defaultdict(list)
        feature_token_counter = defaultdict(Counter)
        for event in events:
            feature_values[int(event["feature_id"])].append(float(event["strength"]))
            feature_token_counter[int(event["feature_id"])][event["token_str"]] += 1

        rows = []
        for feature_id, strengths in sorted(feature_values.items()):
            strengths_tensor = torch.tensor(strengths, dtype=torch.float32)
            top_tokens = feature_token_counter[feature_id].most_common(5)
            top_tokens_text = " | ".join(f"{tok}:{cnt}" for tok, cnt in top_tokens)
            rows.append(
                {
                    "feature_id": feature_id,
                    "count_active": int(strengths_tensor.numel()),
                    "mean_strength": float(strengths_tensor.mean().item()),
                    "max_strength": float(strengths_tensor.max().item()),
                    "p95_strength": float(torch.quantile(strengths_tensor, 0.95).item()),
                    "top_tokens": top_tokens_text,
                }
            )

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "feature_id",
                    "count_active",
                    "mean_strength",
                    "max_strength",
                    "p95_strength",
                    "top_tokens",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        # Cleaner aggregate view: one row per (feature, token) pair.
        token_pair_path = output_csv.with_name(output_csv.stem + "_token_pairs.csv")
        pair_rows = []
        pair_counter = defaultdict(Counter)
        pair_strength_sum = defaultdict(lambda: defaultdict(float))
        pair_strength_max = defaultdict(lambda: defaultdict(float))

        for event in events:
            feature_id = int(event["feature_id"])
            token = event["token_str"]
            strength = float(event["strength"])
            pair_counter[feature_id][token] += 1
            pair_strength_sum[feature_id][token] += strength
            pair_strength_max[feature_id][token] = max(pair_strength_max[feature_id][token], strength)

        for feature_id in sorted(pair_counter.keys()):
            for token, cnt in pair_counter[feature_id].most_common(30):
                pair_rows.append(
                    {
                        "feature_id": feature_id,
                        "token_str": token,
                        "count": int(cnt),
                        "mean_strength": float(pair_strength_sum[feature_id][token] / max(cnt, 1)),
                        "max_strength": float(pair_strength_max[feature_id][token]),
                    }
                )

        with token_pair_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["feature_id", "token_str", "count", "mean_strength", "max_strength"],
            )
            writer.writeheader()
            writer.writerows(pair_rows)

        summary_lines = [
            f"Model: {self.model_name}",
            f"Tokenizer: {self.tokenizer_name}",
            f"Target layer: {events[0]['target_layer'] if events else ''}",
            f"Total samples: {len(set(e['sample_idx'] for e in events)) if events else 0}",
            f"Total sequences: {len(set(e['sequence_idx'] for e in events)) if events else 0}",
            f"Total events: {len(events)}",
            f"Unique active features: {len(feature_values)}",
            f"Token-pair csv: {token_pair_path}",
        ]
        if rows:
            top_rows = sorted(rows, key=lambda r: r["count_active"], reverse=True)[:20]
            summary_lines.append("Top features by count_active:")
            for row in top_rows:
                summary_lines.append(
                    f"feature={row['feature_id']} count={row['count_active']} "
                    f"mean={row['mean_strength']:.6f} max={row['max_strength']:.6f} p95={row['p95_strength']:.6f} "
                    f"top_tokens={row['top_tokens']}"
                )

        output_txt = Path(summary_txt_path)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    def run(self):
        mapping_cfg = self._cfg_section("feature-mapping")
        count = int(mapping_cfg.get("count", 2000))
        top_k = int(mapping_cfg.get("top_k", 10))
        threshold = float(mapping_cfg.get("strength_threshold", 0.0))
        base_mapping_dir = Path(default_feature_mapping_dir(self.config))
        event_output_path = mapping_cfg.get(
            "event_output_path",
            str(base_mapping_dir / "feature_mapping_events.csv"),
        )
        summary_txt_path = mapping_cfg.get(
            "summary_output_path",
            str(base_mapping_dir / "feature_mapping_summary.txt"),
        )
        summary_csv_path = mapping_cfg.get(
            "summary_csv_output_path",
            str(base_mapping_dir / "feature_mapping_feature_summary.csv"),
        )
        checkpoint_path = mapping_cfg.get("model_path") or default_checkpoint_path(self.config)
        show_token_progress = bool(mapping_cfg.get("show_token_progress", True))

        input_dim = self.infer_input_dim()
        sae_model = self.load_sae_model(input_dim=input_dim)

        sae_model.eval()
        events = []

        with torch.no_grad():
            rows = tqdm(
                self._iter_activation_token_rows(),
                total=count,
                desc="Feature mapping",
                disable=not show_token_progress,
            )
            for row in rows:
                sample_idx = int(row["sample_idx"])
                if sample_idx >= count:
                    break

                x = row["activation"]
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                elif x.ndim > 2:
                    x = x.view(-1, x.shape[-1])

                x = x.to(self.device)

                _, f = sae_model(x)
                f_1d = f[0]

                top_indices, top_values = self.select_active_features(
                    f_1d=f_1d,
                    top_k=top_k,
                    threshold=threshold,
                )

                for rank, (feature_id, strength) in enumerate(zip(top_indices, top_values), start=1):
                    token_id = int(row["token_id"])
                    events.append(
                        {
                            "sample_idx": sample_idx,
                            "sequence_idx": int(row["sequence_idx"]),
                            "sequence_local_idx": int(row["sequence_local_idx"]),
                            "token_idx_in_chunk": int(row["token_idx_in_chunk"]),
                            "token_idx_in_sequence": int(row["token_idx_in_sequence"]),
                            "token_id": token_id,
                            "token_str": self._decode_token_for_csv(token_id),
                            "prompt_preview": row["prompt_preview"],
                            "target_layer": row["target_layer"],
                            "chunk_id": row["chunk_id"],
                            "chunk_path": row["chunk_path"],
                            "feature_id": int(feature_id),
                            "strength": float(strength),
                            "rank": rank,
                            "device": str(self.device),
                            "checkpoint": checkpoint_path,
                        }
                    )

        self.write_events_csv(events=events, file_path=event_output_path)
        self.write_summary_files(
            events=events,
            summary_csv_path=summary_csv_path,
            summary_txt_path=summary_txt_path,
        )

        print(f"Events written to {event_output_path}")
        print(f"Feature summary csv written to {summary_csv_path}")
        print(f"Run summary written to {summary_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature mapping with SAE.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()

    config = ConfigManager.from_file(args.config)
    mapping = FeatureMapping(config)
    mapping.run()