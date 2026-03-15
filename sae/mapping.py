import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config_manager import ConfigManager
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

    def _iter_activation_token_rows(self):
        sample_idx = 0
        for data_path in self._dataset_paths():
            payload = torch.load(data_path, map_location="cpu")
            activations = payload.get("activations")
            tokens = payload.get("tokens")
            if activations is None or tokens is None:
                raise ValueError(f"Chunk {data_path} must contain both 'activations' and 'tokens'.")

            if int(activations.shape[0]) != int(tokens.shape[0]):
                raise ValueError(
                    f"Row/token mismatch in {data_path}: activations={activations.shape[0]} tokens={tokens.shape[0]}"
                )

            for row_idx in range(int(activations.shape[0])):
                yield {
                    "sample_idx": sample_idx,
                    "token_idx_in_chunk": row_idx,
                    "activation": activations[row_idx],
                    "token_id": int(tokens[row_idx].item()),
                }
                sample_idx += 1

    def get_model_config(self, model_path):
        base_config = torch.load(model_path, map_location="cpu").get("config", {})

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
        first_payload = torch.load(first_path, map_location="cpu")
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
            "token_idx_in_chunk",
            "token_id",
            "token_str",
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

        summary_lines = [
            f"Total samples: {len(set(e['sample_idx'] for e in events)) if events else 0}",
            f"Total events: {len(events)}",
            f"Unique active features: {len(feature_values)}",
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
        event_output_path = mapping_cfg.get("event_output_path", "reports/feature_mapping_events.csv")
        summary_txt_path = mapping_cfg.get("summary_output_path", "reports/feature_mapping_summary.txt")
        summary_csv_path = mapping_cfg.get("summary_csv_output_path", "reports/feature_mapping_feature_summary.csv")
        checkpoint_path = mapping_cfg.get("model_path", "")
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
                            "token_idx_in_chunk": int(row["token_idx_in_chunk"]),
                            "token_id": token_id,
                            "token_str": self._decode_token(token_id),
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