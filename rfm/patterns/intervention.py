from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

from rfm.layout import resolve_requested_targets, sanitize_layer_name
from rfm.patterns.modeling import AxisMonitor, AxisProbe, AxisProbeState, DirectionResult
from rfm.patterns.spec import ContrastAxisSpec
from rfm.steering.hook import HFSteeringHook


def layer_payload_from_result(result: dict[str, Any], target: str) -> dict[str, Any]:
    return {
        "feature_scores": result["layer_feature_scores"].get(target, []),
        "aggregation": {
            "selected_method": result["selected_aggregation"],
            "benchmark": result["aggregation_benchmark"],
        },
        "feature_pool": result["feature_pools"].get(target, {}),
        "thresholds": result["thresholds"].get(target, {}),
    }


def analysis_payload_from_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "alignment_report": result.get("alignment_report", {}),
        "aggregation_benchmark": result.get("aggregation_benchmark", []),
        "selected_aggregation": result.get("selected_aggregation"),
        "model_metrics": result.get("model_metrics", {}),
        "stable_motifs": result.get("stable_motifs", []),
        "stable_interactions": result.get("stable_interactions", []),
        "feature_importance": result.get("feature_importance", []),
        "decision_tree": result.get("decision_tree", {}),
        "intervention_effects": result.get("intervention_effects", []),
        "coactivation_stats": result.get("coactivation_stats", []),
    }
    if "motif_candidates" in result:
        payload["motif_candidates"] = result["motif_candidates"]
    if "causal_validation" in result:
        payload["causal_validation"] = result["causal_validation"]
    return payload


def motif_members(motif: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(motif.get("kind", "single"))
    if kind == "interaction":
        members = list(motif.get("members", []) or [])
        return [
            {
                "layer": str(member["layer"]),
                "feature_id": int(member["feature_id"]),
            }
            for member in members
            if isinstance(member, dict) and "layer" in member and "feature_id" in member
        ]
    layer_name = motif.get("layer")
    feature_id = motif.get("feature_id")
    if layer_name is None or feature_id is None:
        return []
    return [{"layer": str(layer_name), "feature_id": int(feature_id)}]


def grouped_motif_members(motif: dict[str, Any]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for member in motif_members(motif):
        grouped.setdefault(member["layer"], []).append(int(member["feature_id"]))
    return grouped


def motif_feature_configs(
    motif: dict[str, Any],
    *,
    action: str,
    alpha: float = 5.0,
) -> dict[str, list[dict[str, Any]]]:
    action = str(action).strip().lower()
    if action not in {"ablate", "amplify"}:
        raise ValueError(f"Unsupported motif action: {action}")

    mode = "ablate" if action == "ablate" else "add"
    configs: dict[str, list[dict[str, Any]]] = {}
    for member in motif_members(motif):
        configs.setdefault(member["layer"], []).append(
            {
                "feature_id": int(member["feature_id"]),
                "alpha": float(alpha),
                "mode": mode,
            }
        )
    return configs


def register_feature_interventions(
    *,
    model,
    sae_models: dict[str, torch.nn.Module],
    layer_feature_configs: dict[str, list[dict[str, Any]]],
) -> list[Any]:
    handles = []
    for layer_name, feature_configs in layer_feature_configs.items():
        sae_model = sae_models.get(layer_name)
        if sae_model is None:
            raise KeyError(f"No SAE model loaded for layer: {layer_name}")
        for feature_cfg in feature_configs:
            handles.append(
                HFSteeringHook.apply(
                    hf_model=model,
                    target_layer=layer_name,
                    sae_model=sae_model,
                    feature_id=int(feature_cfg["feature_id"]),
                    alpha=float(feature_cfg.get("alpha", 0.0)),
                    mode=str(feature_cfg.get("mode", "add")),
                )
            )
    return handles


def load_model_and_tokenizer(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.get("model_name", "gpt2")
    local_files_only = bool(
        config.get(
            "patterns.causal_intervention.local_files_only",
            config.get("extraction.local_files_only", True),
        )
    )
    device = config.get(
        "patterns.causal_intervention.device",
        config.get(
            "steering.device",
            config.get(
                "extraction.device",
                config.get("train.device", "cuda" if torch.cuda.is_available() else "cpu"),
            ),
        ),
    )
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype_str = config.get("extraction.dtype", "float32")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(dtype_str).lower(), torch.float32)
    if device == "cpu" and dtype in {torch.bfloat16, torch.float16}:
        dtype = torch.float32
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    pretrained_ref = model_name
    if local_files_only:
        from huggingface_hub import snapshot_download

        pretrained_ref = snapshot_download(repo_id=model_name, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_ref,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ref, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, device


def _deception_artifact_path(config, *parts: str) -> Path:
    from rfm.deception.utils import deception_run_dir

    return Path(deception_run_dir(config, *parts))


def axis_monitor_from_bundle(
    *,
    axis_spec: ContrastAxisSpec,
    bundle: dict[str, Any],
    ensemble_method: str = "weighted_average",
    config=None,
) -> AxisMonitor | None:
    bundle_layers = dict(bundle.get("layers", {}) or {})
    directions = {}
    probes = {}
    for layer_name, payload in bundle_layers.items():
        direction_payload = payload.get("direction")
        if isinstance(direction_payload, dict) and "direction" in direction_payload:
            directions[layer_name] = DirectionResult.from_dict(direction_payload)
        probe_payload = payload.get("probe_state")
        if isinstance(probe_payload, dict) and "weight" in probe_payload:
            probe = AxisProbe(axis_spec=axis_spec)
            probe.state = AxisProbeState.from_dict(probe_payload)
            probes[layer_name] = probe

    if config is not None:
        direction_path = _deception_artifact_path(config, "directions", "directions.pt")
        if direction_path.exists():
            payload = torch.load(direction_path, map_location="cpu", weights_only=False)
            for layer_name, direction_payload in dict(payload or {}).items():
                if layer_name in directions:
                    continue
                if isinstance(direction_payload, dict):
                    directions[layer_name] = DirectionResult.from_dict(direction_payload)
                elif isinstance(direction_payload, DirectionResult):
                    directions[layer_name] = direction_payload

        target_layers = list(dict.fromkeys([*bundle_layers.keys(), *resolve_requested_targets(config)]))
        for layer_name in target_layers:
            if layer_name in probes:
                continue
            probe_path = _deception_artifact_path(config, "probes", f"{sanitize_layer_name(layer_name)}.pt")
            if not probe_path.exists():
                continue
            probe = AxisProbe(axis_spec=axis_spec)
            probe.load(probe_path)
            probes[layer_name] = probe

    if not directions and not probes:
        return None

    monitor_payload = dict(bundle.get("monitor", {}) or {})
    thresholds_raw = monitor_payload.get("thresholds", {})
    if (not thresholds_raw) and config is not None:
        report_path = _deception_artifact_path(config, "monitor", "monitor_report.json")
        if report_path.exists():
            thresholds_raw = dict(json.loads(report_path.read_text(encoding="utf-8")) or {}).get("thresholds", {})
    if isinstance(thresholds_raw, dict):
        thresholds = {str(layer): float(value) for layer, value in thresholds_raw.items()}
    else:
        thresholds = {layer_name: float(thresholds_raw) for layer_name in bundle_layers}

    return AxisMonitor(
        axis_spec=axis_spec,
        directions=directions,
        probes=probes,
        thresholds=thresholds,
        ensemble_method=ensemble_method,
    )


def apply_causal_validation(
    result: dict[str, Any],
    effect_rows: list[dict[str, Any]],
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(result)
    candidates = list(updated.get("motif_candidates", []) or [])
    if not candidates:
        candidates = [dict(item) for item in updated.get("stable_motifs", []) or []]

    effect_lookup = {str(row.get("name")): row for row in effect_rows}
    for item in candidates:
        effect = effect_lookup.get(str(item.get("name")))
        if effect is not None:
            item["causal_validation"] = effect
        if item.get("status") == "stable" and (effect is None or not bool(effect.get("supports_causal_effect", False))):
            item["status"] = "hypothesis"
            item["status_reason"] = "causal_validation_failed" if effect else "causal_validation_missing"

    updated["motif_candidates"] = candidates
    updated["stable_motifs"] = [item for item in candidates if item.get("status") == "stable"]
    updated["stable_interactions"] = [
        item
        for item in updated["stable_motifs"]
        if str(item.get("kind", "")) == "interaction"
    ]
    updated["intervention_effects"] = effect_rows
    updated["causal_validation"] = dict(summary or {})
    return updated


def _chunk_metadata(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("metadata", {}) or {})


def load_intervention_prompts(
    config,
    *,
    chunk_dirs: dict[str, str | Path] | None = None,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], str]:
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()

    scenario_path = None
    if hasattr(config, "get"):
        scenario_path = config.get("deception.scenario_generator.cache_path")

    if scenario_path:
        from rfm.deception.deception_dataset import DeceptionDataset

        if Path(scenario_path).exists():
            try:
                dataset = DeceptionDataset(config=config, mode="paired")
                dataset.load()
                for row in dataset.iter_scenarios():
                    question = str(row.get("question", "")).strip()
                    if not question or question in seen:
                        continue
                    seen.add(question)
                    prompts.append(
                        {
                            "prompt": question,
                            "category": row.get("category", "unknown"),
                            "difficulty": row.get("difficulty", "unknown"),
                            "source": row.get("source", "dataset"),
                        }
                    )
                    if len(prompts) >= max(int(limit), 1):
                        break
                if prompts:
                    return prompts, "scenario_dataset"
            except Exception:
                prompts = []

    for chunk_dir in (chunk_dirs or {}).values():
        for path in sorted(Path(chunk_dir).glob("*.pt")):
            metadata = _chunk_metadata(path)
            questions = list(metadata.get("questions", []) or metadata.get("prompts", []) or [])
            categories = list(metadata.get("categories", []) or [])
            difficulties = list(metadata.get("difficulties", []) or [])
            for index, question in enumerate(questions):
                question = str(question).strip()
                if not question or question in seen:
                    continue
                seen.add(question)
                prompts.append(
                    {
                        "prompt": question,
                        "category": categories[index] if index < len(categories) else "unknown",
                        "difficulty": difficulties[index] if index < len(difficulties) else "unknown",
                        "source": "chunk_metadata",
                    }
                )
                if len(prompts) >= max(int(limit), 1):
                    return prompts, "chunk_metadata"
    return prompts, "unavailable"


def _system_prompt_for_label(config, axis_spec: ContrastAxisSpec, label: str) -> str | None:
    if label == axis_spec.endpoint_a:
        return config.get(
            "patterns.causal_intervention.endpoint_a_system_prompt",
            config.get(f"deception.extraction.system_prompt_{axis_spec.endpoint_a}", None),
        )
    return config.get(
        "patterns.causal_intervention.endpoint_b_system_prompt",
        config.get(f"deception.extraction.system_prompt_{axis_spec.endpoint_b}", None),
    )


def _generation_kwargs(config) -> dict[str, Any]:
    max_new_tokens = int(config.get("patterns.causal_intervention.max_new_tokens", config.get("generation.max_new_tokens", 96)))
    temperature = float(config.get("patterns.causal_intervention.temperature", 0.0))
    top_p = float(config.get("patterns.causal_intervention.top_p", config.get("generation.top_p", 1.0)))
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs


class MotifCausalValidator:
    def __init__(
        self,
        *,
        config,
        axis_spec: ContrastAxisSpec,
        bundle: dict[str, Any],
        sae_models: dict[str, torch.nn.Module],
        chunk_dirs: dict[str, str | Path] | None = None,
        model=None,
        tokenizer=None,
        device: str | None = None,
    ):
        self.config = config
        self.axis = axis_spec
        self.bundle = bundle
        self.sae_models = sae_models
        self.chunk_dirs = chunk_dirs or {}
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_kwargs = _generation_kwargs(config)
        self.max_prompts = int(config.get("patterns.causal_intervention.max_prompts", 4))
        self.amplify_alpha = float(config.get("patterns.causal_intervention.amplify_alpha", 5.0))
        self.min_effect = float(config.get("patterns.causal_intervention.min_effect", config.get("patterns.intervention_min_shift", 0.01)))
        ensemble_method = config.get("deception.monitor.ensemble_method", config.get("patterns.causal_intervention.ensemble_method", "weighted_average"))
        self.monitor = axis_monitor_from_bundle(
            axis_spec=axis_spec,
            bundle=bundle,
            ensemble_method=ensemble_method,
            config=config,
        )
        self.prompts, self.prompt_source = load_intervention_prompts(config, chunk_dirs=self.chunk_dirs, limit=self.max_prompts)

    def available(self) -> tuple[bool, str]:
        if self.monitor is None:
            return False, "monitor_unavailable"
        if not self.prompts:
            return False, f"prompt_source_{self.prompt_source}"
        return True, "ready"

    def _ensure_runtime(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(self.config)

    def _run_generation(self, prompt: str, system_prompt: str | None, layer_feature_configs: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
        self._ensure_runtime()
        handles = []
        if layer_feature_configs:
            handles = register_feature_interventions(
                model=self.model,
                sae_models=self.sae_models,
                layer_feature_configs=layer_feature_configs,
            )
        try:
            return self.monitor.generate_with_monitoring(
                self.model,
                self.tokenizer,
                prompt=prompt,
                system_prompt=system_prompt,
                **self.generation_kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

    def evaluate(self, motifs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ok, status = self.available()
        if not ok:
            return [], {"status": "skipped", "reason": status, "evaluated": 0, "supported": 0}

        rows = [self.evaluate_motif(motif) for motif in motifs]
        supported = sum(1 for row in rows if row.get("supports_causal_effect"))
        return rows, {
            "status": "ok",
            "reason": "",
            "evaluated": len(rows),
            "supported": supported,
            "prompt_source": self.prompt_source,
            "max_prompts": len(self.prompts),
            "monitor_backend": "generation_monitor",
        }

    def evaluate_motif(self, motif: dict[str, Any]) -> dict[str, Any]:
        sign = str(motif.get("sign", "endpoint_b"))
        if sign == "endpoint_a":
            primary_label = self.axis.endpoint_a
            opposite_label = self.axis.endpoint_b
        else:
            primary_label = self.axis.endpoint_b
            opposite_label = self.axis.endpoint_a

        primary_system_prompt = _system_prompt_for_label(self.config, self.axis, primary_label)
        opposite_system_prompt = _system_prompt_for_label(self.config, self.axis, opposite_label)
        ablation_configs = motif_feature_configs(motif, action="ablate", alpha=self.amplify_alpha)
        amplify_configs = motif_feature_configs(motif, action="amplify", alpha=self.amplify_alpha)

        primary_pairs = []
        opposite_pairs = []
        for prompt_record in self.prompts:
            prompt = str(prompt_record["prompt"])
            baseline_primary = self._run_generation(prompt, primary_system_prompt)
            ablated_primary = self._run_generation(prompt, primary_system_prompt, layer_feature_configs=ablation_configs)
            baseline_opposite = self._run_generation(prompt, opposite_system_prompt)
            amplified_opposite = self._run_generation(prompt, opposite_system_prompt, layer_feature_configs=amplify_configs)
            primary_pairs.append((prompt_record, baseline_primary, ablated_primary))
            opposite_pairs.append((prompt_record, baseline_opposite, amplified_opposite))

        baseline_primary_mean = float(sum(item[1]["contrast_probability"] for item in primary_pairs) / max(len(primary_pairs), 1))
        ablated_primary_mean = float(sum(item[2]["contrast_probability"] for item in primary_pairs) / max(len(primary_pairs), 1))
        baseline_opposite_mean = float(sum(item[1]["contrast_probability"] for item in opposite_pairs) / max(len(opposite_pairs), 1))
        amplified_opposite_mean = float(sum(item[2]["contrast_probability"] for item in opposite_pairs) / max(len(opposite_pairs), 1))

        ablation_shift = ablated_primary_mean - baseline_primary_mean
        amplification_shift = amplified_opposite_mean - baseline_opposite_mean
        if sign == "endpoint_a":
            aligned_ablation_shift = ablation_shift
            aligned_amplification_shift = -amplification_shift
        else:
            aligned_ablation_shift = -ablation_shift
            aligned_amplification_shift = amplification_shift

        response_change_rate_primary = float(
            sum(
                1
                for _, baseline, intervention in primary_pairs
                if baseline["response"].strip() != intervention["response"].strip()
            )
            / max(len(primary_pairs), 1)
        )
        response_change_rate_opposite = float(
            sum(
                1
                for _, baseline, intervention in opposite_pairs
                if baseline["response"].strip() != intervention["response"].strip()
            )
            / max(len(opposite_pairs), 1)
        )

        examples = []
        for prompt_record, baseline, intervention in (primary_pairs[:1] + opposite_pairs[:1]):
            examples.append(
                {
                    "prompt": prompt_record["prompt"],
                    "category": prompt_record.get("category", "unknown"),
                    "difficulty": prompt_record.get("difficulty", "unknown"),
                    "baseline_response": baseline["response"][:400],
                    "intervention_response": intervention["response"][:400],
                    "baseline_probability": round(float(baseline["contrast_probability"]), 6),
                    "intervention_probability": round(float(intervention["contrast_probability"]), 6),
                }
            )

        return {
            "name": motif.get("name"),
            "kind": motif.get("kind"),
            "sign": sign,
            "members": motif_members(motif),
            "validation_backend": "generation_monitor",
            "samples_evaluated": len(self.prompts),
            "prompt_source": self.prompt_source,
            "primary_label": primary_label,
            "opposite_label": opposite_label,
            "baseline_primary_mean": round(baseline_primary_mean, 6),
            "ablated_primary_mean": round(ablated_primary_mean, 6),
            "ablation_monitor_shift": round(ablation_shift, 6),
            "aligned_ablation_shift": round(aligned_ablation_shift, 6),
            "baseline_opposite_mean": round(baseline_opposite_mean, 6),
            "amplified_opposite_mean": round(amplified_opposite_mean, 6),
            "amplification_monitor_shift": round(amplification_shift, 6),
            "aligned_amplification_shift": round(aligned_amplification_shift, 6),
            "response_change_rate_primary": round(response_change_rate_primary, 6),
            "response_change_rate_opposite": round(response_change_rate_opposite, 6),
            "supports_causal_effect": bool(
                max(aligned_ablation_shift, aligned_amplification_shift) >= self.min_effect
            ),
            "examples": examples,
        }
