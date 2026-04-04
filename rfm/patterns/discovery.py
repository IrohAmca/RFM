from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rfm.patterns.data import _records_from_metadata, SequenceRecord, aggregate_sequence_activations, validate_layer_alignment
from rfm.patterns.spec import ContrastAxisSpec


def _bh_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running = 1.0
    for rank, (index, p_value) in enumerate(reversed(indexed), start=1):
        corrected = min(running, p_value * n / max(n - rank + 1, 1))
        running = corrected
        adjusted[index] = float(min(max(corrected, 0.0), 1.0))
    return adjusted


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    change_idx = np.where(np.diff(y_score))[0]
    idx = np.r_[change_idx, y_true.size - 1]
    tpr = np.r_[0.0, tps[idx] / positives, 1.0]
    fpr = np.r_[0.0, fps[idx] / negatives, 1.0]
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(tpr, fpr))


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


class PatternDiscoveryAnalyzer:
    def __init__(self, sae_models: dict[str, torch.nn.Module], *, axis_spec: ContrastAxisSpec, device: str = "cuda"):
        self.axis = axis_spec
        self.sae_models = sae_models
        self.device = device
        for model in sae_models.values():
            model.to(device)
            model.eval()

    def _encode_layer(self, sae, activations: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
        all_features = []
        with torch.no_grad():
            for start in range(0, activations.shape[0], batch_size):
                batch = activations[start: start + batch_size].to(self.device)
                _, feats = sae(batch)
                all_features.append(feats.detach().cpu())
        return torch.cat(all_features, dim=0)

    def _load_and_encode_layer(self, chunk_dir: str | Path, target: str) -> dict[str, Any]:
        files = sorted(Path(chunk_dir).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No activation chunks found in {chunk_dir}")

        token_features = []
        token_lengths: list[int] = []
        records: list[SequenceRecord] = []
        seq_offset = 0
        sae = self.sae_models[target]
        for chunk_id, path in enumerate(files):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            chunk_records = _records_from_metadata(metadata, chunk_id=metadata.get("chunk_id", chunk_id), sequence_offset=seq_offset)
            feats = self._encode_layer(sae, payload["activations"].float())
            token_features.append(feats)
            token_lengths.extend([record.token_length for record in chunk_records])
            records.extend(chunk_records)
            seq_offset += len(chunk_records)

        return {
            "token_features": torch.cat(token_features, dim=0),
            "token_lengths": token_lengths,
            "records": records,
        }

    def _cv_splits(self, groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            full = np.arange(groups.shape[0], dtype=np.int64)
            return [(full, full)]
        folds = max(2, min(int(n_splits), int(unique_groups.size)))
        try:
            from sklearn.model_selection import GroupKFold

            splitter = GroupKFold(n_splits=folds)
            indices = np.arange(groups.shape[0], dtype=np.int64)
            return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(indices, groups=groups)]
        except Exception:
            order = np.argsort(groups, kind="mergesort")
            chunks = np.array_split(order, folds)
            splits = []
            for test_idx in chunks:
                mask = np.ones(groups.shape[0], dtype=bool)
                mask[test_idx] = False
                train_idx = np.where(mask)[0]
                splits.append((train_idx, test_idx))
            return splits

    def _feature_threshold(self, values: torch.Tensor) -> float:
        nonzero = values[values > 0]
        if nonzero.numel() == 0:
            return 0.0
        return float(torch.quantile(nonzero, 0.95).item())

    def _feature_score_rows(self, seq_features: torch.Tensor, y: np.ndarray) -> list[dict[str, Any]]:
        endpoint_a_mask = torch.from_numpy((y == 0).astype(np.bool_))
        endpoint_b_mask = torch.from_numpy((y == 1).astype(np.bool_))
        a_values = seq_features[endpoint_a_mask]
        b_values = seq_features[endpoint_b_mask]
        rows = []
        p_values = []
        mw = None
        try:
            from scipy.stats import mannwhitneyu as _mw

            mw = _mw
        except Exception:
            mw = None

        for feature_id in range(seq_features.shape[1]):
            a_col = a_values[:, feature_id]
            b_col = b_values[:, feature_id]
            mean_a = float(a_col.mean().item())
            mean_b = float(b_col.mean().item())
            delta = mean_b - mean_a
            var_a = float(a_col.var(unbiased=False).item())
            var_b = float(b_col.var(unbiased=False).item())
            pooled_std = math.sqrt(
                max(
                    (
                        max(a_col.shape[0] - 1, 0) * var_a
                        + max(b_col.shape[0] - 1, 0) * var_b
                    )
                    / max(a_col.shape[0] + b_col.shape[0] - 2, 1),
                    1e-12,
                )
            )
            effect_size = delta / max(pooled_std, 1e-8)
            threshold = self._feature_threshold(seq_features[:, feature_id])
            if threshold > 0:
                rate_a = float((a_col >= threshold).float().mean().item())
                rate_b = float((b_col >= threshold).float().mean().item())
            else:
                rate_a = float((a_col > 0).float().mean().item())
                rate_b = float((b_col > 0).float().mean().item())
            interaction_candidate_score = float(seq_features[:, feature_id].var(unbiased=False).item()) * min(rate_a, rate_b)
            p_value = 1.0
            if mw is not None:
                try:
                    _, p_value = mw(a_col.numpy(), b_col.numpy(), alternative="two-sided")
                except Exception:
                    p_value = 1.0
            p_values.append(float(p_value))
            rows.append(
                {
                    "feature_id": int(feature_id),
                    "endpoint_a_mean": round(mean_a, 6),
                    "endpoint_b_mean": round(mean_b, 6),
                    "delta": round(delta, 6),
                    "effect_size": round(effect_size, 6),
                    "activation_rate_a": round(rate_a, 6),
                    "activation_rate_b": round(rate_b, 6),
                    "threshold": round(threshold, 6),
                    "interaction_candidate_score": round(interaction_candidate_score, 6),
                    "p_value": float(p_value),
                }
            )

        q_values = _bh_adjust(p_values)
        for row, q_value in zip(rows, q_values):
            row["q_value"] = float(q_value)
        rows.sort(key=lambda item: abs(float(item["effect_size"])), reverse=True)
        return rows

    def _candidate_pool(
        self,
        score_rows: list[dict[str, Any]],
        *,
        top_endpoint_a: int,
        top_endpoint_b: int,
        top_interaction: int,
    ) -> dict[str, list[int]]:
        endpoint_b = [row["feature_id"] for row in score_rows if float(row["delta"]) > 0][:top_endpoint_b]
        endpoint_a = [row["feature_id"] for row in score_rows if float(row["delta"]) < 0][:top_endpoint_a]
        chosen = set(endpoint_a) | set(endpoint_b)
        interaction = [
            row["feature_id"]
            for row in sorted(score_rows, key=lambda item: float(item.get("interaction_candidate_score", 0.0)), reverse=True)
            if row["feature_id"] not in chosen
        ][:top_interaction]
        combined = list(dict.fromkeys(endpoint_a + endpoint_b + interaction))
        return {
            "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b,
            "interaction": interaction,
            "combined": combined,
        }

    def _fit_classifier(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, *, sparse: bool) -> dict[str, Any]:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except Exception:
            weight = x_train[y_train == 1].mean(axis=0) - x_train[y_train == 0].mean(axis=0)
            bias = -float(np.dot(weight, 0.5 * (x_train[y_train == 1].mean(axis=0) + x_train[y_train == 0].mean(axis=0))))
            test_score = x_test @ weight + bias
            y_pred = (test_score >= 0).astype(np.int64)
            return {
                "scaler": None,
                "clf": None,
                "coefficients": weight,
                "intercept": bias,
                "f1": _binary_f1(y_test, y_pred),
                "auc": _roc_auc(y_test, test_score),
                "scores": test_score,
            }

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        clf = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            penalty="l1" if sparse else "l2",
            solver="liblinear",
            random_state=42,
        )
        clf.fit(x_train_scaled, y_train)
        y_score = clf.predict_proba(x_test_scaled)[:, 1]
        y_pred = (y_score >= 0.5).astype(np.int64)
        return {
            "scaler": scaler,
            "clf": clf,
            "coefficients": clf.coef_[0].copy(),
            "intercept": float(clf.intercept_[0]),
            "f1": _binary_f1(y_test, y_pred),
            "auc": _roc_auc(y_test, y_score),
            "scores": y_score,
        }

    def _build_single_matrix(
        self,
        seq_by_layer: dict[str, torch.Tensor],
        feature_pools: dict[str, dict[str, list[int]]],
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        matrices = []
        names = []
        specs = []
        for layer_name, seq_features in seq_by_layer.items():
            selected = feature_pools[layer_name]["combined"]
            if not selected:
                continue
            matrices.append(seq_features[:, selected].numpy())
            for feature_id in selected:
                feature_name = f"{layer_name}::F{feature_id}"
                names.append(feature_name)
                specs.append({"kind": "single", "layer": layer_name, "feature_id": int(feature_id), "name": feature_name})
        if not matrices:
            n_rows = next(iter(seq_by_layer.values())).shape[0]
            return np.zeros((n_rows, 0), dtype=np.float32), [], []
        return np.concatenate(matrices, axis=1), names, specs

    def _build_interaction_matrix(
        self,
        seq_by_layer: dict[str, torch.Tensor],
        feature_pools: dict[str, dict[str, list[int]]],
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        layers = list(seq_by_layer)
        columns = []
        names = []
        specs = []
        for index, layer_a in enumerate(layers):
            ids_a = feature_pools[layer_a]["combined"]
            if not ids_a:
                continue
            for layer_b in layers[index + 1:]:
                ids_b = feature_pools[layer_b]["combined"]
                if not ids_b:
                    continue
                for feature_a in ids_a:
                    col_a = seq_by_layer[layer_a][:, feature_a].numpy()
                    for feature_b in ids_b:
                        col_b = seq_by_layer[layer_b][:, feature_b].numpy()
                        names.append(f"{layer_a}::F{feature_a} x {layer_b}::F{feature_b}")
                        specs.append(
                            {
                                "kind": "interaction",
                                "members": (
                                    {"layer": layer_a, "feature_id": int(feature_a)},
                                    {"layer": layer_b, "feature_id": int(feature_b)},
                                ),
                                "name": names[-1],
                            }
                        )
                        columns.append((col_a * col_b).reshape(-1, 1))
        if not columns:
            n_rows = next(iter(seq_by_layer.values())).shape[0]
            return np.zeros((n_rows, 0), dtype=np.float32), [], []
        return np.concatenate(columns, axis=1), names, specs

    def _benchmark_method(self, seq_by_layer: dict[str, torch.Tensor], y: np.ndarray, groups: np.ndarray, cv_folds: int) -> dict[str, Any]:
        folds = self._cv_splits(groups, cv_folds)
        f1_scores = []
        auc_scores = []
        for train_idx, test_idx in folds:
            feature_pools = {}
            for layer_name, seq_features in seq_by_layer.items():
                score_rows = self._feature_score_rows(seq_features[train_idx], y[train_idx])
                feature_pools[layer_name] = self._candidate_pool(score_rows, top_endpoint_a=4, top_endpoint_b=4, top_interaction=2)
            x_train, _, _ = self._build_single_matrix({layer: values[train_idx] for layer, values in seq_by_layer.items()}, feature_pools)
            x_test, _, _ = self._build_single_matrix({layer: values[test_idx] for layer, values in seq_by_layer.items()}, feature_pools)
            result = self._fit_classifier(x_train, y[train_idx], x_test, y[test_idx], sparse=False)
            f1_scores.append(result["f1"])
            auc_scores.append(result["auc"])
        return {
            "cv_f1_mean": round(_metric_summary(f1_scores)["mean"], 6),
            "cv_f1_std": round(_metric_summary(f1_scores)["std"], 6),
            "cv_auc_mean": round(_metric_summary(auc_scores)["mean"], 6),
            "cv_auc_std": round(_metric_summary(auc_scores)["std"], 6),
        }

    def _coactivation_stats(
        self,
        seq_by_layer: dict[str, torch.Tensor],
        thresholds_by_layer: dict[str, dict[int, float]],
        interactions: list[dict[str, Any]],
        y: np.ndarray,
    ) -> list[dict[str, Any]]:
        positive_mask = y == 1
        negative_mask = y == 0
        rows = []
        fisher_exact = None
        try:
            from scipy.stats import fisher_exact as _fisher_exact

            fisher_exact = _fisher_exact
        except Exception:
            fisher_exact = None

        p_values = []
        for item in interactions:
            members = item.get("members", [])
            if len(members) != 2:
                continue
            left, right = members
            left_values = seq_by_layer[left["layer"]][:, left["feature_id"]]
            right_values = seq_by_layer[right["layer"]][:, right["feature_id"]]
            left_threshold = thresholds_by_layer[left["layer"]][left["feature_id"]]
            right_threshold = thresholds_by_layer[right["layer"]][right["feature_id"]]
            both = ((left_values >= left_threshold) & (right_values >= right_threshold)).numpy()
            rate_positive = float(both[positive_mask].mean()) if positive_mask.any() else 0.0
            rate_negative = float(both[negative_mask].mean()) if negative_mask.any() else 0.0
            risk_difference = rate_positive - rate_negative
            pos_count = int(both[positive_mask].sum())
            neg_count = int(both[negative_mask].sum())
            pos_rest = int(positive_mask.sum()) - pos_count
            neg_rest = int(negative_mask.sum()) - neg_count
            odds_ratio = ((pos_count + 0.5) * (neg_rest + 0.5)) / max((pos_rest + 0.5) * (neg_count + 0.5), 1e-8)
            log_odds = float(math.log(max(odds_ratio, 1e-8)))
            p_value = 1.0
            if fisher_exact is not None:
                try:
                    _, p_value = fisher_exact([[pos_count, pos_rest], [neg_count, neg_rest]])
                except Exception:
                    p_value = 1.0
            p_values.append(float(p_value))
            rows.append(
                {
                    "name": item["name"],
                    "endpoint_b_rate": round(rate_positive, 6),
                    "endpoint_a_rate": round(rate_negative, 6),
                    "risk_difference": round(risk_difference, 6),
                    "log_odds_ratio": round(log_odds, 6),
                    "p_value": float(p_value),
                }
            )
        q_values = _bh_adjust(p_values)
        for row, q_value in zip(rows, q_values):
            row["q_value"] = float(q_value)
        rows.sort(key=lambda row: abs(float(row["log_odds_ratio"])), reverse=True)
        return rows

    def _decision_tree_report(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str],
        cv_folds: int,
        stability_min_fraction: float,
        max_depth: int,
    ) -> dict[str, Any]:
        try:
            from sklearn.tree import DecisionTreeClassifier, export_text
        except Exception:
            return {"error": "scikit-learn required for decision tree reporting"}

        folds = self._cv_splits(groups, cv_folds)
        rule_counts = defaultdict(int)
        rules_by_fold = []
        f1_scores = []
        for train_idx, test_idx in folds:
            clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=42)
            clf.fit(x[train_idx], y[train_idx])
            y_pred = clf.predict(x[test_idx])
            f1_scores.append(_binary_f1(y[test_idx], y_pred))
            rules_text = export_text(clf, feature_names=feature_names, max_depth=max_depth)
            rules_by_fold.append(rules_text)
            for line in rules_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("|---") and "class:" not in stripped:
                    rule_counts[stripped] += 1

        stable_rules = []
        hypothesis_rules = []
        min_count = max(1, math.ceil(len(folds) * stability_min_fraction))
        for rule, count in sorted(rule_counts.items(), key=lambda item: (-item[1], item[0])):
            payload = {
                "rule": rule,
                "count": int(count),
                "stability": round(count / max(len(folds), 1), 6),
            }
            if count >= min_count:
                stable_rules.append(payload)
            else:
                hypothesis_rules.append(payload)

        return {
            "cv_f1_mean": round(_metric_summary(f1_scores)["mean"], 6),
            "cv_f1_std": round(_metric_summary(f1_scores)["std"], 6),
            "stable_rules": stable_rules,
            "hypothesis_rules": hypothesis_rules,
            "rules_by_fold": rules_by_fold,
        }

    def _intervention_effects(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model: dict[str, Any],
        feature_specs: list[dict[str, Any]],
        promoted: list[dict[str, Any]],
        *,
        min_shift: float,
    ) -> list[dict[str, Any]]:
        clf = model.get("clf")
        scaler = model.get("scaler")
        coefficients = np.asarray(model.get("coefficients", np.zeros(x.shape[1], dtype=np.float32)))
        if clf is None or scaler is None:
            return []

        baseline_scores = clf.predict_proba(scaler.transform(x))[:, 1]
        baseline_pred = (baseline_scores >= 0.5).astype(np.int64)
        baseline_f1 = _binary_f1(y, baseline_pred)
        spec_lookup = {spec["name"]: (index, spec) for index, spec in enumerate(feature_specs)}
        effects = []
        for item in promoted:
            name = item["name"]
            if name not in spec_lookup:
                continue
            index, spec = spec_lookup[name]
            x_ablated = x.copy()
            x_amplified = x.copy()
            x_ablated[:, index] = 0.0
            x_amplified[:, index] = x_amplified[:, index] * 1.5
            ablated_scores = clf.predict_proba(scaler.transform(x_ablated))[:, 1]
            amplified_scores = clf.predict_proba(scaler.transform(x_amplified))[:, 1]
            ablated_pred = (ablated_scores >= 0.5).astype(np.int64)
            amplified_pred = (amplified_scores >= 0.5).astype(np.int64)
            endpoint_b_shift = float(ablated_scores[y == 1].mean() - baseline_scores[y == 1].mean()) if np.any(y == 1) else 0.0
            endpoint_a_shift = float(ablated_scores[y == 0].mean() - baseline_scores[y == 0].mean()) if np.any(y == 0) else 0.0
            effects.append(
                {
                    "name": name,
                    "kind": spec["kind"],
                    "validation_backend": "classifier_proxy",
                    "ablation_endpoint_b_shift": round(endpoint_b_shift, 6),
                    "ablation_endpoint_a_shift": round(endpoint_a_shift, 6),
                    "ablation_f1_delta": round(_binary_f1(y, ablated_pred) - baseline_f1, 6),
                    "amplification_f1_delta": round(_binary_f1(y, amplified_pred) - baseline_f1, 6),
                    "supports_causal_effect": bool(abs(endpoint_b_shift) >= min_shift),
                    "supports_causal_effect_proxy": bool(abs(endpoint_b_shift) >= min_shift),
                    "coefficient": round(float(coefficients[index]), 6) if index < coefficients.shape[0] else 0.0,
                }
            )
        return effects

    def analyze(
        self,
        chunk_dirs: dict[str, str | Path],
        *,
        aggregation_candidates: list[str] | None = None,
        cv_folds: int = 5,
        top_endpoint_a: int = 12,
        top_endpoint_b: int = 12,
        top_interaction: int = 8,
        stability_min_fraction: float = 0.6,
        max_tree_depth: int = 5,
        min_interaction_gain: float = 0.005,
        intervention_min_shift: float = 0.01,
    ) -> dict[str, Any]:
        layers = list(chunk_dirs)
        if not layers:
            raise ValueError("Pattern discovery requires at least one layer with activations.")

        encoded_layers = {layer: self._load_and_encode_layer(chunk_dirs[layer], layer) for layer in layers}
        alignment_report = validate_layer_alignment(
            {layer: payload["records"] for layer, payload in encoded_layers.items()},
            self.axis,
        )
        records = encoded_layers[layers[0]]["records"]
        y = np.array([self.axis.label_lookup.get(record.label, -1) for record in records], dtype=np.int64)
        valid_mask = y >= 0
        if not valid_mask.all():
            y = y[valid_mask]
            for layer in layers:
                encoded_layers[layer]["records"] = [record for keep, record in zip(valid_mask, encoded_layers[layer]["records"]) if keep]
                encoded_layers[layer]["token_lengths"] = [record.token_length for record in encoded_layers[layer]["records"]]
            records = encoded_layers[layers[0]]["records"]
        groups = np.array([record.pair_id for record in records], dtype=np.int64)

        aggregation_candidates = aggregation_candidates or ["mean", "topk_mean_4", "lastk_mean_8", "max"]
        seq_cache: dict[str, dict[str, torch.Tensor]] = {}
        benchmarks = []
        for method in aggregation_candidates:
            seq_by_layer = {
                layer: aggregate_sequence_activations(
                    encoded_layers[layer]["token_features"],
                    encoded_layers[layer]["token_lengths"],
                    method=method,
                )
                for layer in layers
            }
            if not valid_mask.all():
                keep_mask = torch.from_numpy(valid_mask.astype(np.bool_))
                seq_by_layer = {
                    layer: values[keep_mask]
                    for layer, values in seq_by_layer.items()
                }
            benchmark = self._benchmark_method(seq_by_layer, y, groups, cv_folds=cv_folds)
            benchmark["method"] = method
            benchmarks.append(benchmark)
            seq_cache[method] = seq_by_layer

        def _method_order(item: dict[str, Any]) -> tuple[float, float, int]:
            preference = {"mean": 0, "topk_mean_4": 1, "lastk_mean_8": 2, "max": 3}
            return (
                float(item.get("cv_f1_mean", 0.0)),
                float(item.get("cv_auc_mean", 0.0)),
                -preference.get(str(item.get("method")), 100),
            )

        best = max(benchmarks, key=_method_order)
        selected_method = str(best["method"])
        seq_by_layer = seq_cache[selected_method]

        layer_feature_scores = {layer: self._feature_score_rows(seq_by_layer[layer], y) for layer in layers}
        thresholds_by_layer = {
            layer: {row["feature_id"]: float(row["threshold"]) for row in layer_feature_scores[layer]}
            for layer in layers
        }
        feature_pools = {
            layer: self._candidate_pool(
                layer_feature_scores[layer],
                top_endpoint_a=top_endpoint_a,
                top_endpoint_b=top_endpoint_b,
                top_interaction=top_interaction,
            )
            for layer in layers
        }

        x_single, single_names, single_specs = self._build_single_matrix(seq_by_layer, feature_pools)
        x_interactions, interaction_names, interaction_specs = self._build_interaction_matrix(seq_by_layer, feature_pools)
        x_full = np.concatenate([x_single, x_interactions], axis=1) if x_interactions.size else x_single
        full_names = single_names + interaction_names
        full_specs = single_specs + interaction_specs

        folds = self._cv_splits(groups, cv_folds)
        base_f1 = []
        base_auc = []
        full_f1 = []
        full_auc = []
        coefficient_counts = defaultdict(int)
        coefficient_signs = defaultdict(list)
        coefficient_magnitude = defaultdict(list)
        for train_idx, test_idx in folds:
            base_result = self._fit_classifier(x_single[train_idx], y[train_idx], x_single[test_idx], y[test_idx], sparse=False)
            full_result = self._fit_classifier(x_full[train_idx], y[train_idx], x_full[test_idx], y[test_idx], sparse=True)
            base_f1.append(base_result["f1"])
            base_auc.append(base_result["auc"])
            full_f1.append(full_result["f1"])
            full_auc.append(full_result["auc"])
            coefficients = np.asarray(full_result["coefficients"])
            for name, value in zip(full_names, coefficients.tolist()):
                if abs(value) <= 1e-8:
                    continue
                coefficient_counts[name] += 1
                coefficient_signs[name].append(1 if value >= 0 else -1)
                coefficient_magnitude[name].append(abs(value))

        required_count = max(1, math.ceil(len(folds) * stability_min_fraction))
        promoted = []
        for spec in full_specs:
            name = spec["name"]
            count = coefficient_counts.get(name, 0)
            if count == 0:
                continue
            signs = coefficient_signs[name]
            sign_consistency = abs(sum(signs)) == len(signs)
            status = "stable" if count >= required_count and sign_consistency else "hypothesis"
            promoted.append(
                {
                    **spec,
                    "stability": round(count / max(len(folds), 1), 6),
                    "count": int(count),
                    "mean_abs_coefficient": round(float(np.mean(coefficient_magnitude[name])), 6),
                    "sign": "endpoint_b" if sum(signs) >= 0 else "endpoint_a",
                    "status": status,
                }
            )

        promoted.sort(key=lambda item: (item["status"] != "stable", -item["stability"], -item["mean_abs_coefficient"], item["name"]))
        model_metrics = {
            "base": {
                "cv_f1_mean": round(_metric_summary(base_f1)["mean"], 6),
                "cv_f1_std": round(_metric_summary(base_f1)["std"], 6),
                "cv_auc_mean": round(_metric_summary(base_auc)["mean"], 6),
                "cv_auc_std": round(_metric_summary(base_auc)["std"], 6),
            },
            "full": {
                "cv_f1_mean": round(_metric_summary(full_f1)["mean"], 6),
                "cv_f1_std": round(_metric_summary(full_f1)["std"], 6),
                "cv_auc_mean": round(_metric_summary(full_auc)["mean"], 6),
                "cv_auc_std": round(_metric_summary(full_auc)["std"], 6),
            },
            "interaction_gain_f1": round(_metric_summary(full_f1)["mean"] - _metric_summary(base_f1)["mean"], 6),
        }

        full_model = self._fit_classifier(x_full, y, x_full, y, sparse=True)
        coefficients = np.asarray(full_model["coefficients"])
        feature_importance = []
        for spec, coef in sorted(zip(full_specs, coefficients.tolist()), key=lambda item: abs(item[1]), reverse=True):
            feature_importance.append(
                {
                    "name": spec["name"],
                    "kind": spec["kind"],
                    "importance": round(abs(float(coef)), 6),
                    "sign": "endpoint_b" if coef >= 0 else "endpoint_a",
                }
            )

        stable_interactions = [item for item in promoted if item["kind"] == "interaction" and item["status"] == "stable"]
        if model_metrics["interaction_gain_f1"] < min_interaction_gain:
            for item in stable_interactions:
                item["status"] = "hypothesis"
        motif_candidates = [dict(item) for item in promoted]

        intervention_effects = self._intervention_effects(
            x_full,
            y,
            full_model,
            full_specs,
            [item for item in promoted if item["status"] == "stable"],
            min_shift=intervention_min_shift,
        )
        stable_promoted = [item for item in promoted if item["status"] == "stable"]
        stable_interactions = [item for item in stable_promoted if item["kind"] == "interaction"]
        coactivation = self._coactivation_stats(seq_by_layer, thresholds_by_layer, stable_interactions, y)
        decision_tree = self._decision_tree_report(
            x_full,
            y,
            groups,
            full_names,
            cv_folds=cv_folds,
            stability_min_fraction=stability_min_fraction,
            max_depth=max_tree_depth,
        )

        return {
            "alignment_report": alignment_report,
            "aggregation_benchmark": benchmarks,
            "selected_aggregation": selected_method,
            "layer_feature_scores": layer_feature_scores,
            "feature_pools": feature_pools,
            "thresholds": thresholds_by_layer,
            "model_metrics": model_metrics,
            "motif_candidates": motif_candidates,
            "stable_motifs": stable_promoted,
            "stable_interactions": stable_interactions,
            "feature_importance": feature_importance[:100],
            "decision_tree": decision_tree,
            "intervention_effects": intervention_effects,
            "coactivation_stats": coactivation,
            "causal_validation": {
                "status": "proxy_only",
                "reason": "classifier_space_validation",
                "evaluated": len(intervention_effects),
                "supported": sum(1 for item in intervention_effects if item.get("supports_causal_effect")),
            },
            "feature_specs": full_specs,
        }
