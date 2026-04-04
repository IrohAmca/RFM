"""LLM-based interpretation of top deception SAE features."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger("rfm.deception.deception_autointerp")


SYSTEM_PROMPT = """You are an AI interpretability researcher analyzing Sparse Autoencoder (SAE) features from a causal language model trained to detect deception.

You will be shown examples from a deception-vs-honest contrast experiment. Each example contains a question paired with an honest response and a deceptive response. The SAE feature you are analyzing activates more strongly during the DECEPTIVE response than the honest one.

Your task: Write a concise 1–2 sentence interpretation of what this specific SAE feature encodes or detects. Focus on:
- What linguistic, semantic, or rhetorical pattern the feature captures
- What specifically makes the deceptive examples different from honest ones in a way this feature would detect
- Be concrete and specific — avoid vague phrases like "detects deception in general"

Respond with ONLY the interpretation, no preamble or labels."""


class DeceptionFeatureAutoInterp:
    """Find top-activating deceptive contexts and interpret SAE features via LLM."""

    def __init__(
        self,
        sae_model,
        chunk_dir: str | Path,
        scenarios_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.sae = sae_model
        self.sae.to(device)
        self.sae.eval()
        self.chunk_dir = Path(chunk_dir)
        self.device = device
        self._question_to_scenario: dict[str, dict] = {}
        if scenarios_path:
            self._load_scenarios(Path(scenarios_path))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_scenarios(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Scenarios file not found: %s", path)
            return
        scenarios = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        scenarios.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._question_to_scenario = {
            s.get("question", ""): s for s in scenarios if "question" in s
        }
        logger.info("Loaded %d scenarios from %s", len(self._question_to_scenario), path)

    def _encode_chunks(self, batch_size: int = 2048) -> tuple[torch.Tensor, list[dict]]:
        """Encode all activation chunks through the SAE.

        Returns
        -------
        seq_features : Tensor [N_seqs, hidden_dim]  — mean feature activation per sequence
        seq_records  : list[dict]  — one record per sequence with label/pair_id/question
        """
        files = sorted(self.chunk_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt activation files found in {self.chunk_dir}")

        all_feature_means: list[torch.Tensor] = []
        all_records: list[dict] = []

        for file_path in files:
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
            activations = payload["activations"].float()  # [N_tokens, d_model]
            metadata = payload.get("metadata", {})
            labels = list(metadata.get("labels", []))
            token_lengths = [int(x) for x in metadata.get("token_lengths", [])]
            pair_ids = list(metadata.get("pair_ids", []))
            questions = list(metadata.get("questions", []))

            # Encode tokens through SAE in mini-batches
            all_tok_features: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, activations.shape[0], batch_size):
                    batch = activations[start: start + batch_size].to(self.device)
                    _, feats = self.sae(batch)
                    all_tok_features.append(feats.cpu())
            tok_features = torch.cat(all_tok_features, dim=0)  # [N_tokens, hidden_dim]

            # Aggregate per sequence (mean over tokens belonging to that sequence)
            offset = 0
            for i, length in enumerate(token_lengths):
                seg = tok_features[offset: offset + length]
                offset += length
                if seg.numel() == 0:
                    continue
                all_feature_means.append(seg.mean(dim=0))
                all_records.append(
                    {
                        "label": labels[i] if i < len(labels) else "unknown",
                        "pair_id": pair_ids[i] if i < len(pair_ids) else -1,
                        "question": questions[i] if i < len(questions) else "",
                    }
                )

        if not all_feature_means:
            raise ValueError("No sequences could be encoded — check chunk contents.")

        seq_features = torch.stack(all_feature_means, dim=0)  # [N_seqs, hidden_dim]
        logger.info("Encoded %d sequences, feature dim=%d", seq_features.shape[0], seq_features.shape[1])
        return seq_features, all_records

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def find_top_contexts(
        self,
        feature_ids: list[int],
        top_k: int = 8,
    ) -> dict[int, list[dict]]:
        """For each feature, find the top-k DECEPTIVE sequences where it fires most.

        Parameters
        ----------
        feature_ids : list of SAE feature indices to analyse
        top_k       : how many top examples to retrieve per feature

        Returns
        -------
        dict mapping feature_id → list of context dicts (sorted by activation desc)
        """
        seq_features, seq_records = self._encode_chunks()

        deceptive_idx = [
            i for i, r in enumerate(seq_records) if r["label"] == "deceptive"
        ]
        if not deceptive_idx:
            logger.warning("No deceptive sequences found; using all sequences.")
            deceptive_idx = list(range(len(seq_records)))

        results: dict[int, list[dict]] = {}
        for fid in feature_ids:
            acts = seq_features[:, fid]
            deceptive_acts = acts[deceptive_idx]
            k = min(top_k, len(deceptive_idx))
            top_local = torch.topk(deceptive_acts, k=k).indices.tolist()
            top_global = [deceptive_idx[i] for i in top_local]

            contexts = []
            for idx in top_global:
                rec = seq_records[idx]
                q = rec["question"]
                scenario = self._question_to_scenario.get(q, {})
                contexts.append(
                    {
                        "question": q,
                        "honest_answer": scenario.get("honest_answer", ""),
                        "deceptive_answer": scenario.get("deceptive_answer", ""),
                        "category": scenario.get("category", "unknown"),
                        "difficulty": scenario.get("difficulty", "unknown"),
                        "activation": float(acts[idx].item()),
                    }
                )
            results[fid] = contexts

        return results

    def interpret_features(
        self,
        feature_contexts: dict[int, list[dict]],
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        base_url: str = "https://api.groq.com/openai/v1",
        request_delay: float = 2.5,
        existing_results: dict[int, str] | None = None,
    ) -> dict[int, str]:
        """Call LLM to interpret each feature from its activating examples.

        Parameters
        ----------
        feature_contexts  : output of find_top_contexts
        api_key           : LLM provider API key
        model             : model identifier
        base_url          : OpenAI-compatible base URL
        request_delay     : seconds between requests (rate-limit guard)
        existing_results  : previously computed interpretations to skip

        Returns
        -------
        dict mapping feature_id → interpretation string
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(api_key=api_key, base_url=base_url)
        results: dict[int, str] = dict(existing_results or {})

        for fid, contexts in feature_contexts.items():
            if fid in results:
                logger.debug("Feature %d already interpreted — skipping.", fid)
                continue
            if not contexts:
                results[fid] = "[No contexts available]"
                continue

            # Build user message with up to 6 examples
            examples_text = []
            for i, ctx in enumerate(contexts[:6], 1):
                examples_text.append(
                    f"Example {i} [category={ctx['category']}, activation={ctx['activation']:.3f}]\n"
                    f"  Q:         {ctx['question'][:250]}\n"
                    f"  Honest:    {ctx['honest_answer'][:250]}\n"
                    f"  Deceptive: {ctx['deceptive_answer'][:250]}"
                )

            user_content = (
                f"Feature ID: {fid}\n\n"
                + "\n\n".join(examples_text)
                + "\n\nWhat does this SAE feature encode?"
            )

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                )
                results[fid] = resp.choices[0].message.content.strip()
                logger.info("✓ Feature %d interpreted", fid)
            except Exception as exc:
                results[fid] = f"[Error: {exc}]"
                logger.warning("Feature %d interpretation failed: %s", fid, exc)

            if request_delay > 0:
                time.sleep(request_delay)

        return results

    def interpret_features_locally(
        self,
        feature_contexts: dict[int, list[dict]],
        *,
        model,
        tokenizer,
        max_new_tokens: int = 96,
        request_delay: float = 0.0,
        existing_results: dict[int, str] | None = None,
    ) -> dict[int, str]:
        """Interpret features using a locally loaded causal LM."""

        def _user_content(feature_id: int, contexts: list[dict]) -> str:
            examples_text = []
            for i, ctx in enumerate(contexts[:6], 1):
                examples_text.append(
                    f"Example {i} [category={ctx['category']}, activation={ctx['activation']:.3f}]\n"
                    f"  Q:         {ctx['question'][:250]}\n"
                    f"  Honest:    {ctx['honest_answer'][:250]}\n"
                    f"  Deceptive: {ctx['deceptive_answer'][:250]}"
                )
            return (
                f"Feature ID: {feature_id}\n\n"
                + "\n\n".join(examples_text)
                + "\n\nWhat does this SAE feature encode?"
            )

        def _prompt_text(user_content: str) -> str:
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            if callable(apply_chat_template):
                try:
                    return apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except Exception:
                    pass
            return f"System: {SYSTEM_PROMPT}\nUser: {user_content}\nAssistant:"

        device = next(model.parameters()).device
        results: dict[int, str] = dict(existing_results or {})

        for fid, contexts in feature_contexts.items():
            if fid in results:
                logger.debug("Feature %d already interpreted locally; skipping.", fid)
                continue
            if not contexts:
                results[fid] = "[No contexts available]"
                continue

            prompt_text = _prompt_text(_user_content(fid, contexts))
            encoded = tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max(int(max_new_tokens), 16),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            response_ids = output_ids[:, input_ids.shape[1]:]
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
            if "</think>" in response:
                response = response.split("</think>", 1)[1].strip()
            elif response.startswith("<think>"):
                response = ""
            cleaned = " ".join(part.strip() for part in response.splitlines() if part.strip()).strip()
            results[fid] = cleaned or "[No interpretation returned]"
            logger.info("✓ Feature %d interpreted locally", fid)

            if request_delay > 0:
                time.sleep(request_delay)

        return results
