"""Think / Retrieve / Revise agent loop built on top of the generic agent framework.

This loop reuses the existing sampling infrastructure to execute a structured three
stage interaction:

1. Ask the policy to emit a `<think>` block with intermediate claims.
2. Retrieve contextual memories conditioned on the thinking step and provide them
   back to the model before requesting a `<revise>` block.
3. Finally, request an `<actions>` block that forecasts concrete next actions.

Memories are backed by the in-memory `InMemoryBM25Temporal` retriever introduced
for the GRPO trainer so we do not rely on external services during rollout.
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import time
from typing import Any, Iterable, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (AgentLoopBase,
                                                     AgentLoopMetrics,
                                                     AgentLoopOutput, register)
from verl.retrievers import InMemoryBM25Temporal
from verl.trainer.utils_traces import jaccard_ngrams
from verl.utils.profiler import simple_timer


def _ensure_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _extract_tag_contents(text: str, tag: str) -> list[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return [match.strip() for match in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


@register("think_retrieve_revise_agent")
class ThinkRetrieveReviseAgentLoop(AgentLoopBase):
    """Agent loop that realises the think / retrieve / revise / act template."""

    _retriever: InMemoryBM25Temporal | None = None
    _retriever_lock: asyncio.Lock | None = None

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        super().init_class(config, tokenizer, processor, **kwargs)

        rollout_cfg = config.actor_rollout_ref.rollout
        loop_cfg = rollout_cfg.get("think_retrieve_revise", {})

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = rollout_cfg.prompt_length
        cls.response_length = rollout_cfg.response_length

        cls.default_n_claims = loop_cfg.get("default_n_claims", 3)
        cls.default_future_len = loop_cfg.get("default_future_len", 3)

        cls.max_think_tokens = loop_cfg.get("max_think_tokens", 256)
        cls.max_revise_tokens = loop_cfg.get("max_revise_tokens", 256)
        cls.max_actions_tokens = loop_cfg.get("max_actions_tokens", 192)

        cls.stop_think = _ensure_list(loop_cfg.get("stop_think", "</think>"))
        cls.stop_revise = _ensure_list(loop_cfg.get("stop_revise", "</revise>"))
        cls.stop_actions = _ensure_list(loop_cfg.get("stop_actions", "</actions>"))

        cls.retriever_top_k = loop_cfg.get("retriever_top_k", 6)
        cls.time_decay_lambda = loop_cfg.get("time_decay_lambda", 0.0)
        cls.memory_namespace = loop_cfg.get("memory_namespace", "default")
        cls.dedup_threshold = loop_cfg.get("dedup_threshold", 0.85)

        cls._retriever = InMemoryBM25Temporal()
        cls._retriever_lock = asyncio.Lock()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        extra_fields: dict[str, Any] = {}

        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        image_data = (kwargs.get("multi_modal_data") or {}).get("image")

        n_claims = int(kwargs.get("n_claims", self.default_n_claims))
        future_len = int(kwargs.get("future_len", self.default_future_len))
        namespace = kwargs.get("memory_namespace", self.memory_namespace)
        now_ts = int(kwargs.get("now_ts", math.floor(time.time())))

        request_id = uuid4().hex

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        with simple_timer("generate_sequences", metrics):
            # THINK -----------------------------------------------------------------
            think_instruction = self._build_think_instruction(n_claims)
            # Combine original context with think instruction in a single user turn
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + think_instruction
            else:
                messages.append({"role": "user", "content": think_instruction})
            prompt_ids = await self._build_prompt_ids(messages, image_data)
            prompt_ids_initial = prompt_ids  # keep the first-stage prompt as rollout prompt

            think_sampling = self._stage_sampling(
                sampling_params,
                stop=self.stop_think,
                max_tokens=self.max_think_tokens,
                remaining=self.response_length - len(response_ids),
            )
            think_output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=think_sampling,
                image_data=image_data,
            )
            think_tokens = think_output.token_ids
            think_text = self.tokenizer.decode(think_tokens, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": think_text})

            response_ids.extend(think_tokens)
            response_mask.extend([1] * len(think_tokens))
            if think_output.log_probs:
                response_logprobs.extend(think_output.log_probs)

            extra_fields["think_output"] = think_text

            # RETRIEVE --------------------------------------------------------------
            retrieved_context = await self._retrieve_context(namespace, think_text, now_ts)
            extra_fields["retrieved_context"] = retrieved_context

            revise_instruction = self._build_revise_instruction(retrieved_context, n_claims)
            messages.append({"role": "user", "content": revise_instruction})
            prompt_ids = await self._build_prompt_ids(messages, image_data)

            revise_sampling = self._stage_sampling(
                sampling_params,
                stop=self.stop_revise,
                max_tokens=self.max_revise_tokens,
                remaining=self.response_length - len(response_ids),
            )
            revise_output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=revise_sampling,
                image_data=image_data,
            )
            revise_tokens = revise_output.token_ids
            revise_text = self.tokenizer.decode(revise_tokens, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": revise_text})

            response_ids.extend(revise_tokens)
            response_mask.extend([1] * len(revise_tokens))
            if revise_output.log_probs:
                response_logprobs.extend(revise_output.log_probs)

            extra_fields["revise_output"] = revise_text

            await self._maybe_store_claims(namespace, revise_text, now_ts)

            # ACTIONS ---------------------------------------------------------------
            actions_instruction = self._build_actions_instruction(future_len)
            messages.append({"role": "user", "content": actions_instruction})
            prompt_ids = await self._build_prompt_ids(messages, image_data)

            actions_sampling = self._stage_sampling(
                sampling_params,
                stop=self.stop_actions,
                max_tokens=self.max_actions_tokens,
                remaining=self.response_length - len(response_ids),
            )
            actions_output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=actions_sampling,
                image_data=image_data,
            )
            actions_tokens = actions_output.token_ids
            actions_text = self.tokenizer.decode(actions_tokens, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": actions_text})

            response_ids.extend(actions_tokens)
            response_mask.extend([1] * len(actions_tokens))
            if actions_output.log_probs:
                response_logprobs.extend(actions_output.log_probs)

            extra_fields["actions_output"] = actions_text

        # Truncate / align lengths ---------------------------------------------------
        if len(response_ids) > self.response_length:
            response_ids = response_ids[: self.response_length]
            response_mask = response_mask[: self.response_length]
            if response_logprobs:
                response_logprobs = response_logprobs[: self.response_length]

        metrics_model = AgentLoopMetrics(
            generate_sequences=metrics.get("generate_sequences", 0.0),
            tool_calls=0.0,
        )

        num_turns = sum(1 for msg in messages if msg["role"] in {"user", "assistant"})

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        # ---------------------------------------------------------------- Reward
        def _sum_log_probs(values: Optional[list[float]]) -> Optional[float]:
            if not values:
                return None
            return float(sum(values))

        think_loglik = _sum_log_probs(think_output.log_probs)
        revise_loglik = _sum_log_probs(revise_output.log_probs)
        actions_loglik = _sum_log_probs(actions_output.log_probs)

        extra_fields["think_log_likelihood"] = think_loglik
        extra_fields["revise_log_likelihood"] = revise_loglik
        extra_fields["actions_log_likelihood"] = actions_loglik

        gt_text = kwargs.get("ground_truth") or kwargs.get("solution")

        return AgentLoopOutput(
            prompt_ids=prompt_ids_initial,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            num_turns=num_turns,
            metrics=metrics_model,
            extra_fields=extra_fields,
        )

    async def _build_prompt_ids(self, messages: list[dict[str, Any]], image_data: Optional[list[Any]]) -> list[int]:
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            input_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return input_ids

    def _stage_sampling(
        self,
        base_sampling: dict[str, Any],
        *,
        stop: list[str],
        max_tokens: Optional[int],
        remaining: int,
    ) -> dict[str, Any]:
        sampling = dict(base_sampling)
        if stop:
            existing = sampling.get("stop")
            merged = list(existing) if isinstance(existing, list) else _ensure_list(existing)
            merged.extend(stop)
            sampling["stop"] = merged
        if remaining <= 0:
            sampling["max_new_tokens"] = 0
        else:
            current_limit = sampling.get("max_new_tokens")
            candidate = min(remaining, max_tokens) if max_tokens else remaining
            sampling["max_new_tokens"] = candidate if current_limit is None else min(candidate, current_limit)
        return sampling

    async def _retrieve_context(self, namespace: str, think_text: str, now_ts: int) -> str:
        stripped = " ".join(content.strip() for content in _extract_tag_contents(think_text, "think"))
        stripped = stripped or _strip_tags(think_text)
        async with self._retriever_lock:
            hits = self._retriever.query(
                namespace=namespace,
                query=stripped,
                top_k=self.retriever_top_k,
                cutoff_ts=now_ts,
                time_decay_lambda=self.time_decay_lambda or None,
            )
        if not hits:
            return "- (none)"
        return "\n".join(f"- {hit['text']}" for hit in hits if hit.get("text"))

    async def _maybe_store_claims(self, namespace: str, revise_text: str, now_ts: int) -> None:
        claims = _extract_tag_contents(revise_text, "claim")
        if not claims:
            return
        async with self._retriever_lock:
            for claim in claims:
                cleaned = claim.strip()
                if not cleaned:
                    continue
                if self._is_duplicate(namespace, cleaned):
                    continue
                self._retriever.add(
                    namespace=namespace,
                    text=cleaned,
                    now_ts=now_ts,
                    utility=1.0,
                    metadata={"source": "revise"},
                    bucket_key=(namespace, hash(cleaned.lower())),
                )

    def _is_duplicate(self, namespace: str, candidate: str) -> bool:
        existing: Iterable = self._retriever.iter_namespace(namespace)
        for doc in existing:
            if jaccard_ngrams(doc.text, candidate, n=3) >= self.dedup_threshold:
                return True
        return False

    @staticmethod
    def _build_think_instruction(n_claims: int) -> str:
        return (
            "Analyze the previous conversation and produce exactly "
            f"{n_claims} hypotheses about the user's likely next steps. "
            "Return them within a <think> block and wrap each hypothesis using "
            "<claim>...</claim> tags. Do not add commentary outside of the block."
        )

    @staticmethod
    def _build_revise_instruction(retrieved_context: str, n_claims: int) -> str:
        return (
            "Retrieved context (time-aware):\n"
            f"{retrieved_context}\n\n"
            "Refine your earlier hypotheses using this context. Emit the updated "
            f"{n_claims} claims inside a <revise> block, each surrounded by <claim> tags. "
            "Avoid any text outside of the block."
        )

    @staticmethod
    def _build_actions_instruction(future_len: int) -> str:
        return (
            f"Based on your revised claims, forecast exactly {future_len} concrete next actions the user will take. "
            "Return the actions inside an <actions> block, each enclosed in its own <action>...</action> tag, "
            "with no additional text."
        )
