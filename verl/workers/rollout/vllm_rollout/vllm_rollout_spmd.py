# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import os
import pickle
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from types import MethodType
from typing import Any, Generator, Sequence

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.retriever import DistributedRetriever, InMemoryBM25Temporal
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.retrieval import jaccard_ngrams, mmr_select
from verl.utils.torch_functional import (get_response_mask,
                                         pad_2d_list_to_length)
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (VLLM_LORA_INT_ID,
                                                     VLLM_LORA_NAME,
                                                     VLLM_LORA_PATH,
                                                     get_vllm_max_lora_rank)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(model_config.lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id


        self.think_cfg = getattr(config, "think_revise", None)
        if self.think_cfg and self.think_cfg.enable:
            dedup_fn = partial(jaccard_ngrams, n=3)
            retriever_factory = partial(
                InMemoryBM25Temporal,
                dedup_threshold=self.think_cfg.dedup_jaccard,
                dedup_sim_fn=dedup_fn,
            )
            self._dist_retr = None
            DistributedRetriever(
                retriever_factory=retriever_factory,
                default_namespace=self.think_cfg.memory_namespace,
                shared=self.think_cfg.share_across_workers,
                actor_name=self.think_cfg.actor_name,
            )
        else:
            self._dist_retr = None

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """

        if self.think_cfg and self.think_cfg.enable:
            return self._generate_sequences_think_revise(prompts)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _generate_sequences_think_revise(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)

        eos_token_id = prompts.meta_info["eos_token_id"]
        prompt_length = idx.size(1)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            raise ValueError("think/revise loop requires 'raw_prompt_ids' in non_tensor_batch.")

        raw_prompt_ids_arr = non_tensor_batch.pop("raw_prompt_ids")
        raw_prompt_ids_list = [list(map(int, seq)) for seq in raw_prompt_ids_arr.tolist()]

        if "multi_modal_data" in non_tensor_batch:
            multi_modal_data_arr = non_tensor_batch.pop("multi_modal_data")
            multi_modal_list = list(multi_modal_data_arr.tolist())
        else:
            multi_modal_list = [None] * batch_size

        think_cfg = self.think_cfg
        tokenizer = self.tokenizer

        def _extract_int_field(field_name: str, default_val: int) -> list[int]:
            arr = non_tensor_batch.get(field_name)
            if arr is None:
                return [default_val] * batch_size
            values = arr.tolist()
            result: list[int] = []
            for value in values:
                try:
                    result.append(int(value))
                except (TypeError, ValueError):
                    result.append(default_val)
            return result

        n_claims_list = _extract_int_field(think_cfg.n_claims_field, think_cfg.default_n_claims)
        future_len_list = _extract_int_field(think_cfg.future_len_field, think_cfg.default_future_len)

        ts_arr = non_tensor_batch.get(think_cfg.timestamp_field)
        default_ts = int(prompts.meta_info.get("global_steps", 0))
        if ts_arr is None:
            now_ts_list = [default_ts] * batch_size
        else:
            now_ts_list = []
            for value in ts_arr.tolist():
                try:
                    now_ts_list.append(int(value))
                except (TypeError, ValueError):
                    now_ts_list.append(default_ts)

        instruction_tokens_list: list[list[int]] = []
        current_prompt_tokens: list[list[int]] = []
        for base_tokens, n_claims in zip(raw_prompt_ids_list, n_claims_list, strict=True):
            instruction_text = "\n" + think_cfg.think_instruction.format(n_claims=n_claims)
            instruction_tokens = tokenizer.encode(instruction_text, add_special_tokens=False)
            instruction_tokens_list.append(instruction_tokens)
            current_prompt_tokens.append(list(base_tokens) + instruction_tokens)


        think_ids_list, think_logps_list = self._run_vllm_prompts(
            current_prompt_tokens,
            multi_modal_list,
            # stop=think_cfg.think_stop,
            max_tokens=think_cfg.think_max_tokens,
        )

        think_texts = [
            tokenizer.decode(ids, skip_special_tokens=False) if ids else ""
            for ids in think_ids_list
        ]

        if self._dist_retr and think_cfg.retriever_top_k > 0:
            hits_lists = self._dist_retr.query_batch(
                queries=think_texts,
                cutoff_ts_list=now_ts_list,
                top_k=think_cfg.retriever_top_k,
                time_decay_lambda=think_cfg.time_decay_lambda,
                namespaces=[think_cfg.memory_namespace] * batch_size,
            )
            selected_hits = []
            sim_fn = lambda a, b: jaccard_ngrams(a, b, n=3)
            for hits in hits_lists:
                if not hits:
                    selected_hits.append([])
                    continue
                items = [(hit.get("text", ""), float(hit.get("score", 0.0)), hit) for hit in hits]
                chosen = mmr_select(items=items, sim_fn=sim_fn, top_m=think_cfg.memory_top_m, alpha=think_cfg.mmr_alpha)
                selected_hits.append(chosen)
        else:
            selected_hits = [[] for _ in range(batch_size)]

        retrieved_texts = []
        for hits in selected_hits:
            if not hits:
                retrieved_texts.append("- (none)")
            else:
                retrieved_texts.append("\n".join(item[0] for item in hits if item[0]))

        revise_suffix_tokens_list: list[list[int]] = []
        current_prompt_tokens_stage2: list[list[int]] = []

        for prompt_tokens, think_ids, retrieved, n_claims in zip(
            current_prompt_tokens,
            think_ids_list,
            retrieved_texts,
            n_claims_list,
            strict=True,
        ):
            suffix_text = "<|im_end|>\n<|im_start|>user\n" + think_cfg.revise_instruction.format(
                retrieved=retrieved,
                n_claims=n_claims,
            )
            suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
            revise_suffix_tokens_list.append(suffix_tokens)
            current_prompt_tokens_stage2.append(prompt_tokens + think_ids + suffix_tokens)

        revise_ids_list, revise_logps_list = self._run_vllm_prompts(
            current_prompt_tokens_stage2,
            multi_modal_list,
            # stop=think_cfg.revise_stop,
            max_tokens=think_cfg.revise_max_tokens,
        )

        revise_texts = [
            tokenizer.decode(ids, skip_special_tokens=False) if ids else ""
            for ids in revise_ids_list
        ]

        actions_suffix_tokens_list: list[list[int]] = []
        current_prompt_tokens_stage3: list[list[int]] = []

        for prompt_tokens, revise_ids, future_len in zip(
            current_prompt_tokens_stage2,
            revise_ids_list,
            future_len_list,
            strict=True,
        ):
            suffix_text = "<|im_end|>\n<|im_start|>user\n" + think_cfg.actions_instruction.format(future_len=future_len)
            suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
            actions_suffix_tokens_list.append(suffix_tokens)
            current_prompt_tokens_stage3.append(prompt_tokens + revise_ids + suffix_tokens)

        actions_ids_list, actions_logps_list = self._run_vllm_prompts(
            current_prompt_tokens_stage3,
            multi_modal_list,
            # stop=think_cfg.actions_stop,
            max_tokens=think_cfg.actions_max_tokens,
        )

        if self._dist_retr:
            rows = []
            for revise_text, now_ts in zip(revise_texts, now_ts_list, strict=True):
                claims = self._extract_claims(revise_text)
                if not claims:
                    fallback = revise_text.strip()
                    if fallback:
                        claims = [fallback]
                for claim in claims:
                    rows.append(
                        {
                            "text": claim,
                            "now_ts": now_ts,
                            "utility": 0.0,
                        }
                    )
            if rows:
                self._dist_retr.add_candidates(rows, visible_delay=think_cfg.visible_delay)

        response_sequences: list[list[int]] = []
        mask_sequences: list[list[int]] = []
        logprob_sequences: list[list[float]] = []
        need_logprobs = self.config.calculate_log_probs
        max_response_length = self.config.response_length

        for (
            instruction_tokens,
            think_ids,
            think_logps,
            revise_suffix_tokens,
            revise_ids,
            revise_logps,
            actions_suffix_tokens,
            actions_ids,
            actions_logps,
        ) in zip(
            instruction_tokens_list,
            think_ids_list,
            think_logps_list,
            revise_suffix_tokens_list,
            revise_ids_list,
            revise_logps_list,
            actions_suffix_tokens_list,
            actions_ids_list,
            actions_logps_list,
            strict=True,
        ):
            seq = (
                list(instruction_tokens)
                + list(think_ids)
                + list(revise_suffix_tokens)
                + list(revise_ids)
                + list(actions_suffix_tokens)
                + list(actions_ids)
            )
            mask = (
                [0] * len(instruction_tokens)
                + [1] * len(think_ids)
                + [0] * len(revise_suffix_tokens)
                + [1] * len(revise_ids)
                + [0] * len(actions_suffix_tokens)
                + [1] * len(actions_ids)
            )
            if need_logprobs:
                log_seq = (
                    [0.0] * len(instruction_tokens)
                    + [float(v) for v in think_logps]
                    + [0.0] * len(revise_suffix_tokens)
                    + [float(v) for v in revise_logps]
                    + [0.0] * len(actions_suffix_tokens)
                    + [float(v) for v in actions_logps]
                )
            else:
                log_seq = []

            if len(seq) > max_response_length:
                seq = seq[:max_response_length]
                mask = mask[:max_response_length]
                if need_logprobs:
                    log_seq = log_seq[:max_response_length]

            response_sequences.append(seq)
            mask_sequences.append(mask)
            if need_logprobs:
                logprob_sequences.append(log_seq)

        response = pad_2d_list_to_length(
            response_sequences,
            pad_token_id=self.pad_token_id,
            max_length=max_response_length,
        ).to(idx.device)

        response_mask = pad_2d_list_to_length(
            mask_sequences,
            pad_token_id=0,
            max_length=max_response_length,
        ).to(idx.device)
        response_mask = response_mask.to(attention_mask.dtype)

        if need_logprobs:
            rollout_log_probs = pad_2d_list_to_length(
                logprob_sequences,
                pad_token_id=0.0,
                max_length=max_response_length,
            ).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)
        else:
            rollout_log_probs = None

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = (response != self.pad_token_id).to(attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
            },
            batch_size=batch_size,
        )

        if need_logprobs and rollout_log_probs is not None:
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _run_vllm_prompts(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        multi_modal_list: Sequence | None,
        max_tokens: int,
        stop: str | None = None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        vllm_inputs = []
        multi_modal_list = multi_modal_list or [None] * len(prompt_token_ids)
        for tokens, multi_modal in zip(prompt_token_ids, multi_modal_list, strict=True):
            entry = {"prompt_token_ids": list(tokens)}
            if multi_modal is not None:
                entry["multi_modal_data"] = multi_modal
            vllm_inputs.append(entry)

        sampling_kwargs = {"max_tokens": max_tokens}
        if stop:
            sampling_kwargs["stop"] = [stop]
        if self.config.calculate_log_probs:
            sampling_kwargs["logprobs"] = max(1, getattr(self.sampling_params, "logprobs", 0) or 0)

        with self.update_sampling_params(**sampling_kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        ids_list: list[list[int]] = []
        logps_list: list[list[float]] = []
        need_logprobs = self.config.calculate_log_probs
        for output in outputs:
            sample_output = output.outputs[0]
            ids = list(sample_output.token_ids)
            ids_list.append(ids)
            if need_logprobs:
                curr_logps: list[float] = []
                for token_id, logprob_dict in zip(ids, sample_output.logprobs):
                    if token_id in logprob_dict:
                        curr_logps.append(float(logprob_dict[token_id].logprob))
                    elif logprob_dict:
                        curr_logps.append(float(next(iter(logprob_dict.values())).logprob))
                    else:
                        curr_logps.append(0.0)
                logps_list.append(curr_logps)
            else:
                logps_list.append([])

        return ids_list, logps_list

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        if not text:
            return []
        pattern = re.compile(r"<claim>(.*?)</claim>", flags=re.IGNORECASE | re.DOTALL)
        claims = []
        for match in pattern.findall(text):
            cleaned = match.strip()
            if cleaned:
                claims.append(f"<claim>{cleaned}</claim>")
        return claims

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import \
                patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(model_config.lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        # https://github.com/vllm-project/vllm/issues/25171
        if config.layered_summon or config.expert_parallel_size > 1:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import \
                patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
