#!/usr/bin/env bash

# Example launch script for the think/retrieve/revise PPO LoRA run.
# This script includes all configurations from ppo_think_revise_lora.yaml inline.

# Start Ray head node with specified IP
# ray start --head \
#   --port=6767 \
#   --node-ip-address=127.0.0.1
export HF_CACHE_DIR=/scr/biggest/oshaikh/hf_cache
export HF_HOME=/scr/biggest/oshaikh/hf_cache
export TRANSFORMERS_CACHE=/scr/biggest/oshaikh/hf_cache
export RAY_DEBUG=legacy
export RAY_DEBUG_EXTERNAL=True
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.num_cpus=5 \
  trainer.total_epochs=1 \
  trainer.experiment_name="test-lora-trainer" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  ++trainer.val_before_train=False \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.exclude_modules='.*visual.*' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.think_revise._target_=verl.workers.config.ThinkReviseConfig \
  +actor_rollout_ref.rollout.think_revise.enable=true \
  +actor_rollout_ref.rollout.think_revise.share_across_workers=true \
  +actor_rollout_ref.rollout.think_revise.actor_name=verl_think_retriever_default \
  +actor_rollout_ref.rollout.think_revise.retriever_top_k=8 \
  +actor_rollout_ref.rollout.think_revise.memory_top_m=3 \
  +actor_rollout_ref.rollout.think_revise.time_decay_lambda=0.0005 \
  +actor_rollout_ref.rollout.think_revise.dedup_jaccard=0.9 \
  +actor_rollout_ref.rollout.think_revise.n_claims_field=n_claims \
  +actor_rollout_ref.rollout.think_revise.future_len_field=future_len \
  +actor_rollout_ref.rollout.think_revise.timestamp_field=ts \
  +actor_rollout_ref.rollout.think_revise.default_n_claims=3 \
  +actor_rollout_ref.rollout.think_revise.default_future_len=3 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.freeze_vision_tower=true \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  data.train_files="/scr/biggest/oshaikh/pack_data/train.parquet" \
  data.val_files="/scr/biggest/oshaikh/pack_data/validation.parquet" \
  data.train_batch_size=16 \
  data.dataloader_num_workers=1 \
  +data.add_generation_prompt=false \
  +data.strip_final_special_token=true \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  reward_model.enable=false \
  reward_model.reward_manager=batch \
  custom_reward_function.path=examples/ppo_trainer/think_revise_lora/reward.py \
  custom_reward_function.name=compute_score
