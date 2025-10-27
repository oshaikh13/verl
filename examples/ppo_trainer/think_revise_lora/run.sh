#!/usr/bin/env bash

# Example launch script for the think/retrieve/revise PPO LoRA run.
# Override these env vars to point at your actual assets before launching.
export POLICY_BASE="${POLICY_BASE:-~/models/qwen/Qwen2-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-~/data/rlhf/demo/train.parquet}"
export VAL_DATA="${VAL_DATA:-~/data/rlhf/demo/val.parquet}"
export RUN_NAME="${RUN_NAME:-think-revise-lora-demo}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

python -m verl.trainer.main_ppo \
  --config-path verl/trainer/config \
  --config-name ppo_think_revise_lora \
  trainer.total_epochs=1 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.model.path="${POLICY_BASE}" \
  actor_rollout_ref.model.exclude_modules='.*visual.*' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.freeze_vision_tower=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  trainer.experiment_name="${RUN_NAME}"
