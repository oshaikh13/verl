#!/bin/bash
# Training script for Think-Retrieve-Revise agent loop with text overlap reward

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

set -x

# Configuration
PROJECT_DIR="$(pwd)"
DATASET_NAME="pack_data"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # e.g., "Qwen/Qwen2.5-7B-Instruct"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# in addtion, upgrade vllm to 0.11.0

# Wandb configuration
export WANDB_API_KEY="${WANDB_API_KEY:-25dd9b773838ca659e1a9d3cc8ef206dae0a3275}"
export WANDB_MODE="${WANDB_MODE:-online}"  # Set to "offline" if you want to log offline
export WANDB_DIR="${WANDB_DIR:-./wandb_logs}"
# export RAY_DEBUG_POST_MORTEM=1

# Start Ray cluster on specific port (will skip if already running on that port)
RAY_PORT=42012
RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
RAY_NUM_CPUS=16
RAY_NUM_GPUS=2
echo "Starting/connecting to Ray cluster on port ${RAY_PORT}..."
# RAY_DEBUG=legacy ray start --head --port=${RAY_PORT} --num-cpus=${RAY_NUM_CPUS} --num-gpus=${RAY_NUM_GPUS} --dashboard-host=0.0.0.0 2>&1 | grep -v "Ray runtime started" || true
ray start --head --port=${RAY_PORT} --num-cpus=${RAY_NUM_CPUS} --num-gpus=${RAY_NUM_GPUS} --dashboard-host=0.0.0.0 2>&1 | grep -v "Ray runtime started" || true

# Create agent loop config
AGENT_CONFIG_PATH="$PROJECT_DIR/recipe/think_retrieve_revise/agent_config.yaml"
cat > "$AGENT_CONFIG_PATH" << EOF
- name: think_retrieve_revise_agent
  _target_: verl.experimental.agent_loop.think_retrieve_revise_agent_loop.ThinkRetrieveReviseAgentLoop
EOF

# Training command
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/scr/biggest/oshaikh/$DATASET_NAME/train.parquet \
    data.val_files=/scr/biggest/oshaikh/$DATASET_NAME/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.trust_remote_code=True \
    data.dataloader_num_workers=2 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules=\'.*visual.*,visual.*\' \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    +actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_CONFIG_PATH \
    +actor_rollout_ref.rollout.think_retrieve_revise.default_n_claims=3 \
    +actor_rollout_ref.rollout.think_retrieve_revise.default_future_len=3 \
    +actor_rollout_ref.rollout.think_retrieve_revise.max_think_tokens=256 \
    +actor_rollout_ref.rollout.think_retrieve_revise.max_revise_tokens=256 \
    +actor_rollout_ref.rollout.think_retrieve_revise.max_actions_tokens=192 \
    +actor_rollout_ref.rollout.think_retrieve_revise.retriever_top_k=6 \
    +actor_rollout_ref.rollout.think_retrieve_revise.time_decay_lambda=0.1 \
    +actor_rollout_ref.rollout.think_retrieve_revise.dedup_threshold=0.85 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='think_retrieve_revise_experiment' \
    trainer.experiment_name="trr_${DATASET_NAME}_lora_${TIMESTAMP}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    trainer.default_local_dir="/scr/biggest/oshaikh/checkpoints/think_retrieve_revise_experiment/trr_${DATASET_NAME}_lora_${TIMESTAMP}" \
    custom_reward_function.path=$PROJECT_DIR/recipe/think_retrieve_revise/reward_function.py \
    custom_reward_function.name=compute_score \
    +ray_kwargs.ray_init.address=127.0.0.1:42012