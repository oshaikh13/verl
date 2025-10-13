#!/bin/bash
# Training script for Think-Retrieve-Revise agent loop with text overlap reward

set -x

# Configuration
PROJECT_DIR="$(pwd)"
DATASET_NAME="your_dataset"
MODEL_PATH="your_model_path"  # e.g., "Qwen/Qwen2.5-7B-Instruct"

# Create agent loop config
AGENT_CONFIG_PATH="$PROJECT_DIR/recipe/think_retrieve_revise/agent_config.yaml"
cat > "$AGENT_CONFIG_PATH" << EOF
- name: think_retrieve_revise_agent
  _target_: verl.experimental.agent_loop.think_retrieve_revise_agent_loop.ThinkRetrieveReviseAgentLoop
EOF

# Training command
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=\$HOME/data/$DATASET_NAME/train.parquet \
    data.val_files=\$HOME/data/$DATASET_NAME/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules='["gate_proj","up_proj","down_proj"]' \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_CONFIG_PATH \
    actor_rollout_ref.rollout.think_retrieve_revise.default_n_claims=3 \
    actor_rollout_ref.rollout.think_retrieve_revise.default_future_len=3 \
    actor_rollout_ref.rollout.think_retrieve_revise.max_think_tokens=256 \
    actor_rollout_ref.rollout.think_retrieve_revise.max_revise_tokens=256 \
    actor_rollout_ref.rollout.think_retrieve_revise.max_actions_tokens=192 \
    actor_rollout_ref.rollout.think_retrieve_revise.retriever_top_k=6 \
    actor_rollout_ref.rollout.think_retrieve_revise.time_decay_lambda=0.1 \
    actor_rollout_ref.rollout.think_retrieve_revise.dedup_threshold=0.85 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='think_retrieve_revise_experiment' \
    trainer.experiment_name='trr_${DATASET_NAME}_lora_$(date +%Y%m%d_%H%M%S)' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    custom_reward_function.path=$PROJECT_DIR/recipe/think_retrieve_revise/reward_function.py \
    custom_reward_function.name=compute_score
