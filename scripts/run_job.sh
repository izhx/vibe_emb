#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/share/envs/embt/bin/python}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"

# SenseCore injects the new SENSECORE_* names for multi-node jobs.  Keep the
# old WORLD_SIZE/RANK names as fallbacks so jobs created with the old runtime
# specification continue to work.  NUM_GPUS is retained for local execution.
NNODES="${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-0}}"
NPROC_PER_NODE="${SENSECORE_ACCELERATE_DEVICE_COUNT:-${NUM_GPUS:-1}}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"


run_train_task() {
  export CUDA_VISIBLE_DEVICES
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  # export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  # export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

  local -a cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    -m vibe_emb.train
    --config "${CONFIG}"
  )

  if [ "${OVERWRITE_OUTPUT_DIR}" = "true" ]; then
    cmd+=(--overwrite_output_dir)
  fi

  # Keep the YAML output_dir by default. Set OUTPUT_DIR explicitly when a run
  # should be redirected without editing the task config.
  if [ -n "${OUTPUT_DIR}" ]; then
    cmd+=(--output_dir "${OUTPUT_DIR}")
  fi

  if [ -n "${MAX_STEPS}" ]; then
    cmd+=(--max_steps "${MAX_STEPS}")
  fi

  if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi

  printf '%q ' "${cmd[@]}"
  printf '\n'
  cd "${ROOT_DIR}"
  exec "${cmd[@]}"
}


CONFIG="${CONFIG:-${ROOT_DIR}/configs/train_f2llm_stage2_full.yaml}"
run_train_task
