#!/usr/bin/env bash


#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data8/zhangxin/.conda/envs/viet/bin/python}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"


run_train_task() {
  export CUDA_VISIBLE_DEVICES
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"

  local -a cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${NUM_GPUS}"
    -m vibe_emb.train
    --config "${CONFIG}"
    --overwrite_output_dir
  )

  # Keep the YAML output_dir by default. Set OUTPUT_DIR explicitly when a run
  # should be redirected without editing the task config.
  if [ -n "${OUTPUT_DIR}" ]; then
    cmd+=(--output_dir "${OUTPUT_DIR}")
  fi

  if [ -n "${MAX_STEPS}" ]; then
    cmd+=(--max_steps "${MAX_STEPS}")
  fi

  printf '%q ' "${cmd[@]}"
  printf '\n'
  cd "${ROOT_DIR}"
  exec "${cmd[@]}"
}


CONFIG="${CONFIG:-${ROOT_DIR}/configs/train_full.yaml}"
# OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/vibe-embedder-full}"
NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
run_train_task

