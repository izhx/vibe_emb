#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_GPUS="${NUM_GPUS:-4}"
MAX_EXAMPLES_PER_DATASET="${MAX_EXAMPLES_PER_DATASET:-100000}"
MAX_STEPS="${MAX_STEPS:-}"

# Core model/data length settings.
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../data/raw/Qwen2.5-0.5B}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-320}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-512}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-4}"

# Large-batch contrastive training settings.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-0}"
NEGATIVES_CROSS_DEVICE="${NEGATIVES_CROSS_DEVICE:-True}"

# Optimizer/runtime settings.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/ds_stage1.json}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
GRADIENT_CHECKPOINTING_KWARGS="${GRADIENT_CHECKPOINTING_KWARGS:-}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

source "${SCRIPT_DIR}/common_train_args.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/qwen2_5-0b5-embedder-multitask}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${DATA_CACHE_DIR}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${NUM_GPUS}"
  -m FlagEmbedding.finetune.embedder.decoder_only.base
  "${common_model_args[@]}"
  "${common_data_args[@]}"
  --max_example_num_per_dataset "${MAX_EXAMPLES_PER_DATASET}"
  "${common_training_args[@]}"
  --output_dir "${OUTPUT_DIR}"
  --overwrite_output_dir
)

if [ -n "${MAX_STEPS}" ]; then
  cmd+=(--max_steps "${MAX_STEPS}")
fi

printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
