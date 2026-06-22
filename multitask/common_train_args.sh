#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data8/zhangxin/.conda/envs/pt28/bin/python}"
export PYTHONPATH="${ROOT_DIR}/ref_repo/FlagEmbedding:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/data/raw/Qwen2.5-0.5B}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/multitask/cache/hf}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${ROOT_DIR}/multitask/cache/datasets}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/multitask/ds_stage1.json}"
NEGATIVES_CROSS_DEVICE="${NEGATIVES_CROSS_DEVICE:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
GRADIENT_CHECKPOINTING_KWARGS="${GRADIENT_CHECKPOINTING_KWARGS:-}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-32}"

TRAIN_DATA=(
  "${ROOT_DIR}/data/train/toolret-train/toolret.jsonl"
  "${ROOT_DIR}/data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl"
  "${ROOT_DIR}/data/train/codesearchnet/codesearchnet.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/fever.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/fiqa.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/hotpotqa.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/msmarco.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/nq.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/squadv2.jsonl"
  "${ROOT_DIR}/data/train/embeddings-fine-tuning/trivia.jsonl"
)

common_model_args=(
  --model_name_or_path "${MODEL_PATH}"
  --cache_dir "${CACHE_DIR}"
  --use_lora True
  --lora_rank "${LORA_RANK:-32}"
  --lora_alpha "${LORA_ALPHA:-64}"
  --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj
  --save_merged_lora_model "${SAVE_MERGED_LORA_MODEL:-False}"
)

common_data_args=(
  --train_data "${TRAIN_DATA[@]}"
  --cache_path "${DATA_CACHE_DIR}"
  --train_group_size "${TRAIN_GROUP_SIZE:-8}"
  --query_max_len "${QUERY_MAX_LEN:-320}"
  --passage_max_len "${PASSAGE_MAX_LEN:-512}"
  --pad_to_multiple_of 8
  --query_instruction_for_retrieval 'Given a query, retrieve passages that are relevant to the query.'
  --query_instruction_format 'Instruct: {}\nQuery: {}'
  --same_dataset_within_batch True
  --small_threshold 0
  --drop_threshold 0
)

common_training_args=(
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --optim "${OPTIM:-adamw_torch}"
  --bf16 "${BF16:-True}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-64}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --dataloader_drop_last True
  --warmup_ratio "${WARMUP_RATIO:-0.1}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --save_steps "${SAVE_STEPS:-1000}"
  --temperature "${TEMPERATURE:-0.02}"
  --sentence_pooling_method last_token
  --normalize_embeddings True
  --sub_batch_size "${SUB_BATCH_SIZE}"
  --report_to none
)

if [ "${GRADIENT_CHECKPOINTING}" = "True" ] || [ "${GRADIENT_CHECKPOINTING}" = "true" ] || [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
  common_training_args+=(--gradient_checkpointing)
fi

if [ -n "${GRADIENT_CHECKPOINTING_KWARGS}" ]; then
  common_training_args+=(--gradient_checkpointing_kwargs "${GRADIENT_CHECKPOINTING_KWARGS}")
fi

if [ -n "${DDP_FIND_UNUSED_PARAMETERS}" ]; then
  common_training_args+=(--ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}")
fi

if [ "${DEEPSPEED_CONFIG}" != "none" ] && [ -n "${DEEPSPEED_CONFIG}" ]; then
  common_training_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [ "${NEGATIVES_CROSS_DEVICE}" = "True" ] || [ "${NEGATIVES_CROSS_DEVICE}" = "true" ] || [ "${NEGATIVES_CROSS_DEVICE}" = "1" ]; then
  common_training_args+=(--negatives_cross_device)
fi
