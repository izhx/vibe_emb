#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data8/zhangxin/.conda/envs/viet/bin/python}"
BASE_MODEL="${BASE_MODEL:-data/raw/Qwen2.5-0.5B}"
REFERENCE_ADAPTER="${REFERENCE_ADAPTER:-results/vibe-embedder-full/checkpoint-1000}"
BASIC_ADAPTER="${BASIC_ADAPTER:-results/vibe-emb-basic-from-ckpt1000/checkpoint-3000}"
CODE_ADAPTER="${CODE_ADAPTER:-results/vibe-emb-code-from-ckpt1000/checkpoint-1328}"
INST_ADAPTER="${INST_ADAPTER:-results/vibe-emb-inst-from-ckpt1000/checkpoint-1500}"
TOOL_ADAPTER="${TOOL_ADAPTER:-results/vibe-emb-tool-from-ckpt1000/checkpoint-1631}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/merges}"
MERGE_SPACE="${MERGE_SPACE:-adapter}"  # adapter | delta-w | full-model
BASIC_WEIGHT="${BASIC_WEIGHT:-1.0}"
CODE_WEIGHT="${CODE_WEIGHT:-1.0}"
INST_WEIGHT="${INST_WEIGHT:-1.0}"
TOOL_WEIGHT="${TOOL_WEIGHT:-1.0}"
LAMBDA_SCALE="${LAMBDA_SCALE:-1.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
SAFE_MERGE_SPACE="${MERGE_SPACE//-/_}"
DEFAULT_OUTPUT="${OUTPUT_ROOT}/multislerp-4task-${MERGE_SPACE}-equal-lambda${LAMBDA_SCALE}"
if [[ "$MERGE_SPACE" == "adapter" ]]; then
  DEFAULT_OUTPUT="${OUTPUT_ROOT}/multislerp-4task-equal-lambda${LAMBDA_SCALE}"
fi
MERGED_OUTPUT="${MERGED_OUTPUT:-${MERGED_ADAPTER:-$DEFAULT_OUTPUT}}"

if [[ "${LOG_TO_RUN_DIR:-1}" == "1" ]]; then
  mkdir -p "$MERGED_OUTPUT"
  exec > >(tee -a "$MERGED_OUTPUT/run.log") 2>&1
fi

MERGE_ARGS=(
  --merge-space "$MERGE_SPACE"
  --base-model "$BASE_MODEL"
  --reference-adapter "$REFERENCE_ADAPTER"
  --adapter "basic=$BASIC_ADAPTER"
  --adapter "code=$CODE_ADAPTER"
  --adapter "inst=$INST_ADAPTER"
  --adapter "tool=$TOOL_ADAPTER"
  --weight "basic=$BASIC_WEIGHT"
  --weight "code=$CODE_WEIGHT"
  --weight "inst=$INST_WEIGHT"
  --weight "tool=$TOOL_WEIGHT"
  --output-dir "$MERGED_OUTPUT"
  --lambda-scale "$LAMBDA_SCALE"
  --torch-dtype "$TORCH_DTYPE"
)

if [[ "${NO_SAFE_SERIALIZATION:-0}" == "1" ]]; then
  MERGE_ARGS+=(--no-safe-serialization)
fi

"$PYTHON_BIN" tools/merge_multi_slerp.py \
  "${MERGE_ARGS[@]}"

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  EVAL_TASKS="${EVAL_TASKS:-ToolRetRetrieval AppsRetrieval CosQA CodeSearchNetRetrieval}"
  EVAL_MODEL_ARGS=(--model_name_or_path "$BASE_MODEL" --adapter "$MERGED_OUTPUT")
  if [[ "$MERGE_SPACE" == "full-model" ]]; then
    EVAL_MODEL_ARGS=(--model_name_or_path "$MERGED_OUTPUT")
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" -u -m vibe_eval.run_mteb \
    "${EVAL_MODEL_ARGS[@]}" \
    --tasks $EVAL_TASKS \
    --output_folder "${EVAL_OUTPUT:-results/mteb_eval_merges/${SAFE_MERGE_SPACE}}" \
    --overwrite_results
fi
