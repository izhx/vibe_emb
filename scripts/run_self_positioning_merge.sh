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

BASIC_DATASETS="${BASIC_DATASETS:-${BASIC_DATASET:-data/train/embeddings-fine-tuning/msmarco.jsonl}}"
CODE_DATASETS="${CODE_DATASETS:-${CODE_DATASET:-data/train/codesearchnet/codesearchnet.jsonl}}"
INST_DATASETS="${INST_DATASETS:-${INST_DATASET:-data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl}}"
TOOL_DATASETS="${TOOL_DATASETS:-${TOOL_DATASET:-data/train/toolret-train/toolret.jsonl}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/merges}"
SEARCH_STEPS="${SEARCH_STEPS:-1000}"
EXAMPLES_PER_DATASET="${EXAMPLES_PER_DATASET:-8000}"
MAX_SOURCE_LINES="${MAX_SOURCE_LINES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-3}"
MU="${MU:-0.05}"
MERGE_DEVICE="${MERGE_DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-320}"
PASSAGE_MAX_LENGTH="${PASSAGE_MAX_LENGTH:-512}"
SEED="${SEED:-13}"
LOG_STEPS="${LOG_STEPS:-10}"

MERGED_ADAPTER="${MERGED_ADAPTER:-${OUTPUT_ROOT}/self-positioning-4task-steps${SEARCH_STEPS}-mu${MU}}"

if [[ "${LOG_TO_RUN_DIR:-0}" == "1" ]]; then
  mkdir -p "$MERGED_ADAPTER"
  exec > >(tee -a "$MERGED_ADAPTER/run.log") 2>&1
fi

MERGE_DATASET_ARGS=()
SMOKE_DATASET_ARGS=()

append_dataset_args() {
  local name="$1"
  local specs="$2"
  local matched_any=0
  local spec
  for spec in $specs; do
    if [[ "$spec" == *[\*\?\[]* ]]; then
      local matches=()
      mapfile -t matches < <(compgen -G "$spec" | sort)
      if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "No files matched dataset glob for ${name}: ${spec}" >&2
        exit 1
      fi
      local match
      for match in "${matches[@]}"; do
        MERGE_DATASET_ARGS+=(--dataset "${name}=${match}")
        SMOKE_DATASET_ARGS+=(--dataset "$name" "$match")
        matched_any=1
      done
    else
      MERGE_DATASET_ARGS+=(--dataset "${name}=${spec}")
      SMOKE_DATASET_ARGS+=(--dataset "$name" "$spec")
      matched_any=1
    fi
  done
  if [[ "$matched_any" -eq 0 ]]; then
    echo "No dataset files configured for ${name}" >&2
    exit 1
  fi
}

append_dataset_args basic "$BASIC_DATASETS"
append_dataset_args code "$CODE_DATASETS"
append_dataset_args inst "$INST_DATASETS"
append_dataset_args tool "$TOOL_DATASETS"

"$PYTHON_BIN" tools/merge_self_positioning.py \
  --reference-adapter "$REFERENCE_ADAPTER" \
  --adapter "basic=$BASIC_ADAPTER" \
  --adapter "code=$CODE_ADAPTER" \
  --adapter "inst=$INST_ADAPTER" \
  --adapter "tool=$TOOL_ADAPTER" \
  "${MERGE_DATASET_ARGS[@]}" \
  --output-dir "$MERGED_ADAPTER" \
  --base-model "$BASE_MODEL" \
  --device "$MERGE_DEVICE" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --search-steps "$SEARCH_STEPS" \
  --log-steps "$LOG_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --examples-per-dataset "$EXAMPLES_PER_DATASET" \
  --max-source-lines "$MAX_SOURCE_LINES" \
  --train-group-size "$TRAIN_GROUP_SIZE" \
  --max-length "$MAX_LENGTH" \
  --query-max-length "$QUERY_MAX_LENGTH" \
  --passage-max-length "$PASSAGE_MAX_LENGTH" \
  --learning-rate "$LEARNING_RATE" \
  --mu "$MU" \
  --seed "$SEED"

if [[ "${RUN_SMOKE_EVAL:-0}" == "1" ]]; then
  SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-16}"
  SMOKE_MAX_NEGATIVES="${SMOKE_MAX_NEGATIVES:-3}"
  SMOKE_DEVICE="${SMOKE_DEVICE:-$MERGE_DEVICE}"
  SMOKE_DTYPE="${SMOKE_DTYPE:-$DTYPE}"
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-${MERGED_ADAPTER}/smoke_eval.json}"

  "$PYTHON_BIN" scripts/eval_adapter_retrieval_smoke.py \
    --checkpoint "$BASIC_ADAPTER" \
    --checkpoint "$CODE_ADAPTER" \
    --checkpoint "$INST_ADAPTER" \
    --checkpoint "$TOOL_ADAPTER" \
    --checkpoint "$MERGED_ADAPTER" \
    "${SMOKE_DATASET_ARGS[@]}" \
    --model-name-or-path "$BASE_MODEL" \
    --device "$SMOKE_DEVICE" \
    --dtype "$SMOKE_DTYPE" \
    --examples "$SMOKE_EXAMPLES" \
    --max-negatives "$SMOKE_MAX_NEGATIVES" \
    --batch-size "${SMOKE_BATCH_SIZE:-16}" \
    --max-length "${SMOKE_MAX_LENGTH:-$MAX_LENGTH}" \
    --trust-remote-code \
    --output "$SMOKE_OUTPUT"
fi

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  EVAL_TASKS="${EVAL_TASKS:-NanoMSMARCORetrieval NanoNQRetrieval AppsRetrieval CodeSearchNetRetrieval CosQA Core17InstructionRetrieval ToolRetRetrieval}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" -u -m vibe_eval.run_mteb \
    --model_name_or_path "$BASE_MODEL" \
    --adapter "$MERGED_ADAPTER" \
    --tasks $EVAL_TASKS \
    --output_folder "${EVAL_OUTPUT:-results/mteb_eval_self_positioning}" \
    --overwrite_results \
    --trust_remote_code
fi
