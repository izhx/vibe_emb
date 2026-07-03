# Embedding Training Framework

This document describes the implemented lightweight training framework in `vibe_emb/`.

## Summary

The framework trains bi-encoder embedding models on FlagEmbedding-style json/jsonl data with `query`, `pos`, and `neg` fields. It keeps the training loop close to `transformers.Trainer`, while handling contrastive batches, multi-dataset configuration, cross-device negatives, full fine-tuning, and optional PEFT.

Runtime target:

- Python: `/data8/zhangxin/.conda/envs/viet/bin/python`
- `transformers==5.8.1`
- `accelerate==1.13.0`
- `torch==2.8.0`
- `datasets==4.8.5`
- `peft==0.19.1` or compatible is required when `model.peft_config` is set.

## Structure

- `vibe_emb/train.py`: YAML entrypoint and `EmbeddingTrainRunner`.
- `vibe_emb/arguments.py`: dataclass config sections.
- `vibe_emb/data.py`: FlagEmbedding data loading and deterministic multi-dataset batch planning.
- `vibe_emb/collator.py`: tokenization and batch metadata passthrough.
- `vibe_emb/modeling.py`: AutoModel wrapper, pooling, loss, cross-device gather, optional PEFT.
- `vibe_emb/trainer.py`: minimal `Trainer` subclass for loss and saving.
- `configs/example_train.yaml`: smoke configuration.
- `configs/train_full.yaml`: multitask full-training configuration adapted from `multitask/train_full.sh`.
- `scripts/train_smoke.sh`: local smoke launcher.
- `scripts/train_full.sh`: multitask launcher for the new framework.
- `scripts/verify_dataset_batch_plan.py`: distributed data-plan consistency checker.

## Data And Sampling

Each Trainer dataset item is one prebuilt rank-local contrastive batch. `per_device_train_batch_size` must be `1`, and `gradient_accumulation_steps` must be `1`.

`dataloader_num_workers` must be `0`. Consumption counters are updated inside `__getitem__`; worker-process dataset copies would make Trainer-side stats misleading.

### Trainer And Accelerate Boundary

This framework intentionally does not rely on accelerate to split a normal DataLoader batch across ranks. The custom `EmbeddingTrainer.get_train_dataloader()` uses `SequentialSampler`, `batch_size=1`, and the simple collator path. Each distributed process has its own Python dataset object, and that object already knows `process_index` and `world_size` from `TrainingArguments`.

The split happens inside `MultiDatasetBatchDataset.__getitem__`:

1. all ranks hold the same `batch_plan[step]`, containing unsliced global record indices;
2. `per_rank = len(global_indices) // world_size`;
3. rank `r` takes `global_indices[r * per_rank : (r + 1) * per_rank]`;
4. the collator tokenizes only that rank-local shard.

`per_device_train_batch_size=1` is therefore a structural constraint, not a performance knob. In old FlagEmbedding-style CLI settings, `--per_device_train_batch_size 64` means 64 raw examples per rank. In this framework the equivalent is `data.default_batch_size: 64` or `dataset.batch_size: 64`, while `training.per_device_train_batch_size` stays `1`.

The sampler generates a global batch plan per epoch:

- one batch contains examples from exactly one configured dataset;
- global batch size is `dataset.batch_size * world_size`;
- every rank uses the same seed and plan, then slices its own shard by `process_index`;
- incomplete global batches are dropped to keep distributed shapes consistent.

The global plan must be identical on every rank. Cross-device negatives use `all_gather`, so q/p tensor shapes must match at each step. The same step also needs the same dataset-specific `loss_kwargs`, `model_kwargs`, `no_in_batch_neg`, and `train_group_size`; otherwise contrastive targets can point at wrong offsets or distributed collectives can hang.

The implementation guarantees this by building the plan with `seed + epoch` on every rank, shuffling dataset order and record indices in the same way, storing unsliced `global_indices`, and slicing those indices only in `__getitem__` by `process_index`.

Positive/negative passage sampling is also rank-consistent. Each record uses a stable hash seed derived from the global seed, epoch, dataset key, batch index, global position inside the batch, and record index. The seed does not include `process_index`, so local rank execution is equivalent to sampling the whole global batch first and then slicing it.

Dataset instruction formatting supports both global defaults and dataset-level overrides:

- `data.default_query_instruction`
- `data.default_passage_instruction`
- `data.default_query_instruction_format`
- `data.default_passage_instruction_format`
- `dataset.query_instruction`
- `dataset.passage_instruction`
- `dataset.query_instruction_format`
- `dataset.passage_instruction_format`

Dataset-level values win when set; otherwise the data defaults are used. Record-level `prompt` wins over both and is used as the query instruction when present.

`query_instruction` and `passage_instruction` can be a string or a list of strings. A list is consumed as a deterministic round-robin queue: for a dataset, the first global sample uses item 0, the next uses item 1, and so on. The implementation computes this from the dataset-local global sample position instead of mutating a Python queue. That keeps every rank consistent under distributed slicing and makes local rank execution equivalent to building the full global batch first and then slicing it. For passages, one selected passage instruction is applied to the whole positive/negative group of a query.

Example:

```yaml
data:
  default_query_instruction:
    - Given a query, retrieve relevant passages.
    - Represent this query for retrieval.
  default_passage_instruction: null
  datasets:
    - name: toolret
      path: data/train/toolret-train/toolret.jsonl
```

Dataset sampling controls how many records enter each epoch before global batches are formed:

- `sample_size`: absolute number of records after deterministic shuffle;
- `sample_factor`: proportional factor, default `1.0`;
- `0 < sample_factor < 1.0`: downsample;
- `sample_factor > 1.0`: upsample by appending deterministic shuffled passes, allowing repeated records.

`sample_size` and `sample_factor != 1.0` are mutually exclusive. `repeat_number` is not supported; use `sample_factor > 1.0` for upsampling. Sampling is rank-independent and happens before dropping incomplete global batches.

Dataset consumption logging prints both plan and actual consumption:

- `planned_global_batches` / `planned_global_instances`: full epoch plan, global across all ranks;
- `consumed_local_batches` / `consumed_local_instances`: batches and instances actually fetched by this rank;
- logs are emitted when epochs are planned and on Trainer `logging_steps` from rank 0.

Supported input record:

```json
{"query": "...", "pos": ["..."], "neg": ["..."], "prompt": "...", "type": "normal"}
```

Optional score fields are `pos_scores` and `neg_scores`.

### Distributed Plan Test

Run this check whenever dataset sampling logic changes:

```bash
/data8/zhangxin/.conda/envs/viet/bin/python -m torch.distributed.run \
  --nproc_per_node 2 \
  scripts/verify_dataset_batch_plan.py \
  --config configs/example_train.yaml \
  --steps 8
```

The checker does not load the model. It initializes each rank's `MultiDatasetBatchDataset`, gathers step metadata with `torch.distributed.all_gather_object`, and verifies:

- every rank sees the same `dataset_key` for the same step;
- every rank sees the same unsliced `global_indices`;
- rank-local shards concatenate back to the global batch in rank order;
- rank-local shards are disjoint positional chunks, even when upsampling allows repeated record ids;
- produced query/passage counts match `local_batch_size` and `train_group_size`.

This directly tests the invariant required by cross-device negatives: each distributed step is one shared global dataset batch, partitioned into disjoint per-rank chunks.

## Model And Loss

`EmbeddingModel` wraps `AutoModel.from_pretrained`.

Pooling:

- `cls`
- `mean`
- `last_token`

Loss behavior:

- default in-batch negative cross entropy;
- optional cross-device negatives gather both query and passage reps with `torch.distributed.all_gather`;
- `no_in_batch_neg=true` computes only within each query's local positive/negative group;
- optional KL distillation when teacher scores are present;
- the positive passage is always group index `0`.

For cross-device negatives, each rank computes the full global query-by-passage score matrix. Gathered non-local reps are constants in the local autograd graph, while this rank's gathered slot keeps gradients. The returned loss is the local rank's full-global-matrix cross entropy and is passed to Trainer/DDP directly; the framework does not multiply it by `world_size`. DDP will average gradients across ranks, so this intentionally keeps the effective gradient scale under Trainer's standard distributed semantics. If changing gather/loss reduction strategy later, compare learning-rate sensitivity and loss scale explicitly.

Dataset-specific `loss_kwargs` and `model_kwargs` are passed through each batch. They can override safe runtime behavior such as `temperature`, `pooling`, and `normalize_embeddings`.

## PEFT, Adapter Warm-Start, And Saving

Full fine-tuning works without PEFT config. When `model.peft_config` is set, startup imports PEFT and wraps the base model with `get_peft_model`.

Supported v1 PEFT config:

```yaml
model:
  peft_config:
    type: LoraConfig
    params:
      r: 32
      lora_alpha: 64
      lora_dropout: 0.0
      target_modules: [q_proj, k_proj, v_proj, o_proj]
      bias: none
```

`params` maps directly to PEFT's native `LoraConfig`; the framework does not maintain its own LoRA parameter dataclass.

To start a new run from an already trained adapter, set `model.peft_adapter_name_or_path` to an adapter checkpoint directory and keep `model.model_name_or_path` pointing at the base model:

```yaml
model:
  model_name_or_path: data/raw/Qwen2.5-0.5B
  peft_adapter_name_or_path: results/vibe-embedder-full/checkpoint-1000
```

This is a warm-start path, not a Trainer checkpoint resume. The loader calls `PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True, torch_device="cpu")`, so adapter weights are initialized from the checkpoint on CPU and remain trainable, while optimizer, scheduler, RNG, and `global_step` start fresh. Keeping the adapter load on CPU avoids PEFT/safetensors creating an early CUDA context on visible `cuda:0`; Trainer/DDP still moves the complete model to the process-local GPU. The adapter checkpoint must contain `adapter_config.json`. If `peft_adapter_name_or_path` and `peft_config` are both set, the adapter checkpoint's config is authoritative and `peft_config` is only preserved in saved experiment config for human reference.

Do not set `model.model_name_or_path` directly to an adapter-only checkpoint such as `results/vibe-embedder-full/checkpoint-1000`. That directory contains PEFT adapter files plus Trainer state, not a full `AutoModel` checkpoint.

Saving uses the model wrapper:

- full fine-tune: base model `save_pretrained`;
- PEFT: adapter standard save by default;
- tokenizer and training args are saved on rank 0;
- `model.save_merged_lora_model=true` additionally writes `output_dir/merged` when PEFT supports `merge_and_unload`.

Saved YAML config files use escaped newlines for strings such as `Instruct: {}\nQuery: {}`. This is only a readability choice; YAML block/folded newlines parse back to the same Python string, but escaped newlines are easier to copy into future configs.

## Running

Smoke run:

```bash
scripts/train_smoke.sh
```

Direct run:

```bash
/data8/zhangxin/.conda/envs/viet/bin/python -m vibe_emb.train --config configs/example_train.yaml
```

Multitask run adapted from `multitask/train_full.sh`:

```bash
scripts/train_full.sh
```

Toolret-only warm-start from the full run's `checkpoint-1000`:

```bash
/data8/zhangxin/.conda/envs/viet/bin/python -m vibe_emb.train \
  --config configs/train_toolret_from_lora.yaml
```

Do not pass `--resume_from_checkpoint` for this case. `--resume_from_checkpoint` is for strict continuation of the same Trainer run: it restores optimizer, scheduler, RNG, and Trainer state and is not the right mechanism when changing the dataset to only `toolret`.

The new config keeps the main old settings: Qwen2.5-0.5B, LoRA rank 32 / alpha 64, target projection and MLP modules, bf16, gradient checkpointing, learning rate `1e-4`, warmup ratio `0.1`, one epoch, cross-device negatives, `train_group_size=4`, and up to `100000` records per dataset. It does not enable DeepSpeed by default in v1 because the framework is first validating the Trainer-native path; `TrainingArguments` can still accept `deepspeed` later if the environment and saving behavior are validated.

Previous FlagEmbedding log `results/train.log` shows a 4-GPU run reaching step 2247/3565 in about 5 hours before receiving SIGHUP, with step time commonly around 5-10 seconds and losses often in the `0.1-1.0` range after warmup. Use this only as a rough throughput/metric reference: the new framework gathers both q and p and leaves loss scaling to Trainer/DDP standard averaging, while FlagEmbedding may differ in gather and loss reduction details.

Historical framework smoke result, using:

```bash
CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 MAX_STEPS=2 \
  OUTPUT_DIR=results/vibe-embedder-full-smoke \
  scripts/train_full.sh
```

Observed result before the later removal of explicit `loss * world_size`:

- 10 datasets loaded and `7139` global batches generated for `world_size=2`;
- per-rank dataset batch size was `64`, so each optimizer step used `128` queries and `512` passages globally with `train_group_size=4`;
- two training steps completed in `18.04s` trainer runtime after data loading, around `9s/step`;
- losses were `20.22` and `13.62`, final `train_loss=16.92`;
- PEFT saving wrote adapter files at the final output and `checkpoint-2`, with no merged checkpoint by default.

Trainer's printed `train_samples_per_second` is not meaningful as raw-example throughput in this framework because Trainer sees one prebuilt dataset item per optimizer step. Compute real query throughput manually as `dataset.batch_size * world_size / step_time`.

Useful overrides:

- `--output_dir`
- `--model_name_or_path`
- `--max_steps`
- `--resume_from_checkpoint`
- `--overwrite_output_dir`

Use `--resume_from_checkpoint` only for exact Trainer checkpoint resume. For adapter warm-start with changed data or training schedule, configure `model.peft_adapter_name_or_path` and leave `resume_from_checkpoint` unset.

## Test Scenarios

- Import all framework modules.
- Parse `configs/example_train.yaml`.
- Build dataset plan for one and multiple datasets.
- Verify dataset planned/consumed stats appear in logs.
- Run one forward/backward smoke step on a tiny config.
- Run `torchrun --nproc_per_node=1` for `max_steps=2`.
- Run `torchrun --nproc_per_node=2` when GPUs are available to validate cross-device gather.
- Run `scripts/verify_dataset_batch_plan.py` with `torch.distributed.run` to validate distributed global-batch consistency without loading a model.
- Confirm no `peft_config` uses full fine-tuning and `peft_config.type=LoraConfig` wraps with PEFT.
- Confirm `peft_adapter_name_or_path` loads a saved adapter as trainable and saves adapter-only checkpoints by default.
