# F2LLM Stage-2 PyTorch DataLoader Prefetch Implementation Plan

状态：CP-1/CP-2/CP-4 和 AC-11 已完成；CP-3 checkpoint resume 与 AC-12 性能门槛待验证  
日期：2026-07-20  
依据：`aidoc/f2llm/spec/f2llm_stage2_dataloader_prefetch_spec.md`  
本计划只描述实施顺序，不包含代码或配置修改。

## 1. 实施结果

完成本计划后，训练框架应支持两条明确路径：

1. 历史同步路径：`dataloader_num_workers=0`、`prefetch_factor=null`；
2. 有界预取路径：`dataloader_num_workers>0`、每 worker 按 resolved factor 提前准备完整
   rank-local 对比学习 batch。

两条路径共享同一个 deterministic batch plan、Arrow reader、collator、模型和 Trainer。
prefetch 只改变数据准备与 GPU 计算的重叠方式，不改变实际训练 batch。框架默认继续使用同步路径；
`configs/train_f2llm_stage2_full.yaml` 是否切到 worker 1，必须在代码正确性验收和 QuarkFS A/B
通过后作为独立提交决定。

目标数据流为：

```text
SequentialSampler(batch_idx)
        |
        v
DataLoader worker
  MultiDatasetBatchDataset.__getitem__
  -> ArrowStorePool.get/get_records
  -> _build_batch
  -> EmbeddingCollator.__call__
        |
        v
ordered DataLoader result queue
        |
        v
EmbeddingTrainer.training_step
  -> main-process consumption acknowledgement
  -> strip private batch metadata
  -> existing model/loss path
```

## 2. 当前实现基线

### 2.1 已有能力

- `vibe_emb.data.MultiDatasetBatchDataset.refresh_epoch()` 生成所有 rank 一致的 compact
  batch plan；`__getitem__()` 再按 `process_index` 切 rank-local shard。
- `_build_batch()` 的正负例随机性来自稳定的 seed、epoch、dataset、batch position 和 record
  identity，不依赖进程调度。
- `vibe_emb.record_store.ArrowStorePool` 是每进程有界 LRU；
  `IndexedArrowRecordStore` 按 unit lazy mmap query/corpus。
- `vibe_emb.collator.EmbeddingCollator` 在 collate 阶段完成 query/passage tokenizer 和 padding。
- `vibe_emb.trainer.EmbeddingTrainer.get_train_dataloader()` 使用 `SequentialSampler`、
  `batch_size=1` 和自定义 collator。
- Transformers 5.8 的 epoch 调用顺序是 callback `on_epoch_begin()` 先执行，随后才创建
  DataLoader iterator；在 `persistent_workers=false` 时，可利用这个顺序让新 worker 继承最新
  epoch plan。

### 2.2 当前阻塞点

| 文件/符号 | 当前行为 | 对实施的影响 |
| --- | --- | --- |
| `vibe_emb/train.py::_build_training_args` | 拒绝所有 `dataloader_num_workers != 0` | 必须在统计迁移后解除硬限制，并补组合校验 |
| `vibe_emb/trainer.py::EmbeddingTrainer.get_train_dataloader` | 不传 factor/persistent/in-order/worker init | YAML factor 会被忽略，worker>0 只能落到 PyTorch 隐式默认 |
| `vibe_emb/data.py::MultiDatasetBatchDataset.__getitem__` | 在 worker 可执行路径中直接增加 consumed counters | worker 副本不可见，且 prefetched batch 会被提前计数 |
| `vibe_emb/data.py::log_consumption_stats` | 当前函数体是 `pass` | spec 要求的外部可观察 consumption 日志尚不存在 |
| `vibe_emb/collator.py::EmbeddingCollator.__call__` | 丢弃 `dataset_key` 等内部元数据 | 主训练进程无法确认本步应增加哪个 dataset counter |
| `vibe_emb/callbacks.py::DatasetStatsCallback` | 调用空的 dataset logger | 需要接入真实 consumption 和 wait metrics |
| `vibe_emb/train.py::EmbeddingTrainRunner.__init__` | `resolved_config.yaml` 的 training 部分仍写原始 `training_raw` | 隐式 factor=2 不会出现在 resolved artifact |
| `scripts/verify_dataset_batch_plan.py::_rank_step_snapshot` | 直接调用 `dataset[step]` | 没有经过真实 DataLoader worker/queue/collator 路径 |
| `tests/test_indexed_arrow_store.py` | 已覆盖 lazy/LRU、task defaults、双 rank plan | 可复用 Arrow fixture，但尚无 multiprocessing、epoch、resume 测试 |

## 3. 具体文件和符号变更总览

### 3.1 生产代码

| 文件 | 修改或新增符号 | 计划职责 |
| --- | --- | --- |
| `vibe_emb/train.py` | `_build_training_args` | 设置默认值；校验 worker/factor/persistent 组合；移除 worker 必须为 0 的旧限制 |
| `vibe_emb/train.py` | 新增 `_resolved_training_config` 或等价 helper | 将实际 worker、factor、persistent、pin-memory 写入 `resolved_config.yaml` |
| `vibe_emb/train.py` | `EmbeddingTrainRunner._load_trainer` | 将 trainer/telemetry 引用传给 stats callback |
| `vibe_emb/train.py` | `EmbeddingTrainRunner.run` | 结束时落盘每 rank wait/RSS/consumption summary；保持 checkpoint 保存顺序 |
| `vibe_emb/trainer.py` | 新增 `create_train_dataloader` | 生产 Trainer 和验证脚本共享的 DataLoader 构造入口 |
| `vibe_emb/trainer.py` | 新增 `DataLoaderWaitTracker` | 记录 consumer 获取 batch 的等待样本并生成平均值/p95 |
| `vibe_emb/trainer.py` | `EmbeddingTrainer.get_train_dataloader` | 调用统一构造入口并打印最终生效配置 |
| `vibe_emb/trainer.py` | `EmbeddingTrainer.training_step` | 在主进程确认 consumption、剥离私有 metadata，再进入父类训练逻辑 |
| `vibe_emb/trainer.py` | `EmbeddingTrainer.get_batch_samples` | 在 `super().get_batch_samples()` 外计时，定义 DataLoader consumer wait |
| `vibe_emb/data.py` | `MultiDatasetBatchDataset.__getitem__` | 删除 consumed counter 修改；返回只读的私有 batch metadata |
| `vibe_emb/data.py` | 新增 `MultiDatasetBatchDataset.record_consumed` | 在主进程按 dataset key/epoch/local count 更新 counters |
| `vibe_emb/data.py` | `format_consumption_stats`、`log_consumption_stats` | 恢复稳定、可检索、包含 scope 的消费日志 |
| `vibe_emb/collator.py` | `EmbeddingCollator.__call__` | 原样携带私有 metadata，不将其转成模型 tensor |
| `vibe_emb/callbacks.py` | `DatasetRefreshCallback.on_epoch_begin` | 保留先 refresh 后 iterator 的顺序，并增加可观察 epoch 日志 |
| `vibe_emb/callbacks.py` | `DatasetStatsCallback` | 输出主进程 consumption 和当前日志窗口 wait 平均值/p95 |

`vibe_emb/arguments.py` 不新增 DataLoader 字段：这些字段已经属于 Transformers
`TrainingArguments`。`data.arrow_prefetch_units` 和 `arrow_local_cache_dir` 不参与本计划。

`vibe_emb/record_store.py` 原则上不改生产逻辑。LRU 上限和 close 行为优先通过现有公开状态及
测试进程观测验证；只有测试无法证明 worker 清理时，才增加只读诊断属性，不增加预取逻辑。

### 3.2 测试和验证工具

| 文件 | 修改或新增内容 |
| --- | --- |
| `tests/f2llm_arrow_test_utils.py` | 从现有测试抽取小型 Arrow manifest/unit 生成 helper，避免提交二进制 fixture |
| `tests/test_indexed_arrow_store.py` | 改用共享 helper；保留现有 lazy/LRU/双 rank 回归；增加 5-unit eviction/close 检查 |
| `tests/test_dataloader_prefetch.py` | 新增配置、参数下传、ahead-of-consumption、统计、等价、epoch、清理和 resume 测试 |
| `tests/fixtures/prefetch_records.jsonl` | 小型、可重复使用的 legacy JSON 数据；通过 `sample_factor` 产生足够的双 rank steps |
| `tests/fixtures/prefetch_train.yaml` | CPU/gloo 双 rank smoke 配置；worker 1、factor 2、persistent false |
| `scripts/verify_dataset_batch_plan.py` | 新增 `--through-dataloader`，通过 `create_train_dataloader` 获取 batch |
| `scripts/f2llm/dataloader_prefetch_benchmark.py` | `prepare` 生成 baseline/candidate YAML；`compare` 自动执行 AC-12 阈值并写 JSON |
| `tests/test_dataloader_prefetch_benchmark.py` | 验证 benchmark compare 的阈值边界和非零退出语义 |

### 3.3 配置和文档

| 文件 | 计划修改时机 |
| --- | --- |
| `vibe_emb/embedding_training_framework.md` | worker 支持合入时更新，不再写死 worker 必须为 0；说明统计确认点和 persistent 限制 |
| `aidoc/f2llm/f2llm_stage2_indexed_arrow_format.md` | 澄清 DataLoader batch prefetch 已支持时，Arrow unit prefetch/node-local cache 仍是后续工作 |
| `aidoc/f2llm/f2llm_stage2_training_code_plan.md` | 更新相关完成状态并链接 spec/implementation plan，不重写历史设计章节 |
| `configs/train_f2llm_stage2_full.yaml` | 只在 AC-1 至 AC-12 全部完成后，以独立 rollout 提交修改 |
| `scripts/run_job.sh` | 预计不改；验证 `TOKENIZERS_PARALLELISM=false` 保持生效 |

## 4. 关键设计决定

### 4.1 TrainingArguments 规范化和 DataLoader 边界校验

`_build_training_args()` 负责对原始配置做尽早失败，并在 worker 大于 0、factor 为 `None` 时，
在构造 `TrainingArguments` 前将 PyTorch 隐式默认值明确写为 factor 2。worker 0 仍要求 factor
为 `None`；persistent true、factor 0/负数或字段类型错误均在模型加载前失败。

组合校验只存在于 `_build_training_args()`。生产训练入口和
`scripts/verify_dataset_batch_plan.py` 都先通过该函数构造 `TrainingArguments`；
`create_train_dataloader()` 是内部构造 helper，直接使用已经规范化的参数，不再重复校验。

不增加 `TrainDataLoaderSettings`。worker/factor/persistent/pin-memory 已经由
`TrainingArguments` 和实际 DataLoader 暴露；multiprocessing context 是构造时固定的 `spawn`
策略。启动日志和 summary 从创建完成的 DataLoader 读取实际属性，避免配置预测值与运行对象
漂移。也不输出 `prefetch_capacity`：它只是 worker 与 factor 的理论乘积，不是实际队列深度。

### 4.2 DataLoader 构造契约

`create_train_dataloader(dataset, collator, args)` 必须保留：

- `SequentialSampler(dataset)`；
- `batch_size=1`；
- `drop_last=args.dataloader_drop_last`；
- `pin_memory=args.dataloader_pin_memory`；
- `in_order=True`；
- non-persistent workers；
- module-level、可 pickle 的 worker init function；
- 与 Transformers 当前 `seed_worker` 等价的 worker seed 初始化。

worker 0 路径不向旧 PyTorch 传入不必要的 multiprocessing-only 参数。worker 大于 0 时才传
显式 `prefetch_factor` 和 worker init；直接调用者传入 factor `None` 时使用 PyTorch 默认值 2。
该 DataLoader 继续不交给 Accelerate 做 batch split；
dataset 已按 rank 生成 local shard，调用 `accelerator.prepare()` 会引入 double-sharding 风险。

默认保持 PyTorch 有序返回。不得启用 `in_order=False` 来换吞吐，因为不同 rank 的完成顺序可能
导致 dataset/task collective 顺序不一致。

### 4.3 Consumption acknowledgement

`MultiDatasetBatchDataset.__getitem__()` 返回的 batch 增加私有 mapping，例如：

```text
_batch_metadata:
  dataset_key
  epoch
  batch_idx
  local_instances
```

该 mapping 满足以下规则：

- 只包含小型 Python 标量，不复制 query/passages/global indices；
- `EmbeddingCollator.__call__()` 原样携带；
- `EmbeddingTrainer.training_step()` 在调用父类前 pop；
- metadata 永远不传入 `EmbeddingModel.forward()`，不改变公开模型签名；
- trainer 调用 `train_dataset.record_consumed(...)` 后再进入现有训练路径；
- `record_consumed()` 校验 dataset key 存在、epoch 等于主进程当前 epoch、count 为正；
- stale epoch metadata 立即失败，可作为误启用 persistent workers 的第二道保护。

`__getitem__()` 不再有任何 consumed 副作用。这样 worker 提前请求、Trainer skip 和
`max_steps` 丢弃都不会污染主进程统计。

### 4.4 Wait time 定义

`EmbeddingTrainer.get_batch_samples()` 围绕父类实现计时。由于当前强制
`gradient_accumulation_steps=1`，一次调用对应一个实际 Trainer batch。指标定义为：

```text
batch_wait = 返回 get_batch_samples 前后的 monotonic wall time
```

该值包含主进程从 DataLoader iterator 获取有序结果的等待和少量 Python 调度开销；不包含
随后 model forward/backward，也不声称是纯存储 I/O 时间。

`DataLoaderWaitTracker` 同时维护：

- 当前 logging window 的精确 count、mean 和 local query count，以及最近 4096 个 p95 样本；
- 整个进程生命周期的样本；
- count、mean、p95；
- 对应实际训练的 local query count。

p95 使用固定算法，例如排序后的 nearest-rank，避免不同依赖库给出不同插值结果。
`DatasetStatsCallback.on_log()` 只通过普通 logger/sidecar 输出，不再次调用 `Trainer.log()`，
避免 callback 递归。

### 4.5 Telemetry artifact

每个 rank 写独立文件，避免多个进程竞争同一文件：

```text
<output_dir>/dataloader_metrics_rank_<rank>.jsonl
<output_dir>/dataloader_metrics_rank_<rank>_summary.json
```

每条 JSONL 对应一个 Trainer logging window，至少包含：

```text
rank, global_step, epoch, wait_count, mean_batch_wait_ms,
p95_batch_wait_ms, consumed_local_batches, consumed_local_instances
```

summary 额外包含 resolved DataLoader settings、进程峰值 RSS、总 wait 和总 query count。
rank 文件是运行 telemetry，不进入 checkpoint，也不参与 resume offset。`train.log` 同时保留一条
可检索的人类可读摘要。

### 4.6 Batch fingerprint

测试和 smoke 工具对以下内容做稳定 JSON 编码后计算 SHA-256：

- epoch、batch index、dataset key；
- global/local record indices；
- 最终 query 文本；
- 最终 positive/negative passage 文本；
- train group size。

生产训练默认不为每个 batch 写 fingerprint，避免大规模运行增加日志和同步开销。fingerprint
JSONL 只由测试、`verify_dataset_batch_plan.py` 和 resume smoke 生成。

## 5. 实施顺序

各阶段必须顺序执行。任何阶段的退出条件未满足时，不进入下一阶段，也不修改完整训练配置。

### 阶段 0：冻结基线和测试夹具

涉及文件：

- `tests/f2llm_arrow_test_utils.py`；
- `tests/test_indexed_arrow_store.py`；
- `tests/fixtures/prefetch_records.jsonl`；
- `tests/fixtures/prefetch_train.yaml`。

实施内容：

1. 从 `tests/test_indexed_arrow_store.py` 抽取 `_sha`、Arrow schema、`_write_unit` 和
   `_manifest` 为共享测试 helper；不改变已有断言。
2. helper 支持按参数生成至少 5 个 unit、每 unit 足够 20-step 测试的 query。
3. 增加稳定 batch fingerprint helper，仅用于测试。
4. 保存 worker 0、seed 固定时 epoch 0/1 的前 20 个 fingerprint 作为测试运行时基准；不提交
   与当前实现细节无关的大型 golden 文件。
5. 创建小型 JSONL/YAML 双 rank fixture，不依赖网络、正式 80 GB 数据或 GPU。

本阶段测试：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_indexed_arrow_store.py
```

退出条件：现有测试数和语义不减少；worker 0 的 batch plan/fingerprint 基线稳定；
`git diff --check` 通过。

### 阶段 1：迁移 consumption accounting，仍保持 worker 0

涉及文件和符号：

- `vibe_emb/data.py::_DatasetStats`；
- `MultiDatasetBatchDataset.__getitem__`；
- 新增 `MultiDatasetBatchDataset.record_consumed`；
- `format_consumption_stats`、`log_consumption_stats`；
- `vibe_emb/collator.py::EmbeddingCollator.__call__`；
- `vibe_emb/trainer.py::EmbeddingTrainer.training_step`；
- `vibe_emb/callbacks.py::DatasetStatsCallback`；
- `tests/test_dataloader_prefetch.py`。

实施内容：

1. 从 `__getitem__()` 删除 consumed counter 更新。
2. 增加 `_batch_metadata`，由 collator 原样传到 trainer。
3. 在 `training_step()` 主进程中 pop metadata、调用 `record_consumed()`，然后调用父类方法。
4. 恢复 `log_consumption_stats()` 的实际日志，明确 planned 是 global scope、consumed 是当前
   rank/local scope。
5. 测试证明：单独调用 dataset/collator 不增加 consumed；调用一个 training step 恰好增加一次；
   metadata 未传入 model。
6. 保留 `_build_training_args()` 对非零 worker 的拒绝，确保本阶段不会开放不完整功能。

本阶段测试：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py \
  tests/test_indexed_arrow_store.py \
  tests/test_modeling.py
```

退出条件：worker 0 前 20 个 fingerprint 与阶段 0 一致；`max_steps=3` 精确记录 3 个实际 batch；
模型 forward 参数集合不变。

### 阶段 2：参数解析和真实 DataLoader prefetch

涉及文件和符号：

- `vibe_emb/train.py::_build_training_args`；
- 新增 `_resolved_training_config`；
- `create_train_dataloader`；
- `EmbeddingTrainer.get_train_dataloader`；
- `tests/test_dataloader_prefetch.py`。

实施内容：

1. 增加 worker/factor/persistent 的显式组合校验，错误发生在 tokenizer/model 加载前。
2. 删除“worker 必须为 0”的旧检查；保留 batch size 和 gradient accumulation 限制。
3. 构造真实 `DataLoader` 并显式使用有序交付、factor、non-persistent worker 和 pin memory。
4. worker 1/factor null 解析并记录为 factor 2；worker 0 日志记录 factor null。
5. `resolved_config.yaml` 写实际生效值，而不是只写输入 YAML。
6. 使用 multiprocessing Event/Queue 编写 ahead-of-consumption 测试；测试不得只依赖 sleep。
7. 检查 DataLoader 对象本身的 `num_workers`、`prefetch_factor`、`persistent_workers`，不以
   YAML 或 `TrainingArguments` 值代替实际下传验证。

本阶段测试：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py \
  tests/test_indexed_arrow_store.py
```

退出条件：SPEC AC-1 至 AC-4 和 AC-7 通过；所有无效配置在 worker spawn 前非零失败；
worker 0 回归通过。

### 阶段 3：Epoch、分布式、LRU 清理和 resume 等价

涉及文件和符号：

- `vibe_emb/callbacks.py::DatasetRefreshCallback.on_epoch_begin`；
- `scripts/verify_dataset_batch_plan.py::_rank_step_snapshot`、`main`；
- `tests/test_dataloader_prefetch.py`；
- `tests/test_indexed_arrow_store.py`。

实施内容：

1. 为 `verify_dataset_batch_plan.py` 增加 `--through-dataloader`；复用生产
   `create_train_dataloader()`，collator 使用 module-level、可 pickle 的 single-item unwrap。
2. 从返回的 `_batch_metadata.batch_idx` 映射回主进程 batch plan，验证 global/local indices。
3. 验证 worker 0 和 worker 1 在 epoch 0/1 的 query/passage/fingerprint 完全一致。
4. 验证 callback refresh 发生在新 epoch iterator 前；persistent true 仍被拒绝。
5. 用 5-unit、LRU=2 的动态 Arrow fixture 触发 eviction；正常耗尽和提前 break 后检查 worker
   liveness、pool 上界和 mmap close。
6. 使用 toy model/本地 tokenizer stub 运行 Trainer 12-step 与 5+resume-to-12 测试；不下载模型。
7. resume 验收将 phase 1 的 5 个实际 batch 与 phase 2 的 7 个实际 batch 合并，再与无中断
   12-step counters 比较；单次恢复进程的 counters 仍只表示该进程实际训练的 batch，不在
   checkpoint 中伪造历史计数。
8. 给 subprocess/gloo 测试设置明确超时，超时视为 collective/worker hang 并打印子进程日志。

本阶段测试：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py \
  tests/test_indexed_arrow_store.py

PYTHONPATH=. /mnt/share/envs/embt/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  scripts/verify_dataset_batch_plan.py \
  --config tests/fixtures/prefetch_train.yaml \
  --steps 20 \
  --through-dataloader
```

退出条件：SPEC AC-5、AC-6、AC-8、AC-9、AC-10 全部通过；双 rank 命令在超时内打印
`OK: verified 20 distributed dataloader steps` 并退出 0。

### 阶段 4：Wait telemetry 和 benchmark 工具

涉及文件和符号：

- `vibe_emb/trainer.py::DataLoaderWaitTracker`；
- `EmbeddingTrainer.get_batch_samples`；
- `vibe_emb/callbacks.py::DatasetStatsCallback`；
- `vibe_emb/train.py::EmbeddingTrainRunner.run`；
- `scripts/f2llm/dataloader_prefetch_benchmark.py`；
- `tests/test_dataloader_prefetch.py`；
- `tests/test_dataloader_prefetch_benchmark.py`。

实施内容：

1. 对父类 `get_batch_samples()` 计时；记录当前 window 和全程统计。
2. callback 按 Trainer logging cadence 输出稳定日志和 rank-specific JSONL。
3. 训练结束时写 summary JSON；异常退出时尽最大可能 flush 已完成窗口，不覆盖 checkpoint。
4. 记录 resolved settings、wait count/mean/p95、local query count 和 rank RSS。
5. benchmark `prepare` 从目标 YAML 生成两份只改变以下字段的 resolved YAML：
   - baseline：worker 0/factor null/block 1；
   - candidate：worker 1/factor 2/block 8。
6. benchmark `compare` 读取两次运行的所有 rank metrics，丢弃 steps 1-20，计算 steps 21-200
   的 wait、step time、query/s、RSS，并按 AC-12 自动返回 0/非零。
7. threshold tests 覆盖恰好 20%、95%、2 GiB 边界以及缺 rank、缺 step、NaN、运行错误。

本阶段测试：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py \
  tests/test_dataloader_prefetch_benchmark.py
```

退出条件：日志和 JSON 字段可以机器解析；p95 算法固定；AC-12 comparator 对通过/失败 fixture
返回正确退出码；telemetry 开启后 worker 0 fingerprint 不变。

### 阶段 5：文档同步和 20-step 真实 smoke

涉及文件：

- `vibe_emb/embedding_training_framework.md`；
- `aidoc/f2llm/f2llm_stage2_indexed_arrow_format.md`；
- `aidoc/f2llm/f2llm_stage2_training_code_plan.md`；
- `scripts/run_job.sh`，只验证不预期修改。

实施内容：

1. 更新 worker、factor、统计确认点、persistent 限制和资源上界说明。
2. 明确 PyTorch DataLoader batch prefetch 已实现时，Arrow unit prefetch 和 node-local cache
   仍不在本需求范围内。
3. 使用完整 Indexed Arrow manifest、真实 tokenizer/model 和独立 `/tmp` 输出目录运行 20 steps。
4. 检查有限 loss、worker 退出、wait/consumption 日志、summary JSON 和 checkpoint 重新识别。
5. smoke 先使用临时 candidate YAML，不修改正式 `configs/train_f2llm_stage2_full.yaml`。

本阶段验证：

```bash
CONFIG=<candidate-yaml> \
MAX_STEPS=20 \
OUTPUT_DIR=/tmp/f2llm-prefetch-smoke \
OVERWRITE_OUTPUT_DIR=true \
bash scripts/run_job.sh
```

同时执行：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py \
  tests/test_dataloader_prefetch_benchmark.py \
  tests/test_indexed_arrow_store.py \
  tests/test_modeling.py

PYTHONPATH=. /mnt/share/envs/embt/bin/python -m py_compile \
  vibe_emb/train.py vibe_emb/trainer.py vibe_emb/data.py \
  vibe_emb/collator.py vibe_emb/callbacks.py \
  scripts/verify_dataset_batch_plan.py \
  scripts/f2llm/dataloader_prefetch_benchmark.py

git diff --check
```

退出条件：SPEC AC-11 通过；相关文档与实现无冲突；未修改正式训练配置。

### 阶段 6：QuarkFS A/B 和配置 rollout

涉及文件：

- benchmark 工具生成的 baseline/candidate YAML 和 metrics artifacts；
- `configs/train_f2llm_stage2_full.yaml`，仅当 comparator 退出 0 时修改。

实施内容：

1. 在同一节点数、GPU/rank 数、CPU 配额、数据快照和 seed 下各运行 200 steps。
2. 保存命令、resolved YAML、train log、所有 rank metrics 和 comparator JSON。
3. comparator 自动执行 AC-12；人工复核运行环境一致和无外部共享盘异常。
4. 通过后单独修改正式配置：worker 1、factor 2、persistent false、block 8。
5. 未通过则正式配置保持 worker 0；功能代码和测试可以合入，但文档记录候选未 rollout。
6. 配置提交后再从已有 checkpoint 做一次短 resume smoke，证明 worker 模式切换不影响恢复。

本阶段验证命令由 benchmark `prepare` 输出，比较命令固定为：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python \
  scripts/f2llm/dataloader_prefetch_benchmark.py compare \
  --baseline-output <baseline-output-dir> \
  --candidate-output <candidate-output-dir> \
  --output <comparison.json>
```

退出条件：comparison JSON 包含所有 AC-12 字段；正式配置修改与 comparator 退出 0 在同一
rollout 证据中可追溯。

## 6. 测试矩阵

| 验收项 | 主要测试/命令 | 必须观察的结果 |
| --- | --- | --- |
| AC-1 | `test_dataloader_prefetch.py` worker 0 两次运行 | resolved factor null；前 20 fingerprint 相同 |
| AC-2 | 实际 DataLoader 属性断言 | worker=1、factor=2、persistent false |
| AC-3 | 参数化 config 初始化测试 | 三种非法组合均在模型/worker 前失败 |
| AC-4 | multiprocessing Event/Queue 测试 | factor 2 时主消费前两个任务 ahead；消费后补一个任务 |
| AC-5 | 动态 Indexed Arrow worker 0/1 对比 | dataset/index/text/group fingerprint 逐项相同 |
| AC-6 | 2-rank `--through-dataloader` | 20 step 对齐、无 overlap、退出 0、无 hang |
| AC-7 | toy Trainer max_steps=3 | counters 精确为 3 batch/12 instances |
| AC-8 | 两 epoch、两次 seed 重跑 | 跨 epoch 不同；同 epoch 可复现；worker 0/1 一致 |
| AC-9 | 12 vs 5+resume | step 6-12 fingerprint/shape 一致，合计 counters 一致 |
| AC-10 | 5 unit、LRU=2、提前 break | pool 不超 2；worker 退出；mmap 关闭 |
| AC-11 | 真实模型/Arrow 20-step smoke | finite loss、日志/metrics/checkpoint 均存在且可识别 |
| AC-12 | QuarkFS 200-step A/B comparator | wait/throughput/RSS/错误条件全部机器判定 |

测试必须使用 `/mnt/share/envs/embt`。单元和双 rank fixture 不访问网络；真实模型 smoke 使用
已有本地模型和 Arrow 数据。任何涉及网络的补充操作按仓库约定清理全部大小写代理变量，但本
需求预期不需要网络。

## 7. 兼容策略

1. **默认兼容**：`_build_training_args()` 继续默认 worker 0；旧 YAML 无需增加 factor 字段。
2. **模型兼容**：私有 `_batch_metadata` 在 `training_step()` 中删除，不修改
   `EmbeddingModel.forward()`。
3. **数据兼容**：不修改 manifest、Arrow schema、fingerprint、sample/doc ID 或 sampling plan。
4. **顺序兼容**：始终使用 `SequentialSampler` 和有序交付；seed 不引入 worker ID。
5. **checkpoint 兼容**：worker/factor/queue 不写入 checkpoint state；worker 0/1 可交叉 resume。
6. **统计兼容**：counter 含义收紧为“实际进入 training step”，不再表示 dataset 被请求；日志中
   明确 scope。resume 的不同进程统计通过 benchmark/test 聚合，不伪造历史。
7. **配置兼容**：显式非法组合失败，不静默把 factor 丢弃或把 persistent 降级为 false。
8. **环境兼容**：以当前 PyTorch 2.10/Transformers 5.8 为目标，只使用公开 DataLoader 参数。
9. **Accelerate 边界**：不让 Accelerate 再切 DataLoader batch，保持现有 rank-local dataset
   设计。
10. **配置 rollout 隔离**：功能代码与 F2LLM 正式配置分提交；性能不达标时无需代码回滚。

## 8. 风险与缓解

| 风险 | 触发方式 | 缓解和验证 |
| --- | --- | --- |
| worker 副本统计丢失 | counter 仍在 `__getitem__` 修改 | 阶段 1 先迁移到 `training_step`，开放 worker 前完成 AC-7 |
| prefetched batch 被误计 | max_steps/异常时队列有未消费结果 | main-process acknowledgement；测试提前停止时 worker 已准备额外 batch |
| stale epoch plan | persistent worker 保留旧 dataset | 启动拒绝 persistent；metadata epoch 二次校验；AC-8 |
| rank 顺序漂移/collective hang | unordered return 或 worker failure | `in_order=True`、SequentialSampler、2-rank timeout smoke |
| Arrow LRU 资源倍增 | worker 数乘以 `arrow_max_open_units` | 初始只 rollout worker 1；记录 RSS/worker；AC-10/AC-12 |
| QuarkFS 请求放大 | 多 rank 同时 prime queue | factor 固定 2、block 8、真实共享盘 A/B；未通过不改配置 |
| CPU 过量订阅 | 每 rank tokenizer worker | 保持 `TOKENIZERS_PARALLELISM=false`；记录拓扑和 query/s |
| fork 后库状态问题 | tokenizer/model/CUDA 初始化后创建 worker | 在真实 smoke 检查 worker crash/hang；multiprocessing context 需人工决定 |
| spawn 序列化开销 | 强制 spawn 时 pickle dataset/tokenizer | 在 A/B 中记录 cold start；fixture 验证 picklability |
| wait 指标误读 | 指标含 queue/Python 而非纯 I/O | 文档固定定义；只用于同环境 A/B，不声称存储分解 |
| logging_steps=1 时单窗口 p95 无意义 | 一个窗口只有一个样本 | sidecar 保留逐窗口数据；comparator 聚合 steps 21-200 后算 p95 |
| worker 清理依赖 GC | Trainer 提前 break | AC-10 检查存活子进程；必要时仅在有公开生命周期点时加显式 close |
| resume counter 语义混淆 | phase 2 只看到新进程训练量 | test/comparator 合并 phase artifacts；checkpoint 不保存 prefetch counter |
| telemetry 写盘影响性能 | 每 step 写 JSONL | 每 rank 独立缓冲写；A/B 同时开启；必要时按 logging window 批量 flush |
| 工作区已有无关改动 | 提交时混入其他任务 | 每 checkpoint 只 stage 本计划列出的路径，提交前逐文件审 diff |

## 9. 提交检查点

每个检查点必须可独立回滚并保持测试绿色。不要提交失败测试，也不要把正式配置 rollout 混入
功能提交。

### CP-1：Consumption semantics

建议提交主题：`refactor(train): acknowledge dataset consumption in trainer`

包含：

- dataset metadata 和 `record_consumed`；
- collator metadata passthrough；
- trainer main-process acknowledgement；
- 恢复 consumption logging；
- worker 0 测试及共享 Arrow test helper。

不包含：解除 worker 0 限制。

提交前检查：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py tests/test_indexed_arrow_store.py tests/test_modeling.py
git diff --check
```

### CP-2：Bounded DataLoader prefetch

建议提交主题：`feat(train): support bounded dataloader batch prefetch`

包含：

- config validation/resolution；
- DataLoader factory/settings；
- worker/factor/persistent 下传；
- resolved settings 日志/YAML；
- AC-1 至 AC-4 和 AC-7 测试。

不包含：正式 F2LLM 配置修改。

### CP-3：Distributed and resume verification

建议提交主题：`test(train): verify prefetch ordering epoch and resume invariants`

包含：

- `--through-dataloader`；
- 双 rank fixture；
- epoch、fingerprint、LRU cleanup 和 resume 测试；
- 为测试暴露的最小只读诊断接口（如果确有需要）。

提交前必须运行双 rank 命令，不能只运行 pytest 单进程用例。

### CP-4：Telemetry and benchmark gate

建议提交主题：`feat(train): report dataloader wait and benchmark prefetch`

包含：

- wait tracker、rank JSONL/summary；
- benchmark prepare/compare；
- threshold 测试；
- 框架和 F2LLM 文档同步。

不包含：A/B 结果产物和正式配置修改。大型日志、临时 YAML、模型 checkpoint 不提交。

### CP-5：F2LLM configuration rollout

建议提交主题：`perf(f2llm): enable validated dataloader prefetch profile`

前置条件：AC-11/AC-12 证据齐全且 comparator 退出 0。

只包含：

- `configs/train_f2llm_stage2_full.yaml` 的 worker/factor/persistent/block 变更；
- 必要的短说明或链接到不提交大产物的 benchmark summary。

该提交是首选回滚点。出现生产回归时先回滚 CP-5，保留 CP-1 至 CP-4 的通用能力。

## 10. 需要人工决定的事项

以下事项无法只从当前代码静态决定。未标注“阻塞”的项目不影响 CP-1/CP-2 开发，但必须在对应
阶段前确认。

### D-1：DataLoader multiprocessing context（阻塞 CP-2 最终定型）

候选：

- 保持 Linux/PyTorch 默认 `fork`：启动快、复制开销小，但 worker 在 tokenizer/model/CUDA
  初始化后创建，需要重点验证 fork 后库状态；
- 显式 `spawn`：隔离 CUDA/fork 风险，但要序列化 dataset、tokenizer/collator，启动和内存开销
  更高；
- `forkserver`：折中，但会增加部署和可用性假设。

建议先用默认 `fork` 完成功能测试，同时用真实 smoke 比较 `fork` 与 `spawn` 的启动稳定性；
最终只保留一个生产默认，不按机器隐式切换。

### D-2：峰值 RSS 的统计口径（阻塞 AC-12）

需要确定 2 GiB 阈值比较：

1. 仅训练 rank 主进程 RSS；还是
2. rank 主进程加其 DataLoader worker process tree 的总 RSS。

建议采用第二种，它更能反映 prefetch 真实成本；如果使用 `psutil`，需决定是否将其声明为正式
运行依赖，或只让 benchmark 工具使用当前环境已有的 psutil。

### D-3：QuarkFS A/B 的生产拓扑和执行窗口（阻塞阶段 6）

需要指定：节点数、每节点 rank 数、CPU quota、同机其他负载要求、数据 cache 冷/热状态、运行
时段和结果保存位置。spec 阈值已确定，但拓扑不一致会让比较失效。

### D-4：是否增加第三个局部性对照组（不阻塞功能）

spec 的 baseline 与 candidate 同时改变 worker 和 `unit_block_batches`，能够判断最终组合是否值得
rollout，但不能区分收益来自 worker 还是 block locality。建议人工决定是否额外运行：

```text
worker 0, factor null, unit_block_batches 8
```

该组不改变 AC-12 判定，只帮助后续选择更简单的优化。

### D-5：Telemetry artifact 保留策略（阻塞 CP-4 运维细节）

需要决定 rank JSONL 是否对所有完整训练默认保留，以及是否设置大小/轮转上限。当前配置
`logging_steps=1` 时，两 epoch 长训练会产生大量记录。建议保留 window aggregates、避免逐 batch
明细文本，并在完整运行前估算文件大小。

### D-6：正式配置启用审批（阻塞 CP-5）

即使 comparator 退出 0，仍需人工确认 A/B 没有共享存储异常、抢占或其他外部干扰，然后批准
修改 `configs/train_f2llm_stage2_full.yaml`。未批准时框架能力可合入，但配置保持 worker 0。

### D-7：Spec 状态字段维护（不阻塞代码）

用户已确认 spec 内容，但 spec 文件当前状态仍写为 `Draft`。需要人工决定在实施开始前是否将其
改为 `Accepted`，以及由谁维护验收证据链接；本计划不自行修改已确认 spec。

## 11. 最终交付检查

实施完成时必须同时满足：

- CP-1 至 CP-4 的相关测试全部绿色；
- 双 rank DataLoader smoke 实际执行并退出 0；
- worker 0 与 worker 1 的 epoch/batch/resume fingerprint 等价；
- consumption 不包含 prefetched-but-untrained batch；
- resolved YAML、启动日志和实际 DataLoader 属性一致；
- 20-step 真实模型 smoke 有明确退出码和 finite loss；
- QuarkFS A/B comparator 生成机器可读结果；
- 正式配置是否 rollout 与 comparator/人工审批一致；
- `py_compile`、相关 pytest、`git diff --check` 全部通过；
- 未提交 `/tmp` smoke checkpoint、训练日志、benchmark 大产物、cache 或无关工作区修改。

只要 AC-5/6/7/8/9 任一正确性项失败，就停止性能优化和配置 rollout；不得以吞吐提升覆盖数据
顺序、统计或 resume 漂移。

## 12. 实施记录（2026-07-20）

### 12.1 检查点状态

| 检查点 | 状态 | 实施结果 |
| --- | --- | --- |
| CP-1 | 完成 | consumption 从 worker 可执行的 `__getitem__` 迁到主进程 `training_step`；私有 metadata 不进入模型 |
| CP-2 | 完成 | 支持 worker/factor/pin-memory；拒绝 persistent；worker 0 保持默认；resolved YAML 和启动日志记录实际值 |
| CP-3 | 部分完成 | 20-step 双 rank、epoch、20-batch Indexed Arrow 等价、LRU/worker 清理和 installed Accelerate resume offset 已验证；真实 Trainer checkpoint resume 未运行 |
| CP-4 | 完成 | 每 rank JSONL/summary、wait/consumption/RSS、A/B prepare/compare 和阈值测试已实现 |
| CP-5 | 部分完成 | 用户明确要求后已完成两卡 20-step smoke、54-step 中止观察、block 1 的 100-step prefetch factor 2/4/off 和 block 8 无 prefetch 100-step；QuarkFS 200-step A/B 尚未执行 |

`configs/train_f2llm_stage2_full.yaml` 在实施前已有 `arrow_max_open_units=32`、专用 output directory
等工作区修改。随后用户在 2026-07-20 明确要求修改正式配置并启动两卡观察，因此本次追加了
worker 1、factor 2、persistent false，并先将 `unit_block_batches` 设为 8；已有字段保持不动。完成
20-step smoke 后，用户又将 `unit_block_batches` 改为 1 并要求单独观察。当前正式 YAML 因而保留
block 1，但该值未通过 AC-12，不能视为已验收的 rollout 候选。随后用户要求关闭 prefetch 做
同口径对照，正式 YAML 当前恢复为 worker 0/factor null/block 1；该状态是实验后的 baseline，
不是对 candidate 的 AC-12 验收结论。用户随后又要求在保持 prefetch 关闭时改为 block 8；该次
观察使用 worker 0/factor null/block 8，用于 locality 对照，仍不是最终 rollout 结论。
用户最新又要求测试 block 1/factor 4，因此正式 YAML 当前为 worker 1/factor 4/block 1；该状态仅是
最近一次实验输入，不代表 AC-12 已通过或已经批准完整训练 rollout。

### 12.2 已定设计事项

1. **D-1 选择 `spawn`**：早期用默认 `fork` 跑 worker 测试虽然通过，但 Python 3.12 明确警告
   多线程进程中 `fork()` 可能造成 child deadlock。训练 DataLoader 在 tokenizer、模型和 CUDA
   初始化后才创建 worker，因此生产构造固定使用 `spawn`。代价是本环境每次 worker 冷启动约
   50--90 秒；该启动成本必须在真实 smoke/A/B 中继续观察。
2. **D-2 使用 rank process tree RSS**：环境存在 `psutil` 时采样 rank 主进程和递归 worker 的
   当前 RSS，并保存运行期间采样最大值；没有 `psutil` 时退化为 Linux `ru_maxrss` 的 rank
   主进程峰值。summary 的 `rss_scope` 明确记录所用口径，不把 `psutil` 设为硬依赖。
3. **D-5 当前保留 window aggregates**：每个 Trainer logging window 写一条 rank-specific
   JSONL，每 100 行 flush；每次训练进程以覆盖模式创建 JSONL，避免复用 output directory 时
   混入旧行。跨 resume invocation 的比较由调用方保存和聚合各阶段 artifact。
4. lifetime 和当前 logging window 的 wait count、mean、local query count 都精确累计；二者的
   p95 都只保留最近 4096 个样本，并写 `p95_sample_count`。这样即使关闭 logging 或把日志间隔
   设得很大，也不会按 batch 无界保存 Python 样本。

### 12.3 相对原计划的偏离

1. 阶段 3 原计划运行 toy Trainer 的 `12 step` 与 `5 + checkpoint resume to 12`。当前改为直接
   使用安装版本 Trainer 实际调用的 `accelerate.skip_first_batches`，验证 worker 1 下
   `5 + 7` 与无中断 12-batch 的 dataset/global/local index、文本 fingerprint、collator shape
   和 consumption 合计一致。原因是虚构 toy checkpoint 不能覆盖当前 PEFT adapter checkpoint
   加载语义；真实模型 checkpoint 验证又属于本轮明确禁止的昂贵训练。因此 AC-9 的数据偏移
   部分有自动化证据，但 checkpoint 保存/加载、loss 和真实 global-step 恢复仍留给 AC-11。
2. 初次实施没有执行真实 smoke/A/B，以遵守“不要启动昂贵的完整训练”。随后用户明确要求修改
   正式配置并启动两卡观察，因此补做了限制为 20 optimizer steps 的 AC-11 smoke；没有扩展到
   完整两 epoch。QuarkFS 200-step baseline/candidate A/B 仍未执行，AC-12 尚未验收。
3. 原计划建议先用默认 `fork` 再人工定型；实施中因可复现的 Python 3.12 deadlock warning 直接
   选择 `spawn`，没有保留按机器切换的隐式分支。选择理由和性能风险见 12.2。
4. 没有创建计划中的物理 git commits。当前是包含其他任务改动的共享 dirty worktree，且本次
   用户未要求提交；CP-1 至 CP-4 仅作为逻辑审查检查点记录，避免把无关文件混入提交。
5. 实际运行和代码复审后，用户确认 `TrainDataLoaderSettings` 与 `prefetch_capacity` 没有独立
   运行时职责。实现改为在 `_build_training_args()` 显式规范化和校验 worker/factor/persistent，
   `create_train_dataloader()` 只消费已规范化参数，并从实际 DataLoader 属性生成日志和 summary。
   该调整删除了主路径的重复 resolved object、重复边界校验和理论 capacity 输出；worker/factor
   下传、ahead-of-consumption、顺序、consumption、wait/RSS telemetry 等功能与行为测试保持
   不变。SPEC、计划主体和测试断言已同步更新，不通过删除功能测试绕过失败。

### 12.4 实际验证证据

实施前基线：

```text
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_indexed_arrow_store.py tests/test_modeling.py
9 passed, 4 warnings in 58.04s
```

当前修订的相关 pytest 分为非 worker 和 worker 两组执行，以便单独观察慢速 `spawn` 生命周期：

```text
34 passed, 5 deselected, 4 warnings in 54.77s
5 passed, 14 deselected in 356.01s
```

两组的选择集互补，合计覆盖当前相关的 39 个 pytest case；没有删除、skip 或放宽失败测试。
其中外部可观察断言包括：启动日志 resolved settings、worker ahead-of-consumption、精确
`3 batches / 12 instances`、worker 0/1 前 20 batch fingerprint、5-unit LRU=2、文件立即
重命名/删除、epoch 重建、resume offset、JSONL/summary 和 A/B 阈值非零失败语义。

双 rank 命令：

```text
timeout 300s env CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
  /mnt/share/envs/embt/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 scripts/verify_dataset_batch_plan.py \
  --config tests/fixtures/prefetch_train.yaml --steps 20 --through-dataloader

OK: verified 20 distributed dataloader steps from tests/fixtures/prefetch_train.yaml \
with world_size=2, epoch=0.
```

该 verifier 只验证 DataLoader 构造和 rank plan，不调用 `training_step`，所以它打印的
`consumed_local_batches=0` 是预期结果，不作为 AC-7 证据；AC-7 由 worker consumption pytest
验证。

以下静态检查也已退出 0：

```text
python -m py_compile <本计划涉及的 production/script/test Python 文件>
git diff --check
```

### 12.5 仍需人工决定或执行

- 确定 D-3 的 QuarkFS A/B 拓扑、cache 冷热状态和 artifact 保存位置，再执行 AC-12。
- 决定是否增加 D-4 的 `worker 0 + block 8` 第三对照组。
- 审核长期训练 JSONL 保留策略；当前每 logging window 一行、每次 invocation 覆盖同 rank 文件。
- 正式配置经过用户要求的观察后当前为 worker 1/factor 4/block 1；prefetch candidate 在
  AC-12 完成前仍是实验性组合，不应直接用于完整训练 rollout。
- D-7 仍未处理：已确认 SPEC 的状态字段是否由 `Draft` 改为 `Accepted` 由文档维护者决定。

### 12.6 两卡真实 smoke 结果（2026-07-20）

执行命令使用本地 Qwen3-0.6B、完整 395-unit manifest、GPU 0/1，并通过 CLI 将训练限制为
`max_steps=20`；正式 YAML 的两 epoch设置没有被运行到底。结果：

- torchrun 退出码 0，NCCL 2.28.9 初始化成功，无 OOM、collective、worker 或 Arrow 错误；
- 两个 rank 均使用 worker 1/factor 2/spawn，并精确消费 20 batch、640 local query；
- 20 个记录 loss 全部有限，范围 5.946--9.302，最终 `train_loss=7.6071`；
- 第一个 batch wait 约 3176 ms，包含 worker 冷启动；steps 2--20 平均 wait 为
  rank 0 `0.206 ms`、rank 1 `0.169 ms`，最大值为 `0.562/0.427 ms`；
- 观察到两卡训练中利用率 100%，显存约 45.5/51.2 GiB；rank+worker process-tree 的逐步
  RSS 最大观测约 4.86/4.88 GiB；
- `results/f2llm-s2-prefetch/checkpoint-20` 被 `get_last_checkpoint()` 识别，root/checkpoint 的
  PEFT config 均可本地重载，optimizer/scheduler/RNG/trainer state 均存在；
- 正常退出后 GPU 显存归零，torchrun、rank 和 DataLoader worker 均无残留。

真实运行还发现并修复两个 telemetry 边界：Trainer 最终 aggregate log 会对 step 20 再触发一次
无 batch 的 `on_log`，原实现会写重复零样本行；summary 原实现把最终 RSS 写到
`peak_rss_bytes`，丢失训练中 worker 存活时的采样峰值。修复后无新 wait sample 的同-step log
被忽略，summary 使用累计 `_peak_rss_bytes`。同时将每步 395-source consumption 长日志收敛为
aggregate totals；per-source planned 详情仍在 epoch plan 日志中保留。

修复后的 callback/benchmark 回归为 `11 passed in 51.84s`；实际 worker consumption 用例为
`1 passed in 113.71s`。中间一次该用例因仍匹配旧逗号格式而失败，数值输出为正确的
`3 batches / 12 instances`；测试随后更新为新 aggregate 日志的精确格式并原样重跑通过，未删除
或弱化数值断言。

### 12.7 block 8 中止观察与 block 1 两卡 100-step 结果（2026-07-20）

用户先要求按 block 8 运行 200 steps，随后在 step 54 完成后要求停止，并把
`unit_block_batches` 改为 1 后重新运行 100 steps。两次都使用 worker 1/factor 2、
spawn、GPU 0/1 和完整 395-unit manifest，分别写入独立 output directory，未续跑 20-step
smoke checkpoint：

- block 8 的 200-step 请求在 step 54 后收到人工 `SIGINT`，torchrun 退出码 1 是预期的
  `KeyboardInterrupt`，不是 worker/NCCL/Arrow 失败；该运行的 metrics/summary 已保留，但因
  `save_steps=500` 没有模型 checkpoint；
- block 8 的已完成 54 steps 中，去掉首步冷启动后的 wait mean 为 rank 0 `0.164 ms`、rank 1
  `0.170 ms`，p95 为 `0.196/0.223 ms`，最大值为 `0.440/0.312 ms`，没有 >1 ms 尖峰；
- block 1 的 100-step 运行正常退出 0，`train_runtime=720.3 s`、`0.139 step/s`、
  `train_loss=6.068`；100 个 loss 全部有限，范围 `0.673--10.988`；
- block 1 两个 rank 均精确写出 100 个唯一 step、消费 100 local batches/3200 local instances；
  warm wait mean 为 `403.126/401.605 ms`，p50 为 `0.117/0.121 ms`，p95 为
  `103.014/93.930 ms`，p99 约 `11.53/11.50 s`，最大值为 `13.736/13.686 s`；
- block 1 的 rank 0 在 steps `2/47/51/76/87` 分别出现约
  `9.859/1.028/11.486/3.788/13.736 s` wait，rank 1 在相同步出现对应尖峰；去掉首步后每 rank
  累计等待约 `39.9/39.8 s`，约占 720.3 秒训练 runtime 的 5.5%；
- block 1 的 rank process-tree peak RSS 为 `19.84/19.85 GB`，block 8 的 54-step 部分运行为
  `6.08/6.03 GB`。该 RSS 是进程树 RSS 求和，可能重复计算共享映射，且两次运行步数不同；只能
  作为随机跨 unit 更快扩大 Arrow LRU 工作集的风险信号，不能当作严格内存 A/B；
- `checkpoint-100` 和 root adapter 均可通过 `PeftConfig.from_pretrained()` 本地读取，base model
  为 Qwen3-0.6B，LoRA `r=64/alpha=64`；`get_last_checkpoint()` 正确返回 `checkpoint-100`；
  正常退出后两卡显存归零，无 torchrun、rank 或 worker 残留。

这两次运行的 batch 顺序、样本长度分布和完成步数不同，不能用于 AC-12 的吞吐判定，也不能
证明 block 8 的端到端吞吐提升比例；AC-12 仍需同口径 200-step baseline/candidate comparator。
但 block 1 在两个 rank 的相同步发生 5 次稳态 I/O 尖峰，而 block 8 的部分运行无 >1 ms 尖峰，
已经表明当前 `arrow_max_open_units=32 + worker 1 + factor 2` 下 block 1 会破坏 unit locality，当前
不应把 block 1 视为 prefetch rollout 的推荐组合。

### 12.8 block 1 关闭 prefetch 的两卡 100-step 对照（2026-07-20）

在 12.7 的 block 1 运行完成后，用户要求关闭 prefetch 再跑一次。正式 YAML 改为 worker 0、
factor null、persistent false，保持 block 1、seed、模型、完整 manifest、GPU 0/1 和其余训练参数
不变；新运行使用独立 output directory，未续跑任何 checkpoint。日志确认实际 DataLoader 为
`num_workers=0/prefetch_factor=null/multiprocessing_context=none`。

- 运行正常退出 0，`train_runtime=883.8 s`、`0.113 step/s`、`train_loss=6.0672`；100 个 loss
  全部有限，两个 rank 均精确写出 100 个唯一 step，并消费 100 local batches/3200 instances；
- prefetch-on 同口径 block 1 运行是 `720.3 s`、`0.139 step/s`。关闭后总时长增加 `163.5 s`
  或 `22.7%`，吞吐下降约 `18.7%`；反向表述为开启 prefetch 将 baseline runtime 缩短约
  `18.5%`，吞吐提高约 `23.0%`；
- prefetch-off 的 warm wait mean 为 rank 0/1 `2518.840/2525.676 ms`，p50 为
  `883.391/899.914 ms`，p95 为 `10.350/10.272 s`，p99 约 `20.42/20.41 s`，最大值为
  `26.040/25.964 s`；99 个 warm step 全部 >1 ms，43 个 >1 s，累计 wait 为
  `249.4/250.0 s`；
- prefetch-on 的对应 warm wait mean 为 `403.126/401.605 ms`，p50 约 `0.12 ms`，累计 wait
  `39.9/39.8 s`。因此 worker 1/factor 2 把每 rank 约 `209.5/210.3 s` 的同步 DataLoader wait
  从主训练路径隐藏或并行化；其约 50 秒 spawn 冷启动抵消了部分短跑收益，但 100 steps 后净收益
  仍为 163.5 秒；
- 10 秒 GPU utilization 采样在同步 I/O 间隙捕获到 SM 利用率从 100% 降到 1%/8% 的时刻，和
  telemetry 的 10--26 秒 wait 尖峰一致；
- prefetch-off 的 rank process-tree peak RSS 为 `18.81/18.80 GB`，prefetch-on 为
  `19.84/19.85 GB`。当前组合的 prefetch 内存代价约 `1.0 GB/rank`；RSS 仍可能重复计算共享
  映射，只用于同口径相对比较；
- `checkpoint-100` 和 root adapter 均可通过 `PeftConfig.from_pretrained()` 本地读取，LoRA
  `r=64/alpha=64`，`get_last_checkpoint()` 正确返回 `checkpoint-100`；结束后 GPU 显存归零，
  无 torchrun、rank 或 worker 残留。

该 100-step on/off 比较使用相同 block 1 batch plan，比 12.7 的不同 block 部分运行更适合判断
PyTorch DataLoader prefetch 的实际作用；结果明确支持 prefetch 能减少同步读取拖慢。但它仍不满足
AC-12 规定的 200 steps、丢弃前 20 steps 和正式 comparator 门槛，且 block 1 本身的 unit locality
较差，因此推荐组合仍需用 block 8 执行正式 baseline/candidate A/B。

### 12.9 block 8 关闭 prefetch 的两卡 100-step 结果（2026-07-20）

用户进一步要求保持 worker 0/factor null，只把 `unit_block_batches` 从 1 改为 8，再执行双卡
100 steps。模型、seed、完整 manifest、GPU 0/1 和其他训练参数保持不变，运行写入独立 output
directory，未续跑已有 checkpoint；日志确认 DataLoader 仍为 worker 0/factor null。

- 运行正常退出 0，`train_runtime=633.4 s`、`0.158 step/s`、`train_loss=5.5367`；100 个 loss
  全部有限，范围 `0.966--9.305`，两个 rank 均精确写出 100 个唯一 step，并消费
  100 local batches/3200 instances；
- warm wait mean 为 rank 0/1 `347.413/344.430 ms`，p50 为 `195.438/192.558 ms`，p95 为
  `1.095/1.106 s`，p99 为 `2.738/2.708 s`，最大值为 `5.306/5.283 s`，累计 wait 为
  `34.39/34.10 s`；每 rank 仅 6 个 warm step >1 s；
- block 1 无 prefetch 的 warm wait mean 为约 `2.52 s`、p50 约 `0.89 s`、p95 约 `10.3 s`、
  最大约 `26.0 s`、累计约 `249--250 s/rank`。block 8 将累计同步 wait 降低约 86%，并把昂贵的
  unit 打开集中到 block 边界；同一 unit 内常见 wait 约 `0.01--0.4 s`；
- block 8 无 prefetch 的 rank process-tree peak RSS 为 `5.53/5.53 GB`，block 1 无 prefetch 为
  `18.81/18.80 GB`。block 8 在前 100 steps 触碰的并发 unit 工作集更小；RSS 仍可能重复计算
  共享映射，仅用于同口径趋势判断；
- block 8 与 block 1 的 100-step runtime 分别为 `633.4/883.8 s`，前者短 `250.4 s`；但 block
  参数会改变前 100 steps 的样本和长度顺序，这个总时间差包含计算量组合差异，不能把 28.3%
  runtime 降幅全部归因于 locality；wait 分布是更直接的 I/O 证据；
- 对相同 block 8 前 54 steps，prefetch-off/on 的 telemetry elapsed 分别为约 `365.95/395.04 s`；
  prefetch-on 的 warm wait 几乎为零，而 off 累计 wait 约 `23.5/23.2 s/rank`，但 spawn 冷启动
  约 50--60 秒，在该短跑中超过了已隐藏的 I/O，因此 on 反而慢约 29 秒。完整长训练中启动成本
  会被摊薄，不能据此否定 prefetch；
- `checkpoint-100` 和 root adapter 均可通过 `PeftConfig.from_pretrained()` 本地读取，LoRA
  `r=64/alpha=64`，`get_last_checkpoint()` 正确返回 `checkpoint-100`；结束后 GPU 显存归零，
  无 torchrun、rank 或 worker 残留。

该结果说明 block 8 本身已经大幅缓解同步 I/O，且短跑中 worker spawn 成本不可忽略。由于各次
运行按顺序执行、没有严格控制 OS page cache，且正式 AC-12 要比较 steps 21--200 的稳态窗口，
最终是否同时启用 prefetch 仍需按 200-step comparator 判定；该次结果记录时正式 YAML 为
worker 0/factor null/block 8，之后的最新配置状态见 12.10。

### 12.10 block 1、prefetch factor 4 的两卡 100-step 结果（2026-07-20）

用户要求把 block 1 的预取深度从 factor 2 增加到 factor 4，再执行双卡 100 steps。正式 YAML
改为 worker 1/factor 4/persistent false/block 1；模型、seed、完整 manifest、GPU 0/1 和其他训练
参数保持不变，运行写入独立 output directory，未续跑已有 checkpoint。resolved YAML 和启动日志
均确认实际 DataLoader 为 `num_workers=1/prefetch_factor=4/spawn`。

- torchrun 正常退出 0，`train_runtime=699.7 s`、`0.143 step/s`、`train_loss=6.0675`；100 个 loss
  全部有限，范围 `0.673--10.998`；两个 rank 均精确写出 steps 1--100 的 100 个唯一 telemetry
  行，并消费 100 local batches/3200 local instances；
- 去掉 step 1 的 worker 冷启动后，warm wait mean 为 rank 0/1 `122.449/123.605 ms`，p50 为
  `0.149/0.121 ms`，p95 为 `0.237/0.187 ms`，p99 为 `2.030/2.074 s`，最大值为
  `9.754/9.769 s`，累计 wait 为 `12.12/12.24 s`；
- factor 4 仅在 steps `2/51/87` 出现 >1 ms wait：rank 0 分别约
  `9.754 s/0.481 s/1.872 s`，rank 1 分别约 `9.769 s/0.539 s/1.917 s`。相同 block 1 顺序下，
  factor 2 在 steps `2/47/51/76/87` 分别出现 5 个尖峰；factor 4 消除了 steps 47/76 的队列耗空，
  并把 steps 51/87 的 rank 0 wait 从 `11.486/13.736 s` 降到 `0.481/1.872 s`；
- factor 2 的 warm wait 累计为 `39.91/39.76 s/rank`，factor 4 为 `12.12/12.24 s/rank`，减少
  约 `69.6%/69.2%`。端到端 runtime 从 `720.3 s` 降至 `699.7 s`，缩短约 `20.5 s` 或
  `2.9%`，吞吐从 `0.139` 提高到 `0.143 step/s`；相对无 prefetch 的 `883.8 s/0.113 step/s`，
  factor 4 缩短 runtime 约 `20.8%`、提高吞吐约 `26.5%`；
- factor 4 的 rank process-tree peak RSS 为 `20.59/20.58 GB`，factor 2 为
  `19.84/19.85 GB`，预取容量从 2 增至 4 的观测内存代价约 `0.74 GB/rank`。RSS 求和仍可能
  重复计算共享映射，这个数值只用于同口径相对比较；
- `checkpoint-100` 和 root adapter 均可通过 `PeftConfig.from_pretrained()` 离线读取，LoRA
  `r=64/alpha=64`，`get_last_checkpoint()` 正确返回 `checkpoint-100`；结束后两卡显存归零，
  无 torchrun、rank 或 worker 残留；
- 保存 checkpoint 时日志出现一次 `terminate called without an active exception`，但主 torchrun
  随后完整输出最终训练指标并退出 0，root/checkpoint adapter 和训练状态文件均可读取。当前将其
  记录为 worker/底层库退出阶段的非致命告警；若长跑中转为非零退出或产物缺失，需单独诊断。

factor 4 在相同 block 1 batch plan 下比 factor 2 更能吸收长尾读取延迟，但 block 1 仍会快速扩大
Arrow unit 工作集并占用约 20.6 GB/rank 的进程树 RSS。三次 block 1 运行是顺序执行，没有严格
清空或固定 OS page cache，factor 4 也只有单次样本；因此 `2.9%` 的端到端差异不能视为稳定收益。
本次仍不满足 AC-12 的 200 steps、丢弃前 20 steps 和正式 comparator 门槛，正式 YAML 当前保留
用户最后要求的 worker 1/factor 4/block 1 仅用于实验复现，不构成完整训练 rollout 建议。

### 12.11 实施后 DataLoader settings 精简（2026-07-20）

根据用户对已运行代码的复审，删除 `TrainDataLoaderSettings` 和
`resolve_train_dataloader_settings()`。`_build_training_args()` 现在独自在配置入口完成组合校验，
并将 worker 大于 0、factor 为 null 的情况显式规范化为 factor 2；内部
`create_train_dataloader()` 直接消费构造完成的 `TrainingArguments`。主训练路径由原先最多四次
settings 解析收敛为一次配置校验。

`prefetch_capacity` 同时从 settings、启动日志、rank summary、SPEC 和验收断言中移除。该值只是
worker 数与 factor 的理论乘积，并不测量运行时队列占用。启动日志与 summary 改为读取创建完成的
DataLoader 的实际 `num_workers`、`prefetch_factor`、`persistent_workers`、`pin_memory` 和
`multiprocessing_context`；wait、consumption、RSS 以及 ahead-of-consumption telemetry 不变。

测试没有删除或跳过 prefetch 行为断言。更新后的实际验证为：

```text
tests/test_dataloader_prefetch.py
19 passed in 396.97s

tests/test_dataloader_prefetch_benchmark.py tests/test_indexed_arrow_store.py
15 passed in 9.39s

scripts/verify_dataset_batch_plan.py --steps 20 --through-dataloader
OK: verified 20 distributed dataloader steps with world_size=2, epoch=0.

py_compile（production/test/verifier/benchmark）和 git diff --check
均退出 0；正式 YAML 构造出的 TrainingArguments 为 worker 1/factor 4/persistent false/pin-memory true。
```

完整 DataLoader 测试继续覆盖隐式 factor 2 落到实际 loader、非法组合在
`TrainingArguments` 前失败、spawn worker 提前准备、worker 0/1 batch 等价、Arrow LRU 与 mmap
关闭、epoch refresh、resume offset、实际 consumption 和机器可读 wait/RSS summary。此次精简未
启动新的模型训练，也不改变此前实验 checkpoint 或结果目录。

随后用户进一步确认无需在 `trainer.py` 重复调用 validator，组合校验已完全内联到
`_build_training_args()`。该最终状态补跑配置/实际 loader/日志/summary 定向回归，结果为
`11 passed, 8 deselected in 42.75s`；双 rank 20-step production DataLoader verifier 再次退出 0。
