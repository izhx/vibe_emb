# F2LLM Stage-2 PyTorch DataLoader Prefetch Spec

状态：Draft  
日期：2026-07-20  
目标配置：`configs/train_f2llm_stage2_full.yaml`

## 1. 目标

为 F2LLM stage-2 Indexed Arrow 训练增加可配置的 PyTorch
`DataLoader` 多进程 batch prefetch，使下一批数据的以下 CPU 工作可以与当前批次的
GPU 计算重叠：

1. 按 batch plan 解析当前 sampling unit；
2. 打开或复用 Arrow mmap；
3. 批量读取 query、positive 和 negative 文本；
4. 构造 rank-local 对比学习 batch；
5. tokenize、padding 并生成 CPU tensor；
6. 在 `pin_memory=true` 时准备 pinned-memory batch。

功能必须保持当前训练的数据选择、batch 顺序、跨 rank 对齐、断点续训语义和
checkpoint 格式不变。框架通用默认值仍保持 `dataloader_num_workers=0`；
F2LLM stage-2 完整训练配置只有在通过本文定义的正确性和性能验收后，才启用建议值：

```yaml
data:
  unit_block_batches: 8

training:
  dataloader_num_workers: 1
  dataloader_prefetch_factor: 2
  dataloader_persistent_workers: false
  dataloader_pin_memory: true
```

其中 `prefetch_factor` 的单位是“每个 worker 提前处理的完整 DataLoader batch”，
不是样本数，也不是 Arrow unit 数。按照当前数据模型，一个 DataLoader batch 已经是一个
完整的 rank-local 对比学习 batch。

## 2. 背景

当前 F2LLM stage-2 数据位于 `/mnt/share` 的 QuarkFS/FUSE 文件系统，通过
`data/processed/f2llm_stage2_80k_arrow/manifest.json` 读取。当前数据快照包含 395 个
启用的 sampling unit、17,733,362 条 query，query/corpus Arrow 文件合计约 80 GB。

现有读取路径已经具备以下能力：

- manifest-only 启动和 unit 首次访问时 lazy open；
- query/corpus Arrow memory map；
- 每进程有界 `ArrowStorePool` LRU；
- 一个 rank-local batch 内批量 `take()` query 和去重后的 corpus doc；
- 由操作系统 page cache 被动复用已访问文件页。

这些能力减少了全量加载和重复打开，但不等于异步 prefetch。当前配置为：

```yaml
data:
  arrow_max_open_units: 32
  arrow_prefetch_units: 0
  unit_block_batches: 1

training:
  dataloader_num_workers: 0
  dataloader_pin_memory: true
```

`num_workers=0` 时，Arrow 读取、batch 构造和 tokenizer 都在训练主进程中同步执行。
PyTorch 不允许在该模式下设置非空 `prefetch_factor`。当前训练入口还主动拒绝
`dataloader_num_workers != 0`，直接原因是 consumption counters 在 dataset
`__getitem__` 中更新：worker 持有 dataset 副本，worker 内的更新不会回到主进程；
同时，提前执行的 `__getitem__` 也不等于该 batch 已实际进入训练。

当前自定义 `EmbeddingTrainer.get_train_dataloader()` 没有传递
`dataloader_prefetch_factor` 或 `dataloader_persistent_workers`。如果只删除现有
`num_workers` 检查，那么 `num_workers>0` 会使用 PyTorch 默认的
`prefetch_factor=2`，但 YAML 中显式配置的 factor 仍不会生效，统计语义也会错误。

## 3. 范围

本需求包括：

1. 在自定义训练 DataLoader 中支持并传递：
   - `dataloader_num_workers`；
   - `dataloader_prefetch_factor`；
   - `dataloader_persistent_workers`；
   - `dataloader_pin_memory`。
2. 对上述字段执行与当前数据模型一致的组合校验，并在启动日志中输出最终生效值。
3. 保持 DataLoader 有序返回；不同 worker 完成顺序不得改变 Trainer 消费的 batch 顺序。
4. 将 consumption accounting 从 worker 执行的 `__getitem__` 请求时刻，移动到训练主进程
   确认该 batch 实际进入训练的时刻。
5. 保证非 persistent worker 在每个 epoch 使用主进程刚生成的最新 batch plan。
6. 保证多 worker 下每个 worker 的 Arrow reader/LRU 独立且有界，并在 worker 退出时释放。
7. 增加单进程、多 worker、双 rank、epoch refresh、提前停止和 resume 验证。
8. 为 `configs/train_f2llm_stage2_full.yaml` 建立 `num_workers=0` 与候选
   `num_workers=1, prefetch_factor=2` 的 A/B 验证流程。
9. 将 `unit_block_batches` 调整为适合 batch prefetch 的局部性值；初始候选值为 8。
10. 输出足以判断 prefetch 是否生效以及是否缓解等待的运行时信息，包括：
    - 最终 DataLoader worker/factor/persistent/pin-memory 和 multiprocessing context；
    - 实际进入训练的 batch/instance consumption counters；
    - 主进程获取下一批数据的等待时间统计，至少包含观测窗口内的平均值和 p95。

## 4. 非范围

本需求不包括：

1. 实现 `data.arrow_prefetch_units`。该字段继续保持 `0`，不得用它表示 DataLoader
   batch prefetch。
2. 实现 `data.arrow_local_cache_dir` 或将完整 unit 提前复制到 node-local NVMe。
3. 主动调用 `madvise`、`readahead` 或扫描整个 Arrow 文件来预热操作系统 page cache。
4. 将完整 80 GB 数据预加载到内存。
5. 改变 Indexed Arrow 文件格式、manifest schema、fingerprint 或 doc ID。
6. 改变 sampling、正负例选择、instruction 选择、batch size、train group size 或 loss。
7. 支持 `dataloader_persistent_workers=true`。在引入显式 epoch state 同步协议前，该组合必须
   启动失败，而不是静默降级。
8. 优化 CPU 到 GPU 的异步传输。`pin_memory=true` 保留，但 non-blocking H2D 属于独立需求。
9. 替换 `transformers.Trainer` 或重写训练循环。
10. 对 Windows/macOS multiprocessing 或非 Linux 训练环境提供生产保证。

## 5. 输入

### 5.1 配置输入

训练配置接受以下字段：

| 字段 | 类型 | 框架默认值 | 规则 |
| --- | --- | --- | --- |
| `training.dataloader_num_workers` | 非负整数 | `0` | `0` 表示同步读取；F2LLM 候选值为 `1` |
| `training.dataloader_prefetch_factor` | `null` 或正整数 | `null` | worker 为 0 时必须为 `null`；worker 大于 0 且为 `null` 时解析为 PyTorch 默认值 `2` |
| `training.dataloader_persistent_workers` | 布尔值 | `false` | 本需求只允许 `false` |
| `training.dataloader_pin_memory` | 布尔值 | `true` | 沿用现有语义 |
| `data.unit_block_batches` | 正整数 | `1` | F2LLM stage-2 候选值为 `8` |
| `data.arrow_max_open_units` | 正整数 | `32` | 上限是每个 worker/进程的 unit LRU 上限 |

配置解析后必须形成唯一的 resolved settings。启动日志不得只打印用户原始值；当
`num_workers>0` 且 factor 省略时，日志必须打印实际值 `2`。

### 5.2 数据和运行时输入

- Indexed Arrow manifest 及其 enabled unit descriptors；
- 当前 epoch 的 deterministic compact batch plan；
- `seed`、`process_index`、`world_size`；
- dataset/task 对应的 local batch size、train group size、max length；
- Trainer 的 `max_steps`、epoch 和 resume checkpoint；
- Linux DataLoader worker process 与共享文件系统运行环境。

## 6. 输出

功能输出包括：

1. 与当前单进程读取路径结构相同的模型输入 batch；
2. 与基准运行相同的每步 dataset key、global indices、rank-local indices、query 数和
   passage 数；
3. 启动时一条可检索的 resolved DataLoader 日志，至少包含：

   ```text
   num_workers=1 prefetch_factor=2 persistent_workers=false pin_memory=true multiprocessing_context=spawn
   ```

4. 只统计实际进入训练步骤的 consumed batch/instance 日志；被预取但丢弃的 batch 不计入；
5. DataLoader consumer wait 的平均值和 p95，用于 A/B 判断是否仍被数据准备拖慢；
6. worker 读取异常的原始 unit/path 信息和非零训练退出码；
7. 不新增或修改训练数据文件，不改变 checkpoint 文件布局。

## 7. 约束

1. `training.per_device_train_batch_size` 必须继续为 `1`；dataset 的一个 item 已经是完整
   rank-local 对比 batch。
2. `gradient_accumulation_steps` 继续为 `1`，本需求不得改变该结构约束。
3. 所有 rank 必须消费相同的 batch index 和 dataset key，collective 调用顺序不得变化。
4. DataLoader 必须保持 `in_order=true` 或等价的有序交付行为。
5. 正负例随机性必须继续只由稳定的 sample/batch/epoch identity 决定，不得依赖 worker ID、
   worker 完成顺序或进程调度。
6. `dataloader_persistent_workers=true` 必须在启动阶段失败。
7. consumption counters 只能在主训练进程确认 batch 被训练时递增，不能在 worker
   `__getitem__` 中递增。
8. prefetch 队列中因 `max_steps`、异常或 epoch 结束被丢弃的 batch 不得计入 consumed。
9. 每个 worker 的 open unit 数不得超过 `arrow_max_open_units`；一个 unit 通常对应 query 和
   corpus 两个 mmap。
10. 资源评估必须按以下上界进行，而不能继续把 `arrow_max_open_units` 当作每 rank 的固定上限：

    ```text
    max_open_units_per_rank = num_workers * arrow_max_open_units
    max_mmaps_per_rank ~= max_open_units_per_rank * 2
    ```

    `num_workers=1` 时与当前每 rank 一个同步 reader pool 的 mmap 上界相同；worker 大于 1
    时会线性增加。
11. `TOKENIZERS_PARALLELISM=false` 的现有 launcher 行为保持不变，防止每个 worker 再创建
    无界 tokenizer 线程池。
12. 不得依赖 worker 内修改 Python dataset 对象后能被主进程观察到。
13. worker 初始化、关闭和异常传播不得吞掉 Arrow size/schema/fingerprint 错误。
14. 当前生产文件系统是共享 FUSE；增加 worker 数必须通过 A/B 数据证明收益，不能只依据
    本地磁盘结果修改完整训练配置。

## 8. 边界情况

### 8.1 配置组合

- `num_workers=0, prefetch_factor=null`：合法，行为与当前版本一致。
- `num_workers=0, prefetch_factor=2`：启动失败，错误信息指出 factor 需要 worker。
- `num_workers=1, prefetch_factor=null`：合法，resolved factor 为 `2`。
- `num_workers>0, prefetch_factor<=0`：启动失败。
- `persistent_workers=true`：无论 worker 数为何均启动失败，并说明 epoch plan 尚不能同步到
  persistent dataset 副本。
- YAML 显式 factor 必须真正传到 DataLoader；不得被自定义 Trainer 静默忽略。

### 8.2 Epoch 边界

- epoch 0 结束后，未消费的 prefetched batch 必须被丢弃；
- epoch 1 创建 iterator 前，worker 必须获得 epoch 1 的 batch plan 和 epoch 值；
- epoch 1 的 sample/negative/instruction 序列应与同 seed 单进程基准完全一致；
- 不允许 worker 持有 epoch 0 的 dataset 副本并继续向 epoch 1 供数。

### 8.3 提前停止与恢复

- 提前停止时，即使 worker 已经准备了尚未消费的 batch，也只能统计实际训练的 batch；
- checkpoint 保存前已经预取但未训练的 batch 不得改变 resume offset；
- 从 epoch 中间 resume 后，第一条实际训练 batch 必须与无中断运行在同一 global step 的 batch
  fingerprint 一致；
- worker 预取深度不得进入 checkpoint，也不得成为恢复正确性的条件。

### 8.4 分布式和任务差异

- worker 完成速度不同时仍按 sampler index 顺序返回；
- 任一 rank 的 worker 失败时训练必须失败退出，不能让其他 rank 永久等待 collective；
- retrieval/clustering 的 passage 数通常高于 classification，prefetch 内存评估必须覆盖
  `train_group_size=8` 和 max length 1024 的最坏配置；
- 最后不足 global batch 的记录仍按现有 batch plan 丢弃；prefetch 不得补齐或重新采样；
- `sample_factor>1` 导致的合法重复 record 不能被误判为 worker 重复供数。

### 8.5 共享存储与资源压力

- 冷启动时 PyTorch 会立即派发 `workers * factor` 个任务，必须考虑多 rank 同时访问多个 unit
  造成的 metadata/read burst；
- worker 数增加后吞吐可能下降。队列更深只能吸收延迟抖动，不能提高共享盘持续带宽；
- worker 被系统终止、文件描述符不足或 mmap 打开失败时，日志必须包含 worker 异常及相关路径；
- 正常结束、提前停止和异常退出均不得遗留训练 worker 进程。

## 9. 兼容性

1. 未配置新字段的历史 YAML 继续解析为 `num_workers=0`、`prefetch_factor=null`，保持同步路径。
2. 框架级默认不变；只有 `configs/train_f2llm_stage2_full.yaml` 在验收通过后切换候选值。
3. Indexed Arrow manifest、unit 文件、sample ID、doc ID 和 fingerprint 完全兼容，无需重建数据。
4. 已有 checkpoint 可以在 worker 0 或 worker 1 模式下恢复；新的 checkpoint 也可以回滚到
   worker 0 模式继续训练。
5. 训练 step 数、optimizer/scheduler 状态、checkpoint 命名和输出目录布局不变。
6. 相同 seed、world size 和 checkpoint 下，worker 0 与 worker 1 的实际训练 batch fingerprint
   必须一致。
7. 继续兼容当前 `/mnt/share/envs/embt` 环境中的 PyTorch 2.10 和 Transformers 5.8；
   不依赖未公开的 DataLoader API。
8. `arrow_prefetch_units` 和 `arrow_local_cache_dir` 的现有字段含义不变，不将它们作为
   DataLoader prefetch 的兼容别名。

## 10. 验收标准

以下每一项都必须能够从命令退出码、测试断言、结构化日志或结果文件直接判断通过/失败。

### AC-1：历史默认行为

给定未设置 `dataloader_prefetch_factor` 且 `dataloader_num_workers=0` 的测试配置，训练初始化
成功，启动日志包含：

```text
num_workers=0 prefetch_factor=null persistent_workers=false
```

同 seed 连续运行两次，前 20 个实际训练 batch fingerprint 完全相同。

### AC-2：有效 prefetch 配置真正下传

给定 `num_workers=1, prefetch_factor=2, persistent_workers=false`，从训练进程实际创建的
DataLoader 读取到：

```text
num_workers == 1
prefetch_factor == 2
persistent_workers is False
```

启动日志同时包含实际 DataLoader 的 worker/factor/persistent 值。测试不得只检查 YAML 或
`TrainingArguments` 值。

### AC-3：无效组合启动失败

以下配置分别运行配置解析/训练初始化时必须返回非零退出码，并包含可定位字段的错误信息：

1. `num_workers=0, prefetch_factor=2`；
2. `num_workers=1, prefetch_factor=0`；
3. `num_workers=1, persistent_workers=true`。

失败必须发生在模型加载和 worker 创建前。

### AC-4：确实发生 ahead-of-consumption

使用可控测试 dataset 阻塞主消费端，在创建 DataLoader iterator 后且主进程尚未请求第一个
训练 batch 前，worker 侧观测到最多 2 个 batch 已开始或完成准备；解除阻塞并消费一个 batch
后，队列会继续补充后续 batch。测试通过进程安全 event/queue 断言，不以固定 sleep 时间作为
唯一判据。

### AC-5：训练 batch 等价

对同一 Indexed Arrow 测试 manifest、seed、world size 和 epoch，分别使用 worker 0 与
worker 1/factor 2 获取前 20 个实际训练 batch。以下序列必须逐项相等：

- dataset key；
- global record indices；
- 每个 rank 的 local record indices；
- query count；
- passage count；
- train group size；
- sample/positive/negative 选择 fingerprint。

### AC-6：跨 rank 对齐

使用 2 个 gloo rank 和 worker 1/factor 2 执行至少 20 步。每一步必须满足：

- 两个 rank 的 dataset key 相同；
- 两个 rank 的 global indices 相同；
- local indices 是不重叠的连续切片，其拼接等于 global indices；
- query/passage shape 符合 local batch size 和 train group size；
- 进程在超时内以退出码 0 结束，不发生 collective hang。

### AC-7：Consumption accounting 准确

在 local batch size 为 4、`max_steps=3`、worker 1/factor 2 的测试中，即使 worker 已请求
第 4、5 个 batch，训练结束日志必须精确报告：

```text
consumed_local_batches=3
consumed_local_instances=12
```

主进程日志不得报告 0、4 或 5 个 consumed batch。

### AC-8：Epoch refresh

运行 2 个 epoch 后：

- epoch 0 和 epoch 1 的 batch fingerprint 序列不同；
- 相同 seed 的第二次完整运行分别复现两个 epoch 的序列；
- worker 1 结果与 worker 0 基准逐 epoch 相同；
- 日志能证明每个 epoch 使用对应 epoch number 创建 iterator/worker。

### AC-9：Resume 等价

在相同测试配置上比较：

1. 无中断训练到 step 12；
2. 训练到 step 5 保存 checkpoint，再从该 checkpoint 继续到 step 12。

两次运行 step 6 至 12 的 batch fingerprint、loss 输入 shape、global step 和最终
consumption counters 必须一致。两次命令均以退出码 0 结束。

### AC-10：LRU 和 worker 清理

使用至少 5 个 unit、`arrow_max_open_units=2` 的测试数据遍历能够触发 eviction 的 batch：

- 任一 worker 同时打开的 unit 不超过 2；
- eviction 后对应 query/corpus mmap 被关闭；
- DataLoader 正常结束和提前停止后，父进程检查不到仍存活的 DataLoader worker；
- 测试进程可以立即重命名/删除测试 Arrow 文件，不存在未释放句柄。

### AC-11：真实配置 smoke

使用 `configs/train_f2llm_stage2_full.yaml` 的候选配置执行至少 20 个 optimizer steps：

- 训练以退出码 0 结束；
- 每步 loss 为有限值；
- 日志包含 resolved DataLoader settings、消费统计和 batch wait 统计；
- 不出现 worker crash、Arrow schema/size error、collective timeout 或 CUDA OOM；
- 输出 checkpoint/metrics 可以被当前训练入口重新识别。

### AC-12：共享存储 A/B 性能门槛

在同一机器配额、同一数据快照、同一 seed、同一 world size 下，分别用以下设置运行 200 步，
前 20 步作为 warm-up 不计入：

- baseline：worker 0、factor null、`unit_block_batches=1`；
- candidate：worker 1、factor 2、`unit_block_batches=8`。

结果必须记录 steps 21-200 的：平均/p95 batch wait、平均 step time、query/s、各 rank 峰值 RSS、
worker 数和运行环境。候选配置进入 `train_f2llm_stage2_full.yaml` 的条件是：

1. p95 batch wait 至少下降 20%，或者 baseline 的 p95 batch wait 已低于 10 ms；
2. query/s 不低于 baseline 的 95%；
3. 无 worker/Arrow/collective 错误；
4. 每 rank 峰值 RSS 增量不超过 2 GiB；
5. consumption 和 batch fingerprint 验收全部通过。

若未满足以上条件，功能代码仍可保留，但目标配置必须继续使用 worker 0。

## 11. 验证方式

实现完成后至少执行以下验证。这里列出的测试文件和参数是实现交付的一部分；命令必须可以
直接运行，不依赖人工读取 Python 对象。

### 11.1 静态和单元测试

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q \
  tests/test_dataloader_prefetch.py \
  tests/test_indexed_arrow_store.py
```

`tests/test_dataloader_prefetch.py` 至少覆盖 AC-1 至 AC-5、AC-7、AC-8 和 AC-10。

### 11.2 双 rank 数据计划 smoke

扩展 `scripts/verify_dataset_batch_plan.py`，使其可以通过真实 DataLoader worker 路径取 batch，
而不是直接调用 dataset `__getitem__`：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  scripts/verify_dataset_batch_plan.py \
  --config tests/fixtures/prefetch_train.yaml \
  --steps 20 \
  --through-dataloader
```

预期退出码为 0，并打印：

```text
OK: verified 20 distributed dataloader steps
```

### 11.3 Resume 集成测试

提供自动化测试或 smoke 脚本，分别执行无中断 12 步和 5+7 步恢复运行，并在结果目录写出
batch fingerprint JSONL。验证命令必须对两个 JSONL 的 step 6-12 做机器比较，差异时返回
非零退出码。

### 11.4 真实 Indexed Arrow smoke

使用独立输出目录运行，避免覆盖正式结果：

```bash
CONFIG=configs/train_f2llm_stage2_full.yaml \
MAX_STEPS=20 \
OUTPUT_DIR=/tmp/f2llm-prefetch-smoke \
OVERWRITE_OUTPUT_DIR=true \
bash scripts/run_job.sh
```

验证者必须保存退出码和日志，并检查 AC-11 中列出的所有可观察项。该命令会写临时模型产物，
不作为单元测试的一部分。

### 11.5 QuarkFS A/B

提供一个只改变 worker/factor/unit block 三个字段的 benchmark 入口或生成两份 resolved YAML，
依次执行 AC-12 的 200-step baseline/candidate。结果至少写出以下机器可读字段：

```json
{
  "num_workers": 1,
  "prefetch_factor": 2,
  "unit_block_batches": 8,
  "measured_steps": 180,
  "mean_batch_wait_ms": 0.0,
  "p95_batch_wait_ms": 0.0,
  "mean_step_time_ms": 0.0,
  "queries_per_second": 0.0,
  "peak_rss_bytes_per_rank": []
}
```

A/B 工具必须根据 AC-12 自动返回 0/非零退出码，不能只生成需要人工判断的图表。

## 12. 风险

1. **共享存储放大**：所有 rank 同时启动 worker 可能增加 QuarkFS 随机读和 metadata 请求，
   导致吞吐下降而不是上升。
2. **CPU 过量订阅**：每个 rank 增加 tokenizer worker；多 GPU 节点上的总进程数按
   `ranks * workers` 增长。
3. **内存和文件句柄增长**：每个 worker 拥有独立 Arrow LRU、mmap、page table 和 batch
   queue；worker 数大于 1 时资源线性增长。
4. **统计提前**：如果仍在 `__getitem__` 中记 consumed，prefetch 会把未训练 batch 算进去。
5. **旧 epoch 数据**：误启用 persistent worker 会让 worker dataset 停留在旧 epoch plan。
6. **分布式死锁**：任一 rank 因 worker 异常或供数顺序不同而缺少 batch，其他 rank 可能阻塞在
   cross-device negative collective。
7. **冷启动变慢**：iterator 创建时会根据 worker 数和 factor 立即派发多个 batch 任务，随机
   unit 设置下会产生打开文件和 page fault burst。
8. **队列内存**：max length 1024、local batch 32、group size 8 时，一个 prefetched retrieval
   batch 含 32 条 query 和 256 条 passage；factor 或 worker 过大会增加共享内存和 pinned memory。
9. **Resume 漂移**：若把“已请求”误当成“已训练”，checkpoint offset 会跳过 prefetched batch。
10. **版本行为变化**：PyTorch 对 multiprocessing、pin-memory thread 和 DataLoader iterator 的
    内部实现可能变化；实现只能依赖公开参数和有序交付契约。

## 13. 回滚

### 13.1 配置回滚

首选回滚不需要修改数据或 checkpoint，只需将目标配置恢复为：

```yaml
data:
  unit_block_batches: 1

training:
  dataloader_num_workers: 0
  dataloader_prefetch_factor: null
  dataloader_persistent_workers: false
```

回滚后必须能直接从 prefetch 模式生成的 checkpoint 继续训练。不得要求重新生成 Arrow 数据或
清理现有 checkpoint。

### 13.2 触发条件

出现以下任一情况应回滚目标训练配置到 worker 0：

- AC-5、AC-6、AC-7、AC-8 或 AC-9 任一正确性验收失败；
- candidate query/s 低于 baseline 的 95%；
- worker crash、collective hang 或新的间歇性 resume 失败；
- 每 rank 峰值 RSS 增加超过 2 GiB，或节点出现文件描述符/共享内存不足；
- p95 batch wait 没有改善且共享存储请求压力明显上升。

### 13.3 代码回滚边界

如果问题只发生在 F2LLM 生产参数，保留通用 DataLoader worker 支持和测试，仅回滚 YAML。
如果 worker 0 的历史配置也出现行为变化，则回滚以下代码边界：

1. 自定义 DataLoader 参数下传；
2. consumption accounting 的主进程确认逻辑；
3. 新增的 wait telemetry。

回滚不得影响 Indexed Arrow reader、batch plan、模型、loss 或 checkpoint 保存逻辑。回滚完成后
必须重新执行 AC-1 和至少一个 20-step worker 0 smoke，证明历史同步路径恢复。
