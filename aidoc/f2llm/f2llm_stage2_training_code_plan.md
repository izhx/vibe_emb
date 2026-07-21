# F2LLM-v2 / ML-Embed Second-stage 训练代码修改计划

## 当前实现状态（2026-07-20）

实施顺序的第 1--4 项已有第一版实现：配置支持 Indexed Arrow manifest 和 task
defaults；record store 使用 manifest-only descriptor、lazy mmap 和有界 unit LRU；
batch builder 对当前 rank 的 query 索引执行一次批量读取；`unit_block_batches` 已实现，
值为 1 时保持旧的逐 batch 混排。旧 JSON/JSONL 路径继续使用 `JsonRecordStore`。

真实三 unit smoke 中，manifest 加载后打开的 store 数为 0；依次访问三个 unit、设置
`arrow_max_open_units=2` 时，常驻 store 始终不超过 2，显式 `close()` 后为 0。task
defaults 分别解析为 retrieval `8/false`、clustering `10/true`、classification
`2/true`（数值依次为 `train_group_size/no_in_batch_neg`）。

F2LLM retrieval loss 和跨卡数值验证已于 2026-07-20 完成；MRL 仍未实现。完整 stage-2
长训和新 loss 的 MTEB 对比尚未执行。

### F2LLM retrieval loss 实施与 smoke（2026-07-20）

冻结语义和逐文件实施计划见：

- [retrieval loss spec](./spec/f2llm_stage2_retrieval_loss_spec.md)
- [retrieval loss implementation plan](./plan/f2llm_stage2_retrieval_loss_implementation_plan.md)

当前实现新增 `training.retrieval_loss_mode: legacy | f2llm`，默认 `legacy`。F2LLM retrieval
计算 query-local hard CE 加 positive-only in-batch CE；两个 CE 使用独立 normalization，在同一
helper 中合计后只 backward 一次。多卡只 gather positives，不 gather queries 或完整 passage
groups。classification/clustering 继续由 `no_in_batch_neg=true` 使用原有 own-group CE。

正式配置 `configs/train_f2llm_stage2_full.yaml` 已显式启用 `f2llm`，并改用独立输出目录
`results/f2llm-s2-lora64-f2llm-loss`，不会自动续接 legacy checkpoint。真实 smoke 使用新增的
`configs/train_f2llm_stage2_loss_smoke.yaml` 和
`scripts/f2llm/verify_retrieval_loss_smoke.py`。

验证结果：

- loss/config/Gloo distributed 与既有 Arrow/prefetch 联合回归：`41 passed in 394.05s`；
- 单卡 Qwen3-0.6B BF16：`hard=2.1378059`、`in_batch=0.2648869`、
  `total=2.4026928`，392 个 LoRA 梯度 tensor 全部有限；
- 两卡 Qwen3-0.6B BF16：每 rank 的 score shape 为 `[2,4]`，每 rank 只 gather 一次
  `[2,1024]` positive tensor；loss 分项与梯度全部有限；
- Trainer 单卡 44 steps 和两卡 31 steps 均实际经过 retrieval batch、保存最终 adapter；
- 两个 adapter 均通过 `maybe_apply_peft()` 重载并识别 392 个可训练 LoRA tensor。

smoke 输出位于 `/tmp/vibe_emb_f2llm_loss_smoke_single_20260720` 和
`/tmp/vibe_emb_f2llm_loss_smoke_ddp2_20260720`，仅作为临时验收产物。正式长训是否从 base/stage-1
开始或以 checkpoint-12000 adapter warm start，及是否调整 learning rate，需要在独立实验中
人工决定；不得切换 loss 后直接 resume 旧 optimizer/scheduler。

### 端到端单卡 smoke 结果（2026-07-14）

使用 `/mnt/share/envs/embt` 环境和 `/mnt/share/models/Qwen3-0.6B`，已经完成一次
60 optimizer-step 的真实 LoRA 训练：

- 配置：`configs/train_f2llm_stage2_smoke.yaml`；
- 一键入口：`bash scripts/run_f2llm_stage2_smoke.sh`；
- Arrow profile：`data/processed/f2llm_stage2_smoke_arrow/`，17 MiB；
- 训练输出：`results/f2llm-stage2-arrow-smoke/`；
- 训练耗时 22.94 秒，最终汇总 loss 1.4463；
- 实际消费：`dala` 46 batches / 92 queries，`cedr` 12 batches / 24 queries，
  `sts22` 2 batches / 4 queries；三种 task type 都完成了 forward、loss、backward 和
  optimizer step；
- 成功保存 checkpoint-30、checkpoint-60 和最终 PEFT adapter，每个 adapter 权重约
  20.2 MB。

该 smoke 证明了 selection → Indexed Arrow → manifest-only loader → lazy mmap/LRU →
task-aware group size → tokenizer/collator → Qwen3 LoRA 反向 → checkpoint 保存的单卡
闭环。它不等价于完整 second-stage 训练；多卡验证结果见下一节。

### 两卡 cross-device negatives smoke（2026-07-14）

两张 H100 上使用 `torch.distributed.run --nproc_per_node 2` 完成了 174 optimizer
steps：

- 配置：`configs/train_f2llm_stage2_smoke_2gpu.yaml`；
- 入口：`bash scripts/run_f2llm_stage2_smoke_2gpu.sh`；
- 输出：`results/f2llm-stage2-arrow-smoke-2gpu/`；
- NCCL 2.28.9 初始化成功，rank 0/1 分别绑定 cuda:0/cuda:1；
- 两个 rank 由相同 seed 独立生成 2,818 个 global batches，global batch size 为 4，
  每个 rank 连续消费其中 2 条 query；
- `negatives_cross_device=true`。前 172 steps 的 classification/clustering 使用
  `no_in_batch_neg=true`；step 173--174 消费 retrieval `sts22`，因此实际进入
  `_dist_gather_tensor` 的 query/passage all-gather 和全局 target-offset loss 分支；
- rank 0 最终统计：`dala` 94 batches / 188 local queries，`cedr` 78 / 156，
  `sts22` 2 / 4；
- 训练耗时 52.45 秒，3.317 steps/s，汇总 loss 1.6604；
- checkpoint-87、checkpoint-174 和最终 PEFT adapter 均成功保存。

训练无 collective hang、shape mismatch、target 越界、NaN 或重复保存冲突。至此单节点
两卡的 Arrow loader、DDP batch 对齐、NCCL cross-device negatives、反向和保存闭环已
验证；尚未覆盖两节点场景。

### 全量训练数据入口（2026-07-14）

3-unit Arrow 仍保留用于快速回归测试；正式 second-stage 配置已切到全量
`data/processed/f2llm_stage2_80k_arrow/manifest.json`：

- 配置：`configs/train_f2llm_stage2_full.yaml`；
- 入口：`bash scripts/run_f2llm_stage2_full.sh`，默认使用 GPU 0、1；
- 395 个 unit、17,733,362 个有效 query、90,313,442 个 unit-local corpus 文档；
- loader 使用 manifest-only 启动、lazy mmap、最多 4 个打开 unit 和 8-batch unit block；
- `scripts/verify_dataset_batch_plan.py` 已从全量配置实际构造计划并读取 3 个 batch，
  未在启动时 eager 打开全部 Arrow。
- 使用同一全量配置在 H100 上完成 1 个真实 optimizer step，生成 8,866,625 个
  global batches 的 epoch plan，实际完成 forward/backward/保存，loss 为 3.25；产物位于
  `results/f2llm-stage2-full-smoke/`。

正式训练的 batch size、训练步数/backbone checkpoint 仍应按具体实验预算覆盖；该配置给出
一个可运行的一 epoch 基线，而不是声称等同于论文未公开的硬件规模与全局 batch。

### NCCL 2.28.9 启动修复与 1024-token 配置（2026-07-15）

本节点的 HPC-X 环境会加载 NCCL RDMA/SHARP plugin。`scripts/train.sh` 原先默认设置
`NCCL_IB_DISABLE=1`，在 NCCL 2.28.9 + CUDA 13 下会让两 rank 的 Qwen DDP 参数同步在
初始化阶段触发 glibc `free(): double free detected in tcache 2`。最小
`init_process_group + all_reduce + Qwen DDP` 已证明：不设置该变量成功，单独设置为 1
稳定复现。因此训练入口不再强制 `NCCL_IB_DISABLE` 或 `NCCL_P2P_DISABLE`；如特定机器
确实需要禁用，应由调用方显式设置并自行验证。

全量配置当前使用每卡 batch size 8、query/passage 最大长度 1024。首个 clustering batch
包含每卡 8 query 和 80 passage，关闭 checkpointing 时会占满 H100 80GB。正式配置因此
启用 `gradient_checkpointing=true` 和 `sub_batch_size=2`。修复后原样执行
`bash scripts/train.sh` 的两卡 1-step smoke 成功：1,108,237 个 global batch plan，
runtime 28.92 秒，loss 4.4307，并成功保存 checkpoint 和最终 adapter。

### 中断续训与数据一致性验证（2026-07-14）

原实现的 PEFT checkpoint 不能直接续训：`EmbeddingModel.save()` 只保存
`adapter_model.safetensors`，而 Transformers 5.8 默认 `_load_from_checkpoint()` 只查找
完整模型 checkpoint 文件。optimizer、scheduler、RNG 和 trainer state 虽然存在，模型
加载会先报缺少 `model.safetensors`。`EmbeddingTrainer._load_from_checkpoint()` 现已增加
adapter-only PEFT 恢复路径，模型加载后仍由 Trainer 恢复 optimizer/scheduler/RNG。

验证采用同一份 8-step 配置：

1. 连续训练 8 steps，并在 step 4 保存 checkpoint；
2. 新进程从 checkpoint-4 恢复，执行 step 5--8；
3. 对比每一步 loss/grad norm、trainer state 和最终 adapter。

结果：恢复 run 的 step 5--8 loss、grad norm、learning rate 与连续 run 相同；最终 392
个 LoRA tensor 全部逐元素 bitwise equal，最大绝对差为 0；两个最终 adapter 的 SHA-256
均为 `48498816cbcab06927463aa59f6cdeac9997458e6fe1b6b55d1ea43d270d67ad`。

同样对两卡 174-step run 做了 checkpoint-87 恢复实验。恢复后的 step 88--174
完成 DDP/NCCL 训练，并再次经过 retrieval cross-device negatives；最终 392 个 LoRA
tensor 与连续两卡 run 全部 bitwise equal，最大绝对差为 0，两个 adapter SHA-256 均为
`9d9ad3bfb597c609b80cc621229beb2c409d60f30eb9460ee00e769544cf251a`。

数据顺序能够恢复的原因是：每个 epoch 的 global batch plan 仅由 seed、epoch、manifest
unit 顺序/count、world size、batch/sample 配置和 `unit_block_batches` 确定；rank 只切分
相同 global indices；正负例选择又由 seed、epoch、unit key、batch index、global
position 和 record index 确定，不依赖进程内可变 RNG。Trainer 默认
`ignore_data_skip=false`，恢复时跳过 checkpoint 前已经完成的 global batches。

一致性成立的必要条件：

- 使用完整 Trainer checkpoint 目录恢复，不能把最终 adapter 当作 resume checkpoint；
- seed、world size、manifest 内容及 unit 顺序、batch size、sample size/factor、task
  defaults、`unit_block_batches` 和 tokenizer/model 配置不变；
- `max_steps`/scheduler 总步数应在首次启动时就确定，不能先按较短 schedule 跑完再增加；
- 保持 `ignore_data_skip=false`；`dataloader_num_workers=0` 仍是默认路径，若启用已验证的
  non-persistent DataLoader prefetch，则 worker 0/1 必须通过相同 batch fingerprint 和
  resume offset 验证；
- NumPy/PyTorch/Transformers/PEFT 版本和确定性相关设置尽量不变。

当前仍有以下风险：checkpoint 未内嵌 manifest/config fingerprint，修改数据或配置后恢复
不会主动拒绝；改变 world size 会改变 global batch 边界及 dropped tail；跨 epoch resume
依赖 Trainer 正确恢复 epoch 并触发 `refresh_epoch()`；消费统计不进入 checkpoint，因此
恢复 run 的统计只覆盖恢复后的 batch；若在两次 checkpoint 之间中断，只能回退到上一个
完整 checkpoint。`scripts/train.sh` 现支持 `RESUME_FROM_CHECKPOINT`，并可通过
`OVERWRITE_OUTPUT_DIR=false` 启用同目录 last-checkpoint 自动发现。

## 1. 目标与边界

本计划负责让 `vibe_emb` 读取 Indexed Arrow 数据，区分 retrieval、clustering、two-way classification 三类任务，并按照参考实现计算不同的负例 loss 和 Matryoshka Representation Learning（MRL）。

上游数据契约：

- [数据采样计划](./f2llm_stage2_sampling_plan.md)
- [Indexed Arrow 数据格式](./f2llm_stage2_indexed_arrow_format.md)

本轮不实现：

- Matryoshka Layer Learning（MLL）；
- Matryoshka Embedding Learning（MEL）；
- 教师模型 embedding MSE 蒸馏；
- 训练时修改或重新采样原始 Parquet；
- 对发布数据重复添加 instruction。

现有 JSON/JSONL 与 legacy contrastive loss 必须继续可用。

## 2. 当前实现差异

当前 `vibe_emb` 的默认 in-batch loss 将当前 batch 的所有 positive 和 hard negatives 拼成一个 passage 矩阵，再让每个 query 对整个矩阵做一次 CE。

参考 F2LLM/ML-Embed 实现采用两个独立 loss：

1. `hard_loss`：每个 query 只与自己的 positive 和显式 hard negatives 比较；
2. `in_batch_loss`：每个 query 只与 batch 中所有样本的 positive 比较。

retrieval 将两者相加。clustering 和 classification 只使用 `hard_loss`，避免 batch 内同类文本成为 false negative。两种 retrieval 计算方式数学上不等价，因此必须新增 task-aware 路径，同时保留旧路径兼容已有实验。

## 3. 配置接口

### 3.1 `DatasetConfig`

新增：

```python
task_type: Optional[str] = None
data_format: str = "auto"
```

允许的 `task_type`：

- `retrieval`
- `clustering`
- `classification`

`data_format`：

- `auto`：根据 path/manifest 判断；
- `json`：现有 JSON/JSONL；
- `indexed_arrow`：新格式。

保留 `no_in_batch_neg`，不弃用。为区分“未设置”和“显式 false”，类型改为 `Optional[bool]`：

- 显式值优先；
- 未设置且有 task type 时使用 task 默认；
- 未设置且没有 task type 时保持 legacy 默认 `false`。

`train_group_size` 同样保持显式配置最高优先级。

### 3.2 `DataConfig`

允许两种数据入口同时存在：

```yaml
data:
  indexed_dataset_manifest: data/processed/ml_embed_stage2_arrow/manifest.json
  datasets: []
```

解析规则：

- `datasets` 和 `indexed_dataset_manifest` 至少存在一个；
- manifest 中每个启用 sampling unit 展开为一个 `_LoadedDataset`；
- 可以同时追加旧 JSONL datasets；
- unit ID 与显式 dataset name 全局不可重复。

新增 task 默认配置：

```yaml
data:
  task_defaults:
    retrieval:
      train_group_size: 8
      no_in_batch_neg: false
    clustering:
      train_group_size: 10
      no_in_batch_neg: true
    classification:
      train_group_size: 2
      no_in_batch_neg: true
```

对应 1 正例加 7/9/1 个 hard negatives。

新增 Indexed Arrow 运行时配置：

```yaml
data:
  arrow_open_mode: lazy
  arrow_max_open_units: 4
  arrow_prefetch_units: 1
  arrow_verify_mode: manifest
  arrow_local_cache_dir: null
  unit_block_batches: 8
```

- `arrow_open_mode` 初版只允许 `lazy`；不提供生产环境 eager-open 全部 unit 的默认路径；
- `arrow_max_open_units` 是每个训练进程的 unit 上限，一个 unit 通常对应两个 Arrow mmap；
- `arrow_prefetch_units` 计入上述上限，初版允许 0 或 1；
- `arrow_verify_mode` 支持 `manifest | lazy | full`，训练默认 `manifest`，完整 checksum 扫描通过离线命令执行；
- `arrow_local_cache_dir` 可选，用于共享存储到 node-local NVMe 的 fingerprint cache；
- `unit_block_batches` 控制 unit 调度局部性。全局 dataclass 默认保持 `1` 以兼容旧 JSONL 行为，second-stage 示例显式使用 `8`。

### 3.3 `EmbedTrainingExtras`

新增：

```python
use_mrl: bool = False
mrl_min_dim: int = 8
mrl_dims: Optional[List[int]] = None
mrl_weighting: str = "f2llm"
```

现有 `temperature` 默认值不全局修改，避免改变旧实验。ML-Embed second-stage 示例 YAML 显式设置 `temperature: 0.05`。

## 4. Record store 抽象

将 `_LoadedDataset.records: List[Dict]` 替换为 store descriptor 与统一只读接口。Indexed Arrow unit 在 manifest 解析后只创建轻量 descriptor，不能在 `_LoadedDataset` 构造阶段打开数据文件：

```python
@dataclass(frozen=True)
class ArrowUnitDescriptor:
    unit_id: str
    query_count: int
    corpus_count: int
    task_type: str
    metadata_path: str
    queries_path: str
    corpus_path: str
    queries_file_size: int
    corpus_file_size: int
    compression: str
    query_record_batch_count: int
    corpus_record_batch_count: int
    metadata_fingerprint: str
    queries_fingerprint: str
    corpus_fingerprint: str
```

manifest 必须内联上述运行时字段，因此创建 descriptor 时不读取 120 个 unit 的 `metadata.json` 或 IPC footer。

统一 store 接口：

```python
class RecordStore(Protocol):
    def __len__(self) -> int: ...
    def get_records(self, indices: Sequence[int]) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...
```

实现：

- `JsonRecordStore`：包装当前 `_read_json_records` 结果，行为不变；
- `IndexedArrowRecordStore`：读取一个 unit 的 `queries.arrow` 和 `corpus.arrow`；
- `ArrowStorePool`：持有 descriptor 到 store 的 lazy factory 和有界 LRU；`_LoadedDataset` 通过 pool 取得当前 unit store，不自行永久持有已打开 reader。

### 4.1 Arrow store 生命周期

- profile 初始化时只读取 manifest，不实例化 `IndexedArrowRecordStore`。
- 某个 unit 第一次被 batch 访问时，pool 才创建 store、读取 unit metadata/IPC footer，并校验 manifest 声明的文件大小、schema、compression 和 fingerprint 声明一致性；lazy-open 不重算完整文件 checksum。
- 根据 record batch 累计范围，将 query index 分组后批量 `take`。
- 收集当前 batch 所需的 positive/negative doc ID，去重后批量读取 corpus。
- 将 doc ID 映射回 query group，返回兼容内部逻辑的：

```python
{
    "query": str,
    "pos": [positive_text],
    "neg": [negative_text, ...],
    "lang": optional_str,
}
```

- unit 文件按需 memory-map；使用有上限的 LRU 文件句柄缓存。
- `arrow_max_open_units=4` 时，每个 rank 通常最多常驻 8 个 query/corpus mappings；prefetch unit 也计入 4 个 unit 上限。
- LRU eviction 同时关闭 query/corpus reader、memory map 和文件句柄。返回值必须是普通 Python 字符串和标量，不能让 Arrow array/buffer 引用逃逸 store 生命周期。
- worker 数仍保持 0，避免多个进程复制状态并破坏 consumption stats。
- dataset/trainer 结束时关闭所有 memory maps。

共享存储上可选启用 node-local cache。每节点 local rank 依据 fingerprint 原子写入完整 unit 文件，其他本地 rank 同步后复用；本地副本必须通过文件大小和 fingerprint 校验，且不得改变 manifest、doc ID 或 batch plan。该优化不作为正确性的依赖。

### 4.2 Batch 构建

`MultiDatasetBatchDataset.__getitem__` 改为一次批量调用 `get_records(local_indices)`，不逐条随机访问 Arrow。

保持现有不变量：

- 所有 rank 使用相同 global batch plan；
- 同一步来自同一个 sampling unit；
- rank 只取连续的 local slice；
- positive/negative 选择 seed 不包含 process index；
- partial global batch 继续丢弃。

### 4.3 Unit-block batch plan

现有实现先生成每个 dataset 的 batch，再把所有 batch 逐个完全打乱。对于 120 个 unit，这会让小 LRU 几乎每步 miss。新增 block 调度：

1. 每个 unit 仍按原有 seed 生成确定性 query permutation 和完整 global batches；
2. 按 `unit_block_batches` 将该 unit 的 batch 切成连续 block；
3. 使用所有 rank 相同的 epoch seed 打乱 block，而不是打乱单个 batch；
4. block 内保持该 unit 的 batch 顺序；
5. final partial global batch 的丢弃规则保持不变。

`unit_block_batches=1` 必须与当前逐 batch 混排行为兼容。second-stage 默认建议 8；4--8 更强调任务交错，16--32 更强调共享存储和 mmap 局部性。

block 调度不能改变每个 unit 的 selected query multiset、rank-local slice、instruction/正负例随机种子或 consumption stats。所有 rank 必须生成逐步相同的 `(unit_id, global_indices)`。

### 4.4 Batch-plan 内存

manifest-only 启动不加载 query/corpus 文本，但当前全局 plan 仍可能为约 8.3M query 保存 `int64` 索引，约 66 MB/rank。实现时优先：

- query count 小于 `2^32` 时使用 `uint32` permutation；
- block plan 只保存紧凑的 `(unit_idx, start_batch, batch_count)` 描述；
- unit permutation 按需生成或使用紧凑数组，不能为每个 batch 创建大量 Python 小对象。

这项优化不改变随机采样语义。若采用无数组的伪随机置换，必须单独证明无重复、全覆盖和跨 rank 一致，初版不强制采用。

## 5. Task-aware group size 与负例开关

解析优先级：

1. dataset/unit 显式 `train_group_size` 或 `no_in_batch_neg`；
2. `data.task_defaults[task_type]`；
3. 现有 `default_train_group_size` 和 legacy `no_in_batch_neg=false`。

合法性校验：

- retrieval/clustering 至少需要 1 个可用 negative；
- classification 默认 group size 必须为 2；允许显式覆盖用于实验，但仍不能超过可构造范围；
- 负例不足时沿用现有确定性重复采样；
- metadata 声明的负例数量必须与 Arrow query 实际数据一致；
- batch 输出携带 `task_type`、resolved group size 和 resolved `no_in_batch_neg`。

## 6. Collator 数据流

`EmbeddingCollator` 在现有字段基础上透传：

```python
task_type: Optional[str]
```

其他行为保持：

- query/passages 分别 tokenize；
- passage 布局继续是每条 query 的 positive 在前，随后是 hard negatives；
- `append_eos_token=true` 时为所有序列保证最后一个 token 是 EOS；
- second-stage Indexed Arrow unit 默认不再应用 query/passage instruction formatter；
- 旧 JSONL 仍可以使用现有 instruction 配置。

## 7. Loss 重构

### 7.1 Legacy path

`task_type is None` 时调用现有 `_contrastive_loss`，保持：

- 默认联合 passage matrix；
- `no_in_batch_neg=true` 时仅使用 query-local group；
- 现有 teacher score distillation 行为；
- 旧配置的 loss 数值和 score shape。

### 7.2 Hard-negative loss

将 passage embeddings reshape 为：

```text
[local_query_count, group_size, hidden_size]
```

每个 query 与自己 group 内的 passages 做 cosine similarity / temperature，target 恒为 group 第 0 列：

```text
hard_loss = CE([q_i · p_i_pos, q_i · p_i_neg_1, ...], target=0)
```

其他 query 的 positive 和 negatives 均不进入该 softmax。

### 7.3 Positive-only in-batch loss

只抽取每个 group 的第 0 个 passage 作为 positive matrix：

```text
positive_reps = p_reps.reshape(batch, group, dim)[:, 0, :]
```

单卡时：

```text
scores = q_local @ positive_local.T / temperature
targets = arange(local_batch)
```

多卡训练时：

- gather 每个 rank 的 `positive_reps`，不 gather hard negatives；
- 每个 rank 只保留本地 `q_reps` 并计算 `[local_queries, global_positives]`；
- target 为 `rank * local_batch + local_index`；
- gather 列表中本 rank positive 使用原 tensor 替换，以保留本地 passage 梯度；
- 远端 positives 作为常量 negatives；
- loss 交由 DDP 按标准方式平均梯度，不额外乘 world size。

### 7.4 三类任务组合

默认：

| task type | hard loss | positive-only in-batch loss |
|---|---:|---:|
| retrieval | 是 | 是 |
| clustering | 是 | 否 |
| classification | 是 | 否 |

实现上以 resolved `no_in_batch_neg` 决定是否添加 in-batch loss，因此显式配置仍可覆盖 task 默认。

```text
loss = hard_loss
if not no_in_batch_neg:
    loss += in_batch_loss
```

task type 用于默认配置、校验和日志分类，不额外建立三套重复模型代码。

## 8. MRL

### 8.1 默认维度

当 `use_mrl=true` 且 `mrl_dims=null`：

- 从 `mrl_min_dim=8` 开始生成二次幂；
- 保留所有小于完整 hidden size 的维度；
- 最后加入完整 hidden size；
- 去重后按降序计算。

显式 `mrl_dims` 必须：

- 全部为正整数；
- 不重复；
- 不超过模型 hidden size；
- 自动补充完整 hidden size，或在严格校验模式下要求用户明确包含。初版采用自动补充并记录 resolved dims。

### 8.2 每维归一化

即使完整 embedding 已归一化，截断后仍必须重新归一化：

```python
q_dim = F.normalize(q_reps[..., :dim], dim=-1)
p_dim = F.normalize(p_reps[..., :dim], dim=-1)
```

### 8.3 参考权重

对降序后的第 `n` 个维度（从 1 开始）：

```text
weight(dim, n) = 1 / (n * sqrt(full_dim / dim))
```

hard loss 和 in-batch loss 分别在每个维度计算并应用相同权重，再按任务规则相加。不对权重和再次归一化，以匹配参考代码。

`use_mrl=false` 时 resolved dims 只有完整 hidden size，权重为 1。

## 9. 输出与日志

`EmbeddingOutput` 可增加可选诊断字段：

```python
hard_loss: Optional[Tensor]
in_batch_loss: Optional[Tensor]
```

总 `loss` 仍是 Trainer 反向传播入口。训练日志至少记录：

- 总 loss；
- 当前 dataset/sampling unit 和 task type；
- resolved group size、temperature、MRL dims；
- 各 task type 的 consumed batch/query 数；
- retrieval hard/in-batch loss 的滑动平均。

resolved config 额外保存：

- Indexed Arrow manifest 路径和 fingerprint；
- profile/catalog/schema version；
- 实际 unit/query 数；
- task defaults；
- resolved MRL 配置；
- MKQA/SIB200 是否被包含。
- Arrow open/verify mode、LRU/prefetch 上限、unit block 大小和本地 cache 路径。

## 10. Second-stage 示例配置

新增 YAML 示例，关键配置为：

```yaml
data:
  indexed_dataset_manifest: data/processed/ml_embed_stage2_arrow/manifest.json
  arrow_open_mode: lazy
  arrow_max_open_units: 4
  arrow_prefetch_units: 1
  arrow_verify_mode: manifest
  arrow_local_cache_dir: null
  unit_block_batches: 8
  default_query_instruction: null
  default_passage_instruction: null
  append_eos_token: true
  same_dataset_within_batch: true
  task_defaults:
    retrieval:
      train_group_size: 8
      no_in_batch_neg: false
    clustering:
      train_group_size: 10
      no_in_batch_neg: true
    classification:
      train_group_size: 2
      no_in_batch_neg: true

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  temperature: 0.05
  negatives_cross_device: true
  use_mrl: true
  mrl_min_dim: 8
  mrl_weighting: f2llm
```

模型路径指向 stage-1 checkpoint 或用户指定 backbone；本计划不硬编码具体 checkpoint。

## 11. 测试计划

### 11.1 配置与兼容性

- 只有旧 JSONL datasets 的配置解析和行为不变。
- 只有 manifest、以及 manifest+JSONL 混合输入均可解析。
- dataset/unit name 冲突时报错。
- task 默认与显式覆盖优先级正确。
- 未设置 task type 时继续进入 legacy loss。
- Arrow open/verify mode、LRU/prefetch 上限和 `unit_block_batches` 非法值会在启动时失败。
- 全局默认 `unit_block_batches=1` 不改变旧 JSONL 的 batch 顺序；second-stage 显式值 8 生效。

### 11.2 Arrow loader

- Indexed Arrow 与等价 JSON fixture 产生相同内部 record。
- 随机 query index、跨 record batch index 和重复 doc ID 均能正确读取。
- `__getitem__` 使用批量 store API，不退化为逐条打开文件。
- manifest-only 初始化不打开任何 unit metadata、Arrow 文件或 IPC footer。
- 文件大小/schema/引用越界错误在目标 unit 第一次 lazy-open 时失败；`full` 离线模式能在训练前发现所有 unit 错误。
- LRU 打开与关闭 unit 后读取结果一致。
- store/prefetch 总数始终不超过 `arrow_max_open_units`，eviction 后 file descriptor 和 mmap 均释放。
- `get_records()` 返回对象不持有 Arrow buffer，store eviction 后内容仍可使用。
- 120-unit fixture 不会因 profile 初始化 eager 打开 240 个文件/rank。
- `compression=none` 和 `zstd` fixture 均可读取，benchmark 单独记录随机读取吞吐和读放大。

### 11.3 Loss 数值

使用固定小 tensor 手算：

- hard loss 只包含 query 自己的 group；
- retrieval in-batch 只包含所有 positives；
- 其他样本 hard negatives 不进入 in-batch softmax；
- retrieval 总 loss 等于两者之和；
- clustering/classification 默认只有 hard loss；
- 显式 `no_in_batch_neg=false/true` 能覆盖 task 默认；
- legacy path 数值保持现状。

### 11.4 跨卡

使用两进程 Gloo CPU 测试：

- 两个 rank 的 global positive 顺序一致；
- target offset 正确；
- 每个 rank 只计算本地 query 行；
- 本地 query 和 positive 有梯度；
- 所有 rank 执行相同 unit、group size 和 collectives；
- 不因 clustering/classification 跳过 gather 而造成 collective 次序不一致。

### 11.5 MRL

- 默认维度生成正确；
- 非法、重复和越界维度被拒绝；
- 截断 embedding 重新归一化；
- 权重与参考公式一致；
- full-dimension-only 与关闭 MRL 数值一致；
- MRL 下 hard/in-batch loss 均按全部维度计算且梯度有限。

### 11.6 Batch plan

扩展 `scripts/verify_dataset_batch_plan.py`，每步额外检查：

- sampling unit 和 task type 在各 rank 一致；
- resolved group size 和 `no_in_batch_neg` 一致；
- query/passage 数量符合 group size；
- retrieval 执行跨卡 gather，其他默认任务不执行；
- local shards 拼接后重建 global batch。
- `unit_block_batches=1` 与旧逐 batch shuffle 输出一致；大于 1 时每个 block 只包含一个 unit。
- 不同 block 大小不改变各 unit query multiset、global/local batch 切分和 consumption count。
- 120-unit、8-rank 模拟的逐步 unit/block 顺序完全一致。
- query permutation 使用紧凑 dtype 后仍无重复、全覆盖且跨 rank 一致。

### 11.7 多节点启动与审计

- 模拟 2 节点、每节点 8 rank 时，每个节点仅 local rank 执行可选路径 preflight/node-local cache 发布，其余 rank 复用结果。
- manifest-only 初始化的磁盘读取量不随 corpus 总字节数增长。
- 按 `sample_id` inspect/export 能还原来源 shard/row 和完整训练 tuple。
- tuple-local patch 重建后只改变目标 sample；共享 `doc_id` 的其他引用保持原文。

## 12. 实施顺序

PyTorch DataLoader batch prefetch 的独立需求、约束和实施记录见：

- `aidoc/f2llm/spec/f2llm_stage2_dataloader_prefetch_spec.md`；
- `aidoc/f2llm/plan/f2llm_stage2_dataloader_prefetch_implementation_plan.md`。

该能力与本节原计划中的 Arrow unit prefetch/node-local cache 不同，不改变 manifest 或
record-store 格式。

1. 扩展配置 dataclass 和 manifest-only descriptor 解析，但保持旧测试通过。
2. 引入 record store 抽象、lazy `ArrowStorePool`、有界 LRU 和显式清理。
3. 让 batch builder 批量读取 store，并透传 task type。
4. 实现 unit-block batch plan 和紧凑索引，验证所有 rank 的 plan 完全一致。
5. 拆出 hard loss 和 positive-only in-batch loss。
6. 实现 task-aware loss 路径及两 rank 测试。
7. 实现 MRL 维度、归一化和权重。
8. 增加日志、resolved config、second-stage 示例 YAML 和审计 CLI。
9. 扩展 batch-plan/多节点启动验证并执行小规模 smoke train。

## 13. 交付条件

完成标准：

- 旧 JSONL 训练可继续运行；
- Indexed Arrow profile 能按 unit 构造 deterministic distributed batch；
- manifest-only + lazy LRU 启动不加载或打开全部 unit，常驻 store 数不超过配置上限；
- unit-block 调度在所有 rank 上一致，并保持每个 unit 的采样集合不变；
- 三类任务使用正确的默认负例规则；
- retrieval loss 与参考实现的 hard+positive-only-in-batch 结构一致；
- MRL 公式和维度通过数值测试；
- 训练样本可按稳定 `sample_id` 审计，patch 通过单 unit 重建进入训练产物；
- 单卡 smoke train 和两 rank batch/loss 测试通过。
