# F2LLM-v2 / ML-Embed Second-stage 数据采样计划

## 1. 目标与边界

本计划负责从 `data/F2LLM-v2/` 的原始 Parquet 中确定 second-stage 训练所使用的行，并产出可复现、可审计的采样索引和稳定来源标识。它不负责文档去重、Arrow 数据构建、tokenize 或训练。

默认采用 ML-Embed second-stage profile：以论文附录列出的 121 个数据来源为 catalog 基础，对每个采样单位最多保留 100,000 条 query。论文报告最终训练规模约为 8.3M，但公开文件、论文来源口径和参考代码之间存在差异，因此实际总量只作为审计指标，不通过全局缩放强行凑齐。

相关文档：

- [Indexed Arrow 数据格式](./f2llm_stage2_indexed_arrow_format.md)
- [训练代码修改计划](./f2llm_stage2_training_code_plan.md)
- [ML-Embed 论文](https://arxiv.org/pdf/2605.15081)
- [F2LLM-v2 论文](https://arxiv.org/pdf/2603.19223)

## 2. 数据组织模型

采样 catalog 明确区分三级概念。

### 2.1 Source

`source_id` 表示论文附录中的数据来源或由该来源生成的任务条目，例如 WebFAQ、OCGI、Amazon Reviews。ML-Embed profile 只启用论文对应的 121 个来源；后续可以增加完整 F2LLM-v2 profile，但不能依赖模糊的文件名前缀动态生成。

### 2.2 Sampling unit

`sampling_unit_id` 是实际应用 100k 上限、独立构造训练 batch 的单位：

- 同一份数据因文件大小拆成多个 `partN` 时，多个文件属于同一个 sampling unit。
- 论文明确按语言子集独立采样的数据，每个语言分别是一个 sampling unit。
- 同一上游数据生成的不同任务格式，如果论文将其作为不同任务条目，则分别建 unit。

示例：

```yaml
sources:
  - source_id: ocgi
    task_type: retrieval
    sampling_units:
      - sampling_unit_id: ocgi
        shards:
          - ocgi.parquet
          - ocgi_part2.parquet

  - source_id: webfaq
    task_type: retrieval
    sampling_units:
      - sampling_unit_id: webfaq_eng
        shards: [webfaq_eng.parquet]
      - sampling_unit_id: webfaq_zho
        shards: [webfaq_zho.parquet]
```

### 2.3 Physical shard

`physical_shards` 是 `data/F2LLM-v2/` 中实际存在的 Parquet 文件。每个被启用的文件必须在 profile 内恰好属于一个 sampling unit。以下情况在 dry-run 阶段直接报错：

- catalog 引用了不存在的文件；
- 同一个文件被两个 unit 重复引用；
- profile 声明必须覆盖的文件没有映射；
- URL 编码名称或特殊字符名称未显式映射，例如 `C%23`；
- 同一个 unit 的 shard 任务类型或核心 schema 不兼容。

## 3. Catalog 与 profile

新增一个版本化 catalog 文件，记录：

- catalog schema version；
- `source_id`、`sampling_unit_id` 和 shard 列表；
- `task_type: retrieval | clustering | classification`；
- 语言、方向或编程语言子集；
- 默认采样上限；
- profile 是否启用该 source/unit；
- 数据污染排除标记和原因；
- 论文表格名称与本地文件名之间的显式映射。

默认 profile 为 `ml_embed_stage2`：

- sample limit：每 sampling unit 100,000；
- seed：42；
- 默认排除 MKQA 和 SIB200；
- 允许通过命令行显式恢复被排除的 unit；
- 实际总量偏离约 8.3M 时警告，但继续生成采样计划。

catalog 是来源归属的唯一事实来源。文件名规则只能用于生成或校验 catalog 草案，不能在正式采样时隐式改变分组。

## 4. 确定性采样算法

### 4.1 统一逻辑行号

对一个 sampling unit 的 shard 按 catalog 中的固定顺序排列。仅通过 Parquet metadata 获取各 shard 行数，并构造连续的逻辑行号空间：

```text
shard_0: [0, rows_0)
shard_1: [rows_0, rows_0 + rows_1)
...
```

因此 OCGI 的两个 shard 是先合并为一个逻辑总体，再从总体中采 100k，而不是每个文件各采 100k 或固定各取 50k。

### 4.2 随机数

每个 unit 使用独立、稳定的随机种子：

```text
unit_seed = stable_hash(global_seed, profile_id, sampling_unit_id)
```

使用 NumPy `Generator(PCG64)` 在 `[0, total_rows)` 上均匀、无放回抽样：

- `total_rows <= sample_limit`：选择全部逻辑行；
- `total_rows > sample_limit`：选择 `sample_limit` 个逻辑行；
- 采样结果排序后再映射回 shard 和 shard-local row index，以便后续按 row group 顺序读取。

相同 catalog、profile、seed 和 Parquet metadata 必须生成完全相同的索引。文件系统枚举顺序不得参与随机性。

### 4.3 Row-group 定位

利用每个 Parquet row group 的累计行数，把 shard-local row index 映射到：

```text
(shard_path, row_group_index, row_offset_in_group)
```

采样阶段不读取文本列。正式 Arrow 构建时只读取命中的 row group，并从中 `take` 被选行，避免全量扫描 500 个大文件。

### 4.4 稳定样本 ID

每个选中原始行都定义一个与训练文本内容无关的稳定 `sample_id`：

```text
sample_id = BLAKE2b-128(
  dataset_release_id,
  normalized_shard_relative_path,
  shard_local_row_index
)
```

约束：

- `dataset_release_id` 标识本次 F2LLM-v2 发布快照，并记录在 catalog 和 selection manifest 中；
- shard 路径相对 `data_root` 规范化，不能使用机器相关的绝对路径；
- 不加入 query/passage/negative 文本、profile ID、sampling seed 或选择顺序；在同一 release namespace 内应用文本 patch、切换 profile 和重新采样都不会改变同一原始行的 ID；
- 二进制形式固定为 16 bytes；JSON、日志和 patch 文件使用 32 位小写十六进制字符串；
- 同一 `dataset_release_id` 内出现重复 ID 时直接失败。

Arrow builder 可以由 shard 相对路径和 shard-local row index 重算 `sample_id`，不需要在 `.npz` 中重复保存 ID。该 ID 是后续审计、diff 和数据修订的稳定主键；Arrow 中的 `query_id` 仍只是某次构建的 unit-local 物理行号。

## 5. 采样计划输出

采样脚本输出一个中间目录：

```text
<sampling_plan_dir>/
├── selection_manifest.json
├── sampling_report.json
├── catalog_snapshot.yaml
└── indices/
    ├── ocgi.npz
    ├── webfaq_eng.npz
    └── ...
```

### 5.1 `selection_manifest.json`

记录：

- schema version；
- dataset release ID、profile ID、catalog version 和 catalog digest；
- 原始数据根目录；
- global seed、默认 sample limit；
- unit 列表及其 task type、语言、启用状态；
- unit 对应的 shard 顺序、行数、metadata fingerprint；
- unit 的索引文件相对路径；
- 排除项和显式覆盖项。

### 5.2 Unit 索引文件

每个 `.npz` 使用 shard 名作为 key，value 为已排序的 shard-local `int64` 行索引。JSON 不直接承载数百万行索引。

索引文件必须满足：

- 每个数组严格递增且无重复；
- 索引范围落在对应 shard 行数内；
- 所有 shard 数组长度之和等于 unit 的 selected count；
- 对少于上限的 unit，索引必须覆盖全部行。

shard key 必须与 catalog 中规范化的相对路径一一对应。`selection_manifest.json` 同时记录每个 shard 的稳定 `shard_id`、相对路径和 metadata fingerprint，供 Arrow builder 写入行级来源字段。

### 5.3 `sampling_report.json`

逐 unit 记录：

- source ID、sampling unit ID、task type、语言；
- 原始总行数、sample limit、最终选择数；
- 各 shard 原始行数和命中数；
- 是否排除以及原因；
- schema 摘要和 warning；
- selection fingerprint；
- `sample_id` 数量、唯一性校验结果和生成算法版本。

报告末尾汇总 profile 的 source 数、unit 数、physical shard 数、三类任务数量和最终 query 总数。

## 6. CLI 设计

建议新增入口：

```bash
python scripts/f2llm/prepare_f2llm_stage2_sampling.py \
  --data-root data/F2LLM-v2 \
  --catalog configs/data/f2llm_v2_sources.yaml \
  --profile ml_embed_stage2 \
  --output-dir data/processed/ml_embed_stage2_selection \
  --seed 42 \
  --dry-run
```

正式生成去掉 `--dry-run`。其他参数：

- `--include-contaminated`：恢复 MKQA、SIB200；
- `--include-unit` / `--exclude-unit`：显式覆盖 profile；
- `--sample-limit`：全局覆盖 100k 上限；
- `--allow-count-mismatch` 不需要作为开关，因为默认策略已经是警告继续；
- `--force`：仅允许覆盖完整且校验通过的旧采样计划；
- `--unit`：只生成或核验指定 unit。

dry-run 只打印并输出统计草案，不写索引文件。

## 7. 错误与告警策略

直接失败：

- catalog/profile 不合法；
- shard 缺失、重复归属或 metadata 无法读取；
- unit 内核心字段不兼容；
- classification 缺少 `negative_1`；
- retrieval/clustering 缺少预期负例列；
- 索引越界、重复或计数不一致。

警告继续：

- 最终总量不等于论文约 8.3M；
- profile 中某个论文来源在当前发布版本不可用，但 catalog 明确标记为 optional；
- unit 小于 100k，因此全量保留；
- 可选字段如 `lang`、`label` 在部分 shard 中不存在。

## 8. 测试与验收

- 两个 `partN` fixture 合计只应用一次采样上限。
- 两个语言 unit 分别应用采样上限。
- 相同 seed 输出逐字节一致的索引；不同 seed 改变大数据集的选择。
- 多 shard 采样与把 shard 先逻辑拼接后直接采样的结果一致。
- 小于上限时选择全部行且不改变行号。
- row-group 定位能准确还原选中原始记录。
- 同一原始行在不同 profile、sample limit 和 seed 下生成相同 `sample_id`。
- 在同一 dataset release namespace 内，文本字段不直接参与 `sample_id` 计算；改变 dataset release ID 会改变 ID namespace。
- `sample_id`、shard 相对路径和 shard-local row index 能唯一定位选中原始记录。
- catalog 缺文件、重复文件、未映射文件和特殊文件名映射都有覆盖测试。
- MKQA/SIB200 默认排除，显式恢复后重新进入报告。
- 总量不匹配只产生 warning，不导致非零退出。
- `selection_manifest.json`、`.npz` 和 report 的计数完全一致。

## 9. 交付条件

本阶段完成的标志是：ML-Embed profile 能对本地数据完成 dry-run 和索引生成，所有 unit 均有可复现选择结果、稳定 `sample_id` 与原始行审计记录；尚不生成训练可读的 query/corpus Arrow 文件。
