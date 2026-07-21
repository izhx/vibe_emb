# F2LLM-v2 Second-stage Indexed Arrow 数据格式设计

## 当前实现状态（2026-07-14）

格式的第一阶段实现已经落地：

- `scripts/f2llm/build_f2llm_indexed_arrow.py`：从 selection manifest 按 unit 构建
  `queries.arrow`、去重后的 `corpus.arrow`、`metadata.json` 和 profile
  `manifest.json`；支持 `--unit`、`--force`、`none|zstd`、固定 record batch 上限、
  临时 SQLite 去重和 unit 目录原子替换。局部 unit 重建会保留 manifest 中其他已构建
  unit。
- `scripts/f2llm/f2llm_indexed_arrow.py`：提供 `verify`、`verify --full` 和按稳定
  `sample_id` 的 `inspect`。
- `vibe_emb/record_store.py`：提供 manifest-only descriptor、按首次访问 mmap 的
  `IndexedArrowRecordStore` 和有界 `ArrowStorePool` LRU。
- `vibe_emb/data.py`：可以通过 `data.indexed_dataset_manifest` 载入 profile，同时保留
  原有 JSON/JSONL dataset；batch 内对 query/corpus 做批量读取。

真实 selection smoke 数据位于 `/tmp/f2llm_arrow_smoke`（临时验证产物，不提交）：

- retrieval `sts22`：389 query / 1,380 corpus；
- clustering `cedr`：4,376 query / 29,473 corpus；
- classification `dala`：6,508 query / 2 corpus。

三者已通过 full fingerprint、连续 query/doc ID、引用范围、sample ID 唯一性和
inspect round-trip 校验。完整 profile 当前包含 17,733,362 条有效 query；patch、export、
diff、节点本地 cache 和 Arrow unit 级异步 prefetch 仍属于后续工作。训练侧已经另外支持 PyTorch
DataLoader 的完整 batch prefetch；它不会提前复制完整 unit，也不使用
`arrow_prefetch_units`。

## 1. 目标与边界

本计划负责把采样计划选中的原始 Parquet 行转换为训练可读的数据格式。设计采用“每个 sampling unit 一个目录、query 与 corpus 分离、文档去重、不预 tokenize”。Arrow 是由原始 Parquet、selection plan 和可选 patch 编译出的只读训练产物，不作为人工编辑的 source of truth。

格式必须同时满足：

- 两节点、16 卡等分布式训练只读取轻量 manifest 即可生成 batch plan，启动时不打开或加载全部 unit；
- unit 数量较多时，每个训练进程的 mmap、文件描述符和 Arrow reader 数量都有明确上限；
- 任意训练样本可以追溯到原始 Parquet 行，修订通过可审计 patch 和单 unit 原子重建完成。

本阶段不决定采哪些行，也不实现 contrastive loss。输入和下游契约分别见：

- [数据采样计划](./f2llm_stage2_sampling_plan.md)
- [训练代码修改计划](./f2llm_stage2_training_code_plan.md)

## 2. 目录布局

一个 profile 的最终目录结构为：

```text
<output_dir>/
├── manifest.json
├── sampling_report.json
├── provenance/
│   ├── catalog_snapshot.yaml
│   ├── selection_manifest.json
│   └── patches.jsonl              # 没有修订时可不存在
└── units/
    ├── ocgi/
    │   ├── queries.arrow
    │   ├── corpus.arrow
    │   └── metadata.json
    ├── webfaq_eng/
    │   ├── queries.arrow
    │   ├── corpus.arrow
    │   └── metadata.json
    └── banking77/
        ├── queries.arrow
        ├── corpus.arrow
        └── metadata.json
```

多个 physical shards 可以构建成同一个 unit 目录。例如 `ocgi.parquet` 与 `ocgi_part2.parquet` 的选中行共同写入 `units/ocgi/`。不同 sampling unit 不共享 query 或 corpus 文件。

这样可以：

- 单独重建、验证或移动某个 unit；
- 避免单个超大 corpus 文件损坏后重建整个 profile；
- 训练时按需 memory-map unit，保持 same-dataset batching；
- 使用 unit-local doc ID，省去全局巨型索引。

不为了减少 mmap 数量而把多个 unit 打包进同一个大文件。mmap 数量由训练侧 lazy-open 和有界 LRU 控制；保持 unit 物理隔离更利于独立校验、修订、缓存和故障恢复。

`provenance/` 保存本次构建使用的 catalog、selection manifest 和非空 patch 日志快照。大体积 selection `.npz` 可以保留在上游采样目录，但 `manifest.json` 必须记录其路径和 fingerprint；`queries.arrow` 中的行级来源字段应能在没有 `.npz` 的情况下定位原始行。

## 3. Arrow schema

格式版本初始定义为 `indexed_arrow_v1`。

### 3.1 `queries.arrow`

| 字段 | Arrow 类型 | 必需 | 含义 |
|---|---|---:|---|
| `query_id` | `int64` | 是 | unit 内连续行号，从 0 开始 |
| `sample_id` | `fixed_size_binary[16]` | 是 | 由数据发布快照、原始 shard 相对路径和 shard-local row index 生成的稳定样本 ID |
| `source_shard_id` | `uint16` | 是 | 指向本 unit metadata 的 `source_shards` 表 |
| `source_row_index` | `int64` | 是 | 原始 Parquet shard 内的物理行号 |
| `query` | `large_string` | 是 | 已包含发布方 instruction 的原始 query |
| `positive_doc_id` | `int64` | 是 | 指向本 unit `corpus.arrow` 的正例 |
| `negative_doc_ids` | `list<int64>` | 是 | 指向本 unit corpus 的全部显式负例 |
| `lang` | `string` | 否 | 原始数据提供的语言代码 |

初版只支持单个 positive，因为 F2LLM-v2 发布 Parquet 使用单个 `passage` 字段。未来如需多个 positive，应升级 schema version，不能在同一版本中改变字段类型。

`query_id` 是构建产物的物理行号，重建后允许变化；审计、patch 和跨版本 diff 必须使用稳定 `sample_id`。不把 `source_id`、`sampling_unit_id`、`task_type` 重复写入每一行；它们属于 unit metadata。

### 3.2 `corpus.arrow`

| 字段 | Arrow 类型 | 必需 | 含义 |
|---|---|---:|---|
| `doc_id` | `int64` | 是 | unit 内连续行号，从 0 开始 |
| `text` | `large_string` | 是 | positive 或 negative 的原始文本 |

`doc_id` 必须等于 corpus 的物理行号。loader 可以直接使用 Arrow `take`，无需构造 `doc_id -> row` Python dict。

## 4. 文本保留与去重规则

- query、passage 和 negative 均保留原始字符串，不 trim、不 Unicode normalize、不删除 instruction。
- 文档仅在当前 sampling unit 内精确去重。
- 不跨 unit 去重；相同文本出现在不同任务或语言 unit 中时分别保存。
- query 不去重，每条选中的原始数据仍是一条训练 query。
- 空字符串、null、非字符串 positive/negative 直接视为数据错误。
- `negative_1` 到最大连续 `negative_N` 按数字顺序保存；列编号出现空洞时失败，避免静默漏负例。
- classification 通常只有一个 negative；retrieval/clustering 通常有 24 个。格式本身不硬编码数量，由 metadata 记录实际 `available_negative_count`。

原始数据中已经存在的 `Instruct: ...\nQuery: ...` 必须原样保留。新格式不保存额外 prompt，也不允许 loader 默认再次添加 instruction。

corpus 去重意味着一个 `doc_id` 可能被多个 query 引用。审计工具必须报告文档引用数；普通数据修订不能直接覆盖一个共享 `doc_id`，否则可能无意修改多条训练样本。

## 5. 去重和 doc ID 分配

每个 unit 独立构建，按 selection plan 中稳定的 shard 顺序和选中行顺序处理。

为避免把最多约 2.5M 个长文档一次放入 Python dict，使用临时磁盘索引：

1. 创建 unit 专属临时目录和临时 SQLite 数据库。
2. 对选中 query 分批读取 `passage` 和所有 `negative_N`。
3. 对文本计算稳定内容摘要，批量执行 insert-or-ignore。
4. 数据库同时保存完整文本，并对摘要冲突执行原文比较；发现不同文本摘要相同则失败。
5. 首次插入顺序决定连续的 unit-local `doc_id`。
6. 由 selection plan 的 shard 相对路径和 shard-local row index 重算并校验 `sample_id`，同时写入 `source_shard_id` 和 `source_row_index`。
7. 批量查询每条 query 对应的 positive/negative doc ID，写入临时 `queries.arrow`。
8. 按 `doc_id` 顺序流式导出 `corpus.arrow`。

临时数据库只用于构建，不进入最终数据目录。SQLite 参数应优先保证构建正确性和可恢复性；可以关闭不必要的同步开销，但不能在生成文件校验前删除数据库。

## 6. Arrow IPC 写入

- 使用 Arrow IPC file format，而不是 stream format，便于 memory map 和读取 record batch 元数据。
- `queries.arrow` 和 `corpus.arrow` 以固定上限的 record batch 分块写入，避免构建期大内存峰值。
- 字符串使用 `large_string`，避免大型 corpus 的 32-bit offset 限制。
- query 引用使用 `int64`，避免未来扩展时受 32-bit 行数限制。
- 是否启用 IPC 压缩作为构建参数写入 metadata；面向随机训练访问的默认值为 `none`。
- loader 不依赖固定 record batch 大小，而是根据 IPC metadata 建立累计行区间。

不默认使用 ZSTD 的原因是：随机命中某个字符串时，压缩 IPC 往往需要解压其所在 record batch 的完整 buffer；一个训练 batch 的 doc ID 跨多个 record batch 时会产生明显读放大。未压缩 IPC 更适合 mmap 按页读取，也能利用同一节点多个 rank 共享的文件页缓存。

ZSTD 仍作为存储空间优先或归档场景的可选项。启用压缩时，builder 应同时接受 `record_batch_target_bytes`，优先按未压缩目标字节数切分（建议 benchmark 8--32 MiB），而不是只按固定行数切分。生产配置必须在实际共享存储上比较 `none` 与 `zstd` 后确定。

## 7. `metadata.json`

每个 unit 的 metadata 至少包含：

```json
{
  "schema_version": "indexed_arrow_v1",
  "dataset_release_id": "f2llm-v2:<release-fingerprint>",
  "source_id": "ocgi",
  "sampling_unit_id": "ocgi",
  "task_type": "retrieval",
  "query_count": 100000,
  "corpus_count": 1234567,
  "available_negative_count": 24,
  "physical_shards": ["ocgi.parquet", "ocgi_part2.parquet"],
  "sample_limit": 100000,
  "sampling_seed": 42,
  "source_shards": [
    {
      "shard_id": 0,
      "path": "ocgi.parquet",
      "row_count": 2000000,
      "metadata_fingerprint": "..."
    },
    {
      "shard_id": 1,
      "path": "ocgi_part2.parquet",
      "row_count": 1500000,
      "metadata_fingerprint": "..."
    }
  ],
  "ipc_compression": "none",
  "queries_file_size": 123456789,
  "corpus_file_size": 2345678901,
  "selection_fingerprint": "...",
  "patches_fingerprint": null,
  "queries_fingerprint": "...",
  "corpus_fingerprint": "..."
}
```

还应记录：

- catalog/profile 版本；
- Arrow/pyarrow 版本；
- optional column 列表；
- 构建时间和构建脚本版本；
- query/corpus record batch 数量；
- query/corpus record batch 累计行区间和未压缩目标字节数；
- 文档去重前后的数量；
- `sample_id` 生成算法版本、唯一性校验结果；
- corpus 文档引用数统计，包括最大值和被多个 query 共享的文档数。

## 8. Profile `manifest.json`

profile manifest 是训练 loader 的入口，至少包含：

- schema version、profile ID、catalog digest；
- sampling report 相对路径；
- unit 有序列表；
- 每个 unit 的目录、metadata 相对路径、task type；
- 每个 unit 的 query/corpus 相对路径、行数、文件大小、IPC compression、record batch 数量和 fingerprint；
- 默认启用状态；
- MKQA/SIB200 等排除信息；
- profile 总 query 数和总 corpus 数；
- dataset release ID、selection manifest 路径/fingerprint、catalog snapshot 和 patch log fingerprint。

所有路径以 manifest 所在目录为基准。移动完整输出目录后无需修改绝对路径。

manifest 必须内联训练启动和 batch-plan 构建需要的运行时字段。训练进程不能为了获取 query count、task type、文件路径或 compression，在启动阶段依次打开 120 个 unit 的 `metadata.json` 或 Arrow footer。

文件 fingerprint 是离线构建和审计依据。训练启动默认不重新扫描、hash 全部 Arrow 文件；完整 checksum 校验由独立 `verify --full` 完成。loader 在某个 unit 第一次打开时校验文件大小、IPC footer、schema、metadata/manifest 一致性和引用范围。

manifest 只在所有目标 unit 构建和校验完成后写入。部分 unit 构建模式可以更新已有 manifest，但必须先验证未重建 unit 的 fingerprint。

## 9. 原子构建与恢复

单 unit 构建流程：

1. 在 `units/.tmp-<unit>-<id>/` 中构建所有临时文件。
2. 校验 Arrow schema、行数、doc ID 和引用完整性。
3. 写入并 fsync `metadata.json`。
4. 将临时目录原子 rename 为 `units/<unit>/`。
5. 已存在目标时，只有显式 `--force` 或审计 patch 重建流程才允许先备份再替换。

中断后：

- 正式 unit 目录保持可读；
- `.tmp-*` 可以由下一次运行识别、清理或恢复；
- 不能出现缺少 metadata 却被 manifest 引用的 unit。

修订已有数据时，推荐写入新的 profile 输出目录。若原地替换单 unit，必须先完成临时目录构建和完整校验，再原子替换 unit，并最后原子更新 profile manifest；任何时刻 manifest 都不能引用一半新、一半旧的同一 unit 文件。

## 10. 构建 CLI

建议入口：

```bash
python scripts/f2llm/build_f2llm_indexed_arrow.py \
  --selection-manifest data/processed/ml_embed_stage2_selection/selection_manifest.json \
  --output-dir data/processed/ml_embed_stage2_arrow
```

支持：

- `--unit <id>`：只构建指定 unit，可重复传入；
- `--force`：替换已经完整构建的目标 unit；
- `--record-batch-size`：覆盖默认 batch 行数；
- `--record-batch-target-bytes`：设置未压缩 record batch 目标字节数；
- `--compression zstd|none`：默认 `none`；
- `--patches <path>`：应用基于稳定 `sample_id` 的修订日志；
- `--keep-build-db`：调试时保留临时去重数据库；
- `--verify-only`：只校验现有 Arrow 数据。

## 11. 多节点 Loader 访问契约

训练代码只依赖以下行为：

- manifest 可枚举所有启用 unit 及其 task type；
- `queries.arrow` 可按 query row index 批量读取；
- `positive_doc_id` 和每个 `negative_doc_id` 都是 corpus 有效行号；
- `corpus.arrow` 可按 doc ID 批量 `take`；
- query 的负例顺序稳定，训练阶段再按 epoch seed 选择需要的 hard negatives；
- unit 文件构建完成后只读，不在训练时修改。

### 11.1 Manifest-only 启动

训练启动阶段只读取 profile `manifest.json` 并创建轻量 `UnitDescriptor`。不能为所有 unit 预先实例化 Arrow store、打开 `metadata.json`、读取 IPC footer、建立 mmap 或扫描文件 checksum。

batch plan 只能依赖 manifest 中的 unit 顺序、query count、task type、batch size、sampling seed 和配置覆盖生成。每个 rank 使用相同输入独立得到一致的 global plan，不需要广播训练文本。

启动校验分层：

- `manifest`（训练默认）：校验 manifest schema、自身 fingerprint 和配置一致性，不扫描大文件；
- `lazy`：在 unit 第一次使用时校验该 unit 的 metadata、文件大小、footer、schema 和引用；
- `full`（离线/显式）：读取全部 unit 并重算 checksum、验证所有引用。

共享存储的全量 preflight 如需检查所有路径，应由每节点一个 local rank 执行并把结果同步给同节点其他 rank，避免 16 个进程同时重复发起 metadata 请求。

### 11.2 Lazy mmap 与有界 LRU

- `IndexedArrowRecordStore` 在 unit 第一次被实际 batch 访问时才创建；
- 每个训练进程维护有上限的 unit LRU，建议默认缓存 4 个 unit，即通常最多 8 个 query/corpus Arrow mmap；
- 可预取下一个 unit，但预取也计入 LRU 上限；
- eviction 必须关闭 Arrow reader、memory map 和文件句柄；
- `get_records()` 返回前将本 batch 需要的字符串转换为普通 Python 对象，不能把引用 mmap buffer 的 Arrow array 暴露到 store 外；
- DataLoader worker 的框架默认保持 0；显式启用 batch prefetch 时，每个 worker 拥有独立的
  store cache 和 mmap，因此资源上界按 `workers * arrow_max_open_units` 计算；
- trainer 正常结束和异常清理路径都必须调用 `close()`。

若 120 个 unit 的两个文件被所有 8 个 rank eager mmap，则约为 240 mappings/rank、1920 mappings/node；这是节点总量，不是单进程 `vm.max_map_count`。通常不会立即耗尽物理内存，但会增加文件描述符、共享存储 metadata 请求、页表和虚拟地址空间，并放大随机页缓存抖动，因此不能作为默认实现。

### 11.3 调度局部性

仅有小 LRU 而 batch plan 在 120 个 unit 间逐 batch 完全随机切换时，缓存会频繁失效。训练侧应把每个 unit 已生成的 batch 划分为固定大小 block，再以相同 seed 打乱 block：

```text
webfaq_eng: 8 batches -> ocgi: 8 batches -> banking77: 8 batches -> ...
```

建议 second-stage 默认 `unit_block_batches=8`，并允许 1（保持旧的逐 batch 混排）、4、16、32 等值。它只改变 unit 之间的交错粒度，不改变每个 unit 选择的 query、same-dataset batching、rank-local slice 或正负例随机种子。所有 rank 必须执行相同 block 顺序。

### 11.4 共享存储与节点本地缓存

同一节点的多个 rank mmap 相同 inode 时可以共享内核文件页缓存，但 Arrow footer、Python 对象、页表和解压缓冲区仍是进程私有。格式不得把 mmap 作为唯一读取后端；同一 IPC 文件也应能通过普通文件读取或可选的 node-local cache 打开。

如果共享文件系统的小随机读较慢，可以由每节点 local rank 依据 manifest fingerprint 把即将使用的完整 unit 预取到本地 NVMe，其他本地 rank 在 barrier 后打开本地副本。该功能是可选优化，不能改变 unit fingerprint、doc ID 或 batch plan。

## 12. 审计与修改

### 12.1 审计工具

至少提供以下只读接口：

- `inspect --unit <id> --sample-id <hex>`：输出 query、展开后的 positive/negatives、doc 引用数、task type 和原始 shard/row；
- `export --unit <id> --output <jsonl>`：按 `sample_id` 导出可读训练 tuple；
- `verify --unit <id>` 和 `verify --full`：校验 schema、ID、引用、fingerprint 和 provenance；
- `diff --profile-a <path> --profile-b <path>`：按稳定 `sample_id` 比较新增、删除和文本变化。

审计输出使用普通 JSON/JSONL；不要求用户直接读取或修改 Arrow 二进制文件。

### 12.2 Patch 语义

修订记录使用可读 `patches.jsonl`，主键为 `sample_id`。初版支持：

- `exclude_sample`；
- `replace_query`；
- `replace_positive`；
- `replace_negative`，必须指定原 negative position；
- `replace_all_negatives`。

每条 patch 至少包含：

```json
{
  "patch_id": "20260713-0001",
  "sample_id": "0123456789abcdef0123456789abcdef",
  "operation": "replace_negative",
  "negative_index": 2,
  "before_fingerprint": "...",
  "text": "replacement text",
  "reason": "manual audit correction"
}
```

`before_fingerprint` 用于防止 patch 静默应用到已经变化的 tuple。patch 文件按行顺序确定性应用；同一 `sample_id`/字段出现相互冲突的操作时默认失败，只有语义明确的顺序操作才能显式允许。构建 metadata 记录 patch 文件 fingerprint、已应用/跳过/失败计数和 patch tool 版本。

普通 patch 表达的是某个 query tuple 上的边修改。builder 应用 patch 后重新执行 unit-local corpus 去重并重新分配 doc ID。禁止默认把 `replace_positive` 或 `replace_negative` 翻译成原地覆盖共享 corpus row。

如确实需要修改所有引用同一文本的样本，必须使用独立的显式全局操作，并在执行前报告受影响的 unit、sample 和引用数量。

### 12.3 重建与版本化

标准修改流程：

1. 使用 `inspect` 或 `export` 确认原始样本及来源；
2. 将变更追加到版本化 patch 文件；
3. 只重建受影响的 unit；
4. 完整校验新 unit，并输出 patch fingerprint 和 diff report；
5. 原子发布新 unit 和 manifest，优先生成新的 profile 版本目录。

训练 loader 不动态应用 patch。patch 必须在训练前编译进 Arrow，保证热路径性能、数据 fingerprint 稳定和实验可复现。

## 13. 验证与测试

- 多 shard unit 的选中行全部进入同一个 unit 目录。
- unit 内重复文档只写入 corpus 一次，所有引用指向同一 doc ID。
- 相同文本跨 unit 时分别存在。
- `doc_id == corpus row index`，且从 0 连续到 `corpus_count - 1`。
- 所有 positive/negative doc ID 均在范围内并能还原原文。
- query 顺序与 selection plan 的稳定顺序一致。
- classification 的单负例和 retrieval/clustering 的多负例均能 round-trip。
- 可选 `lang` 在缺失时不影响统一 loader 接口。
- 构建中断不会污染已经完成的 unit。
- 指定 `--unit` 能独立重建且不修改其他 unit fingerprint。
- memory-map 后按随机 query/doc ID 批量读取结果正确。
- metadata、manifest、Arrow 实际行数和 sampling report 完全一致。
- `sample_id` 能定位到 catalog shard 和原始 Parquet row，跨 profile 保持稳定。
- manifest-only 启动不打开任意 unit Arrow 文件或 metadata。
- LRU 实际打开 unit 数不超过配置上限，eviction 后不遗留 mmap/file descriptor。
- 120-unit fixture 在 8-rank 模拟下不会 eager 打开 1920 个节点级映射。
- unit-block 调度在所有 rank 上顺序一致，且不改变各 unit 的 query 集合。
- `compression=none` 和 `zstd` 均能 round-trip；随机访问 benchmark 单独报告读放大和吞吐。
- patch 后只重建目标 unit；共享 corpus 文档的 tuple-local 修改不会影响其他引用。
- `inspect`、`export` 和 profile diff 能按 `sample_id` 还原修订前后差异。

## 14. 交付条件

本阶段完成的标志是：采样计划中的每个启用 unit 都能生成独立、校验通过、可按原始行审计和单独重建的 query/corpus Arrow 数据；profile manifest 能作为训练 loader 的唯一启动入口；两节点多卡训练可用 manifest-only + lazy LRU 快速启动，且不依赖预 tokenize、全局 corpus 索引或 eager 打开全部 unit。

## 15. 全量 80k profile 构建结果（2026-07-14）

`f2llm_stage2_80k` selection 已全部编译到
`data/processed/f2llm_stage2_80k_arrow/`：

- 395/395 个启用 sampling unit，覆盖 351 retrieval、33 clustering、11 classification；
- selection 共 17,733,380 行，构建后保留 17,733,362 个有效 query；
- FiQA2018 的 4 条空 positive 和 TwentyNewsgroups 的 14 条空 positive 被剔除；
- 21 个 unit 共过滤 1,111 个空或非字符串 negative，过滤计数写入 unit metadata；
- unit-local corpus 去重后共有 90,313,442 条文档；
- 无压缩 Arrow 总占用约 75 GiB；profile manifest 是
  `data/processed/f2llm_stage2_80k_arrow/manifest.json`。

构建命令：

```bash
/mnt/share/envs/embt/bin/python scripts/f2llm/build_f2llm_indexed_arrow.py \
  --selection-manifest data/processed/f2llm_stage2_80k_selection/selection_manifest.json \
  --output-dir data/processed/f2llm_stage2_80k_arrow \
  --compression none \
  --build-db-dir /tmp \
  --resume
```

`--build-db-dir` 将 SQLite 去重临时库放在节点本地盘，避免共享文件系统随机 I/O；
`--resume` 只复用 profile、release 和 selection fingerprint 全部匹配且文件完整的原子 unit。
构建器保持 query/positive 严格非空，对无效 query tuple 做可计数剔除，并只过滤无效
negative。训练前校验命令：

```bash
/mnt/share/envs/embt/bin/python scripts/f2llm/f2llm_indexed_arrow.py \
  --manifest data/processed/f2llm_stage2_80k_arrow/manifest.json verify --full
```
