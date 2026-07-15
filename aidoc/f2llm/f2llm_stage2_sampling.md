# F2LLM-v2 / ML-Embed second-stage 采样复现

本仓库按 `aidoc/f2llm/f2llm_stage2_sampling_plan.md` 对 `data/F2LLM-v2/` 的公开 Parquet
快照生成可复现的 second-stage 行索引。采样阶段不读取训练文本，也不生成 Arrow 或
tokenize 产物。

## 事实来源与口径

- `configs/data/f2llm_v2_sources.yaml` 是来源、sampling unit 和物理 shard 归属的唯一
  事实来源。`ml_embed_stage2` 的 121 个来源来自 ML-Embed 附录 Table 8/9；
  `f2llm_stage2_80k` 的 157 个来源来自 F2LLM-v2 数据表，完整覆盖发布快照的 500 个
  Parquet。
- `ml_embed_stage2` 每个 sampling unit 最多选择 100,000 行；`f2llm_stage2_80k`
  最多选择 80,000 行。两者均采用 global seed 42、均匀无放回采样。
- `partN` 是同一 unit 的物理分片，合并逻辑行空间后只应用一次上限。
- 已独立发布为多个语言、方向或编程语言文件的数据，每个文件是独立 unit。
- 为复现 F2LLM-v2 报告的约 18M，80k profile 将 UNPC、ParaCrawl、CLIRMatrix 的
  方向 shards 分别合并成三个论文级 source 后应用上限；若将这些方向文件各采 80k，
  总量会膨胀至约 25M，与论文不符。
- 公开快照把 MELA、ScaLA、DaLA 顺序拼接在 `mela.parquet`。catalog 使用论文报告的
  精确边界 `[0, 40267)`、`[40267, 168738)`、`[168738, 175246)` 恢复三个来源。
- 数据集快照为 Hugging Face revision
  `d520b8ad02c86d5e5611441c6196ff65d8888927`。ML-Embed profile 默认按当前 MTEB
  建议排除 MKQA/SIB200；F2LLM-v2 80k profile 为复现原论文训练混合而保留它们。

论文附录 121 行的 Size 总和为 50,513,802；直接按附录行限采 100k 时是
8,317,566。计划要求语言/方向物理子集独立成为 unit，因此本地 profile 默认得到
23,747,341 条，而不是通过全局缩放强行凑成 8.3M。该偏差会明确写入 warning。

F2LLM-v2 的 157 行 Size 总和为 60,147,938，与本地 500 个 Parquet metadata 总行数
完全一致。按上述 source grouping 每项限采 80k 后得到 17,733,380，和论文“约 18M”
相差 1.48%，不做补采或全局缩放。

## 运行

先做 metadata-only dry-run：

```bash
python scripts/f2llm/prepare_f2llm_stage2_sampling.py \
  --data-root data/F2LLM-v2 \
  --catalog configs/data/f2llm_v2_sources.yaml \
  --profile ml_embed_stage2 \
  --output-dir /tmp/ml_embed_stage2_selection_dry \
  --seed 42 \
  --dry-run
```

生成正式索引：

```bash
python scripts/f2llm/prepare_f2llm_stage2_sampling.py \
  --data-root data/F2LLM-v2 \
  --catalog configs/data/f2llm_v2_sources.yaml \
  --profile ml_embed_stage2 \
  --output-dir data/processed/ml_embed_stage2_selection \
  --seed 42
```

生成 F2LLM-v2 157-source / 80k 的第二份索引：

```bash
python scripts/f2llm/prepare_f2llm_stage2_sampling.py \
  --data-root data/F2LLM-v2 \
  --catalog configs/data/f2llm_v2_sources.yaml \
  --profile f2llm_stage2_80k \
  --output-dir data/processed/f2llm_stage2_80k_selection \
  --seed 42
```

输出目录包括 `selection_manifest.json`、`sampling_report.json`、catalog 快照和
`indices/*.npz`。NPZ 中 key 是规范化 shard 相对路径，value 是严格递增的物理
shard-local `int64` 行号。

常用覆盖参数：

- `--include-contaminated`：恢复所有 MKQA/SIB200 unit。
- `--include-unit ID` / `--exclude-unit ID`：显式覆盖单个 unit。
- `--unit ID`：只生成或检查指定 unit。
- `--sample-limit N`：覆盖默认上限。
- `--force`：仅在旧 manifest、report 和所有旧 NPZ 均通过结构、范围、递增性和计数
  校验后覆盖旧产物。

## 当前 seed=42 产物

### ML-Embed 121-source / 100k

- catalog 来源：121（默认启用 119，污染排除 2）。
- sampling unit：409（默认启用 382）。
- 唯一物理 shard：417。
- 选择 query：23,747,341。
- task query：retrieval 21,308,110；clustering 2,102,782；classification 336,449。
- 正式目录：`data/processed/ml_embed_stage2_selection/`，共 385 个文件、
  193,220,535 bytes。
- 相同 seed 全量覆盖重建后的目录 SHA-256：
  `7eec0addafd17cd016dbbf87d7f9e602ed98a824d6acabc779438dc481b869f3`。

### F2LLM-v2 157-source / 80k

- catalog 来源：157，全部启用。
- sampling unit：395。
- 唯一物理 shard：500，覆盖完整公开快照。
- 选择 query：17,733,380。
- task query：retrieval 15,236,346；clustering 2,012,916；classification 484,118。
- 正式目录：`data/processed/f2llm_stage2_80k_selection/`，共 398 个文件、
  145,543,765 bytes。
- 相同 seed 全量覆盖重建后的目录 SHA-256：
  `3adb684b86d43ccbd32b4c59f60cba1e44ddc7fb1e78425961ce5bafd4444406`。

测试命令：

```bash
pytest -q tests/test_prepare_f2llm_stage2_sampling.py
```

测试覆盖多 shard 合并上限、profile 级 source regrouping、语言 unit 独立上限、同 seed
逐字节一致、不同 seed
变化、小 unit 全量保留、逻辑行映射、row-group 定位、稳定 sample ID、dry-run、重复
shard 检测，以及同一公开文件中不重叠来源区间的校验。
