# MindSmallReranking 评测路径与优化设计

本文记录本仓库当前环境中 MTEB 2.14.9 的实际行为。方案只在仓库内安装
runtime patch，不修改 `site-packages/mteb` 下的任何文件。

## 范围与保分数约束

优化必须保持以下内容不变：

- dataset instruction 和模型 prompt 处理后，实际送入 tokenizer 的 query/document
  文本；
- checkpoint、tokenizer、截断长度、pooling、归一化和相似度函数；
- 每条 reranking query 对应的候选文档集合；
- query ID、document ID，以及 `max_over_subqueries` 使用的 subquery 分组关系；
- MTEB 结果结构和所有指标，而不只是 main score。

实现应位于 `vibe_eval/`，以显式安装的 runtime patch 或本地 search wrapper
接入。不得直接编辑已安装的 MTEB。

## Retrieval 的当前评测路径

对于只实现 EncoderProtocol 的模型，`AbsTaskRetrieval._evaluate_subset()` 会创建
`RetrievalEvaluator`，并用 `SearchEncoderWrapper` 包装模型。

实际调用链为：

```text
AbsTaskRetrieval.evaluate
  -> AbsTaskRetrieval._evaluate_subset
    -> RetrievalEvaluator.__call__
      -> SearchEncoderWrapper.index(corpus)
      -> SearchEncoderWrapper.search(queries)
        -> create_dataloader(queries, prompt_type=query)
        -> model.encode(query_loader)
        -> _full_corpus_search(...)
          -> create_dataloader(corpus chunk, prompt_type=document)
          -> model.encode(corpus_loader)
          -> model.similarity(query_embeddings, corpus_embeddings)
          -> torch.topk 和 Python heap 聚合
    -> RetrievalEvaluator.evaluate
      -> calculate_retrieval_scores
      -> task_specific_scores
```

没有外部 index backend 时，`SearchEncoderWrapper.index()` 只保存 corpus，不会
立即编码。corpus 会在 `_full_corpus_search()` 中按 50,000 条分块编码。

## Reranking 的当前评测路径

MTEB 对 reranking 复用了 retrieval 的 task、evaluator 和 search wrapper。
真正决定执行分支的不是 `metadata.type == "Reranking"`，而是
`data_split["top_ranked"]` 是否非空：

```text
SearchEncoderWrapper.search
  -> 编码全部 query
  -> 如果 top_ranked 非空：
       _rerank_documents
         -> 编码 corpus
         -> 对每条 query：
              只 gather 该 query 的候选文档 embedding
              计算 query-candidate 相似度
              在候选集合内执行 torch.topk
     否则：
       _full_corpus_search
```

因此，即使任务类型是 Reranking，只要 `top_ranked` 加载失败，MTEB 仍会静默
退化为全库 retrieval。

## MindSmallReranking 数据规模

当前缓存 revision 为：

```text
227478e3235572039f4f7661840e059f31ef6eb1
```

已验证的 test split 规模如下：

| 内容 | 数量 |
| --- | ---: |
| qrels Arrow 原始行数 | 97,006,943 |
| 分组后的 query ID / query 行数 | 2,362,514 |
| 不重复 query text | 37,162 |
| corpus 文档数 | 5,277 |
| 每条 query 的平均候选数 | 约 41.1 |

不重复 query text 只占全部 query 的约 1.57%，同样的模型计算平均被重复约
63.6 次。

qrels 包含正例和 score=0 的负例，而不是只有正例。对原始 Arrow shard 的直接
对比表明，抽样 query 的 qrels document ID 与缓存 `top_ranked` 中的候选集合和
顺序完全一致。这是离线候选恢复能够成立的基础。

## 三段耗时的根因

### 1. GPU encoding 前的 DataLoader 准备

MTEB 2.14.9 为 text query 创建 DataLoader 前会调用
`_combine_queries_with_instruction_text()`：

```python
texts = dataset["text"]
dataset = dataset.add_column("query", texts)
```

`dataset["text"]` 会把 2,362,514 条 Arrow 字符串整体转成 Python list，然后
再复制成 `query` 列。该操作发生在日志 `Searching queries...` 之后、
`model.encode()` 之前，因此此时没有进度条，也没有 GPU 工作。

`f20b6-fix.log` 对应进程在该阶段约使用两个 CPU core、18.2 GB RSS、GPU 0%，
并在同一日志位置停留超过三小时。去掉本仓库旧的 `_collect_texts()` 并不会去掉
这一 MTEB 上游物化操作。

本地验证表明，直接复用 Arrow string column 添加等价的 `query` 列，在完整
2,362,514 行上约需 0.36 秒，样本内容与原函数一致。

### 2. query 重复编码

模型收到 2,362,514 行 query，但其中只有 37,162 种文本。在 task prompt、模型
prompt 和截断配置完全相同时，相同最终文本必然得到相同 embedding。旧的完整
运行在这里耗时约 40 到 45 分钟。

### 3. 离线运行没有加载 top_ranked

严格离线时：

```python
datasets.get_dataset_config_names(
    "mteb/MindSmallReranking",
    revision="227478e...",
)
```

只返回 `['default']`，即使本地缓存实际存在 `corpus`、`queries` 和
`top_ranked`。`RetrievalDatasetLoader` 因此不会调用 `_load_top_ranked()`。

日志也证实了这一点：

- 没有 `Loading top ranked subset: top_ranked`；
- 没有 `Reranking pre-ranked documents...`；
- `Searching queries...` 后实际进入全库搜索。

错误路径把约 9700 万个候选 pair 扩大为：

```text
2,362,514 queries * 5,277 documents = 12,466,985,378 pairs
```

随后还要为每条 query 取 top 1,000，并构造巨大的 tensor 和 Python 结果字典。
这是 encoding 后搜索阶段的主要开销，也不是该任务预期的候选重排语义。

### 4. 指标聚合

MTEB 先对全部 subquery 调用 `calculate_retrieval_scores()`，其中包括
`pytrec_eval`、MRR、confidence score 和 nAUC。随后 MindSmall 再调用
`max_over_subqueries()`，按 base query 合并 subquery 结果，再计算一遍 retrieval
指标。

旧完整运行在 18:20:31 进入 `Evaluating retrieval scores...`，到 19:26:06
结束，仅指标构造和聚合就约耗时 65 分钟。该耗时基于错误全库搜索产生的超大
结果；恢复候选重排后，指标输入会显著缩小。

## 默认方案：按 impression 保存紧凑数据

默认实现不再把 236 万条 subquery 和约 9700 万行 qrels 作为评测输入保存。
`vibe_eval.tasks.mind_small_compact_data` 从原 revision 做一次严格转换，并固定
写入：

```text
/mnt/share/emb/mteb_fix/mindsmallrerank
```

紧凑格式包含三个 Parquet 文件：

- `queries.parquet`：只保存 37,162 条不重复 query text，以文本 SHA-256 作为稳定 key；
- `corpus.parquet`：保存原始 5,277 篇文档；
- `impressions.parquet`：每个 impression 一行，保存完整 subquery ID、对应 query
  key、共享 `top_ranked` 候选 ID，以及共享 qrel ID/score。

同一 impression 的所有 subquery 必须具有完全相同的候选 ID、顺序和 qrel score，
否则构建直接失败。构建还会逐 query 检查 qrels 与 `top_ranked` 去重后的候选及
顺序完全一致、引用的 query/corpus 均存在，并在 manifest 中记录行数和文件
SHA-256。完整数据校验发现 `top_ranked` 中存在重复 document ID，因此它不能从
qrels 无损恢复：紧凑格式仍保存一份 impression 级候选列表，但不再为每个
subquery 重复保存；qrels 同样从 impression 级数据恢复成 MTEB 原有结构。

本地 `vibe_eval.tasks.mind_small_reranking.MindSmallReranking` 只编码唯一 query 和
corpus。打分时按 impression 取回该组 subquery embedding 与共享候选 embedding，
调用模型原有 `similarity()` 和 `torch.topk()`，然后恢复：

```text
results: subquery ID -> {candidate document ID -> model score}
qrels:   subquery ID -> {candidate document ID -> relevance score}
```

恢复后仍调用 MTEB 原版 `RetrievalEvaluator.evaluate()`、
`calculate_retrieval_scores()` 和 MindSmall 的 `max_over_subqueries()`，不重写指标。

## 保留方案：旧 runtime patch

旧实现保留在 `vibe_eval/tasks/mind_small_reranking_patch.py`，用于回退和差分
验证。该模块同时负责 compact/legacy 模式解析、本地 task 替换、任务排序和
legacy runtime hook 安装。`vibe_eval/mteb_patches.py` 只保留通用 query
DataLoader patch 与其他 reranking task 的候选加载 patch。

### Patch A：Arrow-native query 准备

在所有正式评测启动时，runtime 替换 MTEB 模块中已导入的
`_combine_queries_with_instruction_text` callable：

- 没有 dataset instruction 时，直接从底层 Arrow column 添加 `query` 别名；
- 有 `instruction` 列时，完整委托给 MTEB 原函数，不进入 Arrow 快速路径。

该优化适用于所有通过 `create_dataloader(..., prompt_type=query)` 的 text query
任务，包括 Retrieval 和 Reranking，而不只是 MindSmall。收益取决于 query 数量；
小任务差异很小，大规模无 instruction query 可以避免整列 Python 字符串物化。
单测比较快速路径前后的列名、行顺序和全部字段值，并验证 instruction 数据完整
回退原函数。

### Patch B：恢复 reranking 候选集合

只有同时满足以下条件才恢复候选：

- task 是 `MindSmallReranking`；
- revision 是已验证 revision；
- `top_ranked` 为空；
- qrels 同时包含正例和 score=0 的候选负例。

patch 直接复用 qrels 内层 mapping 的 document key 顺序，避免再次读取约 5 GB
的 `top_ranked` Arrow 数据，也避免复制约 9700 万个 ID。任何前置条件不成立都
直接报错，不允许 Reranking 任务静默执行全库 retrieval。

### Patch C：只编码唯一 query text

使用 Arrow dictionary encoding 得到 37,162 个唯一 query text 以及原行到唯一
文本的 inverse index。唯一文本经过原 MTEB DataLoader 和原模型 `encode()`，
task prompt、模型 prompt、tokenizer 与 pooling 均保持不变。

patch 不展开 2,362,514 × embedding_dim 的 query embedding 矩阵，从而避免
1024 维 float32 情况下约 9.7 GB 的重复 embedding。

### Patch D：只计算候选文档分数

5,277 篇 corpus 文档只编码一次。唯一 query embedding 分块调用模型原有
`similarity()`，得到唯一 query 到 corpus 的分数；输出阶段只提取每个 query
候选集合内的分数，并恢复 MTEB 需要的结构：

```text
query ID -> {candidate document ID -> float score}
```

普通 Retrieval 和其他 Reranking task 全部委托给原始
`SearchEncoderWrapper.search()`。

### Patch E：第一版保留 MTEB 指标实现

第一版不改 `calculate_retrieval_scores()`、`max_over_subqueries()` 或
`pytrec_eval` 调用。候选恢复后，结果从每条 query 最多 1,000 篇缩小到平均约
41 篇，应先重新 profile，再判断是否有必要优化指标阶段。

## 接入方式与环境变量

环境变量 `VIBE_MIND_SMALL_RERANKING_MODE` 控制实现：

```text
未设置或 compact：用本地紧凑 task（默认）
legacy：          用上游 task，并安装现有 runtime patch
```

其他值直接报错。两种模式对外仍使用同一个任务名 `MindSmallReranking`。默认模式
要求紧凑数据已经构建并通过 manifest 校验。评测入口不强制覆盖结果；需要重算时
应先显式改名备份原结果文件，或由调用方传入已有的覆盖参数。

`run_mteb.py` 不展开模式分支，只调用
`patch_mind_small_reranking_tasks(tasks)`；所有 MindSmall 专属选择和安装逻辑均
封装在上述 task patch 模块中。

legacy installer 具有以下约束：

- 幂等安装；
- 校验 MTEB 版本为 2.14.9；
- 保存原始 callable，普通任务可完整回退；
- 记录 patch 安装、候选恢复和 query 去重数量；
- 不识别的 dataset revision 直接拒绝执行假设性恢复。

## 验证门槛

1. Arrow-native query 处理与 MTEB 原函数做差分比较，覆盖有/无 instruction。
2. 完整 test split 校验 qrels 和 `top_ranked` 的候选集合及顺序，而不只抽样。
3. 构造包含多个 base query、重复文本、正负候选和分数 tie 的确定性切片。
4. 使用完全相同的预计算 embedding，分别运行原始正确 reranking 和 patch。
5. 比较每个 query-document score 和所有输出指标。计算顺序没变时要求精确相等；
   浮点运算顺序变化时，至少要求序列化后的 MTEB 指标完全相等。
6. 在无资源竞争条件下运行一次完整任务，记录各阶段 wall time、CPU、GPU 和
   peak RSS。
7. 完整结果 JSON 必须与正确但未优化的候选 reranking reference 对比。历史上的
   全库 retrieval 结果不是有效 reference。

## 预期效果

- Arrow patch：encoding 前的 query 转换由小时级降至秒级；
- 唯一文本编码：query 模型计算量约降低 63.6 倍；
- 正确候选重排：相似度规模由约 124.7 亿 pair 降到约 9700 万 pair；
- 指标输入：每条 query 由最多 1,000 个结果降到平均约 41 个结果。

这些是结构性估算。最终性能结论必须来自无 GPU/CPU/内存竞争的完整运行，并
明确区分数据加载、准备、编码、候选打分和指标计算时间。
