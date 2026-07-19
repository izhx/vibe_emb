# TwitterHjerneRetrieval indexed query 行数错误与修复

本文记录 `MTEB(Multilingual, v2)` 评测在 `TwitterHjerneRetrieval` 上触发的
query DataLoader 行数错误、实际数据特征、根因、仓库内修复和验证结果。

本文基于当前评测环境：

- MTEB 2.14.9；
- Hugging Face `datasets` 来自 `/mnt/share/envs/embt`；
- 数据集 `mteb/TwitterHjerneRetrieval` revision
  `97ad55673cf9746f8e4b3aaa92b1bb92d82e52db`；
- 严格离线复用 `/mnt/share/emb/mteb_cache` 中的数据缓存。

## 问题现象

运行完整 multilingual v2 benchmark：

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
nohup python -u -m vibe_eval.run_mteb \
  --benchmarks 'MTEB(Multilingual, v2)' \
  --model_name_or_path /mnt/share/models/F2LLM-v2-0.6B \
  --batch_size 128 \
  --max_length 8192 \
  >> results/mteb_eval/f20b6-mm.log 2>&1 &
```

评测在 `TwitterHjerneRetrieval` 的 query DataLoader 创建阶段失败：

```text
ValueError: Failed to concatenate on axis=1 because tables don't have the same number of rows
```

异常来自仓库 runtime patch 中的 query 列构造，而不是模型 forward、GPU、batch
size、最大长度或相似度计算。日志中的 `73/131` 是 benchmark task 进度，不是
query 行号。

## 实际数据特征

从当前离线缓存加载该任务得到：

| 数据 | 行数 |
| --- | ---: |
| 原始 query | 78 |
| qrels 覆盖的不同 query | 77 |
| MTEB 过滤后的 query 逻辑视图 | 77 |
| 过滤后 Dataset 的底层 Arrow 物理表 | 78 |

没有正例、因而被过滤掉的 query ID 是 `63`。

这个数据状态不是缓存损坏。MTEB 的
`_filter_queries_without_positives()` 会根据 qrels 构造保留位置，再执行：

```python
queries = queries.select(indices)
```

Hugging Face Dataset 的 `select()` 可以只增加一层 `_indices` 映射，而不立即
重写底层 Arrow 表。因此此时：

```text
len(dataset) == 77
dataset.data.num_rows == 78
dataset._indices is not None
```

前者是评测应该看到的逻辑行数，后两项描述其物理存储状态。

## 根因

为避免在 MindSmall 等大规模 query 数据上把整列物化成 Python list，仓库曾将
MTEB 的 query 列构造替换为 Arrow 快路径：

```python
return dataset.add_column("query", dataset.data.column("text"))
```

问题在于 `dataset.data.column("text")` 直接访问底层物理 Arrow 表，不应用
Dataset 的 `_indices` 映射。在 TwitterHjerne 数据上，该表达式返回 78 行。

另一方面，`Dataset.add_column()` 检测到 `_indices` 后，会先调用
`flatten_indices()`，把 Dataset 本身展平成逻辑上的 77 行，然后再横向拼接传入
的列。最终参与拼接的两侧分别为 77 行和 78 行，因此抛出行数不一致异常。

调用链如下：

```text
AbsTaskRetrieval._evaluate_subset
  -> _filter_queries_without_positives
    -> queries.select(indices)                 # 逻辑 77，物理 78
  -> SearchEncoderWrapper.search
    -> create_dataloader
      -> _prepare_dataset
        -> _combine_queries_arrow_native
          -> dataset.data.column("text")       # 错误读取物理 78 行
          -> dataset.add_column("query", ...)
            -> flatten_indices()               # Dataset 展平为 77 行
            -> concat_tables(axis=1)            # 77 != 78，失败
```

MTEB 原始实现使用 `dataset["text"]`，会尊重逻辑索引，因此不会发生这个错误；
但它会返回 Python 对象，不符合本仓库为大数据 query 保留 Arrow 路径的目的。

## 仓库内修复

修复位于 [`vibe_eval/mteb_patches.py`](../../vibe_eval/mteb_patches.py)：

```python
texts = dataset.with_format("arrow")["text"]
return dataset.add_column("query", texts)
```

`with_format("arrow")["text"]` 同时满足两个条件：

1. 通过 Dataset 逻辑视图读取，正确应用 `_indices`，本例返回 77 行；
2. 返回 Arrow `ChunkedArray`，不会把整列转换成 Python list。

因此修复保持原有优化目标，也没有改变 query 文本、query 顺序、ID、prompt、
模型编码或指标语义。安装目录中的 MTEB 和 `datasets` 文件均未修改。

## 回归测试

回归测试位于
[`tests/test_mteb_patches.py`](../../tests/test_mteb_patches.py)，使用三行物理表并
执行 `select([0, 2])`，显式构造：

```text
逻辑行数 = 2
物理行数 = 3
_indices 存在
```

测试要求修复实现与 MTEB 原始实现具有相同的列名和完整数据内容。

已执行的验证：

```text
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q tests/test_mteb_patches.py
12 passed, 4 warnings
```

TwitterHjerne 真实离线缓存 smoke test：

```text
input logical rows:  77
input physical rows: 78
output rows:         77
query matches text: true
```

此外：

- `python -m py_compile vibe_eval/mteb_patches.py tests/test_mteb_patches.py` 通过；
- `git diff --check` 通过。

## 重跑

修复后可直接重跑原命令。`vibe_eval.run_mteb` 默认使用 MTEB 的
`only-missing` 结果缓存策略，已有结果的任务会被跳过；失败的
`TwitterHjerneRetrieval` 会重新执行，不需要删除此前已完成的结果。

仅调整 `batch_size`、`max_length` 或 offline 环境变量不能修复这个异常，因为
失败发生在 query DataLoader 的列拼接阶段，早于 query 模型编码。
