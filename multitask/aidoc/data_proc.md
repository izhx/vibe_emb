# 检索训练数据处理记录

## 目标格式

本轮数据统一转换为 FlagEmbedding retrieval finetune JSONL 格式，每行一个样本：

```json
{"query": "...", "pos": ["..."], "neg": ["..."], "prompt": "...", "type": "normal"}
```

其中 `query` 是检索请求，`pos` 是正例文档列表，`neg` 是负例文档列表，`prompt` 是任务指令，`type` 沿用 FlagEmbedding 示例数据的训练类型字段。`prompt` 字段只在原始数据提供任务指令时写入；没有原始 prompt 的数据不写 `prompt` 字段，让训练代码回退到默认指令。CodeSearchNet 只有 1 个负例，`type` 设为 `only_1neg`。

Python 环境：`/data8/zhangxin/.conda/envs/pt28`。

## 数据源

下载清单见 `data/dl.sh`，本轮涉及的数据源：

| 数据源 | 本地原始路径 | 本轮输出 |
| --- | --- | --- |
| `mangopy/ToolRet-Training-20w` | `data/raw/ToolRet-Training-20w` | `data/train/toolret-train/toolret.jsonl` |
| `samaya-ai/msmarco-w-instructions` | `data/raw/msmarco-w-instructions` | `data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl` |
| `sentence-transformers/codesearchnet` | `data/raw/codesearchnet` | `data/train/codesearchnet/codesearchnet.jsonl` |
| `mteb/apps` | `data/raw/apps` | `data/train/apps/apps.jsonl` |
| `CoIR-Retrieval/cosqa` | `data/raw/cosqa` | `data/train/cosqa/cosqa.jsonl` |
| `hanhainebula/reason-embed-data` | `data/raw/reason-embed-data` | 已下载，当前没有生成本轮 FlagEmbedding JSONL 产物 |
| `lightonai/embeddings-fine-tuning` | `data/raw/embeddings-fine-tuning` | `data/train/embeddings-fine-tuning/*.jsonl` |

转换脚本保存在各输出目录下：

- `data/train/toolret-train/convert_toolret_to_flagembedding.py`
- `data/train/msmarco-w-instructions/convert_msmarco_w_instructions_to_flagembedding.py`
- `data/train/codesearchnet/convert_codesearchnet_to_flagembedding.py`
- `data/train/apps/convert_apps_to_flagembedding.py`
- `data/train/cosqa/convert_cosqa_to_flagembedding.py`
- `data/train/embeddings-fine-tuning/convert_embeddings_fine_tuning_to_flagembedding.py`

## 处理要点

### ToolRet-Training-20w

- 读取 parquet 字段：`query`、`prompt`、`positive`、`negative`。
- 保留条件：`query` 非空，且至少有 1 个正例和 1 个负例。
- 正例：直接使用原始 `positive` 列表。
- 负例：直接使用原始 `negative` 列表，不做额外采样；输出中每条样本固定 15 个负例。
- prompt：使用原始 `prompt` 字段，缺失时置为空字符串。

### msmarco-w-instructions

- 只保留 instruction 样本：`has_instruction == true` 且 `query_id` 包含 `-instruct`。
- query：优先使用 `only_query`，否则回退到 `query`。
- prompt：使用 `only_instruction`。
- 正例：从 `positive_passages` 中取 passage text，若 title 有效则拼接 `title + text`。
- 负例：合并 `negative_passages` 和 `new_negatives`，去重后输出；不输出任何分数。

### CodeSearchNet

- 读取 `comment` 和 `code`。
- query：使用自然语言 `comment`。
- 正例：当前行 `code`。
- 负例：使用相邻下一条样本的 `code`；最后一条用首条 `code` 回绕。若相邻代码完全相同则跳过该样本。
- prompt：原始数据没有 prompt 字段，输出不写 `prompt` 字段，让训练代码使用默认指令。
- 输出 `type` 为 `only_1neg`，每条样本 1 个正例、1 个负例。

### APPS

- 输入结构：`queries/*.parquet`、`corpus/*.parquet`、`data/{split}-*.parquet`。
- 当前转换 `train` split，query/corpus 都只使用 `partition == "train"` 的行。
- query：使用 `queries.text`，即编程题题面。
- 正例：使用 qrels 中 `corpus-id` 对应的 `corpus.text`，即 Python 解答代码。
- 负例：从同一 `train` partition 的其他代码中，按 query 稳定随机采样 15 个。
- 随机种子：`42`；采样 key 包含 `seed`、`query_id`，保证可复现。
- prompt：原始数据没有 prompt 字段，输出不写 `prompt` 字段，让训练代码使用默认指令。
- 输出 `type` 为 `normal`。

### CosQA

- 输入结构：`queries/*.parquet`、`corpus/*.parquet`、`data/{split}-*.parquet`。
- 当前转换 `train` split，query/corpus 都只使用 `partition == "train"` 的行。
- `train` qrels 中每个 query 只有一条记录，其中 `score=1` 有 9,020 条，`score=0` 有 10,584 条；转换时只把 `score >= 1` 的 pair 作为正例。
- query：使用 `queries.text`，即自然语言代码检索请求。
- 正例：使用 qrels 中 `corpus-id` 对应的 `corpus.text`，即 Python 代码。
- 负例：从同一 `train` partition 的其他代码中，按 query 稳定随机采样 15 个。
- 随机种子：`42`；采样 key 包含 `seed`、`query_id`，保证可复现。
- prompt：原始数据没有 prompt 字段，输出不写 `prompt` 字段，让训练代码使用默认指令。
- 输出 `type` 为 `normal`。

### embeddings-fine-tuning

- 数据按子集分别输出：`fever`、`fiqa`、`hotpotqa`、`msmarco`、`nq`、`squadv2`、`trivia`。
- 输入结构：`queries/{split}-*.parquet`、`documents/{split}-*.parquet`、`scores/{split}-*.parquet`。
- `scores.document_ids` 视为按相关性排序的候选文档列表。
- 正例：使用 `document_ids[0]` 对应的文档。
- 负例：从排序后的负例 rank 10 到 rank 100 中，按 query 稳定随机采样 10 个文档。
- 随机种子：`42`；采样 key 包含 `seed`、`split`、`query_id`，保证可复现。
- 只输出文档文本，不输出 score。
- 若 query、正例文档或采样出的任一负例文档缺失，则跳过该样本。
- prompt：原始数据没有 prompt 字段，输出不写 `prompt` 字段，让训练代码使用默认指令。

## 数据统计

统计基于已生成的 `data/train/**/*.jsonl`。所有输出样本均包含非空 `query`、`pos`、`neg`。

| 输出文件 | 样本数 | type | 正例数 | 负例数 | prompt 数 |
| --- | ---: | --- | --- | --- | ---: |
| `data/train/toolret-train/toolret.jsonl` | 208,826 | `normal` | min 1 / avg 2.11 / max 10 | min 15 / avg 15.00 / max 15 | 208,826 |
| `data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl` | 489,243 | `normal` | min 1 / avg 1.00 / max 1 | min 25 / avg 32.15 / max 33 | 489,243 |
| `data/train/codesearchnet/codesearchnet.jsonl` | 1,375,067 | `only_1neg` | min 1 / avg 1.00 / max 1 | min 1 / avg 1.00 / max 1 | 0 |
| `data/train/apps/apps.jsonl` | 5,000 | `normal` | min 1 / avg 1.00 / max 1 | min 15 / avg 15.00 / max 15 | 0 |
| `data/train/cosqa/cosqa.jsonl` | 9,020 | `normal` | min 1 / avg 1.00 / max 1 | min 15 / avg 15.00 / max 15 | 0 |
| `data/train/embeddings-fine-tuning/fever.jsonl` | 140,082 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |
| `data/train/embeddings-fine-tuning/fiqa.jsonl` | 14,112 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |
| `data/train/embeddings-fine-tuning/hotpotqa.jsonl` | 169,996 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |
| `data/train/embeddings-fine-tuning/msmarco.jsonl` | 532,751 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |
| `data/train/embeddings-fine-tuning/nq.jsonl` | 152,132 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |
| `data/train/embeddings-fine-tuning/squadv2.jsonl` | 130,255 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |
| `data/train/embeddings-fine-tuning/trivia.jsonl` | 741,436 | `normal` | min 1 / avg 1.00 / max 1 | min 10 / avg 10.00 / max 10 | 0 |

合计已生成训练样本：3,967,920 条。

## 复现命令

脚本默认路径仍保留当时开发时的默认值；当前原始数据位于 `data/raw/...`，复现时建议显式传入输入路径：

```bash
/data8/zhangxin/.conda/envs/pt28/bin/python data/train/toolret-train/convert_toolret_to_flagembedding.py \
  --input-dir data/raw/ToolRet-Training-20w/ToolRet-Training-20w \
  --output data/train/toolret-train/toolret.jsonl \
  --overwrite

/data8/zhangxin/.conda/envs/pt28/bin/python data/train/msmarco-w-instructions/convert_msmarco_w_instructions_to_flagembedding.py \
  --input-dir data/raw/msmarco-w-instructions/data \
  --output data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl \
  --overwrite

/data8/zhangxin/.conda/envs/pt28/bin/python data/train/codesearchnet/convert_codesearchnet_to_flagembedding.py \
  --input-dir data/raw/codesearchnet/pair \
  --output data/train/codesearchnet/codesearchnet.jsonl \
  --overwrite

/data8/zhangxin/.conda/envs/pt28/bin/python data/train/apps/convert_apps_to_flagembedding.py \
  --input-dir data/raw/apps \
  --output data/train/apps/apps.jsonl \
  --num-negatives 15 \
  --seed 42 \
  --overwrite

/data8/zhangxin/.conda/envs/pt28/bin/python data/train/cosqa/convert_cosqa_to_flagembedding.py \
  --input-dir data/raw/cosqa \
  --output data/train/cosqa/cosqa.jsonl \
  --min-positive-score 1 \
  --num-negatives 15 \
  --seed 42 \
  --overwrite

/data8/zhangxin/.conda/envs/pt28/bin/python data/train/embeddings-fine-tuning/convert_embeddings_fine_tuning_to_flagembedding.py \
  --input-dir data/raw/embeddings-fine-tuning \
  --output-dir data/train/embeddings-fine-tuning \
  --num-negatives 10 \
  --neg-rank-start 10 \
  --neg-rank-end 100 \
  --seed 42 \
  --overwrite
```
