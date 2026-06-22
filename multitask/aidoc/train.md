# Qwen2.5-0.5B 多任务检索训练记录

## 实验目标

使用 `data/train` 下已经转换好的 FlagEmbedding retrieval JSONL 数据，基于本地模型 `data/raw/Qwen2.5-0.5B` 训练 decoder-only embedder。实验脚本和辅助代码放在 `multitask`，模型输出放在 `results`。

Python 环境：

```bash
/data8/zhangxin/.conda/envs/pt28
```

## 输入数据

训练脚本显式传入每个 JSONL 文件，而不是传目录。这样在 `same_dataset_within_batch=True` 时，每个 JSONL 会被 FlagEmbedding 当作一个独立 dataset。

| 数据集 | 路径 | 样本数 | 备注 |
| --- | --- | ---: | --- |
| ToolRet | `data/train/toolret-train/toolret.jsonl` | 208,826 | 工具检索数据 |
| msmarco-w-instructions | `data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl` | 489,243 | 只保留 instruction query |
| CodeSearchNet | `data/train/codesearchnet/codesearchnet.jsonl` | 1,375,067 | `type=only_1neg` |
| FEVER | `data/train/embeddings-fine-tuning/fever.jsonl` | 140,082 | reason embedding 数据 |
| FiQA | `data/train/embeddings-fine-tuning/fiqa.jsonl` | 14,112 | reason embedding 数据 |
| HotpotQA | `data/train/embeddings-fine-tuning/hotpotqa.jsonl` | 169,996 | reason embedding 数据 |
| MSMARCO | `data/train/embeddings-fine-tuning/msmarco.jsonl` | 532,751 | reason embedding 数据 |
| NQ | `data/train/embeddings-fine-tuning/nq.jsonl` | 152,132 | reason embedding 数据 |
| SQuADv2 | `data/train/embeddings-fine-tuning/squadv2.jsonl` | 130,255 | reason embedding 数据 |
| TriviaQA | `data/train/embeddings-fine-tuning/trivia.jsonl` | 741,436 | reason embedding 数据 |

总计 3,953,900 条样本。

## 脚本

Smoke run：

```bash
bash multitask/train_smoke.sh
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 MAX_EXAMPLES_PER_DATASET=100000 bash multitask/train_full.sh
```

关键默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `same_dataset_within_batch` | `True` | batch 内样本来自同一个 JSONL |
| `train_group_size` | `8` | 每条样本 1 正 7 负；`only_1neg` 数据例外 |
| `per_device_train_batch_size` | full `64`，smoke `2` | 在 same-dataset 模式下作为每张卡每步的 query 数 |
| `gradient_accumulation_steps` | `1` | 对比学习不开梯度累积，保证负例池来自同一个 micro-step |
| `sub_batch_size` | full `32` | 只切分模型 encode 前向，不改变对比学习 batch |
| `query_max_len` | full `320`，smoke `256` | ToolRet 和 msmarco-w-instructions 有较长 query |
| `passage_max_len` | full `512`，smoke `256` | 减少长文档系统性截断 |
| `max_example_num_per_dataset` | smoke `32`，full `100000` | 每个 JSONL 的样本上限 |
| LoRA | rank `32`，alpha `64` | 训练 Qwen attention 和 MLP projection |

DeepSpeed 配置使用 `multitask/ds_stage1.json`。FlagEmbedding 示例里的 `ds_stage1.json` 在当前 `deepspeed==0.18.6` 与 `pydantic==2.x` 下会因为 `bf16` 配置包含 `loss_scale` 等额外字段而报错；本实验配置保留 ZeRO stage 1，并把 `bf16` 简化为 `{"enabled": "auto"}`。

## 采样策略分析

FlagEmbedding 的 `AbsEmbedderSameDatasetTrainDataset` 在 `same_dataset_within_batch=True` 时，会分别加载每个 JSONL，记录每个 dataset 的样本索引，并按 dataset 内部样本构造 batch。每个 batch 只来自一个 dataset。

每个 epoch 开始时，loader 会 shuffle dataset 顺序、shuffle 每个 dataset 内部样本，再把所有 dataset 的 batch 列表合并后整体 shuffle。因此，跨 dataset 的 batch 顺序是混合的，但每个 dataset 在一个 epoch 中贡献的 batch 数近似正比于该 dataset 的样本量。

这个默认策略会带来配比风险：如果直接全量训练，CodeSearchNet、TriviaQA、MSMARCO 等大数据集贡献的 batch 会明显更多，小数据集如 FiQA 权重会很低。当前脚本用统一的 `max_example_num_per_dataset` 对每个 JSONL 设置上限，作为简单的数据配平策略。这个策略会截断大数据集、保留小数据集全部或大部分样本；如果后续需要精确配比，需要增加预采样脚本或自定义 sampler。

负例采样方面，每条训练样本在 dataloader 中随机选 1 个 positive，并从 `neg` 中随机选 `train_group_size - 1` 个 negative；如果负例不足，会重复采样。`codesearchnet` 标记了 `type=only_1neg`，FlagEmbedding 会把该 batch 的 `train_group_size` 改成 2，也就是 1 正 1 负。


当前 FlagEmbedding embedder 这条路径主要有这些：

  1. same_dataset_within_batch=False
      - 默认普通模式。
      - 所有 JSONL 先 concatenate_datasets 合成一个大 dataset。
      - Trainer 随机按样本采样，batch 里可以混多个数据集。
      - 配比严格近似按样本量来，大数据集会主导。

  2. same_dataset_within_batch=True
      - 我们现在用的模式。
      - 每个 JSONL 单独作为一个 dataset，每个 batch 只来自一个 dataset。
      - 但 epoch 内每个 dataset 贡献的 batch 数还是约等于 len(dataset) / batch_size，所以本质仍接近按数据量配比。
      - 它解决的是“batch 内不混数据集”，不是“数据集均衡采样”。

  3. max_example_num_per_dataset
      - 每个 dataset 截断到同一个上限。
      - 这是官方代码里最直接可用的配平手段。
      - 缺点是首次仍会完整解析 JSONL，然后再 select；而且小数据集不会被上采样。

  4. small_threshold / drop_threshold
      - 只在传目录时有意义。
      - 小于 small_threshold 的小 JSONL 会被合并成一个 dataset；合并后如果还小于 drop_threshold 就丢弃。
      - 这是为了处理很多小文件，不是通用加权采样。

  5. 数据内 batch_size 字段
      - loader 会读每个 dataset 的 batch_size 列第一条值，覆盖该 dataset 的 batch size。
      - 这会改变该 dataset 的 step 数：batch size 越小，该 dataset 每 epoch step 越多。
      - 可以间接调权，但不太干净，需要改 JSONL 或预处理数据。

  结论：FlagEmbedding 没有看到类似 dataset_weights、sampling_probs、temperature sampling 这种显式多数据集权重采样参数。对我们当前实
  验，合理选择是：

  - 先用 same_dataset_within_batch=True 保证 batch 内同源。
  - 用 max_example_num_per_dataset 限制大数据集，比如 50k/100k。
  - 如果要更精确配比，最好在 multitask 里做一个预采样数据目录，按目标比例为每个 JSONL 生成 capped/upsampled 版本，再喂给
    FlagEmbedding。

## 长度分析和 batch 试跑

长度统计脚本：

```bash
/data8/zhangxin/.conda/envs/pt28/bin/python multitask/analyze_data_lengths.py --sample-size 5000
```

统计使用 Qwen2.5 tokenizer，对 query 使用训练时的 `<instruct>{prompt}\n<query>{query}` 格式；doc 统计每条样本的第一个 positive 和最多 7 个 negative。

| 数据集 | 样本数 | query avg | query p95 | query p99 | query >256 | doc avg | doc p95 | doc p99 | doc >256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToolRet | 208,826 | 122.9 | 271 | 378 | 5.7% | 219.3 | 582 | 861 | 27.3% |
| msmarco-w-instructions | 489,243 | 130.0 | 261 | 307 | 5.6% | 86.7 | 150 | 188 | 0.1% |
| CodeSearchNet | 1,375,067 | 38.2 | 66 | 76 | 0.0% | 148.0 | 448 | 1005 | 12.1% |
| FEVER | 140,082 | 31.1 | 39 | 45 | 0.0% | 247.6 | 648 | 912 | 36.3% |
| FiQA | 14,112 | 34.5 | 44 | 50 | 0.0% | 215.8 | 595 | 998 | 27.5% |
| HotpotQA | 169,996 | 45.6 | 71 | 95 | 0.0% | 84.3 | 186 | 251 | 0.9% |
| MSMARCO | 532,751 | 28.0 | 33 | 36 | 0.0% | 81.4 | 148 | 187 | 0.1% |
| NQ | 152,132 | 29.4 | 34 | 37 | 0.0% | 134.5 | 292 | 1144 | 6.7% |
| SQuADv2 | 130,255 | 34.4 | 42 | 47 | 0.0% | 172.4 | 308 | 411 | 11.4% |
| TriviaQA | 741,436 | 38.6 | 58 | 71 | 0.0% | 146.5 | 180 | 219 | 0.4% |

结论：

- `query_max_len=256` 基本够用，但 ToolRet 和 msmarco-w-instructions 有约 5%-6% query 超过 256；建议正式训练用 `query_max_len=320`。
- `passage_max_len=256` 对 ToolRet、FEVER、FiQA、CodeSearchNet、SQuADv2 截断偏明显；建议正式训练用 `passage_max_len=512`。
- 512 仍不能覆盖所有长尾文档，尤其 NQ、CodeSearchNet、ToolRet 的超长尾；但 512 能显著减少 256 带来的系统性截断，同时单卡显存压力仍很低。

`max_example_num_per_dataset=100000` 后，FiQA 为 14,112 条，其余 9 个数据集各 100,000 条，总计 914,112 条。按当前 full 默认 `per_device_train_batch_size=64`、4 卡、`gradient_accumulation_steps=1` 计算，常规数据集每步全局 query batch 为 256，约 3,571 optimizer steps；FiQA 和 `only_1neg` 数据集会按自身样本数和有效 group size 产生不同 step 数。

多卡问题已重新定位：

| 配置 | 结果 |
| --- | --- |
| 原生 DDP，LoRA，默认 reentrant `--gradient_checkpointing` | 2 卡会报 `Expected to mark a variable ready only once` |
| 原生 DDP，LoRA，`gradient_checkpointing_kwargs={"use_reentrant": false}` | 2 卡和 4 卡可跑 |
| DeepSpeed ZeRO stage1，LoRA，默认 reentrant `--gradient_checkpointing` | 2 卡和 4 卡可跑 |
| DeepSpeed ZeRO stage1，4 卡，query 320，passage 512，cross-device negatives | 5 step 成功 |

结论：正式训练使用 DeepSpeed ZeRO stage1。它能绕开原生 DDP + LoRA + reentrant checkpointing 的 ready-twice 错误，同时保留 gradient checkpointing。实测 2,3,4,5 四卡组合稳定，后续默认使用 `CUDA_VISIBLE_DEVICES=2,3,4,5`。

单卡 profile 结果：

| 配置 | steps | train runtime | steps/s | query/s | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| bs=4，query 320，passage 512 | 20 | 33.36s | 0.600 | 2.40 | 稳定，显存约 5.8GB |
| bs=8，query 320，passage 512 | 10 | 22.90s | 0.437 | 3.50 | 稳定，吞吐高于 bs=4 |
| bs=16，query 320，passage 512 | 5 | 22.25s | 0.225 | 3.60 | 稳定，但单 step 明显变慢，吞吐只略高于 bs=8 |

早期单卡 profile 用于确认单机链路和长度设置，不再作为正式训练建议。当前正式训练优先使用 4 卡 ZeRO stage1：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
NUM_GPUS=4 \
MAX_EXAMPLES_PER_DATASET=100000 \
PER_DEVICE_TRAIN_BATCH_SIZE=64 \
GRADIENT_ACCUMULATION_STEPS=1 \
SUB_BATCH_SIZE=32 \
QUERY_MAX_LEN=320 \
PASSAGE_MAX_LEN=512 \
bash multitask/train_full.sh
```

## Batch size 与 sub-batch size

在 `same_dataset_within_batch=True` 下，FlagEmbedding 的 Runner 会先用用户传入的 `per_device_train_batch_size` 构造 same-dataset batch，然后把 Trainer 外层的 `per_device_train_batch_size` 改成 1。因此这里的 `PER_DEVICE_TRAIN_BATCH_SIZE` 表示每张卡每步从同一个 JSONL 取多少条 query，而不是普通 Trainer dataloader 的外层 batch。

当前 full 默认：

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `NUM_GPUS` | 4 | 建议配合 `CUDA_VISIBLE_DEVICES=2,3,4,5` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 64 | 每张卡每步 64 条 query |
| `GRADIENT_ACCUMULATION_STEPS` | 1 | 不做梯度累积 |
| `TRAIN_GROUP_SIZE` | 8 | 每条 query 1 正 7 负 |
| `SUB_BATCH_SIZE` | 32 | encode 阶段每次前向的文本数 |

常规数据集上，每张卡每步有 64 条 query 和 `64 * 8 = 512` 条 passage。4 卡且 `negatives_cross_device=True` 时，一个 micro-step 的全局负例池为 256 条 query 对 2048 条 passage。`gradient_accumulation_steps=1` 时，optimizer update batch 也是 256 条 query。

`sub_batch_size` 不改变对比学习 batch，也不改变 in-batch negative pool。它只在模型 encode 阶段把已经构造好的 query/passages 切成小块前向，再把 embedding concat 回完整 batch 后计算同一个 loss。例如当前每卡 64 query、512 passage：

| `sub_batch_size` | 每卡 query forward 次数 | 每卡 passage forward 次数 | 实测 |
| ---: | ---: | ---: | --- |
| 8 | 8 | 64 | 可训练，但约 48s/step |
| 32 | 2 | 16 | 可训练，约 14-15s/step |

因此正式训练不建议用 `SUB_BATCH_SIZE=8`，除非更大 batch 或更长序列导致 OOM。当前建议从 `SUB_BATCH_SIZE=32` 开始手动调试；如果显存仍充足，可以继续尝试 64，或设置 `SUB_BATCH_SIZE=0` 关闭 sub-batch 切分。

## 当前 smoke run

GPU 环境：

- 8 x NVIDIA RTX A6000，每张显存约 48GB。
- `/data8/zhangxin/.conda/envs/pt28` 中 `torch==2.8.0+cu128`，`torch.cuda.is_available()` 为 `True`。

第一次执行命令：

```bash
MAX_STEPS=2 MAX_EXAMPLES_PER_DATASET=8 SAVE_STEPS=1000 bash multitask/train_smoke.sh
```

第一次运行结果：模型和 10 个 JSONL 都能加载，GPU 可用；失败点在 DeepSpeed 示例配置与当前版本不兼容，报错为 `DeepSpeedBF16Config` 不接受 `loss_scale`、`initial_scale_power`、`loss_scale_window`、`hysteresis`、`min_loss_scale`。已改为使用 `multitask/ds_stage1.json`。

另外，`max_example_num_per_dataset` 不会避免首次解析原始 JSONL。FlagEmbedding 当前实现是先用 `datasets.load_dataset('json', data_files=...)` 解析完整文件，再 `select` 小样本。因此第一次 smoke run 仍会为所有大 JSONL 构建 cache；本次 cache 在 `multitask/cache` 下约 22GB，后续重复运行会复用 cache，启动更快。

第二次执行同一命令，使用修正后的 `multitask/ds_stage1.json`，smoke run 成功：

| 项 | 结果 |
| --- | --- |
| 训练步数 | 2 steps |
| 样本限制 | 每个 JSONL 最多 8 条 |
| 训练 loss | 4.28125 |
| step 1 loss | 4.1875 |
| step 2 loss | 4.375 |
| 训练耗时 | 11.37 秒，不含前置模型和数据加载 |
| 输出目录 | `results/qwen2.5-0.5b-embedder-smoke` |
| checkpoint | `results/qwen2.5-0.5b-embedder-smoke/checkpoint-2` |

输出目录中包含 LoRA adapter、tokenizer、训练参数和 `embedding/emb.pth`。这说明当前数据格式、同数据集 batch 采样、Qwen2.5-0.5B 加载、LoRA 训练和保存链路已经跑通。
