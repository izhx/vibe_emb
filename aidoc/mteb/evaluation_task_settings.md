# MTEB 评测任务设置与本地运行约定

本文记录本仓库运行 MTEB 文本评测时需要统一检查的任务设置，包括输入长度、
本地缓存、严格离线兼容、dataset revision 和任务专属优化。具体任务的实现细节仍
放在各自文档中，本文只提供运行前的统一入口。

当前代码和 runtime patch 以 **MTEB 2.14.9** 为验证边界。升级 MTEB 后，必须
重新检查任务定义、数据 revision、私有 loader API、结果聚合方式和现有 patch，
不能直接假设行为兼容。

## 输入长度约定

### 常规评测基线

本地常规评测约定使用 8192 tokens。这里的“8192”是运行约定，不是当前 CLI 的
代码默认值。`vibe_eval.run_mteb` 当前的 `--max_length` 默认是 2048：

```text
--max_length 2048
```

query 和 corpus 统一使用 `max_length`，常规评测应显式设置：

```bash
python -m vibe_eval.run_mteb \
  ... \
  --max_length 8192
```

比较不同模型或不同运行的结果时，必须记录这三个值。长度设置不同的结果不能直接
解释为模型性能差异。

### `LEMBPasskeyRetrieval` 长上下文例外

`LEMBPasskeyRetrieval` 包含从 `test_256` 到 `test_32768` 的多个长度子集。8192
长度不能覆盖任务的完整设置。模型本身支持时，至少应让 corpus 覆盖 32768；对
F2LLM-v2-0.6B，MTEB 注册的模型上限是 40960，因此建议使用：

```bash
python -m vibe_eval.run_mteb \
  --tasks LEMBPasskeyRetrieval \
  ... \
  --max_length 40960
```

如果模型、tokenizer、显存或 attention 实现不支持 32768 以上长度，就无法完整
覆盖该任务。此时仍可运行较短长度，但报告中必须注明截断上限，并按各个 split
查看结果，不能只用聚合后的 task main score 与长上下文官方结果比较。

## 本地缓存与严格离线评测

当前数据缓存基于 MTEB 2.14.9，默认缓存目录为：

```text
/mnt/share/emb/mteb_cache
```

完成缓存后，正式评测使用严格离线环境：

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m vibe_eval.run_mteb ...
```

“Hub snapshot 已存在”和“`datasets` 能在离线模式解析到正确的 processed config”
是两件不同的事。原始 Parquet 文件已经下载，不代表 `load_dataset()` 能在严格离线
时找到 MTEB 请求的 config、split 或 data directory。

需要补数据时，可以临时取消离线模式，但必须显式移除代理变量。本仓库环境中建议
使用如下边界：

```bash
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY='*' no_proxy='*' HF_HUB_DISABLE_XET=1 \
  /mnt/share/envs/embt/bin/python -m vibe_eval.run_mteb \
  --no_model \
  --tasks <task-name>
```

补齐后应重新启用三个 offline 环境变量做 smoke test，确认评测没有隐式访问网络。

## 当前离线兼容 patch

仓库通过显式安装的 runtime patch 适配 MTEB 2.14.9，不直接修改
`site-packages/mteb`。主要处理如下：

| 场景 | 当前处理 | 实现位置 |
| --- | --- | --- |
| `CodeEditSearchRetrieval` 严格离线无法解析按语言组织的 raw Parquet | 从固定 revision 的 Hub raw-file cache 直接读取 Parquet，绕过离线 `datasets` 对合成 config 的错误解析 | `vibe_eval/tasks/code_edit_search_retrieval.py` |
| `BelebeleRetrieval` 上游未指定多配置数据集的 config | 按选中的语言对计算所需语言，为 `mteb/belebele` 显式指定并去重加载 122 个语言 config，再保持上游语义构造 376 个评测语言对 | `vibe_eval/tasks/belebele_retrieval.py` |
| Retrieval 离线 config discovery 选择不存在的 `default` qrels | 在错误信息证明缓存中存在 `qrels` 时，受限回退到该 config | `vibe_eval/mteb_patches.py` |
| Reranking 离线加载时漏掉 `top_ranked` | 对当前选中的 reranking dataset/revision/subset 直接恢复候选集合，并拒绝空候选 | `vibe_eval/mteb_patches.py` |
| 大规模或 indexed query 的 Python list 物化开销及逻辑行/物理行错位 | 对无 instruction 的 query 使用 Arrow-native 列处理；有 instruction 时保留上游路径 | `vibe_eval/mteb_patches.py` |
| `MindSmallReranking` 数据和计算规模过大 | 使用本地 compact task，或显式切换 legacy patch | `vibe_eval/tasks/mind_small_reranking*.py` |

`CodeEditSearchRetrieval`、`BelebeleRetrieval` 的替换和 MindSmall task 选择发生在
`vibe_eval.run_mteb.build_tasks()`；通用 query、qrels 和 reranking patch 在正式
构造模型前安装。所有 patch 都应保持版本校验和受限触发条件，不能把某个 task 的
缓存假设无条件应用到其他任务。

### `BelebeleRetrieval` 修复与验证状态

MTEB 2.14.9 的上游实现直接调用 `load_dataset()` 加载 `mteb/belebele`，没有为具有
122 个语言 config 的数据集指定 `name`。本仓库的子类根据任务选中的语言对计算唯一
语言集合，对每种语言执行显式 config 加载，并复用上游的 query、corpus 和 qrels
构造语义；任务替换同时保留 benchmark 设置的 `seed`、`hf_subsets` 和 eval split。

该修复已经完成严格离线数据加载和正式评测验证。F2LLM-v2-0.6B 的结果文件为：

```text
results/mteb_eval/models__F2LLM-v2-0.6B/no_revision_available/BelebeleRetrieval.json
```

结果使用 MTEB 2.14.9 和 dataset revision
`979a211276faa22f671e69d096634193567cfd05`，包含全部 376 个语言对；日志
`results/mteb_eval/f20b6-mm.log` 在 2026-07-19 03:35:33 明确记录
`Finished evaluation for BelebeleRetrieval`。因此该任务不再属于当前未解决问题。

相关详细文档：

- [MindSmallReranking 评测路径与优化设计](mind_small_reranking_optimization.md)
- [TwitterHjerne indexed query 修复](twitter_hjerne_indexed_query_fix.md)
- [MTEB instruction 解析机制](instruction_resolution.md)

## 尚不能由通用 patch 自动解决的离线问题

部分任务会在 evaluator 的后处理阶段再次调用 `load_dataset()`，需要主数据以外的
隐藏 config。例如 `News21InstructionRetrieval` 和
`Robust04InstructionRetrieval` 还需要 `qrel_diff`。这类 config 应先按上面的
无代理联网方式单独补齐，再回到离线模式运行。

新增离线 patch 前应先区分：

1. 数据文件确实缺失；
2. raw Hub snapshot 已存在，但 processed config 未生成；
3. MTEB 离线 config discovery 选择错误；
4. task 实现缺少 config、split 或 revision；
5. evaluator 在主加载流程之外还有隐藏的数据依赖。

只有能明确证明预期数据语义且触发条件足够窄时，才应增加 runtime patch。

## Dataset revision 与结果可比性

MTEB task 的 dataset revision 会更新。即使任务名、split 和 `hf_subset` 相同，
两份结果也可能来自不同 revision。对比本地与官方结果时应至少保存并检查：

```text
task_name
split
hf_subset
dataset_revision
mteb_version
max_length
```

结果连接键使用 `task_name + split + hf_subset`，但这个键只负责找到对应分数，
不能证明两边数据完全相同。revision 不一致时，应把差异标记为“可能包含数据版本
变化”，并进一步检查数据规模、字段、split 和 task metadata；不要直接归因于模型。

缓存预热和正式评测应尽量使用 task metadata 固定的 revision。升级 MTEB 或重新
下载数据后，应重新审计缓存中的 revision，而不是把旧结果和新数据混在同一次汇总
中。

## 任务专属优化原则

任务优化必须保持以下评测语义不变：

- 实际送入 tokenizer 的文本、instruction 和截断长度；
- query、document、qrels 和候选集合；
- task 定义的 split、subset 和 dataset revision；
- pooling、归一化、相似度函数和最终指标；
- 分布式场景下各 rank 的 batch 数、collective 顺序和结果聚合。

当前最重要的任务专属优化是 `MindSmallReranking`。它包含候选恢复、query 去重、
compact 数据格式以及保留上游指标语义等约束，统一记录在
[MindSmallReranking 评测路径与优化设计](mind_small_reranking_optimization.md)，
不要在通用 MTEB loader 中复制 MindSmall 专属逻辑。

## 运行前检查清单

1. 确认使用 `/mnt/share/envs/embt`，并确认 MTEB 版本为 2.14.9。
2. 明确任务列表、split、subset 和 dataset revision。
3. 常规任务显式设置 `max_length` 为 8192；长上下文任务单独调整。
4. 对 `LEMBPasskeyRetrieval` 检查模型是否支持至少 32768，并按 split 审计分数。
5. 确认缓存目录和 offline 环境变量，补数据时显式清除全部代理变量。
6. 从日志确认预期 patch 已安装，并确认 reranking 实际加载了 `top_ranked`。
7. 结果比较时同时检查长度、MTEB 版本和 dataset revision。
8. 对失败任务记录确切 exception、缺失 config/revision 和是否能在无代理联网模式
   下补齐，不要只记录“离线失败”。
