# F2LLM Stage-2 Retrieval Loss Spec

状态：Implemented and validated
日期：2026-07-20
目标配置：`configs/train_f2llm_stage2_full.yaml`

## 1. 目标

为 `vibe_emb` 增加一个显式启用的 F2LLM retrieval loss 模式。该模式严格计算：

```text
retrieval_loss = hard_loss + positive_only_in_batch_loss
```

其中：

- `hard_loss` 只比较每个 query 自己的 positive 和显式 hard negatives；
- `positive_only_in_batch_loss` 只比较所有 rank 的 positive，不把其他 query 的 hard
  negatives 放入该 softmax；
- 两项使用独立归一化分母，但在同一个 loss helper 中通过两个 `logsumexp` 计算，并只执行一次
  backward；
- 多卡时只 gather positive embeddings，不 gather query 或 hard-negative embeddings。

本需求不为 classification、clustering 建立新的模型分支。两类数据继续通过已经 resolved 的
`no_in_batch_neg=true` 使用现有 query-local group CE；在不考虑 MRL 且 group size 相同的前提下，
该路径已经等价于 F2LLM 的 full-dimension `hard_loss`。

## 2. 当前基线与差异

### 2.1 当前 `vibe_emb`

当前 passage 布局为：

```text
[q0_pos, q0_neg_1, ..., q1_pos, q1_neg_1, ...]
```

行为由 resolved `no_in_batch_neg` 控制：

- `true`：从完整 score matrix 中选出 query 自己的连续 group，再计算一次 CE；
- `false`：query 与 batch 内所有 positive 和所有 hard negatives 组成一个联合 passage matrix，
  计算一次 CE；多卡时 gather query 和全部 passage embeddings。

因此当前 classification/clustering 的 full-dimension hard loss 结构已经正确；差异只在
`no_in_batch_neg=false` 的 retrieval 路径。

### 2.2 F2LLM retrieval

F2LLM 对 retrieval 分别计算：

```text
hard_loss(q_i, [p_i_pos, p_i_neg_1, ...])
positive_only_in_batch_loss(q_i, [p_0_pos, p_1_pos, ...])
```

然后相加。该目标与把两组候选合并后计算一个共享 softmax 不等价。

## 3. 范围

本需求包括：

1. 增加全局 retrieval loss mode 配置，并保持历史默认行为不变。
2. 保留现有 `no_in_batch_neg=true` 路径，不修改 classification/clustering loss。
3. 新增精确等价于“两次 CE 相加”的 F2LLM retrieval loss helper。
4. 单卡时计算 local hard loss 和 local positive-only in-batch loss。
5. 多卡时只 gather positives，并保持本 rank positive 的梯度。
6. 为 total、hard 和 positive-only in-batch loss 提供可测试的独立输出。
7. 保留当前 teacher-score distillation 语义：distillation 只作用于 query-local group，且只加
   一次。
8. 增加单卡数值、梯度、兼容性和两 rank 分布式测试。
9. 在 F2LLM stage-2 完整训练配置中显式启用新模式；历史配置未显式设置时继续使用 legacy
   loss。

## 4. 非范围

本需求不包括：

1. 实现 MRL、MLL、MEL 或教师模型 embedding MSE distillation。
2. 修改 classification/clustering 的 `no_in_batch_neg` 默认值。
3. 修改 classification/clustering/retrieval 的 `train_group_size`。
4. 修改采样、Indexed Arrow、batch plan、collator passage 排列或 tokenizer 行为。
5. 把 `task_type` 透传到 `EmbeddingModel.forward()`。loss 路由继续只依赖 resolved
   `no_in_batch_neg` 和显式 loss mode。
6. 让一个共享 softmax 近似两个独立 CE。
7. 为了减少一次 CE API 调用而构造 hard-positive 笛卡尔积或 padded `2B x K` logits。
8. 修改现有 checkpoint 文件布局或实现通用 manifest/config fingerprint guard。
9. 重新运行完整训练或 MTEB 评估；运行属于代码验收后的独立步骤。

## 5. 配置接口

在 `training` section、`EmbedTrainingExtras` 中增加：

```yaml
training:
  retrieval_loss_mode: legacy
```

允许值：

| 值 | 行为 |
| --- | --- |
| `legacy` | 保持当前联合 passage-matrix CE，作为框架默认值 |
| `f2llm` | 当 `no_in_batch_neg=false` 时使用 hard + positive-only in-batch loss |

规则：

1. 默认值必须是 `legacy`，未修改的 YAML 和旧 JSON/JSONL 训练数值保持不变。
2. 非法值在模型初始化前失败，错误信息列出允许值。
3. `resolved_config.yaml` 必须写入最终 `retrieval_loss_mode`。
4. `no_in_batch_neg=true` 时始终只计算 hard loss；`retrieval_loss_mode` 不得强制增加
   in-batch loss。
5. `no_in_batch_neg=false, retrieval_loss_mode=legacy` 完全进入现有实现。
6. `no_in_batch_neg=false, retrieval_loss_mode=f2llm` 进入本需求新增路径。
7. 本轮不支持 per-dataset loss-mode override。一个训练进程中的所有
   `no_in_batch_neg=false` batch 使用同一模式。

F2LLM stage-2 完整配置在实现和测试通过后显式设置：

```yaml
training:
  retrieval_loss_mode: f2llm
```

该配置变更代表训练目标发生变化。已有 checkpoint-12000 及此前 checkpoint 均视为
`legacy` loss 产物，不得在切换为 `f2llm` 后以相同 optimizer/scheduler 状态直接续训；新目标
必须使用新的输出目录开始训练。若只加载旧 adapter 权重作为 warm start，应明确视为新实验并
重建 optimizer/scheduler。

## 6. 数学定义

设：

- local query 数为 `B`；
- world size 为 `W`；
- global positive 数为 `N = B * W`；
- group size 为 `G`，第 0 个 passage 是 positive；
- temperature 为 `T`；
- query embedding 为 `q_i`；
- 本地 passage group 为 `p_i,g`；
- gather 后的 global positives 为 `p_j,+`。

hard logits：

```text
H_i,g = dot(q_i, p_i,g) / T              shape: [B, G]
```

positive-only in-batch logits：

```text
P_i,j = dot(q_i, p_j,+) / T              shape: [B, N]
```

targets：

```text
hard_target_i = 0
in_batch_target_i = process_rank * B + i
```

每个 query 的 loss：

```text
hard_loss_i = logsumexp(H_i, :) - H_i,0
in_batch_loss_i = logsumexp(P_i, :) - P_i,in_batch_target_i
total_loss_i = hard_loss_i + in_batch_loss_i
```

最终 reduction：

```text
hard_loss = mean(hard_loss_i)
in_batch_loss = mean(in_batch_loss_i)
loss = hard_loss + in_batch_loss
```

该表达式与以下实现的数值和梯度等价：

```python
F.cross_entropy(hard_logits, zeros) + F.cross_entropy(in_batch_logits, targets)
```

它不是单一 softmax。两个独立 `logsumexp` 是目标定义的一部分，不能通过把候选拼接成一行并
增加 mask 消除。

### 6.1 为什么单一 masked CE 不成立

把 own hard group 和 global positives 合并后，一次 CE 得到：

```text
logsumexp(H_i union P_i) - positive_score
```

目标 loss 则是：

```text
logsumexp(H_i) + logsumexp(P_i) - 2 * positive_score
```

两者通常不相等。理论上可以对 `H_i,g + P_i,j` 构造 `G * N` 个笛卡尔积 logits 后做一次
CE，但其计算和内存复杂度远高于两个独立归一化，禁止在本需求中采用。

## 7. 单卡数据流

`p_reps` 先按现有 passage-layout invariant reshape：

```text
groups = p_reps.reshape(B, G, D)
positive_reps = groups[:, 0, :]
```

计算：

```text
hard_logits = einsum("bd,bgd->bg", q_reps, groups) / T
in_batch_logits = q_reps @ positive_reps.T / T
```

不得先计算 `q_reps @ p_reps.T` 再用 mask 截取两组 logits。后一种写法虽然可以得到正确的
两个 normalization set，但会无意义地计算其他 query 的 hard-negative scores。

loss reduction 使用 FP32 logits 或 FP32 reduction，避免 BF16 `logsumexp` 的精度和溢出风险；
不得改变 embedding 输出 dtype。

## 8. 多卡数据流

### 8.1 Collective

只有 positive embeddings 进入 collective：

```text
positive_global = gather_with_local_grad(positive_local)
```

约束：

1. 不 gather `q_reps`。
2. 不 gather 完整 `p_reps` 或 hard negatives。
3. gather 列表中本 rank slot 必须替换回原始 `positive_local` tensor，使本地 positive 保留梯度。
4. 其他 rank positives 作为常量 negatives，不跨 rank 建立 autograd graph。
5. 每个 rank 只计算自己的 `B` 行 query loss。
6. local batch size 必须在所有 rank 相同；现有 deterministic batch plan 和 drop-last 规则负责
   保证该条件。
7. loss 在每个 rank 内按 local query mean；随后由 DDP 正常平均参数梯度，不额外乘或除
   world size。

### 8.2 Collective 顺序

当前 batch plan 保证所有 rank 在同一步消费相同 dataset/unit，并解析出相同
`no_in_batch_neg`。因此：

- retrieval + `f2llm`：所有 rank 同时执行一次 positive gather；
- classification/clustering：所有 rank 同时跳过 gather；
- legacy 模式：保持当前 collective 顺序不变。

若未来允许 rank 间 task 不一致，必须先增加显式 task synchronization；本需求不支持该情况。

## 9. Teacher-score distillation

当前 teacher scores 描述 query 自己的 positive/hard-negative group，不包含其他 query 的
positive。因此 `f2llm` 模式下：

```text
loss = hard_ce + positive_only_in_batch_ce + local_group_distill_loss
```

规则：

1. distillation 只对 `hard_logits` 计算一次；
2. 不为 positive-only in-batch logits 合成或扩展 teacher targets；
3. `EmbeddingOutput.hard_loss` 和 `in_batch_loss` 只报告各自 CE，不把 distillation 重复计入
   任一分项；total `loss` 包含 distillation；
4. legacy 和 `no_in_batch_neg=true` 的 teacher 行为保持不变。

## 10. 输出契约

`EmbeddingOutput` 增加可选诊断字段：

```python
hard_loss: Optional[Tensor] = None
in_batch_loss: Optional[Tensor] = None
```

语义：

- legacy 模式允许两项均为 `None`，保持原路径无额外计算；
- `no_in_batch_neg=true` 时 `hard_loss` 为 query-local CE，`in_batch_loss=None`；
- F2LLM retrieval 时两项均存在，且不含 teacher distillation；
- `loss` 始终是 Trainer backward 使用的完整标量。

`scores` 字段继续保持单 Tensor：

- legacy 路径不变；
- `no_in_batch_neg=true` 返回 hard logits；
- F2LLM retrieval 返回 positive-only in-batch logits。

hard logits 不新增到公共输出，单元测试直接调用 loss helper 或检查 `hard_loss`。训练日志分项聚合
属于后续可观测性需求；本需求只保证模型输出提供无歧义的数据源。

## 11. 性能与内存

记 global positive 数为 `N`、group size 为 `G`：

| 实现 | score 数量（每 rank） | 跨卡传输 |
| --- | ---: | --- |
| 当前 legacy joint | `B * N * G` | queries + 全部 passages |
| F2LLM 两组归一化 | `B * G + B * N` | positives only |
| padded 单次 CE 调用 | `2 * B * max(G, N)` | positives only |
| 笛卡尔积单次 CE | `B * G * N` | positives only |

本需求采用第二行。Transformer forward/backward 通常远重于 CE API 调用；优化重点是减少 passage
gather 和无用 score，而不是把两个 normalization 强行包装成一个 CE kernel。

## 12. 预计代码变更

| 文件 | 变更 |
| --- | --- |
| `vibe_emb/arguments.py` | 在 `EmbedTrainingExtras` 增加 `retrieval_loss_mode`，默认 `legacy` |
| `vibe_emb/modeling.py` | 增加 mode 校验、F2LLM loss helper、positive-only gather 和诊断输出 |
| `configs/train_f2llm_stage2_full.yaml` | 验收通过后显式设置 `retrieval_loss_mode: f2llm`，并使用新的输出目录 |
| `tests/test_embedding_training_loss.py` | 增加固定 tensor、兼容性、梯度和 distributed loss 测试 |
| `aidoc/f2llm/f2llm_stage2_training_code_plan.md` | 实现后更新 loss 状态并链接本文 |

本需求不需要修改 `vibe_emb/data.py` 或 `vibe_emb/collator.py`；现有
`no_in_batch_neg`、passage group 布局和 rank 对齐信息已经足够完成 loss 路由。

## 13. 测试要求

### 13.1 配置与兼容性

1. 未配置 `retrieval_loss_mode` 时解析为 `legacy`。
2. `legacy` 的单卡和多卡 path、score shape、loss 数值保持当前行为。
3. 非法 mode 在训练启动前失败。
4. resolved config 包含最终 mode。

### 13.2 单卡数值

使用固定 FP32 tensor 手工验证：

1. hard logits 只包含 query 自己的 group。
2. in-batch logits 只包含 positives。
3. 其他 query 的 hard negatives 不进入任一非本地 hard group。
4. 自定义 `logsumexp` 结果与两次 `F.cross_entropy` 相加一致。
5. total loss 等于 hard 和 in-batch 两项之和。
6. 合并候选后的单一 union CE 与目标 loss 不相等，防止实现错误退化。
7. query、positive 和 hard-negative gradients 均存在且有限。

### 13.3 Classification/clustering 回归

1. `no_in_batch_neg=true` 在 `legacy` 和 `f2llm` mode 下产生相同 loss 和 score。
2. 该路径不调用 distributed gather。
3. classification group size 2 和 clustering 任意合法 group size 均只计算 own-group CE。

### 13.4 Teacher scores

1. distillation 只作用于 hard logits。
2. total loss 等于 hard CE、in-batch CE 和一次 distillation 之和。
3. 不构造 global teacher targets。

### 13.5 两 rank Gloo 测试

1. 每个 rank 只 gather `[B, D]` positive tensor，而不是 `[B * G, D]` passages 或 query。
2. global positive 顺序为 rank-major，target offset 为 `rank * B + local_index`。
3. 每个 rank 只计算 local query rows。
4. 本 rank query、positive 和 hard negatives 有有限梯度。
5. 两个 rank 执行相同 collective 次序，无 hang。
6. 分布式 loss 与按相同“remote positives 不回传梯度”语义构造的 reference 数值一致。

## 14. 验收标准

- AC-1：历史配置不增加字段时，legacy loss 数值与改动前一致。
- AC-2：`no_in_batch_neg=true` 的 classification/clustering 路径数值不变且不 gather。
- AC-3：单卡 F2LLM retrieval loss 与两次标准 CE 相加误差不超过 FP32 测试容差。
- AC-4：两 rank 下只 gather positives，targets、loss 和梯度测试通过。
- AC-5：F2LLM path 的其他 query hard negatives 不影响当前 query 的 loss。
- AC-6：teacher distillation 只增加一次 local-group distill loss。
- AC-7：非法 mode 在加载大模型前失败，resolved config 记录最终 mode。
- AC-8：`configs/train_f2llm_stage2_full.yaml` 显式启用新模式并使用不同于 legacy
  checkpoint 的输出目录。
- AC-9：相关单测、`py_compile` 和 `git diff --check` 通过。
- AC-10：使用小模型完成单卡和两卡 smoke；日志无 collective hang、shape mismatch、target
  越界或 NaN。

## 15. 实施顺序

1. 增加配置字段、允许值校验和配置解析测试。
2. 冻结 legacy path 回归测试，不重写现有 `_contrastive_loss`。
3. 实现 local hard logits 和 loss helper。
4. 实现 positive 提取、单卡 in-batch logits 和精确 `logsumexp` loss。
5. 接入 positives-only distributed gather 和 rank-offset targets。
6. 接入 teacher distillation 与 `EmbeddingOutput` 诊断字段。
7. 完成单卡、classification/clustering、teacher 和两 rank 测试。
8. 更新 F2LLM full config 到新 mode 和新输出目录。
9. 执行单卡/两卡 smoke，记录命令、loss 分项和验证结果。
10. 更新主训练计划的实现状态；完整训练和 MTEB 对比作为后续实验执行。

## 16. 实施与验收结果（2026-07-20）

实现已按本文冻结的语义完成：

- `vibe_emb/arguments.py` 新增 `retrieval_loss_mode=legacy|f2llm`，默认保持 `legacy`，非法值在
  加载模型前失败；
- `vibe_emb/modeling.py` 在 `f2llm` path 中计算两个独立 FP32 `logsumexp` normalization，
  多卡只 gather positives，本地 query rows 参与 loss；
- `EmbeddingOutput` 暴露 `hard_loss` 和 `in_batch_loss`；teacher distillation 只加在 local
  hard logits 上一次；
- `no_in_batch_neg=true` 的 classification/clustering own-group CE 保持不变；
- `configs/train_f2llm_stage2_full.yaml` 显式设置 `retrieval_loss_mode: f2llm`，输出切换到
  `results/f2llm-s2-lora64-f2llm-loss`；
- 新增 `configs/train_f2llm_stage2_loss_smoke.yaml` 和
  `scripts/f2llm/verify_retrieval_loss_smoke.py`，用于真实 Qwen3-0.6B retrieval
  forward/backward 验证。

自动化验证：

```text
41 passed in 394.05s
```

该联合回归包含 17 个 loss/config/distributed 测试以及既有 Indexed Arrow、batch plan 和
DataLoader prefetch 测试。`py_compile` 和相关文件的 `git diff --check` 同时通过。

真实模型验证使用 `/mnt/share/models/Qwen3-0.6B`、BF16、LoRA r64 和 H100：

- 单卡 `sts22` retrieval：`scores=[2,2]`，`hard_loss=2.1378059`，
  `in_batch_loss=0.2648869`，`loss=2.4026928`，分项残差为 0；392 个 LoRA 梯度 tensor
  全部有限；
- 两卡 `sts22` retrieval：每个 rank 的 `scores=[2,4]`，每个 rank 恰好 gather 一次
  `[2,1024]` tensor；即只 gather 两个 local positives。两 rank 的分项 loss、总 loss 和
  392 个 LoRA 梯度 tensor 全部有限；
- Trainer 单卡完成 44 steps，第 43 个 optimizer step 为 retrieval，最终
  `train_loss=1.8323`；两卡完成 31 steps，第 30 个 optimizer step 为 retrieval，最终
  `train_loss=1.6519`；
- 两个 smoke adapter 均通过生产 `maybe_apply_peft()` warm-start path 重载，各包含 392 个
  可训练 LoRA tensor。

smoke 产物位于
`/tmp/vibe_emb_f2llm_loss_smoke_single_20260720` 和
`/tmp/vibe_emb_f2llm_loss_smoke_ddp2_20260720`。它们是临时验收产物，不是正式实验 checkpoint。
完整 stage-2 长训、learning-rate A/B、checkpoint-12000 adapter warm start 和 MTEB 对比仍是
后续实验，不属于本次实现完成条件。
