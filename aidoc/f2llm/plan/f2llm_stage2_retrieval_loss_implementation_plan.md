# F2LLM Stage-2 Retrieval Loss Implementation Plan

状态：Implemented and validated
日期：2026-07-20
依据：`aidoc/f2llm/spec/f2llm_stage2_retrieval_loss_spec.md`

## 0. 执行摘要（2026-07-20）

CP-1 至 CP-5 已完成：配置默认兼容、F2LLM 双 normalization、positives-only distributed
gather、teacher/legacy/no-in-batch 回归、full config 启用和真实单卡/两卡 smoke 均已通过。

实际新增或修改的 loss 范围文件：

- `vibe_emb/arguments.py`；
- `vibe_emb/modeling.py`；
- `tests/test_embedding_training_loss.py`；
- `configs/train_f2llm_stage2_full.yaml`；
- `configs/train_f2llm_stage2_loss_smoke.yaml`；
- `scripts/f2llm/verify_retrieval_loss_smoke.py`；
- 本 plan、对应 spec 和主训练文档。

联合自动化回归为 `41 passed in 394.05s`。真实 Qwen3-0.6B verifier 的单卡 loss 为
`2.4026928 = 2.1378059 + 0.2648869`；两卡每个 rank 只 gather 一次 `[2,1024]`
positive tensor，输出 score shape 为 `[2,4]`。Trainer 分别完成单卡 44 steps 和两卡
31 steps，并实际经过 retrieval batch；两个最终 adapter 均通过生产 warm-start 路径重载。

实施中没有修改 sampling、Arrow schema/store、batch-plan 语义、collator passage layout、MRL
或评估代码。工作树原有的 prefetch 改动保持原样，联合测试证明 loss patch 未破坏其测试。

本次实施采用了 plan 中的推荐项：保留当前 prefetch 工作树、正式配置使用
`results/f2llm-s2-lora64-f2llm-loss`、提交稳定 smoke YAML、使用两张空闲 H100、只由模型输出和
verifier 记录 loss 分项，并保持 `learning_rate=5e-5` 不在代码中缩放 loss。仍需实验负责人决定的
事项只有正式长训初始化来源（base/stage-1 或 checkpoint-12000 adapter warm start）、是否进行
learning-rate/gradient-norm A/B，以及何时启动完整长训和 MTEB 对比。

## 1. 实施结果

完成后的训练框架具有两条显式、互不混淆的 retrieval loss 路径：

1. `retrieval_loss_mode=legacy`：保持当前 query 对联合 passage matrix 的单 CE；
2. `retrieval_loss_mode=f2llm`：计算 query-local hard loss，加上 global
   positive-only in-batch loss。

classification/clustering 继续由 resolved `no_in_batch_neg=true` 进入现有 query-local group
CE，不新增 `task_type` 模型分支。F2LLM 多卡路径只 gather positive embeddings，每个 rank 只计算
local query rows，然后由 DDP 正常平均参数梯度。

最终代码结构如下：

```text
resolved no_in_batch_neg
        |
        +-- true ----------------> existing own-group hard CE
        |
        +-- false + legacy ------> existing joint passage-matrix CE
        |
        +-- false + f2llm -------> hard CE + positive-only in-batch CE
                                      |
                                      +-- single rank: local positives
                                      +-- multi rank: gather positives only
```

## 2. 当前工作树与实施边界

当前工作树同时存在 DataLoader prefetch 相关未提交改动，涉及：

- `vibe_emb/train.py`；
- `vibe_emb/trainer.py`；
- `vibe_emb/data.py`；
- `vibe_emb/collator.py`；
- `vibe_emb/callbacks.py`；
- `configs/train_f2llm_stage2_full.yaml`；
- `aidoc/f2llm/f2llm_stage2_training_code_plan.md`；
- `tests/` 下的 prefetch 测试与夹具。

retrieval loss 的核心生产代码只需要修改 `vibe_emb/arguments.py` 和
`vibe_emb/modeling.py`。实施期间不得顺手重写 prefetch、batch accounting、collator metadata 或
Trainer telemetry。共享配置和主训练文档只在 loss 核心及分布式测试全部通过后修改，并作为独立
检查点处理。

开始实施前必须保存当前工作树状态和相关测试基线。不得通过 reset、checkout 或覆盖文件来清理
现有改动。

## 3. 具体文件和符号

### 3.1 `vibe_emb/arguments.py`

修改：

- 新增模块级允许值常量，例如：

  ```python
  RETRIEVAL_LOSS_MODES = frozenset({"legacy", "f2llm"})
  ```

- `EmbedTrainingExtras.retrieval_loss_mode: str = "legacy"`；
- `EmbedTrainingExtras.__post_init__()`：校验 mode，并在错误中输出实际值和允许值。

约束：

- 校验必须在 `parse_sections()` 构造 dataclass 时发生，早于 tokenizer/base model 加载；
- 不把字段加入 `TrainingArguments`；现有 `parse_sections()` 会依据 dataclass fields 自动从
  `training_raw` 中移除该字段；
- 直接调用 `EmbedTrainingExtras(retrieval_loss_mode=...)` 时也必须执行相同校验；
- 不在 `vibe_emb/config.py` 复制第二份 mode 校验。

### 3.2 `vibe_emb/config.py`

预计不修改生产代码。

需要通过测试确认：

- `parse_sections()` 将 `retrieval_loss_mode` 放入 `EmbedTrainingExtras`；
- 该字段不会进入返回的 Transformers training arguments mapping；
- `to_plain_dict(training_extras)` 会把默认或显式 mode 写入 resolved config 数据源。

如果当前解析行为无法满足上述测试，才允许做最小修复；不得借机调整未知字段或其他 section 的
兼容策略。

### 3.3 `vibe_emb/modeling.py`

#### `EmbeddingOutput`

在现有字段之后追加：

```python
hard_loss: Optional[Tensor] = None
in_batch_loss: Optional[Tensor] = None
```

追加而不是插入，避免改变 legacy `ModelOutput` 已有非空字段的相对顺序。

#### 新增内部结果类型 `_ContrastiveLossResult`

建议使用 module-private dataclass：

```python
@dataclass
class _ContrastiveLossResult:
    scores: Tensor
    loss: Tensor
    hard_loss: Optional[Tensor] = None
    in_batch_loss: Optional[Tensor] = None
```

它只用于模型内部传递总 loss 和诊断分项，不作为公开 API。若实现时选择等价的 typed tuple，必须
保持同样字段语义，禁止返回位置含义不清的四元 tuple。

#### 新增 `_mean_nll_from_logits`

建议签名：

```python
@staticmethod
def _mean_nll_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    ...
```

职责：

- 将 loss reduction 输入转换为 FP32；
- 计算 `logsumexp(logits, -1) - target_logit`；
- 对 query 维取 mean；
- 不改变返回给 `EmbeddingOutput.scores` 的 logits dtype；
- 测试其与 `F.cross_entropy(logits.float(), targets)` 数值和梯度一致。

#### 新增 `_passage_groups`

建议签名：

```python
@staticmethod
def _passage_groups(q_reps: Tensor, p_reps: Tensor) -> Tensor:
    ...
```

职责：

- 校验 query 数大于 0；
- 校验 passage 数能被 query 数整除；
- 校验 query/passage hidden size 一致；
- 返回 `[B, G, D]` view/reshape；
- 保持第 0 列为 positive 的既有 invariant。

不得在这里重新归一化 full-dimension embeddings；当前 encoder 已按 model config 处理 normalize，
MRL 不在本需求范围。

#### 新增 `_f2llm_retrieval_loss`

建议签名：

```python
def _f2llm_retrieval_loss(
    self,
    q_reps: Tensor,
    p_reps: Tensor,
    teacher_scores: Optional[Tensor],
    temperature: float,
) -> _ContrastiveLossResult:
    ...
```

职责和固定顺序：

1. 通过 `_passage_groups()` 得到 `[B, G, D]`；
2. 用 `einsum("bd,bgd->bg", ...) / temperature` 计算 local hard logits；
3. 提取 `positive_local = groups[:, 0, :]`；
4. 仅在 `self.negatives_cross_device and self.training` 时调用
   `_dist_gather_tensor(positive_local)`；
5. 用 local queries 对 local/global positives 计算 in-batch logits；
6. hard targets 全为 0；in-batch targets 单卡为 local index，多卡为
   `process_rank * B + local_index`；
7. 用 `_mean_nll_from_logits()` 分别计算两个 normalization；
8. 若存在 teacher scores，只对 hard logits 增加一次现有 `distill_loss()`；
9. 返回 `scores=in_batch_logits`、完整 total、纯 hard CE、纯 in-batch CE。

禁止：

- gather `q_reps`；
- gather 完整 `p_reps`；
- 使用 current full passage score matrix 后再 mask；
- 把 hard candidates 和 global positives 拼成一个共享 softmax；
- 对两个分项再次除以 2 或 world size。

#### 修改 `_contrastive_loss`

将返回值从不具名 `(scores, loss)` 收敛为 `_ContrastiveLossResult`，但按以下边界控制改动：

1. `no_in_batch_neg=true` 分支保留当前 score 构造、target、CE 和 distillation 操作顺序；仅把
   teacher 前的纯 CE 保存为 `hard_loss`；
2. `no_in_batch_neg=false, mode=legacy` 的单卡和跨卡代码保持原样，只包装返回值；
3. `no_in_batch_neg=false, mode=f2llm` 调用 `_f2llm_retrieval_loss()`；
4. legacy branch 的 `hard_loss`、`in_batch_loss` 均为 `None`；
5. `no_in_batch_neg=true` 不得因为 mode 为 `f2llm` 执行 positive gather。

这里不将当前 `_contrastive_loss` 全面拆成多层通用 loss framework；只增加明确 dispatcher 和
F2LLM helper，避免扩大回归面。

#### 修改 `forward`

- 从 `_contrastive_loss()` 接收 `_ContrastiveLossResult`；
- 构造 `EmbeddingOutput` 时透传 `hard_loss`、`in_batch_loss`；
- `dataset_name`、`train_group_size` 仍不参与 loss 路由；
- temperature 的 per-dataset `loss_kwargs` override 继续生效；
- query/passage encoder 和 sub-batch 路径不改。

### 3.4 `tests/test_embedding_training_loss.py`

新增独立测试文件，不复用当前面向 `vibe_eval` 的 `tests/test_modeling.py`，避免训练/评估模型测试
混在一起。

建议 module-level fixtures/helpers：

- `_minimal_model(mode, negatives_cross_device=False)`：不加载真实 pretrained model；
- `_reference_two_ce(...)`：只在测试中实现明确的标准 CE reference；
- `_fixed_reps(...)`：返回可区分 positive、own hard、other hard 的固定 tensor；
- `_distributed_loss_worker(...)`：可被 `torch.multiprocessing` spawn/pickle 的顶层函数；
- `_run_two_rank_case(...)`：使用临时 file store 初始化 Gloo，并收集每 rank 结果。

不得通过复制生产 `logsumexp` 公式来构造唯一 reference；至少一个主断言必须直接使用两次
`F.cross_entropy`。

### 3.5 `configs/train_f2llm_stage2_full.yaml`

只在所有自动化测试通过后修改：

```yaml
training:
  retrieval_loss_mode: f2llm
  output_dir: <new-output-dir>
```

该文件当前与 DataLoader prefetch 工作重叠。修改前必须人工确认：

- prefetch 最终采用的 worker/factor/unit-block 参数；
- 新 loss 实验的最终输出目录名；
- 是否从 base model 开始，或把旧 adapter 作为新 optimizer 的 warm start。

loss 提交不得覆盖、回滚或暗中重写已有 prefetch 参数。

### 3.6 文档

实现完成后修改：

- `aidoc/f2llm/f2llm_stage2_training_code_plan.md`：链接 spec/plan，更新第 5、6 项状态和验证结果；
- 本文：按 checkpoint 更新完成状态、命令和实际测试结果；
- spec：只更新状态和经过实现验证后确需澄清的事实，不改变已确认目标。

文档更新必须与对应代码 checkpoint 同步，不提前声称 smoke 或多卡验证完成。

## 4. 分阶段实施顺序与测试

### 阶段 0：冻结基线与冲突边界

操作：

1. 记录 `git status --short` 和 loss/prefetch 相关 diff；
2. 确认 `vibe_emb/arguments.py`、`vibe_emb/modeling.py` 当前没有未合并的用户改动；
3. 运行现有 Indexed Arrow、prefetch 和配置解析相关测试，建立改动前基线；
4. 用固定 tensor 记录当前三类行为：
   - `no_in_batch_neg=true` own-group CE；
   - legacy single-rank joint CE；
   - legacy distributed joint CE 的 score shape/target 规则。

测试命令：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_indexed_arrow_store.py \
  tests/test_dataloader_prefetch.py
```

退出条件：

- 基线命令通过，或已有失败被完整记录且确认与 retrieval loss 无关；
- 没有覆盖用户未提交改动；
- 新测试文件名与现有测试不冲突。

### 阶段 1：配置接口

生产修改：

- `EmbedTrainingExtras.retrieval_loss_mode`；
- `EmbedTrainingExtras.__post_init__()`。

测试：

1. 默认 mode 为 `legacy`；
2. 显式 `f2llm` 正确解析；
3. `retrieval_loss_mode` 不进入 Transformers training args；
4. `to_plain_dict(extras)` 包含 resolved mode；
5. 非法值构造 dataclass 和 `parse_sections()` 均失败；
6. 错误在任何模型加载 helper 被调用前发生。

测试命令：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_embedding_training_loss.py -k 'config or mode'
```

退出条件：配置测试全部通过，历史 YAML 不增加字段时仍能解析。

### 阶段 2：legacy 回归护栏和内部返回契约

生产修改：

- `EmbeddingOutput` 追加诊断字段；
- 新增 `_ContrastiveLossResult`；
- `_contrastive_loss()` 和 `forward()` 改用具名内部结果。

此阶段不接入 F2LLM 计算，只包装现有分支。

测试：

1. 固定 tensor 下，legacy single-rank scores 和 loss 与改动前 reference 一致；
2. `no_in_batch_neg=true` scores、total 和 teacher behavior 一致；
3. teacher 为空时 `hard_loss == total loss`；teacher 非空时 total 等于 hard CE 加一次 distill；
4. legacy `EmbeddingOutput` 的已有字段顺序和属性访问不变；
5. legacy 分支诊断字段为 spec 约定值。

测试命令：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_embedding_training_loss.py -k 'legacy or no_in_batch or teacher'
```

退出条件：legacy 数值回归通过后才允许实现新目标。

### 阶段 3：单卡 F2LLM loss

生产修改：

- `_mean_nll_from_logits()`；
- `_passage_groups()`；
- `_f2llm_retrieval_loss()` 的单卡部分；
- `_contrastive_loss()` 增加 mode dispatcher。

测试：

1. hard logits shape 为 `[B, G]`；
2. in-batch logits shape 为 `[B, B]`；
3. total 等于两次标准 CE 相加；
4. `_mean_nll_from_logits()` 与 FP32 `F.cross_entropy` 的 loss/gradient 一致；
5. 修改其他 query 的 hard negative 不影响当前 query 对应的 hard/in-batch loss；
6. union CE 明确与目标 loss 不相等；
7. query、positive、hard negative 的梯度存在且有限；
8. BF16 logits 使用 FP32 reduction 后无 NaN/Inf；
9. passage 数不可整除、空 query、hidden size 不一致时尽早失败；
10. `no_in_batch_neg=true` 在 `legacy/f2llm` 两个 mode 下结果相同且不调用 gather；
11. teacher distillation 只加在 hard logits 上一次。

测试命令：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_embedding_training_loss.py -k 'not distributed'
```

退出条件：spec AC-1、AC-2、AC-3、AC-5、AC-6、AC-7 通过。

### 阶段 4：两 rank positives-only gather

生产修改：

- `_f2llm_retrieval_loss()` 接入现有 `_dist_gather_tensor(positive_local)`；
- 不改 legacy `_dist_gather_tensor()` 调用和行为。

Gloo 测试使用两个独立进程，不能只 mock `world_size`：

1. 每 rank local `B`、group size `G` 和 hidden size `D` 固定且相同；
2. 包装/记录 `_dist_gather_tensor()` 输入，断言唯一输入 shape 为 `[B, D]`；
3. 断言没有 gather `[B * G, D]` passages，也没有 gather `[B, D]` query 的第二次调用；
4. global positives 顺序为 rank-major；
5. targets 为 `rank * B + arange(B)`；
6. 每 rank scores shape 为 `[B, B * W]`，不是 `[B * W, B * W * G]`；
7. loss 与按 remote positives detached 语义构造的 reference 一致；
8. local query、positive、hard-negative gradients 均有限；
9. remote positive 不建立跨进程 autograd graph；
10. classification/clustering case 在两 rank 上共同跳过 collective；
11. worker 设置超时并在失败后销毁 process group，防止 pytest 永久挂起。

测试命令：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_embedding_training_loss.py -k distributed
```

退出条件：spec AC-4 通过，无 hang、target offset 或 gradient scaling 异常。

### 阶段 5：全量自动化回归

测试命令：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_embedding_training_loss.py \
  tests/test_indexed_arrow_store.py \
  tests/test_dataloader_prefetch.py
```

静态检查：

```bash
/mnt/share/envs/embt/bin/python -m py_compile \
  vibe_emb/arguments.py \
  vibe_emb/modeling.py
```

```bash
git diff --check
```

退出条件：相关自动化测试和静态检查全部通过；loss 改动未影响 prefetch、batch plan 或 Indexed
Arrow 测试。

### 阶段 6：配置启用与文档同步

在人工决定新输出目录和 prefetch 合并顺序后：

1. `configs/train_f2llm_stage2_full.yaml` 增加 `retrieval_loss_mode: f2llm`；
2. 将 `output_dir` 改为不会命中任何 legacy checkpoint 的新目录；
3. 不改变 group size、temperature、学习率、max length 或 sampling；
4. 保留已经验收的 prefetch settings，不借 loss 变更重新调参；
5. 更新 spec/plan/main training doc 的状态和已执行验证。

配置检查：

```bash
PYTHONPATH=. /mnt/share/envs/embt/bin/python -c \
  'from vibe_emb.config import load_yaml_config, parse_sections; raw=load_yaml_config("configs/train_f2llm_stage2_full.yaml"); print(parse_sections(raw)[3])'
```

退出条件：输出显示 `retrieval_loss_mode='f2llm'`，新 output dir 不存在或不包含 legacy
checkpoint，resolved config 测试通过。

### 阶段 7：单卡与两卡 smoke

smoke 必须使用包含 retrieval unit 的小 profile 或显式小 batch 配置，不能恰好只消费
classification/clustering 而误判新路径已覆盖。

最低要求：

1. 单卡至少 2 optimizer steps，其中至少 1 个 retrieval batch；
2. 两卡至少 2 optimizer steps，其中至少 1 个 retrieval batch；
3. 记录 total、hard、in-batch loss，三者有限且 total 符合分项关系；
4. 两卡日志证明只执行 positives-only gather 对应的 score shape；
5. checkpoint 和最终 adapter 能重新加载；
6. smoke output 使用独立临时/实验目录，不复用 checkpoint-12000 目录；
7. 不在 smoke 阶段改变学习率、loss 权重或 group size 来“修正”数值。

若当前仓库没有稳定的小 profile smoke YAML，需要人工决定是：

- 新增一个提交内的 `configs/train_f2llm_stage2_loss_smoke.yaml`；或
- 使用测试临时目录生成 YAML，不把机器路径写入仓库。

退出条件：spec AC-8、AC-9、AC-10 通过，文档记录实际命令、环境、GPU 数、steps、loss 分项和
产物路径。

## 5. 兼容策略

### 5.1 配置兼容

- `retrieval_loss_mode` 默认 `legacy`；所有未设置该字段的历史配置保持当前行为；
- 字段属于 `EmbedTrainingExtras`，不会传入 `TrainingArguments`；
- 旧 `resolved_config.yaml` 没有该字段时解释为 legacy，不回写旧产物；
- 不提供旧字段别名或模糊字符串兼容；非法值直接失败。

### 5.2 Loss 兼容

- legacy single-rank/cross-device 分支不改变 score 候选、target、reduction 和 teacher 语义；
- `no_in_batch_neg=true` 优先于 retrieval loss mode，classification/clustering 不变化；
- `temperature`、embedding normalize 和 per-dataset `loss_kwargs` 语义不变；
- `train_group_size`、positive-at-offset-zero 和 flattened passages invariant 不变。

### 5.3 输出/API 兼容

- `EmbeddingOutput` 只在末尾增加可选字段；已有 `loss/scores/q_reps/p_reps` 名称不变；
- legacy retrieval path 的新诊断字段保持 `None`；`no_in_batch_neg=true` path 按 spec
  暴露纯 own-group CE `hard_loss`，`in_batch_loss=None`；
- F2LLM mode 的 `scores` 明确定义为 positive-only in-batch logits；
- Trainer 仍只使用 `outputs.loss` 反向，不修改 optimizer/scheduler/checkpoint 格式。

### 5.4 Checkpoint 兼容

- 代码可以读取旧模型/adapter checkpoint，因为没有新增模型参数；
- 旧 Trainer checkpoint 只能在 `retrieval_loss_mode=legacy` 下语义等价 resume；
- 从旧 checkpoint 切换成 `f2llm` 后继续 optimizer/scheduler 属于目标漂移，禁止作为 resume；
- 如人工选择 adapter warm start，必须新建 output dir 和 optimizer/scheduler，并在实验记录中明确。

### 5.5 并行 prefetch 工作兼容

- loss 不依赖 `_batch_metadata`、worker 数或 prefetch factor；
- prefetch 不得改变同一步的 `no_in_batch_neg`、group size 或 rank-local passage layout；
- retrieval loss 测试至少回归一次 worker 0 路径；prefetch 自身 worker 1 测试继续由其独立计划覆盖；
- shared full config 的最终修改集中到阶段 6，避免两条工作流互相覆盖。

## 6. 风险与缓解

| 风险 | 影响 | 缓解/检测 |
| --- | --- | --- |
| 把两组候选拼成一个 softmax | 目标函数错误 | 固定 tensor 对比两次 CE，并断言 union CE 不相等 |
| 误 gather 全部 passages 或 queries | 通信和 score 规模回到 legacy | 两 rank 记录 gather 输入 shape 和调用次数 |
| rank target offset 错误 | 正例被当负例，loss 异常 | 两 rank 使用可识别 positive tensor，逐 rank 检查 targets |
| 各 rank 进入不同 branch | collective hang | 依赖现有相同 batch plan；Gloo timeout；smoke 覆盖 retrieval 和 no-in-batch step |
| DDP loss 被额外缩放 | 学习率等效变化 | local mean 后不乘 world size；与 detached-remote reference 对比梯度 |
| FP16/BF16 `logsumexp` 精度不足 | NaN、Inf 或数值漂移 | loss reduction 转 FP32；BF16 测试检查有限值 |
| teacher distillation 被加两次 | 总 loss 偏大 | 分项 identity 测试，teacher 只接受 hard logits |
| 修改 no-in-batch path | classification/clustering 回归 | mode 交叉测试，固定 scores/loss reference |
| `scores` 语义在新 mode 不清晰 | 调试调用方误读 | spec 固定为 positive-only logits；legacy 不变 |
| 旧 checkpoint 直接续训新 loss | optimizer history 与目标不一致 | 新 output dir；启动前人工审查 resume/warm-start 参数 |
| 新 loss 总量约为两项之和 | 与 legacy loss 尺度、LR 不可直接比较 | 不在代码中擅自除 2；smoke 后由实验负责人决定是否调 LR |
| full config 与 prefetch diff 冲突 | 覆盖参数或混淆实验变量 | 配置修改独立 checkpoint；实施前后审查 exact diff |
| 测试只覆盖 classification/clustering | 新 retrieval path 实际未运行 | fixture/smoke 明确断言 `in_batch_loss is not None` |
| distributed pytest 残留进程 | CI/本机挂起 | file-store/Gloo timeout、finally destroy group、join 超时后终止子进程 |

## 7. 提交检查点

以下是逻辑提交边界，不授权当前计划阶段实际提交或 push。

### CP-0：Spec 和实施计划

内容：

- retrieval loss spec；
- 本实施计划。

门槛：文档边界、配置名称、checkpoint 语义和人工决定项确认。

建议提交信息：

```text
docs: specify F2LLM retrieval loss implementation
```

### CP-1：配置接口与兼容测试

内容：

- `EmbedTrainingExtras.retrieval_loss_mode`；
- mode validation；
- 配置解析和 legacy 默认测试。

门槛：阶段 1 测试通过，未修改模型数值路径。

建议提交信息：

```text
feat: add configurable retrieval loss mode
```

### CP-2：单卡 loss 核心

内容：

- typed loss result；
- output diagnostics；
- legacy regression guard；
- hard + positive-only in-batch 单卡实现；
- teacher 和梯度测试。

门槛：阶段 2、3 测试通过，legacy/分类/聚类回归通过。

建议提交信息：

```text
feat: implement F2LLM retrieval loss
```

### CP-3：Positives-only distributed loss

内容：

- positives-only gather；
- rank-offset targets；
- 两 rank Gloo 数值、shape、gradient 测试。

门槛：阶段 4、5 全部通过，无 distributed hang。

建议提交信息：

```text
test: verify distributed F2LLM retrieval loss
```

如果 production distributed 接线与测试无法独立，允许 CP-2/CP-3 合并，但 code review 中仍需按
两个逻辑阶段展示 diff 和测试证据。

### CP-4：F2LLM 配置启用

内容：

- full config 显式 `retrieval_loss_mode=f2llm`；
- 新 output dir；
- 保留最终确认的 prefetch 参数；
- 主训练文档状态更新。

门槛：人工决定配置冲突项；阶段 5 通过；config diff 单独审查。

建议提交信息：

```text
config: enable F2LLM retrieval loss for stage 2
```

### CP-5：Smoke 验证记录

内容：

- 单卡/两卡 smoke 配置或可复现命令；
- 实际 loss 分项、score shape、checkpoint reload 结果；
- spec/plan 状态更新。

门槛：阶段 7 全部通过。

建议提交信息：

```text
docs: record F2LLM retrieval loss validation
```

## 8. 每个检查点的统一检查

每个 checkpoint 都执行：

1. `git status --short`：确认没有混入无关数据、结果或 prefetch 改动；
2. `git diff --stat` 和精确 diff review；
3. `git diff --check`；
4. 本 checkpoint 的 targeted pytest；
5. 修改 Python 时执行相关 `py_compile`；
6. 确认没有新增 `__pycache__`、`.pytest_cache`、临时日志、模型或训练产物；
7. 配置 checkpoint 确认 output dir 不会命中 legacy checkpoint；
8. distributed checkpoint 确认子进程全部退出。

未经用户明确要求，不执行 commit、push 或 PR。

## 9. 需要人工决定的事项

### HD-1：与 DataLoader prefetch 工作的合并顺序

建议：先完成/冻结 prefetch 核心代码和测试，再实现 loss；两者的生产核心文件基本不重叠，最后
统一处理 `configs/train_f2llm_stage2_full.yaml` 和主训练文档。

需要决定：

- prefetch 是否先形成独立 commit；
- loss implementation 是否基于当前未提交 prefetch 工作树继续；
- CP-4 配置由哪项工作负责最终合并。

### HD-2：新实验输出目录

必须选择一个从未包含 legacy checkpoint 的目录。不能复用：

- `results/f2llm-s2-lora64`；
- `results/f2llm-stage2-full`；
- 任何能够被 last-checkpoint 自动发现为 legacy run 的目录。

建议命名包含 loss 语义，例如：

```text
results/f2llm-s2-lora64-f2llm-loss
```

最终名称由实验负责人确认。

### HD-3：初始化方式

选择之一：

1. 从 `/mnt/share/models/Qwen3-0.6B` 或指定 stage-1 backbone 全新训练；
2. 从 checkpoint-12000 的 adapter 做 warm start，但新建 optimizer/scheduler 和 output dir；
3. 不允许把 checkpoint-12000 当作 Trainer resume 后切换 loss。

该选择会显著改变实验解释，必须写入训练记录。

### HD-4：Smoke 配置载体

选择：

- 提交一个稳定的小 profile smoke YAML，便于以后回归；或
- 只在 `/tmp` 生成机器本地 smoke 配置，文档记录生成参数。

若新增仓库配置，不能硬编码临时机器路径或复用正式 output dir。

### HD-5：Smoke 和正式训练资源

需要确认：

- 可用 GPU 数和具体设备；
- 单卡/两卡 smoke 的模型路径；
- 是否允许读取完整 Arrow profile，还是使用小 profile；
- smoke steps 和最大可接受运行时间；
- 正式训练是否在本实现任务内启动。

### HD-6：Loss 分项日志是否随本次实现交付

spec 只要求 `EmbeddingOutput` 暴露 `hard_loss`/`in_batch_loss`，持续聚合到 Trainer/TensorBoard
属于后续可观测性范围。

建议：本次只完成模型输出和 smoke 打印；正式滑动平均日志单独设计，避免与当前 prefetch telemetry
改动耦合。若人工要求本次同时加入训练日志，需要先补充日志聚合、DDP reduction 和 callback
契约，再实施。

### HD-7：正式实验超参数是否保持不变

新 loss 是两个 CE 之和，数值尺度和 legacy joint CE 不同。代码不得擅自除 2 或调整学习率。

需要在 smoke/短跑后由实验负责人决定：

- 是否保持 `learning_rate=5e-5`；
- 是否需要 gradient norm/loss 曲线对比；
- 是否先做固定 steps 的 legacy-vs-f2llm A/B；
- MRL 和 clustering group size 10 是否作为后续独立变量。

## 10. 完成定义

只有同时满足以下条件，implementation 才能标记完成：

1. CP-1 至 CP-3 的代码和自动化测试完成；
2. legacy、classification、clustering 数值回归通过；
3. 单卡 F2LLM loss 与两次 CE reference 一致；
4. 两 rank 只 gather positives，target、loss 和梯度验证通过；
5. teacher distillation 只作用于 local hard group 一次；
6. full config 的 mode、output dir 和 prefetch settings 经人工确认；
7. 单卡和两卡 retrieval smoke 通过，adapter/checkpoint 可重新加载；
8. spec、plan 和主训练文档只记录实际完成的验证；
9. 未修改采样、Arrow、batch plan、collator passage layout、MRL 或评估逻辑；
10. 未提交缓存、日志、模型权重或大体积结果。

完整长训和 MTEB 对比不属于代码 implementation 的完成门槛；如果用户要求在同一任务中继续，必须
作为独立实验阶段记录配置、资源、进度、失败和最终覆盖结果。
