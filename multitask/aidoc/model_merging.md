# 多任务 embedding 的 SLERP/self-positioning model merging 实验说明

## 目标

当前有 4 个从同一起点继续训练出的 LoRA embedding adapter：

| 任务 | adapter |
|---|---|
| basic | `results/vibe-emb-basic-from-ckpt1000/checkpoint-3000` |
| code | `results/vibe-emb-code-from-ckpt1000/checkpoint-1328` |
| inst | `results/vibe-emb-inst-from-ckpt1000/checkpoint-1500` |
| tool | `results/vibe-emb-tool-from-ckpt1000/checkpoint-1631` |

共同起点是：

```text
results/vibe-embedder-full/checkpoint-1000
```

目标是在不重新训练全模型的情况下，把 4 个任务 adapter 的能力合到一个 adapter 中。这里按论文思路在 LoRA adapter 参数空间做 task-vector merging：

```text
v_i = theta(adapter_i) - theta(adapter_0)
theta_merged = theta(adapter_0) + lambda * V
```

其中 `adapter_0` 是共同起点，`V` 是多个任务向量经过 SLERP 后得到的方向，`lambda` 控制最终 task vector 的范数。

## 原理

### Task vector

每个任务模型都从同一个 `M_0` 继续训练，因此可以把任务能力近似表示成参数差：

```text
v_i = theta(M_i) - theta(M_0)
```

在当前工程里 checkpoint 是 PEFT LoRA adapter，`tools/merge_multi_slerp.py` 现在支持三种固定权重 Multi-SLERP 路径：

| `--merge-space` | 融合对象 | 产物 |
|---|---|---|
| `adapter` | LoRA checkpoint 中的 `lora_A.weight` / `lora_B.weight` 参数差 | PEFT adapter |
| `delta-w` | 每个 LoRA 还原出的 dense `delta W = B @ A * scale` | full HF model |
| `full-model` | 每个 LoRA 先 `merge_and_unload()` 到 base 后的完整模型权重 | full HF model |

`adapter` 是原始低成本路径；`delta-w` 更接近模型实际函数更新；`full-model` 用完整权重空间做同一类 task-vector merge。传入 `--reference-adapter` 时，三者都会以共同起点作为 task-vector 原点。

### SLERP 方向融合

两个 task vector 的 spherical linear interpolation 为：

```text
slerp(v_i, v_j, t)
  = sin((1 - t) * alpha) / sin(alpha) * v_i
  + sin(t * alpha) / sin(alpha) * v_j
```

其中 `alpha` 是两个向量的夹角。实现里夹角不是逐 tensor 计算，而是把所有 LoRA tensor 当作一个全局向量来计算 dot product 和 norm，更接近论文里的整体参数向量。

多个任务向量按 LMX-Merge 的 self-positioning 思路迭代合并。设可学习方向权重为 `a_i`，第 `i` 个任务加入时：

```text
V_1 = v_1
V_i = slerp(V_{i-1}, v_i, t_i)
t_i = a_i / (mean(a_1 ... a_{i-1}) + a_i)
```

默认用前面任务权重的均值作为已合并方向的权重，对应论文 5.1 中的 `sum(a_i)/(N-1)` 写法。脚本也支持 `--previous-weight sum`。

### alpha/lambda 搜索

论文不是手动定 `t` 和 `lambda`，而是在小 probe dataset `D_t` 上解优化问题：

```text
min_{a_i, lambda}  mean_{I in D_t} L_CL(I; theta_0 + lambda * V(a)) + mu * lambda
```

本实现中：

- `a_i` 和 `lambda` 都用 `softplus(raw)` 参数化，保证为正。
- `L_CL` 使用训练同款对比学习交叉熵：query 对所有 positive/negative passages 打分，target 是本 query 的 positive。
- `mu * lambda` 是范数正则，防止 probe set 上把 task vector 放得过大。
- 只优化少量 merge 超参，不更新 base model 或 LoRA adapter 本身。

关键实现点是 `torch.func.functional_call`：每个 step 根据当前 `a_i/lambda` 生成 differentiable merged LoRA 参数，然后用这些参数做一次前向和 loss 反传。这样梯度能从 loss 回到 `a_i/lambda`，不会因为直接覆盖 `module.weight.data` 而断图。

## 代码入口

新增和修正的脚本：

| 文件 | 用途 |
|---|---|
| `tools/merge_self_positioning.py` | 多 adapter SLERP self-positioning 搜索，保存最终 PEFT adapter |
| `tools/merge_multi_slerp.py` | 固定权重 Multi-SLERP merge 工具，支持 `adapter` / `delta-w` / `full-model` 三种空间 |
| `scripts/run_merge_experiment.sh` | 调用 `merge_multi_slerp.py`，默认指向具体 checkpoint，可用 `MERGE_SPACE=...` 切换融合路径 |
| `scripts/run_self_positioning_merge.sh` | self-positioning 的 shell 入口，默认跑 4-task 融合，可选 smoke eval/MTEB eval |
| `scripts/eval_adapter_retrieval_smoke.py` | 本地小样本 retrieval sanity eval，query prompt 处理与训练格式对齐 |

`merge_self_positioning.py` 支持同一类数据传多个 jsonl：

```bash
--dataset basic=data/train/embeddings-fine-tuning/fever_seed1337_5000.jsonl \
--dataset basic=data/train/embeddings-fine-tuning/msmarco_seed1337_5000.jsonl \
--dataset code=data/train/codesearchnet/codesearchnet_seed1337_5000.jsonl \
--dataset code=data/train/cosqa/cosqa_seed1337_5000.jsonl
```

语义是：先按 dataset name 分组，同名 jsonl 在组内合并并 reservoir sampling，然后每个不同 name 各采 `--examples-per-dataset` 条。也就是说，`basic` 下面有 7 个 jsonl、`code` 下面有 2 个 jsonl 时，最终仍然是 basic/code 各采同样数量，类别之间保持均匀。

`run_self_positioning_merge.sh` 也支持多子集变量，值可以是空格分隔路径或 glob：

```bash
BASIC_DATASETS='data/train/embeddings-fine-tuning/*_seed1337_5000.jsonl' \
CODE_DATASETS='data/train/codesearchnet/*_seed1337_5000.jsonl data/train/cosqa/*_seed1337_5000.jsonl' \
INST_DATASETS='data/train/msmarco-w-instructions/*_seed1337_5000.jsonl' \
TOOL_DATASETS='data/train/toolret-train/*_seed1337_5000.jsonl' \
SEARCH_STEPS=1000 \
EXAMPLES_PER_DATASET=8000 \
bash scripts/run_self_positioning_merge.sh
```

`merge_self_positioning.py` 的主要产物：

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json / tokenizer_config.json / chat_template.jinja
merge_config.json
search_log.jsonl
```

## 已运行实验

### 0. 固定权重 Multi-SLERP full-model 对照

已用 4 个任务 adapter、共同起点 `results/vibe-embedder-full/checkpoint-1000`、等权重和 `lambda=1.0` 产出两种 full model：

| 路径 | 输出目录 | 说明 |
|---|---|---|
| `delta-w` | `results/merges/multislerp-4task-delta-w-equal-lambda1.0` | 对每个 LoRA 先还原 dense `delta W`，在 delta-W task-vector 空间 Multi-SLERP，再加回 base model |
| `full-model` | `results/merges/multislerp-4task-full-model-equal-lambda1.0` | 每个 LoRA 先 merge 到 base，得到 full model 后在完整权重空间 Multi-SLERP |

两个目录都是标准 HF full model，包含 `config.json`、`model.safetensors`、tokenizer 文件、`merge_config.json` 和 `run.log`。已用 `QwenDecoderOnlyEmbedder(checkpoint=<dir>, device="cpu", dtype="fp32")` 做过 encode smoke check，输出维度均为 `(2, 896)`。

复现命令：

```bash
MERGE_SPACE=delta-w \
MERGED_OUTPUT=results/merges/multislerp-4task-delta-w-equal-lambda1.0 \
bash scripts/run_merge_experiment.sh

MERGE_SPACE=full-model \
MERGED_OUTPUT=results/merges/multislerp-4task-full-model-equal-lambda1.0 \
bash scripts/run_merge_experiment.sh
```

### 1. 最小链路验证

命令使用 4 个 adapter、4 个 probe dataset，每类 2 条样本，3 个搜索 step：

```bash
/data8/zhangxin/.conda/envs/viet/bin/python tools/merge_self_positioning.py \
  --reference-adapter results/vibe-embedder-full/checkpoint-1000 \
  --adapter basic=results/vibe-emb-basic-from-ckpt1000/checkpoint-3000 \
  --adapter code=results/vibe-emb-code-from-ckpt1000/checkpoint-1328 \
  --adapter inst=results/vibe-emb-inst-from-ckpt1000/checkpoint-1500 \
  --adapter tool=results/vibe-emb-tool-from-ckpt1000/checkpoint-1631 \
  --dataset basic=data/train/embeddings-fine-tuning/msmarco.jsonl \
  --dataset code=data/train/codesearchnet/codesearchnet.jsonl \
  --dataset inst=data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl \
  --dataset tool=data/train/toolret-train/toolret.jsonl \
  --output-dir results/merges/self-positioning-4task-smoke-steps3 \
  --base-model data/raw/Qwen2.5-0.5B \
  --device cuda:0 \
  --dtype bf16 \
  --trust-remote-code \
  --search-steps 3 \
  --batch-size 2 \
  --examples-per-dataset 2 \
  --train-group-size 3 \
  --max-length 96 \
  --query-max-length 96 \
  --passage-max-length 96
```

结果证明搜索链路可运行，`a_i/lambda` 均有梯度更新，并成功保存 adapter。

### 2. 小规模 self-positioning 实验

实际探索命令：

```bash
/data8/zhangxin/.conda/envs/viet/bin/python tools/merge_self_positioning.py \
  --reference-adapter results/vibe-embedder-full/checkpoint-1000 \
  --adapter basic=results/vibe-emb-basic-from-ckpt1000/checkpoint-3000 \
  --adapter code=results/vibe-emb-code-from-ckpt1000/checkpoint-1328 \
  --adapter inst=results/vibe-emb-inst-from-ckpt1000/checkpoint-1500 \
  --adapter tool=results/vibe-emb-tool-from-ckpt1000/checkpoint-1631 \
  --dataset basic=data/train/embeddings-fine-tuning/msmarco.jsonl \
  --dataset code=data/train/codesearchnet/codesearchnet.jsonl \
  --dataset inst=data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl \
  --dataset tool=data/train/toolret-train/toolret.jsonl \
  --output-dir results/merges/self-positioning-4task-probe32-steps20-mu005 \
  --base-model data/raw/Qwen2.5-0.5B \
  --device cuda:0 \
  --dtype bf16 \
  --trust-remote-code \
  --search-steps 20 \
  --log-steps 5 \
  --batch-size 4 \
  --examples-per-dataset 8 \
  --max-source-lines 20000 \
  --train-group-size 4 \
  --max-length 160 \
  --query-max-length 160 \
  --passage-max-length 160 \
  --learning-rate 5e-3 \
  --mu 0.05
```

输出目录：

```text
results/merges/self-positioning-4task-probe32-steps20-mu005
```

搜索结果：

| 参数 | 值 |
|---|---:|
| final lambda | 0.9714875 |
| alpha basic | 1.0243485 |
| alpha code | 1.0056348 |
| alpha inst | 0.9874166 |
| alpha tool | 1.0002508 |
| last loss | 0.3228751 |
| last probe loss | 0.2742472 |

Task-vector 统计：

| 项 | 值 |
|---|---:|
| LoRA tensors | 336 |
| LoRA params | 17,596,416 |
| norm basic | 13.884598 |
| norm code | 9.207748 |
| norm inst | 11.475445 |
| norm tool | 9.425951 |
| merged norm | 10.293937 |

任务向量两两 cosine 都很小：

| pair | cosine |
|---|---:|
| basic-code | 0.016257 |
| basic-inst | 0.043079 |
| basic-tool | 0.013979 |
| code-inst | 0.015770 |
| code-tool | 0.013562 |
| inst-tool | 0.013381 |

这说明四个任务更新在 LoRA 参数空间大致接近正交，简单线性平均容易改变范数，SLERP 更适合作为首选融合方式。

## 本地 proxy eval

命令：

```bash
/data8/zhangxin/.conda/envs/viet/bin/python scripts/eval_adapter_retrieval_smoke.py \
  --checkpoint results/vibe-emb-basic-from-ckpt1000/checkpoint-3000 \
  --checkpoint results/vibe-emb-code-from-ckpt1000/checkpoint-1328 \
  --checkpoint results/vibe-emb-inst-from-ckpt1000/checkpoint-1500 \
  --checkpoint results/vibe-emb-tool-from-ckpt1000/checkpoint-1631 \
  --checkpoint results/merges/self-positioning-4task-probe32-steps20-mu005 \
  --dataset basic data/train/embeddings-fine-tuning/msmarco.jsonl \
  --dataset code data/train/codesearchnet/codesearchnet.jsonl \
  --dataset inst data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl \
  --dataset tool data/train/toolret-train/toolret.jsonl \
  --examples 8 \
  --max-negatives 3 \
  --device cuda:0 \
  --dtype bf16 \
  --batch-size 16 \
  --max-length 160 \
  --trust-remote-code \
  --output results/merges/smoke_eval_self_positioning_4task_probe32_steps20.json
```

四个 dataset 平均结果：

| checkpoint | mean hit@1 | mean MRR | mean margin |
|---|---:|---:|---:|
| basic parent | 0.9375 | 0.9688 | 0.1834 |
| code parent | 0.8750 | 0.9323 | 0.2180 |
| inst parent | 0.8750 | 0.9297 | 0.1855 |
| tool parent | 0.9375 | 0.9609 | 0.1951 |
| self-positioning merge | 0.9375 | 0.9609 | 0.2345 |

分任务 hit@1：

| checkpoint | basic | code | inst | tool |
|---|---:|---:|---:|---:|
| basic parent | 0.875 | 1.000 | 1.000 | 0.875 |
| code parent | 0.750 | 1.000 | 0.875 | 0.875 |
| inst parent | 0.625 | 1.000 | 1.000 | 0.875 |
| tool parent | 0.750 | 1.000 | 1.000 | 1.000 |
| self-positioning merge | 0.750 | 1.000 | 1.000 | 1.000 |

这个 proxy eval 只用于快速检查融合是否明显坏掉。样本只有每类 8 条，不能作为最终结论。当前小实验显示：merged adapter 保住了 code/inst/tool 小样本 hit@1，平均 margin 高于四个 parent，但 basic 小样本 hit@1 低于 basic parent。

## 技术要点和注意事项

1. 当前是 adapter-space merging，不是 dense model-space merging。它更便宜，和 PEFT 推理兼容，但严格性低于把每个 adapter merge 到 base 后对全量权重做 task vector。
2. SLERP 的夹角按全局 LoRA 向量计算，避免逐 tensor SLERP 导致每层方向尺度不一致。
3. 搜索过程中不保存中间模型。每步动态构造 merged adapter 参数，通过 `functional_call` 前向，梯度只更新 `raw_alphas/raw_lambda`。
4. Probe dataset 的 query 格式应与训练一致：如果 row 有 `prompt`，使用 `Instruct: {prompt}\nQuery: {query}`；否则使用默认检索 instruction。
5. `--max-source-lines` 只用于快速实验。正式实验建议对全量数据做 reservoir sampling，或者提前构造固定 probe jsonl，保证可复现。
6. `mu` 需要 sweep。论文试了 `0.00, 0.05, 0.10`，当前小实验只跑了 `0.05`。

## 建议的下一步正式实验

1. 构造固定 probe set：每类任务至少 512 到 2000 条，正式复现可接近论文的 32k。
2. 跑 1000 step self-positioning：

```bash
for mu in 0.00 0.05 0.10; do
  /data8/zhangxin/.conda/envs/viet/bin/python tools/merge_self_positioning.py \
    --reference-adapter results/vibe-embedder-full/checkpoint-1000 \
    --adapter basic=results/vibe-emb-basic-from-ckpt1000/checkpoint-3000 \
    --adapter code=results/vibe-emb-code-from-ckpt1000/checkpoint-1328 \
    --adapter inst=results/vibe-emb-inst-from-ckpt1000/checkpoint-1500 \
    --adapter tool=results/vibe-emb-tool-from-ckpt1000/checkpoint-1631 \
    --dataset basic=data/train/embeddings-fine-tuning/msmarco.jsonl \
    --dataset code=data/train/codesearchnet/codesearchnet.jsonl \
    --dataset inst=data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl \
    --dataset tool=data/train/toolret-train/toolret.jsonl \
    --output-dir "results/merges/self-positioning-4task-steps1000-mu${mu}" \
    --base-model data/raw/Qwen2.5-0.5B \
    --device cuda:0 \
    --dtype bf16 \
    --trust-remote-code \
    --search-steps 1000 \
    --batch-size 32 \
    --examples-per-dataset 8000 \
    --train-group-size 4 \
    --learning-rate 5e-3 \
    --mu "$mu"
done
```

3. 对每个 candidate 跑 MTEB 子集，至少覆盖：

```text
NanoMSMARCORetrieval
NanoNQRetrieval
AppsRetrieval
CodeSearchNetRetrieval
CosQA
Core17InstructionRetrieval
ToolRetRetrieval
```

4. 如果正式 MTEB 显示某个任务被牺牲，再尝试调整 probe 配比，而不是手调 alpha。self-positioning 的优势就是把任务权重选择留给 probe loss。
