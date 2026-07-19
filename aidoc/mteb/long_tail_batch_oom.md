# MTEB 编码阶段 Decoder KV Cache 导致的 OOM

## 现象

使用 Qwen3 0.6B 基座模型和 F2LLM LoRA adapter，以 `--batch_size 128`、
`--max_length 2048` 严格离线评测 `MTEB(Code, v1)` 时，在编码
`AppsRetrieval` 语料库的过程中发生 OOM。OOM 信息显示：

- PyTorch 已分配 60.15 GiB；
- 已保留但尚未分配 16.69 GiB；
- GPU 仅剩 1.68 GiB；
- Qwen3 RMSNorm 申请 2.00 GiB 内存失败。

当时没有其他进程占用 GPU。错误发生在模型 forward 阶段，而不是 adapter
加载、相似度检索或结果缓存阶段。

## 主要根因：Embedding 编码时启用了生成缓存

adapter 使用的基座模型是 `/mnt/share/models/Qwen3-0.6B`，其配置中
`use_cache=true`。此前 `QwenDecoderOnlyEmbedder` 调用 decoder 时没有覆盖该
配置，因此 Transformers 创建了 `DynamicCache`。虽然 embedding 评测不会复用
历史 key/value，模型仍然保留了全部 28 层的 key/value 张量。

对于该模型，OOM 输入形状下单层 FP16 KV cache 的大小为：

```text
2（K 和 V）* 128 batch * 8 KV heads * 2048 tokens * 128 head dim * 2 bytes
= 每层 1 GiB
```

28 层共保留 28 GiB cache。其他 activation 和显存分配器碎片随后耗尽了 80 GiB
显存。申请 2 GiB 失败也与堆栈中的形状吻合：query projection 的形状为
`[128, 2048, 16, 128]`，FP16 下占 1 GiB；Qwen3 RMSNorm 使用 FP32 计算方差，
同样形状的 FP32 张量正好占 2 GiB。

这也解释了为什么此前同等参数规模的模型能够正常评测。之前评测使用的
`/mnt/share/models/F2LLM-v2-0.6B/config.json` 明确设置了 `use_cache=false`，
而此次 adapter 对应的基座模型开启了该配置。

### 直接显存测试

使用实际合并后的 adapter、FlashAttention 2、2,048 tokens 和 batch size 16
进行测试，结果如下：

- `use_cache=true`：产生 28 层 cache，共占 3.5 GiB；每层结束时的已分配显存
  从 1.392 GiB 增长到 4.83 GiB，峰值为 5.455 GiB；
- `use_cache=false`：没有 KV cache；每层结束时的已分配显存稳定在约
  1.33 GiB，峰值为 1.959 GiB。

当 batch size 增加到 128 时，实测的 3.5 GiB cache 按比例增长为前面计算的
28 GiB。

## 次要诱因：长尾样本造成的 Padding 放大

MTEB 的 `batch_size` 在 tokenization 之前限制样本数量。`AppsRetrieval`
语料的 token 长度中位数是 112，p95 是 495，但 8,765 篇文档中有 17 篇达到
2,048 tokens。实际观察到的一个 128 样本 batch 只包含 27,352 个真实 token，
却因为其中一个 2,048-token 文档而被 padding 到 262,144 tokens。

padding 放大让不必要的 KV cache 达到了最坏形状，是此次 OOM 的直接诱因，
但不是主要显存问题。对于 0.6B embedding 模型，在使用 FlashAttention 且关闭
KV cache 后，80 GiB H100 可以处理这个 padded shape。

## 修复方案

单卡和多卡 wrapper 现在都会在每次 embedding forward 时显式传入
`use_cache=False`。这是 embedding 评测的固定约束：下游只使用当前 hidden
states，不会消费生成缓存，因此没有必要保留 KV cache。

wrapper 还提供了一个默认关闭的 padded-token 显存保护机制：

1. 对一个 MTEB DataLoader batch 只进行一次 tokenize 和 padding；
2. 用 `batch_size * sequence_length` 计算实际模型输入的 padded token 数；
3. 未启用保护机制或者当前 batch 未超过上限时，只调用一次 `_encode`；
4. 超过上限时，沿 batch 维度切分 padded tensor；每个 micro-batch 会先删除其中
   整列都是 padding 的区间，再进行编码，最后按原始顺序拼接 embedding。

`vibe_eval.run_mteb` 通过 `--max_batch_tokens` 暴露该配置，默认值为 `None`，
即默认不拆分。`--batch_size` 仍然表示 MTEB 的样本数量上限。启用保护机制后，
只要单条 padded 序列本身不超过上限，每次 forward 的 padded token 数就不会
超过 `max_batch_tokens`。关闭 decoder cache 仍然是本次问题的根本修复。

## 验证

token budget 回归测试验证了以下行为：micro-batch 拆分不会改变输出顺序；每个
micro-batch 会去除多余 padding；每次模型 forward 都使用 `use_cache=False`。

随后在关闭 token micro-batching（`max_batch_tokens=None`）的情况下，重新运行
此前失败的真实 `AppsRetrieval` 语料切片。wrapper 实际执行了一个
`[128, 2048]` 模型输入，按原始顺序返回 128 x 1,024 embedding，并确认模型输出
中不存在 `past_key_values`。CUDA 峰值显存为 7.677 GiB allocated、
10.812 GiB reserved。

完整的 `AppsRetrieval` 严格离线评测也已经在
`--batch_size 128 --max_length 2048` 下通过。
