# MTEB instruction 解析机制

本文记录 MTEB instruction 的来源、解析优先级和 passage 特殊处理。

本文基于当前环境中的 MTEB 2.14.9：

- `mteb.models.abs_encoder.AbsEncoder`
- `mteb.models.instruct_wrapper.InstructSentenceTransformerModel`

## `InstructSentenceTransformerModel.encode()` 的处理流程

`encode()` 首先调用：

```python
instruction = self.get_task_instruction(task_metadata, prompt_type)
```

`get_task_instruction()` 分为两步：

1. `get_instruction()` 从不同来源中选择 instruction 文本。
2. 如果配置了 `instruction_template`，使用 `format_instruction()` 将文本格式化为模型输入前缀。

一种常见的 instruction 模板为：

```text
Instruct: {instruction}
Query: 
```

最终输入是格式化后的 instruction 前缀与原始文本直接拼接：

```text
Instruct: Retrieve relevant documents.
Query: <原始文本>
```

## Instruction 来源与优先级

`AbsEncoder.get_instruction()` 的实际优先级从高到低如下：

1. `prompts_dict[task_metadata.name]`
   - 精确任务名对应的专用 instruction。
   - 例如 `prompts_dict["ExampleRetrievalTask"]`。
2. `prompts_dict[task_metadata.type]`
   - 任务类型对应的通用 instruction。
   - 例如 `prompts_dict["Classification"]` 或 `prompts_dict["Clustering"]`。
3. `prompts_dict[prompt_type.value]`
   - 输入角色对应的通用 instruction。
   - 例如 `prompts_dict["query"]` 或 `prompts_dict["document"]`。
   - 仅在 `prompt_type` 非空时参与匹配。
4. `task_metadata.prompt`
   - 任务元数据自身定义的 prompt。
   - 字符串直接使用；字典则根据 `prompt_type.value` 选择 `query` 或 `document`。
5. `AbsTask.abstask_prompt`
   - 如果上述来源均为空，MTEB 根据任务名加载任务类，使用任务类的默认 `abstask_prompt`。

简化表示：

```text
任务名专用 prompt
  > 任务类型 prompt
  > query/document 通用 prompt
  > TaskMetadata.prompt
  > AbsTask.abstask_prompt
```

这里记录的是 MTEB 2.14.9 中 `get_instruction()` 的实际代码路径。不要仅根据方法 docstring 推断优先级；升级 MTEB 后应重新检查安装版本的实现。

## `prompts_dict` 与 `model_prompts` 的区别

两个属性名称相似，但用途不同：

### `prompts_dict`

- 由 MTEB wrapper 显式传入。
- 保存任务名、任务类型和 query/document 对应的 instruction 文本。
- `InstructSentenceTransformerModel.encode()` 经由 `get_task_instruction()` 实际使用该属性。

### `model_prompts`

- 表示模型自身携带的 SentenceTransformers prompt 配置。
- 主要由 `get_prompt_name()` 和 `get_prompt()` 使用。
- 当前 `InstructSentenceTransformerModel.encode()` 的 instruction 解析路径不会直接使用它替代 `prompts_dict`。

因此，在复现某个 instruct embedding 模型的 MTEB 结果时，应检查该模型注册信息实际传入的 `prompts_dict`，不能只依赖模型目录中的 SentenceTransformers prompts。

## Passage/document 的特殊处理

`InstructSentenceTransformerModel.encode()` 在选出 instruction 后还会检查：

```python
if (
    not self.apply_instruction_to_passages
    and prompt_type == PromptType.document
):
    instruction = None
```

当模型配置 `apply_instruction_to_passages=False` 时：

- query 使用任务专用 instruction；
- document/passage 不添加 instruction；
- 分类、聚类等 `prompt_type=None` 的任务仍可使用任务名或任务类型 instruction。

最后一点很重要：不能把“不是 query”简单等价为“不使用 instruction”。否则分类和聚类任务可能漏掉预期的 instruction。

## 示例

### 精确任务名优先

假设 `prompts_dict` 中包含某个任务的精确配置：

```python
prompts_dict["ExampleRetrievalTask"] = "Retrieve relevant documents."
```

该配置优先于任务类型或 query/document 通用 instruction。

### 回退到任务类型

如果某个分类任务没有精确任务名配置，但字典中包含 `Classification`，则使用：

```python
prompts_dict["Classification"]
```

### 回退到 query

如果任务名和任务类型均未命中，且 `prompt_type == PromptType.query`，则继续尝试：

```python
prompts_dict["query"]
```

### Document 强制无 instruction

即使 instruction 解析阶段命中了 `prompts_dict["document"]`，当 `apply_instruction_to_passages=False` 时，最终 document 输入仍不添加 instruction。

## 升级 MTEB 时的检查项

升级 MTEB 后至少重新确认：

1. `AbsEncoder.get_instruction()` 的来源优先级是否变化。
2. `get_task_instruction()` 的模板格式化行为是否变化。
3. `InstructSentenceTransformerModel.encode()` 是否仍在 document 阶段清空 instruction。
4. 目标模型 `prompts_dict` 的任务名和任务类型配置是否变化。
5. MTEB 传入 wrapper 的 `prompt_type` 是否仍区分 `query`、`document` 和 `None`。
