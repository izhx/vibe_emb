# AGENTS.md

本项目是 Qwen2.5-0.5B 系列 embedding / retrieval 训练、评估和 LoRA adapter 合并实验仓库。根目录 `AGENTS.md` 是本文件的软链接。

## 重要目录

- `vibe_emb/`: 训练框架核心代码。
  - `train.py`: YAML 配置训练入口，负责加载 tokenizer/model/dataset/trainer，并保存 resolved config。
  - `arguments.py`: `model`、`data`、`training_extras` 配置 dataclass 定义。
  - `config.py`: YAML 读取、配置分段和 dataclass 构建。
  - `data.py`: 多数据集 batch plan、same-dataset batching、instruction 格式和采样逻辑。
  - `collator.py`: 将预构造 batch tokenize 成 query/passage features。
  - `modeling.py`: base model / PEFT LoRA 加载、embedding pooling、contrastive loss。
  - `trainer.py`: 自定义 `Trainer`，配合 embedding batch 和 loss 输出。
- `vibe_eval/`: MTEB 评估代码。
  - `modeling.py`: `QwenDecoderOnlyEmbedder`，加载 base model + PEFT checkpoint 并实现 MTEB encode 接口。
  - `run_mteb.py`: MTEB 评估入口，包含默认 retrieval task 列表。
  - `tasks/toolret.py`: 本地 ToolRet retrieval task。
- `scripts/`: 训练、评估、adapter 合并的 shell 入口和检查脚本。
  - `merge_lora.py`: 将单个 PEFT LoRA adapter merge 成完整模型权重。
  - `eval_adapter_retrieval_smoke.py`: 小样本 retrieval sanity eval。
  - `verify_dataset_batch_plan.py`: 检查分布式 batch plan 是否跨 rank 对齐。
  - `summarize_mteb_main_score.py`: 汇总 MTEB 结果主指标。
  - `run_merge_experiment.sh`、`run_self_positioning_merge.sh`、`run_eval.sh`: 常用实验入口。
- `tools/`: 独立工具脚本。
  - `merge_multi_slerp.py`: 多 adapter Multi-SLERP 合并工具，支持 reference adapter 的 task-vector 空间。
  - `merge_self_positioning.py`: 基于 probe dataset 搜索 adapter 合并权重并保存 merged adapter。
- `configs/`: 训练 YAML 示例和各任务 LoRA warm-start 配置。
- `multitask/`: 多任务数据、实验脚本和记录文档。
- `data/`、`results/`: 本地数据和实验产物，通常较大，不要无关改动。

## 重要文档

- `vibe_emb/embedding_training_framework.md`: 训练框架、数据采样、PEFT warm-start、保存和测试场景说明。
- `multitask/aidoc/data_proc.md`: 检索训练数据处理记录和复现命令。
- `multitask/aidoc/train.md`: 多任务训练配置、采样策略、batch 分析和 smoke run 记录。
- `multitask/aidoc/model_merging.md`: adapter SLERP / self-positioning merge 原理、脚本入口和实验结果。
- `README.md`: 环境依赖的简要记录。

## 常用命令

- 训练入口通常走 shell 包装脚本，例如 `bash scripts/train.sh` 或 `bash scripts/train_task.sh`；底层是 `python -m vibe_emb.train --config <yaml>`。
- MTEB 评估入口是 `python -m vibe_eval.run_mteb --checkpoint <adapter_dir>`，也可用 `bash scripts/run_eval.sh`。
- 四任务 Multi-SLERP merge 默认入口是 `bash scripts/run_merge_experiment.sh`。
- self-positioning merge 默认入口是 `bash scripts/run_self_positioning_merge.sh`。

## 维护注意事项

- PEFT adapter checkpoint 不是完整 `AutoModel` checkpoint。训练 warm-start 使用 `model.peft_adapter_name_or_path` 指向 adapter，`model.model_name_or_path` 仍指向 base model。
- `MultiDatasetBatchDataset` 返回的每个 item 已经是一个 rank-local contrastive batch；训练配置中 `per_device_train_batch_size` 必须保持为 `1`，`gradient_accumulation_steps` 也必须为 `1`。
- 分布式训练时各 rank 的 dataset/batch plan 必须一致，否则 cross-device negatives 可能挂住或 target offset 错位。
- 修改训练数据、采样或 collator 后，优先跑 `scripts/verify_dataset_batch_plan.py` 或小规模 smoke 训练。
- 修改评估或 adapter 合并代码后，优先用 smoke eval 或加载生成的 `adapter_model.safetensors` 做快速验证。
- 避免提交 `__pycache__/`、大体积 `data/`、`results/` 产物，除非明确需要保存某次实验记录。
