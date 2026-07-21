from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import transformers
import yaml
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .callbacks import DatasetRefreshCallback, DatasetStatsCallback, TrainingProgressCallback
from .collator import EmbeddingCollator
from .config import load_yaml_config, parse_sections, to_plain_dict
from .data import MultiDatasetBatchDataset
from .modeling import EmbeddingModel, build_base_model, maybe_apply_peft
from .trainer import EmbeddingTrainer

logger = logging.getLogger(__name__)


def _set_cuda_device_from_local_rank() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


class _ReadableYamlDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.SafeDumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_ReadableYamlDumper.add_representer(str, _str_presenter)


def _setup_logging(output_dir: str, process_index: int) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if process_index == 0:
        handlers.append(logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a", encoding="utf-8"))
    logging.basicConfig(
        format="%(asctime)s|%(name)s:%(lineno)d|%(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if process_index == 0 else logging.WARNING,
        handlers=handlers,
        force=True,
    )


def _apply_cli_overrides(raw: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    raw = dict(raw)
    raw.setdefault("model", {})
    raw.setdefault("training", {})
    if args.output_dir:
        raw["training"]["output_dir"] = args.output_dir
    if args.model_name_or_path:
        raw["model"]["model_name_or_path"] = args.model_name_or_path
    if args.max_steps is not None:
        raw["training"]["max_steps"] = args.max_steps
    if args.resume_from_checkpoint:
        raw["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.overwrite_output_dir:
        raw["training"]["overwrite_output_dir"] = True
    return raw


def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=_ReadableYamlDumper, allow_unicode=True, sort_keys=False)


def _build_training_args(training_raw: Dict[str, Any]) -> TrainingArguments:
    training_raw = dict(training_raw)
    overwrite_output_dir = bool(training_raw.pop("overwrite_output_dir", False))
    training_raw.setdefault("output_dir", "results/vibe-embedder")
    training_raw.setdefault("do_train", True)
    training_raw.setdefault("per_device_train_batch_size", 1)
    training_raw.setdefault("gradient_accumulation_steps", 1)
    training_raw.setdefault("dataloader_num_workers", 0)
    training_raw.setdefault("dataloader_persistent_workers", False)
    training_raw.setdefault("remove_unused_columns", False)
    training_raw.setdefault("report_to", "none")
    training_raw.setdefault("disable_tqdm", True)
    if training_raw["gradient_accumulation_steps"] != 1:
        raise ValueError("gradient_accumulation_steps must be 1 for contrastive embedding training.")
    if training_raw["per_device_train_batch_size"] != 1:
        raise ValueError("per_device_train_batch_size must be 1 because dataset items are pre-batched.")
    # Validate once at the configured training entrypoint, before
    # TrainingArguments initializes devices or any DataLoader is created.
    num_workers = training_raw["dataloader_num_workers"]
    factor = training_raw.get("dataloader_prefetch_factor")
    persistent = training_raw["dataloader_persistent_workers"]
    pin_memory = training_raw.get("dataloader_pin_memory", True)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("training.dataloader_num_workers must be a non-negative integer.")
    if factor is not None and (isinstance(factor, bool) or not isinstance(factor, int)):
        raise ValueError("training.dataloader_prefetch_factor must be null or a positive integer.")
    if not isinstance(persistent, bool):
        raise ValueError("training.dataloader_persistent_workers must be a boolean.")
    if not isinstance(pin_memory, bool):
        raise ValueError("training.dataloader_pin_memory must be a boolean.")
    if persistent:
        raise ValueError(
            "training.dataloader_persistent_workers=true is not supported because epoch batch plans "
            "are refreshed in the main-process dataset."
        )
    if num_workers == 0 and factor is not None:
        raise ValueError(
            "training.dataloader_prefetch_factor must be null when dataloader_num_workers=0."
        )
    if num_workers > 0 and factor is not None and factor <= 0:
        raise ValueError("training.dataloader_prefetch_factor must be a positive integer.")
    if num_workers > 0 and factor is None:
        training_raw["dataloader_prefetch_factor"] = 2
    args = TrainingArguments(**training_raw)
    setattr(args, "vibe_overwrite_output_dir", overwrite_output_dir)
    return args


def _resolved_training_config(training_raw: Dict[str, Any], args: TrainingArguments) -> Dict[str, Any]:
    resolved = dict(training_raw)
    resolved.update(
        {
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
            "dataloader_persistent_workers": args.dataloader_persistent_workers,
            "dataloader_pin_memory": args.dataloader_pin_memory,
        }
    )
    return resolved


class EmbeddingTrainRunner:
    """Construct and run one YAML-configured embedding training process."""
    def __init__(self, config_path: str, cli_args: argparse.Namespace) -> None:
        raw = _apply_cli_overrides(load_yaml_config(config_path), cli_args)
        self.raw_config = raw
        self.model_args, self.data_args, training_raw, self.training_extras = parse_sections(raw)
        self.training_args = _build_training_args(training_raw)
        _setup_logging(self.training_args.output_dir, self.training_args.process_index)
        transformers.utils.logging.set_verbosity(self.training_args.get_process_log_level())

        if self.training_args.process_index == 0:
            _save_yaml(os.path.join(self.training_args.output_dir, "training_config.yaml"), raw)
            _save_yaml(
                os.path.join(self.training_args.output_dir, "resolved_config.yaml"),
                {
                    "model": to_plain_dict(self.model_args),
                    "data": to_plain_dict(self.data_args),
                    "training": _resolved_training_config(training_raw, self.training_args),
                    "training_extras": to_plain_dict(self.training_extras),
                },
            )

        logger.warning(
            "Process rank: %s, world size: %s, local rank: %s, device: %s, n_gpu: %s, distributed: %s, bf16: %s, fp16: %s",
            self.training_args.process_index,
            self.training_args.world_size,
            self.training_args.local_rank,
            self.training_args.device,
            self.training_args.n_gpu,
            self.training_args.world_size > 1,
            self.training_args.bf16,
            self.training_args.fp16,
        )
        logger.info("Training parameters: %s", self.training_args)

        set_seed(self.training_args.seed)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.train_dataset = self._load_train_dataset()
        self.collator = EmbeddingCollator(
            self.tokenizer,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            append_eos_token=self.data_args.append_eos_token,
        )
        self.trainer = self._load_trainer()

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            trust_remote_code=self.model_args.trust_remote_code,
            use_fast=self.model_args.use_fast_tokenizer,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer has neither pad_token nor eos_token; please configure a pad token.")
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer pad_token was missing; using eos_token as pad_token.")
        if self.model_args.tokenizer_padding_side:
            if self.model_args.tokenizer_padding_side not in {"left", "right"}:
                raise ValueError(
                    "model.tokenizer_padding_side must be 'left' or 'right'; "
                    f"got {self.model_args.tokenizer_padding_side!r}."
                )
            tokenizer.padding_side = self.model_args.tokenizer_padding_side
            logger.info("Tokenizer padding_side set to %s.", tokenizer.padding_side)
        return tokenizer

    def _load_model(self):
        base_model = build_base_model(self.model_args, bf16=self.training_args.bf16, fp16=self.training_args.fp16)
        base_model = maybe_apply_peft(base_model, self.model_args)
        model = EmbeddingModel(base_model, self.model_args, self.training_extras)
        if self.training_args.gradient_checkpointing and self.model_args.gradient_checkpointing_enable_input_grads:
            model.enable_input_require_grads()
        return model

    def _load_train_dataset(self):
        return MultiDatasetBatchDataset(
            self.data_args,
            self.training_extras,
            seed=self.training_args.seed,
            process_index=self.training_args.process_index,
            world_size=self.training_args.world_size,
        )

    def _load_trainer(self):
        trainer = EmbeddingTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.collator,
            processing_class=self.tokenizer,
        )
        trainer.add_callback(DatasetRefreshCallback(self.train_dataset))
        self.dataset_stats_callback = DatasetStatsCallback(self.train_dataset, trainer)
        trainer.add_callback(self.dataset_stats_callback)
        trainer.add_callback(TrainingProgressCallback())
        return trainer

    def run(self) -> None:
        last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not getattr(self.training_args, "vibe_overwrite_output_dir", False)
        ):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                logger.info("Checkpoint detected; resuming from %s", last_checkpoint)

        checkpoint = self.training_args.resume_from_checkpoint or last_checkpoint
        try:
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            self.trainer.save_model()
            metrics = train_result.metrics
            metrics["train_batches"] = len(self.train_dataset)
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
        finally:
            # on_train_end normally writes these artifacts. Repeat finalize in
            # the runner's finally path so an exception still preserves the
            # completed wait/consumption observations.
            self.dataset_stats_callback.finalize(self.training_args, self.trainer.state)
            self.train_dataset.close()
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    args = parser.parse_args()
    _set_cuda_device_from_local_rank()
    EmbeddingTrainRunner(args.config, args).run()


if __name__ == "__main__":
    main()
