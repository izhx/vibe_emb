from __future__ import annotations

import logging
import os
import time
from collections import deque
from functools import partial
from math import ceil
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker

logger = logging.getLogger(__name__)


def create_train_dataloader(dataset, collator, args) -> DataLoader:  # noqa: ANN001
    """Create the ordered DataLoader for dataset-prebatched training items."""
    num_workers = args.dataloader_num_workers
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.per_device_train_batch_size,
        "sampler": SequentialSampler(dataset),
        "collate_fn": collator,
        "drop_last": args.dataloader_drop_last,
        "num_workers": num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "persistent_workers": False,
        "in_order": True,
    }
    if num_workers > 0:
        # Production TrainingArguments already contains the explicit resolved
        # factor. Keep the PyTorch default for direct callers that pass None.
        kwargs["prefetch_factor"] = (
            2 if args.dataloader_prefetch_factor is None else args.dataloader_prefetch_factor
        )
        # Workers are created after tokenizer/model/CUDA initialization. Spawn
        # avoids inheriting those threaded/native runtimes through os.fork().
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["worker_init_fn"] = partial(
            seed_worker,
            num_workers=num_workers,
            rank=args.process_index,
        )
    return DataLoader(**kwargs)


class DataLoaderWaitTracker:
    """Bounded p95 samples plus exact window and lifetime aggregates."""

    def __init__(self, recent_capacity: int = 4096, window_capacity: int = 4096) -> None:
        if recent_capacity <= 0 or window_capacity <= 0:
            raise ValueError("Wait sample capacities must be positive.")
        self._window: deque[tuple[float, int]] = deque(maxlen=window_capacity)
        self._recent: deque[tuple[float, int]] = deque(maxlen=recent_capacity)
        self._window_count = 0
        self._window_seconds = 0.0
        self._window_local_instances = 0
        self._total_count = 0
        self._total_seconds = 0.0
        self._total_local_instances = 0

    def record(self, seconds: float, local_instances: int) -> None:
        sample = (max(0.0, float(seconds)), max(0, int(local_instances)))
        self._window.append(sample)
        self._recent.append(sample)
        self._window_count += 1
        self._window_seconds += sample[0]
        self._window_local_instances += sample[1]
        self._total_count += 1
        self._total_seconds += sample[0]
        self._total_local_instances += sample[1]

    @staticmethod
    def _metrics(samples: deque[tuple[float, int]]) -> dict[str, float | int]:
        if not samples:
            return {
                "wait_count": 0,
                "mean_batch_wait_ms": 0.0,
                "p95_batch_wait_ms": 0.0,
                "local_instances": 0,
            }
        waits = sorted(seconds for seconds, _ in samples)
        p95_index = max(0, ceil(0.95 * len(waits)) - 1)
        return {
            "wait_count": len(waits),
            "mean_batch_wait_ms": 1000.0 * sum(waits) / len(waits),
            "p95_batch_wait_ms": 1000.0 * waits[p95_index],
            "local_instances": sum(count for _, count in samples),
        }

    def snapshot_window(self, reset: bool = False) -> dict[str, float | int]:
        metrics = self._metrics(self._window)
        metrics.update(
            {
                "wait_count": self._window_count,
                "mean_batch_wait_ms": (
                    1000.0 * self._window_seconds / self._window_count
                    if self._window_count
                    else 0.0
                ),
                "local_instances": self._window_local_instances,
                "p95_sample_count": len(self._window),
            }
        )
        if reset:
            self._window.clear()
            self._window_count = 0
            self._window_seconds = 0.0
            self._window_local_instances = 0
        return metrics

    def snapshot_total(self) -> dict[str, float | int]:
        recent = self._metrics(self._recent)
        recent.update(
            {
                "wait_count": self._total_count,
                "mean_batch_wait_ms": (
                    1000.0 * self._total_seconds / self._total_count if self._total_count else 0.0
                ),
                "local_instances": self._total_local_instances,
                "p95_sample_count": len(self._recent),
            }
        )
        return recent


class EmbeddingTrainer(Trainer):
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.dataloader_wait_tracker = DataLoaderWaitTracker()
        self.train_dataloader_info: Optional[dict[str, Any]] = None
        super().__init__(*args, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """Load adapter-only PEFT checkpoints saved by EmbeddingModel.save.

        Transformers' default loader only recognizes full-model checkpoint
        filenames. Our normal checkpoints intentionally contain
        adapter_model.safetensors instead, while optimizer/scheduler/RNG state
        is still managed by Trainer after this hook returns.
        """
        adapter_config = os.path.join(resume_from_checkpoint, "adapter_config.json")
        adapter_weights = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if not (os.path.isfile(adapter_config) and os.path.isfile(adapter_weights)):
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        target = model or self.model
        if not hasattr(target, "model"):
            raise TypeError(f"Expected EmbeddingModel wrapper while loading {resume_from_checkpoint}")
        peft_model = target.model
        try:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Resuming an adapter checkpoint requires peft and safetensors") from exc
        if not hasattr(peft_model, "peft_config"):
            raise ValueError(
                f"Checkpoint {resume_from_checkpoint} is a PEFT adapter, but the configured model is not PEFT"
            )
        state_dict = load_file(adapter_weights, device="cpu")
        load_result = set_peft_model_state_dict(peft_model, state_dict, adapter_name="default")
        missing = [key for key in getattr(load_result, "missing_keys", []) if "lora_" in key]
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing or unexpected:
            raise ValueError(
                f"Adapter checkpoint mismatch for {resume_from_checkpoint}: "
                f"missing_lora={missing[:5]}, unexpected={unexpected[:5]}"
            )
        logger.info("Loaded PEFT adapter checkpoint from %s", resume_from_checkpoint)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: ANN001
        del kwargs
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, model, inputs, num_items_in_batch=None):  # noqa: ANN001
        metadata = inputs.pop("_batch_metadata", None)
        if metadata is None:
            raise ValueError("Training batch is missing required _batch_metadata.")
        if not hasattr(self.train_dataset, "record_consumed"):
            raise TypeError("EmbeddingTrainer requires a train_dataset with record_consumed(metadata).")
        self.train_dataset.record_consumed(metadata)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def get_batch_samples(self, epoch_iterator, num_batches, device):  # noqa: ANN001
        started = time.monotonic()
        batch_samples, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        elapsed = time.monotonic() - started
        local_instances = sum(
            int(batch.get("_batch_metadata", {}).get("local_instances", 0))
            for batch in batch_samples
        )
        self.dataloader_wait_tracker.record(elapsed, local_instances)
        return batch_samples, num_items

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        dataloader = create_train_dataloader(self.train_dataset, self.data_collator, self.args)
        context = dataloader.multiprocessing_context
        self.train_dataloader_info = {
            "num_workers": dataloader.num_workers,
            "prefetch_factor": dataloader.prefetch_factor,
            "persistent_workers": dataloader.persistent_workers,
            "pin_memory": dataloader.pin_memory,
            "multiprocessing_context": context.get_start_method() if context is not None else None,
        }
        logger.info(
            "Creating sequential train DataLoader with %d planned batches: "
            "num_workers=%d prefetch_factor=%s "
            "persistent_workers=%s pin_memory=%s multiprocessing_context=%s",
            len(self.train_dataset),
            dataloader.num_workers,
            "null" if dataloader.prefetch_factor is None else dataloader.prefetch_factor,
            str(dataloader.persistent_workers).lower(),
            str(dataloader.pin_memory).lower(),
            self.train_dataloader_info["multiprocessing_context"] or "none",
        )
        return dataloader

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        del state_dict
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "save"):
            raise TypeError(f"Model {type(model).__name__} does not implement save(output_dir).")
        model.save(output_dir)
        if self.processing_class is not None and self.is_world_process_zero():
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
