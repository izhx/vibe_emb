from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer

logger = logging.getLogger(__name__)


class EmbeddingTrainer(Trainer):
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

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # MultiDatasetBatchDataset already returns a complete contrastive batch.
        # Use a plain SequentialSampler and batch_size=1 so Trainer does not
        # reshuffle or stack pre-batched items and break the distributed plan.
        logger.info("Creating sequential train DataLoader with %d planned batches", len(self.train_dataset))
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

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
