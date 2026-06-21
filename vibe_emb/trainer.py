from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer

logger = logging.getLogger(__name__)


class EmbeddingTrainer(Trainer):
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
