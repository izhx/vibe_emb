from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class EmbeddingCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError("EmbeddingCollator expects dataset-prebatched features and Trainer batch_size=1.")
        feature = features[0]

        queries = self.tokenizer(
            feature["queries"],
            truncation=True,
            max_length=feature["query_max_len"],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            feature["passages"],
            truncation=True,
            max_length=feature["passage_max_len"],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        teacher_scores = feature.get("teacher_scores")
        if teacher_scores is not None:
            teacher_scores = torch.tensor(teacher_scores, dtype=torch.float32)

        return {
            "queries": queries,
            "passages": passages,
            "teacher_scores": teacher_scores,
            "dataset_name": feature["dataset_name"],
            "no_in_batch_neg": feature["no_in_batch_neg"],
            "loss_kwargs": feature.get("loss_kwargs") or {},
            "model_kwargs": feature.get("model_kwargs") or {},
            "sub_batch_size": feature.get("sub_batch_size", 0),
            "train_group_size": feature.get("train_group_size"),
        }
