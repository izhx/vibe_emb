from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class EmbeddingCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8
    append_eos_token: bool = False

    def _tokenize(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        if not self.append_eos_token:
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("append_eos_token=true requires tokenizer.eos_token_id to be set.")
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length - 1,
            padding=False,
        )
        for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            input_ids.append(eos_token_id)
            attention_mask.append(1)
        return self.tokenizer.pad(
            tokenized,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError("EmbeddingCollator expects dataset-prebatched features and Trainer batch_size=1.")
        feature = features[0]

        queries = self._tokenize(feature["queries"], feature["query_max_len"])
        passages = self._tokenize(feature["passages"], feature["passage_max_len"])

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
