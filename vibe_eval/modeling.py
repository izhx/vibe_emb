from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from mteb.models.model_meta import ModelMeta
from mteb.models.model_meta import ScoringFunction
from mteb.types import PromptType
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_QUERY_INSTRUCTION = (
    "Given a query, retrieve passages that are relevant to the query."
)
DEFAULT_QUERY_INSTRUCTION_FORMAT = "Instruct: {}\nQuery: {}"


class QwenDecoderOnlyEmbedder:
    """MTEB encoder wrapper for FlagEmbedding decoder-only LoRA checkpoints."""

    def __init__(
        self,
        base_model: str,
        checkpoint: str,
        *,
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        query_max_length: int | None = None,
        corpus_max_length: int | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        query_instruction_format: str = DEFAULT_QUERY_INSTRUCTION_FORMAT,
        use_task_prompts: bool = True,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        self.base_model = base_model
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_max_length = query_max_length or max_length
        self.corpus_max_length = corpus_max_length or max_length
        self.query_instruction = query_instruction
        self.query_instruction_format = query_instruction_format
        self.use_task_prompts = use_task_prompts
        self.normalize_embeddings = normalize_embeddings
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = self._resolve_dtype(dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        base = AutoModel.from_pretrained(base_model, **model_kwargs)
        if base.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(base, checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.mteb_model_meta = ModelMeta(
            loader=None,
            name='/'.join(Path(checkpoint).parts[-2:]),
            revision=None,
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=max_length,
            embed_dim=getattr(base.config, "hidden_size", None),
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=True,
            training_datasets=None,
        )

    def encode(
        self,
        inputs: DataLoader | Iterable[str] | list[str],
        *,
        task_metadata: Any | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: PromptType | str | None = None,
        batch_size: int | None = None,
        show_progress_bar: bool | None = None,
        **_: Any,
    ) -> np.ndarray:
        texts = self._collect_texts(inputs)
        is_query = self._is_query(prompt_type)
        if is_query:
            instruction = self._query_instruction_for_task(task_metadata)
            texts = [
                self.query_instruction_format.format(instruction, text)
                for text in texts
            ]

        effective_batch_size = batch_size or self.batch_size
        max_length = self.query_max_length if is_query else self.corpus_max_length
        batches = range(0, len(texts), effective_batch_size)
        if show_progress_bar:
            batches = tqdm(batches, desc="Encoding", leave=False, mininterval=20)

        embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in batches:
                batch_texts = texts[start : start + effective_batch_size]
                features = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                features = {k: v.to(self.device) for k, v in features.items()}
                outputs = self.model(**features, return_dict=True)
                reps = self._last_token_pool(
                    outputs.last_hidden_state,
                    features["attention_mask"],
                )
                if self.normalize_embeddings:
                    reps = F.normalize(reps, dim=-1)
                embeddings.append(reps.cpu())

        if not embeddings:
            return np.empty((0, self.mteb_model_meta.embed_dim or 0), dtype=np.float32)
        return torch.cat(embeddings, dim=0).float().numpy()

    def encode_queries(self, queries: Iterable[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, prompt_type=PromptType.query, **kwargs)

    def encode_corpus(self, corpus: Iterable[str | dict[str, Any]], **kwargs: Any) -> np.ndarray:
        texts = [self._text_from_item(item) for item in corpus]
        return self.encode(texts, prompt_type=PromptType.document, **kwargs)

    def similarity(self, embeddings1: Any, embeddings2: Any) -> torch.Tensor:
        return torch.as_tensor(embeddings1) @ torch.as_tensor(embeddings2).T

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> torch.Tensor:
        return (torch.as_tensor(embeddings1) * torch.as_tensor(embeddings2)).sum(dim=-1)

    def _resolve_dtype(self, dtype: str) -> torch.dtype | None:
        if dtype == "auto":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if self.device.type == "cuda":
                return torch.float16
            return None
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    def _query_instruction_for_task(self, task_metadata: Any | None) -> str:
        if not self.use_task_prompts or task_metadata is None:
            return self.query_instruction
        prompt = getattr(task_metadata, "prompt", None)
        if isinstance(prompt, dict):
            return prompt.get("query") or self.query_instruction
        return self.query_instruction

    @staticmethod
    def _is_query(prompt_type: PromptType | str | None) -> bool:
        if prompt_type is None:
            return False
        value = getattr(prompt_type, "value", prompt_type)
        return str(value).lower() == "query"

    def _collect_texts(self, inputs: DataLoader | Iterable[str] | list[str]) -> list[str]:
        if isinstance(inputs, DataLoader):
            texts: list[str] = []
            for batch in inputs:
                if isinstance(batch, dict):
                    texts.extend([self._text_from_item(x) for x in batch["text"]])
                else:
                    texts.extend([self._text_from_item(x) for x in batch])
            return texts
        return [self._text_from_item(item) for item in inputs]

    @staticmethod
    def _text_from_item(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            text = item.get("text", "")
            title = item.get("title", "")
            if title:
                return f"{title} {text}".strip()
            return str(text)
        return str(item)

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]
