from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from mteb.models.model_meta import ModelMeta
from mteb.models.model_meta import ScoringFunction
from mteb.types import PromptType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_QUERY_INSTRUCTION = (
    "Given a query, retrieve passages that are relevant to the query."
)
DEFAULT_QUERY_INSTRUCTION_FORMAT = "Instruct: {}\nQuery: {}"


class QwenDecoderOnlyEmbedder:
    """MTEB encoder wrapper for decoder-only embedding checkpoints."""

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        model_name_or_path: str | None = None,
        adapter_name_or_path: str | None = None,
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
        model_path, adapter_path = self._resolve_model_paths(
            checkpoint=checkpoint,
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
        )
        self.checkpoint = checkpoint or model_path
        self.model_name_or_path = model_path
        self.adapter_name_or_path = adapter_path
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

        tokenizer_sources = (
            [adapter_path, model_path] if adapter_path is not None else [model_path]
        )
        self.tokenizer = self._load_tokenizer(
            tokenizer_sources,
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

        base = AutoModel.from_pretrained(model_path, **model_kwargs)
        if base.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer))
        self.model = self._maybe_load_adapter(base, adapter_path)
        self.model.to(self.device)
        self.model.eval()

        display_path = adapter_path or model_path
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=self._display_name(display_path),
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

    @classmethod
    def _resolve_model_paths(
        cls,
        *,
        checkpoint: str | None,
        model_name_or_path: str | None,
        adapter_name_or_path: str | None,
    ) -> tuple[str, str | None]:
        model_path = model_name_or_path or checkpoint
        if adapter_name_or_path is not None:
            if not model_path:
                raise ValueError(
                    "model_name_or_path is required when adapter_name_or_path is set."
                )
            cls._validate_full_model_path(model_path)
            return model_path, adapter_name_or_path

        if model_path is not None:
            cls._validate_full_model_path(model_path)
            return model_path, None

        raise ValueError("model_name_or_path or checkpoint is required.")

    @staticmethod
    def _looks_like_adapter_path(path: str) -> bool:
        path_obj = Path(path)
        return (
            path_obj.is_dir()
            and (path_obj / "adapter_config.json").is_file()
            and not (path_obj / "config.json").is_file()
        )

    @classmethod
    def _validate_full_model_path(cls, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.is_dir():
            return
        if cls._looks_like_adapter_path(path):
            raise ValueError(
                f"{path} looks like a PEFT adapter-only checkpoint. "
                "Evaluate a full model directory, or pass it as adapter_name_or_path "
                "with model_name_or_path set to the base model."
            )

    @staticmethod
    def _load_tokenizer(
        sources: Iterable[str | None],
        *,
        trust_remote_code: bool,
    ):
        errors: list[Exception] = []
        tried: list[str] = []
        for source in sources:
            if source is None:
                continue
            tried.append(source)
            try:
                return AutoTokenizer.from_pretrained(
                    source,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:  # pragma: no cover - depends on local checkpoint contents
                errors.append(exc)
        message = f"Failed to load tokenizer from any of: {', '.join(tried)}"
        if errors:
            raise RuntimeError(message) from errors[-1]
        raise RuntimeError(message)

    @staticmethod
    def _maybe_load_adapter(model: Any, adapter_name_or_path: str | None):
        if adapter_name_or_path is None:
            return model
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("adapter_name_or_path requires `peft` to be installed.") from exc
        model = PeftModel.from_pretrained(
            model,
            adapter_name_or_path,
            is_trainable=False,
        )
        if not hasattr(model, "merge_and_unload"):
            raise RuntimeError(
                f"PEFT adapter {adapter_name_or_path} does not support merge_and_unload for evaluation."
            )
        return model.merge_and_unload()

    @staticmethod
    def _display_name(path: str) -> str:
        parts = Path(path).parts
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return path

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
