from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.codefuse_models import f2llmv2_prompts_dict
from mteb.models.model_meta import ModelMeta
from mteb.models.model_meta import ScoringFunction
from mteb.types import BatchedInput, PromptType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)


DEFAULT_QUERY_INSTRUCTION = (
    "Given a query, retrieve passages that are relevant to the query."
)
DEFAULT_QUERY_INSTRUCTION_FORMAT = "Instruct: {instruction}\nQuery: "


class QwenDecoderOnlyEmbedder(AbsEncoder):
    """MTEB encoder wrapper for decoder-only embedding checkpoints."""

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        model_name_or_path: str | None = None,
        adapter_name_or_path: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        max_length: int = 512,
        max_batch_tokens: int | None = None,
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
        self.max_length = max_length
        if max_batch_tokens is not None and max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive or None.")
        self.max_batch_tokens = max_batch_tokens
        self._reported_token_split = False
        self.query_instruction = query_instruction
        self.use_task_prompts = use_task_prompts
        self.instruction_template = query_instruction_format
        self.prompts_dict = f2llmv2_prompts_dict
        self.apply_instruction_to_passages = False
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
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: Any | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: PromptType | str | None = None,
        show_progress_bar: bool | None = None,
        **_: Any,
    ) -> np.ndarray:
        instruction = self._instruction_for_task(task_metadata, prompt_type)
        if not isinstance(inputs, DataLoader):
            raise TypeError("MTEB encoders require a DataLoader[BatchedInput].")
        progress = tqdm(
            total=self._input_length(inputs),
            desc="Encoding",
            leave=False,
            mininterval=20,
            unit="texts",
            disable=not show_progress_bar,
        )

        embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in inputs:
                if not isinstance(batch, dict) or "text" not in batch:
                    raise TypeError(
                        "MTEB text encoders expect each DataLoader batch to be a "
                        "BatchedInput mapping containing a 'text' field."
                    )
                # texts = batch["text"]
                # if not isinstance(texts, (list, tuple)) or not all(
                #     isinstance(text, str) for text in texts
                # ):
                #     raise TypeError(
                #         "MTEB BatchedInput['text'] must be a sequence of strings."
                #     )
                batch_texts = list(batch["text"])
                if instruction is not None:
                    batch_texts = [instruction + text for text in batch_texts]
                features = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = features["input_ids"]
                batch_size = input_ids.size(0)
                sequence_length = input_ids.size(1)
                token_num = batch_size * sequence_length
                max_batch_tokens = self.max_batch_tokens
                if max_batch_tokens is None or token_num <= max_batch_tokens:
                    batch_embeddings = self._encode(features)
                else:
                    batch_embeddings = self._encode_split_batch(
                        features,
                        max_batch_tokens,
                    )

                embeddings.append(batch_embeddings)
                progress.update(len(batch_texts))
        progress.close()

        if not embeddings:
            return np.empty((0, self.mteb_model_meta.embed_dim or 0), dtype=np.float32)
        return torch.cat(embeddings, dim=0).float().numpy()

    def _encode(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        features = {key: value.to(self.device) for key, value in features.items()}
        # Embedding evaluation consumes only the current hidden states. Decoder KV
        # cache otherwise retains K/V tensors for every layer and grows linearly
        # with batch size and sequence length.
        outputs = self.model(
            **features,
            use_cache=False,
            return_dict=True,
        )
        reps = self._last_token_pool(
            outputs.last_hidden_state,
            features["attention_mask"],
        )
        if self.normalize_embeddings:
            reps = F.normalize(reps, dim=-1)
        return reps.cpu()

    def _encode_split_batch(
        self,
        features: dict[str, torch.Tensor],
        max_batch_tokens: int,
    ) -> torch.Tensor:
        input_ids = features["input_ids"]
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        token_num = batch_size * sequence_length
        micro_batch_size = max(max_batch_tokens // sequence_length, 1)
        num_forwards = (
            batch_size + micro_batch_size - 1
        ) // micro_batch_size
        if not self._reported_token_split:
            logger.warning(
                "Splitting an MTEB batch of %d texts (%d padded tokens) "
                "into %d forwards capped by max_batch_tokens=%d.",
                batch_size,
                token_num,
                num_forwards,
                max_batch_tokens,
            )
            self._reported_token_split = True

        split_embeddings: list[torch.Tensor] = []
        for start in range(0, batch_size, micro_batch_size):
            end = start + micro_batch_size
            micro_features = {
                key: value[start:end]
                for key, value in features.items()
            }
            micro_features = self._trim_batch_padding(micro_features)
            split_embeddings.append(self._encode(micro_features))
        return torch.cat(split_embeddings, dim=0)

    @staticmethod
    def _trim_batch_padding(
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        attention_mask = features["attention_mask"]
        active_columns = attention_mask.bool().any(dim=0).nonzero(as_tuple=False)
        if active_columns.numel() == 0:
            return features

        start = active_columns[0].item()
        end = active_columns[-1].item() + 1
        if start == 0 and end == attention_mask.size(1):
            return features

        batch_size, sequence_length = attention_mask.shape
        return {
            key: value[:, start:end]
            if (
                value.ndim >= 2
                and value.size(0) == batch_size
                and value.size(1) == sequence_length
            )
            else value
            for key, value in features.items()
        }

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

    def _instruction_for_task(
        self,
        task_metadata: Any | None,
        prompt_type: PromptType | str | None,
    ) -> str | None:
        if not self.use_task_prompts:
            if not self._is_query(prompt_type):
                return None
            return self.format_instruction(self.query_instruction, prompt_type)
        if task_metadata is None:
            if not self._is_query(prompt_type):
                return None
            return self.format_instruction(self.query_instruction, prompt_type)

        instruction = self.get_task_instruction(task_metadata, prompt_type)
        if (
            not self.apply_instruction_to_passages
            and prompt_type == PromptType.document
        ):
            return None
        return instruction or None

    @staticmethod
    def _is_query(prompt_type: PromptType | str | None) -> bool:
        if prompt_type is None:
            return False
        value = getattr(prompt_type, "value", prompt_type)
        return str(value).lower() == "query"

    @staticmethod
    def _input_length(inputs: Any) -> int | None:
        if isinstance(inputs, DataLoader) and hasattr(inputs, "dataset"):
            try:
                return len(inputs.dataset)
            except TypeError:
                return None
        try:
            return len(inputs)
        except TypeError:
            return None

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
