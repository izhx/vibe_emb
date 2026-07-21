from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

InstructionConfig = Optional[Union[str, List[str]]]
RETRIEVAL_LOSS_MODES = frozenset({"legacy", "f2llm"})


@dataclass
class ModelConfig:
    """Base-model, tokenizer, pooling, and optional PEFT construction options."""
    model_name_or_path: str
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    tokenizer_padding_side: Optional[str] = None
    torch_dtype: Optional[str] = None
    pooling: str = "last_token"
    normalize_embeddings: bool = True
    peft_adapter_name_or_path: Optional[str] = None
    peft_config: Optional[Dict[str, Any]] = None
    save_merged_lora_model: bool = False
    gradient_checkpointing_enable_input_grads: bool = True


@dataclass
class DatasetConfig:
    """Overrides for one legacy JSON dataset.

    Indexed Arrow units obtain the equivalent runtime fields from their
    manifest descriptors and ``DataConfig.task_defaults``.
    """
    name: str
    path: str
    query_instruction: InstructionConfig = None
    query_instruction_format: Optional[str] = None
    passage_instruction: InstructionConfig = None
    passage_instruction_format: Optional[str] = None
    query_max_len: Optional[int] = None
    passage_max_len: Optional[int] = None
    train_group_size: Optional[int] = None
    batch_size: Optional[int] = None
    sample_size: int = -1
    sample_factor: float = 1.0
    no_in_batch_neg: Optional[bool] = None
    task_type: Optional[str] = None
    data_format: str = "auto"
    shuffle_text: bool = False
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset sources plus batching and Indexed Arrow runtime policy.

    ``task_defaults`` is the central mapping from the canonical F2LLM task
    type to group size and in-batch-negative behavior. Per-dataset values take
    precedence when explicitly configured.
    """
    datasets: List[DatasetConfig] = field(default_factory=list)
    indexed_dataset_manifest: Optional[str] = None
    default_query_max_len: int = 320
    default_passage_max_len: int = 512
    default_train_group_size: int = 8
    default_batch_size: int = 64
    default_query_instruction: InstructionConfig = None
    default_passage_instruction: InstructionConfig = None
    default_query_instruction_format: str = "Instruct: {}\nQuery: {}"
    default_passage_instruction_format: str = "{}{}"
    pad_to_multiple_of: Optional[int] = 8
    append_eos_token: bool = False
    same_dataset_within_batch: bool = True
    cache_dir: Optional[str] = None
    task_defaults: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "retrieval": {"train_group_size": 8, "no_in_batch_neg": False},
            "clustering": {"train_group_size": 10, "no_in_batch_neg": True},
            "classification": {"train_group_size": 2, "no_in_batch_neg": True},
        }
    )
    arrow_open_mode: str = "lazy"
    # The LRU limit is per training process, and one open unit owns both its
    # query and corpus mappings. Keeping it small bounds descriptors/mmaps when
    # a profile contains hundreds of units.
    arrow_max_open_units: int = 32
    arrow_prefetch_units: int = 0
    arrow_verify_mode: str = "manifest"
    arrow_local_cache_dir: Optional[str] = None
    # Consecutive batches from one unit improve mmap/page-cache locality. A
    # value of 1 preserves fully batch-level interleaving for legacy configs.
    unit_block_batches: int = 1


@dataclass
class EmbedTrainingExtras:
    """Embedding-specific training fields removed before TrainingArguments parsing."""
    temperature: float = 0.02
    negatives_cross_device: bool = True
    sub_batch_size: int = 0
    kd_loss_type: str = "kl_div"
    retrieval_loss_mode: str = "legacy"

    def __post_init__(self) -> None:
        if self.retrieval_loss_mode not in RETRIEVAL_LOSS_MODES:
            allowed = ", ".join(sorted(RETRIEVAL_LOSS_MODES))
            raise ValueError(
                f"training.retrieval_loss_mode must be one of {allowed}; "
                f"got {self.retrieval_loss_mode!r}."
            )
