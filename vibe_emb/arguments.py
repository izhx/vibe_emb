from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

InstructionConfig = Optional[Union[str, List[str]]]


@dataclass
class ModelConfig:
    model_name_or_path: str
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    torch_dtype: Optional[str] = None
    pooling: str = "last_token"
    normalize_embeddings: bool = True
    peft_adapter_name_or_path: Optional[str] = None
    peft_config: Optional[Dict[str, Any]] = None
    save_merged_lora_model: bool = False
    gradient_checkpointing_enable_input_grads: bool = True


@dataclass
class DatasetConfig:
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
    no_in_batch_neg: bool = False
    shuffle_text: bool = False
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    datasets: List[DatasetConfig]
    default_query_max_len: int = 320
    default_passage_max_len: int = 512
    default_train_group_size: int = 8
    default_batch_size: int = 64
    default_query_instruction: InstructionConfig = None
    default_passage_instruction: InstructionConfig = None
    default_query_instruction_format: str = "Instruct: {}\nQuery: {}"
    default_passage_instruction_format: str = "{}{}"
    pad_to_multiple_of: Optional[int] = 8
    same_dataset_within_batch: bool = True
    cache_dir: Optional[str] = None


@dataclass
class EmbedTrainingExtras:
    temperature: float = 0.02
    negatives_cross_device: bool = True
    sub_batch_size: int = 0
    kd_loss_type: str = "kl_div"
