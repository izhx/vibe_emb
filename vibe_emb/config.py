from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, Type, TypeVar

import yaml

from .arguments import DataConfig, DatasetConfig, EmbedTrainingExtras, ModelConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _known_dataclass_kwargs(cls: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _build_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    return cls(**_known_dataclass_kwargs(cls, data))


def _warn_unknown_keys(cls: Type[T], data: Dict[str, Any], section: str) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        warnings.warn(f"Ignoring unknown config keys in {section}: {', '.join(unknown)}", stacklevel=2)
        logger.warning("Ignoring unknown config keys in %s: %s", section, ", ".join(unknown))


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML config must be a mapping: {path}")
    return loaded


def parse_sections(raw: Dict[str, Any]) -> tuple[ModelConfig, DataConfig, Dict[str, Any], EmbedTrainingExtras]:
    raw = copy.deepcopy(raw)
    model_raw = raw.get("model") or {}
    data_raw = raw.get("data") or {}
    training_raw = raw.get("training") or {}

    if "model_name_or_path" not in model_raw:
        raise ValueError("Missing required config field: model.model_name_or_path")
    if not data_raw.get("datasets"):
        raise ValueError("Missing required config field: data.datasets")

    model_args = _build_dataclass(ModelConfig, model_raw)

    for i, dataset_raw in enumerate(data_raw["datasets"]):
        _warn_unknown_keys(DatasetConfig, dataset_raw, f"data.datasets[{i}]")
    dataset_args = [_build_dataclass(DatasetConfig, d) for d in data_raw["datasets"]]
    data_raw["datasets"] = dataset_args
    data_args = _build_dataclass(DataConfig, data_raw)

    extras = _build_dataclass(EmbedTrainingExtras, training_raw)
    training_keys = {f.name for f in fields(EmbedTrainingExtras)}
    training_args = {k: v for k, v in training_raw.items() if k not in training_keys}
    return model_args, data_args, training_args, extras


def to_plain_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_plain_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    return obj
