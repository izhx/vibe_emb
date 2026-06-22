#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover - depends on runtime env
    raise RuntimeError("This script requires `peft` to be installed.") from exc


TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    path: Path


def parse_named_path(raw: str) -> AdapterSpec:
    if "=" in raw:
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid adapter spec: {raw!r}. Expected NAME=PATH.")
        return AdapterSpec(name=name, path=Path(path))

    path = Path(raw)
    name = path.name or path.parent.name
    if not name:
        raise ValueError(f"Invalid adapter path: {raw!r}")
    return AdapterSpec(name=name, path=path)


def adapter_weights_path(adapter_dir: Path) -> Path:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return safetensors_path
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.is_file():
        return bin_path
    raise FileNotFoundError(f"Missing adapter_model.safetensors or adapter_model.bin in {adapter_dir}")


def load_adapter(adapter_dir: Path) -> dict[str, torch.Tensor]:
    weights = adapter_weights_path(adapter_dir)
    if weights.suffix == ".safetensors":
        return load_file(str(weights), device="cpu")
    return torch.load(weights, map_location="cpu")


def read_base_model_from_adapter(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    config = read_json(config_path)
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"`base_model_name_or_path` is missing in {config_path}")
    return str(base_model)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def torch_dtype_from_name(name: str | None) -> torch.dtype | str | None:
    if name is None:
        return None
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def validate_compatible(reference: dict[str, torch.Tensor] | None, adapters: list[dict[str, torch.Tensor]]) -> None:
    if not adapters:
        raise ValueError("At least one adapter is required.")

    first_keys = set(adapters[0])
    for idx, state_dict in enumerate(adapters[1:], start=2):
        keys = set(state_dict)
        if keys != first_keys:
            missing = sorted(first_keys - keys)[:10]
            extra = sorted(keys - first_keys)[:10]
            raise ValueError(f"Adapter {idx} has incompatible keys. missing={missing}, extra={extra}")

    if reference is not None and set(reference) != first_keys:
        missing = sorted(first_keys - set(reference))[:10]
        extra = sorted(set(reference) - first_keys)[:10]
        raise ValueError(f"Reference adapter has incompatible keys. missing={missing}, extra={extra}")

    for key in sorted(first_keys):
        shape = adapters[0][key].shape
        for idx, state_dict in enumerate(adapters[1:], start=2):
            if state_dict[key].shape != shape:
                raise ValueError(
                    f"Tensor shape mismatch for {key}: adapter 1 has {tuple(shape)}, "
                    f"adapter {idx} has {tuple(state_dict[key].shape)}"
                )
        if reference is not None and reference[key].shape != shape:
            raise ValueError(
                f"Tensor shape mismatch for {key}: adapter 1 has {tuple(shape)}, "
                f"reference has {tuple(reference[key].shape)}"
            )


def parse_weights(raw_weights: list[str] | None, adapter_specs: list[AdapterSpec]) -> list[float]:
    if not raw_weights:
        return [1.0 for _ in adapter_specs]

    if all("=" in raw for raw in raw_weights):
        by_name: dict[str, float] = {}
        for raw in raw_weights:
            name, value = raw.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Invalid weight spec: {raw!r}. Expected NAME=FLOAT.")
            by_name[name] = float(value)
        missing = [spec.name for spec in adapter_specs if spec.name not in by_name]
        extra = sorted(set(by_name) - {spec.name for spec in adapter_specs})
        if missing or extra:
            raise ValueError(f"Weight names must match adapters. missing={missing}, extra={extra}")
        weights = [by_name[spec.name] for spec in adapter_specs]
    elif any("=" in raw for raw in raw_weights):
        raise ValueError("Use either all positional weights or all NAME=FLOAT weights.")
    else:
        if len(raw_weights) != len(adapter_specs):
            raise ValueError(
                f"Expected {len(adapter_specs)} positional weights, got {len(raw_weights)}."
            )
        weights = [float(raw) for raw in raw_weights]

    for weight in weights:
        if not math.isfinite(weight):
            raise ValueError(f"Weights must be finite, got {weight!r}.")
    return weights


def normalized_weights(weights: list[float], *, normalize: bool, eps: float) -> torch.Tensor:
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if normalize:
        weight_sum = weight_tensor.sum()
        if torch.abs(weight_sum) < eps:
            raise ValueError("Cannot normalize weights because their sum is zero.")
        weight_tensor = weight_tensor / weight_sum
    return weight_tensor


def weighted_linear(tensors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (tensors * weights.view(-1, *([1] * (tensors.ndim - 1)))).sum(0)


# https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/multislerp.py
def multislerp(
    tensors: list[torch.Tensor],
    weights: torch.Tensor,
    *,
    base_tensor: torch.Tensor | None = None,
    eps: float = 1e-8,
    degenerate_fallback: str = "linear",
) -> torch.Tensor:
    """Barycentric interpolation on a hypersphere for one adapter tensor."""
    if len(tensors) == 1:
        return tensors[0]

    output_dtype = base_tensor.dtype if base_tensor is not None else tensors[0].dtype
    stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
    if base_tensor is not None:
        stacked = stacked - base_tensor.float()

    flat = stacked.reshape(stacked.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    unit_tensors = flat / (norms + eps)

    mean = (unit_tensors * weights.view(-1, 1)).sum(0)
    mean_norm = torch.linalg.vector_norm(mean)
    if mean_norm < eps:
        if degenerate_fallback == "linear":
            result = weighted_linear(stacked, weights)
            if base_tensor is not None:
                result = result + base_tensor.float()
            return result.to(output_dtype)
        raise ValueError(
            "The weighted sum of the input tensors is zero. This can happen with "
            "antipodal task vectors and exactly balanced weights. Use different "
            "weights or --degenerate-fallback linear."
        )
    mean = mean / mean_norm

    dots = (unit_tensors * mean).sum(-1, keepdim=True)
    tangent_vectors = unit_tensors - dots * mean
    tangent_result = (tangent_vectors * weights.view(-1, 1)).sum(0)

    tangent_norm = torch.linalg.vector_norm(tangent_result) + eps
    result = mean * torch.cos(tangent_norm) + tangent_result * (
        torch.sin(tangent_norm) / tangent_norm
    )

    avg_norm = (norms.squeeze(-1) * weights).sum()
    result = (result * avg_norm).reshape(stacked.shape[1:])
    if base_tensor is not None:
        result = result + base_tensor.float()
    return result.to(output_dtype)


def merge_tensor_dicts(
    states: list[dict[str, torch.Tensor]],
    weights: torch.Tensor,
    *,
    reference: dict[str, torch.Tensor] | None,
    lambda_scale: float,
    eps: float,
    degenerate_fallback: str,
) -> dict[str, torch.Tensor]:
    validate_compatible(reference, states)
    merged: dict[str, torch.Tensor] = {}
    for key in sorted(states[0]):
        base_tensor = reference[key] if reference is not None else None
        if base_tensor is not None and all(torch.equal(state[key], base_tensor) for state in states):
            value = base_tensor
        elif base_tensor is None and all(torch.equal(state[key], states[0][key]) for state in states[1:]):
            value = states[0][key]
        else:
            value = multislerp(
                [state[key] for state in states],
                weights,
                base_tensor=base_tensor,
                eps=eps,
                degenerate_fallback=degenerate_fallback,
            )
        if reference is not None and lambda_scale != 1.0:
            value = reference[key].float() + lambda_scale * (value.float() - reference[key].float())
            value = value.to(reference[key].dtype)
        merged[key] = value
    return merged


def load_auto_model(
    model_name_or_path: str,
    *,
    cache_dir: str | None,
    torch_dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
):
    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype_from_name(torch_dtype),
    }
    if device_map:
        kwargs["device_map"] = device_map
    return AutoModel.from_pretrained(model_name_or_path, **kwargs)


def load_tokenizer(tokenizer_source: Path | None, base_model: str, *, cache_dir: str | None, trust_remote_code: bool):
    candidates = []
    if tokenizer_source is not None:
        candidates.append(str(tokenizer_source))
    candidates.append(base_model)
    errors: list[str] = []
    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(
                candidate,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Failed to load tokenizer from: " + " | ".join(errors))


def maybe_resize_token_embeddings(model, tokenizer) -> None:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and embeddings.weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


LORA_A_SUFFIXES = (".lora_A.weight", ".lora_A.default.weight")
LORA_B_SUFFIXES = (".lora_B.weight", ".lora_B.default.weight")


def split_lora_key(key: str) -> tuple[str, str] | None:
    for suffix in LORA_A_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "A"
    for suffix in LORA_B_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "B"
    return None


def collect_lora_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    ignored: list[str] = []
    for key, value in state.items():
        parsed = split_lora_key(key)
        if parsed is None:
            ignored.append(key)
            continue
        prefix, side = parsed
        pairs.setdefault(prefix, {})[side] = value
    incomplete = sorted(prefix for prefix, pair in pairs.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise ValueError(f"Incomplete LoRA A/B pairs for prefixes: {incomplete[:10]}")
    if not pairs:
        raise ValueError(f"No LoRA A/B tensors found. Ignored keys: {ignored[:10]}")
    return pairs


def strip_lora_base_prefix(prefix: str) -> str:
    for marker in ("base_model.model.", "base_model."):
        if prefix.startswith(marker):
            return prefix[len(marker) :]
    return prefix


def candidate_base_weight_keys(prefix: str) -> list[str]:
    stripped = strip_lora_base_prefix(prefix)
    candidates = [
        f"{stripped}.weight",
        f"model.{stripped}.weight",
        f"{prefix}.weight",
    ]
    if stripped.startswith("model."):
        candidates.insert(0, f"{stripped.removeprefix('model.')}.weight")
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_base_weight_key(prefix: str, model_state_keys: set[str]) -> str:
    for candidate in candidate_base_weight_keys(prefix):
        if candidate in model_state_keys:
            return candidate
    raise KeyError(
        f"Cannot map LoRA prefix {prefix!r} to a base model weight key. "
        f"Tried: {candidate_base_weight_keys(prefix)}"
    )


def pattern_value(pattern: dict[str, Any], module_name: str, default: Any) -> Any:
    if not pattern:
        return default
    if module_name in pattern:
        return pattern[module_name]
    for key, value in pattern.items():
        if module_name.endswith(f".{key}") or module_name.endswith(key):
            return value
    return default


def lora_scaling(prefix: str, config: dict[str, Any], rank: int) -> float:
    module_name = strip_lora_base_prefix(prefix)
    if module_name.startswith("model."):
        module_name = module_name[len("model.") :]
    rank_value = int(pattern_value(config.get("rank_pattern") or {}, module_name, config.get("r", rank)))
    alpha = float(pattern_value(config.get("alpha_pattern") or {}, module_name, config.get("lora_alpha", rank_value)))
    if config.get("use_rslora"):
        return alpha / math.sqrt(rank_value)
    return alpha / rank_value


def dense_lora_delta_state(
    adapter_dir: Path,
    *,
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    config = read_json(adapter_dir / "adapter_config.json")
    if config.get("use_dora"):
        raise ValueError("delta-w merge does not support DoRA adapters.")
    state = load_adapter(adapter_dir)
    pairs = collect_lora_pairs(state)
    model_state_keys = set(model_state)
    deltas: dict[str, torch.Tensor] = {}
    for prefix, pair in sorted(pairs.items()):
        a = pair["A"].float()
        b = pair["B"].float()
        key = resolve_base_weight_key(prefix, model_state_keys)
        scale = lora_scaling(prefix, config, rank=a.shape[0])
        delta = (b @ a) * scale
        if config.get("fan_in_fan_out"):
            delta = delta.T
        if tuple(delta.shape) != tuple(model_state[key].shape):
            raise ValueError(
                f"Dense delta shape mismatch for {prefix}: got {tuple(delta.shape)}, "
                f"base weight {key} has {tuple(model_state[key].shape)}"
            )
        deltas[key] = delta
    return deltas


def merged_full_model_state(
    base_model: str,
    adapter_dir: Path,
    *,
    tokenizer_source: Path | None,
    cache_dir: str | None,
    torch_dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
) -> dict[str, torch.Tensor]:
    model = load_auto_model(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    tokenizer = load_tokenizer(tokenizer_source or adapter_dir, base_model, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    maybe_resize_token_embeddings(model, tokenizer)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    merged = peft_model.merge_and_unload()
    state = {key: value.detach().cpu() for key, value in merged.state_dict().items()}
    del peft_model, merged, model
    gc.collect()
    return state


def apply_deltas_to_model(model, deltas: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    with torch.no_grad():
        for key, delta in deltas.items():
            if key not in model_state:
                raise KeyError(f"Merged delta key {key!r} is missing from the base model state.")
            model_state[key].add_(delta.to(device=model_state[key].device, dtype=model_state[key].dtype))


def save_full_model(
    model,
    output_dir: Path,
    *,
    tokenizer_source: Path | None,
    base_model: str,
    cache_dir: str | None,
    trust_remote_code: bool,
    safe_serialization: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
    tokenizer = load_tokenizer(tokenizer_source, base_model, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))


def save_full_model_state(
    state: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    base_model: str,
    tokenizer_source: Path | None,
    cache_dir: str | None,
    torch_dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
    safe_serialization: bool,
) -> None:
    model = load_auto_model(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    input_embeddings = model.get_input_embeddings()
    embed_weight = state.get("embed_tokens.weight")
    if embed_weight is None:
        embed_weight = state.get("model.embed_tokens.weight")
    if input_embeddings is not None and embed_weight is not None and input_embeddings.weight.shape[0] != embed_weight.shape[0]:
        model.resize_token_embeddings(embed_weight.shape[0])
    model.load_state_dict(state, strict=True)
    save_full_model(
        model,
        output_dir,
        tokenizer_source=tokenizer_source,
        base_model=base_model,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        safe_serialization=safe_serialization,
    )
    del model
    gc.collect()


def merge_adapters(
    adapters: list[dict[str, torch.Tensor]],
    weights: torch.Tensor,
    *,
    reference: dict[str, torch.Tensor] | None,
    lambda_scale: float,
    eps: float,
    degenerate_fallback: str,
) -> dict[str, torch.Tensor]:
    return merge_tensor_dicts(
        adapters,
        weights,
        reference=reference,
        lambda_scale=lambda_scale,
        eps=eps,
        degenerate_fallback=degenerate_fallback,
    )


def copy_metadata(source_dir: Path, output_dir: Path, *, base_config_source: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_source = base_config_source or source_dir
    for filename in ["adapter_config.json", "README.md"]:
        src = config_source / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)
    for filename in TOKENIZER_FILES:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)


def global_dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in sorted(left):
        total += float(torch.sum(left[key].float() * right[key].float()))
    return total


def global_norm(value: dict[str, torch.Tensor]) -> float:
    return math.sqrt(max(global_dot(value, value), 0.0))


def tensor_stats(
    reference: dict[str, torch.Tensor] | None,
    adapter_specs: list[AdapterSpec],
    adapters: list[dict[str, torch.Tensor]],
    merged: dict[str, torch.Tensor],
) -> dict[str, Any]:
    keys = sorted(merged)
    out: dict[str, Any] = {
        "num_tensors": len(keys),
        "num_parameters": sum(merged[key].numel() for key in keys),
    }

    if reference is None:
        adapter_norms = {spec.name: global_norm(adapter) for spec, adapter in zip(adapter_specs, adapters)}
        out.update(
            {
                "adapter_norms": adapter_norms,
                "merged_norm": global_norm(merged),
            }
        )
        return out

    task_vectors = [
        {key: adapter[key].float() - reference[key].float() for key in keys}
        for adapter in adapters
    ]
    task_norms = {
        spec.name: global_norm(task_vector)
        for spec, task_vector in zip(adapter_specs, task_vectors)
    }
    cosines: dict[str, float | None] = {}
    for (left_idx, left), (right_idx, right) in itertools.combinations(enumerate(task_vectors), 2):
        denom = global_norm(left) * global_norm(right)
        key = f"{adapter_specs[left_idx].name}__{adapter_specs[right_idx].name}"
        cosines[key] = global_dot(left, right) / denom if denom else None

    merged_delta = {key: merged[key].float() - reference[key].float() for key in keys}
    out.update(
        {
            "task_vector_norms": task_norms,
            "task_vector_cosines": cosines,
            "merged_task_vector_norm": global_norm(merged_delta),
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two or more PEFT LoRA adapters with Multi-SLERP."
    )
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter in NAME=PATH form. Repeat for each model to include.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        help=(
            "Relative adapter weight. Repeat in adapter order, or use NAME=FLOAT for "
            "named weights. Defaults to 1.0 for every adapter."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--merge-space",
        choices=["adapter", "delta-w", "full-model"],
        default="adapter",
        help=(
            "Where to run Multi-SLERP: `adapter` merges LoRA A/B tensors and saves a PEFT adapter; "
            "`delta-w` merges dense LoRA delta-W tensors and saves a full HF model; "
            "`full-model` first merges each LoRA into the base model, then merges full model weights."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base HF model path/name for full-model outputs. Defaults to the first adapter config.",
    )
    parser.add_argument(
        "--reference-adapter",
        type=Path,
        default=None,
        help="Optional origin adapter for task-vector Multi-SLERP.",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1.0,
        help="Optional scale for the merged task vector. 1.0 matches standard Multi-SLERP.",
    )
    parser.add_argument(
        "--normalize-weights",
        dest="normalize_weights",
        action="store_true",
        default=True,
        help="Normalize adapter weights to sum to 1 before Multi-SLERP.",
    )
    parser.add_argument(
        "--no-normalize-weights",
        dest="normalize_weights",
        action="store_false",
        help="Use weights as provided.",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Small constant for numerical stability.")
    parser.add_argument(
        "--degenerate-fallback",
        choices=["linear", "error"],
        default="linear",
        help="Behavior when Multi-SLERP has a zero weighted direction for a tensor.",
    )
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=None,
        help="Directory to copy/load tokenizer files from. Defaults to the first adapter.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--torch-dtype",
        default="bf16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Dtype used when loading/saving full-model outputs.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='Optional Transformers device_map for full-model paths, e.g. "auto". Default loads normally.',
    )
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code.")
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Save full models as PyTorch .bin weights instead of safetensors.",
    )
    return parser.parse_args()


def common_metadata(
    args: argparse.Namespace,
    adapter_specs: list[AdapterSpec],
    raw_weights: list[float],
    weights: torch.Tensor,
    reference_dir: Path | None,
    tokenizer_source: Path,
    *,
    base_model: str | None,
) -> dict[str, Any]:
    return {
        "method": "multislerp",
        "merge_space": args.merge_space,
        "output_type": "adapter" if args.merge_space == "adapter" else "full_model",
        "adapters": {spec.name: str(spec.path) for spec in adapter_specs},
        "weights": {spec.name: weight for spec, weight in zip(adapter_specs, raw_weights)},
        "effective_weights": {
            spec.name: float(weight.detach().cpu())
            for spec, weight in zip(adapter_specs, weights)
        },
        "normalize_weights": args.normalize_weights,
        "eps": args.eps,
        "lambda_scale": args.lambda_scale if reference_dir is not None else None,
        "reference_adapter": str(reference_dir) if reference_dir else None,
        "tokenizer_source": str(tokenizer_source),
        "base_model": base_model,
        "torch_dtype": args.torch_dtype if args.merge_space != "adapter" else None,
        "degenerate_fallback": args.degenerate_fallback,
        "safe_serialization": not args.no_safe_serialization if args.merge_space != "adapter" else None,
    }


def main() -> None:
    args = parse_args()
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")
    if not math.isfinite(args.lambda_scale):
        raise ValueError("--lambda-scale must be finite.")

    adapter_specs = [parse_named_path(raw) for raw in args.adapter]
    if len(adapter_specs) < 2:
        raise ValueError("At least two --adapter arguments are required for Multi-SLERP.")
    if len({spec.name for spec in adapter_specs}) != len(adapter_specs):
        raise ValueError("Adapter names must be unique.")

    adapter_specs = [
        AdapterSpec(spec.name, spec.path.expanduser().resolve())
        for spec in adapter_specs
    ]
    raw_weights = parse_weights(args.weight, adapter_specs)
    weights = normalized_weights(raw_weights, normalize=args.normalize_weights, eps=args.eps)

    output_dir = args.output_dir.expanduser().resolve()
    reference_dir = args.reference_adapter.expanduser().resolve() if args.reference_adapter else None
    tokenizer_source = (args.tokenizer_source or adapter_specs[0].path).expanduser().resolve()

    if args.merge_space == "adapter":
        adapters = [load_adapter(spec.path) for spec in adapter_specs]
        reference = load_adapter(reference_dir) if reference_dir else None
        validate_compatible(reference, adapters)
        merged = merge_adapters(
            adapters,
            weights,
            reference=reference,
            lambda_scale=args.lambda_scale,
            eps=args.eps,
            degenerate_fallback=args.degenerate_fallback,
        )
        copy_metadata(tokenizer_source, output_dir, base_config_source=adapter_specs[0].path)
        save_file(merged, str(output_dir / "adapter_model.safetensors"))
        metadata = common_metadata(
            args,
            adapter_specs,
            raw_weights,
            weights,
            reference_dir,
            tokenizer_source,
            base_model=None,
        )
        metadata["mode"] = "multislerp_adapter_task_vector" if reference is not None else "multislerp_adapter_direct"
        metadata["stats"] = tensor_stats(reference, adapter_specs, adapters, merged)
        write_json(output_dir / "merge_config.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    base_model = args.base_model or read_base_model_from_adapter(adapter_specs[0].path)
    trust_remote_code = not args.no_trust_remote_code

    if args.merge_space == "delta-w":
        model = load_auto_model(
            base_model,
            cache_dir=args.cache_dir,
            torch_dtype=args.torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=args.device_map,
        )
        tokenizer = load_tokenizer(tokenizer_source, base_model, cache_dir=args.cache_dir, trust_remote_code=trust_remote_code)
        maybe_resize_token_embeddings(model, tokenizer)
        model_state = model.state_dict()
        delta_states = [dense_lora_delta_state(spec.path, model_state=model_state) for spec in adapter_specs]
        reference_delta = dense_lora_delta_state(reference_dir, model_state=model_state) if reference_dir else None
        merged_delta = merge_tensor_dicts(
            delta_states,
            weights,
            reference=reference_delta,
            lambda_scale=args.lambda_scale,
            eps=args.eps,
            degenerate_fallback=args.degenerate_fallback,
        )
        apply_deltas_to_model(model, merged_delta)
        save_full_model(
            model,
            output_dir,
            tokenizer_source=tokenizer_source,
            base_model=base_model,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
            safe_serialization=not args.no_safe_serialization,
        )
        metadata = common_metadata(
            args,
            adapter_specs,
            raw_weights,
            weights,
            reference_dir,
            tokenizer_source,
            base_model=base_model,
        )
        metadata["mode"] = "multislerp_delta_w_task_vector" if reference_delta is not None else "multislerp_delta_w_direct"
        metadata["stats"] = tensor_stats(reference_delta, adapter_specs, delta_states, merged_delta)
        write_json(output_dir / "merge_config.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    full_states = [
        merged_full_model_state(
            base_model,
            spec.path,
            tokenizer_source=tokenizer_source,
            cache_dir=args.cache_dir,
            torch_dtype=args.torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=args.device_map,
        )
        for spec in adapter_specs
    ]
    reference_full = (
        merged_full_model_state(
            base_model,
            reference_dir,
            tokenizer_source=tokenizer_source,
            cache_dir=args.cache_dir,
            torch_dtype=args.torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=args.device_map,
        )
        if reference_dir
        else None
    )
    merged_state = merge_tensor_dicts(
        full_states,
        weights,
        reference=reference_full,
        lambda_scale=args.lambda_scale,
        eps=args.eps,
        degenerate_fallback=args.degenerate_fallback,
    )
    save_full_model_state(
        merged_state,
        output_dir,
        base_model=base_model,
        tokenizer_source=tokenizer_source,
        cache_dir=args.cache_dir,
        torch_dtype=args.torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=args.device_map,
        safe_serialization=not args.no_safe_serialization,
    )
    metadata = common_metadata(
        args,
        adapter_specs,
        raw_weights,
        weights,
        reference_dir,
        tokenizer_source,
        base_model=base_model,
    )
    metadata["mode"] = "multislerp_full_model_task_vector" if reference_full is not None else "multislerp_full_model_direct"
    metadata["stats"] = tensor_stats(reference_full, adapter_specs, full_states, merged_state)
    write_json(output_dir / "merge_config.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
