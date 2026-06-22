#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover - depends on runtime env
    raise RuntimeError("This script requires `peft` to be installed.") from exc


logger = logging.getLogger(__name__)


def _torch_dtype(name: Optional[str]):
    if name is None or name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --torch-dtype: {name}")


def _read_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"`base_model_name_or_path` is missing in {config_path}")
    return str(base_model)


def _load_tokenizer(adapter_dir: Path, base_model: str, cache_dir: Optional[str], trust_remote_code: bool):
    for model_path in (str(adapter_dir), base_model):
        try:
            return AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except Exception as exc:
            logger.info("Failed to load tokenizer from %s: %s", model_path, exc)
    raise RuntimeError(f"Failed to load tokenizer from adapter dir or base model: {adapter_dir}, {base_model}")


def merge_one(
    adapter_dir: Path,
    output_dir: Optional[Path],
    base_model: Optional[str],
    cache_dir: Optional[str],
    torch_dtype: Optional[str],
    device_map: Optional[str],
    trust_remote_code: bool,
    safe_serialization: bool,
) -> Path:
    adapter_dir = adapter_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise NotADirectoryError(f"Adapter checkpoint is not a directory: {adapter_dir}")

    base_model = base_model or _read_base_model(adapter_dir)
    output_dir = (output_dir or adapter_dir / "merged").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model from %s", base_model)
    model = AutoModel.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=_torch_dtype(torch_dtype),
        device_map=device_map,
    )

    logger.info("Loading LoRA adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)

    logger.info("Merging LoRA weights")
    merged = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s", output_dir)
    merged.save_pretrained(str(output_dir), safe_serialization=safe_serialization)

    tokenizer = _load_tokenizer(adapter_dir, base_model, cache_dir, trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapter checkpoints into full model weights.")
    parser.add_argument(
        "checkpoint",
        nargs="+",
        type=Path,
        help="One or more PEFT adapter checkpoint directories, e.g. results/vibe-embedder-full/checkpoint-2000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Merged model output directory. Only valid with one checkpoint. Default: <checkpoint>/merged.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override base model path/name. Default: adapter_config.json base_model_name_or_path.",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    parser.add_argument(
        "--torch-dtype",
        default="bf16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Dtype used when loading the base model. Default: auto.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='Optional Transformers device_map, e.g. "auto". Default loads normally.',
    )
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code.")
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Save PyTorch .bin weights instead of safetensors.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(format="%(asctime)s|%(levelname)s|%(message)s", level=logging.INFO)
    args = parse_args()
    if args.output_dir is not None and len(args.checkpoint) != 1:
        raise ValueError("--output-dir can only be used with a single checkpoint.")

    for checkpoint in args.checkpoint:
        merged_dir = merge_one(
            adapter_dir=checkpoint,
            output_dir=args.output_dir,
            base_model=args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=not args.no_trust_remote_code,
            safe_serialization=not args.no_safe_serialization,
        )
        logger.info("Done: %s", merged_dir)


if __name__ == "__main__":
    main()
