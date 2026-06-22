#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.func import functional_call
from transformers import AutoModel, AutoTokenizer

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover - depends on runtime env
    raise RuntimeError("This script requires `peft` to be installed.") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

LOGGER = logging.getLogger("self_positioning_merge")


@dataclass(frozen=True)
class NamedPath:
    name: str
    path: Path


@dataclass(frozen=True)
class ProbeExample:
    dataset: str
    query: str
    positives: list[str]
    negatives: list[str]
    prompt: str | None = None


def adapter_weights_path(adapter_dir: Path) -> Path:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return safetensors_path
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.is_file():
        return bin_path
    raise FileNotFoundError(f"Missing adapter_model.safetensors or adapter_model.bin in {adapter_dir}")


def load_adapter(adapter_dir: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    weights = adapter_weights_path(adapter_dir)
    if weights.suffix == ".safetensors":
        state = load_file(str(weights), device=str(device))
    else:
        state = torch.load(weights, map_location=device)
    return {key: value.float() for key, value in state.items()}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_named_path(raw: str) -> NamedPath:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=PATH, got: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty name in NAME=PATH argument: {raw}")
    return NamedPath(name=name, path=Path(path).expanduser().resolve())


def group_named_paths(specs: list[NamedPath]) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    for spec in specs:
        grouped.setdefault(spec.name, []).append(spec.path)
    return list(grouped.items())


def validate_compatible(reference: dict[str, torch.Tensor], adapters: list[dict[str, torch.Tensor]]) -> None:
    expected = set(reference)
    for idx, state in enumerate(adapters, start=1):
        keys = set(state)
        if keys != expected:
            missing = sorted(expected - keys)[:10]
            extra = sorted(keys - expected)[:10]
            raise ValueError(f"Adapter {idx} has incompatible keys. missing={missing}, extra={extra}")
    for key in sorted(expected):
        shape = reference[key].shape
        for idx, state in enumerate(adapters, start=1):
            if state[key].shape != shape:
                raise ValueError(
                    f"Tensor shape mismatch for {key}: reference has {tuple(shape)}, "
                    f"adapter {idx} has {tuple(state[key].shape)}"
                )


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("Softplus-initialized values must be positive.")
    return math.log(math.expm1(value))


def adapter_key_to_param_name(key: str) -> str:
    if not key.endswith(".weight"):
        raise ValueError(f"Unexpected adapter tensor key: {key}")
    return f"{key[:-len('.weight')]}.default.weight"


def copy_metadata(source_dir: Path, output_dir: Path, *, config_source: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["adapter_config.json", "README.md"]:
        src = config_source / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)
    for filename in TOKENIZER_FILES:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)


def global_dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> torch.Tensor:
    dots = [torch.sum(left[key] * right[key]) for key in sorted(left)]
    return torch.stack(dots).sum()


def global_norm(value: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(global_dot(value, value).clamp_min(1e-30))


def linear_combine(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    left_coeff: torch.Tensor,
    right_coeff: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {key: left_coeff * left[key] + right_coeff * right[key] for key in sorted(left)}


def global_slerp(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    t: torch.Tensor,
    *,
    dot_threshold: float,
) -> dict[str, torch.Tensor]:
    left_norm = global_norm(left)
    right_norm = global_norm(right)
    eps = torch.finfo(torch.float32).eps
    if bool((left_norm <= eps).detach().cpu()) or bool((right_norm <= eps).detach().cpu()):
        return linear_combine(left, right, 1.0 - t, t)

    cosine = (global_dot(left, right) / (left_norm * right_norm)).clamp(-1.0, 1.0)
    if bool((torch.abs(cosine) > dot_threshold).detach().cpu()):
        return linear_combine(left, right, 1.0 - t, t)

    theta = torch.acos(cosine)
    sin_theta = torch.sin(theta)
    if bool((torch.abs(sin_theta) <= eps).detach().cpu()):
        return linear_combine(left, right, 1.0 - t, t)

    left_coeff = torch.sin((1.0 - t) * theta) / sin_theta
    right_coeff = torch.sin(t * theta) / sin_theta
    return linear_combine(left, right, left_coeff, right_coeff)


def merge_task_vectors(
    reference: dict[str, torch.Tensor],
    task_vectors: list[dict[str, torch.Tensor]],
    alphas: torch.Tensor,
    lambda_scale: torch.Tensor,
    *,
    dot_threshold: float,
    previous_weight: str,
) -> dict[str, torch.Tensor]:
    if len(task_vectors) != len(alphas):
        raise ValueError("Number of task vectors and alphas must match.")
    merged = task_vectors[0]
    for idx in range(1, len(task_vectors)):
        if previous_weight == "mean":
            left_weight = alphas[:idx].mean()
        elif previous_weight == "sum":
            left_weight = alphas[:idx].sum()
        else:
            raise ValueError(f"Unsupported previous_weight: {previous_weight}")
        right_weight = alphas[idx]
        t = right_weight / (left_weight + right_weight).clamp_min(1e-12)
        merged = global_slerp(merged, task_vectors[idx], t, dot_threshold=dot_threshold)
    return {key: reference[key] + lambda_scale * merged[key] for key in sorted(reference)}


def tensor_stats(
    reference: dict[str, torch.Tensor],
    task_names: list[str],
    adapters: list[dict[str, torch.Tensor]],
    merged: dict[str, torch.Tensor],
) -> dict[str, Any]:
    task_vectors = [{key: state[key] - reference[key] for key in sorted(reference)} for state in adapters]
    task_norms = {name: float(global_norm(vec).detach().cpu()) for name, vec in zip(task_names, task_vectors)}
    cosines: dict[str, float | None] = {}
    for i, left in enumerate(task_vectors):
        for j, right in enumerate(task_vectors[i + 1 :], start=i + 1):
            denom = global_norm(left) * global_norm(right)
            value = None
            if float(denom.detach().cpu()) > 0:
                value = float((global_dot(left, right) / denom).detach().cpu())
            cosines[f"{task_names[i]}__{task_names[j]}"] = value
    merged_delta = {key: merged[key] - reference[key] for key in sorted(reference)}
    return {
        "num_tensors": len(reference),
        "num_parameters": sum(reference[key].numel() for key in reference),
        "task_vector_norms": task_norms,
        "task_vector_cosines": cosines,
        "merged_task_vector_norm": float(global_norm(merged_delta).detach().cpu()),
    }


def read_probe_dataset_group(
    name: str,
    paths: list[Path],
    *,
    limit: int,
    seed: int,
    max_source_lines: int,
) -> list[ProbeExample]:
    examples: list[ProbeExample] = []
    seen = 0
    rng = random.Random(seed)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if max_source_lines > 0 and line_number > max_source_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                positives = row.get("pos") or []
                negatives = row.get("neg") or []
                if not positives or not negatives:
                    continue
                prompt = row.get("prompt")
                example = ProbeExample(
                    dataset=name,
                    query=str(row["query"]),
                    positives=[str(item) for item in positives],
                    negatives=[str(item) for item in negatives],
                    prompt=str(prompt) if isinstance(prompt, str) and prompt.strip() else None,
                )
                seen += 1
                if len(examples) < limit:
                    examples.append(example)
                    continue
                replacement = rng.randrange(seen)
                if replacement < limit:
                    examples[replacement] = example
    if not examples:
        raise ValueError(f"No usable probe examples loaded for dataset {name}: {paths}")
    rng.shuffle(examples)
    return examples


def build_probe_examples(
    datasets: list[NamedPath],
    *,
    examples_per_dataset: int,
    seed: int,
    max_source_lines: int,
) -> list[ProbeExample]:
    all_examples: list[ProbeExample] = []
    for idx, (name, paths) in enumerate(group_named_paths(datasets)):
        examples = read_probe_dataset_group(
            name,
            paths,
            limit=examples_per_dataset,
            seed=seed + idx,
            max_source_lines=max_source_lines,
        )
        LOGGER.info(
            "Loaded %d probe examples for %s from %d jsonl file(s): %s",
            len(examples),
            name,
            len(paths),
            ", ".join(str(path) for path in paths),
        )
        all_examples.extend(examples)
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    return all_examples


def batched_cycle(examples: list[ProbeExample], *, batch_size: int, seed: int) -> Iterable[list[ProbeExample]]:
    rng = random.Random(seed)
    pool = list(examples)
    while True:
        rng.shuffle(pool)
        for start in range(0, len(pool), batch_size):
            batch = pool[start : start + batch_size]
            if len(batch) == batch_size:
                yield batch


def format_query(example: ProbeExample, *, default_instruction: str, instruction_format: str) -> str:
    prompt = example.prompt or default_instruction
    if prompt:
        return instruction_format.format(prompt, example.query)
    return example.query


def build_text_batch(
    examples: list[ProbeExample],
    *,
    train_group_size: int,
    seed: int,
    step: int,
    default_instruction: str,
    instruction_format: str,
) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    passages: list[str] = []
    for idx, example in enumerate(examples):
        rng = random.Random((seed + 1) * 1_000_003 + step * 9176 + idx)
        queries.append(
            format_query(
                example,
                default_instruction=default_instruction,
                instruction_format=instruction_format,
            )
        )
        positives = example.positives
        negatives = example.negatives
        positive = positives[rng.randrange(len(positives))]
        neg_count = train_group_size - 1
        if len(negatives) >= neg_count:
            selected_negatives = rng.sample(negatives, neg_count)
        else:
            selected_negatives = [negatives[rng.randrange(len(negatives))] for _ in range(neg_count)]
        passages.append(positive)
        passages.extend(selected_negatives)
    return queries, passages


def pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def encode_with_params(
    model: PeftModel,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
    *,
    normalize: bool,
) -> torch.Tensor:
    outputs = functional_call(
        model,
        (params, buffers),
        (),
        {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
            "return_dict": True,
        },
    )
    reps = pool_last_token(outputs.last_hidden_state, features["attention_mask"])
    if normalize:
        reps = F.normalize(reps.float(), dim=-1)
    return reps.contiguous()


def compute_probe_loss(
    model: PeftModel,
    tokenizer: Any,
    base_params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    merged_adapter: dict[str, torch.Tensor],
    batch: list[ProbeExample],
    *,
    device: torch.device,
    max_length: int,
    query_max_length: int,
    passage_max_length: int,
    train_group_size: int,
    temperature: float,
    normalize: bool,
    seed: int,
    step: int,
    default_instruction: str,
    instruction_format: str,
) -> torch.Tensor:
    call_params = dict(base_params)
    for key, value in merged_adapter.items():
        call_params[adapter_key_to_param_name(key)] = value

    queries, passages = build_text_batch(
        batch,
        train_group_size=train_group_size,
        seed=seed,
        step=step,
        default_instruction=default_instruction,
        instruction_format=instruction_format,
    )
    q_features = tokenizer(
        queries,
        truncation=True,
        max_length=query_max_length or max_length,
        padding=True,
        return_tensors="pt",
    )
    p_features = tokenizer(
        passages,
        truncation=True,
        max_length=passage_max_length or max_length,
        padding=True,
        return_tensors="pt",
    )
    q_features = {key: value.to(device) for key, value in q_features.items()}
    p_features = {key: value.to(device) for key, value in p_features.items()}
    q_reps = encode_with_params(model, call_params, buffers, q_features, normalize=normalize)
    p_reps = encode_with_params(model, call_params, buffers, p_features, normalize=normalize)
    scores = torch.matmul(q_reps, p_reps.T) / temperature
    targets = torch.arange(q_reps.size(0), device=device, dtype=torch.long) * train_group_size
    return F.cross_entropy(scores, targets)


def resolve_torch_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_base_model(
    base_model: str,
    reference_adapter: Path,
    *,
    device: torch.device,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[Any, PeftModel]:
    tokenizer = AutoTokenizer.from_pretrained(reference_adapter, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    torch_dtype = resolve_torch_dtype(dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    base = AutoModel.from_pretrained(base_model, **model_kwargs)
    if base.get_input_embeddings().weight.shape[0] != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(reference_adapter), is_trainable=False)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def search_and_merge(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference_dir = args.reference_adapter.expanduser().resolve()
    adapter_specs = [parse_named_path(raw) for raw in args.adapter]
    if len(adapter_specs) < 2:
        raise ValueError("At least two --adapter NAME=PATH arguments are required.")
    dataset_specs = [parse_named_path(raw) for raw in args.dataset]
    if not dataset_specs:
        raise ValueError("At least one --dataset NAME=PATH probe dataset is required.")

    LOGGER.info("Loading adapters on %s", device)
    reference = load_adapter(reference_dir, device=device)
    adapters = [load_adapter(spec.path, device=device) for spec in adapter_specs]
    validate_compatible(reference, adapters)
    task_vectors = [{key: state[key] - reference[key] for key in sorted(reference)} for state in adapters]

    probe_examples = build_probe_examples(
        dataset_specs,
        examples_per_dataset=args.examples_per_dataset,
        seed=args.seed,
        max_source_lines=args.max_source_lines,
    )

    LOGGER.info("Loading base model for probe-loss optimization")
    tokenizer, model = load_base_model(
        args.base_model,
        reference_dir,
        device=device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    base_params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    raw_alphas = torch.nn.Parameter(
        torch.full(
            (len(adapter_specs),),
            inverse_softplus(args.init_alpha),
            device=device,
            dtype=torch.float32,
        )
    )
    raw_lambda = torch.nn.Parameter(
        torch.tensor(inverse_softplus(args.init_lambda), device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([raw_alphas, raw_lambda], lr=args.learning_rate)
    batches = batched_cycle(probe_examples, batch_size=args.batch_size, seed=args.seed)
    log_rows: list[dict[str, Any]] = []

    for step in range(1, args.search_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_probe_losses: list[torch.Tensor] = []
        lambda_for_log: torch.Tensor | None = None
        alphas_for_log: torch.Tensor | None = None
        for accum_idx in range(args.grad_accum_steps):
            batch = next(batches)
            alphas = F.softplus(raw_alphas) + 1e-8
            lambda_scale = F.softplus(raw_lambda) + 1e-8
            merged_adapter = merge_task_vectors(
                reference,
                task_vectors,
                alphas,
                lambda_scale,
                dot_threshold=args.dot_threshold,
                previous_weight=args.previous_weight,
            )
            probe_loss = compute_probe_loss(
                model,
                tokenizer,
                base_params,
                buffers,
                merged_adapter,
                batch,
                device=device,
                max_length=args.max_length,
                query_max_length=args.query_max_length,
                passage_max_length=args.passage_max_length,
                train_group_size=args.train_group_size,
                temperature=args.temperature,
                normalize=args.normalize_embeddings,
                seed=args.seed,
                step=(step - 1) * args.grad_accum_steps + accum_idx + 1,
                default_instruction=args.default_query_instruction,
                instruction_format=args.default_query_instruction_format,
            )
            loss = (probe_loss + args.mu * lambda_scale) / args.grad_accum_steps
            loss.backward()
            step_probe_losses.append(probe_loss.detach())
            lambda_for_log = lambda_scale.detach()
            alphas_for_log = alphas.detach()
        optimizer.step()
        if lambda_for_log is None or alphas_for_log is None:
            raise RuntimeError("No gradient accumulation batches were executed.")
        probe_loss_for_log = torch.stack(step_probe_losses).mean()
        loss_for_log = probe_loss_for_log + args.mu * lambda_for_log

        row = {
            "step": step,
            "loss": float(loss_for_log.detach().cpu()),
            "probe_loss": float(probe_loss_for_log.detach().cpu()),
            "lambda": float(lambda_for_log.detach().cpu()),
            "alphas": {
                spec.name: float(value.detach().cpu())
                for spec, value in zip(adapter_specs, alphas_for_log)
            },
        }
        log_rows.append(row)
        if step == 1 or step % args.log_steps == 0 or step == args.search_steps:
            LOGGER.info(
                "step=%d loss=%.6f probe_loss=%.6f lambda=%.6f alphas=%s",
                step,
                row["loss"],
                row["probe_loss"],
                row["lambda"],
                json.dumps(row["alphas"], ensure_ascii=False),
            )

    with torch.no_grad():
        final_alphas = F.softplus(raw_alphas) + 1e-8
        final_lambda = F.softplus(raw_lambda) + 1e-8
        final_merged = merge_task_vectors(
            reference,
            task_vectors,
            final_alphas,
            final_lambda,
            dot_threshold=args.dot_threshold,
            previous_weight=args.previous_weight,
        )

    output_dir = args.output_dir.expanduser().resolve()
    tokenizer_source = (args.tokenizer_source or reference_dir).expanduser().resolve()
    copy_metadata(tokenizer_source, output_dir, config_source=reference_dir)
    save_state = {key: value.detach().cpu().to(reference[key].dtype) for key, value in final_merged.items()}
    save_file(save_state, str(output_dir / "adapter_model.safetensors"))

    metadata = {
        "mode": "self_positioning_slerp_task_vector",
        "reference_adapter": str(reference_dir),
        "adapters": {spec.name: str(spec.path) for spec in adapter_specs},
        "datasets": {
            name: [str(path) for path in paths]
            for name, paths in group_named_paths(dataset_specs)
        },
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "search": {
            "steps": args.search_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "examples_per_dataset": args.examples_per_dataset,
            "max_source_lines": args.max_source_lines,
            "train_group_size": args.train_group_size,
            "temperature": args.temperature,
            "mu": args.mu,
            "init_alpha": args.init_alpha,
            "init_lambda": args.init_lambda,
            "previous_weight": args.previous_weight,
            "dot_threshold": args.dot_threshold,
        },
        "final": {
            "lambda": float(final_lambda.detach().cpu()),
            "alphas": {
                spec.name: float(value.detach().cpu())
                for spec, value in zip(adapter_specs, final_alphas)
            },
            "last_loss": log_rows[-1]["loss"] if log_rows else None,
            "last_probe_loss": log_rows[-1]["probe_loss"] if log_rows else None,
        },
        "stats": tensor_stats(reference, [spec.name for spec in adapter_specs], adapters, final_merged),
    }
    write_json(output_dir / "merge_config.json", metadata)
    with (output_dir / "search_log.jsonl").open("w", encoding="utf-8") as f:
        for row in log_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize SLERP task-vector merging weights on a small probe dataset, "
            "then save the merged PEFT adapter."
        )
    )
    parser.add_argument("--reference-adapter", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Task adapter in NAME=PATH form. Repeat for each task model.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Probe dataset in NAME=JSONL form. Repeat with the same NAME to merge multiple jsonl files "
            "within that class before sampling; each distinct NAME still contributes --examples-per-dataset."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="data/raw/Qwen2.5-0.5B")
    parser.add_argument("--tokenizer-source", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--search-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of probe micro-batches to average before each merge-parameter optimizer step.",
    )
    parser.add_argument("--examples-per-dataset", type=int, default=1024)
    parser.add_argument(
        "--max-source-lines",
        type=int,
        default=0,
        help="Optional cap on source jsonl lines scanned per probe dataset. 0 scans the whole file.",
    )
    parser.add_argument("--train-group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--init-alpha", type=float, default=1.0)
    parser.add_argument("--init-lambda", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=320)
    parser.add_argument("--passage-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--dot-threshold", type=float, default=0.9995)
    parser.add_argument("--previous-weight", choices=["mean", "sum"], default="mean")
    parser.add_argument("--no-normalize-embeddings", dest="normalize_embeddings", action="store_false")
    parser.set_defaults(normalize_embeddings=True)
    parser.add_argument(
        "--default-query-instruction",
        default="Given a query, retrieve passages that are relevant to the query.",
    )
    parser.add_argument(
        "--default-query-instruction-format",
        default="Instruct: {}\nQuery: {}",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s|%(levelname)s|%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    if args.search_steps <= 0:
        raise ValueError("--search-steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive.")
    if args.train_group_size < 2:
        raise ValueError("--train-group-size must be at least 2.")
    metadata = search_and_merge(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
