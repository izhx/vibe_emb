from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .arguments import DataConfig, DatasetConfig, EmbedTrainingExtras

logger = logging.getLogger(__name__)


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    if os.path.isdir(path):
        records: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(path)):
            if name.endswith((".json", ".jsonl")):
                records.extend(_read_json_records(os.path.join(path, name)))
        return records

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            return loaded
        raise ValueError(f"JSON dataset must contain a list: {path}")
    raise ValueError(f"Unsupported dataset file type: {path}")


def _maybe_shuffle_text(text: str, rng: random.Random) -> str:
    if len(text) <= 100:
        return text
    chunk_size = len(text) // 3 + 1
    parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    rng.shuffle(parts)
    return " ".join(parts)


def _stable_int_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _validate_instruction_config(value: Optional[Union[str, List[str]]], field_name: str) -> None:
    if value is None or isinstance(value, str):
        return
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must not be an empty list.")
        bad = [item for item in value if not isinstance(item, str)]
        if bad:
            raise ValueError(f"{field_name} list values must all be strings; got {bad!r}.")
        return
    raise ValueError(f"{field_name} must be a string, a list of strings, or null; got {type(value).__name__}.")


def _select_instruction(value: Optional[Union[str, List[str]]], position: int) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return value[position % len(value)]


@dataclass
class _LoadedDataset:
    key: str
    config: DatasetConfig
    records: List[Dict[str, Any]]


@dataclass
class _DatasetStats:
    planned_global_batches: int = 0
    planned_global_instances: int = 0
    consumed_local_batches: int = 0
    consumed_local_instances: int = 0


class MultiDatasetBatchDataset(Dataset):
    """Dataset whose item is already one rank-local contrastive batch.

    Trainer still sees a normal Dataset, but every __getitem__ result contains
    the full contrastive batch for the current rank. Keep
    per_device_train_batch_size=1 so Trainer does not merge two already-built
    contrastive batches.
    """

    def __init__(
        self,
        data_args: DataConfig,
        training_extras: EmbedTrainingExtras,
        seed: int,
        process_index: int = 0,
        world_size: int = 1,
    ) -> None:
        if not data_args.same_dataset_within_batch:
            raise ValueError("Only same_dataset_within_batch=true is supported in v1.")
        self.data_args = data_args
        self.training_extras = training_extras
        self.seed = seed
        self.process_index = process_index
        self.world_size = world_size
        self.epoch = 0

        _validate_instruction_config(data_args.default_query_instruction, "data.default_query_instruction")
        _validate_instruction_config(data_args.default_passage_instruction, "data.default_passage_instruction")
        self.datasets: List[_LoadedDataset] = []
        seen_names: set[str] = set()
        for cfg in data_args.datasets:
            if cfg.name in seen_names:
                raise ValueError(f"Duplicate dataset name is not allowed: {cfg.name}")
            seen_names.add(cfg.name)
            _validate_instruction_config(cfg.query_instruction, f"data.datasets[{cfg.name}].query_instruction")
            _validate_instruction_config(cfg.passage_instruction, f"data.datasets[{cfg.name}].passage_instruction")
            if cfg.sample_factor <= 0:
                raise ValueError(f"sample_factor must be > 0 for dataset {cfg.name}: {cfg.sample_factor}")
            if cfg.sample_size >= 0 and cfg.sample_factor != 1.0:
                raise ValueError(
                    f"Dataset {cfg.name} cannot set both sample_size and sample_factor. "
                    "Use sample_size for an absolute count or sample_factor for proportional sampling."
                )
            records = _read_json_records(cfg.path)
            if not records:
                raise ValueError(f"Empty dataset: {cfg.name} ({cfg.path})")
            self.datasets.append(_LoadedDataset(key=cfg.name, config=cfg, records=records))
            logger.info("Loaded dataset %s with %d records from %s", cfg.name, len(records), cfg.path)

        self.batch_plan: List[tuple[int, np.ndarray]] = []
        self.batch_instruction_offsets: List[int] = []
        self.stats: Dict[str, _DatasetStats] = {}
        self.refresh_epoch(0)

    def _sample_epoch_indices(self, gen: np.random.Generator, loaded: _LoadedDataset) -> np.ndarray:
        cfg = loaded.config
        record_count = len(loaded.records)
        if cfg.sample_size >= 0:
            # sample_size is an absolute cap and keeps the old behavior: one
            # deterministic permutation, then truncate.
            target_size = min(cfg.sample_size, record_count)
            return gen.permutation(record_count)[:target_size]

        target_size = int(record_count * cfg.sample_factor)
        if target_size <= record_count:
            # sample_factor < 1.0 is downsampling; == 1.0 is a full shuffled epoch.
            return gen.permutation(record_count)[:target_size]

        # sample_factor > 1.0 upsamples the dataset. We append deterministic
        # full permutations until reaching the target, so repeats are allowed
        # but still rank-independent and reproducible for the epoch.
        chunks: List[np.ndarray] = []
        remaining = target_size
        while remaining > 0:
            perm = gen.permutation(record_count)
            take = min(remaining, record_count)
            chunks.append(perm[:take])
            remaining -= take
        return np.concatenate(chunks)

    def refresh_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.stats = {loaded.key: _DatasetStats() for loaded in self.datasets}

        # Every rank must build exactly the same global plan. Cross-device
        # negatives call all_gather on q/p reps, so each step needs matching
        # tensor shapes on all ranks. The step also must come from the same
        # dataset because loss_kwargs/model_kwargs/no_in_batch_neg/train_group_size
        # are dataset-specific. If ranks diverge here, collectives can hang or
        # the contrastive targets can point at the wrong passage offsets.
        #
        # Implementation detail: all ranks use the same seed + epoch, shuffle
        # datasets and record indices in the same order, and store full
        # global_indices in the plan. We only slice by process_index in
        # __getitem__, so the global plan remains identical across ranks. The
        # per-record pos/neg choices also use global batch position seeds, so a
        # rank-local shard is equivalent to slicing an already sampled global
        # batch.
        gen = np.random.default_rng(self.seed + epoch)
        plan: List[tuple[int, np.ndarray, int]] = []
        dataset_batch_counts = {idx: 0 for idx in range(len(self.datasets))}
        dataset_order = gen.permutation(len(self.datasets))
        for dataset_idx in dataset_order:
            loaded = self.datasets[int(dataset_idx)]
            cfg = loaded.config
            per_rank_bs = cfg.batch_size or self.data_args.default_batch_size
            global_bs = per_rank_bs * self.world_size
            if global_bs <= 0:
                raise ValueError(f"Invalid batch_size for dataset {cfg.name}: {per_rank_bs}")

            # sample_size and sample_factor both decide how many record indices
            # enter this epoch before batching. Batch construction may still
            # drop the final partial global batch below.
            idxs = self._sample_epoch_indices(gen, loaded)
            for start in range(0, len(idxs), global_bs):
                batch = idxs[start : start + global_bs]
                if len(batch) == global_bs:
                    dataset_idx_int = int(dataset_idx)
                    # Store a dataset-local batch ordinal before shuffling the
                    # final plan. Instruction lists use it to behave like a
                    # per-dataset global round-robin queue even after batches
                    # are interleaved with other datasets.
                    batch_ordinal = dataset_batch_counts[dataset_idx_int]
                    dataset_batch_counts[dataset_idx_int] += 1
                    plan.append((dataset_idx_int, batch, batch_ordinal))
                    stat = self.stats[loaded.key]
                    stat.planned_global_batches += 1
                    stat.planned_global_instances += len(batch)
                # Drop partial global batches. This keeps each rank's local
                # slice the same size and avoids all_gather shape mismatches.
        gen.shuffle(plan)
        self.batch_plan = [(dataset_idx, batch) for dataset_idx, batch, _ in plan]
        self.batch_instruction_offsets = [
            batch_ordinal * len(batch) for _, batch, batch_ordinal in plan
        ]
        if not self.batch_plan:
            raise ValueError("No full training batches were generated. Lower batch_size or increase sampling.")
        logger.info("Generated %d global batches for epoch %d", len(self.batch_plan), epoch)
        self.log_consumption_stats("planned")

    def __len__(self) -> int:
        return len(self.batch_plan)

    def __getitem__(self, batch_idx: int) -> Dict[str, Any]:
        dataset_idx, global_indices = self.batch_plan[batch_idx]
        loaded = self.datasets[dataset_idx]
        cfg = loaded.config
        per_rank = len(global_indices) // self.world_size
        start = self.process_index * per_rank
        # The plan stores the same global_indices on every rank. Slicing here
        # with process_index gives each rank a disjoint local shard while all
        # ranks still execute the same dataset_key at the same training step.
        local_indices = global_indices[start : start + per_rank]
        local_positions = range(start, start + per_rank)
        stat = self.stats[loaded.key]
        # consumed is intentionally counted in __getitem__: it reflects what
        # Trainer actually asked this rank to train on, which can differ from
        # planned when max_steps stops in the middle of an epoch.
        stat.consumed_local_batches += 1
        stat.consumed_local_instances += len(local_indices)
        return self._build_batch(
            loaded,
            [
                (loaded.records[int(record_idx)], int(record_idx), int(global_position))
                for record_idx, global_position in zip(local_indices, local_positions)
            ],
            batch_idx,
            self.batch_instruction_offsets[batch_idx],
        )

    def format_consumption_stats(self) -> str:
        parts = []
        for key in sorted(self.stats):
            stat = self.stats[key]
            parts.append(
                (
                    f"{key}: planned_global_batches={stat.planned_global_batches}, "
                    f"planned_global_instances={stat.planned_global_instances}, "
                    f"consumed_local_batches={stat.consumed_local_batches}, "
                    f"consumed_local_instances={stat.consumed_local_instances}"
                )
            )
        return " | ".join(parts)

    def log_consumption_stats(self, prefix: str = "consumed") -> None:
        logger.info(
            "Dataset %s stats for epoch %s, rank %s/%s. "
            "planned_* is global-plan scope; consumed_* is local to this rank: %s",
            prefix,
            self.epoch,
            self.process_index,
            self.world_size,
            self.format_consumption_stats(),
        )

    def _build_batch(
        self,
        loaded: _LoadedDataset,
        records: List[tuple[Dict[str, Any], int, int]],
        batch_idx: int,
        instruction_offset: int,
    ) -> Dict[str, Any]:
        cfg = loaded.config
        train_group_size = cfg.train_group_size or self.data_args.default_train_group_size
        if train_group_size < 2:
            raise ValueError(f"train_group_size must be >= 2 for dataset {cfg.name}")

        queries: List[str] = []
        passages: List[str] = []
        teacher_scores: List[float] = []
        has_scores = False

        for record, record_idx, global_position in records:
            # Derive randomness from global sample identity, not process_index.
            # This makes rank-local execution equivalent to first sampling the
            # whole global batch and then slicing it by rank.
            rng = random.Random(
                _stable_int_seed(self.seed, self.epoch, loaded.key, batch_idx, global_position, record_idx)
            )
            query = record["query"]
            instruction_position = instruction_offset + global_position
            query_instruction = cfg.query_instruction
            if query_instruction is None:
                query_instruction = self.data_args.default_query_instruction
            prompt = record.get("prompt") or _select_instruction(query_instruction, instruction_position)
            if prompt:
                query_format = cfg.query_instruction_format or self.data_args.default_query_instruction_format
                query = query_format.format(prompt, query)
            queries.append(query)

            pos = record.get("pos")
            neg = record.get("neg")
            if not isinstance(pos, list) or not pos:
                raise ValueError(f"Dataset {cfg.name} record has no positive passages")
            if not isinstance(neg, list) or not neg:
                raise ValueError(f"Dataset {cfg.name} record has no negative passages")

            pos_idx = rng.randrange(len(pos))
            pos_text = pos[pos_idx]
            if cfg.shuffle_text:
                pos_text = _maybe_shuffle_text(pos_text, rng)
            group = [pos_text]

            neg_count = train_group_size - 1
            if len(neg) < neg_count:
                repeats = math.ceil(neg_count / len(neg))
                neg_indices = rng.sample(list(range(len(neg))) * repeats, neg_count)
            else:
                neg_indices = rng.sample(list(range(len(neg))), neg_count)
            group.extend(neg[i] for i in neg_indices)

            passage_instruction = cfg.passage_instruction
            if passage_instruction is None:
                passage_instruction = self.data_args.default_passage_instruction
            selected_passage_instruction = _select_instruction(passage_instruction, instruction_position)
            if selected_passage_instruction:
                passage_format = cfg.passage_instruction_format or self.data_args.default_passage_instruction_format
                # A query's positive and negatives share one passage
                # instruction. Mixing templates inside the same contrastive
                # group would add an avoidable confound to the target layout.
                group = [passage_format.format(selected_passage_instruction, p) for p in group]
            passages.extend(group)

            if "pos_scores" in record and "neg_scores" in record:
                has_scores = True
                teacher_scores.append(float(record["pos_scores"][pos_idx]))
                teacher_scores.extend(float(record["neg_scores"][i]) for i in neg_indices)

        batch = {
            "queries": queries,
            "passages": passages,
            "teacher_scores": teacher_scores if has_scores else None,
            "dataset_name": cfg.name,
            "dataset_key": loaded.key,
            "no_in_batch_neg": cfg.no_in_batch_neg,
            "query_max_len": cfg.query_max_len or self.data_args.default_query_max_len,
            "passage_max_len": cfg.passage_max_len or self.data_args.default_passage_max_len,
            "loss_kwargs": dict(cfg.loss_kwargs),
            "model_kwargs": dict(cfg.model_kwargs),
            "sub_batch_size": self.training_extras.sub_batch_size,
            "train_group_size": train_group_size,
        }
        return batch


class RefreshEpochCallback:
    """Tiny callback object to avoid importing Trainer types in data.py."""

    def __init__(self, dataset: MultiDatasetBatchDataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.dataset.refresh_epoch(int(state.epoch or 0))
        return control
