from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .arguments import DataConfig, DatasetConfig, EmbedTrainingExtras
from .record_store import ArrowStorePool, JsonRecordStore, RecordStore, load_arrow_manifest

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
    record_count: int
    store: Optional[RecordStore] = None
    arrow_unit_id: Optional[str] = None


@dataclass
class _DatasetStats:
    planned_global_batches: int = 0
    planned_global_instances: int = 0
    consumed_local_batches: int = 0
    consumed_local_instances: int = 0


class _CompactBatchPlan(Sequence[tuple[int, np.ndarray]]):
    """Store an epoch plan without allocating one Python object per batch.

    A full F2LLM epoch contains more than one million global batches.
    Keeping one Python tuple and one NumPy view per batch creates millions of
    native allocations per rank. This class represents the same logical plan
    using three compact structures:

    ``schedule``
        A ``[num_batches, 2]`` uint32 array. Each row stores
        ``(dataset_idx, batch_ordinal)``. Its row number is the shuffled global
        training step. ``batch_ordinal`` is the batch's position *within that
        dataset before cross-dataset/block shuffling*.

    ``permutations``
        One flat array of shuffled record indices per dataset. A dataset's
        ordinal N batch is a slice of this array, rather than a separately
        allocated NumPy array.

    ``global_batch_sizes``
        The number of records in a global batch for each dataset. It equals
        ``per_rank_batch_size * world_size`` and is needed to translate a
        batch ordinal into a slice boundary.

    For example, schedule row ``(7, 3)`` with global batch size 16 resolves to
    ``permutations[7][48:64]``. The returned slice is only a view and is created
    lazily when Trainer requests that step. Every distributed rank owns the
    same schedule and permutation; ``MultiDatasetBatchDataset.__getitem__``
    later selects that rank's disjoint part of the global slice.
    """

    def __init__(
        self,
        schedule: np.ndarray,
        permutations: Dict[int, np.ndarray],
        global_batch_sizes: Dict[int, int],
    ) -> None:
        """Attach the precomputed schedule and its per-dataset lookup tables.

        Construction intentionally performs no expansion or copying: epoch
        planning has already produced arrays with the desired ownership and
        order, so this object is only the lookup facade used by Dataset/Trainer.
        """
        self.schedule = schedule
        self.permutations = permutations
        self.global_batch_sizes = global_batch_sizes

    def __len__(self) -> int:
        """Return the number of global optimizer-step batches in this epoch."""
        return int(len(self.schedule))

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        """Resolve one shuffled step to ``(dataset_idx, global_record_indices)``.

        Negative indices follow normal Python sequence semantics. Slices are
        deliberately unsupported because callers consume one complete
        contrastive batch at a time and supporting slices would encourage
        materializing many batch views again.
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Compact batch plan indices must be integers")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)

        # The ordinal is dataset-local, so convert it to offsets using this
        # dataset's global batch size. The resulting NumPy slice shares memory
        # with the one stored permutation and allocates no record-index copy.
        dataset_idx, batch_ordinal = (int(value) for value in self.schedule[normalized])
        batch_size = self.global_batch_sizes[dataset_idx]
        start = batch_ordinal * batch_size
        return dataset_idx, self.permutations[dataset_idx][start : start + batch_size]

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate in global training-step order, resolving one lazy view at a time."""
        for index in range(len(self)):
            yield self[index]

    def instruction_offset(self, index: int) -> int:
        """Return this batch's dataset-local starting example position.

        Query/passage instruction lists rotate as a per-dataset queue. They
        must therefore use the pre-shuffle dataset-local ordinal, not the
        batch's final global step number. This value is numerically identical
        to the start offset used to slice the dataset permutation.
        """
        dataset_idx, batch_ordinal = (int(value) for value in self.schedule[index])
        return batch_ordinal * self.global_batch_sizes[dataset_idx]


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

        if not data_args.datasets and not data_args.indexed_dataset_manifest:
            raise ValueError("At least one of data.datasets or data.indexed_dataset_manifest is required.")
        if data_args.arrow_open_mode != "lazy":
            raise ValueError("Only data.arrow_open_mode=lazy is supported.")
        if data_args.arrow_prefetch_units not in {0, 1}:
            raise ValueError("data.arrow_prefetch_units must be 0 or 1.")
        if data_args.arrow_prefetch_units >= data_args.arrow_max_open_units:
            raise ValueError("data.arrow_prefetch_units must be smaller than arrow_max_open_units.")
        if data_args.unit_block_batches <= 0:
            raise ValueError("data.unit_block_batches must be positive.")

        _validate_instruction_config(data_args.default_query_instruction, "data.default_query_instruction")
        _validate_instruction_config(data_args.default_passage_instruction, "data.default_passage_instruction")
        self.datasets: List[_LoadedDataset] = []
        self.arrow_store_pool: Optional[ArrowStorePool] = None
        seen_names: set[str] = set()
        if data_args.indexed_dataset_manifest:
            _, descriptors = load_arrow_manifest(data_args.indexed_dataset_manifest)
            self.arrow_store_pool = ArrowStorePool(
                descriptors,
                max_open_units=data_args.arrow_max_open_units,
                verify_mode=data_args.arrow_verify_mode,
            )
            for descriptor in descriptors:
                if descriptor.unit_id in seen_names:
                    raise ValueError(f"Duplicate dataset name is not allowed: {descriptor.unit_id}")
                seen_names.add(descriptor.unit_id)
                cfg = DatasetConfig(
                    name=descriptor.unit_id,
                    path=descriptor.metadata_path,
                    task_type=descriptor.task_type,
                    data_format="indexed_arrow",
                )
                self.datasets.append(
                    _LoadedDataset(
                        key=descriptor.unit_id,
                        config=cfg,
                        record_count=descriptor.query_count,
                        arrow_unit_id=descriptor.unit_id,
                    )
                )
            logger.info("Loaded Indexed Arrow manifest %s with %d unit descriptors", data_args.indexed_dataset_manifest, len(descriptors))
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
            if cfg.data_format not in {"auto", "json"}:
                raise ValueError(f"Explicit datasets currently support data_format=auto|json, got {cfg.data_format}")
            records = _read_json_records(cfg.path)
            if not records:
                raise ValueError(f"Empty dataset: {cfg.name} ({cfg.path})")
            self.datasets.append(
                _LoadedDataset(key=cfg.name, config=cfg, record_count=len(records), store=JsonRecordStore(records))
            )
            logger.info("Loaded dataset %s with %d records from %s", cfg.name, len(records), cfg.path)

        self.batch_plan = _CompactBatchPlan(
            np.empty((0, 2), dtype=np.uint32), {}, {}
        )
        self.stats: Dict[str, _DatasetStats] = {}
        self.refresh_epoch(0)

    def _sample_epoch_indices(self, gen: np.random.Generator, loaded: _LoadedDataset) -> np.ndarray:
        cfg = loaded.config
        record_count = loaded.record_count
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
        # The compact plan is built in two levels. ``permutations`` owns the
        # actual record-index order (one array per dataset), while schedule
        # rows refer to batches by dataset-local ordinal. They are kept apart
        # so shuffling batches never copies the underlying record indices.
        permutations: Dict[int, np.ndarray] = {}
        global_batch_sizes: Dict[int, int] = {}
        schedule_chunks: List[np.ndarray] = []
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
            dataset_idx_int = int(dataset_idx)
            batch_count = len(idxs) // global_bs
            if batch_count:
                if batch_count > np.iinfo(np.uint32).max:
                    raise ValueError(f"Too many batches for compact plan: {loaded.key}={batch_count}")
                # Only full global batches enter the permutation. Keeping the
                # flat array lets ordinal N later resolve to
                # [N * global_bs : (N + 1) * global_bs].
                permutations[dataset_idx_int] = idxs[: batch_count * global_bs]
                global_batch_sizes[dataset_idx_int] = global_bs

                # Before cross-dataset shuffling, ordinals are simply
                # 0..batch_count-1 for this dataset. column_stack creates the
                # compact two-column schedule fragment, not batch index views.
                schedule_chunks.append(
                    np.column_stack(
                        (
                            np.full(batch_count, dataset_idx_int, dtype=np.uint32),
                            np.arange(batch_count, dtype=np.uint32),
                        )
                    )
                )
                dataset_batch_counts[dataset_idx_int] = batch_count
                stat = self.stats[loaded.key]
                stat.planned_global_batches = batch_count
                stat.planned_global_instances = batch_count * global_bs
            # Drop the final partial global batch. This keeps each rank's local
            # slice the same size and avoids all_gather shape mismatches.
        # Concatenation here is cheap relative to the old representation: each
        # batch contributes two uint32 scalars (8 bytes total), independent of
        # global batch size.
        schedule = (
            np.concatenate(schedule_chunks, axis=0)
            if schedule_chunks else np.empty((0, 2), dtype=np.uint32)
        )
        if self.data_args.unit_block_batches == 1:
            # Fully mix individual batches while leaving each dataset's record
            # permutation untouched.
            gen.shuffle(schedule, axis=0)
        else:
            # Shuffle small unit-homogeneous blocks instead of individual
            # batches. This preserves the exact sampled batches while giving
            # ArrowStorePool enough locality to avoid open/evict thrashing.
            block_size = self.data_args.unit_block_batches
            block_chunks: List[np.ndarray] = []
            for dataset_idx in dataset_order:
                dataset_idx_int = int(dataset_idx)
                count = dataset_batch_counts[dataset_idx_int]
                if not count:
                    continue
                # A block descriptor is (dataset_idx, first_batch_ordinal,
                # number_of_batches). The final block may be shorter.
                starts = np.arange(0, count, block_size, dtype=np.uint32)
                lengths = np.minimum(block_size, count - starts).astype(np.uint32)
                block_chunks.append(
                    np.column_stack(
                        (np.full(len(starts), dataset_idx_int, dtype=np.uint32), starts, lengths)
                    )
                )
            blocks = np.concatenate(block_chunks, axis=0)

            # Shuffle block descriptors rather than individual schedule rows.
            # Batches inside a block retain ascending dataset-local ordinals,
            # giving Arrow mmap/LRU several consecutive reads from one unit.
            gen.shuffle(blocks, axis=0)
            schedule = np.empty((sum(dataset_batch_counts.values()), 2), dtype=np.uint32)
            cursor = 0
            for dataset_idx, start, length in blocks:
                length_int = int(length)

                # Expand only the tiny two-column schedule. Record indices are
                # still referenced through ``permutations`` and are not copied.
                schedule[cursor : cursor + length_int, 0] = dataset_idx
                schedule[cursor : cursor + length_int, 1] = np.arange(
                    int(start), int(start) + length_int, dtype=np.uint32
                )
                cursor += length_int
        self.batch_plan = _CompactBatchPlan(schedule, permutations, global_batch_sizes)
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
        store = loaded.store
        if loaded.arrow_unit_id is not None:
            assert self.arrow_store_pool is not None
            store = self.arrow_store_pool.get(loaded.arrow_unit_id)
        assert store is not None
        fetched = store.get_records([int(record_idx) for record_idx in local_indices])
        return self._build_batch(
            loaded,
            [
                (record, int(record_idx), int(global_position))
                for record, record_idx, global_position in zip(fetched, local_indices, local_positions)
            ],
            batch_idx,
            self.batch_plan.instruction_offset(batch_idx),
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
        pass
        # logger.info(
        #     "Dataset %s stats for epoch %s, rank %s/%s. "
        #     "planned_* is global-plan scope; consumed_* is local to this rank: %s",
        #     prefix,
        #     self.epoch,
        #     self.process_index,
        #     self.world_size,
        #     self.format_consumption_stats(),
        # )

    def _build_batch(
        self,
        loaded: _LoadedDataset,
        records: List[tuple[Dict[str, Any], int, int]],
        batch_idx: int,
        instruction_offset: int,
    ) -> Dict[str, Any]:
        """Convert canonical records into the flattened passage-group layout.

        Passage order is ``[positive, negative...]`` for every query. The loss
        relies on this invariant when it computes target column offsets.
        """
        cfg = loaded.config
        # Retrieval normally uses all other passages as in-batch negatives;
        # clustering/classification restrict scoring to the explicit group to
        # avoid false negatives from semantically compatible batch examples.
        task_defaults = self.data_args.task_defaults.get(cfg.task_type or "", {})
        train_group_size = cfg.train_group_size or task_defaults.get("train_group_size") or self.data_args.default_train_group_size
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
            "no_in_batch_neg": (
                cfg.no_in_batch_neg
                if cfg.no_in_batch_neg is not None
                else bool(task_defaults.get("no_in_batch_neg", False))
            ),
            "query_max_len": cfg.query_max_len or self.data_args.default_query_max_len,
            "passage_max_len": cfg.passage_max_len or self.data_args.default_passage_max_len,
            "loss_kwargs": dict(cfg.loss_kwargs),
            "model_kwargs": dict(cfg.model_kwargs),
            "sub_batch_size": self.training_extras.sub_batch_size,
            "train_group_size": train_group_size,
        }
        return batch

    def close(self) -> None:
        for loaded in self.datasets:
            if loaded.store is not None:
                loaded.store.close()
        if self.arrow_store_pool is not None:
            self.arrow_store_pool.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
