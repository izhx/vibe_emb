#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vibe_emb.config import load_yaml_config, parse_sections
from vibe_emb.data import MultiDatasetBatchDataset


def _init_dist_if_needed() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return rank, world_size


def _gather_object(obj: Dict[str, Any], world_size: int) -> List[Dict[str, Any]]:
    if world_size == 1:
        return [obj]
    gathered: List[Dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return [item for item in gathered if item is not None]


def _rank_step_snapshot(dataset: MultiDatasetBatchDataset, step: int) -> Dict[str, Any]:
    dataset_idx, global_indices = dataset.batch_plan[step]
    loaded = dataset.datasets[dataset_idx]
    per_rank = len(global_indices) // dataset.world_size
    start = dataset.process_index * per_rank
    local_indices = global_indices[start : start + per_rank].tolist()

    # Fetch the actual item as Trainer would. This confirms that __getitem__
    # uses the same shard boundaries as the plan metadata inspected above.
    batch = dataset[step]
    return {
        "rank": dataset.process_index,
        "step": step,
        "dataset_key": loaded.key,
        "global_indices": global_indices.tolist(),
        "local_indices": local_indices,
        "query_count": len(batch["queries"]),
        "passage_count": len(batch["passages"]),
        "train_group_size": int(batch["train_group_size"]),
    }


def _verify_step(step_items: List[Dict[str, Any]]) -> None:
    if not step_items:
        raise AssertionError("No rank snapshots were gathered.")
    step_items = sorted(step_items, key=lambda item: item["rank"])
    first = step_items[0]
    global_indices = first["global_indices"]
    dataset_key = first["dataset_key"]

    for item in step_items:
        if item["dataset_key"] != dataset_key:
            raise AssertionError(f"Step {first['step']} dataset mismatch: {step_items}")
        if item["global_indices"] != global_indices:
            raise AssertionError(f"Step {first['step']} global batch mismatch across ranks.")
        if item["query_count"] != len(item["local_indices"]):
            raise AssertionError(f"Step {first['step']} rank {item['rank']} query count does not match local shard.")
        if item["passage_count"] != len(item["local_indices"]) * item["train_group_size"]:
            raise AssertionError(f"Step {first['step']} rank {item['rank']} passage count is inconsistent.")

    reconstructed: List[int] = []
    for item in step_items:
        reconstructed.extend(item["local_indices"])
    if reconstructed != global_indices:
        raise AssertionError(
            f"Step {first['step']} local shards do not reconstruct the global batch. "
            f"expected={global_indices}, got={reconstructed}"
        )
    # Do not require record ids to be globally unique. sample_factor > 1.0
    # intentionally allows repeated records. The invariant here is positional:
    # rank shards are consecutive chunks whose concatenation exactly matches
    # the global index list.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that distributed ranks share the same global dataset batch plan and take disjoint shards."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()

    rank, world_size = _init_dist_if_needed()
    raw = load_yaml_config(args.config)
    _, data_args, training_raw, training_extras = parse_sections(raw)
    seed = int(training_raw.get("seed", 42))

    dataset = MultiDatasetBatchDataset(
        data_args=data_args,
        training_extras=training_extras,
        seed=seed,
        process_index=rank,
        world_size=world_size,
    )
    if args.epoch != 0:
        dataset.refresh_epoch(args.epoch)

    inspected = min(args.steps, len(dataset))
    for step in range(inspected):
        gathered = _gather_object(_rank_step_snapshot(dataset, step), world_size)
        if rank == 0:
            _verify_step(gathered)

    if rank == 0:
        print(
            f"OK: verified {inspected} distributed dataset steps from {args.config} "
            f"with world_size={world_size}, epoch={args.epoch}."
        )
        print(dataset.format_consumption_stats())

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
