#!/usr/bin/env python3
"""Verify and inspect F2LLM Indexed Arrow profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def open_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_file(source).read_all()


def _unit(profile: dict[str, Any], unit_id: str) -> dict[str, Any]:
    for unit in profile["units"]:
        if unit["unit_id"] == unit_id:
            return unit
    raise ValueError(f"Unknown unit: {unit_id}")


def verify_unit(root: Path, unit: dict[str, Any], full: bool) -> dict[str, Any]:
    paths = {name: root / unit[f"{name}_path"] for name in ("metadata", "queries", "corpus")}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Missing {name}: {path}")
    metadata = json.loads(paths["metadata"].read_text())
    if paths["queries"].stat().st_size != unit["queries_file_size"]:
        raise ValueError(f"queries file size mismatch for {unit['unit_id']}")
    if paths["corpus"].stat().st_size != unit["corpus_file_size"]:
        raise ValueError(f"corpus file size mismatch for {unit['unit_id']}")
    queries = open_table(paths["queries"])
    corpus = open_table(paths["corpus"])
    if len(queries) != unit["query_count"] or len(queries) != metadata["query_count"]:
        raise ValueError(f"query count mismatch for {unit['unit_id']}")
    if len(corpus) != unit["corpus_count"] or len(corpus) != metadata["corpus_count"]:
        raise ValueError(f"corpus count mismatch for {unit['unit_id']}")
    corpus_ids = corpus["doc_id"]
    if len(corpus) and (
        pc.min(corpus_ids).as_py() != 0
        or pc.max(corpus_ids).as_py() != len(corpus) - 1
        or pc.count_distinct(corpus_ids).as_py() != len(corpus)
    ):
        raise ValueError(f"non-contiguous doc IDs for {unit['unit_id']}")
    query_ids = queries["query_id"]
    if len(queries) and (
        pc.min(query_ids).as_py() != 0
        or pc.max(query_ids).as_py() != len(queries) - 1
        or pc.count_distinct(query_ids).as_py() != len(queries)
    ):
        raise ValueError(f"non-contiguous query IDs for {unit['unit_id']}")
    positive = queries["positive_doc_id"]
    negatives = pc.list_flatten(queries["negative_doc_ids"])
    for label, values in (("positive", positive), ("negative", negatives)):
        if len(values) and (pc.min(values).as_py() < 0 or pc.max(values).as_py() >= len(corpus)):
            raise ValueError(f"invalid {label} doc reference for {unit['unit_id']}")
    if pc.count_distinct(queries["sample_id"]).as_py() != len(queries):
        raise ValueError(f"duplicate sample IDs for {unit['unit_id']}")
    if full:
        for name in ("metadata", "queries", "corpus"):
            expected = unit[f"{name}_fingerprint"]
            if sha256_file(paths[name]) != expected:
                raise ValueError(f"{name} fingerprint mismatch for {unit['unit_id']}")
    return {"unit_id": unit["unit_id"], "query_count": len(queries), "corpus_count": len(corpus), "ok": True}


def inspect_sample(root: Path, unit: dict[str, Any], sample_id: str) -> dict[str, Any]:
    query_table = open_table(root / unit["queries_path"])
    corpus_table = open_table(root / unit["corpus_path"])
    wanted = bytes.fromhex(sample_id)
    matches = pc.indices_nonzero(pc.equal(query_table["sample_id"], pa.scalar(wanted, type=pa.binary(16))))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one sample {sample_id}; found {len(matches)}")
    row = query_table.take(matches).to_pylist()[0]
    doc_ids = [row["positive_doc_id"], *row["negative_doc_ids"]]
    docs = {item["doc_id"]: item["text"] for item in corpus_table.take(pa.array(doc_ids)).to_pylist()}
    all_ids = query_table["positive_doc_id"].to_pylist() + pc.list_flatten(query_table["negative_doc_ids"]).to_pylist()
    counts = Counter(all_ids)
    reference_counts = {doc_id: counts[doc_id] for doc_id in doc_ids}
    return {
        "unit_id": unit["unit_id"],
        "task_type": unit["task_type"],
        "sample_id": sample_id,
        "source_shard_id": row["source_shard_id"],
        "source_row_index": row["source_row_index"],
        "lang": row["lang"],
        "query": row["query"],
        "positive": docs[row["positive_doc_id"]],
        "negatives": [docs[value] for value in row["negative_doc_ids"]],
        "doc_reference_counts": {str(key): value for key, value in reference_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--unit", action="append", default=[])
    verify.add_argument("--full", action="store_true")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--unit", required=True)
    inspect.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    profile = json.loads(manifest_path.read_text())
    if args.command == "verify":
        wanted = set(args.unit)
        units = [unit for unit in profile["units"] if not wanted or unit["unit_id"] in wanted]
        missing = wanted - {unit["unit_id"] for unit in units}
        if missing:
            raise ValueError(f"Unknown units: {sorted(missing)}")
        print(json.dumps([verify_unit(root, unit, args.full) for unit in units], indent=2))
    else:
        print(json.dumps(inspect_sample(root, _unit(profile, args.unit), args.sample_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
