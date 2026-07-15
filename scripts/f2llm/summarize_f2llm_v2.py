#!/usr/bin/env python3
"""Inventory F2LLM-v2 Parquet files and retain one example per file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def schema_fields(parquet_file: pq.ParquetFile) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in parquet_file.schema_arrow
    ]


def signature(fields: list[dict[str, object]], *, logical: bool) -> tuple[tuple[object, ...], ...]:
    values = [(field["name"], field["type"], field["nullable"]) for field in fields]
    return tuple(sorted(values) if logical else values)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/F2LLM-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/F2LLM-v2_inventory"))
    args = parser.parse_args()

    parquet_paths = sorted(args.input_dir.glob("*.parquet"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    physical_ids: dict[tuple[tuple[object, ...], ...], str] = {}
    logical_ids: dict[tuple[tuple[object, ...], ...], str] = {}
    physical_groups: dict[str, list[str]] = defaultdict(list)
    logical_groups: dict[str, list[str]] = defaultdict(list)

    for path in parquet_paths:
        parquet_file = pq.ParquetFile(path)
        fields = schema_fields(parquet_file)
        physical_signature = signature(fields, logical=False)
        logical_signature = signature(fields, logical=True)
        physical_id = physical_ids.setdefault(
            physical_signature, f"P{len(physical_ids) + 1:02d}"
        )
        logical_id = logical_ids.setdefault(logical_signature, f"L{len(logical_ids) + 1:02d}")
        physical_groups[physical_id].append(path.name)
        logical_groups[logical_id].append(path.name)

        metadata = parquet_file.metadata
        # Polars pushes LIMIT 1 into its Parquet scan. PyArrow's iter_batches may
        # still decode an entire row group for wide files, which is prohibitively
        # expensive for this roughly 564 GB collection.
        first_row = pl.scan_parquet(path).head(1).collect().to_dicts()[0]
        size_bytes = path.stat().st_size
        rows.append(
            {
                "file": path.name,
                "size_bytes": size_bytes,
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "columns": len(fields),
                "physical_schema_id": physical_id,
                "logical_schema_id": logical_id,
                "schema": json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
            }
        )
        samples.append(
            {
                "file": path.name,
                "physical_schema_id": physical_id,
                "logical_schema_id": logical_id,
                "sample": first_row,
            }
        )

    inventory_path = args.output_dir / "inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    total_rows = sum(int(row["rows"]) for row in rows)
    summary_path = args.output_dir / "README.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# F2LLM-v2 数据格式统计\n\n")
        handle.write(
            f"- Parquet 文件：{len(rows):,}\n"
            f"- 总记录数：{total_rows:,}\n"
            f"- 文件总大小：{total_bytes:,} bytes（{format_bytes(total_bytes)}）\n"
            f"- 物理 schema：{len(physical_groups)} 种（字段及字段顺序完全一致）\n"
            f"- 逻辑 schema：{len(logical_groups)} 种（忽略字段顺序）\n"
            "- 非 Parquet 文件：`README.md`（YAML front matter + Markdown）\n\n"
            "明细见 `inventory.csv`；每个 Parquet 的首条记录见 `samples.jsonl`。"
            "样例未截断，原始 Parquet 文件未修改。\n\n"
        )

        handle.write("## 逻辑 schema 汇总\n\n")
        handle.write("| ID | 文件数 | 字段 | 文件 |\n|---|---:|---|---|\n")
        signature_by_id = {value: key for key, value in logical_ids.items()}
        for logical_id, files in logical_groups.items():
            fields = ", ".join(f"`{name}: {kind}`" for name, kind, _ in signature_by_id[logical_id])
            file_list = ", ".join(f"`{name}`" for name in files)
            handle.write(f"| {logical_id} | {len(files)} | {fields} | {file_list} |\n")

        handle.write("\n## 物理 schema 汇总\n\n")
        handle.write("| ID | 文件数 | 字段（保留原顺序） | 文件 |\n|---|---:|---|---|\n")
        signature_by_id = {value: key for key, value in physical_ids.items()}
        for physical_id, files in physical_groups.items():
            fields = ", ".join(f"`{name}: {kind}`" for name, kind, _ in signature_by_id[physical_id])
            file_list = ", ".join(f"`{name}`" for name in files)
            handle.write(f"| {physical_id} | {len(files)} | {fields} | {file_list} |\n")


if __name__ == "__main__":
    main()
