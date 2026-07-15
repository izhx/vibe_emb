#!/usr/bin/env python3
"""Compile an F2LLM selection manifest into unit-local Indexed Arrow files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from scripts.f2llm.prepare_f2llm_stage2_sampling import sample_id_bytes  # noqa: E402

SCHEMA_VERSION = "indexed_arrow_v1"
QUERY_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.binary(16), nullable=False),
        pa.field("source_shard_id", pa.uint16(), nullable=False),
        pa.field("source_row_index", pa.int64(), nullable=False),
        pa.field("query", pa.large_string(), nullable=False),
        pa.field("positive_doc_id", pa.int64(), nullable=False),
        pa.field("negative_doc_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("lang", pa.string()),
    ]
)
CORPUS_SCHEMA = pa.schema(
    [pa.field("doc_id", pa.int64(), nullable=False), pa.field("text", pa.large_string(), nullable=False)]
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode())
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _negative_columns(shard: dict[str, Any]) -> list[str]:
    names = {field["name"] for field in shard["schema"]}
    result = [name for name in names if name.startswith("negative_")]
    return sorted(result, key=lambda value: int(value.removeprefix("negative_")))


def _selected_tables(
    data_root: Path,
    shard: dict[str, Any],
    selected: np.ndarray,
    columns: list[str],
    batch_size: int,
) -> Iterator[tuple[np.ndarray, pa.Table]]:
    """Yield selected physical row numbers and tables without reading unused row groups."""
    parquet = pq.ParquetFile(data_root / shard["relative_path"])
    # Selection archives store physical shard-local row numbers, including
    # catalog row-range offsets for consolidated release files.
    physical = selected
    cursor = 0
    group_start = 0
    for group_id, group_rows in enumerate(shard["row_group_rows"]):
        group_stop = group_start + int(group_rows)
        left = int(np.searchsorted(physical, group_start, side="left"))
        right = int(np.searchsorted(physical, group_stop, side="left"))
        if right > left:
            offsets = pa.array(physical[left:right] - group_start, type=pa.int64())
            table = parquet.read_row_group(group_id, columns=columns).take(offsets)
            selected_rows = physical[left:right]
            for start in range(0, len(table), batch_size):
                yield selected_rows[start : start + batch_size], table.slice(start, batch_size)
            cursor += right - left
        group_start = group_stop
    if cursor != len(selected):
        raise ValueError(f"Could not locate every selected row in {shard['relative_path']}: {cursor}/{len(selected)}")


class CorpusIndex:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute(
            "CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, digest BLOB NOT NULL, text TEXT NOT NULL, UNIQUE(digest, text))"
        )
        self.next_doc_id = 0

    def ids(self, texts: Iterable[str]) -> list[int]:
        result: list[int] = []
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("positive and negative documents must be non-empty strings")
            digest = hashlib.blake2b(text.encode(), digest_size=16, person=b"f2llm-doc-v1").digest()
            row = self.connection.execute(
                "SELECT doc_id FROM docs WHERE digest = ? AND text = ?", (digest, text)
            ).fetchone()
            if row is None:
                collision = self.connection.execute(
                    "SELECT text FROM docs WHERE digest = ? LIMIT 1", (digest,)
                ).fetchone()
                if collision is not None and collision[0] != text:
                    raise ValueError("Document digest collision detected")
                self.connection.execute(
                    "INSERT INTO docs(doc_id, digest, text) VALUES (?, ?, ?)",
                    (self.next_doc_id, digest, text),
                )
                row = (self.next_doc_id,)
                self.next_doc_id += 1
            result.append(int(row[0]))
        return result

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def rows(self, batch_size: int) -> Iterator[list[tuple[int, str]]]:
        cursor = self.connection.execute("SELECT doc_id, text FROM docs ORDER BY doc_id")
        while rows := cursor.fetchmany(batch_size):
            yield [(int(row[0]), str(row[1])) for row in rows]

    def close(self) -> None:
        self.connection.close()


def _ipc_options(compression: str) -> ipc.IpcWriteOptions:
    return ipc.IpcWriteOptions(compression=None if compression == "none" else compression)


def build_unit(
    selection_dir: Path,
    selection: dict[str, Any],
    unit: dict[str, Any],
    output_dir: Path,
    data_root: Path,
    record_batch_size: int,
    compression: str,
    force: bool,
    keep_build_db: bool,
    build_db_dir: Path | None,
) -> dict[str, Any]:
    unit_id = unit["sampling_unit_id"]
    target = output_dir / "units" / unit_id
    if target.exists() and not force:
        raise FileExistsError(f"Unit already exists (use --force): {target}")
    output_dir.joinpath("units").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{unit_id}-", dir=output_dir / "units"))
    # SQLite performs many small random reads/writes while deduplicating the
    # corpus.  Keeping that scratch database off a shared filesystem makes a
    # large build dramatically faster; only the final Arrow files belong in
    # the output directory.
    db_tmp = Path(tempfile.mkdtemp(prefix=f"f2llm-{unit_id}-", dir=build_db_dir))
    db_path = db_tmp / "corpus.sqlite"
    corpus = CorpusIndex(db_path)
    query_path = tmp / "queries.arrow"
    query_count = 0
    query_batches = 0
    dropped_invalid_query_or_positive_count = 0
    filtered_invalid_negative_count = 0
    source_shards: list[dict[str, Any]] = []
    index_path = selection_dir / unit["index_file"]
    try:
        with np.load(index_path, allow_pickle=False) as archive, pa.OSFile(str(query_path), "wb") as sink:
            with ipc.new_file(sink, QUERY_SCHEMA, options=_ipc_options(compression)) as writer:
                for shard_number, shard in enumerate(unit["shards"]):
                    relative_path = shard["relative_path"]
                    if relative_path not in archive.files:
                        raise ValueError(f"Missing {relative_path} in {index_path}")
                    selected = archive[relative_path]
                    negative_columns = _negative_columns(shard)
                    columns = ["query", "passage", *negative_columns]
                    has_lang = any(field["name"] == "lang" for field in shard["schema"])
                    if has_lang:
                        columns.append("lang")
                    for row_numbers, table in _selected_tables(
                        data_root, shard, selected, columns, record_batch_size
                    ):
                        payload = {name: table[name].to_pylist() for name in columns}
                        rows: dict[str, list[Any]] = {name: [] for name in QUERY_SCHEMA.names}
                        for offset, physical_row in enumerate(row_numbers):
                            query = payload["query"][offset]
                            positive = payload["passage"][offset]
                            if not isinstance(query, str) or not query or not isinstance(positive, str) or not positive:
                                dropped_invalid_query_or_positive_count += 1
                                continue
                            negatives = [payload[name][offset] for name in negative_columns]
                            valid_negatives = [text for text in negatives if isinstance(text, str) and text]
                            filtered_invalid_negative_count += len(negatives) - len(valid_negatives)
                            texts = [positive, *valid_negatives]
                            doc_ids = corpus.ids(texts)
                            rows["query_id"].append(query_count)
                            rows["sample_id"].append(
                                sample_id_bytes(selection["dataset_release_id"], relative_path, int(physical_row))
                            )
                            rows["source_shard_id"].append(shard_number)
                            rows["source_row_index"].append(int(physical_row))
                            rows["query"].append(query)
                            rows["positive_doc_id"].append(doc_ids[0])
                            rows["negative_doc_ids"].append(doc_ids[1:])
                            rows["lang"].append(payload["lang"][offset] if has_lang else unit.get("language"))
                            query_count += 1
                        if rows["query_id"]:
                            writer.write_batch(pa.RecordBatch.from_pydict(rows, schema=QUERY_SCHEMA))
                            query_batches += 1
                    source_shards.append(
                        {
                            "shard_id": shard_number,
                            "path": relative_path,
                            "row_count": shard["num_rows"],
                            "physical_num_rows": shard["physical_num_rows"],
                            "row_start": shard.get("row_start", 0),
                            "row_stop": shard.get("row_stop"),
                            "metadata_fingerprint": shard["metadata_fingerprint"],
                        }
                    )
        if query_count + dropped_invalid_query_or_positive_count != int(unit["selected_count"]):
            raise ValueError(
                f"Query accounting mismatch for {unit_id}: {query_count} valid + "
                f"{dropped_invalid_query_or_positive_count} invalid != {unit['selected_count']} selected"
            )
        corpus.commit()
        corpus_path = tmp / "corpus.arrow"
        corpus_batches = 0
        with pa.OSFile(str(corpus_path), "wb") as sink:
            with ipc.new_file(sink, CORPUS_SCHEMA, options=_ipc_options(compression)) as writer:
                for batch in corpus.rows(record_batch_size):
                    writer.write_batch(
                        pa.RecordBatch.from_pydict(
                            {"doc_id": [row[0] for row in batch], "text": [row[1] for row in batch]},
                            schema=CORPUS_SCHEMA,
                        )
                    )
                    corpus_batches += 1
        corpus_count = corpus.count()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset_release_id": selection["dataset_release_id"],
            "profile_id": selection["profile_id"],
            "source_id": unit["source_id"],
            "sampling_unit_id": unit_id,
            "task_type": unit["task_type"],
            "language": unit.get("language"),
            "query_count": query_count,
            "selected_query_count": int(unit["selected_count"]),
            "dropped_invalid_query_or_positive_count": dropped_invalid_query_or_positive_count,
            "filtered_invalid_negative_count": filtered_invalid_negative_count,
            "corpus_count": corpus_count,
            "available_negative_count": max(len(_negative_columns(shard)) for shard in unit["shards"]),
            "physical_shards": [s["relative_path"] for s in unit["shards"]],
            "source_shards": source_shards,
            "sample_limit": unit["sample_limit"],
            "sampling_seed": selection["global_seed"],
            "selection_fingerprint": unit["selection_fingerprint"],
            "ipc_compression": compression,
            "query_record_batch_count": query_batches,
            "corpus_record_batch_count": corpus_batches,
            "queries_file_size": query_path.stat().st_size,
            "corpus_file_size": corpus_path.stat().st_size,
            "queries_fingerprint": sha256_file(query_path),
            "corpus_fingerprint": sha256_file(corpus_path),
            "sample_id_algorithm": selection["sample_id_algorithm"],
            "pyarrow_version": pa.__version__,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(tmp / "metadata.json", metadata)
        metadata["metadata_fingerprint"] = sha256_file(tmp / "metadata.json")
        corpus.close()
        if keep_build_db:
            shutil.copy2(db_path, tmp / "corpus.sqlite")
        shutil.rmtree(db_tmp)
        if target.exists():
            backup = target.with_name(f".{unit_id}.old-{os.getpid()}")
            os.replace(target, backup)
            try:
                os.replace(tmp, target)
            except BaseException:
                os.replace(backup, target)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(tmp, target)
        return metadata
    except BaseException:
        corpus.close()
        shutil.rmtree(db_tmp, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def manifest_unit(metadata: dict[str, Any]) -> dict[str, Any]:
    unit_id = metadata["sampling_unit_id"]
    return {
        "unit_id": unit_id,
        "source_id": metadata["source_id"],
        "task_type": metadata["task_type"],
        "language": metadata.get("language"),
        "enabled": True,
        "directory": f"units/{unit_id}",
        "metadata_path": f"units/{unit_id}/metadata.json",
        "queries_path": f"units/{unit_id}/queries.arrow",
        "corpus_path": f"units/{unit_id}/corpus.arrow",
        **{key: metadata[key] for key in (
            "query_count", "corpus_count", "ipc_compression", "query_record_batch_count",
            "corpus_record_batch_count", "queries_file_size", "corpus_file_size",
            "metadata_fingerprint", "queries_fingerprint", "corpus_fingerprint",
        )},
        **{key: metadata[key] for key in (
            "selected_query_count", "dropped_invalid_query_or_positive_count",
            "filtered_invalid_negative_count",
        ) if key in metadata},
    }


def load_existing_unit(output_dir: Path, selection: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    metadata_path = output_dir / "units" / unit["sampling_unit_id"] / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Cannot resume incomplete unit without metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "profile_id": selection["profile_id"],
        "dataset_release_id": selection["dataset_release_id"],
        "sampling_unit_id": unit["sampling_unit_id"],
        "selection_fingerprint": unit["selection_fingerprint"],
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"Cannot resume stale unit {unit['sampling_unit_id']}: {mismatches}")
    for name in ("queries.arrow", "corpus.arrow"):
        if not (metadata_path.parent / name).is_file():
            raise ValueError(f"Cannot resume incomplete unit missing {name}: {metadata_path.parent}")
    metadata["metadata_fingerprint"] = sha256_file(metadata_path)
    return metadata


def build_manifest(output_dir: Path, selection_path: Path, selection: dict[str, Any], built: list[dict[str, Any]]) -> None:
    built_units = [manifest_unit(value) for value in built]
    existing_path = output_dir / "manifest.json"
    existing_by_id: dict[str, dict[str, Any]] = {}
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing.get("profile_id") != selection["profile_id"]:
            raise ValueError("Existing output manifest belongs to another profile")
        existing_by_id = {unit["unit_id"]: unit for unit in existing.get("units", [])}
    existing_by_id.update({unit["unit_id"]: unit for unit in built_units})
    selection_order = {
        unit["sampling_unit_id"]: index for index, unit in enumerate(selection["units"])
    }
    units = sorted(existing_by_id.values(), key=lambda unit: selection_order[unit["unit_id"]])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": selection["profile_id"],
        "dataset_release_id": selection["dataset_release_id"],
        "catalog_digest_sha256": selection["catalog_digest_sha256"],
        "selection_manifest": os.path.relpath(selection_path, output_dir),
        "selection_manifest_fingerprint": sha256_file(selection_path),
        "total_query_count": sum(unit["query_count"] for unit in units),
        "total_corpus_count": sum(unit["corpus_count"] for unit in units),
        "units": units,
    }
    manifest["manifest_fingerprint"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_json(output_dir / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--unit", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed units matching this selection")
    parser.add_argument("--record-batch-size", type=int, default=4096)
    parser.add_argument("--compression", choices=("none", "zstd"), default="none")
    parser.add_argument("--keep-build-db", action="store_true")
    parser.add_argument(
        "--build-db-dir", type=Path,
        help="Local scratch directory for SQLite corpus deduplication (defaults to system temp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_path = args.selection_manifest.resolve()
    selection = json.loads(selection_path.read_text())
    data_root = (args.data_root or Path(selection["data_root"])).resolve()
    wanted = set(args.unit)
    units = [u for u in selection["units"] if not u["excluded"] and (not wanted or u["sampling_unit_id"] in wanted)]
    missing = wanted - {u["sampling_unit_id"] for u in units}
    if missing:
        raise ValueError(f"Unknown or excluded units: {sorted(missing)}")
    if args.record_batch_size <= 0:
        raise ValueError("--record-batch-size must be positive")
    if args.build_db_dir is not None:
        args.build_db_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for unit in units:
        target = args.output_dir.resolve() / "units" / unit["sampling_unit_id"]
        if args.resume and target.exists():
            print(f"reusing {unit['sampling_unit_id']}", flush=True)
            built.append(load_existing_unit(args.output_dir.resolve(), selection, unit))
            continue
        print(f"building {unit['sampling_unit_id']} ({unit['selected_count']} queries)", flush=True)
        built.append(build_unit(selection_path.parent, selection, unit, args.output_dir.resolve(), data_root,
                                args.record_batch_size, args.compression, args.force, args.keep_build_db,
                                args.build_db_dir))
    build_manifest(args.output_dir.resolve(), selection_path, selection, built)
    print(f"wrote {args.output_dir / 'manifest.json'} with {len(built)} units")


if __name__ == "__main__":
    main()
