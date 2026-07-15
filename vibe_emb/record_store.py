from __future__ import annotations

import json
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc


class RecordStore(Protocol):
    """Random-access source returning the legacy query/pos/neg record shape."""
    def __len__(self) -> int: ...
    def get_records(self, indices: Sequence[int]) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...


class JsonRecordStore:
    """In-memory adapter for legacy JSON/JSONL records."""
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def get_records(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        return [self.records[int(index)] for index in indices]

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class ArrowUnitDescriptor:
    """Manifest-only metadata needed to plan batches without opening Arrow files."""
    unit_id: str
    source_id: str
    task_type: str
    query_count: int
    corpus_count: int
    metadata_path: str
    queries_path: str
    corpus_path: str
    queries_file_size: int
    corpus_file_size: int
    ipc_compression: str
    query_record_batch_count: int
    corpus_record_batch_count: int
    metadata_fingerprint: str
    queries_fingerprint: str
    corpus_fingerprint: str

    @classmethod
    def from_manifest(cls, root: Path, value: Dict[str, Any]) -> "ArrowUnitDescriptor":
        kwargs = {field: value[field] for field in cls.__dataclass_fields__}
        for name in ("metadata_path", "queries_path", "corpus_path"):
            kwargs[name] = str((root / kwargs[name]).resolve())
        return cls(**kwargs)


def load_arrow_manifest(path: str) -> tuple[Dict[str, Any], List[ArrowUnitDescriptor]]:
    """Parse enabled units without touching per-unit metadata or IPC footers."""
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "indexed_arrow_v1":
        raise ValueError(f"Unsupported Indexed Arrow schema: {manifest.get('schema_version')}")
    units = [
        ArrowUnitDescriptor.from_manifest(manifest_path.parent, unit)
        for unit in manifest["units"]
        if unit.get("enabled", True)
    ]
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("Duplicate unit IDs in Indexed Arrow manifest")
    return manifest, units


class IndexedArrowRecordStore:
    """Lazily opened query/corpus pair for a single sampling unit.

    ``read_all`` materializes Arrow table metadata and chunk references, not a
    Python copy of every string. The underlying file pages remain mmap-backed
    and are faulted in as selected rows are read.
    """
    def __init__(self, descriptor: ArrowUnitDescriptor, verify_mode: str = "lazy") -> None:
        self.descriptor = descriptor
        self.query_source: pa.MemoryMappedFile | None = None
        self.corpus_source: pa.MemoryMappedFile | None = None
        self.query_reader: ipc.RecordBatchFileReader | None = None
        self.corpus_reader: ipc.RecordBatchFileReader | None = None
        self.query_table: pa.Table | None = None
        self.corpus_table: pa.Table | None = None
        self._open(verify_mode)

    def _open(self, verify_mode: str) -> None:
        descriptor = self.descriptor
        if os.path.getsize(descriptor.queries_path) != descriptor.queries_file_size:
            raise ValueError(f"queries file size mismatch for {descriptor.unit_id}")
        if os.path.getsize(descriptor.corpus_path) != descriptor.corpus_file_size:
            raise ValueError(f"corpus file size mismatch for {descriptor.unit_id}")
        if verify_mode in {"lazy", "full"}:
            metadata = json.loads(Path(descriptor.metadata_path).read_text())
            for field in ("query_count", "corpus_count", "task_type", "ipc_compression"):
                if metadata[field] != getattr(descriptor, field):
                    raise ValueError(f"metadata/manifest {field} mismatch for {descriptor.unit_id}")
        if verify_mode == "full":
            for path, expected in (
                (descriptor.metadata_path, descriptor.metadata_fingerprint),
                (descriptor.queries_path, descriptor.queries_fingerprint),
                (descriptor.corpus_path, descriptor.corpus_fingerprint),
            ):
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    while chunk := handle.read(8 << 20):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"fingerprint mismatch for {path}")
        self.query_source = pa.memory_map(descriptor.queries_path, "r")
        self.corpus_source = pa.memory_map(descriptor.corpus_path, "r")
        self.query_reader = ipc.open_file(self.query_source)
        self.corpus_reader = ipc.open_file(self.corpus_source)
        if self.query_reader.num_record_batches != descriptor.query_record_batch_count:
            raise ValueError(f"query record batch count mismatch for {descriptor.unit_id}")
        if self.corpus_reader.num_record_batches != descriptor.corpus_record_batch_count:
            raise ValueError(f"corpus record batch count mismatch for {descriptor.unit_id}")
        self.query_table = self.query_reader.read_all()
        self.corpus_table = self.corpus_reader.read_all()
        expected_query_fields = {
            "query_id", "sample_id", "source_shard_id", "source_row_index", "query",
            "positive_doc_id", "negative_doc_ids", "lang",
        }
        if set(self.query_table.schema.names) != expected_query_fields:
            raise ValueError(f"queries schema mismatch for {descriptor.unit_id}")
        if self.corpus_table.schema.names != ["doc_id", "text"]:
            raise ValueError(f"corpus schema mismatch for {descriptor.unit_id}")
        if len(self.query_table) != descriptor.query_count or len(self.corpus_table) != descriptor.corpus_count:
            raise ValueError(f"Arrow row count mismatch for {descriptor.unit_id}")

    def __len__(self) -> int:
        return self.descriptor.query_count

    def get_records(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        """Resolve indexed document IDs and return buffer-independent Python values."""
        if self.query_table is None or self.corpus_table is None:
            raise RuntimeError(f"Store is closed: {self.descriptor.unit_id}")
        normalized = [int(index) for index in indices]
        if any(index < 0 or index >= len(self) for index in normalized):
            raise IndexError(f"query index outside unit {self.descriptor.unit_id}")
        query_rows = self.query_table.take(pa.array(normalized, type=pa.int64())).to_pylist()
        # A corpus row's doc_id is also its dense row offset. Fetch the union
        # once so repeated positives/negatives within this batch are decoded
        # only once.
        doc_ids = sorted(
            {row["positive_doc_id"] for row in query_rows}
            | {doc_id for row in query_rows for doc_id in row["negative_doc_ids"]}
        )
        documents = self.corpus_table.take(pa.array(doc_ids, type=pa.int64())).to_pylist()
        text_by_id = {row["doc_id"]: row["text"] for row in documents}
        return [
            {
                "query": row["query"],
                "pos": [text_by_id[row["positive_doc_id"]]],
                "neg": [text_by_id[doc_id] for doc_id in row["negative_doc_ids"]],
                **({"lang": row["lang"]} if row.get("lang") is not None else {}),
                "sample_id": row["sample_id"].hex(),
            }
            for row in query_rows
        ]

    def close(self) -> None:
        self.query_table = None
        self.corpus_table = None
        self.query_reader = None
        self.corpus_reader = None
        for source_name in ("query_source", "corpus_source"):
            source = getattr(self, source_name)
            if source is not None:
                source.close()
                setattr(self, source_name, None)


class ArrowStorePool:
    """Per-process LRU that bounds simultaneously open sampling units."""
    def __init__(self, descriptors: Sequence[ArrowUnitDescriptor], max_open_units: int = 4, verify_mode: str = "lazy") -> None:
        if max_open_units <= 0:
            raise ValueError("arrow_max_open_units must be positive")
        if verify_mode not in {"manifest", "lazy", "full"}:
            raise ValueError(f"Invalid arrow_verify_mode: {verify_mode}")
        self.descriptors = {descriptor.unit_id: descriptor for descriptor in descriptors}
        self.max_open_units = max_open_units
        self.verify_mode = verify_mode
        self.stores: OrderedDict[str, IndexedArrowRecordStore] = OrderedDict()

    def get(self, unit_id: str) -> IndexedArrowRecordStore:
        """Return an open unit, evicting and closing the least-recently used one."""
        if unit_id in self.stores:
            self.stores.move_to_end(unit_id)
            return self.stores[unit_id]
        descriptor = self.descriptors[unit_id]
        store = IndexedArrowRecordStore(descriptor, self.verify_mode)
        self.stores[unit_id] = store
        while len(self.stores) > self.max_open_units:
            _, evicted = self.stores.popitem(last=False)
            evicted.close()
        return store

    def close(self) -> None:
        while self.stores:
            _, store = self.stores.popitem(last=False)
            store.close()
