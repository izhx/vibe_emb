#!/usr/bin/env python3
"""Build a deterministic, auditable F2LLM-v2 second-stage selection plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from io import BytesIO
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
SAMPLE_ID_ALGORITHM = "blake2b-128-f2llm-locator-v1"
SELECTION_ALGORITHM = "numpy-pcg64-logical-row-v1"
EXPECTED_ML_EMBED_SOURCES = 121


class CatalogError(ValueError):
    """Raised when the catalog or the selected profile is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_digest(value: Any, *, digest_size: int = 32) -> str:
    return hashlib.blake2b(canonical_json(value), digest_size=digest_size).hexdigest()


def normalize_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise CatalogError(f"invalid relative shard path: {value!r}")
    return str(path)


def parse_shard_spec(value: Any) -> tuple[str, int | None, int | None]:
    if isinstance(value, str):
        return normalize_relative_path(value), None, None
    if not isinstance(value, dict) or "path" not in value:
        raise CatalogError("shards must be paths or {path, row_start, row_stop} mappings")
    path = normalize_relative_path(str(value["path"]))
    start = int(value.get("row_start", 0))
    stop_value = value.get("row_stop")
    stop = int(stop_value) if stop_value is not None else None
    if start < 0 or (stop is not None and stop <= start):
        raise CatalogError(f"invalid row range for {path}: [{start}, {stop})")
    return path, start, stop


def sample_id_bytes(dataset_release_id: str, shard_path: str, row_index: int) -> bytes:
    """Return the stable 16-byte ID for one physical source row."""
    shard_path = normalize_relative_path(shard_path)
    hasher = hashlib.blake2b(digest_size=16, person=b"f2llm-row-id-v1")
    for value in (dataset_release_id.encode(), shard_path.encode()):
        hasher.update(len(value).to_bytes(4, "big"))
        hasher.update(value)
    hasher.update(int(row_index).to_bytes(8, "big", signed=False))
    return hasher.digest()


def unit_seed(global_seed: int, profile_id: str, unit_id: str) -> int:
    payload = [int(global_seed), profile_id, unit_id]
    return int.from_bytes(hashlib.blake2b(canonical_json(payload), digest_size=8).digest(), "big")


def select_logical_rows(total_rows: int, sample_limit: int, seed: int) -> np.ndarray:
    if total_rows < 0 or sample_limit <= 0:
        raise ValueError("total_rows must be non-negative and sample_limit must be positive")
    if total_rows <= sample_limit:
        return np.arange(total_rows, dtype=np.int64)
    selected = np.random.Generator(np.random.PCG64(seed)).choice(
        total_rows, size=sample_limit, replace=False
    )
    selected.sort()
    return selected.astype(np.int64, copy=False)


def split_logical_indices(
    indices: np.ndarray, shard_rows: Iterable[int]
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    offset = 0
    for rows in shard_rows:
        end = offset + int(rows)
        left = int(np.searchsorted(indices, offset, side="left"))
        right = int(np.searchsorted(indices, end, side="left"))
        result.append((indices[left:right] - offset).astype(np.int64, copy=False))
        offset = end
    if len(indices) and int(indices[-1]) >= offset:
        raise ValueError("logical row index is outside the combined shards")
    return result


def locate_row_group(row_index: int, row_group_rows: Iterable[int]) -> tuple[int, int]:
    """Map a shard-local row index to (row group index, offset in row group)."""
    offset = 0
    for group_index, rows in enumerate(row_group_rows):
        end = offset + int(rows)
        if offset <= row_index < end:
            return group_index, row_index - offset
        offset = end
    raise IndexError(f"row index {row_index} outside shard with {offset} rows")


def schema_summary(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def core_schema(schema: pa.Schema) -> dict[str, str]:
    names = set(schema.names)
    result: dict[str, str] = {}
    for name in ("query", "passage"):
        if name in names:
            result[name] = str(schema.field(name).type)
    negatives = sorted(
        (name for name in names if name.startswith("negative_")),
        key=lambda name: int(name.removeprefix("negative_")),
    )
    result["negative_count"] = str(len(negatives))
    for name in negatives:
        result[name] = str(schema.field(name).type)
    return result


def parquet_metadata(path: Path, relative_path: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    row_group_rows = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    summary = schema_summary(parquet.schema_arrow)
    fingerprint_payload = {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "num_rows": metadata.num_rows,
        "row_group_rows": row_group_rows,
        "schema": summary,
        "created_by": metadata.created_by,
    }
    return {
        **fingerprint_payload,
        "num_row_groups": metadata.num_row_groups,
        "schema_fingerprint": stable_digest(summary),
        "metadata_fingerprint": stable_digest(fingerprint_payload),
        "core_schema": core_schema(parquet.schema_arrow),
    }


def load_catalog(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    catalog = yaml.safe_load(raw)
    if not isinstance(catalog, dict):
        raise CatalogError("catalog root must be a mapping")
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(
            f"unsupported catalog schema_version {catalog.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not catalog.get("dataset_release_id"):
        raise CatalogError("catalog must define dataset_release_id")
    if not isinstance(catalog.get("sources"), list):
        raise CatalogError("catalog sources must be a list")
    if not isinstance(catalog.get("profiles"), dict):
        raise CatalogError("catalog profiles must be a mapping")
    return catalog, raw


def validate_catalog(catalog: dict[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        profile = catalog["profiles"][profile_id]
    except KeyError as exc:
        raise CatalogError(f"unknown profile: {profile_id}") from exc
    if not isinstance(profile, dict):
        raise CatalogError(f"profile {profile_id!r} must be a mapping")
    if int(profile.get("sample_limit", 0)) <= 0:
        raise CatalogError(f"profile {profile_id!r} has an invalid sample_limit")

    source_ids: set[str] = set()
    unit_ids: set[str] = set()
    shard_owners: dict[str, list[tuple[int | None, int | None, str]]] = {}
    enabled_sources = set(profile.get("enabled_sources", []))
    known_sources: set[str] = set()
    for source in catalog["sources"]:
        if not isinstance(source, dict) or not source.get("source_id"):
            raise CatalogError("every source must be a mapping with source_id")
        source_id = str(source["source_id"])
        if source_id in source_ids:
            raise CatalogError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        known_sources.add(source_id)
        if source.get("task_type") not in {"retrieval", "clustering", "classification"}:
            raise CatalogError(f"invalid task_type for source {source_id}")
        units = source.get("sampling_units", [])
        if not units and not source.get("optional_missing"):
            raise CatalogError(f"source {source_id} has no sampling units")
        for unit in units:
            unit_id = str(unit.get("sampling_unit_id", ""))
            if not unit_id or unit_id in unit_ids:
                raise CatalogError(f"missing or duplicate sampling_unit_id: {unit_id!r}")
            unit_ids.add(unit_id)
            shards = unit.get("shards")
            if not isinstance(shards, list) or not shards:
                raise CatalogError(f"unit {unit_id} must have a non-empty shards list")
            for raw_path in shards:
                shard, start, stop = parse_shard_spec(raw_path)
                owners = shard_owners.setdefault(shard, [])
                for other_start, other_stop, other_unit in owners:
                    if None in (start, stop, other_start, other_stop):
                        raise CatalogError(
                            f"whole shard {shard!r} belongs to both {other_unit!r} and {unit_id!r}"
                        )
                    if max(start, other_start) < min(stop, other_stop):
                        raise CatalogError(
                            f"overlapping {shard!r} ranges belong to {other_unit!r} and {unit_id!r}"
                        )
                owners.append((start, stop, unit_id))
    unknown = enabled_sources - known_sources
    if unknown:
        raise CatalogError(f"profile enables unknown sources: {sorted(unknown)}")
    expected_source_count = int(
        profile.get(
            "expected_source_count",
            EXPECTED_ML_EMBED_SOURCES if profile_id == "ml_embed_stage2" else 0,
        )
    )
    if expected_source_count and len(enabled_sources) != expected_source_count:
        raise CatalogError(
            f"{profile_id} must contain {expected_source_count} paper sources, "
            f"found {len(enabled_sources)}"
        )
    source_by_id = {str(source["source_id"]): source for source in catalog["sources"]}
    overrides = profile.get("source_unit_overrides", {})
    if not isinstance(overrides, dict):
        raise CatalogError(f"profile {profile_id!r} source_unit_overrides must be a mapping")
    for source_id, override_units in overrides.items():
        if source_id not in enabled_sources or source_id not in source_by_id:
            raise CatalogError(f"unit override refers to disabled or unknown source: {source_id}")
        if not isinstance(override_units, list) or not override_units:
            raise CatalogError(f"source unit override for {source_id} must be a non-empty list")
        base_shards = sorted(
            parse_shard_spec(shard)
            for unit in source_by_id[source_id]["sampling_units"]
            for shard in unit["shards"]
        )
        override_shards = sorted(
            parse_shard_spec(shard)
            for unit in override_units
            for shard in unit.get("shards", [])
        )
        if override_shards != base_shards:
            raise CatalogError(
                f"source unit override for {source_id} must cover exactly its catalog shards"
            )
    return profile


def validate_task_schema(task_type: str, metadata: dict[str, Any], unit_id: str) -> None:
    core = metadata["core_schema"]
    for name in ("query", "passage"):
        if core.get(name) != "string":
            raise CatalogError(f"unit {unit_id}: {name} must be a string column")
    if int(core["negative_count"]) < 1:
        raise CatalogError(f"unit {unit_id}: {task_type} data requires negative_1")


def validate_old_plan(output_dir: Path) -> None:
    manifest_path = output_dir / "selection_manifest.json"
    report_path = output_dir / "sampling_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise FileExistsError(
            f"refusing --force because {output_dir} is not a complete sampling plan"
        )
    manifest = json.loads(manifest_path.read_text())
    for unit in manifest.get("units", []):
        index_file = unit.get("index_file")
        if not index_file:
            continue
        path = output_dir / index_file
        if not path.is_file():
            raise FileExistsError(f"refusing --force because old index is missing: {index_file}")
        expected_shards = {
            shard["relative_path"]: shard for shard in unit.get("shards", [])
        }
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(expected_shards):
                    raise ValueError("shard keys differ from the manifest")
                count = 0
                for key in arrays.files:
                    values = arrays[key]
                    shard = expected_shards[key]
                    if values.dtype != np.int64 or values.ndim != 1:
                        raise ValueError(f"{key} does not contain a 1-D int64 array")
                    if len(values) and (
                        int(values[0]) < int(shard.get("row_start", 0))
                        or int(values[-1]) >= int(shard.get("row_stop", shard["num_rows"]))
                        or np.any(values[1:] <= values[:-1])
                    ):
                        raise ValueError(f"{key} contains invalid row indices")
                    count += len(values)
                if count != int(unit["selected_count"]):
                    raise ValueError("selected count differs from the manifest")
        except (OSError, ValueError, KeyError) as exc:
            raise FileExistsError(
                f"refusing --force because old index is invalid: {index_file}: {exc}"
            ) from exc


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def deterministic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ without wall-clock timestamps or unstable key order."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for key in sorted(arrays):
                buffer = BytesIO()
                np.lib.format.write_array(buffer, arrays[key], allow_pickle=False)
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def clean_output_for_force(output_dir: Path) -> None:
    for name in ("selection_manifest.json", "sampling_report.json", "catalog_snapshot.yaml"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    indices = output_dir / "indices"
    if indices.exists():
        shutil.rmtree(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/F2LLM-v2"))
    parser.add_argument(
        "--catalog", type=Path, default=Path("configs/data/f2llm_v2_sources.yaml")
    )
    parser.add_argument("--profile", default="ml_embed_stage2")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/ml_embed_stage2_selection")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--include-contaminated", action="store_true")
    parser.add_argument("--include-unit", action="append", default=[])
    parser.add_argument("--exclude-unit", action="append", default=[])
    parser.add_argument("--unit", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog, catalog_raw = load_catalog(args.catalog)
    profile = validate_catalog(catalog, args.profile)
    sample_limit = args.sample_limit or int(profile["sample_limit"])
    if sample_limit <= 0:
        raise CatalogError("--sample-limit must be positive")
    include_units = set(args.include_unit)
    exclude_units = set(args.exclude_unit)
    only_units = set(args.unit)
    if include_units & exclude_units:
        raise CatalogError("the same unit cannot be both included and excluded")

    existing = args.output_dir.exists() and any(args.output_dir.iterdir())
    if existing:
        if not args.force:
            raise FileExistsError(f"output directory is not empty: {args.output_dir}")
        validate_old_plan(args.output_dir)

    enabled_sources = set(profile["enabled_sources"])
    contaminated_sources = set(profile.get("excluded_contaminated_sources", []))
    source_unit_overrides = profile.get("source_unit_overrides", {})
    known_units = {
        str(unit["sampling_unit_id"])
        for source in catalog["sources"]
        for unit in source_unit_overrides.get(
            str(source["source_id"]), source.get("sampling_units", [])
        )
    }
    unknown_overrides = (include_units | exclude_units | only_units) - known_units
    if unknown_overrides:
        raise CatalogError(f"unknown unit override(s): {sorted(unknown_overrides)}")

    units: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_optional_sources: list[dict[str, str]] = []
    seen_shards: set[tuple[str, int | None, int | None]] = set()
    for source in catalog["sources"]:
        source_id = str(source["source_id"])
        if source_id not in enabled_sources:
            continue
        source_contaminated = source_id in contaminated_sources
        if source.get("optional_missing"):
            missing_optional_sources.append(
                {"source_id": source_id, "reason": str(source["optional_missing"])}
            )
            warnings.append(f"optional paper source unavailable: {source_id}")
            continue
        effective_units = source_unit_overrides.get(source_id, source["sampling_units"])
        for unit in effective_units:
            unit_id = str(unit["sampling_unit_id"])
            if only_units and unit_id not in only_units:
                continue
            excluded_reason: str | None = None
            if source_contaminated and not args.include_contaminated:
                excluded_reason = "benchmark contamination"
            if unit_id in include_units:
                excluded_reason = None
            if unit_id in exclude_units:
                excluded_reason = "explicit --exclude-unit"

            shard_records: list[dict[str, Any]] = []
            for raw_shard in unit["shards"]:
                shard, row_start, row_stop = parse_shard_spec(raw_shard)
                locator = (shard, row_start, row_stop)
                if locator in seen_shards:
                    raise CatalogError(f"selected shard range is mapped more than once: {locator}")
                seen_shards.add(locator)
                path = args.data_root / shard
                if not path.is_file():
                    raise FileNotFoundError(f"catalog shard does not exist: {path}")
                metadata = parquet_metadata(path, shard)
                physical_num_rows = int(metadata["num_rows"])
                effective_start = row_start or 0
                effective_stop = physical_num_rows if row_stop is None else row_stop
                if effective_stop > physical_num_rows:
                    raise CatalogError(
                        f"range [{effective_start}, {effective_stop}) exceeds {shard} "
                        f"with {physical_num_rows} rows"
                    )
                metadata["physical_num_rows"] = physical_num_rows
                metadata["row_start"] = effective_start
                metadata["row_stop"] = effective_stop
                metadata["num_rows"] = effective_stop - effective_start
                validate_task_schema(str(source["task_type"]), metadata, unit_id)
                shard_records.append(
                    {
                        "shard_id": hashlib.blake2b(
                            shard.encode(), digest_size=8, person=b"f2llm-shard-v1"
                        ).hexdigest(),
                        **metadata,
                    }
                )
            schemas = {stable_digest(record["core_schema"]) for record in shard_records}
            if len(schemas) != 1:
                raise CatalogError(f"unit {unit_id} has incompatible core schemas")
            units.append(
                {
                    "source_id": source_id,
                    "paper_name": source["paper_name"],
                    "sampling_unit_id": unit_id,
                    "task_type": source["task_type"],
                    "language": unit.get("language", source.get("language")),
                    "paper_size": source.get("paper_size"),
                    "excluded": excluded_reason is not None,
                    "exclusion_reason": excluded_reason,
                    "shards": shard_records,
                }
            )

    if only_units and not units:
        raise CatalogError("--unit selection did not retain any units")

    output_units: list[dict[str, Any]] = []
    report_units: list[dict[str, Any]] = []
    total_selected = 0
    task_counts: Counter[str] = Counter()
    active_source_ids: set[str] = set()
    for unit in units:
        total_rows = sum(int(shard["num_rows"]) for shard in unit["shards"])
        selected_by_shard: list[np.ndarray]
        if unit["excluded"]:
            selected_by_shard = [np.empty(0, dtype=np.int64) for _ in unit["shards"]]
        else:
            logical = select_logical_rows(
                total_rows,
                sample_limit,
                unit_seed(args.seed, args.profile, unit["sampling_unit_id"]),
            )
            selected_by_shard = split_logical_indices(
                logical, (shard["num_rows"] for shard in unit["shards"])
            )
            selected_by_shard = [
                values + int(shard["row_start"])
                for shard, values in zip(unit["shards"], selected_by_shard, strict=True)
            ]
        selected_count = sum(len(values) for values in selected_by_shard)
        if total_rows <= sample_limit and not unit["excluded"] and selected_count != total_rows:
            raise AssertionError("small unit was not selected in full")
        if total_rows < sample_limit and not unit["excluded"]:
            warnings.append(
                f"unit {unit['sampling_unit_id']} has only {total_rows:,} rows; retaining all"
            )
        per_shard = []
        fingerprint = hashlib.blake2b(digest_size=32, person=b"f2llm-select-v1")
        for shard, values in zip(unit["shards"], selected_by_shard, strict=True):
            if len(values) and (
                int(values[0]) < int(shard["row_start"])
                or int(values[-1]) >= int(shard["row_stop"])
                or np.any(values[1:] <= values[:-1])
            ):
                raise AssertionError(f"invalid selected indices for {shard['relative_path']}")
            fingerprint.update(shard["relative_path"].encode())
            fingerprint.update(values.astype("<i8", copy=False).tobytes())
            per_shard.append(
                {
                    "relative_path": shard["relative_path"],
                    "rows": shard["num_rows"],
                    "physical_rows": shard["physical_num_rows"],
                    "row_start": shard["row_start"],
                    "row_stop": shard["row_stop"],
                    "selected": len(values),
                }
            )

        index_relpath: str | None = None
        if not args.dry_run and not unit["excluded"]:
            index_relpath = f"indices/{unit['sampling_unit_id']}.npz"
        manifest_unit = {
            key: unit[key]
            for key in (
                "source_id",
                "paper_name",
                "sampling_unit_id",
                "task_type",
                "language",
                "excluded",
                "exclusion_reason",
            )
        }
        manifest_unit.update(
            {
                "sample_limit": sample_limit,
                "total_rows": total_rows,
                "selected_count": selected_count,
                "index_file": index_relpath,
                "shards": unit["shards"],
                "selection_fingerprint": fingerprint.hexdigest(),
            }
        )
        output_units.append(manifest_unit)
        report_units.append(
            {
                **{key: manifest_unit[key] for key in manifest_unit if key != "shards"},
                "paper_size": unit["paper_size"],
                "shards": per_shard,
                "schema_summary": unit["shards"][0]["core_schema"],
                "warnings": [],
                "sample_id": {
                    "algorithm": SAMPLE_ID_ALGORITHM,
                    "count": selected_count,
                    "unique": True,
                    "uniqueness_check": "unique normalized locator inputs",
                },
            }
        )
        if not unit["excluded"]:
            total_selected += selected_count
            task_counts[str(unit["task_type"])] += selected_count
            active_source_ids.add(str(unit["source_id"]))
        unit["_selected_by_shard"] = selected_by_shard

    audit_query_count = int(profile.get("paper_audit_query_count", 0))
    if not only_units and audit_query_count and total_selected != audit_query_count:
        warnings.append(
            f"selected query total {total_selected:,} differs from the paper-level "
            f"audit target {audit_query_count:,}; no global scaling was applied"
        )

    catalog_digest = hashlib.sha256(catalog_raw).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "dry_run": args.dry_run,
        "dataset_release_id": catalog["dataset_release_id"],
        "profile_id": args.profile,
        "catalog_version": catalog.get("catalog_version"),
        "catalog_digest_sha256": catalog_digest,
        "data_root": str(args.data_root.resolve()),
        "global_seed": args.seed,
        "default_sample_limit": sample_limit,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sample_id_algorithm": SAMPLE_ID_ALGORITHM,
        "overrides": {
            "include_contaminated": args.include_contaminated,
            "include_units": sorted(include_units),
            "exclude_units": sorted(exclude_units),
            "only_units": sorted(only_units),
        },
        "missing_optional_sources": missing_optional_sources,
        "units": output_units,
    }
    summary = {
        "paper_catalog_source_count": len(enabled_sources),
        "available_source_count": len(
            {unit["source_id"] for unit in units}
        ),
        "active_source_count": len(active_source_ids),
        "unit_count": len(units),
        "active_unit_count": sum(not unit["excluded"] for unit in units),
        "physical_shard_count": len(
            {shard["relative_path"] for unit in units for shard in unit["shards"]}
        ),
        "task_source_counts": dict(
            sorted(
                Counter(
                    source["task_type"]
                    for source in catalog["sources"]
                    if source["source_id"] in enabled_sources
                ).items()
            )
        ),
        "task_unit_counts": dict(
            sorted(Counter(unit["task_type"] for unit in units if not unit["excluded"]).items())
        ),
        "task_selected_counts": dict(sorted(task_counts.items())),
        "selected_query_count": total_selected,
        "paper_audit_query_count": audit_query_count,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "dry_run": args.dry_run,
        "profile_id": args.profile,
        "units": report_units,
        "missing_optional_sources": missing_optional_sources,
        "warnings": warnings,
        "summary": summary,
    }

    if existing:
        clean_output_for_force(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        indices_dir = args.output_dir / "indices"
        indices_dir.mkdir(parents=True, exist_ok=True)
        for unit in units:
            if unit["excluded"]:
                continue
            arrays = {
                shard["relative_path"]: values
                for shard, values in zip(
                    unit["shards"], unit["_selected_by_shard"], strict=True
                )
            }
            final_path = indices_dir / f"{unit['sampling_unit_id']}.npz"
            deterministic_savez(final_path, arrays)
    write_json(args.output_dir / "selection_manifest.json", manifest)
    write_json(args.output_dir / "sampling_report.json", report)
    (args.output_dir / "catalog_snapshot.yaml").write_bytes(catalog_raw)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
