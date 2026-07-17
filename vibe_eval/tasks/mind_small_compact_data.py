from __future__ import annotations

"""Build, validate, and expose the CLI for compact MindSmallReranking data.

The source repeats query text, candidates, and qrels for every subquery. This
module deduplicates query text and stores shared candidates once per impression,
while retaining every ID required to reconstruct the full MTEB inputs.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from datasets import Dataset

COMPACT_DATA_DIR = Path("/mnt/share/emb/mteb_fix/mindsmallrerank")
DEFAULT_CACHE_DIR = Path("/mnt/share/emb/mteb_cache")
FORMAT_VERSION = 1
SOURCE_DATASET = "mteb/MindSmallReranking"
SOURCE_REVISION = "227478e3235572039f4f7661840e059f31ef6eb1"
REQUIRED_FILES = (
    "queries.parquet",
    "corpus.parquet",
    "impressions.parquet",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactMindSmallData:
    """The three compact Arrow tables and manifest loaded for evaluation."""
    queries: pa.Table
    corpus: pa.Table
    impressions: pa.Table
    manifest: dict[str, Any]


def query_key(text: str) -> str:
    """Create a stable content key joining unique queries to impressions."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_query_id(subquery_id: str) -> str:
    """Recover the impression ID from a `base_id_<integer>` subquery ID."""
    base, separator, suffix = subquery_id.rpartition("_")
    if not separator or not base or not suffix.isdigit():
        raise ValueError(
            f"MindSmall subquery ID must end in an integer suffix: {subquery_id!r}"
        )
    return base


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_dir: Path = COMPACT_DATA_DIR) -> dict[str, Any]:
    """Load the manifest and reject mismatched format or source revisions."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Compact MindSmallReranking data is missing at {data_dir}. Build it with:\n"
            "  /mnt/share/envs/embt/bin/python -m "
            "vibe_eval.tasks.mind_small_compact_data build"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported compact MindSmall format version: "
            f"{manifest.get('format_version')!r}"
        )
    if manifest.get("source_dataset") != SOURCE_DATASET:
        raise RuntimeError("Compact MindSmall source dataset does not match.")
    if manifest.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("Compact MindSmall source revision does not match.")
    return manifest


def verify_compact_data(
    data_dir: Path = COMPACT_DATA_DIR,
    *,
    read_tables: bool = True,
) -> CompactMindSmallData | None:
    """Validate file hashes and logical counts, then optionally load all tables."""
    start = perf_counter()
    logger.info("Loading compact MindSmallReranking data from %s", data_dir)
    manifest = load_manifest(data_dir)
    expected_files = manifest.get("files", {})
    for filename in REQUIRED_FILES:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing compact MindSmall file: {path}")
        logger.info("Verifying compact MindSmall file: %s", filename)
        expected_hash = expected_files.get(filename, {}).get("sha256")
        actual_hash = file_sha256(path)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"Compact MindSmall checksum mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    if not read_tables:
        logger.info(
            "Verified compact MindSmallReranking files in %.1fs", perf_counter() - start
        )
        return None

    logger.info("Reading compact MindSmall table: queries.parquet")
    queries = pq.read_table(data_dir / "queries.parquet")
    logger.info("Reading compact MindSmall table: corpus.parquet")
    corpus = pq.read_table(data_dir / "corpus.parquet")
    logger.info("Reading compact MindSmall table: impressions.parquet")
    impressions = pq.read_table(data_dir / "impressions.parquet")
    counts = manifest["counts"]
    actual_counts = {
        "unique_queries": len(queries),
        "corpus_documents": len(corpus),
        "impressions": len(impressions),
        "subqueries": sum(
            len(items.as_py()) for items in impressions["subquery_ids"]
        ),
        "candidate_pairs_compact": sum(
            len(items.as_py()) for items in impressions["candidate_ids"]
        ),
        "qrel_pairs_expanded": sum(
            len(subqueries.as_py()) * len(qrel_candidates.as_py())
            for subqueries, qrel_candidates in zip(
                impressions["subquery_ids"],
                impressions["qrel_candidate_ids"],
                strict=True,
            )
        ),
    }
    if actual_counts != counts:
        raise RuntimeError(
            f"Compact MindSmall counts do not match manifest: "
            f"expected {counts}, got {actual_counts}"
        )
    logger.info(
        "Loaded compact MindSmallReranking data: %d unique queries, %d corpus "
        "documents, %d impressions, %d subqueries in %.1fs",
        actual_counts["unique_queries"],
        actual_counts["corpus_documents"],
        actual_counts["impressions"],
        actual_counts["subqueries"],
        perf_counter() - start,
    )
    return CompactMindSmallData(
        queries=queries,
        corpus=corpus,
        impressions=impressions,
        manifest=manifest,
    )


def _iter_qrel_groups(
    dataset: Dataset,
) -> Iterator[tuple[str, list[str], list[int]]]:
    """Stream 97M qrel rows into query groups without materializing the table."""
    current_query_id: str | None = None
    candidate_ids: list[str] = []
    candidate_scores: list[int] = []
    seen_query_ids: set[str] = set()

    for batch in dataset.data.to_batches(max_chunksize=65_536):
        query_ids = batch.column("query-id").to_pylist()
        corpus_ids = batch.column("corpus-id").to_pylist()
        scores = batch.column("score").to_pylist()
        for qid, corpus_id, score in zip(
            query_ids, corpus_ids, scores, strict=True
        ):
            if current_query_id is None:
                current_query_id = qid
            if qid != current_query_id:
                if qid in seen_query_ids:
                    raise RuntimeError(
                        f"Qrels are not contiguous for query ID {qid!r}."
                    )
                seen_query_ids.add(current_query_id)
                yield current_query_id, candidate_ids, candidate_scores
                current_query_id = qid
                candidate_ids = []
                candidate_scores = []
            candidate_ids.append(corpus_id)
            candidate_scores.append(int(score))

    if current_query_id is not None:
        yield current_query_id, candidate_ids, candidate_scores


def _iter_top_ranked(dataset: Dataset) -> Iterator[tuple[str, list[str]]]:
    """Stream each query's pre-ranked candidates in source row order."""
    for batch in dataset.data.to_batches(max_chunksize=8_192):
        query_ids = batch.column("query-id").to_pylist()
        candidates = batch.column("corpus-ids").to_pylist()
        yield from zip(query_ids, candidates, strict=True)


def _build_tables(
    *,
    queries: Dataset,
    corpus: Dataset,
    qrels: Dataset,
    top_ranked: Dataset,
) -> tuple[pa.Table, pa.Table, pa.Table, dict[str, int]]:
    """Build losslessly restorable compact tables from four source configs."""
    # First map query IDs to content hashes; each distinct text is stored once.
    query_id_to_key: dict[str, str] = {}
    unique_queries: dict[str, str] = {}
    for batch in queries.data.to_batches(max_chunksize=65_536):
        id_column = "id" if "id" in batch.schema.names else "_id"
        query_ids = batch.column(id_column).to_pylist()
        texts = batch.column("text").to_pylist()
        for qid, text in zip(query_ids, texts, strict=True):
            key = query_key(text)
            previous_text = unique_queries.setdefault(key, text)
            if previous_text != text:
                raise RuntimeError(f"SHA-256 collision for query key {key}.")
            if qid in query_id_to_key:
                raise RuntimeError(f"Duplicate query ID: {qid}")
            query_id_to_key[qid] = key

    corpus_id_column = "id" if "id" in corpus.column_names else "_id"
    corpus_ids = corpus[corpus_id_column]
    corpus_id_set = set(corpus_ids)
    if len(corpus_id_set) != len(corpus):
        raise RuntimeError("Compact MindSmall corpus contains duplicate IDs.")

    impression_base_ids: list[str] = []
    impression_subquery_ids: list[list[str]] = []
    impression_query_keys: list[list[str]] = []
    impression_candidate_ids: list[list[str]] = []
    impression_qrel_candidate_ids: list[list[str]] = []
    impression_candidate_scores: list[list[int]] = []

    current_base: str | None = None
    current_subqueries: list[str] = []
    current_query_keys: list[str] = []
    current_candidates: list[str] | None = None
    current_qrel_candidates: list[str] | None = None
    current_scores: list[int] | None = None
    top_iterator = iter(_iter_top_ranked(top_ranked))
    qrel_group_count = 0

    def flush_impression() -> None:
        """Flush one copy of the candidates and qrels shared by an impression."""
        if current_base is None:
            return
        assert current_candidates is not None
        assert current_qrel_candidates is not None and current_scores is not None
        impression_base_ids.append(current_base)
        impression_subquery_ids.append(list(current_subqueries))
        impression_query_keys.append(list(current_query_keys))
        impression_candidate_ids.append(list(current_candidates))
        impression_qrel_candidate_ids.append(list(current_qrel_candidates))
        impression_candidate_scores.append(list(current_scores))

    for qid, candidates, scores in _iter_qrel_groups(qrels):
        qrel_group_count += 1
        try:
            top_qid, top_candidates = next(top_iterator)
        except StopIteration as error:
            raise RuntimeError("top_ranked ended before qrels.") from error
        # top_ranked contains real duplicate document IDs, while qrels are unique
        # by document ID. Validate deduplicated order but preserve the original
        # top_ranked list so candidate scoring semantics remain unchanged.
        if top_qid != qid or list(dict.fromkeys(top_candidates)) != candidates:
            raise RuntimeError(
                f"qrels/top_ranked candidate mismatch for query {qid!r}."
            )
        if qid not in query_id_to_key:
            raise RuntimeError(f"Qrels reference missing query ID {qid!r}.")
        if not candidates:
            raise RuntimeError(f"Query {qid!r} has no candidates.")
        missing_corpus_ids = set(candidates) - corpus_id_set
        if missing_corpus_ids:
            raise RuntimeError(
                f"Query {qid!r} references missing corpus IDs: "
                f"{sorted(missing_corpus_ids)[:3]}"
            )

        base_id = base_query_id(qid)
        if base_id != current_base:
            flush_impression()
            current_base = base_id
            current_subqueries = []
            current_query_keys = []
            current_candidates = top_candidates
            current_qrel_candidates = candidates
            current_scores = scores
        elif (
            top_candidates != current_candidates
            or candidates != current_qrel_candidates
            or scores != current_scores
        ):
            # Compaction is valid only when all subqueries in an impression share
            # exactly the same candidates and relevance labels.
            raise RuntimeError(
                f"Subqueries in impression {base_id!r} do not share qrels."
            )
        current_subqueries.append(qid)
        current_query_keys.append(query_id_to_key[qid])
    flush_impression()

    try:
        extra_top_qid, _ = next(top_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(f"top_ranked has extra query ID {extra_top_qid!r}.")

    if qrel_group_count != len(query_id_to_key):
        raise RuntimeError(
            f"Query/qrels count mismatch: {len(query_id_to_key)} queries, "
            f"{qrel_group_count} qrel groups."
        )

    queries_table = pa.table(
        {
            "query_key": list(unique_queries.keys()),
            "text": list(unique_queries.values()),
        }
    )
    corpus_columns = {
        "id": corpus_ids,
        "text": corpus["text"],
    }
    if "title" in corpus.column_names:
        corpus_columns["title"] = corpus["title"]
    corpus_table = pa.table(corpus_columns)
    impressions_table = pa.table(
        {
            "base_query_id": impression_base_ids,
            "subquery_ids": impression_subquery_ids,
            "query_keys": impression_query_keys,
            "candidate_ids": impression_candidate_ids,
            "qrel_candidate_ids": impression_qrel_candidate_ids,
            "candidate_scores": pa.array(
                impression_candidate_scores,
                type=pa.list_(pa.int32()),
            ),
        }
    )
    counts = {
        "unique_queries": len(queries_table),
        "corpus_documents": len(corpus_table),
        "impressions": len(impressions_table),
        "subqueries": sum(map(len, impression_subquery_ids)),
        "candidate_pairs_compact": sum(map(len, impression_candidate_ids)),
        "qrel_pairs_expanded": sum(
            len(subqueries) * len(qrel_candidates)
            for subqueries, qrel_candidates in zip(
                impression_subquery_ids,
                impression_qrel_candidate_ids,
                strict=True,
            )
        ),
    }
    return queries_table, corpus_table, impressions_table, counts


def _write_build(
    output_dir: Path,
    *,
    queries: pa.Table,
    corpus: pa.Table,
    impressions: pa.Table,
    counts: dict[str, int],
) -> None:
    """Write into a temporary directory and atomically publish a complete build."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        tables = {
            "queries.parquet": queries,
            "corpus.parquet": corpus,
            "impressions.parquet": impressions,
        }
        for filename, table in tables.items():
            pq.write_table(
                table,
                temp_dir / filename,
                compression="zstd",
                row_group_size=4096,
            )
        manifest = {
            "format_version": FORMAT_VERSION,
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "split": "test",
            "counts": counts,
            "files": {
                filename: {
                    "rows": len(table),
                    "sha256": file_sha256(temp_dir / filename),
                }
                for filename, table in tables.items()
            },
        }
        with (temp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        # Never overwrite an existing build; callers must move it explicitly so
        # previous artifacts remain traceable.
        if output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Run verify or remove/move it explicitly before rebuilding."
            )
        os.replace(temp_dir, output_dir)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def build(cache_dir: Path, output_dir: Path) -> None:
    """Build from the pinned source revision and immediately verify the output."""
    os.environ["HF_HOME"] = str(cache_dir)
    from datasets import load_dataset

    revision = SOURCE_REVISION
    queries = load_dataset(
        SOURCE_DATASET, "queries", revision=revision, split="test"
    )
    corpus = load_dataset(SOURCE_DATASET, "corpus", revision=revision, split="test")
    qrels = load_dataset(SOURCE_DATASET, "default", revision=revision, split="test")
    top_ranked = load_dataset(
        SOURCE_DATASET, "top_ranked", revision=revision, split="test"
    )
    tables = _build_tables(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        top_ranked=top_ranked,
    )
    _write_build(
        output_dir,
        queries=tables[0],
        corpus=tables[1],
        impressions=tables[2],
        counts=tables[3],
    )
    verify_compact_data(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify compact MindSmallReranking test data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    build_parser.add_argument("--output-dir", type=Path, default=COMPACT_DATA_DIR)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=COMPACT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args.cache_dir, args.output_dir)
    else:
        verify_compact_data(args.output_dir)
    print(f"{args.command} completed: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
# Usage:
#
# 1. Build compact data with the default cache and output directories:
#    python -m vibe_eval.tasks.mind_small_compact_data build
#
# 2. Validate file hashes, table sizes, and the manifest:
#    python -m vibe_eval.tasks.mind_small_compact_data verify
#
# 3. Override the cache or output directory:
#    python -m vibe_eval.tasks.mind_small_compact_data build \
#        --cache-dir /path/to/hf_cache --output-dir /path/to/output
#    python -m vibe_eval.tasks.mind_small_compact_data verify \
#        --output-dir /path/to/output
"""
