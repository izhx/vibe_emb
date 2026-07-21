#!/usr/bin/env python3
"""Render MTEB benchmark tables from local result revision directories."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

previous_logging_disable = logging.root.manager.disable
logging.disable(logging.WARNING)
try:
    import mteb  # noqa: E402
finally:
    logging.disable(previous_logging_disable)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd
    from mteb.abstasks.abstask import AbsTask
    from mteb.models.model_meta import ModelMeta


class LocalResultCache(mteb.ResultCache):
    """Expose arbitrary local revision directories through ResultCache."""

    def __init__(self, result_dirs: Sequence[Path]) -> None:
        self.result_dirs = [path.resolve() for path in result_dirs]
        # ResultCache.load_results() only needs this attribute for diagnostics once
        # get_cache_paths() is supplied by this local adapter.
        self.cache_path = Path()

    def get_cache_paths(
        self,
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
        tasks: Sequence[str] | Iterable[AbsTask] | None = None,
        require_model_meta: bool = True,
        include_remote: bool = True,
        load_experiments: object = None,
    ) -> list[Path]:
        del models, include_remote, load_experiments
        paths = []
        for result_dir in self.result_dirs:
            if require_model_meta and not (result_dir / "model_meta.json").is_file():
                continue
            paths.extend(
                path
                for path in sorted(result_dir.glob("*.json"))
                if path.name != "model_meta.json"
            )
        return self._filter_paths_by_task(paths, tasks)

    @staticmethod
    def _get_model_name_and_revision_from_path(
        revision_path: Path,
    ) -> tuple[str, str, str | None]:
        model_name, revision, experiment_name = (
            mteb.ResultCache._get_model_name_and_revision_from_path(revision_path)
        )
        return model_name, revision or revision_path.name, experiment_name


def register_local_model_meta(result_dirs: Sequence[Path]) -> None:
    """Make unregistered local models visible to MTEB's summary table."""
    from mteb.benchmarks._create_table import _static_model_meta
    from mteb.models.model_implementations import MODEL_REGISTRY
    from mteb.models.model_meta import ModelMeta

    changed = False
    for result_dir in result_dirs:
        metadata_path = result_dir / "model_meta.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to read {metadata_path}: {exc}") from exc

        model_name = metadata.get("name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"missing model name in {metadata_path}")
        if model_name in MODEL_REGISTRY:
            continue

        MODEL_REGISTRY[model_name] = ModelMeta.model_validate(metadata)
        changed = True

    if changed:
        _static_model_meta.cache_clear()


def load_benchmark_tables(
    result_dirs: Sequence[Path],
    benchmark_names: Sequence[str],
    include_per_task: bool = False,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame | None]]:
    """Use MTEB's own cache loader and benchmark table implementation."""
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The task '.*' is currently in beta\..*",
            )
            register_local_model_meta(result_dirs)
            cache = LocalResultCache(result_dirs)
            tables = []
            for benchmark_name in benchmark_names:
                benchmark = mteb.get_benchmark(benchmark_name)
                results = cache.load_results(
                    tasks=benchmark,
                    require_model_meta=True,
                    include_remote=False,
                )
                summary_table = results.get_benchmark_result()
                per_task_table = None
                if include_per_task:
                    per_task_table = benchmark._create_per_task_table(
                        results._to_results_df(benchmark.tasks)
                    ).to_pandas()
                tables.append((benchmark.name, summary_table, per_task_table))
            return tables
    finally:
        logging.disable(previous_disable)


def render_table(
    tables: Sequence[tuple[str, pd.DataFrame, pd.DataFrame | None]],
) -> str:
    blocks = []
    for benchmark_name, summary_table, per_task_table in tables:
        header = (
            f"Benchmark: {benchmark_name}\n"
            f"MTEB version: {version('mteb')}"
        )
        if per_task_table is None:
            blocks.append(f"{header}\n\n{summary_table.to_string(index=False)}")
        else:
            blocks.append(
                f"{header}\n\n"
                f"Summary:\n{summary_table.to_string(index=False)}\n\n"
                f"Per task:\n{per_task_table.to_string(index=False)}"
            )
    return "\n\n".join(blocks) + "\n"


def render_csv(
    tables: Sequence[tuple[str, pd.DataFrame, pd.DataFrame | None]],
) -> str:
    blocks = []
    for benchmark_name, summary_table, per_task_table in tables:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["Benchmark", benchmark_name])
        writer.writerow(["MTEB version", version("mteb")])
        if per_task_table is None:
            summary_table.to_csv(output, index=False)
        else:
            writer.writerow(["Result type", "Summary"])
            summary_table.to_csv(output, index=False)
            writer.writerow([])
            writer.writerow(["Result type", "Per task"])
            per_task_table.to_csv(output, index=False)
        blocks.append(output.getvalue().rstrip("\n"))
    return "\n\n".join(blocks) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render MTEB's benchmark summary table from local results."
    )
    parser.add_argument(
        "--results",
        dest="result_dirs",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        metavar="DIR",
        help="One or more local MTEB model revision directories",
    )
    parser.add_argument(
        "--benchmark",
        "--benchmarks",
        dest="benchmarks",
        action="append",
        metavar="NAME",
        help="MTEB benchmark name; repeat for multiple benchmarks",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Also include MTEB's per-task table in each benchmark block",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dirs = list(dict.fromkeys(args.result_dirs))
    missing_dirs = [path for path in result_dirs if not path.is_dir()]
    if missing_dirs:
        print(
            "ERROR: result directories do not exist: "
            + ", ".join(str(path) for path in missing_dirs),
            file=sys.stderr,
        )
        return 2

    benchmark_names = list(dict.fromkeys(args.benchmarks or ["MTEB(eng, v2)"]))
    try:
        tables = load_benchmark_tables(
            result_dirs,
            benchmark_names,
            include_per_task=args.per_task,
        )
        content = render_csv(tables) if args.format == "csv" else render_table(tables)
        sys.stdout.write(content)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
