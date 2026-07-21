from __future__ import annotations

"""Repository-wide runtime patches for the installed MTEB version.

These patches cover generic data preparation and offline retrieval behavior.
Task-specific MindSmall logic lives under ``vibe_eval.tasks``.
"""

import fcntl
import json
import logging
import os
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from importlib import import_module
from itertools import count
from pathlib import Path
from threading import Lock
from typing import Any

from datasets import Dataset


logger = logging.getLogger(__name__)

# Runtime patching changes process-global MTEB callables. Keep the original
# functions for delegation and guard each installation so repeated setup is safe.
_query_patch_installed = False
_retrieval_qrels_patch_installed = False
_reranking_patch_installed = False
_e2e_timing_patch_installed = False
_original_combine_queries: Any | None = None
_original_retrieval_load_qrels: Any | None = None
_original_reranking_loader_load: Any | None = None
_original_evaluate_task: Any | None = None
RerankingDatasetKey = tuple[str, str, str | None]
_reranking_dataset_keys: set[RerankingDatasetKey] = set()

_E2E_TIMING_FILENAME = "evaluation_time.json"
_E2E_TIMING_SCHEMA_VERSION = 1
_e2e_timing_thread_lock = Lock()
_e2e_timing_depth: ContextVar[int] = ContextVar("mteb_e2e_timing_depth", default=0)
_e2e_run_id = uuid.uuid4().hex
_e2e_sequence = count(1)
_e2e_mteb_version: str | None = None


def _utc_now() -> str:
    """Return an unambiguous timestamp suitable for the timing artifact."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timing_path(model: Any, task: Any, cache: Any) -> Path | None:
    """Resolve timing output through ResultCache so legacy paths stay intact."""
    if cache is None:
        return None
    model_meta = getattr(model, "mteb_model_meta", None)
    if model_meta is None:
        logger.warning(
            "Cannot record end-to-end timing for %s because the model has no "
            "mteb_model_meta.",
            task.metadata.name,
        )
        return None
    task_result_path = cache.get_task_result_path(
        task_name=task.metadata.name,
        model_name=model_meta,
    )
    return task_result_path.parent / _E2E_TIMING_FILENAME


@contextmanager
def _locked_timing_file(timing_path: Path) -> Iterator[None]:
    """Serialize timing-file updates across benchmark processes."""
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = timing_path.with_name(f".{timing_path.name}.lock")
    # Keep the sidecar file in place: unlinking it could let a new process lock a
    # different inode while an existing process still holds the old one.
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_timing_attempt(
    timing_path: Path,
    task_name: str,
    attempt: dict[str, Any],
) -> None:
    """Merge one attempt and atomically replace the per-model timing file."""
    with _e2e_timing_thread_lock:
        with _locked_timing_file(timing_path):
            if timing_path.exists():
                payload = json.loads(timing_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("schema_version")
                    != _E2E_TIMING_SCHEMA_VERSION
                ):
                    raise ValueError(
                        f"Unsupported end-to-end timing file: {timing_path}"
                    )
            else:
                payload = {
                    "schema_version": _E2E_TIMING_SCHEMA_VERSION,
                    "mteb_version": _e2e_mteb_version,
                    "tasks": {},
                }

            tasks = payload.setdefault("tasks", {})
            task_timing = tasks.setdefault(task_name, {"attempts": []})
            attempts = task_timing.setdefault("attempts", [])
            for index, existing in enumerate(attempts):
                if existing.get("attempt_id") == attempt["attempt_id"]:
                    attempts[index] = attempt
                    break
            else:
                attempts.append(attempt)

            task_timing["latest_attempt"] = attempt
            if attempt["status"] == "success":
                task_timing["latest_successful_attempt"] = attempt
                if not attempt["partial"]:
                    task_timing["latest_full_successful_attempt"] = attempt
            payload["updated_at"] = _utc_now()

            temporary_path = timing_path.with_name(
                f".{timing_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temporary_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_path, timing_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()


def _record_timing_attempt(
    timing_path: Path | None,
    task_name: str,
    attempt: dict[str, Any],
) -> None:
    """Keep telemetry failures from invalidating an otherwise valid result."""
    if timing_path is None:
        return
    try:
        _write_timing_attempt(timing_path, task_name, attempt)
    except Exception:
        logger.exception(
            "Failed to write end-to-end evaluation timing for %s to %s.",
            task_name,
            timing_path,
        )


def _evaluate_task_with_e2e_timing(
    model: Any,
    task: Any,
    *,
    splits: dict[Any, list[Any]],
    co2_tracker: bool | None,
    encode_kwargs: dict[str, Any],
    prediction_folder: Path | None,
    public_only: bool | None,
    cache: Any = None,
    num_proc: int | None = None,
    existing_results: Any = None,
) -> Any:
    """Measure data loading through unloading around MTEB's task evaluator."""
    assert _original_evaluate_task is not None
    depth = _e2e_timing_depth.get()
    depth_token = _e2e_timing_depth.set(depth + 1)
    try:
        # With CO2 tracking enabled, upstream recursively invokes
        # ``_evaluate_task``. Only the outer call represents one task attempt.
        if depth:
            return _original_evaluate_task(
                model,
                task,
                splits=splits,
                co2_tracker=co2_tracker,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                public_only=public_only,
                cache=cache,
                num_proc=num_proc,
                existing_results=existing_results,
            )

        task_name = task.metadata.name
        try:
            timing_path = _timing_path(model, task, cache)
        except Exception:
            logger.exception(
                "Failed to resolve the end-to-end timing path for %s.",
                task_name,
            )
            timing_path = None
        evaluation_time_before = float(
            getattr(existing_results, "evaluation_time", None) or 0.0
        )
        attempt = {
            "attempt_id": uuid.uuid4().hex,
            "run_id": _e2e_run_id,
            "sequence": next(_e2e_sequence),
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "wall_time_seconds": None,
            "mteb_evaluation_time_before_seconds": evaluation_time_before,
            "mteb_evaluation_time_delta_seconds": None,
            "non_evaluation_overhead_seconds": None,
            "data_was_preloaded": bool(getattr(task, "data_loaded", False)),
            "partial": existing_results is not None,
            "split_subset_counts": {
                str(split): len(subsets) for split, subsets in splits.items()
            },
        }
        _record_timing_attempt(timing_path, task_name, attempt)
        started = time.perf_counter()
        try:
            result = _original_evaluate_task(
                model,
                task,
                splits=splits,
                co2_tracker=co2_tracker,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                public_only=public_only,
                cache=cache,
                num_proc=num_proc,
                existing_results=existing_results,
            )
        except BaseException as error:
            attempt.update(
                status="failed",
                finished_at=_utc_now(),
                wall_time_seconds=time.perf_counter() - started,
                error_type=type(error).__name__,
                error_message=str(error)[:2000],
            )
            _record_timing_attempt(timing_path, task_name, attempt)
            raise

        wall_time = time.perf_counter() - started
        evaluation_time_after = getattr(result, "evaluation_time", None)
        status = "success" if evaluation_time_after is not None else "failed"
        evaluation_time_delta = (
            float(evaluation_time_after) - evaluation_time_before
            if evaluation_time_after is not None
            else None
        )
        attempt.update(
            status=status,
            finished_at=_utc_now(),
            wall_time_seconds=wall_time,
            mteb_evaluation_time_delta_seconds=evaluation_time_delta,
            non_evaluation_overhead_seconds=(
                wall_time - evaluation_time_delta
                if evaluation_time_delta is not None
                else None
            ),
        )
        if status == "failed":
            attempt.update(
                error_type=type(result).__name__,
                error_message=str(getattr(result, "exception", result))[:2000],
            )
        _record_timing_attempt(timing_path, task_name, attempt)
        return result
    finally:
        _e2e_timing_depth.reset(depth_token)


def _combine_queries_arrow_native(dataset: Dataset) -> Dataset:
    """Alias the Arrow text column without materializing every value in Python."""
    # Dataset instructions require row-wise text composition. Preserve upstream
    # behavior for that case; the fast path is only valid for plain text queries.
    if "instruction" in dataset.column_names:
        if _original_combine_queries is None:
            raise RuntimeError("The original MTEB query preparation is unavailable.")
        return _original_combine_queries(dataset)
    # Recreate the same output schema as MTEB without materializing the text
    # column as a Python list. Reading through Dataset's Arrow formatter is
    # important here: retrieval may select only queries with positive qrels, so
    # ``dataset.data`` can still contain more physical rows than the logical view.
    if "query" in dataset.column_names:
        dataset = dataset.remove_columns(["query"])
    texts = dataset.with_format("arrow")["text"]
    return dataset.add_column("query", texts)


def _normalized_subset(subset: Any) -> str | None:
    """Normalize MTEB's default subset spelling to the loader's None key."""
    if subset is None or subset == "default":
        return None
    return str(subset)


def _iter_leaf_tasks(tasks: Sequence[Any]) -> Iterator[Any]:
    """Flatten aggregate benchmarks before deriving dataset configurations."""
    for task in tasks:
        if getattr(task, "is_aggregate", False):
            yield from _iter_leaf_tasks(task.tasks)
        else:
            yield task


def _reranking_keys_for_tasks(tasks: Sequence[Any]) -> set[RerankingDatasetKey]:
    """Collect exact dataset revisions/subsets that require ranked candidates."""
    keys: set[RerankingDatasetKey] = set()
    for task in _iter_leaf_tasks(tasks):
        metadata = task.metadata
        # Include both Reranking and InstructionReranking without maintaining a
        # task-name allowlist that would become stale as MTEB adds new tasks.
        if not str(getattr(metadata, "type", "")).endswith("Reranking"):
            continue
        dataset = metadata.dataset
        path = dataset.get("path")
        revision = dataset.get("revision")
        if not isinstance(path, str) or not isinstance(revision, str):
            raise ValueError(
                f"Reranking task {metadata.name} must define dataset path and "
                "revision before its candidate-loading patch can be installed."
            )
        subsets = getattr(task, "hf_subsets", None) or ("default",)
        keys.update(
            (path, revision, _normalized_subset(subset)) for subset in subsets
        )
    return keys


def _load_with_reranking_candidates(loader: Any, num_proc: int | None = None) -> Any:
    """Load top_ranked directly when offline config discovery omitted it."""
    assert _original_reranking_loader_load is not None
    # Always run the original loader first. Only selected reranking configs with
    # missing candidates need the direct-loading fallback below.
    split_data = _original_reranking_loader_load(loader, num_proc=num_proc)
    key = (loader.hf_repo, loader.revision, _normalized_subset(loader.config))
    if key not in _reranking_dataset_keys or split_data["top_ranked"] is not None:
        return split_data
    try:
        # `_load_top_ranked` accepts the already known config/revision and can use
        # cached Arrow data even when Hub config-name discovery is unavailable.
        top_ranked = loader._load_top_ranked(num_proc)
    except Exception as error:
        raise RuntimeError(
            "Reranking evaluation requires top_ranked candidates, but MTEB "
            f"did not discover them for {loader.hf_repo} "
            f"(revision={loader.revision}, subset={loader.config or 'default'}, "
            f"split={loader.split}) and direct loading failed."
        ) from error
    if not top_ranked:
        raise RuntimeError(
            "Reranking evaluation requires non-empty top_ranked candidates for "
            f"{loader.hf_repo} (revision={loader.revision}, "
            f"subset={loader.config or 'default'}, split={loader.split})."
        )
    split_data["top_ranked"] = top_ranked
    logger.info(
        "Loaded %d top_ranked candidate sets directly for offline reranking "
        "dataset %s (subset=%s, split=%s).",
        len(top_ranked), loader.hf_repo, loader.config or "default", loader.split,
    )
    return split_data


def _load_qrels_with_offline_alias(loader: Any, num_proc: int | None = None) -> Any:
    """Retry the cached ``qrels`` config when offline discovery chose default."""
    assert _original_retrieval_load_qrels is not None
    try:
        return _original_retrieval_load_qrels(loader, num_proc=num_proc)
    except ValueError as error:
        message = str(error)
        # Some retrieval repositories store qrels under a literal ``qrels``
        # config, while MTEB's strict-offline config discovery reports only the
        # synthetic default config. Restrict the retry to errors that prove the
        # requested default is absent and a cached qrels config is available.
        if (
            loader.config is not None
            or "config 'default'" not in message
            or "Available configs in the cache:" not in message
            or "'qrels'" not in message
        ):
            raise
        original_configs = loader.dataset_configs
        # The synthetic ``default`` entry is what made upstream select the
        # missing config. Replace it for this retry instead of merely appending
        # qrels, otherwise upstream continues to prefer default.
        loader.dataset_configs = [
            *[config for config in original_configs if config != "default"],
            "qrels",
        ]
        try:
            qrels = _original_retrieval_load_qrels(loader, num_proc=num_proc)
        except Exception:
            # Preserve the first, more accurate failure if the advertised cache
            # cannot actually be loaded.
            raise error
        finally:
            loader.dataset_configs = original_configs
        logger.info(
            "Loaded qrels from cached config 'qrels' after strict-offline "
            "default-config discovery failed for %s.", loader.hf_repo,
        )
        return qrels


def _validate_mteb_version() -> str:
    """Fail closed when private APIs may no longer match the validated layout."""
    import mteb

    if mteb.__version__ != "2.14.9":
        raise RuntimeError(
            "Repository MTEB patches were validated for MTEB 2.14.9, got "
            f"{mteb.__version__}."
        )
    return mteb.__version__


def install_e2e_evaluation_time_patch() -> None:
    """Record per-task wall time beside MTEB's task result JSON files."""
    global _e2e_timing_patch_installed, _e2e_mteb_version
    global _original_evaluate_task
    if _e2e_timing_patch_installed:
        return

    version = _validate_mteb_version()
    evaluate_module = import_module("mteb.evaluate")
    _original_evaluate_task = evaluate_module._evaluate_task

    @wraps(_original_evaluate_task)
    def timed_evaluate_task(*args: Any, **kwargs: Any) -> Any:
        return _evaluate_task_with_e2e_timing(*args, **kwargs)

    evaluate_module._evaluate_task = timed_evaluate_task
    _e2e_mteb_version = version
    _e2e_timing_patch_installed = True
    logger.info(
        "Installed per-task end-to-end evaluation timing patch for MTEB %s.",
        version,
    )


def install_query_dataloader_patch() -> None:
    """Install the Arrow-native patch for text queries without instructions."""
    global _query_patch_installed, _original_combine_queries
    if _query_patch_installed:
        return
    from mteb import _create_dataloaders

    version = _validate_mteb_version()
    _original_combine_queries = _create_dataloaders._combine_queries_with_instruction_text
    _create_dataloaders._combine_queries_with_instruction_text = _combine_queries_arrow_native
    _query_patch_installed = True
    logger.info(
        "Installed generic Arrow-native text-query DataLoader patch for MTEB %s; "
        "datasets with an instruction column retain the original MTEB path.", version,
    )


def install_retrieval_qrels_offline_patch() -> None:
    """Fall back from an absent default config to an advertised qrels cache."""
    global _retrieval_qrels_patch_installed, _original_retrieval_load_qrels
    if _retrieval_qrels_patch_installed:
        return
    from mteb.abstasks import retrieval_dataset_loaders

    version = _validate_mteb_version()
    loader_class = retrieval_dataset_loaders.RetrievalDatasetLoader
    _original_retrieval_load_qrels = loader_class._load_qrels
    loader_class._load_qrels = _load_qrels_with_offline_alias
    _retrieval_qrels_patch_installed = True
    logger.info(
        "Installed strict-offline retrieval qrels config-alias patch for MTEB %s.",
        version,
    )


def install_reranking_top_ranked_patch(tasks: Sequence[Any]) -> None:
    """Load candidate configs required by selected reranking tasks offline."""
    global _reranking_patch_installed, _original_reranking_loader_load
    # Registration is cumulative because callers may evaluate several task sets
    # in the same process after the global loader hook has already been installed.
    keys = _reranking_keys_for_tasks(tasks)
    _reranking_dataset_keys.update(keys)
    if not keys or _reranking_patch_installed:
        return
    from mteb.abstasks import retrieval_dataset_loaders

    version = _validate_mteb_version()
    _original_reranking_loader_load = retrieval_dataset_loaders.RetrievalDatasetLoader.load
    retrieval_dataset_loaders.RetrievalDatasetLoader.load = _load_with_reranking_candidates
    _reranking_patch_installed = True
    logger.info(
        "Installed offline top_ranked loading patch for %d reranking dataset "
        "configuration(s) on MTEB %s.", len(keys), version,
    )


def create_result_cache(mteb_module: Any, output_folder: str | Path) -> Any:
    """Keep the CLI output folder compatible with the legacy MTEB layout."""
    output_path = Path(output_folder)

    class OutputFolderResultCache(mteb_module.ResultCache):
        def get_task_result_path(
            self,
            task_name: str,
            model_name: Any,
            model_revision: str | None = None,
            remote: bool = False,
            experiment_name: str | None = None,
        ) -> Path:
            result_path = super().get_task_result_path(
                task_name=task_name,
                model_name=model_name,
                model_revision=model_revision,
                remote=remote,
                experiment_name=experiment_name,
            )
            if remote:
                return result_path

            relative_path = result_path.relative_to(self.cache_path / "results")
            return output_path / relative_path

    return OutputFolderResultCache(output_path)
