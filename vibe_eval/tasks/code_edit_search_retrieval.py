from __future__ import annotations

"""Strict-offline loader for MTEB's CodeEditSearchRetrieval task."""

import logging
from collections.abc import Sequence
from typing import Any

import datasets
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download
from mteb.tasks.retrieval.code.code_edit_search_retrieval import (
    CodeEditSearchRetrieval as UpstreamCodeEditSearchRetrieval,
)


logger = logging.getLogger(__name__)

CODE_EDIT_SEARCH_TASK_NAME = UpstreamCodeEditSearchRetrieval.metadata.name


class CodeEditSearchRetrieval(UpstreamCodeEditSearchRetrieval):
    """Load pinned raw Parquet files directly when Hugging Face is offline."""

    def _load_local_language(self, language: str) -> Dataset:
        dataset_spec = self.metadata.dataset
        parquet_path = hf_hub_download(
            repo_id=dataset_spec["path"],
            repo_type="dataset",
            revision=dataset_spec["revision"],
            filename=(
                f"{language}/{self._EVAL_SPLIT}-00000-of-00001.parquet"
            ),
            local_files_only=True,
        )
        logger.info(
            "Loading %s/%s from the raw Parquet Hub cache: %s",
            CODE_EDIT_SEARCH_TASK_NAME,
            language,
            parquet_path,
        )

        # Construct the Dataset from the Parquet table itself. Calling
        # datasets.load_dataset(repo_id, data_dir=language) here would make
        # strict-offline mode look for a synthetic default-data_dir=<language>
        # processed-Arrow config instead of resolving the cached hashed config.
        return Dataset(pq.read_table(parquet_path, memory_map=True))

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        if not datasets.config.HF_HUB_OFFLINE:
            return super().load_data(num_proc=num_proc, **kwargs)

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        # Use hf_subsets rather than the module's fixed language list so a
        # language filter already applied by MTEB remains effective.
        for language in self.hf_subsets:
            data = self._load_local_language(language)
            rows = data.select(range(min(1000, len(data))))

            self.queries[language] = {
                self._EVAL_SPLIT: {
                    str(index): row["instruction"]
                    for index, row in enumerate(rows)
                }
            }
            self.corpus[language] = {
                self._EVAL_SPLIT: {
                    str(row["commit"]): {"text": row["diff"]} for row in rows
                }
            }
            self.relevant_docs[language] = {
                self._EVAL_SPLIT: {
                    str(index): {str(row["commit"]): 1}
                    for index, row in enumerate(rows)
                }
            }

        self.data_loaded = True


def patch_code_edit_search_tasks(tasks: Sequence[Any]) -> list[Any]:
    """Replace resolved upstream instances while preserving task filters."""

    patched_tasks: list[Any] = []
    for task in tasks:
        if task.metadata.name != CODE_EDIT_SEARCH_TASK_NAME:
            patched_tasks.append(task)
            continue

        replacement = CodeEditSearchRetrieval(seed=task.seed)
        replacement.hf_subsets = list(task.hf_subsets)
        replacement._eval_splits = getattr(task, "_eval_splits", None)
        patched_tasks.append(replacement)

    return patched_tasks
