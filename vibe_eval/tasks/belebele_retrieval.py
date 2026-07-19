from __future__ import annotations

"""Configuration-aware loader for MTEB's BelebeleRetrieval task."""

from collections.abc import Sequence
from typing import Any

from datasets import Dataset, load_dataset
from mteb.tasks.retrieval.multilingual.belebele_retrieval import (
    BelebeleRetrieval as UpstreamBelebeleRetrieval,
)


BELEBELE_TASK_NAME = UpstreamBelebeleRetrieval.metadata.name
_EVAL_SPLIT = "test"


class BelebeleRetrieval(UpstreamBelebeleRetrieval):
    """Load each Belebele language from its explicit dataset configuration."""

    def _required_languages(self) -> list[str]:
        # MTEB evaluates language pairs, while the Hub dataset exposes one
        # configuration per individual language. Preserve first-use order and
        # load each configuration only once, even when it appears in many pairs.
        return list(
            dict.fromkeys(
                language.replace("-", "_")
                for lang_pair in self.hf_subsets
                for language in self.metadata.eval_langs[lang_pair]
            )
        )

    def load_data(
        self,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        if self.data_loaded:
            return

        self.dataset: dict[str, Dataset] = {
            language: load_dataset(
                name=language,
                **self.metadata.dataset,
                num_proc=num_proc,
                **kwargs,
            )[_EVAL_SPLIT]
            for language in self._required_languages()
        }

        self.queries = {
            lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets
        }
        self.corpus = {
            lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets
        }
        self.relevant_docs = {
            lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets
        }

        for lang_pair in self.hf_subsets:
            languages = self.metadata.eval_langs[lang_pair]
            corpus_language, question_language = (
                language.replace("-", "_") for language in languages
            )
            corpus_data = self.dataset[corpus_language]
            question_data = self.dataset[question_language]

            question_ids: dict[str, int] = {}
            for row in question_data:
                question = row["question"]
                if question not in question_ids:
                    question_ids[question] = len(question_ids)

            link_to_context_id: dict[str, str] = {}
            for row in corpus_data:
                link = row["link"]
                if link in link_to_context_id:
                    continue
                context_id = f"C{len(link_to_context_id)}"
                link_to_context_id[link] = context_id
                self.corpus[lang_pair][_EVAL_SPLIT][context_id] = {
                    "title": "",
                    "text": row["flores_passage"],
                }

            for row in question_data:
                query_id = f"Q{question_ids[row['question']]}"
                self.queries[lang_pair][_EVAL_SPLIT][query_id] = row["question"]

                context_id = link_to_context_id[row["link"]]
                self.relevant_docs[lang_pair][_EVAL_SPLIT].setdefault(
                    query_id,
                    {},
                )[context_id] = 1

        self.data_loaded = True


def patch_belebele_tasks(tasks: Sequence[Any]) -> list[Any]:
    """Replace resolved upstream instances while preserving task filters."""

    patched_tasks: list[Any] = []
    for task in tasks:
        if task.metadata.name != BELEBELE_TASK_NAME:
            patched_tasks.append(task)
            continue

        replacement = BelebeleRetrieval(seed=task.seed)
        replacement.hf_subsets = list(task.hf_subsets)
        replacement._eval_splits = getattr(task, "_eval_splits", None)
        patched_tasks.append(replacement)

    return patched_tasks
