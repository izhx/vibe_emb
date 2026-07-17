from __future__ import annotations

"""Legacy MindSmall runtime optimization and compact/legacy task selection.

The default compact task avoids repeated source data entirely. Legacy mode keeps
the upstream task but patches candidate recovery and query encoding so it remains
usable for differential validation without falling back to full-corpus retrieval.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset
from mteb.types import PromptType


logger = logging.getLogger(__name__)

MIND_SMALL_RERANKING = "MindSmallReranking"
MIND_SMALL_REVISION = "227478e3235572039f4f7661840e059f31ef6eb1"

_mind_patch_installed = False
_original_mind_loader_load: Any | None = None
_original_search: Any | None = None

MIND_SMALL_MODE_ENV = "VIBE_MIND_SMALL_RERANKING_MODE"
MIND_SMALL_COMPACT_MODE = "compact"
MIND_SMALL_LEGACY_MODE = "legacy"


def _load_with_mind_candidates(loader: Any, num_proc: int | None = None) -> Any:
    """Recover legacy candidate sets when offline config discovery misses them."""
    assert _original_mind_loader_load is not None
    split_data = _original_mind_loader_load(loader, num_proc=num_proc)
    if (
        loader.hf_repo == "mteb/MindSmallReranking"
        and loader.revision == MIND_SMALL_REVISION
        and split_data["top_ranked"] is None
    ):
        relevant_docs = split_data["relevant_docs"]
        if not relevant_docs or not any(
            score == 0
            for doc_scores in relevant_docs.values()
            for score in doc_scores.values()
        ):
            raise RuntimeError(
                "MindSmallReranking candidate recovery requires qrels containing "
                "both positive and zero-labelled candidate documents."
            )
        # The qrels mappings already contain the candidate IDs in dataset order.
        # Reuse them as read-only candidate sequences instead of duplicating about
        # 97 million IDs or loading the separate 5 GB top_ranked configuration.
        split_data["top_ranked"] = relevant_docs
        logger.info(
            "Recovered MindSmallReranking candidate sets from qrels because the "
            "offline dataset config discovery did not expose top_ranked."
        )
    return split_data


def _as_candidate_ids(candidate_documents: Any) -> Sequence[str]:
    """Normalize qrel mappings and top-ranked sequences to candidate ID order."""
    if isinstance(candidate_documents, Mapping):
        return tuple(candidate_documents.keys())
    return candidate_documents


def _dictionary_encode_texts(dataset: Dataset) -> tuple[list[str], np.ndarray]:
    """Deduplicate query text in Arrow and return the row-to-unique inverse map."""
    text_column = dataset.data.column("text").combine_chunks()
    encoded = pc.dictionary_encode(text_column)
    if not isinstance(encoded, pa.DictionaryArray):
        raise TypeError(f"Expected DictionaryArray, got {type(encoded).__name__}")
    unique_texts = encoded.dictionary.to_pylist()
    inverse = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return unique_texts, inverse


def _encode_unique_queries(
    wrapper: Any,
    queries: Dataset,
    *,
    task_metadata: Any,
    hf_split: str,
    hf_subset: str,
    encode_kwargs: dict[str, Any],
    num_proc: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each distinct query text once through the normal MTEB DataLoader."""
    from mteb._create_dataloaders import create_dataloader

    unique_texts, inverse = _dictionary_encode_texts(queries)
    unique_queries = Dataset.from_dict(
        {
            "id": [str(index) for index in range(len(unique_texts))],
            "text": unique_texts,
        }
    )
    loader = create_dataloader(
        unique_queries,
        task_metadata=task_metadata,
        prompt_type=PromptType.query,
        num_proc=num_proc,
        **encode_kwargs,
    )
    embeddings = wrapper.model.encode(
        loader,
        task_metadata=task_metadata,
        hf_split=hf_split,
        hf_subset=hf_subset,
        prompt_type=PromptType.query,
        **encode_kwargs,
    )
    logger.info(
        "MindSmallReranking encoded %d distinct query texts for %d query rows.",
        len(unique_texts),
        len(queries),
    )
    return np.asarray(embeddings), inverse


def _encode_mind_corpus(
    wrapper: Any,
    *,
    task_metadata: Any,
    hf_split: str,
    hf_subset: str,
    encode_kwargs: dict[str, Any],
    num_proc: int | None,
) -> np.ndarray:
    """Encode the small shared corpus once with the original document prompt path."""
    from mteb._create_dataloaders import create_dataloader

    if wrapper.task_corpus is None:
        raise ValueError("Corpus must be indexed before searching.")
    loader = create_dataloader(
        wrapper.task_corpus,
        task_metadata=task_metadata,
        prompt_type=PromptType.document,
        num_proc=num_proc,
        **encode_kwargs,
    )
    return np.asarray(
        wrapper.model.encode(
            loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )
    )


def _score_unique_queries(
    model: Any,
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Compute unique-query by corpus scores in bounded similarity blocks."""
    score_chunks: list[np.ndarray] = []
    for start in range(0, len(query_embeddings), chunk_size):
        scores = model.similarity(
            query_embeddings[start : start + chunk_size],
            corpus_embeddings,
        )
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        score_chunks.append(np.asarray(scores, dtype=np.float32))
    if not score_chunks:
        return np.empty((0, len(corpus_embeddings)), dtype=np.float32)
    return np.concatenate(score_chunks, axis=0)


def _build_candidate_results(
    *,
    queries: Dataset,
    inverse: np.ndarray,
    unique_scores: np.ndarray,
    corpus_ids: Sequence[str],
    top_ranked: Mapping[str, Any],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Restore per-query candidate scores from unique-query score rows."""
    corpus_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
    query_ids = queries.data.column("id").combine_chunks()
    results: dict[str, dict[str, float]] = {}

    for row_index, query_id in enumerate(query_ids.to_pylist()):
        # `inverse` maps every repeated source query row to its unique score row.
        candidate_ids = _as_candidate_ids(top_ranked[query_id])
        if len(candidate_ids) > top_k:
            candidate_indices = np.fromiter(
                (corpus_index[doc_id] for doc_id in candidate_ids),
                dtype=np.int64,
                count=len(candidate_ids),
            )
            candidate_scores = unique_scores[inverse[row_index], candidate_indices]
            selected = np.argpartition(candidate_scores, -top_k)[-top_k:]
            candidate_ids = [candidate_ids[index] for index in selected]
            candidate_scores = candidate_scores[selected]
        else:
            candidate_scores = (
                unique_scores[inverse[row_index], corpus_index[doc_id]]
                for doc_id in candidate_ids
            )
        results[query_id] = {
            doc_id: float(score)
            for doc_id, score in zip(candidate_ids, candidate_scores, strict=True)
        }
    return results


def _search_with_mind_optimization(
    wrapper: Any,
    queries: Dataset,
    *,
    task_metadata: Any,
    hf_split: str,
    hf_subset: str,
    top_k: int,
    encode_kwargs: dict[str, Any],
    top_ranked: Mapping[str, Any] | None = None,
    num_proc: int | None = None,
) -> Any:
    """Intercept only MindSmall search and delegate every other task unchanged."""
    assert _original_search is not None
    if task_metadata.name != MIND_SMALL_RERANKING:
        return _original_search(
            wrapper,
            queries,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
        )
    if top_ranked is None:
        # Running this reranking task against the full corpus changes both cost
        # and semantics, so fail explicitly instead of accepting MTEB's fallback.
        raise RuntimeError(
            "MindSmallReranking requires candidate documents; refusing to run it "
            "as full-corpus retrieval."
        )
    if wrapper.index_backend is not None:
        raise RuntimeError(
            "The MindSmallReranking optimization currently requires the default "
            "encoder search path without an external index backend."
        )
    if wrapper.task_corpus is None:
        raise ValueError("Corpus must be indexed before searching.")

    # Preserve the original prompt/tokenization/model path while removing repeated
    # query encoding and restricting output to the task's candidate documents.
    query_embeddings, inverse = _encode_unique_queries(
        wrapper,
        queries,
        task_metadata=task_metadata,
        hf_split=hf_split,
        hf_subset=hf_subset,
        encode_kwargs=encode_kwargs,
        num_proc=num_proc,
    )
    corpus_embeddings = _encode_mind_corpus(
        wrapper,
        task_metadata=task_metadata,
        hf_split=hf_split,
        hf_subset=hf_subset,
        encode_kwargs=encode_kwargs,
        num_proc=num_proc,
    )
    unique_scores = _score_unique_queries(
        wrapper.model,
        query_embeddings,
        corpus_embeddings,
    )
    results = _build_candidate_results(
        queries=queries,
        inverse=inverse,
        unique_scores=unique_scores,
        corpus_ids=wrapper.task_corpus["id"],
        top_ranked=top_ranked,
        top_k=top_k,
    )
    wrapper.task_corpus = None
    return results


def install_mind_small_reranking_patch() -> None:
    """Install the idempotent legacy loader/search hooks for validated MTEB."""
    global _mind_patch_installed, _original_mind_loader_load, _original_search
    if _mind_patch_installed:
        return

    from mteb.abstasks import retrieval_dataset_loaders
    from mteb.models import search_wrappers
    from vibe_eval.mteb_patches import (
        _validate_mteb_version,
        install_query_dataloader_patch,
    )

    version = _validate_mteb_version()
    install_query_dataloader_patch()
    # Save the current callables so non-Mind tasks and nested generic patches can
    # still delegate through the original MTEB behavior.
    _original_mind_loader_load = (
        retrieval_dataset_loaders.RetrievalDatasetLoader.load
    )
    _original_search = search_wrappers.SearchEncoderWrapper.search

    retrieval_dataset_loaders.RetrievalDatasetLoader.load = (
        _load_with_mind_candidates
    )
    search_wrappers.SearchEncoderWrapper.search = _search_with_mind_optimization
    _mind_patch_installed = True
    logger.info(
        "Installed repository-local MindSmallReranking patch for MTEB %s.",
        version,
    )


def resolve_mind_small_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the environment switch, defaulting to the compact implementation."""
    environment = os.environ if environ is None else environ
    mode = environment.get(MIND_SMALL_MODE_ENV, MIND_SMALL_COMPACT_MODE).strip().lower()
    if mode not in {MIND_SMALL_COMPACT_MODE, MIND_SMALL_LEGACY_MODE}:
        raise ValueError(
            f"{MIND_SMALL_MODE_ENV} must be 'compact' or 'legacy', got {mode!r}."
        )
    return mode


def patch_mind_small_reranking_tasks(
    tasks: Sequence[Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> list[Any]:
    """Select the implementation, install legacy hooks, and place MindSmall last."""
    mode = resolve_mind_small_mode(environ)
    patched_tasks = list(tasks)
    if mode == MIND_SMALL_COMPACT_MODE:
        from vibe_eval.tasks.mind_small_reranking import MindSmallReranking

        # Keep the public task name and upstream metadata while replacing only the
        # implementation object selected by MTEB.
        patched_tasks = [
            MindSmallReranking()
            if task.metadata.name == MIND_SMALL_RERANKING
            else task
            for task in patched_tasks
        ]
    elif any(
        task.metadata.name == MIND_SMALL_RERANKING for task in patched_tasks
    ):
        # Legacy hooks are global, so install them only when the selected task set
        # actually contains MindSmall.
        install_mind_small_reranking_patch()

    # Preserve relative order for all other tasks and defer this unusually large
    # task until the end of the evaluation.
    return sorted(
        patched_tasks,
        key=lambda task: task.metadata.name == MIND_SMALL_RERANKING,
    )
