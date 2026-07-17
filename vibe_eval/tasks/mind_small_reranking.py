from __future__ import annotations

"""Run MindSmallReranking from compact data and restore upstream metric inputs."""

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from mteb._create_dataloaders import create_dataloader
from mteb._evaluators.retrieval_evaluator import RetrievalEvaluator
from mteb.abstasks.retrieval import make_score_dict
from mteb.models.models_protocols import EncoderProtocol
from mteb.tasks.reranking.eng.mind_small_reranking import (
    MindSmallReranking as _UpstreamMindSmallReranking,
)
from mteb.types import PromptType

from vibe_eval.tasks.mind_small_compact_data import (
    COMPACT_DATA_DIR,
    CompactMindSmallData,
    verify_compact_data,
)


logger = logging.getLogger(__name__)


class MindSmallReranking(_UpstreamMindSmallReranking):
    """Replace data loading and search while preserving upstream metric semantics."""

    compact_data_dir = COMPACT_DATA_DIR

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Validate the manifest, file hashes, and expanded counts before loading."""
        if self.data_loaded:
            return
        compact = verify_compact_data(Path(self.compact_data_dir))
        assert compact is not None
        self.dataset = {"default": {"test": compact}}
        self.data_loaded = True

    def _evaluate_subset(
        self,
        model: Any,
        data_split: CompactMindSmallData,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode unique text, restore full results/qrels, and run MTEB metrics."""
        if not isinstance(model, EncoderProtocol):
            raise TypeError(
                "Compact MindSmallReranking currently requires an EncoderProtocol "
                f"model, got {type(model)}."
            )

        start = perf_counter()
        logger.info(
            "Running compact MindSmallReranking subset %s/%s: %d unique queries, "
            "%d corpus documents, %d impressions, %d subqueries",
            hf_subset,
            hf_split,
            len(data_split.queries),
            len(data_split.corpus),
            len(data_split.impressions),
            data_split.manifest["counts"]["subqueries"],
        )

        # MTEB expects the query key column to be named `id`. Keep `query_key` on
        # disk because it is also the impression foreign key, and rename only the
        # in-memory evaluation view.
        queries = Dataset(
            data_split.queries.rename_columns(
                ["id" if name == "query_key" else name for name in data_split.queries.column_names]
            )
        )
        corpus = Dataset(data_split.corpus)

        # Queries were deduplicated by text during the build, so each distinct
        # query text is encoded exactly once here.
        logger.info("Encoding %d compact MindSmall queries...", len(queries))
        query_embeddings = np.asarray(
            model.encode(
                create_dataloader(
                    queries,
                    task_metadata=self.metadata,
                    prompt_type=PromptType.query,
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.query,
                **encode_kwargs,
            )
        )
        query_end = perf_counter()
        logger.info(
            "Encoded %d compact MindSmall queries in %.2fs",
            len(queries),
            query_end - start,
        )
        logger.info("Encoding %d compact MindSmall corpus documents...", len(corpus))
        corpus_embeddings = np.asarray(
            model.encode(
                create_dataloader(
                    corpus,
                    task_metadata=self.metadata,
                    prompt_type=PromptType.document,
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )
        )
        corpus_end = perf_counter()
        logger.info(
            "Encoded %d compact MindSmall corpus documents in %.2fs",
            len(corpus),
            corpus_end - query_end,
        )

        # Impressions store stable IDs; these maps resolve them to embedding rows.
        logger.info(
            "Scoring and restoring %d compact MindSmall impressions...",
            len(data_split.impressions),
        )
        query_index = {
            key: index
            for index, key in enumerate(
                data_split.queries["query_key"].to_pylist()
            )
        }
        corpus_index = {
            doc_id: index
            for index, doc_id in enumerate(data_split.corpus["id"].to_pylist())
        }
        results, qrels = self._score_and_restore(
            model=model,
            impressions=data_split.impressions,
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
            query_index=query_index,
            corpus_index=corpus_index,
        )
        restore_end = perf_counter()
        logger.info(
            "Restored scores and qrels for %d subqueries in %.2fs",
            len(results),
            restore_end - corpus_end,
        )

        if prediction_folder:
            logger.info("Saving compact MindSmall predictions to %s", prediction_folder)
            self._save_task_predictions(
                results,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        logger.info("Running retrieval task - Evaluating retrieval scores...")
        # Search is complete, but metrics still go through upstream evaluate() so
        # NDCG/MAP/MRR/nAUC and task-specific aggregation cannot diverge.
        retriever = RetrievalEvaluator(
            corpus=corpus,
            queries=queries,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_ranked=None,
            top_k=self._top_k,
            **kwargs,
        )
        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            hit_rate,
        ) = retriever.evaluate(
            qrels,
            results,
            self.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )
        task_specific_scores = self.task_specific_scores(
            all_scores,
            qrels,
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        metrics_end = perf_counter()
        logger.info(
            "Compact MindSmall timings: query_encode=%.2fs corpus_encode=%.2fs "
            "score_restore=%.2fs metrics=%.2fs total=%.2fs",
            query_end - start,
            corpus_end - query_end,
            restore_end - corpus_end,
            metrics_end - restore_end,
            metrics_end - start,
        )
        logger.info("Running retrieval task - Finished.")
        return make_score_dict(
            ndcg=ndcg,
            _map=_map,
            recall=recall,
            precision=precision,
            mrr=mrr,
            naucs=naucs,
            naucs_mrr=naucs_mrr,
            hit_rate=hit_rate,
            task_scores=task_specific_scores,
            previous_results_model_meta=self._previous_results_model_meta,
        )

    def _score_and_restore(
        self,
        *,
        model: EncoderProtocol,
        impressions: Any,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        query_index: dict[str, int],
        corpus_index: dict[str, int],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
        """Score unique queries in blocks and restore every impression subquery."""
        results: dict[str, dict[str, float]] = {}
        qrels: dict[str, dict[str, int]] = {}
        # The 37k x 5k float32 matrix is about 0.8 GB. Blocked similarity avoids
        # one large GPU tensor and tens of thousands of tiny per-impression kernels.
        all_scores = np.empty(
            (len(query_index), len(corpus_index)),
            dtype=np.float32,
        )
        similarity_batch_size = 512
        for start in range(0, len(query_index), similarity_batch_size):
            stop = min(start + similarity_batch_size, len(query_index))
            score_block = torch.as_tensor(
                model.similarity(
                    query_embeddings[start:stop],
                    corpus_embeddings,
                )
            )
            if score_block.ndim != 2 or score_block.shape != (
                stop - start,
                len(corpus_index),
            ):
                raise RuntimeError(
                    f"Unexpected compact MindSmall score shape "
                    f"{tuple(score_block.shape)}."
                )
            if torch.isnan(score_block).any():
                raise ValueError("NaN scores in compact MindSmall similarity block.")
            all_scores[start:stop] = score_block.detach().float().cpu().numpy()

        # Candidate lists are stored once per impression; expand them back into
        # the per-subquery dictionaries required by MTEB.
        for impression in impressions.to_pylist():
            subquery_ids = impression["subquery_ids"]
            query_indices = [query_index[key] for key in impression["query_keys"]]
            candidate_ids = impression["candidate_ids"]
            candidate_indices = [corpus_index[doc_id] for doc_id in candidate_ids]
            impression_scores = all_scores[np.ix_(query_indices, candidate_indices)]
            shared_qrels = dict(
                zip(
                    impression["qrel_candidate_ids"],
                    impression["candidate_scores"],
                    strict=True,
                )
            )
            for row, query_id in enumerate(subquery_ids):
                # Most candidate lists fit within top_k; rank only oversized lists.
                if len(candidate_ids) <= self._top_k:
                    selected_indices = range(len(candidate_ids))
                else:
                    selected_indices = torch.topk(
                        torch.from_numpy(impression_scores[row]),
                        self._top_k,
                        largest=True,
                    ).indices.tolist()
                results[query_id] = {
                    candidate_ids[index]: float(impression_scores[row, index])
                    for index in selected_indices
                }
                # Qrels are identical within an impression, so sharing this dict is safe.
                qrels[query_id] = shared_qrels
        return results, qrels
