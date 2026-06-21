from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ToolRetRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ToolRetRetrieval",
        description=(
            "ToolRet retrieval benchmark for selecting relevant tool and API "
            "definitions from large tool corpora for natural-language "
            "tool-use tasks."
        ),

        reference="https://arxiv.org/abs/2503.01763",

        dataset={
            "path": "vec-ai/ToolRet",
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "web": ["eng-Latn"],
            "code": ["eng-Latn"],
            "customized": ["eng-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2025-03-03", "2025-03-03"),
        domains=["Programming", "Web", "Written"],
        task_subtypes=["Code retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{shi2025retrieval,
  title={Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
  author={Shi, Zhengliang and Wang, Yuhan and Yan, Lingyong and Ren, Pengjie and Wang, Shuaiqiang and Yin, Dawei and Ren, Zhaochun},
  journal={arXiv preprint arXiv:2503.01763},
  year={2025},
}
""",
        prompt={"query": "Given a natural language task description that requires external tool usage, retrieve the most relevant tool API definitions from the collection that can fulfill the request."},
    )
